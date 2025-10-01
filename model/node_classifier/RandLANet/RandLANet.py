import sys

import torch
import yaml
from torch import Tensor, nn
from torch_geometric.nn import conv

from utils.loading_utils import add_variable_name_prefix_in_config
from utils.point_utils import gather_neighbor

from ..base import BasePointClassifier
from .modules import DilatedResidualBlock, make_mlp


class RandLANet(BasePointClassifier):
    def __init__(
        self,
        feature_dim: int,
        position_dim: int,
        d_in: int,
        encoding_blocks: list[int],
        n_layers: int,
        fc_blocks: list[int],
        num_classes: int,
        batch_size: int,
        do_compile: bool = False,
        *args: torch.Any,
        **kwargs: torch.Any
    ) -> None:
        super().__init__(
            num_classes=num_classes, batch_size=batch_size, *args, **kwargs
        )

        # feature_dim = len(self.input_features)
        # n_layers = self.hparams.get('n_layers', 1)

        # d_in = self.hparams['d_in']

        rel_pos_dim = 1 + position_dim * 3

        self.feature_encoder = nn.Sequential(
            nn.Conv1d(feature_dim, d_in, kernel_size=(1,)),
            nn.BatchNorm1d(d_in),
            nn.LeakyReLU(),
        )

        self.dilated_res_blocks = nn.ModuleList()

        self.decoder_blocks = []
        skip_dim = 0
        fc_in = 0
        for i, d_out in enumerate(encoding_blocks):
            # d_out = block_conf['d_out'] # d_out of the current dilated res block the actual stored dim is 2*d_out
            d_in_decoder, d_out_decoder = d_out * 2 + skip_dim, skip_dim
            if i == 0:
                d_in_decoder = 4 * d_out
                d_out_decoder = 2 * d_out
                fc_in = d_out_decoder

            self.dilated_res_blocks.append(
                DilatedResidualBlock(d_in, d_out, n_layers, rel_pos_dim)
            )

            self.decoder_blocks.append(make_mlp(d_in_decoder, d_out_decoder, n_layers))
            d_in = skip_dim = 2 * d_out

        d_out = d_in

        self.decoder_mlp = make_mlp(d_in, d_out, n_layers)

        self.decoder_blocks.reverse()
        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)

        self.fc_blocks = nn.ModuleList()
        for i, fc_out in enumerate(fc_blocks):
            # fc_out = fc_conf['d_out']
            self.fc_blocks.append(make_mlp(fc_in, fc_out, n_layers))
            fc_in = fc_out

        self.out_mlp = nn.Conv1d(fc_in, self.num_classes, kernel_size=(1,))
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

        if self.num_classes == 2:
            self.out_mlp = nn.Conv1d(fc_in, 1, kernel_size=(1,))
            self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

        self.do_compile = do_compile

    def configure_model(self) -> None:
        super().configure_model()

        if self.do_compile:
            self.feature_encoder = torch.compile(self.feature_encoder, fullgraph=True)
            self.dilated_res_blocks = self.compile_module_list(self.dilated_res_blocks)
            self.decoder_mlp = torch.compile(self.decoder_mlp, fullgraph=True)
            self.decoder_blocks = self.compile_module_list(self.decoder_blocks)
            self.fc_blocks = self.compile_module_list(self.fc_blocks)
            self.out_mlp = torch.compile(self.out_mlp, fullgraph=True)

    def encode(
        self,
        features: Tensor,
        global_xyz: list[Tensor],
        neigh_idxs: list[Tensor],
        sub_idxs: list[Tensor],
    ):

        f_encoder_list = []

        for dilated_block, xyz, neigh_idx, sub_idx in zip(
            self.dilated_res_blocks, global_xyz, neigh_idxs, sub_idxs
        ):
            xyz_t = xyz.permute(0, 2, 1)
            f_encoder_i = dilated_block(xyz_t, features, neigh_idx)

            f_sample_i = self.sample_from_idx(f_encoder_i, sub_idx)
            features = f_sample_i
            if len(f_encoder_list) == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sample_i)

        return features, f_encoder_list

    def decode(
        self, features: Tensor, interp_idxs: list[Tensor], f_encoder_list: list[Tensor]
    ):

        interp_idxs.reverse()

        for decoder, inter_idx, f_encoder_i in zip(
            self.decoder_blocks, interp_idxs, f_encoder_list[1:]
        ):
            f_interp_i = self.nearest_interpolation(features, inter_idx)
            features = decoder(torch.cat([f_interp_i, f_encoder_i], dim=1))

        return features

    def forward(
        self,
        features: Tensor,
        global_xyz: list[Tensor],
        neigh_idxs: list[Tensor],
        sub_idxs: list[Tensor],
        interp_idxs: list[Tensor],
    ):

        features = features.permute(0, 2, 1)
        features = self.feature_encoder(features)  # batch * channel * npoints
        # f_encoder_list = []

        # ### encoding
        features, f_encoder_list = self.encode(
            features, global_xyz, neigh_idxs, sub_idxs
        )

        f_encoder_list.reverse()

        ### MLP
        features = self.decoder_mlp(features)

        ### decoding
        features = self.decode(features, interp_idxs, f_encoder_list)

        # FC
        for fc in self.fc_blocks:
            features = fc(features)

        return self.out_mlp(features)

    @staticmethod
    def sample_from_idx(feature, pool_idx):
        """

        Args:
            feature (_type_): [B, N, d] full input features from B batches, N points and d dimensions/point
            pool_idx (_type_): [B, N', n_max] N' < N, n_max is the number of nearest neighbors of each sampled point
        """
        # feature: batch * channel * npoints
        # pool_idx : batch * N' * num_neigh
        num_neigh = pool_idx.shape[-1]
        pool_feature = gather_neighbor(feature, pool_idx)
        pool_feature = pool_feature.reshape(*pool_feature.shape[:-1], -1, num_neigh)
        pool_feature = pool_feature.max(dim=3).values
        return pool_feature

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """_summary_

        Args:
            feature (_type_): _description_
            interp_idx (_type_): [B, up_num_points, 1] nearest neighbour index
        """

        batch_size = interp_idx.shape[0]
        up_num_point = interp_idx.shape[1]
        d = feature.shape[1]
        interpolated_features = torch.gather(
            feature,
            2,
            interp_idx.reshape(batch_size, up_num_point).unsqueeze(1).repeat(1, d, 1),
        )
        return interpolated_features

    def training_step(self, batch, batch_idx):
        (
            sample_xyz,
            neigh_idxs,
            sub_idxs,
            interp_idxs,
            xyz,
            features,
            hit_labels,
            hit_weights,
            _,
        ) = batch

        output = self(features, sample_xyz, neigh_idxs, sub_idxs, interp_idxs)

        if self.num_classes == 2:
            output = output.squeeze(1)
            hit_labels = hit_labels.to(output.dtype)

        loss = torch.mean(self.loss_fn(output, hit_labels) * hit_weights)

        self.log_val("train_loss", loss, batch_size=self.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):

        (
            sample_xyz,
            neigh_idxs,
            sub_idxs,
            interp_idxs,
            xyz,
            features,
            hit_labels,
            hit_weights,
            _,
        ) = batch

        output = self.forward(features, sample_xyz, neigh_idxs, sub_idxs, interp_idxs)

        if self.num_classes == 2:
            output = output.squeeze(1)
            hit_labels = hit_labels.to(output.dtype)

        loss = torch.mean(self.loss_fn(output, hit_labels) * hit_weights)

        self.log_val("val_loss", loss, batch_size=self.batch_size)

        metrics = self.metrics(output, hit_labels)

        return metrics


class RandLAGCN(RandLANet):
    def __init__(self, hparams, *args: torch.Any, **kwargs: torch.Any) -> None:
        super().__init__(hparams, *args, **kwargs)

        self.graph_convs = nn.ModuleList()

        for i in range(self.hparams.get("n_conv", 1)):

            default_conv = "SAGEConv"
            default_conv_args = {
                "in_channels": hparams["encoding_blocks"][-1]["d_out"] * 2,
                "out_channels": hparams["encoding_blocks"][-1]["d_out"] * 2,
                "project": True,
            }
            conv_module = hparams.get("graph_conv", default_conv)
            conv_args = hparams.get("graph_conv_args", default_conv_args)

            conv_module = getattr(conv, conv_module)

            top_layer_conv = conv_module(**conv_args)

            self.graph_convs.append(top_layer_conv)

        del self.decoder_mlp

        # self.graph_conv = top_layer_conv

    def forward(
        self,
        features: Tensor,
        top_edge_idxs: Tensor,
        global_xyz: list[Tensor],
        neigh_idxs: list[Tensor],
        sub_idxs: list[Tensor],
        interp_idxs: list[Tensor],
    ):

        features = features.permute(0, 2, 1)
        features = self.feature_encoder(features)  # batch * channel * npoints
        # f_encoder_list = []

        # ### encoding
        features, f_encoder_list = self.encode(
            features, global_xyz, neigh_idxs, sub_idxs
        )

        f_encoder_list.reverse()

        ### MLP change to Graph Conv
        # must provde an edge index for top layer
        features = features.permute(0, 2, 1)

        for conv in self.graph_convs:
            features = torch.stack(
                [
                    conv(features[i], top_edge_idxs[i])
                    for i in range(top_edge_idxs.shape[0])
                ],
                dim=0,
            )

        features = features.permute(0, 2, 1)

        ### decoding
        features = self.decode(features, interp_idxs, f_encoder_list)

        # FC
        for fc in self.fc_blocks:
            features = fc(features)

        return self.out_mlp(features)

    def training_step(self, batch, batch_idx):
        (
            sample_xyz,
            neigh_idxs,
            sub_idxs,
            interp_idxs,
            xyz,
            features,
            hit_labels,
            hit_weights,
            top_knn,
        ) = batch

        output = self(features, top_knn, sample_xyz, neigh_idxs, sub_idxs, interp_idxs)

        loss = torch.mean(self.loss_fn(output, hit_labels.long()) * hit_weights)

        self.log_val(
            "train_loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            batch_size=self.hparams["batch_size"],
        )

        return loss

    def validation_step(self, batch, batch_idx):

        (
            sample_xyz,
            neigh_idxs,
            sub_idxs,
            interp_idxs,
            xyz,
            features,
            hit_labels,
            hit_weights,
            top_knn,
        ) = batch

        output = self(features, top_knn, sample_xyz, neigh_idxs, sub_idxs, interp_idxs)

        loss = torch.mean(self.loss_fn(output, hit_labels.long()) * hit_weights)

        self.log_val(
            "val_loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            batch_size=self.hparams["batch_size"],
        )

        metrics = self.metrics(output, hit_labels)

        return metrics


class PointSegMLP(RandLANet):
    def __init__(self, hparams, *args: torch.Any, **kwargs: torch.Any) -> None:
        super().__init__(hparams, *args, **kwargs)

        feature_dim = len(self.input_features)  # + len(self.position_features)

        n_layers = self.hparams.get("n_layers", 1)

        fc_in = self.hparams["d_in"]

        self.feature_encoder = make_mlp(
            feature_dim, fc_in, n_layers
        )  # nn.Sequential(nn.Conv1d(feature_dim, d_in, kernel_size=(1,)), nn.BatchNorm1d(d_in), nn.LeakyReLU())

        self.fc_blocks = nn.ModuleList()
        for fc_conf in self.hparams["fc_blocks"]:
            fc_out = fc_conf["d_out"]
            self.fc_blocks.append(make_mlp(fc_in, fc_out, n_layers, dim="1d"))
            fc_in = fc_out

        self.out_mlp = nn.Conv1d(
            fc_in,
            1 if self.hparams["num_class"] == 2 else self.hparams["num_class"],
            kernel_size=(1,),
        )

        del self.dilated_res_blocks, self.decoder_blocks, self.decoder_mlp

    def forward(
        self,
        features: Tensor,
        global_xyz: list[Tensor],
        neigh_idxs: list[Tensor],
        sub_idxs: list[Tensor],
        interp_idxs: list[Tensor],
    ):

        features = features.permute(0, 2, 1)
        # xyz = global_xyz[0].permute(0,2,1)

        # features = torch.concat([features, xyz], dim=1)
        features = self.feature_encoder(features)  # batch * channel * npoints

        # FC
        for fc in self.fc_blocks:
            features = fc(features)

        return self.out_mlp(features)


if __name__ == "__main__":
    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        config = yaml.full_load(f)

    if not config.get("variable_with_prefix"):
        config = add_variable_name_prefix_in_config(config)

    model = config["model"]
    match model:
        case "RandLAGCN":
            model = RandLAGCN(config)
        case _:
            model = RandLANet(config)

    model.setup()
    model.cuda()
    # with torch.no_grad():
    i = 0
    for batch in model.train_dataloader():
        (
            sample_xyz,
            neigh_idxs,
            sub_idxs,
            interp_idxs,
            xyz,
            features,
            hit_labels,
            hit_weights,
            top_knn,
        ) = batch
        print("Sample xyz sizes: ", [_.shape for _ in sample_xyz])
        print("Sample neighbor idx sizes: ", [_.shape for _ in neigh_idxs])
        print("Sample sub idx sizes: ", [_.shape for _ in sub_idxs])
        print("Sample interp idx sizes: ", [_.shape for _ in interp_idxs])
        print("Input xyz size: ", xyz.shape)
        print("Input feature size: ", features.shape)
        print("Hit label size: ", hit_labels.shape)
        print("Hit weight size: ", hit_weights.shape)
        print("Top KNN", top_knn.shape)

        features = features.cuda()
        top_knn = top_knn.cuda()
        sample_xyz = [_.cuda() for _ in sample_xyz]
        neigh_idxs = [_.cuda() for _ in neigh_idxs]
        sub_idxs = [_.cuda() for _ in sub_idxs]
        interp_idxs = [_.cuda() for _ in interp_idxs]

        with torch.autocast("cuda"):
            match config["model"]:
                case "RandLAGCN":
                    output = model(
                        features, top_knn, sample_xyz, neigh_idxs, sub_idxs, interp_idxs
                    )
                case _:
                    output = model(
                        features, sample_xyz, neigh_idxs, sub_idxs, interp_idxs
                    )
            loss = model.loss_fn(output, hit_labels.long().cuda()) * hit_weights.cuda()
            loss = torch.mean(loss)

            loss.backward()

            for params in model.parameters():
                if torch.isnan(params.grad).any():
                    print(params.grad)

            metrics = model.metrics(output.detach(), hit_labels.cuda())
            print(metrics)

        i += 1
        if i >= 1:
            break
