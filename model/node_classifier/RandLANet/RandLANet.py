import os
import sys

import torch
import yaml
from torch import Tensor, nn
from torch_geometric.nn import conv

from utils.loading_utils import add_variable_name_prefix_in_config
from utils.point_utils import gather_neighbor, knn_search

from ..base import BasePointClassifier
from .modules import DilatedResidualBlock, linear_mlp, make_mlp


def to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [to_device(v, device) for v in obj]
        return tuple(t) if isinstance(obj, tuple) else t
    return obj


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
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            num_classes=num_classes, batch_size=batch_size, *args, **kwargs
        )

        self.save_hyperparameters()

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

        interp_idxs.reverse()

        return features

    def forward(
        self,
        features: Tensor,
        global_xyz: list[Tensor],
        neigh_idxs: list[Tensor],
        sub_idxs: list[Tensor],
        interp_idxs: list[Tensor],
        return_fc_features: bool = False,
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
        fc_fetures = []
        for fc in self.fc_blocks:
            features = fc(features)
            fc_fetures.append(features.clone().detach())

        if return_fc_features:
            return self.out_mlp(features), fc_fetures[-1]
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

    def common_evaluation(self, batch, return_fc_features: bool = False):
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
            _,
        ) = batch

        output = self(features, sample_xyz, neigh_idxs, sub_idxs, interp_idxs, return_fc_features)

        if self.num_classes == 2:
            if type(output) is tuple:
                logits = output[0]
                logits = logits.squeeze(1)
                if return_fc_features:
                    fc_features = output[1]
                    fc_features = fc_features.permute(0, 2, 1)
                    output = (logits, fc_features, *output[2:])
                else:
                    output = (logits, *output[1:])
            else:
                output = output.squeeze(1)

        return output, hit_labels, hit_weights

    def training_step(self, batch, batch_idx):

        output, hit_labels, hit_weights = self.common_evaluation(batch)

        loss = torch.mean(
            self.loss_fn(output, hit_labels.to(output.dtype)) * hit_weights
        )

        return {"loss": loss, "output": output, "hit_labels": hit_labels}

    def validation_step(self, batch, batch_idx):

        output, hit_labels, hit_weights = self.common_evaluation(batch)

        loss = torch.mean(
            self.loss_fn(output, hit_labels.to(output.dtype)) * hit_weights
        )

        return {"loss": loss, "output": output, "hit_labels": hit_labels}

    def test_step(self, batch, batch_idx):

        event_batch, event_idxs, events = batch

        output, hit_labels, _ = self.common_evaluation(event_batch)

        return {
            "output": torch.sigmoid(output).squeeze(),
            "hit_labels": hit_labels.squeeze(),
        }

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):

        # here, beside returning the prediction, we need to
        # rearrange the output to match original order of hits in the event
        event_batch, event_idxs, events = batch

        random_idx = event_batch[-1].squeeze()  # get the random idxs used in dataloader
        reco_idx = torch.empty_like(random_idx)

        reco_idx[random_idx] = torch.arange(0, random_idx.shape[0], device=random_idx.device)

        outputs, hit_labels, _ = self.common_evaluation(event_batch, return_fc_features=True)

        output, fc_features = outputs

        output_reco = output.squeeze()[reco_idx.to(output.device)]
        hit_labels = 1-hit_labels.squeeze()[reco_idx.to(hit_labels.device)]

        hit_score = 1-torch.sigmoid(output_reco)

        fc_features_reco = fc_features.squeeze()[reco_idx.to(fc_features.device)]

        return {"output": hit_score, "hit_labels": hit_labels, "fc_features": fc_features_reco}

    def subsample(self, pc):

        global_xyz, neigh_idxs, sub_idxs, interp_idxs = [], [], [], []

        assert len(self.knn) == len(
            self.sampling_ratio
        ), "The number of KNN samples must be the same as the number of sampling ratio"

        for num_neigh, sampling_ratio in zip(self.knn, self.sampling_ratio):
            neigh_idx = knn_search(pc, pc, num_neigh + 1, engine=self.knn_engine)
            neigh_idx = neigh_idx[:, 1:]  # eliminate self-loop
            sub_points, idx, _ = self.random_sample(pc.shape[0] // sampling_ratio, pc)
            sub_points = sub_points[0]
            pool_idx = neigh_idx[idx, :]
            interp_idx = knn_search(sub_points, pc, 1, engine=self.knn_engine)
            global_xyz.append(pc)
            neigh_idxs.append(neigh_idx)
            sub_idxs.append(pool_idx)
            interp_idxs.append(interp_idx)
            pc = sub_points

        return global_xyz, neigh_idxs, sub_idxs, interp_idxs, pc


class RandLAGCN(RandLANet):

    def __init__(self, hparams, *args, **kwargs) -> None:
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
            on_step=True,
            prog_bar=True,
            batch_size=self.hparams["batch_size"],
        )

        metrics = self.metrics(output, hit_labels)

        return metrics


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
            break
