
import torch
from torch import Tensor, nn

from utils.point_utils import gather_neighbor


def make_mlp(d_in: int, d_out: int, n_layers: int = 1, act_out=True, dim="1d"):

    conv_module = nn.Conv1d
    bn_module = nn.BatchNorm1d
    if dim == "2d":
        conv_module = nn.Conv2d
        bn_module = nn.BatchNorm2d

    modules = [conv_module(d_in, d_out, kernel_size=1)]
    for i in range(n_layers - 1):
        modules.append(nn.LeakyReLU())
        modules.append(conv_module(d_out, d_out, kernel_size=1))
    modules.append(bn_module(d_out))
    if act_out:
        modules.append(nn.LeakyReLU())

    return nn.Sequential(*modules)


class LocSE(nn.Module):
    def __init__(self, rel_pos_dim: int, d_out: int, n_layers=1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.net = make_mlp(rel_pos_dim, d_out // 2, n_layers, True)

    def gather_pos_enc(self, xyz_t: Tensor, neighbor_idx: Tensor):

        # xyz : batch * npoints * channels
        # neighbor_idx : batch * npoints * num_neighbor
        neighbor_xyz = gather_neighbor(xyz_t, neighbor_idx)  # batch * (npoints * num_neighbor) * channels

        num_neighbor = neighbor_idx.shape[-1]

        batch_size, d = xyz_t.shape[0], xyz_t.shape[1]

        xyz_t = (
            xyz_t.repeat(1, num_neighbor, 1).permute(0, 2, 1).reshape(batch_size, -1, d).permute(0, 2, 1)
        )  # batch * (npoints * num_neighbor) * channels

        relative_xyz = xyz_t - neighbor_xyz  # batch* channells * (knn x npoints)
        relative_dist = torch.sqrt(
            torch.sum(torch.pow(relative_xyz, 2), dim=1, keepdim=True)
        )  # batch*(knn x npoints)*1

        return torch.cat(
            [relative_dist, relative_xyz, xyz_t, neighbor_xyz], dim=1
        )  # batch*(1+3+3+3) * (N * num_neighbor)

    def forward(self, xyz_t: Tensor, neighbor_idx: Tensor):
        # xyz : batch * channels * npoints
        # neighbor_idx : batch * npoints * num_neighbor

        relative_position = self.gather_pos_enc(xyz_t, neighbor_idx)  # batch*10*(knn x npoints)

        return self.net(relative_position)  # batch* (d_out //2) * (knn x npoints)


class AttentivePooling(nn.Module):
    def __init__(self, d_in: int, d_out: int, n_layers: int = 1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.feature_converter = nn.Conv2d(d_in, d_in, kernel_size=(1, 1))
        # self.mlp = nn.Sequential(nn.Conv2d(d_in, d_out, kernel_size=(1, 1)), nn.BatchNorm2d(d_out), nn.LeakyReLU())
        self.mlp = make_mlp(d_in, d_out, n_layers, dim="2d")
        # self.aggregator = SumAggregation()

    def forward(self, feature_set):
        # feature_set : batch* d_in (knn * npoints)
        # feature_set = feature_set.view(-1, -1, -1, self.knn)  # batch * d_in * npoints * knn
        att_activation = self.feature_converter(feature_set)  # batch * d_in * npoints * knn
        att_scores = nn.functional.softmax(att_activation, dim=3)

        attended_features = feature_set * att_scores
        aggregate = torch.sum(attended_features, dim=3, keepdim=True)  # batch * d_in  * (npoints) * 1
        point_output = self.mlp(aggregate)

        return point_output.squeeze(-1)  # batch * d_out  * (npoints)


class DilatedResidualBlock(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, d_in, d_out, n_layers=1, rel_pos_dim=10, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.point_encoder_1 = nn.Sequential(
            nn.Conv1d(d_in, d_out // 2, kernel_size=(1,)), nn.BatchNorm1d(d_out // 2), nn.LeakyReLU()
        )
        self.loc_se = LocSE(rel_pos_dim, d_out, n_layers)
        self.att_pooling_1 = AttentivePooling(d_out, d_out // 2, n_layers)

        self.point_encoder_2 = make_mlp(d_out // 2, d_out // 2, n_layers, True)

        # nn.Sequential(
        #     nn.Conv1d(d_out // 2, d_out // 2, kernel_size=(1,)), nn.BatchNorm1d(d_out // 2), nn.LeakyReLU()
        # )
        self.att_pooling_2 = AttentivePooling(d_out, d_out, n_layers)

        self.mlp = make_mlp(
            d_out, d_out * 2, n_layers, False
        )  # nn.Sequential(nn.Conv1d(d_out, d_out * 2, kernel_size=(1,)), nn.BatchNorm1d(d_out * 2))
        self.skip_mlp = make_mlp(
            d_in, d_out * 2, n_layers, False
        )  # nn.Sequential(nn.Conv1d(d_in, d_out * 2, kernel_size=(1,)), nn.BatchNorm1d(d_out * 2))

    def forward(self, xyz_t: Tensor, features_t: Tensor, neighbor_idx: Tensor):

        # features : batch * channels * npoints
        # xyz : batch * channels * npoints
        # neighbor_idx : batch * npoints * num_neighbor

        batch_size, N, num_neighbor = neighbor_idx.shape

        point_features = self.point_encoder_1(features_t)  # batch * d_out // 2 * (npoints * num_neighbor)

        point_features = gather_neighbor(point_features, neighbor_idx)

        f_xyz = self.loc_se(xyz_t, neighbor_idx)  # batch *( d_out // 2) * (npoints * knn)

        point_features = torch.cat([f_xyz, point_features], dim=1)

        point_features = self.att_pooling_1(
            point_features.reshape(batch_size, point_features.shape[1], N, num_neighbor)
        )  # batch * (d_out // 2) * (npoints )

        f_xyz = self.point_encoder_2(f_xyz)  # batch * d_out // 2 * (npoints * knn)

        point_features = gather_neighbor(point_features, neighbor_idx)

        point_features = torch.cat([f_xyz, point_features], dim=1)

        point_features = self.att_pooling_2(
            point_features.reshape(batch_size, point_features.shape[1], N, num_neighbor)
        )  # batch * d_out * (npoints )

        point_features = self.mlp(point_features) + self.skip_mlp(features_t)  # batch * (2 x d_out) * (npoints )

        return nn.functional.leaky_relu(point_features)
