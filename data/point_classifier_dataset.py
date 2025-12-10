import torch
from torch_geometric.data import Data

from model.utils import get_signal_mask
from utils.loading_utils import add_variable_name_prefix_in_config
from utils.point_utils import knn_search

from .event_dataset import EventDataset
from .utils import get_weight_mask, handle_weighting

PI = 3.14159
class PointCloudDataset(EventDataset):

    def __init__(
        self,
        input_dir,
        input_features,
        node_features,
        position_features,
        data_name: str = "",
        num_events=None,
        use_csv=False,
        event_prefix="",
        variable_with_prefix: bool = False,
        node_scales: list[float] = [],
        points_per_batch: int = 1000,
        knn: list[int] = [],
        sampling_ratio: list[int] = [],
        weighting: list[dict] = [{}],
        stage="fit",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        subsampling="random",
        augment_phi: bool = False,
        knn_engine="torch",
        **kwargs,
    ):

        self.input_features = input_features
        self.position_features = position_features
        self.subsampling = subsampling
        self.points_per_batch = points_per_batch
        self.knn = knn
        self.sampling_ratio = sampling_ratio
        self.weighting = weighting
        self.augment_phi = augment_phi
        self.knn_engine = knn_engine
        if not variable_with_prefix:
            self.input_features = add_variable_name_prefix_in_config(
                {"node_features": self.input_features}
            )["node_features"]
            self.position_features = add_variable_name_prefix_in_config(
                {"node_features": self.position_features}
            )["node_features"]

        super().__init__(
            input_dir,
            data_name,
            num_events,
            use_csv,
            event_prefix,
            variable_with_prefix,
            node_features,
            node_scales,
            stage,
            transform,
            pre_transform,
            pre_filter,
            **kwargs,
        )

    def get(self, idx):

        data = super().get(idx)

        if self.use_csv:
            event = data[0]
            event_idx = data[-1]
        else:
            event, event_idx = data

        # augment phi
        if self.augment_phi:
            self._augment_phi(event)

        # transform phi
        self._transform_phi(event)

        hit_labels, hit_weights = self.get_hit_label(event)

        features = sorted([f for f in self.input_features if f not in self.position_features] )

        features = self.gather_features(event, features)

        xyz = self.gather_features(event, self.position_features)

        data, idx, random_idx = self.random_sample(
            self.points_per_batch, xyz, features, hit_labels, hit_weights
        )

        if self.subsampling == "full" and self.points_per_batch > 0:
            start = 0
            end = start + self.points_per_batch
            data = []
            while end < xyz.shape[0]:
                idx = random_idx[start:end]
                data.append(
                    self.sample_from_idx(idx, xyz, features, hit_labels, hit_weights)
                    + [idx]
                )
                start += self.points_per_batch
                end += self.points_per_batch
            idx = random_idx[-self.points_per_batch :]
            data.append(
                self.sample_from_idx(idx, xyz, features, hit_labels, hit_weights)
                + [idx]
            )

            output = []
            for d in data:
                xyz, features, hit_labels, hit_weights, idx = d

                sample_xyz, neigh_idxs, sub_idxs, interp_idxs, top_pc = self.subsample(
                    xyz
                )

                output.append(
                    [
                        sample_xyz,
                        neigh_idxs,
                        sub_idxs,
                        interp_idxs,
                        xyz,
                        features,
                        hit_labels,
                        hit_weights,
                        top_pc,
                        idx,
                    ]
                )

            return output

        xyz, features, hit_labels, hit_weights = data

        sample_xyz, neigh_idxs, sub_idxs, interp_idxs, top_pc = self.subsample(xyz)

        if self.stage in ["predict", "test"]:
            self.unscale_features(event)
            return (
                sample_xyz,
                neigh_idxs,
                sub_idxs,
                interp_idxs,
                xyz,
                features,
                hit_labels,
                hit_weights,
                top_pc,
                random_idx,
                event_idx, 
                event
            )

        return (
            sample_xyz,
            neigh_idxs,
            sub_idxs,
            interp_idxs,
            xyz,
            features,
            hit_labels,
            hit_weights,
            top_pc,
            random_idx,
        )

    def subsample(self, pc):

        global_xyz, neigh_idxs, sub_idxs, interp_idxs = [], [], [], []

        assert len(self.knn) == len(
            self.sampling_ratio
        ), "The number of KNN samples must be the same as the number of sampling ratio"

        for num_neigh, sampling_ratio in zip(
            self.knn, self.sampling_ratio
        ):
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

    @staticmethod
    def _transform_phi(graph: Data):
        # assume that phi is given in multiple of pi, i.e. -1 < phi < 1
        graph["hit_sin_phi"] = torch.sin(graph["hit_phi"] * PI)
        graph["hit_cos_phi"] = torch.cos(graph["hit_phi"] * PI)

    @staticmethod
    def _augment_phi(graph: Data):
        # assume that hit_phi is in the graph
        # assume that phi is given in multiple of pi, i.e. -1 < phi < 1
        # augment phi by a value evenly distributed, -1 < delta phi < 1
        # recompute hit_x and hit_y
        graph.hit_phi = graph.hit_phi + torch.rand(1) * 2
        graph.hit_x = graph.hit_r * torch.cos(graph.hit_phi * PI)
        graph.hit_y = graph.hit_r * torch.sin(graph.hit_phi * PI)

    def get_hit_label(self, event):

        track_edges = event.track_edges

        hit_label = torch.ones_like(event.hit_x, dtype=torch.long) * -1

        hit_weights = torch.ones_like(hit_label, dtype=torch.float)

        for label, config in enumerate(self.weighting):
            w = config["weight"]

            assert (
                w != 1
            ), "weight value must not be exactly 1. If you want to set weight to 1, choose a value close to 1, like 0.998"

            track_weight = handle_weighting(
                event,
                [config],
                pred_edges=track_edges,
                truth=torch.ones_like(track_edges[0]).bool(),
                true_edges=track_edges,
                truth_map=torch.arange(0, track_edges.size(1)),
            )

            assert track_weight.size(0) == track_edges.size(
                1
            ), "Track weights must have the same size as track edges"

            mask = track_weight == w

            hits = track_edges[:, mask].flatten()

            hit_label[hits] = label

            hit_weights[hits] = w

        hit_label[hit_label == (-1)] = label + 1

        # for label, w in enumerate(torch.unique(track_weight).sort().values):
        #     track_mask = (track_weight==w)
        #     hits = track_edges[:, track_mask].flatten()
        #     hit_label[hits] = label
        #     hit_weights[hits] = w

        # hit_label[hit_label==-1] = label + 1

        return hit_label.long(), hit_weights.float()

    def gather_features(self, event, features):
        input_data = torch.stack(
            [event[feature] for feature in features], dim=-1
        ).float()

        return input_data

    @staticmethod
    def random_sample(n: int, *values: torch.Tensor):

        N = values[0].shape[0]

        assert (
            N >= n
        ), f"The number of examples in the subsample {n} must be less than the total number of examples {N}"

        for v in values:
            assert (
                N == v.shape[0]
            ), "All samples must have the same first dimension, also the number of examples"

        random_idx = torch.randperm(N)

        idx = random_idx[:n]

        if n == -1:
            idx = random_idx = torch.arange(N)

        return [v[idx] for v in values], idx, random_idx

    @staticmethod
    def sample_from_idx(idx: torch.Tensor, *values: torch.Tensor):
        return [v[idx] for v in values]


class PointCloudDatasetMetricLearning(PointCloudDataset):
    def __init__(
        self,
        input_dir,
        input_features,
        node_features,
        position_features,
        data_name: str = "",
        num_events=None,
        use_csv=False,
        event_prefix="",
        variable_with_prefix: bool = False,
        node_scales: list[float] = [],
        points_per_batch: int = 1000,
        knn: list[int] = [],
        sampling_ratio: list[int] = [],
        weighting: list[dict] = [{}],
        stage="fit",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        subsampling="random",
        augment_phi: bool = False,
        **kwargs,
    ):
        super().__init__(
            input_dir,
            input_features,
            node_features,
            position_features,
            data_name,
            num_events,
            use_csv,
            event_prefix,
            variable_with_prefix,
            node_scales,
            points_per_batch,
            knn,
            sampling_ratio,
            weighting,
            "predict",
            transform,
            pre_transform,
            pre_filter,
            subsampling,
            augment_phi,
            **kwargs,
        )

    def get(self, idx):

        data = super().get(idx)

        event = data[-1]
        event_idx = data[-2]

        self.scale_features(event)

        track_edges = event.track_edges

        event.track_weights = handle_weighting(
            event,
            weighting_config=self.weighting,
            pred_edges=track_edges,
            truth=torch.ones_like(track_edges[0]).bool(),
            true_edges=track_edges,
            truth_map=torch.arange(0, track_edges.size(1)),
        )

        event.track_signal_mask = event.track_weights > 0

        self.unscale_features(event)

        return (*data[:-2], event_idx, event)


class PointClassifierDatasetNoSubsampling(PointCloudDataset):

    def get(self, idx):

        data = super().get(idx)

        if self.use_csv:
            event = data[0]
            event_idx = data[-1]
        else:
            event, event_idx = data

        # augment phi
        if self.augment_phi:
            self._augment_phi(event)

        # transform phi
        self._transform_phi(event)

        hit_labels, hit_weights = self.get_hit_label(event)

        features = sorted(
            [f for f in self.input_features if f not in self.position_features]
        )

        features = self.gather_features(event, features)

        xyz = self.gather_features(event, self.position_features)

        data, idx, random_idx = self.random_sample(
            self.points_per_batch, xyz, features, hit_labels, hit_weights
        )

        xyz, features, hit_labels, hit_weights = data

        if self.stage in ["predict", "test"]:
            self.unscale_features(event)
            return (
                xyz,
                features,
                hit_labels,
                hit_weights,
                random_idx,
                event_idx,
                event,
            )

        return (
            xyz,
            features,
            hit_labels,
            hit_weights,
            random_idx,
        )
