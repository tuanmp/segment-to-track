import os
import re
import warnings

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningModule
from torch_geometric.data import Dataset
from tqdm import tqdm

from utils import get_pyg_data_keys, handle_hard_node_cuts
from utils.loading_utils import (
    add_variable_name_prefix_in_config,
    add_variable_name_prefix_in_pyg,
    infer_num_nodes,
)


class EventDataset(Dataset):
    """
    The custom default GNN dataset to load graphs off the disk
    """

    def __init__(
        self,
        input_dir: str,
        data_name: str = "",
        num_events=None,
        use_csv: bool=False,
        event_prefix="",
        variable_with_prefix: bool = False,
        node_features: list[str] = [],
        node_scales: list[float] = [],
        stage="fit",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        **kwargs,
    ):
        super().__init__(input_dir, transform, pre_transform, pre_filter)

        self.input_dir = input_dir
        self.data_name = data_name
        self.num_events = num_events
        self.event_prefix = event_prefix
        self.variable_with_prefix = variable_with_prefix
        self.node_features = node_features
        self.node_scales = node_scales
        self.use_csv = use_csv
        self.stage = stage
        self.evt_ids = self.find_evt_ids()

        if not self.variable_with_prefix:
            self.node_features = add_variable_name_prefix_in_config({'node_features': self.node_features})["node_features"]

    def len(self):
        return len(self.evt_ids)

    def get(self, idx):
        """
        Handles the iteration through the dataset. Depending on how dataset is configured for PyG and CSV file loading,
        will return either PyG events (automatically batched by PyG), CSV files (not batched/concatenated), or both (not batched/concatenated).
        Assumes files are saved correctly from EventReader as: event000...-truth.csv, -particles.csv and -graph.pyg
        """
        event_file_root_name = f"event{self.evt_ids[idx]}"
        if self.event_prefix:
            event_file_root_name = f"{self.event_prefix}_{event_file_root_name}"

        event_path = os.path.join(self.input_dir, self.data_name, event_file_root_name)

        graph_path = (
            f"{event_path}-graph.pyg"
            if os.path.exists(f"{event_path}-graph.pyg")
            else f"{event_path}.pyg"
        )
        graph = torch.load(graph_path, weights_only=False)

        graph = self.preprocess_graph(graph)
        if not self.use_csv:
            return graph, self.evt_ids[idx]

        particles = pd.read_csv(f"{event_path}-particles.csv")
        hits = pd.read_csv(f"{event_path}-truth.csv")
        hits = self.preprocess_hits(hits, graph)

        return graph, particles, hits, self.evt_ids[idx]

    def preprocess_graph(self, graph):
        """Preprocess the PyG graph before returning it."""
        if not self.variable_with_prefix:
            graph = add_variable_name_prefix_in_pyg(graph)
        infer_num_nodes(graph)
        # graph = self.apply_hard_cuts(graph)
        self.scale_features(graph)
        return graph

    def preprocess_hits(self, hits, graph):
        """Preprocess the hits dataframe before returning it."""
        # hits = self.apply_hard_cuts(hits, passing_hit_ids=graph.hit_id.numpy())
        return hits

    def find_evt_ids(self):
        """
        Returns a list of all event ids, which are the numbers in filenames that end in .csv and .pyg
        """

        input_data_dir = os.path.join(self.input_dir, self.data_name)
        all_files = os.listdir(input_data_dir) if os.path.isdir(input_data_dir) else []
        all_files = [f for f in all_files if f.endswith(".csv") or f.endswith(".pyg")]
        all_event_ids = sorted(
            list({re.findall("[0-9]+", file)[-1] for file in all_files})
        )

        if len(all_event_ids) == 0:
            warnings.warn(f"No events found in {self.input_dir}/{self.data_name}")

        if self.num_events is not None:
            assert self.num_events <= len(
                all_event_ids
            ), f"Requested {self.num_events} events, but only found {len(all_event_ids)} in {self.input_dir}/{self.data_name}"
            all_event_ids = all_event_ids[: self.num_events]

        # Check that events are present for the requested filetypes
        prefix = self.event_prefix + "_" if self.event_prefix else ""

        csv_event_ids = []
        if self.use_csv:
            csv_event_ids = [
                evt_id
                for evt_id in all_event_ids
                if (f"{prefix}event{evt_id}-truth.csv" in all_files)
                and (f"{prefix}event{evt_id}-particles.csv" in all_files)
            ]

        pyg_event_ids = [
            evt_id
            for evt_id in all_event_ids
            if f"{prefix}event{evt_id}-graph.pyg" in all_files
            or f"{prefix}event{evt_id}.pyg" in all_files
        ]

        if self.use_csv:
            all_event_ids = list(set(csv_event_ids) & set(pyg_event_ids))
        else:
            all_event_ids = pyg_event_ids

        return all_event_ids

    # def apply_hard_cuts(self, event, passing_hit_ids=None):
    #     """
    #     Apply hard cuts to the event. This is implemented by
    #     1. Finding which true edges are from tracks that pass the hard cut.
    #     2. Pruning the input graph to only include nodes that are connected to these edges.
    #     """

    #     if (
    #         self.hparams is not None
    #         and "hard_cuts" in self.hparams.keys()
    #         and self.hparams["hard_cuts"]
    #     ):
    #         assert isinstance(
    #             self.hparams["hard_cuts"], dict
    #         ), "Hard cuts must be a dictionary"
    #         event = handle_hard_node_cuts(
    #             event, self.hparams["hard_cuts"], passing_hit_ids
    #         )

    #     return event

    def scale_features(self, graph):
        """
        Handle feature scaling for the graph
        """

        for feature, scale in zip(self.node_features, self.node_scales):
            graph[feature] = graph[feature] / scale

    def unscale_features(self, graph):
        """
        Unscale features when doing prediction
        """
        for feature, scale in zip(self.node_features, self.node_scales):
            graph[feature] = graph[feature] * scale
