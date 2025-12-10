import os
import sys

import torch
import yaml
from torch import Tensor, nn
from torch_geometric.nn import conv

from utils.loading_utils import add_variable_name_prefix_in_config
from utils.point_utils import gather_neighbor

from ...graph_builder.double_metric_learning import DoubleMetricLearning
from ..base import BasePointClassifier
from .modules import DilatedResidualBlock, linear_mlp, make_mlp
from .RandLANet import RandLANet


class RandLANetML(RandLANet):
    def __init__(
        self,
        feature_dim: int,
        position_dim: int,
        d_in: int,
        encoding_blocks: list[int],
        n_layers: int,
        fc_blocks: list[int],
        emb_hidden: int,
        emb_dim: int,
        emb_layers: int,
        emb_activation: str,
        r_train: float,
        knn_train: int,
        r_val: float,
        knn_val: int,
        margin: float,
        randomisation: float,
        ml_points_per_batch: int,
        ml_weight: float,
        num_classes: int,
        batch_size: int,
        do_compile: bool = False,
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            feature_dim,
            position_dim,
            d_in,
            encoding_blocks,
            n_layers,
            fc_blocks,
            num_classes,
            batch_size,
            do_compile,
            *args,
            **kwargs
        )

        self.ml_head = DoubleMetricLearning(
            in_channels=feature_dim + fc_blocks[-1],
            emb_hidden=emb_hidden,
            emb_dim=emb_dim,
            nb_layers=emb_layers,
            activation=emb_activation,
            r_train=r_train,
            knn_train=knn_train,
            r_val=r_val,
            knn_val=knn_val,
            margin=margin,
            points_per_batch=ml_points_per_batch,
            randomisation=randomisation
        )

        self.save_hyperparameters()

    def fused_forward(
        self,
        features: Tensor,
        global_xyz: list[Tensor],
        neigh_idxs: list[Tensor],
        sub_idxs: list[Tensor],
        interp_idxs: list[Tensor],
        return_fc_features: bool = False,
    ):
        point_cls_output, hidden_rep = super().forward(
            features, global_xyz, neigh_idxs, sub_idxs, interp_idxs, True
        )

        emb_input = torch.cat([features, hidden_rep], dim=-1)

        emb_src, emb_tgt = self.ml_head(emb_input)

        if return_fc_features:
            return point_cls_output, hidden_rep, emb_src, emb_tgt
        return point_cls_output, emb_src, emb_tgt

    def get_embedding(self, in_features, encoded_features):

        x_in = torch.cat([in_features, encoded_features], dim=-1)

        return self.ml_head(x_in)

    def on_train_start(self) -> None:
        self.ml_head.to(self.device)
    
    def on_validation_start(self) -> None:
        self.ml_head.to(self.device)

    def training_step(self, batch, batch_idx):

        event_batch, event_idxs, events = batch

        cls_output, hit_labels, hit_weights = self.common_evaluation(event_batch, return_fc_features=True)
        
        logits, fc_features = cls_output

        cls_loss = torch.mean(
            self.loss_fn(logits, hit_labels.to(logits.dtype)) * hit_weights
        )

        # the 6-th tensor in event_batch is the input features
        ml_in = torch.concat([event_batch[5], fc_features], dim=-1)

        ml_loss = self.ml_head.training_step(ml_in.squeeze(0), events[0])

        tot_loss = cls_loss + ml_loss * self.hparams["ml_weight"]

        return {"loss": tot_loss, "cls_loss": cls_loss, "ml_loss": ml_loss, "output": logits, "hit_labels": hit_labels}

    def on_train_batch_end(
        self, outputs: dict, batch, batch_idx: int
    ) -> None:

        super().on_train_batch_end(outputs, batch, batch_idx)

        self.log_val("train_cls_loss", outputs["cls_loss"], batch_size=self.batch_size)
        self.log_val("train_ml_loss", outputs["ml_loss"], batch_size=self.batch_size)
    
    def validation_step(self, batch, batch_idx):

        event_batch, event_idxs, events = batch

        cls_output, hit_labels, hit_weights = self.common_evaluation(event_batch, return_fc_features=True)
        
        logits, fc_features = cls_output

        cls_loss = torch.mean(
            self.loss_fn(logits, hit_labels.to(logits.dtype)) * hit_weights
        )

        # the 6-th tensor in event_batch is the input features
        ml_in = torch.concat([event_batch[5], fc_features], dim=-1)

        ml_eval_result = self.ml_head.shared_evaluation(ml_in.squeeze(0), events[0], self.hparams["r_val"], self.hparams['knn_val'])
        ml_loss = ml_eval_result["loss"]

        tot_loss = cls_loss + ml_loss * self.hparams["ml_weight"]

        return {"loss": tot_loss, "cls_loss": cls_loss, "ml_loss": ml_loss, "output": logits, "hit_labels": hit_labels, "metrics": ml_eval_result["metrics"]}

    def on_validation_batch_end(
        self, outputs: dict, batch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:

        super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)

        self.log_dict_metrics(outputs['metrics'])
    
    

        
    


