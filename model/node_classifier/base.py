import logging
import os

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import torch
import torchmetrics
import torchmetrics.classification
from torch import nn

from utils.metrics.hit_metrics import HitMetrics
from utils.metrics.score_histo import ScoreHistogram


class BasePointClassifier(L.LightningModule):

    def __init__(
        self,
        num_classes: int,
        batch_size: int,
        knn_engine: str = "torch",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.num_classes = num_classes
        self.batch_size = batch_size

    def setup(self, stage: str) -> None:
        kwargs = {
            "average": "none",
            "task": "binary" if self.num_classes == 2 else "multiclass",
        }
        if self.num_classes > 2: 
            kwargs['num_classes'] = self.num_classes

        if stage in ["fit", "validate"]:
            self.metrics = torchmetrics.MetricCollection(
                {
                    "accuracy": torchmetrics.classification.Accuracy(**kwargs),
                    "recall": torchmetrics.classification.Recall(**kwargs),
                    "f1": torchmetrics.classification.F1Score(**kwargs),
                    "auroc": torchmetrics.classification.AUROC(**kwargs),
                },
                prefix="valid_",
            )
        elif stage in ["test"]:

            self.metrics = torchmetrics.MetricCollection(
                {
                    "auroc": torchmetrics.classification.AUROC(**kwargs),
                    "roc": torchmetrics.classification.ROC(task=kwargs["task"]),
                    "precision_recall": torchmetrics.classification.PrecisionRecallCurve(
                        task=kwargs["task"]
                    ),
                    "score_histogram": ScoreHistogram(density=True, common_norm=False),
                }
            )

            self.hit_metrics = HitMetrics()
        super().setup(stage)

    def log_val(self, name, val, on_epoch=True, on_step=False, batch_size=1, prog_bar=False):
        self.log(name, val, sync_dist=True, on_epoch=on_epoch, on_step=on_step, batch_size=batch_size, prog_bar=prog_bar)

    def log_dict_metrics(self, d, on_epoch=True, on_step=False, batch_size=1, prog_bar=False):
        self.log_dict(d, sync_dist=True, on_epoch=on_epoch, on_step=on_step, batch_size=batch_size, prog_bar=prog_bar)

    def on_validation_epoch_end(self) -> None:
        metrics = self.metrics.compute()
        if self.num_classes == 2:
            [
                self.log_val(metric, val, on_step=False, on_epoch=True)
                for metric, val in metrics.items()
            ]
        else:
            for metric, val in metrics.items():
                [self.log_val(f"{metric}_{i}", val[i], on_step=False, on_epoch=True) for i in range(len(val))]
            self.log_dict_metrics({m : torch.mean(v) for m,v in metrics.items()}, on_epoch=True, on_step=False, batch_size=self.batch_size)
        self.metrics.reset()

    def on_train_batch_end(self, outputs: dict, batch, batch_idx: int) -> None:

        self.log_val("train_loss", outputs["loss"], batch_size=self.batch_size)

    def on_validation_batch_end(
        self, outputs: dict, batch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:

        self.log_val("val_loss", outputs["loss"], batch_size=self.batch_size)

        self.metrics.update(outputs["output"], outputs["hit_labels"])

    def on_test_batch_end(
        self, outputs: dict, batch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:

        event_batch, event_idxs, events = batch

        event = events[0]

        random_idx = event_batch[-1].squeeze()  # get the random idxs used in dataloader
        reco_idx = torch.empty_like(random_idx)

        reco_idx[random_idx] = torch.arange(
            0, random_idx.shape[0], device=random_idx.device
        )

        preds = outputs["output"][reco_idx]
        target = outputs["hit_labels"][reco_idx]

        eta = event.hit_eta.to(preds.device)
        pt = torch.zeros_like(preds).to(preds.device)
        track_edges = event.track_edges.to(preds.device)
        particle_pt = event.track_particle_pt.to(preds.device).to(pt.dtype)

        pt[track_edges[0]] = particle_pt
        pt[track_edges[1]] = particle_pt

        self.metrics.update(1 - preds, 1 - target)
        self.hit_metrics.update(1 - preds, 1 - target, pt, eta)

    @staticmethod
    def compile_module_list(module_list):
        return nn.ModuleList([
            torch.compile(_, fullgraph=True) for _ in module_list
        ])
