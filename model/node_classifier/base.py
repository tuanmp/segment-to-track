import logging
import os

import lightning as L
import matplotlib.pyplot as plt
import scienceplots
import torch
import torchmetrics
import torchmetrics.classification
from torch import nn

plt.style.use(["science", "ieee"])

# from torch.utils.data import DataLoader

# from ...data import point_classifier_dataset

# class BasePointClassifier(L.LightningModule):

#     def __init__(self, hparams, *args: torch.Any, **kwargs: torch.Any) -> None:
#         super().__init__(*args, **kwargs)

#         self.save_hyperparameters(hparams)
#         self.trainset, self.valset, self.testset = None, None, None

#         self.dataset_class = getattr(point_classifier_dataset, self.hparams.get("dataset_class", "PointCloudDataset"))

#         input_features = self.hparams["node_features"]
#         self.position_features = self.hparams["position_features"]

#         self.input_features = [f for f in input_features if f not in self.position_features]

#         kwargs = {
#             "average": 'none',
#             "task": 'binary' if self.hparams['num_class']==2 else 'multiclass'
#         }
#         if self.hparams['num_class'] > 2:
#             kwargs['num_classes'] = self.hparams['num_class']

#         self.metrics = torchmetrics.MetricCollection(
#             {
#                 "accuracy": torchmetrics.classification.Accuracy(**kwargs),
#                 "recall": torchmetrics.classification.Recall(**kwargs),
#                 'f1': torchmetrics.classification.F1Score(**kwargs),
#                 'auroc': torchmetrics.classification.AUROC(**kwargs),
#             },
#             prefix='valid_',
#         )

# def setup(self, stage="fit"):
#     """
#     The setup logic of the stage.
#     1. Setup the data for training, validation and testing.
#     2. Run tests to ensure data is of the right format and loaded correctly.
#     3. Construct the truth and weighting labels for the model training
#     """

#     if stage in ["fit", "predict"]:
#         self.load_data(self.hparams["input_dir"], subsampling="random" if stage=="fit" else "full")

#     elif stage == "test":
#         self.load_data(self.hparams["stage_dir"], subsampling='full')

# def load_data(self, input_dir, subsampling='random'):
#     """
#     Load in the data for training, validation and testing.
#     """
#     input_features = self.hparams["node_features"]
#     position_features = self.hparams["position_features"]

#     input_features = [f for f in input_features if f not in position_features]

#     for data_name, data_num in zip(["trainset", "valset", "testset"], self.hparams["data_split"]):
#         dataset = self.dataset_class(
#             input_dir,
#             self.input_features,
#             self.position_features,
#             data_name,
#             data_num,
#             hparams=self.hparams,
#             subsampling=subsampling
#         )
#         setattr(self, data_name, dataset)

#     logging.info(
#         f"Loaded {len(self.trainset)} training events,"
#         f" {len(self.valset)} validation events and {len(self.testset)} testing"
#         " events"
#     )

# def dataloader(self, dataset: str, shuffle=False):
#     return (
#         DataLoader(
#             getattr(self, dataset),
#             batch_size=self.hparams["batch_size"],
#             num_workers=self.hparams.get("num_workers", 16),
#             drop_last=True,
#             shuffle=shuffle,
#         )
#         if getattr(self, dataset) is not None
#         else None
#     )

# def train_dataloader(self):
#     return self.dataloader("trainset", shuffle=True)

# def val_dataloader(self):
#     return self.dataloader("valset")

# def test_dataloader(self):
#     return self.dataloader("testset")

# def predict_dataloader(self):
#     return [
#         self.train_dataloader(),
#         self.val_dataloader(),
#         self.test_dataloader(),
#     ]

# def configure_optimizers(self):
#     optimizer = [
#         torch.optim.AdamW(
#             self.parameters(),
#             lr=(self.hparams["lr"]),
#             betas=(0.9, 0.999),
#             eps=1e-08,
#             amsgrad=True,
#         )
#     ]
#     scheduler = [
#         {
#             "scheduler": torch.optim.lr_scheduler.StepLR(
#                 optimizer[0],
#                 step_size=self.hparams["patience"],
#                 gamma=self.hparams["factor"],
#             ),
#             "interval": self.hparams["interval"],
#             "frequency": self.hparams["frequency"],
#         }
#     ]
#     return optimizer, scheduler

# def log_val(self, name, val, on_epoch=True, on_step=False, batch_size=1, prog_bar=False):
#     self.log(name, val, sync_dist=True, on_epoch=on_epoch, on_step=on_step, batch_size=batch_size, prog_bar=prog_bar)

# def log_dict_metrics(self, d, on_epoch=True, on_step=False, batch_size=1, prog_bar=False):
#     self.log_dict(d, sync_dist=True, on_epoch=on_epoch, on_step=on_step, batch_size=batch_size, prog_bar=prog_bar)

# def on_validation_epoch_end(self) -> None:
#     metrics = self.metrics.compute()
#     if self.hparams["num_class"] == 2:
#         [self.log_val(metric, val, on_step=False, on_epoch=True) for metric, val in metrics.items() ]
#     else:
#         for metric, val in metrics.items():
#             [self.log_val(f"{metric}_{i}", val[i], on_step=False, on_epoch=True) for i in range(len(val))]
#         self.log_dict_metrics({m : torch.mean(v) for m,v in metrics.items()}, on_epoch=True, on_step=False, batch_size=self.hparams['batch_size'])
#     self.metrics.reset()

# @staticmethod
# def compile_module_list(module_list):
#     return nn.ModuleList([
#         torch.compile(_, fullgraph=True) for _ in module_list
#     ])


class BasePointClassifier(L.LightningModule):

    def __init__(self, num_classes, batch_size, *args: torch.Any, **kwargs: torch.Any) -> None:
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
                    "roc": torchmetrics.classification.ROC(
                        task=kwargs["task"], thresholds=100
                    ),
                    "precision_recall": torchmetrics.classification.PrecisionRecallCurve(
                        task=kwargs["task"], thresholds=100
                    ),
                    "precision_@99pct_recall": torchmetrics.classification.PrecisionAtFixedRecall(
                        task=kwargs["task"], min_recall=0.99
                    ),
                }
            )
        super().setup(stage)

    def log_val(self, name, val, on_epoch=True, on_step=False, batch_size=1, prog_bar=False):
        self.log(name, val, sync_dist=True, on_epoch=on_epoch, on_step=on_step, batch_size=batch_size, prog_bar=prog_bar)

    def log_dict_metrics(self, d, on_epoch=True, on_step=False, batch_size=1, prog_bar=False):
        self.log_dict(d, sync_dist=True, on_epoch=on_epoch, on_step=on_step, batch_size=batch_size, prog_bar=prog_bar)

    def on_validation_epoch_end(self) -> None:
        metrics = self.metrics.compute()
        if self.num_classes == 2:
            [self.log_val(metric, val, on_step=False, on_epoch=True) for metric, val in metrics.items() ]
        else:
            for metric, val in metrics.items():
                [self.log_val(f"{metric}_{i}", val[i], on_step=False, on_epoch=True) for i in range(len(val))]
            self.log_dict_metrics({m : torch.mean(v) for m,v in metrics.items()}, on_epoch=True, on_step=False, batch_size=self.batch_size)
        self.metrics.reset()

    def on_test_epoch_end(self) -> None:

        print("Generating test ROC and Precision-Recall curves...")

        metrics = self.metrics.compute()

        figsize = (6, 5)
        save_dir = "./"
        if self.trainer is not None:
            save_dir = self.trainer.default_root_dir

        print("Plotting ROC curve ...")
        # plot ROC curve
        fpr, tpr, threshold = metrics['roc']
        fig, ax = plt.subplots(figsize=figsize)
        # self.metrics["roc"].plot(score=True, ax=ax)
        ax.plot(fpr.cpu().numpy(), tpr.cpu().numpy(), label=f'ROC curve (AUC = {metrics["auroc"].cpu().item():.2f})')
        ax.plot([0, 1], [0, 1], "k--", label="Random")
        ax.grid(True, linestyle="--", linewidth=1)
        ax.legend()
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        plt.tight_layout()
        save_path = os.path.join(save_dir, "roc_curve.png")
        plt.savefig(save_path)
        print(f"ROC curve saved to: {save_path}")
        plt.close()

        # plot precision-recall curve
        print("Plotting Precision-Recall curve ...")
        precision, recall, thresholds = metrics['precision_recall']
        fig, ax = plt.subplots(figsize=figsize)
        # self.metrics["precision_recall"].plot(ax=ax)
        ax.plot(recall.cpu().numpy(), precision.cpu().numpy(), label="Precision-Recall curve")
        ax.plot([0.99], [metrics['precision_@99pct_recall'][0].cpu().item()], marker='o', markersize=5, label=f'Precision @99pct Recall: {metrics["precision_@99pct_recall"][0].cpu():.2f}\n Score threshold: {metrics["precision_@99pct_recall"][1].cpu():.2f}')
        ax.grid(True, linestyle="--", linewidth=1)
        ax.legend()
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        plt.tight_layout()
        save_path = os.path.join(save_dir, "precision_recall.png")
        plt.savefig(save_path)
        print(f"Precision-Recall curve saved to: {save_path}")
        plt.close()

        self.metrics.reset()
    
    def save_event(self, event, event_idx: str, datatype: str):
        
        save_path = os.path.join(self.save_dir, datatype, f"event_{event_idx}.pyg")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        torch.save(event.cpu(), save_path)

    @staticmethod
    def compile_module_list(module_list):
        return nn.ModuleList([
            torch.compile(_, fullgraph=True) for _ in module_list
        ])
