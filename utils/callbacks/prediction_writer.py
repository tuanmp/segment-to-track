import os
import shutil

import lightning as L
import torch
from genericpath import isfile


class PredictionWriter(L.pytorch.callbacks.BasePredictionWriter):

    def __init__(self, save_dir=None):

        super().__init__()

        self.save_dir = save_dir

    
    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str):
        
        super().setup(trainer, pl_module, stage)

        if self.save_dir is None:
            self.save_dir = os.path.join(trainer.default_root_dir, "predictions")
        if stage == "predict":
            if trainer.is_global_zero:
                os.makedirs(self.save_dir, exist_ok=True)
                shutil.rmtree(self.save_dir)
                os.makedirs(self.save_dir)

    def write_on_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        
        # by default, when full event data is used, we always have batch size = 1
        # this is what we assume here
        
        batch_data, event_idxs, events = batch 

        event_idx = event_idxs[0]  # get the event idx
        event = events[0]  # get the event data
        
        hit_score, hit_labels = prediction["output"], prediction["hit_labels"]
        fc_features = prediction['fc_features'].cpu()
        
        event["hit_score"] = hit_score.cpu()
        event["hit_label"] = hit_labels.cpu()
        for i in range(fc_features.shape[1]):
            event[f"hidden_feature_{i}"] = fc_features[:, i]

        match dataloader_idx:
            case 0:
                datatype = "trainset"
            case 1:
                datatype = "valset"
            case _:
                datatype = "testset"

        self.save_event(event, event_idx, datatype)
    
    def save_event(self, event, event_idx: str, datatype: str):
        
        save_path = os.path.join(self.save_dir, datatype, f"event{event_idx}.pyg")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        torch.save(event.cpu(), save_path)
    
    # def on_predict_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        
    #     # clear the save_dir first
    #     super().on_predict_start(trainer, pl_module)

    #     if trainer.is_global_zero:
    #         for name in os.listdir(self.save_dir):
    #             path = os.path.join(self.save_dir, name)

    #             if os.path.isfile(path):
    #                 os.remove(path)
    #             if os.path.isdir(path):
    #                 shutil.rmtree(path)
    #     return
    
    






