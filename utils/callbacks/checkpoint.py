import os

import lightning as L


class ModelCheckpointCallback(L.pytorch.callbacks.ModelCheckpoint):
    def __init__(
        self,
        stage_dir: str="",
        monitor: str="",
        **kwargs
    ):

        dirpath = os.path.join(stage_dir, 'artifacts')

        suffix = "-" + (os.environ.get("SLURM_JOB_ID") or "") 

        filename = "best" + suffix + "-{" + monitor + ":5f}-epoch-{epoch}"

        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            **kwargs
        )

        self.CHECKPOINT_NAME_LAST = f"last{suffix}"

    # def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
    #     super().on_validation_end(trainer, pl_module)

    # if not self._should_skip_saving_checkpoint(trainer):
    #     trainer.predict(pl_module, trainer.val_dataloaders)
