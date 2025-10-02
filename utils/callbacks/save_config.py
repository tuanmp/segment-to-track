import os

import lightning as L
from lightning import LightningModule, Trainer
from lightning.pytorch.loggers import Logger


class SaveConfigCallback(L.pytorch.cli.SaveConfigCallback):

    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if isinstance(trainer.logger, Logger):
            config = self.parser.dump(self.config, skip_none=False)  # Required for proper reproducibility
            trainer.logger.log_hyperparams({"config": config})
        
        config_path = os.path.join(trainer.default_root_dir, self.config_filename)

        self.parser.save(
            self.config, config_path, skip_none=False, overwrite=self.overwrite, multifile=self.multifile
        )