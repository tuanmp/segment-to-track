import os
from datetime import timedelta
from typing import Literal

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
