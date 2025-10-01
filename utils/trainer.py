import logging
import os

import lightning as L


def get_default_root_dir(stage_dir):
    if (
        "SLURM_JOB_ID" in os.environ
        and "SLURM_JOB_QOS" in os.environ
        and "interactive" not in os.environ["SLURM_JOB_QOS"]
        and "jupyter" not in os.environ["SLURM_JOB_QOS"]
    ):
        return os.path.join(stage_dir, "root_dir", os.environ["SLURM_JOB_ID"])
    else:
        return None

class Trainer(L.Trainer):
    def __init__(
        self,
        stage_dir: str = "",
        **kwargs
    ) -> None:

        default_root_dir = get_default_root_dir(stage_dir)

        logging.info(f"Setting default root dir: {default_root_dir}")

        super().__init__(
            default_root_dir=default_root_dir,
            **kwargs
        )

    
    
    
    
