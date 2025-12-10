import logging
import os
import re
import signal
import sys
from datetime import datetime
from types import FrameType
from typing import Union

import lightning as L
from lightning.pytorch.trainer.connectors.signal_connector import _SignalConnector
from lightning.pytorch.utilities.rank_zero import rank_zero_info

log = logging.getLogger(__name__)
_SIGNUM = Union[int, signal.Signals]

def get_default_root_dir(stage_dir):
    if (
        "SLURM_JOB_ID" in os.environ
        and "SLURM_JOB_QOS" in os.environ
        and "interactive" not in os.environ["SLURM_JOB_QOS"]
        and "jupyter" not in os.environ["SLURM_JOB_QOS"]
    ):
        return os.path.join(stage_dir, "root_dir", os.environ["SLURM_JOB_ID"])
    else:
        return os.path.join(
            stage_dir, "root_dir", datetime.now().strftime("%Y-%m-%d--%H-%M")
        )


class _MySignalConnector(_SignalConnector):
    """
    Only save HPC checkpoint without automatic resubmission
    """

    def _slurm_sigusr_handler_fn(self, signum: _SIGNUM, _: FrameType) -> None:
        rank_zero_info(f"Handling auto-requeue signal: {signum}")

        # save logger to make sure we get all the metrics
        for logger in self.trainer.loggers:
            logger.finalize("finished")

        hpc_save_path = self.trainer._checkpoint_connector.hpc_save_path(
            self.trainer.default_root_dir
        )
        self.trainer.save_checkpoint(hpc_save_path)

        rank_zero_info(f"Saved checkpoint {hpc_save_path}")

        os._exit(0)

        # if self.trainer.is_global_zero:
        #     # find job id
        #     array_job_id = os.getenv("SLURM_ARRAY_JOB_ID")
        #     if array_job_id is not None:
        #         array_task_id = os.environ["SLURM_ARRAY_TASK_ID"]
        #         job_id = f"{array_job_id}_{array_task_id}"
        #     else:
        #         job_id = os.environ["SLURM_JOB_ID"]

        # assert re.match("[0-9_-]+", job_id)
        # cmd = ["scontrol", "requeue", job_id]

        # requeue job
        # log.info(f"requeing job {job_id}...")
        # try:
        #     result = call(cmd)
        # except FileNotFoundError:
        #     # This can occur if a subprocess call to `scontrol` is run outside a shell context
        #     # Re-attempt call (now with shell context). If any error is raised, propagate to user.
        #     # When running a shell command, it should be passed as a single string.
        #     result = call(" ".join(cmd), shell=True)

        # # print result text
        # if result == 0:
        #     log.info(f"Requeued SLURM job: {job_id}")
        # else:
        #     log.warning(f"Requeuing SLURM job {job_id} failed with error code {result}"


class Trainer(L.Trainer):

    def __init__(
        self, stage_dir: str = "", container_mode: bool = False, **kwargs
    ) -> None:

        default_root_dir = get_default_root_dir(stage_dir)

        if kwargs.get("fast_dev_run"):
            default_root_dir = "/tmp"
        else:
            os.makedirs(default_root_dir, exist_ok=True)

        logging.info(f"Setting default root dir: {default_root_dir}")

        super().__init__(
            default_root_dir=default_root_dir,
            **kwargs
        )

        # ask pytorch lightning to save HPC checkpoint but not resubmit the job if running in container
        if container_mode:
            self._signal_connector = _MySignalConnector(self)
