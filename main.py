from lightning.pytorch.cli import LightningCLI

from model.node_classifier.RandLANet.RandLANet import RandLANet
from utils.callbacks.save_config import SaveConfigCallback
from utils.trainer import Trainer


def cli_main():

    cli = LightningCLI(
        RandLANet,
        trainer_class=Trainer,
        save_config_kwargs={"overwrite": True, "save_to_log_dir": False},
        save_config_callback=SaveConfigCallback,
    )

if __name__=='__main__':
    cli_main()
