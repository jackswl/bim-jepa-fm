import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.cli import LightningCLI

from pointjepa.datasets.ifcnet_datamodule import IFCNetCoreDataModule  # allow shorthand notation
from pointjepa.models import PointJepaClassification


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    cli = LightningCLI(
        PointJepaClassification,
        trainer_defaults={
            "default_root_dir": "artifacts",
            "accelerator": "gpu",
            "devices": 1,
            "callbacks": [
                LearningRateMonitor(logging_interval="epoch"),
            ],
        },
        seed_everything_default=42,
        save_config_callback=None,  # https://github.com/Lightning-AI/lightning/issues/12028#issuecomment-1088325894
    )
