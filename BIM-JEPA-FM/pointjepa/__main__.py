import torch
from pytorch_lightning.cli import LightningCLI

from pointjepa.callbacks.log_at_best_val import TrackLinearAccAtMinLossCallback
from pointjepa.callbacks.wandb_checkpoint_logger import WandbModelCheckpointLogger
from pointjepa.models import PointLeJepa

torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    cli = LightningCLI(
        PointLeJepa,
        seed_everything_default=1,
        save_config_callback=None,  # https://github.com/Lightning-AI/lightning/issues/12028#issuecomment-1088325894
    )
