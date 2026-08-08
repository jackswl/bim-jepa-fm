import pytorch_lightning as pl
import wandb
import os

class WandbModelCheckpointLogger(pl.Callback):
    def on_train_end(self, trainer, pl_module):
        if isinstance(trainer.logger, pl.loggers.WandbLogger):
            wandb_run = trainer.logger.experiment

            check_logged_any = False
            for callback in trainer.callbacks:
                if isinstance(callback, pl.callbacks.ModelCheckpoint):
                    if callback.best_model_path:
                        monitor = callback.monitor

                        model_name = f"best_{monitor}"
                        model_path = callback.best_model_path

                        artifact = wandb.Artifact(model_name, type='model',
                                                  description=f"Final model checkpoint for {monitor} at end of training")
                        artifact.add_file(model_path)

                        print(f"Logging {model_name} to wandb artifact: {artifact}.")
                        wandb_run.log_artifact(artifact)
                        check_logged_any = True

            if not check_logged_any:
                print("No checkpoints were saved during training. Logging a dummy artifact.")
