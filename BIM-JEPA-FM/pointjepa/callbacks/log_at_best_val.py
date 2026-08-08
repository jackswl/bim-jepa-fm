from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import Callback

class TrackLinearAccAtMinLossCallback(Callback):
    def __init__(self, monitor_metric_name):
        super().__init__()
        self.best_val_loss = float('inf')
        self.monitor_metric_name = monitor_metric_name

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        val_loss_tensor = trainer.callback_metrics.get('val_loss')

        if val_loss_tensor is None:
            return

        current_val_loss = val_loss_tensor.item()

        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            pl_module.log("best_val_loss", self.best_val_loss, on_step=False, on_epoch=True)

            svm_acc_tensor = trainer.callback_metrics.get(self.monitor_metric_name)

            if svm_acc_tensor is not None:
                best_metric_name = f"best_{self.monitor_metric_name}"
                svm_accuracy = svm_acc_tensor.item()
                pl_module.log(best_metric_name, svm_accuracy, on_step=False, on_epoch=True, sync_dist=True)
