import json
from pathlib import Path
from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy, Precision, Recall, F1Score
from torchmetrics.functional import confusion_matrix

from pointjepa.modules.tokenizer import PointcloudTokenizer
from pointjepa.modules.transformer import TransformerEncoder
from pointjepa.utils import transforms
from pointjepa.utils.checkpoint import extract_model_checkpoint


class PointJepaClassification(pl.LightningModule):
    def __init__(
        self,
        num_points: int = 4096, 
        tokenizer_num_groups: int = 64,
        tokenizer_group_size: int = 32,
        tokenizer_group_radius: float | None = None,
        tokenizer_unfreeze_epoch: int = 0, # ignored; the tokenizer unfreezes with the encoder
        positional_encoding_unfreeze_epoch: int = 0,
        encoder_dim: int = 384,
        encoder_depth: int = 12,
        encoder_heads: int = 6,
        encoder_dropout: float = 0,
        encoder_attention_dropout: float = 0,
        encoder_drop_path_rate: float = 0.2,
        encoder_add_pos_at_every_layer: bool = True,
        encoder_qkv_bias: bool = True,
        encoder_freeze_layers: Optional[List[int]] = None,
        encoder_unfreeze_epoch: int = 0,
        encoder_unfreeze_layers: Optional[List[int]] = None,
        encoder_unfreeze_stepwise: bool = False,
        encoder_unfreeze_stepwise_num_layers: int = 2,
        encoder_learning_rate: Optional[float] = None,
        cls_head: str = "mlp", 
        cls_head_dim: int = 256,
        cls_head_dropout: float = 0.5,
        cls_head_pooling: str = "mean+max", 
        loss_label_smoothing: float = 0.2,
        learning_rate: float = 0.001,
        optimizer_adamw_weight_decay: float = 0.05,
        lr_scheduler_linear_warmup_epochs: int = 10,
        lr_scheduler_linear_warmup_start_lr: float = 1e-6,
        lr_scheduler_cosine_eta_min: float = 1e-6,
        pretrained_ckpt_path: str | None = None,
        pretrained_ckpt_ignore_encoder_layers: List[int] = [],
        train_transformations: List[str] = ["center", "unit_sphere"], 
        val_transformations: List[str] = ["center", "unit_sphere"],
        transformation_scale_min: float = 0.8,
        transformation_scale_max: float = 1.2,
        transformation_scale_symmetries: Tuple[int, int, int] = (1, 0, 1),
        transformation_rotate_dims: List[int] = [1],
        transformation_rotate_degs: Optional[int] = None,
        transformation_translate: float = 0.2,
        transformation_height_normalize_dim: int = 1,
        transformation_jitter_sigma: float = 0.01, 
        transformation_jitter_clip: float = 0.05,
        log_tsne: bool = False,
        log_confusion_matrix: bool = False,
        vote: bool = False,
        vote_num: int = 10,
        num_classes: Optional[int] = None, # supports load_from_checkpoint
        metrics_output_path: str | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.train_acc = None
        self.val_acc = None
        self.vote = vote
        self.val_top_3_acc = None

        self.num_classes = num_classes

        def build_transformation(name: str) -> transforms.Transform:
            if name == "scale":
                return transforms.PointcloudScaling(
                    min=transformation_scale_min, max=transformation_scale_max
                )
            elif name == "center":
                return transforms.PointcloudCentering()
            elif name == "unit_sphere":
                return transforms.PointcloudUnitSphere()
            elif name == "rotate":
                return transforms.PointcloudRotation(
                    dims=transformation_rotate_dims, deg=transformation_rotate_degs
                )
            elif name == "translate":
                return transforms.PointcloudTranslation(transformation_translate)
            elif name == "jitter":
                return transforms.PointcloudJitter(
                    sigma=transformation_jitter_sigma,
                    clip=transformation_jitter_clip,
                )
            elif name == "height_norm":
                return transforms.PointcloudHeightNormalization(
                    transformation_height_normalize_dim
                )
            else:
                raise RuntimeError(f"No such transformation: {name}")

        self.train_transformations = transforms.Compose(
            [transforms.PointcloudSubsampling(num_points)]
            + [build_transformation(name) for name in train_transformations]
        )
        self.val_transformations = transforms.Compose(
            [transforms.PointcloudSubsampling(num_points)]
            + [build_transformation(name) for name in val_transformations]
        )

        # The tokenizer holds the learnable PointNet weights
        self.tokenizer = PointcloudTokenizer(
            num_groups=tokenizer_num_groups,
            group_size=tokenizer_group_size,
            group_radius=tokenizer_group_radius,
            token_dim=encoder_dim,
        )

        self.positional_encoding = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, encoder_dim),
        )

        dpr = [
            x.item() for x in torch.linspace(0, encoder_drop_path_rate, encoder_depth)
        ]
        self.encoder = TransformerEncoder(
            embed_dim=encoder_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            qkv_bias=encoder_qkv_bias,
            drop_rate=encoder_dropout,
            attn_drop_rate=encoder_attention_dropout,
            drop_path_rate=dpr,
            add_pos_at_every_layer=encoder_add_pos_at_every_layer,
        )

        match cls_head_pooling:
            case "cls_token" | "mean+max+cls_token":
                init_std = 0.02
                self.cls_token = nn.Parameter(torch.zeros(encoder_dim))
                nn.init.trunc_normal_(
                    self.cls_token, mean=0, std=init_std, a=-init_std, b=init_std
                )
                self.cls_pos = nn.Parameter(torch.zeros(encoder_dim))
                nn.init.trunc_normal_(
                    self.cls_pos, mean=0, std=init_std, a=-init_std, b=init_std
                )
                self.first_cls_head_dim = (
                    encoder_dim if cls_head_pooling == "cls_token" else 3 * encoder_dim
                )
            case "mean+max":
                self.first_cls_head_dim = 2 * encoder_dim
            case "mean":
                self.first_cls_head_dim = encoder_dim
            case "max":
                self.first_cls_head_dim = encoder_dim
            case _:
                raise ValueError()

        self.loss_func = nn.CrossEntropyLoss(label_smoothing=loss_label_smoothing)

        # num_classes is known ahead of setup() only via load_from_checkpoint
        if self.num_classes is not None:
             self._build_cls_head(self.num_classes)

    def _build_cls_head(self, num_classes: int):
        if hasattr(self, 'cls_head'):
            return 
            
        if self.hparams.cls_head == "mlp":
            self.cls_head = nn.Sequential(
                nn.Linear(
                    self.first_cls_head_dim, self.hparams.cls_head_dim * 2, bias=False
                ),
                nn.BatchNorm1d(self.hparams.cls_head_dim * 2),
                nn.ReLU(),
                nn.Dropout(self.hparams.cls_head_dropout),
                nn.Linear(self.hparams.cls_head_dim * 2, self.hparams.cls_head_dim, bias=False),
                nn.BatchNorm1d(self.hparams.cls_head_dim), 
                nn.ReLU(),
                nn.Dropout(self.hparams.cls_head_dropout),
                nn.Linear(self.hparams.cls_head_dim, num_classes),
            )
        elif self.hparams.cls_head == "linear":
            self.cls_head = nn.Linear(self.first_cls_head_dim, num_classes)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.num_classes is None:
            self.num_classes = self.trainer.datamodule.num_classes
        
        num_classes = self.num_classes

        self._build_cls_head(num_classes)
        
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        if self.vote:
            self.val_vote_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_top_3_acc = Accuracy(
            top_k=3, task="multiclass", num_classes=num_classes)

        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_macc = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
        self.test_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.test_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        # Weighted variants for class-imbalance reporting. Weighted recall equals
        # overall accuracy; it is kept as a sanity check.
        self.test_precision_w = Precision(task="multiclass", num_classes=num_classes, average="weighted")
        self.test_recall_w = Recall(task="multiclass", num_classes=num_classes, average="weighted")
        self.test_f1_w = F1Score(task="multiclass", num_classes=num_classes, average="weighted")
        self.test_per_class_acc = Accuracy(task="multiclass", num_classes=num_classes, average=None)
        self.test_preds = [] 
        self.test_labels = [] 
        self.test_step_times = [] 

        self.val_macc = Accuracy(num_classes=num_classes, average="macro", task="multiclass")
        
        if self.hparams.pretrained_ckpt_path is not None:
             self.load_pretrained_checkpoint(self.hparams.pretrained_ckpt_path)

        if self.hparams.encoder_freeze_layers is not None:
            assert self.hparams.encoder_unfreeze_stepwise == False
            assert isinstance(self.encoder, TransformerEncoder)
            for i in self.hparams.encoder_freeze_layers:
                self.encoder.blocks[i].requires_grad_(False)
        else:
            # Encoder and tokenizer are frozen and unfrozen together
            self.encoder.requires_grad_(False)
            self.tokenizer.requires_grad_(False)

        self.positional_encoding.requires_grad_(False)

        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                logger.watch(self)
                logger.experiment.define_metric("val_acc", summary="last,max")
                logger.experiment.define_metric("val_top_3_acc", summary="last,max")
                logger.experiment.define_metric("val_macc", summary="last,max")

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # points: (B, N, 3)
        tokens: torch.Tensor  # (B, T, C)
        centers: torch.Tensor  # (B, T, 3)
        tokens, centers = self.tokenizer(points)
        pos_embeddings = self.positional_encoding(centers)
        if self.hparams.cls_head_pooling in ["cls_token", "mean+max+cls_token"]:  # type: ignore
            B, T, C = tokens.shape
            tokens = torch.cat(
                [self.cls_token.reshape(1, 1, C).expand(B, -1, -1), tokens], dim=1
            )
            pos_embeddings = torch.cat(
                [self.cls_pos.reshape(1, 1, C).expand(B, -1, -1), pos_embeddings], dim=1
            )
        tokens = self.encoder(tokens, pos_embeddings).last_hidden_state
        match self.hparams.cls_head_pooling:  # type: ignore
            case "cls_token":
                embedding = tokens[:, 0]
            case "mean+max":
                max_features = torch.max(tokens, dim=1).values
                mean_features = torch.mean(tokens, dim=1)
                embedding = torch.cat([max_features, mean_features], dim=-1)
            case "mean+max+cls_token":
                cls_token = tokens[:, 0]
                max_features = torch.max(tokens[:, 1:], dim=1).values
                mean_features = torch.mean(tokens[:, 1:], dim=1)
                embedding = torch.cat([cls_token, max_features, mean_features], dim=-1)
            case "mean":
                embedding = torch.mean(tokens, dim=1)
            case "max":
                embedding = torch.max(tokens, dim=1).values
            case _:
                raise ValueError(f"Unknown cls_head_pooling: {self.hparams.cls_head_pooling}")  # type: ignore
        logits = (
            self.cls_head(embedding)  # type: ignore
            if isinstance(tokens, torch.Tensor)
            else self.cls_head(embedding.F)  # type: ignore
        )
        return logits

    def extract_features(self, points: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the penultimate layer of the MLP head (if available).
        Returns the 256-dim embedding (or cls_head_dim) used for retrieval.
        """
        tokens, centers = self.tokenizer(points)
        pos_embeddings = self.positional_encoding(centers)
        if self.hparams.cls_head_pooling in ["cls_token", "mean+max+cls_token"]:  # type: ignore
            B, T, C = tokens.shape
            tokens = torch.cat(
                [self.cls_token.reshape(1, 1, C).expand(B, -1, -1), tokens], dim=1
            )
            pos_embeddings = torch.cat(
                [self.cls_pos.reshape(1, 1, C).expand(B, -1, -1), pos_embeddings], dim=1
            )
        tokens = self.encoder(tokens, pos_embeddings).last_hidden_state
        
        match self.hparams.cls_head_pooling:  # type: ignore
            case "cls_token":
                embedding = tokens[:, 0]
            case "mean+max":
                max_features = torch.max(tokens, dim=1).values
                mean_features = torch.mean(tokens, dim=1)
                embedding = torch.cat([max_features, mean_features], dim=-1)
            case "mean+max+cls_token":
                cls_token = tokens[:, 0]
                max_features = torch.max(tokens[:, 1:], dim=1).values
                mean_features = torch.mean(tokens[:, 1:], dim=1)
                embedding = torch.cat([cls_token, max_features, mean_features], dim=-1)
            case "mean":
                embedding = torch.mean(tokens, dim=1)
            case "max":
                embedding = torch.max(tokens, dim=1).values
            case _:
                raise ValueError(f"Unknown cls_head_pooling: {self.hparams.cls_head_pooling}")  # type: ignore

        # For the MLP head, drop the final Linear so this returns the penultimate embedding
        if self.hparams.cls_head == "mlp" and isinstance(self.cls_head, nn.Sequential):
            return self.cls_head[:-1](embedding)
        else:
            # Linear or unknown head: fall back to the pooled encoder features
            return embedding

    def extract_patch_features(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract token-level features (patches) from the encoder.
        Returns:
            features: (B, T, D) - Contextualized token features from the last encoder layer.
            centers: (B, T, 3) - The 3D coordinates (centroids) of each token.
        """
        tokens, centers = self.tokenizer(points)
        pos_embeddings = self.positional_encoding(centers)

        if self.hparams.cls_head_pooling in ["cls_token", "mean+max+cls_token"]:  # type: ignore
            B, T, C = tokens.shape
            tokens_input = torch.cat(
                [self.cls_token.reshape(1, 1, C).expand(B, -1, -1), tokens], dim=1
            )
            pos_embeddings_input = torch.cat(
                [self.cls_pos.reshape(1, 1, C).expand(B, -1, -1), pos_embeddings], dim=1
            )
        else:
            tokens_input = tokens
            pos_embeddings_input = pos_embeddings

        encoded_tokens = self.encoder(tokens_input, pos_embeddings_input).last_hidden_state

        # Drop the CLS token so the sequence realigns with 'centers'
        if self.hparams.cls_head_pooling in ["cls_token", "mean+max+cls_token"]:
             encoded_tokens = encoded_tokens[:, 1:, :]

        return encoded_tokens, centers

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        points, label = batch
        points = self.train_transformations(points)
        logits = self.forward(points)

        loss = self.loss_func(logits, label)
        self.log("train_loss", loss, on_epoch=True)

        pred = torch.max(logits, dim=-1).indices
        self.train_acc(pred, label)
        self.log("train_acc", self.train_acc, on_epoch=True)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        points, label = batch
        points = self.val_transformations(points)
        logits = self.forward(points)

        loss = self.loss_func(logits, label)
        self.log("val_loss", loss)

        pred = torch.max(logits, dim=-1).indices
        self.val_acc(pred, label)
        self.log("val_acc", self.val_acc)
        self.val_top_3_acc(logits, label)
        self.log("val_top_3_acc", self.val_top_3_acc)
        self.val_macc(pred, label)
        self.log("val_macc", self.val_macc)

        if self.hparams.vote:  # type: ignore
            logits_list = []
            for _ in range(self.hparams.vote_num):  # type: ignore
                points = self.val_transformations(points)
                logits = self.forward(points)
                logits_list.append(F.softmax(logits, dim=-1))
            mean_logits = torch.mean(torch.stack(logits_list), dim=0)
            vote_pred = torch.max(mean_logits, dim=-1).indices

            self.val_vote_acc(vote_pred, label)
            self.log("val_vote_acc", self.val_vote_acc)

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        points, label = batch
        points = self.val_transformations(points)

        # Per-batch timing, aggregated into inference speed in on_test_epoch_end
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        
        logits = self.forward(points)
        
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        self.test_step_times.append(elapsed_ms / 1000.0)

        pred = torch.max(logits, dim=-1).indices

        self.test_acc.update(pred, label)
        self.test_macc.update(pred, label)
        self.test_precision.update(pred, label)
        self.test_recall.update(pred, label)
        self.test_f1.update(pred, label)
        self.test_precision_w.update(pred, label)
        self.test_recall_w.update(pred, label)
        self.test_f1_w.update(pred, label)
        self.test_per_class_acc.update(pred, label)
        self.test_preds.append(pred.cpu())
        self.test_labels.append(label.cpu())

    def on_test_epoch_end(self) -> None:
        overall_acc = self.test_acc.compute()
        mean_class_acc = self.test_macc.compute()
        precision = self.test_precision.compute()
        recall = self.test_recall.compute()
        f1_score = self.test_f1.compute()
        precision_w = self.test_precision_w.compute()
        recall_w = self.test_recall_w.compute()
        f1_w = self.test_f1_w.compute()
        per_class_acc = self.test_per_class_acc.compute()

        print("\n" + "="*40)
        print(" " * 12 + "TEST RESULTS")
        print("="*40)
        print(f"Overall Accuracy:         {overall_acc.item():.4f}")
        print(f"Mean Class Accuracy:      {mean_class_acc.item():.4f}")
        print(f"Precision (Macro):        {precision.item():.4f}")
        print(f"Recall (Macro):           {recall.item():.4f}")
        print(f"F1 Score (Macro):         {f1_score.item():.4f}")
        print(f"Precision (Weighted):     {precision_w.item():.4f}")
        print(f"Recall (Weighted):        {recall_w.item():.4f}")
        print(f"F1 Score (Weighted):      {f1_w.item():.4f}")
        print("-"*40)
        
        class_map = self.trainer.datamodule.class_to_idx
        class_names = {v: k for k, v in class_map.items()}

        print("--- Per-Class Accuracy ---")
        for i, acc in enumerate(per_class_acc):
            class_name = class_names.get(i, f"Class_{i}")
            print(f"- {class_name:<25}: {acc.item():.4f}")
        
        all_preds = torch.cat(self.test_preds)
        all_labels = torch.cat(self.test_labels)

        total_samples = len(all_labels)
        total_inference_time = sum(self.test_step_times)
        avg_time_per_sample = total_inference_time / total_samples if total_samples > 0 else 0
        samples_per_second = total_samples / total_inference_time if total_inference_time > 0 else 0

        print("\n" + "="*40)
        print(" " * 10 + "INFERENCE SPEED RESULTS")
        print("="*40)
        print(f"Total Samples Processed:  {total_samples}")
        print(f"Total Inference Time:     {total_inference_time:.4f} seconds")
        print(f"Avg. Time per Sample:     {avg_time_per_sample * 1000:.4f} ms") 
        print(f"Samples per Second (FPS): {samples_per_second:.2f}")
        print("="*40 + "\n")

        print("--- Confusion Matrix ---")
        num_classes = self.trainer.datamodule.num_classes
        
        cm = confusion_matrix(
            preds=all_preds, 
            target=all_labels, 
            task="multiclass", 
            num_classes=num_classes
        )
        print(cm)
        print("="*40 + "\n")

        if self.hparams.metrics_output_path is not None:
            metrics_out = {
                "fold_id": getattr(self.trainer.datamodule.hparams, "fold_id", None)
                if hasattr(self.trainer, "datamodule")
                else None,
                "overall_accuracy": overall_acc.item(),
                "mean_class_accuracy": mean_class_acc.item(),
                "precision": precision.item(),
                "recall": recall.item(),
                "f1_score": f1_score.item(),
                "total_samples": total_samples,
                "total_inference_time_sec": total_inference_time,
                "avg_time_per_sample_ms": avg_time_per_sample * 1000,
                "samples_per_second": samples_per_second,
                "per_class_accuracy": {
                    class_names.get(i, f"Class_{i}"): float(acc.item())
                    for i, acc in enumerate(per_class_acc)
                },
            }
            metrics_path = Path(self.hparams.metrics_output_path)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(metrics_out, indent=2))

        self.log_dict({
            "test/overall_accuracy": overall_acc,
            "test/mean_class_accuracy": mean_class_acc,
            "test/precision": precision,
            "test/recall": recall,
            "test/f1_score": f1_score,
            "test/total_inference_time_sec": total_inference_time,
            "test/avg_time_per_sample_ms": avg_time_per_sample * 1000,
            "test/samples_per_second": samples_per_second,
        })
        
        self.test_acc.reset()
        self.test_macc.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()
        self.test_per_class_acc.reset()
        self.test_preds.clear()
        self.test_labels.clear()
        self.test_step_times.clear()

    def configure_optimizers(self):
        assert self.trainer is not None

        encoder_params = []
        other_params = []

        # Encoder and tokenizer form the backbone and share the lower learning rate
        for name, param in self.named_parameters():
            if name.startswith("encoder.") or name.startswith("tokenizer."):
                encoder_params.append(param)
            else:
                other_params.append(param)

        enc_lr: Optional[float] = self.hparams.encoder_learning_rate
        opt = torch.optim.AdamW(
            params=[
                {
                    "params": encoder_params, 
                    "lr": enc_lr if enc_lr is not None else self.hparams.learning_rate
                },
                {"params": other_params},
            ],
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.optimizer_adamw_weight_decay,
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_epochs = self.hparams.lr_scheduler_linear_warmup_epochs
        
        if self.trainer.max_epochs is None or self.trainer.max_epochs == 0:
            if len(self.trainer.datamodule.train_dataloader()) > 0:
                 steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
            else:
                 steps_per_epoch = 1 
        else:
            steps_per_epoch = total_steps // self.trainer.max_epochs

        warmup_steps = min(warmup_epochs * steps_per_epoch, total_steps)
        t_max_cosine = max(1, total_steps - warmup_steps)

        warmup_scheduler = LinearLR(
            opt,
            start_factor=self.hparams.lr_scheduler_linear_warmup_start_lr / self.hparams.learning_rate,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            opt,
            T_max=t_max_cosine,
            eta_min=self.hparams.lr_scheduler_cosine_eta_min
        )
        sequential_scheduler = SequentialLR(
            optimizer=opt,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )

        sched = {
            "scheduler": sequential_scheduler,
            "interval": "step",
        }

        return [opt], [sched]

    def load_pretrained_checkpoint(self, path: str) -> None:
        print(f"Loading pretrained checkpoint from '{path}'.")

        checkpoint = extract_model_checkpoint(path)

        # The pre-training model names the backbone 'student'
        for k in list(checkpoint.keys()):
            if k.startswith("student."):
                new_key = k.replace("student.", "encoder.", 1)
                checkpoint[new_key] = checkpoint.pop(k)

        # Drop the old head and the LeJEPA-only layers, which have no counterpart here
        keys_to_delete = []
        for k in list(checkpoint.keys()):
            if k.startswith("cls_head."):
                keys_to_delete.append(k)
            elif k.startswith("projector.") or k.startswith("sigreg."):
                keys_to_delete.append(k)
            elif k.startswith("encoder.blocks."):
                for i in self.hparams.pretrained_ckpt_ignore_encoder_layers:  # type: ignore
                    if k.startswith(f"encoder.blocks.{i}."):
                        keys_to_delete.append(k)
        
        for k in keys_to_delete:
            del checkpoint[k]

        missing_keys, unexpected_keys = self.load_state_dict(checkpoint, strict=False)  # type: ignore
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")

        # Guard against NaN BatchNorm statistics carried over from pre-training
        for name, m in self.named_modules():
            if isinstance(m, nn.BatchNorm1d):
                if torch.any(torch.isnan(m.running_mean)):  # type: ignore
                    print(f"Warning: NaNs in running_mean of {name}. Setting to zeros.")
                    m.running_mean = torch.zeros_like(m.running_mean)  # type: ignore
                if torch.any(torch.isnan(m.running_var)):  # type: ignore
                    print(f"Warning: NaNs in running_var of {name}. Setting to ones.")
                    m.running_var = torch.ones_like(m.running_var)  # type: ignore

    def on_train_epoch_start(self) -> None:
        if self.hparams.encoder_unfreeze_stepwise:  # type: ignore
            assert isinstance(self.encoder, TransformerEncoder)
            assert self.hparams.encoder_unfreeze_epoch >= 0  # type: ignore
            
            if self.trainer.max_epochs is None:
                print("Warning: trainer.max_epochs is None. Stepwise unfreezing may not work as expected.")
                max_epochs = 1 
            else:
                max_epochs = self.trainer.max_epochs

            unfreeze_epochs = list(
                range(
                    self.hparams.encoder_unfreeze_epoch,  # type: ignore
                    max_epochs,
                    int(
                        (
                            max_epochs
                            - self.hparams.encoder_unfreeze_epoch  # type:ignore
                        )
                        / (
                            (
                                self.hparams.encoder_depth  # type: ignore
                                if self.hparams.encoder_unfreeze_layers is None  # type: ignore
                                else len(
                                    self.hparams.encoder_unfreeze_layers  # type:ignore
                                )
                            )
                            / self.hparams.encoder_unfreeze_stepwise_num_layers  # type: ignore
                        )
                    ),
                )
            )
            if self.trainer.current_epoch in unfreeze_epochs:
                i = unfreeze_epochs.index(self.trainer.current_epoch)
                for j in range(
                    i * self.hparams.encoder_unfreeze_stepwise_num_layers, (i + 1) * self.hparams.encoder_unfreeze_stepwise_num_layers  # type: ignore
                ):
                    unfreeze_layer_idx = -(j + 1)
                    if self.hparams.encoder_unfreeze_layers is not None:  # type: ignore
                        unfreeze_layer_idx: int = self.hparams.encoder_unfreeze_layers[unfreeze_layer_idx]  # type: ignore
                    
                    if abs(unfreeze_layer_idx) <= len(self.encoder.blocks):
                        self.encoder.blocks[unfreeze_layer_idx].requires_grad_(True)
                        self.print(f"Unfreeze encoder layer {unfreeze_layer_idx}")
                    else:
                        self.print(f"Warning: Layer index {unfreeze_layer_idx} out of bounds for stepwise unfreezing.")

        elif self.trainer.current_epoch == self.hparams.encoder_unfreeze_epoch:  # type: ignore
            if self.hparams.encoder_unfreeze_layers is not None:  # type:ignore
                assert isinstance(self.encoder, TransformerEncoder)
                for i in self.hparams.encoder_unfreeze_layers:  # type:ignore
                    if i < len(self.encoder.blocks):
                        self.encoder.blocks[i].requires_grad_(True)
                        print(f"Unfreeze encoder layer {i}")
                    else:
                        print(f"Warning: Layer index {i} out of bounds for unfreezing.")
            else:
                self.encoder.requires_grad_(True)
                # Tokenizer unfreezes with the encoder; tokenizer_unfreeze_epoch is unused
                self.tokenizer.requires_grad_(True)
                print("Unfreeze encoder and tokenizer")

        if self.trainer.current_epoch == self.hparams.positional_encoding_unfreeze_epoch:  # type: ignore
            self.positional_encoding.requires_grad_(True)
            print("Unfreeze positional encoding")