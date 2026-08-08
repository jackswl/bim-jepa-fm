from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from pytorch3d.ops import sample_farthest_points, knn_points
from pytorch3d.ops.utils import masked_gather

from pointjepa.modules.transformer import TransformerEncoder
from pointjepa.modules.tokenizer import PointcloudTokenizer
from pointjepa.utils import transforms


class SIGReg(nn.Module):
    """
    Sketched Isotropic Gaussian Regularizer.
    Expects proj of shape (V, B, D) or (B, D).
    - knots: number of quadrature points (paper recommends 17)
    - num_dirs: number of random directions / slices |A| (paper recommends ~1024)
    - t_max: we integrate symmetrically over [-t_max, t_max] via [0, t_max] + symmetry
    """

    def __init__(self, knots: int = 17, num_dirs: int = 1024, t_max: float = 5.0):
        super().__init__()

        self.num_dirs = num_dirs
        self.t_max = t_max

        t = torch.linspace(0.0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)

        # Trapezoidal rule on [0, t_max], doubled for symmetry over [-t_max, t_max]
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[0] = dt        # endpoints only counted once
        weights[-1] = dt

        # Target characteristic function of N(0,1) at t
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)                       # (knots,)
        self.register_buffer("phi", window)                # (knots,)
        self.register_buffer("weights", weights * window)  # (knots,)

    def forward(self, proj: torch.Tensor, global_step: int | None = None) -> torch.Tensor:
        """
        proj: (V, B, D) or (B, D)
        global_step: used to deterministically seed direction sampling across ranks
        """
        if proj.dim() == 2:
            proj = proj.unsqueeze(0)  # (1, B, D)

        V, N, D = proj.shape

        # Random slicing directions A: (D, num_dirs)
        gen = torch.Generator(device=proj.device)
        if global_step is not None:
            gen.manual_seed(global_step)
        A = torch.randn(
            D, self.num_dirs, device=proj.device, dtype=proj.dtype, generator=gen
        )
        A = A / A.norm(p=2, dim=0, keepdim=True).clamp_min(1e-6)

        x = proj @ A  # (V, N, num_dirs)

        x_t = x.unsqueeze(-1) * self.t  # (V, N, num_dirs, knots)

        # Empirical characteristic function over the N samples
        cos_term = x_t.cos().mean(dim=1)  # (V, num_dirs, knots)
        sin_term = x_t.sin().mean(dim=1)  # (V, num_dirs, knots)

        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if world_size > 1:
            dist.all_reduce(cos_term, op=dist.ReduceOp.SUM)
            dist.all_reduce(sin_term, op=dist.ReduceOp.SUM)
            cos_term = cos_term / world_size
            sin_term = sin_term / world_size

        # Epps-Pulley integrand vs N(0,1)
        err = (cos_term - self.phi).square() + sin_term.square()   # (V, num_dirs, knots)

        statistic = (err @ self.weights) * (N * world_size)   # (V, num_dirs)

        return statistic.mean()


class PointLeJepa(pl.LightningModule):
    """
    LeJEPA-style pretraining for BIM point clouds, built on the Point-JEPA
    backbone (PointcloudTokenizer, TransformerEncoder, transforms.*).

    Differences vs Point-JEPA:
      * NO EMA teacher
      * NO predictor / target / context samplers
      * multi-view input per element: 2 global + 8 local views
      * loss = invariance toward centroid of global views + SIGReg
    """

    def __init__(
        self,
        # tokenizer / encoder
        tokenizer_num_groups: int = 64,
        tokenizer_group_size: int = 32,
        tokenizer_group_radius: float | None = None,
        encoder_dim: int = 384,
        encoder_depth: int = 12,
        encoder_heads: int = 6,
        encoder_dropout: float = 0.0,
        encoder_attention_dropout: float = 0.05,
        encoder_drop_path_rate: float = 0.25,
        encoder_add_pos_at_every_layer: bool = True,
        # optimization
        learning_rate: float = 1e-4,
        optimizer_adamw_weight_decay: float = 0.05,
        lr_scheduler_linear_warmup_epochs: int = 50,
        lr_scheduler_linear_warmup_start_lr: float = 1e-5,
        lr_scheduler_cosine_eta_min: float = 1e-6,
        fix_estimated_stepping_batches: Optional[int] = None,
        # data / augmentations
        train_transformations: List[str] = (
            "scale",
            "center",
            "unit_sphere",
            "rotate",
            "jitter"
        ),
        val_transformations: List[str] = ("center", "unit_sphere"),
        transformation_subsample_points: int = 4096,  # unused if 'subsample' not in list
        transformation_scale_min: float = 0.8,
        transformation_scale_max: float = 1.2,
        transformation_scale_symmetries: Tuple[int, int, int] = (1, 1, 0),
        transformation_rotate_dims: List[int] = (2,), # z axis
        transformation_rotate_degs: Optional[int] = None,
        transformation_translate: float = 0.2,
        transformation_height_normalize_dim: int = 1,
        # LeJEPA
        lejepa_proj_dim: int = 256,
        lejepa_lambda: float = 0.05,  # weight on SIGReg
        sigreg_knots: int = 17,
        sigreg_num_dirs: int = 1024,
        sigreg_t_max: float = 5.0,
        # multi-view
        num_global_views: int = 2,
        num_local_views: int = 8,
        global_neighbor_ratio: float = 0.75,  # 75% of points
        local_num_points: int = 512,          # ~12.5% of 4096
        # absorbs unused kwargs from older configs (predictor, ema, ...)
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        def build_transformation(name: str) -> transforms.Transform:
            if name == "subsample":
                return transforms.PointcloudSubsampling(transformation_subsample_points)
            elif name == "scale":
                return transforms.PointcloudScaling(
                    min=transformation_scale_min,
                    max=transformation_scale_max,
                    symmetries=transformation_scale_symmetries,
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
            elif name == "height_norm":
                return transforms.PointcloudHeightNormalization(
                    transformation_height_normalize_dim
                )
            elif name == "jitter":
                return transforms.PointcloudJitter(
                    sigma=0.01,
                    clip=0.05,
                )
            else:
                raise RuntimeError(f"No such transformation: {name}")

        self.train_transformations = transforms.Compose(
            [build_transformation(name) for name in train_transformations]
        )
        self.val_transformations = transforms.Compose(
            [build_transformation(name) for name in val_transformations]
        )

        self.positional_encoding = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, encoder_dim),
        )

        self.tokenizer = PointcloudTokenizer(
            num_groups=tokenizer_num_groups,
            group_size=tokenizer_group_size,
            group_radius=tokenizer_group_radius,
            token_dim=encoder_dim,
        )

        dpr_encoder = [
            x.item() for x in torch.linspace(0, encoder_drop_path_rate, encoder_depth)
        ]
        self.student = TransformerEncoder(
            embed_dim=encoder_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            qkv_bias=True,
            drop_rate=encoder_dropout,
            attn_drop_rate=encoder_attention_dropout,
            drop_path_rate=dpr_encoder,
            add_pos_at_every_layer=encoder_add_pos_at_every_layer,
        )

        # Tokens are max- and mean-pooled into 2 * encoder_dim before projection
        rep_dim = 2 * encoder_dim
        self.projector = nn.Sequential(
            nn.Linear(rep_dim, 4 * rep_dim, bias=False),
            nn.BatchNorm1d(4 * rep_dim),
            nn.GELU(),
            nn.Linear(4 * rep_dim, 4 * rep_dim, bias=False),
            nn.BatchNorm1d(4 * rep_dim),
            nn.GELU(),
            nn.Linear(4 * rep_dim, lejepa_proj_dim),
        )

        self.sigreg = SIGReg(knots=sigreg_knots, num_dirs=sigreg_num_dirs, t_max=sigreg_t_max)

        self.num_global_views = num_global_views
        self.num_local_views = num_local_views
        self.global_neighbor_ratio = global_neighbor_ratio
        self.local_num_points = local_num_points

    @torch.no_grad()
    def _fps_knn_views(
        self,
        points: torch.Tensor,
        num_seeds: int,
        k_neighbors: int,
    ) -> List[torch.Tensor]:
        """
        points: (B, N, 3)
        returns: list of `num_seeds` tensors, each (B, k_neighbors, 3)
        """
        B, N, _ = points.shape

        # FPS seeds, spatially spread across the element
        _, seed_idx = sample_farthest_points(
            points[:, :, :3].float(), K=num_seeds, random_start_point=True
        )  # (B, num_seeds)
        seed_points = masked_gather(points, seed_idx)  # (B, num_seeds, 3)

        # knn_points(query, support) -> neighbors from 'support' for each query point
        _, _, knn_xyz = knn_points(
            seed_points[:, :, :3].float(),
            points[:, :, :3].float(),
            K=min(k_neighbors, N),
            return_nn=True,
        )  # (B, num_seeds, K, 3)

        views: List[torch.Tensor] = []
        for i in range(num_seeds):
            v = knn_xyz[:, i, :, :]  # (B, K, 3)
            views.append(v)

        return views

    @torch.no_grad()
    def _generate_views(self, points: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Generate 2 global + 8 local views for a batch of BIM elements.

        points: (B, N, 3) (e.g., N = 4096)
        returns:
          global_views: list length G (2), each (B, N_g, 3)
          local_views:  list length L (8), each (B, N_l, 3)
        """
        B, N, _ = points.shape
        device = points.device

        # Globals: large coherent neighbourhoods
        k_global = int(N * self.global_neighbor_ratio)
        global_views = self._fps_knn_views(points, self.num_global_views, k_global)

        # Locals: small neighbourhoods
        local_views = self._fps_knn_views(points, self.num_local_views, self.local_num_points)

        global_views = [g.to(device) for g in global_views]
        local_views = [l.to(device) for l in local_views]

        return global_views, local_views

    def _encode_single_view(self, points: torch.Tensor) -> torch.Tensor:
        """
        points: (B, N_v, 3) after augmentations
        returns: (B, rep_dim) pooled representation for this view
        """
        tokens, centers = self.tokenizer(points)  # (B, T, C), (B, T, 3)
        pos = self.positional_encoding(centers)   # (B, T, C)
        out = self.student(tokens, pos).last_hidden_state  # (B, T, C)

        max_feat = out.max(dim=1).values
        mean_feat = out.mean(dim=1)
        emb = torch.cat([max_feat, mean_feat], dim=-1)  # (B, 2 * encoder_dim)
        return emb

    def _encode_multi_view(self, points: torch.Tensor, train: bool = True) -> torch.Tensor:
        """
        points: (B, N, 3)
        returns: proj of shape (V, B, D_proj)
        where V = num_global_views + num_local_views
        """
        global_views, local_views = self._generate_views(points)
        all_views = global_views + local_views  # length V

        rep_list: List[torch.Tensor] = []
        for v_pts in all_views:
            # shared augmentations for both global & local
            v_aug = self.train_transformations(v_pts) if train else self.val_transformations(v_pts)
            rep = self._encode_single_view(v_aug)  # (B, rep_dim)
            rep_list.append(rep)

        reps = torch.stack(rep_list, dim=0)  # (V, B, rep_dim)

        V, B, D_rep = reps.shape
        proj_flat = self.projector(reps.reshape(V * B, D_rep))  # (V*B, D_proj)
        proj = proj_flat.view(V, B, -1)                        # (V, B, D_proj)
        return proj

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        # labels unused for pretraining
        points, _ = batch
        points = points.to(self.device)

        proj = self._encode_multi_view(points, train=True)  # (V, B, D)

        V, B, D = proj.shape
        G = self.num_global_views

        # Invariance: every view pulled toward the centroid of the global views
        global_proj = proj[:G]             # (G, B, D)
        mu = global_proj.mean(dim=0)       # (B, D)
        inv_loss = (proj - mu.unsqueeze(0)).pow(2).mean()

        sigreg_loss = self.sigreg(proj, global_step=self.global_step)

        lamb = self.hparams.lejepa_lambda
        lejepa_loss = lamb * sigreg_loss + (1.0 - lamb) * inv_loss

        self.log("train/inv_loss", inv_loss, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("train/sigreg_loss", sigreg_loss, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("train/lejepa_loss", lejepa_loss, on_epoch=True, sync_dist=True, prog_bar=True)

        return lejepa_loss

    def validation_step(self, batch, batch_idx: int) -> None:
        points, _ = batch
        points = points.to(self.device)

        proj = self._encode_multi_view(points, train=False)
        V, B, D = proj.shape
        G = self.num_global_views

        global_proj = proj[:G]
        mu = global_proj.mean(dim=0)
        inv_loss = (proj - mu.unsqueeze(0)).pow(2).mean()
        sigreg_loss = self.sigreg(proj, global_step=self.global_step)
        lamb = self.hparams.lejepa_lambda
        lejepa_loss = lamb * sigreg_loss + (1.0 - lamb) * inv_loss

        self.log("val/inv_loss", inv_loss, prog_bar=True)
        self.log("val/sigreg_loss", sigreg_loss, prog_bar=True)
        self.log("val/lejepa_loss", lejepa_loss, prog_bar=True)

    def configure_optimizers(self):
        assert self.trainer is not None

        if self.hparams.fix_estimated_stepping_batches is not None:
            assert (
                self.trainer.estimated_stepping_batches
                == self.hparams.fix_estimated_stepping_batches
            )

        opt = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.optimizer_adamw_weight_decay,
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_epochs = self.hparams.lr_scheduler_linear_warmup_epochs

        if self.trainer.max_epochs is None or self.trainer.max_epochs == 0:
            steps_per_epoch = max(len(self.trainer.datamodule.train_dataloader()), 1)
        else:
            steps_per_epoch = max(total_steps // self.trainer.max_epochs, 1)

        warmup_steps = min(warmup_epochs * steps_per_epoch, total_steps)
        t_max_cosine = max(1, total_steps - warmup_steps)

        warmup_scheduler = LinearLR(
            opt,
            start_factor=self.hparams.lr_scheduler_linear_warmup_start_lr
            / self.hparams.learning_rate,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            opt,
            T_max=t_max_cosine,
            eta_min=self.hparams.lr_scheduler_cosine_eta_min,
        )

        sequential_scheduler = SequentialLR(
            optimizer=opt,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        sched = {
            "scheduler": sequential_scheduler,
            "interval": "step",
        }

        return [opt], [sched]

    def extract_patch_features(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract token-level features (patches) from the encoder.
        Returns:
            features: (B, T, D) - Contextualized token features from the last encoder layer.
            centers: (B, T, 3) - The 3D coordinates (centroids) of each token.
        """
        tokens, centers = self.tokenizer(points)
        pos_embeddings = self.positional_encoding(centers)
        encoded_tokens = self.student(tokens, pos_embeddings).last_hidden_state
        return encoded_tokens, centers
