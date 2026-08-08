import os
import random
from typing import Optional, List

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


class IFCNetCoreDataset(Dataset):
    """A PyTorch Dataset for loading .npy point clouds from the IFCNetCore structure."""
    def __init__(self, file_paths: List[str], class_to_idx: dict):
        super().__init__()
        self.file_paths = file_paths
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        point_cloud = np.load(file_path).astype(np.float32)
        class_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        label = self.class_to_idx.get(class_name, -1)
        return torch.from_numpy(point_cloud), torch.tensor(label, dtype=torch.long)


class IFCNetCoreDataModule(pl.LightningDataModule):
    """A PyTorch Lightning DataModule for the IFCNetCore dataset."""
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 8,
        val_split_ratio: float = 0.3,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        class_dirs = [d for d in os.listdir(self.hparams.data_dir)
                      if os.path.isdir(os.path.join(self.hparams.data_dir, d))]
        class_dirs.sort()
        self.class_to_idx = {name: i for i, name in enumerate(class_dirs)}

        train_val_files: List[str] = []
        test_files: List[str] = []

        for class_name in class_dirs:
            class_path = os.path.join(self.hparams.data_dir, class_name)

            train_path = os.path.join(class_path, "train")
            if os.path.exists(train_path):
                files = [os.path.join(train_path, f) for f in os.listdir(train_path)
                         if f.endswith(".npy")]
                files.sort()  # deterministic ordering
                train_val_files.extend(files)

            test_path = os.path.join(class_path, "test")
            if os.path.exists(test_path):
                files = [os.path.join(test_path, f) for f in os.listdir(test_path)
                         if f.endswith(".npy")]
                files.sort()  # deterministic ordering
                test_files.extend(files)

        rng = random.Random(self.hparams.seed)
        rng.shuffle(train_val_files)

        if self.hparams.val_split_ratio > 0.0:
            def label_of(fp: str) -> int:
                cname = os.path.basename(os.path.dirname(os.path.dirname(fp)))
                return self.class_to_idx[cname]

            y = [label_of(fp) for fp in train_val_files]
            train_files, val_files = train_test_split(
                train_val_files,
                test_size=self.hparams.val_split_ratio,
                random_state=self.hparams.seed,
                stratify=y if len(set(y)) > 1 else None,
            )
        else:
            train_files = train_val_files
            val_files = []

        self.train_dataset = IFCNetCoreDataset(train_files, self.class_to_idx)
        self.val_dataset = IFCNetCoreDataset(val_files, self.class_to_idx)
        self.test_dataset = IFCNetCoreDataset(test_files, self.class_to_idx)

        print(f"IFCNetCoreDataModule setup complete:")
        print(f" - Found {len(self.class_to_idx)} classes.")
        print(f" - Train samples: {len(self.train_dataset)}")
        print(f" - Validation samples: {len(self.val_dataset)}")
        print(f" - Test samples: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def val_dataloader(self):
        # persistent_workers must be off when the val set is empty
        nw = self.hparams.num_workers if len(self.val_dataset) > 0 else 0
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=nw,
            persistent_workers=(nw > 0),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    @property
    def num_classes(self):
        return len(self.class_to_idx)
