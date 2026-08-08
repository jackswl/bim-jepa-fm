import os
from typing import Optional, List
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

class BIMCompNetDataset(Dataset):
    """
    Loads flat .npy files whose label is the last underscore-separated field of the
    filename, e.g. 0_IfcFan_1_IfcFlowMovingDevice.npy -> IfcFlowMovingDevice.
    """
    def __init__(self, file_paths: List[str], class_to_idx: dict):
        super().__init__()
        self.file_paths = file_paths
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        try:
            point_cloud = np.load(file_path).astype(np.float32)

            filename = os.path.basename(file_path)
            name_no_ext = os.path.splitext(filename)[0]
            parts = name_no_ext.split('_')

            if len(parts) > 0:
                class_name = parts[-1]
            else:
                raise ValueError(f"Filename empty or malformed: {filename}")

            label = self.class_to_idx.get(class_name, -1)

            return torch.from_numpy(point_cloud), torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            raise e

class BIMCompNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        test_dir: str,
        batch_size: int = 32,
        num_workers: int = 8,
        val_split_ratio: float = 0.2,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.class_to_idx = {}

    def parse_classes(self, directory):
        """Scans a directory for unique class names. Must match BIMCompNetDataset.__getitem__."""
        classes = set()
        if not os.path.exists(directory):
            return classes

        for f in os.listdir(directory):
            if f.endswith('.npy'):
                name_no_ext = os.path.splitext(f)[0]
                parts = name_no_ext.split('_')

                if len(parts) > 0:
                    class_name = parts[-1]
                    classes.add(class_name)

        return classes

    def setup(self, stage: Optional[str] = None):
        # Union of train and test, so a class present in only one split still gets an index
        print("Scanning classes (using LAST part of filename)...")
        train_classes = self.parse_classes(self.hparams.train_dir)
        test_classes = self.parse_classes(self.hparams.test_dir)

        all_classes = sorted(list(train_classes.union(test_classes)))
        self.class_to_idx = {name: i for i, name in enumerate(all_classes)}

        print(f"Found {len(all_classes)} unique IFC classes.")
        if len(all_classes) > 0:
            print(f"Example classes: {all_classes[:5]}")

        if stage == "fit" or stage is None:
            all_train_files = [
                os.path.join(self.hparams.train_dir, f)
                for f in os.listdir(self.hparams.train_dir) if f.endswith('.npy')
            ]

            all_train_files.sort()  # deterministic ordering

            if self.hparams.val_split_ratio > 0:
                train_files, val_files = train_test_split(
                    all_train_files,
                    test_size=self.hparams.val_split_ratio,
                    random_state=self.hparams.seed
                )
            else:
                train_files = all_train_files
                val_files = []

            self.train_dataset = BIMCompNetDataset(train_files, self.class_to_idx)
            self.val_dataset = BIMCompNetDataset(val_files, self.class_to_idx)

            print(f"Setup Train: {len(self.train_dataset)} | Val: {len(self.val_dataset)}")

        if stage == "test" or stage is None:
            test_files = [
                os.path.join(self.hparams.test_dir, f)
                for f in os.listdir(self.hparams.test_dir) if f.endswith('.npy')
            ]
            test_files.sort()
            self.test_dataset = BIMCompNetDataset(test_files, self.class_to_idx)
            print(f"Setup Test: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.num_workers, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        # persistent_workers must be off when the val set is empty
        nw = self.hparams.num_workers if (self.val_dataset and len(self.val_dataset) > 0) else 0

        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=(nw > 0)
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, pin_memory=True)

    @property
    def num_classes(self):
        return len(self.class_to_idx)
