import os
from typing import Optional, List, Union, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

class CombinedIFCDataset(Dataset):
    """
    A simple dataset that receives a list of file paths and their corresponding labels,
    pre-processed by the DataModule.
    """
    def __init__(self, items: List[Tuple[str, int]]):
        super().__init__()
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        file_path, label = self.items[idx]
        point_cloud = np.load(file_path).astype(np.float32)
        return torch.from_numpy(point_cloud), torch.tensor(label, dtype=torch.long)

class CombinedIFCDataModule(pl.LightningDataModule):
    """
    DataModule that handles multiple IFC datasets with mixed directory structures:
    1. Flat structure: /path/to/data/001_IfcClass.npy
    2. Nested structure: /path/to/data/IfcClass/some_file.npy
    """
    def __init__(
        self,
        data_dirs: Union[str, List[str]],
        batch_size: int = 32,
        num_workers: int = 8,
        val_split_ratio: float = 0.3,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()

    def _get_class_name(self, file_path: str) -> Optional[str]:
        """
        Extracts the class name, handling three layouts:
        - nested folders:   .../IfcBeam/file.npy
        - flat files (old): .../001_IfcDoor.npy
        - flat files (new): .../7_IfcBeam_1.npy
        """
        try:
            parent_dir_name = os.path.basename(os.path.dirname(file_path))
            if parent_dir_name.startswith('Ifc'):
                return parent_dir_name

            filename = os.path.basename(file_path)
            name_without_ext = os.path.splitext(filename)[0]

            parts = name_without_ext.split('_')
            for part in parts:
                if part.startswith('Ifc'):
                    # Standard IFC class names contain no underscores
                    return part

            # Fallback: assume <id>_<class>
            if len(parts) > 1:
                return parts[1]

            return None

        except IndexError:
            return None

    def setup(self, stage: Optional[str] = None):
        all_files = []
        dirs_to_scan = self.hparams.data_dirs
        if isinstance(dirs_to_scan, str):
            dirs_to_scan = [dirs_to_scan]

        print("Scanning for .npy files...")
        for directory in dirs_to_scan:
            if os.path.isdir(directory):
                for root, _, files in os.walk(directory):
                    for file in files:
                        if file.endswith('.npy'):
                            all_files.append(os.path.join(root, file))
            else:
                print(f"Warning: Directory not found, skipping: {directory}")

        if not all_files:
            raise ValueError(f"No .npy files found in provided directories: {dirs_to_scan}")

        file_class_pairs = []
        all_class_names = set()
        for file_path in all_files:
            class_name = self._get_class_name(file_path)
            if class_name:
                all_class_names.add(class_name)
                file_class_pairs.append((file_path, class_name))

        sorted_classes = sorted(list(all_class_names))
        self.class_to_idx = {name: i for i, name in enumerate(sorted_classes)}

        all_items = [(fp, self.class_to_idx[cn]) for fp, cn in file_class_pairs]

        if self.hparams.val_split_ratio > 0.0 and len(all_items) > 1:
            train_items, val_items = train_test_split(
                all_items,
                test_size=self.hparams.val_split_ratio,
                random_state=self.hparams.seed
            )
        else:
            train_items = all_items
            val_items = []

        self.train_dataset = CombinedIFCDataset(train_items)
        self.val_dataset = CombinedIFCDataset(val_items)

        print(f"CombinedIFCDataModule setup complete:")
        print(f" - Found {len(self.class_to_idx)} classes from {len(all_items)} total files.")
        print(f" - Train samples: {len(self.train_dataset)}")
        print(f" - Validation samples: {len(self.val_dataset)}")


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
        if not self.val_dataset or len(self.val_dataset) == 0:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    @property
    def num_classes(self):
        return len(self.class_to_idx) if hasattr(self, 'class_to_idx') else 0
