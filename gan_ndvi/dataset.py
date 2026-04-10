"""
code/gan_ndvi/dataset.py
Paired RGB ↔ NDVI dataset for ControlNet training.
"""

import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, random_split

import albumentations as A
from albumentations.pytorch import ToTensorV2


class RGBtoNDVIDataset(Dataset):
    """
    Paired dataset of RGB satellite / crop images and corresponding NDVI maps.

    Directory structure expected::

        rgb_dir/   0001.png  0002.png  …
        ndvi_dir/  0001.png  0002.png  …   (same filenames)

    Args:
        rgb_dir:   Directory containing RGB images (PNG).
        ndvi_dir:  Directory containing NDVI images (grayscale PNG).
        transform: Optional Albumentations pipeline. If ``None``, a
                   sensible default is applied based on ``mode``.
        mode:      ``'train'`` or ``'val'`` — controls augmentation.
    """

    def __init__(self, rgb_dir: str, ndvi_dir: str,
                 transform=None, mode: str = "train"):
        self.rgb_dir = rgb_dir
        self.ndvi_dir = ndvi_dir
        self.mode = mode
        self.rgb_files = sorted(f for f in os.listdir(rgb_dir) if f.endswith(".png"))

        if transform is not None:
            self.transform = transform
        elif mode == "train":
            self.transform = A.Compose([
                A.Resize(256, 256),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=90, p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2(),
            ], additional_targets={"ndvi": "image"})
        else:
            self.transform = A.Compose([
                A.Resize(256, 256),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2(),
            ], additional_targets={"ndvi": "image"})

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        filename = self.rgb_files[idx]

        rgb = np.array(Image.open(os.path.join(self.rgb_dir, filename)).convert("RGB"))
        ndvi_gray = np.array(Image.open(os.path.join(self.ndvi_dir, filename)).convert("L"))
        # Expand grayscale to 3 channels so Albumentations can handle it uniformly
        ndvi = np.stack([ndvi_gray, ndvi_gray, ndvi_gray], axis=-1)

        if self.transform:
            out = self.transform(image=rgb, ndvi=ndvi)
            rgb = out["image"]          # [3, H, W]  float32 in [-1, 1]
            ndvi = out["ndvi"][0:1]     # [1, H, W]  keep single channel

        return rgb, ndvi


# --------------------------------------------------------------------------- #
# DataLoader factory
# --------------------------------------------------------------------------- #

def get_gan_dataloaders(
    rgb_dir: str,
    ndvi_dir: str,
    batch_size: int = 8,
    num_workers: int = 2,
    train_split: float = 0.8,
):
    """
    Create train and validation DataLoaders for ControlNet training.

    Args:
        rgb_dir:      Path to RGB images.
        ndvi_dir:     Path to paired NDVI images.
        batch_size:   Mini-batch size.
        num_workers:  DataLoader worker processes.
        train_split:  Fraction of data used for training.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    full_dataset = RGBtoNDVIDataset(rgb_dir, ndvi_dir, mode="train")
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
