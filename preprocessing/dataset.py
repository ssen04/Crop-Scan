"""
code/preprocessing/dataset.py
Dataset classes and data-loading utilities for CropScan.
"""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class CropDiseaseDataset(Dataset):
    """
    PyTorch Dataset for crop disease images.

    Args:
        image_paths: List of image file paths.
        mask_paths:  List of mask file paths (optional; required for segmentation mode).
        transform:   Albumentations transform pipeline.
        mode:        ``'classification'`` or ``'segmentation'``.
    """

    def __init__(self, image_paths, mask_paths=None, transform=None, mode="classification"):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))

        if self.mode == "segmentation" and self.mask_paths is not None:
            mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))
            if self.transform:
                out = self.transform(image=image, mask=mask)
                image, mask = out["image"], out["mask"]
            return image, mask

        # Classification mode: derive binary label from folder name
        label = self._label_from_path(self.image_paths[idx])
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label

    @staticmethod
    def _label_from_path(path: str) -> int:
        """Return 0 for healthy, 1 for diseased based on parent folder name."""
        class_name = Path(path).parent.name
        return 0 if "healthy" in class_name.lower() else 1


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #

def get_transforms(img_size: int = 512, mode: str = "train") -> A.Compose:
    """
    Return Albumentations augmentation pipeline.

    Args:
        img_size: Target square image size.
        mode:     ``'train'``, ``'val'``, or ``'test'``.
    """
    if mode == "train":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# --------------------------------------------------------------------------- #
# Dataloaders
# --------------------------------------------------------------------------- #

def prepare_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    img_size: int = 512,
    num_workers: int = 2,
):
    """
    Build train / val / test DataLoaders from a PlantVillage-style directory.

    Expected structure::

        data_dir/
          train/
            ClassName1/  image1.jpg …
            ClassName2/  …

    Args:
        data_dir:    Root directory containing a ``train/`` subfolder.
        batch_size:  Samples per mini-batch.
        img_size:    Resize target (square).
        num_workers: DataLoader worker processes.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_dir = os.path.join(data_dir, "train")
    all_images = []

    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for fname in os.listdir(class_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                all_images.append(os.path.join(class_path, fname))

    print(f"Total images found: {len(all_images)}")

    train_imgs, temp_imgs = train_test_split(all_images, test_size=0.30, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.50, random_state=42)

    print(f"  Train : {len(train_imgs)}")
    print(f"  Val   : {len(val_imgs)}")
    print(f"  Test  : {len(test_imgs)}")

    def _make_loader(paths, mode, shuffle):
        ds = CropDiseaseDataset(paths, transform=get_transforms(img_size, mode))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)

    return (
        _make_loader(train_imgs, "train", shuffle=True),
        _make_loader(val_imgs, "val", shuffle=False),
        _make_loader(test_imgs, "test", shuffle=False),
    )


# --------------------------------------------------------------------------- #
# Quick visualisation helper
# --------------------------------------------------------------------------- #

def visualize_batch(dataloader, num_images: int = 4):
    """Display a batch of images with their labels (uses matplotlib)."""
    import matplotlib.pyplot as plt

    images, labels = next(iter(dataloader))

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    fig, axes = plt.subplots(1, num_images, figsize=(15, 4))
    for i in range(num_images):
        img = (images[i] * std + mean).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].set_title("Healthy" if labels[i] == 0 else "Diseased")
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()
