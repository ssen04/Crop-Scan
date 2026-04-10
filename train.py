"""
train.py
Top-level training entry-point for CropScan.

Examples::

    # 1. Train ControlNet NDVI model
    python train.py --mode controlnet \\
        --rgb_dir  data/processed/gan_training/rgb \\
        --ndvi_dir data/processed/gan_training/ndvi \\
        --save_dir models/controlnet_ndvi \\
        --epochs 10

    # 2. Train binary disease classifier (healthy vs diseased)
    python train.py --mode classifier \\
        --data_dir data/raw/plantvillage_clean \\
        --save_dir models/classifier \\
        --epochs 10
"""

from __future__ import annotations

import argparse
import sys


def train_controlnet(args):
    from code.gan_ndvi.dataset import get_gan_dataloaders
    from code.gan_ndvi.controlnet_trainer import ControlNetNDVITrainer

    print("=" * 60)
    print("ControlNet RGB → NDVI Training")
    print("=" * 60)

    train_loader, val_loader = get_gan_dataloaders(
        rgb_dir=args.rgb_dir,
        ndvi_dir=args.ndvi_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_split=0.8,
    )
    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}")

    trainer = ControlNetNDVITrainer(device=args.device, learning_rate=args.lr)
    trainer.train(train_loader, val_loader, num_epochs=args.epochs,
                  save_dir=args.save_dir)


def train_classifier(args):
    import torch
    import torch.nn as nn
    import torch.optim as optim

    from code.preprocessing.dataset import prepare_dataloaders
    from code.segmentation.models import get_model
    from code.segmentation.classification_trainer import ClassificationTrainer
    from code.utils.config import cfg

    print("=" * 60)
    print("Binary Classifier (Healthy vs Diseased)")
    print("=" * 60)

    train_loader, val_loader, _ = prepare_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=cfg.SEG_IMG_SIZE,
        num_workers=args.num_workers,
    )

    base_model = get_model("unet", num_classes=2)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(base_model.parameters(), lr=args.lr)

    trainer = ClassificationTrainer(
        model=base_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device(args.device),
        save_dir=args.save_dir,
    )
    trainer.train(num_epochs=args.epochs)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="CropScan training script")
    parser.add_argument("--mode", choices=["controlnet", "classifier"],
                        required=True, help="Which model to train")

    # ControlNet args
    parser.add_argument("--rgb_dir",  default="data/processed/gan_training/rgb")
    parser.add_argument("--ndvi_dir", default="data/processed/gan_training/ndvi")

    # Classifier args
    parser.add_argument("--data_dir", default="data/raw/plantvillage_clean")

    # Shared
    parser.add_argument("--save_dir",    default="models")
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--batch_size",  type=int,   default=1)
    parser.add_argument("--lr",          type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int,   default=2)
    parser.add_argument("--device",      default="cuda")

    args = parser.parse_args()

    if args.mode == "controlnet":
        train_controlnet(args)
    else:
        train_classifier(args)


if __name__ == "__main__":
    main()
