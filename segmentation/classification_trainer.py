"""
code/segmentation/classification_trainer.py
Binary classification trainer (healthy vs. diseased).
Uses a segmentation model's encoder as a feature extractor.
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class ClassificationTrainer:
    """
    Trains a binary classifier (healthy=0, diseased=1) by attaching a small
    classification head to the encoder of an ``smp`` segmentation model.

    Args:
        model:        An ``smp`` model whose ``.encoder`` will be used as backbone.
        train_loader: Training DataLoader yielding (image, label) batches.
        val_loader:   Validation DataLoader.
        criterion:    Loss function (e.g. ``nn.BCELoss()``).
        optimizer:    Optimizer applied to backbone + classifier parameters.
        device:       ``torch.device`` to train on.
        save_dir:     Directory for checkpoint files.
    """

    def __init__(self, model, train_loader, val_loader, criterion,
                 optimizer, device, save_dir):
        self.backbone = model.encoder.to(device)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        ).to(device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.val_accs: list[float] = []
        self.best_acc: float = 0.0

    # ------------------------------------------------------------------ #
    # Training epoch
    # ------------------------------------------------------------------ #

    def train_epoch(self) -> float:
        self.backbone.train()
        self.classifier.train()
        running_loss = 0.0

        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.float().unsqueeze(1).to(self.device)

            features = self.backbone(images)[-1]
            outputs = self.classifier(features)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return running_loss / len(self.train_loader)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> tuple[float, float]:
        self.backbone.eval()
        self.classifier.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validating")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)

                features = self.backbone(images)[-1]
                outputs = self.classifier(features)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                predictions = (outputs > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                pbar.set_postfix({"acc": f"{100.*correct/total:.2f}%"})

        return running_loss / len(self.val_loader), 100.0 * correct / total

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        os.makedirs(self.save_dir, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "backbone_state_dict": self.backbone.state_dict(),
            "classifier_state_dict": self.classifier.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accs": self.val_accs,
            "best_acc": self.best_acc,
        }
        torch.save(checkpoint, os.path.join(self.save_dir, "latest.pth"))
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, "best.pth"))
            print(f"  Saved best model  (acc={self.best_acc:.2f}%)")

    # ------------------------------------------------------------------ #
    # Main training loop
    # ------------------------------------------------------------------ #

    def train(self, num_epochs: int):
        print(f"\nStarting training for {num_epochs} epochs on {self.device}")

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}\nEpoch {epoch}/{num_epochs}\n{'='*60}")

            train_loss = self.train_epoch()
            val_loss, val_acc = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            print(f"  Train loss : {train_loss:.4f}")
            print(f"  Val loss   : {val_loss:.4f}")
            print(f"  Val acc    : {val_acc:.2f}%")

            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc

            if epoch % 5 == 0 or is_best:
                self.save_checkpoint(epoch, is_best)

        print(f"\nTraining complete — best accuracy: {self.best_acc:.2f}%")
