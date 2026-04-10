"""
code/segmentation/benchmark.py
Multi-architecture benchmark: trains and evaluates ResNet50, VGG16,
EfficientNet-B0, MobileNetV3-Large, and DenseNet121 side-by-side.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader


class MultiModelBenchmark:
    """
    Train and evaluate multiple image-classification backbones on the same
    crop-disease dataset, then compare accuracy, F1, inference speed, and
    model size.

    Args:
        num_classes: Number of disease / healthy classes to predict.
        device:      ``torch.device`` or device string.
    """

    def __init__(self, num_classes: int = 4, device=None):
        self.num_classes = num_classes
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_configs: dict[str, callable] = {
            "ResNet50":      lambda: tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT),
            "VGG16":         lambda: tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT),
            "EfficientNet":  lambda: tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT),
            "MobileNetV3":   lambda: tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.DEFAULT),
            "DenseNet121":   lambda: tvm.densenet121(weights=tvm.DenseNet121_Weights.DEFAULT),
        }

        self.models: dict[str, nn.Module] = {}
        self.results: dict[str, dict] = {}

    # ------------------------------------------------------------------ #
    # Model construction
    # ------------------------------------------------------------------ #

    def create_model(self, model_name: str) -> nn.Module:
        """
        Instantiate a pre-trained backbone and replace its classification head.

        Args:
            model_name: Key in ``self.model_configs``.

        Returns:
            Model moved to ``self.device``.
        """
        model = self.model_configs[model_name]()

        if "ResNet" in model_name:
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif "VGG" in model_name:
            model.classifier[6] = nn.Linear(
                model.classifier[6].in_features, self.num_classes
            )
        elif "EfficientNet" in model_name:
            model.classifier[1] = nn.Linear(
                model.classifier[1].in_features, self.num_classes
            )
        elif "MobileNet" in model_name:
            model.classifier[3] = nn.Linear(
                model.classifier[3].in_features, self.num_classes
            )
        elif "DenseNet" in model_name:
            model.classifier = nn.Linear(
                model.classifier.in_features, self.num_classes
            )

        return model.to(self.device)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train_model(
        self,
        model_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        save_dir: str = "models",
    ):
        """
        Train a single backbone and save the best checkpoint.

        Args:
            model_name:   Key in ``self.model_configs``.
            train_loader: Training DataLoader.
            val_loader:   Validation DataLoader.
            epochs:       Number of training epochs.
            save_dir:     Directory to save ``best_{model_name}.pth``.
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n{'='*60}\nTraining {model_name}\n{'='*60}")
        model = self.create_model(model_name)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            # -- Train --
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()

            # -- Validate --
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    _, predicted = model(images).max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            val_acc = 100.0 * correct / total
            print(f"  Epoch {epoch:>2}/{epochs}  val_acc={val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    model.state_dict(),
                    os.path.join(save_dir, f"best_{model_name}.pth"),
                )

        # Reload best weights
        model.load_state_dict(
            torch.load(os.path.join(save_dir, f"best_{model_name}.pth"))
        )
        self.models[model_name] = model
        print(f"  Best val acc: {best_val_acc:.2f}%")

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def evaluate_model(self, model_name: str, test_loader: DataLoader) -> dict:
        """
        Evaluate a trained model and record metrics.

        Args:
            model_name:  Key in ``self.models``.
            test_loader: Test DataLoader.

        Returns:
            Dict with keys ``accuracy``, ``precision``, ``recall``,
            ``f1_score``, ``model_size_mb``, ``avg_inference_time_ms``,
            ``confusion_matrix``.
        """
        model = self.models[model_name]
        model.eval()

        all_preds, all_labels, times = [], [], []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                t0 = time.time()
                _, predicted = model(images).max(1)
                times.append(time.time() - t0)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted"
        )
        param_count = sum(p.numel() for p in model.parameters())

        result = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "model_size_mb": param_count * 4 / (1024 ** 2),
            "avg_inference_time_ms": np.mean(times) * 1000,
            "confusion_matrix": confusion_matrix(all_labels, all_preds),
        }
        self.results[model_name] = result
        return result

    # ------------------------------------------------------------------ #
    # Visualisation
    # ------------------------------------------------------------------ #

    def visualize_comparison(self):
        """Plot a 2×2 grid comparing accuracy, F1, speed, and model size."""
        import matplotlib.pyplot as plt

        names = list(self.results)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        def _bar(ax, values, label, colour):
            bars = ax.bar(names, values, color=colour, alpha=0.75, edgecolor="black")
            ax.set_ylabel(label)
            ax.set_title(label, fontweight="bold")
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.2f}", ha="center", va="bottom", fontweight="bold")

        _bar(axes[0, 0], [self.results[m]["accuracy"] * 100 for m in names],
             "Accuracy (%)", "green")
        _bar(axes[0, 1], [self.results[m]["f1_score"] * 100 for m in names],
             "F1 Score (%)", "blue")
        _bar(axes[1, 0], [self.results[m]["avg_inference_time_ms"] for m in names],
             "Inference Time (ms)", "orange")
        _bar(axes[1, 1], [self.results[m]["model_size_mb"] for m in names],
             "Model Size (MB)", "red")

        plt.suptitle("Multi-Model Benchmark Comparison", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def print_summary(self):
        """Print a concise results table to stdout."""
        print(f"\n{'Model':<15} {'Acc%':>6} {'F1%':>6} {'ms':>8} {'MB':>8}")
        print("-" * 50)
        for name, r in self.results.items():
            print(
                f"{name:<15} "
                f"{r['accuracy']*100:>6.2f} "
                f"{r['f1_score']*100:>6.2f} "
                f"{r['avg_inference_time_ms']:>8.1f} "
                f"{r['model_size_mb']:>8.1f}"
            )
