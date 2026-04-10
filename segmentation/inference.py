"""
code/segmentation/inference.py
Inference wrapper for segmentation-based disease detection.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2


class DiseaseDetector:
    """
    Wraps a trained segmentation model for single-image inference.

    Args:
        model:  Trained ``smp`` or Transformers segmentation model.
        device: Device string or ``torch.device``.
    """

    def __init__(self, model, device: str = "cuda"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        self.transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    # ------------------------------------------------------------------ #

    def predict(self, image_path: str):
        """
        Run inference on a single image.

        Args:
            image_path: Path to the input image.

        Returns:
            Tuple of (original_image: ndarray, disease_mask: ndarray,
                      confidence_map: ndarray).
        """
        original = np.array(Image.open(image_path).convert("RGB"))
        tensor = self.transform(image=original)["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            if hasattr(output, "logits"):
                output = output.logits
            output = F.interpolate(output, size=(512, 512), mode="bilinear",
                                   align_corners=False)
            probs = torch.sigmoid(output)
            mask = (probs > 0.5).float()

        disease_mask = mask[0, 0].cpu().numpy()
        confidence = probs[0, 0].cpu().numpy()
        return original, disease_mask, confidence

    # ------------------------------------------------------------------ #

    def visualize(self, image_path: str, save_path: str | None = None) -> float:
        """
        Run inference and display a 3-panel figure (RGB / mask / overlay).

        Args:
            image_path: Path to the input image.
            save_path:  Optional path to save the figure.

        Returns:
            Disease coverage as a percentage of total pixels.
        """
        original, mask, confidence = self.predict(image_path)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(mask, cmap="Reds")
        axes[1].set_title("Disease Mask")
        axes[1].axis("off")

        overlay_mask = np.stack([mask, np.zeros_like(mask), np.zeros_like(mask)], axis=-1)
        overlay = (0.7 * original + 0.3 * overlay_mask * 255).astype(np.uint8)
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay  (red = diseased)")
        axes[2].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

        disease_pct = (mask.sum() / mask.size) * 100
        print(f"\nDisease coverage  : {disease_pct:.2f}%")
        print(f"Mean confidence   : {confidence.mean():.3f}")
        return disease_pct
