"""
code/segmentation/models.py
Segmentation model factory: UNet, DeepLabV3+, SegFormer.
"""

import torch
import segmentation_models_pytorch as smp
from transformers import SegformerForSemanticSegmentation, SegformerConfig


def get_unet(num_classes: int = 2, encoder_name: str = "resnet34",
             encoder_weights: str = "imagenet"):
    """
    UNet with a ResNet34 encoder (default).

    Args:
        num_classes:     Number of output segmentation classes.
        encoder_name:    Encoder backbone (e.g. ``'resnet34'``, ``'resnet50'``).
        encoder_weights: Pre-trained weights source (``'imagenet'`` or ``None``).

    Returns:
        ``smp.Unet`` model instance.
    """
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None,
    )


def get_deeplabv3plus(num_classes: int = 2, encoder_name: str = "resnet50",
                      encoder_weights: str = "imagenet"):
    """
    DeepLabV3+ with a ResNet50 encoder (default).

    Args:
        num_classes:     Number of output segmentation classes.
        encoder_name:    Encoder backbone.
        encoder_weights: Pre-trained weights source.

    Returns:
        ``smp.DeepLabV3Plus`` model instance.
    """
    return smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None,
    )


def get_segformer(num_classes: int = 2, pretrained: bool = True):
    """
    SegFormer-B0 from HuggingFace Transformers.

    Args:
        num_classes: Number of semantic segmentation classes.
        pretrained:  Load weights from ``'nvidia/segformer-b0-finetuned-ade-512-512'``.

    Returns:
        ``SegformerForSemanticSegmentation`` model instance.
    """
    if pretrained:
        return SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
    config = SegformerConfig(num_labels=num_classes)
    return SegformerForSemanticSegmentation(config)


def get_model(model_name: str, num_classes: int = 2):
    """
    Factory function — returns a segmentation model by name.

    Args:
        model_name:  One of ``'unet'``, ``'deeplabv3'``, ``'segformer'``.
        num_classes: Number of output classes.

    Returns:
        Model instance.

    Raises:
        ValueError: If ``model_name`` is not recognised.
    """
    factories = {
        "unet": lambda: get_unet(num_classes),
        "deeplabv3": lambda: get_deeplabv3plus(num_classes),
        "segformer": lambda: get_segformer(num_classes),
    }
    if model_name not in factories:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(factories)}")
    return factories[model_name]()


# --------------------------------------------------------------------------- #
# Quick sanity-check
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    x = torch.randn(2, 3, 512, 512)

    model = get_unet()
    print(f"UNet output      : {model(x).shape}")

    model = get_deeplabv3plus()
    print(f"DeepLabV3+ output: {model(x).shape}")

    print("All models OK.")
