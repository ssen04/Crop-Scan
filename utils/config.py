"""
code/utils/config.py
Central configuration for all CropScan modules.
"""

import torch
import os


class Config:
    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #
    BASE_PATH = os.environ.get("CROPSCAN_BASE", os.path.join(os.path.dirname(__file__), "..", ".."))
    DATA_RAW = os.path.join(BASE_PATH, "data", "raw")
    DATA_PROCESSED = os.path.join(BASE_PATH, "data", "processed")
    MODEL_DIR = os.path.join(BASE_PATH, "models")
    CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")
    OUTPUT_DIR = os.path.join(BASE_PATH, "outputs")

    # ------------------------------------------------------------------ #
    # Supported crops / diseases (PlantVillage label names)
    # ------------------------------------------------------------------ #
    CROPS = ["apple", "grape", "tomato", "corn", "potato"]

    DISEASES = {
        "apple": ["scab", "black_rot", "cedar_rust", "healthy"],
        "grape": ["black_rot", "esca", "leaf_blight", "healthy"],
        "tomato": ["bacterial_spot", "early_blight", "late_blight", "leaf_mold", "healthy"],
    }

    # ------------------------------------------------------------------ #
    # Segmentation training
    # ------------------------------------------------------------------ #
    SEG_IMG_SIZE = 512
    SEG_BATCH_SIZE = 16
    SEG_EPOCHS = 50
    SEG_LR = 1e-4
    SEG_NUM_CLASSES = 2

    MODELS = {
        "unet": "UNet with ResNet34 encoder",
        "deeplabv3": "DeepLabV3+ with ResNet50",
        "segformer": "SegFormer-B0",
        "maskrcnn": "Mask R-CNN",
    }

    # ------------------------------------------------------------------ #
    # ControlNet / GAN training
    # ------------------------------------------------------------------ #
    GAN_IMG_SIZE = 256
    GAN_BATCH_SIZE = 1          # Use 1 with FP16 + AdamW8bit on single GPU
    GAN_EPOCHS = 10
    GAN_LR = 1e-5
    GAN_LAMBDA_L1 = 100         # Kept for reference; ControlNet uses MSE

    # ------------------------------------------------------------------ #
    # Hardware
    # ------------------------------------------------------------------ #
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    NUM_WORKERS = 2
    PIN_MEMORY = True

    # ------------------------------------------------------------------ #
    # Data splits
    # ------------------------------------------------------------------ #
    TRAIN_SPLIT = 0.70
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15

    # ------------------------------------------------------------------ #
    # Training utilities
    # ------------------------------------------------------------------ #
    SAVE_EVERY = 5
    EARLY_STOPPING_PATIENCE = 10

    # ------------------------------------------------------------------ #
    # Augmentation
    # ------------------------------------------------------------------ #
    USE_AUGMENTATION = True
    AUG_ROTATION = 30
    AUG_BRIGHTNESS = 0.2
    AUG_CONTRAST = 0.2


cfg = Config()
