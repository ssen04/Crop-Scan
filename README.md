# CropScan

AI-powered crop disease detection and progression forecasting system.

## Overview

CropScan uses a Stable Diffusion ControlNet model to generate NDVI (vegetation health) maps from ordinary RGB images, then simulates disease spread over time using literature-backed parameters.

**Pipeline:**
1. RGB crop image → ControlNet → NDVI health map
2. NDVI map + disease parameters → 4-week progression simulation
3. Progression frames → MP4 video output

---

## Project Structure

```
cropscan/
├── code/
│   ├── utils/
│   │   └── config.py                  # Central hyperparameter config
│   ├── preprocessing/
│   │   └── dataset.py                 # Dataset classes & dataloaders
│   ├── segmentation/
│   │   ├── models.py                  # UNet / DeepLabV3+ / SegFormer
│   │   ├── classification_trainer.py  # Binary healthy/diseased trainer
│   │   └── inference.py               # DiseaseDetector inference class
│   ├── gan_ndvi/
│   │   ├── dataset.py                 # RGB↔NDVI paired dataset
│   │   ├── controlnet_trainer.py      # ControlNet fine-tuning trainer
│   │   └── inference.py               # ControlNetNDVIInference
│   ├── video_generation/
│   │   └── video_generator.py         # Interpolation & MP4 export
│   └── economics/
│       └── disease_progression.py     # Disease DB + progression engine
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download PlantVillage dataset

```bash
kaggle datasets download -d vipoooool/new-plant-diseases-dataset -p data/raw/plantvillage
```

---

## Usage

### Train the ControlNet NDVI model

```python
from code.gan_ndvi.controlnet_trainer import ControlNetNDVITrainer
from code.gan_ndvi.dataset import get_gan_dataloaders

train_loader, val_loader = get_gan_dataloaders(
    rgb_dir="data/processed/gan_training/rgb",
    ndvi_dir="data/processed/gan_training/ndvi",
    batch_size=1
)

trainer = ControlNetNDVITrainer(device="cuda")
trainer.train(train_loader, val_loader, num_epochs=10, save_dir="models/controlnet_ndvi")
```

### Generate NDVI from an RGB image

```python
from code.gan_ndvi.inference import ControlNetNDVIInference

model = ControlNetNDVIInference("models/controlnet_ndvi/best_controlnet_ndvi.pth")
ndvi_map = model.generate_ndvi("path/to/crop_image.jpg")
ndvi_map.save("ndvi_output.png")
```

### Run disease progression simulation

```python
from code.economics.disease_progression import DiseaseProgressionEngine

engine = DiseaseProgressionEngine("Tomato___Late_blight", treatment="spray")
progression = engine.generate_progression_images(ndvi_map, weeks=[0, 1, 2, 4])
```

### Export progression video

```python
from code.video_generation.video_generator import create_progression_video

create_progression_video(
    progression_images=progression,
    weeks=[0, 1, 2, 4],
    output_path="outputs/progression.mp4",
    crop_name="Tomato Late Blight",
    treatment_name="Spray"
)
```

---

## Models

| Model | Purpose | Base |
|-------|---------|------|
| ControlNet (NDVI) | RGB → NDVI generation | `lllyasviel/sd-controlnet-canny` |
| UNet | Disease segmentation | ResNet34 encoder |
| DeepLabV3+ | Disease segmentation | ResNet50 encoder |
| SegFormer-B0 | Disease segmentation | `nvidia/segformer-b0-finetuned-ade-512-512` |

---

## Disease Database

Literature-backed spread parameters for 14 diseases across tomato, apple, grape, corn, and wheat. Sources include Fry (2008), MacHardy (1996), Hoffman et al. (2004), and others.

---

## Notes

- Training was originally done on Google Colab with an A100 GPU
- ControlNet is fine-tuned with FP16 + AdamW8bit (bitsandbytes) for memory efficiency
- The model generalises cross-scale: trained on EuroSAT satellite (64×64px, ~10m/pixel), validated on PlantVillage close-ups
