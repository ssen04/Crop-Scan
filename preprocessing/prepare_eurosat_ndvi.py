"""
code/preprocessing/prepare_eurosat_ndvi.py
Build paired RGB / NDVI training data from the EuroSAT dataset.

EuroSAT multispectral tiles contain separate Near-Infrared (NIR) and Red bands,
which are used to compute the real NDVI ground truth for ControlNet training.

Usage::

    python -m code.preprocessing.prepare_eurosat_ndvi \\
        --eurosat_dir  data/raw/eurosat \\
        --output_dir   data/processed/gan_training \\
        --max_samples  3000

Output structure::

    data/processed/gan_training/
        rgb/   0001.png  0002.png  …
        ndvi/  0001.png  0002.png  …   (paired grayscale NDVI maps)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import rasterio
    _RASTERIO = True
except ImportError:
    _RASTERIO = False


# --------------------------------------------------------------------------- #
# Core helpers
# --------------------------------------------------------------------------- #

def compute_ndvi(nir: np.ndarray, red: np.ndarray,
                 eps: float = 1e-8) -> np.ndarray:
    """
    Compute NDVI = (NIR - Red) / (NIR + Red), clipped to [-1, 1].

    Args:
        nir: Near-infrared band array (float).
        red: Red band array (float).
        eps: Small constant to avoid division by zero.

    Returns:
        NDVI array in [-1, 1].
    """
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    return np.clip((nir - red) / (nir + red + eps), -1.0, 1.0)


def ndvi_to_grayscale(ndvi: np.ndarray) -> np.ndarray:
    """Rescale NDVI from [-1, 1] to uint8 [0, 255]."""
    return ((ndvi + 1.0) / 2.0 * 255).astype(np.uint8)


def process_tif(tif_path: str, rgb_out: str, ndvi_out: str,
                target_size: int = 64) -> bool:
    """
    Extract RGB and NDVI from a multispectral EuroSAT GeoTIFF.

    EuroSAT Sentinel-2 band order (1-indexed):
      B1=Coastal, B2=Blue, B3=Green, B4=Red, B5-B8=RedEdge/NIR, B9-B13=SWIR etc.

    For RGB we use B4 (Red), B3 (Green), B2 (Blue).
    For NDVI we use B8 (NIR) and B4 (Red).

    Args:
        tif_path:    Input GeoTIFF path.
        rgb_out:     Output path for the RGB PNG.
        ndvi_out:    Output path for the grayscale NDVI PNG.
        target_size: Resize target (square pixels).

    Returns:
        True on success, False on error.
    """
    if not _RASTERIO:
        raise ImportError("rasterio is required: pip install rasterio")

    try:
        with rasterio.open(tif_path) as src:
            # EuroSAT has 13 bands — indices are 1-based in rasterio
            red = src.read(4).astype(np.float32)   # B4
            grn = src.read(3).astype(np.float32)   # B3
            blu = src.read(2).astype(np.float32)   # B2
            nir = src.read(8).astype(np.float32)   # B8

        def _norm(band: np.ndarray) -> np.ndarray:
            lo, hi = band.min(), band.max()
            return ((band - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)

        rgb = np.stack([_norm(red), _norm(grn), _norm(blu)], axis=-1)
        ndvi_arr = ndvi_to_grayscale(compute_ndvi(nir, red))

        Image.fromarray(rgb).resize((target_size, target_size)).save(rgb_out)
        Image.fromarray(ndvi_arr).resize((target_size, target_size)).save(ndvi_out)
        return True

    except Exception as exc:
        print(f"  Skipped {Path(tif_path).name}: {exc}")
        return False


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #

def prepare_eurosat(eurosat_dir: str, output_dir: str,
                    max_samples: int = 3000, target_size: int = 64):
    """
    Walk the EuroSAT directory tree and build paired RGB / NDVI PNGs.

    Args:
        eurosat_dir:  Root EuroSAT directory (contains class sub-folders with .tif files).
        output_dir:   Where to write rgb/ and ndvi/ sub-folders.
        max_samples:  Cap on total pairs generated.
        target_size:  Output image size in pixels (square).
    """
    rgb_dir = os.path.join(output_dir, "rgb")
    ndvi_dir = os.path.join(output_dir, "ndvi")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(ndvi_dir, exist_ok=True)

    tif_paths: list[str] = []
    for root, _, files in os.walk(eurosat_dir):
        for f in files:
            if f.lower().endswith(".tif"):
                tif_paths.append(os.path.join(root, f))

    tif_paths = tif_paths[:max_samples]
    print(f"Found {len(tif_paths)} GeoTIFF files — generating up to {max_samples} pairs")

    success = 0
    for i, tif_path in enumerate(tif_paths):
        fname = f"{i:05d}.png"
        ok = process_tif(
            tif_path,
            rgb_out=os.path.join(rgb_dir, fname),
            ndvi_out=os.path.join(ndvi_dir, fname),
            target_size=target_size,
        )
        if ok:
            success += 1
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(tif_paths)} processed  ({success} OK)")

    print(f"\n✓ Done — {success} paired samples saved to {output_dir}")


# --------------------------------------------------------------------------- #
# CLI entry-point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare EuroSAT RGB/NDVI pairs")
    parser.add_argument("--eurosat_dir",  default="data/raw/eurosat")
    parser.add_argument("--output_dir",   default="data/processed/gan_training")
    parser.add_argument("--max_samples",  type=int, default=3000)
    parser.add_argument("--target_size",  type=int, default=64)
    args = parser.parse_args()

    prepare_eurosat(
        eurosat_dir=args.eurosat_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        target_size=args.target_size,
    )
