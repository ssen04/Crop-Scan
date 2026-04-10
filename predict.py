"""
predict.py
Run the full CropScan inference pipeline on a single image:
  1. Generate NDVI health map (ControlNet)
  2. Simulate 4-week disease progression
  3. Export progression video

Examples::

    python predict.py \\
        --image       path/to/crop.jpg \\
        --checkpoint  models/controlnet_ndvi/best_controlnet_ndvi.pth \\
        --disease     "Tomato - Late Blight" \\
        --treatment   "Full Treatment (Standard Dose)" \\
        --image_type  leaf \\
        --output_dir  outputs/demo
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="CropScan inference pipeline")
    parser.add_argument("--image",      required=True, help="Path to input crop image")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to ControlNet .pth checkpoint")
    parser.add_argument("--disease",    default="Tomato - Late Blight",
                        help="Disease name (key in CROP_DISEASES)")
    parser.add_argument("--treatment",  default="No Treatment",
                        help="Treatment option (key in TREATMENT_OPTIONS)")
    parser.add_argument("--image_type", choices=["leaf", "field"], default="leaf")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--steps",      type=int, default=30,
                        help="DDIM inference steps (fewer = faster)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load NDVI model
    # ------------------------------------------------------------------
    from code.gan_ndvi.inference import ControlNetNDVIInference

    print("\n[1/4] Loading ControlNet NDVI model…")
    ndvi_model = ControlNetNDVIInference(args.checkpoint, device=args.device)

    # ------------------------------------------------------------------
    # 2. Generate NDVI map
    # ------------------------------------------------------------------
    print("[2/4] Generating NDVI health map…")
    ndvi_map = ndvi_model.generate_ndvi(args.image, num_inference_steps=args.steps)
    ndvi_path = os.path.join(args.output_dir, "ndvi_map.png")
    ndvi_map.save(ndvi_path)
    print(f"      Saved → {ndvi_path}")

    # ------------------------------------------------------------------
    # 3. Disease progression simulation
    # ------------------------------------------------------------------
    from code.economics.disease_progression import generate_disease_progression

    print("[3/4] Simulating 4-week disease progression…")
    result = generate_disease_progression(
        rgb_image_path=args.image,
        crop_disease_name=args.disease,
        treatment_level=args.treatment,
        image_type=args.image_type,
        ndvi_model=ndvi_model,
        weeks=[0, 1, 2, 4],
    )

    # Save progression grid
    weeks = result["weeks"]
    progression = result["progression"]
    metrics = result["metrics"]

    fig, axes = plt.subplots(1, len(weeks), figsize=(16, 4))
    for i, week in enumerate(weeks):
        axes[i].imshow(progression[week])
        axes[i].set_title(f"Week {week}", fontsize=12, fontweight="bold")
        axes[i].axis("off")
    fig.suptitle(
        f"{args.disease}  |  {args.treatment}\n"
        f"Diseased area increase: +{metrics['absolute_increase']:,} pixels",
        fontsize=13, fontweight="bold",
    )
    grid_path = os.path.join(args.output_dir, "progression_grid.png")
    plt.tight_layout()
    plt.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved → {grid_path}")

    # ------------------------------------------------------------------
    # 4. Export video
    # ------------------------------------------------------------------
    from code.video_generation.video_generator import create_progression_video

    print("[4/4] Exporting progression video…")
    video_path = os.path.join(args.output_dir, "progression.mp4")
    duration = create_progression_video(
        progression_images=progression,
        weeks=weeks,
        output_path=video_path,
        crop_name=args.disease,
        treatment_name=args.treatment,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  NDVI map        : {ndvi_path}")
    print(f"  Progression grid: {grid_path}")
    print(f"  Progression video: {video_path}  ({duration:.1f}s)")
    print(f"\n  Disease  : {args.disease}")
    print(f"  Treatment: {args.treatment}")
    print(f"  Week-0 diseased pixels : {metrics['week0_pixels']:,}")
    print(f"  Week-4 diseased pixels : {metrics['week4_pixels']:,}")
    print(f"  Absolute increase      : +{metrics['absolute_increase']:,} px")


if __name__ == "__main__":
    main()
