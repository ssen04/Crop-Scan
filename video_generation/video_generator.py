"""
code/video_generation/video_generator.py
Create smooth MP4 disease-progression videos from keyframe PIL images.

Dependencies: opencv-python-headless, numpy, Pillow.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
# Frame interpolation
# --------------------------------------------------------------------------- #

def interpolate_images(img1: Image.Image, img2: Image.Image,
                       num_frames: int) -> list[Image.Image]:
    """
    Create a smooth linear blend between two PIL images.

    Args:
        img1:       Start frame.
        img2:       End frame.
        num_frames: Number of intermediate frames to generate.

    Returns:
        List of ``num_frames`` interpolated PIL Images.
    """
    arr1 = np.array(img1).astype(np.float32)
    arr2 = np.array(img2).astype(np.float32)
    return [
        Image.fromarray((arr1 * (1 - i / num_frames) + arr2 * (i / num_frames)).astype(np.uint8))
        for i in range(num_frames)
    ]


# --------------------------------------------------------------------------- #
# Video export
# --------------------------------------------------------------------------- #

def create_progression_video(
    progression_images: dict[int, Image.Image],
    weeks: list[int],
    output_path: str,
    crop_name: str,
    treatment_name: str,
    frames_per_week: int = 30,
    fps: int = 30,
    hold_seconds: float = 2.0,
) -> float:
    """
    Render a disease-progression MP4 from keyframe PIL images.

    Args:
        progression_images: Dict mapping week number → PIL Image.
        weeks:              Ordered list of week numbers matching the dict keys.
        output_path:        Destination ``.mp4`` file path.
        crop_name:          Displayed in the video overlay (e.g. ``'Tomato Late Blight'``).
        treatment_name:     Displayed in the video overlay.
        frames_per_week:    Interpolation frames between each weekly keyframe.
        fps:                Output video frame rate.
        hold_seconds:       How long to hold the final frame.

    Returns:
        Video duration in seconds.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # ---- Build full frame list ----
    all_frames: list[Image.Image] = []
    for i in range(len(weeks) - 1):
        all_frames.extend(
            interpolate_images(
                progression_images[weeks[i]],
                progression_images[weeks[i + 1]],
                frames_per_week,
            )
        )
    final = progression_images[weeks[-1]]
    all_frames.append(final)
    for _ in range(int(fps * hold_seconds)):
        all_frames.append(final)

    # ---- Week-label boundaries ----
    week_boundaries = [0]
    cumulative = 0
    for _ in range(len(weeks) - 1):
        cumulative += frames_per_week
        week_boundaries.append(cumulative)

    # ---- Write with OpenCV ----
    frame_size = all_frames[0].size          # (W, H)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    font = cv2.FONT_HERSHEY_SIMPLEX
    current_week = weeks[0]

    for idx, frame in enumerate(all_frames):
        bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

        # Determine which week label to show
        for j, boundary in enumerate(week_boundaries):
            if idx >= boundary:
                current_week = weeks[j]

        # Week label with white background box
        week_text = f"Week {current_week}"
        (tw, th), _ = cv2.getTextSize(week_text, font, 1.5, 3)
        cv2.rectangle(bgr, (10, 10), (tw + 30, th + 30), (255, 255, 255), -1)
        cv2.putText(bgr, week_text, (20, th + 20), font, 1.5, (0, 0, 0), 3)

        # Crop + treatment info
        info = f"{crop_name}  |  {treatment_name}"
        cv2.putText(bgr, info, (20, th + 60), font, 0.65, (20, 20, 20), 2)

        writer.write(bgr)

    writer.release()

    duration = len(all_frames) / fps
    print(f"✓ Video saved: {output_path}  ({duration:.1f}s)")
    return duration


# --------------------------------------------------------------------------- #
# Batch helper
# --------------------------------------------------------------------------- #

def create_all_videos(
    all_results: dict,
    output_dir: str,
    weeks: list[int] = None,
    frames_per_week: int = 30,
) -> list[dict]:
    """
    Generate one MP4 per (disease × spray_level) combination.

    Args:
        all_results:   ``{disease_name: {"results": {...}, "params": {...}}}``
                       as produced by the progression pipeline.
        output_dir:    Directory to write MP4 files into.
        weeks:         Week list (default ``[0, 1, 2, 4]``).
        frames_per_week: Interpolation frames per weekly interval.

    Returns:
        List of dicts with keys ``crop``, ``spray_level``, ``filename``,
        ``path``, ``duration``.
    """
    if weeks is None:
        weeks = [0, 1, 2, 4]

    os.makedirs(output_dir, exist_ok=True)
    spray_levels = ["none", "light", "full"]
    spray_labels = {
        "none": "No Treatment",
        "light": "Light Spray (70–75% efficacy)",
        "full": "Full Treatment (88–95% efficacy)",
    }
    video_list: list[dict] = []

    for disease_name, data in sorted(
        all_results.items(),
        key=lambda x: x[1]["params"]["spread_rate_per_week"],
        reverse=True,
    ):
        crop_display = disease_name.replace("___", " - ").replace("_", " ")
        crop_short = disease_name.split("___")[0]
        disease_short = disease_name.split("___")[1]

        for spray in spray_levels:
            filename = f"{crop_short}_{disease_short}_{spray}.mp4"
            path = os.path.join(output_dir, filename)

            duration = create_progression_video(
                progression_images=data["results"]["rgb"][spray],
                weeks=weeks,
                output_path=path,
                crop_name=crop_display,
                treatment_name=spray_labels[spray],
                frames_per_week=frames_per_week,
            )
            video_list.append({
                "crop": crop_display,
                "spray_level": spray_labels[spray],
                "filename": filename,
                "path": path,
                "duration": duration,
            })

    return video_list
