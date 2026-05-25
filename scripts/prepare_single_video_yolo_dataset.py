#!/usr/bin/env python3
"""Build a YOLO dataset from one walkaround video using the current detector.

This script is intentionally a bootstrap tool. It uses the rule-based detector
from count_cylinder_parts_demo.py to create pseudo labels, so the first trained
model validates the training/inference framework rather than true generalization.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MPL_DIR = PROJECT_ROOT / "model_runs" / ".matplotlib"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

from count_cylinder_parts_demo import Detection, detect_end_caps, draw_annotated_frame


CLASS_NAMES = ["part_end_cap"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a single-video YOLO dataset.")
    parser.add_argument(
        "--video",
        default="b7763a0682d156294de373ad97e2c544.mp4",
        help="Input training video.",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/end_cap_single_video",
        help="Output YOLO dataset directory.",
    )
    parser.add_argument("--frame-stride", type=int, default=12, help="Sample every N frames.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means no cap.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument(
        "--bbox-scale",
        type=float,
        default=2.25,
        help="Box side length multiplier relative to detected radius.",
    )
    parser.add_argument(
        "--include-background-piles",
        action="store_true",
        help="Include all large bamboo/cylinder piles in pseudo labels.",
    )
    parser.add_argument(
        "--max-center-std",
        type=float,
        default=43.0,
        help="Same meaning as count_cylinder_parts_demo.py.",
    )
    parser.add_argument(
        "--min-ring-bamboo",
        type=float,
        default=0.17,
        help="Same meaning as count_cylinder_parts_demo.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_dataset(args)
    print(f"dataset_dir={summary['dataset_dir']}")
    print(f"sampled_frame_count={summary['sampled_frame_count']}")
    print(f"train_frame_count={summary['train_frame_count']}")
    print(f"val_frame_count={summary['val_frame_count']}")
    print(f"total_pseudo_labels={summary['total_pseudo_labels']}")
    print(f"data_yaml={summary['data_yaml']}")


def prepare_dataset(args: argparse.Namespace) -> dict:
    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    image_train_dir = output_dir / "images" / "train"
    image_val_dir = output_dir / "images" / "val"
    label_train_dir = output_dir / "labels" / "train"
    label_val_dir = output_dir / "labels" / "val"
    preview_dir = output_dir / "previews"
    for path in [image_train_dir, image_val_dir, label_train_dir, label_val_dir, preview_dir]:
        path.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    sampled_numbers = list(range(0, total_frames, max(1, int(args.frame_stride))))
    if args.max_frames > 0:
        sampled_numbers = sampled_numbers[: args.max_frames]

    val_every = ratio_to_period(float(args.val_ratio))
    frame_records = []
    total_labels = 0
    train_count = 0
    val_count = 0

    for frame_index, frame_number in enumerate(sampled_numbers):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, image = capture.read()
        if not ok:
            continue

        detections = detect_end_caps(
            image=image,
            frame_index=frame_index,
            frame_number=frame_number,
            include_background_piles=bool(args.include_background_piles),
            max_center_std=float(args.max_center_std),
            min_ring_bamboo=float(args.min_ring_bamboo),
        )
        split = "val" if val_every > 0 and frame_index % val_every == 0 else "train"
        image_dir = image_val_dir if split == "val" else image_train_dir
        label_dir = label_val_dir if split == "val" else label_train_dir
        if split == "val":
            val_count += 1
        else:
            train_count += 1

        stem = f"frame_{frame_number:05d}"
        image_path = image_dir / f"{stem}.jpg"
        label_path = label_dir / f"{stem}.txt"
        cv2.imwrite(str(image_path), image)
        label_lines = [
            detection_to_yolo_line(item, width, height, float(args.bbox_scale)) for item in detections
        ]
        label_path.write_text("".join(label_lines), encoding="utf-8")

        if frame_index < 8 or len(detections) >= 50:
            preview_path = preview_dir / f"{stem}.jpg"
            draw_annotated_frame(image, detections, preview_path, f"frame={frame_number} labels={len(detections)}")

        total_labels += len(detections)
        frame_records.append(
            {
                "frame_index": frame_index,
                "frame_number": frame_number,
                "split": split,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "pseudo_label_count": len(detections),
                "detections": [asdict(item) for item in detections],
            }
        )

    capture.release()
    if not frame_records:
        raise RuntimeError("No frames could be sampled from the video.")

    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                f"nc: {len(CLASS_NAMES)}",
                "names:",
                *[f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES)],
                "",
            ]
        ),
        encoding="utf-8",
    )

    summary = {
        "video": str(video_path),
        "dataset_dir": str(output_dir),
        "data_yaml": str(data_yaml),
        "video_info": {
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames,
            "duration_sec": float(total_frames / fps) if fps > 0 else None,
        },
        "config": {
            "frame_stride": int(args.frame_stride),
            "max_frames": int(args.max_frames),
            "val_ratio": float(args.val_ratio),
            "bbox_scale": float(args.bbox_scale),
            "include_background_piles": bool(args.include_background_piles),
            "max_center_std": float(args.max_center_std),
            "min_ring_bamboo": float(args.min_ring_bamboo),
        },
        "sampled_frame_count": len(frame_records),
        "train_frame_count": train_count,
        "val_frame_count": val_count,
        "total_pseudo_labels": total_labels,
        "class_names": CLASS_NAMES,
        "frames": frame_records,
        "notes": [
            "Labels are pseudo labels generated by the existing rule-based detector.",
            "This dataset is intended to validate the model pipeline on one video, not model generalization.",
        ],
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def ratio_to_period(ratio: float) -> int:
    if ratio <= 0:
        return 0
    if ratio >= 1:
        return 1
    return max(2, round(1.0 / ratio))


def detection_to_yolo_line(detection: Detection, width: int, height: int, bbox_scale: float) -> str:
    half_side = max(1.0, detection.radius * bbox_scale / 2.0)
    x1 = max(0.0, detection.x - half_side)
    y1 = max(0.0, detection.y - half_side)
    x2 = min(float(width - 1), detection.x + half_side)
    y2 = min(float(height - 1), detection.y + half_side)

    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    cx = x1 + box_w / 2.0
    cy = y1 + box_h / 2.0
    return f"0 {cx / width:.6f} {cy / height:.6f} {box_w / width:.6f} {box_h / height:.6f}\n"


if __name__ == "__main__":
    main()
