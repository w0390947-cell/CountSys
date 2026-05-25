#!/usr/bin/env python3
"""Extract video frames into a YOLO dataset skeleton for manual labeling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


CLASS_NAMES = ["part_end_cap"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames for manual YOLO labeling.")
    parser.add_argument("--video", default="b7763a0682d156294de373ad97e2c544.mp4")
    parser.add_argument("--output-dir", default="datasets/end_cap_single_video")
    parser.add_argument("--frame-stride", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = extract_frames(args)
    print(f"dataset_dir={summary['dataset_dir']}")
    print(f"sampled_frame_count={summary['sampled_frame_count']}")
    print(f"train_frame_count={summary['train_frame_count']}")
    print(f"val_frame_count={summary['val_frame_count']}")
    print(f"data_yaml={summary['data_yaml']}")


def extract_frames(args: argparse.Namespace) -> dict:
    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    image_train_dir = output_dir / "images" / "train"
    image_val_dir = output_dir / "images" / "val"
    label_train_dir = output_dir / "labels" / "train"
    label_val_dir = output_dir / "labels" / "val"
    for path in [image_train_dir, image_val_dir, label_train_dir, label_val_dir]:
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
    train_count = 0
    val_count = 0

    for frame_index, frame_number in enumerate(sampled_numbers):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, image = capture.read()
        if not ok:
            continue

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
        label_path.touch(exist_ok=True)

        frame_records.append(
            {
                "frame_index": frame_index,
                "frame_number": frame_number,
                "split": split,
                "image_path": str(image_path),
                "label_path": str(label_path),
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
        },
        "sampled_frame_count": len(frame_records),
        "train_frame_count": train_count,
        "val_frame_count": val_count,
        "class_names": CLASS_NAMES,
        "frames": frame_records,
        "notes": [
            "This script only extracts frames and creates empty YOLO label files.",
            "Fill labels manually before training. Empty label files mean no objects in that image.",
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


if __name__ == "__main__":
    main()
