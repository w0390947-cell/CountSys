#!/usr/bin/env python3
"""Count end caps in a video using a trained YOLO model."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MPL_DIR = PROJECT_ROOT / "model_runs" / ".matplotlib"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

from counting.yolo_counter import count_video_with_yolo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO counting on a walkaround video.")
    parser.add_argument("--video", default="b7763a0682d156294de373ad97e2c544.mp4")
    parser.add_argument(
        "--weights",
        default="model_runs/end_cap_single_video/weights/best.pt",
        help="Trained YOLO weights.",
    )
    parser.add_argument("--output-dir", default="model_count_output")
    parser.add_argument("--frame-stride", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = count_video_with_yolo(args)
    print(f"estimated_count={result['estimated_count']}")
    print(f"best_view_sanity_count={result['best_view_sanity_count']}")
    print(f"total_raw_detections={result['total_raw_detections']}")
    print(f"sampled_frame_count={result['sampled_frame_count']}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
