#!/usr/bin/env python3
"""Count end caps in a video using a trained YOLO model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MPL_DIR = PROJECT_ROOT / "model_runs" / ".matplotlib"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

from count_cylinder_parts_demo import blur_score, robust_best_view_count


@dataclass
class YoloDetection:
    frame_index: int
    frame_number: int
    class_id: int
    class_name: str
    x1: float
    y1: float
    x2: float
    y2: float
    score: float


@dataclass
class YoloFrameResult:
    frame_index: int
    frame_number: int
    timestamp_sec: float
    image_path: str
    annotated_path: str
    detection_count: int
    blur_score: float


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
    result = count_video(args)
    print(f"estimated_count={result['estimated_count']}")
    print(f"best_view_sanity_count={result['best_view_sanity_count']}")
    print(f"total_raw_detections={result['total_raw_detections']}")
    print(f"sampled_frame_count={result['sampled_frame_count']}")
    print(f"output_dir={args.output_dir}")


def count_video(args: argparse.Namespace) -> dict:
    video_path = Path(args.video)
    weights_path = Path(args.weights)
    output_dir = Path(args.output_dir)
    keyframe_dir = output_dir / "keyframes"
    annotated_dir = output_dir / "annotated"
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights_path}. Run scripts/train_single_video_yolo.py first."
        )

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: ultralytics. Install project requirements first, for example: "
            "python3 -m pip install -r requirements.txt"
        ) from exc

    model = YOLO(str(weights_path))
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

    frames: list[YoloFrameResult] = []
    detections: list[YoloDetection] = []

    predict_args = {
        "conf": float(args.conf),
        "iou": float(args.iou),
        "imgsz": int(args.imgsz),
        "verbose": False,
    }
    if args.device:
        predict_args["device"] = args.device

    for frame_index, frame_number in enumerate(sampled_numbers):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, image = capture.read()
        if not ok:
            continue

        result = model.predict(source=image, **predict_args)[0]
        frame_detections = parse_yolo_result(result, frame_index, frame_number)
        detections.extend(frame_detections)

        keyframe_path = keyframe_dir / f"frame_{frame_number:05d}.jpg"
        annotated_path = annotated_dir / f"annotated_{frame_number:05d}.jpg"
        cv2.imwrite(str(keyframe_path), image)
        cv2.imwrite(str(annotated_path), draw_boxes(image, frame_detections))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frames.append(
            YoloFrameResult(
                frame_index=frame_index,
                frame_number=frame_number,
                timestamp_sec=float(frame_number / fps) if fps > 0 else 0.0,
                image_path=str(keyframe_path),
                annotated_path=str(annotated_path),
                detection_count=len(frame_detections),
                blur_score=blur_score(gray),
            )
        )

    capture.release()
    if not frames:
        raise RuntimeError("No frames could be sampled from the video.")

    frame_counts = [item.detection_count for item in frames]
    best_view_count = robust_best_view_count(frame_counts)
    result = {
        "video": str(video_path),
        "weights": str(weights_path),
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
            "conf": float(args.conf),
            "iou": float(args.iou),
            "imgsz": int(args.imgsz),
        },
        "estimated_count": int(best_view_count),
        "estimate_policy": "median of the strongest YOLO keyframes after per-frame NMS",
        "best_view_sanity_count": int(best_view_count),
        "total_raw_detections": len(detections),
        "sampled_frame_count": len(frames),
        "per_frame": [asdict(item) for item in frames],
        "detections": [asdict(item) for item in detections],
        "notes": [
            "This count uses a trained YOLO model and the same best-view fusion policy as the demo.",
            "If trained on one video, this result validates the framework but does not prove generalization.",
        ],
    }
    (output_dir / "count_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def parse_yolo_result(result, frame_index: int, frame_number: int) -> list[YoloDetection]:
    names = result.names or {}
    parsed: list[YoloDetection] = []
    if result.boxes is None:
        return parsed

    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    for box, score, class_id in zip(xyxy, scores, classes):
        x1, y1, x2, y2 = [float(value) for value in box]
        parsed.append(
            YoloDetection(
                frame_index=frame_index,
                frame_number=frame_number,
                class_id=int(class_id),
                class_name=str(names.get(int(class_id), class_id)),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                score=float(score),
            )
        )
    return parsed


def draw_boxes(image, detections: list[YoloDetection]):
    canvas = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in [detection.x1, detection.y1, detection.x2, detection.y2]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{detection.class_name} {detection.score:.2f}"
        cv2.putText(canvas, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    cv2.putText(
        canvas,
        f"YOLO detections={len(detections)}",
        (18, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


if __name__ == "__main__":
    main()
