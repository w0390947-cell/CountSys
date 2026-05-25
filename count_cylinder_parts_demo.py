#!/usr/bin/env python3
"""
Demo for approximate counting of bamboo-wrapped cylindrical parts in a walkaround
video.

The pipeline mirrors a multi-view counting system:
1. sample keyframes from the video;
2. detect visible circular/elliptical metal end caps;
3. estimate adjacent camera motion with ORB + homography;
4. project detections into a shared reference plane;
5. cluster projected detections to remove repeated observations.

This is intentionally a demo tuned for b7763a0682d156294de373ad97e2c544.mp4.
It is not a replacement for a trained detector or full SfM/SLAM reconstruction.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN


@dataclass
class Detection:
    frame_index: int
    frame_number: int
    x: float
    y: float
    radius: float
    score: float
    projected_x: float | None = None
    projected_y: float | None = None
    cluster_id: int | None = None


@dataclass
class FrameResult:
    frame_index: int
    frame_number: int
    timestamp_sec: float
    image_path: str
    detection_count: int
    blur_score: float
    pose_ok: bool
    pose_inliers: int
    pose_matches: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Approximate multi-view counting demo for bamboo-wrapped cylindrical parts."
    )
    parser.add_argument(
        "--video",
        default="b7763a0682d156294de373ad97e2c544.mp4",
        help="Input video path.",
    )
    parser.add_argument(
        "--output-dir",
        default="count_demo_output",
        help="Directory for JSON and visualization outputs.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=12,
        help="Sample every N frames. Lower values use more frames and run slower.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional cap on sampled keyframes. 0 means no cap.",
    )
    parser.add_argument(
        "--cluster-eps",
        type=float,
        default=0.0,
        help="DBSCAN radius in projected pixels. 0 uses an adaptive radius.",
    )
    parser.add_argument(
        "--include-background-piles",
        action="store_true",
        help="Count every large bamboo/cylinder pile instead of the dominant foreground pile.",
    )
    parser.add_argument(
        "--save-every-annotated-frame",
        action="store_true",
        help="Save annotation images for every sampled keyframe. By default saves a small subset.",
    )
    parser.add_argument(
        "--max-center-std",
        type=float,
        default=43.0,
        help="Maximum gray-value standard deviation inside a candidate end cap. Lower values are stricter.",
    )
    parser.add_argument(
        "--min-ring-bamboo",
        type=float,
        default=0.17,
        help="Minimum bamboo-colored context around a candidate end cap.",
    )
    return parser.parse_args()


def ensure_clean_output(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "keyframes").mkdir(exist_ok=True)
    (path / "annotated").mkdir(exist_ok=True)


def blur_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def target_region_mask(image: np.ndarray, include_background_piles: bool) -> np.ndarray:
    """Build a rough ROI from bamboo-colored pixels.

    The sample video contains many gravel stones that look like metal end caps.
    Requiring each candidate to be inside a bamboo-rich connected component removes
    most of those false positives. The default keeps the dominant lower component,
    which corresponds to the foreground pile in the sample video.
    """
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)

    bamboo = ((hue >= 7) & (hue <= 35) & (sat > 35) & (val > 65)).astype(np.uint8) * 255
    bamboo = cv2.morphologyEx(
        bamboo,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (25, 11)),
        iterations=2,
    )
    bamboo = cv2.dilate(
        bamboo,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)),
        iterations=1,
    )

    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(bamboo, 8)
    components: list[tuple[float, int]] = []
    for component_id in range(1, component_count):
        area = float(stats[component_id, cv2.CC_STAT_AREA])
        if area < 2500:
            continue
        cy = float(centroids[component_id][1])
        foreground_bonus = 1.0 + max(0.0, cy / height - 0.35) * 2.0
        components.append((area * foreground_bonus, component_id))

    roi = np.zeros((height, width), np.uint8)
    components.sort(reverse=True)
    selected = components if include_background_piles else components[:1]
    for _, component_id in selected:
        roi[labels == component_id] = 255

    roi = cv2.morphologyEx(
        roi,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
        iterations=1,
    )
    return roi


def detect_end_caps(
    image: np.ndarray,
    frame_index: int,
    frame_number: int,
    include_background_piles: bool,
    max_center_std: float,
    min_ring_bamboo: float,
) -> list[Detection]:
    """Detect visible gray metal end caps using Hough circles plus local context."""
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    roi = target_region_mask(image, include_background_piles)

    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.medianBlur(enhanced, 5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=18,
        param1=100,
        param2=23,
        minRadius=8,
        maxRadius=42,
    )
    if circles is None:
        return []

    candidates: list[Detection] = []
    for x_raw, y_raw, radius_raw in np.round(circles[0]).astype(int):
        x = int(x_raw)
        y = int(y_raw)
        radius = int(radius_raw)
        margin = radius * 2
        if x - margin < 0 or y - margin < 0 or x + margin >= width or y + margin >= height:
            continue
        if roi[y, x] == 0:
            continue

        x0 = max(0, x - margin)
        x1 = min(width, x + margin + 1)
        y0 = max(0, y - margin)
        y1 = min(height, y + margin + 1)

        yy, xx = np.ogrid[y0:y1, x0:x1]
        distance = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        inner = distance <= radius * 0.62
        ring = (distance >= radius * 0.95) & (distance <= radius * 1.8)

        patch_hsv = hsv[y0:y1, x0:x1]
        patch_gray = gray[y0:y1, x0:x1]
        inner_sat = patch_hsv[:, :, 1][inner]
        inner_val = patch_hsv[:, :, 2][inner]
        ring_hue = patch_hsv[:, :, 0][ring]
        ring_sat = patch_hsv[:, :, 1][ring]
        ring_val = patch_hsv[:, :, 2][ring]

        center_gray_ratio = float(np.mean((inner_sat < 70) & (inner_val > 115)))
        center_value = float(np.mean(inner_val))
        center_std = float(np.std(patch_gray[inner]))
        ring_bamboo_ratio = float(
            np.mean((ring_hue >= 7) & (ring_hue <= 38) & (ring_sat > 35) & (ring_val > 65))
        )
        ring_dark_ratio = float(np.mean(ring_val < 90))

        if center_gray_ratio <= 0.62:
            continue
        if center_value <= 120 or center_std >= max_center_std:
            continue
        if ring_bamboo_ratio <= min_ring_bamboo or ring_dark_ratio <= 0.03:
            continue

        score = center_gray_ratio + ring_bamboo_ratio - center_std / 120.0
        candidates.append(
            Detection(
                frame_index=frame_index,
                frame_number=frame_number,
                x=float(x),
                y=float(y),
                radius=float(radius),
                score=float(score),
            )
        )

    return non_max_suppression(candidates)


def non_max_suppression(candidates: Iterable[Detection]) -> list[Detection]:
    final: list[Detection] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        keep = True
        for existing in final:
            distance_sq = (candidate.x - existing.x) ** 2 + (candidate.y - existing.y) ** 2
            min_distance = max(candidate.radius, existing.radius)
            if distance_sq <= min_distance**2:
                keep = False
                break
        if keep:
            final.append(candidate)
    return final


def estimate_homography(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    orb: cv2.ORB,
    matcher: cv2.BFMatcher,
) -> tuple[np.ndarray, bool, int, int]:
    keypoints_prev, desc_prev = orb.detectAndCompute(previous_gray, None)
    keypoints_curr, desc_curr = orb.detectAndCompute(current_gray, None)
    if desc_prev is None or desc_curr is None:
        return np.eye(3), False, 0, 0
    if len(keypoints_prev) < 20 or len(keypoints_curr) < 20:
        return np.eye(3), False, 0, 0

    raw_matches = matcher.knnMatch(desc_prev, desc_curr, k=2)
    good_matches = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        first, second = pair
        if first.distance < 0.75 * second.distance:
            good_matches.append(first)

    if len(good_matches) < 12:
        return np.eye(3), False, len(good_matches), 0

    points_prev = np.float32([keypoints_prev[match.queryIdx].pt for match in good_matches])
    points_curr = np.float32([keypoints_curr[match.trainIdx].pt for match in good_matches])
    homography, inlier_mask = cv2.findHomography(points_curr, points_prev, cv2.RANSAC, 4.0)
    if homography is None or inlier_mask is None:
        return np.eye(3), False, len(good_matches), 0

    inliers = int(inlier_mask.sum())
    if inliers < 10:
        return np.eye(3), False, len(good_matches), inliers

    return homography, True, len(good_matches), inliers


def project_detections(
    detections_by_frame: list[list[Detection]],
    transforms: list[np.ndarray],
) -> list[Detection]:
    projected: list[Detection] = []
    for frame_detections, transform in zip(detections_by_frame, transforms):
        for detection in frame_detections:
            point = transform @ np.array([detection.x, detection.y, 1.0], dtype=np.float64)
            if abs(float(point[2])) < 1e-6:
                continue
            px = float(point[0] / point[2])
            py = float(point[1] / point[2])
            if not (-1000 <= px <= 2000 and -1000 <= py <= 2000):
                continue
            detection.projected_x = px
            detection.projected_y = py
            projected.append(detection)
    return projected


def cluster_projected_detections(
    detections: list[Detection],
    cluster_eps: float,
) -> tuple[int, float]:
    if not detections:
        return 0, cluster_eps

    points = np.array([[item.projected_x, item.projected_y] for item in detections], dtype=np.float64)
    radii = np.array([item.radius for item in detections], dtype=np.float64)
    if cluster_eps <= 0:
        cluster_eps = float(np.clip(np.median(radii) * 2.35, 38.0, 60.0))

    labels = DBSCAN(eps=cluster_eps, min_samples=1).fit_predict(points)
    for detection, label in zip(detections, labels):
        detection.cluster_id = int(label)

    return len(set(int(label) for label in labels)), cluster_eps


def robust_best_view_count(frame_counts: list[int]) -> int:
    if not frame_counts:
        return 0
    top_n = max(3, min(5, math.ceil(len(frame_counts) * 0.15)))
    return int(round(float(np.median(sorted(frame_counts)[-top_n:]))))


def draw_annotated_frame(
    image: np.ndarray,
    detections: list[Detection],
    output_path: Path,
    count_text: str,
) -> None:
    canvas = image.copy()
    for detection in detections:
        color = (0, 0, 255)
        center = (int(round(detection.x)), int(round(detection.y)))
        radius = int(round(detection.radius))
        cv2.circle(canvas, center, radius, color, 2)
    cv2.putText(
        canvas,
        count_text,
        (18, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output_path), canvas)


def save_projection_plot(detections: list[Detection], output_path: Path) -> None:
    if not detections:
        return
    points = np.array([[item.projected_x, item.projected_y] for item in detections], dtype=np.float64)
    labels = np.array([item.cluster_id if item.cluster_id is not None else -1 for item in detections])
    plt.figure(figsize=(8, 7))
    plt.scatter(points[:, 0], points[:, 1], c=labels, s=8, cmap="tab20", alpha=0.75)
    plt.gca().invert_yaxis()
    plt.title("Projected end-cap detections after multi-view alignment")
    plt.xlabel("Projected x")
    plt.ylabel("Projected y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def sample_and_count(args: argparse.Namespace) -> dict:
    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    ensure_clean_output(output_dir)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_images: list[np.ndarray] = []
    frame_grays: list[np.ndarray] = []
    frame_results: list[FrameResult] = []
    detections_by_frame: list[list[Detection]] = []

    sampled_numbers = list(range(0, total_frames, max(1, args.frame_stride)))
    if args.max_frames > 0:
        sampled_numbers = sampled_numbers[: args.max_frames]

    for frame_index, frame_number in enumerate(sampled_numbers):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, image = capture.read()
        if not ok:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = detect_end_caps(
            image,
            frame_index=frame_index,
            frame_number=frame_number,
            include_background_piles=bool(args.include_background_piles),
            max_center_std=float(args.max_center_std),
            min_ring_bamboo=float(args.min_ring_bamboo),
        )

        keyframe_name = f"frame_{frame_number:05d}.jpg"
        keyframe_path = output_dir / "keyframes" / keyframe_name
        cv2.imwrite(str(keyframe_path), image)

        frame_images.append(image)
        frame_grays.append(gray)
        detections_by_frame.append(detections)
        frame_results.append(
            FrameResult(
                frame_index=frame_index,
                frame_number=frame_number,
                timestamp_sec=float(frame_number / fps) if fps > 0 else 0.0,
                image_path=str(keyframe_path),
                detection_count=len(detections),
                blur_score=blur_score(gray),
                pose_ok=(frame_index == 0),
                pose_inliers=0,
                pose_matches=0,
            )
        )

    capture.release()
    if not frame_images:
        raise RuntimeError("No frames could be sampled from the video.")

    orb = cv2.ORB_create(2500)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    transforms: list[np.ndarray] = [np.eye(3)]

    for index in range(1, len(frame_grays)):
        homography, pose_ok, matches, inliers = estimate_homography(
            frame_grays[index - 1],
            frame_grays[index],
            orb,
            matcher,
        )
        transforms.append(transforms[-1] @ homography)
        frame_results[index].pose_ok = pose_ok
        frame_results[index].pose_matches = matches
        frame_results[index].pose_inliers = inliers

    projected = project_detections(detections_by_frame, transforms)
    cluster_count, used_eps = cluster_projected_detections(projected, args.cluster_eps)
    frame_counts = [item.detection_count for item in frame_results]
    best_view_count = robust_best_view_count(frame_counts)

    # A pure homography is only a planar approximation. In this walkaround video
    # the pile has strong 3D parallax, so the global projected clusters are kept
    # as a diagnostic rather than treated as the final inventory count.
    estimated_count = int(best_view_count)

    annotated_indices: set[int]
    if args.save_every_annotated_frame:
        annotated_indices = set(range(len(frame_images)))
    else:
        best_indices = sorted(range(len(frame_counts)), key=lambda idx: frame_counts[idx], reverse=True)[:5]
        annotated_indices = set(best_indices + [0, len(frame_images) // 2, len(frame_images) - 1])

    for index in sorted(annotated_indices):
        frame_number = frame_results[index].frame_number
        output_path = output_dir / "annotated" / f"annotated_{frame_number:05d}.jpg"
        draw_annotated_frame(
            frame_images[index],
            detections_by_frame[index],
            output_path,
            count_text=f"frame={frame_number} caps={len(detections_by_frame[index])}",
        )

    save_projection_plot(projected, output_dir / "projected_clusters.png")

    result = {
        "video": str(video_path),
        "video_info": {
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames,
            "duration_sec": float(total_frames / fps) if fps > 0 else None,
        },
        "config": {
            "frame_stride": args.frame_stride,
            "max_frames": args.max_frames,
            "include_background_piles": bool(args.include_background_piles),
            "cluster_eps": used_eps,
            "max_center_std": float(args.max_center_std),
            "min_ring_bamboo": float(args.min_ring_bamboo),
        },
        "estimated_count": estimated_count,
        "estimate_policy": "median of the strongest multi-view keyframes after per-frame de-duplication",
        "global_projection_cluster_count_diagnostic": cluster_count,
        "best_view_sanity_count": best_view_count,
        "total_raw_detections": len(projected),
        "sampled_frame_count": len(frame_results),
        "pose_ok_count": sum(1 for item in frame_results if item.pose_ok),
        "per_frame": [asdict(item) for item in frame_results],
        "detections": [asdict(item) for item in projected],
        "notes": [
            "This demo counts visible metal end caps and deduplicates them by approximate homography projection.",
            "For production, replace the rule-based detector with a trained detector/segmenter and replace homographies with SfM/SLAM.",
            "The default ROI targets the dominant foreground pile in the sample video.",
        ],
    }

    summary_path = output_dir / "count_result.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    result = sample_and_count(args)
    print(f"estimated_count={result['estimated_count']}")
    print(f"global_projection_cluster_count_diagnostic={result['global_projection_cluster_count_diagnostic']}")
    print(f"best_view_sanity_count={result['best_view_sanity_count']}")
    print(f"total_raw_detections={result['total_raw_detections']}")
    print(f"sampled_frame_count={result['sampled_frame_count']}")
    print(f"pose_ok_count={result['pose_ok_count']}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
