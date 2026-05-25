from __future__ import annotations

import math

import cv2
import numpy as np


def blur_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def robust_best_view_count(frame_counts: list[int]) -> int:
    if not frame_counts:
        return 0
    top_n = max(3, min(5, math.ceil(len(frame_counts) * 0.15)))
    return int(round(float(np.median(sorted(frame_counts)[-top_n:]))))
