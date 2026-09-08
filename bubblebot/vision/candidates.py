from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .isolation import CircleCandidate


@dataclass(frozen=True)
class CandidateDetectionResult:
    candidates: list[CircleCandidate]
    confidence: float


def detect_circle_candidates(image: np.ndarray) -> CandidateDetectionResult:
    """Detect saturated, approximately circular bubble-sized image regions."""
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 45, 35]), np.array([179, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw: list[tuple[float, float, float, float]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        (x, y), radius = cv2.minEnclosingCircle(contour)
        circularity = area / max(np.pi * radius * radius, 1.0)
        if area < 90 or radius < 6 or radius > min(width, height) * 0.08 or circularity < 0.42:
            continue
        raw.append((x, y, radius, circularity))
    raw.sort(key=lambda item: (item[1], item[0]))
    candidates = [CircleCandidate(float(x), float(y), float(radius), f"candidate_{index:04d}") for index, (x, y, radius, _) in enumerate(raw)]
    confidence = float(np.median([item[3] for item in raw])) if raw else 0.0
    return CandidateDetectionResult(candidates, max(0.0, min(1.0, confidence)))
