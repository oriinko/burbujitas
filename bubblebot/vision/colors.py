from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np


class CircularCandidate(Protocol):
    x: float
    y: float
    radius: float
    candidate_id: str


@dataclass(frozen=True)
class ColorFeature:
    candidate_id: str
    lab: tuple[float, float, float]


@dataclass
class ColorClassificationResult:
    assignment: dict[str, str] = field(default_factory=dict)
    centroids: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)
    cluster_spread: dict[str, float] = field(default_factory=dict)
    cluster_count: int = 0
    nearest_separation: float = 0.0
    overall_confidence: float = 0.0


def extract_color_feature(image: np.ndarray, candidate: CircularCandidate) -> ColorFeature:
    """Extract robust median Lab from the inner disk, excluding highlights/shadows."""
    height, width = image.shape[:2]
    radius = float(candidate.radius)
    x0 = max(0, int(np.floor(candidate.x - radius)))
    x1 = min(width, int(np.ceil(candidate.x + radius + 1)))
    y0 = max(0, int(np.floor(candidate.y - radius)))
    y1 = min(height, int(np.ceil(candidate.y + radius + 1)))
    patch = image[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    interior = (xx - candidate.x) ** 2 + (yy - candidate.y) ** 2 <= (radius * 0.68) ** 2
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(float)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    valid = interior & (lab[..., 0] > 25) & (lab[..., 0] < 238) & (hsv[..., 1] > 35)
    hsv_pixels = hsv[valid]
    if len(hsv_pixels) < 5:
        hsv_pixels = hsv[interior]
    if not len(hsv_pixels):
        raise ValueError(f"candidate {candidate.candidate_id!r} has no sampleable pixels")
    representative_hsv = np.median(hsv_pixels, axis=0).astype(np.uint8)
    representative_hsv[2] = 200
    normalized_bgr = cv2.cvtColor(np.uint8([[representative_hsv]]), cv2.COLOR_HSV2BGR)
    representative = cv2.cvtColor(normalized_bgr, cv2.COLOR_BGR2LAB)[0, 0]
    return ColorFeature(candidate.candidate_id, tuple(float(value) for value in representative))


def _complete_link_distance(left: list[int], right: list[int], values: np.ndarray) -> float:
    return max(float(np.linalg.norm(values[a] - values[b])) for a in left for b in right)


def cluster_colors(features: list[ColorFeature], max_diameter: float = 50.0) -> ColorClassificationResult:
    """Cluster Lab features with deterministic complete linkage and bounded diameter."""
    result = ColorClassificationResult()
    if not features:
        return result
    ordered = sorted(features, key=lambda item: (item.lab, item.candidate_id))
    values = np.asarray([item.lab for item in ordered], dtype=float)
    clusters: list[list[int]] = [[index] for index in range(len(ordered))]
    while len(clusters) > 1:
        choices = []
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                distance = _complete_link_distance(clusters[left], clusters[right], values)
                if distance <= max_diameter:
                    choices.append((distance, left, right))
        if not choices:
            break
        _, left, right = min(choices)
        clusters[left] = clusters[left] + clusters[right]
        del clusters[right]

    cluster_data = []
    for members in clusters:
        centroid = np.median(values[members], axis=0)
        spread = max(float(np.linalg.norm(values[index] - centroid)) for index in members)
        cluster_data.append((tuple(float(v) for v in centroid), members, spread))
    cluster_data.sort(key=lambda item: item[0])

    centroids = [np.asarray(item[0]) for item in cluster_data]
    separations = [float(np.linalg.norm(a - b)) for i, a in enumerate(centroids) for b in centroids[i + 1 :]]
    nearest_separation = min(separations) if separations else float("inf")
    candidate_confidences = []
    for cluster_index, (centroid_tuple, members, spread) in enumerate(cluster_data):
        color_id = f"COLOR_{cluster_index}"
        result.centroids[color_id] = centroid_tuple
        result.cluster_spread[color_id] = spread
        centroid = np.asarray(centroid_tuple)
        for member in members:
            own_distance = float(np.linalg.norm(values[member] - centroid))
            competing = min((float(np.linalg.norm(values[member] - other)) for i, other in enumerate(centroids) if i != cluster_index), default=max_diameter * 2)
            confidence = max(0.0, min(1.0, 1.0 - own_distance / max(competing, 1e-6)))
            candidate_id = ordered[member].candidate_id
            result.assignment[candidate_id] = color_id
            result.confidence[candidate_id] = confidence
            candidate_confidences.append(confidence)
    result.cluster_count = len(cluster_data)
    result.nearest_separation = nearest_separation
    separation_factor = 1.0 if not np.isfinite(nearest_separation) else min(1.0, nearest_separation / max(max_diameter, 1.0))
    result.overall_confidence = float(np.mean(candidate_confidences)) * separation_factor
    return result



