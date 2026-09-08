from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .isolation import CircleCandidate


@dataclass(frozen=True)
class CandidateDetectionResult:
    candidates: list[CircleCandidate]
    confidence: float
    raw_proposal_count: int = 0
    dominant_radius: float = 0.0
    support_by_candidate: dict[str, float] = field(default_factory=dict)
    radius_distribution: dict[str, int] = field(default_factory=dict)


def _circle_support(gray: np.ndarray, x: float, y: float, radius: float) -> float:
    """Measure normalized radial edge contrast without assuming bubble chroma."""
    height, width = gray.shape
    angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    inner_values = []
    outer_values = []
    for angle in angles:
        inner_x = int(round(x + np.cos(angle) * radius * 0.78))
        inner_y = int(round(y + np.sin(angle) * radius * 0.78))
        outer_x = int(round(x + np.cos(angle) * radius * 1.08))
        outer_y = int(round(y + np.sin(angle) * radius * 1.08))
        if 0 <= inner_x < width and 0 <= inner_y < height and 0 <= outer_x < width and 0 <= outer_y < height:
            inner_values.append(float(gray[inner_y, inner_x]))
            outer_values.append(float(gray[outer_y, outer_x]))
    if not inner_values:
        return 0.0
    contrast = float(np.mean(np.abs(np.asarray(inner_values) - np.asarray(outer_values))))
    return max(0.0, min(1.0, contrast / 55.0))


def _deduplicate(proposals: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float]]:
    kept: list[tuple[float, float, float, float]] = []
    for proposal in sorted(proposals, key=lambda item: item[3], reverse=True):
        x, y, radius, _ = proposal
        duplicate = any(
            np.hypot(x - other_x, y - other_y) < min(radius, other_radius) * 0.55
            and abs(radius - other_radius) < max(radius, other_radius) * 0.4
            for other_x, other_y, other_radius, _ in kept
        )
        if not duplicate:
            kept.append(proposal)
    kept.sort(key=lambda item: (item[1], item[0]))
    return kept


def _radius_diagnostics(radii: np.ndarray, image_scale: float) -> tuple[float, dict[str, int]]:
    if not len(radii):
        return 0.0, {}
    bin_width = max(2.0, image_scale * 0.0025)
    buckets = np.floor(radii / bin_width).astype(int)
    values, counts = np.unique(buckets, return_counts=True)
    winning_bucket = int(values[int(np.argmax(counts))])
    dominant = float(np.median(radii[buckets == winning_bucket]))
    distribution = {
        f"{value * bin_width:.1f}-{(value + 1) * bin_width:.1f}": int(count)
        for value, count in zip(values, counts)
    }
    return dominant, distribution


def detect_circle_candidates(image: np.ndarray) -> CandidateDetectionResult:
    """Propose circles from luminance edges, independent of saturated backgrounds."""
    if image is None or image.size == 0:
        return CandidateDetectionResult([], 0.0)
    height, width = image.shape[:2]
    image_scale = float(min(width, height))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 1.5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(10.0, image_scale * 0.012),
        param1=60,
        param2=30,
        minRadius=max(5, round(image_scale * 0.012)),
        maxRadius=max(8, round(image_scale * 0.045)),
    )
    if circles is None:
        return CandidateDetectionResult([], 0.0)
    proposals = []
    for x, y, radius in circles[0]:
        support = _circle_support(gray, float(x), float(y), float(radius))
        proposals.append((float(x), float(y), float(radius), support))
    deduplicated = _deduplicate(proposals)
    radii = np.asarray([item[2] for item in deduplicated], dtype=float)
    dominant_radius, distribution = _radius_diagnostics(radii, image_scale)
    candidates = [
        CircleCandidate(x, y, radius, f"candidate_{index:04d}")
        for index, (x, y, radius, _) in enumerate(deduplicated)
    ]
    support = {
        candidate.candidate_id: proposal[3]
        for candidate, proposal in zip(candidates, deduplicated)
    }
    confidence = float(np.median(list(support.values()))) if support else 0.0
    return CandidateDetectionResult(
        candidates,
        confidence,
        raw_proposal_count=len(proposals),
        dominant_radius=dominant_radius,
        support_by_candidate=support,
        radius_distribution=distribution,
    )


def draw_candidate_overlay(image: np.ndarray, result: CandidateDetectionResult) -> np.ndarray:
    """Draw raw deduplicated candidate IDs, radii, and edge-support scores."""
    output = image.copy()
    for candidate in result.candidates:
        center = (round(candidate.x), round(candidate.y))
        support = result.support_by_candidate.get(candidate.candidate_id, 0.0)
        cv2.circle(output, center, round(candidate.radius), (0, 255, 0), 2)
        cv2.putText(output, f"{candidate.candidate_id[-4:]} {support:.2f}", (center[0] - 20, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
    summary = f"raw: {result.raw_proposal_count} dedup: {len(result.candidates)} dominant radius: {result.dominant_radius:.1f}"
    cv2.rectangle(output, (5, 5), (620, 35), (0, 0, 0), -1)
    cv2.putText(output, summary, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return output



