from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .colors import ColorFeature, extract_color_feature
from .isolation import CircleCandidate
from .state import DetectionState


@dataclass
class ShooterDetectionResult:
    launcher: dict[str, int] | None = None
    current_candidate: CircleCandidate | None = None
    next_candidate: CircleCandidate | None = None
    current_color: str | None = None
    next_color: str | None = None
    current_confidence: float = 0.0
    next_confidence: float = 0.0
    diagnostics: dict[str, float | str] = field(default_factory=dict)
    current_feature: ColorFeature | None = None
    next_feature: ColorFeature | None = None


def _board_centroids(image: np.ndarray, state: DetectionState) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = {}
    for bubble in state.bubbles:
        candidate = CircleCandidate(bubble.x, bubble.y, bubble.radius, f"board_{bubble.row}_{bubble.col}")
        feature = extract_color_feature(image, candidate)
        grouped.setdefault(bubble.color, []).append(np.asarray(feature.lab, dtype=float))
    return {color: np.median(values, axis=0) for color, values in grouped.items()}


def _match_color(feature: ColorFeature | None, centroids: dict[str, np.ndarray], threshold: float = 50.0) -> tuple[str | None, float]:
    if feature is None or not centroids:
        return None, 0.0
    value = np.asarray(feature.lab, dtype=float)
    distances = sorted((float(np.linalg.norm(value - centroid)), color) for color, centroid in centroids.items())
    distance, color = distances[0]
    if distance > threshold:
        return "UNKNOWN_NEW_COLOR", max(0.0, 1.0 - (distance - threshold) / threshold)
    competing = distances[1][0] if len(distances) > 1 else threshold * 2
    confidence = max(0.0, min(1.0, 1.0 - distance / max(competing, 1e-6)))
    return color, confidence


def _candidate_score(candidate: CircleCandidate, state: DetectionState) -> float:
    board = state.board
    radius = float(state.calibration["radius"])
    center_x = (float(board["left"]) + float(board["right"])) / 2
    width = max(float(board["right"]) - float(board["left"]), radius * 2)
    radius_score = max(0.0, 1.0 - abs(candidate.radius - radius) / radius)
    horizontal = max(0.0, 1.0 - abs(candidate.x - center_x) / (width * 0.65))
    gap = candidate.y - float(board["bottom"])
    vertical = 1.0 if radius * 1.2 <= gap <= width * 1.2 else max(0.0, 1.0 - abs(gap - width * 0.45) / width)
    return 0.45 * radius_score + 0.35 * horizontal + 0.20 * vertical


def detect_shooter(
    image: np.ndarray,
    board_state: DetectionState,
    all_candidates: list[CircleCandidate],
    accepted_candidate_ids: set[str],
) -> ShooterDetectionResult:
    """Detect current/next bubbles after board reconstruction without altering it."""
    if not board_state.bubbles:
        return ShooterDetectionResult(diagnostics={"reason": "missing_board"})
    radius = float(board_state.calibration["radius"])
    board_bottom = float(board_state.board["bottom"])
    board_left = float(board_state.board["left"])
    board_right = float(board_state.board["right"])
    rejected = [
        item for item in all_candidates
        if item.candidate_id not in accepted_candidate_ids
        and item.y > board_bottom + radius * 0.8
        and board_left - radius * 3 <= item.x <= board_right + radius * 3
        and radius * 0.5 <= item.radius <= radius * 1.6
    ]
    if not rejected:
        return ShooterDetectionResult(diagnostics={"reason": "missing_shooter", "candidate_count": 0.0})

    pair_scores = []
    for current in rejected:
        current_score = _candidate_score(current, board_state)
        for next_candidate in rejected:
            if next_candidate is current:
                continue
            distance = float(np.hypot(current.x - next_candidate.x, current.y - next_candidate.y))
            proximity = max(0.0, 1.0 - abs(distance - radius * 2.7) / (radius * 3.0))
            above = 1.0 if next_candidate.y < current.y else 0.35
            next_radius = max(0.0, 1.0 - abs(next_candidate.radius - radius * 0.85) / radius)
            pair_scores.append((current_score + 0.45 * proximity + 0.20 * above + 0.15 * next_radius, current, next_candidate))
    pair_scores.sort(key=lambda item: item[0], reverse=True)

    if pair_scores:
        best_score, current, next_candidate = pair_scores[0]
        second = pair_scores[1][0] if len(pair_scores) > 1 else 0.0
        margin = max(0.0, best_score - second)
        geometry_confidence = max(0.0, min(1.0, best_score / 1.8))
        if margin < 0.08 and len(rejected) > 2:
            geometry_confidence *= 0.55
    else:
        current = max(rejected, key=lambda item: _candidate_score(item, board_state))
        next_candidate = None
        margin = 0.0
        geometry_confidence = max(0.0, min(0.75, _candidate_score(current, board_state)))

    centroids = _board_centroids(image, board_state)
    current_feature = extract_color_feature(image, current)
    next_feature = extract_color_feature(image, next_candidate) if next_candidate else None
    current_color, current_color_confidence = _match_color(current_feature, centroids)
    next_color, next_color_confidence = _match_color(next_feature, centroids)
    current_confidence = min(geometry_confidence, current_color_confidence)
    next_confidence = min(geometry_confidence, next_color_confidence) if next_candidate else 0.0
    return ShooterDetectionResult(
        launcher={"x": round(current.x), "y": round(current.y)},
        current_candidate=current,
        next_candidate=next_candidate,
        current_color=current_color,
        next_color=next_color,
        current_confidence=current_confidence,
        next_confidence=next_confidence,
        diagnostics={"candidate_count": float(len(rejected)), "pair_score": float(pair_scores[0][0]) if pair_scores else 0.0, "score_margin": margin, "launcher_assumption": "current_bubble_center"},
        current_feature=current_feature,
        next_feature=next_feature,
    )
