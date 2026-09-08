from __future__ import annotations

import numpy as np

from .candidates import detect_circle_candidates
from .colors import cluster_colors, extract_color_feature
from .isolation import isolate_board_candidates
from .state import Bubble, DetectionState


def detect_image(image: np.ndarray) -> DetectionState:
    """Compose candidate detection, board isolation, geometry, and color classification."""
    height, width = image.shape[:2]
    detected = detect_circle_candidates(image)
    isolated = isolate_board_candidates(detected.candidates)
    occupied_rows = int(isolated.confidence.get("occupied_rows", 0.0))
    if len(isolated.accepted) < 4 or occupied_rows < 2:
        rejected = [
            {"id": item.candidate_id, "x": item.x, "y": item.y, "radius": item.radius,
             "reason": isolated.rejection_reason.get(item.candidate_id, "no_convincing_board")}
            for item in detected.candidates
        ]
        return DetectionState(
            (width, height), {}, rejected_candidates=rejected,
            confidence={"candidate_detection": detected.confidence, "board_isolation": 0.0,
                        "grid": 0.0, "color_classification": 0.0, "overall_board_state": 0.0, "launcher": 0.0, "shooter": 0.0},
        )

    features = [extract_color_feature(image, item) for item in isolated.accepted]
    colors = cluster_colors(features)
    bubbles = []
    for item in isolated.accepted:
        cell = isolated.cell_by_candidate[item.candidate_id]
        confidence = min(
            float(isolated.confidence.get("overall", 0.0)),
            float(colors.confidence[item.candidate_id]),
        )
        bubbles.append(Bubble(cell[0], cell[1], round(item.x), round(item.y), round(item.radius), colors.assignment[item.candidate_id], confidence))
    bubbles.sort(key=lambda bubble: (bubble.row, bubble.col))

    radius = float(np.median([item.radius for item in isolated.accepted]))
    board = {
        "left": int(np.floor(min(item.x - item.radius for item in isolated.accepted))),
        "right": int(np.ceil(max(item.x + item.radius for item in isolated.accepted))),
        "top": int(np.floor(min(item.y - item.radius for item in isolated.accepted))),
        "bottom": int(np.ceil(max(item.y + item.radius for item in isolated.accepted))),
        "radius": radius,
        "rows": len({bubble.row for bubble in bubbles}),
        "colors": colors.cluster_count,
    }
    rejected = [
        {"id": item.candidate_id, "x": item.x, "y": item.y, "radius": item.radius,
         "reason": isolated.rejection_reason.get(item.candidate_id, "not_board")}
        for item in isolated.rejected
    ]
    grid_confidence = float(isolated.confidence.get("overall", 0.0))
    overall = min(detected.confidence, grid_confidence, colors.overall_confidence)
    calibration = dict(isolated.calibration)
    calibration.update({"row_offset_direction": "canonical odd-r", "board_rows": float(board["rows"])})
    confidence = {
        "candidate_detection": detected.confidence,
        "board_isolation": grid_confidence,
        "grid": grid_confidence,
        "color_classification": colors.overall_confidence,
        "overall_board_state": overall,
        "launcher": 0.0,
        "shooter": 0.0,
    }
    return DetectionState(
        (width, height), board, bubbles, None, None, None,
        calibration, confidence, rejected_candidates=rejected,
    )



