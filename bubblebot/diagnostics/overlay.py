from __future__ import annotations

import cv2
import numpy as np

from bubblebot.vision.state import DetectionState


def draw_debug_overlay(image: np.ndarray, state: DetectionState) -> np.ndarray:
    output = image.copy()
    for bubble in state.bubbles:
        cv2.circle(output, (bubble.x, bubble.y), bubble.radius, (0, 255, 0), 2)
        label = f"{bubble.row},{bubble.col} {bubble.color}"
        cv2.putText(output, label, (bubble.x - bubble.radius, bubble.y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
    for candidate in state.rejected_candidates:
        center = (round(candidate["x"]), round(candidate["y"]))
        cv2.circle(output, center, round(candidate["radius"]), (0, 0, 255), 1)
        cv2.putText(output, candidate.get("reason", "rejected"), (center[0] + 4, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
    if state.board:
        cv2.rectangle(output, (state.board["left"], state.board["top"]), (state.board["right"], state.board["bottom"]), (255, 180, 0), 2)
    summary = (
        f"board bubbles: {len(state.bubbles)}  rows: {state.board.get('rows', 0)}  "
        f"colors: {state.board.get('colors', 0)}  sx: {state.calibration.get('spacing_x', 0):.1f}  "
        f"sy: {state.calibration.get('spacing_y', 0):.1f}  "
        f"isolation: {state.confidence.get('board_isolation', 0):.2f}  "
        f"color: {state.confidence.get('color_classification', 0):.2f}"
    )
    cv2.rectangle(output, (5, 5), (min(output.shape[1] - 5, 780), 34), (0, 0, 0), -1)
    cv2.putText(output, summary, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return output

