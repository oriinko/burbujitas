from __future__ import annotations

import cv2
import numpy as np

from bubblebot.vision.state import DetectionState


def draw_debug_overlay(image: np.ndarray, state: DetectionState) -> np.ndarray:
    output = image.copy()
    for bubble in state.bubbles:
        cv2.circle(output, (bubble.x, bubble.y), bubble.radius, (0, 255, 0), 2)
        cv2.putText(output, f"{bubble.row},{bubble.col} {bubble.color[-1]}", (bubble.x - bubble.radius, bubble.y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    if state.board:
        cv2.rectangle(output, (state.board["left"], state.board["top"]), (state.board["right"], state.board["bottom"]), (255, 180, 0), 2)
    if state.launcher:
        cv2.drawMarker(output, (state.launcher["x"], state.launcher["y"]), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
    return output
