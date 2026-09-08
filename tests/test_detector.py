import numpy as np
import cv2

from bubblebot.vision.detector import detect_image


def test_detect_synthetic_bubbles():
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    for row in range(3):
        for col in range(5):
            x = 180 + col * 36 + (18 if row % 2 else 0)
            y = 80 + row * 31
            cv2.circle(image, (x, y), 14, (0, 0, 230) if (row + col) % 2 else (0, 210, 0), -1)
    state = detect_image(image)
    assert len(state.bubbles) >= 10
    assert state.calibration["radius"] > 0
    assert state.board["right"] > state.board["left"]
