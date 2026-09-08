from __future__ import annotations

import cv2
import mss
import numpy as np


def capture_primary_monitor() -> np.ndarray:
    """Capture the primary monitor as a BGR OpenCV image."""
    with mss.mss() as session:
        monitor = session.monitors[1]
        frame = np.asarray(session.grab(monitor))
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
