from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from .state import Bubble, DetectionState


def _color_name(rgb: np.ndarray, centers: list[np.ndarray]) -> str:
    distances = [float(np.linalg.norm(rgb.astype(float) - center)) for center in centers]
    return f"COLOR_{int(np.argmin(distances))}"


def detect_image(image: np.ndarray) -> DetectionState:
    """Detect colorful circular objects and infer a lattice from their centers.

    This deliberately uses image-relative geometry and HSV saturation/value masks.
    UI-specific mistakes remain visible through confidence values and the overlay.
    """
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 55, 45]), np.array([179, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[int, int, int, np.ndarray]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 120:
            continue
        (x, y), radius = cv2.minEnclosingCircle(contour)
        circularity = area / max(np.pi * radius * radius, 1.0)
        if 8 <= radius <= min(width, height) * 0.08 and circularity >= 0.45:
            px, py = int(round(x)), int(round(y))
            patch = image[max(0, py-2):py+3, max(0, px-2):px+3]
            candidates.append((px, py, int(round(radius)), patch.reshape(-1, 3).mean(axis=0)[::-1]))
    if not candidates:
        return DetectionState((width, height), {}, confidence={"bubbles": 0.0})

    radii = np.array([item[2] for item in candidates], dtype=float)
    radius = float(np.median(radii))
    centers = [np.mean([item[3] for item in candidates], axis=0)]
    if len(candidates) >= 3:
        data = np.float32([item[3] for item in candidates])
        k = min(8, max(2, len(candidates) // 5))
        _, _, cluster_centers = cv2.kmeans(data, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 3, cv2.KMEANS_PP_CENTERS)
        centers = [center for center in cluster_centers]
    x_values = sorted(item[0] for item in candidates)
    y_values = sorted(item[1] for item in candidates)
    spacing_x = float(np.median(np.diff(x_values))) if len(x_values) > 1 else radius * 2
    spacing_y = float(np.median(np.diff(y_values))) if len(y_values) > 1 else radius * 1.73
    spacing_x = max(radius * 1.5, min(radius * 2.4, spacing_x))
    spacing_y = max(radius * 1.3, min(radius * 2.0, spacing_y))
    min_x, min_y = min(x_values), min(y_values)
    bubbles = []
    for x, y, r, rgb in candidates:
        row = max(0, int(round((y - min_y) / spacing_y)))
        offset = spacing_x / 2 if row % 2 else 0
        col = int(round((x - min_x - offset) / spacing_x))
        bubbles.append(Bubble(row, col, x, y, r, _color_name(rgb, centers), min(1.0, 0.55 + circularity_score(r, radius) * 0.45)))
    board = {"left": int(min_x - radius), "right": int(max(x_values) + radius), "top": int(min_y - radius), "bottom": int(max(y_values) + radius), "radius": radius}
    launcher = _find_launcher(image, int(min_y))
    current, next_bubble = _infer_shooter(image, launcher, centers, radius)
    return DetectionState((width, height), board, bubbles, current, next_bubble, launcher, {"radius": radius, "spacing_x": spacing_x, "spacing_y": spacing_y, "row_offset": spacing_x / 2}, {"bubbles": min(1.0, len(bubbles) / 20), "grid": 0.7, "board": 0.6})


def circularity_score(value: float, median: float) -> float:
    return max(0.0, 1.0 - abs(value - median) / max(median, 1.0))


def _find_launcher(image: np.ndarray, top: int) -> dict[str, int] | None:
    h, w = image.shape[:2]
    return {"x": w // 2, "y": min(h - 60, max(top + 300, int(h * 0.84)))}


def _infer_shooter(image: np.ndarray, launcher: dict[str, int] | None, centers: list[np.ndarray], radius: float) -> tuple[str | None, str | None]:
    if launcher is None:
        return None, None
    x, y = launcher["x"], launcher["y"]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    def sample(dx: int, dy: int) -> str | None:
        patch = hsv[max(0, y+dy-int(radius)):y+dy+int(radius), max(0, x+dx-int(radius)):x+dx+int(radius)]
        if patch.size == 0 or float(np.mean(patch[..., 1])) < 35:
            return None
        rgb = cv2.cvtColor(np.uint8([[np.mean(image[max(0, y+dy-2):y+dy+3, max(0, x+dx-2):x+dx+3], axis=(0,1))]]), cv2.COLOR_BGR2RGB)[0, 0]
        return _color_name(rgb, centers)
    return sample(0, 0), sample(0, -int(radius * 2.3))

