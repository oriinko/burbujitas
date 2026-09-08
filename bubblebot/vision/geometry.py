from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class LatticeFit:
    cells: dict[tuple[int, int], tuple[float, float]]
    spacing_x: float
    spacing_y: float
    residual: float
    row_offset_direction: str


def odd_r_neighbors(row: int, col: int) -> set[tuple[int, int]]:
    """Return neighbors for odd-r horizontal layout; odd rows are offset right."""
    if row % 2:
        offsets = ((0,-1),(0,1),(-1,0),(-1,1),(1,0),(1,1))
    else:
        offsets = ((0,-1),(0,1),(-1,-1),(-1,0),(1,-1),(1,0))
    return {(row + dr, col + dc) for dr, dc in offsets}


def fit_lattice(points: list[tuple[float, float]], radius: float = 10.0) -> LatticeFit:
    ys = sorted(y for _, y in points)
    rows: list[list[float]] = []
    for y in ys:
        if not rows or y - float(np.median(rows[-1])) > radius * 0.65:
            rows.append([y])
        else:
            rows[-1].append(y)
    row_y = np.array([np.median(row) for row in rows], dtype=float)
    sy = float(np.median(np.diff(row_y))) if len(row_y) > 1 else radius * 1.73
    gaps = []
    for row in rows:
        xs = sorted(x for x, y in points if min(abs(y - v) for v in row) <= radius * 0.65)
        gaps.extend(float(gap) for gap in np.diff(xs) if gap > radius * 1.2)
    sx = float(np.median(gaps)) if gaps else radius * 2.0

    best = None
    for parity in (0, 1):
        for ox in np.linspace(min(x for x, _ in points) - sx, max(x for x, _ in points) + sx, 81):
            for oy in (row_y[0] - sy, row_y[0], row_y[0] + sy):
                assignments = []
                errors = []
                for x, y in points:
                    row = int(round((y - oy) / sy))
                    offset = sx / 2 if (row + parity) % 2 else 0.0
                    col = int(round((x - ox - offset) / sx))
                    assignments.append((row, col))
                    errors.append(abs(x - (ox + offset + col * sx)) + abs(y - (oy + row * sy)))
                score = float(np.median(errors))
                if best is None or score < best[0]:
                    best = (score, assignments, parity)
    score, assignments, parity = best
    min_row = min(row for row, _ in assignments)
    min_col = min(col for _, col in assignments)
    canonical = {(row - min_row, col - min_col): point for point, (row, col) in zip(points, assignments)}
    direction = "right" if parity == 0 else "left"
    return LatticeFit(canonical, sx, sy, float(score), direction)
