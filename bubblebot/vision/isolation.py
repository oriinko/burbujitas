from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import fit_lattice, odd_r_neighbors


@dataclass(frozen=True)
class CircleCandidate:
    x: float
    y: float
    radius: float
    candidate_id: str = ""


@dataclass
class BoardIsolationResult:
    accepted: list[CircleCandidate] = field(default_factory=list)
    rejected: list[CircleCandidate] = field(default_factory=list)
    cell_by_candidate: dict[str, tuple[int, int]] = field(default_factory=dict)
    residual_by_candidate: dict[str, float] = field(default_factory=dict)
    rejection_reason: dict[str, str] = field(default_factory=dict)
    calibration: dict[str, float] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)


def _key(candidate: CircleCandidate) -> str:
    return candidate.candidate_id or f"{candidate.x:.2f},{candidate.y:.2f}"


def _rows(candidates: list[CircleCandidate], tolerance: float) -> list[list[CircleCandidate]]:
    rows: list[list[CircleCandidate]] = []
    for candidate in sorted(candidates, key=lambda item: item.y):
        center = float(np.median([item.y for item in rows[-1]])) if rows else 0.0
        if not rows or candidate.y - center > tolerance:
            rows.append([candidate])
        else:
            rows[-1].append(candidate)
    return rows


def _lattice_origin(fit) -> tuple[float, float]:
    shifted_right = fit.row_offset_direction == "right"
    origins_x: list[float] = []
    origins_y: list[float] = []
    for (row, col), (x, y) in fit.cells.items():
        offset = fit.spacing_x / 2 if (row % 2 == 1) == shifted_right else 0.0
        origins_x.append(x - col * fit.spacing_x - offset)
        origins_y.append(y - row * fit.spacing_y)
    return float(np.median(origins_x)), float(np.median(origins_y))


def _project(candidate: CircleCandidate, fit, origin_x: float, origin_y: float) -> tuple[tuple[int, int], float]:
    row = int(round((candidate.y - origin_y) / fit.spacing_y))
    shifted_right = fit.row_offset_direction == "right"
    offset = fit.spacing_x / 2 if (row % 2 == 1) == shifted_right else 0.0
    col = int(round((candidate.x - origin_x - offset) / fit.spacing_x))
    expected_x = origin_x + col * fit.spacing_x + offset
    expected_y = origin_y + row * fit.spacing_y
    residual = float(np.hypot(candidate.x - expected_x, candidate.y - expected_y))
    return (row, col), residual


def isolate_board_candidates(candidates: list[CircleCandidate]) -> BoardIsolationResult:
    result = BoardIsolationResult()
    if len(candidates) < 3:
        result.rejected = list(candidates)
        result.rejection_reason = {_key(item): "insufficient_candidates" for item in candidates}
        result.confidence = {"overall": 0.0}
        return result

    radius = float(np.median([item.radius for item in candidates]))
    rows = _rows(candidates, max(2.0, radius * 0.7))
    max_support = max(len(row) for row in rows)
    seed = [item for row in rows if len(row) >= max(3, int(max_support * 0.5)) for item in row]
    if len(seed) < 3:
        result.rejected = list(candidates)
        result.rejection_reason = {_key(item): "no_dominant_band" for item in candidates}
        result.confidence = {"overall": 0.0}
        return result

    fit = fit_lattice([(item.x, item.y) for item in seed], radius)
    origin_x, origin_y = _lattice_origin(fit)
    threshold = max(radius * 0.75, min(fit.spacing_x, fit.spacing_y) * 0.42)
    projected = [(item, *_project(item, fit, origin_x, origin_y)) for item in candidates]

    active_cells = set(fit.cells)
    min_row = min(row for row, _ in active_cells)
    max_row = max(row for row, _ in active_cells)
    extensions: dict[str, tuple[int, int]] = {}

    while True:
        target_row = max_row + 1
        proposed: dict[tuple[int, int], CircleCandidate] = {}
        for item, cell, residual in projected:
            if cell[0] != target_row or residual > threshold or cell in active_cells:
                continue
            if any(neighbor in active_cells for neighbor in odd_r_neighbors(*cell)):
                proposed.setdefault(cell, item)
        if not proposed:
            break
        for cell, item in proposed.items():
            active_cells.add(cell)
            extensions[_key(item)] = cell
        max_row = target_row

    claimed: set[tuple[int, int]] = set()
    for item, cell, residual in projected:
        key = _key(item)
        result.residual_by_candidate[key] = residual
        in_seed_band = min_row <= cell[0] <= max(row for row, _ in fit.cells)
        is_extension = extensions.get(key) == cell
        if residual <= threshold and (in_seed_band or is_extension) and cell not in claimed:
            result.accepted.append(item)
            result.cell_by_candidate[key] = cell
            claimed.add(cell)
        else:
            result.rejected.append(item)
            if cell in claimed:
                result.rejection_reason[key] = "duplicate_cell"
            elif residual > threshold:
                result.rejection_reason[key] = "lattice_residual"
            else:
                result.rejection_reason[key] = "row_band"

    accepted_rows = {row for row, _ in result.cell_by_candidate.values()}
    accepted_residuals = [result.residual_by_candidate[key] for key in result.cell_by_candidate]
    median_residual = float(np.median(accepted_residuals)) if accepted_residuals else float("inf")
    result.calibration = {"radius": radius, "spacing_x": fit.spacing_x, "spacing_y": fit.spacing_y, "row_offset": fit.spacing_x / 2}
    result.confidence = {
        "overall": min(1.0, len(result.accepted) / len(seed)) / (1.0 + median_residual / max(radius, 1.0)),
        "lattice_residual_median": median_residual,
        "accepted_count": float(len(result.accepted)),
        "rejected_count": float(len(result.rejected)),
        "occupied_rows": float(len(accepted_rows)),
        "support_ratio": len(seed) / len(candidates),
        "hypotheses_tested": 1.0,
        "winning_support_count": float(len(seed)),
        "winning_unique_cells": float(len(result.cell_by_candidate)),
        "winning_rows": float(len(accepted_rows)),
        "score_margin": float(len(seed) - len(result.rejected)),
    }
    return result
