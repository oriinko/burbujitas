from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
    core_rows: list[int] = field(default_factory=list)
    row_support: dict[int, int] = field(default_factory=dict)
    duplicate_diagnostics: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class _Projection:
    candidate: CircleCandidate
    cell: tuple[int, int]
    residual: float
    residual_normalized: float
    radius_error: float
    proposal_support: float
    score: float


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


def _row_center(row: list[CircleCandidate]) -> float:
    return float(np.median([candidate.y for candidate in row]))


def _dense_row_runs(
    rows: list[list[CircleCandidate]],
    minimum_support: int,
    radius: float,
) -> list[list[list[CircleCandidate]]]:
    """Return physically contiguous runs of well-supported visual rows."""
    runs: list[list[list[CircleCandidate]]] = []
    current: list[list[CircleCandidate]] = []
    minimum_gap = radius * 1.4
    maximum_gap = radius * 3.0

    for row in rows:
        if len(row) < minimum_support:
            if current:
                runs.append(current)
                current = []
            continue

        if current:
            gap = _row_center(row) - _row_center(current[-1])
            if not minimum_gap <= gap <= maximum_gap:
                runs.append(current)
                current = []
        current.append(row)

    if current:
        runs.append(current)
    return runs


def _select_core_visual_rows(
    rows: list[list[CircleCandidate]],
    radius: float,
) -> tuple[list[list[CircleCandidate]], float]:
    max_support = max(len(row) for row in rows)
    minimum_support = max(3, int(np.ceil(max_support * 0.5)))
    runs = _dense_row_runs(rows, minimum_support, radius)
    if not runs:
        return [], 0.0

    ranked = sorted(
        runs,
        key=lambda run: (len(run), sum(len(row) for row in run)),
        reverse=True,
    )
    winner = ranked[0]
    winner_score = float(sum(len(row) for row in winner))
    runner_up_score = float(sum(len(row) for row in ranked[1])) if len(ranked) > 1 else 0.0
    return winner, winner_score - runner_up_score


def _lattice_origin(fit) -> tuple[float, float]:
    shifted_right = fit.row_offset_direction == "right"
    origins_x: list[float] = []
    origins_y: list[float] = []
    for (row, col), (x, y) in fit.cells.items():
        offset = fit.spacing_x / 2 if (row % 2 == 1) == shifted_right else 0.0
        origins_x.append(x - col * fit.spacing_x - offset)
        origins_y.append(y - row * fit.spacing_y)
    return float(np.median(origins_x)), float(np.median(origins_y))


def _project(
    candidate: CircleCandidate,
    fit,
    origin_x: float,
    origin_y: float,
) -> tuple[tuple[int, int], float]:
    row = int(round((candidate.y - origin_y) / fit.spacing_y))
    shifted_right = fit.row_offset_direction == "right"
    offset = fit.spacing_x / 2 if (row % 2 == 1) == shifted_right else 0.0
    col = int(round((candidate.x - origin_x - offset) / fit.spacing_x))
    expected_x = origin_x + col * fit.spacing_x + offset
    expected_y = origin_y + row * fit.spacing_y
    residual = float(np.hypot(candidate.x - expected_x, candidate.y - expected_y))
    return (row, col), residual


def _candidate_score(
    candidate: CircleCandidate,
    residual: float,
    threshold: float,
    radius: float,
    proposal_support: dict[str, float],
) -> tuple[float, float, float, float]:
    residual_normalized = residual / max(threshold, 1.0)
    radius_error = abs(candidate.radius - radius) / max(radius, 1.0)
    support = min(1.0, max(0.0, proposal_support.get(_key(candidate), 0.0)))
    support_penalty = 1.0 - support if proposal_support else 0.0
    score = residual_normalized + 0.35 * radius_error + 0.10 * support_penalty
    return score, residual_normalized, radius_error, support


def _best_by_cell(
    projections: list[_Projection],
) -> tuple[dict[tuple[int, int], _Projection], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[_Projection]] = {}
    for projection in projections:
        grouped.setdefault(projection.cell, []).append(projection)

    winners: dict[tuple[int, int], _Projection] = {}
    diagnostics: list[dict[str, Any]] = []
    for cell, choices in grouped.items():
        ranked = sorted(choices, key=lambda item: (item.score, item.residual, _key(item.candidate)))
        winners[cell] = ranked[0]
        if len(ranked) > 1:
            diagnostics.append(
                {
                    "cell": cell,
                    "winner": _key(ranked[0].candidate),
                    "losers": [_key(item.candidate) for item in ranked[1:]],
                    "components": {
                        _key(item.candidate): {
                            "total": item.score,
                            "residual": item.residual,
                            "residual_normalized": item.residual_normalized,
                            "radius_error": item.radius_error,
                            "proposal_support": item.proposal_support,
                        }
                        for item in ranked
                    },
                }
            )
    return winners, diagnostics


def _largest_connected_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    remaining = set(cells)
    components: list[set[tuple[int, int]]] = []
    while remaining:
        component: set[tuple[int, int]] = set()
        frontier = [remaining.pop()]
        while frontier:
            cell = frontier.pop()
            component.add(cell)
            for neighbor in odd_r_neighbors(*cell):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    frontier.append(neighbor)
        components.append(component)
    return max(components, key=len, default=set())


def isolate_board_candidates(
    candidates: list[CircleCandidate],
    proposal_support: dict[str, float] | None = None,
) -> BoardIsolationResult:
    """Isolate the dominant contiguous hex board without changing its fitted lattice."""
    result = BoardIsolationResult()
    if len(candidates) < 3:
        result.rejected = list(candidates)
        result.rejection_reason = {_key(item): "insufficient_candidates" for item in candidates}
        result.confidence = {"overall": 0.0}
        return result

    support = proposal_support or {}
    radius = float(np.median([item.radius for item in candidates]))
    visual_rows = _rows(candidates, max(2.0, radius * 0.7))
    core_visual_rows, score_margin = _select_core_visual_rows(visual_rows, radius)
    seed = [candidate for row in core_visual_rows for candidate in row]
    if len(seed) < 3:
        result.rejected = list(candidates)
        result.rejection_reason = {_key(item): "no_dominant_band" for item in candidates}
        result.confidence = {"overall": 0.0}
        return result

    radius = float(np.median([item.radius for item in seed]))
    fit = fit_lattice([(item.x, item.y) for item in seed], radius)
    origin_x, origin_y = _lattice_origin(fit)
    threshold = max(radius * 0.75, min(fit.spacing_x, fit.spacing_y) * 0.42)

    projections: list[_Projection] = []
    for candidate in candidates:
        cell, residual = _project(candidate, fit, origin_x, origin_y)
        result.residual_by_candidate[_key(candidate)] = residual
        if residual <= threshold:
            score, residual_normalized, radius_error, hough_support = _candidate_score(
                candidate,
                residual,
                threshold,
                radius,
                support,
            )
            projections.append(
                _Projection(
                    candidate,
                    cell,
                    residual,
                    residual_normalized,
                    radius_error,
                    hough_support,
                    score,
                )
            )

    winners, result.duplicate_diagnostics = _best_by_cell(projections)
    core_fit_rows = sorted(
        {
            _project(candidate, fit, origin_x, origin_y)[0][0]
            for row in core_visual_rows
            for candidate in row
        }
    )
    result.core_rows = core_fit_rows

    core_cells = {cell for cell in winners if cell[0] in core_fit_rows}
    active_cells = _largest_connected_cells(core_cells)
    extension_cells: set[tuple[int, int]] = set()
    bottom_row = max(row for row, _ in active_cells)

    while True:
        target_row = bottom_row + 1
        proposed = {
            cell
            for cell in winners
            if cell[0] == target_row
            and cell not in active_cells
            and any(neighbor in active_cells for neighbor in odd_r_neighbors(*cell))
        }
        if not proposed:
            break
        active_cells.update(proposed)
        extension_cells.update(proposed)
        bottom_row = target_row

    winner_key_by_cell = {cell: _key(projection.candidate) for cell, projection in winners.items()}
    for candidate in candidates:
        key = _key(candidate)
        cell, residual = _project(candidate, fit, origin_x, origin_y)
        if cell in active_cells and winner_key_by_cell.get(cell) == key:
            result.accepted.append(candidate)
            result.cell_by_candidate[key] = cell
            continue

        result.rejected.append(candidate)
        if residual > threshold:
            result.rejection_reason[key] = "lattice_residual"
        elif cell in winners and winner_key_by_cell[cell] != key:
            result.rejection_reason[key] = "duplicate_cell"
        elif cell[0] > max(core_fit_rows) and cell not in extension_cells:
            result.rejection_reason[key] = "non_adjacent_row"
        else:
            result.rejection_reason[key] = "row_band"

    result.row_support = {
        row: sum(1 for cell in active_cells if cell[0] == row)
        for row in sorted({row for row, _ in active_cells})
    }
    accepted_residuals = [result.residual_by_candidate[key] for key in result.cell_by_candidate]
    median_residual = float(np.median(accepted_residuals)) if accepted_residuals else float("inf")
    core_count = sum(result.row_support.get(row, 0) for row in core_fit_rows)
    hypothesis_count = len(
        _dense_row_runs(
            visual_rows,
            max(3, int(np.ceil(max(len(row) for row in visual_rows) * 0.5))),
            radius,
        )
    )
    result.calibration = {
        "radius": radius,
        "spacing_x": fit.spacing_x,
        "spacing_y": fit.spacing_y,
        "row_offset": fit.spacing_x / 2,
    }
    result.confidence = {
        "overall": min(1.0, core_count / max(len(seed), 1))
        / (1.0 + median_residual / max(radius, 1.0)),
        "lattice_residual_median": median_residual,
        "accepted_count": float(len(result.accepted)),
        "rejected_count": float(len(result.rejected)),
        "occupied_rows": float(len(result.row_support)),
        "support_ratio": core_count / len(candidates),
        "hypotheses_tested": float(hypothesis_count),
        "winning_support_count": float(core_count),
        "winning_unique_cells": float(len(active_cells)),
        "winning_rows": float(len(result.row_support)),
        "score_margin": score_margin,
    }
    return result