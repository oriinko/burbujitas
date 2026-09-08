from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Bubble:
    row: int
    col: int
    x: int
    y: int
    radius: int
    color: str
    confidence: float


@dataclass
class DetectionState:
    image_size: tuple[int, int]
    board: dict[str, int | float]
    bubbles: list[Bubble] = field(default_factory=list)
    current: str | None = None
    next_bubble: str | None = None
    launcher: dict[str, int] | None = None
    calibration: dict[str, float] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)
    rejected_candidates: list[dict[str, Any]] = field(default_factory=list)
    color_centroids: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    shooter_diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = self.__dict__.copy()
        value["bubbles"] = [bubble.__dict__ for bubble in self.bubbles]
        return value

    def console_grid(self) -> str:
        if not self.bubbles:
            return "(no bubbles detected)"
        rows: dict[int, list[Bubble]] = {}
        for bubble in self.bubbles:
            rows.setdefault(bubble.row, []).append(bubble)
        lines = ["grid (odd rows offset):"]
        for row in sorted(rows):
            cells = " ".join(f"{b.color[-1]:>2}" for b in sorted(rows[row], key=lambda b: b.col))
            lines.append(f"R{row:02d}  {'  ' if row % 2 else ''}{cells}")
        return "\n".join(lines)



