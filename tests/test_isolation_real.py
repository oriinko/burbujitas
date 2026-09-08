import json
from pathlib import Path

import pytest

from bubblebot.vision.isolation import CircleCandidate, isolate_board_candidates


FIXTURE = Path(__file__).parent / "fixtures" / "real" / "msn_bubble_001_candidates.json"


def test_real_candidates_isolate_dense_board_and_one_adjacent_tutorial_bubble():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    candidates = [
        CircleCandidate(
            x=float(item["x"]),
            y=float(item["y"]),
            radius=float(item["radius"]),
            candidate_id=item["id"],
        )
        for item in fixture["candidates"]
    ]
    proposal_support = {
        item["id"]: float(item["support"])
        for item in fixture["candidates"]
    }

    result = isolate_board_candidates(candidates, proposal_support)

    assert len(candidates) == 173
    assert len(result.accepted) == 137
    assert len(result.rejected) == 36
    assert result.core_rows == list(range(8))
    assert result.row_support == {
        0: 17,
        1: 17,
        2: 17,
        3: 17,
        4: 17,
        5: 17,
        6: 17,
        7: 17,
        8: 1,
    }
    assert result.calibration["spacing_x"] == pytest.approx(90.0, abs=0.01)
    assert result.calibration["spacing_y"] == pytest.approx(88.8, abs=0.01)

    accepted_ids = {candidate.candidate_id for candidate in result.accepted}
    assert "candidate_0112" in accepted_ids
    assert "candidate_0135" in accepted_ids
    assert "candidate_0109" not in accepted_ids
    assert "candidate_0127" not in accepted_ids
    assert result.rejection_reason["candidate_0109"] == "duplicate_cell"
    assert result.rejection_reason["candidate_0127"] == "duplicate_cell"

    cells = list(result.cell_by_candidate.values())
    assert len(cells) == len(set(cells))