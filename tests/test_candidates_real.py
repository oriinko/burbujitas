from pathlib import Path
import cv2
from bubblebot.vision.candidates import detect_circle_candidates

FIXTURE=Path('tests/fixtures/real/msn_bubble_001.png')

def test_real_fixture_recovers_board_radius_mode_and_recall():
 image=cv2.imread(str(FIXTURE)); result=detect_circle_candidates(image)
 board=[c for c in result.candidates if 120<c.x<1750 and 220<c.y<1000 and 32<c.radius<48]
 assert len(board)>=134
 assert 36<=result.dominant_radius<=43
 assert result.raw_proposal_count>=len(result.candidates)>=136
