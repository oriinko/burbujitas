import cv2, numpy as np, pytest
from bubblebot.vision.detector import detect_image
def make_board(origin=(180,80),sx=36,sy=31,rows=4,cols=6,missing=()):
 im=np.zeros((700,900,3),np.uint8); colors=[(0,0,230),(0,210,0),(230,80,0),(0,210,210)]
 for r in range(rows):
  for c in range(cols):
   if (r,c) in missing: continue
   cv2.circle(im,(origin[0]+c*sx+(sx//2 if r%2 else 0),origin[1]+r*sy),round(sx/2.57),colors[(r+c)%4],-1,cv2.LINE_AA)
 return im
def pos(s): return {(b.row,b.col) for b in s.bubbles}
def test_exact_grid():
 s=detect_image(make_board()); assert pos(s)=={(r,c) for r in range(4) for c in range(6)}; assert s.calibration["spacing_x"]==pytest.approx(36,abs=2); assert s.calibration["spacing_y"]==pytest.approx(31,abs=2)
def test_shifted_scaled_gaps():
 s=detect_image(make_board((311,137),44,38,5,5,((1,2),(3,0)))); assert pos(s)==({(r,c) for r in range(5) for c in range(5)}-{(1,2),(3,0)}); assert s.calibration["spacing_x"]==pytest.approx(44,abs=3); assert s.calibration["spacing_y"]==pytest.approx(38,abs=3)
def test_colors_and_uncertain_launcher():
 s=detect_image(make_board()); assert len({b.color for b in s.bubbles})>=3; assert s.launcher is None; assert s.confidence["launcher"]==0.; assert s.current is None and s.next_bubble is None
import cv2, numpy as np
from bubblebot.vision.detector import detect_image

def contaminated_board():
 im=np.zeros((800,1000,3),np.uint8)
 for r in range(5):
  for c in range(8): cv2.circle(im,(180+c*36+(18 if r%2 else 0),70+r*31),14,((0,0,230) if (r+c)%2 else (0,210,0)),-1)
 cv2.circle(im,(640,700),14,(230,80,0),-1); cv2.circle(im,(690,660),10,(0,210,210),-1); cv2.circle(im,(50,500),12,(255,0,255),-1)
 return im

def test_board_isolation():
 s=detect_image(contaminated_board()); assert len(s.bubbles)==40; assert all(b.y<250 for b in s.bubbles)

def test_four_colors_large_board():
 im=np.zeros((900,1100,3),np.uint8)
 cols=[(0,0,220),(0,200,0),(220,70,0),(0,200,200)]
 for r in range(10):
  for c in range(10): cv2.circle(im,(180+c*36+(18 if r%2 else 0),70+r*31),14,cols[(r+c)%4],-1)
 s=detect_image(im); assert len({b.color for b in s.bubbles})==4
