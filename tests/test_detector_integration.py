from __future__ import annotations

import cv2
import numpy as np
import pytest

from bubblebot.vision.detector import detect_image

COLORS=[(30,30,220),(30,200,40),(210,70,30),(20,200,210),(190,40,180)]

def render(origin=(160,70),sx=38,sy=33,rows=5,cols=8,missing=(),color_count=4,contaminate=False,strong_highlight=True):
 image=np.zeros((700,900,3),np.uint8); expected={}; radius=round(sx/2.7)
 for row in range(rows):
  for col in range(cols):
   if (row,col) in missing: continue
   x=round(origin[0]+col*sx+(sx/2 if row%2 else 0)); y=round(origin[1]+row*sy); base=COLORS[(row*cols+col)%color_count]
   cv2.circle(image,(x,y),radius,tuple(round(v*.4) for v in base),-1,cv2.LINE_AA)
   cv2.circle(image,(x,y),round(radius*.78),base,-1,cv2.LINE_AA)
   if strong_highlight: cv2.circle(image,(x-4,y-4),max(2,radius//4),(245,245,245),-1,cv2.LINE_AA)
   expected[(row,col)]=(row*cols+col)%color_count
 if contaminate:
  for x,y,r,color in [(origin[0]+3*sx,430,radius,COLORS[4]),(origin[0]+4*sx,390,max(7,radius-3),COLORS[4]),(35,250,radius,COLORS[4]),(780,220,radius,COLORS[4]),(origin[0]+11,origin[1]+rows*sy+20,radius,COLORS[4])]:
   cv2.circle(image,(round(x),round(y)),r,color,-1,cv2.LINE_AA)
 return image,expected,radius

def state_signature(state):
 return {(b.row,b.col):b.color for b in state.bubbles}

def test_clean_full_board():
 image,expected,_=render(); state=detect_image(image); assert len(state.bubbles)==40; assert set(state_signature(state))==set(expected); assert state.board['colors']==4

def test_board_with_interior_holes():
 missing={(1,2),(2,4),(3,1)}; image,expected,_=render(missing=missing); state=detect_image(image); assert set(state_signature(state))==set(expected)

def test_sparse_bottom_row():
 missing={(4,c) for c in range(8) if c!=4}; image,expected,_=render(missing=missing); assert set(state_signature(detect_image(image)))==set(expected)

def test_shooter_next_and_ui_excluded():
 image,expected,_=render(contaminate=True); state=detect_image(image); assert len(state.bubbles)==len(expected); assert state.board['colors']==4; assert len(state.rejected_candidates)>=5

def test_clean_contaminated_equivalence():
 clean,_,_=render(); dirty,_,_=render(contaminate=True); a,b=detect_image(clean),detect_image(dirty); assert state_signature(a)==state_signature(b); assert a.calibration['spacing_x']==pytest.approx(b.calibration['spacing_x']); assert a.calibration['spacing_y']==pytest.approx(b.calibration['spacing_y']); assert a.board['colors']==b.board['colors']==4

def test_shifted_board_canonical_equivalence():
 a,_,_=render(); b,_,_=render(origin=(310,145)); assert state_signature(detect_image(a))==state_signature(detect_image(b))

def test_scaled_board_recovers_scale():
 image,expected,radius=render(origin=(200,90),sx=48,sy=41); state=detect_image(image); assert set(state_signature(state))==set(expected); assert state.calibration['spacing_x']==pytest.approx(48,abs=2); assert state.calibration['spacing_y']==pytest.approx(41,abs=2); assert state.board['radius']==pytest.approx(radius,abs=2)

def test_large_four_color_board():
 image,_,_=render(rows=10,cols=10); assert detect_image(image).board['colors']==4

def test_fifth_board_color_discovered():
 image,_,_=render(color_count=5); assert detect_image(image).board['colors']==5

def test_strong_highlights_preserve_colors():
 image,_,_=render(strong_highlight=True); assert detect_image(image).board['colors']==4

def test_near_board_false_positive_rejected():
 clean,_,_=render(); dirty,_,_=render(contaminate=True); assert state_signature(detect_image(clean))==state_signature(detect_image(dirty))

def test_blank_and_small_random_sets_fail_safely():
 blank=np.zeros((500,700,3),np.uint8); assert detect_image(blank).bubbles==[]
 for count in (1,2,3):
  image=blank.copy()
  for i in range(count): cv2.circle(image,(80+i*150,100+i*90),14,COLORS[i],-1)
  state=detect_image(image); assert state.bubbles==[]; assert state.confidence['overall_board_state']==0.0

