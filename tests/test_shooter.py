import cv2
import numpy as np
import pytest

from bubblebot.vision.candidates import detect_circle_candidates
from bubblebot.vision.detector import detect_image
from bubblebot.vision.isolation import isolate_board_candidates
from bubblebot.vision.shooter import detect_shooter
from tests.test_detector_integration import COLORS, render, state_signature


def scene(origin=(160,70),sx=38,sy=33,current=True,next_bubble=True,ui=False,next_scale=.8):
 image,_,radius=render(origin=origin,sx=sx,sy=sy)
 current_xy=(round(origin[0]+5.5*sx),round(origin[1]+10.5*sy))
 next_xy=(round(current_xy[0]+2.2*radius),round(current_xy[1]-2.8*radius))
 if current: cv2.circle(image,current_xy,radius,COLORS[1],-1,cv2.LINE_AA)
 if next_bubble: cv2.circle(image,next_xy,round(radius*next_scale),COLORS[2],-1,cv2.LINE_AA)
 if ui:
  for x,y,r in [(origin[0]-70,current_xy[1]+50,radius),(origin[0]+1*sx,current_xy[1]+100,round(radius*.6)),(origin[0]+7*sx,current_xy[1]-20,radius)]: cv2.circle(image,(round(x),round(y)),r,COLORS[4],-1,cv2.LINE_AA)
 return image,current_xy,next_xy


def run(image):
 state=detect_image(image); detected=detect_circle_candidates(image); isolated=isolate_board_candidates(detected.candidates); ids={c.candidate_id for c in isolated.accepted}; return state,detect_shooter(image,state,detected.candidates,ids)


def expected_colors(state): return state_signature(state)[(0,1)],state_signature(state)[(0,2)]

def test_normal_current_and_next():
 state,result=run(scene()[0]); current,next_color=expected_colors(state); assert result.current_color==current; assert result.next_color==next_color; assert result.launcher is not None

@pytest.mark.parametrize('origin',[(280,70),(160,130),(310,160)])
def test_shifted_scenes(origin):
 state,result=run(scene(origin=origin)[0]); assert (result.current_color,result.next_color)==expected_colors(state)

def test_scaled_scene():
 state,result=run(scene(sx=48,sy=41)[0]); assert (result.current_color,result.next_color)==expected_colors(state)

def test_off_center_launcher_not_image_center():
 image,current,_=scene(origin=(80,70)); _,result=run(image); assert result.launcher['x']==current[0]; assert result.launcher['x']!=image.shape[1]//2

def test_current_and_next_different_colors_match_board_ids():
 state,result=run(scene()[0]); assert result.current_color!=result.next_color; assert (result.current_color,result.next_color)==expected_colors(state)

def test_nearby_ui_does_not_replace_pair():
 state,result=run(scene(ui=True)[0]); assert (result.current_color,result.next_color)==expected_colors(state)

def test_lower_ui_not_selected_as_current():
 image,current,_=scene(ui=True); _,result=run(image); assert result.launcher=={'x':current[0],'y':current[1]}

def test_shooter_aligned_with_board_column():
 image,_,_=scene(); state,result=run(image); assert result.current_candidate is not None; assert len(state.bubbles)==40

def test_only_current_bubble():
 image,current,_=scene(next_bubble=False); state,result=run(image); assert result.launcher=={'x':current[0],'y':current[1]}; assert result.next_candidate is None; assert result.next_confidence==0

def test_missing_shooter():
 state,result=run(scene(current=False,next_bubble=False)[0]); assert result.launcher is None; assert result.current_candidate is None; assert result.next_candidate is None

def test_ambiguous_arrangements_reduce_confidence():
 image,_,_=scene(); cv2.circle(image,(250,430),14,COLORS[1],-1); cv2.circle(image,(280,390),12,COLORS[2],-1); _,result=run(image); assert result.current_confidence<0.8

@pytest.mark.parametrize('scale',[.6,1.15])
def test_next_radius_variation(scale):
 state,result=run(scene(next_scale=scale)[0]); assert result.next_color==expected_colors(state)[1]

def test_board_state_unchanged_by_shooter_processing():
 image,_,_=scene(ui=True); before=detect_image(image); detected=detect_circle_candidates(image); isolated=isolate_board_candidates(detected.candidates); detect_shooter(image,before,detected.candidates,{c.candidate_id for c in isolated.accepted}); after=detect_image(image); assert state_signature(before)==state_signature(after); assert before.board==after.board; assert before.calibration==after.calibration
