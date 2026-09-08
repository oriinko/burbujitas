import cv2
import numpy as np
import pytest

from bubblebot.vision.candidates import detect_circle_candidates


def canvas(size=(800,800),background=(90,50,100)):
 return np.full((size[1],size[0],3),background,np.uint8)

def draw_bubbles(image,centers,radius=24,colors=None,gloss=False,partial=False):
 colors=colors or [(30,30,230),(30,210,40),(235,235,235)]
 for i,center in enumerate(centers):
  color=colors[i%len(colors)]
  if partial:
   cv2.ellipse(image,center,(radius,radius),0,20,330,color,3,cv2.LINE_AA)
  else:
   cv2.circle(image,center,radius,tuple(int(v*.45) for v in color),-1,cv2.LINE_AA)
   cv2.circle(image,center,round(radius*.82),color,-1,cv2.LINE_AA)
  if gloss: cv2.circle(image,(center[0]-7,center[1]-7),5,(250,250,250),-1,cv2.LINE_AA)
 return image

def near_count(result,centers,tolerance=8):
 return sum(any(np.hypot(c.x-x,c.y-y)<=tolerance for c in result.candidates) for x,y in centers)

def test_colored_circles_on_saturated_purple_background():
 centers=[(100+x*75,100+y*70) for y in range(4) for x in range(7)]; r=detect_circle_candidates(draw_bubbles(canvas(),centers)); assert near_count(r,centers)>=27

def test_white_circles_on_saturated_background():
 centers=[(150+i*90,180) for i in range(6)]; r=detect_circle_candidates(draw_bubbles(canvas(),centers,colors=[(240,240,240)])); assert near_count(r,centers)>=5

def test_glossy_highlights_preserve_proposals():
 centers=[(130+i*85,250) for i in range(7)]; r=detect_circle_candidates(draw_bubbles(canvas(),centers,gloss=True)); assert near_count(r,centers)>=6

@pytest.mark.parametrize('size,radius,spacing',[(600,18,60),(1200,36,110)])
def test_scaled_scene(size,radius,spacing):
 centers=[(80+x*spacing,80+y*spacing) for y in range(3) for x in range(4)]; image=canvas((size,size)); r=detect_circle_candidates(draw_bubbles(image,centers,radius)); assert near_count(r,centers,tolerance=radius*.35)>=11

def test_translated_scene():
 centers=[(320+x*75,350+y*70) for y in range(3) for x in range(5)]; r=detect_circle_candidates(draw_bubbles(canvas(),centers)); assert near_count(r,centers)>=14

def test_varied_radii_and_circular_ui_are_proposed():
 centers=[(100,100),(220,100),(340,100),(460,100)]; image=canvas(); draw_bubbles(image,centers[0:2],20); draw_bubbles(image,centers[2:],30); cv2.circle(image,(650,650),34,(220,220,220),4); r=detect_circle_candidates(image); assert near_count(r,centers)>=3; assert any(np.hypot(c.x-650,c.y-650)<10 for c in r.candidates)

def test_saturated_blob_and_rectangle_do_not_merge_background():
 image=canvas(); cv2.rectangle(image,(50,50),(750,750),(120,40,140),-1); cv2.rectangle(image,(200,200),(600,500),(20,200,80),-1); r=detect_circle_candidates(image); assert all(c.radius<40 for c in r.candidates)

def test_partial_noisy_circle_edges():
 centers=[(150+i*90,300) for i in range(5)]; image=draw_bubbles(canvas(),centers,partial=True); noise=np.random.default_rng(4).normal(0,5,image.shape); image=np.clip(image.astype(float)+noise,0,255).astype(np.uint8); r=detect_circle_candidates(image); assert near_count(r,centers,tolerance=12)>=4
