from dataclasses import dataclass
import cv2
import numpy as np
from bubblebot.vision.colors import extract_color_feature, cluster_colors

@dataclass(frozen=True)
class Candidate:
 x: float
 y: float
 radius: float
 candidate_id: str

BASES=[(30,30,220),(30,200,40),(210,70,30),(20,200,210),(190,40,180)]

def rendered(count_colors=4):
 image=np.zeros((1500,1200,3),np.uint8); candidates=[]; rng=np.random.default_rng(7)
 for color,base in enumerate(BASES[:count_colors]):
  for index in range(25):
   x=45+(index%10)*110; y=45+(color*3+index//10)*85; r=28; scale=.82+(index%5)*.08
   shade=tuple(int(np.clip(v*scale,0,255)) for v in base)
   cv2.circle(image,(x,y),r,tuple(int(v*.35) for v in shade),-1,cv2.LINE_AA)
   cv2.circle(image,(x,y),int(r*.78),shade,-1,cv2.LINE_AA)
   cv2.circle(image,(x-8,y-8),8,(245,245,245),-1,cv2.LINE_AA)
   noise=rng.integers(-3,4,size=image[max(0,y-20):y+21,max(0,x-20):x+21].shape,dtype=np.int16)
   patch=image[y-20:y+21,x-20:x+21].astype(np.int16)+noise; image[y-20:y+21,x-20:x+21]=np.clip(patch,0,255).astype(np.uint8)
   candidates.append(Candidate(x,y,r,f"c{color}_{index}"))
 return image,candidates

def classify(count):
 image,candidates=rendered(count); features=[extract_color_feature(image,c) for c in candidates]; return cluster_colors(features)

def test_rendered_four_colors_with_highlights_and_shadows():
 r=classify(4); assert r.cluster_count==4

def test_rendered_fifth_color_is_discovered():
 r=classify(5); assert r.cluster_count==5

def test_strong_highlight_does_not_control_feature():
 image,candidates=rendered(4); r=cluster_colors([extract_color_feature(image,c) for c in candidates]); assert all(list(r.assignment.values()).count(f"COLOR_{i}")==25 for i in range(4))

