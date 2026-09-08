from __future__ import annotations
import cv2
import numpy as np
from .state import Bubble, DetectionState
def _color_name(rgb, centers): return f"COLOR_{int(np.argmin([np.linalg.norm(rgb.astype(float)-c) for c in centers]))}"
def _rows(ys,tol):
 rows=[]
 for y in sorted(ys):
  if not rows or y-np.median(rows[-1])>tol: rows.append([y])
  else: rows[-1].append(y)
 return rows
def _fit(cs,radius):
 rows=_rows([c[1] for c in cs],max(3.,radius*.65)); ry=np.array([np.median(r) for r in rows]); sy=float(np.median(np.diff(ry))) if len(ry)>1 else radius*1.73; gaps=[]
 for row in rows:
  xs=sorted(c[0] for c in cs if min(abs(c[1]-y) for y in row)<=radius*.65); gaps += [d for d in np.diff(xs) if d>radius*1.2]
 sx=float(np.median(gaps)) if gaps else radius*2; sx=max(radius*1.5,min(radius*2.4,sx)); sy=max(radius*1.3,min(radius*2.,sy)); best=None
 for parity in (0,1):
  for ox in np.linspace(min(c[0] for c in cs)-sx,min(c[0] for c in cs)+sx,41):
   for oy in (ry[0]-sy,ry[0],ry[0]+sy):
    ass=[]; res=[]
    for x,y,*_ in cs:
     r=round((y-oy)/sy); off=sx/2 if (r+parity)%2 else 0; col=round((x-ox-off)/sx); ass.append((int(r),int(col))); res.append(abs(y-(oy+r*sy))+abs(x-(ox+off+col*sx)))
    score=float(np.median(res))
    if best is None or score<best[0]: best=(score,ox,oy,parity,ass)
 score,ox,oy,parity,ass=best; base=min((c for c in ass if c[0]==0),key=lambda z:z[1])[1] if any(c[0]==0 for c in ass) else min(c[1] for c in ass); ass=[(r,col-base) for r,col in ass]; conf=max(0.,min(1.,1-score/max(radius*2,1)))
 return [Bubble(rc[0],rc[1],c[0],c[1],c[2],"",conf) for c,rc in zip(cs,ass)],{"radius":radius,"spacing_x":sx,"spacing_y":sy,"row_offset":sx/2,"origin_x":ox,"origin_y":oy,"parity":float(parity)},conf
def _find_launcher(image,board,radius):
 h,w=image.shape[:2]; x0=max(0,int(board["left"]-radius*2)); x1=min(w,int(board["right"]+radius*2)); y0=min(h,int(board["bottom"]+radius*1.5));
 if y0>=h or x1<=x0:return None,0.
 hsv=cv2.cvtColor(image[y0:,x0:x1],cv2.COLOR_BGR2HSV); mask=cv2.inRange(hsv,np.array([0,55,45]),np.array([179,255,255])); contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); objs=[]
 for q in contours:
  (x,y),r=cv2.minEnclosingCircle(q); area=cv2.contourArea(q)
  if radius*.65<=r<=radius*1.5 and area/max(np.pi*r*r,1)>=.4: objs.append((x+x0,y+y0,r))
 if not objs:return None,0.
 x,y,r=max(objs,key=lambda p:p[1]); return {"x":int(round(x)),"y":int(round(y))},min(1.,.5+.5*min(1.,r/radius))
def _sample(image,x,y,r,centers):
 h,w=image.shape[:2]; x0,x1=max(0,int(x-r*.55)),min(w,int(x+r*.55)); y0,y1=max(0,int(y-r*.55)),min(h,int(y+r*.55)); p=image[y0:y1,x0:x1]
 if p.size==0 or float(np.mean(cv2.cvtColor(p,cv2.COLOR_BGR2HSV)[...,1]))<35:return None
 return _color_name(cv2.cvtColor(np.uint8([[p.reshape(-1,3).mean(axis=0)]]),cv2.COLOR_BGR2RGB)[0,0],centers)
def detect_image(image):
 h,w=image.shape[:2]; hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV); mask=cv2.inRange(hsv,np.array([0,55,45]),np.array([179,255,255])); mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8)); contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cs=[]
 for q in contours:
  area=cv2.contourArea(q); (x,y),r=cv2.minEnclosingCircle(q); circ=area/max(np.pi*r*r,1.)
  if area>=120 and 8<=r<=min(w,h)*.08 and circ>=.45:
   px,py=round(x),round(y); p=image[max(0,py-2):py+3,max(0,px-2):px+3]; cs.append((px,py,round(r),p.reshape(-1,3).mean(axis=0)[::-1]))
 if not cs:return DetectionState((w,h),{},confidence={"bubbles":0.,"launcher":0.})
 radius=float(np.median([c[2] for c in cs])); data=np.float32([c[3] for c in cs]); k=min(8,max(2,len(cs)//5)) if len(cs)>=3 else 1
 if k>1: _,_,centers=cv2.kmeans(data,k,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,1.),3,cv2.KMEANS_PP_CENTERS); centers=[c for c in centers]
 else: centers=[data.mean(axis=0)]
 bubbles,cal,grid=_fit(cs,radius)
 for b,c in zip(bubbles,cs): b.color=_color_name(c[3],centers)
 board={"left":int(min(c[0] for c in cs)-radius),"right":int(max(c[0] for c in cs)+radius),"top":int(min(c[1] for c in cs)-radius),"bottom":int(max(c[1] for c in cs)+radius),"radius":radius}; launcher,lc=_find_launcher(image,board,radius); current=_sample(image,launcher["x"],launcher["y"],radius,centers) if launcher else None; nxt=_sample(image,launcher["x"],launcher["y"]-radius*2.25,radius*.8,centers) if launcher else None
 return DetectionState((w,h),board,bubbles,current,nxt,launcher,cal,{"bubbles":min(1.,len(bubbles)/20),"grid":grid,"board":grid,"launcher":lc,"shooter":float(current is not None)*.7+float(nxt is not None)*.3})


