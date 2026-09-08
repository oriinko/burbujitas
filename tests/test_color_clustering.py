import random
import numpy as np
from bubblebot.vision.colors import ColorFeature, cluster_colors

BASES=[(70,185,165),(120,75,180),(180,115,190),(215,125,115)]

def features(bases=BASES, counts=None, variation=5):
 counts=counts or [25]*len(bases); out=[]
 for color,(base,count) in enumerate(zip(bases,counts)):
  for i in range(count):
   delta=((i*3)% (variation*2+1))-variation
   out.append(ColorFeature(f"c{color}_{i}",(base[0]+delta,base[1]+delta*.2,base[2]-delta*.2)))
 return out

def test_one_hundred_samples_four_colors():
 r=cluster_colors(features()); assert r.cluster_count==4; assert sorted(list(r.assignment.values()).count(f"COLOR_{i}") for i in range(4))==[25]*4

def test_brightness_variation_stays_four():
 assert cluster_colors(features(variation=9)).cluster_count==4

def test_distinct_fifth_color():
 assert cluster_colors(features(BASES+[(145,200,75)])).cluster_count==5

def test_uneven_populations_keep_rare_color():
 r=cluster_colors(features(counts=[70,10,6,2])); assert r.cluster_count==4; assert sorted(list(r.assignment.values()).count(x) for x in set(r.assignment.values()))==[2,6,10,70]

def test_order_invariance():
 original=features(); shuffled=list(original); random.Random(42).shuffle(shuffled)
 a=cluster_colors(original); b=cluster_colors(shuffled); assert a.assignment==b.assignment and a.centroids==b.centroids

def test_close_but_distinct_chroma():
 samples=features([(120,145,150),(120,170,125)],counts=[20,20],variation=4); assert cluster_colors(samples,max_diameter=18).cluster_count==2

def test_one_color_and_tiny_counts():
 assert cluster_colors(features([BASES[0]],counts=[1])).cluster_count==1; assert cluster_colors(features(BASES[:2],counts=[1,2])).cluster_count==2
