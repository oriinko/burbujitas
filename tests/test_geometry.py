import cv2, numpy as np, pytest
from bubblebot.vision.geometry import fit_lattice, odd_r_neighbors

def board(sx=36,sy=31,rows=4,cols=6,missing=(),parity=0,origin=(100,50)):
 return [(origin[0]+c*sx+(sx/2 if (r+parity)%2 else 0),origin[1]+r*sy) for r in range(rows) for c in range(cols) if (r,c) not in missing]

def test_geometry_full_shift_scale_and_gaps():
 for pts in [board(),board(origin=(317,141)),board(sx=48,sy=39),board(missing={(1,2),(2,0)}),board(missing={(0,0),(0,1),(3,4),(3,5)}),board(parity=1)]:
  fit=fit_lattice(pts); assert fit.residual<1.; assert fit.spacing_x==pytest.approx(round(max(np.diff(sorted({x for x,y in pts}))),0),abs=20) or fit.spacing_x>0

def test_canonical_rows_and_columns_with_missing_edges():
 a=fit_lattice(board(missing={(0,0),(0,1),(3,4),(3,5)})); b=fit_lattice(board(missing={(0,0),(0,1),(3,4),(3,5)},origin=(500,300)))
 assert min(r for r,c in a.cells)==0 and min(c for r,c in a.cells)==0
 assert set(a.cells)==set(b.cells)

def test_translation_invariance_and_opposite_parity():
 base=fit_lattice(board(missing={(0,0)},parity=0)); shifted=fit_lattice(board(missing={(0,0)},origin=(700,420),parity=0)); opposite=fit_lattice(board(missing={(0,0)},parity=1))
 assert set(base.cells)==set(shifted.cells)
 assert base.row_offset_direction in {"left","right"}
 assert set(opposite.cells)==set(base.cells)

def test_relative_adjacency_survives_canonicalization():
 fit=fit_lattice(board())
 for cell in [(1,1),(1,3),(2,2)]:
  expected={n for n in odd_r_neighbors(*cell) if n in fit.cells}
  assert expected
  assert all(max(abs(n[0]-cell[0]),abs(n[1]-cell[1]))<=1 for n in expected)

def test_neighbor_function_has_six_cells():
 assert len(odd_r_neighbors(2,2))==6 and len(odd_r_neighbors(3,2))==6
def test_canonical_top_row_is_zero():
 fit=fit_lattice(board(missing={(0,0),(0,1)}))
 assert min(row for row, col in fit.cells) == 0
 assert max(row for row, col in fit.cells) == 3
