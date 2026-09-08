from bubblebot.vision.isolation import CircleCandidate, isolate_board_candidates

def board(origin=(100,50),sx=36,sy=31,rows=5,cols=8,missing=()):
 return [CircleCandidate(origin[0]+c*sx+(sx/2 if r%2 else 0),origin[1]+r*sy, sx/2.57, f"b{r}_{c}") for r in range(rows) for c in range(cols) if (r,c) not in missing]

def test_clean_board():
 r=isolate_board_candidates(board(rows=5)); assert len(r.accepted)==40 and not r.rejected

def test_shooter_next_and_ui_rejected():
 cs=board(); cs += [CircleCandidate(244,300,14,"shooter"),CircleCandidate(280,270,10,"next"),CircleCandidate(20,200,12,"ui_left"),CircleCandidate(600,200,12,"ui_right"),CircleCandidate(250,10,12,"ui_top")]
 r=isolate_board_candidates(cs); assert len(r.accepted)==40; assert {c.candidate_id for c in r.rejected}>={"shooter","next","ui_left","ui_right","ui_top"}

def test_sparse_holes_edges_and_offset():
 cs=board(missing={(1,2),(2,3),(4,0),(4,1),(4,2),(4,3),(4,4)})
 r=isolate_board_candidates(cs); assert len(r.accepted)==len(cs)

def test_near_false_positive_and_mini_grid():
 cs=board(); cs += [CircleCandidate(101,210,14,"near"),CircleCandidate(800,100,14,"m1"),CircleCandidate(836,100,14,"m2"),CircleCandidate(818,131,14,"m3")]
 r=isolate_board_candidates(cs); assert len(r.accepted)==40; assert "near" in r.rejection_reason

def test_scaled_shifted_and_aligned_shooter():
 cs=board((311,137),44,38,5,8); cs += [CircleCandidate(311,400,17,"aligned_shooter")]
 r=isolate_board_candidates(cs); assert len(r.accepted)==40; assert "aligned_shooter" in r.rejection_reason

def sparse_bottom(count):
 keep=set(range(3,3+count))
 return board(missing={(4,c) for c in range(8) if c not in keep})

def test_one_bubble_legitimate_bottom_row():
 cs=sparse_bottom(1); r=isolate_board_candidates(cs); assert len(r.accepted)==len(cs)

def test_two_bubble_legitimate_bottom_row():
 cs=sparse_bottom(2); r=isolate_board_candidates(cs); assert len(r.accepted)==len(cs)

def test_three_bubble_legitimate_bottom_row():
 cs=sparse_bottom(3); r=isolate_board_candidates(cs); assert len(r.accepted)==len(cs)

def test_sparse_row_after_one_row_gap_is_rejected():
 cs=board(rows=4); cs += [CircleCandidate(100+3*36,50+5*31,14,"gap")]
 r=isolate_board_candidates(cs); assert "gap" in r.rejection_reason

def test_lattice_aligned_shooter_several_rows_below_is_rejected():
 cs=board(rows=4); cs += [CircleCandidate(100+2*36,50+8*31,14,"shooter_far")]
 r=isolate_board_candidates(cs); assert "shooter_far" in r.rejection_reason
