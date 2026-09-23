import sys, os, numpy as np
sys.path.insert(0, os.path.abspath("."))
from scripts.run_reference_pipeline import sanitize_trajectory, PANDA_VEL

def test_sanitizer_kills_boundary_jumps():
    q = np.zeros((5,7)); q[:,3] = -1.5708; q[2,0]=2.5; q[3,0]=2.5; q[4,0]=2.55
    r = np.array([True,True,True,True,True])
    q2,r2,_ = sanitize_trajectory(q,r,["t"]*5,1/30.)
    assert np.all(np.abs(np.diff(q2,axis=0))/(1/30.) <= 0.9*PANDA_VEL.max()+1e-9)
    assert not r2[2] and r2[0]
