import sys
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.matching import nearest_visual


def test_constant_decimal_cannot_mask_the_nearest_neighbour():
    # Real failure: a constant feature had SD 5.6e-17 and mean .18916684.
    control=np.column_stack([np.full(100,.18916684),np.arange(100.)])
    case=np.array([[.18916684,77.1],[.18916684,12.8]])
    strata=np.vstack([control,case])
    distance,index=nearest_visual(case,control,strata)
    np.testing.assert_array_equal(index[:,0],[77,13])
    assert np.all(distance>0)


def test_distances_agree_with_direct_reference_and_translation():
    rng=np.random.default_rng(5)
    control=rng.normal(size=(90,4));case=rng.normal(size=(25,4))
    strata=np.vstack([control,case]);scale=(strata-strata[0]).std(0,ddof=1)
    reference=cdist(case/scale,control/scale)
    dist,index=nearest_visual(case,control,strata)
    np.testing.assert_array_equal(index[:,0],reference.argmin(1))
    np.testing.assert_allclose(dist[:,0],reference.min(1),rtol=1e-12)
    _,translated=nearest_visual(case+1e6,control+1e6,strata+1e6)
    np.testing.assert_array_equal(index,translated)


def test_all_constant_and_missing_features_are_real_ties():
    d,i=nearest_visual([[.19,np.nan]],[[.19,np.nan],[.19,np.nan]],[[.19,np.nan]]*3)
    assert d[0,0]==0 and i[0,0]==0
