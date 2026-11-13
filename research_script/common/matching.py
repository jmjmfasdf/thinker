"""Stable visual matching; constant features must never create huge offsets."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial import cKDTree
from .io import sha

VERSION = 'centered-direct-distance-v2'


def nearest_visual(case, control, stratum):
    """Return distance and control index, after stratum-only scaling.

    Subtract BEFORE computing SD. A constant decimal's naive SD may be a
    roundoff-sized nonzero, and dividing an uncentered mean by it creates
    values of order 1e16. Dot-product Euclidean distance can then turn
    unrelated images into distance-zero neighbours.
    """
    def filled(x):
        a=np.asarray(x,dtype=np.float64)
        return np.where(np.isnan(a),0.,a)
    values,cases,controls=filled(stratum),filled(case),filled(control)
    if not (np.isfinite(values).all() and np.isfinite(cases).all() and np.isfinite(controls).all()):
        raise ValueError('Nonfinite visual matching input')
    if not len(cases) or not len(controls):raise ValueError('Empty matching group')
    center = values[0]
    centered = values-center
    scale = centered.std(axis=0, ddof=1) if len(values)>1 else np.zeros(values.shape[1])
    threshold = 32*np.finfo(np.float64).eps*np.maximum(1,np.max(abs(values),axis=0))
    use = np.isfinite(scale)&(scale>threshold)
    if not use.any():return np.zeros((len(cases),1)),np.zeros((len(cases),1),dtype=int)
    x = (cases[:,use]-center[use])/scale[use]
    y = (controls[:,use]-center[use])/scale[use]
    distance, index = cKDTree(y).query(x,k=1,workers=1)
    direct = np.linalg.norm(x-y[index],axis=1)
    np.testing.assert_allclose(distance,direct,rtol=1e-10,atol=1e-12)
    return direct[:,None],index[:,None]


def corrected_pair_root(outputs, label):
    """Never silently reuse the confirmed numerically invalid v1 matches."""
    root=Path(outputs)/'02/matching_v2'/label
    path=root/'manifest.json'
    if not path.exists():
        raise RuntimeError('Regenerate stable visual matches before this analysis: '+str(root))
    m=json.loads(path.read_text())
    if m.get('status')!='COMPLETED' or m.get('matching_version')!=VERSION or m.get('matching_source')!=sha(__file__):
        raise RuntimeError('Stale or incomplete corrected matching: '+str(root))
    return root
