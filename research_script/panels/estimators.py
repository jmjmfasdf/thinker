"""COSYNE panel estimators, isolated from historical path/launch side effects.

Adapted verbatim from the two source files listed in panel_mapping.csv.
The local U slope is unrestricted; K is not an observed-curve predictor.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.special import expit
GRID = np.round(np.arange(-4, 4.0001, .05), 8)
BW, DX, RCOND = .30, .01, 1e-8

def moments(m, x, y):
    """Local within-run covariances from all observations in fixed kernel support."""
    low, high = GRID[0]-5*BW, GRID[-1]+5*BW
    inside = (m>=low-DX/2)&(m<=high+DX/2)
    n = len(m)
    m,x,y = m[inside],x[inside],y[inside]
    bins = np.rint((m-low)/DX).astype(int)
    size = int(round((high-low)/DX))+1
    sample = np.rint((GRID-low)/DX).astype(int)
    q = np.column_stack([x,y]); p=x.shape[1]; d=q.shape[1]
    tri = np.triu_indices(d)
    hist = np.empty((size,1+d+len(tri[0])))
    hist[:,0] = np.bincount(bins,minlength=size)
    for i in range(d):
        hist[:,1+i] = np.bincount(bins,weights=q[:,i],minlength=size)
    for col,(i,j) in enumerate(zip(*tri),1+d):
        hist[:,col] = np.bincount(bins,weights=q[:,i]*q[:,j],minlength=size)
    smooth = gaussian_filter1d(hist,BW/DX,axis=0,mode='constant',truncate=5)[sample]
    h=np.arange(-int(5*BW/DX),int(5*BW/DX)+1)
    kernel=np.exp(-.5*(h/(BW/DX))**2);kernel/=kernel.sum()
    w2=fftconvolve(hist[:,0],kernel**2,mode='same')[sample]
    s=smooth[:,0]
    avg=np.divide(smooth,s[:,None],out=np.zeros_like(smooth),where=s[:,None]>1e-15)
    cov=np.zeros((len(GRID),d,d))
    cov[:,tri[0],tri[1]]=avg[:,1+d:]
    cov[:,tri[1],tri[0]]=avg[:,1+d:]
    cov-=avg[:,1:1+d,None]*avg[:,None,1:1+d]
    mass=s/n
    neff=np.divide(s*s,w2,out=np.zeros_like(s),where=w2>1e-15)
    return cov[:,:p,:p]*mass[:,None,None],cov[:,:p,p:]*mass[:,None,None],mass,neff

def solve(g,b,indices):
    """OLS U coefficient after controls, using a scale-invariant Schur complement."""
    g=g[:,indices][:,:,indices];b=b[:,indices]
    diag=np.maximum(np.diagonal(g,axis1=1,axis2=2),0)
    scale=np.sqrt(diag)
    safe=np.where(scale>1e-12,scale,1.)
    corr=g/(safe[:,:,None]*safe[:,None,:])
    target=b/safe[:,:,None]
    cc=corr[:,1:,1:]
    eig,v=np.linalg.eigh(cc)
    keep=eig>np.maximum(eig[:,-1:],1e-12)*RCOND
    inv=np.divide(1.,eig,out=np.zeros_like(eig),where=keep)
    pinv=np.einsum('gij,gj,gkj->gik',v,inv,v,optimize=True)
    nuisance=np.einsum('gi,gij->gj',corr[:,0,1:],pinv,optimize=True)
    residual=1.-np.einsum('gi,gi->g',nuisance,corr[:,0,1:])
    rhs=target[:,0]-np.einsum('gi,gij->gj',nuisance,target[:,1:],optimize=True)
    good=(scale[:,0]>1e-12)&(residual>=1e-5)
    slope=np.divide(rhs,(residual*safe[:,0])[:,None],out=np.full_like(rhs,np.nan),where=good[:,None])
    return slope, residual, keep.sum(1)+1

def gain(y,t,w,mask):
    use=mask&np.isfinite(y)&np.isfinite(t)&(w>0)
    if use.sum()<5:return np.nan
    den=np.sum(w[use]*t[use]**2)
    return max(0.,float(np.sum(w[use]*t[use]*y[use])/den)) if den>1e-20 else np.nan

def basis(v, knots):
    """Nonconstant natural cubic cardinal basis with linear extrapolation."""
    f = CubicSpline(knots, np.eye(len(knots)), bc_type='natural')
    v = np.asarray(v)
    b = f(np.clip(v, knots[0], knots[-1]))
    for mask, endpoint in ((v < knots[0], knots[0]), (v > knots[-1], knots[-1])):
        b[mask] = f(endpoint)+(v[mask]-endpoint)[:, None]*f(endpoint, 1)
    return b[:, 1:]

def fit_model(x, y, weights):
    # Remove constant/redundant nuisance columns without penalizing the K curve.
    gram = x.T@(weights[:, None]*x)
    scale = np.sqrt(np.maximum(np.diag(gram), 0))
    scale[scale < 1e-12] = 1.
    eig, vec = np.linalg.eigh(gram/scale[:, None]/scale[None, :])
    keep = eig > eig.max()*1e-10
    transform = vec[:, keep]/np.sqrt(eig[keep])[None, :]/scale[:, None]
    z = np.ascontiguousarray(x@transform)
    np.testing.assert_allclose(z.T@(weights[:, None]*z), np.eye(keep.sum()), atol=5e-6)
    def objective(theta):
        eta = z@theta
        return (float(weights@(np.logaddexp(0, eta)-y*eta)),
                z.T@(weights*(expit(eta)-y)))
    result = minimize(objective, np.zeros(z.shape[1]), jac=True, method='L-BFGS-B',
                      options=dict(maxiter=900, gtol=2e-8, ftol=1e-12, maxls=40))
    assert result.success and abs(result.jac).max() < 2e-5, (result.message, result.jac)
    beta = transform@result.x
    np.testing.assert_allclose(x@beta, z@result.x, atol=2e-7)
    calibration_error = float(weights@(expit(x@beta)-y))
    assert abs(calibration_error) < 2e-5
    return beta, dict(success=bool(result.success), iterations=int(result.nit),
                      max_gradient=float(abs(result.jac).max()),
                      calibration_error=calibration_error, rank=int(keep.sum()))

