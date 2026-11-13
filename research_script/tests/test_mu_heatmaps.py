"""Mean-of-probabilities, original kernel recovery and zero-mass contracts."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from panels.mu_heatmaps import kernel_surface, mean_surfaces, REFERENCE, axis_limits, marginal


class SmoothHeatmaps(unittest.TestCase):
    def test_group_is_equal_people_not_pooled_counts(self):
        stack=np.array([[[.1,np.nan]],[[.2,.1]],[[.7,.2]],[[.8,.3]],[[np.nan,np.nan]],[[np.nan,np.nan]]])
        z,n=mean_surfaces(stack)
        self.assertAlmostEqual(z[0,0],.45)
        self.assertTrue(np.isnan(z[0,1]))
        self.assertEqual(n.tolist(),[[4,3]])

    def test_exact_original_surface_estimator(self):
        if not REFERENCE.exists(): self.skipTest('Original source unavailable')
        spec=importlib.util.spec_from_file_location('old_mu_surface',REFERENCE)
        mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        rng=np.random.default_rng(10); frames=[]; arrays=[]
        for subject in range(6):
            n=1800+subject*150
            x=rng.normal(size=n); u=rng.exponential(size=n); u[:300]=0
            y=rng.binomial(1,1/(1+np.exp(x-.2*u)))
            d=pd.DataFrame(dict(M_z=x,U_relative=u,U_all=u,outcome=y))
            frames.append(d.rename(columns={'M_z':'M','U_relative':'U','outcome':'switch'}))
            arrays.append(kernel_surface(d,(-3,3),(0,4))[0])
        expected,support=mod.surface(frames,(-3,3),(0,4))
        actual,n=mean_surfaces(np.stack(arrays))
        np.testing.assert_allclose(actual,expected,atol=1e-14,equal_nan=True)
        np.testing.assert_array_equal(n,support)

    def test_zero_updates_are_retained_and_support_is_separate(self):
        d=pd.DataFrame(dict(M_z=np.zeros(100),U_relative=np.zeros(100),U_all=np.zeros(100),
                            outcome=np.tile([0,0,0,1],25)))
        z,n,pos=kernel_surface(d,(-1,1),(0,1))
        self.assertTrue(np.isfinite(z).any())
        np.testing.assert_allclose(z[np.isfinite(z)],.25)
        self.assertEqual(pos.max(),0)
        self.assertEqual(n.max(),100)
        self.assertTrue(np.isnan(z[-1]).all())
        density=marginal(d.U_relative,(0,1))
        self.assertAlmostEqual(np.trapz(density,dx=1/179),1.)

    def test_game_axes_do_not_collapse_when_99_percent_u_is_zero(self):
        d=pd.DataFrame(dict(M_z=np.linspace(-1,1,1000),U_relative=np.r_[np.zeros(999),10.],
            incumbent_action=np.zeros(1000),human_action=np.zeros(1000),scored=np.ones(1000)))
        x,y,fallback=axis_limits([d])
        self.assertTrue(fallback); self.assertEqual(y.tolist(),[0.,10.])


if __name__=='__main__':unittest.main()
