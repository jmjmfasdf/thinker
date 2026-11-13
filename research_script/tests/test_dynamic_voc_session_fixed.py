"""Scientific checks for pooled session-FE fitting and temporal covariance."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import statsmodels.api as sm

PATH = Path(__file__).resolve().parents[1]/"dynamic_voc_session_fixed.py"
SPEC = importlib.util.spec_from_file_location("session_fixed_test", PATH)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class SessionFixedTests(unittest.TestCase):
    def frame(self, n=2000):
        rng = np.random.default_rng(6)
        sessions = np.repeat(np.arange(1,5), n//4)
        x = rng.normal(size=n) + sessions*2
        c = rng.normal(size=n)
        return pd.DataFrame({"x":x,"c":c,"session":sessions,"run_label":sessions.astype(str),
                             "real_step":np.tile(np.arange(n//4),4)})

    def test_linear_matches_explicit_categorical_session_ols(self):
        d = self.frame(); rng = np.random.default_rng(7)
        d["y"] = .3*d.x - 2*d.session + .2*d.c + rng.normal(size=len(d))
        result = m.fit_pooled(d,"y",["x"],["c"],False,lags=(0,))
        x = (d.x-d.x.mean())/d.x.std(ddof=0)
        design = np.column_stack([x, np.ones(len(d)), pd.get_dummies(d.session,drop_first=True), d.c])
        y = (d.y-d.y.mean())/d.y.std(ddof=0)
        fit = sm.OLS(y,design).fit(cov_type="HC1")
        self.assertAlmostEqual(result[0]["mean"],fit.params.iloc[0],places=9)
        self.assertAlmostEqual(result[0]["se"],fit.bse.iloc[0],places=9)
        self.assertEqual(result[0]["n_session_fixed_effects"],3)

    def test_logistic_matches_unpenalized_session_glm(self):
        d = self.frame(); rng = np.random.default_rng(9)
        eta = .4*d.x - .8*d.session + .25*d.c
        d["y"] = rng.binomial(1,1/(1+np.exp(-eta)))
        result = m.fit_pooled(d,"y",["x"],["c"],True,lags=(4,))
        design = np.column_stack([(d.x-d.x.mean())/d.x.std(ddof=0),np.ones(len(d)),pd.get_dummies(d.session,drop_first=True),d.c])
        fit = sm.GLM(d.y,design,family=sm.families.Binomial()).fit()
        self.assertEqual(result[0]["status"],"ok")
        self.assertAlmostEqual(result[0]["mean"],fit.params.iloc[0],places=5)

    def test_hac_matches_direct_lag_products_and_zero_fills_gaps(self):
        influence=np.array([[1.],[2.],[3.],[4.]])
        run=np.array(['a','a','a','b']);step=np.array([0,0,2,0])
        result,n=m.temporal_variances(influence,run,step,(0,1,2))
        # Same decision adds to 3, excluded decision remains 0, run b is separate.
        expected=np.array([3.,0.,3.])
        for lag in (0,1,2):
            v=expected@expected+16
            for k in range(1,lag+1):v+=2*(1-k/(lag+1))*(expected[k:]@expected[:-k])
            self.assertAlmostEqual(result[lag][0],v)
        self.assertEqual(n,3)

    def test_persistent_scores_inflate_temporal_se(self):
        score=np.repeat([1.,-1.],100)[:,None]
        result,_=m.temporal_variances(score,np.array(['r']*200),np.arange(200),(0,32))
        self.assertGreater(result[32][0],10*result[0][0])

    def test_focal_confound_is_not_silently_dropped(self):
        d=self.frame();d['duplicate']=d.x;d['y']=d.x+d.c
        result=m.fit_pooled(d,'y',['x'],['duplicate','c'],False,lags=(0,))
        self.assertEqual(result[0]['status'],'unidentifiable_predictor')

    def test_binary_uninformative_session_is_explicitly_excluded(self):
        d=self.frame();rng=np.random.default_rng(1);d['y']=rng.binomial(1,.4,len(d))
        d.loc[d.session.eq(1),'y']=0
        result=m.fit_pooled(d,'y',['x'],['c'],True,lags=(4,))
        self.assertEqual(result[0]['excluded_constant_outcome_sessions'],[1])
        self.assertEqual(result[0]['n_sessions'],3)

    def test_perfect_separation_is_not_a_finite_significant_effect(self):
        d=self.frame();d['y']=(d.x-2*d.session>0).astype(float)
        result=m.fit_pooled(d,'y',['x'],['c'],True,lags=(4,))
        self.assertEqual(result[0]['status'],'complete_separation')
        self.assertTrue(np.isnan(result[0]['p']))
        self.assertTrue(np.isnan(result[0]['mean']))

    def test_group_inference_uses_subjects_not_decisions(self):
        rows=[]
        for s in range(1,7):
            for lag in m.LAGS:
                rows.append(dict(subject=s,game=0,family='Q3_human',source='ema',population='overall',
                    term='ema_initial',kind='log_odds',hac_lag=lag,mean=s/10,status='ok',n_rows=100000))
        summary=m.group_summary(pd.DataFrame(rows))
        self.assertEqual(summary.n_subjects.iloc[0],6)
        self.assertAlmostEqual(summary['mean'].iloc[0],.35)
        self.assertAlmostEqual(summary.p.iloc[0],.03125)


if __name__ == '__main__':unittest.main()
