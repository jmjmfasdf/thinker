import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.value_update_candidates import features_from_events, METRICS, exact_family


class ValueUpdateCandidates(unittest.TestCase):
    def synthetic(self):
        d=pd.DataFrame(dict(real_step_id=['a','b'],n_actions=[3,3],additional_computations=[3,0],
            incumbent_action=[0,1],human_action=[1,1],U_sum=[3.,0.],U_all=[1.,0.],signed_update_sum=[-1.,0.],
            valid_update_count=[2,0]))
        for a in range(3):
            d[f'model_root_p_{a}']=[(.5,.3,.2)[a],(.2,.6,.2)[a]]
            d[f'root_qs_mean_initial_{a}']=[(1.,2.,0.)[a],0.]
            d[f'root_qs_mean_terminal_{a}']=[(-1.,3.,4.)[a],0.]
            d[f'root_ns_initial_{a}']=[(1.,1.,0.)[a],0.]
            d[f'root_ns_terminal_{a}']=[1.,0.]
        e=pd.DataFrame(dict(real_step_id=['a']*3,branch=[0,1,2],root_q_revision=[-2.,1.,np.nan],
            value_revision_valid=[1,1,0],root_q_after_sampled=[-1.,3.,4.],root_n_after_sampled=[1.,1.,1.]))
        return d,e

    def test_signed_and_absolute_revisions_and_first_evaluation(self):
        d,e=self.synthetic();r=features_from_events(d,e)
        self.assertAlmostEqual(r.signed_mean.iloc[0],-1/3)
        self.assertEqual(r.net_branch_abs_mean.iloc[0],1.)
        self.assertEqual(r.fixed_advantage_mean.iloc[0],1.)
        self.assertEqual(r.signed_balance.iloc[0],1.)
        self.assertEqual(r.event_Q_sum.iloc[0],6.)
        self.assertEqual(r.event_Q_mean.iloc[0],2.)
        self.assertEqual(r.terminal_Q_sum.iloc[0],6.)
        self.assertEqual(r.endpoint_D.iloc[0],3.)
        self.assertTrue(np.isnan(r.endpoint_D.iloc[1]))
        self.assertTrue(r.loc[1,list(METRICS)].eq(0).all())

    def test_human_target_never_defines_candidate_features(self):
        d,e=self.synthetic();r=features_from_events(d,e)
        d.human_action=[2,0];z=features_from_events(d,e)
        np.testing.assert_array_equal(r[list(METRICS)],z[list(METRICS)])

    def test_max_t_keeps_missing_subject_tests_untestable(self):
        rows=[]
        for metric in ['coherent','missing','null']:
            for s in range(1,7):
                value=(1+s*.01) if metric=='coherent' else np.nan if metric=='missing' and s==6 else 0.
                rows.append(dict(subject=s,game=1,population='overall',cohort='all_matched',
                                 metric=metric,representation='within_run_sd',value=value))
        r=exact_family(pd.DataFrame(rows)).set_index('metric')
        self.assertEqual(r.loc['coherent','p_max_t'],2/64)
        self.assertTrue(np.isnan(r.loc['missing','p_max_t']))
        self.assertEqual(r.loc['null','p_max_t'],1.)


if __name__=='__main__':unittest.main()
