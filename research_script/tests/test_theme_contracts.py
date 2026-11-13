"""Scientific invariants for the dynamic thematic analysis."""
import sys
from pathlib import Path
import unittest
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.features import revision_summary,add_history
from common.native import prepare_matrix,event_features
from common.statistics import group_inference
from common.prediction import scores
from analyses.neural import forward
from common.outcomes import risk_set
from analyses.continuity import attach_arrived_mismatch
from analyses.behavior import search_pairs

class ThemeContracts(unittest.TestCase):
    def test_update_denominators_and_zero_budget(self):
        d=pd.DataFrame({'run_label':['r']*3,'real_step':[1,2,3],'real_step_id':['a','b','c'],
                        'additional_computations':[0,4,2],'margin':[0.,0.,4.]})
        e=pd.DataFrame({'run_label':['r']*3,'real_step':[2,2,3],'real_step_id':['b','b','c'],
                        'root_q_revision':[2.,100.,-4.],'value_revision_valid':[1,0,1]})
        z=revision_summary(e,d)
        np.testing.assert_allclose(z.U_sum,[0,2,4]);np.testing.assert_allclose(z.U_all,[0,.5,2])
        self.assertTrue(np.isnan(z.U_valid.iloc[0]));self.assertEqual(z.U_valid.iloc[1],2)
        self.assertAlmostEqual(z.K.iloc[1],.5);self.assertLess(z.K.iloc[2],z.U_all.iloc[2])

    def test_training_only_imputation_and_scaling(self):
        d=pd.DataFrame({'x':[1.,3.,999.],'missing':[2.,np.nan,900.]})
        x,_=prepare_matrix(d,['x','missing'],np.array([1,1,0],bool))
        np.testing.assert_allclose(x[:2,0],[-1,1]);self.assertEqual(x[2,0],997)

    def test_likelihood_known_values_and_zero_floor(self):
        r=scores([0,1],np.array([[.8,.2],[.25,.75]]))
        self.assertAlmostEqual(r['log_score'],(np.log(.8)+np.log(.75))/2)
        self.assertEqual(r['accuracy'],1.)
        r=scores([0],np.array([[0.,1.]]));self.assertEqual(r['zero_probability_count'],1)

    def test_six_subjects_not_checkpoint_replicates(self):
        d=pd.DataFrame([dict(subject=s,game=0,term='x',value=s/10) for s in range(1,7) for cp in range(4)])
        r=group_inference(d,['game','term'])
        self.assertEqual(r.n_subjects.iloc[0],6);self.assertAlmostEqual(r['mean'].iloc[0],.35)
        self.assertAlmostEqual(r.p.iloc[0],.03125)

    def test_missing_subject_not_silently_complete(self):
        d=pd.DataFrame([dict(subject=s,game=0,term='x',value=s/10) for s in range(1,6)])
        r=group_inference(d,['game','term']);self.assertEqual(r.status.iloc[0],'incomplete_subjects')
        self.assertTrue(np.isnan(r.p_max_t.iloc[0]))

    def test_neural_forward_is_causal(self):
        y=forward([10.,11.],np.array([[1.],[1.]]),n_tr=30)
        self.assertLess(np.abs(y[:10]).max(),1e-12)
        self.assertGreater(y[20,0],0)
        impulse=forward([10.],np.array([[1.]]),n_tr=30,impulse_cols=(0,))
        self.assertLess(np.abs(impulse[:10]).max(),1e-12)
        self.assertLess(impulse[-1,0],impulse[16,0])

    def test_search_target_is_the_next_selected_control(self):
        e=pd.DataFrame(dict(run_label=['r']*3,real_step=[1]*3,real_step_id=['a']*3,choice_index=[0,1,2],
            accepted_control=[0,1,2],voluntary=[True]*3,has_next_choice=[True,True,False],search_computation=[1,1,0]))
        e['continue']=[1.,1.,0.]
        z=search_pairs(e)
        self.assertEqual(z.next_continue.tolist(),[1.,0.])
        self.assertEqual(z.next_proceed.tolist(),[0.,0.])

    def test_active_bout_risk_does_not_exclude_other_active_actions(self):
        d=pd.DataFrame(dict(incumbent_action=[0,0,2,2,2,8],human_action=[0,2,2,0,3,6],scored=1,n_actions=9))
        out=risk_set(d,'active_to_noop')
        self.assertEqual(out.outcome.tolist(),[0,1,0,0])
        self.assertEqual(risk_set(d,'noop_to_active').outcome.tolist(),[0,1])

    def test_forecast_error_is_unavailable_before_its_outcome(self):
        d=pd.DataFrame(dict(checkpoint=[1]*4,run_label=['r']*4,episode_id=['e','e','e','new'],
            decision_time=[0.,1.,2.,3.],first_control_native_event=[1,3,5,7]))
        f=pd.DataFrame(dict(run_label=['r'],episode_id=['e'],outcome_available_time=[1.5],
            outcome_available_event=[5],state_comparison_primary=[1],reward_prediction_error=[-2.]))
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/'checkpoint_leave01/sub001_game0';p.mkdir(parents=True)
            f.to_csv(p/'forecast_outcomes.csv.gz',index=False)
            result=attach_arrived_mismatch(d,Path(tmp),1,0)
        self.assertTrue(result.arrived_reward_error.iloc[:2].isna().all())
        self.assertEqual(result.arrived_reward_error.iloc[2],2.)
        self.assertTrue(np.isnan(result.arrived_reward_error.iloc[3]))

if __name__=='__main__':unittest.main()
