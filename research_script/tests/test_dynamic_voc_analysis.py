"""Scientific alignment and validation tests for the dynamic 03 analysis."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

PATH = Path(__file__).resolve().parents[1] / "03_dynamic_voc_mechanisms.py"
SPEC = importlib.util.spec_from_file_location("dynamic_voc_analysis", PATH)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class DynamicVoCTests(unittest.TestCase):
    def path_events(self):
        # Two rewards, a reset, a fresh branch, STOP, then another real decision.
        return pd.DataFrame({
            "run_label":["r"]*6, "real_step":[1,1,1,1,1,2],
            "current_node_depth_pre":[0,1,2,0,1,0],
            "current_node_depth_post":[1,2,0,1,1,1],
            "root_trail_r_pre":[0,2,5.6,0,7,0],
            "root_trail_r_post":[2,5.6,0,7,7,11],
            "cur_r_post":[2,4,0,7,7,11],
            "search_computation":[1,1,1,1,0,1], "actor_reset":[0,0,1,0,0,0]})

    def test_undiscounted_sum_does_not_add_resets_or_other_paths(self):
        before, after, failures = m.reconstruct_reward_sums(self.path_events(), .9)
        np.testing.assert_allclose(before, [0,2,6,0,7,0])
        np.testing.assert_allclose(after, [2,6,0,7,7,11])
        self.assertEqual(failures, 0)

    def test_unknown_or_inconsistent_path_remains_missing(self):
        e = self.path_events(); e.loc[1,"root_trail_r_post"] = 9
        before, after, failures = m.reconstruct_reward_sums(e, .9)
        self.assertTrue(np.isnan(after[1]))
        self.assertTrue(np.isnan(before[2]))
        self.assertEqual(failures, 1)
        self.assertEqual(after[3], 7)

    def test_forced_stop_and_next_choice_alignment(self):
        e = self.path_events().iloc[[0,1,4,5]].copy().reset_index(drop=True)
        e["real_step_id"] = ["s1","s1","s1","s2"]
        e["event_order"] = [0,1,2,0]
        e["is_control_choice"] = 1; e["accepted_control"] = [0,0,2,2]
        e["forced_stop"] = [0,0,1,0]; e["forced_stop_valid"] = 1; e["control_valid"] = 1
        e["search_steps_pre"] = [0,1,2,0]
        e["rollout_return_pre"] = e.root_trail_r_pre + .9 ** e.current_node_depth_pre * 3
        e["rollout_return_post"] = e.root_trail_r_post + .9 ** e.current_node_depth_post * 3
        e["cur_v_pre"] = 3; e["depth_discount_pre"] = .9 ** e.current_node_depth_pre
        for s in m.SOURCES:
            e[f"voc_{s}_q_valid"] = 1
            e[f"voc_{s}_q_continue"] = [5,7,9,99]
            e[f"voc_{s}_q_stop"] = 1
            e[f"voc_{s}_voc"] = [4,6,8,98]
        out, qa = m.event_features(e, .9)
        self.assertEqual(out.voluntary.tolist(), [True,True,False,True])
        self.assertEqual(out.loc[0,"next_voc_ema"], 6)
        self.assertEqual(out.loc[1,"next_voc_ema"], 8)
        self.assertTrue(np.isnan(out.loc[2,"next_voc_ema"]))
        self.assertTrue(np.isnan(out.loc[3,"next_voc_ema"]))
        self.assertEqual(out.loc[0,"bootstrap_return"], e.loc[0,"rollout_return_pre"])
        self.assertEqual(qa["forced_choices"], 1)

    def test_cap_ended_decision_does_not_require_a_fictitious_stop(self):
        e=pd.DataFrame({'run_label':['r','r'],'real_step':[1,2], 'real_step_id':['s1','s2'],
                        'accepted_control':[2,1]})
        d=pd.DataFrame({'run_label':['r','r'],'real_step':[1,2], 'real_step_id':['s1','s2'],
                        'forced_stop':[0,1], 'stop_choice_censored':[0,1]})
        self.assertEqual(m.validate_endings(e,d),1)
        d.loc[1,'stop_choice_censored']=0
        with self.assertRaises(ValueError):m.validate_endings(e,d)

    def test_transition_risk_sets_include_direct_active_switch(self):
        d = pd.DataFrame({"incumbent_action":[0,0,2,2,2], "human_action":[0,2,2,0,3], "scored":1})
        d["human_switch"] = d.incumbent_action.ne(d.human_action).astype(float)
        self.assertEqual(m.risk_set(d,"overall").outcome.tolist(), [0,1,0,1,1])
        self.assertEqual(m.risk_set(d,"active_to_noop").outcome.tolist(), [0,1])
        self.assertEqual(m.risk_set(d,"active_to_active").outcome.tolist(), [0,1])

    def test_enduro_actions_six_through_eight_are_retained(self):
        d = pd.DataFrame({"incumbent_action":[6,7,8,8], "human_action":[6,8,0,1],
                          "scored":1, "n_actions":9, "human_switch":[0,1,1,1]})
        self.assertEqual(len(m.risk_set(d,"overall")),4)
        self.assertEqual(m.risk_set(d,"active_to_active").outcome.tolist(),[0,1,1])
        self.assertEqual(m.risk_set(d,"active_to_noop").outcome.tolist(),[0,1])

    def test_behavior_history_is_causal_and_resets_at_episode_end(self):
        def build(actions):
            d = pd.DataFrame({'run_label':['r']*4,'real_step':[1,2,3,4], 'episode_id':['e']*4,
                              'incumbent_action':[0,0,2,2], 'human_action':actions,
                              'prior_done':[0,0,1,0], 'bout_age':[1,2,1,2], 'run_progress':[0,.1,.2,.3],
                              'root_carried':[0]*4,'root_carried_valid':[1]*4,
                              'additional_computations':[0]*4,'max_path_depth':[0]*4})
            for a in range(9):
                d[f'model_root_p_{a}']=1/9
                d[f'root_qs_mean_initial_{a}']=a
                d[f'root_ns_initial_{a}']=1
            return m.add_behavior_controls(d)
        a=build([0,2,2,0]); b=build([0,0,2,0])
        self.assertEqual(a.loc[1,'recent10'],b.loc[1,'recent10'])
        self.assertEqual(a.loc[2,'recent10'],0)
        self.assertEqual(a.loc[3,'recent10'],0)
        self.assertIn('previous_action_8',a)

    def test_preprocessing_uses_training_rows_only(self):
        a = pd.DataFrame({"x":[1.,3.,5.], "missing":[np.nan,2.,7.]})
        b = a.copy(); b.loc[2] = [1e8,-1e8]
        train = np.array([True,True,False])
        xa, na = m.prepare_matrix(a,["x","missing"],train)
        xb, nb = m.prepare_matrix(b,["x","missing"],train)
        self.assertEqual(na, nb)
        np.testing.assert_array_equal(xa[train],xb[train])

    def test_checkpoint_replicas_are_not_independent_subjects(self):
        records=[]
        for subject in range(1,7):
            for cp in range(1,5):
                for session in range(1,5):
                    records.append(dict(subject=subject, checkpoint=cp, held_out=cp==session,
                        run_label=f's{subject}_ses{session}', game=1, family='Q1_gate', source='ema',
                        population='all', term='voc', kind='beta', value=float(subject)))
        result=m.inference(pd.DataFrame(records),'subject')
        self.assertEqual(result.n_units.tolist(),[6,6])
        np.testing.assert_allclose(result['mean'],3.5)
        np.testing.assert_allclose(result.p,.03125)

    def test_prediction_split_is_full_session_not_only_run(self):
        rng=np.random.default_rng(1); n=200
        d=pd.DataFrame({'x':rng.normal(size=n), 'session':np.repeat([1,2,3,4],50)})
        y=(d.x>0).astype(float).to_numpy()
        test=d.session.eq(2).to_numpy(); train=~test
        pred,qa=m.fit_predict(d,y,['x'],train,test)
        self.assertEqual(len(pred),50)
        self.assertEqual(qa['n_train'],150)
        self.assertGreater(m.roc_auc_score(y[test],pred),.99)


if __name__ == '__main__':
    unittest.main()
