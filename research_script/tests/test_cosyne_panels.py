"""Censoring, same-head endpoints, variable-length support, estimator recovery."""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from panels.data import bouts_from_decisions, competing_risks, native_search, update_composition
from panels.estimators import moments, solve, GRID
from panels.raw_behavior import frame_from_raw


class PanelContracts(unittest.TestCase):
    def test_original_human_frames_and_action_age(self):
        d=frame_from_raw(np.eye(3)[[0,0,1,1,0]],np.array([0.,.1,.2,.3,.4]),
                         [1,0,0,0,0],[0,0,0,0,0],'sub001_ses01_block1_game1')
        self.assertEqual(len(d),5)
        self.assertEqual(d.incumbent_action.tolist(),[-1,0,0,1,1])
        np.testing.assert_allclose(d.age_seconds.iloc[1:],[.1,.2,.1,.2])

    def test_bouts_never_connect_across_episode_boundaries(self):
        d = pd.DataFrame(dict(run_label=['r']*6, segment=[1, 1, 1, 2, 2, 2],
            real_step=[1, 2, 3, 5, 6, 7], human_action=[1, 1, 0, 2, 2, 2],
            time_seconds=[0., 1., 2., 3., 4., 5.]))
        b = bouts_from_decisions(d)
        self.assertEqual(b.route.tolist(), ['to_noop', 'censored', 'censored'])
        self.assertEqual(b.duration_seconds.tolist(), [2., 0., 2.])

    def test_competing_risks_conserve_probability(self):
        b = pd.DataFrame(dict(duration_seconds=[1., 1., 2., 3.],
                             route=['to_noop', 'to_active', 'censored', 'to_noop']))
        # Duplicate independent records to meet plotting support threshold.
        b = pd.concat([b]*10, ignore_index=True)
        r = competing_risks(b, np.array([0., 1., 2., 3.]))
        np.testing.assert_allclose(r.survival+r.to_noop+r.to_active, 1.)
        self.assertEqual(r.survival.iloc[1], .5)
        self.assertEqual(r.to_noop.iloc[1], .25)

    def test_real_head_and_deep_choices_are_not_endpoints(self):
        d = pd.DataFrame(dict(run_label=['r']*3, real_step=[1, 2, 3], real_step_id=['a', 'b', 'c'],
            session=[1]*3, incumbent_action=[0]*3, human_action=[0]*3, U_all=[.1]*3,
            additional_computations=[2, 0, 1], n_actions=[2]*3, scored=[1]*3,
            model_root_p_0=[.6]*3, model_root_p_1=[.4]*3))
        rows = []
        for key, step, head, depth, p, order in [('a', 1, 1, 0, .8, 0), ('a', 1, 1, 2, .1, 1),
                ('a', 1, 1, 0, .6, 2), ('a', 1, 2, 0, .01, 3),
                ('b', 2, 1, 0, .8, 0), ('b', 2, 2, 0, .01, 1),
                ('c', 3, 1, 0, .8, 0), ('c', 3, 2, 0, .01, 1)]:
            rows.append(dict(run_label='r', real_step=step, real_step_id=key, policy_type=head,
                policy_type_valid=1, event_order=order, is_control_choice=int(head == 1),
                current_node_depth_pre=depth, current_node_depth_pre_valid=1,
                actor_action_probability_valid=1, actor_action_probability_0=p, actor_action_probability_1=1-p))
        pe = pd.DataFrame(dict(run_label=['r'], real_step=[1], real_step_id=['a'],
                              root_q_revision=[.1], value_revision_valid=[1], branch=[0]))
        with TemporaryDirectory() as tmp:
            for cp in range(1, 5):
                p = Path(tmp)/f'checkpoint_leave{cp:02d}/sub001_game1'; p.mkdir(parents=True)
                pe.to_csv(p/'persistence_events.csv.gz', index=False)
                pd.DataFrame(rows).to_csv(p/'native_events.csv.gz', index=False)
            ep, _, _ = native_search(d, tmp, 1, 1)
        self.assertEqual(ep.real_step_id.tolist(), ['a'])
        self.assertAlmostEqual(ep.endpoint_change.iloc[0], .2)
        self.assertEqual(ep.root_choices.iloc[0], 2)

    def test_absent_later_computation_is_not_zero_padding(self):
        d = pd.DataFrame(dict(run_label=['r', 'r'], U_all=[1., 1.]))
        events = pd.DataFrame(dict(real_step_id=['a', 'a', 'b'], status2_position=[0, 1, 0],
            branch_type=['strongest_alternative']*3, update=[2., 8., 1.], value_revision_valid=[1]*3,
            run_label=['r']*3))
        pairs = pd.DataFrame(dict(population=['overall'], run_label=['r'], switch_id=['a'], persist_id=['b']))
        c, _ = update_composition(d, events, pairs)
        self.assertEqual(c.status2_position.unique().tolist(), [0])
        self.assertEqual(c.loc[c.branch.eq('strongest_alternative'), 'value'].iloc[0], 1.)

    def test_local_slope_does_not_impose_competition_shape(self):
        rng = np.random.default_rng(3); m = rng.uniform(-5, 5, 12000); u = rng.normal(size=len(m))
        nuisance = rng.normal(size=len(m)); x = np.column_stack([u, m, nuisance])
        y = (2*u+3*m+4*nuisance)[:, None]
        g, b, _, _ = moments(m, x, y)
        slope, _, _ = solve(g, b, [0, 1, 2])
        np.testing.assert_allclose(slope[np.abs(GRID) < 3, 0], 2., atol=1e-8)


if __name__ == '__main__': unittest.main()
