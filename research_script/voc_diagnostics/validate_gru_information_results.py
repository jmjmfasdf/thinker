"""Audit raw artifacts, pairing, held-stream predictions and source bindings."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(root):
    evidence = {}
    missing = []
    for game in ['enduro', 'pong']:
        decoding = root/'v2'/game
        reward = root/'reward_v2'/game
        needed = [decoding/'collection.json', decoding/'decoding_report.json',
                  reward/'manifest.json', reward/'pairs.json',
                  reward/'reward_readout_report.json', reward/'reward_predictions.npz',
                  root/'nonlinear'/game/'report.json']
        absent = [str(p) for p in needed if not p.exists()]
        if absent:
            missing.extend(absent)
            continue
        c = json.loads((decoding/'collection.json').read_text())
        m = json.loads((reward/'manifest.json').read_text())
        assert m['status'] == 'PAIRED_PASS'
        assert m['args']['counts'] == [0, 1, 3, 6]
        assert m['args']['seeds'] == [22301, 22381]
        assert m['args']['batch_size'] == 8 and m['args']['repetitions'] == 2
        assert m['args']['horizon'] == 1536 and m['args']['prefix'] == 128
        assert not m['args']['fit_only']
        for manifest in [c, m]:
            for record in manifest['checkpoint']['files'].values():
                assert sha(record['path']) == record['sha256'], record['path']
            assert manifest['checkpoint']['actor_real_step'] == 100000
        assert c['checkpoint'] == m['checkpoint']
        source = root/'reward_source_v2/research_script/voc_diagnostics/gru_reward_readout.py'
        assert sha(source) == m['script_sha256']
        rows = json.loads((reward/'pairs.json').read_text())
        assert len(rows) == 128
        states = {}
        seen = set()
        task_deltas=[]
        stage_cost_deltas=[]
        for row in rows:
            key = (row['seed'], row['stream'], row['entry']['count'], row['repetition'])
            assert key not in seen
            seen.add(key)
            assert row['prefix'] == 128
            states.setdefault(key[:3], []).append(row)
            seed, stream, k, rep = key
            arms = [json.loads((reward/f'arm_k{k}_{seed}_{rep}_{arm}.json').read_text())
                    for arm in ['stop', 'continue']]
            a, b = arms
            assert a['entries'] == b['entries']
            assert row['entry'] == a['entries'][stream]
            assert a['targets'] == b['targets'] == [k]*8
            assert a['real_action_rng'] == b['real_action_rng'] == 'seed/stream/real_index/repetition; independent of search count'
            for arm in arms:
                assert np.isfinite(arm['entries'][stream]['features']).all()
                assert np.isfinite(arm['entries'][stream]['gru']).all()
                assert 0 <= arm['entries'][stream]['p_continue'] <= 1
                ret=arm['returns'][stream]
                assert ret['task_truncated'] == ret['task_return']
                assert ret['current_stage_cost'] == ret['stage_cost']
                assert abs(ret['stage_cost']-ret['cost']*ret['stage_computations']) < 1e-12
                assert ret['net_truncated'] == ret['task_return']-ret['stage_cost']
            expected = b['returns'][stream]['net_truncated']-a['returns'][stream]['net_truncated']
            assert row['delta_return'] == expected
            task_deltas.append(b['returns'][stream]['task_truncated']-a['returns'][stream]['task_truncated'])
            stage_cost_deltas.append(b['returns'][stream]['current_stage_cost']-a['returns'][stream]['current_stage_cost'])
            assert abs(expected-(task_deltas[-1]-stage_cost_deltas[-1])) < 1e-12
            assert row['tail_bound'] == a['tail_bounds'][stream]+b['tail_bounds'][stream]
            assert row['tail_bound'] < 4e-5
            assert a['returns'][stream]['current_stage_cost'] == 0
            assert b['returns'][stream]['current_stage_cost'] >= .0005
        assert len(states) == 64
        assert all({r['repetition'] for r in values} == {0, 1} for values in states.values())
        # Repetitions must revisit the identical entry state, not merely the
        # same seed/count label. Repeat-STOP checks are asserted in bound code.
        assert all(values[0]['entry'] == values[1]['entry'] for values in states.values())
        for seed in [22301, 22381]:
            for k in [0, 1, 3, 6]:
                assert json.loads((reward/f'repeat_k{k}_{seed}.json').read_text())['status'] == 'PASS'
        prediction = np.load(reward/'reward_predictions.npz')
        means = np.array([np.mean([r['delta_return'] for r in values]) for values in states.values()])
        assert np.array_equal(prediction['target'], means)
        expected_groups = np.array([key[0]*1000+key[1] for key in states])
        assert np.array_equal(prediction['group'], expected_groups)
        assert len(np.unique(expected_groups)) == 16
        assert all(len(np.unique(prediction['fold'][expected_groups == g])) == 1 for g in np.unique(expected_groups))
        r = json.loads((reward/'reward_readout_report.json').read_text())
        assert r['status'] == 'COMPLETE' and r['states'] == 64 and r['pairs'] == 128
        fit = r['fit_provenance']
        assert fit['pairs_sha256'] == sha(reward/'pairs.json')
        for record in fit['source_files']:
            assert sha(record['path']) == record['sha256'], record['path']
        for name, metrics in r['readouts'].items():
            values = prediction[name]
            assert np.isfinite(values).all()
            rmse = float(np.mean((values-means)**2)**.5)
            assert abs(rmse-metrics['rmse']) < 1e-12
        for base in ['full', 'gru']:
            key = base+'_plus_metadata'
            expected = r['readouts'][base]['rmse']-r['readouts'][key]['rmse']
            assert abs(expected-r['direct_metadata_gain'][key]['rmse_improvement']) < 1e-12
        for mode in ['policy', 'forced']:
            raw = [np.load(decoding/f'{mode}_{seed}.npz') for seed in [19201, 19241]]
            p = np.load(decoding/f'{mode}_decoding.npz')
            y = np.c_[np.concatenate([v['count'] for v in raw]), np.concatenate([v['depth'] for v in raw])]
            assert np.array_equal(y, p['target'])
            assert all(len(np.unique(p['fold'][p['stream'] == g])) == 1 for g in np.unique(p['stream']))
            nonlinear=np.load(root/'nonlinear'/game/f'{mode}_predictions.npz')
            assert np.array_equal(nonlinear['target'], y)
            assert np.array_equal(nonlinear['stream'], p['stream'])
            assert np.array_equal(nonlinear['fold'], p['fold'])
            nreport=json.loads((root/'nonlinear'/game/'report.json').read_text())
            assert nreport['status']=='COMPLETE'
            for record in [nreport['source_file'],nreport['modes'][mode]['outer_fold_source']]:
                assert sha(record['path']) == record['sha256'], record['path']
            for j,name in enumerate(['count','depth']):
                error=np.abs(nonlinear['prediction'][:,j]-y[:,j])
                assert abs(float(error.mean())-nreport['modes'][mode]['metrics'][name]['mae'])<1e-12
        for record in c['records']:
            audit = record['gradient_audit']
            assert audit['actual_voc_loss']['gru']['connected'] == 0
            assert audit['actual_voc_loss']['head']['nonzero'] > 0
            assert audit['same_loss_without_detach']['gru']['nonzero'] > 0
            assert audit['policy_loss']['gru']['nonzero'] > 0
            assert audit['tree_encoder_normalization_modules'] == []
        evidence[game] = dict(status='PASS', states=64, pairs=128, independent_streams=16,
            paired_entries='EXACT', repetitions='EXACT', counts=[0,1,3,6],
            checkpoint_sha256=m['checkpoint']['files']['actor']['sha256'],
            maximum_pair_tail_bound=max(row['tail_bound'] for row in rows),
            gradient_checks=len(c['records']),held_stream_prediction_alignment='PASS',
            fitted_source_and_pair_hashes='PASS',reward_target=dict(
                task_change_tolerance=1e-12,
                task_changed_pairs=int(np.count_nonzero(np.abs(task_deltas)>1e-12)),
                maximum_absolute_task_delta=float(np.max(np.abs(task_deltas))),
                mean_task_delta=float(np.mean(task_deltas)),
                mean_current_stage_cost_delta=float(np.mean(stage_cost_deltas)),
                mean_net_delta=float(np.mean([row['delta_return'] for row in rows]))))
    return dict(status='PASS' if len(evidence)==2 and not missing else 'INCOMPLETE',
                games=evidence, missing=missing)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    a=p.parse_args(); result=validate(a.root)
    (a.root/'completion_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
