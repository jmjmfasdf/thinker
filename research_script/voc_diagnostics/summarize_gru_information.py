"""Create a report without treating missing experiments as successful."""
import argparse, csv, json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    a = p.parse_args(); rows=[]; gradients=[]; rewards=[]
    complete=True
    for game in ['enduro','pong']:
        directory=a.root/'v2'/game
        path=directory/'decoding_report.json'
        if not path.exists(): complete=False; continue
        report=json.loads(path.read_text())
        if report['status'] != 'COMPLETE': complete=False; continue
        for meta in sorted(directory.glob('policy_*.json')):
            record=json.loads(meta.read_text()); audit=record['gradient_audit']
            gradients.append(dict(game=game, seed=record['seed'],
                voc_gru_connected=audit['actual_voc_loss']['gru']['connected'],
                voc_head_l2=audit['actual_voc_loss']['head']['l2'],
                undetached_gru_l2=audit['same_loss_without_detach']['gru']['l2'],
                policy_gru_l2=audit['policy_loss']['gru']['l2'],
                normalization_modules=len(audit['tree_encoder_normalization_modules'])))
        for mode,data in report['modes'].items():
            for readout,res in data['readouts'].items():
                for target,score in res['scores'].items():
                    rows.append(dict(game=game,mode=mode,readout=readout,target=target,
                        rows=data['rows'],streams=data['streams'],mae=score['mae'],
                        rounded_accuracy=score['rounded_accuracy']))
        rpath=a.root/'reward_v2'/game/'reward_readout_report.json'
        if not rpath.exists(): complete=False; continue
        reward=json.loads(rpath.read_text())
        if reward['status']!='COMPLETE':complete=False;continue
        for name,comparison in reward['direct_metadata_gain'].items():
            rewards.append(dict(game=game,readout=name,states=reward['states'],pairs=reward['pairs'],
                baseline_rmse=reward['readouts'][name.removesuffix('_plus_metadata')]['rmse'],
                metadata_rmse=reward['readouts'][name]['rmse'],
                improvement=comparison['rmse_improvement'],lo=comparison['ci95'][0],hi=comparison['ci95'][1]))
    for name,data in [('counter_decoding',rows),('gradient_audit',gradients),('reward_metadata_gain',rewards)]:
        if data:
            with (a.root/f'{name}.csv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    lines=['# GRU / VOC information audit', '', 'Status: '+('COMPLETE' if complete else 'PARTIAL — reward or decoding results remain pending.'), '',
           'Frozen final 100k projection checkpoints; Enduro and Pong. All decoder validation holds out whole environment streams.', '',
           '| Game | Sampling | Representation | Count exact accuracy | Depth exact accuracy | Count MAE | Depth MAE |',
           '|---|---|---|---:|---:|---:|---:|']
    for game in ['enduro','pong']:
        for mode in ['policy','forced']:
            for rep in ['gru','full_voc','explicit','constant']:
                selected={r['target']:r for r in rows if r['game']==game and r['mode']==mode and r['readout']==rep}
                if len(selected)==2:
                    c,d=selected['count'],selected['depth']
                    lines.append(f"| {game} | {mode} | {rep} | {c['rounded_accuracy']:.1%} | {d['rounded_accuracy']:.1%} | {c['mae']:.3f} | {d['mae']:.3f} |")
    lines+=['','Gradient evidence uses the production VOC regression loss on real inputs with a deliberately nonzero diagnostic target (baseline + 0.5). These norms test connectivity; they are not reported training-gradient statistics. The undetached and policy-loss controls verify that the GRU can receive gradients.','',
        '| Game | VOC → GRU connected tensors | VOC head gradient norm | Without detach → GRU norm | Policy → GRU norm | Normalization modules |',
        '|---|---:|---:|---:|---:|---:|']
    for r in gradients:
        lines.append(f"| {r['game']} seed {r['seed']} | {r['voc_gru_connected']} | {r['voc_head_l2']:.4f} | {r['undetached_gru_l2']:.4f} | {r['policy_gru_l2']:.4f} | {r['normalization_modules']} |")
    lines+=['','Reward prediction: positive improvement means lower held-out RMSE after adding explicit count/depth.','',
            '| Game | Readout | States / pairs | Original RMSE | With metadata RMSE | RMSE improvement | 95% stream interval |',
            '|---|---|---:|---:|---:|---:|---|']
    for r in rewards:
        lines.append(f"| {r['game']} | {r['readout']} | {r['states']} / {r['pairs']} | {r['baseline_rmse']:.6g} | {r['metadata_rmse']:.6g} | {r['improvement']:.6g} | [{r['lo']:.6g}, {r['hi']:.6g}] |")
    lines+=['','Exploratory nonlinear counter readout (64 hidden units; independently selected epochs on inner training streams). This tests recoverability, not reward utility.','',
            '| Game | Sampling | Count exact accuracy | Depth exact accuracy | Count MAE | Depth MAE |',
            '|---|---|---:|---:|---:|---:|']
    for game in ['enduro','pong']:
        nonlinear=a.root/'nonlinear'/game/'report.json'
        if nonlinear.exists():
            n=json.loads(nonlinear.read_text())
            for mode,record in n['modes'].items():
                c,d=record['metrics']['count'],record['metrics']['depth']
                lines.append(f"| {game} | {mode} | {c['rounded_accuracy']:.1%} | {d['rounded_accuracy']:.1%} | {c['mae']:.3f} | {d['mae']:.3f} |")
        else:
            complete=False
    lines+=['',f'![Counter decoding error]({a.root.resolve() / "counter_decoding_error.png"})',
            'The plot uses equal stream weights within each count and shows ±1 SEM across available streams. Sparse policy tail counts have fewer supporting streams.']
    sensitivity_path=a.root/'counter_sensitivity.json'
    if sensitivity_path.exists():
        sensitivity=json.loads(sensitivity_path.read_text())
        lines+=['','Exploratory forward sensitivity: perturb one clock channel while holding other inputs and the prior GRU state fixed. This measures dependence of the existing online VOC output, not useful or physically valid additional planning.','',
                '| Game | Sampling | Median absolute ΔVOC for count +1 | Median absolute ΔVOC for depth +1 |',
                '|---|---|---:|---:|']
        for game,modes in sensitivity['games'].items():
            for mode,record in modes.items():
                s=record['sensitivity']
                lines.append(f"| {game} | {mode} | {s['count']['median_abs_one_step_q_change']:.6f} | {s['depth']['median_abs_one_step_q_change']:.6f} |")
        lines+=['','The perturbed token can be inconsistent with other tree metadata. Nonzero sensitivity cannot establish value calibration or causal reward improvement.']
    lines+=['','Limits: decodability is distinct from use by the existing head. Reward probes use paired actual environment returns at forced search counts, with the current checkpoint as continuation policy. They do not establish whole-policy reward gains or the causal effect of end-to-end retraining. Intervals condition on fitted out-of-fold predictions.']
    validation_path=a.root/'completion_validation.json'
    validation=json.loads(validation_path.read_text()) if validation_path.exists() else {}
    if complete and validation.get('status') != 'PASS':
        lines[2]='Status: AWAITING VALIDATION — all result reports exist; the raw-artifact audit has not passed.'
        status='AWAITING_VALIDATION'
    else:
        status='COMPLETE' if complete else 'PARTIAL'
        lines[2]='Status: '+('COMPLETE — raw-artifact audit PASS.' if complete else 'PARTIAL — reward or decoding results remain pending.')
    interpretation=a.root/'interpretation.md'
    if interpretation.exists():
        lines+=['',interpretation.read_text().strip()]
    (a.root/'report.md').write_text('\n'.join(lines)+'\n')
    (a.root/'report_status.json').write_text(json.dumps(dict(status=status,
        decoder_rows=len(rows),gradient_checks=len(gradients),reward_comparisons=len(rewards)),indent=2))
    print(a.root/'report.md')


if __name__=='__main__':main()
