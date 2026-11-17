"""Bind normalized query diagnostics, sensitivity and selected-state utility.

This report describes fixed checkpoints; it does not promote within-model
association to useful computation or treat repeated trajectories as new data.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def sign_comparison(rows):
    p=np.clip([r['probability'] for r in rows],0,1)
    d=np.array([r['online_delta'] for r in rows])
    gate,online=p>.5,d>0
    # Exact ties carry no preference and are counted separately.
    valid=(p!=.5)&(d!=0)
    tp=np.sum(valid&gate&online);fn=np.sum(valid&gate&~online)
    tn=np.sum(valid&~gate&~online);fp=np.sum(valid&~gate&online)
    ba=.5*(tp/(tp+fn)+tn/(tn+fp)) if tp+fn and tn+fp else None
    return dict(rows=len(rows),ties=int((~valid).sum()),
        gate_positive_fraction=float(gate.mean()),online_positive_fraction=float(online.mean()),
        non_tie_sign_agreement=float(np.mean(gate[valid]==online[valid])) if valid.any() else None,
        balanced_accuracy_relative_to_gate=ba,
        probability_online_delta_correlation=float(np.corrcoef(p,d)[0,1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    base=args.base.resolve();sources={}
    def bind(path):
        path=Path(path).resolve()
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return path
    def read(path):return json.loads(bind(path).read_text())
    parents=dict(burn8ln='adaptive_planning_encoder_burn8_matched_20261114c',
                 burn16ln='adaptive_planning_encoder_stability_20261114c')
    sensitivity=base/'adaptive_planning_review_20261115_001/query_probability_sensitivity_001'
    plan=read(sensitivity/'plan.json')
    assert plan['conditions']==list(parents)
    regressions=[];counts=[];signs=[];checks=[];histograms=[]
    for condition,name in parents.items():
        root=base/name;observed=root/'query_registered_250k_001'
        done=read(observed/'completion.json');summary=read(observed/f'{condition}_summary.json')
        if done['status']!='COMPLETE' or done['exact_reference_decisions']!=24576:
            raise ValueError('incomplete exact observer validation')
        allocation=read(root/f'evaluation_{condition}/summary.json')
        primary=read(root/'query_predictors_250k_001/audit.json')
        alternative=read(sensitivity/condition/'audit.json')
        if primary['checkpoint']!=alternative['checkpoint'] or primary['folds']!=alternative['folds']:
            raise ValueError('sensitivity changed checkpoint or test folds')
        for scale,audit in [('logodds',primary),('probability',alternative)]:
            if audit['all_decisions']!=24576 or audit['independent_streams']!=16:
                raise ValueError('incomplete regression scope')
            for path,digest in audit['inputs_sha256'].items():
                if hashlib.sha256(bind(path).read_bytes()).hexdigest()!=digest:
                    raise ValueError('regression input changed')
            errors=audit['stream_equal_rmse'];control=errors['count_structure_initial']
            improved=errors['plus_value_updates']
            regressions.append(dict(condition=condition,response_scale=scale,
                decisions=audit['all_decisions'],multi_query_decisions=audit['multi_query_decisions'],
                later_queries=audit['later_optional_queries'],independent_streams=audit['independent_streams'],
                clipped_logodds_rows=audit['clipped_logodds_rows'],
                baseline_rmse=control,with_value_updates_rmse=improved,
                mse_reduction_percent=100*(1-(improved/control)**2),
                mean_mse_improvement=audit['mean_mse_improvement'],
                improvement_ci_low=audit['descriptive_cluster_ci95'][0],
                improvement_ci_high=audit['descriptive_cluster_ci95'][1]))
        counts.extend(dict(condition=condition,**r) for r in summary['by_count'])
        hist=allocation.get('histogram',allocation.get('length_histogram'))
        if sum(hist)!=24576:raise ValueError('incomplete histogram')
        histograms.extend(dict(condition=condition,length=k,fraction=n/24576)
                          for k,n in enumerate(hist) if k)
        with gzip.open(bind(observed/f'{condition}_queries.json.gz'),'rt') as f:queries=json.load(f)
        for label,rows in [('first_optional',[r for r in queries if r['count']==1]),('all_optional',queries)]:
            signs.append(dict(condition=condition,scope=label,**sign_comparison(rows)))
        actor_info=summary['checkpoint']['actor'];model_info=summary['checkpoint']['model']
        for info in (actor_info,model_info):
            if hashlib.sha256(bind(info['path']).read_bytes()).hexdigest()!=info['sha256']:
                raise ValueError('checkpoint bytes changed')
        actor=torch.load(actor_info['path'],map_location='cpu',weights_only=False)
        flags=yaml.safe_load(bind(root/condition/'config.yaml').read_text())
        ema=actor['voc_ema_gate_head_state_dict'];net=actor['actor_net_state_dict']
        scale=flags['voc_gate_temperature']/flags['voc_gate_q_temperature']
        for suffix in ('weight','bias'):
            target=scale*(ema[suffix][0:1]-ema[suffix][1:2])
            if not torch.equal(net['voc_gate_head.'+suffix],target):
                raise ValueError('gate is not exact EMA projection')
        checks.append(dict(condition=condition,checkpoint=actor_info,
            status='BYTE_EXACT_EMA_GATE_PROJECTION_PASS',ema_tau=flags['voc_gate_target_tau'],
            critic_gate_updates=actor['voc_ema_gate_update_count']))
    selected=base/'adaptive_planning_extension_1m_20261114/selected_utility_finalize_001/result'
    selected_done=read(selected/'completion.json');selected_report=read(selected/'summary.json')
    if (selected_done['status']!='ALL_REGISTERED_PAIRS_EXACT_REFERENCE_AND_ACCOUNTING_PASS'
            or selected_done['paired_repetitions']!=64 or selected_done['double_counted']):
        raise ValueError('selected cohort is incomplete or double counted')
    utility_rows=[]
    for group in ('low','high'):
        rows=[r for r in selected_report['states'] if r['cohort']==group]
        utility_rows.append(dict(cohort=group,states=len(rows),
            mean_continue_probability=float(np.mean([r['probability'] for r in rows])),
            mean_net_utility=selected_report[group+'_mean_net_utility'],
            ci_low=selected_report[group+'_mean_net_utility_ci95'][0],
            ci_high=selected_report[group+'_mean_net_utility_ci95'][1],
            states_with_task_difference_exceeding_tail_bound=sum(
                abs(r['task_utility'])>r['maximum_pair_tail_bound'] for r in rows)))
    args.output.mkdir(parents=True,exist_ok=False)
    for filename,rows in [('query_regressions.csv',regressions),('query_count_summary.csv',counts),
                          ('gate_online_signs.csv',signs),('selected_500k_utility.csv',utility_rows)]:
        with (args.output/filename).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    colors={'burn8ln':'#0072B2','burn16ln':'#D55E00'}
    for c,label in [('burn8ln','Burn-in 8'),('burn16ln','Burn-in 16')]:
        rows=[r for r in histograms if r['condition']==c]
        axes[0].plot([r['length'] for r in rows],[100*r['fraction'] for r in rows],'-o',ms=3,color=colors[c],label=label)
        rows=[r for r in counts if r['condition']==c]
        axes[1].plot([r['count'] for r in rows],[r['predicted_continue_mean'] for r in rows],'-',color=colors[c],label=label)
        axes[1].scatter([r['count'] for r in rows],[r['observed_continue_fraction'] for r in rows],s=10,color=colors[c])
    axes[0].set(xlabel='Imagination queries per decision',ylabel='Decisions (%)',
        title='A  Length distribution at 250k',xticks=[1,5,10,15,20,25,30]);axes[0].legend(frameon=False)
    axes[1].set(xlabel='Completed queries (surviving searches)',ylabel='CONTINUE probability',
        title='B  Predicted (line) / observed (points)',ylim=(-.03,1.03),xticks=[1,5,10,15,20,25,29])
    for i,c in enumerate(parents):
        for j,scale in enumerate(('logodds','probability')):
            r=next(r for r in regressions if r['condition']==c and r['response_scale']==scale)
            x=i+(j-.5)*.3;y=r['mse_reduction_percent']
            axes[2].bar(x,y,width=.27,color=colors[c],alpha=1 if j==0 else .5)
            axes[2].text(x,y+.35,f'{y:.1f}%',ha='center',fontsize=9)
    axes[2].set(xticks=[0,1],xticklabels=['Burn-in 8','Burn-in 16'],ylim=(0,24),
        ylabel='Held-out MSE reduction (%)',title='C  Added value-update information')
    axes[2].text(.03,.97,'Solid: log odds\nPale: probability sensitivity',transform=axes[2].transAxes,va='top',fontsize=8)
    fig.suptitle('Enduro sub001 · normalized encoder · one training seed · evaluation epsilon = 0',fontsize=12)
    for ax in axes:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.savefig(args.output/'normalized_query_followup.png',dpi=180)
    fig.savefig(args.output/'normalized_query_followup.pdf');plt.close(fig)
    report=dict(status='COMPLETE_DESCRIPTIVE_FOLLOWUP',created_at=time.time(),sources_sha256=sources,
        gate_projection_checks=checks,selected_utility={k:v for k,v in selected_report.items() if k!='states'},
        findings=['Value-update features predict gate changes beyond the specified count/structure controls in both normalized pilots.',
                  'Probability sensitivity preserves the direction despite logodds clipping in normalized16.',
                  'Gate weights exactly follow EMA; online-critic sign disagreement is not a gate-projection failure.',
                  'Completed selected500k utility has a confidence interval spanning zero; only one state percohort has averaged taskdifference exceeding truncationbound.'],
        limitations=['Each condition reuses its original24576decisions; no new independent trajectories.',
            'Regression is descriptive; value features can still reflect unmodeled mechanics or common inputs.',
            'Count curves condition on surviving searches; pointwise samples differ with querycount.',
            'All-query sign agreement weights long searches more heavily; first-query is reported separately.',
            'Selected utility is a separate500kGRU10checkpoint,not the normalized250kcheckpoint.',
            'Both normalization pilots are Enduro sub001 with one trainingseed and different BCupdate totals.',
            'Burn-in16 efficacy,persistence,andrewardbenefit remain unproven.'])
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    shutil.copy2(__file__,args.output/Path(__file__).name)
    print(json.dumps(dict(output=str(args.output),rows=len(regressions),gate_projection='PASS')))


if __name__=='__main__':main()
