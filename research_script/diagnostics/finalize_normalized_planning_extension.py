"""Validate and summarize every registered normalized8/16 endpoint.

Missing cohorts fail closed. Completion here means evidence is assembled,
not that persistent, useful state-dependent planning has been established.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from summarize_adaptive_planning_evidence import allocation_report, paired_states, utility_report


CONDITIONS={'burn8ln','burn16ln'}


def check_scope(rows, paired=False):
    fields=('seed','stream','count','repetition') if paired else ('seed','stream','decision')
    expected=({(s,i,c,r) for s in (8301,8401) for i in range(8) for c in (1,3,6,12) for r in range(2)}
        if paired else {(s,i,d) for s in (8301,8401) for i in range(8) for d in range(128,1664)})
    actual=[tuple(row[k] for k in fields) for row in rows]
    if len(actual)!=len(expected) or set(actual)!=expected:
        raise ValueError('missing,duplicate,orunregistered evaluation identities')


def check_binding(summary,milestone,expected=None):
    if summary['status']!='COMPLETE':raise ValueError('incomplete evaluation')
    if summary.get('evaluation_epsilon',summary.get('evaluation_train_epsilon'))!=0:
        raise ValueError('evaluation must exclude trainingepsilon')
    binding=summary['checkpoint']
    if set(binding)!={'actor','model'}:raise ValueError('incomplete checkpoint pair')
    for info in binding.values():
        if not milestone<=info['real_step']<milestone+2000:
            raise ValueError('wrong checkpoint milestone')
    if expected is not None:
        for kind in ('actor','model'):
            for key in ('path','sha256','real_step'):
                if binding[kind][key]!=expected[kind][key]:
                    raise ValueError('evaluation differs from audited checkpoint')


def check_registry(manifest,allocations,utilities):
    if {r['condition'] for r in manifest['runs']}!=CONDITIONS or len(manifest['runs'])!=2:
        raise ValueError('both normalized conditions are required')
    if manifest['milestones']!=[500000,1000000] or manifest['utility_endpoint']!=1000000:
        raise ValueError('unregistered endpoint schedule')
    expected={(c,m) for c in CONDITIONS for m in manifest['milestones']}
    keys=[(j['condition'],j['milestone']) for j in allocations['jobs']]
    if set(keys)!=expected or len(keys)!=4:raise ValueError('allocation registry mismatch')
    if set(utilities)!=CONDITIONS:raise ValueError('utility registry mismatch')
    for c,item in utilities.items():
        if item['condition']!=c or item['milestone']!=1000000 or item['pairs']!=128:
            raise ValueError('utility registry scope mismatch')
    return sorted([j['job_id'] for j in allocations['jobs']]+[j['job_id'] for j in utilities.values()])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args();campaign=args.campaign.resolve();sources={};checkpoint_hashes={}
    def bind(path):
        path=Path(path).resolve()
        if str(path) not in sources:
            h=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
            sources[str(path)]=h.hexdigest()
        return path
    def read(path):return json.loads(bind(path).read_text())
    def verify_bytes(binding):
        for info in binding.values():
            path=bind(info['path'])
            if sources[str(path)]!=info['sha256']:raise ValueError('checkpoint bytes changed')
            checkpoint_hashes[str(path)]=sources[str(path)]
    manifest=read(campaign/'manifest.json')
    dependencies=check_registry(manifest,read(campaign/'persistence_evaluation_jobs.json'),read(campaign/'evaluation_jobs.json'))
    parents={};parent_audits={}
    for item in manifest['runs']:
        c=item['condition'];parents[c]=Path(item['parent_config']).parents[1]
        audit=read(item['parent_completion_audit']);parent_audits[c]=audit
        if audit['status']!='FRESH_BUDGET_LOGS_OPTIMIZERS_EVALUATION_AND_WANDB_PASS':
            raise ValueError('parenttrainingaudit did notpass')
        for kind in ('actor','model'):
            for key in ('path','sha256','real_step'):
                if audit['checkpoint'][kind][key]!=item['parent_checkpoint'][kind][key]:
                    raise ValueError('parentbinding mismatch')
        verify_bytes(audit['checkpoint'])
    if args.preflight:
        print(json.dumps(dict(status='FINAL_REPORT_REGISTRY_AND_PARENT_BINDINGS_PASS',dependencies=dependencies,
            expected_allocation_cohorts=6,expected_utility_cohorts=4,sources_sha256=sources)))
        return
    if args.output is None:raise ValueError('--output is required')
    if args.output.exists():raise ValueError('preserveexistingreport')
    training=[];allocations=[];histograms=[];utilities=[];states_all=[];details=[]
    for item in sorted(manifest['runs'],key=lambda r:r['condition']):
        c=item['condition'];parent=parent_audits[c]
        audit=read(campaign/c/'completion.json')
        if (audit['status']!='CONTINUATION_CHECKPOINT_OPTIMIZER_COUNTERS_WANDB_PASS'
                or audit['phase']!='full' or audit['condition']!=c
                or audit['health']['bc_history_status']!='NO_LARGE_NLL_OBSERVED'):
            raise ValueError('fulltrainingaudit did notpass')
        if audit['contract']!=parent['contract'] or audit['wandb']['state']!='finished':
            raise ValueError('training contract orW&B mismatch')
        verify_bytes(audit['checkpoint'])
        logs=bind(Path(item['run_directory'])/'logs.csv')
        if sources[str(logs)]!=audit['logs_sha256']:raise ValueError('auditedtraininglogs changed')
        collector=read(Path(item['run_directory'])/'collector_metrics.json')
        if collector['physical_env_steps']!=audit['physical_collected_transitions']:
            raise ValueError('collector counter mismatch')
        for kind in ('actor','model'):
            if not 1000000<=audit['checkpoint'][kind]['real_step']<1002000:
                raise ValueError('full training endpoint mismatch')
        training.append(dict(condition=c,parent_actor_step=parent['checkpoint']['actor']['real_step'],
            final_actor_step=audit['checkpoint']['actor']['real_step'],final_model_step=audit['checkpoint']['model']['real_step'],
            invocation_learner_transitions=audit['actor_added_transitions'],
            invocation_physical_actions=audit['physical_collected_transitions'],
            invocation_bc_targets=audit['bc_targets'],parent_bc_targets=parent['bc_targets'],
            total_bc_targets=audit['bc_targets']+parent['bc_targets'],
            actor_updates=audit['counters']['imitation_update_count']['logged_delta'],
            max_bc_nll=audit['health']['maximum_bc_nll'],wandb_url=audit['wandb']['url']))
        allocation_rows={}
        for milestone in (250000,500000,1000000):
            folder=(parents[c]/f'evaluation_{c}' if milestone==250000 else campaign/f'allocation_{milestone}_{c}')
            prefix='' if milestone==250000 else c+'_'
            summary=read(folder/(prefix+'summary.json'))
            expected=(parent['checkpoint'] if milestone==250000 else audit['checkpoint'] if milestone==1000000 else None)
            check_binding(summary,milestone,expected);verify_bytes(summary['checkpoint'])
            rows=read(folder/(prefix+'allocation.json'));check_scope(rows);allocation_rows[milestone]=rows
            if not all(np.isfinite(r[k]) for r in rows for k in ('root_value','root_margin','first_optional_delta')):
                raise ValueError('nonfinite allocation valuefeature')
            values=allocation_report(rows);lengths=np.array([r['length'] for r in rows])
            if np.any(lengths!=np.floor(lengths)):raise ValueError('nonintegerimaginationlength')
            histogram=np.bincount(lengths.astype(int),minlength=31).tolist()
            if histogram!=summary.get('histogram',summary.get('length_histogram')):
                raise ValueError('savedhistogram mismatch')
            for key in ('mean_length','min_fraction','interior_fraction','cap_fraction'):
                if not np.isclose(values[key],summary[key],rtol=0,atol=1e-12):raise ValueError('savedallocationstatistic mismatch')
            metadata=dict(condition=c,milestone=milestone,actor_step=summary['checkpoint']['actor']['real_step'])
            allocations.append(dict(**metadata,**{k:values[k] for k in ('decisions','independent_streams','mean_length',
                'min_fraction','interior_fraction','cap_fraction','first_probability_sd',
                'first_probability_after_stream_and_decision_adjustment_sd','optional_sign_change_fraction')}))
            histograms.extend(dict(**metadata,length=k,count=n,fraction=n/24576) for k,n in enumerate(histogram) if k)
            details.append(dict(kind='allocation',**metadata,report=values,checkpoint=summary['checkpoint']))
        for milestone in (250000,1000000):
            folder=(parents[c] if milestone==250000 else campaign)/f'evaluation_{c}'
            summary=read(folder/'summary.json');expected=parent['checkpoint'] if milestone==250000 else audit['checkpoint']
            check_binding(summary,milestone,expected)
            repeated_allocation=read(folder/'allocation.json');check_scope(repeated_allocation)
            if repeated_allocation!=allocation_rows[milestone]:
                raise ValueError('utility and standalone allocation differ on samecheckpoint')
            pairs=read(folder/'pairs.json');check_scope(pairs,paired=True);arms={}
            for row in pairs:
                key=row['seed'],row['count'],row['repetition']
                if key not in arms:arms[key]=read(folder/('arms_%s_%s_%s.json'%key))
            states=paired_states(pairs,arms);report=utility_report(states)
            if report['intervention_real_decision_indices']!=[128] or report['paired_repetitions']!=128:
                raise ValueError('utilityscope mismatch')
            metadata=dict(condition=c,milestone=milestone,actor_step=summary['checkpoint']['actor']['real_step'])
            row=dict(**metadata,states=report['states'],streams=report['independent_streams'])
            for key in ('mean_net_utility','choice_matched_advantage','ema_rmse_improvement_over_zero'):
                row[key]=report[key];row[key+'_ci_low'],row[key+'_ci_high']=report[key+'_ci95']
            utilities.append(row);states_all.extend(dict(**metadata,**r) for r in states)
            details.append(dict(kind='utility',**metadata,report=report,checkpoint=summary['checkpoint']))
    args.output.mkdir(parents=True)
    for name,rows in [('training_summary.csv',training),('allocation_summary.csv',allocations),
                      ('histograms.csv',histograms),('utility_summary.csv',utilities),('paired_states.csv',states_all)]:
        with (args.output/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (args.output/'details.json').write_text(json.dumps(details,indent=2,allow_nan=False)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={250000:'#999999',500000:'#0072B2',1000000:'#D55E00'}
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,c,label in zip(axes,('burn8ln','burn16ln'),('Burn-in 8','Burn-in 16')):
        for milestone in (250000,500000,1000000):
            rows=[r for r in histograms if r['condition']==c and r['milestone']==milestone]
            ax.plot([r['length'] for r in rows],[100*r['fraction'] for r in rows],'-o',ms=3,
                color=colors[milestone],label=f'{milestone/1e6:g}M')
        ax.set(title=label,xlabel='Imagination queries per decision',ylabel='Decisions (%)',xticks=[1,5,10,15,20,25,30],ylim=(-1,101))
        ax.legend(frameon=False);ax.grid(alpha=.15)
    fig.suptitle('Enduro sub001: registered checkpoints, evaluation epsilon = 0')
    fig.savefig(args.output/'length_distributions.png',dpi=180);fig.savefig(args.output/'length_distributions.pdf');plt.close(fig)
    manifest_out=dict(status='EVIDENCE_ASSEMBLED_NOT_GOAL_COMPLETION',completed_at=time.time(),
        goal_complete=False,allocation_cohorts=6,utility_cohorts=4,sources_sha256=sources,
        checkpoint_hashes=checkpoint_hashes,registered_dependencies=dependencies,
        limitations=['One trainingseed,Enduro sub001 allsessions;no acrossseed oracrossgame claim.',
            '250kto1M coldrestart restoresweights,optimizers,counters,BC RNG;collector,recurrentrolloutstate andmodelreplay restart.',
            'ParentsusedRTX3090;both continuationsusedTITAN RTX. Hardwareandrestart confound uninterruptedtraining interpretation.',
            'Differentplanninglengths yielddifferentBCupdate/targetcounts perenvironmentaction;reportedratherthanhidden.',
            'Allocationseeds/indices match,butpolicies visitdifferentobservations;cohortsarenotindependentreplications.',
            '250kutilityuseslegacy mandatorydraw behavior;1M usesrepaired mandatorydraw preservation. Eachpairedstate isinternallymatched;utilitychange isnotacleantraining-onlycontrast.',
            'Choicefrequency-matchedutility doesnotmatchtotalcomputations orprovewholeepisode policyimprovement.',
            'A broadlengthhistogram doesnotbyitself establishusefulstate-dependentplanning. Interpretallregisteredendpoints.'])
    (args.output/'manifest.json').write_text(json.dumps(manifest_out,indent=2,allow_nan=False)+'\n')
    analysis_sources=args.output/'analysis_sources';analysis_sources.mkdir()
    for name in (Path(__file__).name,'summarize_adaptive_planning_evidence.py'):
        shutil.copy2(Path(__file__).parent/name,analysis_sources/name)
    print(json.dumps(dict(status=manifest_out['status'],output=str(args.output),goal_complete=False)))


if __name__=='__main__':main()
