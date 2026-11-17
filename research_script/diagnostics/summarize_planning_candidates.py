"""Aggregate completed isolated diagnostics, verify provenance, and plot results."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime,timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research_script/outputs/00/planning_candidate_analysis_20261114'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    probes=sorted(OUT.glob('gradient*/manifest.json'))
    assert len(probes)==5, f'Need 5 complete probes, got {len(probes)}'
    all_adam=[];all_comp=[];summary=[];checks={};labels={}
    for f in probes:
        m=json.loads(f.read_text());name=f.parent.name
        step=round(m['actor_step']/1e6)
        label=f"{step}M / seed {m['seed']%100}"+(" / later" if m['warmup_calls']>256 else '')
        labels[name]=label
        checks[name+'_actor_unchanged']=sha(m['actor_path'])==m['actor_sha256']
        checks[name+'_model_unchanged']=sha(m['model_path'])==m['model_sha256']
        d=pd.read_csv(f.parent/'candidate_adam.csv');d['probe']=name;d['label']=label;d['checkpoint_m']=step;all_adam.append(d)
        c=pd.read_csv(f.parent/'gradient_components.csv');c['probe']=name;c['label']=label;all_comp.append(c)
        def value(candidate,col,mult=1):return float(d[(d.candidate==candidate)&(d.voc_multiplier==mult)][col].iloc[0])
        summary.append(dict(probe=name,label=label,checkpoint_m=step,seed=m['seed'],warmup=m['warmup_calls'],
            wait_fraction=m['wait_fraction'],real_steps_per_stream=m['real_actions_per_stream'][0],
            raw_voc_to_other_gru_norm=m['gru_voc_to_other_ratio'],
            voc_marginal_update_pct=100*value('original','relative_voc_effect'),
            voc_x10_marginal_update_pct=100*value('original','relative_voc_effect',10),
            voc_x100_marginal_update_pct=100*value('original','relative_voc_effect',100),
            wait_point1_update_change_pct=100*value('wait_0.1','relative_change_vs_production'),
            wait_point1_voc_marginal_pct=100*value('wait_0.1','relative_voc_effect'),
            intrinsic_zero_update_change_pct=100*value('im_both_0','relative_change_vs_production'),
            largest_component=c.sort_values('gru_norm',ascending=False).iloc[0].component,
            seconds=m['seconds']))
        if name.startswith('gradient_6000880_'):
            seed=m['seed']%100
            old=ROOT/f'research_script/outputs/00/planning_learning_deep_review_20261113/probe_6m_seed{seed}'
            prior=pd.read_csv(old/'matched_adam.csv');new=pd.read_csv(f.parent/'matched_adam.csv')
            checks[name+'_reproduces_previous']=bool(np.allclose(prior.select_dtypes('number'),new.select_dtypes('number'),rtol=2e-5,atol=1e-8,equal_nan=True))
    ad=pd.concat(all_adam,ignore_index=True);ad.to_csv(OUT/'candidate_updates.csv',index=False)
    co=pd.concat(all_comp,ignore_index=True);co.to_csv(OUT/'gradient_components.csv',index=False)
    su=pd.DataFrame(summary);su.to_csv(OUT/'gradient_summary.csv',index=False)
    memory=[];memory_meta={}
    for f in sorted(OUT.glob('memory*/memory_context_rows.csv')):
        d=pd.read_csv(f);d['probe']=f.parent.name;memory.append(d)
        memory_meta[f.parent.name]=json.loads((f.parent/'manifest.json').read_text())
        counts=d.groupby(['kind','burnin']).size()
        checks[f.parent.name+'_same_targets_per_burnin']=bool((counts==96).all())
        checks[f.parent.name+'_unique_comparisons']=not d.duplicated(['kind','target_start','burnin','target_in_chunk','stream']).any()
    md=pd.concat(memory,ignore_index=True);md.to_csv(OUT/'memory_comparisons.csv',index=False)
    ms=md.groupby(['probe','kind','burnin']).agg(n=('stream','size'),
        mean_abs_p=('probability_difference',lambda x:x.abs().mean()),
        mean_abs_delta=('delta_difference',lambda x:x.abs().mean()),
        median_relative_hidden_difference=('h_relative_difference','median'),
        sign_flip_fraction=('sign_flip','mean'),mean_real_kl=('real_policy_kl','mean'),
        mean_human_ll_difference=('human_ll_difference','mean')).reset_index()
    ms.to_csv(OUT/'memory_summary.csv',index=False)
    bm6=memory_meta['memory_6m']['batch_metadata'];bm10=memory_meta['memory_10m']['batch_metadata']
    checks['6m_10m_identical_human_targets']=all(bm6[k]==bm10[k] for k in ['source_file','window_start','actions_seq','observation_source_index'])
    before=json.loads((OUT/'source_before.json').read_text())
    checks['production_and_previous_diagnostic_sources_unchanged']=all(sha(k)==v for k,v in before.items())
    support=OUT/'support_v2';sm=json.loads((support/'manifest.json').read_text())
    pop=pd.read_csv(support/'critic_population_controls.csv')
    checks['population_stationary_points']=bool((pop.derivative.abs()<1e-6).all())
    checks['population_common_mode_exact_fit']=bool((pop[pop['mode']=='common_mode'].balanced_q_rmse<1e-6).all())
    assert all(checks.values()),checks
    # All displays are descriptive: batches and within-window decisions are dependent.
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,2,figsize=(12,4.6))
    for name,g in ad[ad.candidate=='original'].groupby('probe'):
        g=g[g.voc_multiplier>0]
        ax[0].plot(g.voc_multiplier,100*g.relative_voc_effect,'o-',label=labels[name])
    ax[0].set(xscale='log',yscale='log',xlabel='VOC-to-GRU gradient multiplier',ylabel='Marginal GRU Adam update change (%)',title='One-update sensitivity at fixed optimizer state')
    ax[0].legend(fontsize=8)
    xx=np.arange(len(su));ax[1].bar(xx-.18,su.wait_point1_update_change_pct,.36,label='WAIT value weight ×0.1')
    ax[1].bar(xx+.18,su.intrinsic_zero_update_change_pct,.36,label='Intrinsic losses ×0')
    ax[1].set_xticks(xx,su.label,rotation=35,ha='right')
    ax[1].set(ylabel='GRU update difference from production (%)',title='Weight changes alter different batches differently');ax[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'gradient_candidates.png',dpi=180);fig.savefig(OUT/'gradient_candidates.pdf');plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for name,g in ms[ms.kind=='first_optional'].groupby('probe'):
        ax[0].plot(g.burnin,100*g.median_relative_hidden_difference,'o-',label=name)
        ax[1].plot(g.burnin,100*g.mean_abs_p,'o-',label=name)
    for z in ax:z.set_xlabel('Burn-in real actions');z.set_xticks([0,1,4,8,16])
    ax[0].set(ylabel='Median hidden-state difference (%)',title='Matched targets; longer context reduces state mismatch')
    ax[1].set(ylabel='Mean absolute P(continue) difference (pp)',title='Gate sensitivity depends on checkpoint and sample')
    ax[1].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'memory_burnin.png',dpi=180);fig.savefig(OUT/'memory_burnin.pdf');plt.close(fig)
    gate=pd.read_csv(support/'gate_count_support.csv');late=gate[gate.interval=='8–10M']
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    xx=np.arange(len(late));ax[0].plot(xx,late.mean_p_continue,'o-',label='Gate probability')
    ax[0].plot(xx,late.actual_continue_fraction,'s--',label='Observed execution')
    ax[0].set(ylabel='Continue probability',ylim=(0,1),title='8–10M: conditional on reaching each search count');ax[0].legend()
    ax[1].bar(xx-.18,late.observed_opposite,.36,label='Observed opposite choices')
    ax[1].bar(xx+.18,late.expected_opposite,.36,label='Expected from gate')
    ax[1].set(ylabel='Opposite-to-preference choices',title='Few samples of the less preferred alternative');ax[1].legend(fontsize=8)
    for z in ax:z.set_xticks(xx,late.count_group);z.set_xlabel('Search count before optional decision')
    fig.tight_layout();fig.savefig(OUT/'gate_support.png',dpi=180);fig.savefig(OUT/'gate_support.pdf');plt.close(fig)
    # Source snapshots distinguish v1 four short probes and v2 extended probe.
    sources=OUT/'analysis_sources'
    for name in ['probe_planning_candidates','probe_bc_burnin_sweep','analyze_planning_support','summarize_planning_candidates','summarize_candidate_training_log','analyze_planning_stage_dynamics']:
        shutil.copy2(ROOT/f'research_script/diagnostics/{name}.py',sources/f'{name}.py')
    (OUT/'qa.json').write_text(json.dumps(dict(status='PASS',checks=checks),indent=2))
    late_stage=pd.read_csv(support/'stage_intervals.csv').iloc[-1]
    alternatives=pd.read_csv(support/'gate_same_rows_alternatives.csv')
    late_alternatives=alternatives[(alternatives.interval=='8–10M')&(alternatives.epsilon==.02)]
    historical=pd.read_csv(OUT/'historical_gradient_summary.csv')
    hmeta=json.loads((OUT/'historical_gradient_manifest.json').read_text())
    within=pd.read_csv(OUT/'gate_within_stage_summary.csv')
    within_late=within[within.interval=='8–10M'].iloc[0]
    checks['historical_snapshot_matches_support']=hmeta['source_sha256']==sm['training_sha256']
    assert checks['historical_snapshot_matches_support']
    (OUT/'qa.json').write_text(json.dumps(dict(status='PASS',checks=checks),indent=2))
    report=f'''학습 변경 후보의 독립 분석 — 2026-11-14

범위와 보존 사항
Enduro sub001, 전체 session, 최소 imagination 1회, cost=0.0001, temperature=0.01,
episode 단위 GRU carry를 쓰는 현재 계보를 분석했다. 대상은 기존 검토의 1,2,3,4,6,7번이며
5번(real head의 직접 관측 경로)은 제외했다. 기존 모델/학습 코드, checkpoint, 실행 job은 수정하지 않았다.
새 진단 파일과 이 결과 폴더만 생성했다. 코드와 checkpoint 해시 및 비교 검증은 qa.json에 기록했다.

1. VOC critic: 구조적 오차 혼입은 재현되지만, 실제 checkpoint의 편향량은 아직 알 수 없다.
현재 Q_C=V+(1-p)D, Q_S=V-pD에서는 V를 고정하므로 공통 가치 오차를 별도로 설명할 자유도가 없다.
같은 상태의 실제 contrast를 알고 있는 대조 실험에서 behavior/current 확률 차이까지 있으면 D가
그 공통 오차를 일부 흡수했다. 예를 들어 p=.01, b=.02, 공통 오차=.1, true D=0일 때
기존 selected-Q 회귀의 최적 D는 +.050761이 된다. production의 Q 재구성과 Huber loss로 검증했다.
공통 오차를 별도로 표현하는 항은 이 예에서 true D와 두 Q를 모두 복원했다. 중요도 가중은 모집단
수준에서 D의 편향을 제거하지만 공통 가치 오차 자체를 표현하지는 못한다.
그러나 반대 행동 표본이 없으면 두 값을 독립적으로 식별할 수 없다. synthetic p=b=.99, N=32에서는
두 행동이 모두 관측된 반복이 약 26%뿐이었다. 공통 항 추가만으로 표본 부족이 해결되지는 않는다.
이 대조 실험은 현재 Enduro의 실제 VOC가 +.05만큼 틀렸다는 의미가 아니다. 실제로는 짝지은
continue/stop 이후의 return과 같은 상태의 두 목표값이 필요하다.
이번 fresh checkpoint probe들은 behavior/current policy가 동일하므로 실제 비동기 학습의 policy lag를
직접 재현하지 않았다. 공통 항과 중요도 가중의 대조는 알려진 true Q를 둔 synthetic 분석이다.

2. WAIT와 전체 가치 loss: WAIT만의 문제로 제한하면 안 된다.
아래는 각 checkpoint로 새로 수집한 짧은 환경 구간이다. B=16, unroll=201, BC=64 targets이며
원래 V-trace target을 고정한 채 WAIT loss만 ×0.1로 했을 때의 가상 Adam 업데이트를 비교했다.
실제 checkpoint의 optimizer moments를 복제했고 전체 actor gradient로 clipping을 계산했다.
6M에서는 WAIT가 가장 큰 GRU gradient였지만, 10M의 초기 환경 표본에서는 SEARCH/REAL 가치 회귀가
더 컸다. historical 8–10M 로그의 WAIT 중앙값은 약 60.9%인데 새 초기 표본은 이보다 훨씬 작았다.
따라서 checkpoint 간 probe 차이를 학습 경과의 평균 효과로 해석하면 안 된다.

{su[['label','wait_fraction','real_steps_per_stream','voc_marginal_update_pct','wait_point1_update_change_pct','intrinsic_zero_update_change_pct','largest_component']].to_string(index=False)}

표의 update_change는 좋은 변화의 비율이 아니라 기존 가중치 업데이트와의 벡터 거리다.
WAIT=0은 민감도 대조 조건이며 production에서 WAIT를 삭제하거나 목표값 전달을 바꾸지 않았다.
추가 10M probe는 warm-up 2048회 뒤 수집했고 전체 task-value 비중을 줄이는 대조 조건도 포함했다.
해당 수치는 candidate_updates.csv의 all_task_value_0.1 / all_task_value_0.01에 있다.
이 추가 표본에서도 WAIT는 약 3.3%로 역사적 학습 분포와 달랐다. WAIT 비중을 ×0.1로 줄였을 때
GRU update 차이는 1.70%였지만 전체 task-value를 ×0.1로 줄이면 73.81%였다. 후자의 VOC marginal
비중은 0.0201→0.0798%로 커졌으나, VOC loss를 줄이는 방향과의 정렬이 함께 좋아진 것은 아니다.
전체 task-value ×0.01에서는 이 batch의 국소 VOC 방향 미분이 오히려 증가 방향이었다. 과거 Adam
moments와 다른 목적함수가 남아 있으므로 비중 축소를 성능 개선으로 읽으면 안 된다.

3. VOC→GRU: 연결은 있으나 현재 한 번의 업데이트에 미치는 추가 영향이 작다.
실제 학습 로그의 29,967 updates를 새로 집계했다. VOC/기타 actor GRU gradient norm의 중앙값은
0–2M 0.382%, 2–4M 0.113%, 4–6M 0.062%, 6–8M 0.047%, 8–10M 0.046%였다.
VOC 자체 norm 중앙값은 1.47→2.42로 늘었으나 기타 actor norm은 377→4,955로 더 크게 늘었다.
신호가 끊긴 것과 상대적으로 작아진 것은 구분해야 한다. gradient cosine의 구간별 중앙값은
거의 0이며 양/음 방향이 혼재했다. 이 비율은 아래의 Adam marginal update 비율과 다른 지표다.

{historical.to_string(index=False)}

voc_marginal_update_pct는 동일한 과거 optimizer moments 아래 이번 VOC gradient를 넣었을 때와
뺐을 때의 GRU Adam update 차이이다. 단순 gradient norm의 비율과 다르다.
VOC 신호를 ×10, ×100으로 가상 확대하면 차이는 커진다. 이것은 목표값의 정확도, 장기 학습 개선,
계산 효용 개선을 입증하지 않는다. 두 조건 모두 과거 VOC가 반영된 optimizer moments를 유지했다.
원래 목표와 오차 구조를 먼저 검증해야 하며, 큰 배율을 바로 적용할 근거로 쓰면 안 된다.

4. gate 선택 표본: 실행은 확률을 따르지만 긴 search에서 반대 선택은 드물다.
전체 학습 로그에서 추출한 optional query는 {sm['optional_rows']}개이며, 8–10M 구간은 {int(late.queries.sum())}개다.
마지막 구간의 완전한 decision은 {int(late_stage.decisions)}개이고 평균 길이는 {late_stage.mean_length:.3f}이다.
한 decision 안에서 계속/중단 선호의 부호가 바뀐 비율은 {100*late_stage.sign_change_fraction:.2f}%였다.
길이 8 이상 {int(late_stage.long_decisions)}개 중 부호 변화 비율은 {100*late_stage.long_sign_change_fraction:.2f}%였고,
cap 이전에 STOP한 {int(late_stage.long_stops)}개 중 {int(late_stage.long_stop_despite_p095)}개는 여전히 P(continue)>.95였다.
이는 확률적 실행에서 나올 수 있는 결과이며 실행 버그나 비최적 budget의 직접 증거는 아니다.
또한 부호가 유지되는 것과 확률이 변하지 않는 것은 다르다. 8–10M에서 실제 단계별 확률은
첫 optional 확률을 고정한 기준보다 실행 log score가 decision당 평균 {within_late.mean_logscore_gain:.3f} nat 높았다.
계산 중 첫 확률에서 5pp 넘게 달라진 decision도 {100*within_late.probability_shift_over_005_fraction:.2f}% 있었다.
이는 실행 분포의 변화에 대한 설명이며 인간 likelihood나 환경 보상 개선 수치는 아니다.

{late.to_string(index=False)}

16–29회 구간은 이미 긴 search에 도달한 표본만 포함한다. 높은 continue 확률을 '더 생각할수록
유리해진다'는 인과 효과로 읽으면 안 된다. 해당 구간의 반대 선택 표본과 p(1-p)가 작다는 점은
두 선택의 가치 차이를 학습하는 데 불리할 수 있다. 같은 상태를 다시 만나고 두 결과를 관측한 것은 아니다.
저장된 EMA delta를 고정하고 tau/epsilon만 바꾼 결과는 다음과 같다.

{late_alternatives.to_string(index=False)}

tau=.025나 epsilon 증가가 덜 선호하는 선택의 예상 표본 수를 늘리는 것은 확인됐다.
새 정책으로 바뀌는 상태 분포, 실제 보상과 효용은 여기서 평가하지 않았다.

6. BC의 짧은 기억: burn-in을 늘리면 상태 차이는 줄지만 gate의 영향은 표본별로 다르다.
8개 human sequence에서 각각 3개 target block, block당 4 actions를 고정했다.
각 probe의 96개 decision은 서로 독립적인 96명이 아니라 같은 sequence 안의 반복 관측이다.
6M과 10M seed14는 정확히 같은 human observations/actions를 사용했다.
reference는 41-edge window 처음부터 carry한 상태이며 full-episode 정답 hidden은 아니다.
다른 tree/query/image feature는 그대로 두고 GRU의 시작 위치만 바꿨다. replanning을 다시 한 것은 아니다.

{ms[ms.kind=='first_optional'].to_string(index=False)}

6M에서는 1→16 burn-in으로 평균 gate 확률 차이가 약 9.09→1.20 percentage points로 줄었다.
10M seed14는 VOC delta가 달라도 확률 포화 때문에 gate 확률은 거의 같았다. seed15에서는 약
1.69→1.10pp였고 중간 길이에서 단조롭지 않았다. 따라서 기억 길이 문제는 실제 존재하지만
항상 큰 STOP/CONTINUE 오류를 만든다고 일반화할 수 없다.
real-head policy KL/사람 행동 LL의 변화도 저장했으나 imitation 학습 자료에서의 fixed-token
민감도이며 일반화 성능이나 계산의 합리적 가치 증거로 사용하지 않는다.

7. imagined value 보상 항: 일관된 지배 항이라는 근거는 없다.
intrinsic policy/value loss를 분리하고 각 항을 줄이거나 제거했을 때의 가상 업데이트를 계산했다.
샘플마다 task/VOC gradient와의 방향이 달랐으며 제거가 항상 VOC와 더 잘 정렬되지는 않았다.
그 항을 제거하면 'VOC를 쓰는 학습'이 바로 커진다는 결과도 없었다. 기존 검토에서 재현한
optimistic model value→positive intrinsic reward 구조는 그대로지만, 실제 optimism의 크기와
환경 return에 미치는 영향은 현재 자료만으로 판단하지 않았다. 10M 시점 계수는 100M schedule
때문에 여전히 약 .9다.

다음 변경의 우선순위에 대한 해석
실제 counterfactual return으로 VOC target의 타당성을 확인하는 것이 우선이다. 그 뒤 별도의
공통 가치 오차 항/두 선택 표본 확보, task-value와 VOC가 공유 GRU에 주는 비중, 더 긴 BC burn-in을
한 항목씩 검증할 근거가 생겼다. WAIT만 줄이기, VOC ×100, intrinsic loss 제거 중 어느 것도
이번 분석만으로 확정할 처방은 아니다. 모델이 상황에 맞게 budget을 학습하는 목표 자체가
불가능하다는 결과도 아니다.

재현과 산출물
candidate_updates.csv, gradient_components.csv, gradient_summary.csv: 실제 체크포인트 기반 가상 업데이트
memory_comparisons.csv, memory_summary.csv: 같은 human target의 context 길이 비교
support_v2/: 실제 로그의 count별 표본 및 알려진 정답의 critic 대조 실험
gradient_candidates.png/pdf, memory_burnin.png/pdf, gate_support.png/pdf
qa.json: production 해시, checkpoint 보존, 이전 6M 진단 재현, target 일치 검증
historical_gradient_rows/summary.csv, historical_gradients.png/pdf: 실제 0–10M 전체 gradient 추이
gate_within_stage_rows/summary.csv: 같은 planning 안에서의 확률 변화와 실행 log score
manifest.json 및 analysis_sources/: 재현 설정과 분석 소스
support/는 가상 epsilon=.1의 경계 반올림을 보정하기 전 결과이므로 support_v2/를 사용한다.
'''
    (OUT/'report.txt').write_text(report)
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='Items 1,2,3,4,6,7; item5 excluded.',
        production_changed=False,training_or_jobs_changed=False,checkpoints_modified=False,
        gradient_probe_manifests=[str(f.relative_to(OUT)) for f in probes],
        memory_probes=list(memory_meta),support='support_v2',
        superseded_output='support: initial diagnostic used floating-point-sensitive 0.05 boundary for hypothetical epsilon=0.1; use support_v2.',
        gradient_source_versions={'four_short_probes':'analysis_sources/probe_planning_candidates_v1.py','later_warmup_probe':'analysis_sources/probe_planning_candidates.py'},
        latest_log_step=sm['last_training_step'],
        limitations=['One Enduro participant and lineage; no across-game or participant generalization.',
            'Copied-gradient/Adam counterfactuals, not retraining or held-out reward benefit.',
            'Fresh probes differ from historical training occupancy; later warmup probe is a sensitivity check.',
            'Memory reference is carried within a 41-edge sampled window, not exact full episode; replanning tokens fixed.',
            'Synthetic critic truth and fixed-row gate alternatives do not identify actual causal computation utility.'])
    manifest['artifact_hashes']={str(f.relative_to(OUT)):sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.suffix in ['.csv','.json','.py','.png','.pdf','.txt'] and f.name!='manifest.json'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(su.to_string(index=False));print(ms[ms.kind=='first_optional'].to_string(index=False));print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
