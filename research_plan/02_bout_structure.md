# 2. Bout Structure Analysis: NOOP의 시간적 구조와 내적 조직

*리뷰어 관점: Planning의 행동적 signature는 단순한 NOOP 비율이 아니라 시간적 구조에서 나타난다. 얼마나 긴 bout인가? 어떻게 시작하고 끝나는가? 어떤 내부 상태가 commit timing과 연결되는가?*

> **섹션 범위 정의**: Section 2는 NOOP bout의 *형태(shape)*를 기술한다. 여기서는 bout length, onset/commit 주변 temporal profile, Short/Long bout 차이, NOOP onset에서 real action commit까지의 trajectory, 그리고 commit hazard를 분석한다. Uncertainty는 bout structure를 설명하는 지표로 사용하며, uncertainty가 NOOP를 *유발하는가*라는 인과 질문은 Section 3, withholding이 실제로 *도움이 되는가*라는 기능 질문은 Section 4에서 다룬다.

## 공통 정의

현재 구현은 `research_script/02_structure_analysis.py`를 기준으로 한다.

- **Real step**: `status == 0`인 row. 실제 환경 action이 실행된 step이다.
- **NOOP bout**: real-step sequence에서 연속된 NOOP action 구간.
- **Commit**: NOOP bout이 끝난 직후의 첫 non-NOOP real action.
- **Short/Long 기준**: Section 1 survival 분석에서 얻은 subject × game별 `Cross2`를 사용한다. 현재 기본 데이터(`sub001`, `SpaceInvaders game 2`)에서는 `Cross2 = 26`이다.
- **Actor metrics**: `entropy_actor`, `margin_actor`는 `status == 0` row의 actor policy에서 계산한다. 이 값은 방금 실행된 real action을 고른 policy의 uncertainty/confidence로 해석한다.
- **Tree/search metrics**: real action이 실행된 `status == 0` row의 tree reps는 real action 이후 상태일 수 있으므로, decision-aligned tree metric은 바로 이전 `status == 2` row에서 가져온 `q_gap_prev_s2`, `rollout_spread_prev_s2`를 사용한다.

현재 메인 figure는 다음과 같다.

| Figure | 파일 | 핵심 질문 |
|--------|------|-----------|
| 2-1C | `fig_2_1C_onset_entropy_bout_length.png` | Bout 시작 시 uncertainty가 bout length를 예측하는가? |
| 2-2 | `fig_2_2_temporal_profiles.png` | Onset/commit 주변 actor uncertainty는 어떤 시간적 모양을 갖는가? |
| 2-2.1 | `fig_2_2_1_temporal_profiles_short_long.png` | 2-2의 temporal profile이 Short/Long bout에서 다르게 나타나는가? |
| 2-3 | `fig_2_3_short_long_metrics.png` | Actor policy metric과 prev-s2 tree metric이 Short/Long bout에서 어떻게 다른가? |
| 2-4 | `fig_2_4_noop_commit_policy_trajectory.png` | NOOP onset부터 real action commit까지 actor policy가 어떻게 변하는가? |
| 2-5 | `fig_2_5_commit_hazard.png` | Actor uncertainty와 search statistics가 commit timing을 얼마나 예측하는가? |

---

## 2-1. Bout Length Distribution and Onset Uncertainty

### 분석 목적

NOOP bout 길이가 단순한 짧은 omission들의 집합인지, 아니면 일부 길게 지속되는 structured bout을 포함하는지 확인한다. 특히 bout onset의 actor entropy가 이후 bout length를 얼마나 설명하는지 본다.

### 구현

`extract_noop_bouts`가 real-step table에서 NOOP bout을 추출하고, Section 1 output의 `Cross2`를 붙여 Short/Long class를 정의한다. `plot_fig_2_1c`는 bout onset의 `entropy_actor`와 `log(1 + bout_length)`의 관계를 그린다.

### 대략적 결과

현재 기본 데이터에서는 총 **2,196 bouts**가 추출된다.

| 지표 | 값 |
|------|----|
| Mean bout length | 8.37 real steps |
| Median bout length | 4 real steps |
| Cross2 | 26 real steps |
| Short bouts | 2,078개, 94.6% |
| Long bouts | 118개, 5.4% |

`fig_2_1C_onset_entropy_bout_length.png`에서 onset entropy와 bout length의 관계는 크지 않다. 기존 출력 기준으로는 약한 음의 관계가 관찰된다. 즉 긴 bout은 높은 entropy spike에서 시작한다기보다, 오히려 낮거나 안정적인 actor policy 상태에서 시작하는 경향이 있다.

### 해석

Bout length는 onset uncertainty 하나로 설명되지 않는다. 이 결과는 "uncertainty가 높아서 바로 NOOP가 시작된다"는 단순 모델보다, bout이 시작된 뒤 내부 과정에서 uncertainty와 confidence가 재조직된다는 해석을 지지한다.

---

## 2-2. Temporal Profiles Around Bout Onset and Commit

### 분석 목적

NOOP bout의 시작과 끝 주변에서 actor uncertainty가 어떤 시간적 패턴을 갖는지 기술한다. 핵심은 두 가지다.

- Onset이 entropy spike와 함께 발생하는가?
- Commit 직전과 직후에 policy uncertainty/confidence가 구조적으로 변하는가?

### 구현

`build_event_tables`가 각 bout의 onset 및 commit을 기준으로 `window_pre = 6`, `window_post = 6` real steps를 정렬한다. `plot_fig_2_2`는 actor entropy/margin의 onset-aligned, commit-aligned profile과 confidence gain 관련 panel을 그린다.

여기서 `entropy_actor`는 `status == 0`의 actor policy에서 계산된 real-action decision metric이다.

### 대략적 결과

**Onset-aligned profile**

- Onset 직전 entropy가 뚜렷하게 상승한다기보다, onset 주변에서는 entropy가 낮거나 안정적인 편이다.
- Bout 내부로 들어간 뒤 entropy가 증가하는 패턴이 더 두드러진다.
- 따라서 NOOP onset은 단순한 high-entropy trigger로 보기 어렵다.

**Commit-aligned profile**

- Commit에 가까워질수록 actor entropy가 상승하고, commit 직후에는 entropy가 떨어지는 rise-and-drop 패턴이 나타난다.
- `margin_actor`는 대체로 entropy와 반대 방향으로 움직인다.
- Commit 직후 entropy 감소는 real action이 uncertainty가 낮은 상태로 transition하는 것처럼 보이게 한다. 다만 기능적 이득 여부는 Section 4에서 control 분석이 필요하다.

**Uncertainty vs. confidence gain**

- Pre/onset uncertainty가 클수록 commit까지의 `-delta entropy`도 커지는 양의 관계가 보인다.
- 즉 높은 uncertainty 상태에서 시작한 bout일수록 commit까지 entropy reduction 폭이 커진다.
- 이 관계는 regression-to-mean 가능성이 있으므로, Section 2에서는 descriptive evidence로만 사용한다.

### 해석

2-2의 핵심은 **onset trigger와 commit resolution이 같은 현상이 아니라는 점**이다. Onset은 high-entropy spike가 아니지만, bout이 진행되고 commit이 가까워지는 동안 actor policy는 uncertainty/confidence의 재조직을 보인다.

---

## 2-2.1. Short/Long Split Temporal Profiles

### 분석 목적

2-2의 temporal profile이 모든 bout에서 동일하게 나타나는지, 아니면 Cross2로 나뉜 Short/Long class에 따라 다른지 확인한다.

### 구현

`plot_fig_2_2_1_short_long`은 `fig_2_2_temporal_profiles.png`와 같은 종류의 subfigure를 Short bout과 Long bout으로 나누어 그린다. Short/Long 구분은 Section 1에서 얻은 `Cross2 = 26`을 사용한다.

### 대략적 결과

- Short bout은 onset 이후 entropy 증가와 commit 부근의 높은 entropy가 더 뚜렷하다.
- Long bout은 평균 entropy가 더 낮고, commit 시점에서도 Short보다 낮은 entropy를 보인다.
- Policy margin은 Long bout에서 더 높게 유지된다.
- Uncertainty와 `-delta entropy`의 양의 관계는 Short/Long 모두에서 유지되는 쪽으로 보이지만, Long bout은 전체적으로 더 안정적인 policy regime에 가깝다.

### 해석

Short/Long은 단순히 같은 과정의 길이 차이가 아니라, 서로 다른 temporal regime일 가능성이 있다. Short bout은 빠른 uncertainty build-up과 commit으로, Long bout은 상대적으로 낮은 entropy와 높은 margin을 유지한 채 실행 시점을 기다리는 패턴으로 해석된다.

---

## 2-3. Short/Long Decision-Aligned Metrics

### 분석 목적

Short/Long bout 차이를 real-action actor policy와 search/tree statistics를 함께 사용해 비교한다. 중요한 점은 metric alignment다.

- Actor entropy와 policy margin은 real action을 고른 `status == 0` actor policy에서 가져온다.
- Q-gap과 rollout spread는 real action 직전의 `status == 2` tree reps, 즉 `prev_s2`에서 가져온다.

이렇게 해야 real action commit을 만든 policy와 search statistics를 같은 decision point에 맞춰 비교할 수 있다.

### 구현

`compute_short_long_metric_tables`가 pre, onset, commit, delta phase별 metric을 Short/Long으로 요약한다. `plot_fig_2_3_short_long_metrics`는 다음 네 가지 panel을 그린다.

- Entropy: `entropy_actor`
- Policy margin: `margin_actor`
- Q-gap: `q_gap_prev_s2`
- Rollout spread: `rollout_spread_prev_s2`

### 대략적 결과

현재 기본 데이터의 핵심 수치는 다음과 같다.

| Metric | Phase | Short | Long |
|--------|-------|-------|------|
| Actor entropy | onset | 0.900 | 0.878 |
| Actor entropy | commit | 1.027 | 0.935 |
| Policy margin | onset | 0.493 | 0.521 |
| Policy margin | commit | 0.391 | 0.541 |
| Prev-s2 Q-gap | onset | 0.131 | 0.107 |
| Prev-s2 Q-gap | commit | 0.129 | 0.134 |
| Prev-s2 rollout spread | onset | 0.547 | 0.515 |
| Prev-s2 rollout spread | commit | 0.544 | 0.499 |

Short bout은 commit 시 actor entropy가 높고 margin이 낮다. 반대로 Long bout은 commit 시 actor entropy가 낮고 margin이 높다. Prev-s2 tree metrics는 Short/Long 차이가 actor policy metric만큼 강하지 않다.

### 해석

가장 중요한 결과는 Long bout이 "더 불확실해서 오래 기다리는 bout"이 아니라는 점이다. Long bout은 오히려 real-action actor policy 기준으로 더 confident한 commit과 연결된다. 따라서 Long bout은 high-uncertainty prolongation이라기보다, **confident delay** 또는 **strategic waiting**에 가깝게 해석하는 것이 더 맞다.

Prev-s2 tree metric의 효과가 약한 것은 최종 직전 tree snapshot 하나만으로는 bout 전체 search process를 충분히 대표하지 못한다는 뜻일 수 있다. 이 점이 2-5의 imaginary search summary 분석으로 이어진다.

---

## 2-4. NOOP Onset to Real Action Commit Policy Trajectory

### 분석 목적

각 NOOP bout에서 NOOP 시작부터 real action commit까지 actor policy가 어떻게 변하는지 직접 본다. Bout마다 길이가 다르므로 x축을 0에서 1로 정규화한다.

이 분석은 다음 질문을 다룬다.

- Short bout과 Long bout은 commit까지 같은 방향으로 policy가 변하는가?
- Long bout은 단순히 entropy가 더 오래 누적되는 형태인가?
- Commit 직전의 actor confidence가 bout class별로 어떻게 다르게 조직되는가?

### 구현

`compute_noop_commit_policy_trajectory`는 각 bout에서 onset부터 commit까지의 real steps를 모은다. 이후 `entropy_actor`, `margin_actor` trajectory를 50개 normalized position으로 보간한다. `plot_fig_2_4_noop_commit_policy_trajectory`는 Short/Long 평균 trajectory와 SEM band를 그린다.

### 대략적 결과

Short bout:

- Entropy는 onset 약 **0.900**에서 commit 약 **1.027**로 증가한다.
- Policy margin은 onset 약 **0.493**에서 commit 약 **0.391**로 감소한다.
- 즉 빠르게 uncertainty가 증가하고 낮은 confidence로 commit하는 양상이다.

Long bout:

- Entropy는 onset 약 **0.878**에서 commit 약 **0.935**로 소폭 증가한다.
- Policy margin은 onset 약 **0.521**에서 commit 약 **0.541**로 유지되거나 약간 증가한다.
- 즉 오래 기다리지만 actor policy는 상대적으로 confident한 상태를 유지한다.

### 해석

2-4는 Short/Long 차이를 가장 직관적으로 보여준다. Short bout은 "uncertainty build-up 후 commit"에 가깝고, Long bout은 "낮은 entropy와 높은 margin을 유지한 채 commit timing을 기다리는 과정"에 가깝다.

따라서 uncertainty와 `-delta entropy`의 양의 관계만 보면 "uncertainty가 높을수록 commit 시 entropy가 줄어든다"는 해석이 가능하지만, 2-4를 함께 보면 Long bout에서는 다른 측면이 보인다. Long bout은 높은 actor uncertainty를 오래 견디는 과정이 아니라, 비교적 confident한 policy 상태에서 withholding이 지속되는 regime일 수 있다.

---

## 2-5. Commit Hazard From Actor and Search Features

### 분석 목적

NOOP bout 내부의 각 real step에서 "다음 real step이 commit인가?"를 예측한다. 이를 통해 commit timing이 단순히 elapsed time으로 설명되는지, actor entropy가 추가 설명력을 갖는지, 그리고 tree/search process가 actor metric 너머의 정보를 제공하는지 본다.

### 구현

`compute_commit_hazard_table`은 NOOP bout 내부의 각 NOOP real step을 한 row로 만들고, target을 `commit_next`로 둔다. 모델 입력은 다음 계열로 나뉜다.

- Elapsed only: `log_elapsed_noop_steps`
- Actor: `entropy_actor`
- Prev-s2 tree: `q_gap_prev_s2`, `rollout_spread_prev_s2`
- Imag search summary: 직전 real step 이후 현재 real step 전까지의 `status == 2` imaginary process 요약
  - `q_gap_imag_mean`, `q_gap_imag_final`, `q_gap_imag_slope`
  - `rollout_spread_imag_mean`, `rollout_spread_imag_final`, `rollout_spread_imag_slope`
  - `imag_cur_action_change_rate`, `imag_cur_action_entropy`

`compute_commit_hazard_models`는 logistic model ablation을 수행하고, `plot_fig_2_5_commit_hazard`는 CV AUC, log-loss improvement, coefficient를 요약한다.

### 대략적 결과

현재 기본 데이터에서는 **18,379 NOOP real steps** 중 **2,196 steps**가 `commit_next = 1`이다.

| Model | CV AUC | CV log loss | Elapsed 대비 log-loss 개선 |
|-------|--------|-------------|----------------------------|
| Elapsed only | 0.591 | 0.361 | 0.000 |
| Actor entropy | 0.635 | 0.353 | 0.008 |
| Prev-s2 tree | 0.592 | 0.361 | 0.000 |
| Actor + prev-s2 tree | 0.635 | 0.353 | 0.008 |
| Imag search summary | 0.599 | 0.359 | 0.002 |
| Actor + all search | 0.639 | 0.351 | 0.010 |

주요 coefficient 방향은 다음과 같다.

- `entropy_actor`는 commit hazard와 양의 관계를 보인다.
- `q_gap_prev_s2`와 `rollout_spread_prev_s2`만으로는 elapsed-only 대비 개선이 거의 없다.
- Imaginary process 전체 요약은 단독으로는 작지만, actor entropy와 결합하면 가장 좋은 성능을 낸다.

### 해석

Commit timing은 최종 prev-s2 tree snapshot 하나보다 actor policy uncertainty에 더 강하게 연결된다. 다만 한 real step 사이에 약 40개 내외의 imaginary/search step이 존재하므로, search process를 단일 마지막 snapshot이 아니라 trajectory summary로 요약하면 약간의 추가 설명력이 생긴다.

따라서 현재 결과는 다음 순서의 claim을 지지한다.

1. Commit hazard의 가장 직접적인 behavioral correlate는 `entropy_actor`다.
2. `q_gap_prev_s2`, `rollout_spread_prev_s2` 같은 final tree metric만으로는 약하다.
3. Imaginary process 전체의 변화량과 다양성은 actor entropy 위에 작은 추가 정보를 제공한다.

이 분석은 Section 3에서 vp net activation, sr net activation, tree reps를 이용한 richer predictive model로 확장할 수 있다.

---

## 보조 출력과 현재 해석상의 위치

현재 스크립트는 메인 figure 외에도 다음 CSV를 저장한다.

| CSV | 역할 |
|-----|------|
| `real_step_metrics.csv` | `status == 0` real-step 단위 metric table |
| `noop_bouts.csv` | NOOP bout onset, commit, length, Short/Long annotation |
| `event_prepost.csv` | onset/commit 주변 pre/onset/commit/delta event table |
| `event_temporal.csv` | onset/commit 기준 windowed temporal table |
| `short_long_event_metrics.csv` | Short/Long event-level metric |
| `short_long_metric_summary.csv` | Short/Long phase별 평균 및 SEM |
| `short_long_metric_tests.csv` | Short/Long 차이 test |
| `noop_commit_policy_trajectory.csv` | 2-4 normalized trajectory source |
| `commit_hazard_steps.csv` | 2-5 hazard model source table |
| `commit_hazard_model_summary.csv` | 2-5 model ablation 결과 |
| `commit_hazard_coefficients.csv` | 2-5 coefficient 요약 |

`bout_internal_trajectory.csv`, `bout_action_stability.csv`, `bout_commit_dynamics.csv`, `episode_stats.csv`, `half_session_stats.csv`도 저장되지만 현재 메인 narrative에서는 보조 자료에 가깝다. 예전의 `fig_2_3_sequential_structure.png`와 `fig_2_4_session_adaptation.png`는 현재 Section 2 main figure에서 제외되었다.

---

## Section 2의 통합 해석

현재 Section 2의 중심 결론은 다음과 같다.

1. **NOOP bout은 길이상 혼합 구조를 갖는다.** 대부분은 Short bout이지만, Cross2 이상으로 이어지는 Long bout이 별도 regime처럼 보인다.
2. **NOOP onset은 high-uncertainty spike가 아니다.** Onset entropy는 bout length를 강하게 예측하지 않고, 긴 bout은 오히려 낮은 entropy에서 시작하는 경향이 있다.
3. **Commit 주변에서는 actor policy의 재조직이 보인다.** Entropy와 margin은 commit 전후로 뚜렷한 temporal profile을 가진다.
4. **Short와 Long은 같은 과정의 길이 차이만은 아니다.** Short bout은 high-entropy, low-margin commit과 연결되고, Long bout은 lower-entropy, higher-margin commit과 연결된다.
5. **Commit timing은 actor entropy가 가장 잘 설명한다.** Final prev-s2 tree metric만으로는 약하지만, imaginary search trajectory summary를 포함하면 actor entropy 위에 작은 추가 설명력이 생긴다.

이 결과를 바탕으로 Section 3에서는 "어떤 내부 representation이 commit hazard와 bout class를 예측하는가?"를 묻고, Section 4에서는 "withholding이 실제 성과나 uncertainty reduction에 기능적으로 기여하는가?"를 검증한다.
