# 섹션별 분석 목적 / 방법 / 목표 결과

---

## Section 1 — Behavioral Phenomenon
**목적**: 인간의 행동 보류(NOOP)가 passive omission이 아니라 *시간 구조와 reward relevance를 가진 행동 현상*임을 행동 데이터만으로 정리한다. 이후 섹션의 계산적·기능적 해석을 위한 descriptive foundation, 즉 C1 claim의 현재 구현 버전을 성립시킨다.

**방법 (6개 층위)**:
- **1-1** Withholding bout schematic + action distribution — subject별 action repertoire와 reference thinker profile 정리
- **1-2** N=6 subject-level robustness 확인 — effect size, NOOP ratio ↔ performance null, meta-analytic direction consistency, ICC 보조 계산
- **1-3** Reward comparison by subject and episode — withholding-preceded vs. not-preceded episode mean reward 비교
- **1-4** Episode-level NOOP ratio vs. withholding benefit — quantity-null 재확인
- **1-5** Fatigue 및 단순 serial dependency 대안 설명 배제 — episode-position density + lag-1 AC
- **1-6** Bout survival curve vs. exponential baseline — random omission 귀무가설 기각

**목표 결과**: 인간의 NOOP 사용은 subject와 game 전반에 걸쳐 일관되며, 총량 자체는 성과를 예측하지 않지만 reward-linked deployment와 비-random bout structure는 분명하다. Fatigue 설명은 약하고, survival 구조는 memoryless omission과 다르다 → C1 성립.

---

## Section 2 — Bout Structure
**목적**: NOOP bout의 *형태(shape)* 를 기술한다 — 길이 분포, onset·commit 주변 시간적 궤적, bout 내부 전환 구조. Uncertainty가 NOOP를 유발하는지(→ Section 3)나 이득이 있는지(→ Section 4)는 다루지 않는다.

**방법**:
- **2-1** Bout length distribution — exponential vs. heavy-tail 이탈, Cross2 기반 Short/Long 유형 분리, state-level uncertainty와 bout length 회귀
- **2-2** Onset·commit aligned temporal profile — z-score 정규화 후 entropy/margin/Q-gap 궤적 추출, rise-and-fall 패턴 formal test
- **2-3** Transition matrix (NOOP↔action), bout-internal uncertainty trajectory (단조감소 여부), cur_action stability
- **2-4** Within-session learning curve — episode 순서별 NOOP 비율 변화, early/late half uncertainty-NOOP coupling 비교

**목표 결과**: **Onset paradox** 확인 (onset은 entropy spike가 아닌 entropy 하강 시점에서 발생), commit 직전 entropy peak + 직후 급락. Short(crisis-response) vs. Long(strategic planning) 두 유형의 혼합 분포 구조.

---

## Section 3 — Computational Interpretation
**목적**: NOOP bout의 형태를 *일으키는* 계산 변수가 무엇인지 검증한다 — uncertainty → NOOP의 인과 방향 및 가장 강력한 예측 변인 식별.

**방법**:
- **3-1** Uncertainty-NOOP coupling: 다양한 uncertainty metric(entropy, Q-gap, search_disagreement 등)과 NOOP probability의 관계, human vs. IL thinker 비교
- **3-2** Mixed-effects logistic regression + AIC/BIC model comparison — 표준화 beta 계수, ROC/AUC, human·thinker 회귀계수 비교
- **3-3** AR(1)~(5) residual test (perseveration 해리), lagged regression / Granger causality (temporal precedence), bout onset entropy → bout length 상관

**목표 결과**: policy entropy/Q-gap이 NOOP를 독립적으로 예측하며, AR 잔차 제거 후에도 uncertainty 기여가 유지됨 → perseveration이 아닌 uncertainty-driven delay 확인 → C2 성립.

---

## Section 4 — Normative Function
**목적**: 행동 보류가 실제로 action quality와 task outcome을 향상시키는지 검증한다 — selection bias를 통제한 상태에서 functional benefit을 확립하여 C3 claim을 성립시킨다.

**방법**:
- **4-1** VRE(Value Revision Error) 및 k-step reward: withholding-preceded vs. non-preceded overt action 비교, uncertainty bin별 상호작용
- **4-2** Propensity score matching (entropy + position + value 동시 통제), dose-response (bout length ~ VRE reduction)
- **4-3** Session-level regression: total score ~ NOOP ratio + pre-NOOP uncertainty (subject random effect 포함)
- **4-4** Counterfactual simulation: human의 NOOP step에서 즉각 행동 시 결과를 thinker로 시뮬레이션

**목표 결과**: NOOP 이후 action의 VRE 유의 감소, k-step reward 증가. Propensity matching 후에도 효과 유지, bout 길이와 benefit 간 dose-response 관계 확인 → C3 성립.

---

## Section 5 — Neural Mechanism
**목적**: delayed commitment와 planning을 매개하는 뇌 회로를 fMRI 데이터로 규명한다 — behavioral mechanism의 신경 기반 확인.

**방법**:
- **5-1** Commitment gating GLM: NOOP 여부·bout length를 regressor로, caudate/putamen·SMA·ACC 반응 분석
- **5-2** Uncertainty-linked planning GLM: entropy·search_disagreement·VRE를 regressor로, hippocampus·vmPFC·dlPFC 반응 분석
- **5-3** Behavior-to-brain bridge: step-level NOOP probability·VRE·k-step reward → brain activation, dissociation test
- **5-4** RSA (thinker latent geometry ↔ brain representation), MVPA (NOOP vs. action commitment 분류)

**목표 결과**: striatum/frontal은 action commitment 시점 phasic response (gating), hippocampus는 NOOP bout 동안 uncertainty-proportional sustained activation (prospective search). 두 신호의 해리 확인.

---

## Section 6 — Representational Mechanism
**목적**: thinker의 latent state (tree search representation) geometry를 분석하여 NOOP 동안의 계산 구조를 규명한다 — planning 결정점에서 탐색(diffusion)과 수렴(drift)의 분리.

**방법**:
- **6-1** Diffusion Maps: tree nodes에 kernel 적용 → Markov chain eigendecomposition, spectral gap (λ₂/λ₁) = action-value conflict 지표, multiscale cluster structure (P^t), diffusion distance
- **6-2** Neural SDE / g(z_t) proxy: imagined node 간 pairwise distance 평균 → diffusion magnitude 대리 지표, NOOP vs. action steps 비교, dose-response
- **6-3** InputDSA / DMDc: A(intrinsic) vs. B(input-driven) dynamics 분리, NOOP 구간 intrinsic dominant 예측, human vs. thinker A/B decomposition 비교

**목표 결과**: NOOP 구간에서 spectral gap 감소(action-value conflict ↑), diffusion magnitude 증가. Action commit 직전 drift dominant 전환. Human과 high-performing thinker의 A/B decomposition 유사 (Anna Karenina principle).

---

## Section 7 — Bidirectional Alignment
**목적**: thinker-human 표현 공간의 양방향 정렬을 분석하여 thinker의 어떤 차원이 인간과 공유된 계산 공간인지 진단한다 — 단방향 RSA(thinker→brain)에서 양방향 비교로 확장.

**방법**:
- **7-1** Forward predictivity (thinker → human NOOP/behavior) vs. Reverse predictivity (behavior → 각 thinker channel) Ridge regression, 비대칭성 정량화
- **7-2** R²_reverse 상위 20% = Common units / 하위 20% = Unique units 식별, NOOP onset-aligned temporal profile 비교, outcome 예측력 비교
- **7-3** Real step별 preceding imaginary window PCA → participation ratio, NOOP vs. action step 비교, bout length 상관
- **7-4** Common/Unique subspace에서 imaginary trajectory 수렴도 비교 (||last − first|| in each subspace)

**목표 결과**: Forward >> Reverse 비대칭성 존재 (thinker에 human이 접근하지 못하는 extra dimension). Common units는 NOOP onset에 반응하고 k-step reward/VRE를 더 잘 예측. NOOP planning window의 effective dimensionality가 action step보다 높음.

---

# 진행 상황 요약 (Status Tracker)

| Section | 분석 항목 | Figure | 상태 |
|---------|-----------|--------|------|
| 1-1 | Withholding bout schematic | **Fig 1-1A** (`fig_1-1_A_withholding_schematic.png`) | ✅ Done |
| 1-1 | Action distribution: S1–S6 + non-IL thinker reference profile | **Fig 1-1B** (`fig_1-1_B_action_distribution.png`) | ✅ Done |
| 1-2 | Subject-level NOOP proportion scatter (피험자×게임) + Cohen's d vs chance | **Fig 1-2A** (`fig_1-2_individual_differences.png` Panel A) | ✅ Done |
| 1-2 | Session reliability: ICC across days per subject × game | Supplementary table (`1-2_icc_by_subject_game.csv`) | ✅ Done |
| 1-2 | NOOP ratio ↔ performance scatter: null result 확인 (dual-axis, Pong/SI) | **Fig 1-2B** (`fig_1-2_individual_differences.png` Panel B) | ✅ Done |
| 1-2 | Meta-analytic direction check: 피험자별 NOOP above chance (Pong vs SI scatter) | **Fig 1-2C** (`fig_1-2_individual_differences.png` Panel C) | ✅ Done |
| 1-3 | Episode-level reward: withholding-preceded vs. not, by subject × game | **Fig 1-3** (`fig_1-3_reward_subject_episode.png`) | ✅ Done |
| 1-3 | Cross-game summary tables: paired test, bout length summary, direction CSV | `1-3_bout_lengths.csv`, `1-3_effect_direction.csv` | ✅ Done |
| 1-4 | NOOP ratio ~ withholding benefit scatter (episode-level) | **Fig 1-4** (`fig_1-4_noopratio_postnoop_reward.png`) | ✅ Done |
| 1-5 | Episode-position NOOP density: 20-bin 연속 곡선 (fatigue 배제) | **Fig 1-5A** (`fig_1-5_alternative_exclusion.png` Panel A) | ✅ Done |
| 1-5 | Episode 3분할(early/mid/late) summary statistics | Supplementary table (`1-4_episode_position_thirds.csv`) | ✅ Done |
| 1-5 | Lag-1 NOOP autocorrelation: boxplot + subject means per game | **Fig 1-5B** (`fig_1-5_alternative_exclusion.png` Panel B) | ✅ Done (descriptive) |
| 1-6 | NOOP bout survival curve: KM per subject × game + exponential baseline | **Fig 1-6** (`fig_1-6_survival_by_subject.png`) | ✅ Done |
| 1-1 | Uncertainty-conditional NOOP rate: human vs. IL thinker per entropy bin (selectivity 차이 정량화) | — | 🔲 New (Section 3 연결) |
| 2-1 | Bout length ~ uncertainty regression (thinker entropy/q_gap, 모델 데이터 필요) | — | 🔲 New |
| 2-1 | Bout length ~ post-action quality | — | 🔲 New |
| 2-2 | Onset-aligned temporal profile | **Fig 2A** | ✅ Done |
| 2-2 | Commit-aligned temporal profile | **Fig 2B** | ✅ Done |
| 2-2 | Rise-then-fall pattern: formal statistical test | — | 🔲 New |
| 2-3 | Transition matrix (NOOP↔action) | — | 🔲 New |
| 2-3 | Bout-internal uncertainty trajectory | — | 🔲 New |
| 2-4 | Within-session learning curve | — | 🔲 New |
| 3-1 | Pre→commit Δentropy / Δmargin / Δq-gap | **Fig 3A** | ✅ Done |
| 3-1 | Pre-uncertainty vs confidence gain | **Fig 3B** | ✅ Done |
| 3-1 | Uncertainty-NOOP coupling: qualitative | — | 🔲 Planned |
| 3-2 | Mixed-effects logistic regression + model comparison | — | 🔲 New |
| 3-3 | AR(1)~AR(5) residual uncertainty test (perseveration 해리, **⚠️ thinker 데이터 완비 후**) | — | 🔲 New |
| 3-3 | Lagged regression / Granger causality (temporal precedence) | — | 🔲 New |
| 3-3 | Bout onset entropy → bout length 상관 | — | 🔲 New |
| 4-1 | VRE: withholding-preceded vs. not | **Fig 4A** | ✅ Done |
| 4-1 | k-step reward: withholding-preceded vs. not | **Fig 4B** | ✅ Done |
| 4-1 | Uncertainty-bin × withholding on outcome | **Fig 4C** | ✅ Done |
| 4-2 | Matched-control comparison (entropy-matched) | **Fig 4D** | ✅ Done |
| 4-2 | Propensity score matching (multi-covariate) | — | 🔲 New |
| 4-2 | Dose-response: bout length ~ VRE reduction | — | 🔲 New |
| 4-3 | Session-level performance ~ NOOP selectivity | — | 🔲 New |
| 4-4 | Counterfactual/ablation simulation | — | 🔲 Optional |
| 5 | fMRI: commitment gating GLM | — | 🔲 Planned |
| 5 | fMRI: uncertainty-linked planning GLM | — | 🔲 Planned |
| 5 | RSA: thinker geometry ↔ brain | — | 🔲 New |
| 6 | Latent state embedding geometry | — | 🔲 Planned |
| 6 | Tree motif analysis | — | 🔲 Planned |
| 6-1 | Diffusion Map on tree nodes: eigendecomposition + diffusion coordinate 시각화 | — | 🔲 New |
| 6-1 | Spectral gap (λ₂/λ₁) 분석: NOOP vs action steps 비교 → action-value conflict 지표 | — | 🔲 New |
| 6-1 | Multiscale cluster structure: P^t (t=8, 64, 1024) 비교 | — | 🔲 New |
| 6-1 | Diffusion distance D_t(real_node, imagined_nodes): NOOP 직전 분포 확산 여부 | — | 🔲 New |
| 6-2 | g(z_t) proxy: imagined node간 pairwise distance 평균 — NOOP vs action 비교 | — | 🔲 New (우선순위 높음) |
| 6-2 | Drift-diffusion 분리 (neural SDE fit): f/g ratio로 commit 직전 drift dominance 검증 | — | 🔲 New |
| 6-2 | g(z_t) ~ NOOP probability 및 bout length dose-response | — | 🔲 New |
| 6-3 | DMDc fit on tree_reps: A (intrinsic), B (input-driven) matrix 추정 | — | 🔲 New |
| 6-3 | SubspaceDMDc: partially observed 설정에서 A, B 추정 개선 | — | 🔲 New |
| 6-3 | A vs B eigenvalue 스펙트럼: NOOP vs action steps 비교 | — | 🔲 New |
| 6-3 | InputDSA state/input score: NOOP 구간 intrinsic dominant 예측 검증 | — | 🔲 New |
| 6-3 | Human vs thinker A/B decomposition InputDSA 비교 (Anna Karenina principle) | — | 🔲 New |

---

# 리뷰어 예상 질문 및 대응 전략

### Q1. "N=6은 너무 작다. 어떻게 일반화할 수 있는가?"
→ **대응**: 피험자 내 일관성 (ICC), bootstrap confidence interval, game-level replication (Pong + Space Invaders를 독립 replication으로 취급), effect size 보고.

### Q2. "NOOP가 많은 것이 planning의 증거인가? Motor inertia일 수 있지 않은가?"
→ **대응**: Uncertainty-selectivity (평균이 아닌 state-contingent), episode-position analysis, autocorrelation 통제, matched-control design.

### Q3. "VRE와 k-step reward 개선이 selection bias일 수 있지 않은가? Better states에서 withhold하기 때문에 더 좋은 결과가 나올 수 있다."
→ **대응**: Propensity score matching (entropy + value + position 동시 통제), dose-response relationship, uncertainty × withholding interaction.

### Q4. "Thinker의 내부 computation이 실제로 human의 planning을 대리하는가? 그 bridge가 충분한가?"
→ **대응**: Imitation learning 후에도 남는 차이가 있음을 보임 → structural difference, RSA analysis (section 5-4, 6).

### Q5. "Neural mechanism이 behavioral mechanism과 독립적인가? 아니면 circular한가?"
→ **대응**: Dissociation test (planning content vs. gating signal), 각 region은 서로 다른 computational regressor에 특이적으로 반응해야 함.

### Q6. "Cross-game generalization이 없다면 game-specific finding이다."
→ **대응**: Section 1-3 및 3-1에서 두 게임 모두에서 동일한 방향의 효과를 보임을 명시.
