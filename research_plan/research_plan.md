# Abstract

## 한국어 abstract

인간은 미래 행동의 결과를 내부적으로 시뮬레이션하는 **model-based reinforcement learning** agent로 자주 개념화된다. 그러나 자연적인 과제에서 나타나는 인간의 행동은 매 시점마다 **overt action**이 선택된다고 가정하는 표준적인 **discrete-time formulation**과 뚜렷하게 다르다. 특히 인간은 의미 있는 빈도로 행동을 보류하며, 이는 적응적 제어가 단지 무엇을 할지를 결정하는 문제를 넘어, 그 행동을 언제 실행할지를 조절하는 과정까지 포함함을 시사한다. 본 연구에서는 이러한 인간 planning의 특징이, 외부 행동의 즉각적인 실행을 지연시키는 동안 내부적인 **model-based computation**을 지속하는 **learned-search agent**로 포착될 수 있는지 검증한다. 이를 위해 Atari gameplay 동안 수집된 행동 및 **fMRI** 데이터를 이용하여, **learned world model**과 **learned search algorithm**에 기반한 agent와 인간의 행동을 비교하고, 증가된 행동 보류가 단순한 수동적 omission이 아니라 **action commitment**의 전략적 지연을 반영하는지 평가한다. 또한 이러한 **delayed commitment**가 **uncertainty** 또는 **task-state geometry**에 따라 선택적으로 동원되는지, 이후의 **action quality**와 **task outcome**을 향상시키는지, 그리고 이러한 과정이 뇌에서 어떻게 표현되는지를 조사한다. 우리는 **hippocampal-prefrontal circuit**이 predictive task structure와 prospective search representation을 부호화하고, **frontostriatal system**이 action commitment의 gating에 기여할 것이라고 가정한다. 이러한 관점은 **overt inaction**을 planning의 계산적으로 의미 있는 구성 요소로 재해석하며, 인간의 **model-based behavior**가 기존의 reinforcement learning agent와 어떻게 구별되는지에 대한 mechanistic account를 제공한다.

## English abstract

Humans are often conceptualized as **model-based reinforcement learning** agents that internally simulate the consequences of future actions. Yet behavior in naturalistic tasks departs markedly from the standard **discrete-time formulation** in which an **overt action** is selected at every step. In particular, humans frequently withhold action, suggesting that adaptive control depends not only on what action to take, but also on when to commit to that action. Here, we test whether this distinctive feature of human planning can be captured by a **learned-search agent** that postpones action commitment while continuing internal **model-based computation**. Using behavioral and **fMRI** data acquired during Atari gameplay, we compare human behavior with agents built on a **learned world model** and **learned search algorithm**, and ask whether elevated action withholding reflects strategic postponement of **action commitment** rather than passive omission. We further examine whether such **delayed commitment** is recruited selectively under **uncertainty** or challenging **task-state geometry**, whether it improves subsequent **action quality** and **task outcome**, and how it is expressed in the brain. We hypothesize that **hippocampal-prefrontal circuits** encode predictive task structure and prospective search representations, whereas **frontostriatal systems** contribute to the gating of action commitment. This framework recasts **overt inaction** as a computationally meaningful component of planning and offers a mechanistic account of how human **model-based behavior** differs from conventional reinforcement learning agents.

---

# 연구 핵심 주장과 논리 흐름

> 이 연구의 중심 claim은 다음 세 단계로 구성된다:
>
> **C1 (현상):** 인간의 행동 보류는 passive omission이 아니라 체계적이고 선택적인 현상이며, IL thinker와 질적으로 구별된다. IL thinker는 NOOP *빈도*를 모방할 수 있지만, uncertainty-contingent *selectivity*는 human에서만 나타나는 전략적 특성이다. \
> **C2 (계산적 해석):** 이 보류는 uncertainty 또는 action-value conflict에 민감하게 반응하는 structured postponement signal이다. \
> **C3 (기능적 이득):** 이러한 지연은 이후의 action quality와 task outcome을 향상시키며, 따라서 planning의 행동적 proxy다.
> 각 claim은 독립적으로 검증 가능해야 하며, 상위 claim은 하위 claim에 논리적으로 의존한다.
> 리뷰어는 C1이 충분히 성립하지 않으면 C2와 C3의 해석 전체를 문제 삼는다.

---

# Main

## 1. Behavioral Phenomenon: 인간의 행동 보류는 체계적이고 선택적이다

*관점: "단순히 NOOP을 많이 누른다"는 것 이상을 보여야 한다. Passive omission과 active postponement를 어떻게 구별하는가? N=6이라는 소규모 표본에서 결론을 내릴 수 있는가?*

> **섹션 구조**: Section 1은 다섯 개의 층위에서 C1(현상의 체계성)을 인간 행동 데이터만으로 확립한다. 모델 데이터 없이 성립하는 주장이어야 하며, Section 2–4의 계산적·기능적 해석을 위한 토대가 된다.
>
> - **1-1**: 기본 현상 확립 (human vs. thinker NOOP gap)
> - **1-2**: 효과의 robustness와 "총량이 아닌 선택적 사용"이 핵심임을 확립
> - **1-3**: 두 게임 간 일반화 (game-specific artifact 기각)
> - **1-4**: 대안 설명(fatigue, perseveration) 배제
> - **1-5**: Bout 구조의 비-random성 확립 (planned delay의 구조적 증거)

---

### 1-1. 기본 현상: Human vs. IL Thinker action distribution

*리뷰어 관점: 비교 대상은 반드시 IL thinker여야 한다. Pretrained thinker와의 차이는 policy 자체의 차이(task 수행 방식)일 수 있으므로, imitation 이후에도 gap이 남는지가 핵심이다. Section 1-1의 모든 비교는 **human vs. IL thinker** 기준이다.*

#### 확보된 결과 (sub001 ses04, SpaceInvaders, 8개 파일, 49,188 real steps)
- Human NOOP: **39.0%**, **IL Thinker** NOOP: **36.0%**, gap: **+3.05pp**
- NOOP bout 총 **2,196개**, 평균 bout 길이 **8.37 real steps**
- Human–IL thinker action distribution JSD 평균: **0.005**
- Gap이 작다는 것은 **IL이 human의 NOOP 빈도를 성공적으로 모방했다는 증거**이며, 남은 residual gap(+3.05pp)이 imitation으로 포착되지 않는 **human-specific withholding의 순수 신호**다.

> **비교 기준 명확화**: pretrained thinker (NOOP ~22%)는 policy 자체가 human과 다르기 때문에 direct NOOP gap 비교에 적합하지 않다. IL thinker는 human action distribution을 모방했기 때문에, 이 기준에서 남은 gap과 selectivity 차이가 **human-specific strategic timing**의 증거다. Pretrained thinker는 별도 대조군으로 Figure 1-1에 함께 표시하되, 해석의 중심은 IL thinker와의 비교다.

#### 의미 및 가설 연결
IL thinker는 human의 NOOP *빈도*를 잘 모방한다(JSD 0.005). 그러나 이후 섹션(Section 3)에서 보이듯 IL thinker는 NOOP *selectivity* — 즉 uncertainty가 높은 state에서 선택적으로 NOOP를 하는 패턴 — 는 human만큼 재현하지 못한다. 이것이 C1의 핵심 주장이다: **IL은 NOOP frequency를 학습하지만, uncertainty-contingent selectivity는 human에서만 나타나는 전략적 특성이다.**

#### 한계 및 후속 필요
- 현재 sub001 단일 피험자 수준 → **N=6 전체 확장 필요** ⚠️
- Human vs. IL thinker의 **uncertainty-conditional NOOP rate 비교**: 같은 uncertainty bin에서 human이 IL thinker보다 더 높은 NOOP rate를 보이는가? (selectivity 차이 정량화, Section 3 연결)
- IL thinker 내부 tree statistics와 residual NOOP gap의 관계 정량화 (Section 2로 연결)

---

### 1-2. 효과의 robustness와 "총량이 아닌 선택적 사용"

*리뷰어 관점: N=6에서 집단 평균만 보여주면 충분하지 않다. 피험자 간 일관성이 없으면 entire effect가 outlier에 의해 driven될 수 있다. 또한 NOOP가 많다는 것이 좋은 것인가, 나쁜 것인가?*

#### 확보된 결과 (N=6, Pong & SpaceInvaders)

**Fig 1-2A — NOOP proportion & effect size:**
| Game | Mean NOOP | Range (sub 1–6) | Cohen's d vs. chance (1/6) |
|------|-----------|-----------------|----------------------------|
| Pong | **86.9%** | 79.0%–91.8% | **d = 13.65** |
| SpaceInvaders | **43.4%** | 38.2%–53.1% | **d = 3.91** |
- 6/6 피험자 모두 두 게임에서 chance 이상 NOOP → effect는 outlier-driven이 아님

**Fig 1-2B — Session reliability (ICC across days):**
- 모든 피험자 × 게임 조합에서 ICC = **−0.28 ~ +0.16** (전부 낮음)
- 피험자별 NOOP 비율은 **trait-stable individual difference가 아니라 session-level state에 따라 유동적**

**Fig 1-2C — NOOP ratio ↔ performance (null):**
| Game | Level | r | p |
|------|-------|---|---|
| Pong | Subject | 0.005 | 0.99 |
| SpaceInvaders | Subject | −0.398 | 0.43 |
- 두 게임 모두 유의하지 않음 → **NOOP 총량은 performance를 예측하지 않는다**

**Fig 1-2D — Episode-level reward: withholding-preceded vs. not:**
- 6/6 피험자 × 2 games에서 일관되게 유의 (**, ***) → NOOP-preceded episode의 reward가 더 높음

**Fig 1-2E — NOOP ratio ↔ withholding benefit (episode-level):**
- Pong: r = −0.01, p = 0.938; SpaceInvaders: r = −0.04, p = 0.482 (둘 다 null)
- 그러나 **benefit > 0이 두 게임 모두에서 일관** → NOOP ratio와 무관하게 withholding 자체는 항상 이득

#### 핵심 주장: 총량이 아닌 사용 자체가 중요하다

세 결과(1-2C, D, E)가 하나의 내러티브를 구성한다:
> - 1-2C: NOOP 총량 ↔ episode score = **null** → "많이 누른다고 성과가 높아지지 않는다"
> - 1-2E: NOOP 총량 ↔ withholding benefit = **null**, but benefit > 0 consistently → "얼마나 많이 쓰든, 쓸 때마다 이득이 있다"
> - 1-2D: NOOP-preceded episode reward > non-preceded → "전략적으로 사용된 NOOP는 episode 결과를 향상시킨다"
>
> **결론: withholding의 총량이 아니라 사용 자체(quality of deployment)가 중요하다.** 1-2C null result에 대한 리뷰어 공격("NOOP는 의미없는 것 아닌가?")을 1-2E가 직접 반박한다.

Non-IL thinker는 NOOP 비율이 낮으면서도 성능이 더 높다는 사실이 이를 뒷받침한다: policy가 이미 sharp한 agent는 deliberation이 불필요하고, policy가 덜 sharp한 human은 withholding을 통해 deliberation을 보상한다. **핵심은 NOOP의 총량이 아니라 selectivity다.**

ICC가 낮다는 사실은 두 가지로 해석된다: (1) session마다 다른 task state geometry에 반응하는 state-reactive 행동 → selectivity 해석 지지, (2) 단순 측정 노이즈 가능성 → Section 3에서 uncertainty-NOOP coupling의 stability로 재확인 필요.

#### 추가로 필요한 연구
- **Selectivity index**: uncertainty-triggered NOOP 비율이 session 성과를 예측하는지 (총 NOOP ratio null과 대비하면 narrative가 강해짐) **[NEW, 우선순위 높음]**

---

### 1-3. 게임 간 일반화

*리뷰어 관점: 하나의 게임에서만 나타나는 현상은 task-specific artifact일 수 있다. 동일한 패턴이 두 게임에서 재현된다면 general mechanism의 증거가 된다.*

#### 확보된 결과 (N=6)

**Fig 1-3A — Effect direction consistency:**
- **6/6 피험자** 모두 Pong과 SpaceInvaders에서 NOOP > chance → 완벽한 방향 일관성

**Fig 1-3B, 1-3C — Bout length & meta-analytic check:**
| Game | N bouts | Mean length | Max |
|------|---------|-------------|-----|
| Pong | 13,038 | **49.5 steps** | 2,899 |
| SpaceInvaders | 60,109 | **9.9 steps** | 490 |
- 두 게임 모두 heavy-tailed distribution (max/mean ≈ 60배 이상)
- 피험자별 NOOP above-chance scatter: 두 게임에서 모두 양의 방향 일관

#### 의미 및 가설 연결
방향 일관성(6/6)은 game-specific artifact 기각의 강력한 증거다. 단 절대 수준 차이(87% vs 43%, mean bout 49 vs 9.9)가 크므로, 두 게임이 **동일한 mechanism**을 반영한다고 단순 주장하기는 어렵다. **SpaceInvaders가 strategic withholding의 clean test bed**이며, Pong은 game-structure contrast 역할로 활용하는 것이 적합하다.

#### 추가로 필요한 연구
- **Uncertainty-NOOP coupling 게임 간 비교**: SI에서 확인된 uncertainty selectivity가 Pong에서도 재현되는지 (Section 3과 연결) **[NEW]**

---

### 1-4. 대안 설명 배제

*리뷰어 관점: NOOP가 많다는 주장의 가장 강한 반론은 "버튼 피로", "motor inertia", "passive omission"이다. 이를 데이터로 배제해야 한다.*

#### 확보된 결과 (N=6)

**Fig 1-4A/B — Fatigue 설명 기각:**
| Game | Early (0–33%) | Late (66–100%) | Early vs. Late p |
|------|--------------|----------------|-----------------|
| Pong | 86.6% | 86.7% | 0.20 (NS) |
| SpaceInvaders | 44.2% | **41.9%** | ≈ 0 (p < 10⁻¹⁰⁰) |
- Pong: episode 전반에 걸쳐 균일 → 피로 효과 없음
- SpaceInvaders: 오히려 후반에 감소 → 피로 예측과 정반대 → **Fatigue/motor inertia 기각** ✅

**Fig 1-4C — Perseveration 설명 (부분 배제):**
- Pong: mean AC = **0.831**; SpaceInvaders: mean AC = **0.813**
- Base-rate 예측치 대비: Pong 0.77 → 실제 0.83 (소폭 초과); SI 0.51 → 실제 0.81 (대폭 초과)
- SpaceInvaders에서 bout 구조만으로 설명되지 않는 serial dependency 존재

높은 AC는 perseveration 증거가 아니다 — bout 구조가 있으면 AC는 자연히 높아진다. 그러나 SI에서 base-rate 대비 잔차 AC가 크다는 것은 두 가지 해석이 경쟁함을 의미한다: (1) active search 지속 vs. (2) passive perseveration. **Formal 해리는 AR-residual uncertainty test(→ Section 3-3)에서 수행한다.**

---

### 1-5. Bout 구조의 비-random성: Survival analysis

*리뷰어 관점: Random omission이라면 bout length가 지수분포를 따라야 한다. 이로부터의 이탈이 "계획된 지연"의 구조적 증거다.*

#### 지수 기준선(Exponential Baseline)을 그리는 이유

NOOP 행동 보류의 "passive vs. active" 여부를 판별하기 위해 **지수분포 기준선**을 설정한다. 만약 NOOP가 완전히 무작위적인 omission, 즉 각 step에서 독립적으로 NOOP 여부를 결정하는 **Poisson process**라면, NOOP bout의 지속 시간(bout length)은 **지수분포(Exponential distribution)**를 따라야 한다. 지수분포는 "memoryless" 성질을 가지며, 현재 이미 t step 동안 NOOP가 지속됐다고 해서 다음 step에도 NOOP를 유지할 확률이 높아지거나 낮아지지 않는다(항상 동일한 hazard rate λ). 따라서 **지수분포 = 랜덤 누락(random omission)의 귀무 가설(null hypothesis)**이며, 이로부터의 이탈이 "bout이 계획된 단위로 구성된다"는 증거가 된다.

#### 지수 모델 추정 방법

지수 기준선의 rate parameter **λ**는 관측된 bout length 데이터의 MLE(Maximum Likelihood Estimation)로 추정한다. 지수분포에서 MLE 추정치는 단순히 **λ̂ = 1 / mean_bout_length**이다. 이로부터 지수 생존함수 **S_exp(t) = exp(−λ̂ · t)**를 계산하여 경험적 KM 곡선과 같은 축에 점선으로 표시한다.
> Kiefer, N. M. (1988). "Economic Duration Data and Hazard Functions." Journal of Economic Literature, 26(2), 646–679. \
> Caballero, R. J., & Engel, E. M. R. A. (1999). "Explaining Investment Dynamics in U.S. Manufacturing: A Generalized (S,s) Approach." Econometrica, 67(4), 783–826.

#### Kaplan-Meier 생존 그래프 작성 방법

Kaplan-Meier(KM) 추정법은 bout length 데이터로부터 비모수적(non-parametric)으로 생존함수 S(t) = P(bout length > t)를 추정한다. 각 bout 종료 시점 t_i에서 "위험에 노출된 bout 수 n_i" 대비 "종료된 bout 수 d_i"를 이용하여 S(t)를 step 함수로 갱신한다: **S(t) = ∏_{t_i ≤ t} (1 − d_i / n_i)**. 에피소드 종료로 인해 bout이 강제 중단되는 경우(censoring)는 해당 시점에서 위험군에서만 제외되고 이벤트로는 계산되지 않는다(censoring rate 0.1~0.6%로 사실상 무시 가능). 각 피험자 × 게임 조합에 대해 별도의 subplot을 생성하고, KM 실선 위에 지수 기준선 점선을 겹쳐 표시한다.

#### 확보된 결과 (N=6, human data only)

**Fig 1-5 — KM survival curve per subject × game:**
- 6 subjects × 2 games, 각 피험자별 subplot, 지수분포 기준선(점선) 비교
- Censoring rate: 0.1~0.6% (에피소드 종료로 인한 강제 중단 — 사실상 무시 가능)
- 두 게임 모두 KM 곡선이 지수 baseline과 명확히 다른 형태 → **random omission 귀무가설 기각**
- 피험자 간 bout length 분포 편차 존재 (Pong mean: 31~72 steps) → individual strategy 반영

**교차 지점 수치 (survival_analysis.py 실행 결과, 2025-04-23):**

| Sub | Game | N bouts | Mean | Cross1 (↓, 위→아래) | Cross2 (↑, 아래→위) | Short% (<Cross2) | Long% (≥Cross2) |
|-----|------|---------|------|---------------------|---------------------|------------------|-----------------|
| 1 | Pong | 2,161 | 57.4 | 13 | 80 | 75.1% | 24.9% |
| 1 | SpaceInvaders | 11,519 | 9.4 | 3 | 27 | 94.4% | 5.6% |
| 2 | Pong | 1,811 | 55.1 | 9 | 81 | 77.0% | 23.0% |
| 2 | SpaceInvaders | 9,784 | 14.7 | 5 | 34 | 89.8% | 10.2% |
| 3 | Pong | 1,564 | 71.8 | 6 | 109 | 81.0% | 19.0% |
| 3 | SpaceInvaders | 8,743 | 9.8 | 3 | 26 | 92.6% | 7.4% |
| 4 | Pong | 3,209 | 31.5 | 2 | 58 | 84.0% | 16.0% |
| 4 | SpaceInvaders | 12,841 | 6.5 | 2 | 17 | 92.4% | 7.6% |
| 5 | Pong | 2,467 | 42.8 | 8 | 61 | 75.7% | 24.3% |
| 5 | SpaceInvaders | 5,510 | 19.2 | 6 | 55 | 94.3% | 5.7% |
| 6 | Pong | 1,885 | 61.5 | 9 | 85 | 75.0% | 25.0% |
| 6 | SpaceInvaders | 11,885 | 6.5 | 2 | 18 | 93.3% | 6.7% |

- **Cross1 범위**: Pong 2~13 steps, SpaceInvaders 2~6 steps (시각적으로 매우 짧아 그림에서 눈에 띄지 않을 수 있음)
- **Cross2 범위**: Pong 58~109 steps, SpaceInvaders 17~55 steps → **heavy tail 시작점, bout 유형 분리 기준선으로 활용 가능**
- Long% (≥Cross2): Pong 16~25%, SpaceInvaders 5~10% → 소수의 bout이 heavy tail 담당

#### "위 → 아래 → 위" 3구간 교차 패턴의 해석

실제 KM 곡선은 **3구간 패턴**을 보인다: 처음에 지수 기준선 위 → Cross1에서 아래로 교차 → Cross2에서 다시 위로 교차. 시각적으로 Cross1(매우 짧은 t)이 잘 보이지 않아 "처음에 아래로 내려갔다가 위로 올라온다"처럼 보이지만, 실제로는 세 구간이 존재한다. 이 패턴은 단순히 "지수분포에서 벗어났다"는 것 이상의 구체적인 메커니즘을 시사한다.

- **구간 1: t < Cross1 — KM > 지수 기준선 (매우 짧은 bout 구간)**:  
  KM이 지수 기준선보다 높다는 것은, 극단적으로 짧은 (1~2 step) bout이 지수 예측보다 *더 적게* 관찰된다는 뜻이다. 즉, 한 번 NOOP가 시작되면 1~2 step에서 즉시 종료되는 경우가 순수 랜덤보다 적다 — bout이 일단 시작되면 최소한의 지속 구조를 가진다는 뜻이다.

- **구간 2: Cross1 < t < Cross2 — KM < 지수 기준선 (중간 bout 구간)**:  
  중간 길이 bout이 지수 예측보다 *더 많이* 종료된다. 이는 2-2에서 확인된 **"Short bouts" (high entropy, crisis-response deliberation)** 유형과 대응된다 — 높은 불확실성 상황에서 빠르게 반응하는 짧은 보류. 이 구간이 전체 bout의 75~94%를 차지한다.

- **구간 3: t ≥ Cross2 — KM > 지수 기준선 (heavy tail 구간)**:  
  KM 곡선이 다시 지수 기준선 위로 올라온다는 것은, 긴 bout이 지수 예측보다 *더 많이* 살아남는다는 뜻이다 — **heavy tail**. Cross2에 도달한 bout은 이후 더 오랫동안 지속될 확률이 지수 예측보다 높다 (increasing survival in hazard terms). 이는 2-2의 **"Long bouts" (low entropy, strategic patient planning)** 유형과 대응된다 — 안정적 상태에서 지속되는 계획적 보류. 전체 bout의 5~25%가 이 구간에 해당한다.

- **교차 패턴 전체의 해석**:  
  두 가지 질적으로 다른 NOOP bout 유형(단기 crisis-response / 장기 strategic planning)이 혼재하는 **혼합 분포(mixture distribution)** 구조를 반영한다. 지수 단일 모델은 이 두 유형을 평균하여 중간에 위치하지만, 실제 데이터는 양 극단(매우 짧거나 매우 긴 bout)에 더 많은 관측값이 몰려 있다. 이는 NOOP가 Poisson 과정처럼 각 step에서 독립적으로 끝날 확률이 일정한 것이 아니라, **bout이 시작되면 그 내부에 지속 구조가 형성됨**을 의미한다. 특히 bout이 일정 시간 지속되면 더 오래 지속될 가능성이 증가한다는 것은, NOOP가 수동적 버튼 미누름이 아니라 **인지적으로 유지되는 active postponement 상태**임을 시사한다. 6명의 피험자 전원에서 이 패턴이 반복된다는 사실은, 이것이 개인 특성이 아닌 **인간 planning의 구조적 특성**임을 강하게 지지한다.

#### 의미 및 가설 연결
1-3에서 텍스트로만 서술한 "heavy-tail → planned delay" 주장을 **피험자별 시각적 증거**로 뒷받침한다. 지수분포에서의 이탈, 특히 초반 단기 초과와 후반 heavy tail이 공존하는 교차 패턴은, 매 step의 NOOP가 독립적으로 결정되지 않고 bout 내부에 시간적 구조가 존재함을 의미한다. 이는 C1(현상의 체계성)과 C2(structured postponement)를 연결하는 bridge 역할을 하며, Section 2-2의 Short/Long bout 이원성과 직접 연결된다.

---

## 2. Bout Structure Analysis: NOOP의 시간적 구조와 내적 조직

*리뷰어 관점: Planning의 행동적 signature는 단순한 비율이 아니라 시간적 구조에서 나타난다. 얼마나 긴 bout? 어떻게 시작하고 끝나는가? 이 구조가 computational demand를 반영하는가?*

> **섹션 범위 정의**: Section 2는 NOOP bout의 *형태(shape)*를 기술한다 — 길이 분포, onset·commit 주변의 시간적 궤적, bout 내부의 전환 구조. Uncertainty 값은 bout 형태를 묘사하는 도구로 쓰이며, uncertainty가 NOOP를 *유발하는가*라는 인과 질문(→ Section 3)이나 withholding이 실제로 *도움이 되는가*라는 기능 질문(→ Section 4)은 다루지 않는다.

### 2-1. Bout length distribution

#### 핵심 질문
- NOOP bout의 길이 분포는 무엇인가? (exponential vs. heavy-tailed → passive vs. active)
- State-level computational demand (uncertainty, Q-gap)가 bout 길이를 예측하는가?

#### 필요한 세부 연구
- **Bout length histogram + survival function** (Kaplan-Meier 스타일): exponential이면 random omission, heavy-tail이면 planned delay 시사 **[V → 이미 Figure 2D에 일부 구현됨]**
- **Bout length ~ pre-uncertainty regression**: uncertainty가 높을수록 더 긴 bout인가? **[NEW]**
- **Human vs. thinker bout length 비교**: Human이 더 variable/longer bout을 보이는가? **[NEW]**

> **Note**: "긴 bout 후 action quality가 높은가"라는 질문은 bout 형태의 *기능적 결과*이므로 Section 4-2(dose-response)에서 다룬다.

---

### 2-2. Temporal profile around bout onset and commit

> **Framing**: 여기서 묻는 것은 "uncertainty의 시간적 모양이 어떻게 생겼는가"이다 — onset 전후, commit 전후에 entropy·margin이 어떤 궤적을 그리는지 기술(describe)한다. 이 패턴이 uncertainty-driven commitment를 *지지하는가*라는 해석은 Section 3-3에서 인과 검증과 함께 다룬다.

#### 확보된 결과 (sub001 ses04, N=2,196 bouts)

**Commit-aligned (baseline 정규화: steps −6~−4 평균 기준):**
| rel_step | Δentropy_actor | 유의성 |
|----------|---------------|--------|
| −6 ~ −4 | ≈ 0 (baseline) | ns |
| −3 | +0.026 | *** |
| −2 | +0.045 | *** |
| −1 | +0.071 | *** |
| **0 (commit)** | **+0.073 (peak)** | *** |
| **+1** | **−0.081 (undershoot)** | *** |
| +6 | −0.030 | *** |

- Rise (−6→0): slope=+0.0147/step, t=17.89, p<0.001 ✅
- Drop (0→+1): Δ=−0.154, t=28.73, p<0.001 ✅
- margin_actor: 완벽한 거울 패턴 (slope=−0.0146, Drop Δ=+0.121, 모두 p<0.001) ✅
- **Post-commit undershoot**: +1 이후 entropy가 pre-bout baseline 아래로 유지됨 → commit action이 game을 더 낮은 uncertainty 상태로 이동시킨다는 functional benefit의 간접 증거

**Onset-aligned — "Onset Paradox" 발견:**
- Pre-onset (−6→−1): entropy가 **하강** 추세 (slope=−0.0064/step, p=0.019)
- onset(0) step: entropy **최저** (0.899, pre-onset −1의 0.904보다도 낮음; 차이 ns)
- +1 (두 번째 NOOP): **+0.123 급등**, t=24.66, p<0.001 → 72.45%의 bout에서 확인
- **결론**: NOOP onset은 entropy spike에 의해 트리거되지 않는다. 인간은 entropy가 낮아지는 순간에 withholding을 시작하고, bout 내부에서 entropy가 축적된다.

**Bout 길이별 분리 — 두 가지 withholding 유형 (임의 기준선):**
| 유형 | pre-bout entropy | commit entropy | post-commit entropy |
|------|-----------------|---------------|---------------------|
| Short (1–3 steps) | 0.983 | **1.051** | 0.915 |
| Medium (4–10) | 0.913 | **1.022** | 0.871 |
| Long (>10 steps) | **0.850** | **0.966** | **0.772** |

- 긴 bout일수록 전반적으로 낮은 entropy context → "많이 불확실할수록 더 오래 기다린다"는 단순 가설과 **반대**

---

**Cross2 기반 원리적 구분 (생존 분석 교차점 활용, sub001 SpaceInvaders Cross2 = 27 steps):**

위의 1–3/4–10/>10 구분은 임의적이다. Section 1-5의 KM 생존 분석에서 도출된 **Cross2 (KM 곡선이 지수 기준선을 다시 위로 교차하는 지점)**는 데이터 기반의 원리적 분리 기준을 제공한다: Cross2 미만은 지수 예측보다 빨리 종료되는 "common/medium" bout, Cross2 이상은 지수 예측보다 오래 살아남는 "heavy-tail strategic" bout.

**sub001 SpaceInvaders (Cross2=27, N=2,196):**
| 유형 | N (%) | pre entropy | onset entropy | commit entropy | Δ entropy | commit margin | Q-gap at onset |
|------|-------|-------------|--------------|----------------|-----------|---------------|----------------|
| Short (<27 steps) | 2094 (95.4%) | 0.905 | 0.900 | **1.027** | **+0.122** | 0.391 | 1.399 |
| Long (≥27 steps) | 102 (4.6%) | 0.877 | 0.869 | **0.920** | **+0.043** | **0.549** | **0.863** |
| Δ (Long−Short) | — | −0.028 (ns) | −0.032 (ns) | **−0.108 (\*\*\*)** | **−0.080 (\*)** | **+0.158 (\*\*\*)** | **−0.536 (\*\*\*)** |

*통계: Welch t-test. commit entropy p=1.9e-7 ***, Δentropy p=0.024 *, commit margin p=5.3e-17 ***, Q-gap onset p=1.4e-4 ***.*

**해석 (Cross2 기준):**
- **Long bouts (≥27 steps, 4.6%)**:
  - **Commit entropy가 유의하게 낮다** (0.920 vs 1.027, ***): 짧은 bout에 비해 행동 결정 시점의 불확실성이 낮음. 긴 기다림 끝에 더 확신에 찬 행동으로 이어짐.
  - **Commit margin이 유의하게 높다** (0.549 vs 0.391, ***): 선택된 행동과 차선책 간의 차이가 더 크다 → 결정이 더 명확하고 단호함.
  - **Onset 시 Q-gap이 유의하게 낮다** (0.863 vs 1.399, ***): NOOP을 시작하는 시점에 행동 가치 간 차이가 작다 → 행동 선택이 덜 긴박한 중립적 상태에서 긴 보류가 시작됨.
  - **Δ entropy가 작다** (+0.043 vs +0.122, *): 짧은 bout에서는 bout 진행 중 entropy가 크게 증가하지만, 긴 bout에서는 entropy가 상대적으로 안정적으로 유지된다.
- **Short bouts (<27 steps, 95.4%)**:
  - Commit 시 entropy가 높고 (1.027), bout 중 entropy가 급증 (+0.122), commit margin이 낮다 (0.391).
  - 진입 시 Q-gap이 크다 (1.399) → 행동 가치 대비가 뚜렷한 상황에서 시작되는 짧은 반응적 보류.

**결론**: Cross2 기반 구분은 임의 기준선(>10)보다 두 유형의 질적 차이를 더 선명하게 분리한다 (commit margin: 0.496 vs 0.549). 두 가지 질적으로 다른 withholding 행동 유형:
  - **Short bouts (crisis-response deliberation)**: 높은 entropy·Q-gap 맥락에서 시작 → bout 중 entropy 급증 → 비교적 낮은 confidence로 commit. 즉각적인 반응이 필요한 상황에서의 단기 보류.
  - **Long bouts (strategic patient planning)**: 중립적 Q-gap 맥락에서 시작 → entropy 안정 유지 → 높은 confidence로 단호하게 commit. 적절한 시점을 기다리는 계획적 보류.

이 이분법은 Section 1-5의 survival curve 교차 패턴(Cross2 이상에서 heavy tail)과 직접 대응되며, 두 유형이 하나의 Poisson 과정이 아닌 **혼합 메커니즘(mixture mechanism)**임을 행동 지표 수준에서 확인한다.

#### 핵심 질문 (업데이트)
- ~~NOOP onset 직전에 uncertainty가 상승하는가?~~ → **Onset paradox로 기각**: onset 자체는 entropy spike가 아님. "무엇이 onset을 트리거하는가?"를 새로운 질문으로 재정식화
- Commit 직전에 action confidence가 올라가는가? → **확인됨** (rise-and-fall 패턴 ✅)
- Post-commit undershoot는 functional benefit을 반영하는가? → Section 4와 연결

#### 필요한 세부 연구
- Real step 기준으로 onset/commit aligned temporal profile 추출 **[✅ Done → Figure 2B/2C]**
- Rise-and-fall formal test (linear trend + paired t) **[✅ Done, 2196 bouts]**
- **Onset trigger 메커니즘 규명**: entropy가 아니라면 무엇이 onset을 결정하는가? → game state (e.g., between-wave pause in SI), margin 절대값, 직전 action의 Q-value 등 **[NEW, 우선순위 높음]**
- **Bout-internal entropy trajectory**: onset(+0) → commit(0) 사이에서 entropy가 단조 증가하는가, 아니면 일정 시점에서 saturate하는가? (긴 bout에서 중간 구간 검증) **[NEW → 2-3과 연결]**
- **Short vs. Long bout의 commit-aligned profile 비교 figure**: 두 유형의 질적 차이를 시각화 **[NEW]**

---

### 2-3. Sequential dependency and transitional structure *(신규)*

*리뷰어 관점: Bout 내부 구조를 이해하면 active search vs. passive waiting을 구별하는 데 도움이 된다.*

> **Framing**: bout 내부에서 uncertainty가 어떻게 변하는지를 *묘사*한다. "단조감소 패턴 = active search 중"이라는 해석적 주장은 Section 3-3(인과 ordering, bout onset 이전 window)과 연결되며, 두 분석은 서로 다른 시간 창(bout 내부 vs. onset 이전)에서 같은 mechanism을 보완적으로 지지한다.

#### 핵심 질문
- NOOP → NOOP 전환 확률 vs. Action → NOOP 전환 확률: 두 상태 간의 전환이 Markovian인가?
- Bout 내에서 시간이 지남에 따라 uncertainty가 감소하는가?
- 같은 bout 내에서 thinker가 내부적으로 action preference를 변경하는지 (cur_action의 변화)

#### 필요한 세부 연구
- **First-order transition matrix** (NOOP→NOOP, NOOP→act, act→NOOP, act→act) 계산 **[NEW]**
- **Bout-internal uncertainty trajectory**: bout 시작 → 끝 방향으로 entropy가 단조감소하는가? **[NEW]**
- **Cur_action stability within bouts**: bout 내에서 thinker가 선호하는 action이 바뀌는 비율 **[NEW]**

---

### 2-4. Within-session and across-session adaptation *(신규)*

*리뷰어 관점: Strategic postponement라면 task structure를 학습함에 따라 변해야 한다. Learning curve가 없다면 단순 habit이다.*

#### 핵심 질문
- Session 초반과 후반 사이에 NOOP 비율이 어떻게 변하는가?
- 게임을 더 잘하게 될수록 (score 상승) NOOP 비율은 어떻게 변하는가?
- Session 내에서 uncertainty-NOOP coupling이 강화되는가? (즉, strategic use가 정교해지는가)

#### 필요한 세부 연구
- **Episode-level NOOP ratio와 cumulative score의 상관** (sub-session learning curve) **[NEW]**
- **Early vs. late episode half 비교**: uncertainty-NOOP 관계가 later half에서 더 강한가? **[NEW]**
- **Cross-session comparison** (ses-01 → ses-04): NOOP 전략의 진화 추적 **[NEW]**

---

## 3. Computational Interpretation: NOOP는 uncertainty-sensitive delayed commitment의 proxy인가?

*리뷰어 관점: Correlation은 causal claim이 아니다. "Uncertainty → NOOP"라는 방향성을 어떻게 확인할 것인가? 또한 어떤 uncertainty 지표가 가장 강력한 예측변인인가?*

> **섹션 범위 정의**: Section 2가 bout의 형태를 기술했다면, Section 3는 그 형태를 *일으키는* 계산 변수가 무엇인지를 묻는다. 분석의 단위는 timestep이며, uncertainty metric → NOOP 발생 확률의 인과 방향을 검증한다. Section 2의 temporal profile이 보여준 rise-and-fall 패턴이 실제로 uncertainty에 의해 구동되는지가 이 섹션의 핵심 질문이다.
>
> **1-4와의 관계**: 1-4의 AR 잔차 분석은 perseveration을 통제한 뒤 uncertainty가 NOOP를 예측하는지를 묻는 분석이며, 3-1·3-2의 사전 조건(confound 제거) 역할을 한다. 결과가 수렴하면 두 섹션에서 모두 인용한다.

### 3-1. Uncertainty-NOOP coupling

#### 핵심 질문
- Human의 NOOP은 단순히 "가능한 action 중 하나를 자주 누른다" 수준인가?
- 아니면 특정 state에서 선택적으로 증가하는 **structured postponement signal**인가?
- Policy uncertainty가 커질수록 NOOP이 증가하는가?

#### 필요한 세부 연구
- Real step별 policy uncertainty와 NOOP probability의 관계 **[]**
- Human vs. pretrained thinker vs. imitation thinker에서 uncertainty-NOOP coupling 비교 **[]**
- 다양한 uncertainty metric과 NOOP의 연관성:
    - **State-side**: world-model uncertainty, rollout disagreement, latent-state novelty, predicted future variance, branch entropy, tree width
    - **Action-side**: chosen-action probability, top-1 minus top-2 policy gap, best-vs-second-best Q gap, commit-step action margin, action entropy의 inverse

---

### 3-2. Formal model comparison *(신규)*

*리뷰어 관점: 어떤 계산 변수가 NOOP를 유발하는가에 대한 systematic model comparison이 없으면, "어느 것이 진짜 원인인가"라는 질문에 답하기 어렵다.*

#### 핵심 질문
- Policy entropy, Q-gap, rollout disagreement, search_disagreement (JSD) 중 어느 변수가 NOOP를 가장 잘 예측하는가?
- 이 변수들 간의 상대적 기여를 quantify할 수 있는가?
- Thinker와 human에서 가장 중요한 예측변인이 다른가?

#### 필요한 세부 연구
- **Mixed-effects logistic regression**: NOOP ~ entropy + q_gap + search_disagreement + rollout_spread + (1|subject) + (1|game), AIC/BIC로 모델 비교 **[NEW]**
- **Standardized beta coefficient 비교**: 어느 uncertainty metric이 가장 강한 독립 예측력을 갖는가? **[NEW]**
- **ROC/AUC analysis**: 각 모델의 NOOP 예측 정확도 비교 **[NEW]**
- **Human과 thinker에서 회귀계수 비교**: species × uncertainty interaction이 있는가? **[NEW]**

---

### 3-3. Causal ordering: What triggers onset, and what drives commitment?

*리뷰어 관점: Cross-sectional correlation은 인과성을 보장하지 않는다. 가능하면 temporal precedence를 보여야 한다.*

> **Onset paradox와의 관계 (2-2에서 이동)**: 2-2 분석에서 onset이 entropy spike에 의해 트리거되지 **않는다**는 것이 확인됐다. Pre-onset entropy는 오히려 하강하고, bout 내부에서 entropy가 축적된다. 따라서 3-3의 인과 질문은 두 층위로 분리되어야 한다:
> 1. **Onset trigger**: 무엇이 withholding을 시작시키는가? (entropy가 아니라면 game context, margin 절대값, 직전 Q-value 등)
> 2. **Commitment trigger**: 무엇이 withholding을 끝내고 action을 실행시키는가? (commit 직전 entropy peak + 직후 급락은 확인됨)
>
> **2-3과의 관계**: 2-3은 bout *내부*에서 entropy가 어떻게 변하는지를 기술한다. 3-3은 onset *이전*과 commit *직전*에서 인과 신호를 검증한다.
>
> **1-4와의 관계**: 1-4에서 확인된 base-rate 초과 lag-1 AC(SpaceInvaders: 실제 0.81 vs 예측 0.51)는 perseveration 대안 설명의 가능성을 열어 두었다. AR-residual test (아래)가 이를 formal하게 검증하여 perseveration과 uncertainty-driven delay를 해리한다.

#### 필요한 세부 연구
- **Onset trigger 후보 분석**: game event markers (e.g., enemy position, between-wave pause), margin 절대값, 직전 k-step reward — 어떤 변수가 onset을 예측하는가? **[NEW, 우선순위 높음]**
- **Commitment trigger 확인 (commit-aligned 패턴의 인과 검증)**: entropy peak at commit → commit은 "entropy가 최고점에서 어떤 threshold를 넘었을 때" 발생하는가, 아니면 "bout duration에 의해 강제되는가?" → bout-length × entropy-at-commit interaction 검증 **[NEW]**
- **AR(1)~AR(5) residual uncertainty test**: lag-k NOOP를 먼저 회귀 제거한 잔차에서 uncertainty(entropy_actor, q_gap)가 NOOP를 독립적으로 예측하는지 확인. k=1~5까지 beta 안정성 추적 → perseveration 통제 후에도 uncertainty 기여가 유지되는가 **[NEW — 1-4에서 이동, thinker 행동 데이터 완비 후 수행]**
- **Granger causality test** 또는 **lagged regression**: t-1 entropy → t NOOP probability (temporal precedence 확인). ⚠️ onset paradox로 인해 onset 시점에서는 유의한 효과가 없을 가능성 — commit 시점에 집중 **[NEW]**
- **Bout onset entropy → bout length**: onset entropy가 낮을수록 bout이 길어지는가? (short vs. long bout의 onset entropy 차이 검증, 2-2 결과와 연결) **[NEW]**

---

## 4. Normative Function: Delayed commitment는 실제로 planning에 이득을 주는가?

*리뷰어 관점: Correlation (withholding precedes better actions)은 selection bias에 취약하다. 더 좋은 state에서 withhold하기 때문에 좋은 결과가 나올 수 있다. Matched control design이 핵심이다.*

> **섹션 범위 정의**: Section 3가 uncertainty가 NOOP를 *유발하는가*를 물었다면, Section 4는 그 withholding이 실제로 *도움이 되는가*를 묻는다. 분석의 단위는 commit action 이후의 outcome(VRE, k-step reward, 세션 점수)이며, withholding의 기능적 이득을 검증한다. Bout 길이와 outcome의 dose-response 관계도 여기서 다룬다 — 이것은 bout의 *형태*(Section 2)를 기술하는 것이 아니라 bout 길이가 *결과*에 미치는 영향이기 때문이다.

### 4-1. Value Revision Error (VRE)와 k-step reward

#### 현재 확보된 지표
- **Value Revision Error (VRE)**: action 선택 시점의 Q와 이후 업데이트된 Q의 차이 → "이 action을 너무 일찍 확신했는가?"
- **k-step reward**: action 이후 k-step 누적 보상합 → "결과적으로 좋았는가?"

#### 핵심 질문
- NOOP 직후에 취한 action은 더 낮은 VRE를 가지는가?
- NOOP를 거친 선택은 더 높은 k-step reward를 가지는가?
- Uncertainty가 높은 state일수록 NOOP 후 benefit이 더 큰가?

#### 필요한 세부 연구
- NOOP preceding vs. non-NOOP preceding action의 VRE 비교 **[V → Figure 4A]**
- 동일한 uncertainty bin 안에서 NOOP 이후 action의 k-step reward 비교 **[V → Figure 4C]**
- Imitation thinker의 internal tree statistics와 VRE의 관계 **[]**
- "Search conflict가 컸던 state일수록 NOOP 이후 benefit이 큰가" 분석 **[]**

---

### 4-2. Matched-control analysis 강화 *(신규)*

*리뷰어 관점: 현재의 matched-control은 entropy-matching만 한다. 더 엄격한 matching이 필요하다.*

#### 핵심 질문
- Entropy뿐 아니라 episode position, game state value, recent reward history를 동시에 matching했을 때도 benefit이 유지되는가?
- Benefit의 effect size는 충분히 크고 일관적인가?
- 더 길게 기다릴수록 더 좋은 결과가 나오는가? (dose-response)

#### 필요한 세부 연구
- **Propensity score matching** (withholding 여부를 outcome, uncertainty + episode position + value estimate를 covariates로): matched pair에서 action quality 비교 **[NEW]**
- **Dose-response relationship**: bout length ~ subsequent k-step reward / VRE reduction → 더 길게 기다릴수록 더 좋아지는가? *(Section 2-1에서 이동)* **[NEW]**
- **Interaction: uncertainty × withholding on outcome**: uncertainty가 높은 state에서만 benefit이 명확한가? **[NEW]**

---

### 4-3. Session-level performance benefit *(신규)*

*리뷰어 관점: Trial-level 효과뿐 아니라 session-level 효과가 있어야 "planning이 실제로 기능한다"는 주장이 강해진다.*

#### 핵심 질문
- 세션 수준에서 NOOP 비율이 높은 세션이 더 높은 점수를 달성하는가?
- 피험자 수준에서 "더 selective한 withholder"가 더 좋은 performance를 보이는가?

#### 필요한 세부 연구
- **Session-level regression**: total score ~ mean NOOP ratio + mean pre-NOOP uncertainty (피험자 random effect 포함) **[NEW]**
- **Selectivity index**: total NOOP count 대비 uncertainty-triggered NOOP 비율이 높은 피험자가 더 잘 하는가? **[NEW]**

---

### 4-4. NOOP ablation (선택적이나 강력한 증거) *(기존 언급 → 구체화)*

반드시 필요하지는 않지만, **있으면 매우 강해진다.**

#### 가능한 접근
- **Counterfactual simulation**: imitation thinker를 이용하여, human의 NOOP step에서 만약 immediately action을 취했다면 어떤 결과가 나왔을지 simulate → 실제 결과와 비교
- **NOOP-masked replay**: human trajectory에서 NOOP를 제거하고 thinker가 재연할 때의 성능 비교 **[NEW]**

---

## 5. Neural Mechanism: Delayed commitment와 planning을 매개하는 뇌 회로

*리뷰어 관점: fMRI claim은 behavioral claim에 의존한다. Section 1-3이 충분히 확립되어야 이 section이 설득력을 갖는다. Region-of-interest에 대한 a priori hypothesis가 명확해야 한다.*

### 5-1. Commitment gating (striatum / frontal)
- **Regressor**: NOOP 여부, commit 전 delay length, bout length
- **후보 영역**: caudate/putamen, supplementary motor area (SMA), anterior cingulate cortex (ACC), pre-SMA
- **예측**: Striatum은 action commitment 시점에서 phasic response, ACC는 conflict 기간 동안 sustained signal

### 5-2. Uncertainty-linked planning (hippocampus / PFC)
- **Regressor**: policy uncertainty (entropy), search_disagreement (JSD), VRE
- **후보 영역**: hippocampus, vmPFC (value), dlPFC (cognitive control), OFC
- **예측**: Hippocampus는 NOOP bout 동안 uncertainty-proportional activation → prospective search representation

### 5-3. Behavior-to-brain bridge
- Trial/step-level NOOP probability → brain activation
- Trial/step-level VRE → brain activation
- Trial/step-level k-step reward expectancy → brain activation
- **Dissociation test**: planning content (hippocampus) vs. gating signal (striatum/frontal) → 이 두 신호가 분리되는가?

### 5-4. Multivariate/decoding analysis *(신규)*
- **RSA (Representational Similarity Analysis)**: thinker의 latent state geometry와 brain representation의 유사도 비교
- **MVPA**: NOOP vs. action commitment의 multivariate classifier → spatial pattern으로 구별 가능한가?

---

## 6. Representational Mechanism: World model과 tree search representation의 geometry

*리뷰어 관점: Section 5의 RSA claim이 있다면, thinker latent geometry에 대한 사전 분석이 필요하다.*

### 핵심 질문
- Thinker의 latent state representation은 task-state geometry를 반영하는가?
- Tree search의 구조를 정량화할 수 있는 motif가 존재하는가?
- 이러한 geometry가 human planning style과 연결되는가?

### 세부 연구 (기존)
- **State embedding geometry** (PCA 후 neighbor structure): branching point 근처에서 state space가 더 복잡한가?
- **Tree search motif**: 실제로 선택된 action과 rejected action 사이의 Q-gap이 NOOP duration을 예측하는가?
- **RSA matrix 구성**: thinker latent state × state → brain region representation matrix와 비교

---

### 6-1. Spectral geometry of tree_reps: Diffusion Maps 접근

> **이론적 배경** (Coifman & Lafon, 2006): 데이터 포인트 간의 local affinity(kernel)로 Markov chain을 구성하면, 그 eigenvector들이 데이터의 "diffusion coordinates"(성격)를 정의한다. Eigenvalue λ_l은 각 기하 구조의 scale별 지속성을 나타낸다. 핵심 논리: **위치(local similarity) → 구조(Markov transition) → 성격(eigenvectors)**. 이것을 tree_reps에 적용한다.

#### 핵심 질문
- 하나의 planning step에서 생성된 tree의 node representation들은 어떤 spectral geometry를 갖는가?
- NOOP bout 동안 tree geometry가 action commit 시점과 구조적으로 다른가?
- Spectral gap (λ_2/λ_1)이 action-value conflict를 반영하는가?

#### 세부 분석
- **Diffusion Map on tree nodes**: 단일 real step의 tree_reps 전체 node에 kernel k(x,y) = exp(-||x-y||²/ε)를 적용 → Markov matrix P → eigendecomposition → diffusion coordinates 시각화 **[NEW]**
- **Spectral gap 분석**: λ_2/λ_1 비율 (두 번째 vs 첫 번째 eigenvalue). Gap이 작으면 두 개의 거의 동등한 future trajectory 클러스터 존재 → action-value conflict 지표. NOOP bout 동안 gap이 action step보다 작은가? **[NEW]**
- **Multiscale cluster structure**: P^t를 t=8, 64, 1024에서 비교 → short-horizon(세부 분기)과 long-horizon(거친 전략) search 구조 분리 **[NEW]**
- **Diffusion distance D_t(real_node, imagined_nodes)**: root(real state)에서 imagined future states까지의 planning distance 분포. NOOP 직전에 이 거리가 더 넓게 퍼지는가? **[NEW]**
- **RSA matrix 구성**: diffusion coordinate 기반 pairwise distance matrix → brain region RSA와 비교 (Section 5-4 연결) **[기존 RSA 항목 구체화]**

#### Section 3 연결
Spectral gap은 Section 3-1의 "state-side uncertainty" 지표 목록(branch entropy, tree width)을 대체·보완하는 새로운 지표로, Section 3-2 formal model comparison에 포함 가능.

---

### 6-2. Stochastic evolution of tree_reps: Neural SDE (scDiffEq) 접근

> **이론적 배경** (Vinyard et al., 2025, Nature MI): single-cell 분화 궤적을 drift(결정론적)와 diffusion(확률적, 상태 의존적)으로 분해하는 neural SDE 프레임워크. 핵심 발견: **다운스트림 fate 결정점(multipotent progenitor)에서 drift와 diffusion magnitude 모두 최대**. Planning의 결정점 = 세포 분화의 분기점 analogy.

#### 핵심 가설
NOOP bout 진입 = multipotent state (여러 action 방향으로 분기 가능 → diffusion ↑)
Action commit = fate commitment (한 direction으로 수렴 → drift dominant)

#### 세부 분석
- **Tree centroid trajectory**: real steps 시퀀스에 걸쳐 tree_reps의 weighted mean (또는 top-k node mean)을 추출, temporal evolution 모델링 **[NEW]**
- **Drift-diffusion 분리**: z_{t+Δt} = z_t + f(z_t)·Δt + g(z_t)·noise 형태로 tree centroid evolution fit. f = drift network (결정론적 planning 방향), g = diffusion network (탐색의 stochasticity) **[NEW]**
- **g(z_t) proxy (저비용 근사)**: neural SDE 없이도, 각 real step에서 imagined nodes 간의 pairwise distance 평균을 diffusion magnitude의 proxy로 사용. NOOP steps vs action steps 비교 **[NEW, 우선순위 높음]**
- **Dose-response**: g(z_t)가 높은 step에서 NOOP probability와 bout length가 더 큰가? (Section 4-2 dose-response와 연결) **[NEW]**
- **Commit 직전 drift dominance 검증**: Section 2-2의 "uncertainty rise-and-fall"을 f/g ratio로 quantify → commit 직전 f(z) ↑, g(z) ↓ 패턴 **[NEW]**

---

### 6-3. Intrinsic vs. input-driven planning dynamics: InputDSA 접근

> **이론적 배경** (Huang, Ostrow et al., 2025): DSA를 non-autonomous system으로 확장. x_{t+1} = Ax_t + Bu_t 에서 A(intrinsic dynamics)와 B(input-to-state mapping)를 DMDc/SubspaceDMDc로 추정. **핵심 발견 (쥐 뇌 데이터)**: evidence accumulation 구간에서 input-driven dynamics dominant → decision-making 구간에서 intrinsic dynamics dominant로 전환. 이 전환이 NOOP bout과 정확히 대응한다는 가설.

#### 핵심 가설
- **NOOP bout** = intrinsic dynamics (A) dominant: 새로운 game observation이 아니라 내부 tree propagation이 state를 주도 → "외부를 기다리는 게 아니라 내부 검색을 계속"
- **Action step / commit 직전** = input-driven dynamics (B) dominant: real game state에 기반해 최종 결정

#### 세부 분석
- **DMDc fit**: tree_reps time series (real steps 기준)에 φ(x_{t+1}) = A·φ(x_t) + B·obs_t 형태로 DMDc 적용. A, B 추정 **[NEW]**
- **SubspaceDMDc**: tree_reps가 partially observed system임을 감안 (imagined nodes는 unobserved 상태 포함) → SubspaceDMDc로 A, B 추정의 정확도 개선 **[NEW]**
- **A vs B eigenvalue 스펙트럼 비교**: NOOP steps vs action steps에서 A의 eigenvalue magnitude (intrinsic) vs B의 singular value (input responsiveness) 비교 **[NEW]**
- **InputDSA_state vs InputDSA_input score**: NOOP 구간에서 state similarity (A) ↑, input similarity (B) ↓ 예측 검증 **[NEW]**
- **Human과 thinker 비교**: high-performing thinker와 human의 A/B decomposition이 유사한가? (InputDSA의 "Anna Karenina principle": 잘하는 agent들은 동적으로 유사) **[NEW, 가장 novel한 contribution]**
- **Section 3 연결**: A/B ratio가 Section 3-2 formal model comparison의 새로운 predictor로 포함. "intrinsic planning dominance index"가 NOOP probability를 예측하는 독립 변인인가? **[NEW]**

---

# 진행 상황 요약 (Status Tracker)

| Section | 분석 항목 | Figure | 상태 |
|---------|-----------|--------|------|
| 1-1 | Withholding bout schematic | **Fig 1-1A** (`fig_1-1_A_withholding_schematic.png`) | ✅ Done |
| 1-1 | Action distribution: human vs. **IL thinker** (bar chart), pretrained thinker 별도 대조군 | **Fig 1-1B** (`fig_1-1_B_action_distribution.png`) | ✅ Done |
| 1-2 | Subject-level NOOP proportion scatter (피험자×게임) + Cohen's d vs chance | **Fig 1-2A** (`fig_1-2_individual_differences.png` Panel A) | ✅ Done |
| 1-2 | Session reliability: ICC across days per subject × game (bar chart) | **Fig 1-2B** (`fig_1-2_individual_differences.png` Panel B) | ✅ Done |
| 1-2 | NOOP ratio ↔ performance scatter: null result 확인 (dual-axis, Pong/SI) | **Fig 1-2C** (`fig_1-2_individual_differences.png` Panel C) | ✅ Done |
| 1-2 | Episode-level reward: withholding-preceded vs. not (6 subjects × 2 games, 일관된 유의성) | **Fig 1-2D** (`fig_1-2_reward_subject_episode.png`) | ✅ Done |
| 1-2 | NOOP ratio ~ withholding benefit scatter (episode-level, null + benefit>0 consistent) | **Fig 1-2E** (`fig_1-2_noopratio_postnoop_reward.png`) | ✅ Done |
| 1-3 | Paired comparison: Pong vs SpaceInvaders NOOP proportion (paired lines) | **Fig 1-3A** (`fig_1-3_cross_game.png` Panel A) | ✅ Done |
| 1-3 | NOOP bout survival function per game (Kaplan-Meier style) | **Fig 1-3B** (`fig_1-3_cross_game.png` Panel B) | ✅ Done |
| 1-3 | Meta-analytic direction check: 피험자별 NOOP above chance (Pong vs SI scatter) | **Fig 1-3C** (`fig_1-3_cross_game.png` Panel C) | ✅ Done |
| 1-4 | Episode-position NOOP density: 20-bin 연속 + 3분할 shading (fatigue 배제) | **Fig 1-4A** (`fig_1-4_alternative_exclusion.png` Panel A) | ✅ Done |
| 1-4 | Episode 3분할(early/mid/late) 별 NOOP proportion bar chart + 통계 | **Fig 1-4B** (`fig_1-4_alternative_exclusion.png` Panel B) | ✅ Done |
| 1-4 | Lag-1 NOOP autocorrelation: boxplot + subject means per game (perseveration 확인) | **Fig 1-4C** (`fig_1-4_alternative_exclusion.png` Panel C) | ✅ Partial |
| 1-5 | NOOP bout survival curve: KM per subject × game + exponential baseline | **Fig 1-5** (`fig_1-5_survival_by_subject.png`) | ✅ Done |
| 1-1 | Uncertainty-conditional NOOP rate: human vs. IL thinker per entropy bin (selectivity 차이 정량화) | — | 🔲 New (Section 3 연결) |
| 2-1 | Residual distribution gap: human vs. IL thinker NOOP gap(+3.05pp) + JSD(0.005) — sub001 ses04 확인됨 | **Fig 1-1D** | ✅ Done (1-1에서 이동) |
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

---

# 7. [DRAFT] Bidirectional Alignment Analysis: Thinker-Human 표현 공간의 양방향 정렬

> **배경**: 기존 RSA (Section 5-4, 6)는 thinker latent geometry → brain representation의 단방향 비교다. 이 섹션은 반대 방향(인간 행동/뇌 신호 → thinker 차원 복원)을 추가하여 thinker의 어떤 차원이 실제로 인간과 공유된 계산 공간인지를 진단한다. 논문(Tian et al., forward/reverse predictivity framework)의 아이디어를 채용.

> **데이터**: `video_stat.npy` → `real_vectors` (status==0, ~3185 real steps, shape per step: (128,6,6)), `tree_reps_vector` (145066, 128), `im_vectors` (status==2, ~124176 imaginary steps). 행동 레이블(NOOP flag, entropy_actor, q_gap, vre_abs_q, k5_reward)은 기존 CSV와 real step index로 join.

---

## 7-1. Forward / Reverse Predictivity 및 비대칭성

#### 핵심 질문
- Thinker activation이 인간 행동(NOOP 여부, uncertainty)을 예측하는가? (forward)
- 인간 행동 신호가 thinker의 각 차원을 역으로 예측하는가? (reverse)
- Forward >> Reverse인 비대칭성이 존재하는가? (thinker의 extra dimension)
- 이 비대칭성이 NOOP real step vs action real step에서 다른가?

#### 분석 방법
- **Forward**: Ridge(X=thinker_128, y=human_noop + behavioral_signals) → R²_forward
- **Reverse**: 각 channel i에 대해 Ridge(X=behavioral_signals, y=thinker_dim_i) → R²_reverse_i 분포
- **Asymmetry**: R²_forward − mean(R²_reverse) → NOOP step vs action step 비교
- 데이터: real_vectors (status==0) global avg pool → (N_real, 128), 행동 레이블 join

---

## 7-2. Common / Unique Unit 식별 및 기능 검증

#### 핵심 질문
- 어떤 thinker 채널이 인간 행동으로부터 선형 복원 가능한가? (common)
- Common units는 NOOP onset에 반응하는가? Unique units와 temporal profile이 다른가?
- Common units가 이후 행동 결과(k5_reward, VRE)를 더 잘 예측하는가?

#### 분석 방법
- R²_reverse 기준 상위 20% → **Common units** (~26채널), 하위 20% → **Unique units**
- **NOOP onset-aligned temporal profile**: real step index 기준 ±k window에서 Common vs Unique 평균 activation 궤적 비교
- **행동 이득 예측력**: Common units activation → k5_reward, vre_abs_q 예측 R² vs Unique units
- **tree search 지표와의 관계**: Common units ~ root_qs_mean variance (action uncertainty); Unique units ~ cur_v 변화량 (imaginary trajectory 내 value update)

---

## 7-3. Imaginary Trajectory의 Effective Dimensionality × NOOP

#### 핵심 질문
- Real step 직전 ~39개의 imaginary step (planning window)에서 thinker representation이 얼마나 많은 차원을 탐색하는가?
- NOOP real step의 planning window가 action real step보다 effective dimensionality가 높은가?
- Bout 길이(withholding 지속)와 planning window의 effective dimensionality는 상관하는가?

#### 분석 방법
- 각 real step i에 대해 preceding status==2 im_vectors 추출 → global avg pool → (~39, 128)
- PCA eigenvalue spectrum → participation ratio: $(\sum \lambda_i)^2 / \sum \lambda_i^2$
- NOOP real step vs action real step의 participation ratio 분포 비교 (Mann-Whitney U)
- Participation ratio ~ bout length 상관 (Spearman r)

---

## 7-4. Common vs Unique 공간에서의 Planning Trajectory 방향성

#### 핵심 질문
- Imaginary trajectory가 Common unit 공간에서는 수렴(uncertainty 해소)하는가?
- Unique unit 공간에서는 다른 패턴(발산 또는 random drift)을 보이는가?
- NOOP bout 동안 두 공간의 trajectory 패턴이 다른가?

#### 분석 방법
- 7-2의 Common/Unique unit index 기반으로 im_vectors를 두 subspace로 분리
- 각 planning window에서 trajectory 수렴도: `||last - first||` in common space vs unique space
- NOOP real step vs action real step에서 수렴도 차이 비교
- **기대**: Common space → 수렴 (active search → commitment); Unique space → no clear convergence

---

## 7 분석 실행 순서 (Draft)

| 순서 | 분석 | 선행 조건 |
|---|---|---|
| 7-0 | tree_reps_vector PCA + NOOP 컬러링 (탐색) | CSV join |
| 7-1 | Forward/Reverse predictivity + asymmetry | CSV join |
| 7-2 | Common/Unique unit 식별 + temporal profile | 7-1 결과 |
| 7-3 | Effective dimensionality × NOOP | CSV join |
| 7-4 | Trajectory 방향성 (Common vs Unique space) | 7-2 결과 |

**공통 선행 조건**: `real_vectors[status==0]`의 real step index와 기존 CSV(is_human_noop, entropy_actor, q_gap, vre_abs_q, k5_reward)의 step index 매핑 확인.
