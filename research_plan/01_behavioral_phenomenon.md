# 1. Behavioral Phenomenon: 인간의 행동 보류는 체계적이고 선택적이다

*관점: "단순히 NOOP을 많이 누른다"는 것 이상을 보여야 한다. Passive omission과 active postponement를 어떻게 구별하는가? N=6이라는 소규모 표본에서 결론을 내릴 수 있는가?*

> **섹션 구조**: Section 1은 다섯 개의 층위에서 C1(현상의 체계성)을 인간 행동 데이터만으로 확립한다. 모델 데이터 없이 성립하는 주장이어야 하며, Section 2–4의 계산적·기능적 해석을 위한 토대가 된다.
>
> - **1-1**: 기본 현상 확립 (human vs. thinker NOOP gap)
> - **1-2**: 효과의 robustness와 "총량이 아닌 선택적 사용"이 핵심임을 확립
> - **1-3**: 두 게임 간 일반화 (game-specific artifact 기각)
> - **1-4**: 대안 설명(fatigue, perseveration) 배제
> - **1-5**: Bout 구조의 비-random성 확립 (planned delay의 구조적 증거)

---

## 1-1. 기본 현상: Human vs. IL Thinker action distribution

*리뷰어 관점: 비교 대상은 반드시 IL thinker여야 한다. Pretrained thinker와의 차이는 policy 자체의 차이(task 수행 방식)일 수 있으므로, imitation 이후에도 gap이 남는지가 핵심이다. Section 1-1의 모든 비교는 **human vs. IL thinker** 기준이다.*

### 확보된 결과 (sub001 ses04, SpaceInvaders, 8개 파일, 49,188 real steps)
- Human NOOP: **39.0%**, **IL Thinker** NOOP: **36.0%**, gap: **+3.05pp**
- NOOP bout 총 **2,196개**, 평균 bout 길이 **8.37 real steps**
- Human–IL thinker action distribution JSD 평균: **0.005**
- Gap이 작다는 것은 **IL이 human의 NOOP 빈도를 성공적으로 모방했다는 증거**이며, 남은 residual gap(+3.05pp)이 imitation으로 포착되지 않는 **human-specific withholding의 순수 신호**다.

> **비교 기준 명확화**: pretrained thinker (NOOP ~22%)는 policy 자체가 human과 다르기 때문에 direct NOOP gap 비교에 적합하지 않다. IL thinker는 human action distribution을 모방했기 때문에, 이 기준에서 남은 gap과 selectivity 차이가 **human-specific strategic timing**의 증거다. Pretrained thinker는 별도 대조군으로 Figure 1-1에 함께 표시하되, 해석의 중심은 IL thinker와의 비교다.

### 의미 및 가설 연결
IL thinker는 human의 NOOP *빈도*를 잘 모방한다(JSD 0.005). 그러나 이후 섹션(Section 3)에서 보이듯 IL thinker는 NOOP *selectivity* — 즉 uncertainty가 높은 state에서 선택적으로 NOOP를 하는 패턴 — 는 human만큼 재현하지 못한다. 이것이 C1의 핵심 주장이다: **IL은 NOOP frequency를 학습하지만, uncertainty-contingent selectivity는 human에서만 나타나는 전략적 특성이다.**

### 한계 및 후속 필요
- 현재 sub001 단일 피험자 수준 → **N=6 전체 확장 필요** ⚠️
- Human vs. IL thinker의 **uncertainty-conditional NOOP rate 비교**: 같은 uncertainty bin에서 human이 IL thinker보다 더 높은 NOOP rate를 보이는가? (selectivity 차이 정량화, Section 3 연결)
- IL thinker 내부 tree statistics와 residual NOOP gap의 관계 정량화 (Section 2로 연결)

---

## 1-2. 효과의 robustness와 "총량이 아닌 선택적 사용"

*리뷰어 관점: N=6에서 집단 평균만 보여주면 충분하지 않다. 피험자 간 일관성이 없으면 entire effect가 outlier에 의해 driven될 수 있다. 또한 NOOP가 많다는 것이 좋은 것인가, 나쁜 것인가?*

### 확보된 결과 (N=6, Pong & SpaceInvaders)

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

### 핵심 주장: 총량이 아닌 사용 자체가 중요하다

세 결과(1-2C, D, E)가 하나의 내러티브를 구성한다:
> - 1-2C: NOOP 총량 ↔ episode score = **null** → "많이 누른다고 성과가 높아지지 않는다"
> - 1-2E: NOOP 총량 ↔ withholding benefit = **null**, but benefit > 0 consistently → "얼마나 많이 쓰든, 쓸 때마다 이득이 있다"
> - 1-2D: NOOP-preceded episode reward > non-preceded → "전략적으로 사용된 NOOP는 episode 결과를 향상시킨다"
>
> **결론: withholding의 총량이 아니라 사용 자체(quality of deployment)가 중요하다.** 1-2C null result에 대한 리뷰어 공격("NOOP는 의미없는 것 아닌가?")을 1-2E가 직접 반박한다.

Non-IL thinker는 NOOP 비율이 낮으면서도 성능이 더 높다는 사실이 이를 뒷받침한다: policy가 이미 sharp한 agent는 deliberation이 불필요하고, policy가 덜 sharp한 human은 withholding을 통해 deliberation을 보상한다. **핵심은 NOOP의 총량이 아니라 selectivity다.**

ICC가 낮다는 사실은 두 가지로 해석된다: (1) session마다 다른 task state geometry에 반응하는 state-reactive 행동 → selectivity 해석 지지, (2) 단순 측정 노이즈 가능성 → Section 3에서 uncertainty-NOOP coupling의 stability로 재확인 필요.

### 추가로 필요한 연구
- **Selectivity index**: uncertainty-triggered NOOP 비율이 session 성과를 예측하는지 (총 NOOP ratio null과 대비하면 narrative가 강해짐) **[NEW, 우선순위 높음]**

---

## 1-3. 게임 간 일반화

*리뷰어 관점: 하나의 게임에서만 나타나는 현상은 task-specific artifact일 수 있다. 동일한 패턴이 두 게임에서 재현된다면 general mechanism의 증거가 된다.*

### 확보된 결과 (N=6)

**Fig 1-3A — Effect direction consistency:**
- **6/6 피험자** 모두 Pong과 SpaceInvaders에서 NOOP > chance → 완벽한 방향 일관성

**Fig 1-3B, 1-3C — Bout length & meta-analytic check:**
| Game | N bouts | Mean length | Max |
|------|---------|-------------|-----|
| Pong | 13,038 | **49.5 steps** | 2,899 |
| SpaceInvaders | 60,109 | **9.9 steps** | 490 |
- 두 게임 모두 heavy-tailed distribution (max/mean ≈ 60배 이상)
- 피험자별 NOOP above-chance scatter: 두 게임에서 모두 양의 방향 일관

### 의미 및 가설 연결
방향 일관성(6/6)은 game-specific artifact 기각의 강력한 증거다. 단 절대 수준 차이(87% vs 43%, mean bout 49 vs 9.9)가 크므로, 두 게임이 **동일한 mechanism**을 반영한다고 단순 주장하기는 어렵다. **SpaceInvaders가 strategic withholding의 clean test bed**이며, Pong은 game-structure contrast 역할로 활용하는 것이 적합하다.

### 추가로 필요한 연구
- **Uncertainty-NOOP coupling 게임 간 비교**: SI에서 확인된 uncertainty selectivity가 Pong에서도 재현되는지 (Section 3과 연결) **[NEW]**

---

## 1-4. 대안 설명 배제

*리뷰어 관점: NOOP가 많다는 주장의 가장 강한 반론은 "버튼 피로", "motor inertia", "passive omission"이다. 이를 데이터로 배제해야 한다.*

### 확보된 결과 (N=6)

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

## 1-5. Bout 구조의 비-random성: Survival analysis

*리뷰어 관점: Random omission이라면 bout length가 지수분포를 따라야 한다. 이로부터의 이탈이 "계획된 지연"의 구조적 증거다.*

### 지수 기준선(Exponential Baseline)을 그리는 이유

NOOP 행동 보류의 "passive vs. active" 여부를 판별하기 위해 **지수분포 기준선**을 설정한다. 만약 NOOP가 완전히 무작위적인 omission, 즉 각 step에서 독립적으로 NOOP 여부를 결정하는 **Poisson process**라면, NOOP bout의 지속 시간(bout length)은 **지수분포(Exponential distribution)**를 따라야 한다. 지수분포는 "memoryless" 성질을 가지며, 현재 이미 t step 동안 NOOP가 지속됐다고 해서 다음 step에도 NOOP를 유지할 확률이 높아지거나 낮아지지 않는다(항상 동일한 hazard rate λ). 따라서 **지수분포 = 랜덤 누락(random omission)의 귀무 가설(null hypothesis)**이며, 이로부터의 이탈이 "bout이 계획된 단위로 구성된다"는 증거가 된다.

### 지수 모델 추정 방법

지수 기준선의 rate parameter **λ**는 관측된 bout length 데이터의 MLE(Maximum Likelihood Estimation)로 추정한다. 지수분포에서 MLE 추정치는 단순히 **λ̂ = 1 / mean_bout_length**이다. 이로부터 지수 생존함수 **S_exp(t) = exp(−λ̂ · t)**를 계산하여 경험적 KM 곡선과 같은 축에 점선으로 표시한다.
> Kiefer, N. M. (1988). "Economic Duration Data and Hazard Functions." Journal of Economic Literature, 26(2), 646–679. \
> Caballero, R. J., & Engel, E. M. R. A. (1999). "Explaining Investment Dynamics in U.S. Manufacturing: A Generalized (S,s) Approach." Econometrica, 67(4), 783–826.

### Kaplan-Meier 생존 그래프 작성 방법

Kaplan-Meier(KM) 추정법은 bout length 데이터로부터 비모수적(non-parametric)으로 생존함수 S(t) = P(bout length > t)를 추정한다. 각 bout 종료 시점 t_i에서 "위험에 노출된 bout 수 n_i" 대비 "종료된 bout 수 d_i"를 이용하여 S(t)를 step 함수로 갱신한다: **S(t) = ∏_{t_i ≤ t} (1 − d_i / n_i)**. 에피소드 종료로 인해 bout이 강제 중단되는 경우(censoring)는 해당 시점에서 위험군에서만 제외되고 이벤트로는 계산되지 않는다(censoring rate 0.1~0.6%로 사실상 무시 가능). 각 피험자 × 게임 조합에 대해 별도의 subplot을 생성하고, KM 실선 위에 지수 기준선 점선을 겹쳐 표시한다.

### 확보된 결과 (N=6, human data only)

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

### "위 → 아래 → 위" 3구간 교차 패턴의 해석

실제 KM 곡선은 **3구간 패턴**을 보인다: 처음에 지수 기준선 위 → Cross1에서 아래로 교차 → Cross2에서 다시 위로 교차. 시각적으로 Cross1(매우 짧은 t)이 잘 보이지 않아 "처음에 아래로 내려갔다가 위로 올라온다"처럼 보이지만, 실제로는 세 구간이 존재한다. 이 패턴은 단순히 "지수분포에서 벗어났다"는 것 이상의 구체적인 메커니즘을 시사한다.

- **구간 1: t < Cross1 — KM > 지수 기준선 (매우 짧은 bout 구간)**:
  KM이 지수 기준선보다 높다는 것은, 극단적으로 짧은 (1~2 step) bout이 지수 예측보다 *더 적게* 관찰된다는 뜻이다. 즉, 한 번 NOOP가 시작되면 1~2 step에서 즉시 종료되는 경우가 순수 랜덤보다 적다 — bout이 일단 시작되면 최소한의 지속 구조를 가진다는 뜻이다.

- **구간 2: Cross1 < t < Cross2 — KM < 지수 기준선 (중간 bout 구간)**:
  중간 길이 bout이 지수 예측보다 *더 많이* 종료된다. 이는 2-2에서 확인된 **"Short bouts" (high entropy, crisis-response deliberation)** 유형과 대응된다 — 높은 불확실성 상황에서 빠르게 반응하는 짧은 보류. 이 구간이 전체 bout의 75~94%를 차지한다.

- **구간 3: t ≥ Cross2 — KM > 지수 기준선 (heavy tail 구간)**:
  KM 곡선이 다시 지수 기준선 위로 올라온다는 것은, 긴 bout이 지수 예측보다 *더 많이* 살아남는다는 뜻이다 — **heavy tail**. Cross2에 도달한 bout은 이후 더 오랫동안 지속될 확률이 지수 예측보다 높다 (increasing survival in hazard terms). 이는 2-2의 **"Long bouts" (low entropy, strategic patient planning)** 유형과 대응된다 — 안정적 상태에서 지속되는 계획적 보류. 전체 bout의 5~25%가 이 구간에 해당한다.

- **교차 패턴 전체의 해석**:
  두 가지 질적으로 다른 NOOP bout 유형(단기 crisis-response / 장기 strategic planning)이 혼재하는 **혼합 분포(mixture distribution)** 구조를 반영한다. 지수 단일 모델은 이 두 유형을 평균하여 중간에 위치하지만, 실제 데이터는 양 극단(매우 짧거나 매우 긴 bout)에 더 많은 관측값이 몰려 있다. 이는 NOOP가 Poisson 과정처럼 각 step에서 독립적으로 끝날 확률이 일정한 것이 아니라, **bout이 시작되면 그 내부에 지속 구조가 형성됨**을 의미한다. 특히 bout이 일정 시간 지속되면 더 오래 지속될 가능성이 증가한다는 것은, NOOP가 수동적 버튼 미누름이 아니라 **인지적으로 유지되는 active postponement 상태**임을 시사한다. 6명의 피험자 전원에서 이 패턴이 반복된다는 사실은, 이것이 개인 특성이 아닌 **인간 planning의 구조적 특성**임을 강하게 지지한다.

### 의미 및 가설 연결
1-3에서 텍스트로만 서술한 "heavy-tail → planned delay" 주장을 **피험자별 시각적 증거**로 뒷받침한다. 지수분포에서의 이탈, 특히 초반 단기 초과와 후반 heavy tail이 공존하는 교차 패턴은, 매 step의 NOOP가 독립적으로 결정되지 않고 bout 내부에 시간적 구조가 존재함을 의미한다. 이는 C1(현상의 체계성)과 C2(structured postponement)를 연결하는 bridge 역할을 하며, Section 2-2의 Short/Long bout 이원성과 직접 연결된다.
