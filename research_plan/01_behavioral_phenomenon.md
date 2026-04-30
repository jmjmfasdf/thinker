# 1. Behavioral Phenomenon: 인간의 행동 보류는 체계적이고 구조를 가진다

*관점: "단순히 NOOP을 많이 누른다"는 것 이상을 보여야 한다. 현재 `01_behavioral_analysis.py`가 실제로 생성하는 figure 흐름에 맞춰, 기술적 현상 정리부터 reward relevance, 대안 설명 배제, survival 구조까지 한 번에 연결한다.*

> **현재 구현된 섹션 구조**: Section 1은 여섯 개의 figure 블록과 몇 개의 보조 CSV를 통해 C1(현상의 체계성)을 행동 데이터만으로 정리한다. 예전의 standalone cross-game figure는 더 이상 생성하지 않으며, 그 핵심인 direction check는 **Fig 1-2C**로 통합되었다.
>
> - **1-1**: withholding bout schematic + action distribution
> - **1-2**: individual differences, performance-null, cross-game direction consistency
> - **1-3**: reward comparison by subject and episode
> - **1-4**: NOOP ratio vs. withholding benefit
> - **1-5**: alternative explanation exclusion
> - **1-6**: bout survival analysis

---

## 1-1. 기본 현상 정리: schematic + action distribution

*리뷰어 관점: 우선 "withholding bout"이 무엇인지 정의해야 하고, 그 다음 인간 행동 repertoire에서 NOOP가 실제로 얼마나 큰 비중을 차지하는지 보여줘야 한다.*

### 현재 파이프라인이 만드는 산출물
- **Fig 1-1A**: `fig_1-1_A_withholding_schematic.png`
- **Fig 1-1B**: `fig_1-1_B_action_distribution.png`
- **CSV**: `1-1_action_distribution_subject_thinker.csv`

### 현재 figure가 말해주는 것
- `1-1A`는 NOOP onset, NOOP end, action commit으로 bout을 정의한다.
- `1-1B`는 **S1–S6 + non-IL Thinker**의 episode-mean action distribution을 게임별로 나란히 보여준다.
- 이 단계의 목적은 "인간 행동에서 NOOP가 실제로 눈에 띄는 행동 범주"라는 점과 "이후 분석에서 bout 단위로 다룰 operational definition"을 먼저 고정하는 것이다.

### 해석
이 단계는 아직 selectivity를 입증하는 단계가 아니다. 대신 Section 1 전체에서 사용할 공통 대상과 표기 체계를 세팅한다. 예전 문서처럼 **human vs. IL thinker residual gap**을 Section 1-1의 핵심 주장으로 두기보다, 현재 구현은 **인간 subject들 + reference thinker의 action profile을 기술적으로 배치하는 단계**에 가깝다.

### 후속 필요
- **human vs. IL thinker** 직접 비교는 별도 분석으로 다시 정리해야 한다.
- uncertainty-conditional NOOP rate나 selectivity 차이는 Section 3에서 formal하게 다루는 편이 현재 파이프라인과 더 잘 맞는다.

---

## 1-2. Individual differences: robust effect, quantity-null, direction consistency

*리뷰어 관점: N=6에서 집단 평균만 보여주면 부족하다. outlier-driven effect가 아닌지, 두 게임 모두에서 같은 방향인지, 그리고 NOOP 총량 자체가 성과를 설명하는지 분리해서 봐야 한다.*

### 현재 파이프라인이 만드는 산출물
- **Fig 1-2**: `fig_1-2_individual_differences.png`
  - `1-2A`: subject-level NOOP proportion
  - `1-2B`: NOOP ratio ↔ performance
  - `1-2C`: meta-analytic direction check
- **보조 CSV**
  - `1-2_subject_game_noop.csv`
  - `1-2_icc_by_subject_game.csv`
  - `1-2_noop_performance_corr.csv`
  - `1-2_cohens_d_vs_chance.csv`

### 확보된 결과 (현재 스크립트 출력 기준)

**Fig 1-2A — Subject-level NOOP proportion**
| Game | Mean NOOP | Range (S1–S6) | Cohen's d vs. chance |
|------|-----------|----------------|----------------------|
| Pong | **0.869** | 0.790–0.918 | **13.65** |
| SpaceInvaders | **0.434** | 0.382–0.531 | **3.91** |

- 6/6 피험자 모두 두 게임에서 chance(1/6)보다 높다.
- 효과는 특정 개인 한 명이 끌고 가는 형태가 아니라, **전 subject에서 방향이 일치하는 현상**이다.

**Fig 1-2B — NOOP ratio ↔ performance**
| Game | Level | r | p |
|------|-------|---|---|
| Pong | Subject | 0.005 | 0.993 |
| SpaceInvaders | Subject | -0.398 | 0.434 |

- 두 게임 모두 **유의하지 않다**.
- 즉, **NOOP 총량 자체는 성과를 직접 예측하지 않는다**.

**Fig 1-2C — Meta-analytic direction check**
- Pong과 SpaceInvaders 모두에서 **6/6 subject가 chance above** 방향에 위치한다.
- 예전 standalone `fig_1-3_cross_game.png`의 핵심 메시지를 이 패널 하나로 압축해 둔 상태다.

**보조 지표 — ICC**
- ICC는 여전히 계산하지만, 더 이상 main figure 패널은 아니다.
- 현재 범위는 **-0.28 ~ +0.16**이며, `1-2_icc_by_subject_game.csv`에 저장된다.
- 해석상으로는 "trait-like 안정성"보다는 **session/state-reactive variability**를 시사한다.

### 핵심 주장
Section 1-2의 메시지는 두 가지다.
- **현상은 robust하다**: 6/6 subject, 2 games에서 방향이 유지된다.
- **총량은 설명 변수가 아니다**: NOOP를 많이 누른다고 성과가 좋은 것은 아니다.

즉 여기서 확보되는 것은 "인간 행동 보류는 존재하고 일관되지만, 단순 빈도로 환원되지는 않는다"는 점이다.

### 후속 필요
- total NOOP ratio가 아닌 **selectivity index**가 성과를 예측하는지 검증해야 한다.
- ICC 해석은 uncertainty-coupling 안정성과 함께 later section에서 재판단하는 것이 적절하다.

---

## 1-3. Reward comparison by subject and episode

*리뷰어 관점: 1-2에서 total NOOP ratio가 성과와 무관하다면, "그럼 NOOP는 의미 없는 행동 아닌가?"라는 반론이 즉시 들어온다. 따라서 quantity-null과 별개로, 실제 withholding deployment가 reward와 연결되는지를 보여줘야 한다.*

### 현재 파이프라인이 만드는 산출물
- **Fig 1-3**: `fig_1-3_reward_subject_episode.png`
- 내부 계산: episode별 `withholding-preceded` vs. `not-preceded` commit reward summary

### figure의 구조
- 6명의 subject 각각에 대해 Pong / SpaceInvaders를 나눠 표시한다.
- 각 episode를 paired line으로 연결하고, `withholding-preceded`와 `not-preceded` episode mean k-step reward를 비교한다.
- 패널별 paired test 결과는 significance bracket으로 바로 표시된다.

### 현재 문서에서의 해석
이 figure는 Section 1에서 **deployment quality**를 가장 직접적으로 보여주는 근거다. `withholding-preceded` condition이 episode-level reward와 연결된다는 점을 subject × game 단위로 시각화하므로, "많이 누르는 것"과 "적절한 타이밍에 쓰는 것"을 분리하는 교두보 역할을 한다.

### 주의할 점
- 이 단계는 여전히 descriptive/paired comparison이다.
- selection bias나 state confound를 완전히 제거하는 것은 Section 4의 matched-control / propensity design 몫이다.

---

## 1-4. NOOP ratio vs. withholding benefit

*리뷰어 관점: reward comparison이 positive라고 해도, 혹시 NOOP를 많이 쓰는 사람이 원래 더 좋은 상태에만 들어가서 생기는 효과일 수 있다. 총량과 benefit의 관계를 별도로 떼어봐야 한다.*

### 현재 파이프라인이 만드는 산출물
- **Fig 1-4**: `fig_1-4_noopratio_postnoop_reward.png`

### 확보된 결과
| Game | r | p |
|------|---|---|
| Pong | -0.01 | 0.938 |
| SpaceInvaders | -0.04 | 0.482 |

- 두 게임 모두 regression slope는 **null**이다.
- 즉, episode-level에서 **NOOP ratio가 높다고 withholding benefit이 커지지 않는다**.

### 의미
Section 1-3과 1-4를 합치면 메시지가 분명해진다.
- `1-3`: withholding-preceded episode reward 비교는 유의한 차이를 보여준다.
- `1-4`: 하지만 그 차이는 **총량**과는 무관하다.

따라서 Section 1의 중간 결론은 다음과 같다.
> **NOOP의 양이 아니라, 언제 어떻게 쓰였는지가 중요하다.**

### 후속 필요
- entropy-conditional benefit, matched-control benefit, bout length dose-response는 Section 4에서 formal test로 확장한다.

---

## 1-5. 대안 설명 배제: fatigue와 단순 serial dependency

*리뷰어 관점: NOOP가 많아 보여도 그게 planning일 필요는 없다. 손 피로, 후반 집중 저하, 단순 반복성(perseveration)으로도 비슷한 패턴이 나올 수 있다.*

### 현재 파이프라인이 만드는 산출물
- **Fig 1-5A**: `fig_1-5_alternative_exclusion.png` Panel A
- **Fig 1-5B**: `fig_1-5_alternative_exclusion.png` Panel B
- **보조 CSV**
  - `1-4_episode_position_thirds.csv`
  - `1-4_noop_autocorrelation.csv`

### 확보된 결과

**Fig 1-5A — Episode-position NOOP density**
| Game | Early | Mid | Late | Early vs. Late |
|------|-------|-----|------|----------------|
| Pong | 0.866 | 0.872 | 0.867 | p = 0.2015 |
| SpaceInvaders | 0.442 | 0.436 | 0.419 | p ≈ 0 |

- Pong은 episode 전반에서 거의 평평하다.
- SpaceInvaders는 후반으로 갈수록 오히려 감소한다.
- 따라서 **fatigue / late-episode omission** 설명과는 맞지 않는다.

**Fig 1-5B — Lag-1 autocorrelation**
- Pong mean AC = **0.831**
- SpaceInvaders mean AC = **0.813**

### 해석
- fatigue는 현재 figure 수준에서도 꽤 강하게 기각된다.
- 반면 lag-1 AC는 **설명 변수라기보다 경고등**에 가깝다. bout 구조가 있으면 AC는 자연스럽게 올라갈 수 있기 때문이다.
- 따라서 현재 `1-5B`는 "perseveration이 완전히 배제되었다"가 아니라, **serial structure가 존재하며 이를 더 엄밀한 AR-residual test로 분해해야 한다**는 단계다.

### 구현상 변화
- 예전 `1-4B`였던 early/mid/late bar plot은 main figure에서 제거되었고, 수치는 `1-4_episode_position_thirds.csv`로 남긴 상태다.

---

## 1-6. Bout survival analysis: random omission이 아닌가?

*리뷰어 관점: passive omission이라면 bout length는 memoryless한 random process에 가까워야 한다. 반대로 survival curve가 exponential baseline에서 체계적으로 벗어나면, bout 내부에 유지 구조가 있다는 뜻이다.*

### 현재 파이프라인이 만드는 산출물
- **Fig 1-6**: `fig_1-6_survival_by_subject.png`
- console summary: subject × game별 `N_bouts`, `Censored%`, `Mean_len`, `Median_len`

### 현재 구현에서 사용하는 비교 기준
- 각 subject × game 조합에 대해 **Kaplan-Meier survival curve**를 그린다.
- episode 끝에서 잘린 bout은 **right censoring**으로 처리한다.
- baseline은 각 조합의 uncensored mean length에서 추정한 **exponential survival**이다.

### 확보된 결과 (현재 스크립트 출력 기준)
- Censoring rate는 **0.1% ~ 0.6%**로 매우 낮다.
- Pong mean bout length는 **31.5 ~ 71.8 steps**
- SpaceInvaders mean bout length는 **6.5 ~ 19.2 steps**
- 두 게임 모두 subject별 survival curve가 exponential baseline과 체계적으로 다르다.

### 문서상 정리
예전 버전의 Cross1 / Cross2 교차 지점 추출은 별도 survival script에서 계산된 2차 분석이었다. 현재 `01_behavioral_analysis.py`의 통합 파이프라인은 **KM curve + exponential baseline + summary table**까지를 main result로 삼는다. 따라서 Section 1 문서에서도 이 범위까지만 핵심 claim으로 두는 것이 현재 구현과 가장 잘 맞는다.

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

### Section 1의 최종 결론
현재 구현된 `01_behavioral_analysis.py`가 행동 데이터만으로 정리해 주는 C1 claim은 다음과 같다.
- 인간은 두 게임에서 일관되게 높은 NOOP 사용을 보인다.
- 그러나 그 총량 자체가 성과를 설명하지는 않는다.
- reward comparison과 benefit scatter를 함께 보면, **withholding은 양이 아니라 deployment의 질**과 연결된다.
- fatigue 설명은 약하고, serial structure와 survival structure는 분명하다.
- survival curve의 비-random성은 NOOP가 단순 omission이 아니라 **시간 구조를 가진 행동 보류**임을 지지한다.
