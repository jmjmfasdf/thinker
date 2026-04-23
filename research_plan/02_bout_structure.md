# 2. Bout Structure Analysis: NOOP의 시간적 구조와 내적 조직

*리뷰어 관점: Planning의 행동적 signature는 단순한 비율이 아니라 시간적 구조에서 나타난다. 얼마나 긴 bout? 어떻게 시작하고 끝나는가? 이 구조가 computational demand를 반영하는가?*

> **섹션 범위 정의**: Section 2는 NOOP bout의 *형태(shape)*를 기술한다 — 길이 분포, onset·commit 주변의 시간적 궤적, bout 내부의 전환 구조. Uncertainty 값은 bout 형태를 묘사하는 도구로 쓰이며, uncertainty가 NOOP를 *유발하는가*라는 인과 질문(→ Section 3)이나 withholding이 실제로 *도움이 되는가*라는 기능 질문(→ Section 4)은 다루지 않는다.

## 2-1. Bout length distribution

### 핵심 질문
- NOOP bout의 길이 분포는 무엇인가? (exponential vs. heavy-tailed → passive vs. active)
- State-level computational demand (uncertainty, Q-gap)가 bout 길이를 예측하는가?

---

## 2-2. Temporal profile around bout onset and commit

> **Framing**: 여기서 묻는 것은 "uncertainty의 시간적 모양이 어떻게 생겼는가"이다 — onset 전후, commit 전후에 entropy·margin이 어떤 궤적을 그리는지 기술(describe)한다. 이 패턴이 uncertainty-driven commitment를 *지지하는가*라는 해석은 Section 3-3에서 인과 검증과 함께 다룬다.

### 확보된 결과 (sub001 ses04, N=2,196 bouts)

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

### 핵심 질문 (업데이트)
-  **Onset paradox로 기각**: onset 자체는 entropy spike가 아님. "무엇이 onset을 트리거하는가?"를 새로운 질문으로 재정식화
- Commit 직전에 action confidence가 올라가는가? → **확인됨** (rise-and-fall 패턴 ✅)
- Post-commit undershoot는 functional benefit을 반영하는가? → Section 4와 연결

### 필요한 세부 연구
- Real step 기준으로 onset/commit aligned temporal profile 추출 **[✅ Done → Figure 2B/2C]**
- Rise-and-fall formal test (linear trend + paired t) **[✅ Done, 2196 bouts]**
- **Onset trigger 메커니즘 규명**: entropy가 아니라면 무엇이 onset을 결정하는가? → game state (e.g., between-wave pause in SI), margin 절대값, 직전 action의 Q-value 등 **[NEW, 우선순위 높음]**
- **Bout-internal entropy trajectory**: onset(+0) → commit(0) 사이에서 entropy가 단조 증가하는가, 아니면 일정 시점에서 saturate하는가? (긴 bout에서 중간 구간 검증) **[NEW → 2-3과 연결]**
- **Short vs. Long bout의 commit-aligned profile 비교 figure**: 두 유형의 질적 차이를 시각화 **[NEW]**

---

## 2-3. Sequential dependency and transitional structure *(신규)*

*리뷰어 관점: Bout 내부 구조를 이해하면 active search vs. passive waiting을 구별하는 데 도움이 된다.*

> **Framing**: bout 내부에서 uncertainty가 어떻게 변하는지를 *묘사*한다. "단조감소 패턴 = active search 중"이라는 해석적 주장은 Section 3-3(인과 ordering, bout onset 이전 window)과 연결되며, 두 분석은 서로 다른 시간 창(bout 내부 vs. onset 이전)에서 같은 mechanism을 보완적으로 지지한다.

### 핵심 질문
- NOOP → NOOP 전환 확률 vs. Action → NOOP 전환 확률: 두 상태 간의 전환이 Markovian인가?
- Bout 내에서 시간이 지남에 따라 uncertainty가 감소하는가?
- 같은 bout 내에서 thinker가 내부적으로 action preference를 변경하는지 (cur_action의 변화)

### 필요한 세부 연구
- **First-order transition matrix** (NOOP→NOOP, NOOP→act, act→NOOP, act→act) 계산 **[NEW]**
- **Bout-internal uncertainty trajectory**: bout 시작 → 끝 방향으로 entropy가 단조감소하는가? **[NEW]**
- **Cur_action stability within bouts**: bout 내에서 thinker가 선호하는 action이 바뀌는 비율 **[NEW]**

---

## 2-4. Within-session and across-session adaptation *(신규)*

*리뷰어 관점: Strategic postponement라면 task structure를 학습함에 따라 변해야 한다. Learning curve가 없다면 단순 habit이다.*

### 핵심 질문
- Session 초반과 후반 사이에 NOOP 비율이 어떻게 변하는가?
- 게임을 더 잘하게 될수록 (score 상승) NOOP 비율은 어떻게 변하는가?
- Session 내에서 uncertainty-NOOP coupling이 강화되는가? (즉, strategic use가 정교해지는가)

### 필요한 세부 연구
- **Episode-level NOOP ratio와 cumulative score의 상관** (sub-session learning curve) **[NEW]**
- **Early vs. late episode half 비교**: uncertainty-NOOP 관계가 later half에서 더 강한가? **[NEW]**
- **Cross-session comparison** (ses-01 → ses-04): NOOP 전략의 진화 추적 **[NEW]**
