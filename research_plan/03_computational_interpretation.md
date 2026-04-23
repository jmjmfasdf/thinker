# 3. Computational Interpretation: NOOP는 uncertainty-sensitive delayed commitment의 proxy인가?

*리뷰어 관점: Correlation은 causal claim이 아니다. "Uncertainty → NOOP"라는 방향성을 어떻게 확인할 것인가? 또한 어떤 uncertainty 지표가 가장 강력한 예측변인인가?*

> **섹션 범위 정의**: Section 2가 bout의 형태를 기술했다면, Section 3는 그 형태를 *일으키는* 계산 변수가 무엇인지를 묻는다. 분석의 단위는 timestep이며, uncertainty metric → NOOP 발생 확률의 인과 방향을 검증한다. Section 2의 temporal profile이 보여준 rise-and-fall 패턴이 실제로 uncertainty에 의해 구동되는지가 이 섹션의 핵심 질문이다.
>
> **1-4와의 관계**: 1-4의 AR 잔차 분석은 perseveration을 통제한 뒤 uncertainty가 NOOP를 예측하는지를 묻는 분석이며, 3-1·3-2의 사전 조건(confound 제거) 역할을 한다. 결과가 수렴하면 두 섹션에서 모두 인용한다.

## 3-1. Uncertainty-NOOP coupling

### 핵심 질문
- Human의 NOOP은 단순히 "가능한 action 중 하나를 자주 누른다" 수준인가?
- 아니면 특정 state에서 선택적으로 증가하는 **structured postponement signal**인가?
- Policy uncertainty가 커질수록 NOOP이 증가하는가?

### 필요한 세부 연구
- Real step별 policy uncertainty와 NOOP probability의 관계 **[]**
- Human vs. pretrained thinker vs. imitation thinker에서 uncertainty-NOOP coupling 비교 **[]**
- 다양한 uncertainty metric과 NOOP의 연관성:
    - **State-side**: world-model uncertainty, rollout disagreement, latent-state novelty, predicted future variance, branch entropy, tree width
    - **Action-side**: chosen-action probability, top-1 minus top-2 policy gap, best-vs-second-best Q gap, commit-step action margin, action entropy의 inverse

---

## 3-2. Formal model comparison *(신규)*

*리뷰어 관점: 어떤 계산 변수가 NOOP를 유발하는가에 대한 systematic model comparison이 없으면, "어느 것이 진짜 원인인가"라는 질문에 답하기 어렵다.*

### 핵심 질문
- Policy entropy, Q-gap, rollout disagreement, search_disagreement (JSD) 중 어느 변수가 NOOP를 가장 잘 예측하는가?
- 이 변수들 간의 상대적 기여를 quantify할 수 있는가?
- Thinker와 human에서 가장 중요한 예측변인이 다른가?

### 필요한 세부 연구
- **Mixed-effects logistic regression**: NOOP ~ entropy + q_gap + search_disagreement + rollout_spread + (1|subject) + (1|game), AIC/BIC로 모델 비교 **[NEW]**
- **Standardized beta coefficient 비교**: 어느 uncertainty metric이 가장 강한 독립 예측력을 갖는가? **[NEW]**
- **ROC/AUC analysis**: 각 모델의 NOOP 예측 정확도 비교 **[NEW]**
- **Human과 thinker에서 회귀계수 비교**: species × uncertainty interaction이 있는가? **[NEW]**

---

## 3-3. Causal ordering: What triggers onset, and what drives commitment?

*리뷰어 관점: Cross-sectional correlation은 인과성을 보장하지 않는다. 가능하면 temporal precedence를 보여야 한다.*

> **Onset paradox와의 관계 (2-2에서 이동)**: 2-2 분석에서 onset이 entropy spike에 의해 트리거되지 **않는다**는 것이 확인됐다. Pre-onset entropy는 오히려 하강하고, bout 내부에서 entropy가 축적된다. 따라서 3-3의 인과 질문은 두 층위로 분리되어야 한다:
> 1. **Onset trigger**: 무엇이 withholding을 시작시키는가? (entropy가 아니라면 game context, margin 절대값, 직전 Q-value 등)
> 2. **Commitment trigger**: 무엇이 withholding을 끝내고 action을 실행시키는가? (commit 직전 entropy peak + 직후 급락은 확인됨)
>
> **2-3과의 관계**: 2-3은 bout *내부*에서 entropy가 어떻게 변하는지를 기술한다. 3-3은 onset *이전*과 commit *직전*에서 인과 신호를 검증한다.
>
> **1-4와의 관계**: 1-4에서 확인된 base-rate 초과 lag-1 AC(SpaceInvaders: 실제 0.81 vs 예측 0.51)는 perseveration 대안 설명의 가능성을 열어 두었다. AR-residual test (아래)가 이를 formal하게 검증하여 perseveration과 uncertainty-driven delay를 해리한다.

### 필요한 세부 연구
- **Onset trigger 후보 분석**: game event markers (e.g., enemy position, between-wave pause), margin 절대값, 직전 k-step reward — 어떤 변수가 onset을 예측하는가? **[NEW, 우선순위 높음]**
- **Commitment trigger 확인 (commit-aligned 패턴의 인과 검증)**: entropy peak at commit → commit은 "entropy가 최고점에서 어떤 threshold를 넘었을 때" 발생하는가, 아니면 "bout duration에 의해 강제되는가?" → bout-length × entropy-at-commit interaction 검증 **[NEW]**
- **AR(1)~AR(5) residual uncertainty test**: lag-k NOOP를 먼저 회귀 제거한 잔차에서 uncertainty(entropy_actor, q_gap)가 NOOP를 독립적으로 예측하는지 확인. k=1~5까지 beta 안정성 추적 → perseveration 통제 후에도 uncertainty 기여가 유지되는가 **[NEW — 1-4에서 이동, thinker 행동 데이터 완비 후 수행]**
- **Granger causality test** 또는 **lagged regression**: t-1 entropy → t NOOP probability (temporal precedence 확인). ⚠️ onset paradox로 인해 onset 시점에서는 유의한 효과가 없을 가능성 — commit 시점에 집중 **[NEW]**
- **Bout onset entropy → bout length**: onset entropy가 낮을수록 bout이 길어지는가? (short vs. long bout의 onset entropy 차이 검증, 2-2 결과와 연결) **[NEW]**
