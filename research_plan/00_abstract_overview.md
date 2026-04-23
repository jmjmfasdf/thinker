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
