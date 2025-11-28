# Parameter recovery와 인간 데이터 피팅 방법 정리

## 1. 이 논문에서의 parameter recovery 개념

일반적인 *parameter recovery*는 “모델로 데이터를 생성한 뒤, 다시 그 모델을 피팅했을 때 원래의 파라미터((\theta_{\text{true}}))를 얼마나 잘 되찾는지”를 보는 절차다.

이 논문에서는 파라미터 자체보다는 **맥락(context)에 따른 행동 패턴**을 대표하는 **행동 파라미터**를 회복하는지에 초점을 맞춘다.

* 두 단계 Markov decision task(MDT)에서 맥락은 주로

  * goal condition (specific vs flexible)
  * state-transition uncertainty (낮음 vs 높음)
    로 정의된다.
* 인간과 모델의 행동 데이터를 **generalized linear model(GLM)**로 피팅하고, 맥락 벡터 (\theta)에 대한 회귀 계수 (\beta)를 “행동 파라미터”로 간주한다.
* 그런 다음

  * 인간 데이터에서 얻은 (\beta_{\text{human}})
  * 인간에 피팅된 모델로부터 시뮬레이션한 데이터에서 얻은 (\beta_{\text{model}})
    를 비교하는 것이 이 논문에서의 **parameter recovery**이다.

즉, 이 논문이 말하는 parameter recovery는

> “모델이 인간 데이터에 피팅된 뒤, **맥락에 따른 행동 프로파일(behavioral profile)**을 얼마나 잘 재현하는가?”

를 정량화하는 절차다. 이때 행동 프로파일은 GLM 계수((\beta))로 표현된다. fileciteturn0file0

---

## 2. 인간 데이터를 가장 잘 설명하도록 모델을 피팅하는 방법

논문은 RL 모델을 인간 데이터에 맞추는 세 가지 학습 regime을 정의한다.

### 2.1 Goal matching (GM)

* **목적**: 인간처럼 행동할 필요는 없고, **단순히 보상을 최대화하는 정책**을 학습.
* 환경: 두 단계 MDT.
* MDP 구성

  * State (S): 현재 위치(state) + goal condition (flexible/specific).
  * Action (A): 각 state에서 Left / Right 2개.
  * Transition (T): state-transition uncertainty 조건에 따라 ((0.9, 0.1)) 또는 ((0.5, 0.5)).
  * Reward (R): ({0, 10, 20, 40}) 중 하나. goal 조건을 만족하면 보상, 아니면 0.
* 학습: 표준 RL (예: DDQN, meta-RL)이 reward signal만 보고 policy를 학습.

GM은 **인간 데이터와 상관없이** “같은 환경에서 최적 정책을 배우는” baseline으로 쓰이고, 인간 설명력 측면에서는 직접적인 피팅은 아니다.

### 2.2 Behavior cloning (BC)

* **목적**: 환경의 reward 구조는 무시하고, **인간 행동을 직접 모방**.
* 각 trial에서

  * 모델 행동이 인간 행동과 같으면 (+1)
  * 다르면 (-1)
    의 보상을 주는 식으로 설정.
* 즉, loss는 거의 supervised learning의 cross-entropy와 유사한 **pure imitation** 구조.

BC는 인간과의 일치도는 높일 수 있지만,

* 환경의 goal / uncertainty 구조를 얼마나 이해했는지는 불분명하고,
* 과적합에 취약하다는 한계가 있다.

### 2.3 Policy matching (PM)

이 논문에서 핵심적인 fitting 방법은 **policy matching (PM)**이다.

PM은 **GM과 BC를 섞은 방식**으로 이해할 수 있다.

1. 에피소드 단위로 에이전트가 두 단계 MDT에서 **보상을 최대화하는 RL 에이전트**로 작동한다 (GM 성격).
2. 하지만 학습에 실제로 사용되는 보상은 **인간 정책과의 유사도**를 반영하는 **terminal reward**이다 (BC 성격).

#### 2.3.1 PM에서의 MDP 보상 정의

에피소드(한 게임)의 끝에서 terminal reward (R_\Omega)를 다음과 같이 정의한다.

[
R_\Omega =
\begin{cases}
k-n, & a_{ag}^1 \neq a_H^1 \ \text{and}\ a_{ag}^2 \neq a_H^2 \
k+n, & a_{ag}^1 = a_H^1 \ \text{and}\ a_{ag}^2 = a_H^2 \
k,   & \text{otherwise}
\end{cases}
]

* (a_{ag}^i): 에이전트가 단계 (i)에서 선택한 행동
* (a_H^i): 같은 trial, 같은 단계에서 인간이 실제로 선택한 행동
* (k>0), (n \ge 0): 상수 (보상 스케일 조절)

해석:

* 두 단계 모두 인간과 같으면 (k+n) (최대 보상)
* 두 단계 모두 다르면 (k-n) (최소 보상)
* 나머지는 (k) (중립)

즉, 에이전트는 **환경의 상태-전이와 보상 구조를 이용해 탐색하면서도**, 에피소드의 성패는 “얼마나 인간과 같은 선택 시퀀스를 만들었는가”로 평가된다.

#### 2.3.2 Likelihood 기반 적합도 평가

각 모델–피험자 쌍에 대해 PM으로 학습을 진행하면서

[
L = p(a_t^{\text{human}} \mid s_t)
]

형태의 **likelihood**를 계산해, 모델이 해당 state에서 인간의 실제 선택을 얼마나 높은 확률로 assign하는지 본다.

* Underfitting test에서는 이 likelihood(혹은 hit-rate)를 사용해 “아예 인간 행동을 따라가지 못하는 모델”을 걸러낸다.
* Deep RL 모델(DDQN, meta-RL, IQN)은 PM-pfcRL1, PM-pfcRL2가 달성한 likelihood를 기준으로 **조기 종료** 조건을 두어, 비슷한 수준으로 맞춰질 때까지 학습한다.

요약하면, **인간 데이터를 가장 잘 설명하도록 모델을 피팅하는 방법**은:

1. 두 단계 MDT 환경에서 RL 에이전트를 돌리며,
2. 에피소드 끝 보상을 인간 행동과의 일치도로 정의하는 **policy matching MDP**를 구성하고,
3. RL 학습을 통해 모델 파라미터를 업데이트하며,
4. 최종적으로 likelihood / hit-rate를 통해 underfitting 여부를 판단하는 것.

이 피팅 절차를 82명의 피험자 × 9개 RL 모델 × 여러 training regime에 대해 수행한다.

---

## 3. “Provably efficient” contextual behavior recoverability (이론적 parameter recovery)

### 3.1 맥락 효과를 행동 파라미터로 표현

각 훈련 단계(0번째: 인간 데이터, 이후: 재훈련 단계 (i))마다, task context 벡터 (\theta_i)와 행동 데이터 (x) 사이의 관계를 **선형 GLM**으로 정의한다.

[
g_i(\theta) = \beta_i^{\top} \theta_i.
]

* (\theta_i): 맥락 벡터 (예: goal condition, state-transition uncertainty 등; 차원 (K_\theta))
* (\beta_i \in \mathbb{R}^{K_\theta}): 해당 단계에서의 **contextual behavior parameter**

0번째 단계 (g_0)는 인간 데이터로부터 얻은 회귀 계수 (\beta_0)이고, 이후 단계 (g_n)은 재훈련된 모델의 시뮬레이션 데이터로부터 얻은 (\beta_n)이다.

### 3.2 파라미터 차이를 요약하는 지표 (S_n)

(n)-번째 재훈련 이후, 인간과 모델 사이의 맥락 효과 차이를 L1 loss로 요약한다.

[
S_n = \sum_{j=1}^{K_\theta} \left| \beta_{0,j} - \beta_{n,j} \right|.
]

* (S_n)이 작을수록, 인간의 맥락 효과와 모델의 맥락 효과가 비슷하다 → **좋은 parameter recovery**.
* 재훈련을 반복하면 overfitting 때문에 (S_n)이 점점 커질 수 있으므로, (\mathbb{P}(S_n \ge \lambda))의 상계를 **overfitting bound**로 사용한다.

### 3.3 단계별 변화량 (X_i)와 부분합 (D_n)

단계 사이의 변화량을 다음과 같이 정의한다.

[
X_i = \sum_{j=1}^{K_\theta} \left| \beta_{i-1,j} - \beta_{i,j} \right|, \quad i=1,\dots,n.
]

그리고 그 부분합

[
D_n = \sum_{i=1}^n X_i
]

을 정의하면, 다음 부등식이 성립한다 (Lemma 1):

[
S_n \le D_n.
]

이는 “전체 drift ((S_n))는 매 단계의 L1 변화량 합((D_n))으로 상계된다”는 단순한 **triangle inequality** 응용이다.

이제 (\mathbb{P}(S_n \ge \lambda) \le \mathbb{P}(D_n > \lambda))이므로, (D_n)의 tail bound를 구하면 곧 overfitting bound가 된다.

### 3.4 Submartingale 가정과 Azuma류 부등식

논문은 다음과 같은 근거 있는 가정을 둔다.

1. (X_i)는 **독립이고, 0 이상이며, 상계 (c)를 가진 random variable**이다.
2. (X_i)의 기대값 (\mu_i = \mathbb{E}[X_i])는 **단조 감소하는 지수열**이다.

   * (\exists r \in (0, 1)) s.t. (\mu_{i+1} \le r,\mu_i) for all (i).

이때

[
D_n = \sum_{i=1}^n X_i
]

은 자연스러운 필트레이션 (\sigma(X_1,\dots,X_{n-1}))에 대해 **submartingale**이 된다 (Lemma 2).

Submartingale에 대해 Azuma–Hoeffding 류의 불등식을 확장하여, 다음 정리를 얻는다 (Theorem – Overfitting Bound):

[
\mathbb{P}(S_n \ge \lambda)
\le \mathbb{P}(D_n > \lambda)
\le C_n \exp\left(-\frac{1}{2}\frac{\lambda^2}{c^2 n}\right),
]

[
C_n = \prod_{i=1}^n \left(1 + \frac{\mu_i}{c}\right).
]

### 3.5 (C_n)의 포화와 “두 번만 재훈련해도 충분하다”는 주장

가정 2에 의해

[
\mu_i \le r^{i-1} \mu_1, \quad 0<r<1
]

이므로, (\mu_i)는 기하급수적으로 줄어들고, 따라서

[
C_n = \prod_{i=1}^n \left(1 + \frac{\mu_i}{c}\right)
]

역시 (n \to \infty)에서 **수렴**한다.

실제로는 (\mu_i)를 모르는 대신, 관측된 (X_i)로부터

[
\hat{\mu}_i = X_i, \quad \hat{r} = \frac{X_2}{X_1}
]

와 같이 추정하고, 이를 이용해 (C_n)의 장기적인 크기를 예측한다.

핵심 직관:

* **재훈련 초기에 (X_1, X_2)가 빠르게 줄어들면** → (\hat{r}<1)이 충분히 작음.
* 이때 (C_n)은 빠르게 포화되고, 이후 재훈련을 더 해도 overfitting bound가 크게 나빠지지 않는다.
* 따라서 **두 번 정도의 재훈련만으로도**, “이 모델이 인간의 맥락 의존적 행동 파라미터를 얼마나 안정적으로 회복하는지”를 판단할 수 있다는 것이 이론적 결과의 실질적인 의미다.

이것이 논문에서 말하는 “provably efficient contextual behavior recoverability test”이며, **parameter recovery를 위해 많은 retraining을 돌릴 필요가 없다는** 점을 수학적으로 정당화한다.

---

## 4. Behavioral profile correlation을 이용한 실질적인 parameter recovery

이론적 bound와 별도로, Methods에서는 보다 전통적인 **behavioral profile correlation 기반 parameter recovery**를 정의한다.

### 4.1 GLM 기반 behavioral profile

두 단계 MDT에서의 행동을, 맥락 벡터 (x)와 choice optimality (y) 사이의 GLM으로 표현한다.

[
y = \beta_1 x_1 + \beta_2 x_2,
]

* (x_1): goal condition (specific vs flexible)
* (x_2): state-transition uncertainty (low vs high)
* (y): choice optimality (해당 trial에서, context를 완전히 아는 “ideal agent”의 행동과 같은 행동을 했으면 1, 아니면 0)

이 GLM을

* 인간 데이터에 적합하여 (\beta_{\text{human}} = (\beta_{1,\text{human}}, \beta_{2,\text{human}}))
* 인간에 피팅된 모델의 시뮬레이션 데이터에 적합하여 (\beta_{\text{model}} = (\beta_{1,\text{model}}, \beta_{2,\text{model}}))

를 얻는다.

### 4.2 Behavioral profile correlation

이제 **parameter recovery 품질**을 다음과 같이 정의한다.

[
\rho = \text{corr}\big(\beta_{\text{human}}, \beta_{\text{model}}\big).
]

* 두 task context 각각에 대해 상관을 보거나,
* 2차원 벡터로 보고 전체 상관을 볼 수 있다.

의미:

* (\rho>0)이고 통계적으로 유의하면,

  * 모델이 **goal 조작**과 **uncertainty 조작**이 인간 행동에 미치는 효과의 **방향과 상대적 크기**를 제대로 재현한다는 뜻이다.
* 단순히 평균 reward나 평균 선택률이 비슷한 것과는 다르게,

  * **“맥락이 바뀔 때 인간이 어떻게 정책을 조정하는지”라는 미세한 패턴**까지 복제하는지를 보는 test다.

논문은 이를 “Overfitting test using behavioral profile correlation (Parameter recovery analysis)”로 명시하며, Evans (2020) 스타일의 parameter recovery 분석을 따르고 있다고 밝힌다.

---

## 5. 이 논문이 주장하는 “가장 인간 데이터를 잘 설명하는 모델” 선정 기준

정리하면, 이 논문에서 특정 모델(pfcRL2)을 “사람 데이터를 가장 잘 설명한다”고 주장하기 위해 사용한 절차는 다음 세 층으로 구성된다.

### 5.1 1단계: Underfitting test (likelihood)

* PM 또는 BC로 피팅한 뒤, 인간 행동을 얼마나 잘 예측하는지 **likelihood / hit-rate**로 평가.
* 아예 인간 행동 패턴을 따라가지 못하는 모델은 여기서 탈락.

### 5.2 2단계: Parameter recovery / Overfitting test

* **Contextual behavior recoverability**

  * 인간 데이터에서 추정한 (\beta_0)과
  * 피팅된 모델을 재훈련/시뮬레이션하여 얻은 (\beta_n)의 차이 (S_n) 또는 behavioral profile correlation (\rho)를 사용.
  * 두 지표 모두 “맥락에 따른 choice optimality 패턴을 얼마나 안정적으로 재현하는가”를 측정.
* 이론적으로는 submartingale bound를 통해 “두 번 정도의 재훈련으로도 overfitting 여부를 진단할 수 있다”고 보이고,
* 실증적으로는 pfcRL2가 가장 좋은 contextual behavior recoverability를 보임.

### 5.3 3단계: Generalization & adaptation test (추가 검증)

* 10개의 새로운 MDT를 생성한 task space에서

  * Generalization: 새로운 task들에서의 normalized reward 평균
  * Adaptation: task switching / context changing 시나리오에서의 choice optimality
* 여기서도 pfcRL2가 다른 model-free, model-based, meta-RL, SR-DYNA, IQN 등을 모두 앞선다.

따라서 이 논문에서의 “사람 데이터를 가장 잘 설명하는 모델”은,

1. 인간 행동에 잘 피팅되고 (underfitting X),
2. 인간의 **맥락 의존적 behavioral profile((\beta))을 안정적으로 회복**하며 (parameter recovery O),
3. 그 파라미터로 unseen task들에서도 좋은 generalization–adaptation 성능을 보이는

**prefrontal RL2 (pfcRL2)**로 결론난다.

이 전체 과정이 바로, 이 논문이 제시하는 **parameter recovery + policy matching 기반 인간 설명력 평가 프레임워크**이다.
