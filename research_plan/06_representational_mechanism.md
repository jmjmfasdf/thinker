# 6. Representational Mechanism: World model과 tree search representation의 geometry

*리뷰어 관점: Section 5의 RSA claim이 있다면, thinker latent geometry에 대한 사전 분석이 필요하다.*

## 핵심 질문
- Thinker의 latent state representation은 task-state geometry를 반영하는가?
- Tree search의 구조를 정량화할 수 있는 motif가 존재하는가?
- 이러한 geometry가 human planning style과 연결되는가?

## 세부 연구 (기존)
- **State embedding geometry** (PCA 후 neighbor structure): branching point 근처에서 state space가 더 복잡한가?
- **Tree search motif**: 실제로 선택된 action과 rejected action 사이의 Q-gap이 NOOP duration을 예측하는가?
- **RSA matrix 구성**: thinker latent state × state → brain region representation matrix와 비교

---

## 6-1. Spectral geometry of tree_reps: Diffusion Maps 접근

> **이론적 배경** (Coifman & Lafon, 2006): 데이터 포인트 간의 local affinity(kernel)로 Markov chain을 구성하면, 그 eigenvector들이 데이터의 "diffusion coordinates"(성격)를 정의한다. Eigenvalue λ_l은 각 기하 구조의 scale별 지속성을 나타낸다. 핵심 논리: **위치(local similarity) → 구조(Markov transition) → 성격(eigenvectors)**. 이것을 tree_reps에 적용한다.

### 핵심 질문
- 하나의 planning step에서 생성된 tree의 node representation들은 어떤 spectral geometry를 갖는가?
- NOOP bout 동안 tree geometry가 action commit 시점과 구조적으로 다른가?
- Spectral gap (λ_2/λ_1)이 action-value conflict를 반영하는가?

### 세부 분석
- **Diffusion Map on tree nodes**: 단일 real step의 tree_reps 전체 node에 kernel k(x,y) = exp(-||x-y||²/ε)를 적용 → Markov matrix P → eigendecomposition → diffusion coordinates 시각화 **[NEW]**
- **Spectral gap 분석**: λ_2/λ_1 비율 (두 번째 vs 첫 번째 eigenvalue). Gap이 작으면 두 개의 거의 동등한 future trajectory 클러스터 존재 → action-value conflict 지표. NOOP bout 동안 gap이 action step보다 작은가? **[NEW]**
- **Multiscale cluster structure**: P^t를 t=8, 64, 1024에서 비교 → short-horizon(세부 분기)과 long-horizon(거친 전략) search 구조 분리 **[NEW]**
- **Diffusion distance D_t(real_node, imagined_nodes)**: root(real state)에서 imagined future states까지의 planning distance 분포. NOOP 직전에 이 거리가 더 넓게 퍼지는가? **[NEW]**
- **RSA matrix 구성**: diffusion coordinate 기반 pairwise distance matrix → brain region RSA와 비교 (Section 5-4 연결) **[기존 RSA 항목 구체화]**

### Section 3 연결
Spectral gap은 Section 3-1의 "state-side uncertainty" 지표 목록(branch entropy, tree width)을 대체·보완하는 새로운 지표로, Section 3-2 formal model comparison에 포함 가능.

---

## 6-2. Stochastic evolution of tree_reps: Neural SDE (scDiffEq) 접근

> **이론적 배경** (Vinyard et al., 2025, Nature MI): single-cell 분화 궤적을 drift(결정론적)와 diffusion(확률적, 상태 의존적)으로 분해하는 neural SDE 프레임워크. 핵심 발견: **다운스트림 fate 결정점(multipotent progenitor)에서 drift와 diffusion magnitude 모두 최대**. Planning의 결정점 = 세포 분화의 분기점 analogy.

### 핵심 가설
NOOP bout 진입 = multipotent state (여러 action 방향으로 분기 가능 → diffusion ↑)
Action commit = fate commitment (한 direction으로 수렴 → drift dominant)

### 세부 분석
- **Tree centroid trajectory**: real steps 시퀀스에 걸쳐 tree_reps의 weighted mean (또는 top-k node mean)을 추출, temporal evolution 모델링 **[NEW]**
- **Drift-diffusion 분리**: z_{t+Δt} = z_t + f(z_t)·Δt + g(z_t)·noise 형태로 tree centroid evolution fit. f = drift network (결정론적 planning 방향), g = diffusion network (탐색의 stochasticity) **[NEW]**
- **g(z_t) proxy (저비용 근사)**: neural SDE 없이도, 각 real step에서 imagined nodes 간의 pairwise distance 평균을 diffusion magnitude의 proxy로 사용. NOOP steps vs action steps 비교 **[NEW, 우선순위 높음]**
- **Dose-response**: g(z_t)가 높은 step에서 NOOP probability와 bout length가 더 큰가? (Section 4-2 dose-response와 연결) **[NEW]**
- **Commit 직전 drift dominance 검증**: Section 2-2의 "uncertainty rise-and-fall"을 f/g ratio로 quantify → commit 직전 f(z) ↑, g(z) ↓ 패턴 **[NEW]**

---

## 6-3. Intrinsic vs. input-driven planning dynamics: InputDSA 접근

> **이론적 배경** (Huang, Ostrow et al., 2025): DSA를 non-autonomous system으로 확장. x_{t+1} = Ax_t + Bu_t 에서 A(intrinsic dynamics)와 B(input-to-state mapping)를 DMDc/SubspaceDMDc로 추정. **핵심 발견 (쥐 뇌 데이터)**: evidence accumulation 구간에서 input-driven dynamics dominant → decision-making 구간에서 intrinsic dynamics dominant로 전환. 이 전환이 NOOP bout과 정확히 대응한다는 가설.

### 핵심 가설
- **NOOP bout** = intrinsic dynamics (A) dominant: 새로운 game observation이 아니라 내부 tree propagation이 state를 주도 → "외부를 기다리는 게 아니라 내부 검색을 계속"
- **Action step / commit 직전** = input-driven dynamics (B) dominant: real game state에 기반해 최종 결정

### 세부 분석
- **DMDc fit**: tree_reps time series (real steps 기준)에 φ(x_{t+1}) = A·φ(x_t) + B·obs_t 형태로 DMDc 적용. A, B 추정 **[NEW]**
- **SubspaceDMDc**: tree_reps가 partially observed system임을 감안 (imagined nodes는 unobserved 상태 포함) → SubspaceDMDc로 A, B 추정의 정확도 개선 **[NEW]**
- **A vs B eigenvalue 스펙트럼 비교**: NOOP steps vs action steps에서 A의 eigenvalue magnitude (intrinsic) vs B의 singular value (input responsiveness) 비교 **[NEW]**
- **InputDSA_state vs InputDSA_input score**: NOOP 구간에서 state similarity (A) ↑, input similarity (B) ↓ 예측 검증 **[NEW]**
- **Human과 thinker 비교**: high-performing thinker와 human의 A/B decomposition이 유사한가? (InputDSA의 "Anna Karenina principle": 잘하는 agent들은 동적으로 유사) **[NEW, 가장 novel한 contribution]**
- **Section 3 연결**: A/B ratio가 Section 3-2 formal model comparison의 새로운 predictor로 포함. "intrinsic planning dominance index"가 NOOP probability를 예측하는 독립 변인인가? **[NEW]**
