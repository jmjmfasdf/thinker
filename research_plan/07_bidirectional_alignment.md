# 7. [DRAFT] Bidirectional Alignment Analysis: Thinker-Human 표현 공간의 양방향 정렬

> **배경**: 기존 RSA (Section 5-4, 6)는 thinker latent geometry → brain representation의 단방향 비교다. 이 섹션은 반대 방향(인간 행동/뇌 신호 → thinker 차원 복원)을 추가하여 thinker의 어떤 차원이 실제로 인간과 공유된 계산 공간인지를 진단한다. 논문(Tian et al., forward/reverse predictivity framework)의 아이디어를 채용.

> **데이터**: `video_stat.npy` → `real_vectors` (status==0, ~3185 real steps, shape per step: (128,6,6)), `tree_reps_vector` (145066, 128), `im_vectors` (status==2, ~124176 imaginary steps). 행동 레이블(NOOP flag, entropy_actor, q_gap, vre_abs_q, k5_reward)은 기존 CSV와 real step index로 join.

---

## 7-1. Forward / Reverse Predictivity 및 비대칭성

### 핵심 질문
- Thinker activation이 인간 행동(NOOP 여부, uncertainty)을 예측하는가? (forward)
- 인간 행동 신호가 thinker의 각 차원을 역으로 예측하는가? (reverse)
- Forward >> Reverse인 비대칭성이 존재하는가? (thinker의 extra dimension)
- 이 비대칭성이 NOOP real step vs action real step에서 다른가?

### 분석 방법
- **Forward**: Ridge(X=thinker_128, y=human_noop + behavioral_signals) → R²_forward
- **Reverse**: 각 channel i에 대해 Ridge(X=behavioral_signals, y=thinker_dim_i) → R²_reverse_i 분포
- **Asymmetry**: R²_forward − mean(R²_reverse) → NOOP step vs action step 비교
- 데이터: real_vectors (status==0) global avg pool → (N_real, 128), 행동 레이블 join

---

## 7-2. Common / Unique Unit 식별 및 기능 검증

### 핵심 질문
- 어떤 thinker 채널이 인간 행동으로부터 선형 복원 가능한가? (common)
- Common units는 NOOP onset에 반응하는가? Unique units와 temporal profile이 다른가?
- Common units가 이후 행동 결과(k5_reward, VRE)를 더 잘 예측하는가?

### 분석 방법
- R²_reverse 기준 상위 20% → **Common units** (~26채널), 하위 20% → **Unique units**
- **NOOP onset-aligned temporal profile**: real step index 기준 ±k window에서 Common vs Unique 평균 activation 궤적 비교
- **행동 이득 예측력**: Common units activation → k5_reward, vre_abs_q 예측 R² vs Unique units
- **tree search 지표와의 관계**: Common units ~ root_qs_mean variance (action uncertainty); Unique units ~ cur_v 변화량 (imaginary trajectory 내 value update)

---

## 7-3. Imaginary Trajectory의 Effective Dimensionality × NOOP

### 핵심 질문
- Real step 직전 ~39개의 imaginary step (planning window)에서 thinker representation이 얼마나 많은 차원을 탐색하는가?
- NOOP real step의 planning window가 action real step보다 effective dimensionality가 높은가?
- Bout 길이(withholding 지속)와 planning window의 effective dimensionality는 상관하는가?

### 분석 방법
- 각 real step i에 대해 preceding status==2 im_vectors 추출 → global avg pool → (~39, 128)
- PCA eigenvalue spectrum → participation ratio: $(\sum \lambda_i)^2 / \sum \lambda_i^2$
- NOOP real step vs action real step의 participation ratio 분포 비교 (Mann-Whitney U)
- Participation ratio ~ bout length 상관 (Spearman r)

---

## 7-4. Common vs Unique 공간에서의 Planning Trajectory 방향성

### 핵심 질문
- Imaginary trajectory가 Common unit 공간에서는 수렴(uncertainty 해소)하는가?
- Unique unit 공간에서는 다른 패턴(발산 또는 random drift)을 보이는가?
- NOOP bout 동안 두 공간의 trajectory 패턴이 다른가?

### 분석 방법
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
