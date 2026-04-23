# 5. Neural Mechanism: Delayed commitment와 planning을 매개하는 뇌 회로

*리뷰어 관점: fMRI claim은 behavioral claim에 의존한다. Section 1-3이 충분히 확립되어야 이 section이 설득력을 갖는다. Region-of-interest에 대한 a priori hypothesis가 명확해야 한다.*

## 5-1. Commitment gating (striatum / frontal)
- **Regressor**: NOOP 여부, commit 전 delay length, bout length
- **후보 영역**: caudate/putamen, supplementary motor area (SMA), anterior cingulate cortex (ACC), pre-SMA
- **예측**: Striatum은 action commitment 시점에서 phasic response, ACC는 conflict 기간 동안 sustained signal

## 5-2. Uncertainty-linked planning (hippocampus / PFC)
- **Regressor**: policy uncertainty (entropy), search_disagreement (JSD), VRE
- **후보 영역**: hippocampus, vmPFC (value), dlPFC (cognitive control), OFC
- **예측**: Hippocampus는 NOOP bout 동안 uncertainty-proportional activation → prospective search representation

## 5-3. Behavior-to-brain bridge
- Trial/step-level NOOP probability → brain activation
- Trial/step-level VRE → brain activation
- Trial/step-level k-step reward expectancy → brain activation
- **Dissociation test**: planning content (hippocampus) vs. gating signal (striatum/frontal) → 이 두 신호가 분리되는가?

## 5-4. Multivariate/decoding analysis *(신규)*
- **RSA (Representational Similarity Analysis)**: thinker의 latent state geometry와 brain representation의 유사도 비교
- **MVPA**: NOOP vs. action commitment의 multivariate classifier → spatial pattern으로 구별 가능한가?
