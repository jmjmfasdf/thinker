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

## 5-5. Subject/game 단위 exploratory RSA pipeline

현재 단계의 목적은 GLM contrast를 만들기보다, 각 subject/game에서 **task state**, **Thinker internal representation**, **ROI BOLD pattern**이 같은 시간 구조를 공유하는지 탐색하는 것이다. 현재 구현은 self-contained 통합 스크립트인 `research_script/07_encoding_analysis.py`에 둔다. 이 스크립트는 기존 `07_encoding_rsa.py`, `08_encoding_loro.py`, `make_figures.py`에 import 의존하지 않고, 필요한 feature construction, RSA, leave-one-run-out encoding, figure 생성을 내부에서 모두 수행한다.

입력은 session/block을 직접 지정하지 않고 다음처럼 subject와 game만 받는다.

```bash
python research_script/07_encoding_analysis.py --subject 1 --game 2
```

이 실행은 해당 subject/game에 속하는 모든 자료를 자동으로 수집한다.

- Behavioral state: `behavioral_data_block/sub-XXX/ses-*/subXXX-sesYY-blockZ-gameG.npz`
- Thinker trace: `test/subXXX/ses-*/subXXX-sesYY-blockZ-gameG_*.npy`
- fMRI BOLD: `/home/jeongmin/fmri/atari/derivatives/ants_mni/subXXX-Y/SessionZ/wfiltered_func_data.nii`

중요하게, 이 분석은 GLM이 아니고 volume/TR 단위의 pattern geometry와 encoding을 직접 다루므로, smoothed image인 `s5_wfiltered_func_data.nii`보다 **unsmoothed filtered image인 `wfiltered_func_data.nii`를 기본 입력으로 사용**한다. Cross et al. (2021)도 voxelwise encoding 분석에서는 fine-grained voxel detail을 보존하기 위해 smoothing을 적용하지 않았고, 5 mm smoothing은 GLM 및 visualization에 사용했다. ROI pattern RSA와 voxel-wise encoding에서는 smoothing이 local pattern structure를 흐릴 수 있으므로, 현재 목적에는 unsmoothed BOLD가 더 적절하다.

분석 단위는 각 block/run이다.

```text
subject-session-block-game
```

각 run은 fMRI 앞/뒤 60 volume을 제외해 최대 480 TR 분석 공간으로 맞춘다. 공통 전처리 산물은 run별 Thinker/RAM feature와 ROI BOLD pattern이며, 이후 RSA branch는 DSM을 만들고 encoding branch는 원 feature/BOLD 행렬을 유지한다.

```text
shared run data:
  features = RAM, Thinker tree_reps/im_vectors/im_vp_vectors, RAM+Thinker
  bold     = left hippocampus, right hippocampus, hippocampus, PFC

RSA branch:
  features/bold -> DSM -> Spearman RSA / partial RSA / RSA permutation

Encoding branch:
  X_runs, Y_runs -> leave-one-run-out ridge encoding / encoding permutation
```

## 5-6. Core claims and analysis layers

이 pipeline에서 만들고 싶은 주장은 세 층으로 분리한다.

1. **Thinker representation은 task state를 반영한다.**
   - Ground truth task state proxy는 RAM DSM이다.
   - `tree_reps`, `im_vectors`, `im_vp_vectors` DSM이 RAM DSM과 유사한지 본다.

2. **일부 ROI의 BOLD representation은 task state를 반영한다.**
   - Left/right hippocampus, PFC, hippocampus-PFC coupling DSM이 RAM DSM과 유사한지 본다.
   - 현재 예시에서는 hippocampus보다 PFC가 더 유망한 후보로 보인다.

3. **일부 ROI의 BOLD representation은 Thinker가 바라보는 task geometry와 닮아 있다.**
   - ROI DSM과 Thinker DSM을 비교한다.
   - 이때 RAM을 통제한 partial RSA도 함께 보고, 단순히 둘 다 RAM을 반영해서 생긴 관계인지 분리한다.

### Representation construction

Thinker representation은 real step 자체가 아니라, **이전 real step 이후부터 현재 real step 직전까지 생성된 non-real/imaginary trajectory 전체**를 현재 real step에 pair한다.

```text
previous real step ... non-real/imaginary steps ... current real step

current real step feature =
  concat(non-real/imaginary step representations between previous and current real step)
```

기본 정의는 real step 사이의 모든 non-real step을 포함하는 것이다.

```text
primary: status != 0
sensitivity: status == 2 only
```

Representation별 처리:

- `tree_reps`: non-empty tree key를 모두 flatten하여 사용한다.
- `im_vectors`: per-step raw vector를 flatten한다. 너무 크면 per-step PCA 후 sequence concat을 유지한다.
- `im_vp_vectors`: `im_vectors`와 동일하게 처리한다.

real step별 feature가 만들어지면 480 TR bin으로 평균하고, column-wise z-score 후 DSM을 만든다.

```text
DSM[i,j] = 1 - corr(feature_i, feature_j)
```

### Target DSMs

각 run마다 다음 target DSM을 준비한다.

- RAM DSM: frame-level RAM feature에 HRF convolution 후 1초 TR binning
- Left hippocampus BOLD pattern DSM
- Right hippocampus BOLD pattern DSM
- Mean hippocampus DSM: left/right hippocampus DSM 평균
- PFC BOLD pattern DSM
- Hippocampus-PFC coupling DSM: 11 TR sliding-window ROI mean correlation의 `abs(r_i - r_j)`

### RSA comparisons

기본 비교는 DSM upper triangle 간 Spearman correlation이다.

```text
rho = Spearman(upper(DSM_A), upper(DSM_B))
```

기본 RSA table:

- Thinker DSM x RAM DSM
- RAM DSM x ROI DSM
- Thinker DSM x ROI DSM
- Thinker DSM x hippocampus-PFC coupling DSM

추가로 partial RSA를 저장한다.

```text
ROI ~ Thinker | temporal lag
ROI ~ RAM | temporal lag
ROI ~ Thinker | RAM + temporal lag
ROI ~ RAM | Thinker + temporal lag
```

이 분석은 “A와 B가 같은 시점쌍을 비슷하게 멀고 가깝게 보는가?”를 묻는다. 직접적인 feature prediction은 별도 encoding 분석으로 둔다.

### Feature-level encoding analyses

DSM-RSA와 별도로 세부 task 정보가 직접 예측되는지도 본다.

Cross et al. (2021)의 encoding 절차를 참고하되, 현재 pipeline의 primary temporal alignment는 우리 방식인 **feature-level canonical HRF convolution 후 1 Hz TR binning**으로 둔다. 구현상 주 분석은 `08_encoding_loro.py`의 원칙을 따른 **leave-one-run-out (LORO) voxelwise encoding**이다.

- Feature는 gameplay frame/step 단위에서 추출한 뒤 canonical HRF로 convolution하고, 그 결과를 1 Hz TR resolution으로 평균한다.
- Cross et al.식 5 s/6 s shifted feature concat 및 RSA 6 s shift는 primary가 아니라 sensitivity/control analysis로 둔다.
- Encoding은 voxelwise ridge regression으로 수행한다.
- Cross-validation은 leave-one-run-out을 기본으로 한다. 예를 들어 run이 11개면 10개 run으로 학습하고 남은 1개 run을 예측하는 절차를 11번 반복한다.
- Ridge alpha는 LORO grid search로 선택한다. 각 alpha에 대해 전체 LORO를 수행하고, 평균 held-out voxelwise Pearson r이 가장 높은 alpha를 사용한다.
- PCA, z-scoring, ridge alpha selection은 train fold 안에서 fit하여 leakage를 피한다.
- Voxel response는 z-score하여 voxel 간 scale 차이를 줄인다.
- Prediction score는 held-out time course의 predicted vs actual Pearson correlation으로 둔다.
- 결과는 voxelwise r을 먼저 계산한 뒤 ROI/model별 mean r, median r, positive voxel fraction, permutation p/q summary로 집계한다.

1. RAM → ROI BOLD encoding
   - `X = RAM HRF TR features`
   - `Y = ROI voxel pattern`
   - task state가 ROI BOLD pattern을 예측하는지 본다.

2. Thinker → ROI BOLD encoding
   - `X = tree_reps / im_vectors / im_vp_vectors`
   - `Y = ROI voxel pattern`
   - Thinker representation이 ROI BOLD pattern을 예측하는지 본다.

3. Incremental encoding
   - `BOLD ~ RAM`
   - `BOLD ~ Thinker`
   - `BOLD ~ RAM + Thinker`
   - `RAM + Thinker`가 `RAM only`보다 나은지 보아, Thinker representation이 task state 이상의 설명력을 갖는지 평가한다.

4. Fold-level diagnostic
   - `encoding_loro_fold_manifest.csv`에는 held-out run별 mean/median r을 저장한다.
   - 이 값은 paper-style figure에서 run 간 variability와 standard error를 그리는 데 사용한다.

## 5-7. Block permutation test

DSM upper triangle은 독립 샘플이 아니고, fMRI/RAM/Thinker feature 모두 강한 temporal autocorrelation을 갖는다. 따라서 단순 random permutation은 부적절하다. 기본 null은 **block permutation**으로 둔다.

```text
480 TR -> 40 TR blocks
block order를 shuffle
DSM_perm = DSM[perm_idx][:, perm_idx]
rho_perm = Spearman(upper(A), upper(DSM_perm))
```

기본 설정:

- `n_perm = 1000`
- primary block size: `40 TR` (Cross et al., 2021의 fMRI encoding/RSA permutation 기준)
- sensitivity block sizes: `20 TR`, `30 TR`, `60 TR`
- 보조 null: circular shift

p-value:

```text
p = (1 + count(rho_perm >= rho_observed)) / (1 + n_perm)
```

양의 alignment를 주 가설로 두면 one-sided p-value를 사용하고, 탐색 결과표에는 two-sided p-value도 함께 저장한다. 여러 representation, ROI, game, block을 동시에 보므로 FDR q-value를 반드시 계산한다.

Encoding 분석도 같은 원리로 held-out BOLD time course의 block order를 shuffle하여 null distribution을 만든다. 단, encoding permutation에서는 LORO fold에서 얻은 prediction은 고정하고, held-out BOLD만 40 TR block 단위로 shuffle한 뒤 predicted-vs-actual Pearson r을 다시 계산한다. 이 방식은 model prediction 자체는 유지하면서 temporal alignment가 우연히 맞는 정도를 추정한다.

Cross et al.의 RSA permutation과 최대한 맞추려면, ROI DSM을 이미 만든 뒤 DSM row/column만 shuffle하는 것보다 **fMRI volume order를 blockwise shuffle한 뒤 ROI DSM을 재구성**하는 방식을 primary로 둔다. 이는 fMRI autocorrelation structure를 보존하면서 model/task DSM과의 temporal alignment를 깨는 방식이다.

## 5-8. Cross et al. (2021) 기준 엄밀성 체크리스트

Cross et al.의 fMRI/encoding/RSA 절차에서 현재 pipeline에 직접 반영할 항목은 다음과 같다.

- **BOLD image**: voxelwise encoding/RSA는 unsmoothed BOLD를 사용한다. Smoothed BOLD는 GLM 또는 visualization용으로만 둔다.
- **Temporal alignment**: primary analysis에서는 model/RAM/Thinker feature를 frame 또는 step 단위에서 canonical HRF로 convolution한 뒤 1 Hz TR resolution으로 평균한다. Cross et al.의 5 s/6 s delay concat 및 6 s shifted RSA는 sensitivity analysis로 함께 보고한다.
- **Dimensionality reduction**: DQN hidden layer는 layer별 100 PC를 사용했다. 우리 분석에서는 `tree_reps`, `im_vectors`, `im_vp_vectors`별 PCA를 쓰되, CV encoding에서는 PCA를 train fold 안에서 fit한다.
- **Encoding model**: voxelwise ridge regression, alpha grid search, leave-one-run-out cross-validation, held-out Pearson correlation을 기본으로 한다.
- **Permutation test**: validation time course 또는 fMRI volumes를 40 TR block 단위로 shuffle하여 autocorrelation을 보존한다. one-sided p-value와 FDR correction을 기본으로 보고한다.
- **RSA metric**: model/fMRI DSM은 correlation distance를 기본으로 한다. RAM/HDF처럼 명시적 low-dimensional task feature는 Euclidean DSM도 함께 보고 metric sensitivity를 확인한다.
- **Model comparison**: 단일 model의 rho뿐 아니라 model 간 차이도 permutation difference distribution으로 검정한다.
- **Control models**: 가능하면 motor regressors, low-level visual/PCA representation, cross-game representation, RAM-only model을 함께 두어 Thinker representation의 unique contribution을 평가한다.

## 5-9. Figures and outputs

subject/game 단위 output 구조:

```text
research_script/outputs/07_encoding_analysis/sub001_game2/
  dsms/
    dsms_<run_label>.npz
  features/
    features_<run_label>.npz
  rsa/
    rsa_manifest.csv
    rsa_partial_manifest.csv
    rsa_permutation_manifest.csv
    rsa_nulls.npz
  encoding_loro/
    encoding_loro_manifest.csv
    encoding_loro_fold_manifest.csv
    encoding_loro_voxel_stats.npz
    encoding_loro_voxel_stats_keys.csv
    plots/
      encoding_loro_mean_r_by_roi.png
      encoding_loro_mean_r_heatmap.png
      encoding_loro_delta_over_ram_heatmap.png
      encoding_loro_positive_voxel_fraction.png
      encoding_loro_best_alpha_heatmap.png
      encoding_loro_summary_plots.pdf
  encoding/
    encoding_manifest.csv  # compatibility copy of LORO summary
  figures/
    dsm_panel_<run_label>.png
    rsa_heatmap_aggregate.png
    perm_nulls.png
    paper/
      new_fig1_strip_plot.png
      new_fig2_heatmap_scaled.png
      new_fig3_fisher_combined.png
      new_fig4_tree_reps_per_run.png
      new_fig5_encoding_comparison.png
  summary.md
```

가능한 한 많은 figure를 저장한 뒤, permutation/FDR 기준으로 유의한 결과를 선별한다.

- Coverage map: subject/game에 포함된 session/block/run 수
- Real/non-real/imaginary interval length histogram
- RAM DSM per run
- Thinker DSM per run
- ROI BOLD DSM per run
- Hippocampus-PFC coupling DSM per run
- Run별 RSA rho heatmap
- Subject/game aggregate RSA heatmap
- Partial RSA heatmap
- Block permutation null distribution for top effects
- p-value/FDR q-value heatmap
- RAM → ROI BOLD encoding score by ROI
- Thinker → ROI BOLD encoding score by ROI
- LORO encoding score by ROI/model
- Incremental LORO encoding heatmap/bar plot: RAM vs Thinker vs RAM+Thinker
- Held-out run별 encoding variability plot
- PCA explained variance curves for `im_vectors` and `im_vp_vectors`
- Significant-results gallery

## 5-10. GLM analysis: deferred

원래 neural mechanism claim을 가장 정석적으로 검증하려면 GLM 분석이 필요하다. 예를 들어 NOOP onset, commitment onset, uncertainty, VRE, search disagreement, reward expectancy 등을 event/parametric regressor로 넣고 hippocampus/PFC/striatum/frontal ROI에서 contrast를 보는 방식이다.

하지만 현재 사정상 GLM 설계, nuisance modeling, run별 design matrix 검증, first-level/second-level inference를 안정적으로 진행하기 어렵다. 따라서 본 단계에서는 GLM claim을 유보하고, 다음 분석을 우선한다.

- unsmoothed `wfiltered_func_data.nii` 기반 ROI pattern RSA
- RAM/Thinker/fMRI DSM alignment
- block permutation 기반 통계 검정
- feature-level leave-one-run-out voxelwise encoding

GLM은 추후 별도 단계에서 다음 항목을 정리한 뒤 진행한다.

- event definition: NOOP onset, commitment, search/imaginary interval
- HRF convolution과 temporal derivative
- motion/nuisance/confound regressors
- run-level design quality check
- first-level contrast
- subject/game-level aggregation
- ROI and whole-brain correction strategy
