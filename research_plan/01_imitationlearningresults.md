# 1. Imitation Learning Results: Human–Thinker 행동 및 task-state bridge

*관점: `01_behavioral_analysis.py`가 인간의 행동 보류 현상을 확립한다면, `01_imitationlearningresults.py`는 인간과 imitation/Thinker의 외현적 행동을 직접 비교하고, Thinker 내부 표현이 실제 Atari task state를 담는지 검증한다. 이 스크립트는 canonical human–Thinker bridge이지만, 현재 결과가 직접 지지하는 범위는 **행동 분포 비교**와 **Thinker representation의 RAM decodability**까지다.*

> **현재 구현된 분석 구조**
>
> - **1-IL-1**: Human vs. Thinker action distribution
> - **1-IL-2**: Real step 직전 imaginary representation 구성
> - **1-IL-3**: Thinker feature와 Atari RAM의 step-level alignment
> - **1-IL-4**: Leave-One-Session-Out Ridge decoding
> - **1-IL-5**: Block permutation과 RAM-slot FDR
> - **1-IL-6**: Pong 결과
> - **1-IL-7**: Space Invaders 결과
>
> 현재 확인된 output은 `sub001`의 Pong과 Space Invaders 결과다. 따라서 아래 수치는 **단일 subject 내부의 4-session 일반화 결과**이며, subject 간 일반화 결과가 아니다.

---

## 분석의 위치와 claim 범위

이 스크립트는 현재 연구의 네 개 canonical script 중 두 번째 bridge에 해당한다.

1. `01_behavioral_analysis.py`: 인간 행동만으로 withholding 현상을 확립한다.
2. `01_imitationlearningresults.py`: 인간–Thinker 행동 비교와 Thinker representation의 task-state validity를 확인한다.
3. `02_structure_analysis_cognitiveuncertainty.py`: bout 내부의 cognitive-uncertainty dynamics를 분석한다.
4. `07_encoding_analysis.py`: Thinker/RAM representation과 fMRI representation의 정렬을 검증한다.

이 위치에서 현재 스크립트가 직접 뒷받침하는 주장은 다음 두 가지다.

- 같은 real step에서 기록된 `human_action`과 `thinker_action`의 전체 분포는 게임별로 비교 가능한 형태로 존재하며, Thinker가 인간 행동의 큰 구조를 어느 정도 재현하면서도 잔차 차이를 남긴다.
- real step 직전의 `im_vp_vectors` trajectory에는 새로운 session에서도 Atari RAM state를 선형적으로 복원할 수 있는 정보가 포함되어 있다.

반대로 현재 스크립트만으로는 다음을 주장할 수 없다.

- 잔차 action-distribution 차이가 strategic withholding 또는 uncertainty selectivity 때문이라는 인과 해석
- Thinker representation과 인간의 뇌 representation이 동일하다는 주장
- RAM decoding이 높은 slot이 NOOP 또는 delayed commitment를 직접 유발한다는 주장
- subject 간 일반화 또는 population-level human–Thinker alignment

---

## 공통 정의와 입력

### Real step

`status == 0`인 row를 실제 게임 환경에 action이 실행된 **real step**으로 정의한다. Action distribution과 RAM target alignment 모두 이 real-step 축을 기준으로 한다.

### Action space

현재 action space는 다음 여섯 범주다.

| ID | Action |
|----|--------|
| 0 | NOOP |
| 1 | FIRE |
| 2 | RIGHT |
| 3 | LEFT |
| 4 | RIGHTFIRE |
| 5 | LEFTFIRE |

`human_action`과 `thinker_action`이 1차원 ID이면 그대로 사용하고, one-hot/logit 형태의 2차원 배열이면 `argmax`로 action ID를 얻는다. 유효 범위 0–5 밖의 값은 count에서 제외한다.

### Thinker representation

RAM decoding에는 각 `.npy`의 `im_vp_vectors`를 사용한다. 공간 차원이 있는 경우 마지막 공간 축들을 평균해 step별 vector로 만든다. 각 real step에는 **직전 real step 이후부터 현재 real step 직전까지의 `status == 2` imaginary steps**를 연결한다.

현재 output에서 real step당 imaginary step은 평균적으로 약 **39개**다.

### RAM target

Ground-truth task-state target은 `behavioral_data_block_old/sub_{subject}/game_{game}/day_{session}/block_{block}/RAM.txt`의 128개 Atari RAM byte다. `.npy` chunk와 RAM은 subject–session–block–game 단위로 묶은 뒤, chunk 순서대로 RAM offset을 이동시키며 정렬한다.

---

## 1-IL-1. Human vs. Thinker Action Distribution

### 분석 목적

Human과 imitation/Thinker가 같은 게임에서 어떤 action repertoire를 사용하는지 기술한다. 특히 NOOP 빈도와 overt action 구성의 큰 틀이 재현되는지, 그리고 imitation 이후에도 어떤 잔차 차이가 남는지 확인한다.

### 구현

`test/{subject}/ses-*/*game{game}_*.npy`의 모든 파일을 모은다. 각 파일에서 `status == 0`인 row만 남기고 human/Thinker action count를 합산한 뒤, 총 real-step 수로 나눈 proportion을 grouped bar chart로 그린다.

### 현재 산출물

- `action_dist_sub001_game1.png`
- `action_dist_sub001_game2.png`

현재 stage-out 결과에서 두 agent의 총 sample 수는 서로 같다.

| Game | Human real steps | Thinker real steps | 현재 figure의 패턴 |
|------|-----------------:|-------------------:|--------------------|
| Pong | 135,667 | 135,667 | 양쪽 모두 NOOP 중심이지만, human의 NOOP 비중이 더 높고 Thinker는 RIGHT/LEFT를 더 많이 사용한다. |
| Space Invaders | 276,175 | 276,175 | 양쪽 모두 여섯 action을 사용한다. Human은 NOOP와 RIGHT/LEFT 비중이 상대적으로 높고, Thinker는 FIRE 및 복합 fire action 비중이 상대적으로 높다. |

### 해석

이 figure는 Human과 Thinker가 완전히 다른 action space를 사용하는 것이 아니라, 게임별 action-distribution 구조를 상당 부분 공유하면서도 체계적인 residual 차이를 보인다는 descriptive bridge다. 다만 현재 스크립트는 action distribution에 대해 JSD, confidence interval, session-level paired test 또는 uncertainty-conditional comparison을 계산하지 않는다.

따라서 현재 figure에서 가능한 가장 강한 표현은 다음과 같다.

> **Thinker는 인간 행동 분포의 게임별 큰 구조를 재현하지만, NOOP와 overt action 사이에는 잔차 분포 차이가 남는다.**

이 잔차를 곧바로 human-specific strategic withholding으로 부르려면 session/subject 단위 통계와 uncertainty-conditional selectivity 분석이 추가로 필요하다.

---

## 1-IL-2. Imaginary Representation Construction

### 분석 목적

Thinker가 overt action 사이에 수행하는 imaginary computation을 real-step 단위 feature로 요약한다. 이 feature가 실제 task state를 담는지 RAM decoding으로 검증하기 위한 representation construction 단계다.

### 구현

각 real step 직전의 imaginary sequence를 세 가지 방식 중 하나로 요약한다.

| Feature mode | 구성 | 용도 |
|--------------|------|------|
| `moments` (default) | mean + std + max + last vector + imaginary-step count | trajectory의 중심, 변동성, 극값, 최종 상태를 함께 보존 |
| `last` | 마지막 imaginary vector + imaginary-step count | 최종 search state만 사용하는 간단한 대조 |
| `concat` | imaginary sequence를 padding 후 전부 concatenate + count | trajectory 순서를 가장 많이 보존하는 고차원 대조 |

현재 stage-out 결과는 default `moments` mode다. Pooled `im_vp_vectors`가 128차원이므로 `128 × 4 + 1 = 513`개 feature가 생성된다.

Imaginary step이 하나도 발견되지 않으면 가장 가까운 이전 imaginary step을 사용하고, 그것도 없으면 현재 real-step representation을 fallback으로 사용한다. 현재 fallback rate는 매우 낮다.

| Game | Mean imaginary steps | Mean fallback rate |
|------|---------------------:|-------------------:|
| Pong | 38.997 | 0.0081% |
| Space Invaders | 38.993 | 0.0181% |

### 해석

현재 feature는 real step 자체의 단일 snapshot이 아니라, 그 action 직전의 internal imaginary trajectory를 요약한다. 따라서 decoding 성공은 Thinker의 직전 search process가 task-state 정보를 유지한다는 증거다. 다만 `moments`는 imaginary sequence의 정확한 시간 순서를 대부분 제거하므로, search trajectory의 방향성이나 수렴 과정을 직접 검증하는 representation은 아니다.

---

## 1-IL-3. Thinker–RAM Step Alignment

### 분석 목적

Thinker feature와 Atari RAM이 같은 real step을 가리키도록 정렬하고, decoding 결과가 단순한 파일 길이 mismatch에서 생기지 않았는지 진단한다.

### 구현

- 파일명에서 subject, session, block, game, chunk를 파싱한다.
- 같은 subject–session–block–game의 chunk를 chunk 번호순으로 정렬한다.
- block의 단일 `RAM.txt`에서 chunk별 real-step 수만큼 연속 구간을 사용한다.
- feature와 남은 RAM row 수 중 더 짧은 길이만 사용한다.
- 사용 row, trimmed feature row, 남은 RAM row, mean imaginary steps, fallback rate를 alignment CSV에 저장한다.

### 현재 alignment 결과

| Game | Aligned real steps | Source chunks | Trimmed feature steps | Sessions |
|------|-------------------:|------------------:|----------------------:|----------|
| Pong | 135,667 | 11 | 0 | 1–4 |
| Space Invaders | 276,175 | 50 | 0 | 1–4 |

현재 결과에서는 feature 쪽에서 잘려 나간 real step이 없다. 이는 최소한 기록된 chunk feature가 RAM target에 순서대로 모두 대응했다는 뜻이다. 다만 alignment는 timestamp나 event marker를 이용한 독립적 검증이 아니라 **파일 순서와 row 순서가 일치한다는 전제**에 기반한다.

---

## 1-IL-4. Session-LORO Ridge Decoding

### 분석 목적

같은 session 안의 시간적 인접성을 이용해 RAM을 외우는 것이 아니라, 새로운 session에도 일반화되는 task-state information이 Thinker representation에 있는지 검증한다.

### 구현

기본 session은 1, 2, 3, 4다. 각 fold에서 한 session 전체를 test set으로 남기고 나머지 세 session으로 학습하는 **Leave-One-Session-Out (LORO)**를 수행한다.

각 fold에서 다음 순서를 따른다.

1. Training session만으로 `StandardScaler`를 fit한다.
2. Training/test feature를 같은 scaler로 변환한다.
3. 128개 RAM address를 동시에 예측하는 multi-output Ridge를 fit한다.
4. Held-out session prediction을 원래 row 위치에 저장한다.

현재 기본 regularization은 `Ridge(alpha=10.0)`이다. 네 session 중 하나라도 없으면 분석을 중단하므로, 현재 결과는 네 session이 모두 포함된 엄격한 session-LORO다.

### 평가 지표

RAM address마다 다음 지표를 저장한다.

- Held-out actual–predicted Pearson `r`
- `R²`
- MAE와 RAM value range로 나눈 normalized MAE
- prediction을 0–255 정수로 반올림한 exact accuracy
- majority-value baseline accuracy와의 차이

Primary figure는 Pearson `r`을 사용한다. 따라서 높은 `r`은 시간에 따른 RAM 변화의 추적 성능을 뜻하며, 모든 byte 값을 정확히 맞춘다는 뜻은 아니다. 특히 rounded accuracy가 majority baseline보다 낮은 slot도 있으므로 **temporal tracking**과 **exact state reconstruction**을 구분해야 한다.

---

## 1-IL-5. Block Permutation and FDR

### 분석 목적

RAM과 Thinker feature 모두 강한 temporal autocorrelation을 가지므로, iid row shuffle보다 보수적인 시간 블록 단위 null을 사용한다. 동시에 128개 RAM address를 검정하므로 multiple-comparison correction을 적용한다.

### 구현

- 각 held-out session prediction은 고정한다.
- 실제 RAM row를 held-out fold 안에서 연속 **40-real-step block** 단위로 섞는다.
- permutation마다 actual–predicted Pearson `r`을 다시 계산한다.
- 관측 `r` 이상인 null `r`의 비율로 one-sided p-value를 계산한다.
- 128개 address에 Benjamini–Hochberg FDR을 적용한다.

현재 결과는 `n_perm = 1,000`, `perm_seed = 0`을 사용한다. 따라서 얻을 수 있는 최소 p-value는 `1 / 1001 ≈ 0.000999`다. 강한 slot 다수가 이 최소값에 모이므로, 이들 사이의 유의도 순위를 세밀하게 구분하기보다는 Pearson `r`, `R²`, MAE 같은 effect-size/accuracy 지표를 함께 봐야 한다.

### 주의할 점

- FDR-significant slot 수는 독립적인 semantic variable 수가 아니다. Atari RAM address끼리 동일하거나 강하게 상관된 값이 있을 수 있다.
- 매우 작은 양의 `r`도 큰 sample size에서는 유의할 수 있다. `q < 0.05`만으로 representation quality를 판단하면 안 된다.
- 현재 permutation은 concatenated held-out rows의 block order를 섞는다. Run/block boundary를 별도 계층으로 보존하는 permutation sensitivity analysis는 아직 없다.

---

## 1-IL-6. Pong RAM Decoding Results

### 현재 결과 요약

| 항목 | 값 |
|------|----:|
| Subject | sub001 |
| Sessions | 1–4 |
| Aligned real steps | 135,667 |
| Feature dimension | 513 |
| RAM targets | 128 |
| FDR-significant slots | 27 |

게임 의미가 명시된 key RAM slot은 모두 FDR-significant다.

| Address | Slot | Pearson r | R² | Normalized MAE | FDR q |
|--------:|------|----------:|---:|---------------:|------:|
| 13 | cpu_score | 0.425 | 0.119 | 0.153 | 0.00111 |
| 14 | player_score | 0.793 | 0.589 | 0.113 | 0.00111 |
| 49 | ball_x | 0.844 | 0.712 | 0.064 | 0.00111 |
| 50 | cpu_y | 0.898 | 0.801 | 0.075 | 0.00111 |
| 51 | player_y | 0.857 | 0.733 | 0.077 | 0.00111 |
| 54 | ball_y | 0.885 | 0.780 | 0.085 | 0.00111 |

### 해석

Thinker의 imaginary `im_vp_vectors` summary는 새로운 session에서도 ball과 paddle position을 강하게 추적한다. 특히 `cpu_y`, `ball_y`, `player_y`, `ball_x`가 모두 높은 `r`과 양의 `R²`를 보인다는 점은 단순한 session ID나 고정 baseline을 넘어, Pong의 현재 공간적 state가 representation에 포함되어 있음을 지지한다.

Score도 decoding되지만 `cpu_score`는 position slot보다 약하다. 따라서 현재 Pong 결과의 가장 강한 주장은 **dynamic visuomotor task-state geometry의 보존**이다.

---

## 1-IL-7. Space Invaders RAM Decoding Results

### 현재 결과 요약

| 항목 | 값 |
|------|----:|
| Subject | sub001 |
| Sessions | 1–4 |
| Aligned real steps | 276,175 |
| Feature dimension | 513 |
| RAM targets | 128 |
| FDR-significant slots | 88 |

Primary key slot 네 개도 모두 FDR-significant다.

| Address | Slot | Pearson r | R² | Normalized MAE | FDR q |
|--------:|------|----------:|---:|---------------:|------:|
| 16 | enemies_y | 0.945 | 0.892 | 0.054 | 0.00101 |
| 17 | enemy_count | 0.964 | 0.929 | 0.059 | 0.00101 |
| 28 | player_x | 0.941 | 0.886 | 0.081 | 0.00101 |
| 73 | num_lives | 0.379 | 0.121 | 0.189 | 0.00101 |

추가로 annotation된 alien bitmap과 game-state slot도 강하게 decoding된다.

| Address range/slot | 대표 Pearson r | 해석 |
|--------------------|------------------:|------|
| 18–23, alien row bitmaps | 0.875–0.920 | alien formation의 row-level 구조가 representation에 포함됨 |
| 42, game_state_flags | 0.907 | discrete game-state 변화 추적 |
| 107, alien_alive_column_mask | 0.881 | 살아 있는 alien column 구조 추적 |
| 13, alien_edge_x_limit | 0.863 | alien formation의 horizontal boundary 정보 추적 |

`internal_scan_counter`와 `internal_cooldown_timer`도 높은 correlation을 보이지만, 현재 trace figure에서는 제외된다. 이 값들은 task-relevant state라기보다 emulator/game-engine 내부 진행 변수일 가능성이 있으므로, main interpretation에서는 enemy configuration, player position, lives처럼 명확한 task-state slot을 우선해야 한다.

### 해석

Space Invaders에서는 enemy count, enemy vertical position, player position, alien bitmap/column structure가 모두 held-out session에서 강하게 복원된다. 이는 Thinker representation이 단순 action tendency만 담는 것이 아니라, 다수의 동적 object 및 game-state 변수를 함께 보존한다는 강한 증거다.

다만 88/128 slot의 FDR significance는 representation이 88개의 독립 개념을 각각 명시적으로 부호화한다는 뜻이 아니다. RAM redundancy와 큰 sample size를 고려하면, key/annotated slot의 effect size와 cross-session prediction trace가 더 해석 가능한 근거다.

---

## Figures and Outputs

스크립트 자체의 기본 저장 위치는 다음과 같다.

```text
research_script/outputs/01_imitationlearningresults/
  figures/
    action_dist_{subject}_game{game}.png
    fig_{game}_session_loro_slot_scores_fdr.png
    fig_{game}_session_loro_fdr_volcano.png
    fig_{game}_session_loro_key_fdr_bars.png
    fig_{game}_session_loro_first_run_traces.png
  results/
    {game}_session_loro_alignment_summary.csv
    {game}_session_loro_slot_decoding_scores.csv
    {game}_session_loro_key_slot_summary.csv
    {game}_session_loro_fdr_significant_slots.csv
    {game}_session_loro_key_slot_predictions.csv
    {game}_session_loro_permutation_nulls.npz
    {game}_session_loro_summary.txt
```

현재 보존된 `sub001` 결과는 SLURM stage-out 구조 아래 있다.

```text
research_script/outputs/01_imitationlearningresults_stageout/
  sub001_game1/
    figures/
    results/
  sub001_game2/
    figures/
    results/
```

각 figure의 역할은 다음과 같다.

| Figure | 역할 |
|--------|------|
| `action_dist_*` | Human과 Thinker의 여섯 action proportion 비교 |
| `*_slot_scores_fdr` | 128개 RAM address의 Pearson r과 FDR 상태를 한눈에 표시 |
| `*_fdr_volcano` | effect size `r`과 FDR q-value를 함께 표시 |
| `*_key_fdr_bars` | annotated 또는 FDR-significant slot의 핵심 결과 요약 |
| `*_first_run_traces` | 첫 source run에서 실제 RAM과 held-out prediction의 시간 추적 비교 |

---

## 현재 구현의 한계와 후속 필요

### 1. 단일 subject 결과

현재 stage-out output은 `sub001`만 포함한다. Session-LORO는 새로운 날에 대한 일반화는 검증하지만, 새로운 사람에 대한 일반화는 검증하지 않는다. 동일 pipeline을 S2–S6에 적용하고, key-slot `r`을 subject × game 단위로 요약해야 한다.

### 2. Imitation provenance 확인 필요

스크립트는 `.npy`의 field 이름인 `thinker_action`을 읽을 뿐, 어떤 checkpoint와 imitation-training configuration에서 생성되었는지는 검증하지 않는다. Human vs. **IL Thinker**라는 논문 수준 표현을 쓰려면 data-generation manifest와 checkpoint provenance를 별도로 고정해야 한다.

### 3. Action distribution의 통계가 없음

현재 action figure는 pooled count만 사용한다. Session/episode를 독립 단위로 둔 confidence interval, paired comparison, JSD 및 subject-level aggregation이 필요하다. Pooled real-step 수를 그대로 독립 표본처럼 검정하면 temporal dependence 때문에 과도하게 작은 p-value가 나올 수 있다.

### 4. Selectivity 분석이 없음

현재 action comparison은 marginal distribution만 본다. 논문의 human-specific strategic withholding claim을 강화하려면 같은 uncertainty 또는 matched task-state bin에서 다음을 비교해야 한다.

- `P(NOOP | uncertainty bin)`의 Human–Thinker 차이
- 동일 RAM state 또는 propensity-matched state에서의 NOOP probability
- Human–Thinker의 bout onset/length/commit structure 차이
- Session별 action-distribution residual과 performance의 관계

### 5. RAM decoding은 task-state validity이지 human alignment가 아님

RAM decoding은 Thinker representation이 task state를 포함한다는 중요한 prerequisite다. 그러나 human fMRI와의 representation alignment는 `07_encoding_analysis.py`의 RSA/LORO encoding에서 별도로 검증해야 한다. 이 두 단계를 섞으면 circular한 claim이 된다.

### 6. Output namespace 충돌 가능성

스크립트 내부 result filename에는 subject ID가 들어가지 않는다. 같은 output directory에서 여러 subject의 같은 game을 순차 또는 병렬 실행하면 result CSV와 RAM figure가 덮어써질 수 있다. 현재 SLURM wrapper는 subject/game별 stage-out directory로 이를 피하지만, direct run을 확장할 때는 output path 자체를 `{subject}_game{game}` 단위로 분리하는 것이 안전하다.

### 7. Action-only 실행 경로가 없음

현재 `main()`은 action distribution을 저장한 직후 RAM decoding을 항상 실행한다. 빠른 행동 figure 재생성이나 RAM data가 없는 dataset을 위해 `--skip-ram-decoding` 같은 option을 두는 것이 유용하다.

---

## 이 분석의 통합 해석

현재 `01_imitationlearningresults.py` 결과가 만드는 human–Thinker bridge는 두 층으로 정리할 수 있다.

1. **Behavioral bridge**: Human과 Thinker는 게임별 action repertoire의 큰 구조를 공유하지만, NOOP와 overt action 구성에는 잔차 차이가 남는다.
2. **Representational bridge prerequisite**: Thinker의 real-step 직전 imaginary representation은 새로운 session에서도 object position, score, enemy configuration, lives 등 실제 Atari task state를 선형적으로 복원한다.

Pong에서는 ball/paddle geometry, Space Invaders에서는 enemy/player configuration이 특히 강하게 decoding된다. 따라서 Thinker internal representation을 이후 cognitive-uncertainty 분석과 fMRI alignment 분석에 사용하는 것은 task-state validity 측면에서 정당화된다.

그러나 현재 결과의 최종 문장은 다음 수준으로 제한하는 것이 정확하다.

> **Thinker는 인간과 비교 가능한 행동 분포를 생성하며, 그 imaginary representation은 session을 넘어 일반화되는 풍부한 Atari task-state 정보를 담는다. 이 결과는 human–Thinker comparison과 후속 neural alignment를 위한 bridge를 제공하지만, human-specific strategic withholding이나 shared neural computation 자체는 후속 conditional behavior 및 fMRI 분석에서 별도로 검증해야 한다.**
