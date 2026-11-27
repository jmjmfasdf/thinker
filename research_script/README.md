# Adaptive future computation and continuous behavioral control

이 디렉터리는 **계산의 배분 → 평가의 갱신 → 행동 제어의 조직화 → 신경 표현**을 검증합니다.
관측된 버튼 입력, 모델의 내부 계산, 인간의 의사결정은 서로 다른 측정 수준입니다.
Imagination은 인간의 사고 자체가 아니라, 행동과 뇌 활동에 대해 검증할 모델 설명변수입니다.

## VoC 관계 분석 확장 (03, 05–10)

[별도 계획·지표 정의](VOC_ANALYSIS_PLAN.md)에 따라 기존 분석에
`선택 전 VoC → 실제 계산에서의 갱신 → 다음 VoC/선택 → 인간 행동` 관계를 추가했습니다.
04의 실제 효용 평가와 11–13 신경 분석은 이번 확장에서 제외합니다.

- visualization과 해시가 맞는 원래 checkpoint/config로 동일 단계 readout을 복원합니다.
  원본 환경을 실행하거나 STOP/CONTINUE로 분기하지 않습니다.
- 기존 번호 진입점을 그대로 실행하면 새 `voc_*` 표와 그림도 생성됩니다.
  `--utility-input`, `--forecast-input`, `--rich-reuse`는 새 분석의 필수 입력이 아닙니다.
- `voc_choices`, `voc_transitions`, `voc_decisions`는 선택 직전·실제 계산 전후·전체 planning을
  구별합니다. `last_choice_voc`와 `endpoint_voc_final`은 특히 상한 종료에서 서로 다릅니다.
- 05는 첫 강제 계산과 자발적 계산의 관계를 분리합니다. 06은 실제 다음 자발적 선택이
  있을 때만 중단 관계를 평가하며, 07은 동일 held-session에서 변수군 제외 비교를 합니다.
  09의 `voc_full / voc_no_voc / voc_no_direction`도 동일 지원 표본으로 비교합니다.
- 10의 패턴 비교는 evaluation view, 0.5초 창을 사용하고 참가자·held fold의 패턴 번호를
  서로 대응시키거나 합쳐 평균하지 않습니다. 패턴으로 나눈 관계 역시 기술적 비교입니다.
- 공통 숫자 캐시는 출력 루트의 `.voc_cache/00/subNNN/game`에 자동 생성합니다. 표본 모드가 바뀌면
  해당 캐시의 서명을 다시 검증하고 필요한 부분을 재생성합니다. 캐시가 없어도 원자료에서
  재생성되며, 기존 `outputs/00` 수치 표는 이 추출기가 덮어쓰지 않습니다.
  각 결과의 `logs/run.json`에 복원 오차·입력 설정·추정 지원 상태가 기록됩니다.
- 그림은 큰 축·범례, 제목 없는 300-dpi PNG/PDF이며 `--render-only`는 저장된 표만 읽습니다.

gate 일치는 실행 검증이고, 큰 가치 갱신은 실제 보상 개선의 증거가 아닙니다.
새 분석은 VoC의 인과적 효용·최적성을 검증하지 않으며, 기존 Thinker가 모든 세션으로
학습됐다는 제한도 유지합니다. 전체 참가자 분석 job은 별도로 실행해야 합니다.

## 분석 순서

### 원본 순서 capture와 분석 표본 (2026-11-26)

`visual_behav.py`는 이제 원본 관측을 건너뛰지 않고 순서대로 replay합니다.
frame stack도 인접 원본 관측으로 구성합니다. 각 episode의 첫 전이부터 저장하며,
첫 두 전이는 `score_mask=False`로 남깁니다. 마지막 관측은 전이의 도착점이므로
존재하지 않는 다음 관측을 만들어 마지막 행의 action을 평가하지 않습니다.
reset/terminal/시계 역행/기존 1/6초 초과 gap에서는 episode를 분리합니다.
학습 loader의 15Hz 기본값이나 checkpoint는 변경하지 않습니다.

새 기본 입력은 `logs/visualization/hxln_10m_allframes`이고, 새 기본 출력은
`research_script/outputs`입니다. 원본 capture와 고정된 실행 소스는 보존합니다.

- 00의 `tables/decisions.csv.gz`, `choices.csv.gz`는 전체 **분석 적격** 자료입니다.
  `real_steps.csv.gz`에는 비채점 초기 전이까지 포함한 모든 저장 real transition의
  원본 frame/time/score mask가 남습니다. 모든 native tensor/event는 원본 Zarr에 유지됩니다.
- 00은 `decisions_15hz.csv.gz`, `choices_15hz.csv.gz`도 함께 만듭니다.
  `sampling_15hz.csv.gz`에는 grid time, 원본 frame/time, 지연, 분석 적격 여부를 저장합니다.
  각 episode의 첫 관측 시각을 원점으로 1/15초 격자를 만들고, 격자를 넘지 않는
  가장 최근 관측을 선택합니다. 중복 선택된 동일 decision은 분석에서는 한 번만 사용합니다.
  부적격 관측이 선택되면 다른 프레임으로 대체하지 않습니다. 따라서 분석 행 수는
  정확히 초당 15개가 아닐 수 있습니다. 선택 규칙/버전은 `logs/run.json`에 기록됩니다.
- 공식 01–13은 `--sample-mode all`(기본) 또는 `--sample-mode 15hz`를 받습니다.
  01은 모델과 독립적으로 원본 행동자료에 같은 causal 선택 규칙을 적용하고 bout를 다시
  구성합니다. 02–13은 00의 선택 decision과 그에 속한 전체 imagination을 사용합니다.
  원본 행동 outcome/전환 시각은 임의로 축약하지 않습니다. 09·10의 feature 배열과
  endpoint 인덱스도 같은 선택으로 정렬하며, 누락된 지원 구간은 연결하지 않습니다.
- 04의 독립 환경 utility 표는 인간 replay 시계가 아닙니다. `15hz` 선택은 matching
  `capture_manifest_sha256`와 `episode,real_event`가 있을 때만 가능하고, 없으면
  명시적으로 `BLOCKED_INPUT`입니다. 새로운 STOP/CONTINUE 실험을 실행하지 않습니다.
- 01 이후 결과는 `all`과 `15hz` 모두 **`outputs/NN/subNNN/game` 및 `outputs/NN/group/game`**에
  저장합니다. 별도의 `native_allframes`나 `samples/<mode>` 폴더를 만들지 않습니다.
  같은 번호·참가자·게임을 다른 표본 모드로 실행하면 검증된 새 결과가 기존 결과를 교체합니다.
  현재 표본 모드는 `logs/run.json`에 기록합니다. 실패한 실행은 이전 성공 결과를 지우지 않습니다.
- `--aggregate`와 `--render-only`에는 저장된 결과와 같은 `--sample-mode`를 지정합니다.
  서로 다른 모드의 개인 결과를 그룹 집계에 섞거나 다른 모드의 결과를 잘못 재그리는 것은 거부합니다.
  표본을 바꾸려면 먼저 개인 분석을 실행해야 합니다.
- 00은 모드와 무관하게 공통 `outputs/00/subNNN/game` 아래에 전체·15Hz 표와 선택 규칙을 모두 보존합니다.
  01의 `--legacy-output-layout`은 이전 명령 호환용으로만 남아 있으며 이제 아무 효과가 없습니다.

```bash
python research_script/00_native_preparation.py --subjects 1 2 3 4
python research_script/02_model_validation.py --subjects 1 2 3 4 --sample-mode all
python research_script/02_model_validation.py --subjects 1 2 3 4 --sample-mode 15hz
```

전체 replay를 사후 15Hz로 선택한 값은 입력부터 15Hz였던 과거 replay와 다릅니다.
중간 원본 관측에서도 RNN·트리가 갱신되기 때문입니다. 과거 capture를 명시적으로
입력하면 `all`도 그 capture에 존재하는 관측만 뜻하며, 누락된 프레임은 복원하지 않습니다.
단위시간당 impulse 합·짧은 행동 bout·누적 변화는 표본 모드에 따라 달라질 수 있으므로
두 모드를 동일한 값의 재표현으로 해석하지 않습니다. 과거 baseline의 집계 정확도도
자동으로 같은 평가 시점에 정렬되는 것은 아닙니다.

아래 파일들이 현재 공식 진입점입니다. 00은 전처리만, 01은 모델 없이 수행하는 인간 행동 분석만 담당합니다.

| 번호 | 진입점 | 질문과 주요 분석 |
| --- | --- | --- |
| 00 | `00_native_preparation.py` | 완료된 native capture의 출처·체크포인트·원본 행동 정렬을 검증하고 공통 수치 표를 준비 |
| 01 | `01_behavioral_analysis.py` | 원본 버튼 시계열, 행동별 bout 길이·생존곡선·전환 구조; Atari와 Mario 지원 |
| 02 | `02_model_validation.py` | 실제 기록된 REAL 정책의 사람 행동 확률·정확도·보정과 학습 조건·학습 곡선 확인 |
| 03 | `03_computation_allocation.py` | 상황별 계산량·깊이, 선택 전 VoC와 자발적 계속/중단, RAM·화면 상태와 계산 배분의 관계 |
| 04 | `04_computation_utility.py` | 기존의 짝지은 STOP/CONTINUE 환경 평가에서 추가 계산의 실제 효용과 VoC 예측력 |
| 05 | `05_evaluation_revision.py` | 상상 중 V/Q의 절대 갱신량·방향·계산당 갱신량과 관측 범위 |
| 06 | `06_search_to_policy.py` | 검색으로 얻은 평가, 이후 계산 제어, 실제 REAL 정책 변화의 연결 |
| 07 | `07_computation_to_behavior.py` | persist/switch box plot, COSYNE 방식의 매끄러운 관계 지도, 행동·전환에 대한 추가 예측력 |
| 08 | `08_evaluation_continuity.py` | 실제 관측 사이의 평가 연속성·변화와 이후 행동; 검증된 예측 자료가 있으면 도착 시 불일치 추가 |
| 09 | `09_evaluation_accumulation.py` | 계산이 지지하게 된 행동이 미래 첫 전환의 도착 행동을 예측하는가; 기존 표현 방향·전체 bout 분석도 보존 |
| 10 | `10_control_patterns.py` | SRN/VPN·actor의 기능적 ACE, 시간 창, 평가 패턴의 유지/전환, 행동 및 계산량 예측; control motif 후보 검증 |
| 11 | `11_neural_encoding.py` | 계산 배분·평가 갱신이 BOLD에 추가 설명력을 갖는가 |
| 12 | `12_neural_decoding.py` | ROI 활동으로 계산 관련 변수를 추가로 예측할 수 있는가 |
| 13 | `13_neural_rsa.py` | 화면·행동·초기 상태·계산량을 통제한 표현 구조의 대응 |

03의 상태 공간 기하 분석은 관측된 RAM 변화의 요약입니다. 인과적 controllability 또는
인지적 불확실성으로 해석하지 않습니다. 10의 held-session budget prediction은 기능적
표현을 필요로 하므로 10에 유지했습니다. 02의 게임 점수는 기존 학습 로그를 읽는 것으로,
새 환경 점수 평가를 실행하지 않습니다.

`00_cache.py`는 00의 호환 진입점입니다. `00_brain_cache.py`는 검증된 BOLD 정렬과
수치 design만 준비합니다. `00_prepare.py`와 atlas 준비 도구는 필요한 원자료 전처리 도구입니다.

## 기본 입력과 저장 구조

모델 분석의 기본 입력은 다음 최신 native 10M visualization입니다.

```
logs/visualization/hxln_10m_allframes/sub001/pong/
logs/visualization/hxln_10m_allframes/sub001/enduro/
logs/visualization/hxln_10m_allframes/sub001/spaceinvaders/
```

다른 참가자도 같은 구조입니다. 기본 실행 대상은 sub001·sub002의 세 Atari 게임이며,
`--subjects 1 2 3 4 5 6`으로 범위를 지정합니다. 완료되지 않았거나 출처가 맞지 않는 자료는
대체 자료로 채우지 않습니다. 01은 원본 행동 자료를 직접 읽으며 모델/capture가 필요 없습니다.

결과 위치는 반복 실행해도 같습니다.

```
research_script/outputs/
  00/sub001/pong/
    tables/decisions.csv.gz
    tables/choices.csv.gz
    tables/real_steps.csv.gz
    tables/sampling_15hz.csv.gz
    tables/decisions_15hz.csv.gz
    tables/choices_15hz.csv.gz
    logs/run.json
    logs/run.log
  07/sub001/pong/
    figures/*.png
    figures/*.pdf
    tables/*.csv.gz
    tables/*.npz
    logs/run.json
    logs/run.log
  07/group/pong/
    figures/...
    tables/...
    logs/...
  .voc_cache/00/sub001/pong/...
```

- PNG/PDF와 재그리기·통계에 필요한 CSV, 필요한 수치 NPZ만 공개 결과로 저장합니다.
- 원본 영상·전체 activation·체크포인트·학습 모델 복제본은 결과 폴더에 쌓지 않습니다.
- 해시와 설정은 `logs/run.json`에 기록하며, 폴더 이름에 날짜·해시·난수를 붙이지 않습니다.
- 임시 계산은 임시 디렉터리에서, 결과 발행용 작업과 잠금은 `outputs/.work/`에서 관리합니다.
- 기존 성공 결과는 새 결과 검증 뒤 교체합니다. 실패하면 기존 성공 결과를 보존합니다.
- 반복 실행마다 결과 archive를 만들지 않습니다. 개편 전 결과의 보관은 아래의 일회성 이관으로 분리합니다.
- 그룹 결과는 게임별로 분리하고 run별 요약을 참가자 안에서 평균한 뒤 참가자를 동일 가중합니다.
  요청한 참가자의 입력 누락 및 유효 표본 수를 보고합니다.
  10의 기존 switch-aligned 표는 held-session 평균까지 저장하므로, 이 공통 scalar
  곡선은 참가자 안에서 세션을 동일 가중한 뒤 참가자를 평균합니다. ACE 축과 motif
  번호는 참가자·fold 사이에서 정렬되지 않아 합쳐 평균하지 않고 별도 패널로 표시합니다.

`logs/run.json`의 `status`는 파일 생성 단계의 상태입니다. `scientific_status`는
분석의 지원 수준을 따로 나타냅니다. 예를 들어 파일 저장은 `COMPLETED`여도
회귀 일부만 추정됐다면 `PARTIAL_FITS`, 정보가 부족하면 `INSUFFICIENT_SUPPORT`입니다.
누락 입력은 `BLOCKED_INPUT`으로 기록하고, 실패를 빈 성공 테이블로 저장하지 않습니다.

## 실행과 재그리기

### 공통 그림 스타일

현재 공식 01–13의 참가자별·그룹별 그림은 `common/figure_style.py`의 동일한
저장 규칙을 적용합니다. 00은 그림을 만들지 않습니다.

- 축 눈금·범례 22pt, 축 이름 26pt; 컬러바 눈금과 과학적 표기의 지수도 포함합니다.
- 대제목·패널 제목과 그림 아래 작은 부연설명은 넣지 않습니다. 축 이름, 범례,
  패널 안의 데이터 주석은 유지하며, 분석 조건·표본 범위·해석 제한은 표/QA/문서에 둡니다.
- PNG는 300dpi, PDF는 편집 가능한 TrueType 글꼴이며 큰 글자에 맞춰 패널 공간을 확보합니다.
- 기존 호환 번호 CLI와 독립 13/13_1/14 렌더러도 같은 스타일을 적용합니다.
- 스타일 변경은 수치 분석 캐시를 무효화하지 않고 렌더 단계만 갱신합니다.
  이미 완료된 그림은 소스 수정만으로 바뀌지 않습니다. 재생성하려면 아래의
  `--render-only`를 명시적으로 실행합니다. 동결된 과거 source snapshot은 수정하지 않습니다.

### 명령 예시

프로젝트 루트에서 dynamic 분석 환경을 사용합니다.

```bash
# 공통 전처리. 02–13도 필요할 때 이를 호출합니다.
/home/jeongmin/miniconda-dynamic/envs/dynamic-thinker/bin/python -B research_script/00_native_preparation.py --subjects 1 2 --games pong enduro spaceinvaders --resume

# 모델이 필요 없는 행동 분석
/home/jeongmin/miniconda-dynamic/envs/dynamic-thinker/bin/python -B research_script/01_behavioral_analysis.py --subjects 1 2 --games pong enduro spaceinvaders mario --resume

# 선택한 연구 단계 실행
/home/jeongmin/miniconda-dynamic/envs/dynamic-thinker/bin/python -B research_script/orchestration/run_themes.py --themes 02 03 05 06 07 08 --subjects 1 2 --games pong enduro spaceinvaders --workers 2 --resume

# 02: 저장된 baseline 비교를 추가 (참가자별 분석 후 그룹 그림)
python -B research_script/02_model_validation.py --subjects 1 2 3 4 --games pong enduro spaceinvaders --baseline-root research_script/outputs/00_baseline_comparison/cluster_results_13219/action_metrics --resume
python -B research_script/02_model_validation.py --subjects 1 2 3 4 --games pong enduro spaceinvaders --aggregate --resume

# 표를 바꾸지 않고 저장된 자료만으로 그림 재생성/재사용
/home/jeongmin/miniconda-dynamic/envs/dynamic-thinker/bin/python -B research_script/07_computation_to_behavior.py --subjects 1 2 --games pong enduro spaceinvaders --render-only

# 참가자별 분석 뒤 게임별 동일 가중 그룹 요약
/home/jeongmin/miniconda-dynamic/envs/dynamic-thinker/bin/python -B research_script/07_computation_to_behavior.py --subjects 1 2 --games pong enduro spaceinvaders --aggregate --resume
```

`--stage prepare`, `--stage analyze`, `--stage render`로 단계를 나눌 수 있습니다.
완료 단계 재사용과 기존 결과 교체는 기본 동작이며 `--resume`을 따로 지정할 필요가 없습니다.
입력·설정·관련 코드·저장 파일의 해시가 같은 단계만 재사용하고, 모드가 바뀌면 같은 경로에서 다시 분석합니다.
그림 코드만 바뀌면 수치 분석을 재실행하지 않습니다. `--render-only`는 저장된
표/수치 배열만 사용하며 원본 capture, 행동 자료, 체크포인트를 열거나 모델을 다시
맞추지 않습니다. 00은 그림을 만들지 않는 전처리 단계입니다.

`--input-root`는 다른 native capture, `--outputs-root`는 별도의 결과 부모 폴더를 지정합니다.
`--behavior-root`로 원본 행동 자료를, `--training-root`로 학습 기록을 지정할 수 있습니다.
04는 `--utility-input`으로 실제 짝지은 평가 CSV와 출처 manifest를 명시해야 합니다.
08의 도착 시 예측 불일치에는 `--forecast-input`의 검증된 예측이 필요합니다.
입력이 없으면 해당 추정값은 결측/지원 불가로 기록하며 다른 지표로 대신하지 않습니다.
10은 `--rich-reuse`로 완료된 기존 rich extraction을 지정할 수 있습니다. 이 경우에도
capture·checkpoint·artifact hash를 확인한 뒤 수치 자료만 재사용합니다.

이 실행 경로들은 학습, visualization 생성, 환경 점수 평가를 제출하지 않습니다.
01–13 전체 실행은 무거울 수 있으므로 필요한 번호와 참가자·게임을 명시하는 것이 좋습니다.

### 02의 저장된 baseline 비교

`--baseline-root`에는 `run_metrics.csv`, `manifest.json`, `qa.json`이 있는 디렉터리를 지정합니다.
요청된 비교 입력은 명시적 예외로 사용하며, 원본 baseline 파일은 수정하지 않습니다.
기존 Thinker 점수는 사용하지 않고 DQN+BC, DRC+BC, fixed-rollout+BC와 현재 native
Dynamic Thinker의 최종 REAL 정책을 비교합니다. 참가자·게임·seed·run 중복 및 run 범위를
검증하고 입력 파일 SHA256을 `baseline_comparison_sources`와 실행 로그에 기록합니다.
현재 비교는 집계 CSV를 그대로 나란히 놓지 않고, baseline 시점별 확률과 native REAL
출력을 **정확히 같은 `(run, source_frame)`**에서 다시 집계합니다.
`--baseline-predictions-root`는 `subNNN_gameG.npz`가 있는 `aligned_arrays` 디렉터리이며,
생략하면 `--baseline-root` 옆의 `full_data_scope_audit/aligned_arrays`를 사용합니다.
이 NPZ의 원본 frame index·시간·현재/직전 행동을 native 및 원본 행동 자료와 검증하고,
세 baseline의 확률이 기존 run별 accuracy·LL·표본 수를 재현하는지 먼저 확인합니다.
과거 Thinker의 `p_thinker`는 사용하지 않습니다. 시점별 파일이 없으면 집계 비교로
조용히 대체하지 않고 실패합니다. 현재 archive는 baseline seed 0만 지원합니다.
`baseline_comparison_alignment`에 run별 공통/제외 개수를,
`baseline_comparison_decisions`에 공통 frame과 네 모델의 정답 여부를 보관합니다.
`all`에서는 baseline의 모든 시점을 포함해야 하며, `15hz`에서는 별도 15Hz native
선택과의 교집합으로 **네 모델 모두** 재집계합니다. 다른 02 지표의 전체 표본은 유지합니다.

그림 `human_action_four_model_boxplot.png/pdf`는 **action accuracy만** 보여줍니다.
y축은 백분율이 아닌 **0–1 accuracy**이며, 라벨은 `Accuracy`입니다.
관측 최솟값보다 낮은 0.05 단위에서 시작하며, 생략한 하단 구간을 물결 표시합니다.
0.6 미만의 관측값도 자르지 않고 전부 표시합니다. 개별 참가자 그림은 seed 평균을 낸 run 점입니다.
**그룹 그림은 참가자별 subfigure(4명이면 2×2)**로 나누며, 각 패널의 네 모델 box는
해당 참가자의 episode/run별 accuracy를 표시합니다. 모든 패널은 같은 y축 범위를 사용합니다.
여기서 episode/run은 baseline과 공유하는 **원본 행동 파일(session/block/game)** 단위로,
Atari life나 개별 프레임을 새 독립 episode로 세지 않습니다. 현재 두 지원 게임은 개인당 11개 run입니다.
요청한 참가자만 포함하며, baseline의 다른 참가자나 과거 Thinker를 섞지 않습니다.
대제목·하단 작은 설명 없이 참가자 식별자만 패널에 표시합니다. 원본 run별 점수, 참가자별 점수, 표본 수와 지원
범위도 CSV로 저장하므로 `--render-only`는 baseline 원자료를 다시 읽지 않습니다.

현재 저장된 baseline은 Pong·Space Invaders만 포함합니다. Enduro는 기존 02 분석은
수행하되 baseline 비교 지원 불가를 coverage/QA에 기록하고 4모델 그림을 만들지 않습니다.
이 그림은 동일 decision의 paired 비교이지만 **학습 자료와 겹치는 기술적 행동 적합도
비교**입니다. 평가 시점을 맞추어도 학습량·계산량을 통제한 비교나 held-out 일반화
검증이 되지는 않습니다. 원본 행동 파일의 현재 SHA256을 기록하되, archive에 없는
과거 이미지 파일 hash의 일치까지 검증했다고 주장하지 않습니다.

`baseline_comparison_accuracy_tests.csv.gz`에는 6개 모델 쌍의 정확도 차이(percentage point),
독립 단위 수, 양측 exact paired sign-flip p값, Holm 보정 p값을 저장합니다.
그룹에서는 참가자별 평균을 짝짓고, 개인에서는 같은 session의 모든 run 차이를 함께
부호 반전합니다(효과 크기는 run 수 가중 session 기여량으로 기존 equal-run 평균과 일치).
프레임이나 seed를 독립 표본으로 늘리지 않습니다. 검정은 단위 간 독립성과 귀무가설하
차이 부호의 교환 가능성을 가정합니다. 평가 시점은 같지만 학습 자료와 겹치므로
공통 시점에서의 집계 정확도 차이에 관한 **탐색적 검정**으로 해석합니다.
기존 참가자 수준 검정과 개별 그림은 6개 비교를 Holm 보정하고 `*` p<.05, `**` p<.01, `***` p<.001,
비유의는 `ns`, 독립 단위 2개 미만은 `NA`로 표시합니다. 참가자 4명의 양측 exact 검정은
최소 raw p값이 .125입니다. 이 검정 결과는 그대로 보존하지만, 새 그룹 그림의 별표에는 사용하지 않습니다.

그룹 패널의 별표는 `baseline_comparison_episode_accuracy_tests.csv.gz`의 **개인 내 episode-paired**
양측 exact sign-flip 검정입니다. 같은 run·동일 프레임에서 평가한 두 모델 accuracy의 차이를
run별 동일 가중하여 검정하며, **각 참가자/게임 패널의 모델 6쌍**을 Holm 보정합니다.
seed·프레임 수나 다른 참가자의 run으로 표본 수를 늘리지 않습니다. 단, 같은 session의
episode들은 의존할 수 있으므로 이는 **episode 독립성 가정하의 탐색적 비교**이지
참가자 모집단에서의 유의성이나 held-out 일반화 증거가 아닙니다. session을 함께 부호 반전하는
보수적 민감도 검정도 `baseline_comparison_session_accuracy_tests.csv.gz`에 저장합니다.
episode 검정에서 별표가 나와도 session 검정에서는 나오지 않을 수 있습니다.
검정은 analyze 단계에서 저장하며, render-only는 저장된 검정 결과만 읽습니다.
기존 개인 02 결과가 있으면 `02_model_validation.py --subjects 1 2 3 4 --games pong spaceinvaders
--sample-mode all --aggregate`로 새 group 표와 그림만 갱신할 수 있습니다. 원본 replay 재추출은 필요 없습니다.

## 계산량·평가 갱신의 정의

한 실제 관측에서 수행한 계산 횟수를 B라 하고, 동일한 SEARCH 단계에서 비교 가능한
계산 전후의 task critic을 V₀,…,V_B라 둡니다.

```
A_V = sum(k=1..B) |V_k - V_(k-1)|   # 누적 절대 갱신량
U_V = A_V / B                       # 실제 계산 1회당 절대 갱신량
D_V = V_B - V_0                     # 방향을 보존한 순 변화
```

첫 imagination도 갱신에 포함합니다. B=0이면 이 계산 내 총량·평균·순 변화는 0입니다.
V가 높다는 것, A_V/U_V가 크다는 것, VoC가 높다는 것은 각각 상태 가치,
실제로 일어난 평가 변화, 앞으로의 추가 계산에 대한 기대라는 서로 다른 의미입니다.

Q는 같은 root-action 통계가 방문된 상태에서 다시 관측됐을 때의 변화만 집계합니다.
방문하지 않은 대안의 placeholder를 실제 가치 0으로 간주하지 않습니다.
A_Q/U_Q와 함께 유효 update 수·coverage를 제시하므로 V와 Q의 관측 범위는 다릅니다.
M은 기존 정의인 초기 정책의 incumbent log probability와 최선 경쟁 행동의 차이입니다.
`K = 4 U_V sigmoid(M) sigmoid(-M)`는 파생 상호작용 지표이며 학습된 VoC가 아닙니다.

STOP 입력으로 인한 변화는 계산 자체의 변화와 분리합니다. 정책 비교는 같은 REAL
단계에서 수행합니다. 실제 관측 사이의 변화와 한 observation 안에서의 갱신도 구분합니다.
계산 제어 분석에는 선택 이전 VoC를 사용하고 강제 최소 계산을 자발적 계속에 포함하지
않습니다. 상한에 도달한 CONTINUE는 실제로 없었던 STOP으로 바꾸지 않습니다.

## 09: 행동별 지지 변화의 전향적 검증

09의 새 주검증은 “계산량이 많으면 전환한다”가 아니라 **“계산이 지지하게 된
특정 행동으로 이후 인간이 전환하는가”**입니다. 기존 표현 방향·bout 분석과
`voc_*` 결과는 그대로 유지하며 새 표·그림은 `action_support_*`로 구분합니다.

- 원래 checkpoint의 동일 REAL-phase 정책 head로 계산 전·후 logits를 복원합니다.
  `voc_decisions`의 `policy_logit_pre_<a>` / `policy_logit_post_<a>`는 확률이 아닌 logits입니다.
  두 벡터만 decision당 추가 저장하며 큰 imagination feature를 중복 보존하지 않습니다.
- 현재 인간 행동 a에 대해 후보 b의 상대 지지 변화는
  `d(b) = (z_post(b)-z_post(a)) - (z_pre(b)-z_pre(a))`입니다. 미래 정답을 보고
  부호를 뒤집거나 표현 축을 고르지 않습니다.
- 기본 입력은 같은 버튼을 유지한 과거 0.5초입니다. `[t-0.5,t)`에서 각 값을
  다음 관측 직전까지 유지하는 시간 가중평균을 사용합니다. 입력과 미래 구간이
  서로 겹치지 않는 landmark를 시간·입력 적격성으로 선택합니다.
- 미래 0.5초의 첫 전환 도착 버튼이 정답이며, 전환이 없으면 현재 버튼입니다.
  원본 행동에서 전환 후 복귀까지 확인합니다. reset/terminal/시간 단절·미래
  관찰 부족은 검열이며 “유지”가 아닙니다. `all`과 `15hz` 모두 같은 원본 정답을 사용합니다.
- 시간 평균 pre/post 상대 logits의 softmax를 그대로 비교합니다. 미래 정답의
  `log(q_post)-log(q_pre)`가 양수면 계산 후 지지가 더 잘 맞습니다.
  **주지표는 실제 전환 구간에서 현재 버튼을 제외하고 재정규화한 목적지 log-score 차이**입니다.
  전체 구간의 유지/전환 목적지 점수는 따로 보고하며 accuracy는 0–1입니다.
  정확한 최대값 동률은 균등 분배하여 행동 ID의 임의적 순서가 점수를 바꾸지 않게 합니다.
  조건부 목적지 점수만 개선되면 “대안 사이의 순위”가 맞아진 것일 수도 있습니다.
  실제 도착 행동의 `destination_relative_update`와 `destination_reinforced_fraction`을
  함께 저장하여 현재 행동에 비해 실제로 지지가 강화되었는지도 구분합니다. 이 둘은
  정답을 사용한 평가 지표이며 예측 입력에 넣지 않습니다.
- 보조 readout은 행동 이력·원본 bout age, 시각 상태와 `visual_change`, 과거 보상,
  초기 정책 지지/불확실성, 초기 VoC·계산량·carry, 부호 없는 갱신과 `abs(d(b))`를
  통제합니다. 방향 모델은 여기에 **모든 후보 행동에 공통인 하나의 자유 계수**로
  `d(b)`를 추가합니다. 음의 계수로 예측이 개선되면 원래 가설의 지지가 아닙니다.
- readout은 session holdout, 학습 session만의 결측 처리·표준화, 고정 L2=.1을 사용합니다.
  비교 모델은 동일 평가 행을 사용합니다. 표본 부족/비수렴은 0 효과와 구분합니다.
  주검증 자체는 방향 계수나 인간 정답을 학습하지 않습니다.
- 개인 점수는 run → session 순으로 균등 평균하고 30초 블록 bootstrap을 사용합니다
  (기본 1,000회, seed 701). Group은 다시 참가자 균등 평균을 사용합니다.
  Group의 mean ± 1.96 SEM은 소수 참가자의 기술적 요약이지 모집단 기전의 검증이 아닙니다.
  서로 다른 protocol ID/표본 모드가 섞이면 group 집계를 중단합니다.

기존 번호 진입점으로 추출부터 그림까지 실행하며 별도 중간 파일 입력은 없습니다:

```bash
python -B research_script/09_evaluation_accumulation.py --subjects 1 2 3 4 --games pong enduro spaceinvaders --sample-mode all
python -B research_script/09_evaluation_accumulation.py --subjects 1 2 3 4 --games pong enduro spaceinvaders --sample-mode all --aggregate
```

표: `action_support_windows`, `predictions`, `coverage`, `protocol`, `fit_qa`,
`run_scores`, `scores`, `destination_counts` (모두 같은 접두사).
Group은 `participant_scores`, `group_scores`, `participant_destinations`,
`group_destinations`를 추가합니다. 원자료 없이 저장 표만으로 재렌더링 가능합니다.
핵심 그림은 `action_support_forecast_gain`, `action_support_direction_increment`,
`action_support_destination_matrix`입니다. 축/범례 22–26pt, 제목·그림 아래 설명 없음입니다.

저장은 기존 `outputs/09/subNNN/game` 및 `outputs/09/group/game`입니다. 수치 재실행 전
완료된 09 결과를 `outputs/.archive/09/<owner>/<game>/<revision>/`에 검증·보존합니다.
새 logits 캐시는 자동 재생성되며 frozen campaign source는 변경하지 않습니다.

현재 Thinker는 네 session 모두로 학습되었습니다. session holdout은 **분석 readout**에만
적용되므로 모델 자체의 미학습 자료 검증은 아닙니다. pre/post readout은 실제 STOP 분기
정책과 같다는 보장이 없으며, 지지 변화는 두 endpoint로 결정됩니다. 이 검증만으로
계산의 인과적 보상 효용·독립적인 누적 메커니즘·최적 VoC를 주장하지 않습니다.

## 시간·예측·신경 분석의 해석

- 원본 행동의 run/episode 경계, 비단조 시간, 공백, switch 후 복귀를 보존합니다.
  드문 replay 표본 사이에서 일어난 switch-and-return을 행동 유지로 취급하지 않습니다.
- scaling, PCA, ACE, nuisance 회귀, mixture 선택과 readout은 학습 세션 안에서만 추정합니다.
  테스트 세션의 미래 행동으로 축의 방향이나 motif 이름을 정하지 않습니다.
- 현재 Thinker 자체는 모든 인간 세션으로 학습했습니다. **분석 readout의 세션 holdout은
  Thinker 학습에서 보지 못한 세션에 대한 검증이 아닙니다.**
- 09는 0.5초 입력/0.5초 미래의 짧은 창을 기본으로 하고, bout 시작 부근에서 고정한
  방향으로 이후 변화를 추적하는 보조 분석도 보존합니다. 미래 결과로 방향을 정하지 않습니다.
  부호 있는 진행과 절대 이동량을 구분하고, endpoint로 설명 가능한 부분을 명시합니다.
  시간 순서를 섞은 대조는 새로운 환경 궤적이나 인과 개입이 아닙니다.
- 10은 단일 상태·긴 계산량만으로 군집을 명명하지 않습니다. 기능적 표현, 시간 창,
  행동과 화면 대조, 세션 holdout, K=1을 포함한 mixture 비교로 패턴의 지원 여부를 봅니다.
  목표/전략이라는 해석은 별도 행동·장면 근거가 있어야 합니다. 연속적 패턴도 유효한 결과입니다.
- 사람 trajectory 위에서 추출한 변수는 먼저 Thinker의 평가입니다. 인간의 목표 또는
  내부 planning을 직접 측정했다고 주장하지 않습니다.
- 11–13은 native 모델 변수로 design을 다시 구성하고, 행동 SHA와 정렬이 확인된 BOLD만
  재사용합니다. 현재 Enduro의 검증된 BOLD 입력은 없어 `BLOCKED_INPUT`입니다.
- 기존 정렬 가정은 **TR 1초, 행동 시작을 volume 60에 대응, 480 TR**입니다.
  scanner trigger로 입증된 정렬은 아닙니다. 실제 행동 시간축에 HRF를 적용하며
  imagination event 각각에 가상의 물리 시간을 부여하지 않습니다.
- encoding/decoding의 예측 성능과 ROI 평균의 기술적 상관, RSA의 표현 대응을 구분합니다.
  fMRI로 개별 imagination의 순서를 분해하거나 GRU를 특정 뇌 영역과 동일시하지 않습니다.

## 기존 번호와 비교 자료

개편 전 소스 사본은 `legacy/research_script/pre_reorganization/source/research_script/`에
보존했습니다. 주요 이동은 다음과 같습니다.

| 이전 | 현재 |
| --- | --- |
| 00 training diagnostics, 01 model fidelity | 02 모델 검증 |
| 03 dynamic control, 09 task-state | 03 계산 배분 |
| 10 computation utility | 04 계산 효용 |
| 02의 V/Q 갱신·행동 관계, 05 prediction | 05 갱신 / 07 행동 관계 |
| 04 search-to-action | 06 검색에서 정책으로 |
| 12 continuous dynamics | 08 관측 간 연속성 |
| 13 directional accumulation, 13_1 whole bout | 09 방향성 누적과 bout 보조 분석 |
| 14 activation/ACE/temporal/motif | 10 제어 패턴 |
| 08 encoding, 06 decoding, 07 RSA | 11 / 12 / 13 |

이전 72 leave-session cache는 기본 입력이 아닙니다. 명시적 비교가 필요한 경우
격리된 보존 소스를 다음 진입점으로 실행합니다. `--legacy-theme`은 **이전 번호**이며
비교 출력은 현재 `research_script/outputs` 바깥에 지정해야 합니다.

```bash
/home/jeongmin/miniconda-dynamic/envs/dynamic-thinker/bin/python -B research_script/orchestration/compare_legacy.py --dataset legacy_leave_session --legacy-theme 02 --subject 1 --game 1 --cache-root research_script/outputs/00_dynamic_cache_10m_20261110 --outputs-root legacy/research_script/comparison_results
```

그 자료의 checkpoint leave-session holdout과 최신 all-session 학습 자료를 같은
독립 표본으로 합치지 않습니다. 검증된 기존 scalar extraction은 해시가 일치할 때만
명시적인 이전 버전 import로 재사용하며 출처를 `00/.../logs/run.json`에 남깁니다.

## 일회성 정리 상태

새 분석 코드는 구현됐지만, 사용자 요청에 따라 실행 중인 구 14 분석을 유지하고
그 분석이 종료된 뒤 기존 폴더를 이관합니다. 기존 번호 결과와 이전 진입점을
일괄 이동하는 작업은 아직 실행하지 않았으며, 자동 이관 작업도 등록하지 않았습니다.
현재 코드가 기존 형식의 결과를 만나면 덮어쓰지 않고 이관 필요를 알립니다.

다음 도구의 기본 동작은 **읽기 전용 이동 계획 확인**입니다.

```bash
/home/jeongmin/miniconda-dynamic/envs/dynamic-thinker/bin/python -B research_script/orchestration/migrate_legacy_layout.py
```

실행 중인 이전 분석이 모두 끝난 뒤 `--apply`로 검토된 목록을
`legacy/research_script/pre_reorganization/`에 한 번만 이관합니다. 실행 중인 이전
분석이 감지되면 도구가 거절하며, 기존 archive에 충돌해도 덮어쓰지 않습니다.
학습·재개·W&B 운영 경로, 원본 데이터, 체크포인트, native visualization, 수입 cache와
호환 심볼릭 링크는 이 연구 결과 이관의 대상이 아닙니다.
