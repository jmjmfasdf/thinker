# Adaptive future computation and continuous behavioral control

이 디렉터리는 **계산의 배분 → 평가의 갱신 → 행동 제어의 조직화 → 신경 표현**을 검증합니다.
관측된 버튼 입력, 모델의 내부 계산, 인간의 의사결정은 서로 다른 측정 수준입니다.
Imagination은 인간의 사고 자체가 아니라, 행동과 뇌 활동에 대해 검증할 모델 설명변수입니다.

## 분석 순서

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
| 09 | `09_evaluation_accumulation.py` | 일정 행동을 유지하는 동안 평가 변화가 방향성을 가지며 누적되는가; 짧은 창 분석과 전체 bout 보조 분석 |
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
logs/visualization/hxln_10m/sub001/pong/
logs/visualization/hxln_10m/sub001/enduro/
logs/visualization/hxln_10m/sub001/spaceinvaders/
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

프로젝트 루트에서 dynamic 분석 환경을 사용합니다.

```bash
# 공통 전처리. 02–13도 필요할 때 이를 호출합니다.
/home/jeongmin/miniconda-dynamic/envs/dynamic-thinker/bin/python -B research_script/00_native_preparation.py --subjects 1 2 --games pong enduro spaceinvaders --resume

# 모델이 필요 없는 행동 분석
/home/jeongmin/miniconda-dynamic/envs/dynamic-thinker/bin/python -B research_script/01_behavioral_analysis.py --subjects 1 2 --games pong enduro spaceinvaders mario --resume

# 선택한 연구 단계 실행
/home/jeongmin/miniconda-dynamic/envs/dynamic-thinker/bin/python -B research_script/orchestration/run_themes.py --themes 02 03 05 06 07 08 --subjects 1 2 --games pong enduro spaceinvaders --workers 2 --resume

# 표를 바꾸지 않고 저장된 자료만으로 그림 재생성/재사용
/home/jeongmin/miniconda-dynamic/envs/dynamic-thinker/bin/python -B research_script/07_computation_to_behavior.py --subjects 1 2 --games pong enduro spaceinvaders --render-only

# 참가자별 분석 뒤 게임별 동일 가중 그룹 요약
/home/jeongmin/miniconda-dynamic/envs/dynamic-thinker/bin/python -B research_script/07_computation_to_behavior.py --subjects 1 2 --games pong enduro spaceinvaders --aggregate --resume
```

`--stage prepare`, `--stage analyze`, `--stage render`로 단계를 나눌 수 있습니다.
`--resume`은 입력·설정·관련 코드·저장 파일의 해시가 일치하는 완료 단계를 재사용합니다.
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
