# 보상 기반 VOC 검증: 현재 checkpoint와 맞출 정의

Planning의 효용은 인간 행동 일치가 아니라 추가 계산이 기대 과제 가치를
계산비용보다 더 높이는지로 평가한다. 인간 likelihood는 별도의 행동 적합도다.
가치 추정치를 사용한다는 사실만으로 실제 선택이 합리적이라고 판정하지 않는다.

## 확인한 critic 목표

48개 checkpoint에 대응하는 frozen `legacy`와 `group` runtime 모두 다음을 사용한다.

- `learn_actor.py:5958`: 과제 보상 discount는 실제 환경 transition에서만 γ,
  SEARCH/STOP/WAIT 내부 단계에서는 1이다. terminal/truncation이면 0이다.
- `learn_actor.py:5963`: think discount는 `~(done | stage_end)`이다.
  즉 **현재 search stage가 끝나면 이후 계산비용을 bootstrap하지 않는다.**
- `learn_actor.py:1209`: net target = task target + think_cost × think target.
- `cenv.pyx:1584`: PROCEED/RESET 계산 1회에 think reward −1이다.
  STOP의 비용은 0이고 stage를 끝낸다. 최대 계산량 도달도 stage를 끝낸다.
- 현재 기본 설정은 γ=0.99, reward_clip=1, think_cost=0.0005이며,
  실험에서는 각 checkpoint의 실제 값을 읽고 일치 여부를 검증해야 한다.

따라서 현재 critic의 평가 대상은 다음과 같다.

\[
U_{\mathrm{current}} = \mathbb{E}\left[
  \sum_{j\ge0}\gamma^j r^{\mathrm{clipped}}_j
  - c\,N_{\mathrm{remaining,current\ stage}}
\right].
\]

이는 모든 미래 decision의 계산비용까지 합산한 전체 metareasoning 목적함수와
같지 않다. 후자는 별도 결과로 보고하고, 현재 ΔQ의 정확도를 판정하는 target과
섞지 않는다. ΔQ가 이미 net value 차이이므로 gate 적용 때 비용을 다시 빼지 않는다.

## 필요한 paired 실험

1. 온라인 환경에서 정상 정책이 만든 search 시작 상태를 고정한다.
   인간 NPZ에는 image/action/reward/terminal/time/RAM이 있지만, RAM만으로
   emulator 전체 상태·RNG·wrapper frame stack을 복원할 수 없다.
2. 같은 seed, 실제 행동 이력, planner 이력, actor recurrent state/RNG를 재구성한다.
   STOP-now와 CONTINUE-once로 분기하고, 후자는 원래 conditional PROCEED/RESET을
   따른 뒤 원래 gate 정책으로 돌아간다. 이는 무조건 k회 계산과 다른 개입이다.
3. 최종 실제 행동 및 그 이후 정책은 각 갈래가 스스로 선택한다.
   미래 human 화면이나 action을 강제로 넣지 않는다.
   Primary action과 conditional PROCEED/RESET의 sampling, gate temperature와
   exploration epsilon을 명시한다. 현재 human replay는 conditional/primary에
   greedy 평가를 사용하므로, 그 결과를 학습 중 sampled policy의 Q 정확도와
   바로 동일시하지 않는다. Critic calibration에는 target policy와 맞는 실행이 필요하다.
4. task reward는 실제 환경에서 얻고 γ를 실제 행동마다 한 번만 적용한다.
   현재 stage가 끝날 때까지 발생한 계산비용만 현재 critic target에 포함한다.
5. terminal까지의 Monte Carlo가 이상적이다. 유한 H를 쓰면 truncated MC와
   γ^H V(s_H)를 더한 bootstrap 결과를 분리한다. 같은 critic의 bootstrap으로
   같은 critic을 검증하면 순환성이 남으므로 독립적인 held-out Monte Carlo가 필요하다.
6. 분기 전에 기록한 EMA/online ΔQ와 paired return 차이를 비교한다.
   calibrated slope, signed utility, uncertainty를 보고한다. Return 차이의
   분산은 같은 seed·공통 난수·동일 초기 상태로 줄이고, 여러 rollout을 반복한다.
7. EMA gate의 효용은 동일 평균 계산량의 random/fixed allocation과 비교한다.
   계산량을 더 쓴 정책의 원보상만 높아진 것을 VOC 제어 개선으로 해석하지 않는다.

## 실행 경로 확인 결과

설치된 EnvPool 0.8.4에는 `quick_save/quick_load`가 없다. Thinker wrapper에
그 이름의 method가 존재해도 실제 backend가 지원한다는 뜻이 아니다.
또 `EnvPoolWrap.reset`은 seed 인자를 전달하지 않으므로 seed는 환경 생성 시
명시해야 한다. 이 부분을 확인하지 않고 `reset(seed=...)`만 사용하는 실험은
서로 다른 seed 실험이 아닐 수 있다.

`check_online_reconstruction.py`는 Enduro/Pong 각각 두 seed에서 같은 환경을
둘씩 생성하고, RGB 4-frame stack·episodic life 설정으로 같은 실제 행동
128개를 실행했다. Reset과 모든 transition의 관측·보상·done·info 해시가
일치했다. 이 검사 자체는 **환경 경로의 재구성 가능성** 검증이다.

후속 `reward_probe.py`는 actor/planner까지 포함해 공통 prefix와 분기 시점의
상태·RNG·tree 관측 fingerprint를 비교하고, STOP 갈래를 다시 실행해 전체
trajectory와 return의 일치를 검증한다. `voc_training_pipeline_20260924`에는
각 게임 4개 상태 × 반복 2회, 256 real-step의 기존 checkpoint 결과가 있다.
이 결과는 모두 유한 구간이며 terminal까지 완주한 return은 없다. 학습 정책과
정확히 맞출 때에는 `--policy-epsilon`을 사용해야 한다. 정기 pilot 평가는 이
옵션을 사용한다. 상세 범위는 `training_pipeline.md`를 참조한다.

원본 checkpoint와 frozen runtime은 변경하지 않았다.
