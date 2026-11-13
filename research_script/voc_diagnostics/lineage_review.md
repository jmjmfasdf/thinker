# v7–v20 lineage 확인 (2026-09-24)

후속 protocol에 명시된 이전 버전의 실패 기록과, 확인 가능한 실제
평가 산출물을 구분했다. 아래의 합격/실패는 각 버전의 당시 기준이며,
현재 48개 5M checkpoint의 성능을 대신 판정하지 않는다.

| 버전 | 확인된 범위와 결과 | 근거 |
|---|---|---|
| v7 | 100k mechanism pass 기록은 있음. 200k checkpoint의 고정 평가에서는 `all_four_behaviors_pass=false`, gap retention=0.380837 | `/work/jeongmin/voc-fixed-evaluations/enduro-voc-v7-200k-heldout-20260827/summary.json`; `VOC_V7_200K_ACCEPTANCE.md` |
| v8 | 이번 조사에서 별도 합격 판정 산출물을 확인하지 못함 | 확인 범위를 넘어 합격/실패를 추정하지 않음 |
| v9 | 200k에서 deep-negative 표본 수 부족으로 실패: full 230<256, late 55<64 | `VOC_V10_POWERED_300K_ACCEPTANCE.md:3` |
| v10 | 300k 방향·강도·stability 기준 실패 | `VOC_V11_EXACT_PROJECTION_300K_ACCEPTANCE.md:12` |
| v11 | exact projection의 전달 성능은 확인됨. 전체 300k 기준은 W3의 deep-negative sampled STOP 12/25 때문에 실패 | `VOC_V12_EPSGREEDY_GATE_300K_ACCEPTANCE.md:12` |
| v12 | 300k에서 sign support, actor AMP skips, 일부 행동 기준, forced-stop 기준 실패 | `VOC_V13_VERSIONED_EPS25_300K_ACCEPTANCE.md:21` |
| v13 | 1.2k integrity wire가 model 종료 대기 문제로 실패 | `VOC_V14_SEALED_MODEL_INPUT_300K_ACCEPTANCE.md:21` |
| v14 | 100k qualification에서 gate retention≈0.98. 전체 판정은 held-out EMA TD RMSE=0.524795>0.5로 실패 | `VOC_V15_HALF_SQUARED_Q_300K_ACCEPTANCE.md:20` |
| v15 | 100k qualification에서 retention≈0.98. RMSE=0.563713으로 실패 | `VOC_V16_COMMON_MODE_Q_300K_ACCEPTANCE.md:45` |
| v16 | 100k에서 sign denominator와 RMSE 기준 실패 | `VOC_V17_HUBER_COMMON_Q_300K_ACCEPTANCE.md`; v14–v19 RCA report |
| v17 | RMSE=0.473962는 통과했지만 sign denominator 기준 실패 | `VOC_V18_ORTHOCD_ADAM_Q_300K_ACCEPTANCE.md:82` |
| v18 | sign denominator와 RMSE=0.621387로 실패 | `VOC_V19_TAU1_Q_300K_ACCEPTANCE.md:56` |
| v19 | 15개 기준 중 14개 통과. RMSE=0.525172로 전체 실패 | `VOC_V20_SCHEMA13_TELEMETRY_300K_ACCEPTANCE.md:58` |
| v20 | telemetry protocol과 1.2k wire의 `finish: complete`는 확인. qualification/primary/fixed 평가 전체 합격은 확인하지 못함 | `/tmp/di-voc-v20-schema13-telemetry-wire-candidate-oQn9WV/runs/enduro-voc-v20-telemetry-tau1-orthocd-adam-eps25-sentinel-wire1200/finish` |

**확인된 전체 합격 버전을 바로 채택할 근거는 없다.** 다만 exact
projection이 teacher→gate 전달을 거의 정확하게 만드는 사실은 별도
기술적 근거다. 이를 현재 checkpoint의 분석용 T2 조건으로 테스트할 수
있지만, teacher의 실제 효용이 입증되었다고 해석할 수는 없다.

v11의 확률적 실행에서 표본 25개의 STOP 빈도가 기준에 못 미친 것과
실행 오류는 구분해야 한다. 후속 protocol도 이 실패를 projection이나
runtime fault로 해석하지 않는다. 과거의 유한 표본 합격 기준을 현재
실험의 과학적 효용 기준과 그대로 동일시하지 않는다.

본 조사는 기존 결과의 재분류, 재학습, checkpoint 교체를 수행하지 않았다.
