"""Summarize the bounded T0/T1 diagnostic; never pool selected points as subjects."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GAMES = {0: "Enduro", 1: "Pong"}
INCLUDED = ["t1_smoke_enduro", "t1_smoke_pong",
            "t1_transitions_verified_enduro", "t1_transitions_verified_pong"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binary_probability(p, action):
    return p[0] if action == 0 else sum(p[1:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=ROOT / "research_script/outputs/voc_diagnostics_20260924")
    args = parser.parse_args()
    root = args.root.resolve()
    pairs, inputs, n_decisions = [], {}, 0
    for name in INCLUDED:
        directory = root / name
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["status"] == "SMOKE_PASS", name
        assert sha(directory / "probe_source.py") == manifest["code_sha256"], name
        inputs[name] = {f: sha(directory / f) for f in ["manifest.json", "decisions.jsonl", "probe_source.py"]}
        rows = [json.loads(line) for line in (directory / "decisions.jsonl").read_text().splitlines()]
        for target in manifest["targets"]:
            group = [r for r in rows if r["target_edge"] == target]
            assert [r["budget"] for r in group] == manifest["budgets"] + [0]
            base = group[0]
            assert base["probability"] == group[-1]["probability"]
            assert group[-1]["repeat_control"]
            for row in group:
                assert row["actual_computations"] == row["budget"]
                assert row["target_human_action_not_executed"]
                for key in ["prefix_sha256", "prefix_calls", "entry_sha256", "tree_observation_sha256"]:
                    assert row[key] == base[key], (name, target, key)
                prob = row["probability"]
                assert all(math.isfinite(p) and 0 <= p <= 1 for p in prob)
                assert math.isclose(sum(prob), 1, abs_tol=1e-12)
            previous, action = base["previous_human_action"], base["human_action"]
            transition = ("NOOP" if previous == 0 else "active") + "_to_" + ("NOOP" if action == 0 else "active")
            assert all(row["transition"] == transition for row in group)
            if "transitions_verified" in name:
                assert (previous == 0) != (action == 0), (name, target)
            p0 = base["probability"]
            for row in group[1:-1]:
                p = row["probability"]
                delta = math.log(max(p[action], 1e-12)) - math.log(max(p0[action], 1e-12))
                assert math.isclose(delta, row["delta_human_log_likelihood"], abs_tol=1e-12)
                record = dict(game=GAMES[manifest["game"]], subject=manifest["subject"],
                              session=manifest["session"], checkpoint=manifest["checkpoint"]["directory"],
                              source=manifest["source"]["source_id"], episode=manifest["episode"],
                              target_edge=target, observation_source_index=row["observation_source_index"],
                              selection="first_directional_boundary" if "transitions_verified" in name else "early_NOOP_smoke",
                              transition=transition, previous_action=previous, human_action=action,
                              budget=row["budget"], base_argmax=base["argmax"], argmax=row["argmax"],
                              argmax_changed=row["argmax"] != base["argmax"],
                              p_human_base=p0[action], p_human=p[action],
                              total_variation=row["total_variation"], delta_human_ll=delta,
                              delta_binary_log_score=math.log(max(binary_probability(p, action), 1e-12))
                              - math.log(max(binary_probability(p0, action), 1e-12)),
                              artifact=name)
                pairs.append(record)
            n_decisions += 1
    csv_path = root / "t1_pairs.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    qa = json.loads((root / "t0/qa.json").read_text())
    assert qa["completed"] == qa["expected"] == 48 and not qa["errors"]
    with (root / "t0/summary.csv").open() as f:
        t0 = [r for r in csv.DictReader(f) if r["split"] == "held_out" and r["scope"] == "all"]
    count = sum(q["valid_choices"] for q in qa["caches"])
    invalid = sum(q["invalid_choices"] for q in qa["caches"])
    lines = [
        "# VOC 진단 결과: T0 완료와 T1 소규모 검증", "",
        "핵심: teacher→gate 전달 문제는 KL로도 확인됐다. 추가 계산이 최종 행동확률과",
        "최선 행동을 바꿀 수 있는 사례도 확인됐다. 따라서 ‘계산의 실제 가치가 거의 0’이라는",
        "단정은 철회해야 한다. 다만 계산이 인간 행동 예측이나 과제 보상을 일관되게 개선하는지는",
        "아직 검증되지 않았다.", "",
        "## 범위", "",
        "- T0: 6명 × Enduro/Pong × leave-session checkpoint 4개 = 48개 캐시 전부.",
        "- 이 분석에는 Space Invaders가 포함되지 않는다.",
        f"- T1: sub1/leave1, 게임별 held-out session 1의 첫 episode에서 총 {n_decisions}개 시점.",
        "  게임별 초기 NOOP 3개와 양방향 전환 각 1개를 선정했다. 각 시점에서 계산량 0/1/3/5와",
        "  반복 0회 조건을 검사했다. 30개 대비를 CSV에 저장했으며 모집단 평균을 추정하지 않았다.",
        "- 재학습·checkpoint 변경·학습 job 제출은 수행하지 않았다.", "",
        "## T0: EMA teacher → gate", "",
        f"RESET을 포함한 {count:,}개 control choice를 계산했다. 유효하지 않은 choice는 {invalid}개다.",
        "Teacher 확률은 sigmoid(EMA ΔQ / checkpoint temperature), 현재 48개 temperature는 0.05다.",
        "다음은 held-out 인간 session 결과이며, run → checkpoint → subject 순으로 동일 가중했다.",
        "±는 6명 사이 SEM이다. 단위는 nat/control choice, 낮을수록 teacher에 가깝다.", "",
        "| Game | KL(teacher ‖ gate) ↓ | KL(teacher ‖ constant 0.5) ↓ |",
        "|---|---:|---:|",
    ]
    for row in t0:
        lines.append(f"| {GAMES[int(row['game'])]} | {float(row['kl_teacher_gate']):.6f} ± {float(row['kl_teacher_gate_sem']):.6f} | {float(row['kl_teacher_neutral']):.6f} |")
    lines += ["", "게임별 평균에서 현재 gate는 항상 0.5인 gate보다도 teacher와 더 다르다.",
              "이는 teacher 추종 문제의 근거이며, teacher가 유용한 판단을 한다는 증거가 아니다.",
              "Pong의 작은 절대 KL도 전달이 충분하다는 의미는 아니다. Teacher 자체가 0.5에 가깝다.",
              "원자료와 training/all-session 집계는 `t0/`에 저장했다.", "",
              "## T1: 같은 시작 상태에서 추가 계산의 효과", "",
              "완전한 C++ planner 복제 API가 없어 각 조건에서 같은 seed로 같은 episode prefix를",
              "새 planner에 재생했다. 분기 전 actor 입력·recurrent state·RNG·기록 가능한 tree 상태의",
              "해시가 일치하고, 반복 0회 조건의 확률도 정확히 재현됐다. 요청 계산량과 실행량이",
              "일치했으며 actor/world-model 가중치와 buffer는 변하지 않았다.", "",
              "현재 인간 행동을 실행에 반영하기 전에 최종 real-head 확률을 읽었다.",
              "아래는 원본 행동에서 선정한 각 방향의 **첫 전환 1개씩**으로, 5회 − 0회 대비다.",
              "ΔLL은 실제 인간 행동의 log probability 차이, binary Δlog score는 NOOP 또는 active에",
              "할당한 확률의 log 차이다. 양수면 해당 관측 행동에 더 높은 확률을 부여한 것이다.",
              "TV는 두 행동분포 사이 거리로, 0이면 동일하다.", "",
              "| Game | Human transition | Edge (0-based) | Human action | Argmax: 0→5 computations | TV | Δ action LL | Δ binary log score |",
              "|---|---|---:|---:|---|---:|---:|---:|"]
    for row in pairs:
        if row["selection"] == "first_directional_boundary" and row["budget"] == 5:
            lines.append(f"| {row['game']} | {row['transition'].replace('_to_', '→')} | {row['target_edge']} | {row['human_action']} | {row['base_argmax']}→{row['argmax']} | {row['total_variation']:.4f} | {row['delta_human_ll']:+.4f} | {row['delta_binary_log_score']:+.4f} |")
    lines += ["", "Pong active→NOOP에서는 0회 조건의 최선 행동 3이 3회·5회에서 NOOP(0)으로 바뀌었다.",
              "5회 조건의 인간 NOOP 확률은 0.3344 → 0.6572다. 반면 Pong NOOP→active에서는",
              "5회 조건의 실제 인간 행동 확률이 0.0486 → 0.0102로 감소했다.",
              "Enduro에서는 이 두 전환의 argmax가 유지됐지만 확률분포는 달라졌다.", "",
              "이 결과는 선택한 시점에서 추가 계산이 출력에 영향을 준다는 근거다. 추가 계산에는",
              "actor의 recurrent update도 포함되므로 아직 world-model search **내용**의 효과로",
              "분리할 수 없다. 기존 tree carry도 두 조건에 공통으로 남아 있어 0회 조건은",
              "‘과거 search까지 없는 agent’가 아니다. 양방향·두 게임의 일관된 개선은 확인되지 않았다.", "",
              "## 인덱스 검증과 제외 기록", "",
              "Cache real_step은 1부터 시작하고 probe target_edge는 0부터 시작한다.",
              "초기 `t1_transitions_enduro`(43/46), `t1_transitions_pong`(34/37)는 변환을 놓쳐",
              "전환 다음 시점이었다. 저장된 실제 label은 active→active/NOOP→NOOP이며 덮어쓰지 않았다.",
              "이 두 디렉터리는 위 표와 `t1_pairs.csv`에서 제외했다. `--first-transitions`가 이제",
              "원본 actions_seq에서 직접 전환을 선택한다. 올바른 edge는 Enduro 42/45, Pong 33/36이다.", "",
              "## v7–v20 검토와 다음 판단", "",
              "확인한 산출물에서는 당시 전체 합격 기준을 통과한 후속 버전을 찾지 못했다.",
              "Exact projection의 teacher 전달 개선은 확인되지만, teacher의 효용을 입증하지 않는다.",
              "세부 근거는 `research_script/voc_diagnostics/lineage_review.md`에 정리했다.", "",
              "1. T1을 사전에 선정한 held-out run과 모든 subject/checkpoint로 확대한다. 양방향 전환과",
              "   지속 시점을 포함하고, 추가 recurrent update 수를 맞춘 대조조건으로 search 내용의 효과를 분리한다.",
              "2. T2에서는 현재 gate와 정확한 EMA 확률 투영을 비교한다. 두 조건의 확률적 실행을 유지하고,",
              "   계산량 차이와 동일 평균 계산량에서의 행동 성능을 함께 본다.",
              "3. T3는 인간 행동 likelihood와 보상−계산비용 효용을 분리한다. 현재 reward 기반 VOC 검증에는",
              "   STOP-now와 CONTINUE-once-then-current-policy의 짝지은 환경 rollout이 필요하다.",
              "4. 이 검증으로 실패한 연결을 특정한 뒤 재학습 변경을 정한다. 현재 결과만으로 search 활용",
              "   강제 학습이나 근시적 VOC로의 교체가 필수라고 결론 내릴 수 없다.", "",
              "현재 학습에는 이미 RL과 인간 imitation이 함께 들어가며 world-model prior는 online actor를",
              "distillation한다. 단순히 ‘RL을 추가’하거나 ‘prior도 독립적인 직접 human BC를 한다’는",
              "설명에 기반해 학습을 변경하면 안 된다.", "",
              "재현 방법과 검증 명령은 `research_script/voc_diagnostics/README.md`에 있다.",
              "각 T1 디렉터리에는 실행 당시 코드 사본과 checkpoint/runtime/source 해시를 보존했다."]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    t0_status="COMPLETE_48_OF_48", t1_status="BOUNDED_SMOKE_ONLY",
                    included=inputs, excluded={"t1_transitions_enduro": "post-boundary indexing mistake",
                                               "t1_transitions_pong": "post-boundary indexing mistake"},
                    n_target_decisions=n_decisions, n_nonzero_budget_contrasts=len(pairs),
                    source_code_sha256=sha(Path(__file__)),
                    t0_inputs={f: sha(root / "t0" / f) for f in ["summary.csv", "qa.json", "manifest.json"]},
                    outputs={f: sha(root / f) for f in ["report.md", "t1_pairs.csv"]})
    (root / "report_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved {root / 'report.md'} and {len(pairs)} paired contrasts")


if __name__ == "__main__":
    main()
