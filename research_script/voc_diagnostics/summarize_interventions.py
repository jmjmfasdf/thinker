"""Aggregate paired intervention outputs without treating events as subjects."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["tv", "argmax_changed", "delta_human_ll", "delta_binary_log_score",
           "delta_computations", "reference_computations", "treatment_computations"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, required=True)
    args = p.parse_args()
    root = args.campaign.resolve()
    plan = json.loads((root / "campaign.json").read_text())
    records, coverage = [], []
    for task in plan["tasks"]:
        folder = root / task["name"]
        if not (folder / "manifest.json").exists():
            coverage.append(dict(task=task["name"], status="INCOMPLETE"))
            continue
        manifest = json.loads((folder / "manifest.json").read_text())
        assert manifest["status"] == "COMPLETE"
        for name, key in [("intervention_probe.py", "code_sha256"), ("forced_budget_probe.py", "helper_sha256")]:
            assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == manifest[key] == plan["script_hashes"][name]
        rows = [json.loads(line) for line in (folder / "decisions.jsonl").read_text().splitlines()]
        for target in manifest["targets"]:
            group = {r["condition"]: r for r in rows if r["target_edge"] == target}
            assert set(group) == {"stop", "full", "frozen", "learned", "ema", "stop_repeat"}
            assert group["stop"]["probability"] == group["stop_repeat"]["probability"]
            for row in group.values():
                for key in ("prefix_sha256", "entry_sha256", "tree_observation_sha256"):
                    assert row[key] == group["stop"][key]
                prob = np.asarray(row["probability"])
                assert np.isfinite(prob).all() and np.all(prob >= 0) and np.all(prob <= 1)
                assert abs(prob.sum() - 1) < 1e-12
            assert group["full"]["token_masks"] == group["frozen"]["token_masks"]
            assert group["full"]["action_schedule"] == group["frozen"]["action_schedule"]
            assert len(set(group["frozen"]["content_hashes"])) == 1
            for left, right in zip(group["learned"]["choices"], group["ema"]["choices"]):
                assert left["uniform"] == right["uniform"]
            for treatment, reference, contrast in [("full", "stop", "full_vs_stop"),
                                                    ("full", "frozen", "full_vs_frozen"),
                                                    ("ema", "learned", "ema_vs_learned")]:
                a, b = group[treatment], group[reference]
                pa, pb = np.asarray(a["probability"]), np.asarray(b["probability"])
                human = a["human_action"]
                binary_a = pa[0] if human == 0 else pa[1:].sum()
                binary_b = pb[0] if human == 0 else pb[1:].sum()
                records.append(dict(game={0:"Enduro",1:"Pong"}[task["game"]], subject=task["subject"],
                    leave_session=task["leave_session"], source=manifest["source"]["source_id"],
                    edge=target, transition=a["transition"], contrast=contrast,
                    tv=float(np.abs(pa-pb).sum()/2), argmax_changed=int(a["argmax"]!=b["argmax"]),
                    delta_human_ll=float(np.log(max(pa[human],1e-12))-np.log(max(pb[human],1e-12))),
                    delta_binary_log_score=float(np.log(max(binary_a,1e-12))-np.log(max(binary_b,1e-12))),
                    delta_computations=a["actual_computations"]-b["actual_computations"],
                    reference_computations=b["actual_computations"], treatment_computations=a["actual_computations"],
                    initial_ema_voc=group["stop"]["choices"][0]["ema_delta_q"],
                    initial_teacher_p=group["stop"]["choices"][0]["teacher_probability"],
                    initial_learned_p=group["stop"]["choices"][0]["learned_probability"]))
        coverage.append(dict(task=task["name"], status="COMPLETE", n_targets=len(manifest["targets"]),
                             missing_strata=",".join(manifest["missing_strata"])))
    pd.DataFrame(coverage).to_csv(root / "coverage.csv", index=False)
    if not records:
        raise RuntimeError("No complete probe results")
    pairs = pd.DataFrame(records)
    pairs.to_csv(root / "paired_scores.csv", index=False)
    subjects = pairs.groupby(["game", "subject", "transition", "contrast"], as_index=False)[METRICS].mean()
    # Each transition category receives equal weight, then subjects equally.
    overall = subjects.groupby(["game", "subject", "contrast"], as_index=False)[METRICS].mean().assign(transition="equal_strata")
    subjects = pd.concat([subjects, overall], ignore_index=True)
    subjects.to_csv(root / "subject_scores.csv", index=False)
    summary = subjects.groupby(["game", "transition", "contrast"])[METRICS].agg(["mean", "sem", "count"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(root / "summary.csv", index=False)
    finished = sum(r["status"] == "COMPLETE" for r in coverage)
    lines = ["# 계산 제어 후속 검증", "",
             f"완료: {finished}/{len(plan['tasks'])} checkpoint. 고유 시점 {len(pairs)//3}개.", "",
             "주요 결과는 계산이 출력에 미치는 영향이다. Planning의 보상−계산비용 효용은 이 표로 판정하지 않는다.",
             "Human likelihood는 별도의 행동 적합도 결과다. Enduro/Pong, 6명, leave-session 4개를 대상으로",
             "각 미학습 session의 첫 run/episode에서 첫 양방향 전환과 지속 시점을 선택했다.",
             "각 방향은 checkpoint 내 1개뿐이다. 전체 run의 대표 표본이나 학습 성능 비교가 아니다.", "",
             "## T1: 계산량과 새로운 정보의 효과", "",
             "Full은 정상 search 3회, stop은 즉시 중단, frozen은 같은 search 행동·갱신 횟수로",
             "actor에 초기 정보만 반복 제공한다. Frozen은 인위적인 입력 개입으로, 최적화된 대체 agent가 아니다.",
             "TV는 행동분포 거리(0이면 동일), argmax change는 가장 높은 확률의 행동이 달라진 비율이다.",
             "각 strata별 checkpoint 평균 → strata 동일 가중 → subject 동일 가중, ±SEM은 subject 사이 값이다.", "",
             "| Game | Contrast | TV | Argmax change |",
             "|---|---|---:|---:|"]
    for _, r in summary[summary.transition.eq("equal_strata") & summary.contrast.ne("ema_vs_learned")].iterrows():
        lines.append(f"| {r.game} | {r.contrast} | {r.tv_mean:.4f} ± {r.tv_sem:.4f} | {r.argmax_changed_mean:.1%} |")
    lines += ["", "## T2: EMA gate의 직접 실행", "",
              "양쪽 모두 확률적으로 실행하고 같은 uniform 난수를 사용했다. Conditional PROCEED/RESET과",
              "상상 행동 policy는 유지했다. 각 시점에서 한 난수열만 사용했으며, 현재 gate로 재생한 공통",
              "이력에서 현재 결정만 개입했다. 전체 실행 정책의 성능이 아니다. **평균 계산량도 맞추지 않았다.**", "",
              "| Game | Learned computations | EMA computations | Δ computations | TV | Argmax change |",
              "|---|---:|---:|---:|---:|---:|"]
    for _, r in summary[summary.transition.eq("equal_strata") & summary.contrast.eq("ema_vs_learned")].iterrows():
        lines.append(f"| {r.game} | {r.reference_computations_mean:.3f} | {r.treatment_computations_mean:.3f} | {r.delta_computations_mean:+.3f} | {r.tv_mean:.4f} | {r.argmax_changed_mean:.1%} |")
    lines += ["", "## 행동 적합도: 양방향 전환별 보조 결과", "",
              "아래는 NOOP/active binary log score 차이(nat/decision), 양수면 관측된 인간 행동에",
              "더 높은 확률을 부여한 것이다. Rational choice 또는 과제 보상 개선을 뜻하지 않는다.", "",
              "| Game | Human transition | Contrast | Δ binary log score |",
              "|---|---|---|---:|"]
    for _, r in summary[summary.transition.isin(["NOOP_to_active", "active_to_NOOP"])].iterrows():
        lines.append(f"| {r.game} | {r.transition} | {r.contrast} | {r.delta_binary_log_score_mean:+.4f} ± {r.delta_binary_log_score_sem:.4f} |")
    lines += ["", "## 해석의 범위와 다음 단계", "",
              "- 출력 변화는 영향의 증거다. 기대 보상이 개선됐다는 증거는 아니다.",
              "- Frozen 대조는 새 search 정보와 그에 따른 행동·reward feedback의 의존성을 검사한다.",
              "  부자연스러운 반복 입력의 분포 변화 가능성이 남는다.",
              "- 기존 VOC의 효용 검증에는 같은 온라인 상태에서 STOP-now와 CONTINUE-once 이후",
              "  기존 정책을 따르는 조건의 보상−계산비용 return 차이가 필요하다. 현재 critic의 비용은",
              "  **현재 search stage가 끝날 때까지**이며, 과제 보상은 이후 실제 행동까지 이어진다.",
              "  모든 미래 stage 비용을 합산한 목표와 혼동하면 안 된다. 비용은 이미 critic",
              "  목표에 포함되므로 ΔQ에서 임의로 다시 빼지 않는다.",
              "- 유한 rollout horizon, discount와 보상 clipping, terminal 처리, 잔여 bootstrap을",
              "  명시하고 그 target을 현재 critic의 정확한 정의와 맞춰야 한다.",
              "- 현재 인간 NPZ의 화면·행동만으로 실제 게임 반사실 보상을 복원하지 않는다.",
              "- 코드, 원본 source/checkpoint 식별, 각 arm의 gate 확률·난수·계산량·상태 해시는",
              "  각 task 디렉터리의 manifest와 decisions.jsonl에 남겼다."]
    lines += ["", "## 방향별 관측 범위", "",
              "첫 256 edge 안에 해당 전환이 없으면 제외하고 누락 사유를 coverage.csv에 남겼다.",
              "아래는 각 strata의 checkpoint 수와 subject 수다. 요약은 사용 가능한 checkpoint와",
              "strata를 동일 가중하며, 누락된 전환을 관측한 것으로 간주하지 않는다.", "",
              "| Game | Stratum | Checkpoints | Subjects |", "|---|---|---:|---:|"]
    for (game, stratum), group in pairs[pairs.contrast.eq("full_vs_stop")].groupby(["game", "transition"]):
        lines.append(f"| {game} | {stratum} | {len(group)} | {group.subject.nunique()} |")
    if finished != len(plan["tasks"]):
        lines.insert(2, "**미완료 결과이며 전체 비교로 해석할 수 없다.**")
    (root / "report.md").write_text("\n".join(lines)+"\n")
    print(f"Reported {finished}/{len(plan['tasks'])} checkpoints, {len(pairs)//3} decisions")


if __name__ == "__main__":
    main()
