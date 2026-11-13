"""One shared definition of action support, revision, competition, and history.

U_all = valid absolute sampled-branch updates summed / ALL computations.
U_valid averages only valid updates. B=0 has zero total/all-event evidence,
undefined U_valid, and an explicit searched indicator. K is not learned VOC.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import expit
from . import native
from .io import sha, signature, write_json
from .paths import output, tag

VERSION = "themes-native-10m-v1"
KEYS = native.KEYS

def revision_summary(events, decisions):
    e = events.copy()
    valid = e.value_revision_valid.eq(1) & np.isfinite(e.root_q_revision)
    e["U_sum"] = e.root_q_revision.abs().where(valid, 0)
    e["signed_update_sum"] = e.root_q_revision.where(valid, 0)
    e["valid_update_count"] = valid.astype(int)
    grouped = e.groupby(KEYS, as_index=False)[["U_sum", "signed_update_sum", "valid_update_count"]].sum()
    d = decisions.merge(grouped, on=KEYS, how="left", validate="one_to_one")
    cols = ["U_sum", "signed_update_sum", "valid_update_count"]
    d[cols] = d[cols].fillna(0)
    b = d.additional_computations
    if (d.valid_update_count > b).any(): raise ValueError("More valid updates than computations")
    d["U_all"] = d.U_sum / b.where(b.gt(0), 1)
    d["U_valid"] = d.U_sum / d.valid_update_count.where(d.valid_update_count.gt(0))
    d["searched"] = b.gt(0).astype(float)
    d["M"] = d.margin
    d["competition_weight"] = 4 * expit(d.M) * expit(-d.M)
    d["K"] = d.U_all * d.competition_weight
    return d

def add_history(d):
    d = d.sort_values(["run_label", "real_step"]).reset_index(drop=True)
    start = (d.run_label.ne(d.run_label.shift()) | d.episode_id.ne(d.episode_id.shift()) |
             d.real_step.diff().ne(1) | d.prior_done.eq(1))
    d["segment"] = start.cumsum()
    for name in ["M", "U_all", "K", "ema_initial", "log_budget", "root_value"]:
        d[f"lag1_{name}"] = d.groupby("segment", sort=False)[name].shift(1)
    visual = [c for c in d if c.startswith("visual_")]
    # Observed change available at this decision, never a future-frame input.
    delta = d.groupby("segment", sort=False)[visual].diff()
    d["visual_change"] = np.sqrt((delta ** 2).mean(axis=1))
    bout = (start | d.incumbent_action.ne(d.incumbent_action.shift())).cumsum()
    d["past_bout_U"] = d.groupby(bout).U_sum.cumsum() - d.U_sum
    for name in ["lag1_M", "lag1_U_all", "lag1_K", "lag1_ema_initial", "lag1_log_budget", "past_bout_U"]:
        d[name] = d[name].where(~start)
    return d

def enrich(path, cp, config, subject, game):
    d, e, qa = native.load_cache(path, cp, config, subject, game)
    a = int(d.n_actions.iloc[0])
    cols = KEYS + ["decision_time", "decision_time_valid", "actor_action_probability_valid",
                   "root_v_initial", "root_v_terminal", "useful_carry", "useful_carry_valid"]
    cols += [f"{prefix}_{j}" for prefix in ["actor_action_probability", "root_qs_mean_terminal", "root_ns_terminal"] for j in range(a)]
    d = d.merge(native.read_columns(path / "native_decisions.csv.gz", cols), on=KEYS, validate="one_to_one")
    frame_cols = KEYS + ["current_frame", "previous_frame"]
    d = d.merge(native.read_columns(path / "model_control_windows.csv.gz", frame_cols), on=KEYS, validate="one_to_one")
    rev = native.read_columns(path / "persistence_events.csv.gz", KEYS + ["root_q_revision", "value_revision_valid"])
    d = add_history(revision_summary(rev, d))
    p = d[[f"actor_action_probability_{j}" for j in range(a)]].to_numpy(float)
    valid = d.actor_action_probability_valid.eq(1).to_numpy()
    if valid.any() and (not np.isfinite(p[valid]).all() or (p[valid] < 0).any() or
                        (p[valid] > 1).any() or not np.allclose(p[valid].sum(1), 1, atol=1e-5)):
        raise ValueError("Invalid recorded real-head probabilities")
    qa.update(revision_definition="U_sum / all native additional computations; invalid updates contribute zero",
              K_definition="4 * U_all * sigmoid(M) * sigmoid(-M)",
              zero_budget_U_all=0, zero_budget_U_valid="undefined",
              real_head_valid=int(valid.sum()), checkpoint_temperature=float(config["voc_gate_q_temperature"]),
              voc_gru_connected=bool(config["voc_gru_connected"]))
    return d, e, qa

def prepare_unit(cache, outputs, subject, game):
    records = native.discover(cache, subject, game, [1, 2, 3, 4])
    root = output(0, outputs) / "derived" / tag(subject, game)
    root.mkdir(parents=True, exist_ok=True)
    sources = {p.name: sha(p) for p in [Path(__file__), Path(native.__file__)]}
    for cp, path, config, binding in records:
        if int(binding['checkpoint_real_step']) != 10_000_000:
            raise ValueError(f'Not a 10M checkpoint: {path}')
        if not np.isclose(float(config['voc_gate_q_temperature']), .01) or not config['voc_gru_connected']:
            raise ValueError(f'Unexpected training condition: {path}')
        folder = root / f"leave{cp:02d}"
        identity = dict(version=VERSION, cache=str(path.resolve()), binding=sha(path / "checkpoint_binding.json"), source=sources)
        token = signature(identity)
        manifest_path = folder / "manifest.json"
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text())
            if old.get("signature") != token: raise ValueError(f"Derived cache settings changed: {folder}")
            if old.get("status") == "COMPLETED" and all((folder / f).is_file() and sha(folder/f)==h for f,h in old["artifacts"].items()):
                continue
        folder.mkdir(exist_ok=True)
        # Verify the actual numeric inputs; completion markers alone are insufficient.
        for name in ["native_decisions.csv.gz", "native_events.csv.gz", "model_control_windows.csv.gz",
                     "visual_controls.npz", "visual_alignment.csv.gz", "persistence_events.csv.gz"]:
            if sha(path/name) != binding["artifacts"][name]["sha256"]: raise ValueError(f"Input hash mismatch: {path/name}")
        d, e, qa = enrich(path, cp, config, subject, game)
        for frame, name in [(d, "decisions.csv.gz"), (e, "events.csv.gz")]:
            frame.to_csv(folder/name, index=False)
        write_json(manifest_path, dict(identity, signature=token, status="COMPLETED", qa=qa,
                   artifacts={n:sha(folder/n) for n in ["decisions.csv.gz", "events.csv.gz"]}))
    return root

def load_unit(outputs, subject, game, *, heldout=True, events=True, checkpoint=None):
    ds, es = [], []
    for cp in ([checkpoint] if checkpoint else range(1, 5)):
        p = output(0, outputs) / "derived" / tag(subject, game) / f"leave{cp:02d}"
        m = json.loads((p/"manifest.json").read_text())
        if m["status"] != "COMPLETED": raise ValueError(f"Incomplete derived cache: {p}")
        d = pd.read_csv(p/"decisions.csv.gz")
        if heldout: d = d.loc[d.session.eq(cp)]
        ds.append(d)
        if events:
            e = pd.read_csv(p/"events.csv.gz")
            if heldout: e = e.loc[e.session.eq(cp)]
            es.append(e)
    d = pd.concat(ds, ignore_index=True)
    if heldout: native.validate_keys(d, "one held-out checkpoint per decision")
    return d, pd.concat(es, ignore_index=True) if es else pd.DataFrame()
