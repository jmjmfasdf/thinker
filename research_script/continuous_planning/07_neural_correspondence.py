"""Export native-time regressors; fMRI fitting is intentionally deferred."""
import numpy as np
import pandas as pd
import common as C


def main():
    args = C.parser(__doc__).parse_args()
    d, _, cache = C.load_cache(args)
    root, manifest, done = C.start_manifest(args, C.MODULES["07"], dict(cache=cache["signature"]))
    if done: return
    manifest["smoke"] = cache["smoke"]
    # No arbitrary TR, scanner offset, HRF or fabricated intra-search event times.
    names = ["decision_id", "subject", "game", "session", "run", "episode", "time", "human_action", "incumbent", "switch",
             "root_value", "root_margin", "budget", "evidence_signed", "evidence_abs", "mismatch_pixel", "history_cumulative"]
    names += [c for c in d if c.startswith(("root_p_", "actor_p_", "old_plan_support_", "visual_"))]
    export = d[names].copy()
    export["duration"] = d.groupby("episode").time.shift(-1)-d.time
    export["scanner_onset"] = np.nan
    export.to_csv(root/"native_time_regressors.csv.gz", index=False)
    C.write_json(root/"neural_contract.json", dict(
        status="EXPORT_ONLY_FMRI_NOT_RUN",
        required_inputs=["BOLD and acquisition metadata", "scanner-to-behavior offset per run", "prespecified anatomical ROIs", "motion/confound files"],
        persistent="Human action and pre-search support: zero-order hold over observed intervals",
        impulse="Human switch: recorded real decision time",
        parametric="Past plan, arrived mismatch, new evidence, budget, terminal Actor: enclosing real decision",
        no_event_timestamps=True,
        planned_primary="Canonical HRF plus temporal derivative, held-out encoding delta R2 by block deletion/refit",
        planned_secondary="FIR 0-16 seconds, descriptive BOLD timing only",
        inference="Six-subject synchronized sign flips; information-block x anatomical ROI correction",
        prohibited="Do not infer computational ordering/information propagation from BOLD lags"))
    C.finish(root, manifest, neural_status="EXPORT_ONLY_FMRI_NOT_RUN", rows=len(export))


if __name__ == "__main__":
    main()
