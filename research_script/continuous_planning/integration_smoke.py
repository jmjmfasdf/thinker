"""Create a clearly marked, subsampled DERIVED-cache integration fixture.

This never samples the production analysis. Fixture manifests are smoke=True
and the full-cohort verifier refuses them. Existing native-cache files are read
only; no capture, checkpoint, or historical analysis is modified.
"""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import common as C


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--rows-per-run", type=int, default=400)
    args = p.parse_args()
    source = args.source
    original = json.loads((source/"manifest.json").read_text())
    name = source.name
    out = args.output_root/"00_native_cache"/name
    if out.exists(): raise FileExistsError("Preserve existing smoke fixture")
    out.mkdir(parents=True)
    d = pd.read_csv(source/"decisions.csv.gz")
    selected = []
    for _, rows in d.groupby("run", sort=False):
        selected.extend(rows.index[np.linspace(0, len(rows)-1, min(len(rows), args.rows_per_run)).astype(int)])
    selected = np.sort(selected)
    mapping = {int(old): new for new, old in enumerate(d.loc[selected, "decision_id"])}
    sampled = d.loc[selected].copy(); sampled["decision_id"] = np.arange(len(sampled))
    sampled.to_csv(out/"decisions.csv.gz", index=False)
    events = pd.read_csv(source/"search_events.csv.gz")
    events = events[events.decision_id.isin(mapping)].copy()
    events["decision_id"] = events.decision_id.map(mapping)
    events.to_csv(out/"search_events.csv.gz", index=False)
    forecasts = pd.read_csv(source/"forecasts.csv.gz")
    forecasts = forecasts[forecasts.origin_id.isin(mapping)].copy()
    forecasts["origin_id"] = forecasts.origin_id.map(mapping)
    for column in ("target_id", "state_target_id"):
        forecasts[column] = forecasts[column].map(mapping).fillna(-1).astype(int)
    forecasts.to_csv(out/"forecasts.csv.gz", index=False)
    forecasts[forecasts.state_target_id>=0].to_csv(out/"realized_forecasts.csv.gz", index=False)
    with np.load(source/"visual.npz", allow_pickle=False) as z:
        np.savez_compressed(out/"visual.npz", lowres=z["lowres"][selected])
    shutil.copy2(source/"feature_contract.json", out/"feature_contract.json")
    manifest = dict(original, signature=C.digest(dict(source=original["signature"], selected=selected)),
                    smoke=True, integration_subsample=True, complete_real=len(sampled), scored_real=int(sampled.scored.sum()))
    C.finish(out, manifest)
    print(f"Integration-only fixture: {len(sampled)} decisions; {out}", flush=True)


if __name__ == "__main__":
    main()
