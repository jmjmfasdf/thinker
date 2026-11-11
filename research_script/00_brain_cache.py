#!/usr/bin/env python3
"""Build the fMRI/TR/ROI cache used by analyses 06, 07, and 08.

This is the only neural-cache build entry point. It reads the shared
``outputs/00_cache`` tables, reconstructs the 02/03/04 core variables, and
builds a native-time forward model before sampling at fMRI TRs. Persistent
P/M/Q0/action states use zero-order hold; transitions use additive exact-time
impulses; R/E/G/A use decision-locked parametric modulation; preceding-window
boxcars are retained as sensitivity blocks. Imaginary events remain explicitly
anchored only to their enclosing real step because event-specific timestamps
were not observed. Analyses 06--08 are read-only consumers. No prior 04 output,
legacy 07 encoding script, im_vectors, or raw topology is required.

Examples
--------
python research_script/00_brain_cache.py --subject 3 --game 1 \
  --roi-set all --max-roi-voxels 4000
python research_script/00_brain_cache.py --subject 3 --game 1 \
  --roi-set all --max-roi-voxels 0
"""

from __future__ import annotations

import argparse

import planning_brain_common as COMMON
import planning_trace_common as TRACE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    COMMON.add_shared_cli_arguments(parser, require_subject_game=False)
    TRACE.add_trace_cli_arguments(parser)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run synthetic cache-contract tests without reading study data.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild matching alignment/process/ROI variants in place.",
    )
    parser.add_argument("--overwrite-alignment", action="store_true")
    parser.add_argument("--overwrite-process", action="store_true")
    parser.add_argument("--overwrite-behavior", action="store_true")
    parser.add_argument("--overwrite-roi", action="store_true")
    parser.add_argument(
        "--planning-trace-only",
        action="store_true",
        help=(
            "Read and validate the existing canonical v4 BOLD/ROI cache, then "
            "build only the separate concat29/mean29 planning-trace v1 sidecar."
        ),
    )
    parser.add_argument(
        "--overwrite-planning-trace",
        action="store_true",
        help=(
            "Replace matching planning-trace v1 sidecars. This never overwrites "
            "the canonical process, BOLD, ROI, RAM, or visual cache."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        result = {
            "planning_brain_common": COMMON.self_test(),
            "planning_trace_common": TRACE.self_test(),
        }
        COMMON.log(f"Brain-cache self-test PASS: {result}")
        return
    if args.subject is None or args.game is None:
        raise SystemExit("--subject and --game are required unless --self-test is used")
    if args.overwrite_planning_trace and not args.planning_trace_only:
        raise SystemExit(
            "--overwrite-planning-trace requires --planning-trace-only; it is "
            "kept separate from every canonical cache overwrite flag"
        )
    if args.planning_trace_only:
        requested_max_runs = args.max_runs
        # The canonical 06--08 adapter normally requires >=2 runs for LORO.
        # Trace construction itself is run-local, so permit an explicit
        # one-run smoke while still exercising the unchanged adapter contract.
        if requested_max_runs == 1:
            args.max_runs = 2
        run_specs, _, audit = COMMON.shared_prepare_from_args(args)
        if requested_max_runs is not None:
            run_specs = run_specs[: int(requested_max_runs)]
        args.max_runs = requested_max_runs
        trace_audit = TRACE.build_subject_trace_cache(
            run_specs,
            audit,
            args.planning_trace_root,
            overwrite=bool(args.overwrite_planning_trace),
        )
        COMMON.log(
            "Planning-trace sidecar complete: "
            f"{trace_audit['trace_root']} | runs={trace_audit['n_runs']} | "
            f"concat={len(TRACE.CONCAT_COMPONENT_NAMES)} | "
            f"mean={len(TRACE.MEAN_COMPONENT_NAMES)} | "
            "canonical-v4-cache-modified=False"
        )
        return
    sessions = COMMON.parse_sessions(args.sessions)
    roi_include = (
        [part.strip() for part in args.roi_include.split(",") if part.strip()]
        if args.roi_include
        else None
    )
    run_specs, _, audit = COMMON.prepare_brain_runs(
        subject=args.subject,
        game=args.game,
        sessions=sessions,
        metrics_root=args.metrics_root,
        trace_root=args.trace_root,
        behav_root=args.behav_root,
        fmri_root=args.fmri_root,
        atlas_root=args.atlas_root,
        cache_root=args.cache_root,
        roi_set=args.roi_set,
        roi_include=roi_include,
        max_roi_voxels=args.max_roi_voxels,
        required_status2=args.required_status2,
        allow_smoothed_fallback=args.allow_smoothed_fallback,
        overwrite_cache=bool(args.overwrite),
        allow_cache_build=True,
        max_runs=args.max_runs,
        include_topology=bool(args.include_topology_cache),
        topology_max_depth=int(args.topology_max_depth),
        feature_root=args.feature_root,
        overwrite_alignment=bool(args.overwrite_alignment),
        overwrite_process=bool(args.overwrite_process),
        overwrite_behavior=bool(args.overwrite_behavior),
        overwrite_roi=bool(args.overwrite_roi),
        legacy_process=bool(args.legacy_process_cache),
    )
    COMMON.log(
        "Brain cache complete: "
        f"{audit['cache_root']} | runs={len(run_specs)} | "
        f"ROIs={len(audit['roi_names'])} | "
        f"feature-source={audit.get('feature_source_token', 'legacy')} | "
        f"topology={audit['include_topology']}"
    )


if __name__ == "__main__":
    main()
