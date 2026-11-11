"""V2 entry point. --render-only cannot reach extraction, fitting or inference."""
from __future__ import annotations

import json
from core import Context, parser, FIGURES, primary_lock


def verify(ctx):
    for name in ('derived','fits','panels'):ctx.verify_seal(ctx.root/name)
    primary_lock(ctx)
    for figure in FIGURES:
        for suffix in ('.png','.pdf'):
            path=ctx.root/'figures'/(figure+suffix)
            if not path.exists() or not path.stat().st_size:raise ValueError(f'Missing figure {path}')
    fit=json.loads((ctx.root/'fits/manifest.json').read_text())
    ctx.C.write_json(ctx.root/'completion.json',dict(status='COMPLETED_WITH_QA',
        new_effects_all_estimable=fit['complete'],figures=len(FIGURES),
        primary_group='LOCKED' if (ctx.root/'provenance/v1_primary_max_t.csv').exists() else 'AWAITING_V1_GROUP',
        visual_review='REQUIRED; never inferred from file existence',urgency='INDEPENDENT_ANNOTATION_PENDING'))


def main():
    args=parser().parse_args()
    if args.render_only:
        if args.stage=='group':
            raise ValueError('--render-only does not recompute group tests; use existing group panel bundles')
        ctx=Context(args)
        from render import render
        render(ctx)
        return
    if args.stage=='group':
        from group_report import run
        run(args);return
    ctx=Context(args)
    if args.stage in ('all','derive'):
        from derive import derive
        derive(ctx)
    if args.stage in ('all','fit'):
        from fitting import fit_all
        fit_all(ctx)
    if args.stage in ('all','tables'):
        from panels import build
        build(ctx)
    if args.stage in ('all','render'):
        from render import render
        render(ctx)
    if args.stage in ('all','verify'):verify(ctx)


if __name__=='__main__':main()
