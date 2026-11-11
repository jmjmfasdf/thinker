"""Separate new inference families, immutable V1 primary family, group panels."""
from __future__ import annotations

import copy
import json
import shutil
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from core import Context, PRIMARY
from render import render

FAMILIES = {
    'process': ['process_switch','process_destination'],
    'actor': [f'{target}_{a}_{b}' for target in ('actor','human') for a,b in (('R0','R1'),('R1','R2'),('R0','R2'))],
    'control': ['stop_evidence','stop_moderation'],
}


def infer(C,frame,family,games=(0,1)):
    keys=[(g,c) for g in games for c in FAMILIES[family]]
    matrix=np.full((6,len(keys)),np.nan)
    for j,(g,c) in enumerate(keys):
        q=frame[(frame.game==g)&(frame.contrast==c)]
        if q.subject.duplicated().any():raise ValueError('Duplicate subject effect')
        for r in q.itertuples():matrix[int(r.subject)-1,j]=r.delta_ll
    statistic,raw,adjusted=C.exact_max_t(matrix)
    rows=[]
    for j,(g,c) in enumerate(keys):
        v=matrix[:,j];valid=np.isfinite(v).all();mean=v.mean() if valid else np.nan
        width=student_t.ppf(.975,5)*v.std(ddof=1)/np.sqrt(6) if valid else np.nan
        rows.append(dict(family=family,game=g,contrast=c,delta_ll=mean,ci_low=mean-width,ci_high=mean+width,
            t=statistic[j],p_exact=raw[j],p_max_t=adjusted[j],n_subjects=int(np.isfinite(v).sum()),
            decision='not estimable' if not valid else 'supported' if mean>0 and adjusted[j]<.05 else 'additional explanation not established',
            reason='' if valid else 'One or more of the six subjects not estimable; no reduced-N test',
            **{f'sub{s+1:03d}':v[s] for s in range(6)}))
    return pd.DataFrame(rows)


def aggregate_panel(data,kind):
    if 'y' not in data or 'x' not in data:return data
    q=data.copy()
    if kind=='tile' and q.row.astype(str).str.startswith('sub').all():
        q['row']=q.subject.map(lambda s:f'sub{s:03d}')
    keys=['subject','game']+[c for c in ('series','axis','row','half') if c in q]
    if '_bin' in q:keys+=['_bin']
    else:keys+=['x']
    # Only units meeting each panel's support criterion enter the equal-run mean.
    agg={'y':('y','mean'),'n':('n','sum'),'valid_runs':('y','count')}
    if 'x' not in keys:agg['x']=('x','mean')
    for col in ('events','forced','resets','bouts','category_count'):
        if col in q:agg[col]=(col,'sum')
    for col in ('coverage','switch','budget','low','high'):
        if col in q:agg[col]=(col,'mean')
    result=q.groupby(keys,dropna=False,sort=False).agg(**agg).reset_index()
    result['unit']=result.subject.map(lambda s:f'sub{s:03d}')
    return result


def run(args):
    contexts=[];effects=[]
    for game in (0,1):
        for subject in range(1,7):
            a=copy.copy(args);a.subject=subject;a.game=game
            c=Context(a);c.verify_seal(c.root/'fits');c.verify_seal(c.root/'panels')
            contexts.append(c);effects.append(pd.read_csv(c.root/'fits/effects.csv'))
    C=contexts[0].C
    source=args.v1_root/'08_group_report/primary_max_t.csv'
    v1manifest=args.v1_root/'08_group_report/manifest.json'
    if not source.exists() or json.loads(v1manifest.read_text())['status']!='COMPLETED':
        raise ValueError('Awaiting canonical V1 group inference; cannot replace it with a new primary test')
    report=args.output_root/'group_report';report.mkdir(exist_ok=True)
    frozen=report/'v1_primary_max_t.csv'
    if frozen.exists() and frozen.read_bytes()!=source.read_bytes():raise ValueError('Primary hash changed')
    shutil.copyfile(source,frozen)
    frame=pd.concat(effects,ignore_index=True)
    frame.to_csv(report/'subject_effects.csv',index=False)
    result=pd.concat([infer(C,frame,f) for f in FAMILIES],ignore_index=True)
    result.to_csv(report/'new_max_t.csv',index=False)
    paired=frame.pivot(index=['subject','family','contrast'],columns='game',values='delta_ll').reset_index()
    paired['delta_ll']=paired[1]-paired[0];paired['game']='Pong minus Enduro'
    differences=pd.concat([infer(C,paired,f,games=('Pong minus Enduro',)) for f in FAMILIES],ignore_index=True)
    differences.to_csv(report/'paired_game_contrasts.csv',index=False)
    for c in contexts:
        # Numeric values and raw input hashes must still equal the immutable V1
        # tables, including NaNs and estimability flags.
        from core import primary_lock
        primary_lock(c)
    for game in (0,1):
        selected=[c for c in contexts if c.args.game==game]
        c=copy.copy(selected[0]);c.root=args.output_root/f'group_game{game}'
        out=c.root/'panels';out.mkdir(parents=True,exist_ok=True)
        for meta_path in sorted((selected[0].root/'panels').glob('*.json')):
            if meta_path.name=='manifest.json':continue
            meta=json.loads(meta_path.read_text());name=meta_path.stem
            metas=[json.loads((s.root/'panels'/(name+'.json')).read_text()) for s in selected]
            for field in ('included_decisions','all_decisions','excluded_left_boundary'):
                if field in meta:meta[field]=sum(m[field] for m in metas)
            if 'included_decisions' in meta:
                meta['fraction_of_all']=meta['included_decisions']/meta['all_decisions']
            elif 'fraction_of_all' in meta:
                meta['fraction_of_all']=float(np.mean([m['fraction_of_all'] for m in metas]))
            if 'budget_depth_equal_fraction' in meta:
                meta['budget_depth_equal_fraction']=float(np.mean([m['budget_depth_equal_fraction'] for m in metas]))
            frames=[pd.read_csv(s.root/'panels'/(name+'.csv')) for s in selected]
            if meta['kind'] in ('paths','objects','timeline','pending'):
                data=frames[0]
            else:data=aggregate_panel(pd.concat(frames,ignore_index=True),meta['kind'])
            data.to_csv(out/(name+'.csv'),index=False)
            meta.update(group_size=6,unit='subject; equal-run aggregation',
                group_inference=str(report/'new_max_t.csv'),primary_inference=str(frozen))
            C.write_json(out/(name+'.json'),meta)
        (c.root/'derived').mkdir(exist_ok=True)
        shutil.copyfile(selected[0].root/'derived/path_images.npz',c.root/'derived/path_images.npz')
        c.seal(out,group_size=6)
        render(c,group=True)
    C.write_json(report/'manifest.json',dict(status='COMPLETED',group_size=6,
        primary_source=str(source),primary_sha256=C.sha(source),v2_primary_sha256=C.sha(frozen),
        new_families={k:len(v)*2 for k,v in FAMILIES.items()},sign_patterns=64,
        posthoc_reanalysis=True,group_incomplete_effects=int(result.delta_ll.isna().sum())))
    lines=['# Continuous planning V2', '', 'Post-hoc observational reanalysis; no claim of independent confirmation.',
           'V1 H1–H3 copied byte-for-byte. New families use 64 synchronized subject sign flips.', '']
    for r in result.itertuples():lines.append(f'- game{r.game}, {r.contrast}: {r.decision}; ΔLL={r.delta_ll:.5g}, adjusted p={r.p_max_t:.5g}, n={r.n_subjects}.')
    (report/'summary.md').write_text('\n'.join(lines)+'\n')
