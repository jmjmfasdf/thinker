"""Verify saved individual/group surfaces and compare old/new support and U.

No significance tests are introduced; matched differences retain the existing
run-equal subject summaries and are not re-estimated from smoothed pixels.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.paths import OUTPUTS, tag, GAMES
from common.io import sha, write_json
from panels.mu_heatmaps import POPS, mean_surfaces, MIN_LOCAL


def main():
    dest=OUTPUTS/'02/heatmaps'; m=json.loads((dest/'manifest.json').read_text())
    errors=[]; support=[]
    assert m['status']=='COMPLETED'
    for path,digest in m['artifacts'].items():
        if not Path(path).exists() or sha(path)!=digest:errors.append('Artifact changed: '+path)
    for game in m['games']:
        root=OUTPUTS/'02/panels'/f'game{game}'
        with np.load(root/'heatmap/surfaces.npz') as group:
            for pop in POPS:
                arrays=[]; pos=[]
                for subject in m['subjects']:
                    p=OUTPUTS/'02/panels'/tag(subject,game)/'heatmap/surfaces.npz'
                    with np.load(p) as person:
                        z=person[pop+'_probability']; n=person[pop+'_local_count']
                        pos.append(person[pop+'_positive_local_count']>=MIN_LOCAL)
                        if np.any(np.isfinite(z)&(n<MIN_LOCAL)):errors.append('Unsupported subject pixel: '+str(p))
                        finite=z[np.isfinite(z)]
                        if len(finite) and (finite.min()<0 or finite.max()>1):errors.append('Probability outside [0,1]: '+str(p))
                        np.testing.assert_array_equal(person[pop+'_x'],group[pop+'_x'])
                        np.testing.assert_array_equal(person[pop+'_y'],group[pop+'_y'])
                        arrays.append(z)
                expected,n=mean_surfaces(np.stack(arrays))
                np.testing.assert_allclose(group[pop+'_probability'],expected,rtol=0,atol=1e-14,equal_nan=True)
                np.testing.assert_array_equal(group[pop+'_support'],n)
                np.testing.assert_array_equal(group[pop+'_positive_support'],np.stack(pos).sum(0))
                support.append(dict(game=game,population=pop,version='restored_COSYNE_estimator',
                    supported_fraction=np.isfinite(expected).mean(),grid=180))
        old=pd.read_csv(root/'tables/mu_group_surface.csv')
        for pop,r in old.groupby('population'):
            support.append(dict(game=game,population=pop,version='previous_dynamic_40x40',
                supported_fraction=r['mean'].notna().mean(),grid=40))
    fixed_path=Path('/home/jeongmin/thinker/research_script/outputs/00_cosyne2027_figure/data/flow_layout_candidate_20261001/fig1/figure1_E_exact_MU_surface.npz')
    if fixed_path.exists():
        with np.load(fixed_path) as fixed:
            for game in (1,2):
                for pop in POPS[:2]:
                    z=fixed[f'game{game}_{pop}_probability']
                    support.append(dict(game=game,population=pop,version='original_fixed_model',
                        supported_fraction=np.isfinite(z).mean(),grid=len(z)))
    pd.DataFrame(support).to_csv(dest/'support_comparison.csv',index=False)
    matched=pd.read_csv(dest/'matched_U_subjects.csv'); rows=[]
    for (game,pop),r in matched.groupby(['game','population']):
        rows.append(dict(game=game,game_name=GAMES[game],population=pop,model='dynamic_10M',
            persist=r.persist.mean(),switch=r['switch'].mean(),difference=r.difference.mean(),
            sem=r.difference.sem(),positive_subjects=int(r.difference.gt(0).sum()),n_subjects=len(r)))
    fixed=pd.read_csv(dest/'previous_fixed_U_subjects.csv')
    for game,r in fixed.groupby('game'):
        rows.append(dict(game=game,game_name=GAMES[game],population='overall',model='original_fixed',
            persist=r.control_mean.mean(),switch=r.case_mean.mean(),difference=r.difference.mean(),
            sem=r.difference.sem(),positive_subjects=int(r.difference.gt(0).sum()),n_subjects=len(r)))
    pd.DataFrame(rows).to_csv(dest/'matched_U_comparison.csv',index=False)
    summary=pd.read_csv(dest/'subject_summary.csv')
    means=summary.groupby(['game','population'])[['zero_budget_fraction','zero_U_fraction','positive_U_fraction']].mean()
    means.to_csv(dest/'group_zero_fractions.csv')
    qa=dict(status='PASSED' if not errors else 'FAILED',errors=errors,
        individual_figures=18,group_figures=3,group_mean_verified=True,
        kernel_matches_original='Synthetic recovery test in tests/test_mu_heatmaps.py',
        interpretation='Smoothed descriptive surfaces do not change the existing matched tests; fixed and dynamic models differ in checkpoints, budgets, cohort and matching')
    write_json(dest/'qa.json',qa)
    print(json.dumps(qa)); print(means.to_string()); print(pd.DataFrame(rows).to_string(index=False))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
