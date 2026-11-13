"""Verify separate-game panel artifacts and report scientific support limits."""
from pathlib import Path
import json
import sys
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.paths import OUTPUTS,GAMES
from common.io import sha,write_json


def main():
    rows=[]; errors=[]; support=[]
    for number in ['01','02','03','04','05','09','10','12']:
        for game in range(3):
            root=OUTPUTS/number/'panels'/f'game{game}';mp=root/'manifest.json'
            if not mp.exists():errors.append(str(mp)+' missing');continue
            m=json.loads(mp.read_text())
            if m['game']!=game or m['subjects']!=list(range(1,7)):errors.append(str(mp)+' cohort mismatch')
            for name,digest in m.get('figures',{}).items():
                path=root/name
                if not path.exists() or sha(path)!=digest:errors.append(str(path)+' hash mismatch')
                if f'_game{game}.' not in path.name:errors.append(str(path)+' game name mismatch')
                if not path.with_suffix('.caption.txt').exists():errors.append(str(path)+' caption missing')
            rows.append(dict(theme=number,game=game,game_name=GAMES[game],status=m['status'],
                             png=len(list((root/'figures').glob('*.png'))),pdf=len(list((root/'figures').glob('*.pdf')))))
            if number=='04':
                f=root/'tables/endpoint_sensitivity_qa.csv'
                if f.exists():
                    d=pd.read_csv(f)
                    support.append(dict(theme=number,game=game,panel='COSYNE Fig3B',
                        eligible_decisions=int(d.n.sum()),participants_with_eligible_decisions=int(d.loc[d.n.gt(0),'subject'].nunique()),
                        status='INSUFFICIENT_ROOT_CHOICES' if d.n.sum()==0 else 'SEE_LOCAL_SUPPORT_TABLE',
                        note='Needs >=2 root choices from the same SEARCH head; undefined is not a zero effect'))
                f=root/'tables/endpoint_sensitivity.csv'
                if f.exists():
                    d=pd.read_csv(f)
                    if len(d):support[-1]['common_M_grid_points']=int(d.groupby('M')['observed'].count().eq(6).sum())
            if number=='02':
                f=root/'tables/k_curves.csv'
                if f.exists():
                    d=pd.read_csv(f)
                    for pop,r in d.groupby('population'):
                        support.append(dict(theme=number,game=game,panel='COSYNE Fig3C '+pop,
                            common_percentiles=int(r.groupby('percentile').probability.count().eq(6).sum()),
                            status='COMMON_SUPPORT_RECORDED',note='Blank percentile intervals are unsupported, not zero probability'))
    dest=OUTPUTS/'00/panel_campaign';dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(dest/'artifact_qa.csv',index=False)
    pd.DataFrame(support).to_csv(dest/'estimation_coverage.csv',index=False)
    write_json(dest/'qa.json',dict(status='PASSED' if not errors else 'FAILED',errors=errors,
        n_theme_games=len(rows),n_png=sum(r['png'] for r in rows),n_pdf=sum(r['pdf'] for r in rows),
        estimator_support_is_reported_separately=True))
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(support).to_string(index=False))
    if errors:raise SystemExit('\n'.join(errors))


if __name__=='__main__':main()
