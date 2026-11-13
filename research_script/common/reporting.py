"""Each theme owns tables, figures, QA and provenance under outputs/NN/."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .statistics import group_inference
from .paths import GAMES, output, tag
from .io import write_json

def save_tables(root,tables):
    (root/'tables').mkdir(parents=True,exist_ok=True)
    for name,frame in tables.items():
        if not len(frame.columns):frame=pd.DataFrame(columns=['status'])
        suffix='.csv.gz' if name=='predictions' else '.csv'
        frame.to_csv(root/'tables'/(name+suffix),index=False)

def figures(root,tables,title):
    folder=root/'figures';folder.mkdir(exist_ok=True)
    if 'gate_curves' in tables and len(tables['gate_curves']):
        fig,axes=plt.subplots(1,2,figsize=(9,3.7),squeeze=False)
        for ax,population in zip(axes[0],['start','continue']):
            d=tables['gate_curves'];d=d.loc[d.population.eq(population)&d.source.eq('ema')]
            ax.plot(d.voc,d.predicted_continue,label='Gate probability')
            ax.plot(d.voc,d.actual_continue,'o-',label='Observed continuation')
            ax.set(xlabel='Pre-choice EMA VOC',ylabel='Continuation probability',title=population,ylim=(0,1));ax.legend(fontsize=8)
        fig.suptitle(title);fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(folder/f'gate_calibration.{ext}',dpi=180)
        plt.close(fig)
    effects=tables.get('effects',pd.DataFrame())
    if len(effects):
        d=effects.loc[effects.hac_lag.eq(128)].copy()
        if len(d):
            fig,ax=plt.subplots(figsize=(8,max(3,len(d)*.23)))
            y=np.arange(len(d));ok=d.status.eq('ok')
            ax.errorbar(d.loc[ok,'mean'],y[ok],xerr=1.96*d.loc[ok,'se'],fmt='o',color='#376c99')
            labels=d.population.astype(str)+' | '+d.term.astype(str)
            labels=labels.where(ok,labels+' ['+d.status.astype(str)+']')
            ax.set_yticks(y,labels,fontsize=7);ax.axvline(0,color='gray',lw=.7)
            ax.set(xlabel='Standardized association; temporal 95% CI',title=title)
            fig.tight_layout()
            for ext in ['png','pdf']:fig.savefig(folder/f'associations.{ext}',dpi=180)
            plt.close(fig)
    scores=tables.get('run_scores',tables.get('actor_run_scores',pd.DataFrame()))
    if len(scores):
        metric='log_score' if 'log_score' in scores else 'cross_entropy'
        if 'population' not in scores:scores=scores.assign(population='Actor soft target')
        pops=scores.population.unique();fig,axes=plt.subplots(1,len(pops),figsize=(max(5,len(pops)*3),4),squeeze=False)
        for ax,pop in zip(axes[0],pops):
            d=scores.loc[scores.population.eq(pop)].groupby('model')[metric].agg(['mean','sem'])
            ax.bar(np.arange(len(d)),d['mean'],yerr=d['sem'].fillna(0),color='#376c99')
            ax.set_xticks(np.arange(len(d)),d.index,rotation=65,ha='right',fontsize=7)
            ax.set(title=pop,ylabel=metric+' (nat/decision)')
        fig.suptitle(title);fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(folder/f'heldout_scores.{ext}',dpi=180)
        plt.close(fig)

def aggregate(number,outputs,subjects,games):
    root=output(number,outputs);all_tables={};missing=[]
    for s in subjects:
        for g in games:
            folder=root/tag(s,g)
            if not (folder/'manifest.json').exists():missing.append(tag(s,g));continue
            import json
            manifest=json.loads((folder/'manifest.json').read_text())
            if manifest.get('status')!='COMPLETED':missing.append(tag(s,g));continue
            for path in (folder/'tables').glob('*.csv'):
                df=pd.read_csv(path)
                if len(df): all_tables.setdefault(path.stem,[]).append(df.assign(subject=s,game=g))
    tables={k:pd.concat(v,ignore_index=True) for k,v in all_tables.items()}
    inference=[]
    if 'effects' in tables:
        effects=tables['effects'];effects=effects.loc[effects.hac_lag.eq(128)].rename(columns={'mean':'value'})
        keys=['game','family','population','term']+(['source'] if 'source' in effects else [])
        for family,part in effects.groupby('family'):
            inf=group_inference(part,keys)
            inf['test_family']=family;inference.append(inf)
    for name in ['run_scores','actor_run_scores']:
        if name not in tables:continue
        d=tables[name];metric='log_score' if name=='run_scores' else 'cross_entropy'
        if 'population' not in d:d=d.assign(population='actor_soft_target')
        keys=['subject','game','run_label','population']
        wide=d.pivot(index=keys,columns='model',values=metric)
        models=sorted(wide.columns);diffs=[]
        for before,after in zip(models[:-1],models[1:]):
            v=(wide[after]-wide[before])*(1 if metric=='log_score' else -1)
            diffs.append(v.rename('value').reset_index().assign(contrast=before+' -> '+after))
        if diffs:
            runs=pd.concat(diffs,ignore_index=True);tables[name+'_increments']=runs
            units=runs.groupby(['subject','game','population','contrast']).value.mean().reset_index()
            tables[name+'_subject_increments']=units
            inf=group_inference(units,['game','population','contrast']);inf['test_family']=name;inference.append(inf)
    if 'matched_run_effects' in tables:
        units=tables['matched_run_effects'].groupby(['subject','game','population','term']).value.mean().reset_index()
        tables['matched_subject_effects']=units
        inf=group_inference(units,['game','population','term']);inf['test_family']='matched';inference.append(inf)
    for name in ['rsa_run_effects','decoding_run_effects','encoding_run_effects','localization_run_effects']:
        if name not in tables:continue
        d=tables[name];keys=['game','roi','term']+(['stage'] if 'stage' in d else [])
        units=d.groupby(['subject',*keys]).value.mean().reset_index()
        tables[name.replace('run','subject')]=units
        inf=group_inference(units,keys);inf['test_family']=name;inference.append(inf)
    if inference:tables['group_inference']=pd.concat(inference,ignore_index=True)
    folder=root/'group';folder.mkdir(parents=True,exist_ok=True);save_tables(folder,tables)
    if 'group_inference' in tables:
        d=tables['group_inference']; d=d.loc[d['mean'].notna()].copy()
        if len(d) and len(d)<150:
            fig,ax=plt.subplots(figsize=(10,max(4,len(d)*.17)))
            ax.errorbar(d['mean'],np.arange(len(d)),xerr=1.96*d['sem'].fillna(0),fmt='o',ms=3)
            labels=d.game.map(GAMES)+' | '+d.get('population',pd.Series('',index=d.index)).astype(str)+' | '+d.get('term',d.get('contrast',pd.Series('',index=d.index))).fillna('').astype(str)
            ax.set_yticks(np.arange(len(d)),labels,fontsize=6);ax.axvline(0,color='gray',lw=.7)
            ax.set_xlabel('Equal-participant mean ± 1.96 SEM (descriptive interval)');fig.tight_layout()
            (folder/'figures').mkdir(exist_ok=True)
            for ext in ['png','pdf']:fig.savefig(folder/'figures'/f'group_effects.{ext}',dpi=180)
            plt.close(fig)
        elif len(d):
            group_keys=['game','test_family']+(['stage'] if 'stage' in d else [])
            for key,part in d.groupby(group_keys,dropna=False):
                game,family=key[:2];stage=str(key[2]) if len(key)>2 and pd.notna(key[2]) else 'default'
                matrix=part.pivot_table(index='roi',columns='term',values='mean',aggfunc='mean')
                fig,ax=plt.subplots(figsize=(8,max(5,len(matrix)*.13)))
                limit=np.nanmax(np.abs(matrix.to_numpy())) or 1
                im=ax.imshow(matrix.to_numpy(),aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
                ax.set_xticks(np.arange(len(matrix.columns)),matrix.columns,rotation=40,ha='right',fontsize=8)
                ax.set_yticks(np.arange(len(matrix)),matrix.index,fontsize=5)
                ax.set_title(f'{GAMES[int(game)]}: {family}\n{stage}; equal-subject effect, no thresholding',fontsize=9)
                fig.colorbar(im,ax=ax);fig.tight_layout();(folder/'figures').mkdir(exist_ok=True)
                for ext in ['png','pdf']:fig.savefig(folder/'figures'/f'{family}_{stage}_game{game}.{ext}',dpi=180)
                plt.close(fig)
    write_json(folder/'manifest.json',dict(status='PARTIAL' if missing else 'COMPLETED',missing=missing,
        subjects=subjects,games=games,independent_unit='subject',expected_subjects=6,
        prediction_aggregation='within-run mean -> equal runs within subject -> equal subjects',
        correction='exact synchronized sign flips, max-|t| within explicitly named families; NA for incomplete family'))
    return not missing
