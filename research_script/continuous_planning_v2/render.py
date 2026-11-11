"""Dedicated legacy-style scientific layouts. No raw extraction, fits or tests."""
from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from core import FIGURES

COLORS = ['#406B87', '#BC704B', '#648866', '#857192', '#A49657', '#555555']
TITLES = {
    '02-F1':'Initial preference, computation and behavioral state',
    '02-F2':'Evaluation and computation before human action switches',
    '02-F3':'Individual consistency of additional behavioral explanation',
    '02-F4':'Terminal Actor output and held-out readouts',
    '02-F5':'Three competing action transitions',
    '02-F6':'Initial evaluation and dynamic computation range',
    '02-F7':'Computation history during ongoing action bouts',
    '03-F1':'Initiation, continuation and stopping',
    '03-F2':'Search-informed terminal Actor readout',
    '03-F3':'Locked Actor readouts: alignment with human behavior',
    '03-S1':'Supplement · native Q-update arithmetic',
    '04-F1':'Evidence before the next computation choice',
    '04-F1b':'Initial preference moderates evidence–STOP relations?',
    '04-F2':'Supplement · early and late search in the long-budget subset',
    '04-F3':'What remains beyond the strong human-behavior baseline?',
    'N1':'Information on the real behavioral timescale',
    'N2':'Earlier imagined paths and later actions / states',
    'N3':'Urgency proxy · independent validation gate',
    '02-S1':'Supplement · repeated-evaluation geometry',
}
LAYOUT = {'02-F1':(2,3),'02-F2':(3,3),'02-F3':(2,2),'02-F4':(1,2),
    '02-F5':(2,3),'02-F6':(2,3),'02-F7':(2,4),'03-F1':(2,2),'03-F2':(1,2),
    '03-F3':(2,2),'03-S1':(1,2),'04-F1':(2,2),'04-F1b':(2,3),
    '04-F2':(2,2),'04-F3':(2,2),'N1':(2,2),'N2':(3,2),'N3':(2,2),'02-S1':(1,2)}


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
        'axes.labelsize':9,'axes.spines.top':False,'axes.spines.right':False,
        'axes.titlepad':10,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
        'axes.prop_cycle':matplotlib.cycler(color=COLORS),'pdf.fonttype':42,'ps.fonttype':42,
        'savefig.facecolor':'white','lines.linewidth':1.7})


def unavailable(ax, reason, pending=False):
    ax.set_axis_off()
    ax.text(.04,.68,'Independent validation pending' if pending else 'Insufficient observed support',fontsize=10,weight='semibold',transform=ax.transAxes)
    ax.text(.04,.50,'\n'.join(textwrap.wrap(str(reason),46)),fontsize=9,va='top',transform=ax.transAxes,color='#666666')


def finite_units(table):
    if 'y' not in table:return 0
    return table.loc[np.isfinite(pd.to_numeric(table.y,errors='coerce')),'unit'].nunique()


def probability_axis(label):
    return label in {'Survival probability','Probability still computing','STOP hazard',
        'Cause-specific one-step hazard','Fraction of decisions','Fraction of all decisions',
        'P(B > 0)','P(additional computation)','Observed frequency'}


def summary(frame, keys, group=False):
    """Descriptive CIs from units, never frames/events. No reduced-N group CI."""
    rows=[]
    for key,q in frame.groupby(keys,dropna=False,sort=False):
        key=key if isinstance(key,tuple) else (key,)
        values=q.groupby('unit').y.mean().dropna().to_numpy()
        row=dict(zip(keys,key));row['mean']=values.mean() if len(values) and (not group or len(values)==6) else np.nan
        row['n_units']=len(values);row['count']=int(q.n.sum())
        ok=(len(values)==6) if group else (len(values)>=2)
        half=student_t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)) if ok else np.nan
        row['lower']=row['mean']-half;row['upper']=row['mean']+half
        rows.append(row)
    return pd.DataFrame(rows)


def curve(ax,data,meta,group=False):
    q=data.copy()
    if 'series' not in q:q['series']='All'
    if 'x' not in q or not finite_units(q):
        unavailable(ax,'Insufficient observed support (minimum 20 per bin); see panel CSV for N.');return
    categorical=not pd.api.types.is_numeric_dtype(q.x)
    if categorical:
        labels=list(dict.fromkeys(q.x.astype(str)))
        # Horizon labels have a scientific order, not a lexical order.
        if labels and all('-' in s and s.split('-')[0].isdigit() for s in labels):labels=sorted(labels,key=lambda s:int(s.split('-')[0]))
        q['xp']=q.x.astype(str).map({v:i for i,v in enumerate(labels)})
        ax.set_xticks(range(len(labels)),labels,rotation=20,ha='right')
    else:q['xp']=q.x
    # Bin means differ by run: align on the bin identifier, not float x mean.
    q['key']=q['_bin'] if '_bin' in q else q.x
    for i,(series,z) in enumerate(q.groupby('series',sort=False)):
        if not finite_units(z):continue
        color=meta.get('color',COLORS[i%len(COLORS)])
        for _,u in z.groupby('unit'):
            u=u.sort_values('xp');ax.plot(u.xp,u.y,color=color,alpha=.19,lw=.7)
        s=summary(z,['key'],group)
        xpos=z.groupby('key',dropna=False).xp.mean()
        s['xp']=s.key.map(xpos);s=s.sort_values('xp')
        drawstyle='steps-post' if meta.get('step') else 'default'
        ax.plot(s.xp,s['mean'],color=color,marker='o',ms=3,label=series,lw=2,drawstyle=drawstyle)
        ax.fill_between(s.xp.to_numpy(float),s.lower.to_numpy(float),s.upper.to_numpy(float),color=color,alpha=.13,lw=0,
            step='post' if meta.get('step') else None)
    if q.series.nunique()>1 or meta.get('legend'):ax.legend(frameon=False,loc='best',fontsize=7)
    absent=[str(s) for s,z in q.groupby('series',sort=False) if not finite_units(z)]
    if absent:ax.text(.02,.97,'Unsupported: '+', '.join(absent),transform=ax.transAxes,va='top',fontsize=7,color='#666666')
    if 'residual' in meta.get('ylabel','').lower() or 'ΔLL' in meta.get('ylabel','') or 'improvement' in meta.get('ylabel','').lower():
        ax.axhline(0,color='#888888',lw=.7,zorder=0)
    if meta['kind']=='calibration':
        ax.plot([0,1],[0,1],ls='--',color='#777777',lw=.8);ax.set(xlim=(0,1),ylim=(0,1))
    if meta.get('markers'):
        n=q.groupby('x')[['forced','resets']].sum()
        forced=n[n.forced>0];reset=n[n.resets>0]
        for x in forced.index:ax.plot(x,1,marker='v',ms=5,color=COLORS[1],transform=ax.get_xaxis_transform(),clip_on=False)
        for x in reset.index:ax.plot(x,.94,marker='|',ms=6,color=COLORS[2],transform=ax.get_xaxis_transform(),clip_on=False)
        ax.text(.02,.88,f'▼ forced N={int(n.forced.sum())}; | RESET N={int(n.resets.sum())}',transform=ax.transAxes,fontsize=7,va='top')


def paired(ax,data,meta,group=False):
    q=data.copy()
    if 'series' not in q:q['series']='All'
    if not finite_units(q):unavailable(ax,'No complete held-out estimates / observed support. See QA.');return
    labels=list(dict.fromkeys(q.x.astype(str)));mapping={v:i for i,v in enumerate(labels)}
    # Model stages are paired across the same units, not separate series.
    if all(z.x.nunique()==1 for _,z in q.groupby('series')) and q.series.nunique()==len(labels):
        q['series']='Paired models'
    q['xp']=q.x.astype(str).map(mapping)
    for i,(label,z) in enumerate(q.groupby('series',sort=False)):
        color=COLORS[i%len(COLORS)];offset=(i-(q.series.nunique()-1)/2)*.09
        for _,unit in z.groupby('unit'):
            unit=unit.sort_values('xp');ax.plot(unit.xp+offset,unit.y,'o' if meta['kind']=='bar' else 'o-',color=color,alpha=.28,lw=.65,ms=2.8)
        s=summary(z,['xp'],group).sort_values('xp')
        ax.errorbar(s.xp+offset,s['mean'],yerr=[s['mean']-s.lower,s.upper-s['mean']],color=color,fmt='D' if meta['kind']=='bar' else 'D-',ms=4,lw=2,capsize=3,label=label)
    ax.set_xticks(range(len(labels)),['\n'.join(textwrap.wrap(s,10 if meta['kind']=='bar' else 19)) for s in labels],rotation=0)
    if q.series.nunique()>1:ax.legend(frameon=False,fontsize=7)


def effect(ax,data,meta,group=False):
    q=data.copy()
    if 'x' not in q or not finite_units(q):unavailable(ax,'Effect not estimable in all required outer folds.');return
    labels=list(dict.fromkeys(q.x.astype(str)))
    ax.set_xlim(-.5,len(labels)-.5)
    for i,label in enumerate(labels):
        z=q[q.x.astype(str)==label].copy();vals=z.groupby('unit').y.mean().dropna()
        ax.scatter(np.full(len(vals),i)+np.linspace(-.12,.12,len(vals)),vals,s=17,color=COLORS[i%len(COLORS)],alpha=.65,zorder=3)
        s=summary(z,['x'],group)
        if len(s):
            r=s.iloc[0]
            ax.errorbar(i,r['mean'],yerr=[[r['mean']-r.lower],[r.upper-r['mean']]],fmt='D',color='#222222',ms=4,capsize=3)
    ax.axhline(0,color='#888888',lw=.8);ax.set_xticks(range(len(labels)),['\n'.join(textwrap.wrap(s,22)) for s in labels],rotation=0)
    if group and meta.get('group_inference'):
        inference=pd.read_csv(meta['group_inference'])
        mapping={'Evidence block':'stop_evidence','R0 → R1':'human_R0_R1','R1 → R2':'human_R1_R2','R0 → R2':'human_R0_R2'}
        for i,label in enumerate(labels):
            key=mapping.get(label,label)
            rows=inference[(inference.contrast==key)&(inference.game==int(data.game.iloc[0]))]
            if len(rows):
                r=rows.iloc[0]
                message=f'p(adj)={r.p_max_t:.3g}; n={int(r.n_subjects)}' if np.isfinite(r.p_max_t) else f'Not estimable; n={int(r.n_subjects)}'
                ax.text(i,.97,message,transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=6.5)


def weights(ax,data,meta,group=False):
    """Horizontal forest plot makes all readout coefficients legible."""
    labels=list(dict.fromkeys(data.x.astype(str)))
    for i,label in enumerate(labels):
        z=data[data.x==label];v=z.groupby('unit').y.mean().dropna()
        color=COLORS[0] if 'unavailable' not in label else COLORS[2]
        ax.scatter(v,np.full(len(v),i)+np.linspace(-.15,.15,len(v)),s=10,alpha=.4,color=color)
        r=summary(z,['x'],group).iloc[0]
        ax.errorbar(r['mean'],i,xerr=[[r['mean']-r['lower']],[r['upper']-r['mean']]],fmt='D',color='#333333',ms=3,capsize=2)
    ax.axvline(0,lw=.7,color='#999999')
    ax.set_yticks(range(len(labels)),labels,fontsize=7)
    ax.set_ylim(len(labels)-.4,-.6)
    ax.set_xlabel('Standardized descriptive weight');ax.set_ylabel('')


def tile(ax,data,meta,group=False):
    if not finite_units(data):unavailable(ax,'No supported cells; individual N and exclusions saved.');return
    q=data.copy()
    if meta['kind'] in ('geometry','revision_geometry'):
        count_rows=7 if meta['kind']=='geometry' else len(meta['y_edges'])-1
        grid=q.groupby(['row','x']).y.mean().unstack('x').reindex(index=range(count_rows),columns=range(len(meta['x_edges'])-1))
        # Only supported subject values enter a group mean; incomplete cells
        # stay masked in group view, with individual data retained in the CSV.
        if group:
            counts=q[q.y.notna()].groupby(['row','x']).unit.nunique().unstack('x').reindex_like(grid)
            grid=grid.where(counts==6)
        positive=meta['panel'] in ('A','B','C') or meta['kind']=='revision_geometry'
        lim=max(.001,np.nanmax(abs(grid.to_numpy()))) if grid.notna().any().any() else .001
        yedges=np.arange(8)-.5 if meta['kind']=='geometry' else meta['y_edges']
        image=ax.pcolormesh(meta['x_edges'],yedges,grid.to_numpy(),cmap='Blues' if positive else 'RdBu_r',
            vmin=0 if positive else -lim,vmax=lim,shading='flat',rasterized=False)
        if meta['kind']=='geometry':ax.set_yticks(range(7),meta['row_labels'])
        ax.set_xticks([-8,-4,0,4,8]);ax.set_xlim(-8,8)
        # Tails remain in source tables; explicit note prevents pretending full coverage.
        ax.text(.01,1.01,'M ∈ [−8, 8]; tails retained in CSV',transform=ax.transAxes,fontsize=6.5,color='#555555')
        if meta['kind']=='revision_geometry':
            ax.set_yscale('symlog',linthresh=.005)
            ax.set_yticks([-.5,-.02,0,.02,.5],['−0.5','−0.02','0','0.02','0.5'])
            supported=np.flatnonzero(grid.notna().any(axis=1).to_numpy())
            if len(supported):ax.set_ylim(yedges[max(0,supported.min())],yedges[min(len(yedges)-1,supported.max()+1)])
    else:
        rows=list(dict.fromkeys(q.row.astype(str)));cols=list(dict.fromkeys(q.x.astype(str)))
        grid=q.assign(row=q.row.astype(str),x=q.x.astype(str)).groupby(['row','x']).y.mean().unstack('x').reindex(index=rows,columns=cols)
        lim=max(1e-6,np.nanmax(abs(grid.to_numpy()))) if grid.notna().any().any() else 1e-6
        image=ax.imshow(grid,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto')
        ax.set_xticks(range(len(cols)),['\n'.join(textwrap.wrap(c,16)) for c in cols],rotation=0)
        labels=[re.sub(r'^sub\d+_','',s).replace('_game0','').replace('_game1','') for s in rows]
        ax.set_yticks(range(len(rows)),labels)
        if len(rows)*len(cols)<=70:
            for i in range(len(rows)):
                for j in range(len(cols)):
                    v=grid.iloc[i,j]
                    if np.isfinite(v):ax.text(j,i,f'{v:.1e}' if meta.get('figure')=='N1' else f'{v:.3g}',ha='center',va='center',fontsize=6,color='white' if abs(v)>.65*lim else '#222222')
        if meta.get('figure')=='N1':
            bands=q.groupby('x',sort=False).first().reindex([float(c) for c in cols])
            if 'low' in bands and 'high' in bands:
                ax.set_xticks(range(len(cols)),[f'{r.low:g}–{r.high:g}' for r in bands.itertuples()],rotation=25,ha='right')
            ax.set_ylabel('Information block')
    ax.figure.colorbar(image,ax=ax,pad=.025,shrink=.85,label=meta.get('color_label','Value'))


def dual(ax,data,meta,group=False):
    for i,(axis,subset) in enumerate(data.groupby('axis',sort=False)):
        target=ax if i==0 else ax.twiny()
        local=dict(meta,kind='line',color=COLORS[i])
        curve(target,subset,local,group)
        target.set_xlabel('Initial root margin (log probability)' if axis=='margin' else 'Normalized root uncertainty',color=COLORS[i])
        if i:target.spines['top'].set_visible(True);target.tick_params(axis='x',colors=COLORS[i]);target.set_xlim(0,1)
    ax.set_title(meta['title'],loc='left',pad=40,fontsize=10)


def timeline(ax):
    ax.set_axis_off()
    xs=np.linspace(.12,.9,4)
    ax.annotate('',xy=(.96,.72),xytext=(.06,.72),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#555555'))
    for i,x in enumerate(xs):
        ax.plot(x,.72,'o',transform=ax.transAxes,color=COLORS[0])
        ax.text(x,.82,f'x(t+{i})' if i else 'x(t)',transform=ax.transAxes,ha='center')
        ax.text(x,.63,f'a(t+{i})' if i else 'a(t)',transform=ax.transAxes,ha='center',color=COLORS[1])
    ax.text(.02,.96,'REAL TIME: observed frames → human actions',transform=ax.transAxes,fontsize=8)
    for j,x in enumerate(np.linspace(.14,.38,4)):
        ax.add_patch(Rectangle((x,.35),.045,.08,transform=ax.transAxes,fc=COLORS[2],alpha=.7))
        ax.text(x+.023,.31,str(j),transform=ax.transAxes,ha='center',fontsize=7)
    ax.text(.1,.2,'IMAGINATION: event order within decision t',transform=ax.transAxes,fontsize=8)
    ax.annotate('Mismatch becomes\navailable at x(t+2)',xy=(xs[2],.72),xytext=(.54,.37),xycoords='axes fraction',
        fontsize=8,arrowprops=dict(arrowstyle='->',color='#777777'),wrap=True)


def path_images(ax,ctx,data):
    ax.set_axis_off()
    source=ctx.root/'derived/path_images.npz'
    if not source.exists():unavailable(ax,'Native path-image cache not available');return
    with np.load(source,allow_pickle=False) as z:
        for i,label in enumerate(('match','divergence')):
            if label+'_real' not in z:continue
            actual=z[label+'_real'];pred=z[label+'_predicted']
            rec=data[data.label==label].iloc[0]
            actions=json.loads(rec.real_actions) if isinstance(rec.real_actions,str) and rec.real_actions.startswith('[') else []
            path=json.loads(rec.path) if isinstance(rec.path,str) else []
            depths=json.loads(rec.prediction_depths) if isinstance(rec.prediction_depths,str) else []
            # Show initial state plus first three forecast depths; full arrays saved.
            width=.15;count=min(4,len(actual));y=.60-i*.50
            ax.text(.0,y+.35,label.capitalize()+f' · t={rec.origin_time:.2f}s',transform=ax.transAxes,fontsize=8)
            for j in range(count):
                a=ax.inset_axes([.17+j*.19,y+.14,width,.14]);a.imshow(np.moveaxis(actual[j],0,-1));a.axis('off')
                a.set_title(f'x(t+{j})',fontsize=6,pad=1)
                if j and j in depths:
                    b=ax.inset_axes([.17+j*.19,y-.055,width,.14]);b.imshow(np.moveaxis(pred[depths.index(j)],0,-1));b.axis('off')
                    b.set_title(f'pred depth {j}',fontsize=6,pad=1)
            ax.text(.01,y+.20,'Actual',transform=ax.transAxes,fontsize=7)
            ax.text(.01,y-.02,'Imagined',transform=ax.transAxes,fontsize=7)
            ax.text(.17,y-.10,'Actual actions '+str(actions[:3])+' | imagined '+str(path[:3]),transform=ax.transAxes,fontsize=6.5)


def objects(ax,ctx,data):
    ax.set_axis_off()
    if data.empty:return
    for i,row in enumerate(data.head(3).itertuples()):
        path=ctx.args.v1_root/'06_urgency_modulation'/ctx.tag/'annotation_images'/(row.frame_id+'.png')
        if not path.exists():continue
        a=ax.inset_axes([.02+i*.33,.24,.3,.57]);a.imshow(plt.imread(path));a.axis('off');a.set_title(f't={row.time:.2f}s',fontsize=8)
        for obj in json.loads(row.predictions):
            a.plot(obj['x'],obj['y'],'+',color='#E6AE38',ms=8)
            a.text(obj['x']+2,obj['y'],obj['role'],color='#E6AE38',fontsize=5)
    ax.text(.03,.12,'Detection candidates only; velocities / deadline proxy not validated.',transform=ax.transAxes,fontsize=8)


def support_label(data,meta,group=False):
    units='subjects' if group else 'runs'
    valid=finite_units(data)
    if meta['kind']=='weights':return f'{valid} {units}; {data.x.nunique()} coefficient types'
    if 'n' not in data or data.empty:return f'{valid} {units}; N=0'
    if 'category_count' in data:
        n=int(data.groupby('unit').n.max().sum())
        return f'{valid} {units}; {n:,} decisions (shared denominator)'
    keys=[k for k in ('axis','series','row','_bin' if '_bin' in data else 'x') if k in data]
    counts=data.groupby(keys,dropna=False).n.sum() if keys else data.n
    lo,hi=int(counts.min()),int(counts.max())
    support=f'{lo:,}' if lo==hi else f'{lo:,}–{hi:,}'
    label=f'{valid} {units}; N per bin/estimate={support}'
    if 'events' in data and meta.get('kind')=='effect':label+=f'; {int(data.events.sum()):,} events'
    if 'fraction_of_all' in meta:label+=f"\nIncluded decisions: {100*meta['fraction_of_all']:.2f}% of eligible"
    if 'excluded_left_boundary' in meta:label+=f"\nUnknown-onset bouts excluded: {meta['excluded_left_boundary']:,}"
    if 'budget_depth_equal_fraction' in meta:label+=f"\nB = depth in {100*meta['budget_depth_equal_fraction']:.1f}% of decisions"
    if 'coverage' in data:
        label+=f'\nCoverage: {100*data.coverage.min():.1f}–{100*data.coverage.max():.1f}% (per bin/run)'
    return label


def trajectory_support(ax,data,group):
    if not len(data):return
    # Each line's N counts actual observed evidence; no unsupported-bin fill.
    rows=[]
    for label,z in data.groupby('series',sort=False):
        values=[];units=[]
        for time in (-4,-3,-2,-1,0):
            q=z[np.isclose(z.x,time,atol=1e-7)]
            values.append(str(int(q.n.sum())));units.append(str(finite_units(q)))
        short={'First evaluation':'First','Re-evaluation':'Repeat'}.get(label,'All')
        rows.append(short+' N: '+' / '.join(values)+'; units: '+' / '.join(units))
    text='At −4 / −3 / −2 / −1 / 0 s\n'+'\n'.join(rows)
    ax.text(0,-.34,text,transform=ax.transAxes,fontsize=6.5,va='top',color='#666666')


def render(ctx,group=False):
    ctx.verify_seal(ctx.root/'panels')
    style();out=ctx.root/'figures';out.mkdir(exist_ok=True)
    summaries=[]
    for name in FIGURES:
        nrow,ncol=LAYOUT[name]
        width=4.6*ncol;height=4.0*nrow
        if name=='02-F2':height=14
        if name=='03-F2':width=12;height=8
        if name=='N1':width=12;height=9
        if name=='N2':width=11;height=15
        if name=='02-F3':
            fig=plt.figure(figsize=(12,8.5),layout='constrained');gs=fig.add_gridspec(2,2,height_ratios=[1.3,1])
            axes=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,:])]
        else:
            fig,ax=plt.subplots(nrow,ncol,figsize=(width,height),layout='constrained');axes=list(np.atleast_1d(ax).ravel())
        title=f'{TITLES[name]}\n'+('Enduro' if ctx.args.game==0 else 'Pong')+' · '+('six subjects' if group else ctx.tag)
        fig.suptitle(title,fontsize=13,weight='semibold')
        for i,ax in enumerate(axes):
            letter=chr(65+i);stem=ctx.root/'panels'/(name+'_'+letter)
            if not stem.with_suffix('.json').exists():raise ValueError(f'Missing specified panel {name}-{letter}')
            meta=json.loads(stem.with_suffix('.json').read_text());data=pd.read_csv(stem.with_suffix('.csv'))
            meta=dict(meta)
            meta['title']=letter+'  '+'\n'.join(textwrap.wrap(meta['title'],49))
            ax.set_title(meta['title'],loc='left',fontsize=10)
            if name=='02-F7' and letter in ('A','E'):meta['step']=True
            if name=='04-F1b':meta['legend']=True
            kind=meta['kind']
            if kind=='pending':unavailable(ax,meta['note'],pending=True)
            elif kind in ('geometry','tile','revision_geometry'):tile(ax,data,meta,group)
            elif kind in ('paired','bar'):paired(ax,data,meta,group)
            elif kind=='primary':
                ax.set_axis_off()
                for k,h in enumerate(('H1','H2','H3')):
                    child=ax.inset_axes([.07+k*.32,.17,.25,.65])
                    effect(child,data[data.x==h],meta,group)
                    child.set_title({'H1':'Current → switch','H2':'Current → destination','H3':'History → switch'}[h],fontsize=9)
                    if not k:child.set_ylabel('ΔLL (nats)')
                    p=[r for r in meta.get('primary_group',[]) if r['contrast']=={'H1':'current_compute_switch','H2':'current_compute_destination','H3':'all_earlier_computation_increment'}[h]]
                    if p and group:
                        val=p[0].get('p_max_t');child.text(.5,.98,'Group adjusted p='+('NA' if val is None else f'{val:.3g}'),transform=child.transAxes,ha='center',va='top',fontsize=7)
            elif kind=='effect':effect(ax,data,meta,group)
            elif kind=='weights':weights(ax,data,meta,group)
            elif kind=='dual':dual(ax,data,meta,group)
            elif kind=='timeline':timeline(ax)
            elif kind=='paths':path_images(ax,ctx,data)
            elif kind=='objects':objects(ax,ctx,data)
            elif kind=='carry':
                ax.set_axis_off()
                for j,col in enumerate(('budget','switch')):
                    a=ax.inset_axes([.06+j*.56,.16,.35,.62]);z=data.copy();z['y']=z[col]
                    paired(a,z,meta,group);a.set_ylabel('Computations' if col=='budget' else 'Switch probability')
            else:curve(ax,data,meta,group)
            if kind not in ('dual','weights'):ax.set_xlabel(meta['xlabel'])
            if kind!='weights' and not (name=='N1' and letter=='B'):ax.set_ylabel(meta['ylabel'])
            if name=='03-F2' and letter=='B':ax.set_ylabel('Actor-policy cross entropy (nats)')
            if name=='02-F2':
                ax.set_xlim(-4,0);ax.set_xticks([-4,-3,-2,-1,0])
                trajectory_support(ax,data,group)
            if probability_axis(meta['ylabel']) and kind not in ('pending','geometry'):
                lo,hi=ax.get_ylim();ax.set_ylim(max(0,lo),min(1,hi))
            if kind not in ('pending','paths','timeline','objects'):
                n=finite_units(data);obs=int(data.n.sum()) if 'n' in data else 0
                if name!='02-F2':
                    ax.text(0,-.25,support_label(data,meta,group),transform=ax.transAxes,ha='left',va='top',fontsize=6.5,color='#666666')
                summaries.append(dict(figure=name,panel=letter,valid_units=n,summed_panel_n=obs,kind=kind))
                ax.grid(axis='x' if kind=='weights' else 'y',alpha=.15,lw=.5)
        foot='Thin lines / dots: '+('subjects' if group else 'source runs')+'; bold: equal-unit mean; band/bar: pointwise 95% '+('subject CI (all six required)' if group else 'descriptive run interval')
        if name in ('02-F2','02-F6','02-F7','04-F2'):foot+='\nConditional / aligned curves are descriptive, not causal effects. Panel CSVs include support and exclusions.'
        if name=='N3':foot='Independent annotation has not passed; urgency inference is deliberately withheld.'
        if name=='02-F3' and not group:foot+='\nH1–H3: unchanged run estimates; six-subject corrected tests are reported in the group figure / locked CSV.'
        if name=='04-F2':foot+='\nRepeated-update panels require an observed repeated update in BOTH halves; empty panels are not zero effects.'
        fig.supxlabel(foot,fontsize=8)
        fig.savefig(out/(name+'.png'),dpi=300,bbox_inches='tight')
        fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
        plt.close(fig)
        print('rendered',ctx.tag,name,flush=True)
    pd.DataFrame(summaries).to_csv(out/'support_audit.csv',index=False)
    ctx.C.write_json(out/'manifest.json',dict(status='COMPLETED',render_only=True,
        panel_manifest_sha=ctx.C.sha(ctx.root/'panels/manifest.json'),figures=list(FIGURES),dpi=300,
        human_visual_review='REQUIRED before full cohort expansion'))
