"""COSYNE Fig2E kernel estimator, with individual and equal-person maps.

The original figure pooled decisions within each person, then averaged the
person-specific probability surfaces where >=4 people had local support.
It did not pool people's event counts and did not require all six people.
U remains the absolute valid revision sum / actual computation count, with
B=0 mapped to U=0. No observations are removed to obtain a smoother picture.
"""
from pathlib import Path
import html
import json
import shutil
import time
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import fftconvolve
from common.io import sha, signature, write_json
from common.paths import GAMES, tag, output
from .data import population

GRID = 180
BANDWIDTH = .04
MIN_LOCAL = 12
MIN_SUBJECTS = 4
POPS = ('noop_to_active', 'active_to_other', 'active_to_noop')
LABELS = ('No input → active', 'Active → any other action', 'Active → no input')
STEM = 'cosyne_fig2E_MU_switch_surface'
REFERENCE = Path('/home/jeongmin/thinker/research_script/outputs/00_cosyne2027_figure/code/archive/flow_layout_candidate_20261001/fig1/build_exact_mu_surface.py')
CONTRACT = dict(grid=GRID, bandwidth_fraction=BANDWIDTH, min_local_decisions=MIN_LOCAL,
    support_radius_in_bandwidths=2, min_subjects=MIN_SUBJECTS,
    within_person='Pool recorded decisions across runs, as in original COSYNE Fig2E',
    group='Arithmetic mean of supported subject probabilities; each subject has equal weight',
    U='Sum of valid absolute sampled-branch root-Q revisions / actual computations; B=0 gives U=0',
    scale='Within-run M z score (ddof=1, retained from theme02); U / run mean U; zero-mean runs retain U=0',
    axes='Game-specific pooled 1st–99th percentiles, shared by all subjects and directions; U axis starts at zero',
    colors='Per-direction maximum group 97th percentile across requested games; shared by individuals and groups',
    caveat='Gaussian smoothing spreads the observed U=0 mass locally; colored positive-U pixels alone do not establish positive-update support')


def mean_surfaces(stack, minimum=MIN_SUBJECTS):
    support = np.isfinite(stack).sum(axis=0)
    average = np.divide(np.nansum(stack, axis=0), support,
        out=np.full(stack.shape[1:], np.nan), where=support >= minimum)
    return average, support


def kernel_surface(frame, xlim, ylim):
    """Original padded histogram-convolution estimator, returned per person.

    Local support counts actual observations in a two-bandwidth ellipse,
    rather than thresholding a normalized smoothed histogram height.
    """
    if np.ptp(xlim) <= 0 or np.ptp(ylim) <= 0:
        raise ValueError('Increasing finite display limits required')
    dx, dy = np.ptp(xlim)/(GRID-1), np.ptp(ylim)/(GRID-1)
    bwx, bwy = BANDWIDTH*np.ptp(xlim), BANDWIDTH*np.ptp(ylim)
    px, py = int(np.ceil(5*bwx/dx)), int(np.ceil(5*bwy/dy))
    xe = xlim[0]-(px+.5)*dx + np.arange(GRID+2*px+1)*dx
    ye = ylim[0]-(py+.5)*dy + np.arange(GRID+2*py+1)*dy
    sigma = (bwy/dy, bwx/dx)
    yy, xx = np.mgrid[-int(np.ceil(2*sigma[0])):int(np.ceil(2*sigma[0]))+1,
                      -int(np.ceil(2*sigma[1])):int(np.ceil(2*sigma[1]))+1]
    disk = ((xx/sigma[1])**2+(yy/sigma[0])**2 <= 4).astype(float)
    sl = np.s_[py:py+GRID, px:px+GRID]
    def hist(f, weights=None):
        return np.histogram2d(f.U_relative, f.M_z, bins=(ye, xe), weights=weights)[0]
    count = hist(frame)
    den = gaussian_filter(count, sigma, mode='constant', truncate=5)[sl]
    num = gaussian_filter(hist(frame, frame.outcome), sigma, mode='constant', truncate=5)[sl]
    local = np.maximum(0, np.rint(fftconvolve(count, disk, mode='same')[sl])).astype(int)
    positive = np.maximum(0, np.rint(fftconvolve(hist(frame.loc[frame.U_all.gt(0)]), disk, mode='same')[sl])).astype(int)
    prob = np.divide(num, den, out=np.full_like(den, np.nan), where=(local >= MIN_LOCAL)&(den > 0))
    np.clip(prob, 0, 1, out=prob)
    return prob, local, positive


def marginal(values, limits):
    dx = np.ptp(limits)/(GRID-1); bw = BANDWIDTH*np.ptp(limits)
    pad = int(np.ceil(5*bw/dx))
    edge = limits[0]-(pad+.5)*dx+np.arange(GRID+2*pad+1)*dx
    hist = np.histogram(values, bins=edge)[0].astype(float)
    smooth = gaussian_filter1d(hist, bw/dx, mode='constant', truncate=5)[pad:pad+GRID]
    area = np.trapz(smooth, dx=dx)
    return smooth/area if area > 0 else np.full(GRID, np.nan)


def load_decisions(outputs, subject, game):
    root = output(2, outputs)/'panels'/tag(subject, game)/'heatmap'
    root.mkdir(parents=True, exist_ok=True)
    sources = {}
    for cp in range(1, 5):
        p = output(0, outputs)/'derived'/tag(subject, game)/f'leave{cp:02d}'
        m = json.loads((p/'manifest.json').read_text())
        if m['status'] != 'COMPLETED': raise ValueError('Incomplete source: '+str(p))
        sources[str(p/'decisions.csv.gz')] = m['artifacts']['decisions.csv.gz']
    identity = signature(dict(sources=sources, loader='mu-heldout-input-v1'))
    meta = root/'input_manifest.json'; cache = root/'decisions.csv.gz'
    if meta.exists() and cache.exists():
        m = json.loads(meta.read_text())
        if m.get('signature') == identity and sha(cache) == m['sha256']:
            return pd.read_csv(cache), sources
    cols = ['session', 'run_label', 'real_step_id', 'M', 'U_all', 'U_sum',
            'valid_update_count', 'additional_computations', 'incumbent_action',
            'human_action', 'n_actions', 'scored']
    frames = []
    for cp, (path, expected) in enumerate(sources.items(), 1):
        if sha(path) != expected: raise ValueError('Source checksum changed: '+path)
        d = pd.read_csv(path, usecols=cols)
        frames.append(d.loc[d.session.eq(cp)])
    d = pd.concat(frames, ignore_index=True)
    if d.duplicated(['run_label','real_step_id']).any(): raise ValueError('Duplicate held-out decision')
    np.testing.assert_allclose(d.U_all, d.U_sum/np.maximum(d.additional_computations, 1), rtol=1e-10, atol=1e-12)
    assert d.loc[d.additional_computations.eq(0), 'U_all'].eq(0).all()
    mg = d.groupby('run_label').M
    d['M_z'] = (d.M-mg.transform('mean'))/mg.transform('std').replace(0, np.nan)
    mean_u = d.groupby('run_label').U_all.transform('mean')
    d['U_relative'] = np.divide(d.U_all, mean_u, out=np.zeros(len(d)), where=mean_u.gt(0))
    d.to_csv(cache, index=False)
    write_json(meta, dict(signature=identity, sources=sources, sha256=sha(cache)))
    return d, sources


def axis_limits(frames):
    allrows = pd.concat([population(d, 'overall') for d in frames])
    finite = allrows.loc[np.isfinite(allrows.M_z)&np.isfinite(allrows.U_relative)]
    xlim = np.quantile(finite.M_z, [.01, .99])
    ymax = float(finite.U_relative.quantile(.99))
    fallback = False
    if ymax <= 0:
        positive = finite.loc[finite.U_relative.gt(0), 'U_relative']
        ymax = float(positive.quantile(.99)) if len(positive) else 1.
        fallback = True
    if np.ptp(xlim) <= 0: xlim = np.array([-1., 1.])
    return xlim, np.array([0., ymax]), fallback


def draw_surface(path, game, who, surfaces, color_limits, summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    fig = plt.figure(figsize=(14.8, 5.2), layout='constrained')
    outer = fig.add_gridspec(1, 3)
    group = who == 'Group'
    for i, (pop, label) in enumerate(zip(POPS, LABELS)):
        a = surfaces[pop]; x, y = a['x'], a['y']
        inner = outer[i].subgridspec(2, 3, height_ratios=[.18, 1], width_ratios=[1, .13, .06])
        ax = fig.add_subplot(inner[1,0]); top = fig.add_subplot(inner[0,0], sharex=ax)
        side = fig.add_subplot(inner[1,1], sharey=ax); cb = fig.add_subplot(inner[1,2])
        cmap = plt.colormaps['magma'].copy(); cmap.set_bad('white')
        im = ax.imshow(np.ma.masked_invalid(a['probability']), origin='lower',
            extent=(x[0],x[-1],y[0],y[-1]), aspect='auto', interpolation='bilinear',
            cmap=cmap,vmin=0,vmax=color_limits[pop],rasterized=True)
        ax.set(xlabel='Initial margin M (within-run z)',ylabel='U / run mean U')
        ax.axvline(0,color='#999999',ls=':',lw=.7)
        if y[-1] >= 1: ax.axhline(1,color='#999999',ls=':',lw=.7)
        if not np.isfinite(a['probability']).any():
            ax.text(.5,.5,'Insufficient local support',transform=ax.transAxes,ha='center')
        color = '#367aa8' if game != 2 else '#dd8a35'
        top.plot(x,a['density_x'],color=color,lw=1)
        top.fill_between(x,a['density_x'],color=color,alpha=.2)
        side.plot(a['density_y'],y,color=color,lw=1)
        side.fill_betweenx(y,a['density_y'],color=color,alpha=.2)
        top.set_axis_off(); side.set_axis_off(); top.set_title(label,fontsize=11)
        fig.colorbar(im,cax=cb,label='Switch probability',format=PercentFormatter(1),extend='max')
        row = summaries.loc[summaries.population.eq(pop)]
        n = int(row.n.sum()); zero = float(row.zero_U_fraction.mean())
        ax.text(.02,.98,f'n={n:,}; U=0: {zero:.1%}',transform=ax.transAxes,va='top',fontsize=8,
            bbox=dict(facecolor='white',edgecolor='none',alpha=.85))
    subtitle = 'Equal-person mean · at least 4 of 6 supported' if group else 'All held-out runs · at least 12 local observations'
    fig.suptitle(f'{GAMES[game]} · {who}\n{subtitle}',fontsize=14,fontweight='bold')
    files = []
    for ext in ['png','pdf']:
        f = path.with_suffix('.'+ext); f.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(f,dpi=200,bbox_inches='tight'); files.append(f)
    plt.close(fig)
    caption = ('COSYNE Fig2E Gaussian kernel probability estimator: 180×180 grid, bandwidth 4% of game-specific axis spans, '
        '12 actual decisions within a two-bandwidth ellipse. All runs pooled within subject; group is the arithmetic mean '
        'of locally supported subject probabilities (minimum 4/6), not pooled event counts. Common game axes and '
        'direction-specific color scales shared across subjects/games. Color saturates at the maximum group 97th percentile; '
        'the upper colorbar triangle marks saturation. U=sum(valid absolute root-Q revisions)/actual computation count; '
        'B=0 yields U=0 and remains included. U divided by its run mean; a run with mean U=0 retains U=0. '
        'Smoothing spreads U=0 mass locally; positive-U color alone does not imply observed positive updates. '
        'Support counts and positive-update local counts are saved. Active→any other includes release of input. '
        'These are descriptive probabilities, with no inference from pixel counts.')
    path.with_suffix('.caption.txt').write_text(caption+'\n')
    return files


def draw_support(root, game, surfaces):
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    fig, axes = plt.subplots(2,3,figsize=(13,7),layout='constrained')
    norm=BoundaryNorm(np.arange(-.5,7.5),7)
    for j,(pop,label) in enumerate(zip(POPS,LABELS)):
        a=surfaces[pop]; extent=(*a['x'][[0,-1]],*a['y'][[0,-1]])
        for i,key in enumerate(['support','positive_support']):
            im=axes[i,j].imshow(a[key],origin='lower',extent=extent,aspect='auto',cmap='viridis',norm=norm,interpolation='nearest')
            axes[i,j].set(title=label,xlabel='M (within-run z)',ylabel='U / run mean U')
    fig.colorbar(im,ax=axes.ravel().tolist(),ticks=range(7),label='Number of subjects')
    fig.suptitle(f'{GAMES[game]} · local sample support\nTop: all decisions; bottom: ≥12 observations with U>0',fontsize=13)
    p=root/'heatmap_support'
    for ext in ['png','pdf']: fig.savefig(p.with_suffix('.'+ext),dpi=150,bbox_inches='tight')
    plt.close(fig)


def archive_file(path, stamp):
    if path.exists():
        dest = path.parent/'archive'/stamp/path.name; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest)


def build(outputs, subjects=(1,2,3,4,5,6), games=(0,1,2)):
    subjects=list(subjects); games=list(games)
    if subjects != [1,2,3,4,5,6]: raise ValueError('Group heatmaps require the six-subject cohort')
    dest=output(2,outputs)/'heatmaps'; dest.mkdir(parents=True,exist_ok=True)
    source=sha(__file__); key=signature(dict(source=source,subjects=subjects,games=games,contract=CONTRACT))
    done=dest/'manifest.json'
    if done.exists():
        previous=json.loads(done.read_text())
        if previous.get('signature')==key and previous.get('status')=='COMPLETED' and all(
                Path(p).exists() and sha(p)==h for p,h in previous['sources'].items()) and all(
                Path(p).exists() and sha(p)==h for p,h in previous['artifacts'].items()):
            return previous
    stamp=time.strftime('%Y%m%dT%H%M%S')
    archive_file(done,stamp)
    write_json(done,dict(status='RUNNING',signature=key,started_at=time.time(),contract=CONTRACT))
    shutil.copy2(__file__,dest/f'source_{stamp}.py')
    sources={}; results={}; summaries=[]; support_rows=[]; axes_audit=[]; matched=[]
    for game in games:
        frames={}
        for subject in subjects:
            d,src=load_decisions(outputs,subject,game); frames[subject]=d; sources.update(src)
            print(f'Heatmap inputs: {tag(subject,game)}, {len(d):,} held-out decisions',flush=True)
        xlim,ylim,fallback=axis_limits(list(frames.values()))
        axes_audit.append(dict(game=game,xlim=xlim,ylim=ylim,positive_quantile_fallback=fallback))
        results[game]={s:{} for s in subjects}; results[game]['group']={}
        for pop in POPS:
            stack=[]; positives=[]; dxs=[]; dys=[]
            for subject,d in frames.items():
                r=population(d,pop); finite=np.isfinite(r.M_z)&np.isfinite(r.U_relative)
                missing=int((~finite).sum()); r=r.loc[finite]
                prob,local,positive=kernel_surface(r,xlim,ylim)
                dx=marginal(r.M_z,xlim); dy=marginal(r.U_relative,ylim)
                item=dict(probability=prob,local_count=local,positive_local_count=positive,
                    x=np.linspace(*xlim,GRID),y=np.linspace(*ylim,GRID),density_x=dx,density_y=dy)
                results[game][subject][pop]=item
                stack.append(prob); positives.append(positive>=MIN_LOCAL); dxs.append(dx); dys.append(dy)
                summaries.append(dict(subject=subject,game=game,population=pop,n=len(r),invalid_axes=missing,
                    switch_probability=r.outcome.mean(),zero_budget_fraction=r.additional_computations.eq(0).mean(),
                    zero_U_fraction=r.U_all.eq(0).mean(),positive_U_fraction=r.U_all.gt(0).mean(),
                    beyond_old_U4=r.U_relative.gt(4).mean(),displayed_fraction=(r.M_z.between(*xlim)&r.U_relative.between(*ylim)).mean(),
                    supported_fraction=np.isfinite(prob).mean()))
            mean,count=mean_surfaces(np.stack(stack))
            results[game]['group'][pop]=dict(probability=mean,support=count,
                positive_support=np.stack(positives).sum(0),x=np.linspace(*xlim,GRID),y=np.linspace(*ylim,GRID),
                density_x=np.nanmean(dxs,axis=0),density_y=np.nanmean(dys,axis=0))
            for threshold in range(1,7):
                support_rows.append(dict(game=game,population=pop,min_subjects=threshold,
                    supported_fraction=(count>=threshold).mean(),
                    positive_supported_fraction=(np.stack(positives).sum(0)>=threshold).mean()))
        for subject in subjects:
            p=output(2,outputs)/'panels'/tag(subject,game)/'tables/matched_means.csv'
            if p.exists():
                sources[str(p)]=sha(p)
                q=pd.read_csv(p); q=q.loc[q.metric.eq('U_relative')]
                matched.append(q.groupby('population')[['persist','switch','difference']].mean().reset_index().assign(subject=subject,game=game))
        print(f'Heatmap surfaces: {GAMES[game]} complete',flush=True)
    color_limits={}
    for pop in POPS:
        quantiles=[]
        for game in games:
            a=results[game]['group'][pop]['probability']; a=a[np.isfinite(a)]
            if len(a):quantiles.append(float(np.quantile(a,.97)))
        color_limits[pop]=max(quantiles+[.01])
    summary=pd.DataFrame(summaries); artifacts={}; gallery=[]
    for game in games:
        for who, surfaces in results[game].items():
            unit=f'game{game}' if who=='group' else tag(who,game)
            root=output(2,outputs)/'panels'/unit; data=root/'heatmap'; data.mkdir(parents=True,exist_ok=True)
            path=root/'figures'/f'{STEM}_game{game}'
            for ext in ['.png','.pdf','.caption.txt']: archive_file(path.with_suffix(ext),stamp)
            subset=summary.loc[summary.game.eq(game)]
            if who!='group':subset=subset.loc[subset.subject.eq(who)]
            label='Group' if who=='group' else f'sub{who:03d}'
            files=draw_surface(path,game,label,surfaces,color_limits,subset)
            npz=data/'surfaces.npz'
            np.savez_compressed(npz,**{f'{pop}_{k}':v for pop,a in surfaces.items() for k,v in a.items()})
            gallery.append(dict(game=game,subject=who,png=str(files[0]),pdf=str(files[1])))
            for p in [*files,path.with_suffix('.caption.txt'),npz]:artifacts[str(p)]=sha(p)
            write_json(data/'manifest.json',dict(status='COMPLETED',source_sha256=source,contract=CONTRACT,
                game=game,subject=who,colors=color_limits,surfaces_sha256=sha(npz),
                figures={str(p):sha(p) for p in files}))
            if who=='group':
                draw_support(data,game,surfaces)
                for ext in ['png','pdf']: artifacts[str(data/f'heatmap_support.{ext}')]=sha(data/f'heatmap_support.{ext}')
        print(f'Heatmap figures: {GAMES[game]} individuals and group saved',flush=True)
    tables=dict(subject_summary=summary,support_coverage=pd.DataFrame(support_rows),
        matched_U_subjects=pd.concat(matched,ignore_index=True) if matched else pd.DataFrame())
    old=Path('/home/jeongmin/thinker/research_script/outputs/00_cosyne2027_figure/data/figure2_model_links_20260930/tables/subject_contrasts.csv')
    if old.exists():
        sources[str(old)]=sha(old); q=pd.read_csv(old); q=q.loc[q.metric.eq('mean_update')]
        tables['previous_fixed_U_subjects']=q
    for name,frame in tables.items():
        p=dest/f'{name}.csv';frame.to_csv(p,index=False);artifacts[str(p)]=sha(p)
    cards=[]
    import os
    for game in games:
        cards.append(f'<h2>{GAMES[game]}</h2>')
        for entry in sorted([x for x in gallery if x['game']==game],key=lambda x:0 if x['subject']=='group' else x['subject']):
            label='Group average (≥4/6 supported)' if entry['subject']=='group' else f"sub{entry['subject']:03d}"
            img=os.path.relpath(entry['png'],dest); pdf=os.path.relpath(entry['pdf'],dest)
            cards.append(f'<article><h3>{label}</h3><a href="{html.escape(pdf)}">PDF</a><img src="{html.escape(img)}" loading="lazy"></article>')
        rel=os.path.relpath(output(2,outputs)/'panels'/f'game{game}'/'heatmap/heatmap_support.png',dest)
        cards.append(f'<details><summary>Local support: all observations vs U&gt;0</summary><img src="{rel}"></details>')
    (dest/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>M–U heatmaps</title>'
        '<style>body{max-width:1500px;margin:30px auto;font:16px/1.5 system-ui;padding:20px;background:#f5f7fa}img{width:100%}article{background:white;padding:15px;margin:20px 0}</style>'
        '<h1>Individual and group M–U switch probability</h1><p>180×180 Gaussian surfaces, same estimator as COSYNE. '
        'Group averages subject probabilities, not observations. Each subject pools all held-out runs. Minimum 4/6 local support. '
        'The U=0 mass is retained and smoothed; consult the support sheet before interpreting positive-U colors. '
        'Common axes within game and common probability colors per transition across games/subjects. Colorbar triangles indicate saturation.</p>'
        +''.join(cards),encoding='utf-8')
    artifacts[str(dest/'index.html')]=sha(dest/'index.html')
    if REFERENCE.exists():sources[str(REFERENCE)]=sha(REFERENCE)
    result=dict(status='COMPLETED',signature=key,completed_at=time.time(),contract=CONTRACT,
        subjects=subjects,games=games,axes=axes_audit,color_limits=color_limits,sources=sources,artifacts=artifacts,
        figures=len(gallery),reference=str(REFERENCE))
    write_json(done,result)
    return result


def append_group_files(root, game, files):
    m=json.loads((root/'heatmap/manifest.json').read_text())
    if m.get('status')!='COMPLETED' or m['source_sha256']!=sha(__file__):
        raise ValueError('Rebuild smooth heatmaps before rendering theme02')
    for p,digest in m['figures'].items():
        if sha(p)!=digest:raise ValueError('Heatmap changed: '+p)
        files.append(p)
