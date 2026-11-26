"""Single deterministic lifecycle for the current 00–13 research sequence."""
from pathlib import Path
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
import ast
import contextlib
import importlib
import inspect
import json
import os
import tempfile

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
from .paths import REPO, OUTPUTS
from .research_inputs import DEFAULT_INPUT, BlockedInput, capture_metadata, prepare_native, sha
from .result_store import ResultStore, ResultStoreError, _verify
from .io import signature, clean

THEMES={'00':'native_preparation','01':'behavioral_analysis','02':'model_validation',
'03':'computation_allocation','04':'computation_utility','05':'evaluation_revision',
'06':'search_to_policy','07':'computation_to_behavior','08':'evaluation_continuity',
'09':'evaluation_accumulation','10':'control_patterns','11':'neural_encoding',
'12':'neural_decoding','13':'neural_rsa'}
MODULES={'01':'human_behavior','02':'validation_allocation','03':'validation_allocation',
'04':'utility_current','05':'revision_behavior','06':'policy_continuity',
'07':'revision_behavior','08':'policy_continuity','09':'temporal_pipeline',
'10':'control_pattern_pipeline','11':'neural_current','12':'neural_current','13':'neural_current'}


def parser(number):
    p=ArgumentParser(description=f'{number}: {THEMES[number]}; latest native visualization; deterministic subject/game outputs')
    p.add_argument('--subjects','--subject',type=int,nargs='+',default=[1,2])
    p.add_argument('--games','--game',nargs='+',choices=['pong','enduro','spaceinvaders','mario'],default=['pong','enduro','spaceinvaders'])
    p.add_argument('--input-root',type=Path,default=DEFAULT_INPUT)
    p.add_argument('--outputs-root',type=Path,default=OUTPUTS)
    p.add_argument('--behavior-root',type=Path)
    p.add_argument('--scalar-cache',type=Path)
    p.add_argument('--training-root',type=Path)
    p.add_argument('--utility-input',type=Path)
    p.add_argument('--forecast-input',type=Path)
    p.add_argument('--rich-reuse',type=Path)
    p.add_argument('--stage',choices=['all','prepare','analyze','render'],default='all')
    p.add_argument('--render-only',action='store_true')
    p.add_argument('--resume',action='store_true',help='Reuse verified stages; changed analysis or plot signatures invalidate only descendants')
    p.add_argument('--aggregate',action='store_true',help='Equal participant summaries from current saved results; no raw extraction')
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--seed',type=int,default=701)
    p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--window',type=float,default=.5);p.add_argument('--horizon',type=float,default=.5)
    p.add_argument('--windows-seconds',type=float,nargs='+',default=[.25,.5,1.])
    p.add_argument('--horizons-seconds',type=float,nargs='+',default=[.25,.5,1.])
    p.add_argument('--basis-samples',type=int,default=2048);p.add_argument('--jacobian-samples',type=int,default=512)
    p.add_argument('--energy',type=float,default=.99);p.add_argument('--max-rank',type=int,default=128)
    p.add_argument('--chunk-size',type=int,default=256);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--null-repeats',type=int,default=100)
    return p


def code_signature(module, render=False):
    source=Path(module.__file__).read_text();tree=ast.parse(source)
    nodes=[n for n in tree.body if (isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and
        (n.name.startswith('render') or n.name.startswith('_render') or n.name.startswith('_draw') or n.name.startswith('_plot'))) == render]
    if render:
        # Plot-only changes cannot invalidate numerical analysis.
        return signature(ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False))
    dependencies=[]
    for name in ('inference.py','prediction.py','outcomes.py','matching.py','features.py','io.py','native.py','research_inputs.py','native_replay.py'):
        dependencies.append(sha(Path(__file__).with_name(name)))
    backend={'temporal_pipeline':['evaluation_accumulation.py','bout_evaluation.py'],
        'control_pattern_pipeline':['control_patterns.py'], 'neural_current':['neural.py'],
        'validation_allocation':['behavior.py'], 'revision_behavior':['behavior.py','replay_themes.py','value_updates.py'],
        'policy_continuity':['replay_themes.py','value_updates.py']}.get(module.__name__.split('.')[-1],[])
    dependencies.extend(sha(Path(module.__file__).with_name(name)) for name in backend)
    if module.__name__.split('.')[-1]!='human_behavior':
        base=Path(__file__).resolve().parents[1]
        for name in ('analyses/replay_themes.py','analyses/value_updates.py','panels/mu_heatmaps.py','panels/data.py','panels/estimators.py'):
            dependencies.append(sha(base/name))
    return signature(dict(code=ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False),dependencies=dependencies))


def read_tables(root):
    import pandas as pd
    result={}
    for p in sorted((Path(root)/'tables').glob('*.csv*')):
        name=p.name.removesuffix('.gz').removesuffix('.csv')
        try:result[name]=pd.read_csv(p)
        except pd.errors.EmptyDataError:result[name]=pd.DataFrame(columns=['status'])
    return result


def write_outputs(stage,tables,arrays):
    import pandas as pd
    for name,frame in tables.items():
        if not len(frame.columns):frame=pd.DataFrame(columns=['status'])
        stage.write_table(name,frame)
    for name,values in arrays.items():stage.write_arrays(name,**values)


def _plot_neural_heatmap(ax, frame, row, column, label, limit=None):
    """Display one saved estimate per cell; absent support stays visibly missing."""
    import numpy as np
    matrix=frame.pivot(index=row,columns=column,values='value').sort_index().sort_index(axis=1)
    values=matrix.to_numpy(float)
    finite=values[np.isfinite(values)]
    scale=limit or (float(np.max(np.abs(finite))) if finite.size else 1.) or 1.
    cmap=__import__('matplotlib').colormaps['RdBu_r'].copy();cmap.set_bad('#eeeeee')
    im=ax.imshow(np.ma.masked_invalid(values),aspect='auto',cmap=cmap,vmin=-scale,vmax=scale)
    ax.set(xticks=np.arange(len(matrix.columns)),xticklabels=[str(v).replace('_',' ') for v in matrix.columns],
           yticks=np.arange(len(matrix.index)),yticklabels=[str(v).replace('_',' ') for v in matrix.index],title=label)
    ax.tick_params(axis='x',rotation=55)
    if values.size<=80:
        for (y,x),v in np.ndenumerate(values):
            if np.isfinite(v):ax.text(x,y,f'{v:.3f}',ha='center',va='center',fontsize=7,
                                      color='white' if abs(v)>.6*scale else 'black')
    return im


def render_neural(root,tables,title):
    """Neural estimates on ROI × session/term maps from saved tables alone.

    Encoding and decoding estimates are evaluated on held-out sessions.
    Localization is descriptive within-run residual correlation. RSA panels
    distinguish the two explicitly fitted control sets; no population inference.
    """
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(root)/'figures';out.mkdir(parents=True,exist_ok=True);written=[]
    def save(fig,name):
        fig.tight_layout()
        for ext in ('png','pdf'):
            path=out/f'{name}.{ext}';fig.savefig(path,dpi=180,bbox_inches='tight');written.append(str(path))
        plt.close(fig)
    for name,metric in [('encoding_run_effects','Held-out ΔR²: evaluation / VOC beyond initial state and amount'),
                        ('decoding_run_effects','Held-out MSE reduction from adding ROI activity'),
                        ('localization_run_effects','Residual ROI-mean correlation (descriptive)'),
                        ('rsa_run_effects','Partial rank RSA correlation (descriptive)')]:
        d=tables.get(name,pd.DataFrame())
        if d.empty:continue
        required={'run_label','session','roi','term','value'}
        if not required.issubset(d):raise ValueError(f'{name} lacks neural plot columns: {sorted(required-set(d))}')
        d=d.loc[np.isfinite(pd.to_numeric(d.value,errors='coerce'))].copy()
        if 'status' in d:d=d.loc[d.status.eq('ok')]
        if d.empty:continue
        participants=['subject'] if 'subject' in d else []
        keys=[*participants,'session','roi','term']+(['stage'] if 'stage' in d else [])
        # First equal runs within session. Never pool voxel or pair counts as weights.
        session=d.groupby(keys,dropna=False).value.mean().reset_index()
        participant=d.groupby([*participants,'roi','term']+(['stage'] if 'stage' in d else []),dropna=False).value.mean().reset_index()
        averaging='Equal runs within participant, then equal participants'
        if name=='encoding_run_effects':
            grouped=len(participants)>0 and session.subject.nunique()>1
            axis='subject' if grouped else 'session';estimates=participant if grouped else session
            for term,z in estimates.groupby('term',sort=False):
                fig,ax=plt.subplots(figsize=(max(6,len(z[axis].unique())*1.2),max(4,len(z.roi.unique())*.28)))
                im=_plot_neural_heatmap(ax,z,'roi',axis,title+'\n'+metric)
                ax.set_xlabel('Participant (equal run means)' if grouped else 'Held-out session');ax.set_ylabel('ROI')
                fig.colorbar(im,ax=ax,label='ΔR² (full − controls)')
                save(fig,'encoding_participants' if grouped else 'encoding_heldout_sessions')
        elif name=='rsa_run_effects':
            mean=participant.groupby(['stage','roi','term'],dropna=False).value.mean().reset_index()
            stages=sorted(mean.stage.unique());limit=float(mean.value.abs().max()) or 1.
            fig,axs=plt.subplots(1,len(stages),squeeze=False,figsize=(7*len(stages),max(4,len(mean.roi.unique())*.28)))
            for ax,stage in zip(axs.ravel(),stages):
                im=_plot_neural_heatmap(ax,mean.loc[mean.stage.eq(stage)],'roi','term',stage.replace('_',' '),limit)
                fig.colorbar(im,ax=ax,label='Partial rank correlation')
            fig.suptitle(title+' · RSA\n'+averaging)
            save(fig,'rsa_control_sets')
        else:
            mean=participant.groupby(['roi','term'],dropna=False).value.mean().reset_index()
            fig,ax=plt.subplots(figsize=(max(7,len(mean.term.unique())*1.4),max(4,len(mean.roi.unique())*.28)))
            im=_plot_neural_heatmap(ax,mean,'roi','term',title+'\n'+metric)
            fig.colorbar(im,ax=ax,label='MSE reduction' if name.startswith('decoding') else 'Residual correlation')
            ax.set_xlabel(averaging)
            save(fig,name.removesuffix('_run_effects')+'_roi_terms')
    return written


def _render_signature(number,module):
    # The dispatcher/renderer are separate from numerical-analysis fingerprints.
    return dict(module=code_signature(module,True),
        delegated=code_signature(importlib.import_module('analyses.'+MODULES[number]),True) if number in MODULES else None,
        dispatcher=signature(inspect.getsource(render)),
        neural=signature([inspect.getsource(render_neural),inspect.getsource(_plot_neural_heatmap)])
               if number in ('11','12','13') else None,version='canonical-v3')


def _qa_metadata(store,qa,tables=None):
    qa=clean(qa or {})
    store.meta['analysis']=qa
    store.meta['scientific_status']=str(qa.get('status','UNSPECIFIED')).upper()
    counts={name:len(frame) for name,frame in (tables or {}).items()}
    if counts:store.meta['table_rows']=counts
    store.log('Analysis QA: '+json.dumps(qa,ensure_ascii=False,sort_keys=True))
    if counts:store.log('Saved table rows: '+json.dumps(counts,sort_keys=True))


def _scientific_status(meta):
    return str(meta.get('scientific_status',meta.get('analysis',{}).get('status','NOT_ANALYZED'))).upper()


def _legacy_guard(root):
    """Never overwrite an old numbered layout while its dependencies are live."""
    root=Path(root)
    if root.exists() and not (root/'logs/run.json').is_file():
        raise ResultStoreError('Existing result has the previous output layout and cannot be overwritten automatically: '
            +str(root)+'. Complete the explicit guarded output migration first; use --outputs-root for an isolated check.')


def render(number,module,root,tables,title):
    if number in ('02','03','05','06','07','08'):module.render(number,root,tables,title)
    elif number in ('11','12','13'):render_neural(root,tables,title)
    else:module.render(root,tables,title)


def analyze(number,module,args,subject,game,prepared,workdir):
    if number=='01':return (*module.analyze(subject,game,behavior_root=args.behavior_root),{})
    if number=='09':return module.analyze(args,subject,game,workdir)
    if number in ('11','12','13'):
        tables,qa=module.run_current_neural(number,prepared['decisions'],args.input_root,subject,game)
        return tables,qa,{}
    if number in ('04','10'):return module.analyze(args,subject,game,prepared,workdir)
    return module.analyze(number,args,subject,game,prepared,workdir)


def _input_identity(number,args,subject,game):
    if number=='01':
        from analyses.human_behavior import discover
        sources=discover(subject=subject,game=game,behavior_root=args.behavior_root)
        # discover returns source dictionaries; hash actual files, not a subject roster.
        files=[]
        for source in sources:
            path=source.get('path',source.get('source')) if isinstance(source,dict) else source
            if path is None:raise ValueError('Behavior discovery returned no source path')
            files.append(dict(role='behavior',path=str(Path(path).resolve()),sha256=sha(path)))
            # Mario exact input combinations come from BK2, not the canonical NPZ.
            if isinstance(source,dict) and source.get('bk2_path'):
                bk=source['bk2_path']
                files.append(dict(role='exact_buttons',path=str(Path(bk).resolve()),sha256=sha(bk)))
        if not files:raise BlockedInput(f'No raw behavioral sources: sub{subject:03d}/{game}')
        return dict(dataset='human_behavior',sources=files)
    return capture_metadata(args.input_root,subject,game)


def _external_signature(number,args,meta,subject,game):
    """Fingerprint read inputs outside the immutable native capture.

    Missing optional inputs have an explicit fingerprint, so their later arrival
    triggers analysis. Only whitelisted scientific files are read; no job env or
    credentials are copied into result metadata.
    """
    records=[]
    def add(path,role):
        path=Path(path)
        record=dict(role=role,path=str(path.resolve()),exists=path.is_file())
        if record['exists']:record['sha256']=sha(path)
        records.append(record)
    if number=='02':
        base=Path(getattr(args,'training_root',None) or REPO/'logs/thinker')
        logical=meta.get('checkpoint',{}).get('logical_xpid')
        folder=base if (base/'logs.csv').is_file() else base/logical if logical else None
        if folder is not None:
            for name in ('logs.csv','logs_model.csv','meta.json','meta_model.json'):add(folder/name,'training')
    if number=='04' and getattr(args,'utility_input',None):
        supplied=Path(args.utility_input)
        csv=supplied/'utility_states.csv' if supplied.is_dir() else supplied
        manifest=csv.parent/'manifest.json'
        for path in (csv,manifest,csv.parent/'utility_scores.csv'):add(path,'utility')
        if manifest.is_file():
            declared=json.loads(manifest.read_text())
            for source in declared.get('utility_sources',[]):
                for key in ('path','pairs_path'):
                    if source.get(key):
                        path=Path(source[key]);add(path if path.is_absolute() else csv.parent/path,'paired_source')
    if number=='08' and getattr(args,'forecast_input',None):
        path=Path(args.forecast_input);add(path,'forecast');add(path.parent/'manifest.json','forecast_provenance')
    if number in ('11','12','13'):
        from analyses.neural_current import input_fingerprint
        records.append(dict(role='current_neural_inputs',provenance=input_fingerprint(subject,game)))
    if number=='10' and getattr(args,'rich_reuse',None):
        path=Path(args.rich_reuse)
        candidates=(path/f'sub{subject:03d}'/game,path/game,path)
        folder=next((p for p in candidates if (p/'extract.done.json').is_file()),path)
        for name in ('audit.json','extract.done.json','extraction.json'):add(folder/name,'rich_provenance')
        if (folder/'extract.done.json').is_file():
            from analyses.control_pattern_pipeline import RICH_FILES
            for name in RICH_FILES:add(folder/name,'rich_artifact')
    return signature(records)


def unit(number,args,subject,game):
    number=f'{int(number):02d}';root=Path(args.outputs_root)/number/f'sub{subject:03d}'/game
    _legacy_guard(root)
    if number=='00':
        if args.render_only or args.stage=='render':
            raise ValueError('00 performs preparation only and has no figures; --render-only / --stage render are unsupported')
        try:prepared=prepare_native(args,subject,game)
        except FileNotFoundError as error:
            with ResultStore(root,dict(theme=number,subject=subject,game=game),resume=True) as store:
                store.blocked_input(str(error))
            return dict(subject=subject,game=game,status='BLOCKED_INPUT',path=str(root),reason=str(error))
        return dict(subject=subject,game=game,status='STAGE_COMPLETE',path=str(prepared['cache']))
    module=importlib.import_module('analyses.'+MODULES[number])
    plot_only=args.render_only or args.stage=='render'
    if plot_only:
        if not (root/'logs/run.json').is_file():raise BlockedInput('No analyzed result to redraw: '+str(root))
        saved=json.loads((root/'logs/run.json').read_text())
        _verify(root,saved,required=('prepare','analyze'))
        identity=saved['identity'];meta=saved.get('input',{})
    else:
        try:meta=_input_identity(number,args,subject,game)
        except FileNotFoundError as error:
            identity=dict(theme=number,subject=subject,game=game)
            with ResultStore(root,identity,resume=True) as store:store.blocked_input(str(error))
            return dict(subject=subject,game=game,status='BLOCKED_INPUT',path=str(root),reason=str(error))
        identity=dict(theme=number,subject=subject,game=game,dataset=meta.get('dataset'),
            source=signature(meta.get('sources') or meta.get('manifest_sha256')))
    settings=clean({k:v for k,v in vars(args).items() if k not in ('subjects','games','workers','resume','stage','render_only','aggregate','outputs_root','input_root')})
    with ResultStore(root,identity,resume=args.resume or plot_only) as store:
        try:
            if not plot_only:
                with store.stage('prepare',signature(dict(input=identity,version='current-input-v2'))):pass
                store.meta['input']=clean(meta)
                if args.stage=='prepare':
                    if number!='01':prepare_native(args,subject,game)
                    store.publish(required_stages=('prepare',))
                    return dict(subject=subject,game=game,status=store.meta['status'],scientific_status=_scientific_status(store.meta),path=str(root))
                qa=None;tables=None
                with store.stage('analyze',signature(dict(code=code_signature(module),
                                 dispatcher=inspect.getsource(analyze),settings=settings,
                                 external=_external_signature(number,args,meta,subject,game)))) as stage:
                    if not stage.reused:
                        prepared=None if number=='01' else prepare_native(args,subject,game)
                        with tempfile.TemporaryDirectory(prefix=f'research-{number}-') as workdir:
                            tables,qa,arrays=analyze(number,module,args,subject,game,prepared,Path(workdir))
                        # Missing inputs must never commit an empty successful analysis stage.
                        if qa.get('status')=='BLOCKED_INPUT':raise BlockedInput(qa.get('reason',str(qa)))
                        write_outputs(stage,tables,arrays)
                if qa is not None:_qa_metadata(store,qa,tables)
                if args.stage=='analyze':
                    store.publish(required_stages=('prepare','analyze'))
                    return dict(subject=subject,game=game,status=store.meta['status'],scientific_status=_scientific_status(store.meta),path=str(root))
            with store.stage('render',_render_signature(number,module)) as stage:
                if not stage.reused:
                    render(number,module,stage.path,read_tables(stage.path),f'{game} · sub{subject:03d}')
            store.publish()
        except FileNotFoundError as error:
            store.blocked_input(str(error))
            return dict(subject=subject,game=game,status='BLOCKED_INPUT',path=str(root),reason=str(error))
    return dict(subject=subject,game=game,status=store.meta['status'],scientific_status=_scientific_status(store.meta),path=str(root))


def _group_scope(number,store):
    if number!='10':return
    qa=dict(store.meta.get('analysis',{}))
    actual=dict(aggregation='Shared scalar switch curves: equal held-session means within participant, then equal participants. Saved curves do not retain run-specific means.',
        representation_aggregation='ACE coordinates are displayed separately for participant and held fold. Candidate profiles stay separate for participant, held fold, and view; neither basis coordinates nor candidate labels are pooled or aligned.',
        plotted_scalar_metrics=['K','U','D','policy_change'],
        figure_summary_tables=['control_switch_scalar_participant','control_switch_scalar_group'])
    if any(qa.get(k)!=v for k,v in actual.items()):
        qa.update(actual);_qa_metadata(store,qa)


def group(number,args):
    from analyses import group_results
    number=f'{int(number):02d}';results=[]
    plot_only=args.render_only or args.stage=='render'
    for game in args.games:
        root=Path(args.outputs_root)/number/'group'/game
        _legacy_guard(root)
        if plot_only:
            if not (root/'logs/run.json').is_file():raise BlockedInput('No analyzed group result to redraw: '+str(root))
            saved=json.loads((root/'logs/run.json').read_text())
            _verify(root,saved,required=('prepare','analyze'));identity=saved['identity']
        else:
            units=[(s,game,Path(args.outputs_root)/number/f'sub{s:03d}'/game) for s in args.subjects]
            missing=[str(r) for _,_,r in units if not (r/'logs/run.json').exists()]
            if missing:raise BlockedInput('Missing participant results: '+', '.join(missing))
            sources={}
            for subject,_,path in units:
                meta=json.loads((path/'logs/run.json').read_text())
                _verify(path,meta,required=('prepare','analyze'))
                # Redrawing or recording a later blocked attempt does not alter numbers.
                sources[str(subject)]=dict(identity=meta['identity'],
                    analyze=meta['stages']['analyze']['signature'],scientific_status=_scientific_status(meta),
                    tables={k:v for record in meta['stages'].values() for k,v in record['artifacts'].items()
                            if k.startswith('tables/')})
            identity=dict(theme=number,game=game,subjects=sorted(args.subjects),sources=sources)
        with ResultStore(root,identity,resume=True) as store:
            if not plot_only:
                with store.stage('prepare',identity):pass
                if args.stage=='prepare':
                    store.publish(required_stages=('prepare',))
                    results.append(dict(game=game,path=str(root),status=store.meta['status'],scientific_status=_scientific_status(store.meta)))
                    continue
                qa=None
                with store.stage('analyze',code_signature(group_results)) as stage:
                    if not stage.reused:
                        tables,qa,arrays=group_results.aggregate(number,args,units)
                        write_outputs(stage,tables,arrays)
                if qa is not None:
                    qa=dict(qa,participant_scientific_status={s:v['scientific_status'] for s,v in sources.items()})
                    statuses=set(qa['participant_scientific_status'].values())
                    if statuses-{'COMPLETED'}:qa['status']='PARTIAL_INPUT_SUPPORT'
                    _qa_metadata(store,qa,tables)
                _group_scope(number,store)
                if args.stage=='analyze':
                    store.publish(required_stages=('prepare','analyze'))
                    results.append(dict(game=game,path=str(root),status=store.meta['status'],scientific_status=_scientific_status(store.meta)))
                    continue
            _group_scope(number,store)
            with store.stage('render',_render_signature(number,group_results)) as stage:
                if not stage.reused:
                    if number in ('11','12','13'):render_neural(stage.path,read_tables(stage.path),f'{game} · equal participant')
                    elif number=='10':
                        from analyses.control_pattern_pipeline import render_group
                        render_group(stage.path,read_tables(stage.path),f'{game} · participant comparison')
                    else:group_results.render(number,stage.path,read_tables(stage.path),f'{game} · equal participant')
            store.publish()
        results.append(dict(game=game,path=str(root),status=store.meta['status'],scientific_status=_scientific_status(store.meta)))
    return results


def _run_unit(task):
    """Let one missing participant/game report independently in a batch."""
    number,args,subject,game=task
    try:return unit(*task)
    except Exception as error:
        return dict(subject=subject,game=game,status='BLOCKED_INPUT' if isinstance(error,FileNotFoundError) else 'FAILED',
                    error_type=type(error).__name__,reason=str(error),
                    path=str(Path(args.outputs_root)/number/f'sub{subject:03d}'/game))


def main(number):
    number=f'{int(number):02d}';p=parser(number);args=p.parse_args()
    if any(s<1 or s>6 for s in args.subjects):p.error('Subjects must be1–6')
    if number!='01' and 'mario' in args.games:p.error('Mario is currently available for model-independent01 only')
    if args.workers<1:p.error('--workers must be positive')
    if number=='00' and (args.render_only or args.stage=='render'):
        p.error('00 has no figures; --render-only / --stage render are unsupported')
    if args.aggregate:
        print(json.dumps(group(number,args),indent=2));return
    tasks=[(number,args,s,g) for s in args.subjects for g in args.games]
    results=[]
    if args.workers==1:
        for task in tasks:
            result=_run_unit(task);results.append(result);print(json.dumps(result),flush=True)
    else:
        # Frozen model packages must be isolated per participant/game.
        import multiprocessing
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures={pool.submit(_run_unit,task):task for task in tasks}
            for future in as_completed(futures):
                result=future.result();results.append(result);print(json.dumps(result),flush=True)
    if any(r['status']=='FAILED' for r in results):raise SystemExit(1)
    if any(r['status']=='BLOCKED_INPUT' for r in results):raise SystemExit(2)
