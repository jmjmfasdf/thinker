"""Exploratory nonlinear decodability probe; outer test streams never tune epochs."""
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import GroupKFold


def fit_predict(x, y, groups, outer_fold_ids, logger=None):
    all_pred=np.full_like(y,np.nan,dtype=float)
    folds=np.full(len(y),-1,dtype=int); choices=[]
    # Reuse saved assignments: library versions may break ties differently
    # when many environment streams contain exactly the same number of rows.
    outer_fold_ids=np.asarray(outer_fold_ids)
    assert outer_fold_ids.shape == (len(y),)
    assert set(np.unique(outer_fold_ids)) == {0,1,2,3}
    assert all(len(np.unique(outer_fold_ids[groups==g]))==1 for g in np.unique(groups))
    for fold in range(4):
        train=np.flatnonzero(outer_fold_ids != fold)
        test=np.flatnonzero(outer_fold_ids == fold)
        assert not set(groups[train]) & set(groups[test])
        inner_train,inner_val=next(GroupKFold(3).split(x[train],y[train],groups[train]))
        fit_rows,val_rows=train[inner_train],train[inner_val]
        def fit(rows, validation=None, epochs=60):
            torch.manual_seed(811+fold)
            mean=x[rows].mean(0);scale=x[rows].std(0);scale[scale<1e-6]=1.
            xm=torch.from_numpy(((x-mean)/scale).astype('float32'))
            yt=torch.from_numpy(y.astype('float32'))
            model=torch.nn.Sequential(torch.nn.Linear(x.shape[1],64),torch.nn.GELU(),torch.nn.Linear(64,2))
            opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            generator=np.random.default_rng(193+fold)
            best=float('inf');best_epoch=1;bad=0
            for epoch in range(1,epochs+1):
                model.train();losses=[]
                for ids in np.array_split(generator.permutation(rows),max(1,int(np.ceil(len(rows)/512)))):
                    pred=model(xm[ids]);loss=(pred-yt[ids]).square().mean()
                    opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
                if validation is not None:
                    model.eval()
                    with torch.no_grad(): score=float((model(xm[validation])-yt[validation]).square().mean())
                    if logger:logger(dict(fold=fold,epoch=epoch,train_mse=float(np.mean(losses)),validation_mse=score))
                    if score<best-1e-6:best=score;best_epoch=epoch;bad=0
                    else:bad+=1
                    if bad>=10:break
            model.eval()
            with torch.no_grad():prediction=model(xm[test]).numpy()
            return prediction,best_epoch
        _,epoch=fit(fit_rows,validation=val_rows)
        prediction,_=fit(train,epochs=epoch)
        all_pred[test]=prediction;folds[test]=fold;choices.append(epoch)
    assert np.isfinite(all_pred).all()
    assert all(len(np.unique(folds[groups==g]))==1 for g in np.unique(groups))
    return all_pred,folds,choices


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--game',choices=['enduro','pong'],required=True)
    a=p.parse_args();torch.set_num_threads(1)
    source=a.root/'v2'/a.game;out=a.root/'nonlinear'/a.game;out.mkdir(parents=True,exist_ok=True)
    import wandb
    run=wandb.init(project='thinker',entity=os.getenv('WANDB_USER','jmme425'),mode='online',
        group='voc-gru-information-20260925',name=f'gru-nonlinear-{a.game}',
        job_type='exploratory-frozen-counter-probe',dir=str(out),config=dict(
            hidden=64,activation='GELU',lr=.001,weight_decay=.0001,max_epochs=60,
            outer_folds=4,inner_split='first GroupKFold(3)',scope='decodability only; post-linear exploratory analysis'))
    report=dict(game=a.game,wandb_url=run.url,scope='exploratory nonlinear counter readout; no utility or end-to-end training claim',modes={},
        source_file=dict(path=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    try:
        for mode in ['policy','forced']:
            chunks=[np.load(source/f'{mode}_{s}.npz') for s in [19201,19241]]
            x=np.concatenate([v['gru'] for v in chunks]);groups=np.concatenate([v['stream'] for v in chunks])
            target=np.c_[np.concatenate([v['count'] for v in chunks]),np.concatenate([v['depth'] for v in chunks])]
            reference_path=source/f'{mode}_decoding.npz'
            reference=np.load(reference_path)
            assert np.array_equal(reference['target'],target)
            assert np.array_equal(reference['stream'],groups)
            logpred,fold,epochs=fit_predict(x,np.log1p(target),groups,reference['fold'],
                logger=lambda row:run.log({f'{mode}/{key}':value for key,value in row.items()}))
            pred=np.maximum(0,np.expm1(np.clip(logpred,-10,10)))
            metrics={}
            for j,name in enumerate(['count','depth']):
                err=pred[:,j]-target[:,j]
                metrics[name]=dict(mae=float(np.abs(err).mean()),rounded_accuracy=float(np.mean(np.rint(pred[:,j])==target[:,j])))
                run.log({f'{mode}/test/{name}/{k}':v for k,v in metrics[name].items()})
            report['modes'][mode]=dict(rows=len(x),streams=len(np.unique(groups)),epochs=epochs,metrics=metrics,
                outer_fold_source=dict(path=str(reference_path.resolve()),sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest()))
            np.savez_compressed(out/f'{mode}_predictions.npz',prediction=pred,target=target,fold=fold,stream=groups)
            print(json.dumps(dict(mode=mode,**report['modes'][mode])),flush=True)
        report['status']='COMPLETE'
        (out/'report.json').write_text(json.dumps(report,indent=2))
        run.summary['status']='COMPLETE'
    finally:run.finish()


if __name__=='__main__':main()
