"""Frozen forward sensitivity to clock channels; no policy-utility claim."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    report={}
    for game in ['enduro','pong']:
        folder=a.root/'v2'/game
        binding=json.loads((folder/'collection.json').read_text())['checkpoint']
        ckpath=Path(binding['files']['actor']['path'])
        assert hashlib.sha256(ckpath.read_bytes()).hexdigest()==binding['files']['actor']['sha256']
        ck=torch.load(ckpath,map_location='cpu',weights_only=False)
        assert ck['flags']['last_layer_n']==0
        weights=ck['actor_net_state_dict']
        cell_weights={key.removeprefix('tree_rep_encoder.cells.0.'):v for key,v in weights.items() if key.startswith('tree_rep_encoder.cells.0.')}
        input_n=cell_weights['weight_ih'].shape[1];hidden=cell_weights['weight_hh'].shape[1]
        cell=torch.nn.GRUCell(input_n,hidden);cell.load_state_dict(cell_weights);cell.requires_grad_(False)
        phase=weights['phase_embedding.weight'][0].numpy()
        qw=(weights['voc_head.weight'][0]-weights['voc_head.weight'][1]).numpy()
        qb=float(weights['voc_head.bias'][0]-weights['voc_head.bias'][1])
        game_report={}
        for mode in ['policy','forced']:
            chunks=[np.load(folder/f'{mode}_{s}.npz') for s in [19201,19241]]
            data={key:np.concatenate([x[key] for x in chunks]) for key in chunks[0].files}
            features=data['features'];h=data['gru'];tokens=data['token']
            # The final vector adds a phase embedding after concatenation.
            scores=[np.max(np.abs(features[:100,s:s+hidden]-phase[s:s+hidden]-h[:100])) for s in range(features.shape[1]-hidden+1)]
            offset=int(np.argmin(scores));assert scores[offset]<2e-6
            head_error=float(np.max(np.abs(features@qw+qb-data['delta'])))
            assert head_error<2e-6
            prev=np.zeros_like(h);last={}
            for i,(stream,decision,k) in enumerate(zip(data['stream'],data['decision'],data['count'])):
                key=(int(stream),int(decision))
                if k:
                    j=last[key];assert data['count'][j]==k-1
                    prev[i]=h[j]
                last[key]=i
            local=torch.from_numpy(qw[offset:offset+hidden])
            changes={key:[] for key in ['count','depth']}
            gradients={key:[] for key in changes}
            reconstruction_error=0.
            for start in range(0,len(h),512):
                stop=min(start+512,len(h));ids=slice(start,stop)
                x=torch.from_numpy(tokens[ids]).requires_grad_(True)
                hp=torch.from_numpy(prev[ids])
                decoded=cell(x,hp)
                reconstruction_error=max(reconstruction_error,float((decoded.detach()-torch.from_numpy(h[ids])).abs().max()))
                sensitivity=torch.autograd.grad((decoded@local).sum(),x)[0]
                for key,column,values in [('count',input_n-3,data['count']),('depth',input_n-2,data['depth'])]:
                    # Positions are the frozen schema's final five metadata
                    # fields: reset, depth_discount, log_count, log_depth, start.
                    assert np.allclose(tokens[ids,column],np.log1p(values[ids]),atol=1e-6)
                    changed=x.detach().clone();changed[:,column]=torch.from_numpy(np.log1p(values[ids]+1).astype('float32'))
                    with torch.no_grad(): shift=(cell(changed,hp)-decoded.detach())@local
                    changes[key].append(shift.numpy())
                    gradients[key].append(sensitivity[:,column].numpy())
            assert reconstruction_error<5e-6
            result={}
            arrays={}
            for key in changes:
                change=np.concatenate(changes[key]);derivative=np.concatenate(gradients[key])
                arrays[key+'_delta_change']=change
                mask=data['count']>0
                result[key]=dict(median_abs_one_step_q_change=float(np.median(np.abs(change))),
                    p95_abs_one_step_q_change=float(np.quantile(np.abs(change),.95)),
                    fraction_abs_change_above_1e_6=float(np.mean(np.abs(change)>1e-6)),
                    nonroot_median_abs_change=float(np.median(np.abs(change[mask]))),
                    median_abs_derivative_per_log_unit=float(np.median(np.abs(derivative))))
            game_report[mode]=dict(rows=len(h),gru_feature_offset=offset,gru_reconstruction_max_error=reconstruction_error,
                q_reconstruction_max_error=head_error,online_delta_sd=float(data['delta'].std()),sensitivity=result)
            np.savez_compressed(a.root/f'sensitivity_{game}_{mode}.npz',**arrays,count=data['count'],depth=data['depth'],stream=data['stream'])
        report[game]=game_report
    final=dict(status='COMPLETE',games=report,scope='Local perturbation of only the log-count or log-depth channel, with other inputs/history fixed. Such inputs need not be realizable search states. Nonzero sensitivity establishes forward dependence, not calibrated VOC or beneficial planning.')
    (a.root/'counter_sensitivity.json').write_text(json.dumps(final,indent=2))
    print(json.dumps(final,indent=2))


if __name__=='__main__':main()
