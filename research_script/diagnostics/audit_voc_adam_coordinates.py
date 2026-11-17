"""Read-only decomposition of raw-head Adam updates into common/contrast terms.

This is an algebraic attribution of the stored moment update, not a causal
claim that optimizer coordinates caused an observed gate distribution.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch


def decompose(m, v, step, group):
    if group.get('amsgrad',False) or group.get('weight_decay',0) or group.get('maximize',False):
        raise ValueError('Only ordinary unregularized Adam is supported')
    m,v=m.double(),v.double()
    if m.shape[0]!=2 or v.shape!=m.shape:
        raise ValueError('Expected two raw-head rows')
    b1,b2=group['betas']
    m=m/(1-b1**step)
    inv=1/(torch.sqrt(v/(1-b2**step))+group['eps'])
    common=(m[0]+m[1])/2
    contrast=(m[0]-m[1])/2
    a=-group['lr']*(inv[0]-inv[1])*common
    b=-group['lr']*(inv[0]+inv[1])*contrast
    direct=-group['lr']*(m[0]*inv[0]-m[1]*inv[1])
    torch.testing.assert_close(a+b,direct,rtol=1e-9,atol=1e-13)
    return a.flatten().numpy(),b.flatten().numpy(),direct.flatten().numpy()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--campaign',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    torch.set_num_threads(1)
    rows=[]
    # A known Adam example: common and contrast derivatives both exist, but
    # equal-sign raw gradients take equal first Adam steps, cancelling D.
    g=torch.tensor([[.6],[.4]],dtype=torch.float64)
    a,b,d=decompose(.1*g,.001*g*g,1,dict(betas=(.9,.999),eps=1e-8,lr=.001))
    assert abs(d[0])<1e-10 and a[0]*b[0]<0
    c=args.campaign.resolve()
    for item in json.loads((c/'manifest.json').read_text())['runs']:
        if item['condition']=='smoke':continue
        run=c.parents[3]/'logs/thinker'/item['xpid']
        paths=list(run.glob('ckp_actor.tar_step_*'))
        selected=[p for p in paths if 50000<=int(p.name.rsplit('_',1)[1])<52000]
        selected += [p for p in paths if int(p.name.rsplit('_',1)[1])>=250000]
        for path in sorted(selected):
            raw=path.read_bytes();ck=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
            if ck['voc_gate_policy_schema_version']!=4:
                raise ValueError('This audit expects ordinary raw-coordinate Adam (schema 4)')
            opt=ck['voc_optimizer_state_dict']
            aa,bb,dd=[],[],[]
            for group in opt['param_groups']:
                for pid in group['params']:
                    state=opt['state'][pid]
                    a,b,d=decompose(state['exp_avg'],state['exp_avg_sq'],int(state['step']),group)
                    aa.extend(a);bb.extend(b);dd.extend(d)
            a,b,d=np.asarray(aa),np.asarray(bb),np.asarray(dd)
            na,nb,nd=np.linalg.norm(a),np.linalg.norm(b),np.linalg.norm(d)
            rows.append(dict(condition=item['condition'],step=int(ck['real_step']),path=str(path),
                sha256=hashlib.sha256(raw).hexdigest(),
                common_moment_to_contrast_update_norm=float(na),
                contrast_moment_to_contrast_update_norm=float(nb),
                net_contrast_update_norm=float(nd),
                common_over_contrast_component=float(na/nb) if nb else None,
                component_cosine=float(a@b/(na*nb)) if na*nb else None,
                net_over_sum_component_norm=float(nd/(na+nb)) if na+nb else None))
    result=dict(rows=rows,known_example='PASS; first-step raw Adam can cancel a nonzero contrast gradient',
        interpretation='Coordinate-dependent moment mixing is measurable; its causal effect on useful planning is untested. These are parameter-update components, not separate reward signals or GRU update attributions.',
        production_changes='NONE',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    for r in rows:
        print(r['condition'],r['step'],'common/contrast',r['common_over_contrast_component'],
              'component cosine',r['component_cosine'],'net/sum',r['net_over_sum_component_norm'])


if __name__=='__main__':
    main()
