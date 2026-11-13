"""Show held-stream decoding error without conflating rounding with information loss."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    fig,axes=plt.subplots(2,2,figsize=(10,6),sharex=True,constrained_layout=True)
    for row,game in enumerate(['enduro','pong']):
        for mode,label,color in [('policy','Policy-visited states','#1864ab'),
                                 ('forced','Forced search budgets','#d9480f')]:
            data=np.load(args.root/'v2'/game/f'{mode}_decoding.npz')
            for column,target in enumerate(['Search count','Rollout depth']):
                truth=data['target'][:,column]
                error=np.abs(data['gru'][:,column]-truth)
                counts=np.unique(truth)
                means=[];sems=[]
                for k in counts:
                    mask=truth==k
                    # Equal stream weight within each count. Shading describes
                    # stream variability of held-out errors, not training-seed CI.
                    stream_means=[error[mask & (data['stream']==g)].mean()
                                  for g in np.unique(data['stream'][mask])]
                    means.append(np.mean(stream_means))
                    sems.append(np.std(stream_means,ddof=1)/np.sqrt(len(stream_means))
                                if len(stream_means)>1 else np.nan)
                means=np.asarray(means);sems=np.asarray(sems)
                axes[row,column].plot(counts,means,label=label,color=color,marker='.',markersize=4)
                axes[row,column].fill_between(counts,np.maximum(0,means-sems),means+sems,color=color,alpha=.15)
                axes[row,column].set(title=f'{game.title()}: {target.lower()}',
                    xlabel=f'True {target.lower()}',ylabel='Mean absolute error (steps)')
                axes[row,column].spines[['top','right']].set_visible(False)
                axes[row,column].grid(alpha=.15)
    axes[0,0].legend(frameon=False,fontsize=9)
    fig.suptitle('Information retained in the frozen GRU\nSeparately fitted linear probes; held-out environment streams',fontsize=12)
    fig.savefig(args.root/'counter_decoding_error.png',dpi=180)
    fig.savefig(args.root/'counter_decoding_error.svg')
    print(args.root/'counter_decoding_error.png')


if __name__=='__main__':main()
