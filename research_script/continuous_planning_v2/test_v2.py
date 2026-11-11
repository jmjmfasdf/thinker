"""Fast unit tests without writing smoke result folders or touching V1."""
import ast
import copy
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from core import bridge, DEFAULT_V1, PRIMARY
from derive import event_metrics
from fitting import SoftReadout, alternative_design
from panels import bout_table, km_curves


class Contracts(unittest.TestCase):
    def synthetic(self,budget,forced=False,reset=False):
        d=pd.DataFrame([dict(decision_id=0,root_value=0.)])
        rows=[]
        for j in range(budget+int(not forced)):
            stop=j==budget
            before=min(j,2) if reset else j
            after=before+int(not stop)
            rows.append(dict(decision_id=0,event=j,event_index=j,root_margin=1.,
                computation=not stop,control=2 if stop else 1 if reset and j==1 else 0,
                actor_reset=bool(reset and j==1),accepted_action=0,depth_pre=0 if reset and j==2 else j if not reset or j<2 else j-2,
                depth_post=0 if reset and j==1 else (j if stop else j+1) if not reset or j<2 else j-1,
                root_n_before_0=float(j),root_n_after_0=float(j+int(not stop)),
                root_q_before_0=1. if j else np.nan,root_q_after_0=1. if j or not stop else np.nan,
                rollout_return=1.,forced_terminal=forced and j==budget-1,stop_choice=int(stop)))
        return pd.DataFrame(rows),d

    def test_budgets_stop_and_prior_information(self):
        for B in (0,1,2,29,30):
            for forced in (False,True) if B else (False,):
                raw,d=self.synthetic(B,forced)
                e,extra=event_metrics(raw,d,1)
                self.assertEqual(int(e.computation.sum()),B)
                self.assertEqual(int(e.stop_choice.sum()),int(not forced))
                self.assertTrue(np.isnan(e.iloc[0].prior_rollout_advantage))
                self.assertEqual(int(extra.process_valid_events.iloc[0]),max(0,B-1))
                if B:self.assertEqual(extra.process_first_events.iloc[0],1)
                else:self.assertTrue(np.isnan(extra.process_abs_total.iloc[0]))
                if B>=2:
                    split=B//2
                    self.assertEqual(split,14 if B==29 else B//2)
                    self.assertEqual(B-split,15 if B==29 else B-B//2)

    def test_reset_path(self):
        raw,d=self.synthetic(3,reset=True)
        raw.loc[1,'depth_post']=2 # RESET can record pre-reset forecast depth.
        e,_=event_metrics(raw,d,1)
        self.assertEqual(int((e.control==1).sum()),1)

    def test_no_cross_decision_evidence(self):
        a,d=self.synthetic(2)
        b=a.copy();b.decision_id=1
        d=pd.concat([d,d.assign(decision_id=1)])
        e,_=event_metrics(pd.concat([a,b]),d,1)
        self.assertTrue(e.groupby('decision_id').head(1).prior_first_gain.isna().all())

    def test_no_cross_episode_bouts(self):
        d=pd.DataFrame(dict(episode=['a']*3+['b']*3,run=['run']*6,time=[0,1,2,0,1,2],human_action=[0,0,1,1,1,1]))
        b=bout_table(d)
        self.assertEqual(b.left_unknown.sum(),2)
        self.assertEqual(b.event.sum(),1)
        self.assertTrue((b.duration>=0).all())

    def test_km_float_ties_and_censoring(self):
        b=pd.DataFrame(dict(run=['r']*4,duration=[.1,.1+1e-13,.3,.5],event=[1,1,0,1]))
        curve=km_curves(b).set_index('x')
        self.assertAlmostEqual(curve.loc[.125,'y'],.5)
        self.assertAlmostEqual(curve.loc[.25,'y'],.5)
        self.assertAlmostEqual(curve.loc[.5,'y'],0.)
        self.assertEqual(curve.loc[.125,'n'],2)

    def test_actor_soft_targets_and_leakage(self):
        rng=np.random.default_rng(13);x=rng.normal(size=(160,3));alt=rng.normal(size=(160,3,2))
        logits=.5*x[:,0,None]*np.array([1,0,-1])+alt[:,:,0]
        p=np.exp(logits-logits.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
        cfg={'max_iter':300,'tolerance':1e-7}
        model=SoftReadout().fit(x,alt,p,np.ones(160)/160,10,cfg)
        pred=model.predict(x,alt)
        self.assertLess(float(-np.mean(np.sum(p*np.log(pred),axis=1))),float(-np.mean(np.sum(p*np.log(p.mean(0)),axis=1))))
        source=Path(__file__).with_name('fitting.py').read_text()
        self.assertIn("drop(columns=['human_action', 'switch'])",source)
        self.assertLess(source.index('pred.to_csv(path, index=False)'),source.index("human = labels.human_action"))

    def test_missing_is_not_zero(self):
        raw=np.array([[[0.,np.nan],[1.,2.]],[[1.,3.],[2.,4.]],[[100.,1e6],[200.,2e6]]])
        train,test,names=alternative_design(raw,np.array([0,1]),np.array([2]),['a','b'])
        self.assertIn('b unavailable',names)
        self.assertTrue(np.isfinite(train).all())
        self.assertGreater(abs(test).max(),100)

    def test_frozen_cv_and_control_guards(self):
        C,M,A=bridge(DEFAULT_V1)
        d=pd.DataFrame(dict(session=[1]*6,checkpoint=['same']*6,run=['a']*3+['b']*3,episode=['e1','e1','e2','e3','e3','e3']))
        for _,tr,te in M.outer_folds(d):self.assertFalse(set(d.loc[tr,'run'])&set(d.loc[te,'run']))
        with self.assertRaises(ValueError):M.check_control_features(M.Spec('bad',('budget',)),'stop')
        _,_,p=C.exact_max_t(np.ones((6,4))*np.array([1,-1,0,2]))
        self.assertTrue(np.isfinite(p).all())
        _,_,p=C.exact_max_t(np.full((6,4),np.nan));self.assertTrue(np.isnan(p).all())

    def test_render_only_does_not_import_fitting(self):
        tree=ast.parse(Path(__file__).with_name('render.py').read_text())
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertNotIn('fitting',imports);self.assertNotIn('derive',imports)

    def test_review_support_and_dual_labels(self):
        from render import support_label,dual,probability_axis
        import matplotlib.pyplot as plt
        q=pd.DataFrame(dict(unit=['r']*5,x=list('abcde'),y=[.2]*5,n=[100]*5,category_count=[20]*5))
        self.assertIn('100 decisions',support_label(q,{'kind':'bar'}))
        self.assertFalse(probability_axis('Terminal incumbent margin (log probability)'))
        self.assertFalse(probability_axis('Human exact-action log probability (nats)'))
        self.assertTrue(probability_axis('STOP hazard'))
        fig,ax=plt.subplots()
        q=pd.DataFrame([dict(unit='r',x=x,y=.5,n=30,series=axis,axis=axis) for axis in ('margin','uncertainty') for x in (0.,1.)])
        dual(ax,q,dict(title='Single title',ylabel='P(B > 0)',kind='dual'))
        self.assertEqual(ax.get_title(loc='left'),'Single title')
        self.assertEqual(ax.get_title(loc='center'),'')
        self.assertNotEqual(fig.axes[0].lines[-1].get_color(),fig.axes[1].lines[-1].get_color())
        plt.close(fig)

    def test_no_reduced_subject_interval(self):
        from render import summary
        table=pd.DataFrame(dict(unit=['s1','s2','s3','s4','s5'],x=[0]*5,y=[1,2,3,4,5],n=[25]*5))
        s=summary(table,['x'],group=True)
        self.assertTrue(s.lower.isna().all())
        table=pd.concat([table,pd.DataFrame(dict(unit=['s6'],x=[0],y=[6],n=[25]))])
        self.assertTrue(summary(table,['x'],group=True).lower.notna().all())

    def test_group_families_and_missing(self):
        from group_report import infer,FAMILIES
        C,_,_=bridge(DEFAULT_V1)
        rows=[dict(subject=s,game=g,contrast=c,delta_ll=.01*s) for s in range(1,7) for g in (0,1) for c in FAMILIES['actor']]
        out=infer(C,pd.DataFrame(rows),'actor')
        self.assertEqual(len(out),12)
        rows[0]['delta_ll']=np.nan
        out=infer(C,pd.DataFrame(rows),'actor')
        self.assertEqual(int(out.delta_ll.isna().sum()),1)

    def test_render_primitives(self):
        import tempfile
        import matplotlib.pyplot as plt
        from render import curve,paired,effect,tile,dual,style
        style()
        q=pd.DataFrame([dict(unit=f'r{u}',run=f'r{u}',x=float(x),y=.1*u+.1*x,n=30,series='Series',_bin=x) for u in range(3) for x in range(4)])
        meta=dict(kind='line',ylabel='ΔLL',title='Test')
        with tempfile.TemporaryDirectory() as tmp:
            for fun in (curve,paired,effect):
                fig,ax=plt.subplots();fun(ax,q,meta);fig.savefig(Path(tmp)/(fun.__name__+'.png'));plt.close(fig)
            fig,ax=plt.subplots();z=q.assign(row=q.unit)
            tile(ax,z,dict(kind='tile',color_label='ΔLL'));fig.savefig(Path(tmp)/'tile.png');plt.close(fig)


if __name__=='__main__':unittest.main(verbosity=2)
