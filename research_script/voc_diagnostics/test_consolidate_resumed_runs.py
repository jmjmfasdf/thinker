import csv
import json
from pathlib import Path
import tempfile
import unittest

from consolidate_resumed_runs import combine_collector, merge_csv, merge_jsonl, archive_tree


class ConsolidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_rollback_boundary_and_committed_targets(self):
        fields = ['_tick','_time','real_step','loss','training/bc_targets_attempted',
                  'training/bc_targets_committed','training/actor_updates_committed']
        old = [[0,1,10,'a',64,64,1],[1,2,20,'b',128,128,2],
               [2,3,30,'abandoned',192,192,3]]
        new = [[0,10,25,'c',64,64,1],[1,11,25,'d',128,128,2],
               [2,12,40,'e',192,192,3]]
        for name,rows in [('old.csv',old),('new.csv',new)]:
            with (self.root/name).open('w',newline='') as f:
                w=csv.writer(f);w.writerow(fields);w.writerows(rows)
        prov={'resumed_actor_step':20,'base_training_counters':{
            'bc_targets_attempted':128,'bc_targets_committed':128,'actor_updates_committed':2}}
        stats=merge_csv(self.root/'old.csv',self.root/'new.csv',self.root/'merged.csv',20,prov,{},True)
        with (self.root/'merged.csv').open() as f:rows=list(csv.DictReader(f))
        self.assertEqual([r['loss'] for r in rows],['a','b','c','d','e'])
        self.assertEqual([int(r['real_step']) for r in rows],[10,20,25,25,40])
        self.assertEqual([int(r['training/bc_targets_committed']) for r in rows],[64,128,192,256,320])
        self.assertEqual([int(r['_tick']) for r in rows],list(range(5)))
        self.assertEqual(stats['superseded_rows'],1)

    def test_episode_duplicates_are_real_parallel_episodes(self):
        for name,rows in [('old.jsonl',[{'attempt':0,'real_step':16,'return_raw':-21.0}]*2),
                          ('new.jsonl',[{'attempt':0,'real_step':16,'return_raw':-21.0}]*2)]:
            (self.root/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
        report=merge_jsonl(self.root/'old.jsonl',self.root/'new.jsonl',self.root/'merged.jsonl',32,16)
        rows=[json.loads(s) for s in (self.root/'merged.jsonl').read_text().splitlines()]
        self.assertEqual(report['rows'],4)
        self.assertEqual([r['real_step'] for r in rows],[16,16,48,48])
        self.assertEqual([r['attempt'] for r in rows],[0,0,1,1])

    def test_controls_keep_segment_identity_and_physical_offset(self):
        row={'attempt':0,'real_step':0,'decision':0,'stream':0,'count':0}
        for name in ['old.jsonl','new.jsonl']:(self.root/name).write_text(json.dumps(row)+'\n')
        merge_jsonl(self.root/'old.jsonl',self.root/'new.jsonl',self.root/'merged.jsonl',80,16,True)
        rows=[json.loads(s) for s in (self.root/'merged.jsonl').read_text().splitlines()]
        self.assertEqual(rows[1]['real_step'],80)
        self.assertEqual(rows[1]['decision'],5)
        self.assertEqual(rows[1]['segment_decision'],0)

    def test_ratios_use_counts_and_missing_denominators_are_not_invented(self):
        a={'physical_env_steps':100,'raw_reward_sum':20,'imagined_transitions':100,
           'max_completed_episode_return':30,'learned_stop_fraction':.2,'reset_fraction':.1}
        b={'physical_env_steps':300,'raw_reward_sum':60,'imagined_transitions':900,
           'max_completed_episode_return':10,'learned_stop_fraction':.8,'reset_fraction':.4}
        for i in range(31):a[f'count_{i}_decisions']=0;b[f'count_{i}_decisions']=0
        a['count_0_decisions']=100;b['count_1_decisions']=300
        merged=combine_collector(a,b)
        self.assertEqual(merged['computations_per_action'],2.5)
        self.assertEqual(merged['zero_search_fraction'],.25)
        self.assertEqual(merged['raw_reward_per_1000_actions'],200)
        self.assertEqual(merged['max_completed_episode_return'],30)
        self.assertIsNone(merged['learned_stop_fraction'])

    def test_archive_records_links_without_recursing(self):
        source=self.root/'source';source.mkdir();(source/'raw').write_bytes(b'original')
        (source/'previous_attempt').symlink_to(source,target_is_directory=True)
        entries=archive_tree(source,self.root/'archive')
        self.assertEqual((self.root/'archive/raw').read_bytes(),b'original')
        self.assertFalse((self.root/'archive/previous_attempt').exists())
        self.assertEqual(sum(r['kind']=='symlink' for r in entries),1)


if __name__=='__main__':unittest.main()
