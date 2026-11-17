from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'diagnostics'))
from finalize_normalized_planning_extension import check_binding,check_registry,check_scope


def test_allocation_identity_audit_rejects_replacement_and_missing_row():
    rows=[dict(seed=s,stream=i,decision=d) for s in (8301,8401) for i in range(8) for d in range(128,1664)]
    check_scope(rows)
    with pytest.raises(ValueError,match='identities'):check_scope(rows[:-1])
    with pytest.raises(ValueError,match='identities'):check_scope([rows[1]]+rows[1:])


def test_pair_scope_preserves_both_repetitions_and_all_counts():
    rows=[dict(seed=s,stream=i,count=c,repetition=r) for s in (8301,8401) for i in range(8) for c in (1,3,6,12) for r in range(2)]
    check_scope(rows,paired=True)
    wrong=deepcopy(rows);wrong[-1]['repetition']=0
    with pytest.raises(ValueError,match='identities'):check_scope(wrong,paired=True)


def test_checkpoint_from_wrong_endpoint_or_different_weights_is_rejected():
    binding={k:dict(path='/fixed/'+k,sha256=k,real_step=1000016) for k in ('actor','model')}
    summary=dict(status='COMPLETE',evaluation_epsilon=0.,checkpoint=binding)
    check_binding(summary,1000000,deepcopy(binding))
    wrong=deepcopy(binding);wrong['actor']['sha256']='other'
    with pytest.raises(ValueError,match='audited checkpoint'):check_binding(summary,1000000,wrong)
    with pytest.raises(ValueError,match='milestone'):check_binding(summary,500000)
    with pytest.raises(ValueError,match='trainingepsilon'):check_binding(dict(summary,evaluation_epsilon=.1),1000000)


def test_registry_requires_both_conditions_all_milestones_and_utilities():
    conditions=['burn8ln','burn16ln']
    manifest=dict(runs=[dict(condition=c) for c in conditions],milestones=[500000,1000000],utility_endpoint=1000000)
    allocations=dict(jobs=[dict(condition=c,milestone=m,job_id=i*2+j+1) for i,c in enumerate(conditions) for j,m in enumerate((500000,1000000))])
    utilities={c:dict(condition=c,milestone=1000000,pairs=128,job_id=i+5) for i,c in enumerate(conditions)}
    assert check_registry(manifest,allocations,utilities)==list(range(1,7))
    with pytest.raises(ValueError,match='allocation registry'):
        check_registry(manifest,dict(jobs=allocations['jobs'][:-1]),utilities)
    with pytest.raises(ValueError,match='utility registry'):
        check_registry(manifest,allocations,{'burn8ln':utilities['burn8ln']})
