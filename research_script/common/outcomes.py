"""Human transition risk sets selected using only the preceding action.

Active-to-NOOP includes active-to-other-active as a non-termination. Selecting
only same-action stays would condition eligibility on the future outcome.
"""
def risk_set(frame,population):
    n=frame.n_actions if 'n_actions' in frame else 6
    valid=(frame.incumbent_action.ge(0)&frame.incumbent_action.lt(n)&
           frame.human_action.ge(0)&frame.human_action.lt(n)&frame.scored.eq(1))
    if population=='overall':keep=valid;y=frame.human_action.ne(frame.incumbent_action)
    elif population=='noop_to_active':keep=valid&frame.incumbent_action.eq(0);y=frame.human_action.ne(0)
    elif population=='active_to_noop':keep=valid&frame.incumbent_action.ne(0);y=frame.human_action.eq(0)
    else:raise ValueError(population)
    result=frame.loc[keep].copy();result['outcome']=y.loc[keep].astype(int)
    return result
