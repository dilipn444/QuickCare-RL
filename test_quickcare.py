import numpy as np
from quickcare import QuickCare, fcfs, priority, run_policy

def test_reproducible_exogenous_traffic():
    a,b=QuickCare(seed=88),QuickCare(seed=88)
    assert np.array_equal(a.arrivals,b.arrivals) and a.durations==b.durations

def test_decision_matches_observation():
    e=QuickCare(seed=13)
    obs,_=e.reset()
    assert np.array_equal(obs[:4]*25,[len(q) for q in e.queues])
    assert set(e.valid_actions()).issubset(set(range(4)))

def test_full_runs_all_scenarios():
    for scenario in ('normal','surge','severity_surge','shortage'):
        for policy in (fcfs,priority):
            metrics,hist=run_policy(lambda env,obs:policy(env),scenario,91)
            assert len(hist)==100 and metrics['started']>0
            assert metrics['unfinished_waiting'] >= 0 and metrics['invalid_actions']==0
            assert metrics['started']>=metrics['completed']

def test_urgent_priority_when_available():
    e=QuickCare(seed=8)
    for i in range(3):e.queues[i].append((0,3))
    assert priority(e)==0

def test_no_scheduling_when_resource_full():
    e=QuickCare(seed=4)
    e.busy=[(2,0,0)]*e.slots
    assert e.valid_actions()==[]

def test_seed_equal_across_policies():
    a,b=QuickCare(seed=7,scenario='surge'),QuickCare(seed=7,scenario='surge')
    assert np.array_equal(a.arrivals,b.arrivals)
