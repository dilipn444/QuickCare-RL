"""QuickCare: educational synthetic scheduling simulation, not clinical decision support."""
from collections import deque
import numpy as np

SEVERITY = ('critical', 'urgent', 'moderate', 'low')
RATES = np.array([0.10, 0.22, 0.42, 0.62])

class QuickCare:
    """Discrete decision epochs. Arrivals/discharges occur BEFORE each action is selected.

    Each timestep permits at most one service admission, even if several slots are free.
    End-of-horizon waiting cases are counted as backlog, not silently discharged.
    """
    def __init__(self, scenario='normal', horizon=100, slots=3, seed=0):
        if scenario not in {'normal', 'surge', 'severity_surge', 'shortage'}:
            raise ValueError('Unknown scenario')
        if horizon < 3 or slots < 1:
            raise ValueError('Invalid horizon or number of slots')
        self.scenario, self.horizon, self.slots, self.seed = scenario, horizon, slots, seed
        self.action_space_n = 4
        self.reset(seed=seed)

    def reset(self, seed=None, options=None):
        if seed is not None: self.seed = seed
        rng = np.random.default_rng(self.seed)
        rates = np.tile(RATES, (self.horizon, 1))
        start, end = self.horizon // 3, 2 * self.horizon // 3
        if self.scenario == 'surge': rates[start:end] *= 1.8
        if self.scenario == 'severity_surge': rates[start:end, :2] *= 2.5
        self.arrivals = rng.poisson(rates)
        self.durations = {(t,cls): rng.integers(2,6,size=int(self.arrivals[t,cls])).tolist()
                          for t in range(self.horizon) for cls in range(4)}
        self.queues = [deque() for _ in range(4)]
        self.busy = []
        self.t = 0
        self.completed = self.started = self.urgent_late = self.invalid = 0
        self.waits, self.urgent_waits, self.queue_hist, self.rewards = [],[],[],[]
        self._prepare_epoch()
        return self.observe(), {}

    def capacity(self):
        if self.scenario == 'shortage' and self.horizon//3 <= self.t < 2*self.horizon//3:
            return max(1,self.slots-1)
        return self.slots

    def _prepare_epoch(self):
        still=[]
        for finish, cls, arrival in self.busy:
            if finish <= self.t: self.completed += 1
            else: still.append((finish, cls, arrival))
        self.busy = still
        for cls in range(4):
            for duration in self.durations[self.t, cls]:
                self.queues[cls].append((self.t, duration))

    def observe(self):
        counts = np.array([len(q) for q in self.queues],dtype=np.float32)
        oldest = np.array([self.t-q[0][0] if q else 0 for q in self.queues],dtype=np.float32)
        free = max(0,self.capacity()-len(self.busy))
        return np.concatenate([counts/25,oldest/30,[free/self.slots]]).astype(np.float32)

    def valid_actions(self):
        if len(self.busy) >= self.capacity(): return []
        return [i for i,q in enumerate(self.queues) if q]

    def step(self, action):
        if self.t >= self.horizon: raise RuntimeError('Episode has finished; reset first')
        valid = self.valid_actions()
        if action in valid:
            arrival,duration=self.queues[action].popleft()
            wait=self.t-arrival
            self.waits.append(wait)
            if action < 2:
                self.urgent_waits.append(wait)
                self.urgent_late += int(wait > 5)
            self.started += 1
            self.busy.append((self.t+duration,action,arrival))
        elif valid:
            self.invalid += 1
        lengths=np.array([len(q) for q in self.queues])
        overdue=sum(sum(self.t-p[0] > 5 for p in self.queues[i]) for i in (0,1))
        reward=-float(np.dot(lengths,[5,3,1,0.5]))/10-0.5*overdue+(0.2 if action in valid else 0)
        self.queue_hist.append(int(sum(lengths)))
        self.rewards.append(reward)
        self.t+=1
        done=self.t==self.horizon
        if not done: self._prepare_epoch()
        return self.observe(),reward,done,False,(self.metrics() if done else {})

    def metrics(self):
        return {'mean_wait_served':float(np.mean(self.waits)) if self.waits else 0.,
                'mean_urgent_wait_served':float(np.mean(self.urgent_waits)) if self.urgent_waits else 0.,
                'urgent_delay_over_5_served':self.urgent_late,
                'mean_queue':float(np.mean(self.queue_hist)) if self.queue_hist else 0.,
                'max_queue':max(self.queue_hist,default=0),
                'started':self.started,'completed':self.completed,
                'unfinished_waiting':sum(map(len,self.queues)),
                'unfinished_in_service':len(self.busy),
                'reward':float(sum(self.rewards)),'invalid_actions':self.invalid}

def fcfs(env):
    valid=env.valid_actions()
    return min(valid,key=lambda i:(env.queues[i][0][0],i)) if valid else 0

def priority(env):
    valid=env.valid_actions()
    return min(valid) if valid else 0

def run_policy(policy, scenario='normal',seed=2026,horizon=100):
    env=QuickCare(scenario=scenario,seed=seed,horizon=horizon)
    obs,_=env.reset();done=False
    while not done:
        action=policy(env,obs)
        obs,_,done,_,_=env.step(action)
    return env.metrics(),env.queue_hist
