"""Baseline comparisons + lightweight tabular Q-learning and DQN training."""
import csv, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from quickcare import QuickCare, fcfs, priority, run_policy

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
random.seed(7);np.random.seed(7);torch.manual_seed(7)
torch.set_num_threads(1)

def state_bin(obs):
    # 4 capped queue sizes, 4 capped oldest waits, 1 free-resource count.
    q = tuple(int(v) for v in np.minimum((obs[:4]*25//3).astype(int), 6))
    w = tuple(int(v) for v in np.minimum((obs[4:8]*30//4).astype(int), 5))
    return q+w+(int(round(obs[8]*3)),)

def train_q(episodes=140):
    q={}; curve=[]
    for ep in range(episodes):
        env=QuickCare(seed=1000+ep%35,scenario=['normal','surge','severity_surge','shortage'][ep%4])
        obs,_=env.reset();done=False;total=0
        eps=max(.08,.9*(.985**ep))
        while not done:
            s=state_bin(obs); valid=env.valid_actions()
            if not valid: a=0
            elif random.random()<eps:a=random.choice(valid)
            else:a=max(valid,key=lambda v:q.get((s,v),0))
            n,r,done,_,_=env.step(a);ns=state_bin(n)
            nextvalid=env.valid_actions()
            target=r+(.95*max((q.get((ns,v),0) for v in nextvalid),default=0) if not done else 0)
            q[s,a]=q.get((s,a),0)+.15*(target-q.get((s,a),0))
            obs=n;total+=r
        curve.append(total)
    return q,curve

class Net(nn.Module):
    def __init__(self):
        super().__init__(); self.layers=nn.Sequential(nn.Linear(9,48),nn.ReLU(),nn.Linear(48,48),nn.ReLU(),nn.Linear(48,4))
    def forward(self,x):return self.layers(x)

def train_dqn(episodes=90):
    online=Net();target=Net();target.load_state_dict(online.state_dict())
    opt=torch.optim.Adam(online.parameters(),lr=.001)
    replay=[]; curve=[];steps=0
    for ep in range(episodes):
        env=QuickCare(seed=3000+ep%30,scenario=['normal','surge','severity_surge','shortage'][ep%4]);s,_=env.reset();done=False;total=0
        eps=max(.08,.85*(.977**ep))
        while not done:
            valid=env.valid_actions()
            if not valid:a=0
            elif random.random()<eps:a=random.choice(valid)
            else:
                with torch.no_grad(): vals=online(torch.tensor(s).float()).numpy()
                a=max(valid,key=lambda v:vals[v])
            ns,r,done,_,_=env.step(a);nv=env.valid_actions()
            replay.append((s.copy(),a,r,ns.copy(),done,nv))
            if len(replay)>6000:replay.pop(0)
            s=ns;total+=r;steps+=1
            if len(replay)>=64 and steps%4==0:
                batch=random.sample(replay,64)
                ss=torch.tensor(np.stack([x[0] for x in batch]),dtype=torch.float32)
                aa=torch.tensor([x[1] for x in batch],dtype=torch.long)
                rr=torch.tensor([x[2] for x in batch],dtype=torch.float32)
                nnxt=torch.tensor(np.stack([x[3] for x in batch]),dtype=torch.float32)
                with torch.no_grad():
                    next_vals=target(nnxt).numpy()
                    best=np.array([max((next_vals[i,v] for v in x[5]),default=0) if not x[4] else 0 for i,x in enumerate(batch)])
                    desired=rr+.95*torch.tensor(best,dtype=torch.float32)
                predicted=online(ss).gather(1,aa[:,None]).squeeze(1)
                loss=nn.functional.smooth_l1_loss(predicted,desired)
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(online.parameters(),5);opt.step()
            if steps%250==0:target.load_state_dict(online.state_dict())
        curve.append(total)
    return online,curve

def main():
    print('Training tabular Q-learning ...',flush=True)
    q,qcurve=train_q()
    print('Training DQN ...',flush=True)
    net,dcurve=train_dqn();net.eval()
    torch.save(net.state_dict(),OUT/'dqn_weights.pt')
    (OUT/'q_table.json').write_text(json.dumps([{'state':list(s),'action':a,'value':v} for (s,a),v in q.items()]))
    with (OUT/'training_curves.csv').open('w',newline='') as f:
        wr=csv.writer(f);wr.writerow(['algorithm','episode','reward'])
        for name,curve in [('Tabular Q-learning',qcurve),('DQN',dcurve)]:
            for i,v in enumerate(curve):wr.writerow([name,i+1,round(v,4)])
    policies={
        'FCFS':lambda e,s:fcfs(e),
        'Severity priority':lambda e,s:priority(e),
        'Tabular Q-learning': lambda e,s: max(e.valid_actions(),key=lambda v:q.get((state_bin(s),v),0)) if e.valid_actions() else 0,
        'DQN':lambda e,s:max(e.valid_actions(),key=lambda v:net(torch.tensor(s).float()).detach().numpy()[v]) if e.valid_actions() else 0,
    }
    rows=[]; trajectories={}
    for scenario in ['normal','surge','severity_surge','shortage']:
        for name,policy in policies.items():
            for seed in [101,102,103,104,105]:
                m,curve=run_policy(policy,scenario,seed)
                rows.append({'scenario':scenario,'policy':name,'seed':seed,**m})
                if seed==101:trajectories[(scenario,name)]=curve
    with (OUT/'evaluation.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4))
    for label, curve in [('Q-learning',qcurve),('DQN',dcurve)]:
        xs=np.arange(len(curve)); window=min(12,len(curve))
        smooth=np.convolve(curve,np.ones(window)/window,mode='valid')
        ax.plot(xs[window-1:],smooth,label=label)
    ax.set(xlabel='Training episode',ylabel='Smoothed episode return',title='Initial RL training (different training lengths)');ax.legend();fig.tight_layout();fig.savefig(OUT/'training.png',dpi=155);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for name in policies:
        selected=[r['mean_queue'] for r in rows if r['scenario']=='surge' and r['policy']==name]
        ax.bar(name,np.mean(selected),yerr=np.std(selected,ddof=1),capsize=3)
    ax.set(ylabel='Mean queued cases (5 seeds)',title='Surge: mean queue length, error bars = SD')
    ax.tick_params(axis='x',labelrotation=15);fig.tight_layout();fig.savefig(OUT/'surge_comparison.png',dpi=155);plt.close(fig)
    means={}
    for scenario in ['normal','surge','severity_surge','shortage']:
        means[scenario]={}
        for name in policies:
            select=[r for r in rows if r['scenario']==scenario and r['policy']==name]
            means[scenario][name]={key:round(float(np.mean([r[key] for r in select])),2) for key in ('mean_wait_served','mean_urgent_wait_served','mean_queue','started','unfinished_waiting','reward')}
    (OUT/'summary.json').write_text(json.dumps(means,indent=2))
    print(json.dumps(means,indent=2),flush=True)

if __name__=='__main__':main()
