"""Extract actual committed losses, rewards and wall time; never claim convergence."""
import argparse
import csv
from pathlib import Path
import statistics as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import read,write,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--progress',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    progress=read(a.progress);steps=[];times=[]
    for ri,records in enumerate(progress['rounds']):
        updates=[];rewards=[]
        for record in records:
            if sha(record['report'])!=record['report_sha256']:raise ValueError('Report changed')
            report=read(record['report']);events=[e for e in report['events'] if e['kind']=='online_update']
            us=[u for e in events for u in e['optimizer_updates']]
            if len(us)!=8:raise ValueError('Missing optimizer updates')
            updates.append(us);rr=[]
            for u,e in zip(us,events):
                if abs(u['loss']-u['policy_loss']-.001*u['kl'])>1e-6:raise ValueError('Unexpected loss formula')
                rv=[r['reward'] for r in e['branch_feedback']['rewards']]
                rr.append(dict(reward=rv[int(np.argmax(e['current_logits']))],parent_reward=rv[int(np.argmax(e['reference_logits']))]))
            rewards.append(rr)
        for k in range(8):
            row=dict(update_attempt=ri*8+k+1,policy_version=updates[0][k]['version_after'],episodes=(ri+1)*8,coverage_epochs=(ri+1)*8/150)
            row.update({key:st.mean(u[k][key] for u in updates) for key in ('loss','policy_loss','kl')})
            row.update({key:st.mean(r[k][key] for r in rewards) for key in ('reward','parent_reward')})
            steps.append(row)
        times.append(max(r['elapsed_seconds'] for r in records))
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'optimizer_steps.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(steps[0]));w.writeheader();w.writerows(steps)
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained');x=np.array([r['update_attempt'] for r in steps])
    for ax,keys in zip(axs.flat,[('loss',),('policy_loss',),('kl',),('reward','parent_reward')]):
        for key in keys:
            y=np.array([r[key] for r in steps]);smooth=np.array([np.mean(y[max(0,i-31):i+1]) for i in range(len(y))])
            line=ax.plot(x,smooth,label=key)[0];ax.plot(x,y,alpha=.18,color=line.get_color(),lw=.5)
        ax.axvline(904,color='black',ls='--',lw=1);ax.set_xlabel('Global update attempt');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('B continuation: p6 boundary at 904; reward is training-context diagnostic')
    for suffix in ('png','pdf'):fig.savefig(a.output/('training_curves.'+suffix),dpi=160)
    write(a.output/'training_summary.json',dict(episodes=progress['completed'],coverage_epochs=progress['completed']/150,
        continuation_wall_hours=progress.get('training_wall_seconds',0)/3600,
        parent_wall_hours=progress.get('parent_training_wall_seconds',0)/3600,
        committed_round_max_sum_hours=sum(times)/3600,
        continuation_round_max_sum_hours=sum(times[113:])/3600,
        reward_is_held_out_eval=False,convergence_claim=False,global_update_attempts=len(steps)))


if __name__=='__main__':main()
