"""Create plots and compact, evidence-based summaries after all runs finish."""
import json
from pathlib import Path
from statistics import mean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

names=['base','rl1','rl2','rl3']
out=Path('report'); out.mkdir(exist_ok=True)
metrics={name:json.loads(Path(f'results/{name}/metrics.json').read_text(encoding='utf-8')) for name in names}
validation={name:json.loads(Path(f'results/{name}/validation.json').read_text(encoding='utf-8')) for name in names}
training={name:[json.loads(s) for s in Path(f'runs/{name}/metrics.jsonl').read_text(encoding='utf-8').splitlines() if 'loss' in json.loads(s)] for name in names[1:]}
fig,axes=plt.subplots(3,2,figsize=(8,7),layout='constrained')
stats={}
for row,name in enumerate(names[1:]):
    log=training[name]
    x=[r['step'] for r in log]
    for col,key in enumerate(['loss','reward']):
        y=[r[key] for r in log]
        axes[row,col].plot(x,y,color='0.7',lw=.7,label='step')
        smooth=[mean(y[max(0,i-9):i+1]) for i in range(len(y))]
        axes[row,col].plot(x,smooth,color='black',lw=1.3,label='mean of last 10')
        axes[row,col].set(title=f'{name.upper()}: {key}',xlabel='Step')
        axes[row,col].grid(alpha=.2)
        if row==0: axes[row,col].legend(fontsize=7)
    stats[name]={'steps':len(log),'first_20_reward':mean(r['reward'] for r in log[:20]),'last_20_reward':mean(r['reward'] for r in log[-20:]),'zero_std_fraction':mean(r['frac_reward_zero_std'] for r in log),'nonzero_gradient_steps':sum(r['grad_norm']>0 for r in log),'clipped_completion_fraction':mean(r['completions/clipped_ratio'] for r in log)}
fig.savefig(out/'training_curves.png',dpi=180)
best=max(names[1:],key=lambda n:metrics[n]['overall'])
benchmark={r['id']:r for r in map(json.loads,Path('bfcl.jsonl').read_text(encoding='utf-8').splitlines())}
preds=[json.loads(s) for s in Path(f'results/{best}/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
errors=[r for r in preds if not r['check']['valid']]
selected=[]
for label,predicate in [
    ('extra_call',lambda r:'irrelevance' in r['category'] and bool(r['calls'])),
    ('missing_call',lambda r:'irrelevance' not in r['category'] and not r['calls']),
    ('simple',lambda r:r['category']=='simple' and bool(r['calls'])),
    ('multiple',lambda r:'multiple' in r['category'] and 'parallel' not in r['category'] and bool(r['calls'])),
    ('parallel',lambda r:'parallel' in r['category'] and bool(r['calls']))]:
    found=next((r for r in errors if predicate(r)),None)
    if found:
        ex=benchmark[found['id']]
        selected.append({'kind':label,**found,'question':ex['question'],'ground_truth':ex['ground_truth']})
summary={'metrics':metrics,'validation':validation,'training':stats,'best_rl':best,'errors_total':len(errors),'error_examples':selected}
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'best_rl':best,'overall':{n:metrics[n]['overall'] for n in names},'training':stats},indent=2))
