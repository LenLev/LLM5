import json
from pathlib import Path
from statistics import mean
from datetime import datetime, timezone

lines=['# Ход работы', '',datetime.now(timezone.utc).isoformat(),'','Используется только физическая GPU 1.','']
for name in ['base','rl1','rl2','rl3']:
    metric=Path(f'results/{name}/metrics.json')
    log=Path(f'runs/{name}/metrics.jsonl')
    if metric.exists():
        m=json.loads(metric.read_text())
        lines.append(f'- {name}: BFCL overall {m["overall"]:.2%}.')
    elif log.exists():
        rows=[json.loads(s) for s in log.read_text().splitlines() if s.strip()]
        rows=[r for r in rows if 'loss' in r]
        if rows:
            r=rows[-1]
            lines.append(f'- {name}: шаг {r["step"]}/100; средняя награда за последние 10 шагов {mean(x["reward"] for x in rows[-10:]):.3f}; шагов с ненулевым градиентом {sum(x["grad_norm"]>0 for x in rows)}.')
    else:
        lines.append(f'- {name}: ожидает запуска.')
text='\n'.join(lines)+'\n'
Path('STATUS.md').write_text(text)
print(text)
