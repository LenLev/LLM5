"""Readable live progress in the VS Code remote terminal (standard library only)."""
import json
import time
from pathlib import Path
from datetime import datetime

print('RLHF: Qwen3.5-2B | Только физическая GPU 1',flush=True)
print('Остановка этого просмотра (Ctrl+C) не останавливает обучение.\n',flush=True)
seen={}
while True:
    for name in ['base','rl1','rl2','rl3']:
        result=Path(f'results/{name}/metrics.json')
        path=Path(f'runs/{name}/metrics.jsonl')
        message=None
        try:
            if result.exists():
                m=json.loads(result.read_text())
                message=f'{name.upper():4}: оценка BFCL = {m["overall"]:.2%}'
            elif path.exists():
                rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                rows=[r for r in rows if 'loss' in r]
                if rows:
                    r=rows[-1]
                    message=f'{name.upper():4}: шаг {r["step"]:3}/100 | награда {r["reward"]:.3f} | loss {r["loss"]:.4f}'
            if message and seen.get(name)!=message:
                print(datetime.now().strftime('%H:%M:%S'),message,flush=True)
                seen[name]=message
        except (OSError,json.JSONDecodeError):
            pass
    time.sleep(3)
