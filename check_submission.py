"""Check completeness and unchanged benchmark files before submission."""
import hashlib
import json
from pathlib import Path

def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]

expected={r['id'] for r in read_rows('bfcl.jsonl')}
assert len(expected)==350
manifest=json.loads(Path('results/manifest.json').read_text())
for path,expected_hash in manifest['sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected_hash,path
for name in ['base','rl1','rl2','rl3']:
    rows=read_rows(f'results/{name}/predictions.jsonl')
    assert len(rows)==350 and {r['id'] for r in rows}==expected,name
    m=json.loads(Path(f'results/{name}/metrics.json').read_text())
    assert m['benchmark_sha256']==manifest['sha256']['bfcl.jsonl']
    val=read_rows(f'results/{name}/validation_predictions.jsonl')
    assert len(val)==128 and len({r['dialogue_id'] for r in val})==128
    if name!='base':
        log=read_rows(f'runs/{name}/metrics.jsonl')
        assert max(r['step'] for r in log)==100
        assert any(r.get('grad_norm',0)>0 for r in log)
print('PASS: 4 complete BFCL evaluations, 4 validation evaluations, 3 training runs; hashes match.')
