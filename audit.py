import hashlib
import json
from pathlib import Path
from importlib.metadata import version

train=[json.loads(x) for x in Path('data/train.jsonl').read_text().splitlines()]
val=[json.loads(x) for x in Path('data/validation.jsonl').read_text().splitlines()]
assert not ({x['dialogue_id'] for x in train}&{x['dialogue_id'] for x in val})
assert not ({x['prompt'] for x in train}&{x['prompt'] for x in val})
assert all(x['prompt'].endswith('<think>\n\n</think>\n\n') or '<|im_start|>assistant' in x['prompt'] for x in train+val)
manifest={'versions':{p:version(p) for p in ['torch','trl','transformers','peft','vllm','datasets']},'sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['evaluate.py','bfcl.jsonl','data/train.jsonl','data/validation.jsonl']},'dialogue_overlap':0,'prompt_overlap':0}
Path('results/manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
