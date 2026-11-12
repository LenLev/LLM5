"""Use the first user/assistant exchange; split whole dialogues before sampling."""
import ast
import json
import random
import re
from pathlib import Path
from collections import Counter
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

REVISION = '6bda777c88d21e5a204703c1ee45597a8fa4f734'
BASE_REVISION = '15852e8c16360a2fea060d615a32b45270f8a8fc'

def schema(x):
    if isinstance(x, list):
        return [schema(v) for v in x]
    if not isinstance(x, dict):
        return x
    out = {k: schema(v) for k, v in x.items() if v is not None}
    if isinstance(out.get('type'), str):
        out['type'] = {'dict':'object','float':'number','tuple':'array','any':'string'}.get(out['type'], out['type'])
    return out

def parse_reference(text, mapping):
    text = text.strip()
    if not text.startswith('['):
        return []
    for old in sorted(mapping, key=len, reverse=True):
        text = re.sub(re.escape(old) + r'(?=\s*\()', mapping[old], text)
    root = ast.parse(text, mode='eval').body
    if not isinstance(root, ast.List):
        raise ValueError('not a call list')
    result = []
    for call in root.elts:
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.args:
            raise ValueError('unsupported call')
        if call.func.id not in mapping.values() or any(k.arg is None for k in call.keywords):
            raise ValueError('unknown function')
        result.append({'name':call.func.id,'arguments':{k.arg:ast.literal_eval(k.value) for k in call.keywords}})
    return result

def main():
    ds = load_dataset('Team-ACE/ToolACE', revision=REVISION, split='train')
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B', revision=BASE_REVISION)
    ids = list(range(len(ds)))
    random.Random(42).shuffle(ids)
    val_ids = set(ids[:1130])
    rows = {'train':[], 'validation':[]}
    rejected = Counter()
    lengths, answers, counts = [], [], []
    seen = set()
    for i, row in enumerate(ds):
        try:
            tools, _ = json.JSONDecoder().raw_decode(row['system'][row['system'].index('[{'):])
            mapping = {f['name']:re.sub(r'\W+', '_', f['name']).strip('_') for f in tools}
            if len(set(mapping.values())) != len(tools):
                raise ValueError('duplicate names')
            tools = [dict(schema(f), name=mapping[f['name']]) for f in tools]
            conv = row['conversations']
            if len(conv)<2 or conv[0]['from']!='user' or conv[1]['from']!='assistant':
                raise ValueError('first exchange')
            if not conv[1]['value'].strip():
                raise ValueError('empty reference')
            ref = parse_reference(conv[1]['value'], mapping)
            prompt = tok.apply_chat_template([{'role':'user','content':conv[0]['value']}], tools=[{'type':'function','function':f} for f in tools], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            size = len(tok.encode(prompt, add_special_tokens=False))
            lengths.append(size)
            answers.append(len(tok.encode(conv[1]['value'])))
            counts.append(len(tools))
            if size > 1536:
                rejected['prompt_over_1536'] += 1
                continue
            if prompt in seen:
                rejected['duplicate_prompt'] += 1
                continue
            seen.add(prompt)
            rows['validation' if i in val_ids else 'train'].append({'prompt':prompt,'reference':json.dumps(ref,ensure_ascii=False),'tools_json':json.dumps(tools,ensure_ascii=False),'dialogue_id':i,'has_call':bool(ref)})
        except (ValueError, SyntaxError, TypeError, KeyError) as e:
            rejected[type(e).__name__] += 1
    Path('data').mkdir(exist_ok=True)
    stats = {'source':'Team-ACE/ToolACE','revision':REVISION,'seed':42,'raw_dialogues':len(ds),'method':'first user/assistant exchange only','rejected':dict(rejected)}
    for name, values in rows.items():
        random.Random(42).shuffle(values)
        stats[name+'_eligible'] = len(values)
        limit = 2048 if name=='train' else 128
        # Keep naturally occurring proportions; no benchmark data enters training.
        values = values[:limit]
        Path(f'data/{name}.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in values))
        stats[name] = {'examples':len(values),'with_call':sum(x['has_call'] for x in values),'without_call':sum(not x['has_call'] for x in values)}
    for name, values in [('prompt_tokens',lengths),('reference_tokens',answers),('tools',counts)]:
        stats[name] = dict(zip(['min','median','p90','max'], map(float,np.percentile(values,[0,50,90,100]))))
    Path('data/eda.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2))
    print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)

if __name__ == '__main__':
    main()
