"""Held-out ToolACE evaluation; never used as training examples."""
import json
import sys
from pathlib import Path
from statistics import mean
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from rewards import score

def main():
    name=sys.argv[1]
    model='Qwen/Qwen3.5-2B' if name=='base' else f'checkpoints/{name}'
    revision='15852e8c16360a2fea060d615a32b45270f8a8fc'
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B',revision=revision)
    rows=[json.loads(line) for line in Path('data/validation.jsonl').read_text().splitlines()]
    engine=LLM(model=model,revision=revision if name=='base' else None,tokenizer='Qwen/Qwen3.5-2B',tokenizer_revision=revision,dtype='bfloat16',language_model_only=True,max_model_len=4096,max_num_seqs=32,gpu_memory_utilization=0.85,enforce_eager=True,seed=42,generation_config='vllm')
    outputs=engine.generate([{'prompt_token_ids':tok.encode(r['prompt'],add_special_tokens=False)} for r in rows],SamplingParams(temperature=0,max_tokens=1024,skip_special_tokens=False))
    results=[]
    for row,out in zip(rows,outputs):
        result=out.outputs[0]
        results.append({'dialogue_id':row['dialogue_id'],'response':result.text,'reference':json.loads(row['reference']),'exact':score(result.text,row['reference'],tools=row['tools_json']) if result.finish_reason!='length' else 0.,'dense':score(result.text,row['reference'],True,row['tools_json']) if result.finish_reason!='length' else 0.,'finish_reason':result.finish_reason})
    path=Path('results')/name
    path.mkdir(parents=True,exist_ok=True)
    (path/'validation_predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results))
    (path/'validation.json').write_text(json.dumps({'n':len(results),'exact':mean(r['exact'] for r in results),'dense':mean(r['dense'] for r in results)},indent=2))
    print((path/'validation.json').read_text())

if __name__ == '__main__':
    main()
