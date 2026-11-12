"""Three independent RL runs, each starting from the same original checkpoint."""
import json
import os
import sys
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForImageTextToText, TrainerCallback
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from rewards import score

MODEL='Qwen/Qwen3.5-2B'
REVISION='15852e8c16360a2fea060d615a32b45270f8a8fc'
CONFIGS = {
    'rl1': {'dense':False, 'learning_rate':1e-5, 'hypothesis':'Exact match provides a simple, strict reward.'},
    'rl2': {'dense':True, 'learning_rate':1e-5, 'hypothesis':'Partial credit gives a signal when exact matches are rare.'},
    'rl3': {'dense':True, 'learning_rate':3e-5, 'hypothesis':'A larger learning rate improves learning within the same step budget.'},
}

class LogCallback(TrainerCallback):
    def __init__(self,path): self.path=path
    def on_log(self,args,state,control,logs=None,**kwargs):
        with self.path.open('a') as f:
            f.write(json.dumps({'step':state.global_step,**(logs or {})})+'\n')

def main():
    name=sys.argv[1]
    cfg=CONFIGS[name]
    output=Path('runs')/name
    output.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,padding_side='left')
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    data=load_dataset('json',data_files={'train':'data/train.jsonl','validation':'data/validation.jsonl'})
    def reward(completions, reference, tools_json, **kwargs):
        values=[score(c,r,cfg['dense'],t) for c,r,t in zip(completions,reference,tools_json)]
        with (output/'reward_examples.jsonl').open('a') as f:
            for c,r,v in zip(completions,reference,values):
                f.write(json.dumps({'response':c,'reference':json.loads(r),'reward':v},ensure_ascii=False)+'\n')
        return values
    model=AutoModelForImageTextToText.from_pretrained(MODEL,revision=REVISION,dtype=torch.bfloat16,attn_implementation='sdpa')
    args=GRPOConfig(
        output_dir=str(output),max_steps=int(os.environ.get('RL_STEPS','100')),
        learning_rate=cfg['learning_rate'],warmup_steps=5,
        per_device_train_batch_size=2,gradient_accumulation_steps=8,
        num_generations=4,max_completion_length=384,
        temperature=0.8,beta=0.0,loss_type='grpo',scale_rewards='group',
        bf16=True,gradient_checkpointing=True,
        use_vllm=True,vllm_mode='colocate',vllm_gpu_memory_utilization=0.35,
        vllm_max_model_length=2048,vllm_enable_sleep_mode=True,
        logging_steps=1,save_strategy='steps',save_steps=50,save_total_limit=2,
        eval_strategy='no',report_to='none',seed=42,data_seed=42,
        mask_truncated_completions=True,
    )
    (output/'config.json').write_text(json.dumps({'experiment':cfg,'training':args.to_dict(),'base_revision':REVISION,'checkpoint_selection':'fixed final step, chosen before BFCL evaluation'},indent=2,default=str))
    trainer=GRPOTrainer(model=model,args=args,processing_class=tok,
        reward_funcs=reward,train_dataset=data['train'],
        peft_config=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.0,task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']),
        callbacks=[LogCallback(output/'metrics.jsonl')])
    trainer.train()
    trainer.save_model(str(output/'adapter'))
    trainer.save_state()
    print('TRAINING_COMPLETE',name,flush=True)

if __name__=='__main__': main()
