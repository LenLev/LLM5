import sys
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer
from train import MODEL, REVISION
name=sys.argv[1]
base=AutoModelForImageTextToText.from_pretrained(MODEL,revision=REVISION,dtype=torch.bfloat16,device_map='cpu')
model=PeftModel.from_pretrained(base,f'runs/{name}/adapter').merge_and_unload()
model.save_pretrained(f'checkpoints/{name}')
AutoTokenizer.from_pretrained(MODEL,revision=REVISION).save_pretrained(f'checkpoints/{name}')
print('MERGE_COMPLETE',name)
