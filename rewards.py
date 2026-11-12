"""Safe parsing: never execute generated calls."""
import json
import ast
import re
from collections import Counter

def canonical(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',',':'))

def parse(text, tools=None):
    text = text.replace('<|im_end|>','').strip()
    if not text:
        return None
    if any(text.count(a)!=text.count(b) for a,b in [('<tool_call>','</tool_call>'),('<function=','</function>'),('<parameter=','</parameter>')]):
        return None
    blocks = re.findall(r'<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>', text, re.S)
    if len(blocks) != text.count('<tool_call>'):
        return None
    if not blocks:
        return None if any(t in text for t in ['<tool','<function','<parameter']) else []
    if isinstance(tools, str):
        tools = json.loads(tools)
    schemas = {t['name']:t.get('parameters',{}).get('properties',{}) for t in (tools or [])}
    result = []
    for name, body in blocks:
        args = {}
        matches = list(re.finditer(r'<parameter=([^>]+)>(.*?)</parameter>',body,re.S))
        rest = re.sub(r'<parameter=([^>]+)>(.*?)</parameter>','',body,flags=re.S)
        if rest.strip():
            return None
        for m in matches:
            key, value = m.group(1), m.group(2).strip()
            if key in args:
                return None
            expected = schemas.get(name.strip(),{}).get(key,{}).get('type')
            if expected == 'string':
                args[key] = value
                continue
            if expected == 'boolean' and value.lower() in ('true','false'):
                args[key] = value.lower() == 'true'
                continue
            try:
                args[key] = json.loads(value)
            except json.JSONDecodeError:
                try:
                    args[key] = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    args[key] = value
        result.append({'name':name.strip(),'arguments':args})
    return result

def score(text, reference, dense=False, tools=None):
    pred = parse(text, tools)
    ref = json.loads(reference) if isinstance(reference,str) else reference
    if pred is None:
        return 0.0
    exact = Counter(map(canonical,pred)) == Counter(map(canonical,ref))
    if not dense or not ref:
        return float(exact)
    if not pred:
        return 0.0
    # One-to-one matching prevents extra or duplicate calls from earning free credit.
    remaining = list(ref)
    total = 0.0
    for call in pred:
        candidates = [(j,r) for j,r in enumerate(remaining) if r['name']==call['name']]
        if not candidates:
            continue
        def similarity(r):
            a,b = call['arguments'], r['arguments']
            keys = set(a)|set(b)
            return sum(k in a and k in b and canonical(a[k])==canonical(b[k]) for k in keys)/max(1,len(keys)) if keys else 1.0
        j, target = max(candidates,key=lambda p:similarity(p[1]))
        total += 0.25 + 0.5*similarity(target)
        remaining.pop(j)
    return 0.25*float(exact) + total/max(len(pred),len(ref))

if __name__=='__main__':
    ref=[{'name':'weather','arguments':{'city':'Moscow'}}]
    good='<tool_call>\n<function=weather>\n<parameter=city>Moscow</parameter>\n</function>\n</tool_call>'
    assert score(good,ref,True)==1
    assert score(good.replace('Moscow','Paris'),ref,True)==0.25
    assert score(good+good,ref,True)<1
    assert score('No suitable tool.',[],True)==1
    assert score('<tool_call>broken',[],True)==0
    typed='<tool_call><function=f><parameter=id>123</parameter><parameter=ok>True</parameter></function></tool_call>'
    tools=[{'name':'f','parameters':{'properties':{'id':{'type':'string'},'ok':{'type':'boolean'}}}}]
    assert parse(typed,tools)==[{'name':'f','arguments':{'id':'123','ok':True}}]
    print('Reward checks passed')
