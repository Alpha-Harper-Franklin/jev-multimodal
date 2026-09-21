"""Real two-image, Choice/Noul/Score, and cache-isolation integration check."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jev_multimodal import Question
from jev_multimodal.cuda_qwen import QwenVision

p=argparse.ArgumentParser()
p.add_argument('--model',required=True); p.add_argument('--manifest',required=True); p.add_argument('--output',required=True)
args=p.parse_args()
rows=[json.loads(s) for s in Path(args.manifest).read_text().splitlines()[:2]]
for row in rows:
    assert hashlib.sha256(Path(row['image_path']).read_bytes()).hexdigest()==row['image_sha256']
engine=QwenVision(args.model,batch_size=3)
questions=[Question.yes_no('person','Considering image 1 only, is there a person visible?'),
           Question('setting','Considering image 2 only, which setting applies?',{'indoor':'Indoors','outdoor':'Outdoors','uncertain':'Cannot determine'}),
           Question.ordinal('text','Considering both images, how much readable text is visible?', ['No readable text','A few readable words','Many readable words'])]
images=[r['image_path'] for r in rows]
shared=engine.judge(images,questions,mode='shared')
independent=engine.judge(images,questions,mode='independent')
batched=engine.judge(images,questions,mode='batched')
reverse=engine.judge(images[::-1],questions,mode='shared')
restored=engine.judge(images,questions,mode='shared')
max_delta=max(abs(a['probabilities'][k]-b['probabilities'][k]) for a,b in zip(shared['answers'],independent['answers']) for k in a['probabilities'])
batch_delta=max(abs(a['probabilities'][k]-b['probabilities'][k]) for a,b in zip(batched['answers'],independent['answers']) for k in a['probabilities'])
restore_delta=max(abs(a['probabilities'][k]-b['probabilities'][k]) for a,b in zip(shared['answers'],restored['answers']) for k in a['probabilities'])
passed=(max_delta<.05 and batch_delta<.05 and restore_delta<.005 and shared['image_sha256']!=reverse['image_sha256'] and shared['metrics']['images']==2
        and shared['metrics']['vision_forward_calls']==1 and [a['type'] for a in shared['answers']]==['noul','choice','score']
        and all(a['choice']==b['choice'] for a,b in zip(shared['answers'],independent['answers'])))
result={'passed':passed,'max_probability_delta':max_delta,'batched_probability_delta':batch_delta,'restore_delta':restore_delta,'shared':shared,'independent':independent,'batched':batched,'reverse':reverse,'restored':restored,
        'scope':'Integration and numerical checks only; no accuracy labels or temporal benchmark'}
out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
with out.open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({k:result[k] for k in ('passed','max_probability_delta','restore_delta','scope')}),flush=True)
if not passed:raise SystemExit(1)
