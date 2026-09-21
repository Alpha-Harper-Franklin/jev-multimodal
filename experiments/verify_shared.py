"""Real-image independent/cache/batch equivalence check before larger evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jev_multimodal import Question
from jev_multimodal.cuda_qwen import QwenVision

parser=argparse.ArgumentParser()
parser.add_argument('--model',required=True)
parser.add_argument('--manifest',required=True)
parser.add_argument('--output',required=True)
args=parser.parse_args()
row=json.loads(Path(args.manifest).read_text().splitlines()[0])
if hashlib.sha256(Path(row['image_path']).read_bytes()).hexdigest()!=row['image_sha256']:
    raise ValueError('Image hash mismatch')
questions=[Question.from_dict(q) for q in row['questions']]
engine=QwenVision(args.model,batch_size=4)
independent=engine.judge(row['image_path'],questions,mode='independent')
shared=engine.judge(row['image_path'],questions,mode='shared')
reordered=engine.judge(row['image_path'],list(reversed(questions)),mode='shared')
by_id={a['id']:a for a in reordered['answers']}
delta=max(abs(a['probabilities'][k]-b['probabilities'][k]) for a,b in zip(independent['answers'],shared['answers']) for k in a['probabilities'])
agreement=all(a['choice']==b['choice'] for a,b in zip(independent['answers'],shared['answers']))
reorder_delta=max(abs(a['probabilities'][k]-by_id[a['id']]['probabilities'][k]) for a in shared['answers'] for k in a['probabilities'])
report={'independent':independent,'shared':shared,'reordered':reordered,'max_probability_delta':delta,
        'choice_agreement':agreement,'reorder_max_probability_delta':reorder_delta,
        'passed':agreement and delta<0.02 and reorder_delta<0.02 and shared['metrics']['vision_forward_calls']==1}
target=Path(args.output); target.parent.mkdir(parents=True,exist_ok=True)
with target.open('x',encoding='utf-8') as stream:
    json.dump(report,stream,indent=2)
print(json.dumps({k:v for k,v in report.items() if k not in ('independent','shared','reordered')}),flush=True)
if not report['passed']:
    raise SystemExit(1)
