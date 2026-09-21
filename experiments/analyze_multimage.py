"""Offline join for paired-image grounding and derived ordinal counts."""
import argparse
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jev_multimodal.metrics import binary_metrics

p=argparse.ArgumentParser()
p.add_argument('--predictions',required=True);p.add_argument('--labels',required=True);p.add_argument('--output',required=True)
a=p.parse_args()
rows=[json.loads(s) for s in Path(a.predictions).read_text().splitlines()]
labels={r['question_id']:r for r in (json.loads(s) for s in Path(a.labels).read_text().splitlines())}
result={'scope':'64 source images paired into 32 independent-photo pairs, not video; ordinal labels derived from two original POPE presence labels','modes':{}}
for mode in ('independent','batched','shared'):
    found=[r for r in rows if r['mode']==mode]
    if len(found)!=32 or len({r['pair_index'] for r in found})!=32:raise ValueError('Incomplete or repeated pairs')
    binary=[];by=[];scores=[];sy=[];score_top=[]
    for row in found:
        mapping={r['id']:r for r in row['mapping']}
        for answer in row['answers']:
            target=mapping[answer['id']]
            ys=[int(labels[k]['label']=='yes') for k in target['original_ids']]
            if target['kind']=='noul':binary.append(answer['noul']);by.append(ys[0])
            else:scores.append(answer['score']);sy.append(sum(ys));score_top.append(int(answer['choice']))
    result['modes'][mode]={'binary_grounding':binary_metrics(binary,by),
        'ordinal':{'n':len(sy),'expected_score_mae':statistics.mean(abs(s-y) for s,y in zip(scores,sy)),
                   'argmax_level_accuracy':sum(s==y for s,y in zip(score_top,sy))/len(sy)},
        'latency_p50_ms_excluding_first_pair':statistics.median(r['metrics']['elapsed_ms'] for r in found if not r['first_pair_cold'])}
Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
