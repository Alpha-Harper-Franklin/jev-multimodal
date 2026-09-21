"""Paired-image grounding and ordinal composition from POPE questions.

Labels are not read here. Ground truth can be joined from the original sidecar.
Pairs are independent photographs, not a video or a temporal benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jev_multimodal import Question
from jev_multimodal.cuda_qwen import QwenVision

p=argparse.ArgumentParser()
p.add_argument('--model',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
args=p.parse_args()
rows=[json.loads(s) for s in Path(args.manifest).read_text().splitlines()]
if len(rows)%2:raise ValueError('Even image count required')
engine=QwenVision(args.model,batch_size=4)
out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
with (out/'predictions.jsonl').open('x') as f:
    for index in range(0,len(rows),2):
        pair=rows[index:index+2]
        for row in pair:
            if hashlib.sha256(Path(row['image_path']).read_bytes()).hexdigest()!=row['image_sha256']:raise ValueError('Hash mismatch')
        questions=[]
        mapping=[]
        for image_index,row in enumerate(pair,1):
            for q in row['questions']:
                ident=f'image{image_index}:{q["id"]}'
                questions.append(Question.yes_no(ident,f'Considering image {image_index} only: '+q['instructions']))
                mapping.append({'id':ident,'original_ids':[q['id']],'kind':'noul'})
            for start in (0,2,4):
                group=row['questions'][start:start+2]
                objects=[]
                for q in group:
                    match=re.fullmatch(r'Is there (?:a|an) (.+) in the image\?',q['instructions'])
                    if not match:raise ValueError('Unexpected POPE question grammar')
                    objects.append(match.group(1))
                ident=f'count{image_index}:{group[0]["id"]}:{group[1]["id"]}'
                question=f'Considering image {image_index} only, how many of these two categories are visibly present: {objects[0]}; {objects[1]}? Count categories, not individual objects.'
                questions.append(Question.ordinal(ident,question,['Neither category is present','Exactly one category is present','Both categories are present']))
                mapping.append({'id':ident,'original_ids':[q['id'] for q in group],'kind':'score'})
        modes=['independent','batched','shared']
        offset=(index//2)%3
        for mode in modes[offset:]+modes[:offset]:
            result=engine.judge([r['image_path'] for r in pair],questions,mode=mode)
            result.update(pair_index=index//2,mapping=mapping,first_pair_cold=index==0)
            f.write(json.dumps(result)+'\n');f.flush()
        print(json.dumps({'pairs_complete':index//2+1,'total_pairs':len(rows)//2}),flush=True)
(out/'complete.json').write_text(json.dumps({'pairs':len(rows)//2,'questions_per_pair':18,'scope':'paired-image grounding and derived ordinal categories; not video'}))
