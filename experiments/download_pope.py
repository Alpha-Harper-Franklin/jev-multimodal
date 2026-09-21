"""Download a deterministic POPE random subset; keep labels outside model input."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import urllib.request

REVISION = '08d957b917e5a378a2f99d35b6293c536a66298b'
ANNOTATIONS = f'https://raw.githubusercontent.com/RUCAIBox/POPE/{REVISION}/output/coco/coco_pope_random.json'
IMAGE_BASE = 'https://s3.amazonaws.com/images.cocodataset.org/val2014/'


def download(url,target):
    if target.exists():
        return target.read_bytes()
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'jev-multimodal-evaluation'}),timeout=25) as response:
        data = response.read()
    if not data:
        raise ValueError('Empty download')
    temporary = target.with_suffix(target.suffix+'.partial')
    temporary.write_bytes(data)
    temporary.replace(target)
    return data


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--images',type=int,default=64)
    parser.add_argument('--seed',type=int,default=20260921)
    args=parser.parse_args()
    base=Path(args.output)
    (base/'images').mkdir(parents=True,exist_ok=True)
    annotation_bytes=download(ANNOTATIONS,base/'pope_random.source.jsonl')
    groups=defaultdict(list)
    for line in annotation_bytes.decode().splitlines():
        row=json.loads(line)
        groups[row['image']].append(row)
    if not 2 <= args.images <= len(groups):
        raise ValueError('Sample size must be between 2 and available image count')
    chosen=random.Random(args.seed).sample(sorted(groups),args.images)
    def image_record(name):
        path=base/'images'/name
        # Path-style access to the official COCO bucket preserves TLS hostname
        # validation; the legacy images.cocodataset.org HTTPS alias can mismatch.
        data=download(IMAGE_BASE+name,path)
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
        return {'image_id':name,'image_path':str(path.resolve()),'image_sha256':hashlib.sha256(data).hexdigest(),
                'questions':[{'id':str(row['question_id']),'type':'noul','instructions':row['text']} for row in groups[name]]}
    with ThreadPoolExecutor(max_workers=4) as pool:
        records=list(pool.map(image_record,chosen))
    labels=[{'question_id':str(row['question_id']),'image_id':name,'label':row['label'],
             'split':'calibration' if i<len(chosen)//2 else 'test'} for i,name in enumerate(chosen) for row in groups[name]]
    (base/'manifest.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records),encoding='utf-8')
    (base/'labels.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in labels),encoding='utf-8')
    receipt={'dataset':'POPE COCO random subset','revision':REVISION,'annotation_url':ANNOTATIONS,
             'annotation_sha256':hashlib.sha256(annotation_bytes).hexdigest(),'seed':args.seed,
             'images':len(records),'questions':len(labels),'image_base':IMAGE_BASE,'split':'first half of seeded sample for calibration; second half for test',
             'images_redistributed':False,'purpose':'development subset; not the full POPE benchmark'}
    (base/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
