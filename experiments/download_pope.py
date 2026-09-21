"""Download a deterministic POPE random subset; keep labels outside model input."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import time
import urllib.request

REVISION = '08d957b917e5a378a2f99d35b6293c536a66298b'
ANNOTATIONS = f'https://raw.githubusercontent.com/RUCAIBox/POPE/{REVISION}/output/coco/coco_pope_random.json'
IMAGE_BASE = 'https://s3.amazonaws.com/images.cocodataset.org/val2014/'


def download(url,target):
    if target.exists():
        return target.read_bytes()
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'jev-multimodal-evaluation'}),timeout=25) as response:
                data = response.read()
            break
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(attempt+1)
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
    parser.add_argument('--variants', nargs='+', choices=['random', 'popular', 'adversarial'], default=['random'])
    args=parser.parse_args()
    base=Path(args.output)
    (base/'images').mkdir(parents=True,exist_ok=True)
    if len(set(args.variants)) != len(args.variants):
        raise ValueError('Duplicate POPE variant')
    groups=defaultdict(list)
    sources = []
    for variant in args.variants:
        url = ANNOTATIONS.replace('coco_pope_random.json', f'coco_pope_{variant}.json')
        annotation_bytes=download(url,base/f'pope_{variant}.source.jsonl')
        sources.append({'variant': variant, 'url': url, 'sha256': hashlib.sha256(annotation_bytes).hexdigest()})
        for line in annotation_bytes.decode().splitlines():
            row=json.loads(line)
            row['variant'] = variant
            row['question_id'] = (variant+':' if len(args.variants)>1 else '')+str(row['question_id'])
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
    records=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for record in pool.map(image_record,chosen):
            records.append(record)
            if len(records)%50==0:
                print(json.dumps({'downloaded_images':len(records),'total':len(chosen)}),flush=True)
    labels=[{'question_id':str(row['question_id']),'image_id':name,'label':row['label'],'variant':row['variant'],
             'split':'calibration' if i<len(chosen)//2 else 'test'} for i,name in enumerate(chosen) for row in groups[name]]
    (base/'manifest.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records),encoding='utf-8')
    (base/'labels.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in labels),encoding='utf-8')
    receipt={'dataset':'POPE COCO '+', '.join(args.variants),'revision':REVISION,'sources':sources,'seed':args.seed,
             'images':len(records),'questions':len(labels),'image_base':IMAGE_BASE,'split':'first half of seeded sample for calibration; second half for test',
             'images_redistributed':False,'purpose':'complete selected COCO variants' if len(chosen)==len(groups) else 'development subset; not the full POPE benchmark'}
    (base/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
