"""Real hosted API baseline: one six-question request per image, no retries."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jev_multimodal import Question
from jev_multimodal.evidence import Evidence, Snapshot, visual_evidence
from jev_multimodal.typesafe import JevClient, JevError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--captions', required=True)
    parser.add_argument('--predictions')
    parser.add_argument('--mode', choices=['caption', 'visual_evidence'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError('Workers must be 1..4')
    captions = [json.loads(s) for s in Path(args.captions).read_text().splitlines()]
    visual = {}
    if args.mode == 'visual_evidence':
        if not args.predictions:
            parser.error('--predictions required for visual_evidence')
        visual = {r['image_id']: r for r in (json.loads(s) for s in Path(args.predictions).read_text().splitlines()) if r['mode'] == 'shared'}
    def run(row):
        questions = [Question.from_dict(q) for q in row['questions']]
        if args.mode == 'caption':
            evidence = (Evidence(row['image_id'], row['image_sha256'], 'image', row['model']+':caption', row['image_sha256'], {'caption': row['caption']}),)
            front_ms = row['elapsed_ms']
        else:
            prediction = visual[row['image_id']]
            if prediction['image_sha256'] != row['image_sha256']:
                raise ValueError('Image mismatch across frontends')
            evidence = visual_evidence(prediction, row['image_id'], row['image_sha256'], questions=questions)
            front_ms = prediction['metrics']['elapsed_ms']
        record = {'image_id': row['image_id'], 'mode': args.mode, 'frontend_ms': front_ms,
                  'cold_first_caption': row['cold_first_image'], 'retries': 0}
        import time
        began = time.perf_counter()
        try:
            record['result'] = JevClient().judge(Snapshot(evidence), questions)
            record['status'] = 'ok'
        except JevError as error:
            record.update(status='error', error=str(error))
        record['api_wall_ms'] = (time.perf_counter()-began)*1000
        record['component_sum_ms'] = front_ms+record['api_wall_ms']
        return record
    with Path(args.output).open('x') as stream, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, record in enumerate(pool.map(run, captions)):
            stream.write(json.dumps(record)+'\n'); stream.flush()
            print(json.dumps({'completed': i+1, 'total': len(captions), 'status': record['status']}), flush=True)


if __name__ == '__main__':
    main()
