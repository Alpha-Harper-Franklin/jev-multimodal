"""Compare hosted pipelines on paired successful images, retaining failures."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jev_multimodal.metrics import binary_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--caption-results', required=True)
parser.add_argument('--evidence-results', required=True)
parser.add_argument('--labels', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
load = lambda path: [json.loads(s) for s in Path(path).read_text().splitlines()]
truth = {r['question_id']: r for r in load(args.labels)}
groups = {'caption_to_jev': load(args.caption_results), 'visual_evidence_to_jev': load(args.evidence_results)}
paired = set.intersection(*[{r['image_id'] for r in rows if r['status'] == 'ok'} for rows in groups.values()])
summary = {'scope': 'POPE 64-image development subset', 'latency_semantics': 'frontend measured remotely plus API measured on client; component sum, not one-process wall time',
           'api_runs': 'separate phases; 2 concurrent API workers per phase; no retries', 'paired_success_images': len(paired), 'pipelines': {}}
for name, rows in groups.items():
    ok = [r for r in rows if r['status'] == 'ok']
    same = [r for r in ok if r['image_id'] in paired]
    entry = {'attempted_images': len(rows), 'successful_images': len(ok),
             'errors': dict(Counter(r.get('error') for r in rows if r['status'] != 'ok')),
             'input_tokens_successful': sum(r['result']['input_tokens'] or 0 for r in ok),
             'api_p50_ms_successful': statistics.median(r['api_wall_ms'] for r in ok),
             'component_sum_p50_ms_successful': statistics.median(r['component_sum_ms'] for r in ok),
             'frontend_p50_ms': statistics.median(r['frontend_ms'] for r in rows),
             'api_returned_models': sorted({r['result']['model'] for r in ok})}
    for split in ('all', 'test'):
        answers = [a for r in same for a in r['result']['answers'] if split == 'all' or truth[a['id']]['split'] == split]
        entry['paired_'+split] = binary_metrics([a['noul'] for a in answers], [int(truth[a['id']]['label'] == 'yes') for a in answers])
    summary['pipelines'][name] = entry
Path(args.output).write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
