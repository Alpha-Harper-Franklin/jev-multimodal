"""Offline label join. No label file is opened in the visual inference process."""
import argparse
import json
from pathlib import Path
import random
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jev_multimodal.metrics import binary_metrics, fit_temperature
from jev_multimodal.schema import normalized


def percentile(values, fraction):
    return sorted(values)[round((len(values)-1)*fraction)]


def analyze(predictions, labels):
    truth = {r['question_id']: r for r in labels}
    if len(truth) != len(labels):
        raise ValueError('Duplicate ground truth ID')
    result = {'scope': '64-image development subset; no accuracy novelty claim', 'modes': {}}
    mode_records = {m: [r for r in predictions if r['mode'] == m] for m in ('independent', 'shared')}
    for mode, records in mode_records.items():
        answers = [a for r in records for a in r['answers']]
        if len(answers) != len(truth) or {a['id'] for a in answers} != set(truth):
            raise ValueError('Missing or repeated predictions')
        calibration = [a for a in answers if truth[a['id']]['split'] == 'calibration']
        test = [a for a in answers if truth[a['id']]['split'] == 'test']
        ys = lambda batch: [int(truth[a['id']]['label'] == 'yes') for a in batch]
        temp = fit_temperature([a['candidate_logits'] for a in calibration], ys(calibration))
        latency = [r['metrics']['elapsed_ms'] for r in records]
        result['modes'][mode] = {
            'all': binary_metrics([a['noul'] for a in answers], ys(answers)),
            'test_raw': binary_metrics([a['noul'] for a in test], ys(test)),
            'test_temperature_scaled': binary_metrics([normalized(a['candidate_logits'], temp)[0] for a in test], ys(test)),
            'fitted_temperature_calibration_only': temp,
            'images': len(records), 'latency_p50_ms': statistics.median(latency),
            'latency_p95_ms': percentile(latency, .95), 'total_ms': sum(latency),
            'vision_forward_calls': sum(r['metrics']['vision_forward_calls'] for r in records)}
    independent = {a['id']: a for r in mode_records['independent'] for a in r['answers']}
    shared = {a['id']: a for r in mode_records['shared'] for a in r['answers']}
    result['parity'] = {'choice_agreement': sum(independent[k]['choice'] == shared[k]['choice'] for k in truth)/len(truth),
                        'max_probability_delta': max(abs(independent[k]['noul']-shared[k]['noul']) for k in truth),
                        'mean_probability_delta': statistics.mean(abs(independent[k]['noul']-shared[k]['noul']) for k in truth)}
    timings = {m: {r['image_id']: r['metrics']['elapsed_ms'] for r in rows} for m, rows in mode_records.items()}
    ids = sorted(timings['shared'])
    ratios = [timings['independent'][k]/timings['shared'][k] for k in ids]
    rng = random.Random(20260921)
    boot = [statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(2000)]
    result['paired_speedup'] = {'median': statistics.median(ratios), 'image_bootstrap_95_ci': [percentile(boot, .025), percentile(boot, .975)]}
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--predictions', required=True); p.add_argument('--labels', required=True); p.add_argument('--output', required=True)
    args = p.parse_args()
    load = lambda path: [json.loads(s) for s in Path(path).read_text().splitlines()]
    summary = analyze(load(args.predictions), load(args.labels))
    Path(args.output).write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
