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


def paired_accuracy(reference, candidate, truth, split):
    """Bootstrap image clusters, keeping the three variants of a photo together."""
    groups = {}
    reference_by_id = {a['id']: a for r in reference for a in r['answers']}
    flips, improved, regressed = 0, 0, 0
    for row in candidate:
        differences = []
        for answer in row['answers']:
            label = truth[answer['id']]
            if split is not None and label['split'] != split:
                continue
            base = reference_by_id[answer['id']]['choice']
            ours = answer['choice']
            delta = int(ours == label['label']) - int(base == label['label'])
            differences.append(delta)
            flips += base != ours
            improved += delta == 1
            regressed += delta == -1
        if differences:
            groups[row['image_id']] = (sum(differences), len(differences))
    values = list(groups.values())
    if not values:
        raise ValueError('Empty paired comparison')
    rng = random.Random(20260921)
    bootstrap = []
    for _ in range(2000):
        sample = rng.choices(values, k=len(values))
        bootstrap.append(sum(v[0] for v in sample)/sum(v[1] for v in sample))
    return {'image_clusters': len(values), 'questions': sum(v[1] for v in values),
            'accuracy_delta': sum(v[0] for v in values)/sum(v[1] for v in values),
            'image_bootstrap_95_ci': [percentile(bootstrap, .025), percentile(bootstrap, .975)],
            'choice_flips': flips, 'improved': improved, 'regressed': regressed}


def analyze(predictions, labels, development_images=None):
    truth = {r['question_id']: r for r in labels}
    if len(truth) != len(labels):
        raise ValueError('Duplicate ground truth ID')
    result = {'scope': 'Recorded POPE COCO images; sample size and variants specified by input artifacts', 'modes': {}}
    development_images = set(development_images or ())
    if development_images:
        result['development_overlap'] = {
            'previously_used_images': len(development_images),
            'calibration_image_overlap': len({r['image_id'] for r in labels if r['split'] == 'calibration'} & development_images),
            'test_image_overlap': len({r['image_id'] for r in labels if r['split'] == 'test'} & development_images)}
    modes = sorted({r['mode'] for r in predictions})
    if not {'independent','shared'} <= set(modes):
        raise ValueError('Both independent and shared records are required')
    mode_records = {m: [r for r in predictions if r['mode'] == m] for m in modes}
    for mode, records in mode_records.items():
        if len({r['image_id'] for r in records}) != len(records):
            raise ValueError('Duplicate image record')
        if any(truth.get(a['id'], {}).get('image_id') != r['image_id'] for r in records for a in r['answers']):
            raise ValueError('Prediction assigned to the wrong image')
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
        if development_images:
            held = [a for a in test if truth[a['id']]['image_id'] not in development_images]
            result['modes'][mode]['test_excluding_development_images'] = {
                'images': len({truth[a['id']]['image_id'] for a in held}),
                'raw': binary_metrics([a['noul'] for a in held], ys(held)),
                'temperature_scaled': binary_metrics([normalized(a['candidate_logits'], temp)[0] for a in held], ys(held))}
        unique_prompts, duplicate_groups, inconsistent_choices, max_delta = 0, 0, 0, 0.0
        for row in records:
            groups = {}
            for answer in row['answers']:
                groups.setdefault(answer['prompt_sha256'], []).append(answer)
            unique_prompts += len(groups)
            for group in groups.values():
                if len(group) > 1:
                    duplicate_groups += 1
                    inconsistent_choices += len({a['choice'] for a in group}) > 1
                    max_delta = max(max_delta, max(a['noul'] for a in group)-min(a['noul'] for a in group))
        result['modes'][mode]['within_image_repeated_prompts'] = {
            'distinct_image_prompt_pairs': unique_prompts, 'repeated_prompt_groups': duplicate_groups,
            'groups_with_choice_disagreement': inconsistent_choices, 'max_probability_range': max_delta}
        variants=sorted({truth[a['id']].get('variant','random') for a in answers})
        result['modes'][mode]['variants']={}
        for variant in variants:
            batch=[a for a in answers if truth[a['id']].get('variant','random')==variant]
            held=[a for a in batch if truth[a['id']]['split']=='test']
            result['modes'][mode]['variants'][variant]={'all':binary_metrics([a['noul'] for a in batch],ys(batch)),
                                                        'test':binary_metrics([a['noul'] for a in held],ys(held))}
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
    result['paired_accuracy'] = {
        reference: {split: paired_accuracy(mode_records[reference], mode_records['shared'], truth,
                                           None if split == 'all' else split)
                    for split in ('all', 'test')}
        for reference in modes if reference != 'shared'}
    if 'batched' in mode_records:
        batched={a['id']:a for r in mode_records['batched'] for a in r['answers']}
        ratios=[timings['batched'][k]/timings['shared'][k] for k in ids]
        boot=[statistics.median(rng.choices(ratios,k=len(ratios))) for _ in range(2000)]
        result['shared_vs_batched']={'median_speedup':statistics.median(ratios),
            'image_bootstrap_95_ci':[percentile(boot,.025),percentile(boot,.975)],
            'choice_agreement':sum(batched[k]['choice']==shared[k]['choice'] for k in truth)/len(truth),
            'max_probability_delta':max(abs(batched[k]['noul']-shared[k]['noul']) for k in truth)}
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--predictions', required=True); p.add_argument('--labels', required=True); p.add_argument('--output', required=True)
    p.add_argument('--development-labels', help='Earlier pilot labels; additionally report test images outside this development set')
    args = p.parse_args()
    load = lambda path: [json.loads(s) for s in Path(path).read_text().splitlines()]
    development_images = {r['image_id'] for r in load(args.development_labels)} if args.development_labels else None
    summary = analyze(load(args.predictions), load(args.labels), development_images)
    Path(args.output).write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
