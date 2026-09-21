"""Offline, task-specific abstention and calibration-shift diagnostics.

Thresholds use calibration labels only. The risk target is empirical, not a
distribution-free guarantee; report coverage and accepted errors together.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jev_multimodal.metrics import binary_metrics, fit_temperature
from jev_multimodal.schema import normalized


def threshold_for(calibration, target_error):
    grid = [.5, .6, .7, .8, .85, .9, .925, .95, .975, .99, .995, .999]
    for threshold in grid:
        accepted = [(p, y) for p, y in calibration if max(p, 1-p) >= threshold]
        if len(accepted) >= 50 and sum((p >= .5) != bool(y) for p, y in accepted)/len(accepted) <= target_error:
            return threshold
    return None


def routing(rows, threshold):
    selected = [(p, y) for p, y in rows if threshold is not None and max(p, 1-p) >= threshold]
    errors = sum((p >= .5) != bool(y) for p, y in selected)
    return {'total': len(rows), 'accepted': len(selected), 'errors': errors,
            'coverage': len(selected)/len(rows),
            'accepted_error_rate': errors/len(selected) if selected else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--predictions', required=True)
    p.add_argument('--labels', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    labels = [json.loads(s) for s in Path(a.labels).read_text().splitlines()]
    truth = {r['question_id']: r for r in labels}
    assert len(truth) == len(labels)
    records = [json.loads(s) for s in Path(a.predictions).read_text().splitlines()]
    answers = [x for r in records if r['mode'] == 'shared' for x in r['answers']]
    assert len(answers) == len(truth) and {x['id'] for x in answers} == set(truth)
    cal = [x for x in answers if truth[x['id']]['split'] == 'calibration']
    test = [x for x in answers if truth[x['id']]['split'] == 'test']
    assert {truth[x['id']]['image_id'] for x in cal}.isdisjoint({truth[x['id']]['image_id'] for x in test})
    result = {'scope': 'Shared backend, calibration and test image groups disjoint within this development dataset; thresholds give no guaranteed risk bound',
              'threshold_grid': [.5,.6,.7,.8,.85,.9,.925,.95,.975,.99,.995,.999],
              'minimum_calibration_acceptances': 50, 'protocols': {}}
    for protocol in ('all_variants', 'random_only'):
        fit = cal if protocol == 'all_variants' else [x for x in cal if truth[x['id']]['variant'] == 'random']
        ys = lambda batch: [int(truth[x['id']]['label'] == 'yes') for x in batch]
        temperature = fit_temperature([x['candidate_logits'] for x in fit], ys(fit))
        pairs = lambda batch: [(normalized(x['candidate_logits'], temperature)[0], y) for x, y in zip(batch, ys(batch))]
        calibration = pairs(fit)
        entry = {'temperature': temperature, 'calibration_questions': len(fit), 'targets': {}, 'test_variants': {}}
        for target in (.1, .05, .02):
            threshold = threshold_for(calibration, target)
            entry['targets'][str(target)] = {'threshold': threshold,
                                             'calibration': routing(calibration, threshold),
                                             'test': routing(pairs(test), threshold),
                                             'test_variants': {}}
        for variant in ('random', 'popular', 'adversarial'):
            batch = [x for x in test if truth[x['id']]['variant'] == variant]
            calibrated = pairs(batch)
            entry['test_variants'][variant] = {
                'raw': binary_metrics([x['noul'] for x in batch], ys(batch)),
                'temperature_scaled': binary_metrics([x[0] for x in calibrated], [x[1] for x in calibrated])}
            for target in entry['targets'].values():
                target['test_variants'][variant] = routing(calibrated, target['threshold'])
        result['protocols'][protocol] = entry
    Path(a.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v['targets'] for k, v in result['protocols'].items()}, indent=2))


if __name__ == '__main__':
    main()
