"""Measure candidate-label order sensitivity without reading accuracy labels."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jev_multimodal import Question
from jev_multimodal.cuda_qwen import QwenVision


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--images', type=int, default=32)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    rows = [json.loads(s) for s in Path(a.manifest).read_text().splitlines()][:a.images]
    assert len(rows) == a.images and a.images > 0
    target = Path(a.output)
    target.mkdir(parents=True, exist_ok=False)
    engine = QwenVision(a.model, batch_size=8)
    comparisons = []
    with (target/'predictions.jsonl').open('x') as output:
        for index, row in enumerate(rows):
            assert hashlib.sha256(Path(row['image_path']).read_bytes()).hexdigest() == row['image_sha256']
            results = {}
            orders = ('yes_first','no_first') if index%2 == 0 else ('no_first','yes_first')
            for order in orders:
                criteria = {'yes':'Yes','no':'No'} if order == 'yes_first' else {'no':'No','yes':'Yes'}
                questions = [Question(q['id'], q['instructions'], dict(criteria)) for q in row['questions']]
                result = engine.judge(row['image_path'], questions, mode='shared')
                result.update(image_id=row['image_id'], candidate_order=order)
                results[order] = {a['id']: a for a in result['answers']}
                output.write(json.dumps(result)+'\n'); output.flush()
            for ident, answer in results['yes_first'].items():
                other = results['no_first'][ident]
                comparisons.append({'image_id':row['image_id'], 'id':ident,
                                    'flip':answer['choice'] != other['choice'],
                                    'probability_delta':abs(answer['probabilities']['yes']-other['probabilities']['yes']),
                                    'first_order_confidence':max(answer['probabilities'].values())})
            print(json.dumps({'completed_images':index+1,'total':len(rows)}),flush=True)
    summary = {'scope':'Candidate-order sensitivity on development images; no accuracy labels, no calibrated certainty claim',
               'images':len(rows),'questions':len(comparisons),
               'choice_flips':sum(x['flip'] for x in comparisons),
               'max_yes_probability_delta':max(x['probability_delta'] for x in comparisons),
               'mean_yes_probability_delta':statistics.mean(x['probability_delta'] for x in comparisons),
               'flips_with_original_confidence_at_least_095':sum(x['flip'] and x['first_order_confidence'] >= .95 for x in comparisons),
               'comparisons':comparisons}
    (target/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__ == '__main__':
    main()
