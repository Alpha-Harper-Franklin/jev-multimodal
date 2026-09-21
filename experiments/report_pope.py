"""Write a measurement report from completed artifacts, with fixed protocol text."""
import argparse
import json
from pathlib import Path


def pct(value):
    return f'{100*value:.2f}%'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', required=True)
    a = p.parse_args()
    root = Path(a.directory)
    summary = json.loads((root/'summary.json').read_text())
    routing = json.loads((root/'routing.json').read_text())
    complete = json.loads((root/'complete.json').read_text())
    assert complete['images'] == 500
    modes = ['independent','batched','shared']
    names = {'independent':'Independent full forwards','batched':'Ordinary full-prompt batches','shared':'Shared visual prefix'}
    rows = summary['modes']
    table = ['| Mode | All accuracy | All F1 | Test accuracy | Request p50 / p95 |',
             '|---|---:|---:|---:|---:|']
    for mode in modes:
        r = rows[mode]
        table.append(f"| {names[mode]} | {pct(r['all']['accuracy'])} | {pct(r['all']['f1'])} | {pct(r['test_raw']['accuracy'])} | {r['latency_p50_ms']:,.1f} / {r['latency_p95_ms']:,.1f} ms |")
    variants = ['| COCO variant | Independent F1 | Ordinary batch F1 | Shared-prefix F1 |','|---|---:|---:|---:|']
    for variant in ('random','popular','adversarial'):
        variants.append('| '+variant+' | '+' | '.join(pct(rows[m]['variants'][variant]['all']['f1']) for m in modes)+' |')
    speed = summary['shared_vs_batched']
    paired = summary['paired_accuracy']['batched']['test']
    overlap = summary['development_overlap']
    held = rows['shared']['test_excluding_development_images']
    unique = rows['shared']['within_image_repeated_prompts']
    raw, scaled = rows['shared']['test_raw'], rows['shared']['test_temperature_scaled']
    target = routing['protocols']['all_variants']['targets']['0.05']
    shifted = routing['protocols']['random_only']['targets']['0.05']
    routing_text = []
    for title, entry in [('All-variant calibration',target),('Random-only calibration',shifted)]:
        selected = entry['test']
        risk = pct(selected['accepted_error_rate']) if selected['accepted_error_rate'] is not None else 'undefined (none accepted)'
        routing_text.append(f"- {title}, empirical 5% calibration error target: chosen threshold `{entry['threshold']}`; test coverage {pct(selected['coverage'])}, accepted errors {selected['errors']}/{selected['accepted']}, accepted error rate {risk}.")
    text = f'''# Complete COCO POPE variants

500 images, 9,000 source question IDs, three inference modes and **27,000 recorded answers**. Every COCO random, popular and adversarial annotation is included. This is the COCO portion of POPE, not all dataset families. No training was performed.

![Measured latency and F1](comparison.png)

## Main comparison

{chr(10).join(table)}

Each request contains all 18 annotated questions for one image. Latency includes loading/preprocessing the image, prefix/cache work, full-vocabulary readout and CUDA synchronization; it excludes initial model loading. Identical question strings occur across variants, so 9,000 IDs are not 9,000 independent problems. There are {unique['distinct_image_prompt_pairs']:,} distinct image/prompt pairs.

Shared versus ordinary batching: **{speed['median_speedup']:.3f}× paired median speedup**, image-bootstrap 95% interval [{speed['image_bootstrap_95_ci'][0]:.3f}, {speed['image_bootstrap_95_ci'][1]:.3f}]. Shared versus independent forwards: {summary['paired_speedup']['median']:.3f}×. Bootstrap resamples image groups within this single run; it does not measure run-to-run or hardware uncertainty.

On the test split, shared-minus-batched accuracy is {100*paired['accuracy_delta']:+.3f} percentage points, paired image-bootstrap interval [{100*paired['image_bootstrap_95_ci'][0]:+.3f}, {100*paired['image_bootstrap_95_ci'][1]:+.3f}]. There are {paired['choice_flips']} choice flips: {paired['improved']} improve and {paired['regressed']} regress. The method reduces repeated computation; a general accuracy improvement is not established.

{chr(10).join(variants)}

Each variant has 3,000 annotated questions. Positive questions and some negatives recur across variants; uncertainty estimates keep each image's variants together.

## Protocol and limits

- Qwen2.5-VL-7B-Instruct; BF16 transformer, SDPA, FP32 final vocabulary projection. Same prompts, candidate letters, image resolution and batch size eight in all modes.
- `independent` recomputes vision and language for every question. `batched` repeats full image/prompt inputs in ordinary length-bucketed microbatches. `shared` computes the image/context prefix once and branches the cache into suffix microbatches. This is not a comparison against vLLM, SGLang or every community implementation.
- One RTX 4090 on a shared host. Other GPUs were occupied by other jobs. All three modes run on the same GPU with rotating mode order and one first-image warmup for every mode.
- The downloader pins POPE annotations, samples the full image list with seed `20260921`, and assigns the first 250 images to calibration and the last 250 to test. Labels and split metadata are read only by offline analysis.
- Of the prior 64 development images, {overlap['calibration_image_overlap']} occur in calibration and {overlap['test_image_overlap']} in test. Removing previously used test images leaves **{held['images']} images / {held['raw']['n']:,} questions**, with shared accuracy **{pct(held['raw']['accuracy'])}**. This additional slice is explicitly a development-overlap check, not a newly collected dataset. Pretraining overlap is unknown.
- BF16 computation is approximate. Shared/ordinary-batched choice agreement is {pct(speed['choice_agreement'])}, maximum yes-probability difference {speed['max_probability_delta']:.6f}. These differences can matter near a decision threshold.
- Within shared-mode requests, {unique['repeated_prompt_groups']} prompt groups repeat; {unique['groups_with_choice_disagreement']} have inconsistent choices across their copies, with maximum probability range {unique['max_probability_range']:.6f}. See raw records rather than assuming identical numerical execution for different microbatch shapes.
- One-image question-count scaling repeats question text intentionally. It is a throughput diagnostic, not additional accuracy evidence.

## Calibration and abstention

Temperature is fit on calibration images only. For shared inference, test Brier is {raw['brier']:.5f} → {scaled['brier']:.5f}, NLL {raw['nll']:.5f} → {scaled['nll']:.5f}, and ten-bin top-label ECE {raw['ece_10_top_label']:.5f} → {scaled['ece_10_top_label']:.5f}. Temperature does not change argmax decisions.

Routing chooses the lowest threshold from a fixed grid meeting an empirical calibration-error target with at least 50 accepted calibration questions. It is not a guaranteed risk bound. The grid, all targets (10%, 5%, 2%) and all variant breakdowns are retained in [routing.json](routing.json).

{chr(10).join(routing_text)}

The random-only fit probes transfer to popular/adversarial negative sampling on disjoint test images. It is a limited sampling-shift diagnostic, not a cross-dataset generalization result. Abstentions are counted, not silently replaced by correct answers from an oracle or a stronger model.

## Artifacts and reproduction

[Raw predictions](predictions.jsonl) · [Summary](summary.json) · [Environment and source hashes](environment.json) · [Dataset receipt](dataset-receipt.json) · [Completion](complete.json) · [Scaling](scaling.json) · [SVG figure](comparison.svg).

```bash
python -m pip install '.[vision,analysis]'
python experiments/download_pope.py --images 500 --variants random popular adversarial --output runs/pope500
python experiments/run_pope.py --model /path/to/Qwen2.5-VL-7B-Instruct --manifest runs/pope500/manifest.jsonl --output runs/full
python experiments/download_pope.py --images 64 --output runs/pope64
python experiments/analyze_pope.py --predictions runs/full/predictions.jsonl --labels runs/pope500/labels.jsonl --development-labels runs/pope64/labels.jsonl --output runs/full/summary.json
python experiments/analyze_routing.py --predictions runs/full/predictions.jsonl --labels runs/pope500/labels.jsonl --output runs/full/routing.json
python experiments/plot_results.py --summary runs/full/summary.json --output runs/full/comparison
```

Use fresh output directories. Images, model weights, private paths and API credentials are not redistributed. These measurements use local visual inference and do not include a Jev API stage; the separate [64-image hosted comparison](../README.md) measures that bridge.
'''
    (root/'README.md').write_text(text, encoding='utf-8')
    print(str(root/'README.md'))


if __name__ == '__main__':
    main()
