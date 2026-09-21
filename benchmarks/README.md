# Historical 64-image pilot and hosted Jev comparison

For the current 500-image, three-mode COCO evaluation, see [the complete report](pope500-all/README.md). This page retains the earlier pilot, API experiments and their failures.

Run date: 2026-09-21. These results support an engineering comparison on a small development subset. They do not establish a new model, state-of-the-art accuracy, temporal competence or driving performance.

## Protocol

- POPE COCO random annotations pinned at `08d957b917e5a378a2f99d35b6293c536a66298b`.
- Sort available image filenames, sample 64 with Python seed `20260921`. Six yes/no questions per image, 384 total.
- First 32 selected images: calibration. Remaining 32: test, 192 questions. Labels/split live in the downloader's sidecar, never in inference input.
- Qwen2.5-VL-7B-Instruct, BF16 transformer, SDPA, processor min pixels 200,704 and max pixels 401,408; final backend uses FP32 vocabulary projection. Microbatch eight suffixes; fixed letter candidates A/Yes and B/No.
- One RTX 4090. Exact Python/Torch/Transformers/config and source hashes are in `environment.json`. Other users had jobs on other GPUs during the FP32 run; these are shared-host measurements.
- Warm up both visual modes, then alternate mode order per image. Timing includes image loading, preprocessing, cache work and CUDA synchronization; excludes model loading. Independent mode uses the same prepared image/prompt tokens but recomputes vision/language for every question.
- No training. Calibration scans a prespecified log-temperature grid from -3 to 3, fits calibration NLL only, and reports test metrics. Pretraining overlap with public images is unknown.

## Final backend: FP32 readout

Raw files: [predictions](pope64-fp32/predictions.jsonl), [environment](pope64-fp32/environment.json), [summary](pope64-fp32/summary.json), [completion](pope64-fp32/complete.json), [first-image canary](pope64-fp32/parity.json).

| Metric | Independent | Shared |
|---|---:|---:|
| Correct / all 384 | 348 | 349 |
| Correct / test 192 | 175 | 175 |
| Test accuracy | 91.15% | 91.15% |
| Latency p50 / p95, ms | 611.31 / 657.74 | 159.50 / 174.74 |
| Visual forward calls, all 64 images | 384 | 64 |
| Test Brier, raw → scaled | 0.07284 → 0.06405 | 0.07252 → 0.06401 |
| Test NLL, raw → scaled | 0.35167 → 0.22202 | 0.35233 → 0.22234 |
| Test top-label ECE, ten equal-width bins | 0.06645 → 0.03393 | 0.06605 → 0.03442 |

Both temperatures fit to 2.7871. This fit was not applied to hosted bridge inputs. It is a diagnostic for this task/configuration, not a general model constant.

Median paired per-image speedup: **3.8118×**, image bootstrap 95% interval [3.7962, 3.8242]. This interval describes image variation within one run, not repeat-run or hardware uncertainty. No statistical accuracy gain is claimed.

Numerical parity is imperfect: **383/384 choices agree**, maximum yes-probability difference **0.02839**, mean **0.000798**. Splitting BF16 transformer computation changes numerical kernels and can flip a near-boundary prediction. The first-image canary passed its 0.02 tolerance, but that tolerance did not hold across all images. `independent` mode reproduces the independent reference path; neither mode is a safety guarantee.

### Scaling diagnostic

One image, question texts repeated with distinct IDs, one timed pass per configuration. This diagnoses scaling, not accuracy or a robust latency distribution.

| Questions | Independent ms | Shared ms | Speed ratio |
|---:|---:|---:|---:|
| 1 | 117.71 | 149.37 | 0.79× |
| 4 | 434.64 | 158.11 | 2.75× |
| 16 | 1695.53 | 241.26 | 7.03× |
| 64 | 6780.38 | 594.26 | 11.41× |

Cache setup has a cost. Do not quote the 64-question ratio as the normal six-question speedup. [Raw scaling](pope64-fp32/scaling.json).

## Real Jev: caption versus compact evidence

Same 64 images and question IDs. Caption baseline uses the same checkpoint, a generic visible-object prompt, greedy generation, at most 160 new tokens. It does not receive task questions while captioning; targeted visual evidence does. This is an application-pipeline comparison, not the best possible question-conditioned caption baseline.

Hosted model requested/returned: `jev-1.13.0`. One request per image, six questions together; two API workers; no retries. API phases were sequential, so service/network differences can confound timing. Captions were generated before the FP32 classifier readout change; generation uses the original model head.

| Metric | Caption → Jev | Compact visual evidence → Jev |
|---|---:|---:|
| Valid responses / attempts | 64 / 64 | 64 / 64 |
| Correct / all 384 | 328 | 349 |
| Correct / test 192 | 163 | 175 |
| Test accuracy | 84.90% | 91.15% |
| Frontend p50 ms | 2238.38 | 159.50 |
| API p50 ms | 827.74 | 955.39 |
| Component-sum p50 ms | 3133.51 | 1104.49 |
| Reported successful input tokens | 47,801 | 82,456 |

Difference: 12 test questions, **6.25 percentage points**, from only 32 test image groups. Ratio of component-sum medians: 2.84×. Component sums combine remote frontend and local API timings; serialization and transfer between machines were not timed as a live service. The first caption is cold and marked. This is not native Jev image latency.

Compact evidence retains source identity, exact candidate probabilities and question text while keeping repeated prompt hashes, raw logits and diagnostics in local artifacts. It does not round probabilities. Input tokens per successful image fell by about 65% compared with the earlier verbose bridge, but remain **73% above captions**. It preserves direct local test accuracy rather than improving it. Direct local inference is faster and needs no API.

Raw records: [captions](pope64/captions.jsonl), [caption API](pope64/caption-jev.jsonl), [compact evidence API](pope64-fp32/compact-evidence-jev.jsonl), [comparison](pope64-fp32/compact-bridge-summary.json).

## Preserved development runs

- `pope64/` visual predictions used BF16 final-head output: p50 588/144 ms, maximum probability delta 0.06146. FP32 reduced the recorded maximum but did not eliminate the single choice flip. Compare timings within each run; source hashes differ.
- `pope64-fp32/evidence-jev.jsonl` is the earlier verbose bridge: 62/64 successes, one transport/JSON error and one HTTP 400, no retries. Its `bridge-summary.json` uses 62 common successful images, not all 64. Failures were not replaced.
- The first paired-run launcher failed on a missing `split` field before writing predictions. The corrected runner reads split information only during offline analysis. That failure produced no scored samples.
- Historical records are retained for audit. Current scripts produce compact-evidence/FP32 results; historical records are not separate independent datasets.

## Reproduce the hosted experiment

After the main README's visual experiment:

```bash
python experiments/caption_pope.py --model /path/to/model --manifest runs/pope64/manifest.jsonl --output runs/captions.jsonl
# Configure TYPESAFE_API_KEY in the process environment.
python experiments/run_jev_bridge.py --captions runs/captions.jsonl --mode caption --output runs/caption-jev.jsonl
python experiments/run_jev_bridge.py --captions runs/captions.jsonl --predictions runs/paired/predictions.jsonl --mode visual_evidence --output runs/evidence-jev.jsonl
python experiments/analyze_bridge.py --caption-results runs/caption-jev.jsonl --evidence-results runs/evidence-jev.jsonl --labels runs/pope64/labels.jsonl --output runs/bridge-summary.json
```

Download receipt: [dataset.json](pope64/dataset.json). Images and credentials are not committed. Prediction IDs join exactly to regenerated labels. Fresh API runs may differ.
