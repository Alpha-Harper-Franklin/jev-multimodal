# Two-image grounding and ordinal decisions

32 ordered pairs built from the earlier 64 POPE random images. Each request contains two independent photographs, 12 image-specific Noul questions and six Score questions. Score asks whether neither, one or both of two named categories are present in a specified image; its labels are sums of the corresponding POPE labels. No labels enter inference.

This is an exploratory extension on reused development images. It is neither a video benchmark nor an evaluation of object-instance counting. The 192 ordinal questions and 384 binary questions are dependent through the same 64 images.

| Mode | Noul accuracy (384) | Ordinal argmax accuracy (192) | Expected-score MAE | Request latency p50 |
|---|---:|---:|---:|---:|
| Independent full forwards | 87.50% | 73.96% | 0.2644 | 3,130 ms |
| Ordinary full-prompt batches | 87.50% | 74.48% | 0.2629 | 3,139 ms |
| Shared image prefix | 87.76% | 73.96% | 0.2629 | 422 ms |

Qwen2.5-VL-7B-Instruct on one RTX 4090, BF16 transformer with FP32 final readout, batch size four. The first pair is excluded from latency because it includes cold execution. Mode order rotates by pair. Accuracy uses all pairs. These measurements show reduced repeated computation; they do not show improved ordinal accuracy.

[Raw predictions](predictions.jsonl) · [Summary](summary.json) · [Completion record](complete.json). The separate [capability canary](../capabilities/batched-parity.json) tests mixed answer types, reordered images, restoration and probability differences. Its cold timings are not performance measurements.

```bash
python experiments/download_pope.py --images 64 --output runs/pope64
python experiments/run_multimage.py --model /path/to/Qwen2.5-VL-7B-Instruct --manifest runs/pope64/manifest.jsonl --output runs/paired-images64
python experiments/analyze_multimage.py --predictions runs/paired-images64/predictions.jsonl --labels runs/pope64/labels.jsonl --output runs/paired-images64/summary.json
```

The measurement uses the checked-in runner's fixed 32-pair protocol. Keep the seed and image count for reproduction.
