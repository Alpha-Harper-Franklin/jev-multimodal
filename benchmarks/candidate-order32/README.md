# Candidate order sensitivity

32 development images, six POPE random questions each, Qwen2.5-VL-7B shared-prefix mode. Every question is scored once with `A=Yes, B=No` and once with `A=No, B=Yes`, using Choice so that candidate order can vary. Image order and question wording are unchanged; execution order alternates by image.

- 1 of 192 choices flips when candidates reverse.
- Maximum yes-probability change: 0.218993; mean: 0.009132.
- No flipped choice had at least 0.95 conditional probability in the original ordering in this sample.

No accuracy labels are read here. This measures sensitivity, not correctness or a validated confidence guarantee. These are reused development images, and a zero high-confidence-flip count in 192 questions does not establish immunity elsewhere.

[Raw predictions](predictions.jsonl) · [Summary and per-question comparisons](summary.json).

```bash
python experiments/check_candidate_order.py --model /path/to/model --manifest runs/pope64/manifest.jsonl --images 32 --output runs/candidate-order32
```

Candidate-order checks were motivated by the `alpha-sys-1` readout audit. No third-party implementation was copied.
