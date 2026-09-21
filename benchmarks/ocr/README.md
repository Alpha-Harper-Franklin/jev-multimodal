# OCR, changed images and conflicting sources

Tesseract 4.1.1 / pytesseract 0.3.13 processed two authored images. Each contains service hours and a refund deadline, with the deadline changed from 30 to 90 days in the second image. These are clean integration fixtures, not a representative OCR benchmark.

![The second authored OCR input](authored-90.png)

All ten words were extracted on each pass and every word box stayed within its image. Reading identical bytes again hit the cache; changing the deadline invalidated it and extracted `90 DAYS`. Cache status across the three calls was `[miss, hit, miss]`. Content hashes are preserved separately from caller-supplied revisions. [Summary](summary.json) · [First evidence](evidence-1.json) · [Cached evidence](evidence-2.json) · [Changed-image evidence](evidence-3.json).

We then combined the 90-day OCR evidence (`source_id=screen`) with the 30-day PDF evidence (`source_id=document`) from the PDF fixture. One real `jev-1.13.0` request returned all four expected outputs:

| Question | Observed output |
|---|---|
| Screen deadline, Choice | 90 days |
| PDF deadline, Choice | 30 days |
| Do the sources disagree, Noul | Yes, 0.98 |
| Number of present screen fields, Score | 2 of 2 |

[Combined input](combined.json) · [Actual API output](jev-response.json). Explicit source-scoped questions are in [the example](../../examples/source_scoped_questions.json). This shows one controlled conflict can be represented and answered. It does not establish general conflict resolution or calibrated end-to-end certainty.

```bash
python -m pip install '.[documents]'
# Supply a local Tesseract runtime and its English language data.
python experiments/validate_ocr.py --tesseract /path/to/tesseract --output runs/ocr
python -m jev_multimodal merge --evidence runs/ocr/evidence-3.json runs/adapters/document.json --output runs/ocr/combined.json
python -m jev_multimodal judge --evidence runs/ocr/combined.json --questions examples/source_scoped_questions.json --output runs/ocr/jev-response.json
```

The recorded Linux runtime used project-local extracted Ubuntu packages, without changing system packages. Initial setup required adding the missing `libgif.so.7` dependency before extraction could run. Timings in the summary are single CPU calls, including a cold first call, and are not a throughput benchmark.
