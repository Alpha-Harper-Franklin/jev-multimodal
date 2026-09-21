# PDF + real speech integration

The local PDF/ASR extractors and a combined Jev request were run, with no mocked extraction or API response. This is a small integration fixture, not an accuracy benchmark.

- PDF: an authored two-page file. Page 1 states service hours and a 30-day refund deadline; page 2 is blank and is correctly marked `needs_ocr`. Extraction retained page numbers.
- Audio: the public `openai/whisper` JFK test clip, decoded locally by faster-whisper-tiny.en on CPU. Both transcript segments retain start/end times. Its 22 normalized reference words were reproduced without word errors in this run; this is not a general ASR accuracy claim.
- The combined snapshot contains four evidence records. A single real Jev request returned the expected decisions for all five authored checks, including `unknown` for unspecified weekend availability. These checks concern each modality separately; they do not prove cross-modal reasoning gains.

[Extraction summary](summary.json) · [Document evidence](document.json) · [Speech evidence](speech.json) · [Combined snapshot](combined.json) · [Jev response](jev-response.json) · [Authored PDF](authored-fixture.pdf).

The audio comes from [`openai/whisper` at 86098128c0b4f24f0e2aa2994de830614b474227](https://github.com/openai/whisper/blob/86098128c0b4f24f0e2aa2994de830614b474227/tests/jfk.flac). Model revision: [`Systran/faster-whisper-tiny.en` at 0d3d19a32d3338f10357c0889762bd8d64bbdeba](https://huggingface.co/Systran/faster-whisper-tiny.en/tree/0d3d19a32d3338f10357c0889762bd8d64bbdeba). Source asset hashes and dependency versions are in the summary. Audio and weights are not redistributed.

```bash
python -m pip install '.[documents,audio]'
python experiments/validate_adapters.py --audio /path/to/jfk.flac --asr-model /path/to/faster-whisper-tiny.en --output runs/adapters
python -m jev_multimodal merge --evidence runs/adapters/document.json runs/adapters/speech.json --output runs/combined.json
# Set TYPESAFE_API_KEY in the process environment.
python -m jev_multimodal judge --evidence runs/combined.json --questions examples/mixed_evidence_questions.json --output runs/decisions.json
```

Recorded extraction took 10 ms for the PDF and 3,233 ms for audio including CPU model loading. The separate API call took 771 ms. These are single integration runs and not latency distributions. OCR remains an optional implemented adapter without a validated Tesseract runtime in this release environment.
