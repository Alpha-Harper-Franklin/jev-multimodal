# Project checks

- Do not infer visual capability from a demo video: inspect what enters the model.
- Ground-truth labels and calibration/test splits belong in sidecars, never inference state.
  Read the manifest schema before adding fields to a runner.
- Keep immutable raw predictions, failed requests, source hashes, and environment records.
- Shared BF16 transformer computation is not bitwise equivalent to independent forwards.
  Check choice flips, probability deltas, reordered questions, and image isolation.
- Community prefix sharing, candidate readout, and temperature scaling are prior art.
- Do not call conditional candidate softmax calibrated correctness without task-specific evidence.
- Optional extractors need real runtime validation before being advertised as validated.
- Do not execute downloaded third-party repositories or copy incompatible licensed code.
- Run `python -m unittest discover -s tests -v` and the installed CLI before publishing.
