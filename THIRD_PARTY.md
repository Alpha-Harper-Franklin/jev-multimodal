# Prior work, licenses, and attribution

The implementation in `jev_multimodal/` was written for this repository. It uses public APIs and independently implements established techniques after source review. No third-party source tree, model weight, or private image is vendored. The audit and machine-readable [catalog](research/source-catalog.json) give immutable references.

| Source | License at reviewed version | Influence |
|---|---|---|
| hr98w/jev-visual | MIT | Shared image/context prefix, cache branches, finite-candidate visual scoring |
| TheoLeeCJ/SemIf | MIT | Prefix token boundary validation, suffix batching and masks |
| OmniJev/PlayJev | Apache-2.0 | Single-token label checks, candidate mass diagnostics, FP32 final projection |
| awlevin/typesafe-computer-use | MIT | Source-preserving perception and cache invalidation |
| gaborishka/jev-canvas | MIT | Acquisition-time alignment and stale result rejection |
| jerryjliu/docjev | Apache-2.0 | Explicit context budgets and page ownership |
| genai-craft/openvons | Apache-2.0 notice in LICENSE | Visual decision heads and separate calibration evaluation |
| sseanliu/Jev-Vision | Apache-2.0 notice in LICENSE | Reviewed tree masks/branch-relative positions; not integrated |
| IamBusy/OpenJev-Vision | Apache-2.0 | Reviewed joint distributions and dependency-shift limitations; not integrated |
| r33drichards/laya-vision | Apache-2.0 code; restricted weights | Reviewed visual scorer/training interface; weights not used |

Some GitHub metadata labels permissive LICENSE notices `NOASSERTION`; actual text was checked where specified. A code license does not automatically license weights or training data.

- `jcpsimmons/jev-macos-loop` is AGPL-3.0. Its architecture was read without copying source into this MIT package.
- Repositories without a detected license, including `alektebel/jev-multimodal`, `nullsilver-labs/alpha-sys-1`, `nabendu82/jev-reflex`, and `trungdq88/youtube-sponsor-detection`, were inspected as public references, not treated as permissively licensed code.
- Laya Vision's described visual weights are noncommercial because of ScienceQA; its Score output is untrained. These weights are not included, downloaded or evaluated here.
- Qwen2.5-VL-7B-Instruct supplies the pretrained backbone; its model card/license applies. No Qwen or TypeSafe weights are redistributed.
- TypeSafe Jev is a separately hosted service. Local Qwen outputs are not TypeSafe Jev outputs.
- POPE annotations are referenced at commit `08d957b917e5a378a2f99d35b6293c536a66298b`. COCO images are downloaded from the official bucket for local experiments and not redistributed. Original image licenses still apply.
- PyTorch, Transformers, Pillow, pypdf, pytesseract/Tesseract and faster-whisper retain their own licenses. Optional packages are not bundled.
- The PDF/ASR integration uses the public `openai/whisper` JFK test clip and a `Systran/faster-whisper-tiny.en` checkpoint. Immutable revisions are recorded in `benchmarks/adapters/README.md`; neither audio nor weights are redistributed. The two-page PDF is an original synthetic extraction fixture.

Third-party benchmark numbers were not reproduced. Our numerical claims come only from the committed `benchmarks/` artifacts.
