# From an engineering baseline to a research contribution

Status: a proposed research protocol, not completed experiments. The released code and public-image measurements establish reusable inference and evidence handling. They do not establish a new learning algorithm or superiority over all Jev projects.

## Research question

Can a decision model recognize when its multimodal evidence is insufficient, contradictory or out of date, and request the most useful new observation within a fixed latency budget?

The unit of evaluation is a decision at a particular time. Image, speech, OCR and environment observations have separate acquisition times and arrival times. Evidence may describe different states even when every individual extractor is correct. This creates a testable problem beyond adding another encoder or sharing a prefix.

## Required data and separation

Use recorded tasks with independently verified outcomes and controllable observation arrival. Store source bytes, acquisition times, delivery times, sensor failures and ground truth separately. Use environment instrumentation only to assign evaluation labels; a pixels-only condition must never receive privileged object identities, poses or future state.

Partition entire scenes, tasks and recording sessions before generating temporal windows. Keep calibration scenes separate from training and test scenes. Report the pretraining-overlap uncertainty of public datasets. Derived delays, frame drops and conflicting observations must be identified as interventions on recordings, rather than new independent examples.

## Comparisons

| Comparison | What it isolates |
|---|---|
| Strong direct VLM with ordinary batching and equal image resolution | Added decision layer versus using the existing model |
| Shared-prefix candidate inference | Computation reuse; an established technique |
| Question-conditioned caption → Jev | Representation choice under the same question access |
| Compact predictions → Jev | Whether Jev adds value beyond transmitting perception scores |
| Learned visual decision head with matched training data | Efficiency from specialization rather than packaging |
| Timestamp gate without learning | Benefits of simple freshness rules |
| Learned evidence quality and observation selection | Proposed research component |
| Oracle current evidence and each single modality | Ceiling and modality contribution |

Community implementations in the [source audit](SOURCE_REVIEW.zh-CN.md) determine the relevant architectural comparisons. Use their pinned implementations and permitted checkpoints only when their runtime and license can be reproduced. A feature table or an author's benchmark claim is not a measured comparison.

## Proposed learning target

For every decision, predict the candidate distribution plus whether to answer, abstain or request a specified observation. Train on independently labeled outcomes with proper scoring losses; supervise evidence sufficiency from controlled source removal and arrival interventions. Train the observation selector against the measured reduction in decision loss per unit of observation cost, using training scenes only. Preserve an explicit no-improvement outcome so that additional sensing is not always rewarded.

This is a hypothesis. It needs ablations against a frozen encoder, a small trained head and nonlearned gates before selecting an architecture. Do not name it a reproduction of TypeSafe RLCD: its training details have not been published.

## Acceptance criteria for a paper claim

- Report incorrect accepted decisions and coverage together, including per-condition results for missing, stale and conflicting evidence. A lower error rate obtained by rejecting almost everything is not sufficient.
- Count perception, serialization, network, decision, observation refresh and action verification in end-to-end latency. Report p50/p95, throughput, memory, token cost and failure rate separately.
- Compare at matched quality/coverage and matched compute/latency budgets. Include paired scene/session uncertainty intervals and multiple training seeds where training is involved.
- Require a held-out task or sensor-shift result. Test whether temperature and acceptance thresholds learned on one condition still work elsewhere.
- For control claims, measure completed tasks in the actual simulator or robot loop. Static images, a UI animation and offline imitation loss do not establish closed-loop success.
- Publish raw failures, exact splits, source/model revisions and runnable evaluation commands. If the learned method does not beat timestamp rules or direct VLM inference, retain that result and revise the claim.

The next research milestone is one independently labeled temporal task with these baselines. More static POPE examples strengthen the engineering measurement; they cannot substitute for that experiment.
