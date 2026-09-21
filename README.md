# VisualIf

**Turn visual observations into typed events for software.**

Define semantic conditions over camera, video, or screen observations and let software consume bounded judgments. The planned architecture uses an external perception component to produce reusable structured state, then Jev to evaluate multiple conditions over that state.

**Status: design-stage project.** This repository currently contains the architecture, a sample event specification, and the evaluation plan. There is no working camera pipeline, event runtime, or Jev adapter yet. Example observations are authored fixtures.

## Example applications

- A robot workflow notices that a grasp did not hold.
- A screen workflow notices that an application is waiting for user action.
- A camera workflow notices a package arrival.
- A recorded presentation is indexed by semantic events.

These are proposed applications, not completed demos.

## Proposed architecture

```text
Camera / video / screen
    -> OCR, detector, or vision model
    -> structured observation with timestamp and provenance
    -> Jev judgments over declared questions
    -> temporal logic, deduplication, and event subscribers
```

Perception is intended to run once for several judgments where reuse is valid. Code handles timestamps, expiry, temporal conditions, and event deduplication. State changes invalidate cached judgments. Missing or stale evidence must remain distinguishable from a negative answer.

Jev itself remains text-only. This project does not modify Jev's weights or provide native image input to its API.

`examples/event.json` is a proposed project-level event specification, not an executable configuration or a TypeSafe HTTP request.

## First milestone

- [ ] A recorded-observation runner and event subscription interface.
- [ ] One perception adapter and a Jev decision adapter.
- [ ] Multiple conditions over a shared observation.
- [ ] Explicit observation freshness, missing evidence, and deduplication.
- [ ] A small camera or screen demonstration with actual timing.
- [ ] A direct-VLM structured-output baseline.

## Evaluation

Compare direct VLM judgments against perception plus Jev, varying the number of questions and state reuse. Include perception latency/cost, API latency/cost, event false positives, missed events, and time to detection. Use labelled end-to-end examples and held-out thresholds.

Jev's confidence does not cover upstream perception failures. A second decision layer may add cost when only one question needs answering; any efficiency advantage must be measured.

## 中文说明

VisualIf 计划把摄像头、视频和屏幕中的语义条件转成程序事件。感知前端生成带时间与来源的结构化观察，Jev 负责有限问题判断，代码负责持续时间、去重与状态过期。

当前只有设计和示例，尚未接入相机或 Jev。主要验证问题是：同一观察服务多个判断时，是否比 VLM 直接输出结构化结果更实用、更经济。它不代表 Jev 本体获得了视觉能力。

## Sources

- [TypeSafe model capabilities](https://docs.typesafe.ai/models)
- [TypeSafe primitives](https://docs.typesafe.ai/primitives)
- [TypeSafe confidence](https://docs.typesafe.ai/confidence)
- [Semantic Router](https://github.com/aurelio-labs/semantic-router): related decision-routing infrastructure.

Independent community project; not affiliated with TypeSafe. MIT licensed; external models and components keep their own licenses.
