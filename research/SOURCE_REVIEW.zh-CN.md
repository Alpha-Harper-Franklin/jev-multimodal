# Jev 多模态开源代码审阅

检索日期：2026-09-21。完整检索式、返回数量、固定 commit、已取得文件及哈希见 [source-catalog.json](source-catalog.json)。通过 GitHub 搜索、三个社区项目目录和源码交叉查找，范围覆盖视觉、屏幕、文档、语音、手势、视频及机器人。

这是有检索上限的源码调查，不声称穷尽全网。宽泛搜索只取得第一页、最多 100 项；名称相似的 JeVois 摄像头项目等属于误匹配。Google 网页搜索在当前网络下未得到可用结果。下面的“代码”链接指向审阅时固定的版本，避免把更新后的实现当成本次证据。没有在本项目环境中复现的第三方数字均视为作者报告。

## 直接影响本次实现的技术

| 项目与源码 | 实际媒体路径、可借鉴机制 | 本项目处理 |
|---|---|---|
| [jev-visual / adapters.py](https://github.com/hr98w/jev-visual/blob/19af545f096e8db4c4dd5d47aed42d92ec252111/jev_visual/adapters.py) | MLX 视觉模型；共享图像前缀、分支缓存、问题批处理、只投影需要的位置 | 迁移思路到 PyTorch/CUDA、Qwen2.5-VL；独立实现，明确是既有方法 |
| [SemIf / shared.py](https://github.com/TheoLeeCJ/SemIf/blob/ca3ba65f142967030ecb453346e94d6f476a69df/src/semif_phase1/shared.py) | 文本共享前缀；处理 BPE 边界、批量缓存、padding 和 position IDs | 加入稳定 token 前缀、右侧 padding、有效最后位置、微批次缓存隔离 |
| [PlayJev / model.py](https://github.com/OmniJev/PlayJev/blob/2d7a0280841e3244a299c9c640efc8cefc46d476/playjev/model.py#L213) | 已训练的像素游戏策略；FP32 最终读出，候选字母单 token 检查；双帧 temporal patch | 采用 FP32 最终投影思路并记录额外显存；不宣称复现其训练或双帧能力 |
| [typesafe-computer-use / perception.py](https://github.com/awlevin/typesafe-computer-use/blob/cc7b5066ae1a07b5e3182e8f87a9b5b6dfdcffc1/typesafe_computer_use/perception.py#L134) | OCR + Accessibility；变化区域重识别，窗口/区域变化使缓存失效；合并保留来源索引 | 实现内容与提取器配置共同决定的有限缓存、来源记录；区域 OCR 增量算法未移植 |
| [jev-canvas / controller.js](https://github.com/gaborishka/jev-canvas/blob/cec7b27a9f9a3cf12e6692ca07dfeb6125f3cccd/src/controller.js#L200) | MediaPipe 手势 + WebSpeech；“这里”出现时记录指向，限制在途请求，拒绝被新输入替代的旧结果 | 保留采集时间、音频区间、source revision；实现 RevisionGate，不把网络返回时间当采集时间 |
| [docjev / windows.py](https://github.com/jerryjliu/docjev/blob/9ed0fe05984ce1906af9272b8b400c8d46520f98/src/jev_docs/windows.py) | OCR 后按页判断；页窗口区分负责输出的页与上下文页，超预算显式失败 | 保留页码和扫描页 needs_ocr 标记；超预算拒绝，不静默截断；自动分窗尚未实现 |
| [openvons / temperature.py](https://github.com/genai-craft/openvons/blob/92f09020d8f5e00b66817b37000d477d9ba65825/openvons/core/temperature.py) | 独立温度校准、向量校准；另有声音命令的 reject 类 | 独立实现温度网格拟合，按图像隔离 calibration/test；不移用别人的温度或阈值 |

## 必须比较、不能再当作新颖性的视觉路线

| 项目与源码 | 代码级发现 | 对选题的影响 |
|---|---|---|
| [laya-vision / vlm.py](https://github.com/r33drichards/laya-vision/blob/86ccec115ef3d72d1851168fc9b7194e8dbed35a/laya/vlm.py#L334) | SmolVLM + 候选 scorer、类型 embedding、act head；可选择候选段双向注意力；已发布权重 | 小模型视觉决策已有实现。代码 Apache-2.0，但所述视觉权重因 ScienceQA 限制为非商用，Score 未训练；不能统称可商用 |
| [openvons / vlm_decision_model.py](https://github.com/genai-craft/openvons/blob/92f09020d8f5e00b66817b37000d477d9ba65825/openvons/vision/vlm_decision_model.py#L81) | 图片前缀一次计算，repeat cache，多问题 suffix 与可训练 head；另有冻结视觉 tower | CUDA 上共享视觉前缀也有先例；不能把本项目实现方式当成首创 |
| [OpenJev-Vision / public_model.py](https://github.com/IamBusy/OpenJev-Vision/blob/4fa973e5212881d704def6818b0def190b094f12/src/openjev/vision/public_model.py#L96) | 冻结 DINOv2，独立、joint、低秩 binding 三类 head；以显式联合分布回答多个事件 | 固定事件空间适合研究一致性，但不是开放世界视觉问答；作者还报告组合泛化的负面结果，应纳入后续比较 |
| [Jev-Vision / vl.py](https://github.com/sseanliu/Jev-Vision/blob/c33f4a0046ce5db8b05273a3c3c673dc39a211ce/model/s1/vl.py#L66) | Qwen3-VL；tree attention 隔离问题分支；每分支 M-RoPE 从视觉前缀末端重启；训练屏幕操作验证 | “多问题一次前向”与操作前后验证已有训练方案。论文需比较环境真实标签和跨网站划分，不能只比较 API 延迟 |
| [alpha-sys-1 / readout.py](https://github.com/nullsilver-labs/alpha-sys-1/blob/d7fe8575cc70a00daa68f6ad9cbe1e5c14eb33e1/sys1/readout.py) | 图文模型候选 token 读出，候选顺序反转测试、label mass；作者报告多域训练模型 | 增加候选顺序敏感性与分布变化测试，比只展示高 confidence 有意义；未检测到仓库许可证，不复制 |
| [Vision-JEV / model.py](https://github.com/arnodjiang/Vision-JEV/blob/916368d3b0e41ce43c4bff407c21dea2e4e28926/src/vision_jev/model.py) | Qwen3.5-0.8B + LoRA + query/key 候选 pointer；候选 bbox 是输入而非定位预测 | 这是经 smoke test 的研究原型，README 明确尚无任务级权重和提升；不能引用为已验证性能方案 |
| [alektebel/jev-multimodal / model.py](https://github.com/alektebel/jev-multimodal/blob/6b9a4e3bf9fd633eed74f894347aec92eb6e033a/jev_mm/model.py#L76) | SigLIP 图文 fusion + pairwise scorer；当前 forward 的候选 logits 由 state 与 options 产生，question embedding 进入 confidence 分支 | 接口有 question 不等于预测实际条件化于 question；本项目把每个问题放入真实 VLM 后缀。无检测到的许可证，不复制 |
| [QwenServe / model_runner.py](https://github.com/zhangjianbang-nb/qwenserve/blob/2bcbbef522d23a0ff55f977b27bfd32b06ba70d8/src/qwenserve/engine/model_runner.py#L79)；[pipeline.py](https://github.com/zhangjianbang-nb/qwenserve/blob/2bcbbef522d23a0ff55f977b27bfd32b06ba70d8/src/qwenserve/mm/pipeline.py#L175) | README 列剪枝、合并、分页 KV 等；固定版本的 HF runner 只传 input_ids，mm pipeline 使用合成视觉 token 占位 | 不把功能列表当作真实像素加速证据。不集成该参考执行路径；视觉 token 剪枝需另做精度与位置编码验证 |

## 名称没有“多模态”也纳入的相邻项目

| 项目 | 真实输入 / 可借鉴点 | 当前判断 |
|---|---|---|
| [jev-ultrafast](https://github.com/browser-use/jev-ultrafast/tree/1231850a0bf1a0c0341fe408ef1668dbbfdfac46) | 浏览器 DOM、持久控制句柄、多个结构化问题 | 默认回路不看截图；支持先保留结构化观测再请求 Jev 的设计 |
| [arc-cua](https://github.com/shhivv/arc-cua/tree/60e85f35fd1b5a136209ca7fd6e8aa01b54ec2cb) | AX/OCR 混合观测、状态新鲜度 | 属于外部感知后接 Jev，不是 Jev 原生视觉 |
| [third-hand](https://github.com/shhivv/third-hand/tree/430394b35dbb44ff8b303bf19da29b0828d92bd2) | 原生 Accessibility、必要时 OCR | 结构化观测优先，减少昂贵感知 |
| [jev-voice-browser](https://github.com/moritzkremb/jev-voice-browser/tree/054db0f3dbf537af63a8117632d3f941ccd520e1) | 实时转录、去抖、批量问题、旧状态处理 | 可借鉴事件调度；API 消费文本，不直接听音频 |
| [jev-mac-voice](https://github.com/brudarko/jev-mac-voice/tree/631afba3b22d4432c86aa9616ce374fdf358fc5e) | 语音浏览器/桌面动作路径 | 外部语音输入，不是音频基础模型 |
| [youtube-sponsor-detection](https://github.com/trungdq88/youtube-sponsor-detection/tree/de01f0568d043035889a296a61ce21e0accc8b16) | 转录行 ID、时间区间、直播文本窗口 | 不把“处理视频”说成理解视频帧；未检测到许可证，不复制 |
| [jevmeter](https://github.com/ChetasLua/jevmeter/tree/cbf8e117b5b8835e3294c3a8ee652c7dfa737a9a) | Whisper 时间戳、句级 Jev 打分、阶段产物复用 | 保留时间定位；离线字幕打分与视听联合推理分开描述 |
| [tax-doc-classifier](https://github.com/kyotofin/tax-doc-classifier/tree/3e95a77f763c6becb78472f8b2ce2f54237f9214) | PDF 文本抽取、层级候选缩小 | PDF 文本不是原生页面视觉；扫描页依赖 OCR |
| [jev-macos-loop](https://github.com/jcpsimmons/jev-macos-loop/tree/1aadc01ef262c3b01460909c5dd102f0d9616ba5) | OmniParser、OCR、AX；感知并行和动作后验证 | AGPL-3.0；只研究结构，不复制到本 MIT 项目 |
| [mobile-jev](https://github.com/droidrun/mobile-jev/tree/395fc222beac4f059f9a0beb337d114a2b066e99) | 手机 UI 元素、HTTP 连接复用、就绪/转场检查 | 截图展示不能证明模型看图 |
| [jev-libero / scene.py](https://github.com/Dimweaker/jev-libero/blob/9b8098a2faccfb561c95ffb3d117ad1688f1c30f/src/jev_libero/scene.py) | MuJoCo geometry/contact bindings，可读内部 mesh/pose | 比较时必须标注模拟器状态权限，不能等同纯视觉 VLA |
| [Jev-as-Policy](https://github.com/YuanKJing/Jev-as-Policy/tree/cac97e79845bf56a5b2ef0e6d4928238bf5caee7) | 机器人策略组合与来源文档 | 作为机器人应用入口；不凭标题认定端到端视觉策略 |
| [jev-drone / tactics.py](https://github.com/RomanSlack/jev-drone/blob/cbeb53ce4f17a06ea490ae43effcdad231143610/tactics.py#L91) | 相机感知转数值 scene；战术 Jev 异步、限频、过期，低层控制独立 | 时间戳和异步判断值得采用；飞行闭环结果不能迁移成通用视觉成绩 |
| [quackd / agent/jev.py](https://github.com/rokbenko/quackd/blob/6073829048b4c8152b7260d3c1d2af62709bfd02/quackd/agent/jev.py#L475) | 从各机器人适配器的允许动作中生成有限候选；完整系统另有视觉模型 | 将“视觉感知”和“便宜的 Jev 决策步骤”分别计时、分别评测 |
| [piper-astra-jev / perception.py](https://github.com/RobotKitAI/piper-astra-jev/blob/10d671e01d467475084033e7af9f6595886ab7b3/piper_llm/perception.py) | Grounding DINO 或 SAM3 + 深度；Jev 消费结构化状态；真实夹爪反馈 | 检测、定位和动作后触觉验证应保留独立证据，不能都归因给 Jev |
| [realtime-vision-decision-agent](https://github.com/Programalyst/realtime-vision-decision-agent/tree/392dcf1071ff517062d56d3909af7a51ef2f690f) | YOLO 检测游戏物体 → 候选动作预处理 → Jev | 外部 detector 的感知上限必须计入端到端成功率；Ultralytics 依赖另有许可 |
| [jev-reflex / context.py](https://github.com/nabendu82/jev-reflex/blob/66bb25d9f7baf00b1a7a0bd49f090df4c4f95087/backend/fusion/context.py) | MediaPipe + Vosk；去重的四秒历史，最多保留八条紧凑上下文 | 适合语音指代与手势同步；无检测到的许可证，不复制 |
| [jev-robot-control](https://github.com/openroboto-ai/jev-robot-control/tree/7a4ed8b72c3c17d7aa790678ed9660df67c10dd3)；[RoboJEV](https://github.com/lykycy123/RoboJEV/tree/aa82be09603032b0e196f2e4aaa7277d54eb7b42) | 结构化模拟器状态 → intent → XYZ/gripper，物理结果验证 | 有视频，但不是像素输入；纳入相邻工作并明确观测权限差别 |

目录还收录了会议转录、幻灯片语音、绘画、网页内容提取等候选，完整固定版本清单见机器可读目录。部分只完成 README/入口筛查，不能算作运行或完整代码审计。三个主要发现入口为 `Anil-matcha/awesome-jev-by-typesafe`、`yibie/awesome-jev`、`cobanov/awesome-jev`，各自版本也在目录中。

## 下一阶段值得做的研究

单纯套一个 VLM 或共享 KV 已经拥挤。更值得检验的假设是：在视频、语音、OCR、动作状态异步到达时，显式建模证据的时间、冲突与缺失，能否在相同计算/延迟预算下减少错误决策。

评测应包含新鲜/陈旧/缺失/互相矛盾的输入，按场景与录制序列分组切分；不能只随机切相邻帧。需要同时比较单模态、直接 VLM、caption→Jev、紧凑证据→Jev、训练决策头和不确定性路由。主指标应包括任务准确率、拒答覆盖率、校准误差、总延迟及资源消耗。必须用环境状态或独立标注验证结果，不能把模型自身的判断当真值。

本次 64 图 POPE 实验只建立可复现工程基线；尚不能支持顶会新颖性、跨模态泛化或控制成功率结论。
