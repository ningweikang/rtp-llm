# ACL Graph 学习计划 · 第二阶段:从「读懂」到「吃透」

> 仓库:`ningweikang/rtp-llm`,分支 `pr-25`(HEAD = f089d978,2026-07-30,PR #23 合并后)
> 前置:阶段一(概念 + 调用链)与全仓库走读已完成,笔记见 `acl-graph-learning-plan.md`、
> `rtp-llm-source-walkthrough.md`(六条调用链)、`acl-graph-call-chain.md`、`notes/stage1.md`、
> `notes/acl-graph-implementation-principle.md`。
> 本阶段目标:把 PR #25 的核心机制从「知道流程」升级为「能解释设计取舍、能回答为什么」。

---

## 0. 现状审计(计划依据)

**代码与笔记无代差**:HEAD 停在 7-30,笔记写于 8 月初,引用行号可信;实际路径是
`rtp_llm/models_py/...`(不是 `rtp_llm/models/...`)。

**已覆盖**:
- 全仓库六条调用链(启动 → HTTP → Backend → Python/C++ Engine → gRPC → NormalEngine 调度)
- ACL Graph 三层架构与捕获/重放主流程、canRun 判据、分桶策略
- 动态 context_lens 的 graph_task_update 机制(principle 笔记第 5 节)

**未覆盖 / 值得深挖**(本计划全部模块都从这里出题):
1. 捕获期的**双路径差异**:warmup / capture / replay 三种状态下,Python 侧实际执行的是
   `ascend_decode.py` 的哪一段代码、输入张量为什么不同
2. `_forward_fia_graph` 里的**看似死代码**(L233-244 的再判断)与**共享 workspace 生命周期**
3. **KV Cache 写路径**在 graph 下的 device 端推导(手写 slot_mapping),与 `npu_scatter_pa_kv_cache`
   非连续视图的关系
4. **C++ shims 的跨平台 stub 设计**、内存保持(MemHold)与图实例的生命周期边界
5. 与 CUDA Graph 版**逐点差异**及原因(memcpy 替代融合 kernel 的代价)
6. PR 演进史:每个 commit 对应哪个真实缺陷(有 80b8aa9e 这类「修 bug」commit 可考古)

---

## 学习方法

- **问题驱动**:每个模块 = 一个必须跨层读代码才能回答的问题;答案写进自己的笔记。
- **三遍法**:先凭已有笔记复述 → 再对着代码逐行核实 → 最后不看代码画图/讲给别人听。
- **quiz 出口**:每个模块末尾有自测题,答不上来就回头重读,不要带着疑问进下一模块。
- **本机无 NPU**:所有验证以静态阅读 + git 考古 + 纯 Python 模拟为主;需要真机跑的
  实验单独标注,攒到有昇腾环境时统一做。

---

## M0 笔记对账与术语表(0.5 天)

**目标**:确认已有笔记没有记错;把散落各篇的术语统一成一张表。

- 通读 5 篇已有笔记,顺手修正与当前代码不符之处(路径、行号、函数名)。
- 产出:`notes/glossary.md` 术语表,至少包含:
  `NPUGraph / GraphCaptureMode::Relaxed / graph_task_group_begin·end / graph_task_update_begin·end /
  handle / workspace(attention workspace)/ lse / TND layout / sparse_mode=3 / block_table /
  slot_mapping / paged KV cache / hybrid KV cache / kv_cache_layer_to_group / D2D·H2H·pinned /
  capture_mem_hold_ / is_cuda_graph 标志`
- 每条术语写一行「是什么 + 出现位置」,禁止抄注释,用自己的话。

**Quiz**:不看代码,说出 canRun() 的五条判据分别防什么(提示:投机解码、prefill、hybrid KV cache 组数、越桶、开关)。

---

## M1 捕获的完整生命周期:一次 NPUGraph 是怎么「录」下来的(1 天)

**核心问题**:从 `PyWrappedModel` 构造到所有桶捕获完成,Python 侧同一份 `forward()`
为什么会被以三种不同状态执行,分别走 `ascend_decode.py` 的哪条分支?

**必读**(按顺序):
1. `rtp_llm/cpp/models/PyWrappedModel.h` L190-313(构造与 runner 创建)、L404-437(initCapture)
2. `rtp_llm/cpp/ascend_graph/ascend_graph_runner.cc`
   - `captureDecode()`(分桶、大→小顺序、每桶独立 `mem_hold_`)
   - `captureOneGraphInstance()`(warmup ×2 → `graphCaptureBegin(Relaxed)` → python forward → `graphCaptureEnd`)
3. `rtp_llm/cpp/ascend_graph/ascend_graph_utils.h`(`AscendGraphInstance`、`AscendGraphMemHold` 到底 hold 了什么)
4. `rtp_llm/cpp/ascend_graph/ascend_graph_device_shims.cc`(graphCaptureBegin/End → torch_npu 哪个 API)
5. `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_decode.py`
   - L101 `prepare_cuda_graph`、L110 `forward`、L177 `AscendDecodeAttnOp.prepare`(host/device 分叉)
   - L205 的分支条件与 L209 `_forward_fia_graph`

**思考题**:
- warmup 时 `torch.npu.is_current_stream_capturing() == False`,为什么 block_table/context_lens
  已经走 device 版(`_d` 张量)?(提示:`is_cuda_graph` 标志在 `prepare()` 里决定了读 host 还是 device)
- 图捕获的铁律是 capture 期间不允许惰性初始化 / 隐式同步——warmup ×2 是在为哪两件事热身?
- `_forward_fia_graph` L233-244 的「非捕获再判断」分支,从当前调用路径看**永远不可达**——
  它是防御还是伏笔?为什么作者要写它?(允许结论是「防御性代码」,但要说出理由)
- `capture_mem_hold_`(最大桶共享)与每桶 `graph_instances_[bs].mem_hold_` 各自负责什么?
  捕获结束后,哪些 tensor 必须活着?谁保证它们活着?

**产出**:一页「捕获时序图」:三态(warmup/capture/replay)各自执行哪段代码、输入从哪来、
  期间发生几次 device 同步。

**Quiz**:如果把 warmup 次数从 2 改成 0,预期会观察到什么故障?为什么不能只 warmup 1 次?
(提示:有些 kernel 首次调用会编译/分配,且可能与后续调用的 workspace 需求不同)

---

## M2 重放与动态更新:图是死的,数据是活的(1-1.5 天)⭐ 最难

**核心问题**:图一旦捕获,地址全部固定;decode 每步的 sequence length 都在变——
「固定图 + 动态输入」这对矛盾是怎么解开的?

**必读**:
1. `ascend_graph_runner.cc`:`forward()` → `prepareInputs()` 全文
   - 每个 `copyTensorSlice` 拷的是什么(D2D vs H2H pinned)、为什么 padding 区要清零、
     真实 bs < 桶 bs 时多出来的行是什么语义
   - `forward_event_.synchronize()` 在 prepareInputs 开头:它在防什么竞态?
2. `ascend_decode.py` L101-108(`prepare_cuda_graph`:从 `sequence_lengths` 算出 `ctx_list`)、
   L270-295(`update_graph_fia`)
3. `git show 80b8aa9e`(strong refs 修长序列精度)——先看 diff,再回答:
   **修复前为什么长序列会算错?** 提示:update 流上重放 `.out()` 时,`q/k_cache/...` 这些
   tensor 若只剩 handle 侧弱引用会发生什么;`self._graph_refs` 里存的是什么。

**思考题**:
- `update_graph_fia` 里 `ctx_list = (seq_lens+1).tolist()` 把 tensor 转成了 Python list——
  为什么 graph 更新路径要的是 list 而不是 device tensor?这个转换发生在 host 还是 device?
  (对照 L234 eager 路径用 `torch.arange` device tensor,说出两种传参方式各自的同步开销)
- `graph_task_update_begin/end` 包住一次新的 `.out()` 调用:这到底是在「改已捕获图的参数」,
  还是在「用新参数把 task 重新登记进 handle」?结合 vllm-ascend 的语义给出你的结论。
- 画出**连续两次 decode forward** 的时序图:prepareInputs(写 buffer)→ replay → 读 hidden_states。
  第二次的 prepareInputs 为什么必须等第一次的 `forward_event_`?如果不等,哪个 buffer 会被踩?

**产出**:`notes/replay-and-update.md` + 时序图(手画或 mermaid)。

**Quiz**:
1. 桶 bs=8、实际 bs=5 时,第 6-8 行的输入是什么?输出会怎么处理?(提示:padding 清零 + 输出只取前 5 行)
2. 长序列精度 bug 的根因用一句话怎么讲?

---

## M3 eager / graph 双路径差异总账 + KV Cache 写路径(1 天)

**核心问题**:同一份注意力实现,为什么 eager 和 graph 的输入管线长得完全不一样?

**必读**:
- `ascend_decode.py` 全文(已熟,这次带着「差异」眼镜看)
- `ascend_attn_params.py`(host 版 `compute_ascend_attn_params` vs device 版推导)
- `ascend_kv_cache_write_op.py` + `ascend_prefill.py`(prefill 用同一套写缓存 op 吗?)
- `rtp_llm/models_py/modules/factory/attention/` 下的 `common.py`(`create_write_cache_store_impl`、`apply_write_cache_store`)
- `git show 6a4b90f3 --stat`、`git show faece0e7`、`git show 88740c4d`(非连续 KV cache 视图三部曲)

**思考题**:
- eager 路径 `block_table` 来自 `kv_cache_kernel_block_id_host`,`context_lens = sequence_lengths + 1`
  (CPU),每次 forward 现算现拷;graph 路径用 `_d` 结尾的 device 张量。**为什么 graph 不能复用 host 输入?**
- `_update_rope_kv_write_params_device`(L72-95)手写 block_index/block_offset 算 slot_mapping:
  对照 eager 用的 `compute_ascend_attn_params`,两者算法等价吗?graph 版为什么要手写而不用现成函数?
  (提示:capture 期间不能有 host→device 同步点 / 数据依赖的 host 计算)
- `kv_cache.kv_cache_base[:, 0].reshape(B, page_size, -1)` 是非连续视图;FIA 为什么能直接吃它?
  `npu_scatter_pa_kv_cache` 在这里的角色是什么?(配合三个 commit 读)
- `AscendDecodeImpl.forward` 里 `need_rope_kv_cache` 两条支路:哪条会把 RoPE 后的 K/V 写进 cache?
  graph 模式下 RoPE 的 `positions_d` 从哪来?(L72-74:`sequence_lengths_plus_1_d - 1`)

**产出**:一张「eager vs graph」双路径对照表(输入来源 / 同步点 / workspace / 输出存放),
  一张 KV Cache 写路径一图流。

**Quiz**:graph 模式下新增一个 token,它的 RoPE position 和 slot_mapping 分别是谁、在哪一步、用什么数据算出来的?

---

## M4 C++ 侧:shims、内存与跨平台设计(1 天)

**核心问题**:为什么 device 操作要包一层 shims?「非昇腾平台编译成 no-op stub」是怎么做到的?

**必读**:
- `ascend_graph_device_shims.h/cc` 全文:每个函数对应 torch_npu / ACL 的哪个 API
  (GraphStream、graphEvent、memcpy、CaptureBegin/End、Replay、Pool)
- `ascend_graph_utils.h` 全文:实例与 MemHold 的析构顺序、GraphPoolHandle 的生命周期
- `ascend_graph/BUILD` 与 `rtp_llm/cpp/models/BUILD`:条件依赖怎么写(USING_ASCEND 宏从哪来)
- 对照:`rtp_llm/cpp/cuda_graph/` 全套(CudaGraphRunner、CaptureMemoryHold、GraphInstance)
- `ascend_graph_runner.cc` 的 `replayAndSyncCheck`(捕获后立即重放验证,验证什么、失败怎么办)

**思考题**:
- 列出 Ascend 版与 CUDA 版 **5 处以上**差异,每个差异回答:「为什么 Ascend 必须这样?」
  (例:为什么用 aclrtMemcpyAsync/tensor.copy_ 而不是 CUDA 版的自定义融合拷贝 kernel?
  这会让每次重放多几次 D2D 拷贝——这是否抵消了部分图收益?)
- `AscendGraphRunner.h` 里保留了 `is_target_verify_`、`sp_steps_`、Bert embedding 字段,
  却删掉了 prefill 相关字段——从字段差异反推 CUDA 版支持而 Ascend 版不支持的场景。
- stub 化设计:如果在非 Ascend 平台误调用 shims,行为是什么?为什么这种「静默失败」是可接受的?

**产出**:`notes/ascend-vs-cuda-graph.md`(差异表 + 每条的取舍理由)。

**Quiz**:为什么 `graph_runner_` 的静态类型是 `GraphBase*` 而不是两个 runner 各自类型?
这个抽象在 canRun/forward 之外还强制了哪些一致性?

---

## M5 git 考古:把 PR 25 的开发过程重走一遍(0.5-1 天)

**必读**(按时间顺序,每个 commit 先 `git show --stat` 再读关键 diff):
```
6fa93f57   feat: add aclgraph(第一版,看它抄了 CudaGraphRunner 的哪些结构)
d02ea8fa   add aclgraph logs(日志埋点在哪,为什么这些点重要)
6a4b90f3   feat: update aclgraph and ascend decode/kv_cache for fia discontinuous
faece0e7   AscendAttn: npu_scatter_pa_kv_cache scatter directly into non-contiguous views
88740c4d   AscendAttn: use npu_scatter_pa_kv_cache with non-contiguous kv cache views
80b8aa9e   fix: use strong refs for graph_task_update tensors to fix long-sequence precision
7c92cd04   Update ascend decode and kv cache write op for aclgraph
3d2cc853   Inline FIA v2 workspace allocation in _forward_fia_graph(HEAD 前最后一个改动)
```

**产出**:`notes/pr25-evolution.md`,格式:
| commit | 解决的问题/现象 | 改法 | 为什么这样改(你的推断) | 遗留疑问 |

**Quiz**:3d2cc853 把 workspace 分配内联进 `_forward_fia_graph` 并缓存成类级 `_shared_workspace`——
  它为什么敢缓存?如果两个桶先后捕获,workspace 大小以哪个为准?(提示:捕获顺序大→小)

---

## M6 外延:回到源头看 vllm-ascend 与 FIA v2(选做,1-2 天)

- 找 vllm-ascend 源码,读 `graph_task_group_begin/end`、`graph_task_update_begin/end` 的原始实现,
  确认本 fork 的用法与上游一致、改了什么(本 fork 的 `AscendDecodeAttnOp` 是简化还是改写?)
- 查 CANN/torch_npu 文档:`npu_fused_infer_attention_score_v2` 的 `out=` + `workspace=` 约定;
  `sparse_mode=3`、`input_layout="TND"` 的含义;`NPUGraph` 捕获的 Relaxed 模式限制
  (什么算子不允许出现在 capture 里)
- 对照 xLLM 的 aclgraph 说明(代码注释提到 xLLM 同样 decode-only),了解业界做法

**产出**:`notes/fia-v2-and-upstream.md`。

---

## 需要真机(NPU)才能做的实验(攒着,标注 ⏳)

1. 起服务 + `--enable_cuda_graph`,观察日志 `Initialize AscendGraphRunner ...` 与分桶捕获顺序
2. 用真实长序列跑一次,验证 80b8aa9e 修复前后数值差异
3. 关闭 graph(eager)与开启 graph 的 decode 时延对比,量化「重放省了多少、memcpy 加了多少」
4. 观察 `_shared_workspace` 在多大 batch size 下会被多个桶复用、显存水位
   (对应 `notes/stage1.md` 提到的 A10 无直接关系——昇腾上以 `npu-smi` 看显存)

---

## 路线图小结

```
M0 对账 + 术语表(0.5d)
M1 捕获生命周期:三态执行 → 哪段代码(1d)
M2 重放 + graph_task_update 动态更新(1-1.5d)← 最难,值得砸时间
M3 eager/graph 双路径 + KV cache 写路径(1d)
M4 C++ shims/内存/跨平台,对照 CUDA 版(1d)
M5 git 考古 PR 演进(0.5-1d)
M6 vllm-ascend / FIA v2 外延(选做)
--------------------------------------------------
合计约 5-7 个学习日(每天 2-3 小时则 2-3 周)
```

每个模块完成标准:笔记落盘(notes/ 下新文件)+ quiz 全部能独立回答 + 能不看代码画出该模块的图。

---

## 第二阶段代码速查(行号已按当前 HEAD 核实)

| 位置 | 模块 | 关键点 |
|---|---|---|
| `rtp_llm/cpp/ascend_graph/ascend_graph_runner.h` | M1/M2/M4 | 字段即设计:无 prefill、有 hybrid KV 组映射 |
| `.../ascend_graph_runner.cc` initCapture/captureDecode/captureOneGraphInstance | M1 | 分桶、大→小、warmup×2 |
| `.../ascend_graph_runner.cc` forward/prepareInputs/replayDecode | M2 | forward_event_ 同步、三类拷贝 |
| `.../ascend_graph_device_shims.*` | M4 | stub 化跨平台设计 |
| `.../ascend_graph_utils.h` | M1/M4 | AscendGraphInstance / MemHold |
| `rtp_llm/models_py/.../ascend_impl/ascend_decode.py` L101/110/177/198/205/209/270/297 | M1-M3 | 三态执行 + 双路径 + update |
| `.../ascend_attn_params.py` | M3 | host 版参数计算(被 device 版替代的那条路) |
| `.../ascend_kv_cache_write_op.py` | M3 | 非连续 KV 视图写入 |
| `rtp_llm/cpp/cuda_graph/`(对照) | M4 | CUDA 参照实现 |
| `rtp_llm/cpp/models/PyWrappedModel.h` | M1 | runner 选择与 initCapture 触发点 |
