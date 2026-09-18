# ACL Graph 代码走读学习方案（NPU 初学者版）

> 适用仓库：`ningweikang/rtp-llm`，分支 `pr-25`（PR #25 Ascend ACL Graph 适配）
>
> 核心思路：**先补 NPU/图执行的概念地基 → 用 CUDA Graph 做参照系 → 再按"入口 → Python 捕获 → C++ 执行"的调用链逐层深入**

---

## 阶段 0：概念准备（半天）

| 概念 | 必懂点 | 参考 |
|---|---|---|
| 昇腾 NPU 编程模型 | 异构计算：Host(CPU) / Device(NPU)，AI Core，DDR/HBM | 华为 CANN 文档 |
| ACL (Ascend Computing Language) | `aclrtStream`、`aclrtMemcpyAsync`、事件同步 | CANN 应用开发指南 |
| torch_npu | PyTorch 的昇腾后端，`torch.npu` API，`c10_npu` C++ 层 | torch_npu 文档 |
| **图执行 (Graph)** | **捕获(capture)→重放(replay)**：一次捕获算子序列，多次零开销重放，消除 Host 调度开销 | NVIDIA CUDA Graph 文档（对照理解） |
| FIA v2 | FlashInfer-Ascend 注意力实现（vllm-ascend 用的推理库） | vllm-ascend 源码 |

> 📌 **关键理解**：ACL Graph ≈ 昇腾版 CUDA Graph。代码注释明确指出本实现"对标 `CudaGraphRunner`，仅 decode 阶段"。代码里反复出现的 `graph_task_group` / `graph_task_update` 是 **vllm-ascend 的图捕获模式**，是本 fork 的独创。

---

## 阶段 1：调用链总览（1 小时）——先看"地图"再进"迷宫"

用 git 历史理解来龙去脉，再建立调用链：

```
启动服务
  └─ rtp_llm/cpp/models/PyWrappedModel.h (L287)
       └─ 构造 AscendGraphRunner (替代 CudaGraphRunner)
            ├─ initCapture() → 捕获 decode 图（按 batch size 分桶）
            │    └─ 调用 Python: forward() → ascend_decode.py
            │         └─ graph_task_group_begin/end 包住 FIA v2 .out()
            └─ forward() → replayGraph() → 重放已捕获的图
                 └─ graph_task_update_begin/end 动态更新 context_lens
```

**本阶段读**：`PyWrappedModel.h` 中 `AscendGraphRunner` 的创建与切换逻辑（搜索 `AscendGraphRunner`、`is_cuda_graph`）。

---

## 阶段 2：Python 侧捕获逻辑（半天）⭐ 建议从这里深入

**读 `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_decode.py`（316 行）**

阅读顺序与重点：

1. `AscendDecodeImpl`（L14）→ `prepare_cuda_graph()`（L101）：捕获入口，`update_graph_fia` 按 batch size 建图
2. `AscendDecodeAttnOp`（L141）→ 注释就是最好的文档：
   - **Capture 时**：`graph_task_group_begin/end` 包住 FIA v2 `.out()`，用预计算 workspace
   - **Replay 时**：`graph_task_update_begin/end` 动态更新 `context_lens`（长序列精度问题见 commit `80b8aa9e`）
3. `forward()`（L110）：`is_cuda_graph` 分支如何区分 eager/图执行

**配合读**：同目录的 `ascend_attn_params.py`（图模式的输入参数如何被固定/更新）、`ascend_kv_cache_write_op.py`（非连续 KV Cache 视图，commit `6a4b90f3` 的改动点）。

---

## 阶段 3：C++ 侧执行核心（1-2 天）⭐ 本项目重头戏

**读 `rtp_llm/cpp/ascend_graph/`（约 1200 行），按依赖顺序：**

```
① ascend_graph_device_shims.h/cc  (87+139 行)  设备抽象层
    → GraphStream(NPUStream)、makeGraphEvent、memcpy、捕获/重放原语
    → 理解 "非昇腾平台编译成 no-op stub" 的跨平台设计

② ascend_graph_utils.h  (128 行)  图实例与内存保持
    → AscendGraphInstance ≈ cuda_graph 的 GraphInstance（NPUGraph 版）
    → AscendGraphMemHold ≈ CaptureMemoryHold（捕获期 tensor 生命周期保持）

③ ascend_graph_runner.h/cc  (117+731 行)  核心 Runner  ← 重点
    → 构造：graph_params、禁用 prefill 图模式
    → initCapture() → captureDecode → captureDecodeOneBatchSize → captureOneGraphInstance
       （分桶捕获：不同 batch size 各建一张图）
    → forward() → replayAndSyncCheck → replayGraph → replayDecode
    → prepareInputs：输入如何从真实请求"搬运"到捕获时固定的内存
```

**阅读技巧**：全程对照 `rtp_llm/cpp/cuda_graph/` 里的 `CudaGraphRunner` —— 两者结构几乎一一对应，差异点就是 ACL Graph 的本质（`NPUGraph` vs `CUDAGraph`、`NPUStream` vs `CUDAStream`、memcpy 替代融合 kernel）。

---

## 阶段 4：串联验证（半天）

1. **跑通最小验证**：用 `benchmark/` 或 `example/` 起服务，`--enable_cuda_graph` 开图模式，观察日志 `Initialize AscendGraphRunner ...` 输出
2. **日志验证捕获/重放**：commit `d02ea8fa`/`6fa93f57` 加过 aclgraph 日志，用 `grep -rn "RTP_LLM_LOG" rtp_llm/cpp/ascend_graph/` 找埋点
3. **git 考古**（强烈推荐）：

   ```bash
   git log --oneline origin/main..pr-25 -- rtp_llm/cpp/ascend_graph/
   git show 6fa93f57 --stat   # feat: add aclgraph（第一版）
   git show 6a4b90f3 --stat   # 非连续 KV Cache 支持
   git show 80b8aa9e          # long-sequence 精度修复
   ```

   按 commit 顺序读，等于把 PR 的开发过程重走一遍。

---

## 阶段 5：拓展（选做）

- 对比 **CUDA Graph 版** 与 ACL Graph 版的性能/限制差异（为什么 ACL 只做 decode）
- 了解 **vllm-ascend** 的 `graph_task_group/update` 原始实现，理解本 fork 的适配取舍
- 阅读设计中引用的 `6-graph-mode/rtp-llm-aclgraph-adaptation-plan.md`（不在本仓库，若有内部文档权限可以找出来，代码注释多处引用它）

---

## 学习路线图小结

```
NPU 基础(CANN/torch_npu/图执行)
        ↓
调用链总览（PyWrappedModel.h → 谁在什么时候用什么 runner）
        ↓
Python 捕获层（ascend_decode.py：graph_task_group 模式）
        ↓
C++ 执行层（device_shims → utils → runner 三步走，对照 CUDA 版）
        ↓
运行验证 + git 考古串起全貌
```

## 代码文件速查表

| 文件 | 行数 | 阶段 | 角色 |
|---|---|---|---|
| `rtp_llm/cpp/ascend_graph/ascend_graph_runner.cc/h` | 731/117 | 3 | 核心 Runner：捕获/重放 |
| `rtp_llm/cpp/ascend_graph/ascend_graph_device_shims.cc/h` | 139/87 | 3 | 设备抽象层（stub 设计） |
| `rtp_llm/cpp/ascend_graph/ascend_graph_utils.h` | 128 | 3 | 图实例 + 内存保持 |
| `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_decode.py` | 316 | 2 | Python 捕获逻辑 |
| `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_attn_params.py` | 115 | 2 | 图模式输入参数 |
| `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_kv_cache_write_op.py` | 41 | 2 | KV Cache 写入（非连续视图） |
| `rtp_llm/cpp/models/PyWrappedModel.h` | — | 1 | Runner 选择入口 |
| `rtp_llm/cpp/cuda_graph/`（对照） | — | 3 | CUDA Graph 参照实现 |
