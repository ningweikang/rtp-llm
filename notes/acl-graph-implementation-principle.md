# RTP-LLM ACL Graph 实现原理（修正版）

> 基于 PR #25（branch `pr-25`）源码第一手解读。
> 工作流：pi 解读 → codex 独立审查 → 本版已吸收 codex 的核对意见（31 条逐点核对中 3 处实质性错误 + 若干表述修正 + 8 个风险点）。
> 源码范围：`rtp_llm/cpp/ascend_graph/`（1202 行）+ `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_decode.py`（316 行）。

---

## 1. 总体定位（修正版）

PR #25 给 RTP-LLM 的昇腾 NPU 路径增加 **decode 阶段的图执行（Graph）模式**：对标 `CudaGraphRunner`，用 `c10_npu::NPUGraph` 把一次 decode 的主体算子序列**分桶捕获成图**，之后重放以**降低被捕获算子序列的逐算子调度开销**。

⚠️ **注意**：不是"零调度开销重放"。每次重放前仍有大量 C++/Python 调度：

- `forward_event_.synchronize()` 等上次 forward 完成
- 逐 tensor 拷贝/清零（`copyTensorSlice`）
- Python `prepare_cuda_graph()` → FIA graph-task update（含 host 数据转换 `.tolist()` 和 `us.synchronize()`）
- 重放后 `slice().clone()` 读输出

图模式省的是**被捕获主体算子序列**的下发开销，不是整个 forward。

## 2. 三层架构（修正版）

```
┌─────────────────────────────────────────────────┐
│ ① device_shims（226 行）— C++ 设备 API 收敛层      │
│    收敛 AscendGraphRunner 的 C++ stream/event/     │
│    memcpy/NPUGraph 调用；非昇腾平台编译成 stub     │
├─────────────────────────────────────────────────┤
│ ② utils（128 行）— 图实例 + 捕获期内存保持          │
│    AscendGraphMemHold / AscendGraphInstance /     │
│    AscendGraphStreamLife / AscendGraphCaptureGuard │
├─────────────────────────────────────────────────┤
│ ③ AscendGraphRunner（848 行）— 捕获/重放编排        │
│    分桶捕获 decode 图 → 重放时搬运输入 → replay      │
└─────────────────────────────────────────────────┘
        │ pybind 调 Python forward
┌─────────────────────────────────────────────────┐
│ ④ ascend_decode.py（316 行）— FIA v2 注意力封装     │
│    graph_task_group_begin/end 捕获                 │
│    graph_task_update_begin/end 运行期更新参数       │
└─────────────────────────────────────────────────┘
```

### ① device_shims：C++ 设备 API 收敛层

- 收敛 AscendGraphRunner 使用的 C++ stream/event/memcpy/NPUGraph 捕获原语（`NPUStream`、`NPUGraph::capture_begin/end/replay`、`aclrtMemcpyAsync`、`aclrtSynchronizeDevice`）。
- **跨平台 stub 设计**：`#if USING_ASCEND` 下实现真实逻辑；非昇腾平台编译成 no-op（`GraphStream` 为 dummy struct，各函数空实现，capture flag 固定 false）。因此 `ascend_graph` 库可被无条件依赖，不破坏 CUDA/ROCm 路径。
- ⚠️ 修正：**不是"整个实现的唯一底层 API 接触层"**。Python 侧（ascend_decode.py）直接大量调用 `torch.npu` / `torch_npu` API（FIA、stream capture、task group/update）；`ascend_graph_runner.cc` 也直接包含 torch_npu 头（正文主要经 shim 调用）。
- `GraphPoolHandle` 是空结构：`graphCaptureBegin` 固定传 `{0,0}` 让 NPUGraph 自动创建 mempool，**不支持 CUDA 式显式 mempool 共享**。
- `graphMemGetInfo()` / `graphMemcpyAsync()` / `graphStreamSynchronize()` 在 runner 主流程**未使用**（shim 是面向未来的 API 面）。

### ② utils：图实例与内存保持

- `AscendGraphMemHold`：按值保存整套 `PyModelInputs`（所有 attention/bert 字段）+ Python attention 对象 + 输出 tensor。作用是**维持捕获期 tensor 的生命周期和地址**——图重放要求所有绑定的 tensor 地址永远不变。不仅是 tensor 生命周期容器，还携带 Python 引用。
- `AscendGraphInstance`：按 batch size 组织的一个图实例 = `NPUGraph` + `mem_hold_`。
- `AscendGraphStreamLife`：RAII，构造时切到捕获流，析构恢复原流。
- `AscendGraphCaptureGuard`：RAII，置/清"捕获中"全局标志。

### ③ AscendGraphRunner：捕获/重放编排

继承 `GraphBase`（复用 CUDA Graph 的基础设施），成员与 CudaGraphRunner 一一对应。核心字段：

- `graph_instances_`：`unordered_map<int, AscendGraphInstance>`，batch size → 图实例
- `capture_range_` / `decode_capture_batch_sizes_`：分桶列表
- `capture_mem_hold_`：最大尺寸的共享捕获内存（各桶从中 slice）
- `capture_stream_`、`forward_event_`：捕获专用流、forward 完成事件

### ④ ascend_decode.py：FIA v2 封装

- `AscendDecodeImpl`（FMHAImplBase 子类）：decode 入口，组合 FIA 注意力 + RoPE + KV cache 写入 + 参数计算。
- `AscendDecodeAttnOp`：FIA v2 算子封装。图模式用 vllm-ascend 的 `graph_task_group` / `graph_task_update` 模式。
- `AscendAttnParams`：图模式输入参数（positions、slot_mapping）。

## 3. 捕获机制（修正版）

### 3.1 分桶

`getDecodeBatchSizesToCapture()`：

- **默认**：`{1, 8, 16, 24, 32}`，48 起每 16 递增，最后补 `max_bs_`（若未包含）。
- **自定义**：若 Python 提供 `decode_capture_batch_sizes_`，直接排序采用——**不强制补 max_bs_、不去重、不校验非法值/超 max_bs_**。

每个桶一张独立图。**每个桶的输入不是独立分配**：各桶 `PyModelInputs` 是最大公共 buffer 的 `[0:bucket_bs]` slice（`prepareCaptureInputs`），共享同一 storage，地址对应这些 slice。

### 3.2 捕获期 tensor 分配

`initCaptureAttentionInputs` 一次性分配（**列举修正版**）：

- `input_ids`、`input_hiddens`（device）
- `input_lengths`（pinned host）+ `input_lengths_d`（device）
- `sequence_lengths`（pinned host，捕获时填 `seq_size_per_block_ - 1`）+ `sequence_lengths_plus_1_d`（device，填 `seq_size_per_block_`）
- **两套** block 表：kernel 版与非 kernel 版，各 device/host 双份
- per-group block 表（仅 `kv_cache_group_num_ > 1` 时）
- `prefix_lengths(_d)`、`padding_offset`、`cu_seqlens(_host)`、`cu_kv_seqlens`、`decode_cu_seqlens_d`
- BERT embedding 辅助输入（combo_position_ids、combo_tokens_type_ids、position_encoding、token_type_embedding、input_embedding_scalar）

输出 buffer 固定按 `{max_num_token_, hidden_size_}` + `model_data_type_` 分配——**不是由 warmup 输出推导 dtype**。

### 3.3 warmup（修正版：次数分层）

总次数 = **1 + 2 × 桶数** 次非捕获 forward，另加每桶 1 次捕获 forward：

1. `initCapture()` 对最大公共输入 warmup **1 次**（runner.cc:407-412）：触发 lazy init。
2. 每桶 `captureOneGraphInstance` 捕获前再 warmup **2 次**（runner.cc:470-479）。

⚠️ 修正：源码注释写 "settle output dtype / kernel lazy init"，但**实际代码没有从 warmup 返回值推导 dtype**——主要是 lazy init。

### 3.4 逐桶捕获（捕获顺序：从大到小 ✅）

`captureDecode()` 从 `capture_range_` 末尾递减循环（**从大到小**捕获，意图是让 mempool 高水位先建立——注意这是设计意图，各图实际各自 `capture_begin({0,0})` 自动建 pool，**并非共享同一显式 mempool**）。

`captureOneGraphInstance` 单桶流程：

1. 桶级 warmup × 2
2. `graphDeviceSynchronize()`：保证捕获时无在途算子
3. `AscendGraphStreamLife`：切到捕获专用流
4. `graphCaptureBegin(graph, {0,0}, Relaxed)`：开始捕获（Relaxed 模式）
5. `AscendGraphCaptureGuard`：置"捕获中"标志
6. 调 Python `forward()`（此时 stream 处于 capture，Python 走 `_forward_fia_graph` 图捕获路径）
7. 输出 `copy_` 到持久 buffer（重放时有稳定地址）
8. `graphCaptureEnd(graph)`
9. 流恢复（RAII）
10. `replayAndSyncCheck`：replay 一次 + `graphDeviceSynchronize()` 验证能跑通

⚠️ 风险（codex 发现）：`graphCaptureBegin` 之后若 Python forward 或 cast 抛异常，直接 rethrow，**`graphCaptureEnd` 不执行**——可能留下未正常结束的 NPUGraph capture/runtime 状态（RAII 只恢复 flag 和 stream）。

⚠️ 验证局限："验证"只确认 replay 能执行并同步完成，**没有与 eager 输出做数值比对**。

## 4. 重放机制（修正版）

### 4.1 canRun 判据（修正版：分两条路径）

**普通 decode 路径**：

1. `enable_graph_` 开
2. 非 prefill（decode-only；prefill 输入 → false → eager）
3. hybrid KV cache 组数匹配（仅当输入 device-by-group 非空时检查；⚠️ 不检查 host-by-group 数量）
4. `tryGetRealGraphDecodeBatchSize`：batch size **向上对齐**到最近桶（`lower_bound` 返回第一个 `>= current_batch_size` 的桶，如 batch=9 用 16 的图）；超过最大桶 → false

**⚠️ target-verify 特殊路径（我初版解读遗漏，codex 抓到）**：

```cpp
if (is_target_verify_) {
    if (inputs.attention_inputs.is_target_verify) {
        return tryGetRealGraphDecodeBatchSize(inputs, state);  // 提前返回
    }
    return false;
}
```

该分支**绕过 `enable_graph_`、`is_prefill`、hybrid 组数检查**——只在 target-verify 模式下使用（投机解码的 verify 阶段），是一个独立旁路。

### 4.2 prepareInputs：把动态数据灌进静态地址

1. **`forward_event_.synchronize()`**：等上次 forward 完成（host 侧 event 同步，防覆盖还没读完的持久 buffer）——注意这是较重的同步。
2. **清零 block 表**：kernel/非 kernel、device/host、per-group 全部 `fill_(0)`——防陈旧 KV block ID 污染下一批。
3. **`copyTensorSlice` 数据搬运**：
   - 直接调 PyTorch `Tensor::copy_`（`non_blocking=true`），**不是直接调 `aclrtMemcpyAsync`**（shim 的 `graphMemcpyAsync()` 在 runner 主流程未使用；`copy_` 在 NPU backend 可能下沉到 ACL memcpy，但源码没直接调用）。
   - 维度处理：只处理前导 size-1 维（`squeeze(0)`）、1 维分支、2 维按前两维 `min` slice。**仅可靠覆盖当前使用的一维/二维输入**；标量 tensor（dim==0）会因 `s.size(0)` 报错，3 维以上只裁剪前两维（剩余维度必须自然匹配）。不是通用 N 维 slice-copy helper。
4. **padding 区域清零**（⚠️ 边界语义修正）：
   - 实际清零范围是 `[current_batch_size, graph_bucket_bs)`（PyTorch 把越界 end 截到该桶 slice 的实际长度），不是真正统一清到 `max_bs_`。
   - 清零的字段：input_lengths(_d)、sequence_lengths、sequence_lengths_plus_1_d、decode_cu_seqlens_d、prefix_lengths(_d)、所有 block 表。
   - ⚠️ **`input_ids`/`input_hiddens` 的桶内 padding 区域不清零**——decode 每请求 1 token 时旧数据可能保留。
5. **计算 `sequence_lengths_plus_1_d`**：engine 不填充该字段，由 sequence_lengths +1 现场算。
6. **调 Python `attn_pyobj.prepare_cuda_graph(attention_inputs)`**：更新 FIA 图参数（见 §5）。

### 4.3 重放与输出

1. `replayDecode(graph_bs)` → `NPUGraph::replay()`：重放整张图。
2. 从持久 buffer `slice(0, 0, seq_len_sum).clone()` 读输出（普通 decode 中 `seq_len_sum = batch_size`）。
3. `forward_event_.record(current_stream)`：记录完成事件，供下一轮 synchronize。

⚠️ 降级语义修正：fallback eager 仅发生在 **canRun 返回 false 的前置判定**（未启用/ prefill/组数不匹配/超桶/`capture_range_` 空）。**捕获、拷贝、update、replay 期间的异常是抛出，不会自动回退 eager**；`canRun` 本身也不检查 `graph_instances_` 中是否真的存在所选 key。

## 5. 动态 context_lens 处理（vllm-ascend 图捕获模式，精确版）

图执行拓扑固定，但每个请求的 KV 长度是动态的。方案是 vllm-ascend 的 `graph_task_group` / `graph_task_update`：

**捕获阶段**（`_forward_fia_graph`，仅当 `torch.npu.is_current_stream_capturing()`）：

1. 预计算 workspace 一次：`npu_fused_infer_attention_score_v2_get_max_workspace(...)` → 转成 `_shared_workspace`（class 级全局，**首次按当时配置分配，之后不重算**）。
2. `graph_task_group_begin(stream)`
3. FIA v2 `.out(workspace=_shared_workspace, out=[out, lse])`——注意 `actual_seq_kvlen` 是**固定值 `[page_size] × batch_size`**（Python list，不是 tensor）。
4. `graph_task_group_end(stream)` 拿到 handle
5. 记录 `_graph_refs`（q/k/v/block_table/atten_mask/out/lse 引用）+ handle

**运行期重放前**（`update_graph_fia`，由 `prepare_cuda_graph` 触发）：

1. host `sequence_lengths`（已被 C++ 搬入持久 buffer 并清零 padding）`+1` → `.tolist()` → `ctx_list`（⚠️ padding 项最终是 **1**，不是 0）
2. 在 class 级共享 update stream（`_shared_update_stream`）上：
   - `graph_task_update_begin(us, handle)`
   - 重新提交 FIA `.out()`，传真实 `actual_seq_kvlen=ctx_list`
   - `graph_task_update_end(us)`
3. `us.synchronize()`（同步操作）

实现效果："只更新 FIA 子任务的动态参数（actual_seq_kvlen），整图拓扑不变"。⚠️ 注意：这个 update 在独立共享 stream 上执行 + 强制同步，会引入 host 等待和调度成本。

⚠️ 风险（codex 发现）：
- `_graph_handles` / `_graph_refs` **只追加不清理**——若同一 attention op 重新捕获，旧 handle 残留并会在后续全部被 update。
- `update_graph_fia` 的 `batch_size` 来自持久输入 `sequence_lengths.size(0)`（= 选中图桶大小，不是真实 batch size）；padding 项依赖 C++ 清零后得到 ctx_list 中的 1。
- `_forward_fia_graph` 内部还有一个"非捕获时 eager FIA"分支（ascend_decode.py:233-244），在当前调用链中近似不可达（防御性/冗余逻辑）。

## 6. 与 CUDA Graph 版差异

| 维度 | CudaGraphRunner | AscendGraphRunner |
|---|---|---|
| 图对象 | `at::cuda::CUDAGraph` | `c10_npu::NPUGraph` |
| 流 | CUDAStream | NPUStream |
| 输入搬运 | 自定义融合 kernel | 逐 tensor `copy_`（未用融合 kernel；显式 `aclrtMemcpyAsync` shim 未被该路径调用） |
| prefill 图模式 | 支持 | **不支持**：构造时 `is_prefill_cuda_graph_mode=true` 直接 throw |
| mempool | 显式共享 | NPUGraph 自动创建（`{0,0}`），`GraphPoolHandle` 空壳 |
| 动态 KV 长度 | （CUDA 版机制） | FIA graph_task_update |

补充：普通运行时收到 `is_prefill` 输入 → `canRun()` false → eager（构造 throw 只是配置级约束；target-verify 分支可能绕过运行时 prefill 检查，见 §4.1）。

## 7. 实现细节与风险清单（汇总）

### 设计细节

- 捕获从大到小（mempool 高水位意图）
- 捕获前 `graphDeviceSynchronize`（捕获拓扑干净）
- `sequence_lengths` 捕获值 = `seq_size_per_block_ - 1`（page_size 当捕获 context_len，保持 FIA workspace 较小，避免大 ctx 时同步成本）
- decode-only：ACL Graph 只做 decode（prefill eager）
- 降级路径：canRun false → eager（仅前置判定）

### 风险/待改进点（codex 审查发现，学习记录用）

1. target-verify 旁路绕过 enable_graph/prefill/hybrid 检查（最值得注意）
2. 捕获异常不执行 `graphCaptureEnd`，可能留下未结束的捕获状态
3. FIA workspace class 级全局共享，换配置/桶不重算（容量风险）
4. `_graph_handles`/`_graph_refs` 只追加不清理
5. 捕获 flag 是进程级无锁 bool（非 thread-local；Python 实际用 `torch.npu.is_current_stream_capturing()` 判断，两套机制未闭环）
6. 捕获验证无 eager 数值比对
7. 自定义 bucket 无去重/范围/非空校验
8. `enable_graph_debug_mode_` 保存但无行为
9. hybrid cache：canRun 只校验 device-by-group，host-by-group 不对称校验
10. padding 语义：input_ids/input_hiddens 桶内 padding 不清零；padding 的 actual_seq_kvlen=1 而非 0
11. `copyTensorSlice` 非通用 N 维拷贝（标量会崩、3 维只裁两维）
12. `prepare_cuda_graph` 每轮 `.tolist()` host 数据转换开销

## 8. 源码位置速查

| 文件 | 行数 | 关键位置 |
|---|---|---|
| `ascend_graph_device_shims.h/cc` | 87/139 | stream/event/memcpy 封装、stub 分支、capture flag |
| `ascend_graph_utils.h` | 128 | AscendGraphMemHold、AscendGraphInstance、RAII guard |
| `ascend_graph_runner.h` | 117 | 字段声明、GraphBase 继承 |
| `ascend_graph_runner.cc` | 731 | canRun(:172-206)、initCapture(:385-419)、captureDecode(:429-463)、captureOneGraphInstance(:466-513)、prepareInputs(:561-705)、forward(:707-723) |
| `ascend_decode.py` | 316 | prepare_cuda_graph(:101-108)、_forward_fia_graph(:219-268)、update_graph_fia(:270-295)、_forward_fia(:297-316) |

---

*生成方式：pi 第一手读码 → codex 独立审查（逐条核对 + 行号依据）→ 修正落盘。审查与解读均在对话内完成，本文件为明确指令下生成的最终版。*
