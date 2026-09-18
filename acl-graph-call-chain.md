# ACL Graph 调用链总结

> 适用分支：`pr-25`
>
> 核心实现：Ascend `c10_npu::NPUGraph`，当前主要用于 decode 阶段。

## 1. 总体流程

ACL Graph 分为两个阶段：

```text
初始化阶段：创建 Runner → 分配固定输入 → 按 batch size 捕获 Graph
推理阶段：canRun() → prepareInputs() → replay Graph → 读取输出
```

完整调用链：

```text
NormalExecutor
  └─ PyWrappedModel::forward()
       ├─ canRun() == false
       │    └─ Python model.forward(..., eager)
       │         └─ AscendDecodeImpl
       │              └─ FIA v2 eager
       │
       └─ canRun() == true
            └─ AscendGraphRunner::forward()
                 ├─ prepareInputs()
                 │    ├─ 更新固定输入 buffer
                 │    └─ prepare_cuda_graph()
                 │         └─ update_graph_fia()
                 │              └─ graph_task_update_begin/end
                 ├─ replayDecode()
                 │    └─ NPUGraph::replay()
                 └─ 读取 hidden_states
```

---

## 2. Executor 入口
### 2.1 创建 PyWrappedModel
**文件：** `rtp_llm/cpp/normal_engine/NormalExecutor.cc:99`
```cpp
model_.reset(new PyWrappedModel(model_init_params, params.py_model));
```
PyWrappedModel 的定位是：C++ 推理引擎与 Python 模型定义之间的桥接层（Adapter/Facade）。  
### 2.2 发起模型推理

**文件：** `rtp_llm/cpp/normal_engine/NormalExecutor.cc:169`

```cpp
model_output = std::move(model_->forward(model_input));
```

调用进入：

```text
NormalExecutor::process()
    └─ PyWrappedModel::forward()
```

---

## 3. PyWrappedModel 是干什么的

> **定位**：C++ 推理引擎与 Python 模型定义之间的桥接层（Adapter/Facade）。
>
> C++ 引擎（Executor）只认识 `GptModelInputs`，Python 模型只认识 `PyModelInputs`。`PyWrappedModel` 负责翻译、持有 Python 模型对象、并管理图模式（ACL/CUDA Graph）的创建与调用。
>
> 类定义：`rtp_llm/cpp/models/PyWrappedModel.h:38`

### 4.1 核心职责

#### ① 持有并初始化 Python 模型

```cpp
py::object py_model_;   // 持有 Python 模型对象（pybind 句柄）
```

- 构造函数里 `py_model_ = py_instance;`，调用 `py_model_.attr("initialize")(init_resources)` 完成 Python 侧初始化（绑定 KV Cache 等）
- 析构时 `py_instance_.release()` 释放引用

#### ② 输入/输出格式转换（C++ 世界 ↔ Python 世界）

`forward()` 里做转换（`PyWrappedModel.cc:443-489`）：

| 转换 | 函数 |
|---|---|
| `GptModelInputs` → `PyAttentionInputs` | `buildPyAttentionInputs()` |
| 权重/KV → `BertEmbeddingInputs` | `buildBertEmbeddingInputs()` |
| KV Cache 绑定到 attention inputs | `setupKVCacheForAttentionInputs()` |
| host tensor → 设备 tensor（pinned） | `tensorHoldHostAndToCuda()` |

#### ③ 图模式（Graph Runner）管理 ← 与 ACL Graph 直接相关

```cpp
GraphBase* graph_runner_{nullptr};   // CudaGraphRunner 或 AscendGraphRunner
```

- **创建**：构造函数里按平台 `new CudaGraphRunner / AscendGraphRunner`（`PyWrappedModel.h:243/287`）
- **捕获**：`graph_runner_->initCapture()`（`PyWrappedModel.h:312`）
- **分流**：`forward()` 里 `canRun()` 决定走图重放还是 eager（`PyWrappedModel.cc:493`）

```text
enable_cuda_graph_ && graph_runner_->canRun(...)
    ├─ true  → graph_runner_->forward(...)   // NPUGraph 重放
    └─ false → py_model_.attr("forward")(...) // Python eager
```

#### ④ 后处理（forward 后半段）

`callForwardPostLayers()`（`PyWrappedModel.cc:582`）：把模型输出的 hidden states 继续处理 —— final layernorm、TP 同步 logits（`tpSyncEmbeddingOrLogits`）、采样前准备等，返回 `GptModelOutputs` 给 Executor。

### 4.2 谁在使用它

| 调用方 | 场景 |
|---|---|
| `NormalExecutor.cc:99` | 主模型（prefill + decode） |
| `MtpExecutor.cc:206/248` | 投机解码的 target / draft 模型 |
| `EmbeddingExecutor.cc:70` | embedding 模型 |

它们统一通过 `model_->forward(model_input)` 调用。

### 4.3 与 ACL Graph 的关系总结

```text
NormalExecutor (C++ 调度)
    ↓ 统一接口 model_->forward()
PyWrappedModel (桥接层)
    ├─ 持有 Python 模型对象
    ├─ 创建 AscendGraphRunner（USING_ASCEND 平台）
    ├─ forward 时 canRun() 分流：图 vs eager
    └─ 图路径 → AscendGraphRunner::forward() → NPUGraph 重放
        eager 路径 → Python model.forward() → ascend_decode.py
```

> **一句话记忆**：`PyWrappedModel` = "Python 模型的 C++ 代理"，向上对 Executor 隐藏 Python 细节，向下对 Python 模型隐藏 C++ 引擎细节，中间插入 Graph Runner 加速 decode。

---

## 4. PyWrappedModel 创建 AscendGraphRunner

**文件：** `rtp_llm/cpp/models/PyWrappedModel.h:190-313`

```cpp
if (enable_cuda_graph_) {
#if USING_ASCEND
    if (is_prefill_cuda_graph_mode) {
        RTP_LLM_LOG_WARNING(
            "Ascend ACL Graph does not support prefill cuda graph mode; "
            "graph_runner_ will not be created, falling back to eager forward.");
    } else {
        GraphParams graph_params;

        graph_params.enable_cuda_graph =
            params.hw_kernel_config.enable_cuda_graph;
        graph_params.max_seq_len = params.max_seq_len;
        graph_params.max_context_batch_size =
            params.concurrency_config.concurrency_limit;
        graph_params.decode_capture_batch_sizes =
            params.hw_kernel_config.decode_capture_batch_sizes;

        graph_runner_ =
            new AscendGraphRunner(graph_params, py_instance);
    }
#endif

    if (graph_runner_ != nullptr) {
        graph_runner_->initCapture();
    }
}
```

主要逻辑：

- `graph_runner_` 是统一接口 `GraphBase*`
- Ascend 平台使用 `AscendGraphRunner`
- ACL Graph 当前主要支持 decode
- prefill graph 模式通常回退到 eager
- `decode_capture_batch_sizes` 决定捕获哪些 batch size

---

## 5. GraphBase 统一接口

**文件：** `rtp_llm/cpp/cuda_graph/cuda_graph_base.h:12-48`

```cpp
struct CudaGraphState {
    int current_batch_size{1};
    int current_real_graph_bs{1};
    int seq_len_sum{0};
};

class GraphBase {
public:
    virtual void initCapture() = 0;

    virtual PyModelOutputs forward(
        const PyModelInputs& inputs,
        CudaGraphState& state) = 0;

    virtual bool canRun(
        const PyModelInputs& inputs,
        CudaGraphState& state) = 0;
};
```

虽然状态结构名仍是 `CudaGraphState`，但 Ascend ACL Graph 复用了这套 CUDA/ROCm/Ascend 通用接口。

---

## 6. AscendGraphRunner 构造

**文件：** `rtp_llm/cpp/ascend_graph/ascend_graph_runner.cc:30-93`

```cpp
AscendGraphRunner::AscendGraphRunner(
    const GraphParams& graph_params,
    py::object py_instance)
    : GraphBase(std::move(py_instance)),
      enable_graph_(graph_params.enable_cuda_graph),
      max_seq_len_(graph_params.max_seq_len),
      decode_capture_batch_sizes_(
          graph_params.decode_capture_batch_sizes),
      capture_stream_(
          ascend_graph::graphGetStreamFromPool(true)),
      forward_event_(
          ascend_graph::makeGraphEvent()) {
    if (graph_params.is_prefill_cuda_graph_mode) {
        throw std::runtime_error(
            "prefill cuda graph mode is not supported");
    }

    py_attn_pyobj_method_ =
        py_instance_.attr("prepare_fmha_impl");
    py_forward_method_ =
        py_instance_.attr("forward");
}
```

Runner 保存两个 Python 方法：

```text
py_attn_pyobj_method_ → Python prepare_fmha_impl()
py_forward_method_    → Python forward()
```

后续 Graph 捕获和推理重放都通过这两个方法进入 Python。

---

# 一、初始化阶段：捕获 Graph

## 7. initCapture() 总入口

**文件：** `rtp_llm/cpp/ascend_graph/ascend_graph_runner.cc:385-419`

```cpp
void AscendGraphRunner::initCapture() {
    if (!enable_graph_) {
        return;
    }

    max_num_token_ = max_bs_ * num_tokens_per_bs_;
    capture_range_ = getDecodeBatchSizesToCapture();

    PyModelInputs inputs;
    inputs.input_ids =
        torch::zeros({max_num_token_}, options_npu_int32_);
    inputs.input_hiddens =
        torch::zeros(
            {max_num_token_, hidden_size_},
            options_npu_float_);

    initCaptureAttentionInputs(
        inputs, max_bs_, num_tokens_per_bs_);

    capture_mem_hold_ =
        ascend_graph::AscendGraphMemHold(
            output, inputs, false);

    initKernelInternalMemory();

    // 预热 Python forward
    auto attn_pyobj =
        py_attn_pyobj_method_(
            capture_mem_hold_.py_model_inputs_, true);
    py_forward_method_(
        capture_mem_hold_.py_model_inputs_, attn_pyobj);

    captureDecode();
}
```

初始化阶段主要完成：

1. 确定 batch size 桶
2. 分配固定输入 tensor
3. 初始化 KV Cache、cu_seqlens 等内部内存
4. 预热 Python forward，触发 kernel 初始化
5. 对每个 batch size 捕获一张 Graph

---

## 8. 生成 batch size 桶

**文件：** `rtp_llm/cpp/ascend_graph/ascend_graph_runner.cc:119-142`

```cpp
std::vector<int>
AscendGraphRunner::getDecodeBatchSizesToCapture() {
    if (!decode_capture_batch_sizes_.empty()) {
        std::sort(
            decode_capture_batch_sizes_.begin(),
            decode_capture_batch_sizes_.end());
        return decode_capture_batch_sizes_;
    }

    std::vector<int> capture_bs;
    for (int i : {1, 8, 16, 24, 32}) {
        if (i <= max_generate_batch_size) {
            capture_bs.push_back(i);
        }
    }
    return capture_bs;
}
```

例如：

```text
capture_range = [1, 8, 16, 32]
实际 batch size = 5  → 使用 batch size=8 的 Graph
```

---

## 9. 按 batch size 捕获 Graph

**文件：** `rtp_llm/cpp/ascend_graph/ascend_graph_runner.cc:429-464`

```cpp
void AscendGraphRunner::captureDecode() {
    for (int bs : capture_range_) {
        graph_instances_.try_emplace(bs);
    }

    // 从大 batch 到小 batch 顺序捕获
    for (int i = capture_range_.size() - 1; i >= 0; i--) {
        int bs = capture_range_[i];

        PyModelInputs inputs;
        prepareCaptureInputs(
            inputs, bs, bs * num_tokens_per_bs_);

        graph_instances_[bs].mem_hold_ =
            createMemHold(
                inputs, bs * num_tokens_per_bs_);

        graph_instances_[bs].mem_hold_.attn_pyobj_ =
            py_attn_pyobj_method_(
                graph_instances_[bs]
                    .mem_hold_
                    .py_model_inputs_, true);

        captureDecodeOneBatchSize(bs);
        replayAndSyncCheck(bs, "batch size");
    }
}
```

每个 batch size 对应一个 `AscendGraphInstance`：

```text
batch size
 ├─ NPUGraph
 ├─ 固定输入 tensor
 ├─ Python attention 对象
 └─ 固定输出 buffer
```

---

## 10. 捕获单张 Graph

**文件：** `rtp_llm/cpp/ascend_graph/ascend_graph_runner.cc:466-509`

```cpp
void AscendGraphRunner::captureOneGraphInstance(
    int key,
    const char* key_type) {

    auto inputs =
        graph_instances_[key].mem_hold_.py_model_inputs_;
    auto attn_pyobj =
        graph_instances_[key].mem_hold_.attn_pyobj_;

    // Warmup
    py_forward_method_(inputs, attn_pyobj);
    py_forward_method_(inputs, attn_pyobj);

    ascend_graph::graphDeviceSynchronize();

    ascend_graph::AscendGraphStreamLife
        stream_life(capture_stream_);

    auto& graph = graph_instances_[key].graph_;

    ascend_graph::graphCaptureBegin(
        graph,
        shared_graph_pool_,
        ascend_graph::GraphCaptureMode::Relaxed);

    auto py_outputs_obj =
        py_forward_method_(inputs, attn_pyobj);
    outputs = py_outputs_obj.cast<PyModelOutputs>();

    graph_instances_[key]
        .mem_hold_
        .decoder_layer_hidden_states_
        .copy_(outputs.hidden_states);

    ascend_graph::graphCaptureEnd(graph);
}
```

调用关系：

```text
graphCaptureBegin()
    └─ Python model.forward()
         └─ AscendDecodeImpl
              └─ FIA v2 Graph
```

---

# 二、Python 侧 Graph 捕获

## 11. 创建 FMHA 实现

**文件：** `rtp_llm/models_py/model_desc/module_base.py:95-103`

```python
def prepare_fmha_impl(
    self, inputs: PyModelInputs,
    is_cuda_graph: bool = False,
):
    fmha_impl = AttnImplFactory.get_fmha_impl(
        self.config,
        self.parallelism_config,
        self.weight,
        inputs.attention_inputs,
        self.fmha_config,
        is_cuda_graph,
    )
    return fmha_impl
```

工厂设置 Graph 标记：

**文件：** `rtp_llm/models_py/modules/factory/attention/attn_factory.py:144-149`

```python
attn_inputs.is_cuda_graph = is_cuda_graph
```

捕获阶段调用时：

```text
is_cuda_graph = True
```

---

## 12. AscendDecodeImpl 更新 Graph 输入

**文件：** `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_decode.py:101-108`

```python
def prepare_cuda_graph(self, attn_inputs):
    self.attn_inputs = attn_inputs
    self.fmha_impl.prepare(attn_inputs)

    batch_size = attn_inputs.sequence_lengths.size(0)
    seq_lens = attn_inputs.sequence_lengths[:batch_size]
    ctx_list = (
        seq_lens.to(torch.int32) + 1
    ).tolist()

    self.fmha_impl.update_graph_fia(
        ctx_list, batch_size)
```

这个函数由 C++ 在每次重放前调用，用于更新 FIA 的动态上下文长度。

---

## 13. FIA v2 Graph 捕获

### 13.1 判断 Graph 或 eager

**文件：** `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_decode.py:198-205`

```python
def forward(self, q, kv_cache, use_graph=False):
    if use_graph and torch.npu.is_current_stream_capturing():
        return self._forward_fia_graph(
            q, kv_cache, block_table, context_lens)

    return self._forward_fia(
        q, kv_cache, block_table, context_lens)
```

### 13.2 捕获 graph task

**文件：** `ascend_decode.py:249-262`

```python
stream = torch.npu.current_stream()

torch.npu.graph_task_group_begin(stream)

torch_npu.npu_fused_infer_attention_score_v2.out(
    query=q,
    key=k_cache,
    value=v_cache,
    block_table=block_table,
    workspace=shared_workspace,
    out=[out, lse],
)

handle = torch.npu.graph_task_group_end(stream)
```

捕获过程：

```text
graph_task_group_begin()
    └─ FIA v2 .out()
graph_task_group_end()
```

---

# 三、推理阶段：选择 Graph 或 eager

## 14. PyWrappedModel::forward()

**文件：** `rtp_llm/cpp/models/PyWrappedModel.cc:491-518`

```cpp
CudaGraphState graph_state;

if (enable_cuda_graph_ &&
    graph_runner_->canRun(
        py_model_inputs, graph_state)) {

    py_model_inputs
        .attention_inputs
        .is_s_padded = true;

    py_model_outputs =
        graph_runner_->forward(
            py_model_inputs, graph_state);
} else {
    held_attn_pyobj_ =
        py_model_.attr("prepare_fmha_impl")(
            py_model_inputs, false);

    auto py_model_forward =
        py_model_.attr("forward");

    auto outputs =
        py_model_forward(
            py_model_inputs,
            held_attn_pyobj_);
}
```

分支关系：

```text
canRun() == true
    └─ AscendGraphRunner::forward()
         └─ replay Graph

canRun() == false
    └─ Python model.forward(..., eager)
         └─ FIA v2 eager
```

---

## 15. canRun() 判断条件

**文件：** `rtp_llm/cpp/ascend_graph/ascend_graph_runner.cc:144-205`

```cpp
bool AscendGraphRunner::canRun(
    const PyModelInputs& inputs,
    CudaGraphState& state) {

    if (is_target_verify_) {
        if (inputs.attention_inputs.is_target_verify) {
            return tryGetRealGraphDecodeBatchSize(
                inputs, state);
        }
        return false;
    }

    if (!enable_graph_) {
        return false;
    }

    // Prefill 回退 eager
    if (inputs.attention_inputs.is_prefill) {
        return false;
    }

    // Hybrid KV Cache 组数检查
    if (!inputs.attention_inputs
             .kv_cache_kernel_block_id_device_by_group.empty()) {
        const size_t group =
            inputs.attention_inputs
                .kv_cache_kernel_block_id_device_by_group.size();

        if (kv_cache_group_num_ <= 0) {
            return false;
        }
        if (group != static_cast<size_t>(kv_cache_group_num_)) {
            return false;
        }
    }

    return tryGetRealGraphDecodeBatchSize(
        inputs, state);
}
```

batch size 选择：

**文件：** `ascend_graph_runner.cc:144-169`

```cpp
auto it = std::lower_bound(
    capture_range_.begin(),
    capture_range_.end(),
    state.current_batch_size);

if (it == capture_range_.end()) {
    return false;
}

state.current_real_graph_bs = *it;
state.seq_len_sum = state.current_batch_size;
```

示例：

```text
capture_range = [1, 8, 16, 32]

实际 batch = 5   → 使用 Graph 8
实际 batch = 16  → 使用 Graph 16
实际 batch = 40  → 超出范围，走 eager
```

---

# 四、推理阶段：准备输入并重放

## 16. AscendGraphRunner::forward()

**文件：** `rtp_llm/cpp/ascend_graph/ascend_graph_runner.cc:708-728`

```cpp
PyModelOutputs AscendGraphRunner::forward(
    const PyModelInputs& inputs,
    CudaGraphState& state) {

    prepareInputs(inputs, state);

    replayDecode(
        state.current_real_graph_bs);

    outputs.hidden_states =
        graph_instances_[
            state.current_real_graph_bs]
        .mem_hold_
        .decoder_layer_hidden_states_
        .slice(0, 0, state.seq_len_sum)
        .clone();

    forward_event_.record(
        ascend_graph::graphGetCurrentStream());

    return outputs;
}
```

调用顺序：

```text
forward()
 ├─ prepareInputs()
 ├─ replayDecode()
 ├─ 读取固定输出 buffer
 └─ record forward_event_
```

---

## 17. prepareInputs() 数据搬运

**文件：** `rtp_llm/cpp/ascend_graph/ascend_graph_runner.cc:561-705`

等待上一次推理完成：

```cpp
forward_event_.synchronize();

const size_t graph_idx =
    state.current_real_graph_bs;

auto& py_model_inputs =
    graph_instances_[graph_idx]
        .mem_hold_
        .py_model_inputs_;
```

清理旧 KV Cache block table：

```cpp
py_model_inputs.attention_inputs
    .kv_cache_kernel_block_id_device
    .fill_(0);

py_model_inputs.attention_inputs
    .kv_cache_kernel_block_id_host
    .fill_(0);
```

复制输入：

```cpp
copyTensorSlice(
    inputs.input_ids,
    py_model_inputs.input_ids);

copyTensorSlice(
    inputs.attention_inputs.input_lengths_d,
    py_model_inputs.attention_inputs.input_lengths_d);

copyTensorSlice(
    inputs.attention_inputs
        .kv_cache_kernel_block_id_device,
    py_model_inputs.attention_inputs
        .kv_cache_kernel_block_id_device);
```

更新 Python Attention：

**文件：** `ascend_graph_runner.cc:701-705`

```cpp
attn_pyobj.attr(
    "prepare_cuda_graph")(
        py_model_inputs.attention_inputs);
```

调用链：

```text
C++ prepareInputs()
    └─ AscendDecodeImpl.prepare_cuda_graph()
         └─ AscendDecodeAttnOp.update_graph_fia()
```

---

## 18. 动态更新 FIA Graph 参数

**文件：** `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_decode.py:270-294`

```python
def update_graph_fia(self, ctx_list, batch_size):
    if not self._graph_handles:
        return

    us = torch.npu.Stream()

    with torch.npu.stream(us):
        for i, handle in enumerate(self._graph_handles):
            torch.npu.graph_task_update_begin(
                us, handle
            )

            torch_npu.npu_fused_infer_attention_score_v2.out(
                query=q,
                key=k_cache,
                value=v_cache,
                actual_seq_kvlen=ctx_list,
                workspace=shared_workspace,
                out=[out, lse],
            )

            torch.npu.graph_task_update_end(us)

    us.synchronize()
```

作用：

```text
每次 replay 前
    ├─ 更新真实 context length
    ├─ 更新 FIA graph task
    └─ 保持 Graph 结构不变，只更新动态参数
```

---

## 19. 重放 Graph

**文件：** `rtp_llm/cpp/ascend_graph/ascend_graph_runner.cc:516-527`

```cpp
void AscendGraphRunner::replayDecode(int bs) {
    replayGraph(bs);
}

void AscendGraphRunner::replayGraph(int key) {
#if USING_ASCEND
    ascend_graph::graphReplay(
        graph_instances_[key].graph_);
#endif
}
```

最终进入 device shim：

```cpp
graph.replay();
```

---

# 五、底层 Ascend Graph 封装

## 20. Device Shims 接口

**文件：** `rtp_llm/cpp/ascend_graph/ascend_graph_device_shims.h:34-79`

```cpp
using GraphStream = c10_npu::NPUStream;

GraphStream graphGetStreamFromPool(bool high_priority);
GraphStream graphGetCurrentStream();

void graphCaptureBegin(
    c10_npu::NPUGraph& graph,
    GraphPoolHandle pool,
    GraphCaptureMode mode);

void graphCaptureEnd(c10_npu::NPUGraph& graph);
void graphReplay(c10_npu::NPUGraph& graph);
```

## 21. Device Shims 实现

**文件：** `rtp_llm/cpp/ascend_graph/ascend_graph_device_shims.cc:43-105`

```cpp
GraphStream graphGetStreamFromPool(bool high_priority) {
    return c10_npu::getStreamFromPool(
        high_priority,
        c10_npu::current_device());
}

void graphCaptureBegin(
    c10_npu::NPUGraph& graph,
    GraphPoolHandle,
    GraphCaptureMode mode) {

    graph.capture_begin(
        {0, 0},
        toAclMode(mode));
}

void graphCaptureEnd(c10_npu::NPUGraph& graph) {
    graph.capture_end();
}

void graphReplay(c10_npu::NPUGraph& graph) {
    graph.replay();
}
```

底层对应关系：

```text
GraphStream       → c10_npu::NPUStream
Graph capture     → c10_npu::NPUGraph::capture_begin/end
Graph replay      → c10_npu::NPUGraph::replay
Device sync       → aclrtSynchronizeDevice
Stream sync       → aclrtSynchronizeStream
```

---

# 六、核心数据结构

## 22. AscendGraphRunner

**文件：** `rtp_llm/cpp/ascend_graph/ascend_graph_runner.h:32-103`

```cpp
class AscendGraphRunner : public GraphBase {
public:
    void initCapture() override;

    PyModelOutputs forward(
        const PyModelInputs& inputs,
        CudaGraphState& state) override;

    bool canRun(
        const PyModelInputs& inputs,
        CudaGraphState& state) override;

private:
    std::unordered_map<
        int,
        ascend_graph::AscendGraphInstance>
        graph_instances_;

    ascend_graph::AscendGraphMemHold
        capture_mem_hold_;
};
```

## 23. AscendGraphInstance

**文件：** `rtp_llm/cpp/ascend_graph/ascend_graph_utils.h:24-82`

```cpp
class AscendGraphInstance {
public:
    c10_npu::NPUGraph graph_;
    AscendGraphMemHold mem_hold_;
};
```

每个 batch size 都保存一套独立的：

```text
NPUGraph
固定输入 tensor
Python attention 对象
固定输出 buffer
```

---

# 七、关键代码位置速查表

| 功能 | 文件 | 行号 |
|---|---|---:|
| PyWrappedModel 类定义 | `rtp_llm/cpp/models/PyWrappedModel.h` | 38 |
| 创建 PyWrappedModel | `rtp_llm/cpp/normal_engine/NormalExecutor.cc` | 99 |
| 发起模型 forward | `rtp_llm/cpp/normal_engine/NormalExecutor.cc` | 169 |
| 输入转换（PyAttentionInputs 等） | `rtp_llm/cpp/models/PyWrappedModel.cc` | 443-489 |
| 后处理 callForwardPostLayers | `rtp_llm/cpp/models/PyWrappedModel.cc` | 582 |
| 创建 AscendGraphRunner | `rtp_llm/cpp/models/PyWrappedModel.h` | 251-312 |
| Graph/Eager 分支 | `rtp_llm/cpp/models/PyWrappedModel.cc` | 491-518 |
| GraphBase 接口 | `rtp_llm/cpp/cuda_graph/cuda_graph_base.h` | 12-48 |
| Runner 构造 | `rtp_llm/cpp/ascend_graph/ascend_graph_runner.cc` | 30-93 |
| batch size 桶生成 | `ascend_graph_runner.cc` | 119-142 |
| Graph 是否可运行 | `ascend_graph_runner.cc` | 144-205 |
| 初始化捕获 | `ascend_graph_runner.cc` | 385-419 |
| 捕获所有 batch Graph | `ascend_graph_runner.cc` | 429-464 |
| 捕获单张 Graph | `ascend_graph_runner.cc` | 466-509 |
| 准备 replay 输入 | `ascend_graph_runner.cc` | 561-705 |
| replay 入口 | `ascend_graph_runner.cc` | 708-728 |
| replay Graph | `ascend_graph_runner.cc` | 516-527 |
| Python FMHA 创建 | `models_py/model_desc/module_base.py` | 95-103 |
| 设置 `is_cuda_graph` | `models_py/modules/factory/attention/attn_factory.py` | 144-149 |
| Python Graph 输入更新 | `ascend_decode.py` | 101-108 |
| FIA Graph 捕获 | `ascend_decode.py` | 198-262 |
| FIA Graph 动态更新 | `ascend_decode.py` | 270-294 |
| NPU Graph 封装 | `ascend_graph_device_shims.h/.cc` | 34-105 |
| Graph 实例和内存保持 | `ascend_graph_utils.h` | 24-82 |

---

# 八、核心结论

ACL Graph 的核心思想是：

> **捕获阶段固定算子执行流程和内存地址；重放阶段只把新请求的数据拷贝到固定 buffer，更新动态 FIA 参数，再调用 `NPUGraph::replay()`。**

因此，推理阶段的主要工作变成：

```text
真实请求数据
    └─ copy 到 Graph 固定输入 buffer
         └─ 更新 context length / KV block table
              └─ NPUGraph::replay()
                   └─ 读取固定输出 buffer
```

这可以减少每次推理时 Python 和 Host 侧的算子调度开销。
