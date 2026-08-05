# fused_gdn_gating 算子 GPU-NPU 精度比对指南

本文档总结 `fused_gdn_gating`（Gated Delta Net 门控）算子 NPU-vs-GPU 比对的实践经验，供后续该算子维护和类似 elementwise 门控算子比对参考。通用方法论参见同目录下的 `gpu_npu_comparison_guide.md`。

> **重要**：该算子的 NPU 实现在 **vllm-ascend** 中（`vllm_ascend/ops/triton/fused_gdn_gating.py` 的 `fused_gdn_gating_patch` Triton kernel），**不在** `fla_npu.ops.ascendc`。因此验证直接调用 vllm-ascend 的 Triton 算子，参考 vllm-ascend 目录下其他算子的测试方式（如 `test_fused_gdn_gating.py`）。

## 整体流程

```
GPU 黄金数据 (.pt)  →  恢复非连续 stride (a/b)  →  解析输入/输出  →  调用 vllm-ascend fused_gdn_gating Triton kernel  →  NPU 执行  →  与 golden 比对
```

## 1. 理解 GPU 数据格式

### 数据来源

GPU 端通过 `torch.save()` 导出的 `.pt` 文件位于 `<workspace>/sample/fused_gdn_gating/`，采用 `inputs`/`outputs`/`input_meta` 分层结构：

```python
data = torch.load(path, map_location="cpu", weights_only=False)
# 顶层 key: op_name, mode, param_names, inputs, outputs, input_meta, model_state
```

### 典型数据内容

| key | 类型 | 值 |
|-----|------|----|
| `op_name` | str | `"fused_gdn_gating"` |
| `mode` | str | `"decode_seq1"` / `"prefill_seq32"` / `"prefill_seq2047"` |
| `param_names` | list | `['A_log', 'a', 'b', 'dt_bias', 'beta', 'threshold']` |
| `inputs/A_log` | tensor | shape=(H,), dtype=float16, stride=(1,), contiguous=True |
| `inputs/a` | tensor | shape=(S, H), dtype=float16, **stride=(64, 1)**，见下文非连续说明 |
| `inputs/b` | tensor | shape=(S, H), dtype=float16, **stride=(64, 1)**，与 `a` 相同 |
| `inputs/dt_bias` | tensor | shape=(H,), dtype=float16, stride=(1,), contiguous=True |
| `outputs[0]` | tensor | shape=(1, S, H), dtype=float32 — 门控 `g` |
| `outputs[1]` | tensor | shape=(1, S, H), dtype=float16 — `beta` |
| `input_meta` | dict | 记录 A_log/a/b/dt_bias 的原始 shape/stride/dtype/contiguous |

### 三个 sample 的具体 shape

| 文件 | S（token 数） | H（head 数） | `a`/`b` stride | `a`/`b` 是否非连续 | `g` shape | `beta` shape |
|------|--------------|-------------|----------------|-------------------|-----------|--------------|
| `decode_seq1.pt` | 1 | 32 | (64, 1) | 否（dim0=1 时 stride 任意仍算连续） | (1, 1, 32) | (1, 1, 32) |
| `prefill_seq32.pt` | 32 | 32 | (64, 1) | **是**（dim0=32，行 stride 64 > 32） | (1, 32, 32) | (1, 32, 32) |
| `prefill_seq2047.pt` | 2047 | 32 | (64, 1) | **是** | (1, 2047, 32) | (1, 2047, 32) |

### 与 chunk_local_cumsum / causal_conv1d 数据格式的差异

| 维度 | chunk_local_cumsum | causal_conv1d | fused_gdn_gating |
|------|-------------------|---------------|------------------|
| 输出结构 | 单个 tensor | 单个 tensor / inplace_outputs | **list 两个 tensor**（`g` fp32 + `beta` fp16） |
| 标量参数 | `reverse`/`scale` 为 None 表示默认值 | 嵌入 inputs | `beta`/`threshold` 记录在 param_names，但 inputs 不直接存 |
| `a`/`b` 非连续性 | `g` 常连续 | `x`/`conv_state` 常非连续 | **prefill 时 `a`/`b` 非连续**（stride=(64,1)） |
| 是否需 layout 转置 | (B,T,H)→(B,H,T) | (D,S)→(S,D) | **无需转置**，GPU/NPU 布局一致 |
| NPU 算子来源 | fla_npu.ops.ascendc | fla_npu.ops.ascendc | **vllm_ascend Triton kernel** |

### `input_meta` 与 stride 恢复（关键）

`a`/`b` 在 GPU 上是 cache buffer 的**视图**：底层每行 64 个 float16，但该算子只用到前 `H=32` 个 head，因此行 stride 是 64 而非 32。`torch.save()` 以连续形式保存会丢失这个 stride，必须从 `input_meta` 恢复：

```python
a = _restore_strided_tensor(inputs["a"], meta.get("a", {}))
# 恢复后: a.shape=(S, 32), a.stride()=(64, 1) — 非连续
```

> **注意**：`decode_seq1.pt` 中 `a` 的 dim0=1，此时 `stride=(64,1)` 会被 `torch` 判定为连续（大小为 1 的维度 stride 无关紧要），因此 `input_meta` 中 `contiguous=True`。但 `prefill_seq32` / `prefill_seq2047` 中 dim0>1，`stride=(64,1)` 是**真非连续**。恢复逻辑对两种情况都应保留。

## 2. GPU kernel 的数学语义

GPU 端实现来自 `rtp_llm/rtp_llm/models_py/triton_kernels/fla/gdn_gating.py` 的 `fused_gdn_gating_kernel`：

```python
x = a.to(tl.float32) + dt_bias.to(tl.float32)
softplus_x = tl.where(beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x)
g  = -tl.exp(A_log.to(tl.float32)) * softplus_x     # float32 输出
beta_out = tl.sigmoid(b.to(tl.float32))             # 输出 dtype = b.dtype (float16)
```

即：

- $g = -\exp(A\_log) \cdot \text{softplus}(a + dt\_bias)$（softplus 带 `beta`/`threshold` 阈值分支，默认 `beta=1.0, threshold=20.0`）
- $\beta = \text{sigmoid}(b)$

两个输出都带一个 leading singleton batch 维，最终 shape 为 `(1, S, H)`。

### 关键：GPU kernel 处理非连续输入

GPU kernel 通过 `stride_ab`（batch 维 stride）显式支持非连续 `a`/`b`：

```python
stride_ab = a.stride(0)   # = 64（cache buffer 行 stride），用于非连续寻址
assert stride_ah == 1 and stride_bh == 1, "stride_ah must be 1"
```

这解释了为何 dump 中 `a`/`b` 的 stride 是 `(64, 1)`——它们是从更宽的 cache buffer 切出来的视图。

## 3. 算子在 vllm-ascend 中的实现

vllm-ascend 的 Triton kernel 位于 `vllm_ascend/ops/triton/fused_gdn_gating.py`：

- `fused_gdn_gating_patch(A_log, a, b, dt_bias, beta=1.0, threshold=20.0)`：Triton kernel，NPU 上的实际执行算子
- 它按**连续** `(S, H)` 偏移寻址 `a`/`b`，因此调用前需 `.contiguous()`

```python
from vllm_ascend.ops import triton as triton_ops

# ``vllm_ascend.ops.triton`` 是空 __init__ 的 namespace 包，函数在子模块中
_fused_gdn_gating_patch = triton_ops.fused_gdn_gating.fused_gdn_gating_patch
_init_device_properties_triton = triton_ops.triton_utils.init_device_properties_triton
```

### 与 `KdaGateCumsum` 的区别

`flash-linear-attention-npu` 中的 `KdaGateCumsum`（`npu_kda_gate_cumsum`）在 `use_gate_in_kernel=True` 时也实现了 `-exp(A_log)*softplus(g+dt_bias)`，但它是 **gate 激活 + chunk-local cumsum** 融合算子：

```text
gk = chunk_local_cumsum(-exp(A_log) * softplus(g + dt_bias)) / ln(2)
```

- 输出是累积后的 `gk`（÷ln(2)），不是逐元素 `g`
- **无 `beta = sigmoid(b)` 输出**
- 输入布局为 `[B, H, T, K]` head-major，与 GPU `fused_gdn_gating` kernel 的 `[S, H]` 不同

因此验证 GPU 的 `fused_gdn_gating` kernel 时，应调用 vllm-ascend 的 `fused_gdn_gating_patch`，而非 `KdaGateCumsum`。

## 4. 测试实现

测试文件位于 `rtp-llm/example/ascendc_npu/test_npu_fused_gdn_gating_gpu_golden.py`，核心结构：

```python
@unittest.skipIf(not torch.npu.is_available(), "NPU is not available")
class TestFusedGdnGatingGpuGolden(unittest.TestCase):
    rtol = 5e-2
    atol = 5e-2

    @classmethod
    def setUpClass(cls):
        # fused_gdn_gating_patch 内部调用 get_vectorcore_num()，需先初始化设备属性
        _init_device_properties_triton()

    def _run_case(self, filename: str) -> None:
        case = _load_gpu_case(filename)

        # 校验 golden 合法性
        self.assertEqual(tuple(case["g_expected"].shape), (1,) + tuple(case["a"].shape))
        self.assertEqual(case["g_expected"].dtype, torch.float32)
        self.assertEqual(case["beta_expected"].dtype, torch.float16)

        # Triton kernel 按连续 (S, H) 偏移寻址，恢复的非连续视图需先 .contiguous()
        a = case["a"].contiguous().npu()
        b = case["b"].contiguous().npu()

        g, beta = _fused_gdn_gating_patch(
            A_log=case["A_log"].npu(),
            a=a,
            b=b,
            dt_bias=case["dt_bias"].npu(),
        )
        torch.npu.synchronize()

        self.assertTensorClose(g, case["g_expected"])
        self.assertTensorClose(beta, case["beta_expected"])
```

三个用例覆盖 decode（S=1）与 prefill（S=32 / S=2047）。

## 5. 常见错误

### 5.1 输出结构当作单个 tensor

`data["outputs"]` 是 **list**，不是单个 tensor：

```python
# ✗ 错误：把 outputs 当单个 tensor
g_expected = data["outputs"]

# ✓ 正确：outputs 是 list，分别取 g 和 beta
g_expected = data["outputs"][0]      # (1, S, H) fp32
beta_expected = data["outputs"][1]   # (1, S, H) fp16
```

### 5.2 未恢复 `a`/`b` 的非连续 stride

prefill 时 `a`/`b` 是 stride=(64,1) 的非连续视图。不恢复 stride，就丢失了 GPU 端的真实内存布局：

```python
# ✗ 错误：直接用保存的连续 tensor
a = inputs["a"]

# ✓ 正确：从 input_meta 恢复原始 stride
a = _restore_strided_tensor(inputs["a"], meta.get("a", {}))
```

### 5.3 忘记 `.contiguous()` 传给 Triton kernel

vllm-ascend 的 `fused_gdn_gating_patch` 按**连续** `(S, H)` 偏移寻址，不处理非连续 stride。恢复后的非连续视图必须 `.contiguous()`：

```python
# ✗ 错误：直接把非连续视图传进去
g, beta = _fused_gdn_gating_patch(a=a, b=b, ...)

# ✓ 正确：先 .contiguous()
g, beta = _fused_gdn_gating_patch(a=a.contiguous(), b=b.contiguous(), ...)
```

### 5.4 未初始化 Triton 设备属性

`fused_gdn_gating_patch` 内部调用 `get_vectorcore_num()`，它断言设备属性已初始化。必须在调用前执行 `init_device_properties_triton()`，且应放在 `setUpClass`（而非 `__main__`）以兼容 pytest 发现：

```python
# ✗ 错误：只在 __main__ 初始化，pytest 发现时不会执行
if __name__ == "__main__":
    init_device_properties_triton()

# ✓ 正确：setUpClass 中初始化
@classmethod
def setUpClass(cls):
    init_device_properties_triton()
```

### 5.5 在 float16 下计算 `a + dt_bias`

kernel 显式 `.to(tl.float32)` 后再相加。若在 fp16 下计算，长序列上可能溢出（A 变 `-inf`）。这是 kernel 内部语义，参考实现应保持一致。


## 6. 调试技巧

1. **确认算子来源**：`fused_gdn_gating` 在 vllm-ascend 中（`fused_gdn_gating_patch`），不在 `fla_npu.ops.ascendc`。先用 `from fla_npu.ops.ascendc._aclnn_ctypes import ASCENDC_CTYPES_OPS` 确认算子是否在该命名空间
2. **确认 `a`/`b` 的 stride**：打印 `a.stride()`，若是 `(64,1)` 且 shape dim0>1，说明是 cache buffer 的非连续视图
3. **注意 decode 与 prefill 的连续性差异**：`decode_seq1` 的 `a` 因 dim0=1 被判为连续，`prefill` 的 `a` 是非连续。不要假设所有 sample 一致
4. **区分输出列表索引**：`outputs[0]` 是 `g`（fp32），`outputs[1]` 是 `beta`（fp16），顺序别搞反
5. **初始化 Triton 设备属性**：`fused_gdn_gating_patch` 依赖 `get_vectorcore_num()`，务必先 `init_device_properties_triton()`
6. **阈值分支 softplus 精度**：`beta*x <= threshold` 时用 `log1p(exp(...))`，否则用线性 `x`，避免 exp 溢出

## 7. GPU 与 NPU 布局策略对比

| 维度 | GPU dump | NPU（vllm-ascend Triton） |
|------|---------|------------------|
| `A_log` / `dt_bias` | (H,) fp16 连续 | 同 (H,) fp16，无需转换 |
| `a` / `b` 布局 | (S, H) fp16，prefill 时 stride=(64,1) 非连续 | 同 (S, H)，恢复 stride 后 `.contiguous().npu()` |
| 输出 `g` | (1, S, H) fp32 | 同 (1, S, H) fp32 |
| 输出 `beta` | (1, S, H) fp16 | 同 (1, S, H) fp16 |
| layout 转置 | 无 | **无需转置**（与 chunk_local_cumsum 不同） |
| 算子来源 | — | **vllm-ascend `fused_gdn_gating_patch`**（非 fla_npu.ops.ascendc） |
| 非连续性 | `a`/`b` 非连续（cache 视图） | 恢复 stride 后需 `.contiguous()` 再传给 Triton kernel |

**结论**：`fused_gdn_gating` 是逐元素门控算子，其 NPU 实现在 vllm-ascend 中。比对的核心难点不在 layout 转换（GPU/NPU 布局一致），而在三点——(1) 正确恢复 `a`/`b` 的非连续 stride；(2) 调用前 `.contiguous()` 以满足 Triton kernel 的连续寻址；(3) 先初始化 Triton 设备属性。测试通过直接调用 vllm-ascend 算子并与 GPU golden 数据对比完成验证。
