下面整理成 Markdown 文档格式：

```markdown
# NVIDIA CUDA Graph 原理总结：Capture → Replay

## 1. CUDA Graph 概述

CUDA Graph 是 NVIDIA 提供的一种 GPU 执行优化机制。

核心思想：

> 将一次已经确定的 GPU 工作流（kernel 执行顺序、依赖关系、参数和内存绑定）提前捕获（Capture），生成可执行 Graph，之后通过 Replay 方式重复执行，从而减少 CPU 对 GPU kernel 的逐算子调度开销。

传统 CUDA 执行模式：

```

CPU
|
| cudaMemcpyAsync()
|
| Kernel Launch
|
| Kernel Launch
|
| Kernel Launch
|
GPU
|
Kernel1
Kernel2
Kernel3

```

每个 kernel 都需要 CPU Runtime 参与调度。

CUDA Graph：

```

第一次执行:

CPU
|
| Stream Capture
↓
CUDA Graph
|
| Kernel1
| Kernel2
| Kernel3
|
Instantiate
↓
Executable Graph

后续执行:

CPU
|
| cudaGraphLaunch()
↓
GPU

Kernel1
Kernel2
Kernel3

```

Replay 阶段 CPU 不再逐个提交 kernel。

---

# 2. CUDA Graph 生命周期

CUDA Graph 主要包含三个阶段：

```

Capture
|
↓
cudaGraph_t

|
↓

Instantiate

|
↓

cudaGraphExec_t

|
↓

Replay

|
↓

cudaGraphLaunch()

````

---

# 3. Capture：捕获阶段

## 3.1 Capture 的作用

Capture 的作用：

> 将 CUDA Stream 中发生的一系列 GPU 操作记录为一个静态执行图。

示例：

```cpp
cudaStreamBeginCapture(stream);

cudaMemcpyAsync();

kernel_A<<<...>>>();

kernel_B<<<...>>>();

cudaStreamEndCapture(stream,&graph);
````

CUDA 不仅执行这些操作，同时记录：

```
Graph:

Memcpy
  |
  ↓
Kernel_A
  |
  ↓
Kernel_B
```

生成：

```
cudaGraph_t
```

---

# 4. Capture 捕获的内容

## 4.1 Kernel 执行顺序

Graph 保存：

```
Node1:
Matrix Multiplication

Node2:
LayerNorm

Node3:
Attention

Node4:
AllReduce
```

以及它们之间的依赖：

```
MatMul
   |
   ↓
Norm
   |
   ↓
Attention
```

也就是：

* kernel topology
* execution dependency

---

## 4.2 Kernel 参数

例如：

```cpp
matmul<<<grid,block>>>
(
 A,
 B,
 C,
 N
)
```

Capture 后记录：

```
Kernel:
matmul

Arguments:

A = 0x100000
B = 0x200000
C = 0x300000

N = 4096
```

---

## 4.3 Memory Address Binding

CUDA Graph 最关键的限制：

> Graph 绑定的是 tensor 的 memory address(pointer)，而不是 tensor 名称。

例如：

第一次：

```
Q tensor:

address = 0x100000
```

Graph:

```
Attention Node

Input:
0x100000
```

Replay 时：

如果：

```
Q tensor:

address = 0x900000
```

Graph 仍然访问：

```
0x100000
```

因此会产生错误。

所以 CUDA Graph 要求：

* tensor storage 固定
* memory allocation 固定
* pointer 地址稳定

---

# 5. Instantiate：生成可执行 Graph

Capture 得到：

```
cudaGraph_t
```

它只是 Graph 描述。

通过：

```cpp
cudaGraphInstantiate()
```

生成：

```
cudaGraphExec_t
```

区别：

| 对象              | 作用           |
| --------------- | ------------ |
| cudaGraph_t     | Graph结构描述    |
| cudaGraphExec_t | 优化后的可执行Graph |

真正执行的是：

```
cudaGraphExec_t
```

---

# 6. Replay：重放阶段

Replay 调用：

```cpp
cudaGraphLaunch(graphExec, stream)
```

执行已经捕获好的 Graph。

---

## 6.1 Eager执行模式

传统：

```
CPU

launch kernel1

launch kernel2

launch kernel3

launch kernel4


GPU

kernel1

kernel2

kernel3

kernel4
```

CPU需要：

* 调用 Runtime
* 设置参数
* 提交kernel

---

## 6.2 Graph执行模式

```
CPU

cudaGraphLaunch()


GPU

kernel1

kernel2

kernel3

kernel4
```

CPU只提交一次。

---

# 7. CUDA Graph 为什么更快？

主要收益：

## 减少 CPU Kernel Launch 开销

传统：

```
1000 kernels

↓

1000次 CPU 调度
```

Graph：

```
1000 kernels

↓

1次 Graph Launch
```

减少：

* Python 调度
* PyTorch dispatcher
* CUDA runtime launch
* driver interaction

---

# 8. CUDA Graph 的三个本质

## 本质1：执行拓扑固定

Capture 固化：

```
kernel顺序

dependency关系

stream关系
```

不能：

```
增加kernel

删除kernel

改变执行路径
```

---

## 本质2：Replay 绕过 CPU 调度

传统：

```
CPU
 |
kernel launch
 |
GPU
```

Graph：

```
CPU
 |
Graph Launch
 |
GPU执行完整pipeline
```

CPU overhead 大幅降低。

---

## 本质3：输入输出资源需要静态化

Graph要求：

```
Tensor address固定

Memory layout固定

Execution path固定
```

因此动态量需要特殊处理。

---

# 9. 动态输入如何处理？

LLM 推理中：

动态：

```
batch size

sequence length

context length

KV cache offset
```

但是 Graph 需要：

```
固定shape

固定memory

固定kernel路径
```

常用方案：

---

## 方案1：固定 Shape

例如：

```
batch=8

seq_len=8192
```

所有请求使用该 Graph。

---

## 方案2：多个 Graph Cache

例如：

```
Graph1:

batch=1


Graph2:

batch=2


Graph3:

batch=4


Graph4:

batch=8
```

请求：

```
batch=5
```

padding：

```
batch=8
```

执行：

```
Graph4 replay
```

---

## 方案3：Graph Update

部分参数可以更新：

例如：

```
seq_len

kernel参数

Memcpy size
```

通过：

```
cudaGraphExecKernelNodeSetParams()
```

但是不能修改：

```
kernel数量

dependency

memory allocation
```

---

# 10. CUDA Graph 与 LLM 推理

Transformer Decode：

Eager:

```
Token

 |
Attention

 |
MLP

 |
AllReduce

 |
Next Token
```

每个 token：

重复 kernel 调度。

---

Graph:

第一次：

```
Decode step

↓

Capture

↓

Graph:

Attention
MLP
AllReduce
```

之后：

```
Token1

Graph Replay


Token2

Graph Replay


Token3

Graph Replay
```

---

# 11. vLLM / TensorRT-LLM 中的 Graph 使用

通常不会只有一个 Graph。

原因：

LLM 输入动态。

因此维护：

```
Graph Pool

batch=1 graph

batch=2 graph

batch=4 graph

batch=8 graph
```

请求根据：

* batch size
* sequence length
* decode阶段

选择对应 Graph。

---

# 12. Profiler 中如何识别 Graph 是否生效

## Eager模式

典型：

```
PyTorch

 |
CANN/CUDA Runtime

 |
Kernel

Kernel

Kernel
```

表现：

* Host Runtime 调用多
* kernel间存在间隙
* CPU调度明显

---

## Graph模式

典型：

```
PyTorch

 |
Graph Execute

 |
GPU Nodes

 |
Kernel

Kernel

Kernel
```

表现：

* Graph Execute / Replay事件
* Host runtime减少
* Kernel连续执行
* GPU idle减少

---

# 13. 总结

CUDA Graph 可以概括为：

> Capture 是把一次 GPU 执行过程录制成静态 Graph；Replay 是直接执行这个 Graph，跳过 CPU 对每个 kernel 的调度。

核心收益：

```
减少 CPU overhead
提高 GPU利用率
降低小batch推理延迟
```

核心限制：

```
执行路径需要固定

Memory address需要固定

动态shape需要通过：

- 多Graph
- padding
- update机制

解决
```

对于大模型推理：

CUDA Graph 的本质不是改变模型计算，而是：

> 把重复发生的 GPU 执行流程从“每次重新调度”变成“一次录制，多次播放”。

```
```
