# RTP-LLM 源码走读：从服务启动到模型推理的完整调用链

> 角色：讲师版源码导读
>
> 适用分支：`pr-25`
>
> 说明：本文以当前 checkout 的源码为准，行号是本次走读时的参考位置；后续代码变更后行号可能变化。本文重点分析语言模型推理主链路，并在最后单独展开 Ascend ACL Graph Decode 链路。

---

## 1. 先建立整体认识

RTP-LLM 是一个生产级大模型推理引擎，核心目标不是训练模型，而是把 HuggingFace 等格式的模型加载到 GPU/NPU 上，处理在线请求，并通过动态批处理、KV Cache、多卡并行和硬件图模式提升吞吐与延迟表现。

它采用 **Python 模型定义 + C++ 推理引擎 + Python/C++ 服务层** 的混合架构：

- Python：模型配置、模型类、tokenizer、权重加载、部分硬件算子调用。
- C++：请求调度、推理循环、KV Cache、采样、多卡通信、RPC 和底层模型执行。
- FastAPI/Uvicorn：对外提供 HTTP/OpenAI 风格接口。
- gRPC：Frontend 与 Backend 之间传递推理请求和流式结果。
- CUDA/ROCm/Ascend：提供不同硬件上的 kernel、stream、graph 和通信能力。

总体关系如下：

```text
Client
  │ HTTP / OpenAI API
  ▼
FrontendApp / FastAPI
  │ tokenizer、chat template、请求校验、流式响应
  ▼
FrontendWorker / Pipeline
  │ gRPC client
  ▼
C++ Backend RPC Server
  │ QueryConverter、请求入队、输出轮询
  ▼
NormalEngine
  │ 后台 loop
  ▼
Scheduler
  │ FIFO 或 Batch Decode
  ▼
NormalExecutor
  │ gather input → model forward → sample → dispatch output
  ▼
PyWrappedModel
  │ C++ 输入转换 + Python forward + Graph/Eager 分流
  ▼
Python Model / Attention / Custom Ops
  │
  ├─ eager：普通 Python forward
  └─ graph：CudaGraphRunner 或 AscendGraphRunner
```

---

## 2. 第一条调用链：服务启动

### 2.1 启动入口

源码位置：

- `rtp_llm/start_server.py:186-194`
- `rtp_llm/cli/serve.py:12-45`

传统启动方式是：

```bash
python -m rtp_llm.start_server
```

CLI 方式是：

```bash
rtp-llm serve <model_path>
```

`start_server.py:186` 的 `main()` 主要做三件事：

1. `setup_args()` 解析参数和环境变量。
2. `setup_and_configure_server()` 创建完整的 `PyEnvConfigs`。
3. 调用 `start_server(py_env_configs)`。

关键代码位置：

```python
# rtp_llm/start_server.py:186-194

def main():
    py_env_configs = setup_args()
    setup_and_configure_server(py_env_configs)
    start_server(py_env_configs)
```

### 2.2 启动 Backend 和 Frontend

源码位置：`rtp_llm/start_server.py:42-184, 192-258`

`start_server()` 会创建全局并发控制器和 `ProcessManager`，随后根据角色启动 Backend 与 Frontend：

```text
start_server()
  ├─ init_controller()
  ├─ ProcessManager()
  ├─ start_backend_server_impl()
  │    └─ multiprocessing.Process(start_backend_server)
  └─ start_frontend_server_impl()
       └─ multiprocessing.Process(start_frontend_server)
```

Backend 启动后通过 pipe 报告初始化成功，Frontend 通过 `/health` 或 gRPC health check 检查 Backend 是否可用。

### 2.3 多卡 Backend 启动

源码位置：

- `rtp_llm/start_backend_server.py:35-150`：单 rank 启动
- `rtp_llm/start_backend_server.py:274-350`：多 rank 启动

核心函数是 `local_rank_start()`：

```text
local_rank_start()
  ├─ set_parallelism_config()
  ├─ setup_cuda_device_and_accl_env()
  ├─ set_global_controller()
  ├─ BackendManager(py_env_configs)
  ├─ backend_manager.start()
  └─ backend_manager.serve_forever()
```

这里的 `cuda` 命名是历史兼容命名，实际代码也会根据平台使用 Ascend NPU 或 ROCm。设备类型判断可关注：

- `rtp_llm/device/device_type.py`
- `rtp_llm/device/device_impl.py`
- `rtp_llm/models/base_model.py:118-136` 的 `_get_device_str()`

多卡场景会为每个 local rank 创建进程，并通过分布式配置设置：

- `WORLD_RANK`
- local rank
- TP/DP rank
- NCCL/HCCL 通信配置

---

## 3. 第二条调用链：Frontend 初始化和 HTTP 请求入口

### 3.1 FrontendApp 启动

源码位置：

- `rtp_llm/start_frontend_server.py`
- `rtp_llm/frontend/frontend_app.py:60-162`

调用链：

```text
start_frontend_server()
  └─ FrontendApp(py_env_configs)
       ├─ FrontendServer(...)
       ├─ EngineConfig.create()
       ├─ get_world_info()
       └─ GrpcClientWrapper(...)

FrontendApp.start()
  ├─ FrontendServer.start()
  ├─ create_app()
  ├─ socket.bind(0.0.0.0:server_port)
  └─ uvicorn.Server.run()
```

`FrontendApp` 的主要职责是：

- 创建 FastAPI 应用。
- 创建 gRPC 客户端连接 Backend。
- 等待 Backend 健康检查通过。
- 绑定对外 HTTP 端口。
- 管理优雅退出和正在处理的请求。

### 3.2 FrontendServer 初始化

源码位置：`rtp_llm/frontend/frontend_server.py:43-129`

`FrontendServer.start()` 会再次创建模型配置和 tokenizer，但这里主要用于请求侧处理，不负责真正加载 GPU 模型权重：

```text
FrontendServer.start()
  ├─ ModelFactory.create_model_config()
  ├─ FrontendWorker(...)
  │    ├─ TokenizerFactory.create()
  │    ├─ EngineConfig.create()
  │    ├─ get_world_info()
  │    └─ Pipeline(...)
  ├─ LANGUAGE_MODEL → OpenaiEndpoint
  └─ other task → EmbeddingEndpoint
```

相关源码：

- `rtp_llm/frontend/frontend_worker.py:83-143`
- `rtp_llm/openai/openai_endpoint.py:39-100`
- `rtp_llm/embedding/embedding_endpoint.py`

### 3.3 HTTP 路由

源码位置：`rtp_llm/frontend/frontend_app.py:164-360`

主要路由包括：

| 路由 | 源码位置 | 作用 |
|---|---|---|
| `POST /` | `frontend_app.py:313-324` | 兼容普通 inference 请求 |
| `POST /chat/completions` | `frontend_app.py:326-336` | Chat Completion |
| `POST /v1/chat/completions` | `frontend_app.py:326-336` | OpenAI 兼容接口 |
| `POST /v1/batch/chat/completions` | `frontend_app.py:338-349` | 批量 Chat 请求 |
| `POST /batch_infer` | `frontend_app.py:351-360` | 批量推理 |
| `GET /v1/models` | `frontend_app.py:289-292` | 查询模型列表 |
| `/health`、`/health_check` | `frontend_app.py:199-222` | 健康检查 |

### 3.4 Chat Completion 请求处理

以 `/v1/chat/completions` 为例：

```text
HTTP POST /v1/chat/completions
  └─ FrontendApp.create_app().chat_completion()
       └─ FrontendServer.chat_completion()
            └─ OpenaiEndpoint.chat_completion()
                 ├─ chat renderer 构造 prompt
                 ├─ tokenizer 编码
                 ├─ GenerateConfig 生成
                 └─ BackendRPCServerVisitor / ModelRpcClient
                      └─ gRPC GenerateStreamCall
```

`OpenaiEndpoint` 负责把 OpenAI 风格的 messages 转换为模型需要的 prompt，并在返回时把 token 结果重新组织成 OpenAI response。

重点源码：

- `rtp_llm/frontend/frontend_server.py:270-302`
- `rtp_llm/openai/openai_endpoint.py` 中的 `chat_completion()` 及相关 renderer 逻辑
- `rtp_llm/openai/renderer_factory.py`
- `rtp_llm/openai/renderers/`

### 3.5 普通 inference 请求处理

普通请求从 `FrontendServer.inference()` 进入：

```text
FrontendApp.inference()
  └─ FrontendServer.inference()
       └─ FrontendWorker.inference(**req)
            ├─ RequestExtractor.extract_request()
            ├─ tokenizer / request config
            └─ FrontendWorker._inference()
                 └─ _yield_generate()
                      └─ Pipeline / BackendRPCServerVisitor
```

重点源码：

- `rtp_llm/frontend/frontend_server.py:228-256`
- `rtp_llm/frontend/frontend_worker.py:194-259`
- `rtp_llm/server/backend_rpc_server_visitor.py:20-80`

对于 Prefill/Decode 分离部署，`BackendRPCServerVisitor.route_ips()` 还会根据请求、KV Cache key、master service 和角色信息选择 Prefill/Decode Backend，位置在：

- `rtp_llm/server/backend_rpc_server_visitor.py:120-224`

---

## 4. 第三条调用链：Backend 初始化

### 4.1 BackendManager.start()

源码位置：`rtp_llm/server/backend_manager.py:26-139`

核心调用链：

```text
BackendManager.start()
  ├─ DistributedServer.start()
  ├─ EngineConfig.create()
  ├─ init_distributed_environment()       # world_size > 1 时
  ├─ get_world_info()
  ├─ update_worker_addrs()
  ├─ ModelFactory.create_model_config()
  ├─ ModelFactory.update_engine_config_from_model_config()
  ├─ create_propose_model_config()         # 可选
  ├─ ModelFactory.from_model_configs()
  └─ engine.start()
```

这里是 Backend 真正开始准备模型的地方。Frontend 侧的模型配置主要用于 tokenizer 和协议处理；Backend 侧会加载权重、初始化 Python model、创建 C++ engine。

### 4.2 ModelFactory 创建模型

源码位置：

- `rtp_llm/model_factory_register.py`
- `rtp_llm/model_factory.py:56-114`
- `rtp_llm/model_factory.py:173-245`
- `rtp_llm/model_factory.py:247-330`

模型注册机制如下：

```text
rtp_llm/models/__init__.py
  └─ import 各种模型类
       └─ register_model(...)
            └─ _model_factory[model_type] = model_cls
```

根据 `model_type`，`ModelFactory` 调用：

```python
model_cls._create_config(ckpt_path)
model_cls.from_config(...)
```

模型类型映射还支持根据 HuggingFace 的 `architectures` 和 repo 名称推导 RTP-LLM model type，相关代码在：

- `rtp_llm/model_factory_register.py:46-119`

### 4.3 BaseModel.load()

源码位置：`rtp_llm/models/base_model.py:139-190`

核心流程：

```text
BaseModel.load()
  ├─ _may_init_multimodal()
  ├─ _init_custom_module()
  ├─ create_model_loader()
  ├─ model_weights_loader.load_weights(device)
  ├─ WeightManager(...)
  └─ _create_python_model()
```

其中：

- `ModelLoader` 负责读取和转换权重。
- `WeightManager` 负责运行时权重管理。
- `_create_python_model()` 创建 Python 侧模型结构。
- `BaseModel._get_device_str()` 根据当前硬件返回 `cuda:N`、`npu:N` 或 `hip:N`。

权重加载入口：

- `rtp_llm/model_loader/loader.py:65-90`
- SafeTensors、量化、LoRA 等具体逻辑位于 `rtp_llm/model_loader/`。

### 4.4 创建 EngineConfig

源码位置：`rtp_llm/config/engine_config.py:17-190`

`EngineConfig` 是 Python 配置向 C++ 配置传递的聚合对象，包含：

- ParallelismConfig
- RuntimeConfig
- KVCacheConfig
- HWKernelConfig
- FMHAConfig
- MoEConfig
- SpeculativeExecutionConfig
- PDSepConfig
- GrpcConfig
- CacheStoreConfig
- LoadConfig

它在 `EngineConfig.create()` 中把 `PyEnvConfigs` 转换为 C++ binding 对象，并设置 Prefill/Decode 分离相关配置。

---

## 5. 第四条调用链：Python Engine 到 C++ Engine

### 5.1 创建 LanguageCppEngine

源码位置：

- `rtp_llm/model_factory.py:173-245`
- `rtp_llm/async_decoder_engine/engine_creator.py:18-50`
- `rtp_llm/async_decoder_engine/rpc_engine.py:18-73`

`ModelFactory.from_model_configs()` 最终调用 `create_engine()`：

```text
create_engine()
  ├─ torch.ops.rtp_llm.init_engine(alog_conf_path)
  └─ model.task_type == LANGUAGE_MODEL
       └─ LanguageCppEngine(...)
```

语言模型使用 `LanguageCppEngine`，Embedding 模型使用 `EmbeddingCppEngine`。

### 5.2 LanguageCppEngine._start()

源码位置：`rtp_llm/async_decoder_engine/rpc_engine.py:51-73`

```python
LanguageCppEngine._start()
  ├─ RtpLLMOp.start()
  └─ ft_op.start_http_server(...)
```

Python 封装层：`rtp_llm/ops/rtp_llm/rtp_llm_op.py:13-43`

```python
RtpLLMOp.start()
  └─ self.ft_op.init(model, engine_config, ...)
```

其中 `self.ft_op` 是 C++ pybind 暴露的 `RtpLLMOp` 对象。

### 5.3 C++ RtpLLMOp::init()

源码位置：`rtp_llm/cpp/pybind/multi_gpu_gpt/RtpLLMOp.cc:106-137`

```text
RtpLLMOp::init()
  ├─ initModel()
  │    ├─ 读取 model_config
  │    ├─ 读取 engine_config 各配置
  │    ├─ WeightsConverter.createGptWeights()
  │    └─ 生成 EngineInitParams
  ├─ initProposeModel()                   # 可选
  └─ 启动线程 initRPCServer()
```

这里把 Python 对象中的以下内容转换给 C++：

- `model_config`
- `weight.weights`
- `weight.global_weights`
- `py_model`
- `weight_manager`
- `py_eplb`
- 各类 C++ 配置对象

`initModel()` 的详细位置：`RtpLLMOp.cc:139-276`。

### 5.4 C++ RPC Server 初始化

源码位置：`rtp_llm/cpp/pybind/multi_gpu_gpt/RtpLLMOp.cc:283-335`

```text
RtpLLMOp::initRPCServer()
  ├─ role == PREFILL/DECODE
  │    └─ RemoteRpcServiceImpl
  └─ otherwise
       └─ LocalRpcServiceImpl

service.init()
  └─ LocalRpcServer::init()
       └─ new NormalEngine(...)
```

`LocalRpcServer::init()` 位置：

- `rtp_llm/cpp/model_rpc/LocalRpcServer.cc:14-57`

它会在释放 Python GIL 后创建 `NormalEngine`，这很重要，因为模型初始化和推理循环不应长期占用 Python GIL。

---

## 6. 第五条调用链：gRPC 请求进入 C++ Engine

### 6.1 LocalRpcServer::GenerateStreamCall()

源码位置：`rtp_llm/cpp/model_rpc/LocalRpcServer.cc:144-189`

请求处理链：

```text
ModelRpcClient
  └─ gRPC GenerateStreamCall
       └─ LocalRpcServer::GenerateStreamCall()
            ├─ QueryConverter::transQuery()
            ├─ 多模态特征处理
            ├─ engine_->enqueue(input)
            └─ pollStreamOutput(...)
```

关键代码：

```cpp
auto input = QueryConverter::transQuery(&input_pb);
generate_context.setStream(engine_->enqueue(input));
pollStreamOutput(...);
```

输入转换代码：

- `rtp_llm/cpp/model_rpc/QueryConverter.cc`
- `rtp_llm/cpp/model_rpc/QueryConverter.h`

输出轮询会不断调用 `stream->nextOutput()`，再通过 `QueryConverter::transResponse()` 转成 protobuf 并写回 gRPC stream。

### 6.2 BatchGenerateCall()

源码位置：`LocalRpcServer.cc:191-260`

批量请求路径：

```text
BatchGenerateCall()
  ├─ 多次 prepareInput()
  ├─ engine_->batchEnqueue(inputs)
  ├─ collectStreamOutput()
  └─ QueryConverter::transResponse()
```

注意：当前实现会按 streams 顺序串行收集最终结果，代码注释中也指出混合长度 batch 的并行收集仍有改进空间。

### 6.3 Prefill/Decode 分离

当角色为 `PREFILL` 或 `DECODE` 时，C++ 使用 `RemoteRpcServiceImpl`，其初始化继承 Local RPC 逻辑，同时增加：

- Cache Store
- worker 地址
- peer 信息
- 远程 KV Cache 交换

源码位置：

- `rtp_llm/cpp/model_rpc/RemoteRpcServer.cc:7-100`
- `rtp_llm/cpp/model_rpc/RemoteRpcServiceImpl.*`
- `rtp_llm/cpp/cache/connector/`
- `rtp_llm/cpp/disaggregate/`

---

## 7. 第六条调用链：NormalEngine 调度与执行

### 7.1 NormalEngine 初始化

源码位置：`rtp_llm/cpp/normal_engine/NormalEngine.cc:47-121`

构造函数主要完成：

```text
NormalEngine::NormalEngine()
  ├─ 可选 warm up
  ├─ initCacheManager()
  ├─ initExecutor()
  ├─ initScheduler()
  └─ startLoop()
```

相关成员定义：`rtp_llm/cpp/normal_engine/NormalEngine.h:25-104`。

### 7.2 KV Cache 初始化

源码位置：

- `NormalEngine.cc:277-365` 附近的 `initCacheManager()`
- `rtp_llm/cpp/cache/KVCacheManager.h:25-132`
- `rtp_llm/cpp/cache/CacheConfigCreator.*`

`KVCacheManager` 负责：

- 初始化 KV Cache 显存块。
- 分配和释放 block。
- block copy。
- 记录可用 token/block 数量。
- 处理 Prefix Cache。
- 与 Prefill/Decode 分离的 Cache Connector 协作。
- 支持混合 attention 的多组 KV Cache。

可以把它理解成推理引擎的显存资源管理器。

### 7.3 Scheduler

源码位置：

- `rtp_llm/cpp/normal_engine/NormalEngine.cc:144-155`
- `rtp_llm/cpp/engine_base/schedulers/SchedulerBase.h`
- `rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h`
- `rtp_llm/cpp/engine_base/schedulers/BatchDecodeScheduler.h`

调度器根据配置选择：

```text
use_batch_decode_scheduler == true
  └─ BatchDecodeScheduler
otherwise
  └─ FIFOScheduler
```

`FIFOScheduler` 内部维护：

- waiting streams
- loading cache streams
- running streams
- new streams

调度时不仅考虑请求先后，还会考虑：

- 最大生成 batch size
- 最大 batch token 数
- KV Cache 可用空间
- 输入长度
- 请求取消和超时
- Prefill/Decode 混合状态

### 7.4 后台推理循环

源码位置：`rtp_llm/cpp/normal_engine/NormalEngine.cc:368-502`

```text
NormalEngine::startLoop()
  └─ 创建 normal_engine_loop 线程
       └─ NormalEngine::loop()
            └─ while (running_)
                 └─ step()
```

`step()` 的主逻辑：

```text
step()
  ├─ scheduler_->schedule()
  ├─ DP 场景补 fake stream
  ├─ executor_->process(streams)
  ├─ profiler tick
  └─ reportMetrics()
```

重点代码：

- `NormalEngine.cc:388-400`：循环线程
- `NormalEngine.cc:437-502`：单步执行
- `NormalEngine.cc:405-435`：enqueue/batchEnqueue

---

## 8. 第七条调用链：NormalExecutor 到模型 forward

### 8.1 NormalExecutor 初始化

源码位置：`rtp_llm/cpp/normal_engine/NormalExecutor.cc:24-117`

```text
NormalExecutor::NormalExecutor()
  ├─ 创建 Sampler
  ├─ 创建 ExpertBalancer（MoE + EPLB 时）
  ├─ 构造 GptModelInitParams
  ├─ new PyWrappedModel(...)
  └─ 创建 NormalBatchStreamProcessor
```

当 `params.py_model` 存在时：

```cpp
model_.reset(new PyWrappedModel(model_init_params, params.py_model));
```

### 8.2 NormalExecutor::process()

源码位置：`rtp_llm/cpp/normal_engine/NormalExecutor.cc:119-234`

完整链路：

```text
NormalExecutor::process(streams)
  ├─ StreamGroups(streams)
  ├─ batch_stream_processor_->gatherModelInput()
  ├─ tpSyncModelInputs()
  ├─ cache_manager_->blockBatchCopy()
  ├─ model_->forward(model_input)
  ├─ Sampler::forward()
  └─ batch_stream_processor_->dispatch()
```

其中：

1. `gatherModelInput()` 把多个 GenerateStream 组织成一次模型输入。
2. `tpSyncModelInputs()` 确保 TP rank 之间使用一致的输入。
3. `blockBatchCopy()` 处理 KV Cache block 迁移。
4. `model_->forward()` 执行真正的 Transformer forward。
5. rank 0 负责采样和结果分发。
6. `dispatch()` 将 token、状态和辅助信息写回各个 stream。

主要协作类：

- `rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.*`
- `rtp_llm/cpp/models/Sampler.*`
- `rtp_llm/cpp/engine_base/stream/GenerateStream.*`
- `rtp_llm/cpp/engine_base/stream/StreamCacheResource.*`

---

## 9. 第八条调用链：PyWrappedModel 的输入转换和执行分流

### 9.1 PyWrappedModel 的角色

源码位置：

- 类定义：`rtp_llm/cpp/models/PyWrappedModel.h:38-131`
- 构造函数主体：`PyWrappedModel.h:124-330`
- forward：`rtp_llm/cpp/models/PyWrappedModel.cc:443-540`

`PyWrappedModel` 是 C++ Engine 和 Python Model 之间的适配器：

```text
C++ GptModelInputs
  └─ PyWrappedModel
       └─ PyModelInputs
            └─ Python model.forward()
```

它负责：

- 持有 Python model 对象。
- 调用 Python `initialize()`。
- 将 C++ 输入转换成 Python binding 输入。
- 绑定 KV Cache。
- 处理 host/device tensor 拷贝。
- 选择 eager 或 graph 路径。
- 对 hidden states 做后处理、layernorm、logits 和 TP 同步。

### 9.2 PyWrappedModel::forward()

源码位置：`PyWrappedModel.cc:443-529`

核心步骤：

```text
PyWrappedModel::forward(inputs)
  ├─ holdInputsHostBuffers()
  ├─ buildPyAttentionInputs()
  ├─ buildBertEmbeddingInputs()
  ├─ setupKVCacheForAttentionInputs()
  ├─ calculatePaddingOffset()
  ├─ fusedCopy()
  ├─ 构造 PyModelInputs
  ├─ graph_runner_->canRun()
  │    ├─ true  → graph_runner_->forward()
  │    └─ false → py_model.prepare_fmha_impl(..., false)
  │                 py_model.forward(...)
  ├─ hidden_states.clone()
  └─ callForwardPostLayers()
```

输入转换重点位置：`PyWrappedModel.cc:462-487`。

Graph/Eager 分流位置：`PyWrappedModel.cc:491-518`。

源码中的关键分支：

```cpp
if (enable_cuda_graph_ && graph_runner_->canRun(py_model_inputs, graph_state)) {
    py_model_outputs = graph_runner_->forward(py_model_inputs, graph_state);
} else {
    held_attn_pyobj_ = py_model_.attr("prepare_fmha_impl")(py_model_inputs, false);
    auto outputs = py_model_.attr("forward")(py_model_inputs, held_attn_pyobj_);
}
```

虽然变量名仍然叫 `cuda_graph`，但 Ascend 编译路径实际会使用 `AscendGraphRunner`。

---

## 10. Python 模型 forward 链路

### 10.1 Python 模型接口

Python 模型通常需要提供：

- `initialize(init_resources)`
- `prepare_fmha_impl(inputs, is_cuda_graph)`
- `forward(inputs, fmha_impl)`

通用实现可以从以下位置开始阅读：

- `rtp_llm/models_py/model_desc/module_base.py:94-151`
- `rtp_llm/models_py/modules/factory/attention/attn_factory.py:138-232`

### 10.2 Attention 工厂

`prepare_fmha_impl()` 会通过 `AttnImplFactory` 选择实际 attention 实现，并把 `is_cuda_graph` 写入 attention input：

```python
# models_py/modules/factory/attention/attn_factory.py:138-149
attn_inputs.is_cuda_graph = is_cuda_graph
```

因此 Graph 模式不仅是 C++ 的行为，Python attention 也需要知道当前是否处于图捕获/重放路径。

### 10.3 普通 eager 路径

```text
PyWrappedModel::forward()
  └─ py_model.prepare_fmha_impl(inputs, False)
  └─ py_model.forward(inputs, fmha_impl)
       └─ model layers
            └─ attention / FFN / MoE
                 └─ PyModelOutputs
```

Eager 路径的优点是兼容性好，可以处理动态形状和未被 Graph 覆盖的请求；缺点是每一步都会产生更多 Python/算子调度开销。

---

## 11. ACL Graph 专项调用链

当前分支的 Ascend Graph 代码位于：

- `rtp_llm/cpp/ascend_graph/`
- `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_decode.py`
- `rtp_llm/cpp/models/PyWrappedModel.h`

### 11.1 创建 AscendGraphRunner

源码位置：`rtp_llm/cpp/models/PyWrappedModel.h:190-313`

编译条件：

```cpp
#if USING_ASCEND
    graph_runner_ = new AscendGraphRunner(graph_params, py_instance);
#endif
```

当前 ACL Graph 的特点：

- 复用 `GraphBase` 和 `GraphParams`。
- 使用 `c10_npu::NPUGraph`。
- 主要用于 Decode。
- Prefill Graph 不支持时，`graph_runner_` 不创建，回退到 eager。

构造函数相关源码：

- `rtp_llm/cpp/ascend_graph/ascend_graph_runner.h:32-111`
- `rtp_llm/cpp/ascend_graph/ascend_graph_runner.cc:30-99`

### 11.2 GraphBase 抽象

源码位置：`rtp_llm/cpp/cuda_graph/cuda_graph_base.h:12-48`

```cpp
class GraphBase {
public:
    virtual void initCapture() = 0;
    virtual PyModelOutputs forward(...) = 0;
    virtual bool canRun(...) = 0;
};
```

`CudaGraphState` 这个名称是历史命名，但它被 CUDA、ROCm 和 Ascend Graph 共用。

### 11.3 ACL Graph 初始化和捕获

入口：`rtp_llm/cpp/ascend_graph/ascend_graph_runner.cc:385-419`

```text
AscendGraphRunner::initCapture()
  ├─ enable_graph_ 检查
  ├─ 计算 max_num_token_
  ├─ getDecodeBatchSizesToCapture()
  ├─ initCaptureAttentionInputs()
  ├─ 初始化持久化 capture memory
  ├─ initKernelInternalMemory()
  ├─ Python warmup forward
  └─ captureDecode()
```

批大小桶生成：`ascend_graph_runner.cc:119-142`

如果没有显式配置，代码会按预设的 batch size 生成捕获列表，例如：

```text
[1, 8, 16, 24, 32, ... , max_batch_size]
```

完整捕获链：

```text
captureDecode()
  └─ captureDecodeOneBatchSize(bs)
       └─ captureOneGraphInstance(bs, ...)
            ├─ Python warmup forward
            ├─ graphDeviceSynchronize()
            ├─ 切换 capture stream
            ├─ graphCaptureBegin(NPUGraph)
            ├─ Python model.forward()
            │    └─ AscendDecodeImpl
            │         └─ AscendDecodeAttnOp._forward_fia_graph()
            └─ graphCaptureEnd(NPUGraph)
```

相关源码：

- `ascend_graph_runner.cc:425-509`
- `ascend_graph_utils.h:22-82`

### 11.4 ACL Graph 的 Attention 实现

源码位置：`rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_decode.py`

`AscendDecodeImpl` 位于第 14 行，主要负责：

- RoPE
- KV Cache 写入
- FIA v2 attention
- Graph/Eager attention 分流

关键函数：

| 函数 | 位置 | 作用 |
|---|---:|---|
| `AscendDecodeImpl.prepare_cuda_graph()` | `ascend_decode.py:101-108` | 重放前准备 attention 输入并更新 FIA graph 参数 |
| `AscendDecodeImpl.forward()` | `ascend_decode.py:110-132` | 完成 QKV、RoPE、KV Cache 写入和 attention |
| `AscendDecodeAttnOp.forward()` | `ascend_decode.py:198-207` | 根据当前 stream 是否 capturing 选择路径 |
| `_forward_fia_graph()` | `ascend_decode.py:209-268` | 捕获 FIA v2 graph task |
| `update_graph_fia()` | `ascend_decode.py:270-295` | 更新动态 context length |
| `_forward_fia()` | `ascend_decode.py:297-316` | eager FIA v2 路径 |

Graph 捕获时使用：

```python
torch.npu.graph_task_group_begin(stream)
torch_npu.npu_fused_infer_attention_score_v2.out(...)
handle = torch.npu.graph_task_group_end(stream)
```

因此 ACL Graph 并不是简单地把整个 Python 函数“录制”下来，还需要 attention 算子本身支持 graph task 的参数更新。

### 11.5 Graph 可运行条件

源码位置：`rtp_llm/cpp/ascend_graph/ascend_graph_runner.cc:144-205`

`canRun()` 主要检查：

1. 是否开启 Graph。
2. 是否为 Decode；Prefill 直接回退 eager。
3. Speculative Decoding 的 target verify 条件是否满足。
4. Hybrid KV Cache group 数量是否匹配。
5. 实际 batch size 是否落在已捕获范围内。

batch size 选择逻辑：

```text
capture_range = [1, 8, 16, 32]
实际 batch = 5
  └─ lower_bound() → 选择 Graph 8

实际 batch = 40
  └─ 超出最大捕获范围 → canRun() == false → eager
```

### 11.6 Graph 重放

源码位置：

- `ascend_graph_runner.cc:516-527`：`replayGraph()` / `replayDecode()`
- `ascend_graph_runner.cc:561-705`：`prepareInputs()`
- `ascend_graph_runner.cc:708-728`：`forward()`

重放链：

```text
AscendGraphRunner::forward()
  ├─ prepareInputs()
  │    ├─ 等待上一次 Graph 完成
  │    ├─ 拷贝 input_ids
  │    ├─ 拷贝 sequence lengths
  │    ├─ 拷贝 KV block table
  │    ├─ 设置固定输入的 batch slice
  │    └─ attn_pyobj.prepare_cuda_graph()
  │         └─ AscendDecodeAttnOp.update_graph_fia()
  ├─ replayDecode()
  │    └─ ascend_graph::graphReplay()
  │         └─ NPUGraph::replay()
  └─ 读取并 clone hidden_states
```

核心设计是：

```text
动态请求数据
  └─ copy 到固定 Graph buffer
       └─ 更新 context length 和 KV block table
            └─ NPUGraph::replay()
```

### 11.7 Device Shims

源码位置：

- `rtp_llm/cpp/ascend_graph/ascend_graph_device_shims.h`
- `rtp_llm/cpp/ascend_graph/ascend_graph_device_shims.cc:42-136`

封装关系如下：

| RTP-LLM 抽象 | Ascend 实现 |
|---|---|
| `GraphStream` | `c10_npu::NPUStream` |
| `graphGetStreamFromPool()` | `c10_npu::getStreamFromPool()` |
| `graphGetCurrentStream()` | `c10_npu::getCurrentNPUStream()` |
| `graphCaptureBegin()` | `NPUGraph::capture_begin()` |
| `graphCaptureEnd()` | `NPUGraph::capture_end()` |
| `graphReplay()` | `NPUGraph::replay()` |
| `graphDeviceSynchronize()` | `aclrtSynchronizeDevice()` |
| `graphStreamSynchronize()` | `aclrtSynchronizeStream()` |
| `graphMemcpyAsync()` | `aclrtMemcpyAsync()` |

非 Ascend 平台提供空实现/stub，因此其他平台可以无条件依赖该模块，而不会真正实例化 Ascend Graph。

---

## 12. 一次 Decode 请求的完整时序

下面把最重要的 Decode 请求压缩成一条可以跟读源码的链路：

```text
1. Client
   └─ POST /v1/chat/completions

2. FrontendApp
   └─ frontend_app.py:326-336

3. FrontendServer
   └─ frontend_server.py:270-302

4. OpenaiEndpoint
   ├─ chat template
   ├─ tokenizer
   └─ GenerateConfig

5. FrontendWorker / Pipeline
   └─ BackendRPCServerVisitor / ModelRpcClient

6. C++ LocalRpcServer
   └─ LocalRpcServer.cc:144-189
       ├─ QueryConverter::transQuery()
       └─ engine_->enqueue(input)

7. NormalEngine
   └─ NormalEngine.cc:405-435
       └─ scheduler_->enqueue(stream)

8. NormalEngine loop
   └─ NormalEngine.cc:388-400
       └─ step()

9. Scheduler
   └─ scheduler_->schedule()

10. NormalExecutor
    └─ NormalExecutor.cc:119-234
        ├─ gatherModelInput()
        ├─ tpSyncModelInputs()
        └─ model_->forward()

11. PyWrappedModel
    └─ PyWrappedModel.cc:443-529
        ├─ 构造 PyModelInputs
        ├─ graph_runner_->canRun()
        └─ AscendGraphRunner::forward()

12. AscendGraphRunner
    └─ ascend_graph_runner.cc:708-728
        ├─ prepareInputs()
        ├─ update_graph_fia()
        └─ NPUGraph::replay()

13. NormalExecutor
    ├─ Sampler::forward()
    └─ dispatch()

14. LocalRpcServer
    └─ pollStreamOutput()
        └─ gRPC stream 返回 GenerateOutputsPB

15. Frontend
    └─ token decode / response renderer
        └─ HTTP 流式返回客户端
```

---

## 13. 关键模块速查表

| 层次 | 目录/文件 | 主要职责 |
|---|---|---|
| 启动 | `rtp_llm/start_server.py` | 解析配置、启动进程、健康检查 |
| Frontend | `rtp_llm/frontend/` | HTTP、OpenAI 协议、tokenizer、流式响应 |
| 请求路由 | `rtp_llm/server/backend_rpc_server_visitor.py` | DP、Prefill/Decode、Master 路由 |
| 模型工厂 | `rtp_llm/model_factory.py` | 创建 ModelConfig、模型和 Engine |
| 模型注册 | `rtp_llm/model_factory_register.py` | model type/HF architecture 映射 |
| 权重加载 | `rtp_llm/model_loader/` | SafeTensors、量化、LoRA、权重转换 |
| Python 模型 | `rtp_llm/models/`、`rtp_llm/models_py/` | 模型结构和 Python 算子 |
| Python Engine | `rtp_llm/async_decoder_engine/` | Python 到 C++ Engine 的封装 |
| PyBind | `rtp_llm/cpp/pybind/` | Python/C++ 边界 |
| RPC | `rtp_llm/cpp/model_rpc/` | gRPC 请求、流式输出、PD 分离 |
| 主引擎 | `rtp_llm/cpp/normal_engine/` | 调度循环、Executor、采样 |
| Scheduler | `rtp_llm/cpp/engine_base/schedulers/` | 请求队列和动态批处理 |
| KV Cache | `rtp_llm/cpp/cache/` | block 分配、复制、缓存连接器 |
| C++/Python 模型桥 | `rtp_llm/cpp/models/PyWrappedModel.*` | 输入转换、forward、Graph/Eager 分流 |
| CUDA Graph | `rtp_llm/cpp/cuda_graph/` | CUDA/ROCm Graph 基础实现 |
| ACL Graph | `rtp_llm/cpp/ascend_graph/` | Ascend NPUGraph 捕获和重放 |
| Ascend Attention | `rtp_llm/models_py/modules/factory/attention/ascend_impl/` | FIA v2、KV Cache、Graph task 更新 |

---

## 14. 建议的学生阅读顺序

### 第一阶段：只看主干

1. `README_cn.md`
2. `rtp_llm/start_server.py`
3. `rtp_llm/frontend/frontend_app.py`
4. `rtp_llm/server/backend_manager.py`
5. `rtp_llm/model_factory.py`
6. `rtp_llm/async_decoder_engine/rpc_engine.py`

目标：回答“服务如何启动、模型在哪里创建、Frontend 和 Backend 如何分工”。

### 第二阶段：看 C++ 推理循环

1. `rtp_llm/cpp/pybind/multi_gpu_gpt/RtpLLMOp.cc`
2. `rtp_llm/cpp/model_rpc/LocalRpcServer.cc`
3. `rtp_llm/cpp/normal_engine/NormalEngine.cc`
4. `rtp_llm/cpp/normal_engine/NormalExecutor.cc`
5. `rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.*`
6. `rtp_llm/cpp/cache/KVCacheManager.*`

目标：回答“一个请求如何进入 scheduler，如何变成一次模型执行”。

### 第三阶段：看 Python/C++ 桥接

1. `rtp_llm/cpp/models/PyWrappedModel.h`
2. `rtp_llm/cpp/models/PyWrappedModel.cc`
3. `rtp_llm/models_py/model_desc/module_base.py`
4. `rtp_llm/models_py/modules/factory/attention/attn_factory.py`

目标：回答“C++ 输入如何交给 Python 模型，hidden states 如何返回”。

### 第四阶段：看 ACL Graph

1. `rtp_llm/cpp/cuda_graph/cuda_graph_base.h`
2. `rtp_llm/cpp/ascend_graph/ascend_graph_runner.h`
3. `rtp_llm/cpp/ascend_graph/ascend_graph_runner.cc`
4. `rtp_llm/cpp/ascend_graph/ascend_graph_device_shims.*`
5. `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_decode.py`

目标：回答“Graph 何时创建、何时捕获、何时判断可用、如何更新动态输入并重放”。

---

## 15. 走读结论

这份代码的主线可以概括为：

```text
HTTP 请求
  → Python Frontend 做协议和 tokenizer 处理
  → gRPC 传给 C++ Backend
  → NormalEngine 调度请求和 KV Cache
  → NormalExecutor 组织 batch
  → PyWrappedModel 调用 Python 模型
  → eager 或 Graph 执行 Transformer
  → C++ 采样并产生 token
  → gRPC/HTTP 流式返回
```

ACL Graph 只是这条主链中的一个优化分支：

```text
PyWrappedModel::forward()
  ├─ 条件不满足：Python eager forward
  └─ 条件满足：AscendGraphRunner
       ├─ 固定 buffer
       ├─ 更新 KV block table/context length
       ├─ 更新 FIA graph task
       └─ NPUGraph::replay()
```

理解这两条主线后，再去看具体模型的 layer 实现、量化 kernel、MoE 通信和 Prefill/Decode 分离，就不会迷失在大量底层代码中。

> 最重要的阅读方法：先沿着 `Frontend → RPC → NormalEngine → NormalExecutor → PyWrappedModel` 走通一条请求，再深入某个优化点。不要一开始就从某个 attention kernel 或设备 shim 反向阅读整个仓库。
