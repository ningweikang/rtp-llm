 阶段一：ACL Graph 总调用链                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                           
 ### 1. 入口 —— PyWrappedModel.h 构造函数（L190–313）                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                           
 ```cpp                                                                                                                                                                                                                                                                    
   if (enable_cuda_graph_) {                                                                                                                                                                                                                                               
       #if USING_CUDA || USING_ROCM                                                                                                                                                                                                                                        
           graph_runner_ = new CudaGraphRunner(...);      // CUDA 平台                                                                                                                                                                                                     
       #elif USING_ASCEND                                                                                                                                                                                                                                                  
           if (is_prefill_cuda_graph_mode) {                                                                                                                                                                                                                               
               // ACL Graph 仅支持 decode → 直接警告并跳过，走 eager                                                                                                                                                                                                       
           } else {                                                                                                                                                                                                                                                        
               graph_runner_ = new AscendGraphRunner(graph_params, py_instance);  // L287                                                                                                                                                                                  
           }                                                                                                                                                                                                                                                               
       #endif                                                                                                                                                                                                                                                              
       ...                                                                                                                                                                                                                                                                 
       graph_runner_->initCapture();   // L312：捕获阶段在这里触发                                                                                                                                                                                                         
   }                                                                                                                                                                                                                                                                       
 ```                                                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                           
 关键点：                                                                                                                                                                                                                                                                  
 - 平台由编译宏 USING_ASCEND 分流，GraphBase* graph_runner_ 是统一接口                                                                                                                                                                                                     
 - ACL Graph 刻意拒绝 prefill 图模式（构造器里也 double-check 抛异常），只做 decode                                                                                                                                                                                        
 - graph_params 关键字段：decode_capture_batch_sizes（分桶列表）、max_context_batch_size、tokens_per_block                                                                                                                                                                 
                                                                                                                                                                                                                                                                           
 ### 2. initCapture() —— 捕获阶段（L404–437）                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                           
 ```                                                                                                                                                                                                                                                                       
   initCapture()                                                                                                                                                                                                                                                           
    ├─ capture_range_ = getDecodeBatchSizesToCapture()   // 分桶：1,8,16,24,32,48,64...max_bs                                                                                                                                                                              
    ├─ 分配捕获期持久 tensor（input_ids、kv block tables、cu_seqlens 等）                                                                                                                                                                                                  
    ├─ 一次 warmup forward（触发 kernel 惰性初始化、确定 dtype）                                                                                                                                                                                                           
    └─ captureDecode()                                                                                                                                                                                                                                                     
        ├─ 先 try_emplace 全部桶（占位）                                                                                                                                                                                                                                   
        ├─ 按 大→小 顺序（先大桶确定 mempool 高水位）                                                                                                                                                                                                                      
        │    prepareCaptureInputs() → 从共享 capture_mem_hold_ 切片出该桶输入                                                                                                                                                                                              
        │    captureOneGraphInstance(bs)     ← 每个桶一张图                                                                                                                                                                                                                
        │       ├─ WarmUp：py_forward_method_ 跑 2 次                                                                                                                                                                                                                      
        │       ├─ graphCaptureBegin(graph, pool, Relaxed)   ← NPUGraph 捕获开始                                                                                                                                                                                           
        │       │    └─ py_forward_method_(inputs, attn_pyobj)   ← Python forward                                                                                                                                                                                          
        │       │         └─ ascend_decode.py L249: torch.npu.graph_task_group_begin/end                                                                                                                                                                                   
        │       └─ graphCaptureEnd(graph)                                                                                                                                                                                                                                  
        └─ replayAndSyncCheck(bs)  // 捕获后立即重放验证                                                                                                                                                                                                                   
 ```                                                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                           
 ### 3. 推理期 forward() —— PyWrappedModel.cc L493 切换                                                                                                                                                                                                                    
                                                                                                                                                                                                                                                                           
 ```cpp                                                                                                                                                                                                                                                                    
   if (enable_cuda_graph_ && graph_runner_->canRun(py_model_inputs, graph_state)) {                                                                                                                                                                                        
       py_model_inputs.attention_inputs.is_s_padded = true;                                                                                                                                                                                                                
       py_model_outputs = graph_runner_->forward(...);    // 图路径                                                                                                                                                                                                        
   } else {                                                                                                                                                                                                                                                                
       py_model_.attr("forward")(...)                     // eager 路径（fallback）                                                                                                                                                                                        
   }                                                                                                                                                                                                                                                                       
 ```                                                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                           
 canRun() 判定（L135–169，失败即回退 eager）：                                                                                                                                                                                                                             
 1. 投机解码 target verify 路径特殊处理                                                                                                                                                                                                                                    
 2. !enable_graph_ → false                                                                                                                                                                                                                                                 
 3. is_prefill → false（decode-only 的核心约束）                                                                                                                                                                                                                           
 4. hybrid KV cache 组数不匹配 → false                                                                                                                                                                                                                                     
 5. tryGetRealGraphDecodeBatchSize()：实际 bs 在 capture_range_ 里做 lower_bound 向上取桶（bs=5 用 bs=8 的图），超出最大桶 → false                                                                                                                                         
                                                                                                                                                                                                                                                                           
 ### 4. 图路径 forward() —— 重放（L635–666）                                                                                                                                                                                                                               
                                                                                                                                                                                                                                                                           
 ```                                                                                                                                                                                                                                                                       
   AscendGraphRunner::forward(inputs, state)                                                                                                                                                                                                                               
    ├─ prepareInputs()     ← 把真实请求数据拷进捕获时固定的持久 buffer                                                                                                                                                                                                     
    │    ├─ forward_event_.synchronize()    // 等上次重放完成，防止覆盖未消费的 buffer                                                                                                                                                                                     
    │    ├─ 清零 block tables（防脏数据污染）                                                                                                                                                                                                                              
    │    ├─ copyTensorSlice() × N：D2D 拷贝（input_ids、cu_seqlens、block tables…）                                                                                                                                                                                        
    │    ├─ H2H pinned 拷贝（input_lengths、sequence_lengths…）                                                                                                                                                                                                            
    │    ├─ padding 区清零（真实 bs < 桶 bs 时）                                                                                                                                                                                                                           
    │    └─ attn_pyobj.attr("prepare_cuda_graph")(attn_inputs)   ← 关键！Python 侧更新                                                                                                                                                                                     
    ├─ replayDecode(bs) → replayGraph(bs) → graphReplay(NPUGraph)   // 零开销重放                                                                                                                                                                                          
    └─ 从持久 buffer 切片 clone 出 hidden_states                                                                                                                                                                                                                           
        forward_event_.record(...)          // 供下次 prepareInputs 等待                                                                                                                                                                                                   
 ```                                                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                           
 ### 5. Python 侧闭环（ascend_decode.py）                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                           
 - Capture 时（L249/261）：torch.npu.graph_task_group_begin(stream) / graph_task_group_end(stream) 包住 FIA v2 .out()                                                                                                                                                      
 - Replay 时（L282/294）：torch.npu.graph_task_update_begin(us, handle) / graph_task_update_end(us) 动态更新 context_lens（这正是 commit 80b8aa9e 修的长序列精度问题）                                                                                                     
 - prepare_cuda_graph(attn_inputs)（L101）：每次重放前被 C++ 调用，把新拷贝的输入同步给 attention 实现                                                                                                                                                                     
                                                                                                                                                                                                                                                                           
 ────────────────────────────────────────────────────────────────────────────────                                                                                                                                                                                          
                                                                                                                                                                                                                                                                           
 阶段一总结：一图流                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                           
 ```                                                                                                                                                                                                                                                                       
   启动 → PyWrappedModel 构造                                                                                                                                                                                                                                              
           ├─ [USING_ASCEND] new AscendGraphRunner → initCapture()                                                                                                                                                                                                         
           │     └─ 按 batch size 分桶 → 每桶 warmup×2 + NPUGraph 捕获                                                                                                                                                                                                     
           │           └─ capture 内跑 Python forward → graph_task_group_begin/end 包 FIA v2                                                                                                                                                                               
           └─ 推理 forward()                                                                                                                                                                                                                                               
                 ├─ canRun()?  prefill/超桶/组数不匹配 → eager fallback                                                                                                                                                                                                    
                 └─ 是 → prepareInputs()（拷贝数据到持久 buffer + prepare_cuda_graph 更新）                                                                                                                                                                                
                         → graphReplay() 零开销重放 → 读回 hidden_states                                                                                                                                                                                                   
 ```                                                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                           
 核心设计思想：捕获期把所有输入/输出固定在持久内存上，重放期只做"搬数据进固定槽位 + 重放 + 搬结果出来"，把 Python/kernel 调度开销全部消除 —— 与 CUDA Graph 的 capture/replay 思想完全一致，区别仅在于用 c10_npu::NPUGraph + aclrtMemcpyAsync。    


 ACL Graph 调用链总结                                                                                   
                                                                                                        
 ACL Graph 的运行过程分为两条路径：                                                                     
                                                                                                        
 ```text                                                                                                
   服务/Executor                                                                                        
     └─ PyWrappedModel                                                                                  
          ├─ 初始化阶段：创建 Runner → 捕获各 batch size 的 Graph                                       
          └─ 推理阶段：canRun()                                                                         
               ├─ 可以运行 → prepareInputs() → replay Graph                                             
               └─ 不可以运行 → Python eager forward                                                     
 ```                                                                                                    
                                                                                                        
 ────────────────────────────────────────────────────────────────────────────────                       
                                                                                                        
 1. Executor 调用模型                                                                                   
                                                                                                        
 ### 代码位置                                                                                           
                                                                                                        
 rtp_llm/cpp/normal_engine/NormalExecutor.cc:99                                                         
                                                                                                        
 ```cpp                                                                                                 
   model_.reset(new PyWrappedModel(model_init_params, params.py_model));                                
 ```                                                                                                    
                                                                                                        
 模型推理调用：                                                                                         
                                                                                                        
 rtp_llm/cpp/normal_engine/NormalExecutor.cc:169                                                        
                                                                                                        
 ```cpp                                                                                                 
   model_output = std::move(model_->forward(model_input));                                              
 ```                                                                                                    
                                                                                                        
 进入：                                                                                                 
                                                                                                        
 ```text                                                                                                
   NormalExecutor::process()                                                                            
       └─ PyWrappedModel::forward()                                                                     
 ```                                                                                                    
                                                                                                        
 ────────────────────────────────────────────────────────────────────────────────                       
                                                                                                        
 2. PyWrappedModel 创建 AscendGraphRunner                                                               
                                                                                                        
 ### 代码位置                                                                                           
                                                                                                        
 rtp_llm/cpp/models/PyWrappedModel.h:190-313                                                            
                                                                                                        
 核心逻辑：                                                                                             
                                                                                                        
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
           graph_params.max_seq_len =                                                                   
               params.max_seq_len;                                                                      
           graph_params.max_context_batch_size =                                                        
               params.concurrency_config.concurrency_limit;                                             
           graph_params.decode_capture_batch_sizes =                                                    
               params.hw_kernel_config.decode_capture_batch_sizes;                                      
                                                                                                        
           graph_runner_ = new AscendGraphRunner(graph_params, py_instance);                            
       }                                                                                                
   #endif                                                                                               
                                                                                                        
       if (graph_runner_ != nullptr) {                                                                  
           graph_runner_->initCapture();                                                                
       }                                                                                                
   }                                                                                                    
 ```                                                                                                    
                                                                                                        
 关键点：                                                                                               
                                                                                                        
 - graph_runner_ 类型是统一接口 GraphBase*                                                              
 - Ascend 使用 AscendGraphRunner                                                                        
 - ACL Graph 当前主要支持 decode                                                                        
 - prefill 通常回退到 eager                                                                             
 - decode_capture_batch_sizes 决定需要捕获哪些 batch size                                               
                                                                                                        
 ────────────────────────────────────────────────────────────────────────────────                       
                                                                                                        
 3. GraphBase 统一接口                                                                                  
                                                                                                        
 ### 代码位置                                                                                           
                                                                                                        
 rtp_llm/cpp/cuda_graph/cuda_graph_base.h:12-48                                                         
                                                                                                        
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
                                                                                                        
 虽然结构体名称仍叫 CudaGraphState，但 Ascend ACL Graph 也复用了这套接口。                              
                                                                                                        
 ────────────────────────────────────────────────────────────────────────────────                       
                                                                                                        
 4. AscendGraphRunner 构造                                                                              
                                                                                                        
 ### 代码位置                                                                                           
                                                                                                        
 rtp_llm/cpp/ascend_graph/ascend_graph_runner.cc:30-93                                                  
                                                                                                        
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
                                                                                                        
 这里保存了两个 Python 方法：                                                                           
                                                                                                        
 ```text                                                                                                
   py_attn_pyobj_method_ → Python prepare_fmha_impl()                                                   
   py_forward_method_    → Python forward()                                                             
 ```                                                                                                    
                                                                                                        
 后续捕获和重放都通过这两个对象进入 Python。                                                            
                                                                                                        
 ────────────────────────────────────────────────────────────────────────────────                       
                                                                                                        
 一、初始化阶段：捕获 Graph                                                                             
                                                                                                        
 5. initCapture() 总入口                                                                                
                                                                                                        
 ### 代码位置                                                                                           
                                                                                                        
 rtp_llm/cpp/ascend_graph/ascend_graph_runner.cc:385-419                                                
                                                                                                        
 ```cpp                                                                                                 
   void AscendGraphRunner::initCapture() {                                                              
       if (!enable_graph_) {                                                                            
           return;                                                                                      
       }                                                                                                
                                                                                                        
       max_num_token_ = max_bs_ * num_tokens_per_bs_;                                                   
                                                                                                        
       capture_range_ =                                                                                 
           getDecodeBatchSizesToCapture();                                                              
                                                                                                        
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
           capture_mem_hold_.py_model_inputs_,                                                          
           attn_pyobj);                                                                                 
                                                                                                        
       captureDecode();                                                                                 
   }                                                                                                    
 ```                                                                                                    
                                                                                                        
 初始化阶段主要做四件事：                                                                               
                                                                                                        
 1. 确定 batch size 桶                                                                                  
 2. 分配固定输入内存                                                                                    
 3. Python forward 预热                                                                                 
 4. 对每个 batch size 捕获一张图                                                                        
                                                                                                        
 ────────────────────────────────────────────────────────────────────────────────                       
                                                                                                        
 6. 生成 batch size 桶                                                                                  
                                                                                                        
 ### 代码位置                                                                                           
                                                                                                        
 rtp_llm/cpp/ascend_graph/ascend_graph_runner.cc:119-142                                                
                                                                                                        
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
   实际 batch size：5                                                                                   
   捕获的 Graph：8                                                                                      
 ```                                                                                                    
                                                                                                        
 运行时使用大于等于实际 batch size 的最小桶。                                                           
                                                                                                        
 ────────────────────────────────────────────────────────────────────────────────                       
                                                                                                        
 7. 按 batch size 捕获 Graph                                                                            
                                                                                                        
 ### 代码位置                                                                                           
                                                                                                        
 rtp_llm/cpp/ascend_graph/ascend_graph_runner.cc:429-464                                                
                                                                                                        
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
               inputs,                                                                                  
               bs,                                                                                      
               bs * num_tokens_per_bs_);                                                                
                                                                                                        
           graph_instances_[bs].mem_hold_ =                                                             
               createMemHold(                                                                           
                   inputs,                                                                              
                   bs * num_tokens_per_bs_);                                                            
                                                                                                        
           graph_instances_[bs].mem_hold_.attn_pyobj_ =                                                 
               py_attn_pyobj_method_(                                                                   
                   graph_instances_[bs]                                                                 
                       .mem_hold_                                                                       
                       .py_model_inputs_,                                                               
                   true);                                                                               
                                                                                                        
           captureDecodeOneBatchSize(bs);                                                               
           replayAndSyncCheck(bs, "batch size");                                                        
       }                                                                                                
   }                                                                                                    
 ```                                                                                                    
                                                                                                        
 每个 batch size 对应一个：                                                                             
                                                                                                        
 ```cpp                                                                                                 
   std::unordered_map<int, AscendGraphInstance>                                                         
       graph_instances_;                                                                                
 ```                                                                                                    
                                                                                                        
 定义位置：                                                                                             
                                                                                                        
 rtp_llm/cpp/ascend_graph/ascend_graph_runner.h:90                                                      
                                                                                                        
 ────────────────────────────────────────────────────────────────────────────────                       
                                                                                                        
 8. 捕获单张 Graph                                                                                      
                                                                                                        
 ### 代码位置                                                                                           
                                                                                                        
 rtp_llm/cpp/ascend_graph/ascend_graph_runner.cc:466-509                                                
                                                                                                        
 ```cpp                                                                                                 
   void AscendGraphRunner::captureOneGraphInstance(                                                     
       int key,                                                                                         
       const char* key_type) {                                                                          
                                                                                                        
       auto inputs =                                                                                    
           graph_instances_[key].mem_hold_                                                              
               .py_model_inputs_;                                                                       
                                                                                                        
       // Warmup                                                                                        
       py_forward_method_(inputs, attn_pyobj);                                                          
       py_forward_method_(inputs, attn_pyobj);                                                          
                                                                                                        
       ascend_graph::graphDeviceSynchronize();                                                          
                                                                                                        
       ascend_graph::AscendGraphStreamLife                                                              
           stream_life(capture_stream_);                                                                
                                                                                                        
       auto& graph =                                                                                    
           graph_instances_[key].graph_;                                                                
                                                                                                        
       ascend_graph::graphCaptureBegin(                                                                 
           graph,                                                                                       
           shared_graph_pool_,                                                                          
           ascend_graph::GraphCaptureMode::Relaxed);                                                    
                                                                                                        
       auto py_outputs_obj =                                                                            
           py_forward_method_(inputs, attn_pyobj);                                                      
                                                                                                        
       outputs =                                                                                        
           py_outputs_obj.cast<PyModelOutputs>();                                                       
                                                                                                        
       graph_instances_[key]                                                                            
           .mem_hold_                                                                                   
           .decoder_layer_hidden_states_                                                                
           .copy_(outputs.hidden_states);                                                               
                                                                                                        
       ascend_graph::graphCaptureEnd(graph);                                                            
   }                                                                                                    
 ```                                                                                                    
                                                                                                        
 这里真正开始 ACL Graph 捕获：                                                                          
                                                                                                        
 ```text                                                                                                
   graphCaptureBegin()                                                                                  
       └─ Python model forward()                                                                        
            └─ Ascend attention                                                                         
                 └─ FIA v2                                                                              
 ```                                                                                                    
                                                                                                        
 ────────────────────────────────────────────────────────────────────────────────                       
                                                                                                        
 二、Python 侧 Graph 捕获                                                                               
                                                                                                        
 9. Python prepare_fmha_impl                                                                            
                                                                                                        
 ### 代码位置                                                                                           
                                                                                                        
 rtp_llm/models_py/model_desc/module_base.py:95-103                                                     
                                                                                                        
 ```python                                                                                              
   def prepare_fmha_impl(                                                                               
       self,                                                                                            
       inputs: PyModelInputs,                                                                           
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
                                                                                                        
 工厂会设置：                                                                                           
                                                                                                        
 ### 代码位置                                                                                           
                                                                                                        
 rtp_llm/models_py/modules/factory/attention/attn_factory.py:144-149                                    
                                                                                                        
 ```python                                                                                              
   attn_inputs.is_cuda_graph = is_cuda_graph                                                            
 ```                                                                                                    
                                                                                                        
 因此捕获时：                                                                                           
                                                                                                        
 ```text                                                                                                
   is_cuda_graph = True                                                                                 
 ```                                                                                                    
                                                                                                        
 Python attention 会进入图模式。                                                                        
                                                                                                        
 ────────────────────────────────────────────────────────────────────────────────                       
                                                                                                        
 10. AscendDecodeImpl 处理 Graph 输入                                                                   
                                                                                                        
 ### 代码位置                                                                                           
                                                                                                        
 rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_decode.py:101-108                       
                                                                                                        
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
                                                                                                        
 该函数由 C++ 每次 replay 前调用，用于更新 FIA 的动态上下文长度。                                       
                                                                                                        
 ────────────────────────────────────────────────────────────────────────────────                       
                                                                                                        
 11. FIA v2 Graph 捕获                                                                                  
                                                                                                        
 ### 代码位置                                                                                           
                                                                                                        
 rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_decode.py:198-262                       
                                                                                                        
 ```python                                                                                              
   def forward(self, q, kv_cache, use_graph=False):                                                     
       if use_graph and torch.npu.is_current_stream_capturing():                                        
           return self._forward_fia_graph(                                                              
               q, kv_cache, block_table, context_lens)                                                  
                                                                                                        
       return self._forward_fia(                                                                        
           q, kv_cache, block_table, context_lens)                                                      
 ```                                                                                                    
                                                                                                        
 Graph 捕获部分：                                                                                       
                                                                                                        
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
                                                                                                        
 调用位置：                                                                                             
                                                                                                        
 ```text                                                                                                
   _forward_fia_graph()                                                                                 
       └─ graph_task_group_begin()                                                                      
       └─ FIA v2 .out()                                                                                 
       └─ graph_task_group_end()                                                                        
 ```                                                                                                    
                                                                                                        
 ────────────────────────────────────────────────────────────────────────────────                       
                                                                                                        
 三、推理阶段：选择 Graph 或 Eager                                                                      
                                                                                                        
 12. PyWrappedModel::forward                                                                            
                                                                                                        
 ### 代码位置                                                                                           
                                                                                                        
 rtp_llm/cpp/models/PyWrappedModel.cc:443-518                                                           
                                                                                                        
 ```cpp                                                                                                 
   GptModelOutputs                                                                                      
   PyWrappedModel::forward(                                                                             
       const GptModelInputs& inputs) {                                                                  
                                                                                                        
       auto py_model_inputs =                                                                           
           PyModelInputs({                                                                              
               token_ids,                                                                               
               input_hiddens,                                                                           
               attention_inputs,                                                                        
               bert_embedding_inputs                                                                    
           });                                                                                          
                                                                                                        
       CudaGraphState graph_state;                                                                      
                                                                                                        
       if (enable_cuda_graph_ &&                                                                        
           graph_runner_->canRun(                                                                       
               py_model_inputs,                                                                         
               graph_state)) {                                                                          
                                                                                                        
           py_model_inputs                                                                              
               .attention_inputs                                                                        
               .is_s_padded = true;                                                                     
                                                                                                        
           py_model_outputs =                                                                           
               graph_runner_->forward(                                                                  
                   py_model_inputs,                                                                     
                   graph_state);                                                                        
       } else {                                                                                         
           held_attn_pyobj_ =                                                                           
               py_model_.attr(                                                                          
                   "prepare_fmha_impl")(                                                                
                       py_model_inputs,                                                                 
                       false);                                                                          
                                                                                                        
           auto py_model_forward =                                                                      
               py_model_.attr("forward");                                                               
                                                                                                        
           auto outputs =                                                                               
               py_model_forward(                                                                        
                   py_model_inputs,                                                                     
                   held_attn_pyobj_);                                                                   
       }                                                                                                
   }                                                                                                    
 ```                                                                                                    
                                                                                                        
 分支关系：                                                                                             
                                                                                                        
 ```text                                                                                                
   canRun() == true                                                                                     
       └─ AscendGraphRunner::forward()                                                                  
            └─ replay Graph                                                                             
                                                                                                        
   canRun() == false                                                                                    
       └─ Python prepare_fmha_impl(..., false)                                                          
       └─ Python model.forward()                                                                        
            └─ eager FIA                                                                                
 ```                                                                                                    
                                                                                                        
 ────────────────────────────────────────────────────────────────────────────────                       
                                                                                                        
 13. canRun() 判断条件                                                                                  
                                                                                                        
 ### 代码位置                                                                                           
                                                                                                        
 rtp_llm/cpp/ascend_graph/ascend_graph_runner.cc:144-205                                                
                                                                                                        
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
                                                                                                        
       // Hybrid KV Cache 组数不匹配时回退                                                              
       if (group_count_mismatch) {                                                                      
           return false;                                                                                
       }                                                                                                
                                                                                                        
       return tryGetRealGraphDecodeBatchSize(                                                           
           inputs, state);                                                                              
   }                                                                                                    
 ```                                                                                                    
                                                                                                        
 batch size 桶选择：                                                                                    
                                                                                                        
 ### 代码位置                                                                                           
                                                                                                        
 rtp_llm/cpp/ascend_graph/ascend_graph_runner.cc:144-169                                                
                                                                                                        
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
                                                                                                        
 例如：                                                                                                 
                                                                                                        
 ```text                                                                                                
   capture_range = [1, 8, 16, 32]                                                                       
                                                                                                        
   实际 batch = 5  → 使用 Graph 8                                                                       
   实际 batch = 16 → 使用 Graph 16                                                                      
   实际 batch = 40 → 超出范围，走 eager                                                                 
 ```                                                                                                    
                                                                                                        
 ────────────────────────────────────────────────────────────────────────────────                       
                                                                                                        
 四、推理阶段：准备输入并重放                                                                           
                                                                                                        
 14. AscendGraphRunner::forward                                                                         
                                                                                                        
 ### 代码位置                                                                                           
                                                                                                        
 rtp_llm/cpp/ascend_graph/ascend_graph_runner.cc:708-728                                                
                                                                                                        
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
                                                                                                        
 ────────────────────────────────────────────────────────────────────────────────                       
                                                                                                        
 15. prepareInputs() 数据搬运                                                                           
                                                                                                        
 ### 代码位置                                                                                           
                                                                                                        
 rtp_llm/cpp/ascend_graph/ascend_graph_runner.cc:561-705                                                
                                                                                                        
 核心操作：                                                                                             
                                                                                                        
 ```cpp                                                                                                 
   forward_event_.synchronize();                                                                        
                                                                                                        
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
                                                                                                        
 复制设备端输入：                                                                                       
                                                                                                        
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
                                                                                                        
 ascend_graph_runner.cc:701-705                                                                         
                                                                                                        
 ```cpp                                                                                                 
   attn_pyobj.attr(                                                                                     
       "prepare_cuda_graph")(                                                                           
           py_model_inputs.attention_inputs);                                                           
 ```                                                                                                    
                                                                                                        
 这会进入：                                                                                             
                                                                                                        
 ```text                                                                                                
   C++ prepareInputs()                                                                                  
       └─ AscendDecodeImpl.prepare_cuda_graph()                                                         
            └─ AscendDecodeAttnOp.update_graph_fia()                                                    
 ```                                                                                                    
                                                                                                        
 ────────────────────────────────────────────────────────────────────────────────                       
                                                                                                        
 16. 动态更新 FIA Graph 参数                                                                            
                                                                                                        
 ### 代码位置                                                                                           
                                                                                                        
 rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_decode.py:270-294                       
                                                                                                        
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
       └─ 更新真实 context length                                                                       
       └─ 更新 FIA graph task                                                                           
       └─ 解决不同请求长度下的注意力参数变化                                                            
 ```                                                                                                    
                                                                                                        
 ────────────────────────────────────────────────────────────────────────────────                       
                                                                                                        
 17. 重放 Graph                                                                                         
                                                                                                        
 ### 代码位置                                                                                           
                                                                                                        
 rtp_llm/cpp/ascend_graph/ascend_graph_runner.cc:516-527                                                
                                                                                                        
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
                                                                                                        
 最终调用：                                                                                             
                                                                                                        
 ```cpp                                                                                                 
   graph.replay();                                                                                      
 ```                                                                                                    
                                                                                                        
 ────────────────────────────────────────────────────────────────────────────────                       
                                                                                                        
 五、底层 Ascend Graph 封装                                                                             
                                                                                                        
 18. Device Shims                                                                                       
                                                                                                        
 ### 头文件位置                                                                                         
                                                                                                        
 rtp_llm/cpp/ascend_graph/ascend_graph_device_shims.h:34-79                                             
                                                                                                        
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
                                                                                                        
 ### 实现位置                                                                                           
                                                                                                        
 rtp_llm/cpp/ascend_graph/ascend_graph_device_shims.cc:43-105                                           
                                                                                                        
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
                                                                                                        
 这些封装将上层逻辑映射到：                                                                             
                                                                                                        
 ```text                                                                                                
   c10_npu::NPUGraph                                                                                    
   c10_npu::NPUStream                                                                                   
   aclrtMemcpyAsync                                                                                     
   aclrtSynchronizeDevice                                                                               
 ```                                                                                                    
                                                                                                        
 ────────────────────────────────────────────────────────────────────────────────                       
                                                                                                        
 六、核心数据结构                                                                                       
                                                                                                        
 AscendGraphRunner                                                                                      
                                                                                                        
 ### 位置                                                                                               
                                                                                                        
 rtp_llm/cpp/ascend_graph/ascend_graph_runner.h:32-103                                                  
                                                                                                        
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
                                                                                                        
 AscendGraphInstance                                                                                    
                                                                                                        
 ### 位置                                                                                               
                                                                                                        
 rtp_llm/cpp/ascend_graph/ascend_graph_utils.h:24-82                                                    
                                                                                                        
 ```cpp                                                                                                 
   class AscendGraphInstance {                                                                          
   public:                                                                                              
       c10_npu::NPUGraph graph_;                                                                        
       AscendGraphMemHold mem_hold_;                                                                    
   };                                                                                                   
 ```                                                                                                    
                                                                                                        
 每个 batch size 保存：                                                                                 
                                                                                                        
 ```text                                                                                                
   batch size                                                                                           
    ├─ NPUGraph                                                                                         
    ├─ 固定输入 tensor                                                                                  
    ├─ Python attention 对象                                                                            
    └─ 固定输出 buffer                                                                                  
 ```                                                                                                    
                                                                                                        
 ────────────────────────────────────────────────────────────────────────────────                       
                                                                                                        
 完整调用链                                                                                             
                                                                                                        
 ```text                                                                                                
   NormalExecutor.cc                                                                                    
     └─ model_->forward()                                                                               
          ↓                                                                                             
   PyWrappedModel.cc:443                                                                                
     └─ build PyModelInputs                                                                             
     └─ graph_runner_->canRun()                                                                         
          ├─ false                                                                                      
          │    └─ Python model.forward(..., eager)                                                      
          │         └─ AscendDecodeImpl                                                                 
          │              └─ FIA v2 eager                                                                
          │                                                                                             
          └─ true                                                                                       
               ↓                                                                                        
   AscendGraphRunner::forward()                                                                         
     ├─ prepareInputs()                                                                                 
     │    ├─ 复制 input_ids / sequence_lengths / KV block table                                         
     │    ├─ 清理 padding 区域                                                                          
     │    └─ prepare_cuda_graph()                                                                       
     │         └─ update_graph_fia()                                                                    
     │              └─ graph_task_update_begin/end                                                      
     │                                                                                                  
     ├─ replayDecode()                                                                                  
     │    └─ replayGraph()                                                                              
     │         └─ c10_npu::NPUGraph::replay()                                                           
     │                                                                                                  
     └─ 读取 hidden_states                                                                              
 ```                                                                                                    
                                                                                                        
 最重要的三个代码入口                                                                                   
                                                                                                        
 ```text                                                                                                
   1. 创建 Runner                                                                                       
      rtp_llm/cpp/models/PyWrappedModel.h:251-312                                                       
                                                                                                        
   2. 判断是否使用 Graph                                                                                
      rtp_llm/cpp/models/PyWrappedModel.cc:491-518                                                      
                                                                                                        
   3. 捕获与重放核心                                                                                    
      rtp_llm/cpp/ascend_graph/ascend_graph_runner.cc                                                   
      - initCapture(): 385                                                                              
      - captureDecode(): 429                                                                            
      - captureOneGraphInstance(): 466                                                                  
      - prepareInputs(): 561                                                                            
      - forward(): 708                                                                                  
 ```                                                                                                    
                                                                                                        
 核心设计可以概括为：                                                                                   
                                                                                                        
 │ 捕获阶段固定算子和内存地址；重放阶段只更新固定输入 buffer 和动态 FIA 参数，然后调用                  
 │ NPUGraph::replay()。                                                                                 
