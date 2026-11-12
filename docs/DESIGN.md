# GaudiRT 运行时与多卡通信架构设计

> **文档状态说明**：本文档原位取代早期（Phase 1 前）关于基础脚手架与裸 SCAL 探针的阶段草稿，作为 `GaudiRT` 现行 `PrivateUse1` 设备后端、分段图编译执行器与原生多卡通信子系统的唯一设计规范。历史撤回结论与全栈里程碑状态见顶层 `ROADMAP.md`，硬件与 SDK 踩坑台账见顶层 `docs/gaudi2-pitfalls.md`。

---

## 1. 架构定位与硬边界

`GaudiRT` 是面向 Gaudi2 构建的零历史包袱 PyTorch 2 原生设备后端（`PrivateUse1` 设备名 `gaudirt`）与图编译执行运行时：

1. **零旧桥隔离**：运行进程设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，C++ 源码与构建产物不引入任何 `gaudi-pytorch-bridge` 头文件或动态库，Python 侧禁止加载 `habana_frameworks` 或 `PT_HPU_LAZY_MODE`；启动与运行期对进程模块表与动态库映射执行硬核查。
2. **明确契约与快速失败（Fail-Fast）**：所有形状越界、设备索引不匹配、未声明算子或非法通信组在入口处直接报错终止，不做静默回退（Fallback），不在解码热循环内部堆砌重复的防御性检查。
3. **跨模型与跨引擎复用**：设备上下文（`Context`）、显存块缓存分配器（`BlockCachingAllocator`）、Recipe 编译缓存（`RecipeCache`）、分段图执行器（`SegmentedRunner`）以及分布式后端（`ProcessGroupGaudiRT`）全部建立在标准 PyTorch 2 `PrivateUse1` / `torch.compile` / `c10d::Backend` 契约之上，不绑死单一模型结构或单一上层服务框架。

### 1.1 模块分层与职责边界

`GaudiRT` 彻底清除早期单模手写图、裸 SCAL 探针与历史兼容壳，按单一职责收敛为四层：

- **C++ 通用运行时底座（`include/gaudirt/{common,runtime}/`、`src/*.cpp`）**：`DeviceRuntime`（单设备与计算/H2D/D2H 流生命周期）、`BlockCachingAllocator` + `GraphPool`（流序块式显存分配与分段图复用池）、`PinnedHostPool`（锁页主机 DMA 池）、`EventPool` / `StreamEvent`（零稳态创建的可复用硬件事件）与 `RecipeCache`（带编译器/算子库/构图二进制 SHA256 指纹的多进程原子落盘缓存）。
- **C++ 统一构图与分段桥接层（`src/bridge/`）**：全仓唯一 `synGraphCreate` 入口收敛至 RAII `SynapseGraphBuilder`（`graph_builder.{hpp,cpp}`），统一服务于 13 档静态桶编译子图与冷路径微型算子；`fx_op_lowering.{hpp,cpp}` 承载 FX 算子节点到 Synapse 节点的下沉（含 `128×128` Block-FP8 小桶/大桶片上 SRAM MME 路径与词表分片 `shard_embedding`），`bucket_program.{hpp,cpp}` 负责 13 档静态桶缓存与快速发射（图外 `bucket_linear` 直接复用同一 `BucketProgram`），`paged_attention*.{hpp,cpp}` / `paged_metadata.{hpp,cpp}` 承载 4D 片上归约分页注意力，`segmented_runner.{hpp,cpp}` 与 `communicator.{hpp,cpp}` / `process_group.{hpp,cpp}` 承载分段图调度与 HCCL 分布式后端。
- **Python 编译前端（`python/gaudirt/`）**：`compiler.py`（`torch.compile` 入口、Dynamo 守卫与静态签名缓存）、`fx_passes.py`（`VocabParallelEmbedding` 折叠与集合通信算子函数式归一化）、`op_lowering.py`（扁平分发表驱动的 `FX -> BucketProgram` 下沉）、`partitioner.py`（基于边界秩的凸集 DAG 划分与 `ASAP Start + ALAP Wait` 拓扑调度）与 `segment_binding.py`（C++ 守卫托管下的静态张量直绑快路径）。

---

## 2. 多卡通信的第一性原理

在 Gaudi2 硬件上，MME（矩阵引擎）与 TPC（向量引擎）通过片内计算流执行算子，而机内 Scale-Up 互联（每卡 24×100GbE RDMA NIC 配合片上 EDMA）拥有独立的硬件队列与直接显存读写通道：

1. **硬件本源**：只要通信引擎与计算引擎操作无数据依赖的不同显存缓冲，并通过设备侧硬件流事件（`synEventRecord` + `synStreamWaitEvent`）完成跨流握手，硬件队列即可并发推进集合通信与矩阵/向量计算，全程无需主机 CPU 介入同步。
2. **旧路径症结**：传统惰性桥接路径把集合通信留在 Python 逐算子分发层，每次通信前后触发子图切碎、主机侧流同步或临时事件创建/销毁；词法上紧跟通信算子的 `wait_tensor` 又把异步句柄立刻拉回同步等待，彻底堵死了跨算子重叠空间。
3. **根本解法**：
   - **通信进入编译图**：集合通信算子与计算子图、PagedAttention 同级进入 `SegmentedRunner` 指令流，成为显式的 `CollectiveStart` 与 `CollectiveWait` 原语，消除热路径 Python 胶水与每次调用的事件分配。
   - **数据依赖决定等待时机，而非词法位置**：编译器剥离词法紧邻的 `wait_tensor`，仅保留真实消费者数据边，按 **`ASAP Start`（生产者一结束立即发车）+ `ALAP Wait`（首个真实消费者执行前才等待）** 对分段有向无环图（DAG）做拓扑调度。
   - **自适应单流/双流下沉**：当模型本身是纯串行单链（如标准 Dense Transformer 的 `O-Proj -> AllReduce -> AddRMSNorm`）且中间不存在独立可并发计算段时，强行切双流只会白交两次跨流事件握手税；仅当 `CollectiveStart` 与 `CollectiveWait` 之间确实调度进了独立计算段（如共享专家分支、并行分支或双微批流水）时，才下沉到独立通信流并发执行。

---

## 3. 多设备绑定与并发编译安全

### 3.1 逻辑设备到物理卡号的确定性映射

多卡进程组（如 Tensor Parallelism 各 Worker）按逻辑设备索引 `gaudirt:0 .. gaudirt:{N-1}` 寻址，物理机器上可能仅有部分模块空闲：

- **可见模块解析**：启动时解析环境变量 `HABANA_VISIBLE_MODULES`（如 `"0,1"`），建立进程内只读映射表 `visible_modules[logical_idx] -> physical_module_id`，`device_count()` 返回可见模块数；未设置时默认暴露单模块 `[0]`。
- **延迟绑定单进程单卡上下文**：框架插件加载阶段（`init_backend`）仅完成后端注册，**不提前抢占默认物理卡**。待 Worker 进程调用 `set_device(logical_idx)` 确定自身逻辑设备后，首次设备操作才通过 `synDeviceAcquireByModuleId(visible_modules[logical_idx])` 获取对应物理模块，并将 `ID` 与 `HLS_MODULE_ID` 同步写入设备运行时配置。同一进程一旦完成物理卡绑定，禁止跨卡切换逻辑设备索引。
- **全链路设备索引对齐**：分配器、图内存池、动态形状校验、分桶编译与算子分发统一校验张量所在设备与当前上下文 `device_index` 一致，消除硬编码 `gaudirt:0` 限制。

### 3.2 多进程共享 `RecipeCache` 原子落盘

TP 多进程首次冷启动编译同构子图时，多个 Rank 会同时向同一 `GAUDIRT_RECIPE_CACHE` 目录写入相同哈希键的 `.recipe` 与 `.meta` 文件：

- 临时文件统一命名为 `<target>.tmp.<pid>.<seq>`，写完并校验落盘字节数后通过 POSIX 原子 `rename` 覆盖目标路径，彻底消除多进程共享 `.tmp` 文件名引发的竞态报错或半截文件读取。

---

## 4. `CommunicatorRegistry` 与 `ProcessGroupGaudiRT`

`GaudiRT` 在 C++ 层直接实现 `c10d::Backend` 子类 `ProcessGroupGaudiRT`，通过 `torch.distributed.Backend.register_backend("gaudirt", ...)` 注册为 `gaudirt` 设备的原生分布式后端：

```
PyTorch c10d Store (TCPStore)
        │  Rank 0 调用 hcclGetUniqueId(&uid)（动态临时端口，禁用固定 HCCL_COMM_ID）
        │  广播 1032B hcclUniqueId -> 全组调用 hcclCommInitRank
        ▼
CommunicatorRegistry (进程内单例，按 group_name / pg_uid 索引)
        │
        └── CommGroup {
              hcclComm_t      comm;             // HCCL 通信域句柄
              synStreamHandle comm_stream;      // 由 Context::streams 托管的通信流
              StreamEvent     compute_to_comm;  // 预分配可复用：计算流 -> 通信流事件（含未消费等待防重录保护）
              StreamEvent     comm_to_compute;  // 预分配可复用：通信流 -> 计算流事件（含未消费等待防重录保护）
              uint64_t        barrier_address;  // 常驻零初始化 256B 设备缓冲，承载单 FP32 AllReduce 设备栅栏
              uint64_t        scratch_address;  // 按需扩容复用的通信暂存区（字节对齐打包 / 小批量 AllGather）
            }
```

核心契约：

1. **零端口冲突握手**：不设置全局 `HCCL_COMM_ID`（规避默认固定端口冲突 `P-RT-08`）。子组内 `rank == 0` 直接调用 `hcclGetUniqueId(&uid)` 由内核分配空闲临时端口，通过 `c10d::Store` 按唯一键 `"gaudirt_hccl_uid/<group_id>/<seq>"` 分发给同组其余进程完成 `hcclCommInitRank`。
2. **适配 Scale-Up `libhcl.so` 物理能力边界**：
   - **设备栅栏**：当前 Scale-Up NIC 模式下 `hcclBarrier` 返回 `hcclUnsupported (status=2)`，故每个 `CommGroup` 预留零初始化 `barrier_address`，通过 1 元素 FP32 `hcclAllReduce(..., hcclSum, comm, comm_stream)` 实现纯设备流栅栏；
   - **数据类型映射**：`libhcl.so` 仅实现 `hcclBfloat16`、`hcclFloat16`、`hcclFloat32`。算术归约（`AllReduce` / `ReduceScatter`）对非上述浮点类型立即 Fail-Fast 报错；纯字节搬运（`AllGather` / `AllToAll` / `Broadcast` / `Send` / `Recv`）按字节粒度打包为 `hcclFloat32` / `hcclBfloat16` 传输（奇数单字节经 `scratch` 补齐 2 字节）；
   - **组调用隔离**：`hcclGroupStart() ... hcclGroupEnd()` 内部仅放置 `hcclSend` / `hcclRecv`，本卡自拷贝（`synMemCopyAsync`）严格置于组调用之外，避免与延迟提交的 HCCL P2P 操作产生同流竞态。
3. **完整的 Eager 与编译图共享上下文**：同一个 `CommGroup` 同时服务于编译期之外的常规 `torch.distributed` 集合通信以及编译图内的 `SegmentedRunner` 指令流；`comm_stream` 由 `Context::streams` 统一登记，使 `GraphPool` 流栅栏与 `synchronize()` 自动覆盖全部通信流。
4. **异步设备流等待语义**：常规 `ProcessGroupGaudiRT` 集合操作在 `compute_stream` 记录 `compute_to_comm` 事件、在 `comm_stream` 等待并提交 `hccl*` 调用后记录 `comm_to_compute` 事件（可复用 `StreamEvent` 若前次记录尚未被等待则先在目标流消费，满足 Synapse 事件重录前置条件）；返回的 `WorkGaudiRT::wait()` 默认在调用方 `compute_stream` 插入设备侧 `synStreamWaitEvent`，保持主机 CPU 异步发射不被阻塞，同时对输入/输出缓冲执行 `record_stream(comm_stream)` 防止跨流提前复用。

---

## 5. 编译器 `ASAP Start + ALAP Wait` 调度与自适应双流下沉

### 5.1 集合通信前端归一化

`fx_passes.py` 同时识别标准 PyTorch 函数式集合通信与 SGLang 通信算子族，统一归一化为无副作用的函数式图节点：

- `torch.ops._c10d_functional.{all_reduce, all_gather_into_tensor, reduce_scatter_tensor, all_to_all_single}` + `wait_tensor`
- `torch.ops.sglang.{outplace_all_reduce, inplace_all_reduce, reg_all_gather_into_tensor, reg_reduce_scatter_tensor, reg_all_to_all_single}`

其中词法上的 `wait_tensor(y)` 不生成任何同步节点，直接折叠为对集合通信输出槽位的数据依赖边；诸如 `empty + reg_all_gather_into_tensor` 或 `inplace_all_reduce` 的原地写语义在分段边界处转化为干净的输入槽位 $\rightarrow$ 输出槽位函数式转换，且对词表分片尾维拼接 `[W * B, V_local] -> reshape(W, B, V_local) -> movedim(0, 1) -> reshape(B, W * V_local)` 直接折叠为 `AllGatherDimLast`（当 `B == 1` 时零拷贝重塑为 `[1, W * V_local]`，当 `B > 1` 时由重排子图完成块转置）。

对于 `VocabParallelEmbedding` 产生的词表分片掩码查表模式（区间比较 `>= start`、`< end`、偏移减法、越界归零掩码与 `embedding`），编译器识别并下沉为单节点 `FxOp::shard_embedding`，并在编译与静态绑定阶段按完整词表上限校验输入 Token ID。

### 5.2 凸集子图划分与 `ASAP Start + ALAP Wait` 拓扑调度

为消除源码书写顺序（词法顺序）对并行度的束缚，编译器在分段时采用**基于边界秩的凸集划分（Convex DAG Partitioner）**，将仅按词法相邻但互不依赖的平行分支合并进各自的凸集计算段，严格保持粗化分段图的无环性（DAG），再对三类原语节点——**纯计算段（`RunSegment`）**、**注意力段（`PagedAttention`）** 与 **集合通信对（`CollectiveStart` / `CollectiveWait`）**——执行确定性拓扑调度：

1. **立即发车（ASAP Start）**：一旦某个集合通信的输入槽位由前置计算段产出，立即发射 `CollectiveStart(coll_id)`，使通信请求第一时间进入硬件队列。
2. **推迟等待（ALAP Wait）**：只要就绪队列中还存在**不依赖在途集合通信结果**的独立 `RunSegment` 或 `PagedAttention`，优先发射全部就绪计算/注意力段；直到所有无依赖计算均已排空、下一步必须执行消费该通信结果的段（或图返回输出）时，才发射对应的 `CollectiveWait(coll_id)`。

### 5.3 `SegmentedRunner` 自适应单流/双流执行

`SegmentedRunner::set_outputs()` 在固化指令序列时静态扫描每一对 `(CollectiveStart(c), CollectiveWait(c))`：

| 拓扑形态 | 典型场景 | 执行流选择 | 同步开销 |
|---|---|---|---|
| **紧邻型**（`Start(c)` 与 `Wait(c)` 之间无任何计算/注意力指令） | 标准串行 Dense 层 `O-Proj -> AllReduce -> AddRMSNorm`、`Down-Proj -> AllReduce -> NextLayer` | 直接在主计算流 `compute_stream` 上发射 `hccl*` | **零跨流事件开销**，硬件流天然保序 |
| **重叠型**（`Start(c)` 与 `Wait(c)` 之间夹有 ≥1 条独立计算/注意力指令） | 共享专家与路由专家并行、并行 Attention+MLP、双微批交错流水 | `Start(c)` 在 `compute_stream` 记录事件并切入 `comm_stream` 发射 `hccl*`；中间计算在 `compute_stream` 并发运行；`Wait(c)` 在 `compute_stream` 等待 `comm_to_compute` 事件 | 仅使用通信域预分配的两个可复用 `StreamEvent`，稳态 **0 次事件创建/销毁** |

---

## 6. 分桶感知通信、显存生命周期与退场契约

### 6.1 物理分桶缓冲与精确逻辑传输量

`GaudiRT` 计算段使用静态分桶缓冲（物理容量 `M_bucket * H`），而张量的逻辑尺寸始终保持真实 Token 数 `[M, H]`：

- **精确字节传输**：所有集合通信严格按逻辑元素数 `M * H`（而非物理桶上限 `M_bucket * H`）向 HCCL 提交传输长度，避免在非满桶形状下浪费互联带宽或污染有效前缀外的尾部分区。
- **确定性归约精度**：全归约路径对跨卡部分和保持确定性归约语义；在小批量延迟敏感区间支持 `AllGather + 确定性秩序 FP32 累加`（严格按 `rank 0 + rank 1 + ... + rank {W-1}` 顺序在 FP32 中累加再转回目标精度），保证所有 Rank 输出逐位一致（Bit-Exact），彻底杜绝多步解码因跨卡舍入微差导致的轨迹分叉。

### 6.2 显存槽位钉住与跨流安全

在 `SegmentedRunner` 的槽位复用分析（`ReusableOutputs`）中：

- 集合通信输入槽位 `src_slot` 的最后使用点延伸至对应的 `CollectiveWait(c)`（而不仅是 `CollectiveStart(c)`），确保通信在途期间后续计算段绝不会覆写发送缓冲；
- 当指令走双流重叠模式（`comm_stream`）时，自动对 `src_slot` 与 `dst_slot` 底层块登记 `record_stream(comm_stream)`，保证动态分配器与图内存池在 `comm_stream` 消费完毕前不回收底层物理块。

### 6.3 退场生命周期与防死锁铁律

严格遵循底层驱动与 HCCL 析构顺序契约（`P-RT-06`、`P-RT-21`）：

1. **正常退场顺序**：同步 `compute_stream` 与全部活跃 `comm_stream` $\rightarrow$ 释放图静态绑定槽位与显存池租约 $\rightarrow$ 按创建逆序调用 `hcclCommDestroy` 销毁通信器、销毁预分配 `StreamEvent` 与 `comm_stream` $\rightarrow$ 最后释放 `DeviceRuntime`（`synDeviceRelease`、`synDestroy`）。
2. **异常退场防死锁**：若进程因未捕获 C++/Python 异常退出（`std::uncaught_exceptions() > 0` 或 `PyErr_Occurred()`），析构与清理钩子**严禁发起任何跨卡集合同步或屏障等待**，直接快速终止本进程，避免单卡报错触发对端卡在 HCCL 内无限挂死。
