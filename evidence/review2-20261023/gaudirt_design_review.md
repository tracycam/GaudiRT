# GaudiRT 设计审查报告（software-autopsy.md + DESIGN.md）

> 审查时间：2026-10-23。审查依据：源码、`libSynapse.so` 二进制反汇编（SHA256 `6796f09d…`，与真机一致），以及本次在 `192.168.100.153` 上新做的一次纯 C++ 真机实验（module 7，独占 flock，系统 SDK `1.24.2-481`，无 LD_PRELOAD，结束后八卡恢复 768MiB/0%）。
> 下文“事实”指我复核过的源码、二进制或实测；“推断”和“假说”会单独标明。

---

## 0. 结论先行

**裁决：需修改后再冻结。现在不能进入 Gate 0 的生产代码阶段。**

主要原因有三条：

1. **立项主线（Layer 1 驻留实例）还没有证明打在关键路径上。** 同一份 `production70` 原始数据里，只改 CPU 调度策略（uclamp），步长就从 13.19/13.37 ms 降到 11.31/11.32 ms。sibling 仓库已有的设备 trace 还显示：在高频 CPU 策略下，相邻 recipe 之间的空闲共 2.70 ms，其中约 **1.90 ms 发生在八个 rank 的对应通信调用都已返回之后**；可归因到“最后一个 rank 还没提交”的只有约 0.51 ms。DESIGN 引用的是最差那一组（13.37 ms），既没用 11.3 ms 的数据，也没引用这份设备空闲拆解。
2. **四层里有三层半是在重做已有代码**（见 §3）：L1_A 和 L3-R2 对应 `PreparedRegionProgram`，以及已经退役的 `fixed_plan.cpp`；L2 对应 `gaudi-kernels-public` 里的 `bf16/fp8/mxfp4_linear` 和 `mxfp4_compact`；L3-R1 对应已有的 `_prepared_decode_input`（生产统计里 `input_layout_hits=1386`）。
3. **真正没人负责的瓶颈没有被列入任何一层：** 每个 token 有 141 次 TP 集合通信，带来设备侧的交接和小消息时延；recipe 内部有 2.06 ms 空闲；计算 union 5.73 ms 离 HBM roofline 还有多远，目前没有量化。

---

## 1. 本次新增的真机实测：在途上限出在哪个资源（DESIGN Gate 0 第 2 项）

做法：[prequeue_capacity_probe.cpp](file:///home/rate/.gemini/antigravity-cli/brain/4a518f11-bc3c-4a4b-9d72-8b43839ae657/scratch/prequeue_capacity_probe.cpp)。先用一个 48 节点、8192³ BF16 GEMM 链组成的 gate recipe 堵住设备（单独运行约 122 ms），再连续提交 4000 个小 recipe，逐次记录 `synLaunchWithExternalEventsExt` 的耗时。gdb 作为父进程，在第一次长阻塞时抓取全线程调用栈。原始输出见 [plain.log](file:///home/rate/.gemini/antigravity-cli/brain/4a518f11-bc3c-4a4b-9d72-8b43839ae657/scratch/remote-run/plain.log) 和 [gdb.log](file:///home/rate/.gemini/antigravity-cli/brain/4a518f11-bc3c-4a4b-9d72-8b43839ae657/scratch/remote-run/gdb.log)。

| 小 recipe 配置 | 绑定数 | 首次阻塞序号（2 次重复） | 阻塞时长 | 阻塞前 p50（无完成回收的状态） |
|---|---:|---:|---:|---:|
| 1 节点 add | 3 | 573 / 573 | 118 ms（≈ gate 剩余时间） | 6.6–7.0 µs |
| 两个不同 recipe 交替 | 3 | 571 / 571 | 118 ms | 6.7–7.0 µs |
| 8 节点 | 24 | 573 / 573 | 116 ms | 10.8–11.3 µs |
| 32 节点 | 96 | 573 / 573 | 108 ms | 24.1–25.4 µs |

阻塞时的调用栈（事实）：

```
synLaunchWithExternalEventsExt → … → QueueComputeScal::launch → RecipeLauncher::launch
 → prepareForDownload → getMappedMemoryInfo → LaunchTracker::waitForCompletionCopy
 → RecipeLauncher::checkCompletionCopy → ScalStreamBase::longSoWait
 → Scal::completionGroupWait → hlthunk_wait_for_interrupt（ioctl 睡眠等待）
```

**结论：**
- 卡住的是 **`MappedMemMgr` 这条路径**。等待的是**最老一次 launch 的 copy（下载）完成**，不是 compute 完成。不是 ArcHbm/HbmGlbl，不是 LaunchTracker 槽位，也不是 CCB 信用。autopsy 的首选假说成立。
- 上限是**按次数计，不按字节计**：绑定数从 3 涨到 96、节点数从 1 涨到 32，上限都是 ≈573（另加 gate 本身）。不同 recipe 越多，上限略降（573 → 571）。推断：池里还有按 recipe 常驻的条目。
- 和历史 bridge 实验（568/570）吻合，说明 collective 不占这个池。
- **单次 launch 成本 ≈ 6.2 µs 固定 + 0.19 µs/绑定**（阻塞前、没有完成回收时）。生产 recipe 读 9–18 个、写 5–16 个，单次约 9–13 µs。
- 还没覆盖的边界：真实层 recipe 的 patchable section 更大，可能一次占多个 DataChunk，上限可能低于 573。这项必须用生产 recipe 复测。

> [!IMPORTANT]
> 这个上限 ≈ 573 次 compute launch ≈ **3.97 个 decode 步**。生产配置 `async_scheduling=False`，下一步要等主机采样结果才开始，在途最多 1 步（约 144 次 launch），**永远碰不到这个上限**。所以“问题二”只有在“设备自治的多步执行”模式下才成立，而 DESIGN 没有把这个模式列为目标。

---

## 2. Q1 事实与归因核查

### 2.1 硬件对标表

| 条目 | 判定 | 说明 |
|---|---|---|
| 432/865 TFLOPS、96GB/2.45 TB/s、48MB SRAM（`sramSize=50331648` 已实测） | 基本属实 | — |
| “TPC ~11 TFLOPS FP32”，并与 CUDA FP32 比较 | **口径不清** | 没标出处；Gaudi2 公开的向量算力口径多是 BF16。表里混用了 BF16 和 FP32 两种口径 |
| 910B 各项参数 | 来源缺失 | 区间值没有出处，建议删除或给出处 |
| RoCE 262.5 GB/s 对比 NVLink 300 GB/s | **口径错位，且漏了关键维度** | HLS-2 是全互联，**每对 rank 只有 3×100GbE = 37.5 GB/s**。decode 的 141 次小消息集合通信受**时延**限制，不受带宽限制，表里完全没有时延这一维 |
| “48MB 全可编排” | **漏了边界** | `gaudi-kernels-public` 把 RMW 解码 tile 硬限在 16 MiB（[mxfp4_compact.cpp](file:///home/rate/llm/Gaudi-LLM/gaudi-kernels-public/csrc/ops/mxfp4_compact.cpp)、[bf16_linear.cpp](file:///home/rate/llm/Gaudi-LLM/gaudi-kernels-public/csrc/ops/bf16_linear.cpp)）；二进制里有 `GCFG_RMW_SECTION_MAX_SIZE_BYTES`、`GCFG_MAX_RMW_TENSOR_BYTES`、`GCFG_SCRATCHPAD_SRAM_MAX_SIZE`、`GCFG_SRAM_SLICER_MAX_CAPACITY_BYTES`，默认值待读取 |

### 2.2 Synapse 图编译器与 SRAM（autopsy §二.1.1）

- `sram_mme_probe` 证明：一个 **2 KB**（64×16 BF16）的 RMW 中间张量被放进了 SRAM，并且结果逐位正确。属实。
- **缺对照组**：没有跑 RMW=false 的对照，无法证明是 RMW 让它进了 SRAM。GC 本身就有自动 SRAM 切片器和 Layered Brain（见上面那些 `GCFG_LAYERED_BRAIN_*` / `GCFG_SRAM_SLICER_*`），本来也可能把它放进 SRAM。
- **前提没验证**：问题四说“Aten 逐算子 lowering 没用上 SRAM”。生产 144 个 recipe 的 post-graph 从未导出，SRAM 和 DRAM 各占多少没有统计。

### 2.3 RecipeLauncher 反汇编（autopsy §二.2 问题一）

| autopsy 的说法 | 二进制事实 | 判定 |
|---|---|---|
| `LaunchTracker::checkForCompletion (0x171e1f0)` 是 `RecipeLauncher::launch` 的第 1 步 | 0x171e1f0 实际是 `RecipeLauncher::checkCompletionCompute`。`LaunchTracker::checkForCompletion` 在 **0x16d6130**，由 `QueueComputeScal::launch`（0x1770202）在**持有流互斥锁的情况下**、构造 RecipeLauncher **之前**调用 | **符号和调用层级都错了** |
| `validateSectionsInfo (0x1720a00)` | 导出入口是 0x170f8b0；0x1720a00 落在 `dfaLogDescription` 后面一段未导出的局部代码里 | 地址错 |
| 调用顺序 prepare → download → scalEnqueue → validate | 实际：validateEvents → prepareForDownload → downloadToDev → 间接调用 → updateSfgLongSos → scalEnqueue → validateSectionsInfo | 基本属实 |
| scalEnqueue 末尾“挂载槽位释放回调” | 0x17247c8/0x17247e1 是先取本次 longSo targetValue，再对两个内存管理器做 vtable+0x10 调用。符号里有 `NBuffQueueAllocator::setLongSo`。推断：这是**按 long-SO 代际标记回收**，不是回调 | 措辞失准 |
| 状态 2 时跳到 0x171f1c0 等待 | 判断的是 `this+0x68`/`this+0x98` 两个条目状态，不是返回值；等待走 `LaunchTracker::waitForCompletionCopy`（本次实测已抓到） | 机制属实，细节需修正 |
| “无状态接口，缺少驻留机制” | **言过其实**。`MappedMemMgr::getAddrForId(…EntryIds…)` / `removeId(RecipeSeqId)` 按 recipe 缓存映射条目，`HbmGlblMemMgr→NBuffAllocator` 也按 recipe 缓存。作者自己的 breakdown 里 `patchAll` 只有 **0.074 µs（0.5%）**，`memcpyToMapped` 只有 **0.032 µs** | **归因错误**：Patch 不是成本大头 |

**插桩 breakdown 里真正的大头**（13.785 µs/次）：
- `QueueComputeScal::launch` **以外**约 2.96 µs（synSingleton、DeviceCommon、StreamsContainer、Stream::addJob 这几层，公开 API 绕不开）
- checkForCompletion 1.83 µs、scalEnqueue 1.87 µs、pdmaDownload 1.48 µs、validateSectionsInfo 1.05 µs、analyzeTensors 0.83 µs

驻留实例理论上最多能省 analyze、validate、getAddr、PDMA 等合计 ≈ 4–6 µs/次，**不会让单次成本降一个数量级**。

### 2.4 计时口径（问题一、问题二）

- **11.1 µs** 是 3 绑定 toy recipe、batch=256 的均值。按本次测出的成本模型，生产 recipe（约 25 绑定）每次约 11–13 µs。
- **4.65 ms/步** 来自 `sdk1241`、observer 插桩、双卡 64×64 合成负载，测的是 API 墙钟的包含时间。文档有标注，但 DESIGN 仍拿它当动机。它**不是关键路径**：高频策略下 NativePlanSubmit 为 5.18–5.40 ms，而设备计算跨度 10.48 ms，主机提交本来就和设备执行重叠（见 [CPU-DEPENDENCY-AND-OVERLAP-20261011.md](file:///home/rate/llm/Gaudi-LLM/gaudi-kernels-public/docs/CPU-DEPENDENCY-AND-OVERLAP-20261011.md)）。
- **13.37 ms** 是 `production70` 中 host policy 关闭那一组。同一文件里 host policy 开启的两组是 **11.31/11.32 ms**（我从 `runs[1..2].resident_p50_ms` 读出）。基线是挑出来的，DESIGN 里所有收益都可能和 CPU 频率效应混在一起。
- **被忽略的反证**：[BRIDGE-INTEGRATION-20261004.md](file:///home/rate/llm/Gaudi-LLM/gaudi-kernels-public/docs/BRIDGE-INTEGRATION-20261004.md) 记录过“forward 提交更早返回了，但输出同步等得更久，采样仍约 12.8 ms，模型没有提速”。autopsy 和 DESIGN 都没引用这条。

### 2.5 HCL（autopsy §二.1.2）

- **源码版本不匹配**：`HCL-provenance.json` 写明公开源码只到 **1.21.0-555**，装的运行时是 1.24.x。autopsy 把 1.21 的源码当成 1.24 的证据，没有加版本说明。
- `libSynapse.so 1.24.2` 导入了 `hcl::HclPublicStreams::streamWaitEvent(…, hcl::syncInfo)`，说明“long-SO 设备侧等待”这个接口形态在 1.24 二进制里仍然存在。这一点可以佐证，但只是间接证据。
- **漏掉的成本**：
  - prepared 路径把 collective 提交到 **compute 流**上（`call.collective->submit(…, stream)`，[NativeExecutionPlan.cpp:L200-L213](file:///home/rate/llm/Gaudi-LLM/gaudi-pytorch-bridge/backend/habana_device/NativeExecutionPlan.cpp#L200-L213)）。每次 HCL 调用前都要先走 `synStreamSyncHCLStreamHandle → DeviceCommon::syncHCLStreamHandle`，往 Synapse 流里加一个 `EventNetwork` 任务（二进制事实），**每个 token 产生 141 次 compute↔HCL 的设备侧交接**。
  - HCL 单次的主机成本（插桩下约 15.8 µs）≥ compute launch，可是**没有任何一层负责它**。
  - HCL 自己也有主机阻塞路径（`CyclicBufferManager::moveToNextDivision → waitOnValue`，见 [DEVICE-PROGRAM-FEASIBILITY-20261011.md](file:///home/rate/llm/Gaudi-LLM/gaudi-kernels-public/docs/DEVICE-PROGRAM-FEASIBILITY-20261011.md)）。

### 2.6 Bridge 与 NativeExecutionPlan（问题三）

- **`Frame::instantiate` 的粒度说错了**：它是**每次 `replay()` 每个 region 一次**（frames 向量在 repetition 循环外面）。生产里每步 2 次，地址缓存命中 2794/2796；不是每次 launch 一次。区域内每次 launch 只多一个循环，直接调 `synLaunchWithExternalEventsExt`，用的是缓存好的绑定（[L194-L214](file:///home/rate/llm/Gaudi-LLM/gaudi-pytorch-bridge/backend/habana_device/NativeExecutionPlan.cpp#L194-L214)）。Bridge 每步固定开销主要是 binding preflight：高频策略下 98–127 µs，约占步长 1%。
- **模式分裂不是性能问题**：生产路径是 Lazy + HPUGraph + NativeExecutionPlan（`executor=pytorch`、`step_graph=True`），`HabanaGraphModule` 根本不在关键路径上。
- “144 计算 + 141 通信”、`adapter_replays=0`：属实。“计算子图已接近下限”在当前算法结构下也属实。但反过来说明：**通信次数和单次通信时延才是可以动的杠杆**，文档没有往这个方向推。

### 2.7 问题四（算子选型）

把 MoE 走 `tpc_vector` / `tpc_folded_scale_tail` 说成“退化”，是**错误归因**。`gaudi-kernels-public` 已经有基于实测的选型：
- [CONTRACTS.md](file:///home/rate/llm/Gaudi-LLM/gaudi-kernels-public/docs/CONTRACTS.md)：BF16 方案选了“MME/SRAM/尾行计划，TPC 候选未胜出”；MXFP4 方案是“TPC 解码/SRAM/BF16 MME”。
- [MXFP4_DUAL_PATH_PLAN.md](file:///home/rate/llm/Gaudi-LLM/gaudi-kernels-public/docs/MXFP4_DUAL_PATH_PLAN.md)：“Small M may compute in TPC; larger M decodes bounded BF16 SRAM tiles for MME”。

decode 时 M 很小，MME 利用率极低（probe 里 `mme_compute_utilization=0.03125`），TPC GEMV 可能本来就是 HBM 带宽受限下的合理选择。

---

## 3. Q2 有没有重复造轮子

| DESIGN 模块 | 已有实现 | 判定 |
|---|---|---|
| L0 哈希锁、ABBA、插桩探针 | `review-*` 脚本、`gaudi-kernels-public/csrc/diagnostics/{submission_timing,launch_fault,compute_events}.cpp`、`bridge_model_test.py --*-abba` | 整理成工具可以，**不应该算一层** |
| **L1_A** 预绑定 `synLaunchTensorInfoExt` 表 + 共享 workspace | `PreparedRegionProgram`（缓存绑定、`max_workspace_size`）；`gaudi-kernels-public/tools/validation/retired/fixed_plan.cpp`（“still calls the CPU APIs for each command”，**已退役**） | **重写** |
| L1_B 驻留实例 | 没有现成实现，但 Synapse 已经按 recipe 缓存静态和映射条目（§2.3） | 新的，可是收益上限没量化，另有致命暗礁（§4） |
| **L2** synGraphHandle + RMW SRAM 块构建器、MoE 块 | `bf16_linear.cpp`（`sram_intermediate`）、`fp8_linear.cpp`、`mxfp4_linear.cpp`、`mxfp4_compact.cpp`（`plan_tiles` / `append_grouped`，16 MiB 预算）；`MOE_EXPERT_DISPATCH_CHECKPOINT` 里“594 decoder outputs, all placed in SRAM” | **改名重写**。DESIGN 给的示例 “TPC 量化解包 → RMW SRAM → MME” 就是 `append_decode_mme_tile` |
| **L3-R1** 固定地址原位输入槽位 | MimoHPUWorker 的 `_prepared_decode_input`（`input_layout_hits=1386`）、bridge 的 retained leases | **重写** |
| **L3-R2** 计算 + HCL 统一 StepExecutionPlan | `NativeExecutionPlan`（RAW/WAR/WAW 依赖、serial region、communicator FIFO 进出保护、HCCL 队列一次提交）。DESIGN 自己也写了“把 for 循环搬出 Bridge 收益极其有限” | **重写**，而且和 DESIGN 自己的准则冲突 |
| 无人认领 | 141 次 collective 的设备侧交接和小消息时延；recipe 内 2.06 ms 空闲；计算 union 和 roofline 的差距 | **真实缺口** |

---

## 4. Q3 驻留计算实例（ResidentComputeInstance）的暗礁

下面各条按“事实（二进制或实测）→ 风险”组织。

1. **完成计数耦合**：`QueueComputeScal::launch` 在流互斥锁里依次做 `checkForCompletion → RecipeLauncher → LaunchTracker::add`；`libscal` 导出 `scal_completion_group_{get,set}_expected_ctr`。如果绕开 QueueComputeScal 直接往**同一个** scal 流提交，Synapse 主机侧维护的 CG 期望值就和设备 long-SO 计数**对不上**，后续 `synEventRecord`/`synStreamSynchronize`/HCL `streamWaitEvent` 的 target 全会错，表现是静默的数据竞争。如果改用私有流，就得占用一个 completion group，而 CG 在 scal 配置里是静态分配的，Synapse 已经全部占用。
2. **内存池回收是按代际 long-SO 做的**：`HbmGlblMemMgr::getAddr → NBuffAllocator::alloc`，scalEnqueue 结束时按本次 targetValue 标记。若要在环形池或 N-buffer 里“钉住”一个块，要么把轮转堵死，要么被复用后覆盖。所以只能自己 `synDeviceMalloc` 驻留内存。
3. **自建内存会推翻“不拼 ECB”原则**：驻留描述符一旦放在私有地址，基地址更新（`addBaseAddresses`）、ECB list 分派、alloc barrier（`addAllocBarrierForLaunch`）都得自己发 ARC scheduler 命令。这正是 DESIGN §一.1 判定为“风险极高且无必要”的事，同时还被版本不一致的 ARC 协议头卡住。**设计内部自相矛盾。**
4. **patchable 二进制块里是否含每次 launch 都变的状态，目前未知**：BFM 里有 `process_eng_reset_soset`、`process_eng_sfg` 这类 opcode。如果动态 ECB 或描述符里嵌了每次 launch 的同步对象、SFG 值，那么多个在途实例共用一份描述符就会互相踩。必须先实测“相同绑定下的 patchable 字节是否逐位相同”。
5. **SFG/外部事件**：`updateSfgLongSos` 每次 launch 都会跑。SFG 关闭时 prepared 路径可以排除它，但 Gate 1 必须证明回退判定是确定的。
6. **在途上限会转移**：本次实测证明当前第一个上限在 MappedMemMgr，而它被占用正是因为每次 launch 都要下载。驻留实例去掉这次下载后，上限会转移到 ArcHbm、LaunchTracker、CCB 或 HCL 的 cyclic buffer。每换一层都要重新抓调用栈。
7. **HCL 协同**：prepared 路径里 collective 和 compute 共用一条流，每次 collective 都会插入 `EventNetwork` 同步任务。私有提交路径必须正确参与这个 long-SO 序列，否则 HCL 等的目标值不对。
8. **异常与生命周期**：设备错误或 TDR（`ScalCompletionGroupBase::tdr`）、`synRecipeDestroy`/`synDeviceRelease` 时如果还有驻留实例在用、部分提交后失败（bridge 有 poison 机制）、进程退出时的排空，这些 DESIGN 都没有定义。
9. **维护成本**：靠私有结构偏移（如 `this+0x68/0x98/0x100/0x108`）和哈希锁定运行，**任何 SDK 小版本更新都会让它失效**。DESIGN 没有给出退出策略。

---

## 5. Q4 门禁评估

**结构性缺陷：**
- **门禁按层排序，没按瓶颈大小排序**。L2 只依赖 synGraphCompile，和 L1 正交，“冻结 L1 才能写 L2”是人为加的耦合。
- **没有量化的继续/终止门槛**：每道 gate 只要求“出报告”，没有写“低于多少收益就砍掉这一层”。
- PDMA 字节数、预排深度这些 ROI 关键量放在 Gate 1，**应该提前到 Gate 0**。不然就是先写实现再看收益。
- Gate 1 缺：在 bridge 进程内共存的测试、和 HCL 共用流的测试、异常注入测试；Gate 3 才测 long-SO 回绕已经太晚。
- 所有 ABBA 都没把 **CPU 调度策略（uclamp/governor）设为受控变量**，而这一个变量就能造成 2.16 ms 的差异。

**Gate 0 应该补的必测实验（都不需要写生产代码）：**

| # | 实验 | 回答的问题 | 继续/终止门槛（建议） |
|---|---|---|---|
| G0-1 | **设备下限实验**：70 层 TP8，每个 rank 先用 delay gate 堵住设备，预排 1 个完整步（144 + 142 次命令 < 573，可行），放开 gate 后用设备时间戳测纯设备步长 T_dev | 所有主机侧优化（L1 + L3）收益的**理论上限** = T_step − T_dev | T_step − T_dev < 1 ms 时，L1_B 不立项 |
| G0-2 | 固定 uclamp=1024 或 performance governor，重测 T_step（与 11.3 ms 对齐） | 去掉 CPU 频率混杂 | 后续所有 ABBA 都必须固定这个策略 |
| G0-3 | 用现成的 `BYPASS_ANALYZE_TENSORS` / `BYPASS_VALIDATE_SECTIONS` 跑整模 oracle ABBA，并逐位校验输出 | 驻留实例在 CPU 侧能省多少的上界 | TPOT 改善 < 3% 时，L1_B 不立项 |
| G0-4 | 用生产的 144 个真实 recipe 复测本次的在途上限探针 | 大 recipe 会不会占多个 DataChunk，上限是否更低 | — |
| G0-5 | 相同绑定连续 launch，用 gdb 在 `pdmaDownload` 处 dump patchable 源缓冲，比较逐位是否一致 | 驻留描述符是否**可能**成立 | 不一致时，L1_B 直接否决 |
| G0-6 | TP8 小消息 AllGather（生产 payload）的设备时延；同一条 generic 流和分离流上 compute↔HCL 的交接时延 | 那 1.90 ms 的构成 | 用于给“通信 track”排优先级 |
| G0-7 | 导出 144 个生产 recipe 的 post-graph，统计 SRAM/DRAM 分配、MME 利用率、recipe 内空闲；对每个 rank 的激活权重字节数算 HBM roofline | L2 的前提是否成立 | 计算 union 已接近 roofline（如 > 80%）时，L2 不立项 |
| G0-8 | vLLM 现成的 `async_scheduling` ABBA（不和投机解码同时开） | 零新代码能隐藏多少主机开销 | — |
| G0-9 | 读出 `GCFG_RMW_SECTION_MAX_SIZE_BYTES` / `GCFG_SCRATCHPAD_SRAM_MAX_SIZE` / `GCFG_HBM_GLOBAL_MEM_SIZE_MEGAS` 的默认值，并查有没有能放大映射池的配置项 | 有没有配置项能直接解决问题二或 SRAM 预算 | — |

---

## 6. Q5 最终裁决与修改方案

**裁决：需修改后再冻结。**

**必须修改：**
1. **更正 autopsy 的事实错误**：第 2 节表里标为“错/言过其实/口径错位”的条目（符号地址、无状态、Patch 成本、HCL 源码版本、Frame 粒度、13.37 vs 11.31、RoCE 口径与时延、问题四的归因）。
2. **把已有证据并入**：TRACE-CPU-LATENCY、CPU-DEPENDENCY-AND-OVERLAP（设备空闲拆解）、BOUNDED-CAPACITY、BRIDGE-INTEGRATION（“提交更早返回 ≠ 提速”），以及本报告 §1 的在途上限实测。
3. **按瓶颈预算重写立项依据**：以高频策略下约 11.3–11.5 ms 的一步为基准，拆成计算 union 5.73、recipe 内空闲 2.06、通信后空闲 1.90、提交偏斜 0.51、首个计算前 0.92 ms。**每一层都要写明它攻击哪一项、理论上限是多少。**
4. **删掉重复模块**：
   - L1_A：直接用 `PreparedRegionProgram` 当基线；
   - L2：并入 `gaudi-kernels-public`，不新建 `gaudirt::blocks`；
   - L3-R1/R2：如有需要，在 `NativeExecutionPlan` 或 worker 上扩展，不另起 runner。
5. **L1_B 改成“有条件立项”**：只有 G0-1、G0-3、G0-5 全部通过才继续，并补齐 §4 第 1–3、8、9 条的设计（CG 归属、私有内存与 ARC 命令的矛盾怎么解、异常处理、SDK 升级的退出策略）。
6. **新增“通信 track”**（G0-6 之后）：减少 collective 次数，或降低单次时延和交接成本。这是目前证据里最大的、没人负责的非计算项。
7. **门禁改成预算驱动**，每道门写明继续/终止阈值；所有 ABBA 固定 CPU 策略。

---

## 附：本次审查产物

- 在途上限探针源码：[prequeue_capacity_probe.cpp](file:///home/rate/.gemini/antigravity-cli/brain/4a518f11-bc3c-4a4b-9d72-8b43839ae657/scratch/prequeue_capacity_probe.cpp)
- 运行脚本：[run.sh](file:///home/rate/.gemini/antigravity-cli/brain/4a518f11-bc3c-4a4b-9d72-8b43839ae657/scratch/remote-run/run.sh)
- 环境与哈希：[env.txt](file:///home/rate/.gemini/antigravity-cli/brain/4a518f11-bc3c-4a4b-9d72-8b43839ae657/scratch/remote-run/env.txt)
- 原始输出：[plain.log](file:///home/rate/.gemini/antigravity-cli/brain/4a518f11-bc3c-4a4b-9d72-8b43839ae657/scratch/remote-run/plain.log)、[gdb.log](file:///home/rate/.gemini/antigravity-cli/brain/4a518f11-bc3c-4a4b-9d72-8b43839ae657/scratch/remote-run/gdb.log)
- 反汇编辅助脚本：[dis.py](file:///home/rate/.gemini/antigravity-cli/brain/4a518f11-bc3c-4a4b-9d72-8b43839ae657/scratch/dis.py)（导出符号与 PLT 解析）
- 远端目录：`/home/ai/gaudi-experiments/gaudirt-review2-20261023/`
