# Gaudi2 硬件底子与现有软件栈实证解剖（Software Autopsy）

> **文档说明与证据准则**：
> 本文档严格区分 **【无插桩配对计时】**、**【独立运行的 Profiler / Trace 采样】**、**【静态反汇编与符号表事实】** 与 **【基于第一性原理的估算/推断】**，绝不混用不同运行的测量口径。
> 全部证据归档清单见 [`GaudiRT/evidence/README.md`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/evidence/README.md)，包含：
> - 第一轮真机复核：[`GaudiRT/evidence/review-20261023/`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/evidence/review-20261023/summary.json)
> - 第二轮历史记录与反汇编摘要：[`GaudiRT/evidence/runtime-audit-20261023/`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/evidence/runtime-audit-20261023/prequeue-audit.json)
> - `libSynapse.so` (`1.24.2-481`) 静态反汇编与符号表：[`GaudiRT/evidence/design-review-20261023/`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/evidence/design-review-20261023/synapse-symbols.txt)
> - 第三轮在途上限纯 C++ + GDB 真机实测与两轮审查：[`GaudiRT/evidence/review2-20261023/`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/evidence/review2-20261023/gaudirt_design_review_round2.md)
> - 70 层 TP8 模型历史实测与时间线报告：[`production70-first-principles-a/result.json`](file:///home/rate/llm/Gaudi-LLM/instance-assets/Logic-20261018/production70-first-principles-a/result.json)、[`CPU-DEPENDENCY-AND-OVERLAP-20261011.md`](file:///home/rate/llm/Gaudi-LLM/gaudi-kernels-public/docs/CPU-DEPENDENCY-AND-OVERLAP-20261011.md)、[`BRIDGE-INTEGRATION-20261004.md`](file:///home/rate/llm/Gaudi-LLM/gaudi-kernels-public/docs/BRIDGE-INTEGRATION-20261004.md)、[`BOUNDED-CAPACITY-20261011.md`](file:///home/rate/llm/Gaudi-LLM/gaudi-kernels-public/docs/BOUNDED-CAPACITY-20261011.md)、[`BRIDGE-REPEATED-REPLAY-20261011.md`](file:///home/rate/llm/Gaudi-LLM/gaudi-kernels-public/docs/BRIDGE-REPEATED-REPLAY-20261011.md)

---

## 一、物理算力、带宽与通信延迟对标（同口径客观参数）

在相同模型、相同精度与相同访存量前提下，Intel Gaudi2 (96GB) 与主流 AI 加速卡的客观物理参数对比如下：

| 核心物理维度 | Intel Gaudi2 (96GB) | NVIDIA A100 (80GB SXM) | NVIDIA H100 (80GB SXM) | **客观物理能力定位与约束** |
| :--- | :--- | :--- | :--- | :--- |
| **制程与功耗** | 7nm / 600W | 7nm / 400W | 4nm / 700W | 与 A100 同代 7nm 制程，比 H100 落后半代 |
| **BF16 / FP16 矩阵算力** | **432 TFLOPS** (2×MME) | 312 TFLOPS | 989 TFLOPS | A100 的 **138%**；H100 的 **43.7%** |
| **FP8 矩阵算力** | **865 TFLOPS** (E4M3/E5M2) | 不支持原生 FP8 | 1,979 TFLOPS | 具备原生 FP8 优势（相对 A100）；为 H100 FP8 的 **43.7%**。**注：Gaudi2 MME 不支持原生 FP4/MXFP4，MXFP4 权重必须在 TPC 上解码** |
| **向量/非矩阵算力 (SIMD)** | **24×TPC** (2048-bit VLIW, 标称 ~11 TFLOPS FP32 / ~44 TFLOPS BF16) | 108×SM (~19.5 TFLOPS FP32) | 132×SM (~67 TFLOPS FP32) | 向量核数量有限（24 核），重度依赖 TPC 解码或归约时易撞上 TPC 算力瓶颈 |
| **HBM 显存容量与带宽** | **96 GB HBM2e @ 2.45 TB/s** | 80 GB @ 2.0 TB/s | 80 GB @ 3.35 TB/s | 标称带宽为 A100 的 **122.5%**、H100 SXM 的 **73.1%** |
| **片上高速缓存 (SRAM)** | **48 MB SRAM** (`sramSize = 50,331,648 B`) | 40 MB L2 | 50 MB L2 | 容量与 H100 L2 相当，采用软件显式管理（非硬件透明 Cache）；编译器对单个 RMW Section 默认有大小上限（`gaudi-kernels-public` 实测硬限 `16 MiB`） |
| **机内互联总带宽与点对点带宽** | **24×100GbE RoCEv2**（单卡总计 300 GB/s）<br/>8 卡机内 Scale-Up 用 21 口（聚合 **262.5 GB/s 单向**），**每对 Rank 间仅 3×100GbE = 37.5 GB/s 单向** | 300 GB/s 单向<br/>(600 GB/s 双向 NVLink 交换) | 450 GB/s 单向<br/>(900 GB/s 双向 NVLink 交换) | 聚合带宽为 A100 的 **87.5%**，但无 NVSwitch 全互连交换芯片，**任意两卡直接带宽仅 37.5 GB/s** |
| **小消息集合通信延迟** *(Decode 关键维度)* | **片上 RoCEv2 + HCL 流同步交接**<br/>（每步 141 次小包 AllReduce/AllGather，受协议栈、NIC 往返与跨流同步固定延迟主导） | NVLink P2P / NCCL | NVLink + NVLS 硬件多播归约 | Decode 单步 141 次通信每次传输仅数 KB~数十 KB，**物理耗时由“单次小消息固定延迟 × 141”主导，而非峰值带宽** |

---

## 二、70 层 TP8 模型稳态 Decode 的真实账单（严格区分测量口径）

### 1. 无插桩配对计时基线：CPU 策略（`uclamp`）影响约 `2.05 ~ 2.16 ms/step`
在 [`production70-first-principles-a/result.json`](file:///home/rate/llm/Gaudi-LLM/instance-assets/Logic-20261018/production70-first-principles-a/result.json) 与 [`CPU-DEPENDENCY-AND-OVERLAP-20261011.md:7-14`](file:///home/rate/llm/Gaudi-LLM/gaudi-kernels-public/docs/CPU-DEPENDENCY-AND-OVERLAP-20261011.md#L7-L14) 的两轮独立同进程无插桩对照中：
* **`production70-first-principles-a/result.json`（ABBA 四轮）**：
  * `runs[0], runs[3]`（`host_policy_enabled = false`, `uclamp = 0`）：**`13.366 / 13.370 ms/step`**（`74.82 / 74.80 TPS`）；
  * `runs[1], runs[2]`（`host_policy_enabled = true`, 全线程 `uclamp = 1024`）：**`11.319 / 11.310 ms/step`**（**`88.35 / 88.42 TPS`**），同进程配对节省 **`2.05 ms/step`**。
* **`CPU-DEPENDENCY-AND-OVERLAP-20261011.md`（独立配对运行）**：
  * 全线程 `uclamp` 从 `0` 提到 `1024`：合并 TPS 从 **`73.19` 升至 `86.91 TPS`**，单 Token 节省 **`2.156 ms`**（[`BRIDGE-REPEATED-REPLAY-20261011.md:54-61`](file:///home/rate/llm/Gaudi-LLM/gaudi-kernels-public/docs/BRIDGE-REPEATED-REPLAY-20261011.md#L54-L61) 复测亦为 `72.31/73.56 -> 86.86/86.64 TPS`）。
* **严谨归因**：源报告明确指出“**主要可复现因素是 CPU 调度/调频策略**”（`uclamp` 同时影响 Linux 调度器核心放置与 `schedutil` 频率选择，两者在实验中未单独拆离）。但这 `~2.1 ms` 明确属于主机操作系统策略而非 Synapse/Bridge 代码缺陷。因此，**后续所有 Gate 实验必须锁定 `uclamp = 1024` 与 NUMA 亲和性作为不可变的基线前提**。

### 2. Profiler / SDK Trace 下的时间线分布（独立采样运行，各项中位数不可直接相加）
在 [`CPU-DEPENDENCY-AND-OVERLAP-20261011.md:18-37`](file:///home/rate/llm/Gaudi-LLM/gaudi-kernels-public/docs/CPU-DEPENDENCY-AND-OVERLAP-20261011.md#L18-L37) 中，研究者在**完成无插桩计时之后，单独开启 SDK Trace 与 CPU Profiler** 采集了 Rank 0 连续 3 个 SDK Step 的各项中位数（注：**带 Profiler 采样的数字不能等同于无插桩 `11.31 ms` 运行本身，且各项中位数非互斥预算、不可直接相加**）：

| 观测维度（来自独立 SDK / CPU Trace 运行，取 3 步中位数） | 原 CPU 策略 (`uclamp=0`) | 高性能策略 (`uclamp=1024`) | 物理含义与边界说明 |
| :--- | ---: | ---: | :--- |
| **SDK Trace 采样总跨度** | — | **`11.530 ms`** | 首个计算前 `0.920 ms`，计算跨度 `10.481 ms`，末个计算后 `0.141 ms` |
| **① 物理 TPC / MME 计算 Union** | `5.727 ms` | **`5.728 ms`** | 物理计算引擎活跃并集（约为下节估算的 HBM 权重带宽下限 `2.1 ~ 2.5 ms` 的 **`2.3 ~ 2.7 倍`**） |
| **② 单个 Recipe 内部无 TPC/MME 区间** | `2.059 ms` | **`2.056 ms`** | 已编译 Recipe 内部无 TPC/MME 活动的区间（注：PDMA/EDMA 可能与其它工作重叠，不全是全引擎空闲；修改外部 Launch API 无法消除） |
| **③ 相邻 Recipe 之间无计算区间** | `7.473 ms` | **`2.702 ms`** | 相邻计算 Recipe 之间的间隙（涵盖每步 141 次集合通信与流同步） |
| ↳ 子项 3a：**8 个 Rank 对应通信调用均已返回后**的间隙 | — | **`1.900 ms`** | 已包含在 `2.702 ms` 内。主机侧通信 API 已返回，设备端在执行 `synStreamSyncHCLStreamHandle` (`EventNetwork`) 流同步交接与 RoCE 小包传输 |
| ↳ 子项 3b：**最慢 Rank 尚未开始对应通信调用**的间隙 | — | **`0.507 ms`** | 已包含在 `2.702 ms` 内。真正发生在最后一个 Rank 尚未进入对应 HCCL 调用前的**主机提交偏斜（Submission Skew）窗口** |
| **④ 首个计算前 (`0.920 ms`) + 末个计算后 (`0.141 ms`)** | — | **`1.061 ms`** | Step 边界等待上一 Token 采样结果、输入准备与首算子启动 |
| **CPU Profiler 下 `NativePlanSubmit` 中位数** | `8.91 ~ 9.44 ms` | **`5.18 ~ 5.40 ms`** | 含 SDK/背压的主机墙钟时间，**与设备端 `10.481 ms` 计算跨度高度重叠**；其中 Retained 绑定预检约 **`98 ~ 127 µs`** |
| **另一次无 CPU Profiler 的主机侧嵌套观测** | — | `sample_tokens` `11.073 ms` | 含输入准备 `0.282 ms`、`graph.replay` `5.514 ms`、结果回读（含等设备完成）`4.872 ms`；调用互相嵌套且与设备重叠，不可相加 |

---

## 三、现有栈各层的真实能力与病灶审计

### 1. 现有栈已经具备且不应重复建设的能力（防膨胀红线）

| 能力模块 | 现有代码位置与证据状态 | 审计结论与限定条件 |
| :--- | :--- | :--- |
| **Synapse 图编译器 (`synGraphCompile`) 与显式 RMW SRAM 分配** | [`sram_mme_probe.cpp`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/evidence/review-20261023/sram_mme_probe.cpp) 与 [`mxfp4_compact.cpp:36-47`](file:///home/rate/llm/Gaudi-LLM/gaudi-kernels-public/csrc/ops/mxfp4_compact.cpp#L36-L47) 验证了通过 `synSectionSetPersistent(false)` + `synSectionSetRMW(true)` 可将 `2 KB` (`[64, 16]` BF16) 中间张量分配在 SRAM (`0x1000fffffd080000`)。 | **复用 `synGraphCompile`**，绝不手写底层私有 ARC/ECB 指令。注：当前探针仅验证了 `2 KB` 张量且缺 `RMW=false` 对照，真实子图受 `GCFG_RMW_SECTION_MAX_SIZE_BYTES`（`16 MiB`）等编译器容量约束，需在 Gate 0 补齐边界实测。 |
| **预绑定 `synLaunchTensorInfoExt` 与 C++ 区域重放 (`PreparedRegionProgram`)** | [`NativeExecutionPlan.cpp`](file:///home/rate/llm/Gaudi-LLM/gaudi-pytorch-bridge/backend/habana_device/NativeExecutionPlan.cpp) 已实现捕获期绑定冻结、单次 HCCL 队列提交 `144` 计算 + `141` 通信命令（`adapter_replays = 0`），并在 [`BRIDGE-REPEATED-REPLAY-20261011.md`](file:///home/rate/llm/Gaudi-LLM/gaudi-kernels-public/docs/BRIDGE-REPEATED-REPLAY-20261011.md) 中新增了 `graph.replay_many(N)` 与存储一致性保护。 | **严禁在 `GaudiRT` 中重写预绑定 `synLaunch` 循环执行器**（避免重蹈 `gaudi-kernels-public` 已删除的 `fixed_plan.cpp` 覆辙）。 |
| **固定地址 Decode 输入槽位** | `MimoHPUWorker` 的 `_prepared_decode_input` 已实现固定槽位复用（`input_layout_hits = 1386, full_copy_fallbacks = 0`）。 | **无需重复开发固定输入池**。 |
| **HCL (`libhcl.so`) 跨流同步与集合通信** | 本地参考源码（[`hcl_graph_sync.cpp:434`](file:///home/rate/llm/Gaudi-LLM/instance-assets/numerics-contract-20260925/workspace/deps/HCL/hcl/src/platform/gen2_arch_common/hcl_graph_sync.cpp#L434)，注：**该源码树版本为公开版 `1.21.0-555`，而装机运行库为 `1.24.2-481`**）及 `1.24.2-481` 符号表表明，HCL 内部通过硬件 Long-SOB 与 Monitor 实现设备侧同步。 | **不重写 HCL**；但需如下文 §三.4 所述，准确区分 Bridge 区域事件与 HCL 内部触发的 `EventNetwork`。 |

---

### 2. 纯 C++ `synLaunchWithExternalEventsExt` 的真实开销与在途上限（[`review2-20261023`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/evidence/review2-20261023/remote-run/plain.log) 实测）

在 [`prequeue_capacity_probe.cpp`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/evidence/review2-20261023/prequeue_capacity_probe.cpp)（系统 `1.24.2-481`、Module 7、纯 C++ 无插桩）中，先发射一个 `122.13 ~ 122.19 ms` 的大矩阵乘 Gate Recipe 堵住设备，随后在同一 Compute Stream 上连续发射小 Recipe，得到 [`plain.log`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/evidence/review2-20261023/remote-run/plain.log) 与 [`gdb.log`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/evidence/review2-20261023/remote-run/gdb.log) 的精确数据：

1. **三组节点/绑定规模与双 Recipe 交替的实测原始数据**：

   | 实验配置 (`plain.log`) | TPC 节点数 | 绑定张量数 | 阻塞前单次 Launch P50 (`pre_block_p50_us`) | 首次阻塞的 0-based 序号 (`first_block_index`) | 阻塞调用耗时 (`max_us`) |
   | :--- | ---: | ---: | ---: | ---: | ---: |
   | `small_1node_3bind` (rep 0 / 1) | 1 | 3 | **`6.62 / 7.02 µs`** | **`573`**（即前 573 次 `0..572` 成功，**第 574 次**阻塞） | **`118.31 / 118.17 ms`** |
   | `small_1node_3bind_alt2recipes` (rep 0 / 1) | 1 (双 Recipe 交替) | 3 | **`6.99 / 6.74 µs`** | **`571`**（即前 571 次 `0..570` 成功，**第 572 次**阻塞） | **`118.25 / 118.36 ms`** |
   | `small_8node_24bind` (rep 0 / 1) | 8 | 24 | **`11.26 / 10.82 µs`** | **`573`**（前 573 次成功，**第 574 次**阻塞） | **`115.59 / 115.97 ms`** |
   | `small_32node_96bind` (rep 0 / 1) | 32 | 96 | **`25.36 / 24.10 µs`** | **`573`**（前 573 次成功，**第 574 次**阻塞） | **`107.60 / 108.24 ms`** |

   * **关于 Launch 耗时斜率的严谨说明**：以上三组单 Recipe 配置（`1 节点/3 绑定`、`8 节点/24 绑定`、`32 节点/96 绑定`）满足 `绑定数 = 3 × 节点数`。虽然这三点在数值上可拟合为约 `6.2 µs + 0.19 µs × 绑定数`，但由于**节点数与绑定数完全共线**，现有 3 个数据点无法分离出“每增加 1 个绑定”与“每增加 1 个节点”各自的独立耗时，需在 Gate 0 中补充“固定 1 节点、仅改变绑定数”的对照实验。

2. **阻塞时的精确 GDB 调用栈与库偏移**（[`gdb.log:81-105`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/evidence/review2-20261023/remote-run/gdb.log#L81-L105)，`libSynapse.so` 加载基址 `0x7ffff4000000`）：
   ```text
   #0  __GI___ioctl (fd=20, request=3224921155)
   #1  hlthunk_wait_for_interrupt_by_handle_with_timestamp () from libhl-thunk.so
   #2  hlthunk_wait_for_interrupt_by_handle () from libhl-thunk.so
   #3  Scal::completionGroupWait(...) () from libscal.so
   #4  0x00007ffff556162c (+0x156162c) in ScalCompletionGroupBase::longSoWait(...)
   #5  0x00007ffff55bf721 (+0x15bf721) in ScalStreamBase::longSoWait(...)
   #6  0x00007ffff571dab1 (+0x171dab1) in RecipeLauncher::checkCompletionCopy(unsigned long)
   #7  0x00007ffff56d8ecf (+0x16d8ecf) in LaunchTracker::waitForCompletionCopy()
   #8  0x00007ffff571f1d8 (+0x171f1d8) in RecipeLauncher::getMappedMemoryInfo()
   #9  0x00007ffff5723264 (+0x1723264) in RecipeLauncher::prepareForDownload(LaunchInfo const&)
   #10 0x00007ffff5724cbb (+0x1724cbb) in RecipeLauncher::launch(LaunchInfo const&)
   #11 0x00007ffff577031d (+0x177031d) in QueueComputeScal::launch(LaunchInfo const&)
   ...
   #21 0x00007ffff5141550 (+0x1141550) in synLaunchWithExternalEventsExt
   ```
   * **严谨定性**：实验直接证明了在该预排队负载下，**`RecipeLauncher::getMappedMemoryInfo` (`+0x171f1d8`) $\rightarrow$ `LaunchTracker::waitForCompletionCopy` (`+0x16d8ecf`) $\rightarrow$ `RecipeLauncher::checkCompletionCopy` (`+0x171dab1`) 是最先触发的主机阻塞路径**（等待最老一次在途 Launch 的 Copy/Download 完成中断），而其它资源池在该测试中并非首个耗尽点。这与八卡合成实验（[`BOUNDED-CAPACITY-20261011.md:29-31`](file:///home/rate/llm/Gaudi-LLM/gaudi-kernels-public/docs/BOUNDED-CAPACITY-20261011.md#L29-L31)）中“3 步（426 次调用）PASS、4 步（568 次调用）在 `peak_call_index = 570` 处触发 ~810 ms 阻塞”完全吻合。
   * **工程适用边界**：单步 70 层 Decode 含 `144` 次计算 Launch，因此**单步同步执行乃至 $\le 3$ 步连续预排队（$\le 432$ 次计算 Launch）均不会触碰 `571 ~ 573` 的在途上限**；该上限仅在 **$\ge 4$ 步完全预排队（不补命令）** 时才会触发。

---

### 3. 为什么“在 `libSynapse.so` 外部绕开 `QueueComputeScal` 的驻留实例（`L1_B`）”不可行？（直接裁决砍掉）

结合静态反汇编（[`design-review-20261023/recipe-launch.asm`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/evidence/design-review-20261023/recipe-launch.asm)、[`recipe-stages.asm`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/evidence/design-review-20261023/recipe-stages.asm)、[`queue-launch.asm`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/evidence/design-review-20261023/queue-launch.asm)）与符号表（[`synapse-symbols.txt`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/evidence/design-review-20261023/synapse-symbols.txt)），从外部绕过 `RecipeLauncher::launch` 存在三大死结：

1. **Completion Group (`CG`) 计数器失步**：`QueueComputeScal::launch` 内部维护主机侧预期的 SCAL Completion Group 目标值。外部绕过会导致 Synapse 主机侧记录的 `CG` 计数与设备实际 Long-SOB 计数失步，静默破坏 `synEventRecord`、`synStreamSynchronize` 与 HCL `streamWaitEvent`。
2. **内部池按 Long-SOB 代际回收**：在 `RecipeLauncher::scalEnqueue` 末尾（`0x17247c8`, `0x17247e1`，见 [`recipe-stages.asm:462-470`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/evidence/design-review-20261023/recipe-stages.asm#L462-L470)），代码通过两个管理对象的虚表指针（`vtable + 0x10`）发生间接虚调用，传入 `(longSo.m_index, longSo.m_targetValue, true)`（导出符号表中对应签名为 `NBuffQueueAllocator::setLongSo` 与 `NBuffDeviceAllocator::setLongSo`），由此推断内部池槽位按 Long-SOB 完成代际自动回收复用。钉死槽位会阻塞池轮转，而改用外部 `synDeviceMalloc` 则无法走 `scalEnqueue`。
3. **外部内存发令迫使手写未公开 ARC 调度包**：若使用外部 HBM 存放描述符，就必须自行向 SCAL 流拼装私有 ARC 调度包（`addBaseAddresses` / `addEcbListWithBarrierAndSend`），这直接违反了“不手写未公开固件协议包”的铁律；同时，在 `11.31 ms` 时间线中，真正因最慢 Rank 主机提交滞后产生的间隙仅 `0.507 ms`。

**裁决**：**在设计阶段直接砍掉“外部 Hack `libSynapse.so` 绕过 `RecipeLauncher` 的驻留实例（`L1_B`）”**，不在 Gate 0 中浪费时间编写注定触犯暗礁三的私有协议 Hack。

---

### 4. `EventNetwork` 的真实归属层（纠正跨层归因错误）

此前将每步 141 次通信交接中的 `EventNetwork` 归因于 `gaudi-pytorch-bridge` 的 `NativeExecutionPlan.cpp` 是错误的：
* **源码与反汇编事实**：
  * **Bridge 层**：`NativeExecutionPlan.cpp` 中的 `PreparedRegionProgram` 仅在每个编译区域的**入口和出口**调用 `record_and_wait_for_event`（70 层模型每步 2 个区域，**每步仅约 4 次 Bridge 事件**）。
  * **HCL $\rightarrow$ Synapse 层**：每次调用 `hcclAllReduce` / `hcclAllGather`（见 `hccl.cpp` 第 567/665/711/.../969 行）时，是 **`libhcl.so` 内部首先调用 `syncHCLStreamHandle` $\rightarrow$ `synStreamSyncHCLStreamHandle`**，再由 `libSynapse.so` 的 `DeviceCommon::syncHCLStreamHandle` 构造 `EventNetwork` 并通过 `StreamsContainer::addJob` 排入队列。
* **工程推论**：仅修改 Bridge 的 `NativeExecutionPlan.cpp` 最多只能影响区域边界的 ~4 次事件，**触及不到 141 次集合通信内部由 HCL 触发的 `synStreamSyncHCLStreamHandle` (`EventNetwork`)**。要压缩 `2.702 ms`（含全员提交后 `1.900 ms`）的通信间隙，必须首先在 Gate 0 中实测分离“HCL `synStreamSyncHCLStreamHandle` 交接开销”与“RoCE 小包物理通信耗时”，或从图结构上减少每步集合通信次数。

---

## 四、权重带宽下限估算与三大互斥优化方向（对准 `11.31 ms` 时间线）

### 1. 70 层 MiMo TP8 单 Token 权重读取下限估算（第一性原理 Roofline）
根据模型 `config.json` 与量化配置（70 层、TP8、无专家并行、Batch=1、4K 上下文），每张卡（Rank）每生成一个 Token 必须从 HBM 读取的字节数估算如下：

| 权重/状态分项 | 精度与配置说明 | 每 Rank 每 Token 读取量估算 | 占总读取量比例 |
| :--- | :--- | ---: | ---: |
| **Attention `o_proj`** | **BF16（位于 `ignored_layers`，共 71 项未量化）** | $70 \times 25.2\text{ MB} \approx \mathbf{1.76\text{ GB}}$ | **~33.8%** *(单项最大)* |
| **Attention `qkv_proj`** | FP8（Q: $128\times 192$, K: $8\times 192$, V: $8\times 128$） | $70 \times 20.8\text{ MB} \approx \mathbf{1.46\text{ GB}}$ | ~28.1% |
| **MoE 激活专家权重** | 69 层 MoE，Top-8 激活，`store_dtype=mxfp4`（连缩放因子约 `4.25 bit/param`） | $69 \times 20.0\text{ MB} \approx \mathbf{1.38\text{ GB}}$ | ~26.5% |
| **第 0 层 Dense MLP + Router + `lm_head` (1/8) + KV 读取** | 混合精度 + 4K 上下文 KV 读取 | $\approx \mathbf{0.60\text{ GB}}$ | ~11.5% |
| **合计每 Rank 每 Token 读取量** | — | **$\approx 5.20\text{ GB}$** | **100%** |

* **HBM 带宽时间下限**：
  * 按标称峰值带宽 `2.45 TB/s` 计：$5.20\text{ GB} / 2.45\text{ TB/s} \approx \mathbf{2.12\text{ ms}}$；
  * 按实际可达有效带宽（85%，`~2.08 TB/s`）计：$5.20\text{ GB} / 2.08\text{ TB/s} \approx \mathbf{2.50\text{ ms}}$。
* **关键物理推论**：
  1. 当前物理计算 Union（`5.728 ms`）约为有效带宽下限（`2.50 ms`）的 **`~2.3 倍`**（相对峰值 `2.12 ms` 约 `2.7 倍`）。
  2. **Gaudi2 MME 不支持 FP4，MXFP4 专家权重无论走纯 TPC GEMV 还是“TPC 解码 $\rightarrow$ SRAM $\rightarrow$ MME”，解包与反量化都必须在 24 个 TPC 上完成**。这解释了为何 `gaudi-kernels-public` 此前在小 $M$ 下尝试 `mxfp4_compact` v2/v3（TPC 解码后交 MME）曾因解码吞吐与引擎交接开销被速度门拒绝。因此，**绝不能预设“把 MXFP4 算子改成 MME + SRAM 就一定更快”**，必须先由 Gate 0 的逐 Recipe 剖析定位真正的低效点。
  3. **未量化的 BF16 `o_proj`（`1.76 GB`，占 34%）是单项最大的带宽消耗源**，也是评估计算/访存收益时不可忽视的头号大户。

---

### 2. 去重后的三大真实攻关桶（不设重复分桶、不做违背统计口径的简单相加）

将重复计入 `2.702 ms` 的 `0.507 ms` 发射偏斜归并回通信与流交接桶后，所有后续工作严格围绕以下三个不重叠的物理问题展开：

| 攻关赛道 | 对应 Profiler 时间线指标（注：独立采样中位数，非互斥可加预算） | 核心科学问题与攻关点（全部须先过 Gate 0 实验裁决） |
| :--- | :--- | :--- |
| **赛道 A：Recipe 内外计算效率与内部气泡** | • 物理计算 Union：**`5.728 ms`**（对比带宽下限 `2.1 ~ 2.5 ms`）<br/>• Recipe 内无 TPC/MME 区间：**`2.056 ms`** | 通过 `G0-6` 导出 144 个 Recipe 的逐个耗时与字节账单，查明 `5.73 ms` 与 `2.06 ms` 究竟花在哪些具体算子（如 BF16 `o_proj`、MoE TPC 解码/归约、Attention 还是内部 DMA 等待），仅对实测可省 $\ge 0.20\text{ ms/step}$ 的目标做针对性子图/内核优化。 |
| **赛道 B：141 次通信、流同步交接与发射偏斜** | • 相邻 Recipe 间无计算区间：**`2.702 ms`**<br/>&nbsp;&nbsp;↳ 含 8 卡通信提交后间隙 **`1.900 ms`**<br/>&nbsp;&nbsp;↳ 含最慢 Rank 未提交偏斜 **`0.507 ms`** | 通过 `G0-5` 实测拆解 `2.702 ms` 中：① 141 次 `libhcl.so` $\rightarrow$ `synStreamSyncHCLStreamHandle` (`EventNetwork`) 同步开销；② 141 次 8 卡 RoCE 小消息物理网络延迟；③ 主机提交偏斜（`0.507 ms`）。据此决定是调整通信/流同步方式还是减少每层集合通信次数。 |
| **赛道 C：Step 边界跨 Token 反馈屏障** | • 首个计算前：**`0.920 ms`**<br/>• 末个计算后：**`0.141 ms`** | 先通过零代码的 **`G0-3`（vLLM `async_scheduling` ABBA 对照）** 检验主机侧重叠能否直接消除首算子前间隙；再对比基于现有 `replay_many`（[`BRIDGE-REPEATED-REPLAY-20261011.md`](file:///home/rate/llm/Gaudi-LLM/gaudi-kernels-public/docs/BRIDGE-REPEATED-REPLAY-20261011.md)）的 $\le 3$ 步设备端状态反馈闭环（避开 `571 ~ 573` 在途上限）。 |
