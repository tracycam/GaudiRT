# GaudiRT 架构设计规范（Evidence-Driven Runtime Architecture & Boundary Specification）

> **模块职责边界铁律（Strict Three-Layer Responsibility Boundary）**：
>
> - **`gaudi-kernels`（[`/home/rate/llm/Gaudi-LLM/gaudi-kernels-public`](file:///home/rate/llm/Gaudi-LLM/gaudi-kernels-public)）**：负责 TPC/MME 内核实现、算子融合、权重量化计算（如 `o_proj` FP8）、单核 TPC 算子多核化（1-core $\to$ 24-core）与 SRAM 分块。`GaudiRT` 探针发现的算子级瓶颈（[`g0-6-recipe-roofline-audit.json`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/evidence/gate0-20261023/g0-6-recipe-roofline-audit.json)）作为证据移交算子库，**算子融合与权重量化不属于 `GaudiRT` 主线**。
> - **`Bridge / vLLM runner`（[`/home/rate/llm/Gaudi-LLM/gaudi-pytorch-bridge`](file:///home/rate/llm/Gaudi-LLM/gaudi-pytorch-bridge)）**：负责框架接入、模型构图、输入准备、采样回传与请求调度。
> - **`GaudiRT`（[`/home/rate/llm/Gaudi-LLM/GaudiRT`](file:///home/rate/llm/Gaudi-LLM/GaudiRT)）**：聚焦**面向编译产物的通用运行时执行能力**——编译产物实例化与重放、计算与集合通信统一提交、在途资源（`MappedMemMgr` Chunk 信用）管理、完成回收与确定性背压。复用已有机制，**只解决经过验证的运行时缺口，并产生可验证的硬件收益**，绝不把 `PreparedRegionProgram` 简单搬出来再包装一次，也不写耦合 `libSynapse.so` 私有 ABI 偏移的代码。

---

## 一、核心问题回答：现有运行时缺少什么通用能力？`GaudiRT` 以什么机制补齐？收益多少？

通过对 70 层 TP8 生产轨迹（8 卡 `sdk-scope-1-rank0..7.jsonl` 及 `production70-first-principles-a/result.json`）与 [`NativeExecutionPlan.cpp`](file:///home/rate/llm/Gaudi-LLM/gaudi-pytorch-bridge/backend/habana_device/NativeExecutionPlan.cpp) / [`NativeDependencyPlan.h`](file:///home/rate/llm/Gaudi-LLM/gaudi-pytorch-bridge/backend/habana_device/NativeDependencyPlan.h) 的逐微秒对齐审计，现有运行时（`NativeExecutionPlan` + `PreparedRegionProgram`）在稳态 `11.31 ms` 步长内，真正因**运行时机制缺口**损失的设备时间集中在前 `2.0 ms`（步首主机准备停顿 `~0.95 ms` + 步中 `Command 5` 跨流断层导致的 `~0.77 ms` 早期设备气泡），外加多步在途提交缺乏精确信用背压管理。

`GaudiRT` 针对这 **3 个经过验证的运行时缺口（`RT-1`、`RT-2`、`RT-3`）** 提供通用机制并完成了 8 卡 Gaudi2 真机 ABBA 验证（[`rt-1-2-3-runtime-gaps.json`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/evidence/gate0-20261023/rt-1-2-3-runtime-gaps.json)）：

| 运行时缺口编号 | 现有运行时 (`NativeExecutionPlan`) 的具体缺陷 | `GaudiRT` 补齐的通用机制 | 8 卡 Gaudi2 真机实测收益 (`uclamp=1024`, ABBA) |
| :--- | :--- | :--- | :--- |
| **`RT-1`**<br/>**混合命令断层与跨流切分缺口** | 生产单步 `286` 个命令中，`command 5` 同时包含 `compute: true` 与 `collectives: 1`（内部 `o_proj` 写出的张量紧接着被同命令的 `AllGather` 读取）。现有 [`NativeDependencyPlan.h:67`](file:///home/rate/llm/Gaudi-LLM/gaudi-pytorch-bridge/backend/habana_device/NativeDependencyPlan.h#L67) 与 [`PreparedRegionProgram::compile`](file:///home/rate/llm/Gaudi-LLM/gaudi-pytorch-bridge/backend/habana_device/NativeExecutionPlan.cpp#L122) 拒绝混合命令，导致全步被劈成 **2 个 Region（`[0..4]` 与 `[6..285]`）**：<br/>1. 在 `command 4 -> 5 -> 6` 处触发 `leave()` + `publish(Region 0)`；<br/>2. `command 5` 退回旧 `RecipeLauncher::Launch`，其集合通信因 `replay_stream == nullptr` 被迫切到独立的 `comm_stream`（`stream 6`），插入 4 次跨流 `synEventRecord` / `synStreamWaitEvent`；<br/>3. 导致 8 卡设备在 `k=0..5` 处产生 **`644 ~ 806 µs`（均值 `~770 µs`）的早期启动气泡**。 | **[`UnifiedRegionLowerer`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/include/gaudirt/runtime/unified_region_lowerer.hpp)**：<br/>识别具有内部 RAW 依赖（`compute_writes` $\cap$ `collective_reads` $\neq \emptyset$）且无外部张量事件的混合命令，将其在编译期正交拆解为两个连续原子子调用（`SubCallKind::Compute` + `SubCallKind::Collective`），使全步 `286` 个逻辑命令（`145` 计算 + `142` 通信）100% 合并为 **`1` 个统一单流 Region（`[0..285]`，`coalesced_collectives: 141/142 -> 142/142`）**，彻底消除步中跨流切换与二次 Region 发布。 | • 早期窗口（`Commands 0..7`）设备耗时从 **`751.2 ~ 754.6 µs`** 降至 **`159.4 ~ 161.2 µs`**，8 卡一致消除 **`591.8 ~ 594.3 µs`（`~0.59 ms`）** 早期设备气泡；<br/>• 结合 `RT-2` 后，单步全图设备执行跨度净省 **`600.8 ~ 1,076.5 µs`（`0.60 ~ 1.08 ms/步`）**。 |
| **`RT-2`**<br/>**步首/步尾 $O(N_{\text{bindings}})$ 全量预检与地址事件表扫描缺口** | 现有运行时在每步入口和每个 Region 进出时：<br/>1. 遍历 `2,391` 个 `native_boundaries_` 与 `2,392` 个 `RetainedTensorBinding::matches`；<br/>2. 在 `PreparedRegionProgram::replay` 入口对 `2,392` 个依赖地址执行 `stream_event_manager::get_wait_events` 哈希查找、排序去重；<br/>3. 在 `PreparedRegionProgram::publish` 出口对 `2,392` 个地址执行 `add_producer` 哈希查找与 `event::remove_device_ptr` 线性删除（2 个 Region 时每步执行 2 轮）。<br/>这也是生产轨迹中每步 `enq0` 前 **`863 ~ 1,018 µs` 主机发射延迟**的核心来源。 | **[`GenerationEpochTracker`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/include/gaudirt/runtime/generation_epoch_tracker.hpp)**：<br/>引入单调递增的绑定代数（`binding_generation_`）与外部流写入代数（`external_stream_epoch_`）。当静态图绑定未发生重绑且无外部流写入 Region 足迹地址时：<br/>1. 绑定校验退化为 **`O(1)` 代数比对**；<br/>2. 同流连续重放依赖硬件 FIFO 保序，免除 `2,392` 地址等待扫描；<br/>3. 步尾以 **单个 Region Epoch 事件（`O(1)`）** 替代 `2,392` 次 `events_by_addr_` 哈希更新与线性移除。 | • 独立 ABBA 实测（`2,392` 绑定与地址）：每步主机预检 + 事件表维护耗时从 **`990.44 µs`** 降至 **`1.07 µs`**，净省 **`989.37 µs/步`（加速比 `923.9x`）**；<br/>• 8 卡整步主机提交耗时净省 **`734.4 ~ 1,092.8 µs/步`**。 |
| **`RT-3`**<br/>**Recipe 复用拓扑下的在途 Chunk 信用与确定性背压缺口** | `libSynapse.so` 的 `MappedMemMgr` 存在硬上限（每队列 `576` 个 DataChunk，门控占 `1` 个后余 `575` 个），耗尽时在 `getMappedMemoryInfo -> LaunchTracker::waitForCompletionCopy` 发生百毫秒级硬阻塞。现有运行时无在途信用感知，盲目调用 `replay_many(K)` 在超限时会在步中随机卡死主机提交线程。 | **[`InFlightCreditManager`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/include/gaudirt/runtime/inflight_credit_manager.hpp) & [`CompiledRegionExecutor`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/include/gaudirt/runtime/compiled_region_executor.hpp)**：<br/>1. 基于 `G0-7` 证明的精确硬件守恒律：<br/>$$2\,N_{\text{distinct}} + N_{\text{in\_flight}} \le 575$$<br/>2. 审计证实生产模型每步 `145` 次计算仅复用 **`12` 个不同 `synRecipeHandle`**（MoE 与 Attention 层间高度复用），故真实模型无阻塞上限为 $575 - 2\times 12 = \mathbf{551}$ 次（**`3.80` 个完整步**，而非 144 独立 Recipe 假想下的 `287` 次）；<br/>3. 按步绑定完成事件，通过 `synEventQuery` 非阻塞回收已完成步信用，并在超预算前于**步边界**执行确定性背压等待。 | • 真机复现 12-Recipe 复用拓扑（`145` 计算/步）：理论预测首次阻塞点 **`551`**，真机实测首次阻塞点 **`551`（零误差，`3.80` 步）**；<br/>• 开启 `InFlightCreditManager` 后，连续提交 `K = 1, 2, 3, 4` 步（最高 `580` 次 Launch > `551`）全程 **`mid_step_waitForCompletionCopy_blocks = 0`**，单次 Launch 最大耗时仅 **`39.59 µs`**（对比无信用管理的 `1,444.36 µs` 阻塞）。 |

---

## 二、现有资产映射与职责划分总表

| 模块/方向 | 所属责任层 | 架构裁决 |
| :--- | :--- | :--- |
| **混合命令单流 Region 统一降维 (`RT-1`)** | **`GaudiRT`**（核心机制）$\rightarrow$ 赋能 `Bridge` | 由 [`UnifiedRegionLowerer`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/include/gaudirt/runtime/unified_region_lowerer.hpp) 实现并验证，消除 `command 5` 断层。 |
| **代数守卫绑定预检与 `O(1)` Epoch 围栏 (`RT-2`)** | **`GaudiRT`**（核心机制）$\rightarrow$ 赋能 `Bridge` | 由 [`GenerationEpochTracker`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/include/gaudirt/runtime/generation_epoch_tracker.hpp) 实现并验证，消除每步 `2,392` 项哈希/线性扫描。 |
| **Recipe 复用感知的在途信用与背压管理 (`RT-3`)** | **`GaudiRT`**（核心机制） | 由 [`InFlightCreditManager`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/include/gaudirt/runtime/inflight_credit_manager.hpp) 与 [`CompiledRegionExecutor`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/include/gaudirt/runtime/compiled_region_executor.hpp) 实现并验证。 |
| **外部绕过 `RecipeLauncher` 的私有 ABI 驻留实例 (`L1_B`)** | 已否决 | 反汇编（[`design-review-20261023/*.asm`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/evidence/design-review-20261023/recipe-launch.asm)）证实破坏 `CG` 完成计数与内存池回收，**彻底砍掉**。 |
| **TPC/MME 算子融合、1-core 算子多核化、`o_proj` FP8 量化** | **`gaudi-kernels`** | `GaudiRT` 已通过 `G0-6`（[`g0-6-recipe-roofline-audit.json`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/evidence/gate0-20261023/g0-6-recipe-roofline-audit.json)）完成证据定位并移交 `gaudi-kernels`；**从 `GaudiRT` 主线撤下**。 |
| **`decode_graph.py` 异步调度准入与 Sampler/Input 编排** | **`Bridge / vLLM runner`** | `GaudiRT` 已通过 `G0-3/G0-4`（[`g0-3-4-host-and-path-audit.json`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/evidence/gate0-20261023/g0-3-4-host-and-path-audit.json)）定位 3 处 `use_async_scheduling` 门禁并移交上层。 |

---

## 三、架构数据流与组件关系图

```mermaid
flowchart TD
    subgraph Upper["上层框架与算子库 (非 GaudiRT 主线职责)"]
        GK["gaudi-kernels:<br/>TPC/MME 内核、算子融合、o_proj 量化、SRAM 分块"]
        BR["Bridge / vLLM Runner:<br/>模型构图、输入准备、采样与请求调度"]
    end

    subgraph GRT["GaudiRT: 面向编译产物的通用运行时核心 (include/gaudirt/runtime/)"]
        URL["RT-1: UnifiedRegionLowerer<br/>混合 Compute+Collective 命令内联 RAW 识别与原子拆解<br/>286 命令 -> 1 个统一 Region (145 Compute + 142 Collective)"]
        GET["RT-2: GenerationEpochTracker<br/>O(1) 代数守卫绑定校验 + 单流 FIFO 免等待 + O(1) 步尾 Epoch 围栏<br/>替代每步 2,392 绑定/地址哈希扫描 (990 us -> 1.07 us)"]
        ICM["RT-3: InFlightCreditManager<br/>精确守恒律: 2 * N_distinct + N_in_flight <= 575<br/>12 复用 Recipe -> 551 次上限 (3.80 步) + synEventQuery 非阻塞回收 + 步边界背压"]
        CRE["CompiledRegionExecutor<br/>统一编译产物实例化、单步/多步重放与在途生命周期引擎"]
        URL --> CRE
        GET --> CRE
        ICM --> CRE
    end

    subgraph HW["底层公开 SDK 接口 (零私有 ABI 偏移依赖)"]
        SYN["libSynapse.so (synLaunchWithExternalEventsExt / synEvent*)"]
        HCL["libhcl.so (hcclAllGather / hcclAllReduce on compute_stream)"]
    end

    GK -. "编译产物 (synRecipeHandle)" .-> URL
    BR -. "捕获命令流与绑定快照" .-> URL
    CRE ==> SYN
    CRE ==> HCL
```

---

## 四、`GaudiRT` 仓库结构

```text
GaudiRT/
├── AGENTS.md                                         # 项目工程准则与方法学约束（零实现细节）
├── CMakeLists.txt                                    # C++20 运行时核心、单元测试与硬件探针构建配置
├── docs/
│   ├── software-autopsy.md                           # 硬件参数、11.31ms 基线、573/571 阻塞栈、5.2GB 权重下限与现有栈审计
│   └── DESIGN.md                                     # 本设计规范：三层职责边界、RT-1/RT-2/RT-3 核心机制与真机验证收益
├── include/gaudirt/
│   ├── common/
│   │   ├── host_policy.hpp                           # Linux sched_setattr (uclamp_min=1024)、CPU 亲和性与无插桩环境校验
│   │   ├── synapse_raii.hpp                          # Synapse 设备/流/带时间戳事件对 (EVENT_COLLECT_TIME) RAII 封装
│   │   └── abba_harness.hpp                          # 同起止边界 ABBA 交错采样与分位数统计工具
│   └── runtime/
│       ├── unified_region_lowerer.hpp                # RT-1: 混合计算+通信命令原子降维与单流 Region 合并器
│       ├── generation_epoch_tracker.hpp              # RT-2: O(1) 代数守卫绑定预检与 Region Epoch 围栏追踪器
│       ├── inflight_credit_manager.hpp               # RT-3: MappedMemMgr (575 - 2*N_distinct) 在途信用与步边界背压管理器
│       └── compiled_region_executor.hpp              # 集成 RT-1/RT-2/RT-3 的通用编译产物重放执行器
├── tests/
│   └── runtime_unit_tests.cpp                        # GaudiRT 运行时核心机制确定性单元测试 (gaudirt_runtime_unit_tests)
├── probes/
│   ├── prequeue_capacity_probe.cpp                   # 定位 573/571 次 MappedMemMgr/waitForCompletionCopy 阻塞上限
│   ├── g0_2_launch_and_sram_probe.cpp                # G0-2: 正交解耦绑定/节点 Launch 成本 & SRAM RMW 16 MiB 容量边界
│   ├── g0_5_collective_handoff_probe.cpp             # G0-5/G0-1: 8 卡 FP32 AllGather vs AllReduce & 同边界普通/预排队对比
│   ├── g0_7_distinct144_prequeue_probe.cpp           # G0-7/G0-1: 144 独立 Recipe 预排队与 MappedMemMgr 守恒定律发现
│   └── rt_1_2_3_runtime_gaps_probe.cpp               # RT-1/RT-2/RT-3: 8 卡真机运行时缺口闭环验证探针
├── scripts/
│   ├── g0_3_4_host_and_path_audit.py                 # G0-3/G0-4: async_scheduling 三处门禁与 uclamp=1024 主机阶段审计
│   └── g0_6_recipe_roofline_audit.py                 # G0-6: 70L 144 Recipe 算子级 Roofline 审计（移交 gaudi-kernels 的证据）
└── evidence/
    ├── README.md                                     # 各证据子目录来源、实验条件与关键文件索引
    ├── review-20261023/                              # 第一轮真机复核
    ├── runtime-audit-20261023/                       # 第二轮历史预排队记录与库哈希核对
    ├── design-review-20261023/                       # libSynapse.so (1.24.2-481) 静态反汇编 (.asm) 与导出符号表
    ├── review2-20261023/                             # 第三轮 Module 7 纯 C++ 在途上限实测与两轮评审报告
    └── gate0-20261023/                               # Gate 0 (G0-1 ~ G0-7) 及 RT-1/RT-2/RT-3 完整 JSON 证据与 SUMMARY.md
```
