# Gate 0 与运行时缺口验证报告（`gate0-20261023`）

- **实验环境**：`192.168.100.153`（8 × Intel Gaudi2，`hl-smi` 8 卡空闲，SDK `1.24.2-481` / `79e284b`）
- **控制条件**：所有真机探针统一锁死 `uclamp_min = 1024`（[`host_policy.hpp`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/include/gaudirt/common/host_policy.hpp)），禁用一切 Profiler / `LD_PRELOAD` 插桩，采用 ABBA 交错采样与相同起止事件边界（`[first_op_start, last_op_end]`）。

---

## 一、模块职责边界重申（撤下跨界项，聚焦运行时核心）

根据架构职责划分：
1. **`gaudi-kernels`**：负责 TPC/MME 内核、算子融合、权重量化（如 `o_proj` FP8）、1-core TPC 算子多核化与 SRAM 分块。原本列为 P1（算子融合）与 P2（`o_proj` 量化）的工作**已从 `GaudiRT` 主线中正式撤下**，`GaudiRT` 将 [`g0-6-recipe-roofline-audit.json`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/evidence/gate0-20261023/g0-6-recipe-roofline-audit.json) 作为定位证据移交 `gaudi-kernels`。
2. **`Bridge / vLLM runner`**：负责框架接入、模型构图、输入准备、采样和请求调度。`GaudiRT` 将 [`g0-3-4-host-and-path-audit.json`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/evidence/gate0-20261023/g0-3-4-host-and-path-audit.json) 中定位的 3 处 `use_async_scheduling` 快路径退出门禁移交上层。
3. **`GaudiRT`**：严格聚焦**面向编译产物的通用运行时执行能力**——实例化与重放、计算和通信统一提交、在途资源管理、完成回收与背压。

---

## 二、`GaudiRT` 三大已验证运行时缺口（`RT-1 / RT-2 / RT-3`）与真机实测收益

产物文件：
- 探针源码：[`probes/rt_1_2_3_runtime_gaps_probe.cpp`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/probes/rt_1_2_3_runtime_gaps_probe.cpp)
- 运行时核心头文件：
  - [`include/gaudirt/runtime/unified_region_lowerer.hpp`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/include/gaudirt/runtime/unified_region_lowerer.hpp)
  - [`include/gaudirt/runtime/generation_epoch_tracker.hpp`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/include/gaudirt/runtime/generation_epoch_tracker.hpp)
  - [`include/gaudirt/runtime/inflight_credit_manager.hpp`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/include/gaudirt/runtime/inflight_credit_manager.hpp)
  - [`include/gaudirt/runtime/compiled_region_executor.hpp`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/include/gaudirt/runtime/compiled_region_executor.hpp)
- 单元测试：[`tests/runtime_unit_tests.cpp`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/tests/runtime_unit_tests.cpp)
- 8 卡实测数据：[`evidence/gate0-20261023/rt-1-2-3-runtime-gaps.json`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/evidence/gate0-20261023/rt-1-2-3-runtime-gaps.json)

### 1. `RT-1`：混合命令单流统一降维（消除 `Command 5` 跨流断层与 `~0.77 ms` 早期设备气泡）
- **缺口成因**：70 层生产模型单步 `286` 个命令中，`command 5`（第 1 层 Attention `o_proj` + `AllGather`）是唯一同时含 `compute: true` 与 `collectives: 1` 的混合命令。现有 `NativeDependencyPlan` 与 `PreparedRegionProgram::compile` 将其视为屏障，把单步切分为 `Region 0 [0..4]`、`Command 5`（旧路径回退，通信被迫切到 `comm_stream`）和 `Region 1 [6..285]`，在步首设备最饥饿阶段插入 `leave()`、`publish(Region 0)`、跨流等待与二次 `2,392` 地址扫描，造成 8 卡一致的 `644 ~ 806 µs` 早期设备气泡。
- **`GaudiRT` 机制（[`UnifiedRegionLowerer`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/include/gaudirt/runtime/unified_region_lowerer.hpp)）**：校验混合命令内部 `compute_writes` 到 `collective_reads` 的 RAW 闭合性后，将其正交拆解为 `Compute` + `Collective` 两个原子子调用，使 `286` 个命令（`287` 个子调用：`145` 计算 + `142` 通信）合并为 **`1` 个统一 Region（`[0..285]`，`coalesced_collectives: 141/142 -> 142/142`）**。
- **8 卡 ABBA 实测结果**：
  - 早期窗口（`Commands 0..7`）设备执行跨度：8 卡从 **`751.23 ~ 754.63 µs`** 降至 **`159.40 ~ 161.23 µs`**，每卡稳定消除 **`591.83 ~ 594.31 µs`（`~0.59 ms`）** 的早期跨流与主机断层气泡；
  - 整步（`286` 命令）设备执行跨度：8 卡净省 **`600.84 ~ 1,076.45 µs/步`**。

### 2. `RT-2`：`O(1)` 代数守卫绑定预检与单 Epoch 围栏（消除每步 `2,392` 绑定/地址扫描）
- **缺口成因**：现有运行时在每步入口遍历 `2,392` 个 `RetainedTensorBinding::matches`，并在每个 Region 进出时对 `2,392` 个依赖地址执行 `events_by_addr_` 哈希表查找（`get_wait_events`）与生产者覆盖/线性数组移除（`add_producer` $\to$ `remove_device_ptr`），导致每步 `enq0` 前主机侧迟滞 `~0.95 ms`。
- **`GaudiRT` 机制（[`GenerationEpochTracker`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/include/gaudirt/runtime/generation_epoch_tracker.hpp)）**：当静态图绑定代数（`binding_generation_`）与外部流写入代数（`external_stream_epoch_`）未变时，将绑定检查、同流重放入口等待和出口事件发布全部收敛为 **`O(1)` 代数核验 + 单个步尾 Epoch 事件记录**。
- **真机 ABBA 实测结果**：
  - 2,392 绑定预检 + 2-Region 事件表维护耗时从 **`990.44 µs`** 降至 **`1.07 µs`**，单步净省 **`989.37 µs`（`923.9x` 加速）**；
  - 8 卡整步主机提交耗时从 `3,833.85 ~ 5,475.77 µs` 降至 `3,091.17 ~ 4,383.02 µs`，每步净省 **`734.36 ~ 1,092.75 µs`**。

### 3. `RT-3`：Recipe 复用拓扑下的在途 Chunk 信用管理与确定性背压
- **缺口成因**：`libSynapse.so` 的 `MappedMemMgr` 遵循精确守恒律：
  $$\text{first\_block\_idx}(N_{\text{distinct}}) = 575 - 2\,N_{\text{distinct}}$$
  虽然 70 层生产模型每步发射 `145` 次计算，但轨迹审计证实每步仅使用 **`12` 个不同 `synRecipeHandle`**（MoE 与 Attention 跨层复用同一编译产物）。因此真实模型的无阻塞在途上限并非 144 独立 Recipe 假设下的 `287` 次，而是 $575 - 2\times 12 = \mathbf{551}$ 次（**`3.80` 个完整步**）。但当连续预排步数达到 `K = 4`（`580` 次 > `551`）且缺乏信用管理时，仍会在第 `552` 次调用（`idx = 551`）处撞上 `waitForCompletionCopy` 硬阻塞。
- **`GaudiRT` 机制（[`InFlightCreditManager`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/include/gaudirt/runtime/inflight_credit_manager.hpp)）**：实时追踪在途不同 `synRecipeHandle` 引用计数（$2\,N_{\text{distinct}}$）与在途 Launch 数（$N_{\text{in\_flight}}$），在每步入口通过 `synEventQuery` 非阻塞回收已完成步的 `145` 个信用，并在信用不足时于**步边界**执行确定性同步等待。
- **真机实测结果**：
  - 12-Recipe 复用拓扑（`145` 计算/步）无节流测试：理论预测 `first_block_idx = 551`，真机实测 **`551`（零误差，`3.80` 步）**，验证 `K = 1, 2, 3` 步（`145, 290, 435` 次计算）在不触发任何阻塞的情况下可一次排入；
  - 开启 `InFlightCreditManager` 后测试 `K = 1, 2, 3, 4` 步（最高 `580` 次 Launch）：全程 **`mid_step_waitForCompletionCopy_blocks = 0`**，峰值占用 Chunk 稳定受控在 `314 <= 559`，单次 Launch 最大耗时仅 **`39.59 µs`**。

---

## 三、前期 Gate 0 基础探针结论归档（`G0-1` ~ `G0-7`）

| 门禁编号 | 实验名称与产物文件 | 实测核心指标 | 归属模块与结论 |
| :--- | :--- | :--- | :--- |
| **`G0-1` & `G0-7`** | **在途 `MappedMemMgr` 守恒定律发现**<br/>[`g0-7-distinct144-prequeue.json`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/evidence/gate0-20261023/g0-7-distinct144-prequeue.json) | 5 组独立 Recipe 数测试点零误差验证 $\text{first\_block\_idx}(N_{\text{distinct}}) = 575 - 2\,N_{\text{distinct}}$（`573, 503, 431, 295, 287`）。 | **归属 `GaudiRT`（`RT-3`）**：直接转化为 [`InFlightCreditManager`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/include/gaudirt/runtime/inflight_credit_manager.hpp) 的信用预算定律。 |
| **`G0-2`** | **Launch 成本正交解耦与 SRAM `RMW` 容量上限**<br/>[`g0-2-launch-sram.json`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/evidence/gate0-20261023/g0-2-launch-sram.json) | • $\text{Launch}(\mu\text{s}) \approx 6.20 + 0.227\,N_{\text{bind}} + 0.019\,N_{\text{node}}$；<br/>• SRAM `RMW=true` 容量上限为 **`16 MiB`**（`20 MiB` 返回 `synFail`）。 | **基础物理参数**：支撑 `GaudiRT` Launch 开销建模，SRAM 边界移交 `gaudi-kernels`。 |
| **`G0-3` & `G0-4`** | **`async_scheduling` 路径门禁与 `uclamp=1024` 主机剖析**<br/>[`g0-3-4-host-and-path-audit.json`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/evidence/gate0-20261023/g0-3-4-host-and-path-audit.json) | 定位 `decode_graph.py:77`、`runner.py:424`、`runner.py:666` 三处因 `use_async_scheduling` 退出 `StepGraphCapture` 的门禁。 | **移交 `Bridge / vLLM runner`**：后续异步或设备反馈必须在保持 `StepGraphCapture` 快路径下开启。 |
| **`G0-5`** | **8 卡集合通信交错跨度与 `FP32 AllGather` 对比**<br/>[`g0-5-collective-handoff.json`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/evidence/gate0-20261023/g0-5-collective-handoff.json) | `FP32 hcclAllGather` 单次 `10.32 µs`（比 `BF16 AllReduce` 快 45%）；连续提交时硬件 `EventNetwork` 交接边际开销仅 `0.18 µs/次`。 | **归属 `GaudiRT`（`RT-1`）**：证明只要消除 `Command 5` 跨流断层、保持同流连续提交，HCL 交接开销极小。 |
| **`G0-6`** | **70 层 144 Recipe 算子级 Roofline 与子图内空泡审计**<br/>[`g0-6-recipe-roofline-audit.json`](file:///home/rate/llm/Gaudi-LLM/GaudiRT/evidence/gate0-20261023/g0-6-recipe-roofline-audit.json) | 定位 MoE TPC 算子超额（`+1.00 ms`）、`o_proj` BF16 带宽墙（`913 µs`，达 Roofline 92.6%）及 1-core TPC 算子导致的子图内切换空泡。 | **移交 `gaudi-kernels`**：作为算子库内核优化与量化改造的定量依据，脱离 `GaudiRT` 主线。 |
