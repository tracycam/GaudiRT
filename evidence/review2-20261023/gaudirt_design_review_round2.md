# GaudiRT 设计第二轮审查（更新后的 software-autopsy.md / DESIGN.md）

> 审查时间 2026-10-23 03:17。逐条核对了 `evidence/review2-20261023/`、gdb 原始日志、`libSynapse.so 1.24.2-481` 导出符号、HCL 源码，以及模型 `config.json`。

## 0. 裁决

**方向同意，文档暂不冻结。修完下列 B 级证据完整性问题后，即可冻结并进入 Gate 0。**

这一版的骨架是对的：
- 现有资产映射与防膨胀红线
- 按 11.31 ms 时间线的瓶颈桶来划分攻关方向
- Gate 0 为纯实验，并带 Kill 阈值
- 驻留实例降级为条件触发
- 删除重复层，`GaudiRT` 只保留探针

但 autopsy §三.2 里出现了**证据中不存在的数据**。在一份标榜“只信实测”的文档里，这类问题会直接让结论不可信，必须先修。

---

## 1. B 级：证据完整性（阻塞冻结）

| # | 文档写法 | 原始证据（[plain.log](file:///home/rate/llm/Gaudi-LLM/GaudiRT/evidence/review2-20261023/remote-run/plain.log)、[gdb.log](file:///home/rate/llm/Gaudi-LLM/GaudiRT/evidence/review2-20261023/remote-run/gdb.log)） | 处置 |
|---|---|---|---|
| B1 | 绑定数 “3, 16, 32, 64, 96”，对应耗时 “6.75 / 9.09 / 12.63 / 18.17 / 24.65 µs”；“TPC 节点数 1, 8, 32” | 只跑过 3 组：**3 绑定/1 节点**（6.62–7.02 µs）、**24 绑定/8 节点**（10.82–11.26 µs）、**96 绑定/32 节点**（24.10–25.36 µs）。16/32/64 这三个点和对应数值**在任何证据里都找不到** | 删除这些点，或者补跑并归档 |
| B2 | “前 572 次快速完成，第 573 次（0-based idx=572）阻塞” | `first_block_index=573`（0-based），也就是**前 573 次成功，第 574 次阻塞**；两个 recipe 交替时为 571 | 更正差一错误，并补上交替场景的 571 |
| B3 | “阻塞约 105–107 ms” | 107.6–118.4 ms | 更正 |
| B4 | gdb 栈帧地址 `0x1724962 / 0x1723015 / 0x171f1db / 0x171deec / 0x17226d3` | 用库基址 `0x7ffff4000000` 换算后，实际返回地址是：`launch 0x1724cbb`、`prepareForDownload 0x1723264`、`getMappedMemoryInfo 0x171f1d8`、`waitForCompletionCopy 0x16d8ecf`、`checkCompletionCopy 0x171dab1`。**五个地址一个都对不上**，而且 `waitForCompletionCopy` 根本不在 0x171xxxx 这一段 | 照 gdb.log 改写，或者只写符号名、不写地址 |
| B5 | 暗礁二称 scalEnqueue 末尾调用 “`HbmGlblMemMgr::setLongSo` 与 `ArcHbmMemMgr::setLongSo`” | 这两个符号**不存在**。导出表里只有 `NBuffQueueAllocator::setLongSo` 和 `NBuffDeviceAllocator::setLongSo`；0x17247c8/0x17247e1 处是 **vtable+0x10 的间接调用** | 降级为“推断：经 vtable 按 long-SO 代际标记回收” |
| B6 | “根因是 `DataChunk` 槽位”“既不是 CCB 也不是 HBM 池字节耗尽” | 实测只证明了 **MappedMemMgr → waitForCompletionCopy 这条路径是该负载下最先触发的上限**。DataChunk 粒度没有验证；其它池只能说“不是第一个耗尽的” | 措辞收窄 |

> [!NOTE]
> **我对首轮报告的自我更正**：首轮报告里的成本公式 “6.2 µs + 0.19 µs/绑定” 存在**混杂**。探针里绑定数始终等于 3 × 节点数，0.19 µs 实际上是“每个绑定连同其所属节点”的合并斜率，无法拆成绑定和节点两部分。而且这个线性拟合只有 3 个点。若要保留此公式，G0 需要补一组“节点数固定、只变绑定数”的探针。

---

## 2. 归因错误与内部矛盾（冻结前修正）

### 2.1 `EventNetwork` 归错了层（影响 G0-5 和 Gate 2 第 2 项）

- **事实**：HCL 每个集合通信 API（`hccl.cpp` 第 567/665/711/…/969 行）进入后，先调用 `syncHCLStreamHandle → synStreamSyncHCLStreamHandle`，再由 `libSynapse.so` 里的 `DeviceCommon::syncHCLStreamHandle` 构造 `EventNetwork` 并 `StreamsContainer::addJob`。也就是说，**每次 collective 的 EventNetwork 是 HCL 调 Synapse 时生成的，Bridge 并没有插入它**。
- Bridge 自己的跨流事件只出现在每个 region 的入口和出口（`record_and_wait_for_event`），每步 2 个 region，共约 4 次，不是 141 次。
- **后果**：“直接在 `NativeExecutionPlan.cpp` 精简 EventNetwork”打不到那 141 次交接。G0-5 的分支 ① 应改为测量“HCL 在 generic 流上提交”与“HCL 原生流 + 显式 long-SO 等待”的差异，并承认若要去掉它，需要换一种 HCL/Synapse 的用法（可能涉及非公开路径）。

### 2.2 时间桶账目

- **Bucket D 的 0.51 ms 重复计数**：它已经包含在 Bucket B 的 2.70 ms 里（1.900 + 0.507 + 0.295）。
- **桶不能相加**：源文档写明“各项分别取中位数，不能要求精确相加，也不是互斥预算”（[CPU-DEPENDENCY-AND-OVERLAP:L30-L32](file:///home/rate/llm/Gaudi-LLM/gaudi-kernels-public/docs/CPU-DEPENDENCY-AND-OVERLAP-20261011.md#L30-L32)）。所谓“合计 5.5–6.0 ms 可优化空间”没有推导过程，应删除。
- **测量口径混用**：时间线表里的 SDK/CPU trace 来自**另一次开着 profiler 的运行**，不是 11.31 ms 那次无插桩运行。文档开头又声明“严格基于真机无插桩实测”。表格必须标注“profiler trace、另一次运行、中位数不可相加”。
- **TPS 区间也混了两次运行**：“86.91–88.42 TPS”中，86.91 来自 TRACE 文档的那次运行，88.42 来自 `production70`；`production70` 的链接还指向了 command-count-audit，而不是包含 host policy 的原始 json。
- **“完全来自 CPU 唤醒/频率爬升”说过头了**：源文档的措辞是“主要可复现因素”。uclamp 同时影响调度放置和频率，两者并未分离。

### 2.3 “权重带宽下限 ~1.7 ms”没有出处（P1 的立论依据）

我用模型 `config.json` 做了一个估算。这只是估算，前提是：TP8、无专家并行、batch=1、4K 上下文。

| 项 | 每 rank 每 token 读取量 |
|---|---:|
| QKV 投影（FP8，q 128×192、k 8×192、v 8×128） | 70 × 20.8 MB ≈ 1.46 GB |
| **o_proj（BF16，`ignored_layers` 共 71 项未量化）** | **70 × 25.2 MB ≈ 1.76 GB（约 34%）** |
| MoE（69 层，top-8，`store_dtype=mxfp4`，约 4.25 bit/参数） | 69 × 20 MB ≈ 1.38 GB |
| 第 0 层 dense MLP、router、lm_head（1/8）、KV 读取 | ≈ 0.6 GB |
| **合计** | **≈ 5.2 GB → 峰值带宽下约 2.1 ms，按 85% 可达带宽约 2.5 ms** |

由此推出两点：
- 计算 union 5.73 ms 相对下限是**约 2.3–2.7 倍**，不是 3.4 倍。
- 第一性原理上的提醒：Gaudi2 的 MME 不支持 FP4，MXFP4 专家无论走 TPC GEMV 还是“TPC 解码 → SRAM → MME”，**解码都要在 TPC 上完成**。按 2.45 TB/s 喂满 MXFP4，每秒约 4.9 T 个参数，每个参数要一次 FMA 加若干解码操作，所需 TPC 吞吐很可能超过 24 个 TPC 的能力。换成 MME + SRAM 只省掉了 FMA，省不掉解码。
- 另外，**BF16 的 o_proj 是单项最大的读取量**。这是数值契约层面的决策，也是一个潜在的大杠杆。

所以 Gate 1 第 1 项“把纯 TPC 路径重构为 MME + RMW SRAM”必须改写成**以 G0-6 的逐 recipe 字节数/耗时为前提的假说**，不能预设结论。另外它和 `gaudi-kernels-public` 已有的小 M 选型、compact v2/v3 被速度门拒绝的负结果直接冲突，需要先说明为什么这次会不一样。

### 2.4 DESIGN 门禁的内部矛盾

| 位置 | 矛盾 | 建议 |
|---|---|---|
| G0-1 | 只定义了 Δ < 0.30（砍）和 Δ ≥ 0.50（继续），**0.30–0.50 之间没有规定**。“首算子开始到末算子结束”需要设备时间戳，开 profiler 会违反 G0-9 | 补上灰区规则；用 `synEventElapsedTime`（带计时标志的事件，打在 gate 结束和步末）代替 profiler；分别和设备跨度 10.48 ms、墙钟 11.31 ms 两个口径比较 |
| G0-2 vs autopsy 暗礁三 | autopsy 已断定外部驻留必须手写 ARC 包，而这正是 G0-2 的 Kill 条件。要测 G0-2，就得先写这个被禁止的 hack，或者它注定失败 | 二选一：现在就砍掉 L1_B；或者先写明一条**不需要手写 ARC 包**的候选机制，再谈测试 |
| G0-3 | “最快合法发射路径”没有定义。现成的 bypass 开关是通过 LD_PRELOAD 插桩库实现的，本身带开销 | 明确 oracle 的实现方式，并说明它的插桩开销如何扣除 |
| G0-4 vs Gate 2 第 1 项 | G0-4 规定只有 ≥ 0.15 ms/step 的子项才立项，但 Gate 2 第 1 项针对的 preflight 实测只有 0.098–0.127 ms，**按自己的门槛不该做** | 删除，或改为“视 G0-4 结果而定” |
| G0-7 / Gate 2 第 3 项 | 设备端多步反馈需要设备侧采样和 EOS 判定，还要改 vLLM 调度器；现有 2 卡/8 卡原型（BOUNDED-DEVICE-FEEDBACK / BOUNDED-CAPACITY）没有被引用。**零代码的 vLLM `async_scheduling` 对照被删掉了**，而它正是检验“首算子前 0.92 ms 能否靠主机重叠消除”的最便宜实验 | 恢复 async_scheduling 的 ABBA（注明它与投机解码不能同时开）；引用已有的设备反馈原型 |
| Gate 2 冻结标准 | 三个动作合在一起只要求 “< 11.11 ms”，无法分清哪个动作贡献了多少 | 每个动作单独做 ABBA |

---

## 3. 首轮意见中仍未吸收的项（建议修正，不阻塞）

- **硬件表**：TPC 的 FP32/BF16 口径、910B 缺出处；RoCE 每对 rank 只有 3×100GbE = 37.5 GB/s，且**缺小消息时延这一维**。Bucket B 的 1.90 ms 恰恰取决于这个缺失的维度。
- **SRAM 探针缺 RMW=false 对照**：“稳定分配在 SRAM”只在一个 2 KB 张量上成立。还应读出 `GCFG_RMW_SECTION_MAX_SIZE_BYTES` / `GCFG_SCRATCHPAD_SRAM_MAX_SIZE` 的默认值（`gaudi-kernels-public` 硬限 16 MiB）。
- **HCL 版本**：引用的 HCL 源码是公开版 1.21.0-555，装的运行时是 1.24.x，需要注明。
- **目录**：`evidence/design-review-20261023/` 和 `review2-20261023/` 的作用不清，需要在 README 或 manifest 里说明。我首轮误建的 `evidence/gate0-baseline/` 和 review2 内容逐字节相同，已删除；按 DESIGN 约定，下一步产物统一放到 `gate0-20261023/`。

---

## 4. 最小修改清单（完成后即可冻结并进入 Gate 0）

1. 按 §1 修正 B1–B6。
2. 修正 §2.1 的 EventNetwork 归属，并相应改写 G0-5 和 Gate 2 第 2 项。
3. 时间线表标注口径；删除 Bucket D 的重复计数和“5.5–6.0 ms 可优化空间”；用带公式的 roofline 估算替换 “1.7 ms”，或改为“待 G0-6 计算”。
4. 处理 §2.4 表中的五处矛盾：G0-1 灰区与计时方法、G0-2 的去留、G0-3 的 oracle 定义、G0-4 与 Gate 2 第 1 项、恢复 async_scheduling。
5. 把 Gate 1 第 1 项改写成“以 G0-6 为前提”的假说，并写明 MXFP4 的 TPC 解码约束。
