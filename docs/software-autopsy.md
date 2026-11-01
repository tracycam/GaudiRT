# 软件与硬件账单：保留事实

本文只保留后续决策需要的历史事实。原始报告、反汇编、日志、探针源码和负结果已迁移到 `../instance-assets/gaudirt-archive-20261025/GaudiRT/`，校验清单位于 `../instance-assets/gaudirt-archive-20261025/manifest/`。本文不再把历史探针结果表述为生产能力。

## 真实模型基线

- 历史 70 层 TP8 Decode 在高性能主机策略下约为 11.31 ms/token；另一组 CPU 策略对照约为 13.37 ms/token。主机策略影响必须固定，不能把它误算成新代码收益。
- 独立 Profiler 采样曾观察到 5.728 ms TPC/MME 活跃并集、2.056 ms Recipe 内无计算区间、2.702 ms 相邻 Recipe 间无计算区间。它们来自带采样的运行，区间互相包含或重叠，不能求和形成收益承诺。
- 同一采样中的主机 `NativePlanSubmit` 与设备执行高度重叠。主机 API 时间减少不等于端到端 TPOT 同量减少。
- 真实路径审查记录的绑定预检约为 98–127 us。RT-2 使用模拟器得到的 923.9x 不是该真实路径的加速比，已撤回。

## 编译与存储边界

- 复杂算子继续复用官方 `synGraphCompile`，不重写闭源 MME 切分、Recipe 生成和量化编译逻辑。
- Gaudi2 有 48 MiB SRAM；既有实验中单个 RMW Section 的 16 MiB 限制不能外推成“所有真实中间结果都能留片上”。真实图的占用和分块必须由编译结果验证。
- MXFP4 权重需要 TPC 解码，不能假设改成 MME 或 SRAM 路径必然加速。
- BF16 `o_proj` 是历史权重读取账单中的重要部分，但 FP8 化必须独立验证校准、数值误差和最终模型输出。

## 发射、通信与信用

- `synLaunchWithExternalEventsExt` 的成本随 Recipe 和绑定情况变化；玩具 Recipe 的单次耗时不能替代真实 Recipe。
- 合成实验支持 `2 * N_distinct + N_in_flight <= 575` 的 MappedMemMgr 信用关系。12 个不同 Recipe 时得到 551 的上限也属于该实验拓扑，不能未经验证套到所有真实模型。
- 每次 HCCL 调用内部的同步属于 HCL/Synapse 路径，不是只改上层 Bridge 就能消除的事件。
- 换发射器不会改变 Recipe 内部的计算，也不会减少集合通信的物理传输次数。裸 SCAL 微基准结果不能直接外推到真实模型 TPOT。

## 已完成的底层协议实验

- P1 证明了特定单 Recipe 的自管 HBM 重放和输出一致性。
- P2a 证明了特定 8 卡 FP32 AllGather 的自主网络重放。
- P3a 证明了 `add -> AllGather -> add` 的闭环与双 CCB 回绕；32,638 步实验覆盖 long-SO chunk1 进位，全部八 Rank 输出、guard 和 hash 链通过。
- 这些结果仅保留为底层协议知识。当前主线不为旧碎 Recipe 建设通用裸加载器，也不把这些结果称为完整 Graph 或生产运行时。

## 后续裁决方式

下一阶段必须直接实现真实模型一层，不再新增玩具探针。大 Recipe、SRAM、FP8 和静态执行计划分别测量，基准始终是当前真实路径。只有真实输出满足事先约定的精度标准且设备时间或 TPOT 净下降，才继续扩大范围。
