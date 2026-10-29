# GaudiRT 验收基线与后续路线

## 已验收基线

以下结论只覆盖列出的真实切片，不外推为整模或 TPOT 收益。

- Type A（layers.0，纯 RMSNorm、Full-Attention、Dense MLP）：八卡零旧桥 C++ 闭环通过。`run-layer0-v14` 的最终层输出最大行相对误差为 `1.178e-5`；`run-dense-serial-v2` 的最终层输出八卡逐位一致。
- Type C（layers.1，SWA+MXFP4 MoE）：`run-swa-moe-v3` 八卡逐位一致。
- Type B（layers.7，Full-Attention+MXFP4 MoE）：`run-type-b-v4` 八卡闭环通过，最终层输出最大行相对误差为 `3.644e-4`。
- 整数 `tensorId` 发射：各类 Recipe 均在初始化期建立 launch 数组，运行期只更新地址，不再按名字查找。
- 最小 PrivateUse1：`private-use1-v11` 的八进程普通 H2D/D2H 拷贝、arena 分配、整层 `torch.compile(fullgraph=True)`、输出和 KV 验收通过。
- 统一 Phase 2 薄桥：`phase2-v16` 在每 Rank 一个零旧桥进程内验收 Type A/B/C，11 个动态张量全部显式入参，FX 每层只有一个 `gaudirt.layer` 节点，三类 KV 均逐位一致。Type A/B 最终输出最大行相对误差分别为 `1.178e-5`、`3.644e-4`，Type C 输出逐位一致。权重在初始化期注册，运行期只做地址绑定，不同层型共享独立 Recipe。
- N-lane Dense 变体已否决：正确但整层 p50 比真实 W8A16 MME 对照慢 29.4%，活跃实现已回退，证据保存在 `layer0-abba-v3`。

所有性能结论只适用于相应切片。历史 `11.31 ms/token` 生产基线与切片耗时不得直接拼接或外推。

## 尚未闭环

- 合成动态步的输出曾达到数值门，但 KV 未逐位通过，该路线不作为合格证据，也不再继续投入。
- Type C 的同 manifest 注册保持了 Recipe 身份，但这只是复用机制检查，不代表 layers.2 的生产数值已验收。
- `.recipe` 磁盘缓存、真实多层连续步进、70 层整模、真实服务负载和端到端 TPOT ABBA 尚未完成。
- ARC 固件增强仍须先证明公开 SDK 的具体瓶颈，经过同源 x86 仿真与独立恢复验证后才可上真机。

## Phase 2 后续

1. 实现 `.recipe` 冷编译落盘与暖进程反序列化，按图源码、内核与几何指纹隔离缓存。
2. 用真实第二层权重验证同构 Recipe 的换权重执行，不能只做同 manifest 机制检查。
3. 接入真实多层、连续步进，再验收 70 层整模与端到端净收益。

## Phase 3 与 Phase 4 边界

- Phase 3：完成 70 层整模和真实负载 ABBA，先证明正确性和端到端净收益，再扩展前端通用性。
- Phase 4：只对已测明的设备端动态控制瓶颈进行 ARC 固件改造，保持公开 Synapse/HCCL 路径为默认实现，不以固件工作掩盖前置缺口。
