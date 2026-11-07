# ROADMAP.md — 全栈工程设计与阶段路线图

本文是 `Gaudi-LLM` 项目的唯一工程设计与路线总纲，工程纪律、版本卫生与文档规范服从同目录 [`AGENTS.md`](AGENTS.md)，实测硬件、编译器与运行时踩坑总台账统一维护于 [`docs/gaudi2-pitfalls.md`](docs/gaudi2-pitfalls.md)。子仓或历史文档若与本文冲突，一律以本文为准。

---

## 一、总体目标与全栈四层分工

抛弃 21 万行 `gaudi-pytorch-bridge` 的 LazyTensor / JIT 动态追踪遗产，**同时拒绝“每换一个模型手写几千行 C++ Synapse 图”的零复利作坊路线**，围绕现代 PyTorch 2 分段静态编译与核心融合原语构建干净、显式、快速失败（Fail-Fast）的通用自研推理栈，直接承接 **SGLang 全模型生态（200+ `models/*.py`）** 与 **DeepInfer（`DeepSeek-V4.1-Flash`）**。前期 `models/mimo/` 仅作为验证 `GaudiRT` 底层机制的探路靶子（已收口封板，不再扩写单模专属构图）。全栈严格拆为四层，禁止跨层混写或把具体模型常量塞进通用底座：

1. **算子原语层（`gaudi-kernels-public/`）**
   剥离单模型硬编码维度与窄契约（如零 bias 限制），坚持**“在硬件物理能力内压榨最高效率”的软硬协同原则**——凡硬件具备原生满速通路的（BF16 MME `432 TFLOPS`、整张量安全缩放 IEEE-FP8 MME `865 TFLOPS`）直接跑满原生硬件；凡硬件无原生细粒度块缩放或无 FP4 MME 的（`MXFP4`、`UE8M0` `MXFP8`、`128×128` Block-FP8，对标 NVIDIA 上 Marlin / SGLang INT4 在线反量化喂 Tensor Core 路线），将已验证高效软核通用化（小 $M$ 走 24 核 TPC 寄存器流式解包乘加吃满 `2.45 TB/s` HBM 带宽；大 $M$ 走 TPC 在 48 MB SRAM 内按 Tile 流式反量化/重缩放后双缓冲喂满 MME，严禁 HBM 全量展开 FP32 慢路径）。全库收敛出 **5 大参数化标准原语家族**，合并打包进单一 provider `.so`（遵守 Synapse `<=15` 库硬上限 `P-RT-14`）并通过 `GC_KERNEL_PATH` 按需覆盖；已达标的官方算子直接由编译层调用，不盲目重造：
   - **Attention 与混合线性状态原语**：通用参数化 Paged GQA / MHA / SWA（Prefill varlen + Decode，含前置 QK-Norm 等变体）、新一代混合线性注意力（如 `Qwen 3.8 Next` 的 `Gated DeltaNet` 线性循环核与状态更新），以及面向 `DeepSeek-V4.1-Flash` 的 Paged / Sparse FlashMLA + `DeepSelect` 动态索引器；
   - **通用量化 Linear / DeepGEMM 软硬协同原语**：统一覆盖 BF16、Block-FP8、`MXFP4` 及 `UE8M0` `MXFP8×MXFP8` / `MXFP8×MXFP4`（解除零 bias 限制，内部封装 OCP `float8_e4m3fn` 到 Gaudi2 硬件 IEEE-Specials FP8 的安全幅值/指数转换 `P-NUM-01` 与按 $M$ 自动分派的 TPC 直算 / SRAM-MME 流水路径），对上层模型完全透明；
   - **Fused MoE 原语**：参数化 `num_experts / top_k` 与评分函数（softmax / sigmoid + bias / sqrtsoftplus）的 Router + 按专家分桶 Permute + Grouped MME/TPC GEMM + 安全指数 SwiGLU（`tpc_math_safe.h`）；
   - **Fused Norm / RoPE / Sampler 原语**：FP32 补偿归约 `AddRMSNorm` / `LayerNorm`、多变体 `RotaryEmbedding`、设备端 `Top-K / Top-P / Multinomial` 采样与结构化解码位掩码；
   - **专项结构原语**：如 `DeepSeek-V4.1-Flash` 的 4 流残差 `mHC`（`post` / `sinkhorn` 迭代归一化）。
2. **运行时底座（`GaudiRT/`）**
   只保留与具体模型无关的通用 C++20 机制：设备/流/HCCL 生命周期（`DeviceRuntime`）、带编译器与算子库二进制 SHA256 指纹的 `.recipe` 注册序列化与零同步静态绑定表（`RecipeRegistry` / `PreparedLaunch`）、在途信用管理（`InFlightCreditManager`）、权重加载辅助，以及仅供后续底层旁路与固件协同使用的队列协议（`scal/`）。面向通用生态在底座补齐三项核心能力：
   - **Stream-ordered 块式缓存显存分配器**：支持单张量 `free` 合并回收、跨流事件安全复用及静态桶执行专用池（`GraphPool`），取代仅支持线性累加的 bump `HbmArena`；
   - **零拷贝视图与异步 DMA 引擎**：支持固定主机内存池（规避重复 `synHostMap` 崩溃 `P-RT-15`）及 `non_blocking` H2D/D2H 流同步，统一承载异步调度与常驻百 GB 级主机 DRAM 词表（如 `203 GB` `Engram`）的稀疏 Gather 后锁页小包异步 H2D；
   - **多通信域集合通信底座**：解除单机 `TP8` 硬编码，按实际负载演进支持单机 Scale-Up `TP / DP / EP` 子通信组及 `AllReduce`、`AllGather`、`ReduceScatter`、`AllToAll`（跨机 Scale-Out 在多机 RoCE 环境具备后独立扩展）。
3. **通用编译桥接层（`GaudiRT` `PrivateUse1` + C++ 底层全透明静态桶 `FX -> Synapse` 编译器）**
   - **热路径（C++ 底层全透明 13 档静态桶编译重放，对 Python 零侵入）**：规避 `DATA_TENSOR_DYNAMIC` 全跨度动态图带来的 48 MB SRAM 最坏容量排程退化与每次发射的主机 SMF 动态 Patch 开销，将分桶逻辑彻底下沉封死在 C++ 桥接与执行层内部（`BucketDispatcher`），使上层 Python 模型与调度器像使用 CUDA 一样直接传入任意动态 token 数 `M`：
     1. **单次符号捕图 + `O(1)` 选桶**：Python 侧 `torch.compile(..., dynamic=True)` 仅做 1 次 FX 捕图；C++ 运行期按 `std::bit_ceil(M)` 映射至 `{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}` 共 13 个静态桶 `.recipe`（编译期自动为所有 `synTensor` / `synNode` 注入 `(subgraph_hash, bucket_M, op_idx)` 唯一前缀根除 `P-RT-12/22/23/31` 同名串形坑，且常驻 Recipe 总数远低于 `P-RT-05` `575` 槽位上限）；
     2. **物理 `Storage` 留桶余量、逻辑 `Tensor` 保精确 `[M, ...]`**：权重与 KV 池保持精确字节分配零浪费；瞬态激活输出底层 `c10::Storage` 按 `M_bucket * row_bytes` 分配而对外保留精确 `[M, H]` 形状与 `numel()`（使 `hccl` 集合通信天然只传真实的 `M` 行数据），下游子图输入经 `storage().nbytes()` 校验实现 99% 零拷贝直绑，偶发外部窄视图自动走流内亚微秒级 D2D 暂存对齐；
     3. **有状态原语硬件 `valid_rows` 截断**：`paged_attn`、`fused_moe`、线性注意力状态更新等 C++ 自定义原语统一经 H2D 元数据头传入 `valid_rows = M`，TPC 核对 `row >= valid_rows` 直接跳过 KV 写入与专家计数，彻底免除上层插件构造假页表（scratch page）与拼块缓冲（`P-RT-17`）。
   - **冷路径（零散图外胶水即用即发）**：针对权重加载切片转置、初始化预计算等非热循环操作，提供 `as_strided` 零拷贝视图与用完即释或小范围 `DATA_TENSOR_DYNAMIC` 微型缓存算子直接入流发射，以及标准 `c10d` `ProcessGroupGaudiRT`；
   - **铁律边界**：候选进程物理隔离旧桥（前置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，严禁 `#include`/链接 `gaudi-pytorch-bridge`、严禁加载 `habana_frameworks` 或启用 `PT_HPU_LAZY_MODE`）；零 LazyTensor 动态图追踪、零静默回退（未实现算子或越界输入立即 Fail-Fast 报错）。
4. **模型与服务生态层（`sglang` 极薄 OOT 平台插件、`DeepInfer` 原生引擎对接、`vllm-gaudi` 对照封存）**
   因静态分桶与尾行隔离已在第 2、3 层 C++ 内部完全透明吸收，上层无需任何 Padding/凑整侵入，直接零改动复用上游框架模型定义（如 `南北阁 33B`、`Qwen 3.8 Next`、`Hy4` 等 SGLang 200+ `models/*.py`）、RadixCache 前缀树、混合线性状态池、Chunked Prefill 与投机解码，仅通过极薄的 Out-Of-Tree `SRTPlatform` 插件或 `DeepInfer` C++ `zos` 引擎绑定接入；前期 `vllm-gaudi/zero_bridge` 单模路径封板留作底层回归基线。

---

## 二、闭源 SDK 与底层部件处置死规矩

Intel 将离线图编译器与在线发射运行时打包在同一个 `libSynapse.so` 中，讨论与编码必须按职责严格拆开，执行以下固定边界：

| 部件 | 处置决定 | 具体边界与理由（基于当前 SDK `1.24.x` 与真实目标负载） |
| :--- | :---: | :--- |
| **1. 官方算子库 (`libtpc_kernels.so`)** | **通用算子复用；<br/>核心原语替换** | Core ATen 基础逐元素/变形算子直接由 `FX -> Synapse` 编译层复用官方 GUID；对缺失或慢于屋脊线的 5 大核心原语用 `gaudi-kernels-public` 单一打包 `.so` 覆盖。 |
| **2. 离线图编译器 (`synGraphCompile` / `libSynapseMme.so`)** | **永远不重写** | 任何阶段都不自写 `EcbGraphCompiler` 去替代 MME 切分、SRAM 双缓冲排程与 ECB 打包。只把它当 `FX -> Synapse` 编译层的后端汇编器调用；若过度合图导致排程倒退则保留分段切点，唯一标准是真机耗时变快。 |
| **3. 在线发射器 (`synLaunchWithExternalEventsExt`)** | **Phase 1–7 原样调用；<br/>Phase 8 局部旁路** | 主线直接调公开发射接口配合 `PreparedLaunch`；仅在 Phase 8 经实测账单证实公开接口吃不掉发射/同步瓶颈时，吃 `synGraphCompile` 编出的 `.recipe` 用 `scal/` 局部旁路下发，绝不演变成重写通用裸加载器。 |
| **4. 集合通信库 (`libhcl.so` / `hccl*`)** | **Phase 1–7 原样调用；<br/>Phase 8 按需局部旁路** | 直接通过 `ProcessGroupGaudiRT` 调公开 `hcclAllReduce` / `hcclAllGather` / `hcclReduceScatter` / `hcclAlltoAll` / SendRecv；仅在 Phase 8 证实需跨计算+通信片上循环时按需接管对应通信 CCB。 |
| **5. 队列层与内核驱动 (`libscal.so` / `libhl-thunk.so` / `habanalabs.ko`)** | **当前原样复用** | 在当前已测单机 8 卡静态稳态推理路径下不是主要瓶颈，不改内核驱动，不重写 `libscal.so`；若后续跨机 EP 或新映射路径出现新证据再重审。 |
| **6. 片上 ARC 固件 (`/opt/habanalabs/engines_fw/*.bin`)** | **Phase 1–7 不动；<br/>Phase 8 定点增强** | 仅针对通用服务闭环实测后仍吃不掉的设备端动态控制瓶颈改造未签名的引擎/调度固件；严禁触碰带硬件签名的 `habanalabs-firmware-odm` 板级底层固件。 |

---

## 三、当前真实状态与测量纪律（截至 2026-11-06）

### 已证实的资产与基线
- **Phase 5A 现状**：`GaudiRT` `0c4d16f`、算子包 `ddfe12e`、SGLang 通用钩子 `ac0d513` 和 OOT 插件 `9e84412` 保持分支隔离，模型定义无改动。无状态 FX 分段与有状态算子独立；已撤除多次 GEMM 精修、纯 TPC 注意力回退和旧全表 MME 注意力。新的分页分块注意力尚未接入，入口在写 KV 前报未实现，**当前版本不能推理**，不能把旧版功能闭环当作本版验收。
- **已验证的局部底座**：类型化 `1D..5D` 构图、原生多输出 `split`、静态桶及零拷贝视图已做单卡图检查；`M=1/3/17` 的多维图冷暖逐字节一致，暖载 18 张、重编 0，最大行相对 FP64 误差 `5.28e-8`，冷暖收据 `964e538f3c21833f98837afcce18dd627b6db71ec03c9aa35850fda0aa68316f`、`f88d279df84f615bf4896084af0d5ab5768f82e87a295de955402aa1222e75d0`。目标宽度 `7168` 的两层投影及拆分最大行误差 `0.000379147`，纯 FP32 底噪 `0.000367940`；这不包含完整 Decoder 层，收据 `2c9ea49b618189c394ffbfa2db94b23da2bbac6692fbe3bf0ec515bd5b2d9274`。持久事件、轻量 `DataPtr` 和暖子图发射的并发局部检查通过，但整模稳态事件、分配和锁开销未验证，见 `P-RT-64`；embedding 的桶尾处理见 `P-RT-68`。
- **注意力局部算术与未解决风险**：两块未分页 MME 图在 `56/8`、`head_dim=128` 的 `M=137` 样本上，相对独立 FP64 / 同块 BF16-PV 最大行误差分别为 `0.0026753 / 0.000263724`；`M=1`、有效 KV `4089` 时为 `0.00264149 / 0.00012085`，两组冷暖逐字节一致、暖重编 0。后者 200 次暖发射设备流区间平均 `0.0343714 ms`，含提交间隙、不含分页 Gather、KV 写入或调度；数值及计时收据 `f2ae6697d0376fcc22edc5ea9379f5f588ba039aeadf14f918cf537d32d7e8aa`、`4b5fa7696268d74b2c45b2b549881f330a878f7bc41040b3d6668a66963b985a`、`c5f0ecea54327d3d31b7a849f1723b083677e060b8ad74504074109c0fbac38d`。固定 KV 大描述符仅在额外隔离的两个 KV 池实验中证明 `2048/8192` 槽的冷暖写后读取及哨兵正确，不能外推整模布局，见 `P-RT-67`。将 KV 写入、Gather 与两块算术合图的首次发射触发 TPC 地址 `0` 页故障，驱动发起复位并终止进程，退出码 `137`；节点及根因未知，事后占用回基线不证明后续计算可用，见 `P-RT-70` 和故障收据 `724a3e048ae4a99bdd93df08a74cd3db6cb2b65bef18edf8c4046dcff343e8b1`。
- **数值与生命周期未通过项**：宽度 `64`、`M=65` 的 RMSNorm 单算子 / 连接线性与残差的最大行相对 FP64 误差 `0.00322201 / 0.00363345`，均超过 `0.003` 门槛，见 `P-NUM-28`。本轮归档中 294 个候选 Python 进程有干净入口与退出记录；一次早期 `split` 实验只有入口且退出码 `139`、根因未知，不能算生命周期通过，汇总 `0db650384d8efd62cf1267e5bef81828c675229e6b72b81cf325c623bb6d8313`，见 `P-RT-69`。现行构建的 `ldd` 局部检查未见旧桥，整模主进程及 Worker 尚未重验。
- **52 层 33B 旧版数值与状态**：目标权重 29 文件共 `66,260,069,914` 字节，14 个分片与发布方 SHA256 一致，收据 `46cf6179c0be09be6b366ab11cbbde9af53ae359c34924b72504172ec0511c10`。旧版固定在线前缀记录 52 层最大行误差 `0.000474189`、条件化 KL `0.000042036`，但当时 FP32/FP64 参考在层内含 BF16 舍入；**撤回其满足现行纯 FP32 完整层及 KL 门槛的结论**，旧口径收据 `b913a80c50c4e4a722a99aa3949bcecfa0bca62b5a5c2755124dc4411369f907` 仅作诊断。旧版 `137/1053/4096` 输入各 128 步的冷暖 logits、KV 写入与终态逐字节一致，目标、历史及未触碰槽审计通过；末次前向冷编 60 张、暖载 60 张且重编 0，冷进程退出后磁盘共 62 张，收据 `2124b9ece2cb5e09aa952692c16cc7a3a97231f6ff9b18205d3c61e7a18d64f5`、`e9c5d9595fa0164db9b4cbe725c4ed6f51b0987c7d5605ada6266750bae7d182`。这些均非当前版本成绩。
- **旧版无审计性能基线**：SGLang Engine 单卡 `TP=1`、贪心、完整 128 token 预热、计时窗口零新编图、关闭 RadixCache 与 Chunked Prefill 时，`Prompt=1024/4096, Output=128` 的 TTFT 为 `11021.235 / 144203.557 ms`、TPOT 为 `60.261 / 99.638 ms/token`；分配器峰值 `64.026 GiB`，计时后常驻 Recipe `71 / 73` 张。收据 `0557e69ed0700deb0d17b0540f762fb81423a176b1b1918b6ed7c851c95f8f80`。当前版本的端到端数值、状态、服务能力及性能均未测。
- **本轮收据**：仓外 `instance-assets/phase5a-generic-slice/refactor-native-receipts-v1/receipts.sha256` 覆盖 1170 个记录和诊断文件，`sha256sum -c` 全部通过；清单 SHA256 `c2660d6442a9e75921c868368a14d9af399377ac9fd54f45d237734c8a7f8957`。清单只证明归档完整性，不代表阶段验收通过。
- **70 层真实模型参考包已冻结**（`instance-assets/mimo-layer1-20261026/reference-v1/`）：
  - **全模型无插桩稳态基线**：70 层 TP8、Batch=1、Decode M=1、context=4096 下，两轮 128-token 稳态 p50 为 **`11.284 ms`** 与 **`11.306 ms`**（`88.41` / `88.32 token/s`），输出序列逐 token 一致。
  - **第 1 层真值与权重清单**：8 Rank 经独立交叉核验与 `plain` 逐位一致（`max_abs = 0.0`）的阶段真值，以及每 Rank 15 个张量的 `GAUDIRT_WEIGHTS_V1` 权重清单已就绪。
- **真实 layer1 C++ 数值已在 8 Rank 通过**：Attention、Post-Residual、Norm、Router Top-8、MoE local 和最终 `layer_output` 均与冻结参考逐位一致；50 次设备事件采样的各 Rank p50 为 `347.696–362.569 µs`。这些只是当前实现的单层测量，**不是相对生产路径的提速结论**。
- **RMW 实验已撤回**：两轮 ABBA 的候选方向不稳定且出现反转，当前代码不再包含该开关或默认启用 RMW；原始记录保留在 `instance-assets/mimo-layer1-20261026/layer1-sram-*`。
- **生产 standalone 单层基线已合格**：`instance-assets/mimo-layer1-20261026/performance-standalone-v1/baseline.json` 记录了 12 次预热、64 次样本和完整数值校验，逐样本 TP8 最大设备耗时的 p50 为 **`507.9285 µs`**；这是静态单层 replay，不是 TPOT。
- **单层 ABBA 已证实候选净收益**：生产 A1/A2 p50 为 `507.9285 / 520.596 µs`，C++ BF16 B1/B2 为 `345.606 / 302.7715 µs`；两臂均优于两臂生产对照，p50 均值下降约 **36.96%**，生产对照漂移约 `2.49%`。证据在 `instance-assets/mimo-layer1-20261026/performance-cpp-abba-v1/abba.json`，**只限静态单层，尚未证明整模或生产闭环**。
- **FP8 候选已被数值门禁否决并撤回**：Rank 5 的 o_proj `max_row_relative_l2 = 2.0488925%` 超过冻结的 `2%` 上限，且 8 Rank 的 post-residual 和归一化输出均出现位差异。未启动 FP8 性能测量，未降低阈值；失败证据在 `instance-assets/mimo-layer1-20261026/oproj-fp8-v2-safe-scale/numeric-failure.json`。
- **共享 Recipe 跨层事件归因已核实**：`instance-assets/mimo-layer1-20261026/recipe-attribution-v1/attribution.json` 对 8 Rank × 3 样本验证了层尾 Recipe 每次发射重复 60 次、跨层边界 Recipe 重复 59 次。`model/N` 是静态节点名而非运行时层号；旧 `2,384.5 µs` 是跨层聚合，不是单层耗时。
- **MoE 在线 Loader 字节等价已证实**：`instance-assets/mimo-layer1-20261026/moe-loader-equivalence-v1/validation.json` 验证 8 Rank 共 32 个实际生产 `gp/gs/dp/ds` 缓冲区与离线原始载荷逐字节一致，并固定生产 Adapter、packing 实现和 checkpoint 索引哈希。
- **薄 PyTorch 2 Bridge 的单层数值门已通过**：`instance-assets/mimo-layer1-20261026/bridge-layer1-v1/validation.json` 及 `run-v21/` 证明 8 Rank 各一次受限 FX 后端编译只包含一个 `gaudirt.layer1` 计算节点和两个元组取项节点；真实生产权重、输入、第二 residual 与完整 KV 均逐位一致。当前只支持 B1/M1、position4096，设备、分配器、流与 workspace 仍借用生产上下文，**不是独立 PrivateUse1 替换、同构层复用、整模闭环或 TPOT 成绩**。
- **两层同构 Recipe 复用已通过**：`instance-assets/mimo-layer1-20261026/bridge-layer1-v1/validation-reuse.json` 及 `run-reuse-v3/` 证明同一 compiled FX 函数执行真实第 1、2 层时只编译一次，rebind 前后 Attention/MoE recipe 指针不变，两层输出、residual 和完整 KV 均逐位一致。该结果仍限 B1/M1、position4096 的 `swa128` 层型，**未覆盖全部层型、`.recipe` 磁盘缓存、动态形状或整模**。
- **Full-Attention + MoE 数值参考探针已跑过，但实现违反零旧桥红线须重做**：`instance-assets/mimo-layer1-20261026/bridge-layer1-v1/validation-full.json` 及 `run-full-v2/` 验证了 layers.7 的独立 RoPE 表与无 sink 数值契约，但其实现把单层拆成半截接口并退回旧桥 Python Lazy 执行 `flat_pa`（未走独立 `GaudiRT` 闭环与受限 `torch.compile`），**仅可作为数值探针参考，严禁作为正式层实现保留或扩 70 层**。70 层实际包含三类层型：layers.0（纯 RMSNorm + Full-Attention + Block-FP8 `W8A16` Dense MLP）、9 个 Full-Attention MoE 层、60 个 `swa128` MoE 层。

- **零旧桥 Dense MLP 切片已通过**：`instance-assets/gaudirt-zero-bridge-20261028/run-dense-v4/validation.json` 记录 8 Rank 的 normalized、gate_up、图内 SwiGLU、residual 和最终输出均逐位一致；二进制只链接公开 SDK，无旧桥依赖。权重由纯 CPU checkpoint 按 Rank 分片并打包。**该结果只覆盖 layers.0 的 Dense MLP，不包括 Full-Attention、整层 FX、PrivateUse1 或性能收益。**
- **三类层的统一零旧桥 PrivateUse1 已验收**：`instance-assets/gaudirt-zero-bridge-20261028/phase2-v16/validation.json` 证明每 Rank 一个候选进程内可执行 Type A/B/C，11 个动态张量全部显式入参，FX 每层只有一个 `gaudirt.layer` 节点，三类 KV 均逐位一致。Type A/B 最终输出最大行相对误差分别为 `1.178e-5`、`3.644e-4`，Type C 输出逐位一致。权重初始化期注册，运行期仅换地址，不同层型持有独立 Recipe。
- **Recipe 冷编译与暖加载已验收**：`phase2-v18-cold` 每 Rank 实际序列化九张模型 Recipe，`phase2-v19-warm` 的新进程每 Rank 实际反序列化九张且零再保存，八 Rank 指纹一致且数值/KV/FX 全绿。72 个 Recipe 文件仅保留在远端；**该结果不是加载时延或整模性能结论**。
- **真实两层零旧桥复用已验收**：`phase2-v22-real-reuse/validation.json` 证明单个 FX 图直连 layers.1 和 layers.2，复用同一 SWA Attention/MoE Recipe 并换用真实异层权重，八 Rank 的两层输出、residual 和 KV 全部逐位一致。layers.2 使用真实 128 槽 SWA 窗口，窗口外显式置零，**不冒充生产全量 KV 初态**。
- **显式 HBM 预算与 safetensors 流式直传已验收**：`phase2-v24-stream/validation.json` 证明候选进程使用显式 8 GiB 配置，CPU 从 checkpoint 切片/打包后直接 H2D，不读取权重 `.bin`；真实两层输出和 KV 在八 Rank 逐位一致。**该结果只证明两层流式加载机制，不是完整 70 层闭环或 TPOT 结论。**
- **70 层零旧桥设备端步图（Device Step-Graph）、TPC `v_exp_f32` 边界修正与 128 步状态审计已完成**（候选分支 `feature/mimo-swa-exp-boundary` `76826f2` + 算子分支 `feature/gaudirt-swa-exp-correct` `fd75cd6`）：
  - 定位并修复了 Gaudi2 TPC `v_exp_f32` 在分界点处的硬件算术缺陷（如 `exp(-ln(2))` 返回 `1.0` 而非 `0.5`，单卡掩码 Softmax 最大绝对误差 `0.166667 -> 3.39e-6`），已覆盖 MoE SwiGLU 门控、Full-Attention Softmax 与 SWA Softmax，并将 SWA 融合 Norm 残差归约与 QKV value scaling 统一为 FP32。
  - 8 Rank × 70 层 × 128 步逐槽位 KV 写入与保持审计全部通过，未触碰 KV 区域逐位不变（`max_abs = 0.0`），8 Rank 生成序列与无插桩运行逐 token 一致（收据 SHA256：`ff968f35a54adb5742e29136899ac17fcedc647c7d3314be2a65af62209572dc`）。
  - 首步全词表对独立 CPU FP32 参考 `max_kl = 0.0007722`、`top1_match = 1.0`（`relative_l2 = 0.0205696`，归一化能量差 `calc_diff = 2.1e-4`）；对独立直接舍入 CPU FP64 参考 `max_kl = 0.0037558`（`relative_l2 = 0.0236053`）。无插桩单跑 70 层 TP8 耗时为 `p50 = 11.0613 ms`、`mean = 11.1134 ms`（收据 SHA256：`bb6c5be08e99ec5597feb26a970f90f467615cf5021329c1c84c52a4a3f2bcc8`，**仅限禁用 RecipeCache 的单跑，尚未作为最终 ABBA 成绩**）。
- **70 层 MoE 离散路由与 BF16 层边界的浮点物理噪声底已实测确立**：
  - 在同一 4096-token 输入与同一 BF16 层间存储契约下，**纯 CPU FP32 参考对直接舍入纯 CPU FP64 参考**自身的首步全词表 `relative_l2 = 0.024443`（`max_kl = 0.003246`），并在 69 个 MoE 层中的 **16 层**选出不同 Top-8 专家集合（收据 SHA256：`befaaaf5c5656ddd15adbe68a7b6a99a4d059b2034bb3f42a896b8a1f7055cf9`）。
  - 在冻结同输入、同 Top-8 专家的单层对拍（Layer 39）中，纯 CPU FP32 对 CPU FP64 的输出相对误差已达 `0.002891`，而设备候选对 CPU FP64 为 `0.002438`（收据 SHA256：`e3bf5bc7dadd7c03fc943adbabcd3649c792cd9493682a94f8d2c2b770d8b7ea`、`0a3fe791d0e92f14856936302b2695578a5eb4d5182e933e4fe9faa50f859eff`）。
  - 原第四节对跨求值树外部参考设定的“单层 `<= 1e-3`”与“70 层滚动全词表裸 logits `relative_l2 <= 0.02`”硬线，低于同精度 IEEE 浮点归约与离散 Top-K 路由本身的物理噪声底（纯 CPU FP32 自身即不达标），且偏离仓内原契约（`tools/validation/executor/fp32_quality_contract.py`：`max_kl <= 0.01`、`raw_logit_relative_l2 = 'diagnostic'`）与主流 GPU 引擎分层标准，已在第四节公开订正。
- **Phase 3 完整 70 层零旧桥独立解码闭环与 4 臂 ABBA 已完成验收**（收口分支 `feature/mimo-phase3-closeout` `61241ef` + 算子分支 `feature/gaudirt-swa-exp-correct` `fd75cd6`）：
  - **70 层冻结同输入路由与算术全过**：8 Rank × 70 层独立检查通过，69/69 个 MoE 层均选中 FP64 参考专家集合，单层最大 `max_row_relative_l2 = 0.0024381`、最大 `max_abs = 0.03125`（收据 SHA256：`36e8472ddf295499288e01c2ce124ae3737ac50c828c4b925e2fdf71989ed6a5`）。
  - **TP8 词表切分 `lm_head` 与构建指纹 `RecipeCache` 冷/热逐位等价全过**：词表按 TP8 切分为每卡 `[6144, 19072]` + 流内 `hcclAllGather`，单卡（设备事件 p50 `897.493 -> 214.124 µs`，收据 SHA256：`2e3e4afc1efc040192cde0e0451e076eacbc9c587c324ab1ff05e1f7d726e242`）与 8 卡 AllGather（收据 SHA256：`a2ad4ca77752008778e328f7330cc1034b60188d8b4ab080339c5241db2cf56c`）均逐位零误差；70 层冷编译每 Rank 保存 10 张自编译 Recipe，暖进程仅加载 10 张、重编 0 张，首步 logits、128 步 token 与终态 KV 与已审计候选逐位一致（收据 SHA256：`d85a2d7a03d3691efe378ab7b7434796bd87c46f885b756b1fa6ed806ebc3498`、`ab1d75721324c1564a6d45e2170738ff6c8631d3ceb44aa7fa9743cbc9cad826`），独立观察图 69 个 MoE 层 × 128 步冷/热有序路由 ID 逐位一致（收据 SHA256：`3e86346d72bd1e7da5747fa8dddad56c7d56065e6e0c0756ec5de3c6b93331ba`）。
  - **无插桩 4 臂独立解码 ABBA 胜出**：`Batch=1, Decode M=1, initial context=4096`（16 步预热 + 128 步采样）下，生产 `A1/A2`（完整 `vLLM engine.step()`）p50 为 **`11.4145 / 11.2448 ms`**（mean `11.4652 / 11.2902 ms`，漂移 `-1.49%`）；零旧桥候选 `B1/B2`（动态元数据 + TP8 70 层整模 + 设备选词 + 生产客户端 `OutputProcessor`）p50 为 **`10.5119 / 10.6084 ms`**（mean `10.5317 / 10.6225 ms`，漂移 `+0.92%`）。候选两臂均快于两臂生产对照与历史 `11.284 ms` 门槛，p50 均值净省 **`0.7695 ms/step`（`-6.79%`，`94.70` vs `88.26 tok/s`）**；四臂主进程退出码全为 `0`，候选进程零旧桥依赖，80 张缓存文件未变（收据 SHA256：`93c16cb993b1ff0c4f8242e1e70914045c19b1518c01c921e2bc9be0f270a5aa`、`070b03c9a97ce8f01c0cc7fb81cc16beaa5eed814d6bb279b1fb511a45c0ea4b`）。

- **Phase 4.1 底座防呆、生命周期固化与第 6 步平票核实已完成**（`feature/gaudirt-safe-math-prefill` `502c072`、`feature/mimo-phase4-serving` `2f6a401`）：
  - **TPC 安全指数公共头落地**：`csrc/tpc/common/tpc_math_safe.h` 统一收敛三类算子的 `v_exp_f32` 分界点修正，750 点硬件边界回归与单卡新旧输出 **100% 逐位一致（`max_abs = 0.0`）**（收据 SHA256：`0765f89bfa6f35204d3f41549c641748dd53481f6cc052fc0874530cfaa66d33`、`d10ab28752b4b30be1696087f5b2d841aad5a65b6fa0dec95bf94c883d72ba3d`，详载 `docs/gaudi2-pitfalls.md` `P-TPC-01`）。
  - **TP8 退出生命周期封装与旧桥物理隔离**：跨 Rank 退出栅栏、流排空与编译图/`PrivateUse1` 张量先于 `DeviceRuntime` 释放顺序封装入 `private_use1.cpp` 与 `runner.py`，4 轮 8 Rank 延迟退出冒烟与后续整模冷/暖运行全程 `exit 0` 零原生终止告警（收据 SHA256：`4f3f48a4641529fbd4e5b70be9e944bc4df93e16895f01f696b9be147c159181`）；强制前置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 封堵 PyTorch 导入时自动挂载旧桥（`P-RT-09`，收据 SHA256：`7020be912798a121c8adc8526dfdb7f793f44cf0192a1a66657c65dd27c943e1`）。
  - **Phase 3 第 6 步（`pos=4102`）分叉真相查明**：同冻结 KV 下，候选、关闭整步图的生产 Eager、独立 CPU FP32 与独立 CPU FP64 四方 Top-1 **全部选 `51273`**（候选对 FP32/FP64 的 `max_kl = 0.001166 / 0.001401`）；该步前两名 `51273` vs `35085` 在 FP64 下仅差 `0.08301`（小于该幅值 BF16 间隔 `0.125`），开启生产整步图时两词 BF16 logits 同为 `22.75` 平票而按较小 ID 选 `35085`（详见 `P-RT-16`，收据 SHA256：`ed12ae92fe1655d33144e1fb79cd7b2a234723f13191af4679f92f977fa79968`、`75ccef5c70ffffb6d80d05b8a44347e158e88959d8c3e74a998837df7a84f416`、`b981ec9d6ec9530183385a2e6aeb815d17fd429534bebcbae9f2e57ee4e92c55`）。
- **Phase 4.2 零旧桥 `M=32` 分块 Prefill 与空 KV `4096 Prompt -> 128 步 Decode` 独立闭环已完成验收**（`feature/mimo-phase4-serving` `8fc3b3b` + `feature/gaudirt-safe-math-prefill` `d23e585`）：
  - **算子打包、补偿舍入与三类层 TP8 / 长上下文 FP64 全过**：将自研 TPC provider 静态合并进单一打包库以满足 Synapse 15 库硬上限（`P-RT-14`），并修复 QKV 归约在 BF16 中点处的二次舍入偏差（`P-NUM-09`）。三类异构层（Type A/B/C）8 Rank 空 KV 首块 `M=32` 对 32 次串行 `M=1` 输出与 KV **100% 逐位一致**（收据 SHA256：`5602ae037ea82e48f39b37e53f7eb203a0df310ecf27edfe6c6d00f112f53318`、`f746d975cfc9f3d56f27ca8c7186d680b45ede9310e87fa7be3d9f2576031cf0`）；真实首块 32 行全行对独立 CPU FP64 最大行相对 L2 分别为 `3.15e-4 / 6.24e-4 / 2.75e-4`（512/512 个 MoE 路由集合全匹配，收据 SHA256：`0c1da8bac37b7394edcf526b531dda3496172e2604f4d1675ca487b42199fc30`、`9d847418e4bfa8ee78c0ca0c21ed55db99b1727f7ad17f1cfe998de01f09f446`）；真实 `pos=4095` 长上下文在线 KV（Layer 7 含 4095 行 Full KV、Layer 8 含 127 行 SWA KV）对 FP64 整层相对 L2 仅 **`2.41e-5 / 1.42e-5`** 且路由一致，33 行 `gk_norm_block128_grid24_fp32_v1` 融合编码与残差逐位匹配 FP64（收据 SHA256：`43d317abfaf2627ccd869cfe396637f6e8531855b075d18020d03d71a91d49ad`、`3b9051c41e2a68a87b4be92d6ec3985c3d7170ca3fd83f86f05b20fa3e08da9e`、`b06fc76c2dd8565f2b4a3fc9fc1f4d902d470cf1d809e43559c7dcdaa9438998`）。
  - **70 层全零空 KV 端到端冷/暖确定性与逐槽位 KV 审计全过**：不加载旧桥、不借离线 KV 快照，从全零空 KV 接收 4096-token 真实 Prompt 完成 `128 × M=32` Prefill + 128 步 Decode；冷编译每 Rank 保存 20 张自编译 Recipe，独立暖进程每 Rank 加载 20 张、重编 0 张，首步 logits、129 个生成 token 与 140 个终态 KV 缓冲区冷暖 **100% 逐位一致**（收据 SHA256：`9f7dd5b21605e082dcea06f4df9d855ff65c48e7afcfd5070271c2367248097f`、`424daa81971473fb224c1af27fc315f59b37a1811a42d5d88ec63c1b6497011a`）；8 Rank 逐步插桩审计全部 128 个 Prefill 分块 + 128 个 Decode 步，已写槽位零覆盖、未触碰槽位恒零（`max_abs = 0.0`），生成英文长段连贯可读，全进程 `exit 0` 释放八卡（收据 SHA256：`55654b029f2758db1b91fd4f405596ad6108284bef0dbc53b6797ef2e6924fec`）。
  - **空 KV 4096 长前缀 70 层 FP64 与混合精度参考对照（不再为长前缀另跑纯 FP32 参考）**：① 固定候选在线前缀 KV 重算末 token 70 层，候选对 CPU FP64 的 **`max_kl = 0.00620 <= 0.01`**、Top-1 一致（同条件下混合精度 CPU 参考（FP64 RMSNorm/Router/RoPE/Attention，FP32 线性层与 Rank 求和）对 FP64 为 `0.02178`，收据 SHA256：`0d10c115448317bec44c7320dc978e3937d6078555807b682bd9c0855703727a`、`269557b1fe256cc310d1f6f908f5b8de1a1037a7195aafec33ce8f6e98ac1a36`）；② 从空 KV 独立滚完 4096 token × 70 层（复合 `282,624` 次离散 Top-8 路由写入在线 KV），**零硬件参与的混合精度 CPU 参考对纯 CPU FP64**为 **`max_kl = 0.043453`**（`relative_l2 = 0.122038`，翻转 32/69 层，第 8 层起翻），而 **Gaudi2 零旧桥候选对同一份纯 CPU FP64 真值**的 **`max_kl = 0.034032`**（`relative_l2 = 0.128379`，Top-1 与 Top-4 排序 `[576, 1096, 342, 151645]` 100% 一致，仅翻转 28/69 层且前 17 个 MoE 层全匹配），比该混合精度参考贴近 FP64 达 **`21.7%`**（此前误称纯 FP32，已撤回；源码与结果复核收据 SHA256 `78d449fd87991a36a0ef1ea8da37820b27d25fcf6df2c19b98f1d31d83bf652a`）（收据 SHA256：`96276b2a260b5640cb9c3a7f435a9976178a5afbcb3f05e8a30853a71265a839`、`a79278bd63dfe4967f5ec6c9e5db88085a610a1a8b5aad28e5a5699c4179ea53`、`a048ff3d98880ad4845162a43f8cd5dd27dedcfc51c07a1c31b911d0696e5bbd`、`c5c679d889776ea147099538bdf038b223fa2ca2fc6317dd5fe7af8b9867a236`）。
- **Phase 4.3 单流服务路径与同口径 ABBA 已验证**（`vllm-gaudi` `ad7f4f0f`，实测 `GaudiRT` `66cd1a6`，算子包 `d23e585`）：
  - **服务与状态**：真实 vLLM V1 调度、分页块表、8 Worker IPC 和 `OutputProcessor` 贯通；空 KV 4096-token Prompt 的首步 logits、129 个输出 token 与独立 Runner 逐位一致，暖运行 8 Rank 70 层最终 KV 逐位一致。冷、暖分别每 Rank 保存 20 张和加载 20 张自编译 Recipe，暖运行保存 0 张。收据 SHA256：`66fcf3c28741bf21444e89b094f5b55a556308f08a616286d4ac5879e59e2510`、`505f94136d141698fbf2dacba522d74afdb4dc6d034ec993b0cb4940458690d7`。
  - **完整 `engine.step()` 服务 ABBA**：每臂 1 次预热、3 次正式请求；A1/B1/B2/A2 的 4096-token TTFT p50/mean 分别为 **`2570.86/2650.66`、`8436.30/8436.56`、`8435.89/8436.04`、`2567.78/2567.92 ms`**，128 步 TPOT p50/mean 分别为 **`11.522/11.520`、`11.016/11.103`、`10.952/11.069`、`11.384/11.407 ms`**。生产 A1→A2 的 p50/mean 漂移：TTFT `-0.12%/-3.12%`、TPOT `-1.20%/-0.99%`；候选 B1→B2：TTFT `-0.005%/-0.006%`、TPOT `-0.58%/-0.31%`。两次候选 token 逐位一致，候选与生产从第 13 个输出 token 起分叉。全部 Worker 与 EngineCore 退出码为 0、候选缓存未变、八卡释放；`same_complete_production_serving_path_proven = true`。汇总收据 SHA256：`7e4907c876dd8f38064fd47ef619df6cf41fdf08c85107c580e7cf905d2dc4a6`。
  - **性能边界**：两臂 p50 的简单均值下，候选 TPOT 约快 `4.1%`，但 TTFT 约为生产的 **`3.28×`**、增加约 `5.87 s`；这是明确的 Prefill 退化，不能宣称端到端全面加速，瓶颈原因尚待诊断。生产空 KV 首步对同一 CPU FP64 的 `max_kl = 0.056061`、Top-1 一致，仅作长前缀横向参考（收据 SHA256：`e0e53f85a8b584b6c87cfd5468f3394502a2df15d2b381c92eeb6884a343cb0b`）。
- **Phase 4.4 单流服务与性能闭环已验证**（Prefill 静态桶 `M<=2048`、32 倍数长度；`GaudiRT` `aabf3ab`、算子包 `f127cec`、`vllm-gaudi` `2a91c397`）：
  - **真实单流服务性能**：M=2048 上限、空 KV 4096-token Prompt、每臂 1 次预热和 3 次 128 步请求，完整 `engine.step()` 无插桩 A1/B1/B2/A2 配对；A 为生产调度 `533`，B 为零旧桥原生调度 `2048`。四臂 TTFT p50/mean 为 `2570.877/2646.591`、`2374.533/2374.566`、`2374.031/2374.054`、`2570.326/2570.539 ms`；TPOT p50/mean 为 `11.406/11.411`、`10.959/11.057`、`10.981/11.124`、`11.369/11.394 ms`。两臂 p50 均值分别为候选 `2374.282 / 10.970 ms`、生产 `2570.602 / 11.387 ms`，候选分别快 `7.64% / 3.67%`。A1→A2 的 TTFT/TPOT p50 漂移 `-0.021% / -0.328%`、mean 漂移 `-2.874% / -0.154%`；B1→B2 的 p50 漂移 `-0.021% / +0.201%`、mean 漂移 `-0.022% / +0.608%`。A1 的三个 TTFT 样本中有一个 `2798.874 ms`，故其 mean 漂移不能仅按 p50 判断。两次候选各自及跨臂的 token 一致，候选与生产从第 13 个输出 token 分叉；所有 EngineCore/Worker 退出码均为 0、候选缓存未变、八卡释放（设备收据 SHA256 `5e6ebee0ea060b69634bc4b1c13a1276139bc28595f50fb1b817a1c447e34069`）。汇总收据 SHA256 `51c05a86b07bbb51e4ddeda048bc1c2b9d31ef1e87b326ab0c8b49b5a9e5c47a`。
  - **质量与范围边界**：真实服务对 `256 / 2080 / 3104` token 各完成 12 个生成 token 的变长冒烟，4096 token 完成 128 步；不表示所有长度均已实测。空 KV 4096 全前缀对独立 FP64 `max_kl=0.046614`、Top-1 一致；当时使用的 `0.04997` 门限源自混合精度 CPU 参考，该跨归约树全前缀 KL 仅作诊断，现行硬门禁见第四节；历史服务及性能结果不受此口径修正影响；冷暖缓存 42 张/Rank，首步 logits、129 token 和最终 KV 逐位一致，128 步未触碰 KV 区 `max_abs=0`。70 层设备 TP8 归约逐位一致，69 层冻结同输入设备 Router 的非临界专家集合均匹配 FP64；同前缀候选条件化 `max_kl=0.011019507 <= 0.012683427`，L34 按第四节 SWA 双 BF16 分解条款通过。收据 SHA256：归约 `51fa8a6219f60432cbf7ec914aa8ed54786515cd36283950c4f3d125a8b20fa7`、Router `cab910925106b94b61f33ff853bfc9846332ea48a8bc883934a89dac3ea4a936`、冻结对拍 `019f6742da11a201896031e22afbd62f02efcf462cdaf3965288ebe1274e96aa`、条件化包络 `4d600285cf05ddac3b1836ec5f35a2f011338841df63e30958963fc6d595d240`。候选 `ldd` 无旧桥依赖且无未解析库，主进程及 Worker 初始化/退出执行模块和映射检查，收据 SHA256 `d27fe5ad4dfdcc34cf4673be7b1727b60d4d105acb877403a3d710538f1842de`。当前范围为单流、贪婪生成、Prompt 为 `32..4096` 的 32 倍数、Prefill 分块 `M<=2048`、最多 128 步 Decode；尚未覆盖非 32 倍数尾块 padding、`Batch > 1` 并发与通用采样。

- **`adaptive-infra` 既有分支的真实服务验证**（仅限 MiMo、TP8、贪婪生成；`GaudiRT` `dfa8b30`、算子 `89c3508`、`vllm-gaudi` `d32c7cba`）：
  - **数值与状态**：8 Rank 的 M=1、M=2048 末位置 70 层冻结同输入检查及 B4/B8 共 12 个请求行的 840 项冻结单层检查通过；后一组 828 项设备 Router 专家集合均匹配 FP64。4096 同前缀末步 `max_kl=0.00718409`、Top-1 一致；冷跑每 Rank 保存 43 张 Recipe、暖跑每 Rank 加载 43 张且重编 0 张，首步 logits、129 token 和 70 层终态 KV 冷暖逐位一致，128 步未触碰槽位 `max_abs=0`。M=2048 检查的真实冻结输入仅覆盖末位置，其余行是合成输入；不能据此宣称所有 Prefill 行都对过 FP64。收据 SHA256：M=1 `75b209c780ab1ee9936f3365e832bf1c104778dabee455b8724778c5d5ebd736`、M=2048 `bbab92dfa519832551d3456f76dc04945f5b88bc06122ef78b4cbdd2b3e77bc7`、批次算术 `6f78ab11f878e8f3b6ed78c5e907691e5d658ec913be0c80ff48959479566364`、批次 Router `14e8431ec2971ad4026194993f4206aaa8fa3501cd74d7a41c103f57c9652702`、条件化 `a7c4854ae35f253e3b94400bab195d511a437d868756db6caef028edf239538c`、暖缓存 `eeaf6991b30ee63b9548ed2426ad3ccb48a20793c936a7106d4a3f27ad366514`。
  - **页隔离与范围**：真实服务测过 Prompt 长度 `1 / 17 / 85 / 533 / 2055 / 3058 / 4096`；同前缀尾行对照及未触碰 KV 审计通过。另用 3800-token 请求交错执行 B4/B8，8 Rank 各记录 434 次页释放和 186 次后续复用，524 步未触碰区逐位不变；长前缀 B8 同前缀逐行对照本次未触发，短前缀 B4/B8 的 96 项对照逐位一致。收据 SHA256 `51dbff7f62b218ab4e6cc8bf17d7c50f8155170fd1a00a5e4f537477a46a9e36`、`10fbba369590722033d9f4bba2cee2e07c06df2407a7b767bb48b500f50a5c39`、`2d9285433a2c593de4f317aa51d53e7340238a085dd6698d8802b6f70675d783`。
  - **单流完整 `engine.step()` A1/B1/B2/A2**：4096 Prompt、128 Decode 步、生产预算 533、候选预算 2048；TTFT p50/mean 为 `2567.699/2568.396`、`2036.368/2036.907`、`2037.933/2037.886`、`2568.111/2568.480 ms`，TPOT p50/mean 为 `11.421/11.444`、`11.077/11.129`、`11.291/11.411`、`11.437/11.476 ms`。候选两臂 p50 均值相对生产 TTFT 下降 `20.67%`、TPOT 下降 `2.15%`；候选 TPOT 两轮均未达到此前 `11.05 ms` 绝对目标，且两轮漂移 `1.94%`。生产有未预热警告，但该次日志缺逐请求时间边界，无法证明警告全部只在预热期。收据 SHA256 `5a3bf406eaef9f7b25b5514d514aaca31b51aa63f1eaa724e37e4ccf4478a87e`。
  - **真实连续补位 B4/B8 对照**：每臂一次预热及三次正式运行，分别处理 8/16 条不同长度请求，每条 Decode 至少 128 步，完成后补入新请求；完整 `engine.step()` 窗口包含补位 Prefill 和尾部排空。B4 的 A1/B1/B2/A2 总解码吞吐为 `178.14 / 189.65 / 190.70 / 178.08 token/s`，批量 TPOT p50/mean 为 `15.221/21.466`、`16.666/20.130`、`16.685/20.018`、`15.237/21.470 ms`，批次墙钟 p50/mean 为 `6204.17/6204.11`、`5651.72/5652.26`、`5656.24/5660.51`、`6202.96/6205.62 ms`；候选吞吐均值提升 `6.78%`，但步骤 p50 均值增加 `9.50%`。B8 的四臂吞吐为 `222.90 / 209.16 / 208.66 / 227.26 token/s`，批量 TPOT p50/mean 为 `20.536/34.186`、`24.835/36.517`、`24.802/36.627`、`20.395/33.528 ms`，批次墙钟 p50/mean 为 `9667.65/9784.12`、`10241.32/10239.07`、`10223.93/10222.99`、`9581.72/9580.32 ms`；候选吞吐均值低 `7.18%`。B8 生产 A2 的 8 条未预热警告处于秒级日志的预热／正式交界，无法证明严格预热门禁通过，故 B8 仅作带此限制的实测对照，不宣称最终验收通过。B4/B8 汇总收据 SHA256 `44602a9f175d419e523c371d8b8f8b7a9f880bdc0122fbcc13a73633db52ccfe`、`ee1f7335940a564a4a4b3279dd41772b27052d13bec2989575447edb45463c02`。
  - **隔离与遗留边界**：各臂客户端、EngineCore 和 8 Worker 退出码均为 0，候选暖缓存未变，测后八卡释放；候选 `ldd` 无旧桥依赖，运行期执行模块与映射隔离检查。生产对照使用仓外隔离副本移除关闭状态的诊断分支和加载计时打印，保留生产计算与图配置（`P-BENCH-07`）。小桶全改 FP32 AllReduce 的 8 Rank 单层 M=1 实测使 L34/L48 p50 分别增加 `18.88% / 18.58%`，已撤回，故小桶仍用 AllGather、大桶用 AllReduce；收据 SHA256 `7d69ba721b75b47d6a5181caea5260aa63a7fff76775df603f60f7dddf94ee20`。采样期间 `hl-smi` 在 B8 加载时报告可见显存 `free=0`，没有足够证据给出整轮峰值余量；通用采样、通用多模型编译与彻底拆分模型执行文件仍未完成。

### 尚未完成与严禁越界宣称的事项
1. **前期单模探路边界与转向封板说明**：
   - Phase 1–4.4 围绕 MiMo-70L 的手工 C++ 构图路径已跑通零旧桥单流服务闭环并完成底层机制（`DeviceRuntime`、`RecipeRegistry`、`PreparedLaunch`、`tpc_math_safe.h`、15 库静态合并、物理隔离门禁）的真机证成；`adaptive-infra` 分支的单流和 B4 连续补位已有整模对照，B8 已测但生产 A2 的预热／正式边界仍有歧义，详见上方实测范围。
   - **自 2026-11-03 起，停止继续扩写 MiMo 专属手写 C++ 模型图**，研发主线全面转向通用 `PrivateUse1` + `FX -> Synapse` 分段编译桥与 5 大参数化算子原语，直接面向 SGLang 全模型生态与 DeepInfer。
2. **通用底座（承接 SGLang 新一代模型族与 DeepInfer）当前真实缺口**：
   - **通用 `PrivateUse1` 范围**：现有块回收、`GraphPool`、已测视图、异步与设备间拷贝和部分图外算子；未支持的冷路径输入直接报错，完整冷路径和标准 `c10d` 多通信域 `ProcessGroup` 尚未完成。历史 MiMo 路径已封存，非本次修改对象；
   - **C++ 透明静态桶 `FX -> Synapse` 编译器仅局部落地**：已有 13 桶的线性、残差、融合归一化、QKV、RoPE 和 MLP 图；旧版曾按 Paged Attention 切分并验证两层前向与 52 层模型，当前版本注意力禁用，不能延用这些整模结论。通用广播、视图、强类型节点和其他算子尚未全覆盖；外部设备索引与元数据校验仍有同步，SGLang 调度器自身元数据已按流异步处理；
   - **通用软硬协同算子原语与新一代模型结构缺口**：旧版 Paged GQA 的 TPC 和全表 MME 路径曾用于完整模型基线；两条旧路径已删除，固定分块注意力及整模重验尚缺，RMSNorm 另有未通过的单算子与连接图样本。前期 `Block-FP8` / `MXFP4` 软核仍带 MiMo 维度硬编码与“零 bias”限制（`P-NUM-09`）。通用 SWA、混合线性注意力（如 `Gated DeltaNet`）、设备端通用采样器、Paged/Sparse FlashMLA + `DeepSelect`、`UE8M0` `MXFP8×MXFP8 / MXFP8×MXFP4` 软硬协同 DeepGEMM（TPC 流式直算 + 48 MB SRAM 在线反量化重缩放喂 MME）、`mHC`（`post` / `sinkhorn`）及 `203 GB` 主机 DRAM `Engram` 稀疏 Gather 异步 H2D 均尚未落地；
   - 本地散落的 20 余个历史实验 worktree 待归并清理。
3. **仓内 `DeepInfer/` 现有 `feat/gaudi2` 分支（`4da1dac` / `6539279`）的真实定性（严禁冒充可用后端）**：
   - 该分支与 `GaudiRT` 零连接，且 `deepinfer/platform.py` 直接导入 `habana_frameworks.torch` 并通过猴子补丁转调 `torch.hpu.*` / `HPUGraph`，**直接违反零旧桥红线**；
   - 其 `deep_gemm` 与 `flash_mla` 实为将 `UE8M0` / FP8 / FP4 在 HBM 上拆包反量化为 FP32 大张量后调 `torch.matmul` / `torch.bmm` 的功能模拟慢路径，`csrc/gaudi_utils.cpp` 的 `cuda_copy_async` 为主机 `std::memcpy` 占位、`get_current_stream()` 硬返回 `0`。**该分支仅可视为接口形状参考，严禁宣称已完成 Gaudi2 适配，底层算子与运行时对接须基于 `GaudiRT` 重做并清退旧桥依赖。**
4. **候选路径严禁混入旧桥（物理隔离门禁）**：
   - 旧桥（`gaudi-pytorch-bridge` / `habana_frameworks` / `PT_HPU_LAZY_MODE=1`）仅允许用于离线真值采集与 ABBA 对照臂（Arm A）独立进程。
   - 候选臂（Arm B）主进程及全部服务框架（SGLang / DeepInfer / vLLM）Worker 子进程必须前置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，C++ 编译产物 `ldd` 不得链接旧桥库、源码不得 `#include` 旧桥头，Python 进程强制校验 `'habana_frameworks' not in sys.modules` 且 `/proc/self/maps` 无旧桥映射；严禁暴露半截接口退回旧桥算子。
5. **单层与整模测量边界不得混用**：
   - `11.284 / 11.306 ms` 与 `10.5119 / 10.6084 ms` 是 70 层整模独立解码步指标，禁止除以 70 充当单层指标；Profiler 诊断中的 `2,384.5 µs` 已证实是跨层共享 Recipe 的重复事件聚合，严禁作为单层基线。
   - 不再使用缺乏统一实验支撑的模糊比例归因，只陈述具体环节实测毫秒/微秒数。

---

## 四、数值验收标准（对齐物理噪声底与主流 GPU 引擎通用分层契约）

所有数值对拍必须在性能计时之外独立运行（禁用平均误差掩盖局部坏点，全张量必须 `finite == true`）：

1. **同算术路径与缓存复用（纯执行器搬迁、地址绑定替换、未改变浮点求值树的合图、`scal/` 发射旁路或定制固件控制流、`RecipeCache` 冷编译 vs 暖加载、服务框架封装 vs 独立 Runner 同分桶同页表）**：
   - 各并行 Rank 对应输出、KV / 线性循环状态写入物理槽位与内容、Router Top-K ID（若含 MoE）、首步全词表 logits 及连续多步贪婪解码 token 序列必须 **逐位一致（bit-exact，`max_abs = 0.0`）**。
2. **改变浮点累加树但保持原数据类型语义（算子融合、MME/SRAM 分块切分、异构设备 vs 外部 CPU FP32/FP64 参考）**：
   - **单算子 / 全模型逐层冻结同输入单层验收（隔离跨层误差与离散路由翻转，对齐 FlashAttention 双基准与 FlashInfer 算子容差）**：
     - **MoE 层离散路由条款**（纯 Dense 层免检本项）：非临界样本（第 $K$ 与第 $K+1$ 专家路由分差显著高于 BF16 舍入粒度，如 $\Delta > 10^{-3}$）的 Router Top-$K$ 专家集合必须与 FP64 参考完全一致；
     - **逐层算术误差条款**（MoE 层在同专家集合下核验）：`CPU_FP32_same_boundary` 仅在算子或模块的外部输入、输出存储边界使用 BF16，内部计算全程纯 FP32，不得加入候选内部的 BF16 概率等近似。单层输出对独立 FP64 参考的逐行 `max_row_relative_l2` 必须满足 **`<= max(3e-3, 2.0 * err(CPU_FP32_same_boundary, CPU_FP64))`** 且绝对上限 `<= 1e-2`，同时报告逐元素 `max_abs`。历史 MiMo 中仅采用 `2×BF16` 概率分解的受控低精度分解层使用专门上限 `<= 3.5e-3`，不适用于本次 Phase 5A。（历史实例：MiMo-70L L34 修正后纯 FP32 同存储边界参考对 FP64 误差 `0.0`，候选 `0.003061274` 按该专门条款通过，收据 SHA256 `019f6742da11a201896031e22afbd62f02efcf462cdaf3965288ebe1274e96aa`。）单层检查不能替代整模质量、状态池与端到端性能验收。
     - **孤立无残差 Paged Attention 算子**：同时满足对独立 CPU FP64 的 `max_row_relative_l2 <= 1e-2`，以及对与设备相同 KV 分块、BF16 舍入分块指数和 BF16 `V`、FP32 在线归约的独立 CPU 参考 `max_row_relative_l2 <= 1e-3`。两个参考及候选均须逐项有限，分别报告误差；该 BF16-PV 参考只用于此孤立算子门槛，绝不混入完整 Decoder 层的纯 FP32 底噪。完整层须含 `o_proj + residual` 并严格遵守上一条；整模条件化 KL 门槛不变。不得增加第二次 `P@V` 来迎合孤立样本。
   - **全模型端到端验收（区分冻结同输入单层、固定在线前缀的滚动条件化末步与空状态长前缀，详见 `P-NUM-06`）**：
     - **输出分布与选词（硬阻断）**：
       - **固定在线前缀末步全模型滚动条件化**：先通过冻结同输入逐层检查；固定同一在线 KV / 线性状态前缀及末 token，独立运行 FP64 与同存储边界纯 FP32 多个合法归约树，并记录同前缀、已通过单层验收的 `M=1` Decode 回放。定义 `ref_floor = max(max_kl_trees(FP64 || CPU_FP32_same_boundary), max_kl(FP64 || Validated_M1_same_prefix))`，候选必须满足 `finite == true`、`max_kl(FP64 || Candidate) <= max(0.01, 1.15 * ref_floor)`，非平票样本 `top1_match == 1.0`；若 FP64 前两名分差小于该幅值一个 BF16 间隔，须命中参考 Top-2。不得借用不同前缀底噪。（历史实例：Phase 4.4 MiMo-70L 树 A/B/C/D 分别为 `0.007758 / 0.006465 / 0.002391 / 0.011029`，同前缀 `M=1` 为 `0.010877`，门限 `0.012683`，候选 `0.011020` 通过；包络收据 SHA256 `4d600285cf05ddac3b1836ec5f35a2f011338841df63e30958963fc6d595d240`，`M=1` 回放 SHA256 `e5fb4619cd8d2de4b530c0b538b3c79b5b0b6f1e7536fc467e3a89709c9cc13c`。Phase 4.2 另一前缀的 `0.02178` 是混合精度参考，不是纯 FP32。）
       - **空状态长前缀端到端状态（硬阻断）**：全词表必须有限；冷编译与暖加载的首步 logits、生成 token 和终态 KV / 线性状态必须逐位一致；连续 `>= 128` 步须通过逐槽位状态写入及未触碰区 `max_abs = 0.0` 审计，所有并行 Rank 每步选词一致且文本连贯无退化。任意动态长度输入在同一在线前缀下，末有效行须与 `M=1` 满足单步数值契约；同算术路径必须逐位一致。禁止为新优化或新长度启动多小时全模型 CPU FP64/FP32 空前缀滚动参考；历史 MiMo-70L 4096 空 KV 独立 FP64 的 KL / Top-5 仅作诊断（第 1 个 MoE 层已有 `5.22e-6` 临界路由翻转，收据 SHA256 `81e8b8cba485cd5028fc84bc5daf259941bacb020723d59aee33d794b601c3cd`；此前将 `0.043453` 误称纯 FP32 已撤回，复核收据 SHA256 `78d449fd87991a36a0ef1ea8da37820b27d25fcf6df2c19b98f1d31d83bf652a`）。
     - **全词表裸 Logits 距离（诊断监控，非跨求值树硬阻断）**：有同输入参考时记录 `relative_l2`、`max_row_relative_l2` 与归一化能量差 `calc_diff = 1 - 2<x,y>/(||x||^2+||y||^2)`；单步跨归约树超出 `0.03` 时排查，不为长前缀另跑多小时参考。
     - **多步状态与重跑确定性（硬阻断）**：连续 `>= 128` 步自回归生成中，各并行 Rank 每步选词必须全卡一致，且开启/关闭逐步审计插桩的生成 token 序列逐位一致；每层每步目标 KV / 线性状态物理槽位写入正确、历史槽位后续步零篡改、未触碰区域全程逐位不变（`max_abs = 0.0`），生成文本连贯无循环退化。
3. **新增低精度量化变更（如 BF16 投影转 FP8 / MXFP4）**：
   - 必须事先写明 scale 粒度与校准契约（静态 block/channel scale 或动态激活量化规则），不得无校准强转。
   - 单层同算子误差通过门禁（`max_row_relative_l2 <= 2e-2`）后，必须在**完整目标模型**上同时通过上述第 2 条全模型分布门禁、`>= 128` 步逐槽位状态审计与多步生成质量验收；任一门禁不过立即回退，禁止事后放宽。

---

## 五、分阶段执行路线

不再扩充外围参考采集或架构综述，**自 Phase 5 起全面终止单模型手写 C++ Synapse 构图，所有新工程一律长在可复利的通用底座与原语主干上**，淘汰过时模型靶子，每阶段绑定现代真实上游模型做端到端验收：

### Phase 1–4（前期底座探路阶段：已完成封板，停止单模定制扩写）

- **Phase 1：真实单层 C++ 数值与性能闭环（已完成）**
  完成真实模型第 1 层（Attention + MoE + TP8 集合通信衔接）8 Rank 逐 stage 验收、同口径 standalone 单层 ABBA 计时（净降约 `36.96%`），以及 MoE 在线 Loader 逐字节等价证明与共享 Recipe 跨层事件归因核实。
- **Phase 2：受限 `PrivateUse1` 独立底座、全层型闭环与同构复用（`B=1, M=1` 范围已完成）**
  切断旧桥头文件与 `HPU` 寄生，落地 `DeviceRuntime` / `HbmArena` / `RecipeRegistry` / `PreparedLaunch` 预构建整数 `tensorId` 零同步绑定表、同构层 `synRecipeHandle` 复用与带构建指纹的 `.recipe` 磁盘缓存。
- **Phase 3：完整 70 层零旧桥独立解码闭环与 ABBA 验收（已完成：`61241ef` / `fd75cd6`）**
  通过纯 CPU `safetensors` 流式加载器与独立底座完成 70 层冻结同输入路由/算术、冷/暖逐位等价、128 步 KV 状态审计及无插桩 4 臂 ABBA 解码胜出（`10.5119 / 10.6084 ms` vs `11.4145 / 11.2448 ms`）。
- **Phase 4：底座防呆收敛、零旧桥 Prefill 与 `vllm-gaudi` 单流服务闭环（4.1–4.4 已按各自范围验证封板）**
  1. **4.1 底座防呆（`502c072` / `2f6a401`）**：落地公共安全数学头 `tpc_math_safe.h`（`P-TPC-01`）、TP8 退出栅栏与销毁顺序封装（`P-RT-06`）、`TORCH_DEVICE_BACKEND_AUTOLOAD=0` 物理隔离（`P-RT-09`）及第 6 步平票核实（`P-RT-16`）。
  2. **4.2 最小分块 Prefill 与空 KV 闭环（`8fc3b3b` / `d23e585`）**：合并 15 库上限内的单一 TPC 算子包（`P-RT-14`），修复 QKV 中点二次舍入（`P-NUM-09`），打通 70 层空 KV `4096 Prompt -> 128 步 Decode` 冷/暖逐位一致闭环。
  3. **4.3 & 4.4 `vllm-gaudi` 单流 `M<=2048` 服务性能闭环（`aabf3ab` / `f127cec` / `2a91c397`）**：基于改造前 Rank 0 插桩拆账（Dense / MoE / SWA / Full-Attention / HCCL 流区间 `1208.8 / 988.4 / 797.2 / 380.3 / 310.7 ms`，插桩墙钟 `3697.2 ms`，区间可重叠禁相加，收据 SHA256 `7fb7ecfa79dd892c294de56606be0e7dd8212d7fc2d91a2b36e5651f57c69722`），打通按专家分桶 MME `batch_gemm` 大桶 Prefill 与真实 `vLLM engine.step()` 服务路径，无插桩 4 臂 ABBA 两轮 p50 均值达 TTFT `2374.282 ms` vs `2570.602 ms`（`-7.64%`）、TPOT `10.970 ms` vs `11.387 ms`（`-3.67%`），全进程 `exit 0` 释放八卡。*（注：`adaptive-infra` 并发小批分支不再单独做 MiMo 整模定制收尾，相关算子修复与分页经验直接并入 Phase 5 通用底座。）*

---

### Phase 5：通用 `PrivateUse1` 运行时、C++ 透明静态桶 `FX -> Synapse` 编译器与 SGLang 现代 Dense 模型族闭环（底座部分完成，现行首要主线）

**验证靶子**：直接采用上游 `sglang` 原生现代模型代码（首发 **`南北阁 33B`（`Nanbeige` 33B）** 等主流 Dense GQA 模型，覆盖 BF16 & Block-FP8、带/不带 QKV bias 与 `TP=1/8`），**严禁为目标模型编写任何专属 C++ 层图代码，严禁在上层 Python 插件塞分桶 Padding 逻辑**。内部严格按两级垂直切片推进：

1. **Phase 5A（进行中：旧版 52 层单卡基线已验证；当前版本所有注意力调用明确报错，固定分块 MME 注意力、数值修复、强类型构图、底座去内耗及重新验收待完成）**：
   - **显存与基础视图底座**：落地 **Stream-ordered Block Caching Allocator**（单张量 `free` 合并回收 + `GraphPool`）、`as_strided` / `view` / `slice` 零拷贝视图、D2D 拷贝、固定主机内存池异步 H2D/D2H 与 `synEvent` 同步，以及二进制指纹 `RecipeCache` 下沉；
   - **C++ 底层 13 档透明静态桶 `FX -> Synapse` 编译器（`BucketDispatcher`）**：实现 `torch.compile(..., dynamic=True)` 单次捕图 $\rightarrow$ C++ 按 `std::bit_ceil(M)` 路由 13 档静态桶 `.recipe`（自动注入 `(subgraph_hash, bucket_M, op_idx)` 唯一张量/节点名）+ `Storage` 物理桶余量与逻辑 `[M, H]` 视图直绑 + 图外冷路径微算子即用即发，首批覆盖 Dense 模型必需的 ~15–20 个 Core ATen 算子；
   - **最小通用 Dense 原语与极薄 `sglang-gaudirt` 单卡闭环**：落地支持 `valid_rows` 硬件截断的固定分块 MME **Paged GQA Attention**（非空前缀 Extend/Prefill + Decode），支持块页、RadixCache 和 Chunked Prefill，通用 `AddRMSNorm` / `RoPE`；重新验证单卡 `TP=1` 上游模型冷暖逐位确定性、第四节数值与无插桩性能。
2. **Phase 5B（横向扩展 `TP=8` 并发、通用量化软硬核与完整服务能力）**：
   - **通信与完整 Core ATen 覆盖**：实现标准 `c10d` `ProcessGroupGaudiRT`（覆盖当前真实消费的单机 `TP / DP` 子通信组，不在本阶段空造无消费者的死分支），补齐完整 ~40 个 Core ATen 降维节点；
   - **去 MiMo 化的通用量化 Linear 与设备端 Sampler**：将 `Block-FP8` 等量化软硬核彻底剥离单模维度与零 bias 限制（封装 `P-NUM-01` IEEE-FP8 安全转换与小 $M$ TPC / 大 $M$ SRAM-MME 双路径），补齐 SWA 变体与**设备端通用 Sampler**（temperature / top-k / top-p / 结构化解码掩码），打通 SGLang `RadixCache` 与 `Chunked Prefill` 高并发混合批；
   - **结项门禁**：`南北阁 33B` 等至少两款异构现代 Dense 模型零模型代码改动跑通 `TP=8` 冷/暖确定性、第四节分层数值门禁、高并发混合批长稳压测（全进程 `exit 0` 零泄漏）与同口径端到端吞吐/延迟验收，成批解锁 SGLang 标准 Dense 模型族。

### Phase 6：混合线性注意力、通用参数化 Fused MoE 与新一代混合架构模型族闭环（待建）

**验证靶子**：上游 `sglang` 新一代混合注意力与高稀疏 MoE 模型——**`Qwen 3.8 Next`**（混合线性注意力 `Gated DeltaNet` + 标准 GQA + 高稀疏 MoE + MTP）与 **`Hy4`（混元 4）**（前置 QK-Norm / 异构注意力变体 + 大规模 MoE），覆盖 BF16 / Block-FP8 / `MXFP4` 及单机 `TP / EP` 组合。
1. **混合线性注意力原语与异构状态池**：实现支持 `valid_rows` 截断的 **`Gated DeltaNet` 线性循环核**（chunkwise Prefill + recurrent Decode）及前置 QK-Norm 等 Attention 变体，在 `sglang-gaudirt` 对接 **Linear Recurrent State + Paged KV 混合状态缓存池**。
2. **通用参数化 `GaudiFusedMoE` 与单机 Scale-Up EP**：将前期 `MXFP4` / `Block-FP8` MoE 软硬协同核彻底通用化（参数化 `num_experts / top_k`、softmax / sigmoid + bias 路由、按专家分桶 Permute、小 $M$ TPC 流式直算与大 $M$ 48 MB SRAM 在线反量化喂 MME Grouped GEMM、FP32 安全指数 SwiGLU），并在 `ProcessGroupGaudiRT` 上打通单机 8 卡 Scale-Up `hcclAlltoAll` Dispatch/Combine，接入 SGLang `MoERunnerBackend`（跨机 Scale-Out `AllToAll` 视多机 RoCE 环境独立扩展，不阻塞单机 8 卡闭环）。
3. **结项门禁**：`Qwen 3.8 Next` 与 `Hy4` 零模型代码改动通过第四节离散路由与混合状态分层数值门禁（冻结同输入专家匹配 + 条件化 KL 包络 + `>= 128` 步 KV/线性状态无串扰审计）及并发服务吞吐验收。

### Phase 7：MLA + `UE8M0` 软硬协同低精度合龙——打通零旧桥 `DeepInfer`（`DeepSeek-V4.1-Flash`）（待建）

**验证靶子**：聚焦契合单机 `8×96 GB` HBM + 主机 DRAM 物理容量账的新一代 **`DeepSeek-V4.1-Flash`**，在 `GaudiRT` 上原生跑通零旧桥 `DeepInfer`（并反哺 SGLang 同源 MLA/`UE8M0` 原语），彻底清退 `DeepInfer/feat/gaudi2` 中的旧桥模拟慢路径。
1. **MLA、SWA Bounded Replay 与 `UE8M0` 软硬协同核心算子**：
   - 实现 **Paged MLA / Sparse FlashMLA**（`head_dim = 576/512`，含 `528B` FP8+UE8M0 与 `288B` FP4 压缩分页 KV 格式）、**SWA bounded replay** 及 **`DeepSelect` 动态变长 Top-K 索引器**；
   - 实现 **`UE8M0` Blockwise `MXFP8×MXFP8` / `MXFP8×MXFP4` DeepGEMM 与 MegaMoE**：遵循硬件物理能力最大化原则——小 $M$ 走 24 核 TPC 寄存器流式解包乘加，大 $M$ 走 TPC 在 48 MB SRAM 内按 Tile 在线解包重缩放（含 `P-NUM-01` OCP-FN 到 Gaudi2 IEEE-FP8 安全幅值转换或受控分量分解）后双缓冲喂满 MME，严禁退回 HBM 全量 FP32 反量化慢路径。
2. **`DeepInfer` 专属原语与 C++20 `zos` 引擎原生对接**：
   - 补齐 4 流残差 **`mHC`（`post` / `sinkhorn` 迭代归一化）** 与 `sqrtsoftplus` 路由门控 TPC 核；
   - 基于 Phase 5 的固定主机内存池与异步 `h2d_stream`，打通 **常驻主机 DRAM 的 `203 GB` `Engram` 词表**按命中 ID 稀疏 Gather 后的锁页小包异步 H2D 与设备计算重叠；
   - 重写 `DeepInfer/csrc/gaudi_utils.cpp` 与 `graph_timer_gaudi.cpp`，将 C++20 `zos` 零开销异步调度器及 `DSpark / NextN` 变长多步投机验证直接绑定至 `GaudiRT`（天然复用 C++ 透明静态桶），删除全部 `habana_frameworks` 导入与 `cuda_migration` 猴子补丁。
3. **结项门禁**：物理隔离校验全绿（零旧桥符号/映射），通过第四节第 2、3 条分层数值与长序列状态门禁，完成真实负载端到端 TTFT / TPOT 与吞吐验收。

### Phase 8：通用负载硬件效率深挖——主机态 `scal/` 发射旁路与受控的片上 ARC 固件增强（后置，按账单两级立项）

在 Phase 5–7 通用服务闭环稳定后，对真实多模型负载（尤其 `DSpark/NextN` 多步投机采样与高稀疏 MoE）拆出“MME/TPC 本体计算 vs 集合通信物理传输 vs `synLaunch` 外壳与设备动态控制泡沫”的无插桩/设备事件账单；对上层编译器、算子库与插件保持 **100% 零侵入**（仅替换 `GaudiRT` 最底层发射实现），按需分两级推进：
1. **Phase 8.1：纯主机态 `scal/` 队列发射旁路（不改片上 `.bin`）**
   针对 `synLaunch` 每次 `22~33 µs` 外壳开销暴露为真实瓶颈的小延迟场景，直接消费 `synGraphCompile` 产出的 `.recipe`，通过 `scal/`（`ccb_writer`）向硬件 CCB 队列直接下发指令包，将主机提交压至微秒级。
2. **Phase 8.2：受控的未签名片上 ARC 固件增强（`sched_*.bin` / `eng_*.bin`）**
   仅针对公开路径证实吃不掉的**设备端动态控制瓶颈**（如 MTP 投机验证片上条件循环 `COND_LOOP`、高稀疏 MoE 空专家片上跳过 `SKIP_IF_ZERO`、计算与片上 `24×100GbE` 网卡硬件 SOB 直连）改造未签名引擎/调度固件（严禁触碰带硬件签名的 `habanalabs-firmware-odm` 板载底层固件）：
   - **隔离与防挂死铁律**：严禁真机盲试引发硬件死锁与整机 8 卡驱动 Reset；必须严格走完“收益账单立项 $\rightarrow$ SDK 同源 x86 模拟器 `lib*_bfm.so` 全边界通过 $\rightarrow$ 固件二进制纳入 `RecipeCache` SHA256 指纹 $\rightarrow$ 进程级 `HABANA_SCAL_BIN_PATH` 独占设备加载（退出即复原，不碰系统目录）$\rightarrow$ 输出相对公开路径 **100% 逐位一致（`max_abs = 0.0`）** 与 ABBA 净收益验收”，无端到端净收益立即撤回。
