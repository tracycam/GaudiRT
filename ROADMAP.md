# ROADMAP.md — 全栈工程设计与阶段路线图

本文是 `Gaudi-LLM` 项目的唯一工程设计与路线总纲，工程纪律、版本卫生与文档规范服从同目录 [`AGENTS.md`](AGENTS.md)，实测硬件、编译器与运行时踩坑总台账统一维护于 [`docs/gaudi2-pitfalls.md`](docs/gaudi2-pitfalls.md)。子仓或历史文档若与本文冲突，一律以本文为准。

---

## 一、总体目标与全栈四层分工

抛弃 21 万行 `gaudi-pytorch-bridge` 的 LazyTensor / JIT 重放遗产，围绕大模型推理构建干净、显式、快速失败（Fail-Fast）的自研软件栈。全栈严格拆为四层，禁止跨层混写或把具体模型常量塞进通用运行时核心：

1. **算子层（`gaudi-kernels-public/`）**
   针对当前 SDK（`1.24.x`）与已测目标负载中缺失或未达标的算子，用 `tpc-clang` 编写高性能融合 TPC 内核（Block-FP8、SWA/Attention、MXFP4 MoE、多核 Norm 等），通过 `GC_KERNEL_PATH` 排在前面按需覆盖；已达标的官方算子不盲目重造。
2. **运行时与编译底座（`GaudiRT/`）**
   只保留与具体模型无关的通用 C++20 机制：设备/流/HCCL 生命周期（`DeviceRuntime`）、显存池（`HbmArena`）、官方 `.recipe` 注册序列化与绑定表（`RecipeRegistry`）、在途信用管理（`InFlightCreditManager`）、带 SHA256 校验的通用权重清单加载（`WeightManifest`），以及仅供后续固件协同使用的底层队列协议（`scal/`）。特定模型的图拼装代码必须与 `gaudirt::runtime` 核心解耦。
3. **桥接前端（待建薄 PyTorch 2 Bridge）**
   基于 PyTorch 2（`PrivateUse1` + `torch.compile` FX 图捕获 + 显式静态 `StepGraph`），把模型子图降维对接到 `GaudiRT` 与自研算子库。候选进程独立自管设备、显存与通信（严禁 `#include` 或链接旧 `gaudi-pytorch-bridge`、严禁加载 `habana_frameworks` 或启用 `PT_HPU_LAZY_MODE`，旧桥仅限独立对照进程）；零 LazyTensor、零层内切碎退回 Python 算子、零逐算子重放扫描、越界输入直接报错终止，不留静默回退（Fallback）。
4. **模型与服务层（`vllm-gaudi/` 及模型 Runner）**
   负责请求调度、KV Cache 物理页管理、采样循环与模型装配。

---

## 二、闭源 SDK 与底层部件处置死规矩

Intel 将离线图编译器与在线发射运行时打包在同一个 `libSynapse.so` 中，讨论与编码必须按职责严格拆开，执行以下固定边界：

| 部件 | 处置决定 | 具体边界与理由（基于当前 SDK `1.24.x` 与单机 8 卡目标负载） |
| :--- | :---: | :--- |
| **1. 官方算子库 (`libtpc_kernels.so`)** | **按实测替换主干** | 对当前目标模型中实测缺失或慢于屋脊线的算子，用 `gaudi-kernels-public` 自研 `.so` 覆盖；不把“官方算子慢”外推成所有算子的永久结论。 |
| **2. 离线图编译器 (`synGraphCompile` / `libSynapseMme.so`)** | **永远不重写** | 任何阶段都不自写 `EcbGraphCompiler` 去替代 MME 切分、SRAM 双缓冲排程与 ECB 打包。只把它当后端汇编器调用；**不预设“每半层只能有一个大 Recipe”**，若合图导致编译器排程倒退则保留切分点，唯一标准是真机耗时变快。 |
| **3. 在线发射器 (`synLaunchWithExternalEventsExt`)** | **Phase 1–4 原样调用；<br/>Phase 5 局部旁路** | Phase 1–4 直接调公开发射接口；仅在 Phase 5 经实测账单证实公开接口吃不掉发射/同步瓶颈时，吃 `synGraphCompile` 编出的 `.recipe` 用 `scal/` 局部旁路下发，绝不演变成重写通用裸加载器。 |
| **4. 集合通信库 (`libhcl.so` / `hccl*`)** | **Phase 1–4 原样调用；<br/>Phase 5 按需局部旁路** | Phase 1–4 直接调公开 `hcclAllReduce` / `hcclAllGather`；Phase 5 若仅改引擎固件做层内动态跳过则不碰它，仅在需要跨计算+通信片上循环时接管对应通信 CCB。 |
| **5. 队列层与内核驱动 (`libscal.so` / `libhl-thunk.so` / `habanalabs.ko`)** | **当前原样复用** | 在当前已测单机 8 卡静态稳态推理路径下不是主要瓶颈，不改内核驱动，不重写 `libscal.so`；若后续跨机或新映射路径出现新证据再重审。 |
| **6. 片上 ARC 固件 (`/opt/habanalabs/engines_fw/*.bin`)** | **Phase 1–4 不动；<br/>Phase 5 定点增强** | 仅针对公开路径与真实服务闭环实测后仍吃不掉的设备端动态控制瓶颈改造未签名的引擎/调度固件；严禁触碰带硬件签名的 `habanalabs-firmware-odm` 板级底层固件。 |

---

## 三、当前真实状态与测量纪律（截至 2026-10-31）

### 已证实的资产与基线
- **`GaudiRT` 底座已清理收敛**：历史探针、模拟基线与大体积证据已移出 Git 归档至 `instance-assets/gaudirt-archive-20261025/`（含 SHA256 清单），仓内仅保留通用底座。
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
  - **空 KV 4096 长前缀 70 层全量 FP64 三角核验完成（候选精度优于纯 CPU FP32）**：① 固定候选在线前缀 KV 重算末 token 70 层，候选对 CPU FP64 的 **`max_kl = 0.00620 <= 0.01`**、Top-1 一致（同条件下 CPU FP32 对 FP64 为 `0.02178`，收据 SHA256：`0d10c115448317bec44c7320dc978e3937d6078555807b682bd9c0855703727a`、`269557b1fe256cc310d1f6f908f5b8de1a1037a7195aafec33ce8f6e98ac1a36`）；② 从空 KV 独立滚完 4096 token × 70 层（复合 `282,624` 次离散 Top-8 路由写入在线 KV），**零硬件参与的纯 CPU FP32 对纯 CPU FP64 自身**即达 **`max_kl = 0.043453`**（`relative_l2 = 0.122038`，翻转 32/69 层，第 8 层起翻），而 **Gaudi2 零旧桥候选对同一份纯 CPU FP64 真值**的 **`max_kl = 0.034032`**（`relative_l2 = 0.128379`，Top-1 与 Top-4 排序 `[576, 1096, 342, 151645]` 100% 一致，仅翻转 28/69 层且前 17 个 MoE 层全匹配），比纯 IEEE CPU FP32 贴近 FP64 达 **`21.7%`**（收据 SHA256：`96276b2a260b5640cb9c3a7f435a9976178a5afbcb3f05e8a30853a71265a839`、`a79278bd63dfe4967f5ec6c9e5db88085a610a1a8b5aad28e5a5699c4179ea53`、`a048ff3d98880ad4845162a43f8cd5dd27dedcfc51c07a1c31b911d0696e5bbd`、`c5c679d889776ea147099538bdf038b223fa2ca2fc6317dd5fe7af8b9867a236`）。

### 尚未完成与严禁越界宣称的事项
1. **严禁把 Phase 3 / Phase 4.2 独立 Runner 成绩冒充为“完整生产服务（第 4 层 `vllm-gaudi`）已收口”**：
   - Phase 3 的 4 臂 ABBA（`10.5119 / 10.6084 ms` vs `11.4145 / 11.2448 ms`）未经过 `vLLM` 调度器与进程间 IPC（口径限定收据 SHA256：`3a1588c0e449fa8df2f3bacc2d860dced6de624322890e7c2308ed77df80d218`）；Phase 4.2 的空 KV `Prefill -> 128 步 Decode` 尚未做无插桩服务级测速，**当前尚无同 `vLLM engine.step()` 口径下的服务级 TTFT 与 TPOT 验收成绩**。
   - **当前唯一主线阻塞项即 Phase 4.3**：接入 `vllm-gaudi` 原生零旧桥 Platform / Worker / ModelRunner，走通真实调度器、分页 KV 块表映射、进程间 IPC 与 `OutputProcessor` 的完整对称 `engine.step()` 闭环及 4 臂 ABBA。多并发 `Batch > 1` 分桶、非 `32` 整数倍变长尾块掩码与通用随机采样留待单流服务闭环达标后按需扩展。
2. **待收敛的工程债与受限契约边界**：
   - 本地散落的 20 余个历史实验 worktree 尚未在 Phase 4 终验后归并清理；`models/mimo/private_use1.cpp` 仍耦合 MiMo 模型常量；
   - `P-NUM-09` 补偿 QKV 中点舍入内核受“零 bias 且准备后 scale 正常数下界”的模型契约约束（流式入口已强校，次正规数残差人工用例不适用，禁宣称通用算子）；
   - 生产对照进程关闭时其框架自带子进程 `SIGTERM` 与共享内存警告，候选在接入 `vLLM` 多进程执行器后必须验证主进程与全部 Worker 子进程干净退出且八卡释放。
3. **候选路径严禁混入旧桥（物理隔离门禁）**：
   - 旧桥（`gaudi-pytorch-bridge` / `habana_frameworks` / `PT_HPU_LAZY_MODE=1`）仅允许用于离线真值采集与 ABBA 对照臂（Arm A）独立进程。
   - 候选臂（Arm B）主进程及全部 `vLLM` Worker 子进程必须前置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，C++ 编译产物 `ldd` 不得链接旧桥库、源码不得 `#include` 旧桥头，Python 进程强制校验 `'habana_frameworks' not in sys.modules` 且 `/proc/self/maps` 无旧桥映射；严禁导入旧 `vllm_gaudi/platform.py` 或暴露半截接口退回旧桥算子。
4. **单层与整模测量边界不得混用**：
   - `11.284 / 11.306 ms` 与 `10.5119 / 10.6084 ms` 是 70 层整模独立解码步指标，禁止除以 70 充当单层指标；Profiler 诊断中的 `2,384.5 µs` 已证实是跨层共享 Recipe 的重复事件聚合，严禁作为单层基线。
   - 不再使用缺乏统一实验支撑的模糊比例归因，只陈述具体环节实测毫秒/微秒数。

---

## 四、数值验收标准（对齐物理噪声底与主流 GPU 引擎分层契约）

所有数值对拍必须在性能计时之外独立运行（禁用平均误差掩盖局部坏点，全张量必须 `finite == true`）：

1. **同算术路径与缓存复用（纯执行器搬迁、地址绑定替换、未改变浮点求值树的合图、`RecipeCache` 冷编译 vs 暖加载、`vLLM` 服务封装 vs 独立 Runner 同分桶同页表）**：
   - 8 Rank 对应输出、KV 写入物理槽位与内容、Router Top-K ID、首步全词表 logits 及连续多步贪婪解码 token 序列必须 **逐位一致（bit-exact，`max_abs = 0.0`）**。
2. **改变浮点累加树但保持原数据类型语义（算子融合、MME/SRAM 分块切分、异构设备 vs 外部 CPU FP32/FP64 参考）**：
   - **单算子 / 冻结同输入单层验收（隔离跨层路由翻转，对齐 FlashAttention 双基准与 FlashInfer 算子容差）**：
     - 非临界样本（第 $K$ 与第 $K+1$ 专家路由分差显著高于 BF16 舍入粒度，如 $\Delta > 10^{-3}$）的 Router Top-$K$ 专家集合必须与 FP64 参考完全一致；
     - 同专家集合下，单层输出对独立 FP64 参考的逐行 `max_row_relative_l2` 必须满足 **`<= max(3e-3, 2.0 * err(CPU_FP32_same_boundary, CPU_FP64))`**（且绝对上限不超过同类 BF16 融合算子 `1e-2` 门禁），同时报告逐元素 `max_abs`。严禁设定低于同精度参考自身重排噪声底（实测单层可达 `2.89e-3`）的伪精度阈值；若出现硬件指令边界异常（如特殊函数分界点失真），必须定位到具体算子修复。
   - **全模型 70 层端到端验收（按滚动离散路由决策规模区分固定前缀单步与空 KV 长前缀两档包络，详见 `P-NUM-06`）**：
     - **输出分布与选词（硬阻断）**：
       - **固定前缀单步 Teacher-Forced（`1 × 69` 次离散 MoE 决策，如冻结前缀 KV 单步解码或固定在线前缀 KV 的条件化末步重算）**：对独立全模型 FP64（或同契约 FP32）参考必须满足 **`finite == true`**、**`max_kl <= 0.01`**，非临界平票样本首步 `top1_match == 1.0`（若 FP64 前两名分差小于该幅值 1 个 BF16 间隔，须命中参考 Top-2）。
       - **全零空 KV 长前缀滚动 Prefill（如 `L = 4096` 复合 `2.8×10^5` 次离散 MoE 决策写入在线 KV）**：鉴于零硬件参与的纯 CPU `FP32 vs FP64` 自身在该长度下已存在 `max_kl = 0.043453`（翻转 32/69 层）的固有轨迹分叉底噪，验收必须先通过单层长上下文同输入 FP64 门禁与上述**固定在线前缀末步条件化 `max_kl <= 0.01`**（排除算子或 KV 读写缺陷），在此前提下全前缀对独立全模型 FP64 参考必须满足 **`finite == true`**、**`max_kl(FP64 || Candidate) <= max(0.01, 1.15 * max_kl(FP64 || CPU_FP32_same_boundary))`**（不得劣于同存储边界 IEEE FP32 参考包络），且非平票样本 `top1_match == 1.0`、Top-1 命中参考 Top-5。
     - **全词表裸 Logits 距离（诊断监控，非跨求值树硬阻断）**：必须同时记录 `relative_l2`、`max_row_relative_l2` 与归一化能量差 `calc_diff = 1 - 2<x,y>/(||x||^2+||y||^2)`。单步跨归约树超出 `0.03` 或长前缀超出 `1.2 * err(CPU_FP32, CPU_FP64)` 时触发异常排查。
     - **多步 KV 状态与重跑确定性（硬阻断）**：连续 `>= 128` 步自回归生成中，8 Rank 每步选词必须全卡一致，且开启/关闭逐步审计插桩的生成 token 序列逐位一致；每层每步目标 KV 物理槽位写入正确、历史槽位后续步零篡改、未触碰 KV 区域全程逐位不变（`max_abs = 0.0`），生成文本连贯无循环退化。
3. **新增低精度量化变更（如 BF16 `o_proj` 转 FP8）**：
   - 必须事先写明 scale 粒度与校准契约（静态 block/channel scale 或动态激活量化规则），不得无校准强转。
   - 单层同算子误差通过门禁（`max_row_relative_l2 <= 2e-2`）后，必须在完整 70 层模型上同时通过上述第 2 条全模型分布门禁、128 步逐槽位 KV 状态审计与多步生成质量验收；任一门禁不过立即回退，禁止事后放宽。

---

## 五、分阶段执行路线

不再扩充外围参考采集或架构综述，严格按以下顺序串行推进，每步只改一个主变量并做 ABBA 验收：

### Phase 1：真实单层 C++ 数值与性能闭环（已完成）
1. 完成真实模型第 1 层（Attention + MoE + TP8 集合通信衔接）的可运行实现，按第四节第 1/2 条数值标准通过 8 Rank 逐 stage 验收。
2. 建立同口径 standalone 单层基线，在关闭 Profiler 条件下做配对 ABBA 计时。
3. **单变量拆分验收**：合图粒度（大 Recipe vs 保留切分）、SRAM 驻留、FP8 量化（如 `o_proj`）逐项独立测试；合图或 SRAM 负收益则立即回退或保留切分，FP8 按第四节第 3 条独立校准并留待全模型多步验证。
4. **结项门禁（Exit Gate，非可选补作业）**：补齐 **MoE 权重与生产 Loader 逐字节等价证明** 及 **共享 Recipe 跨层事件归属核实**。这两项未完成前不阻塞研发编码推进，但**严禁宣称“生产单层已闭环”或“单层性能归因已完成”**。

### Phase 2：独立薄 PyTorch 2 Bridge、全层型 C++ 闭环与同构复用（`B=1, M=1` 解码范围已完成）
1. **彻底切断旧桥依赖，落地独立 `PrivateUse1` + `torch.compile(..., fullgraph=True)` 薄前端**：删掉对旧桥头文件的 `#include`、链接与 `HPU` 分发寄生，由 `GaudiRT` 独立自管设备、流、`HbmArena` 与 `hccl`；把 `mimo_*` 模型特定代码从 `include/gaudirt/runtime/` 解耦；全部三种异构层（含 `layers.0` 图内 SwiGLU 与 10 个 Full-Attention 层的 C++ 图/算子）必须在 `GaudiRT` + 自研算子内完整闭环，严禁层内切碎退回 Python 调旧桥算子。
2. 打通初始化期预注册整数 `tensorId` 零同步绑定表（严禁运行期切层调 `synchronize_compute`）、同构层 `synRecipeHandle` 复用与 `.recipe` 磁盘缓存；支持集合之外的输入直接报错，不做静默回退。

### Phase 3：完整 70 层零旧桥独立解码闭环与 ABBA 验收（已完成：`61241ef` / `fd75cd6`）
1. 通过独立薄 Bridge 驱动完整 70 层 TP8 模型，使用**不依赖 `habana_frameworks` 的纯 CPU `safetensors` 按 Rank 流式切片/打包加载器**直接灌入 `GaudiRT` `HbmArena`（遵守已证实字节等价的 FP8 / MXFP4 规则），严禁拉起旧桥进程借权重。初始化显式提供覆盖权重、10 层 Full KV、60 层 SWA KV 与 workspace 的单卡 HBM 预算；70 层及 Embedding、Final RMSNorm、TP8 词表切分 `lm_head` 的权重在 CPU 内存中切片/打包后直接 H2D，禁止生成全量 `.bin` 副本。
2. 完成 70 层构建指纹 `RecipeCache` 冷编译/暖加载逐位等价验收（第四节第 1 条），按第四节第 2 条完成 70 层冻结同输入路由/算术与连续 128 步状态验收，并在无插桩同机器策略下与独立进程运行的旧桥生产基线完成 4 臂（A-B-B-A）独立解码 + 客户端输出处理 TPOT 验收（`10.5119 / 10.6084 ms` vs `11.4145 / 11.2448 ms`）。

### Phase 4：底座防呆收敛、零旧桥 Prefill 与 `vllm-gaudi` 完整服务闭环（4.1、4.2 已完成；当前冲刺 4.3）
1. **里程碑 4.1：底座防呆与工程清理（已完成：`502c072` / `2f6a401`）**：
   - 将已在三类算子重复实现的 TPC `v_exp_f32` 分界点安全指数修正下沉为 `gaudi-kernels` 公共安全数学头文件 `tpc_math_safe.h`（750 点边界单测与逐位回归通过）；
   - 将多 Rank 退出栅栏、流同步、编译图与 `PrivateUse1` 设备张量先于 `DeviceRuntime` 销毁的顺序封装进 C++/Runner 生命周期方法，强制 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 物理隔离旧桥；
   - 将质量评测脚本同步至第四节校准契约，查明第 6 步选词分叉系 `0.0830` logit（小于 BF16 `0.125` 间隔）平票区在生产整步图下的并列选词差异（`P-RT-16`）。
2. **里程碑 4.2：零旧桥最小 Prefill（`M=32` 静态分桶）与空 KV 端到端自洽闭环（已完成：`8fc3b3b` / `d23e585`）**：
   - 合并 15 库上限内的自研 TPC 算子包（`P-RT-14`），修复 QKV BF16 中点二次舍入（`P-NUM-09`），打通不加载旧桥、不借离线 KV 快照的 70 层 `M=32` 分块 Prefill 图与分页 KV 写入；
   - 验证从空 KV 接收 4096-token 真实 Prompt 跑通 `Prefill -> 128 步 Decode`，冷/暖 20 张 `RecipeCache` 逐位一致、128 块 + 128 步 KV 槽位零越界，条件化末步 `max_kl = 0.00620 <= 0.01`、全前缀相对独立 70 层 FP64 `max_kl = 0.034032 < 0.043453`（优于同契约纯 CPU FP32 参考）。
3. **里程碑 4.3（当前执行）：接入第 4 层（`vllm-gaudi` 原生零旧桥 Platform / Worker / ModelRunner）与严格同 `engine.step()` 口径 4 臂 ABBA**：
   - 在独立干净 worktree 上将零旧桥后端（`GaudiRT` + 自研算子包）接入真实 `vLLM` V1 引擎（`LLMEngine` / `EngineCore` 调度器、分页 KV 块表分配、`MultiprocExecutor` 进程间 IPC 与 `OutputProcessor`），主进程与全部 8 个 Worker 子进程彻底物理隔离 `habana_frameworks`（绕开旧 `vllm_gaudi/platform.py` 导入）；
   - 验证通过 `vLLM` 服务入口跑出的 4096 Prompt 首步 logits、128 步解码 token 序列与终态 KV 与里程碑 4.2 独立 Runner **100% 逐位一致（`max_abs = 0.0`，第四节第 1 条）**，且主/子进程优雅关闭全 `exit 0`、无残留共享内存泄漏、八卡完全释放；
   - 在完全对称的完整服务调用链（`vLLM` `engine.step()` / `LLM.generate` 同计时边界、同主机绑核策略、零插桩）下完成**单流 `4096-token TTFT` + `128-step 稳态 TPOT` 的 4 臂（A1-B1-B2-A2）ABBA 终验**，用实测数据把 `same_complete_production_serving_path_proven` 翻转为 `true`。单流服务闭环达标后，再按需扩展变长尾块 padding 与有限静态 `Batch > 1` 分桶。

### Phase 5：单步硬件效率深挖与受控的 ARC 固件 / `scal/` 队列增强
1. **先拆清真机微秒账单，用尽公开接口空间再动底层**：对稳态单步耗时（当前 `~10.56 ms` vs 纯有效 HBM 带宽底线 `~2.7 ms`）拆出“MME/TPC 算子本体计算 vs `141` 次 `hcclAllGather` 通信 vs `140+` 次 `synLaunch` 跨图同步间隙”的实测账单；属算子利用率或公开合图可解的先在算子/图内解决，**仅针对公开 `synGraphCompile + synLaunch + hccl` 证实无法消除的设备端动态控制或队列提交瓶颈立项**。
2. 严格走完“定位瓶颈与收益账单 $\rightarrow$ `lib*_bfm.so` 同源 x86 仿真通过 $\rightarrow$ 独立恢复方案与审批 $\rightarrow$ `HABANA_SCAL_BIN_PATH` 独占设备 ABBA 验收”闭环；无端到端净收益则撤回。
