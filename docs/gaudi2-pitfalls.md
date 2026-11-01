# Gaudi2 硬件、编译器、数值与运行时踩坑总台账

本文是 `Gaudi-LLM` 项目的**唯一踩坑总台账**（服从 [`../AGENTS.md`](../AGENTS.md) 第五节记坑纪律；现行架构与阶段状态以 [`../ROADMAP.md`](../ROADMAP.md) 为准）。
每条记录严格区分 **`[已证实]`** 与 **`[待测]`**，按 **「现象与触发条件 $\rightarrow$ 真实根因 $\rightarrow$ 硬性铁律解法 $\rightarrow$ 证据落点」** 四要素归档。凡新发现底层坑点，修复交付时必须同步更新本表，并优先下沉为公共代码防呆护栏。

---

## 一、TPC 硬件算术、指令集与编译器坑（`P-TPC`）

### `P-TPC-01`：`v_exp_f32` 在分界点处的硬件算术失真 `[已证实]`
- **现象与触发**：单层同输入同专家集合下，Layer 44 Rank 7 的 MoE 门控激活相对 FP64 误差异常飙至 `0.001311`（其余 Rank 贴近 FP64）；单卡掩码 Softmax 在特定负输入处的最大绝对概率误差高达 **`0.166667`**。
- **真实根因**：Gaudi2 TPC 的 `v_exp_f32` 指令在区间约化分界点存在硬件算术缺陷——例如对 `x = -ln(2)` 返回 **`1.0`** 而非 **`0.5`**。凡直接调用裸 `v_exp_f32` 计算 SiLU/Sigmoid 或 Softmax 的算子均受波及。
- **硬性铁律解法**：严禁在算子中裸调单次 `v_exp_f32`；必须采用经边界验证的二次指数分解修正（已在 MoE FP32 Gate、Full-Attention Softmax 与 SWA Softmax 三处实测将单卡最大概率误差压至 `3.39e-6`、Rank 7 MoE 误差压至 `6.41e-5`），并抽成全算子库统一复用的公共安全数学头文件（配 750+ 分界邻域回归单测）。
- **证据落点**：`gaudi-kernels` 公共头 `csrc/tpc/common/tpc_math_safe.h`（提交 `7fd1f90`）及 750 点设备回归 `tests/tpc_safe_exp_boundary.cpp` / `tools/run_tpc_safe_exp_boundary.py`（提交 `502c072`）；750 个边界样本与三种算子单卡数学快筛收据 SHA256：`d10ab28752b4b30be1696087f5b2d841aad5a65b6fa0dec95bf94c883d72ba3d`。三种新旧内核编译 `.text` 和单卡测试的完整输出张量均逐位一致（`max_abs=0.0`），收据 SHA256：`0765f89bfa6f35204d3f41549c641748dd53481f6cc052fc0874530cfaa66d33`。此前收据：`e40cd3a74f785f4d5c5dfd0dd640fefd8b3ad2e6ff076aec18da48686461ba26`、`286c537925034796eeeb6ce159e58f7fa5264f0a5408af6fc97c8a05f9416fd4`。

### `P-TPC-02`：标量 `SLM` 与向量 `VLM` 地址空间物理隔离导致 `-O2` 删加载 `[已证实]`
- **现象与触发**：标量解码后写入本地内存、再用向量加载 `v_f32_ld_l_v_b` 读出时数据恒为 `0.0`；开启 `-O2` 后编译器直接删除上游 packed/scale 向量加载指令。
- **真实根因**：TPC 的标量本地内存（`SLM`）与向量本地内存（`VLM`）是两块物理不通的独立地址空间。往 `SLM` 写再从 `VLM` 读在数据流上无有效消费者，触发编译器合法的死代码消除（DCE），并非编译器“乱丢指令”。
- **硬性铁律解法**：严禁用 `SLM <-> VLM` 往返搬运数据；跨标量/向量路径一律走文档化的谓词寄存器搬运（predicated scalar-to-vector move）或纯向量流水线，并用冲激输入 + `tpc-llvm-objdump` 验证 `LD_G -> MAC` 完整消费链。
- **证据落点**：`instance-assets/single-card/20260923T081629Z/tpc/`、`20260923T083422Z/tpc/`。

### `P-TPC-03`：启用 LUT / SFU 后可用 `VLM` 容量腰斩至 `16 KiB` `[已证实]`
- **现象与触发**：编译参数带 `-vlm 32` 可正常编过，但上真机运行 `24 KiB` activation tile 时直接触发设备侧异常或主机侧内存报错崩溃。
- **真实根因**：TPC 硬件在启用特殊函数单元（SFU）/ 查表（LUT）模式时，会将硬件表空间切走一半 `VLM`，实际可用 `VLM` 上限降为 **`16 KiB`**。
- **硬性铁律解法**：凡使用 SFU/LUT 的 TPC 内核一律按 **`<= 16 KiB`**（`-vlm 16`）硬约束规划本地向量工作集，禁止假设 32 KiB 常驻可用。
- **证据落点**：`instance-assets/kernel-overheads-20260927/`。

### `P-TPC-04`：同一 VLIW 指令包内两个大立即数互相覆盖编码 `[已证实]`
- **现象与触发**：汇编/编译零报错，但运行期循环步长错乱（如 `add I5, 128` 实际执行效果变成 `+32768`）。
- **真实根因**：Gaudi2 VLIW 指令包（bundle）的共享立即数槽位有限，同一拍内塞入两个大立即数时编码位域发生覆盖。
- **硬性铁律解法**：循环步长与大偏移常量在循环外预装入标量寄存器，手写或关键内联汇编改动后必须用 `tpc-llvm-objdump --triple=tpc --mcpu=gaudi2` 核对反汇编字面值。
- **证据落点**：`instance-assets/kernel-first-principles-20260927/`。

### `P-TPC-05`：`-O2` 将冷分支 `if-convert` 成主路径谓词指令 `[已证实]`
- **现象与触发**：在循环内写极低概率触发的边界兜底分支（fallback），结果正常路径稳态 VPU/SPU 指令数与耗时显著上升。
- **真实根因**：TPC `-O2` 默认对分支做 `if-conversion`（无 `jmpr` 硬件跳转跳过），冷分支被转为带谓词指令每拍照常发射，白占 VPU 槽位。
- **硬性铁律解法**：热循环内部禁塞分支兜底；将整块主循环与尾块（tail）在循环外显式拆成两段无分支流水。
- **证据落点**：`instance-assets/kernel-overheads-20260927/`。

### `P-TPC-06`：MXFP4 裸 `DATA_PACKED_MXFP4` 不做硬件译码，nibble 解包需两步对齐 `[已证实]`
- **现象与触发**：直接用 typed tensor descriptor 加载 `DATA_PACKED_MXFP4` 读出的仍是原始打包字节；仅做 `UNPCK_4_TO_8` 时 512 个有序码出现奇偶通道错位。
- **真实根因**：`DATA_PACKED_MXFP4` 仅为主机侧张量标记，硬件加载管线不自动做 E2M1/E8M0 浮点反量化；load-pipe `4->8` 解包后必须配合 `convert.u4 all_lanes` 才能对齐奇偶 nibble 通道，F32->BF16 双半交织置换需在离线打包阶段逆向吸收。
- **硬性铁律解法**：按 `U8` 原始字节加载，遵循 `UNPCK_4_TO_8 -> convert.u4` 或经全 512 码对拍的成对 256-entry LUT 译码；权重静态置换一律在离线/加载期吸收，不留给在线解码核。
- **证据落点**：`docs/mxfp4-verdict.md`、`instance-assets/mimo-layer1-20261026/moe-loader-equivalence-v1/`。

### `P-TPC-07`：LUT 基址与函数 ID 错配踩零地址，及 `LOOKUP_AND_MAC` 语义误读 `[已证实]`
- **现象与触发**：使用 LUT 时触发设备访问地址 `0` 错误；或误把 `LOOKUP_AND_MAC` 当向量点积使用。
- **真实根因**：`set_lut_32` 配 32-entry 基址却调用 256-entry 函数 ID 会越界踩零；256-entry 成对表须配 `set_lut_256`（base `0xCD8/0xCDC`，表项为连续 16-bit halfword）；而 `LOOKUP_AND_MAC` 硬件语义是分段二次多项式 `C0 + x*(C1 + C2*x)`，并非线性点积。
- **硬性铁律解法**：使用 256-entry LUT 前必须先跑满全 256 键单测验证基址与函数 ID 配对；不臆猜未验证 intrinsic 名字语义。
- **证据落点**：`instance-assets/kernel-first-principles-20260927/`。

### `P-TPC-08`：`shuffle` 仅限自身 16-word 组内且 `bit7` 为通道使能位 `[已证实]`
- **现象与触发**：调用 `v_u8_shuffle_b` / `u32 shuffle` 输出全零，或跨组收集（gather）数据错乱。
- **真实根因**：控制向量方向字节的 **`bit7` 是输出使能位**（`bit7=0` 且 `income=0` 时直接填零），且硬件映射范围严格限制在各自的 **16-word 组内**，不支持跨组任意重排。
- **硬性铁律解法**：组内重排必须置位 `bit7`；跨组重排改走 `unpack + convert` 或双输入置换指令，上线前跑全通道冲激映射单测。
- **证据落点**：`instance-assets/kernel-first-principles-20260927/`。

### `P-TPC-09`：Index-Space 合并 Tile 漏遍历 `get_index_space_size()` `[已证实]`
- **现象与触发**：输出张量预填哨兵值（sentinel）后，跑完内核发现大量输出行仍保留哨兵未更新。
- **真实根因**：图编译器可能合并相邻逻辑 tile，`get_index_space_offset()` 只是当前核分到区间的起点，单次调用的跨度由 `get_index_space_size()` 给出；且逻辑 128 tiles 不自动保证跑满全部 24 个物理 TPC 核。
- **硬性铁律解法**：内核外层必须完整循环遍历 `[offset, offset + size)`；单测输出缓冲区必须预填非零/NaN 哨兵校验全覆盖，并用活跃核计数确认 24 核负载均衡。
- **证据落点**：`MANIFEST.md`、`instance-assets/kernel-coverage-20260926/`。

### `P-TPC-10`：TPC 反汇编与手册工具链误导 `[已证实]`
- **现象与触发**：普通 GNU `objdump` 把 TPC ELF 反汇编成乱码 x86 助记符；临时 `_slots.o` 报 `<unknown>`；`man tpc-clang` 声称 `-march` 只支持 `dali`（初代 Goya）。
- **真实根因**：`habanalabs-tools` 随包 man 手册严重过期；反汇编必须走专用 LLVM 后端并显式指定 Gaudi2 目标与含 `.tpc_metadata` 的对象文件。
- **硬性铁律解法**：编译固定传 `-mcpu=gaudi2 -O2`，反汇编固定用 `tpc-llvm-objdump --triple=tpc --mcpu=gaudi2`，内建函数原型一律以 `tpc-clang -E` 实提的 `tpc-defs.h` / `tpc-special.h` 为准。
- **证据落点**：[`../MANIFEST.md`](../MANIFEST.md) 第 3–4 节。

### `P-TPC-11`：静态 VPU 指令条数大幅下降但真机墙钟几乎不动 `[已证实]`
- **现象与触发**：静态反汇编显示 VPU 指令减少 53%，真机算子耗时仅降 8%。
- **真实根因**：小 `M` 解码核多受限于 LSU 访存延迟、长依赖链 stall 或核心数未打满，静态指令数不等于运行期关键路径周期。
- **硬性铁律解法**：禁止拿反汇编指令行数报喜；唯一裁判是真机同形状设备事件与端到端墙钟。
- **证据落点**：`instance-assets/kernel-overheads-20260927/`。

---

## 二、量化格式、浮点舍入与数值对拍坑（`P-NUM`）

### `P-NUM-01`：Gaudi2 硬件 FP8 `E4M3` 为 IEEE-Specials，与 Checkpoint `OCP-FN` 同名不同义 `[已证实]`
- **现象与触发**：对 OCP `float8_e4m3fn` 权重直接调设备侧原生类型转换时，顶带数值（`0x78..0x7e` / `0xf8..0xfe`）直接炸出 `Inf` / `NaN`。
- **真实根因**：Gaudi2 硬件 FP8 E4M3 遵循 IEEE 特殊值编码（指数全 1 含多个非有限码，有限饱和顶码为 `0x77 / 0xf7`），而主流训练产出的 OCP `e4m3fn` 仅 `0x7f / 0xff` 为 NaN、其余皆为合法有限大数。
- **硬性铁律解法**：权重加载期（CPU/离线）必须按全 256 码表显式完成 OCP-FN 到硬件安全表示的解码/映射；在线激活若进原生 FP8 MME 必须先卡安全幅值带，严禁在解码热循环里为避坑塞通用查表慢分支。
- **证据落点**：[`mxfp4-verdict.md`](mxfp4-verdict.md)、`gaudi-kernels/python/gaudi_kernels/block_fp8.py`。

### `P-NUM-02`：MXFP4 `E8M0` `byte0 = 2^-127` 与 32 元素分组边界 `[已证实]`
- **现象与触发**：小幅值输入出现精确 50% 相对误差；或均匀 scale 测试全过、换成真实非均匀 scale 后每隔 16/32 列成片算错。
- **真实根因**：OCP MX `E8M0` 无符号指数定义为 `2^(b - 127)`，故 `byte = 0` 是 `2^-127`（FP32 位型 `0x00400000`）而非 `2^-128`；且每 **32 个 K 元素**（16 个打包字节）共享 1 个 `E8M0` 字节。用全常数 scale 做单测会完全掩盖分组步长错位。
- **硬性铁律解法**：单测必须包含非均匀奇偶跳变 scale 与 `x = 2^120` 放大探针（隔离 subnormal 冲零干扰），全 256 个 `E8M0` 码值逐一校验。
- **证据落点**：[`mxfp4-verdict.md`](mxfp4-verdict.md)、`instance-assets/p2-mxfp4-salvage-20260923/`。

### `P-NUM-03`：设备 FP32 Subnormal 整结果归零（Flush-to-Zero）`[已证实]`
- **现象与触发**：当完整浮点结果落入 FP32 非正规数区间（`< 2^-126`）时，CPU FP32/FP64 参考为非零极小值，Gaudi2 设备输出精确为 `0.0`。
- **真实根因**：硬件执行单元对非正规结果做 Flush-to-Zero。
- **硬性铁律解法**：仅允许对“整个数学输出本身处于 FP32 subnormal 区间”的样本做具名豁免；严禁借此放宽正常量级输出的误差阈值。
- **证据落点**：[`mxfp4-verdict.md`](mxfp4-verdict.md)。

### `P-NUM-04`：PyTorch CPU `FP64 -> BF16` 经 `FP32` 中转的双重舍入假阳性 `[已证实]`
- **现象与触发**：拿 PyTorch CPU `tensor_fp64.to(torch.bfloat16)` 当“黄金真值”对拍时，某个距离 BF16 中点 `1.28e-8` 的 QKV 样本与设备输出差 1 ULP，误判为设备算错。
- **真实根因**：当前环境下 PyTorch 的 `float64 -> bfloat16` 先舍入到 `binary32` 再舍入到 `bfloat16`（Double Rounding），在临界中点处会选到与 IEEE 直接 `binary64 -> BF16` 最近偶数舍入相反的邻居，而设备选的才是直接舍入正确值。
- **硬性铁律解法**：构建高精度 CPU 参考流时，从 `FP64` 落 `BF16` 层间契约必须使用 IEEE 直接位级最近偶数舍入函数，禁止直接调 `.to(torch.bfloat16)` 充当裁判。
- **证据落点**：`GaudiRT` `docs/mimo-numeric-correction.md`；收据 SHA256：`d30f789b4b234dc19c281c472196775da1b5d3db08e833b9ab7efe3`。

### `P-NUM-05`：融合 RMSNorm 残差提前舍入 BF16 与 SWA Value 缩放精度丢失 `[已证实]`
- **现象与触发**：早期融合 `grid24` Norm 把 `hidden + residual` 先舍入成 BF16 再算均方根，以及 SWA Value 写入前用 BF16 做缩放，导致写入 KV 的相对误差升至 `0.00157 ~ 0.00387`，70 层累积后首步 KL 恶化。
- **真实根因**：Transformer 数学契约与 GPU FlashInfer 标杆一致要求残差相加后的归一化方差归约及 Value scaling 在 FP32 寄存器内完成后再落 BF16 存储。
- **硬性铁律解法**：层边界残差存储为 BF16，但算子内部 `residual_add + RMSNorm` 累加与 QKV Value 缩放一律保持 FP32 计算后再写回 BF16 KV。
- **证据落点**：收据 SHA256：`1a53142c52377f3eb54f12452cfbb7c78f7ba43e1dff8700f8e9a3996611e2d1`、`350e5e4306c15215aec7961a937dbb707b2dcb57f1d482a0781a675de2a59c1c`。

### `P-NUM-06`：70 层离散 Top-K MoE 跨求值树的固有路由翻转底噪（单步与长前缀两档包络）`[已证实]`
- **现象与触发**：① 固定相同 4096 前缀 KV 做单步 70 层解码时，首步全词表裸 logits 对外部 CPU FP32 参考的 `relative_l2` 卡在 `0.02057`，压不进拍脑袋定的 `0.02` 线；② 从全零空 KV 独立滚完 4096-token Prompt × 70 层 Prefill 时，全部单算子与长上下文单层均对齐 FP64，但全前缀首步 logits 对混合 CPU FP32 参考的 `max_kl` 升至 `0.04392`，超出固定前缀单步的 `0.01` 门禁。
- **真实根因**：384 选 8 的离散 Top-K 路由存在阶跃不连续性，且随滚动 token 数呈复利放大：
  - **固定前缀单步（仅 `1 × 69 = 69` 次路由决策）**：在同一 4096 前缀 KV 与 BF16 层边界下，**纯 CPU FP32 对纯 CPU FP64 自身**的首步 `relative_l2 = 0.024443`、`max_kl = 0.003246`，翻转 **16/69** 个 MoE 层；固定候选在线前缀 KV 只重算末 token 70 层时，候选对 FP64 `max_kl = 0.00620 <= 0.01`，而 CPU FP32 对 FP64 已达 `0.02178`。
  - **全零空 KV 4096 长前缀滚动（`4096 × 69 = 282,624` 次路由决策写入在线 KV）**：任意前缀 token 发生 `1e-6` 级临界专家翻转都会写入 KV Cache，经因果 Attention 级联扰动后续数千位置与更深层。实测同一 4096 Prompt 下，**零硬件参与的纯 CPU FP32 对纯 CPU FP64 自身**的 `max_kl = 0.043453`（`relative_l2 = 0.122038`，末 token 翻转 **32/69** 层，第 8 层即在 `< 1e-6` 分差处先翻）；而 **Gaudi2 零旧桥候选（`8fc3b3b`/`d23e585`）对纯 CPU FP64** 的 **`max_kl = 0.034032`**（`relative_l2 = 0.128379`，Top-1 与 Top-4 排序 `[576, 1096, 342, 151645]` 100% 一致，仅翻转 **28/69** 层且 Layer 1～17 全匹配），比纯 CPU FP32 贴近 FP64 达 `21.7%`。对长前缀跨归约树死套 `0.01` 等于要求 BF16/FP8 硬件比 IEEE CPU FP32 高 4.35 倍精度，违反物理下限。生产与候选第 6 步平票证据另见 `P-RT-16`。
- **硬性铁律解法**：严格执行 [`../ROADMAP.md`](../ROADMAP.md) 第四节分层标准——① 同算术路径/冷热缓存必须逐位一致（`max_abs = 0.0`）；② 跨求值树单层（含长上下文在线 KV）必须**冻结同输入**核验路由匹配 + 同专家相对误差低于物理门槛；③ 固定前缀单步（及空 KV 前缀条件化末步）70 层硬阻断为 `max_kl <= 0.01`；④ 空 KV 长前缀全量滚动硬阻断要求在单层与条件化末步 `<= 0.01` 全过的前提下，`max_kl(FP64 || Candidate) <= max(0.01, 1.15 * max_kl(FP64 || CPU_FP32_same_boundary))` 且 `finite`、`top1_match` 与逐槽位 KV 零越界全部通过。
- **证据落点**：单步收据 SHA256：`befaaaf5c5656ddd15adbe68a7b6a99a4d059b2034bb3f42a896b8a1f7055cf9`、`36e8472ddf295499288e01c2ce124ae3737ac50c828c4b925e2fdf71989ed6a5`、`ed12ae92fe1655d33144e1fb79cd7b2a234723f13191af4679f92f977fa79968`；条件化末步收据 SHA256：`0d10c115448317bec44c7320dc978e3937d6078555807b682bd9c0855703727a`、`269557b1fe256cc310d1f6f908f5b8de1a1037a7195aafec33ce8f6e98ac1a36`；4096 全前缀空 KV FP64 参考、候选对 FP64 及 FP32 对 FP64 收据 SHA256：`96276b2a260b5640cb9c3a7f435a9976178a5afbcb3f05e8a30853a71265a839`、`a79278bd63dfe4967f5ec6c9e5db88085a610a1a8b5aad28e5a5699c4179ea53`、`a048ff3d98880ad4845162a43f8cd5dd27dedcfc51c07a1c31b911d0696e5bbd`。

### `P-NUM-07`：MoE 重复专家 ID 坍缩与 `0 * Inf/NaN` 掩码污染 `[已证实]`
- **现象与触发**：随机路由单测含同行重复专家 ID 时，官方 MoE 算子误差飙到 `60%+`；或对未命中槽位做 `* 0.0` 掩码、对全掩码注意力块做 `max` 减法时炸出 `NaN`。
- **真实根因**：官方算子对同行重复专家采取 `last-slot-wins`（覆盖而非累加）；IEEE 浮点中 `0.0 * (±Inf / NaN) = NaN`、`-inf - (-inf) = NaN`。
- **硬性铁律解法**：Router Top-K 输出保证专家 ID 互异（若有填充槽位须用合法唯一虚拟 ID 并走条件选择掩码 `select`，严禁用乘零掩盖未初始化内存）；全掩码注意力块显式归零贡献，Attention Sink 只计入一次。
- **证据落点**：[`mxfp4-verdict.md`](mxfp4-verdict.md) 第 3.3 节。

### `P-NUM-08`：MiMo QKV 分段 `128×128` Block-FP8 Scale 对齐与 `weight_scale_inv` 符号陷阱 `[已证实]`
- **现象与触发**：把融合 `qkv_proj`（`[27136, 6144]`）当成全局 `[Q, K, V]` 切分或按名字对 `weight_scale_inv` 做除法时，权重幅值暴涨至 `1e5` 或跨段 scale 错位。
- **真实根因**：Checkpoint 中 `qkv_proj` 按 TP8 预排为 8 个 `[Q_i, K_i, V_i]` 块（每块 `3392` 行 = `26 * 128 + 64`，对应 `27` 行 `128×128` 块缩放因子，8 块合计 `216` 行 scale）；且名为 `weight_scale_inv` 的 F32 张量实测值域为 `1e-4` 量级，语义是**乘法解量化因子 `w = w_q * scale`**，而非除数。
- **硬性铁律解法**：按每 Rank 独立的 `[3392, 6144]` 与 `[27, 48]` 切片加载并做乘法解量化，严禁拼成全局段重切或做除法。
- **证据落点**：[`mxfp4-verdict.md`](mxfp4-verdict.md) 第 2 节、`ROADMAP.md` 第三节。

---

### `P-NUM-09`：QKV 归约落在 BF16 中点时丢失残差造成二次舍入 `[已证实，修复限模型契约]`
- **现象与触发**：真实 Prompt 的 Type A 第三行有一个 QKV 元素落在 BF16 中点附近；旧图选错相邻值后，单层相对 FP64 的行误差放大到 `0.004888`。
- **真实根因**：Neumaier 的 FP32 累加值与补偿值先合并为 FP32，再转 BF16 时，前一次舍入可能恰好落在 BF16 中点并抹掉真实数值所在的一侧；该样本的 FP64 值低于中点 `3.740e-8`。人工构造的 FP32 次正规数残差设备用例仍失败，具体运算环节未单独判定。
- **硬性铁律解法**：模型 QKV 归约在 BF16 中点保留并使用补偿残差；该实现仅用于零 bias、准备后 scale 有下界的模型契约，不作为任意次正规数输入的通用算子。流式权重入口校验 scale 与 bias；直接提供设备权重的调用者须遵守同一契约。变更后必须按真实输入、独立数学参考和全模型质量门禁验证。
- **证据落点**：`gaudi-kernels` `csrc/tpc/block_fp8_reduce_isa/round_bf16_compensated.c`（`d23e585`）；流式入口检查见 `GaudiRT` `models/mimo/private_use1.cpp`（`8fc3b3b`）。目标模型全部 725760 个准备后 QKV scale 的最小值为 `1.8687e-6`（收据 SHA256：`0ad93ae7bd1f94e6196243172dc08cb0b683cd0b2d76fbcb907a48d930c164e8`）；32 行、8 Rank 独立 FP64 对拍收据 SHA256：`0c1da8bac37b7394edcf526b531dda3496172e2604f4d1675ca487b42199fc30`；正常与次正规数设备用例收据 SHA256：`44f79750da03505516cc724235239ca356622445a0f889f92ab3e214f2bc7e82`。

## 三、Synapse 图编译、运行时与生命周期坑（`P-RT`）

### `P-RT-01`：自定义 TPC `.so` 缺入口符号或漏配官方 `libtpc_kernels.so` 兜底 `[已证实]`
- **现象与触发**：`synNodeCreate` 报 `No Entry point GetShapeInference`；或只把 `GC_KERNEL_PATH` 设成自研 `.so` 后连基础算子都报 `GlueCode 400`。
- **真实根因**：Synapse Kernel DB 要求自定义算子库完整导出 `GetKernelGuids`、`InstantiateTpcKernel`、`GetShapeInference`、`GetSupportedDataLayout`、`GetLibVersion` 五个 C 符号；裸装环境默认未设 `GC_KERNEL_PATH`，覆盖时仍需保留官方库在冒号列表末尾提供通用算子。
- **硬性铁律解法**：构建期用 `nm -D` 强校五符号齐全；运行期固定设置 `GC_KERNEL_PATH=<ours_1.so>:...:/usr/lib/habanalabs/libtpc_kernels.so`。
- **证据落点**：[`redeploy.md`](redeploy.md) 第 4 节。

### `P-RT-02`：Persistent Section 混绑与运行期动态按名查表 `[已证实]`
- **现象与触发**：图编译报 `checkInputPersistence` / launch section mismatch，或每步发射在主机侧反复按字符串查找张量 ID 浪费 CPU 时间。
- **真实根因**：外部可读写张量未各自绑定独立 persistent section、混用 `synTensorSetHostPtr` 与 section 地址修补，或在解码热循环里重复调元数据查询。
- **硬性铁律解法**：每个外部动态/权重张量分配独立 persistent section；编译/加载期一次性通过 `synTensorRetrieveIds` 固化整数 `tensorId` 静态绑定数组，解码步只改地址槽位直接发射。
- **证据落点**：`GaudiRT` `src/recipe_registry.cpp`。

### `P-RT-03`：“声明放 SRAM”被编译器刷回 HBM 及 RMW `16 MiB` 限制 `[已证实]`
- **现象与触发**：构图时给中间张量设了 SRAM 提示，结果性能没涨甚至在两轮 ABBA 中方向反转。
- **真实根因**：`synGraphCompile` 后置调度 Pass 可能根据容量/切片决策插入 DMA 落回 HBM，且单个 RMW Section 存在 `16 MiB` 容量限制；盲目强开 RMW 或跨算子大合图可能导致编译器排程倒退。
- **硬性铁律解法**：绝不把“代码里写了放 SRAM”当成物理驻留证据；一切 SRAM 驻留与合图粒度改动必须通过单变量 ABBA 实测净收益裁决，负收益或不稳定立即撤回（如已撤回的 Layer 1 RMW 实验）。
- **证据落点**：[`../ROADMAP.md`](../ROADMAP.md) 第三节、`GaudiRT` `docs/software-autopsy.md`。

### `P-RT-04`：`synLaunch` 外壳 CPU 开销与设备重叠错觉 `[已证实]`
- **现象与触发**：底层 `scal_stream_submit` 写硬件队列仅 `0.38 µs`，但 `synLaunchWithExternalEventsExt` 单次耗时 `22 ~ 33 µs`（含每次堆分配 `RecipeLauncher`、重复校验与关不掉的内部环形日志原子锁）；反过来，单独优化某段主机 CPU 耗时后，整模 TPOT 未必等额下降。
- **真实根因**：当设备计算/通信耗时大于主机下发耗时时，主机 `launch` 与设备执行在流水线上高度重叠——省掉被掩盖的主机 CPU 时间不会直接缩短设备关键路径；但当每步切碎成近 300 次微型算子与集合通信往返（如 70 层 × 2 次 `hcclAllGather`）时，跨图/跨流同步气泡就会暴露成真瓶颈。
- **硬性铁律解法**：不拿“主机 API 耗时降了多少倍”邀功，先用设备事件拆清“算子计算 vs 集合通信 vs 跨图同步气泡”，再决定合图还是动底层队列。
- **证据落点**：[`gaudi2-hardware-vs-software-autopsy.md`](gaudi2-hardware-vs-software-autopsy.md)、`GaudiRT` `docs/software-autopsy.md`。

### `P-RT-05`：公开 Synapse 发射路径的在途信用上限（`<= 575`）`[已证实]`
- **现象与触发**：无节制异步连续塞入大量不同 Recipe 发射而不回收事件时，触发 `MappedMemMgr` 资源耗尽阻塞或失败。
- **真实根因**：公开路径映射内存槽位有限，合成负载实测受 `2 * N_distinct_recipes + N_in_flight <= 575` 约束。
- **硬性铁律解法**：由 `InFlightCreditManager` 管理在途发射信用并按步/按窗口回收完成事件，同构层强制复用同一 `synRecipeHandle`（把 70 层压缩为约 10 张去重 Recipe）。
- **证据落点**：`GaudiRT` `include/gaudirt/runtime/inflight_credit_manager.hpp`、`docs/DESIGN.md`。

### `P-RT-06`：退出阶段原生终止与设备生命周期 `[现象已证实，根因待核实]`
- **现象与触发**：128 步解码与结果落盘全部正确完成，但进程退出阶段 Rank 0 偶发打印原生终止信息甚至以退出码 `1` / `134` 异常终止（如首轮未修退出的 ABBA `B2`）。
- **真实根因**：尚未确定。Python 局部作用域或 `torch._dynamo` 缓存仍持有设备张量、或其他 Rank 尚未完成通信，均是需要验证的生命周期假设；仅凭调整清理顺序后运行成功不能证明具体触发链。
- **硬性铁律解法**：多卡进程退出前必须先过跨 Rank 完成栅栏、同步全部设备流、显式释放编译图与设备张量引用，最后销毁 `DeviceRuntime`；该顺序必须封装进正式 Runner / C++ 生命周期管理，并将**退出码 `== 0` 且零原生终止告警**列为硬验收门禁。
- **证据落点**：`GaudiRT` 提交 `21b69b9` 的 `models/mimo/runner.py`、`models/mimo/private_use1.cpp` 与 `tests/private_use1_lifecycle_smoke.py`；四轮 8 Rank、延迟一 Rank 的空权重冒烟通过（CPU 编译图及 `PrivateUse1` 张量，非真实整模），收据 SHA256：`4f3f48a4641529fbd4e5b70be9e944bc4df93e16895f01f696b9be147c159181`。此前运行收据：`175f6c06c7bbd7f1bb404c431fbe6ed346579480be139f861c45ed48517b4b03`、`93c16cb993b1ff0c4f8242e1e70914045c19b1518c01c921e2bc9be0f270a5aa`，不单独证明崩溃根因。

### `P-RT-07`：`.recipe` 磁盘缓存未绑定编译器与算子库指纹导致脏读 `[已证实]`
- **现象与触发**：修改了 TPC 内核 `.so` 或构图 C++ 代码后复用旧磁盘 `.recipe` 缓存，静默跑出旧算子结果。
- **真实根因**：离线序列化的 `.recipe` 内嵌了编译时刻的 TPC ELF 二进制与图结构，不随外部 `.so` 更新而自动失效。
- **硬性铁律解法**：缓存目录路径必须包含构图扩展二进制、`libSynapse.so`、`libSynapseMme.so` 及 `GC_KERNEL_PATH` 全部算子库文件的 SHA256 组合摘要；不匹配直接走冷编或报错。严禁为此手搓 HMAC 签名或 `0400` 权限封条等“安全剧场”。
- **证据落点**：`GaudiRT` `models/mimo/private_use1.cpp`（提交 `febf823`）、[`../AGENTS.md`](../AGENTS.md) 三.7。

### `P-RT-08`：HCCL 默认端口 `12345` 冲突与单机 Scale-Up 驱动边界 `[已证实]`
- **现象与触发**：分布式初始化莫名卡死/握手失败；或裸机 `modprobe habanalabs_ib` 报未知符号。
- **真实根因**：旧桥导入 HCCL 时默认抢占 `12345` 端口（常与本机监控 agent 冲突）；单机 8 卡内部 24×100GbE Scale-Up 互联由 `habanalabs_cn` / `habanalabs_en` 驱动承载，无需加载依赖外部 IB 栈的 `habanalabs_ib`。
- **硬性铁律解法**：零旧桥 `GaudiRT` 直接通过显式 `hcclGetUniqueId` 交换通信句柄，不走默认端口猜测；单机环境不装 `habanalabs_ib`。
- **证据落点**：[`redeploy.md`](redeploy.md) 第 3–4 节。

### `P-RT-09`：PyTorch 与插件入口自动加载已安装的旧桥 `[已证实]`
- **现象与触发**：在未配置隔离变量的 CPU 离线评测进程中，导入 PyTorch 时自动尝试加载 `habana_frameworks.torch`；此外 `vllm_gaudi` 默认插件入口（`vllm_gaudi/platform.py` 第 7 行与 `patches.py`）顶层无条件导入 `habana_frameworks.torch`。
- **真实根因**：当前环境的 PyTorch 设备后端入口点包含已安装的旧桥自动加载器，导入 PyTorch 本身即可触发它；同时旧版 `vllm_gaudi.register()` 直接拉入旧 `HpuPlatform`。
- **硬性铁律解法**：候选主进程、Worker 子进程与离线评测入口均必须在导入 PyTorch 前设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`；`vLLM` 零旧桥服务模式在插件入口按配置直接返回独立的零旧桥 Platform / Worker 类，严禁导入旧 `HpuPlatform` 或 `patches.py`，并在初始化期强校 `sys.modules` 与 `/proc/self/maps` 零旧桥映射。
- **证据落点**：`GaudiRT` `models/mimo/runner.py`、`vllm_gaudi/platform.py`；离线现象与隔离验证收据 SHA256：`7020be912798a121c8adc8526dfdb7f793f44cf0192a1a66657c65dd27c943e1`。

### `P-RT-10`：当前 `PrivateUse1` 不支持设备张量 `narrow` `[已证实]`
- **现象与触发**：对已分配的设备张量调用 `narrow(...).cpu()`，八卡小张量冒烟均抛出 `aten::as_strided` 未实现异常。
- **真实根因**：当前受限的 `PrivateUse1` 实现没有注册 `as_strided`；PyTorch 的 `narrow` 依赖该算子。该现象与设备数据数值错误无关。
- **硬性铁律解法**：不使用静默回退；离线 KV 核验使用显式检查范围的连续槽位读取接口，只有在真实服务需要时才实现并验证通用视图。
- **证据落点**：`GaudiRT` 提交 `2f6a401` 的 `model_kv_slots`；8 Rank 70 层前六步 KV 与已审计结果逐字节一致，七步整模退出码为 0（收据 SHA256：`ed12ae92fe1655d33144e1fb79cd7b2a234723f13191af4679f92f977fa79968`）；失败观察收据 SHA256：`ea3c737a85046ae65d7175e5de90f1ebfc84fd6290f9304da0a26340b064c8a2`。

### `P-RT-11`：生产诊断的 eager 标志不控制仓库整步图 `[开关行为已证实]`
- **现象与触发**：生产侧第 6 步同 KV 回放设置了 vLLM `enforce_eager=True`，但 8 Rank 的逐层回放检查报告缺失，进程以 `1` 退出；未得到同输入结果。
- **真实根因**：生产 Runner 的 `_step_graph_enabled()` 根据独立的 `decode_graph` 配置或显式覆盖值决策，此次配置为 `step`；vLLM 的 eager 标志不会改变该开关。图可能绕过逐层 Python 钩子，但这次漏捕的具体触发链仍需验证。
- **硬性铁律解法**：逐层生产诊断必须明确关闭仓库整步图，并确认逐层钩子与目标位置确实命中；诊断不得当作无插桩性能成绩。
- **证据落点**：源码 `gaudi_kernels/serving/runner.py` 的 `_step_graph_enabled()`；失败及开关快筛收据 SHA256：`4f3ed10ed5fe166ea16245372bd9314c21206b3e24879c9a93057bb17bc2cd8a`；显式关闭后的 8 Rank、70 层回放命中目标位置并正常退出，收据 SHA256：`75ccef5c70ffffb6d80d05b8a44347e158e88959d8c3e74a998837df7a84f416`。

### `P-RT-12`：跨桶复用临时张量名称导致形状串用 `[已证实]`
- **现象与触发**：同一进程先构建 SWA M=1 图，再构建 M=32 图；M=32 的 reshape 创建或输出投影编译失败，日志显示临时张量仍为 M=1 形状。
- **真实根因**：无 section 的 `synTensorCreate` 在这些图中按名称复用了先前创建的张量；日志报告名称已映射，即使新描述符请求了不同形状。
- **硬性铁律解法**：不同静态桶的临时张量使用不同名称，并在同进程同时构建旧桶和新桶做设备快筛；不能仅检查传入描述符。
- **证据落点**：`GaudiRT` `models/mimo/mimo_attention.cpp`；失败日志与修复后单卡真实权重对拍收据 SHA256：`87c2ddab1e7052f5906071263b4ded130af9921ab662e144bc457e444808dd1d`。 同一问题也影响同进程的 gathered/direct ModelHead：整模首跑在编译时报告张量已分配，按图类型区分名称后单卡双图输出逐位一致（`GaudiRT` `04f08b9`，失败与快筛收据 SHA256：`76a39093957434f6f9b97ec8b05148a8328523f1f08a230a23e70169ce283354`）。首次异常退出存在原生堆损坏；独立 8 Rank 重现旧编译异常并显式执行扩展关闭后均正常退出、无原生终止消息（收据 SHA256：`74cebf8c62c3bd5b0d250f40e8f3f1bea4c19cec069ca458a8c973a3e7927411`）。该局部重现不包含 70 层状态，不能确定原堆损坏根因。

### `P-RT-13`：逐行偏移持久化张量地址触发段对齐错误 `[已证实]`
- **现象与触发**：参考单行图从连续 I32 元数据块以 4 字节递增地址发射，第二次 `synLaunch` 返回 `synFailedSectionValidation`。
- **真实根因**：该图的持久化段地址要求 `0x80` 字节对齐，4 字节偏移不满足运行时校验。
- **硬性铁律解法**：为独立发射的元数据使用对齐的段地址，或在图内读取同一对齐缓冲区的偏移；不得把任意字节偏移直接当作段基址。
- **证据落点**：`GaudiRT` `tests/mimo_swa_prefill_real_weight_test.cpp`；失败日志与修复后快筛收据 SHA256：`87c2ddab1e7052f5906071263b4ded130af9921ab662e144bc457e444808dd1d`。


### `P-RT-14`：TPC 共享库超过 15 个导致运行时初始化失败 `[已证实]`
- **现象与触发**：Prefill 与 Decode 分别配置算子库后，8 Rank 均在 `synInitialize` 阶段失败，尚未构图或加载权重。
- **真实根因**：当前 Synapse Kernel DB 的 `GC_KERNEL_PATH` 最多接受 15 个 TPC 库；超过上限时日志明确报告 `Maximum number of supported TPC libs is 15`。
- **硬性铁律解法**：把实际使用的自研算子 provider 合并为一个具备完整五入口的静态打包库，保留官方通用算子库，并在运行前核对库数量；缓存指纹覆盖整个打包库。
- **证据落点**：`gaudi-kernels` `7486745` / `d23e585`；17 库失败、15 库三类层 8 Rank 初始化和单卡 Embedding 回归收据 SHA256：`2f5114f880c7d78dbfe39fc959fe1ad443f711e11b9b1611b47d21442caa88a3`；完整 70 层空 KV 冷/暖闭环收据 SHA256：`9f7dd5b21605e082dcea06f4df9d855ff65c48e7afcfd5070271c2367248097f`、`424daa81971473fb224c1af27fc315f59b37a1811a42d5d88ec63c1b6497011a`。

### `P-RT-15`：对 `synHostMalloc` 指针再次映射并解除映射导致主机访问崩溃 `[已证实]`
- **现象与触发**：整模 Prefill 的图已全部提交并同步，回读 4 字节 token 后访问固定内存触发段错误；没有获得 logits 或数值验收结果。
- **真实根因**：`synHostMalloc` 已提供可供 DMA 使用的固定内存，误用为普通主机内存调用 `synHostMap` / `synHostUnmap`；反汇编栈定位到解除映射后读取该指针。
- **硬性铁律解法**：直接用 `synMemCopyAsync` 指向该固定内存，等待对应流或事件完成，再读取并最终用 `synHostFree` 释放；普通主机内存才使用显式映射助手。
- **证据落点**：`GaudiRT` `db77929`；整模失败记录及单卡四次直接 DMA 回归收据 SHA256：`aa912692b59fd4d18c075b7c7e132ddb9de73c580b7dc549c0547bd3d5ad1f7d`；修复后 8 Rank 70 层空 KV Prefill + 128 步 Decode 冷/暖正常完成并退出收据 SHA256：`9f7dd5b21605e082dcea06f4df9d855ff65c48e7afcfd5070271c2367248097f`、`424daa81971473fb224c1af27fc315f59b37a1811a42d5d88ec63c1b6497011a`。

### `P-RT-16`：生产整步图与 eager 在同 KV 下选词不同 `[现象已证实，根因待查]`
- **现象与触发**：Phase 3 第 6 步，8 Rank 全部 70 层有效 KV 注入并回读一致后，生产整步图选择 `35085`；关闭该图的生产 eager 与两份独立 CPU 参考选择 `51273`。
- **真实根因**：图 hidden 的完整 BF16 词表中两个 token 均为 `22.75`，`argmax` 选择较小 ID；相同 hidden 经独立 FP64 词表头仍选该 ID，未发现选词头选错。同 KV 的独立全模型 FP32 / FP64 前两名差分别为 `0.18463 / 0.08301`，后者小于该数值区间 BF16 的 `0.125` 间隔；两者均选择另一 ID。图 hidden 的上游偏差具体来自路由还是算术仍待查，不能仅凭平票认定原因；对 FP32 / FP64 参考的 KL 为 `0.01379 / 0.00751`。
- **硬性铁律解法**：不得把 eager 重放结果当作整步图的数值结论；检查平票时必须用实际 `argmax`，不能把 `topk` 的平票返回顺序当作贪心选词。生产对照与服务验收必须保持图配置一致，并在同输入下单独核验图路径及其完整退出。上游问题修复前不宣称根因已查清。
- **证据落点**：生产图同 KV 重放收据 SHA256：`bbd6fffeaabbfa9d9f76b92e5520503391fca0e780f734fc92d4dce541aebad3`；同 hidden 的 FP64 词表头与全模型参考对拍收据 SHA256：`bf244c58e2f3f87a23161a4eca1e9eab228c750a5276bf449a90ae2464d66bfa`；同 KV 前两名比较收据 SHA256：`b981ec9d6ec9530183385a2e6aeb815d17fd429534bebcbae9f2e57ee4e92c55`；生产 eager 收据 SHA256：`75ccef5c70ffffb6d80d05b8a44347e158e88959d8c3e74a998837df7a84f416`。生产诊断主进程退出码为 0，但框架日志有子进程 SIGTERM 和共享内存警告。

### `P-RT-17`：生产 Prefill 调度分块不保证是自研静态桶整数倍 `[整模已验证]`
- **现象与触发**：真实 vLLM 调度器对 4096-token 请求先下发 505 token，要求每次调度都可被 32 整除的候选 Worker 因此在首块报错。
- **真实根因**：生产分块边界由调度策略和预留规则决定，`max_num_scheduled_tokens=512` 只是预算，不能推断实际分块长度或与算子桶对齐。
- **硬性铁律解法**：保留生产调度与页分配，跨调度调用暂存不足 32 的尾部 token，按顺序拼成静态桶；仅在完整提示已处理时产生首 token，禁止用空结果冒充推理。整模须核对输出和状态等价。
- **证据落点**：失败日志 SHA256：`0f5c70b9e9e33867915bf432019575fab0f28690442164dd879ec96116961e4c`；`vllm-gaudi` `ed5caaa9` 的 505-token 分块顺序单测通过。70 层服务冷、暖运行的首步 logits 与 129 个输出 token 均与独立 Runner 逐位一致；暖运行的 8 Rank、70 层终态 KV 也逐位一致。收据 SHA256：`66fcf3c28741bf21444e89b094f5b55a556308f08a616286d4ac5879e59e2510`、`505f94136d141698fbf2dacba522d74afdb4dc6d034ec993b0cb4940458690d7`。

### `P-RT-18`：EngineCore 的外层关闭期限短于 TP8 Worker 清理 `[现象已证实]`
- **现象与触发**：轻量服务请求已完成，默认关闭时主进程返回 0，但 5 秒后 EngineCore 被强杀，并报告 9 个共享内存对象待清理。
- **真实根因**：Worker 退出预算与 EngineCore 进程管理器的关闭预算是两层独立期限；只设置前者不会延长后者，TP8 的设备同步和退出栅栏可能超过默认 5 秒。
- **硬性铁律解法**：服务生命周期使用 vLLM 的正式关闭接口，同时为 EngineCore 和 Worker 设置足以完成同步及栅栏的期限；检查全部进程退出码和共享内存，而不只看主进程返回值。
- **证据落点**：默认期限失败日志 SHA256：`e99341fbcfcb27f8bd07b1d1253ea6fba4fd5a1420ac3bde93bfed0c00fe10eb`；延长期限后仍需修复下述 EngineCore 清理问题。两项修复后的轻量运行 8 Worker、EngineCore 和主进程退出码均为 0，8 卡释放，无强杀及共享内存告警；日志 SHA256：`a0a434c75162c7dad879fa2fc862ce3448a21dd178c5fae000bad80de623a09a`。

### `P-RT-19`：无设备的 EngineCore 调用通用加速器缓存清理失败 `[已证实]`
- **现象与触发**：8 个 GaudiRT Worker 均以 0 退出后，EngineCore 在关闭期间调用 `torch.accelerator.empty_cache()` 抛出无可用加速器异常，EngineCore 退出码为 1。
- **真实根因**：零旧桥 EngineCore 仅持有 CPU 调度及 IPC 状态，设备与分配器由 Worker 进程持有；当前 PyTorch 通用加速器查询不将该进程的 `PrivateUse1` 注册视作已选加速器。
- **硬性铁律解法**：候选平台的 EngineCore 关闭路径仅清理其自身分布式环境；Worker 仍负责设备同步、跨 Rank 栅栏和资源释放。不得忽略异常或仅凭父进程退出码宣称通过。
- **证据落点**：逐进程退出码和异常日志 SHA256：`3bfbcf40108a5ed5fc835c446261545b8e95ff16af56c3efc4eedf9906088152`；修复 `vllm-gaudi` `194a91ff` 后的 8 Rank 轻量服务和逐进程退出码日志 SHA256：`a0a434c75162c7dad879fa2fc862ce3448a21dd178c5fae000bad80de623a09a`。该轻量测试使用合成输出，不代表整模推理等价。

### `P-RT-20`：Worker“gracefully”日志不证明数值退出码为零 `[已证实]`
- **现象与触发**：完整服务计时的 EngineCore 以 0 退出，日志称所有 Worker 已正常退出，但没有提供各 Worker 的退出码。
- **真实根因**：执行器的该日志仅判断 Worker 是否仍存活，并未检查每个子进程的 `exitcode`；非零退出同样可能使存活检查结束。
- **硬性铁律解法**：验收要单独记录各 Worker 的实际退出状态，不得用该日志或父进程退出码代替。性能运行可在计时结束后的关闭阶段被动读取 `/proc` 僵尸进程等待状态；未捕获到状态时必须标为缺失。
- **证据落点**：8 Rank 轻量测试中，被动观察的全部状态均为 0，与框架内部记录的 8 个退出码一致；观察收据 SHA256：`3cfa2c11f67468f67068e8ba1202471ca5e4a456ecb0871470b5ab6f4d3235e8`，日志 SHA256：`1a65460f60e56c8b8c92a968b2868b732128cbe2836725d22dcbc335287de83f`。

---

## 四、性能测量与评测防骗坑（`P-BENCH`）

### `P-BENCH-01`：跨层共享 Recipe 导致 Profiler 单层耗时虚高近 60 倍 `[已证实]`
- **现象与触发**：Profiler 抓包里看到名为 `model/N` 的 Recipe 耗时高达 **`2,384.5 µs`**，误以为某一层慢得离谱。
- **真实根因**：同构层复用同一 `synRecipeHandle` 时，`model/N` 只是编译期静态图名而非运行期层号；实测层尾共享 Recipe 每步重复发射 60 次、跨层边界 Recipe 重复 59 次，`2,384.5 µs` 是 **59～60 层累积聚合时间**，不是单层耗时。
- **硬性铁律解法**：严禁引用跨层聚合 trace 数字充当单层基线；单层对照必须在独立同输入 standalone harness 下关闭 Profiler 实测。
- **证据落点**：[`../ROADMAP.md`](../ROADMAP.md) 第三节、`instance-assets/mimo-layer1-20261026/recipe-attribution-v1/attribution.json`。

### `P-BENCH-02`：未消费输出被编译器裁除制造虚假高带宽 `[已证实]`
- **现象与触发**：微基准测出 `5.936 TB/s` 远超物理峰值 `2.45 TB/s` 的荒谬带宽。
- **真实根因**：未被下游消费者读取的输出、重复拷贝或同源随机种子被图编译器 DCE / CSE 静默消除，实际并未执行全部访存。
- **硬性铁律解法**：所有基准测试的输出张量必须被下游消费并在计时外做全量数值校验；高于硬件物理屋脊线的成绩一律先按假阳性排查（真实独立种子实测为 `2.761 TB/s` SRAM）。
- **证据落点**：[`../AGENTS.md`](../AGENTS.md) 一.4。

### `P-BENCH-03`：带插桩/Profiler 耗时当成绩及重叠 Span 直接相加 `[已证实]`
- **现象与触发**：挂着 hook 或 HLTV profiler 跑出的耗时比真实无插桩慢数倍，或把多个并行/嵌套区间耗时直接相加得出大于总墙钟的假账单。
- **真实根因**：插桩改变同步与发射节奏，且 MME、TPC、DMA 与主机提交在时间轴上高度重叠；此外设备原始时间戳约为 50MHz 计数器而非核心主频周期。
- **硬性铁律解法**：带 trace / 逐步 KV 审计的运行仅用于定性与定位，**测速必须 100% 关闭插桩**；收益预算只认关键路径墙钟缩减量。
- **证据落点**：[`../AGENTS.md`](../AGENTS.md) 二.4、`GaudiRT` `docs/software-autopsy.md`。

### `P-BENCH-04`：计时起止口径与主机 CPU 策略不对称 `[已证实]`
- **现象与触发**：同一模型仅因主机 CPU 调频/绑核策略不同，TPOT 从 `11.31 ms` 漂到 `13.37 ms`（差 `18%`）；或候选只计纯设备发射、漏掉客户端输出解码（约 `0.15 ~ 0.35 ms`）乃至调度/IPC 开销，造成不对等胜出。
- **真实根因**：单步 `~10 ms` 级解码对主机线程唤醒、NUMA 亲和及前端处理极其敏感。
- **硬性铁律解法**：ABBA 四臂必须锁定同一主机 CPU 亲和策略与清晰对称的起止边界；若候选尚未接入完整 `vLLM` 调度与 IPC（如 Phase 3 独立解码器），必须在收据与文档中如实标注 `same_complete_production_serving_path_proven = false`，严禁冒充完整线上服务成绩。
- **证据落点**：收据 SHA256：`3a1588c0e449fa8df2f3bacc2d860dced6de624322890e7c2308ed77df80d218`。

### `P-BENCH-05`：评测脚本 `NaN` 比较放行漏洞（Fail-Open）`[已证实]`
- **现象与触发**：算子输出含 `NaN` / `Inf`，但判定脚本写 `if err > tol: fail()`，因 IEEE 中 `NaN > tol` 为 `False` 而以退出码 `0` 假通过。
- **真实根因**：负向不等式判断对 `NaN` 天然漏网，且 Python `json.dumps` 默认可能写出非标准 `NaN` 或崩溃。
- **硬性铁律解法**：所有数值门禁必须前置 `torch.isfinite(out).all()`，并采用正向断言 `assert finite and err <= tol`，失败强制非零退出。
- **证据落点**：[`../ROADMAP.md`](../ROADMAP.md) 第四节。

### `P-BENCH-06`：设备初始化覆盖 vLLM 的 NUMA 亲和与调度策略 `[已证实]`
- **现象与触发**：候选 70 层冷启动使用生产 `vllm_kwargs(diagnostics=False)` 后，Worker 主线程仍分别被固定到 CPU 0、4、8、12、16、20、24、28；部分 Rank 不在 vLLM 选定的 NUMA CPU 集合内，Worker 的 `uclamp.min` 为 1024 而 EngineCore 为 0。
- **真实根因**：`PrivateProgram` 初始化 `DeviceRuntime` 时自行设置 `rank × 4` 的 CPU 和默认利用率夹紧，覆盖了 Worker 继承的生产放置策略；之后创建的线程也可能继承该差异。
- **硬性铁律解法**：服务 Worker 的设备初始化必须原样继承 vLLM 的 CPU 和调度策略；验证各 Rank 及相关线程的实际亲和和夹紧值。修复并实测前，此候选结果不得用于同口径 ABBA。
- **证据落点**：冷启动时 `/proc` 与生产放置配置观察收据 SHA256：`7d62e1acda6b6b38e16ca6d437cf2254ec6dcf083e005ae31f63078daab6f9b2`；`GaudiRT` `66cd1a6` 修复后，8 Rank 主线程亲和均与 vLLM 放置一致，检查到的线程均在对应 NUMA 集合内且 `uclamp.min=0`，全部进程退出码为 0、8 卡释放，轻量日志 SHA256：`18981ff868b52e7f71a0427fbbea21f9c078aad2bffb65d11b4b0c8c08271ec9`。两次均为非计时运行，不是性能结论。

---

## 五、已证伪与已撤回的历史结论（严禁死灰复燃）

| 已作废的历史说法 | 证伪事实与现行定论 |
| :--- | :--- |
| “抛弃 `synGraphCompile` 自写通用 `EcbGraphCompiler` 与裸加载器” | **已废止**。MME 切分与 SRAM 双缓冲排程继续复用离线 `synGraphCompile` + 内容指纹 `.recipe` 缓存；仅在公开发射证实吃不掉的瓶颈上按需局部用 `scal/`。 |
| “RT-2 绑定优化提速 `923.9x`（`990.44 µs -> 1.07 µs`）” | **已撤回**。分母来自劣质模拟基线，真实旧路径绑定预检仅约 `98 ~ 127 µs`。严禁打假靶子。 |
| “某单层耗时高达 `2,384.5 µs`” | **已证伪**（见 `P-BENCH-01`）。实为共享 Recipe 跨 59～60 层的聚合计数；真实单层生产 standalone p50 为 `507.9 ~ 520.6 µs`，自研 C++ 单层为 `302.8 ~ 345.6 µs`，70 层融合均摊约 `151 µs`/层。 |
| “70 层跨求值树全词表裸 logits `relative_l2` 必须 `<= 0.02`，或空 KV 4096 长前缀滚动 `KL` 必须 `<= 0.01`” | **已订正**（见 `P-NUM-06`）。单步下纯 CPU FP32 对 FP64 裸 `relative_l2` 即达 `0.02444`（翻转 16 层）；空 KV 4096 长前缀下纯 CPU FP32 对 FP64 自身 `max_kl` 即达 `0.043453`（翻转 32 层），而候选对 FP64 为 `0.034032`（优于 CPU FP32）。已按物理底噪订正为单层同输入门禁 + 固定前缀单步 `KL <= 0.01` + 空 KV 长前缀不劣于 IEEE `FP32 vs FP64` 包络门禁 + 逐步 KV 硬门禁。 |
| “Layer 1 开启 SRAM RMW 必然提速 / `o_proj` 直接转 FP8 必然可用” | **已撤回**。RMW 两轮 ABBA 方向反转不稳定；无重校准 `o_proj` FP8 在 Rank 5 超出 `2%` 单层误差门禁，两者均已从主线移除。 |
