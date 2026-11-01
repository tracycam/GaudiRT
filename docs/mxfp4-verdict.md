# MXFP4 packed 路径判决书（P2 历史取证归档 · 2026-09-23）

> **[!NOTE] 历史专项取证文档**
> 本文归档 2026-09-23 对 checkpoint 量化结构、E8M0 编码与官方 `mxfp4_fused_weights` 语义的专项实测结论。原始 `.json/.log` 工件已按仓库卫生规范迁至 `instance-assets/p2-mxfp4-salvage-20260923/`（SHA256 不变）；共性坑点已并入 `gaudi2-pitfalls.md`，现行架构与阶段状态以根目录 `../ROADMAP.md` 为准。

状态：**P2 最终版（salvage closeout 2026-09-23）**。远端实例已被回收销毁（唯一一次存在性探测 `ssh -o BatchMode=yes -o ConnectTimeout=5 <instance> true` → banner exchange timeout，exit 255，2026-09-23；此后无任何远端访问）。全部测量工件在销毁前已回传本地 **`instance-assets/p2-mxfp4-salvage-20260923/`**（原 `docs/evidence-p2/`，sha256 与该目录 README 表逐一吻合，完整性=实测），本判决书数字均可回溯至这些工件。
judged by **CPU fp32**（`JUDGE: cpu_fp32`，oracle = `vllm-gaudi/tests/unit_tests/oracle/mxfp4_ref.py` 语义：E2M1 LUT `[±0,.5,1,1.5,2,3,4,6]`、低 nibble=偶元素、E8M0=`2^(b-127)` 字面读法；oracle 文件 sha256 `1d16b651…`）。
本文件允许记录绝对实测数字（AGENTS.md 的"无绝对性能数字"约束只针对 commit message/PR；本文件与 evidence-p2 按约定留在 workspace，**不入 git 树**）。

**出处标签（provenance，逐表/逐行给源；单元格内保留 实测/待测 标签）**：
- `S` = **salvaged-from-local-artifacts 2026-09-23**（本 lane **无会话历史可用**——"session history 含测量输出"前提已作废；数字直接取自本地存证 `docs/evidence-p2/<file>`，sha256 与该目录 README 表逐一核对吻合，括号内为 sha256 前 8 位。原始测量=2026-09-22/23 远端 device 0 输出）
- `R` = **recomputed-local 2026-09-23**（本地 CPU 复算，脚本/条件随数字给出）
- `A` = **解析推算**（公式可复算，非测量）
- `L` = **LOST-待测（重部署可复现）**（无存证；见 §9 Evidence-Loss Inventory）

交叉核对：另一并行 lane 正从原 run session 提取数字；两 lane 应在 Oracle gate 2 前对齐，**对齐基准=本文件的 `docs/evidence-p2/` 存证**（本地 sha 校验过的 raw source）。
运行环境（`S`）：远端 8×Gaudi2/HL-225（96GB HBM/卡），实验固定 **device 0**（P2 单卡实验约定）；`/data/venv`（torch 2.11.0a0+git009b5f6），eager（`PT_HPU_LAZY_MODE=0`）+ `torch.compile(backend="hpu_backend")`。
脚本（本仓库 `vllm-gaudi/tools/mxfp4/` + `tools/bench/`，已提交 branch `oracle-mxfp4-cpu`；远端副本 `/data/mxfp4/`、`/data/p2run/` 随机器销毁，代码本体无损）：
`checkpoint_sample.py`（任务 1）、`vendor_op_ab.py`（任务 2）、`e8m0_probe.py`（任务 3）、`cpu_oracle.py`/`vendor_op.py`（共享判定/调用胶水）、`combine_probe.py`（任务 2 根因）、`tools/bench/harness.py`（任务 4）。
记录纪律：**compile-failure / numeric / performance 三类记录分列**；每条数字标 实测/待测 + 出处标签；FAIL 行保留可见。

## 0. 问题（binding questions）
1. vendor `torch.ops.hpu.mixture_of_experts.mxfp4_fused_weights` 能否在 Gaudi2 上以 MiMo 真实形状（gate/up `[2048,6144]`、down `[6144,2048]`）正确工作？（编译失败/数值/性能三分）
2. `weight_scale` vs `weight_scale_inv` 的语义（scale 变体）到底是什么？
3. E8M0 保留码 `{0x00,0x01,0x7F,0x80,0xFD,0xFF}` 的解码行为（0/inf/nan/`2^(b-127)`）？
4. packed 常驻 vs BF16 展开的 HBM 行为（residency delta + 运行峰值）？

## 1. 方法（method）
- 全部数值对照同一 CPU oracle（fp32 裁判）；设备输入的 same-source 纪律：oracle 计算用**设备实际所见值**（x/router weights 经 bf16 roundtrip 后的 fp32 值；权重从 packed 字节按 oracle 语义解码）。
- w12 半序（gate/up 谁过 silu、contiguous/even-odd 行布局）不靠猜：op 每 case **只跑一次**，输出对 **4 个 oracle 布局假设**（contig/evenodd × silu_first/silu_second）分别判定 —— 1 次观测、4 判别行，无重试循环。
- Path A = packed vendor op（`torch.compile(backend="hpu_backend")`，module-level 具名入口 `mxfp4_fused_fwd`，镜像 `extension/ops.py::_mxfp4_fused_fwd` 调用形态，非 bias 变体）。
  Path B = BF16 展开常驻 + 逐专家 eager GEMM（naive fused_moe 参考路径）。
- 计时：`time.perf_counter()` + 显式 `torch.hpu.synchronize()` 双侧夹持；warmup 5 + iters 20，报 min/median。
- 内存：`torch.hpu.memory_stats()`（habana 键 `InUse/MaxInUse/…`）+ `torch.hpu.max_memory_allocated()`（运行峰值）+ `hl-smi -q -d MEMORY`（原始行）；每个字段记录来源 API。
- 通过阈值：`max_rel_err = max|Δ| / max|y_ref| ≤ 2e-2`（bf16 计算链 vs fp32 裁判；rms 一并报告）。

## 2. 任务 1 — 真实 checkpoint 抽检（`S`: sample_report.json `ea81671a`；mmap，`checkpoint_sample.py`）

### 2.1 格式结构（per-shard header + 采样；shards：`model_pp0_ep0_shard0.safetensors`（expert 0..2 + backbone，1817 tensors，34,362,380,864 B data）、`model_pp0_ep1_shard0.safetensors`（expert 3..5，1242 tensors，4,151,181,312 B data）；index `save_format=mxfp4`，`tp_size=8`，total_size 566,027,990,784 B）
| 张量族 | dtype | shape（实测） | scale 伴张量 | 量化方案 |
|---|---|---|---|---|
| experts.{e}.{gate,up}_proj.weight | U8 packed | `[2048, 3072]`（=2048×6144/2） | `weight_scale` U8 `[2048, 192]` | **MXFP4**（2×E2M1/byte + E8M0/32 元素） |
| experts.{e}.down_proj.weight | U8 packed | `[6144, 1024]`（=6144×2048/2） | `weight_scale` U8 `[6144, 64]` | **MXFP4** |
| self_attn.qkv_proj.weight | **F8_E4M3** | `[27136, 6144]` | `weight_scale_inv` **F32** `[216, 48]` | FP8 + per-128×128-block F32 scale |
| layers.0 mlp.{gate,up}_proj.weight | **F8_E4M3** | `[16384, 6144]` | `weight_scale_inv` F32 `[128, 48]` | 同上 |
| layers.0 mlp.down_proj.weight | **F8_E4M3** | `[6144, 16384]` | `weight_scale_inv` F32 `[48, 128]` | 同上 |
| self_attn.o_proj.weight | BF16 | `[6144, 16384]`（v-only：128×128 v_head_dim；layer30 采样 256 行） | — | 未量化（=幅值锚） |
| mlp.gate.weight / e_score_correction_bias | BF16 / F32 | `[384, 6144]` / `[384]`（69 个 MoE 层） | — | router + DeepSeek 式 score 校正偏置 |
| embed_tokens / lm_head | BF16 | `[152576, 6144]` | — | vocab=152576（原"待测"项，本次实测） |
（另有 `visual.*` 视觉塔 BF16 张量族（patch_embed 为 5D `[1280,3,2,16,16]`），非本判决 4 问范围，仅记录存在=实测。）

**结构性发现**（vs `gptoss_mxfp4.py` loader 期望；源=sample_report `loader_expectation_findings`，defects=[]）：
- **L1**（结构）loader 期望 GPT-OSS 融合键 `.w13_weight/.w13_weight_scale/.w2_weight/.w2_weight_scale`；MiMo 是 per-expert `gate/up/down_proj` + `weight_scale` → P3 loader 需 remap + 把 gate/up **按行**stack 成 w12。
- **L2** packed 布局规则（`MoeMXFP4Matmul.set_weight/set_scale`：`(rows, cols/2)` u8 + `(rows, cols/32)` u8）对全部 6 个 expert 张量族 **成立**（`layout_ok=True`，defects=0，实测；layer1/expert0 + layer40/expert3 双样本）。
- **L4**（结构）`weight_scale_inv` 变体在 `gptoss_mxfp4.py` 中完全未处理 → P3 需为 qkv/dense-mlp/MTP 新增逻辑（FP8 解量化，非 MXFP4）。
- **L5**（结构）qkv scale 网格 `[216, 48]` implies 27648 行块（216×128）vs weight 27136 行（=212×128）——多出 512 行（段映射 **待测**；per-segment padding 与 in-shard 整体 padding 两种解释不可区分，**"512 行对齐"不得编码为事实**，见 §8 L5 条）。dense gate/up/down 网格整除干净。

### 2.2 expert `weight_scale`（MXFP4 / E8M0 uint8）值域（`S`: sample_report；256 行采样，layer1/expert0）
| 张量 | scale byte min..max | byte mean | 指数范围（b-127） | ≥250 占比 | decoded \|w\| absmean / absmax |
|---|---|---|---|---|---|
| gate_proj | 114..121 | 116.0（116.027） | -13..-6 | 0.00% | 7.5e-4 / 9.4e-2 |
| up_proj | 114..118 | 115.9（115.923） | -13..-9 | 0.00% | 8.0e-4 / 1.2e-2 |
| down_proj | 116..119 | 117.1（117.065） | -11..-8 | 0.00% | 1.8e-3 / 2.3e-2 |
layer40/expert3（ep1 shard）gate/up scale byte mean=120.1（实测，120.116/120.142）——两 shard 一致量级。
无 ≥250 尾部字节（reserved-code 邻域在真实数据中未出现；全部 12 个采样张量 `pct_ge_250=0.0`）。

### 2.3 `weight_scale_inv`（F32）值分布 + 语义裁决证据（`S`: sample_report `adjudication_evidence`/`scale_inv_samples`）
| 张量 | shape | 值 absmean / absmax | frac_pow2（=2^-x 结构？） | 符号 |
|---|---|---|---|---|
| qkv_proj.weight_scale_inv | `[216, 48]` F32 | 1.73e-4 / 5.63e-4 | **0.0**（不是 2 的幂） | 全正（finite_frac 1.0，mean=absmean） |
| layers.0.gate_proj.weight_scale_inv | `[128, 48]` F32 | 1.39e-4 / 3.18e-4 | **0.0** | 全正 |

解量化幅值候选（`inv_semantics_stats`，per-128×128-block，JUDGE: cpu_fp32）：
| 候选 | qkv implied \|w\| absmean | dense gate implied \|w\| absmean | 锚（o_proj BF16 \|w\| absmean，layer30） |
|---|---|---|---|
| `w = w_q * scale` | **0.0175**（0.017478） | **0.0064**（0.006403） | **0.0132**（0.013180） |
| `w = w_q / scale` | 3.61e5 | 4.48e5 | — |
→ 见 §5 裁决 ②。

## 3. 任务 2 — vendor op A/B（MiMo 真实形状；`S`: ab_full.json `60f410cf`（N1/N2/N3+memory）、ab_N3u.json `24e8d030`、ab_N1.json `da2079ee`（首跑，被 ab_full 取代，保留））

**op schema（实测捕获，ab_full/ab_N1/ab_N3u environment 行一致）**：
`hpu::mixture_of_experts.mxfp4_fused_weights(Tensor hidden_states, Tensor expert_routing_table, Tensor router_weights, Tensor[] w12, Tensor[] w3, Tensor[] d_scale_w12, Tensor[] d_scale_w3, *, int block_size, bool permuted_weights, str activation, int experts_min, int experts_max, bool is_fp4, int chunk_size=0, int total_experts=0) -> Tensor`

### 3.1 记录类：compile（`S`: ab_full/ab_N3u compile_status 行）
| case | Path A packed vendor op（torch.compile hpu_backend，module-level 具名入口） | Path B BF16 展开+GEMM |
|---|---|---|
| N1 (E1,T4,top1) | OK | OK |
| N2 (E8,T16,top4) | OK | OK |
| N3 (E64,T32,top8) | OK | OK |
| N3u (E64,T32,top8, unique ids) | OK | OK |

compile-failure 记录数 = **0**。eager 原生调用被 frontend_blocklist:[eager] 挡住（ops.py 文档）——本判决一律走 compile 路径（Gate-1 纪律：具名函数入口，无 stdin/lambda 图）。

### 3.2 记录类：numeric（`S`: ab_full/ab_N3u numeric 行；`JUDGE: cpu_fp32`；阈 max_rel_err ≤ 2e-2；判定列=与哪一布局假设吻合）
| case | contig_silu_first | contig_silu_second | evenodd_silu_first | evenodd_silu_second | 判定 |
|---|---|---|---|---|---|
| N1 A / B | **A 3.5e-3 ✓**（0.003540）/ B 3.5e-3 ✓ | 1.00 ✗ | 1.52 ✗ | 1.43 ✗ | contig_silu_first |
| N2 A（random ids） | 6.6e-1 ✗（0.6581）/ B **6.1e-3 ✓**（0.00608） | ✗ | ✗ | ✗ | 见 §3.3 根因（fixture 重复 id） |
| N3 A（random ids） | 6.2e-1 ✗（0.6198）/ B **8.6e-3 ✓**（0.00857） | ✗ | ✗ | ✗ | 同上 |
| **N3u A（unique ids）** | **A 4.4e-3 ✓**（0.00442）/ B 7.6e-3 ✓（0.00765） | 1.01 ✗ | 1.32 ✗ | 1.34 ✗ | **contig_silu_first，64 专家 top-8 组合形态 PASS** |

（数字为 max_rel_err；A-vs-B 直接互校（ab_full/ab_N3u `A_vs_B_device_crosscheck`）：N1 1.1e-3、N3u 7.9e-3 —— packed kernel 与 BF16 展开路径在 bf16 噪声内一致。）

**w12 半序判定（结构性结论）**：`contig_silu_first` 唯一吻合 ⇒ w12 = **按行 concat(gate, up)**（gate 在前 I 行），`activation="silu"` = `h = silu(gate_part) * up_part` —— 与 vLLM `silu_and_mul`（先半 silu）约定一致。evenodd 布局假设与两个半序被 100%+ 误差否决（判别力充分）。

### 3.3 N2/N3 mismatch 根因证伪（`S`: combine.json `6b3de89d`；`combine_probe.py`，一变量一 case，各跑一次）
| case | 变量 | 变体→max_rel_err | 命中 |
|---|---|---|---|
| F1 (top1, rw=0.37) | rw 应用方式 | rw_as_is **4.7e-3 ✓** / rw_normalized_to_1 0.63 ✗ / rw_squared 1.70 ✗ | **rw 原样乘**（不再归一化、不平方） |
| F2 (top4, ids 唯一+升序) | 多槽 combine | accumulate **4.1e-3 ✓** | **标准累加** |
| F3 (top4, ids=[e1,e1,e2,e2]) | 行内重复 id | accumulate_all 0.57 ✗ / collapse_last_wins **4.6e-3 ✓** / collapse_first_wins 0.29 ✗ | **同行重复 id 坍缩为最后一个槽的贡献** |

⇒ N2/N3 的 numeric_mismatch 根因 = fixture 随机 id 同行重复触发 op 的 last-slot-wins 语义，与 oracle 累加语义分叉。**这是 op 的组合语义契约，不是 kernel 数值缺陷**：MiMo 真实 top-k 路由（384 选 8）id 天然唯一，永不触发。N2/N3 FAIL 行按记录纪律保留在 `ab_full.json`（fixture 契约违规的证据）；契约合规下（F2 / N3u / harness）数值全 PASS。
fixture 含重复 id 的行数统计（旧稿"约 59%/约 52%"无存证计算，**已废弃**；替换为 `R` 复算）：seed 1234 按 `make_case` 抽取流本地重放 —— **N2 = 9/16 行（56.3%）、N3 = 17/32 行（53.1%）** 含同行重复 id；解析期望（均匀放回）N2 59.0%、N3 36.6%（`A`）。复算条件=本地 numpy PCG64 流与远端一致（NEP-19 稳定流，**未逐字节校验=待测**）；fixture 终态本身 `L`（未持久化）。

### 3.4 记录类：performance（`S`: ab_full/ab_N3u performance 行；sync-correct：perf_counter + `torch.hpu.synchronize` 双侧；warmup 5 + iters 20；device 0）

**口径标签（Gate-2 P1-C，逐项分标）**：Path B = **naive per-expert eager BF16 baseline**（逐专家 eager GEMM，**非优化基线**）；**valid samples = 20/20**（每 case iters 20，无剔除；FAIL/离群行保留可见）；加速比=同统计量 B/A，**逐统计量分标**，不取单一"口径为准"。
| case | Path A min / median (ms) | Path B min / median (ms) | A 加速（min 口径，B/A） |
|---|---|---|---|
| N1 | 0.58 / 0.62 | 2.63 / 2.81 | 4.56× |
| N2 | 2.10 / 2.24 | 10.45 / 11.56 | 4.97× |
| N3 | 14.65 / 15.04 | 72.29 / 76.96 | 4.93× |
| N3u | 13.00 / 31.33 | 69.54 / 73.23 | 5.35× |
（加速比=两 min 相除，`R` 复算自 min/median 原值；精确 min：A/B = 0.5765/2.6287、2.1033/10.4485、14.6519/72.2857、12.9972/69.5380 ms。
**N3u 加速比分标（`R` 复算自 ab_N3u all_ms，同统计量 B/A）：min 5.35×（5.3502）/ median 2.34×（2.3374）/ mean 1.76×（1.7570）**。
离群保留可见：N3 Path A all_ms 含 2 个 ≥40ms 样本（51.8、41.4）；**N3u Path A all_ms 含 9 个 ≥40ms 样本**（45.99–78.55，双稳态：前 11 个 13.0–33.5ms、后 9 个 45.99–78.6ms；旧稿"2 个 40ms+ 离群"记错对象=实为 N3，已更正）——min 只是其中一个统计量（分标见上）；"stable ~5×" 稳定加速口径 **RETRACTED（撤回）**；稳定性/双稳态归因=**待测**（P4 条款：alternating A/B order + repeats + device-state recording）。）

### 3.5 记录类：memory（64 专家真实形状；`S`: ab_full.json memory 行；API：`torch.hpu.memory_stats`（InUse）+ `torch.hpu.max_memory_allocated`（峰值）+ `hl-smi -q -d MEMORY`（旧采集仅含 banner——脚本只截取 stdout 前 8 行的**采集截断缺陷**，非 SMI 环境故障结论；明细行未入档，InUse 口径以前两者为准））
| 状态 | InUse (bytes) | Δ | 理论对照（`A`） |
|---|---|---|---|
| baseline（输入小张量） | 230,427,008 | — | — |
| + packed 64 专家（w12+w3+scales） | 1,513,884,032 | **+1,283,457,024（1.283 GB）** | **精确相等**：20,054,016 B/专家（w12 4096×3072 + w3 6144×1024 + d_scale_w12 4096×192 + d_scale_w3 6144×64）× 64 = 1,283,457,024 B |
| + BF16 展开（gate/up/down 全量） | 6,346,116,992 | +4,832,232,960 = **4,831,838,208 B（expanded net delta）+ 394,752 B（packed 快照后创建的输入张量）** | **expanded net delta = 理论精确值 4,831,838,208 B**（2 B/元素：37,748,736 元素/专家 × 2 × 64）；+394,752 B **精确分解**：x 32×6144×2=393,216 + ids 32×8×4=1,024 + rw 32×8×2=512（旧稿"余 +394,752 B=分配器对齐余量"归因错误，**已更正并撤回**：该余量恰为这三个输入张量，不是分配器对齐） |
| packed op 运行峰值（**packed 与 BF16 共驻状态**下测得） | 6,346,903,424 | = 共驻常驻 +**786,432 B** | 峰值口径（逐字）：**"in the packed-and-BF16-co-resident experiment, the HPU allocator reports +786,432 B peak for the packed call; no GB-scale full expansion was observed"** |
| 展开路径运行峰值（同共驻状态） | 6,346,829,696 | = 共驻常驻 +712,704 B | 同上共驻口径（`S`） |
| 释放后（after_free_weights_keep_inputs） | 230,821,760 | ≈ baseline ✓ | — |
（旧稿"理论 packed 1.206GB + 视图开销"表述不准，已更正：实测增量与 packed 张量字节数**逐字节相等**（含 scale 张量），不存在额外"视图开销"。口径注记：packed→expanded 窗口内还创建了 ~4 KiB int64 索引张量（`A`），InUse 未单独可见——allocator 记账口径局限；**本表数字不外推到 op 内部 scratch**。）

⇒ **峰值口径（逐字，P1-B2 修订）**："in the packed-and-BF16-co-resident experiment, the HPU allocator reports +786,432 B peak for the packed call; no GB-scale full expansion was observed"（`S`）。**RETRACTED（撤回）**：旧稿"packed 路径不做临时展开 / 无任何临时解量化"的全称断言——allocator 记账不覆盖 vendor op 的内部 scratch（其内部 TPC/MME 切分=**UNMEASURED/未测**，见 §6 fallback 规则），**不得由 allocator 数字外推**。可保留的结论仅限：**未观测到 GB 级全量展开物化**。展开/BF16 路径的 HBM 成本 = 常驻 ~4× + 同量级读带宽。

## 4. 任务 3 — E8M0 保留码探测（`S`: e8m0_v2.json `23639e1f`（现行）+ e8m0.json `1f967f4f`（v1 缺陷记录，FAIL 可见）；e8m0_probe v2；v1 缺陷=0×inf 污染 0xFF 行、atol 过宽误标 0x00/0x01）

### 4.1 w3 侧线性探针（ratio = 该 scale 字节在被测路径中的**有效贡献比**；若链路无 flush 才等于 decoded(b)——机制=待测，见下；`JUDGE: cpu_fp32` 字面参考 `2^(b-127)`）
| byte | ratio 观测 | 字面参考 | 命中？ | 结论（**仅被测字节**；测试集合={0x00,0x01,0x7F,0x80,0xFD,0xFF}，其余 250 字节=**待测**） |
|---|---|---|---|---|
| 0x00 | **0.0**（amplified-C 下仍 0.0） | 5.88e-39 | ✗ | **0x00 → zero effective contribution in the tested path**（实测）；机制=**待测**：reserved-zero-decoder 与 subnormal-flush-before-multiply 在这些观测下**不可区分**，不选机制（旧稿"decoder 对 0x00 返回 0（机制实测，非 flush）"属越权断言，已撤回） |
| 0x01 | v1 C=1 观测 0.0；v2 amplified-C 观测 1.1754944e-38（精确命中 2^-126） | 2^-126 = 1.1754944e-38 | ✓ 精确（amplified 口径） | 观测=实测（amplified-C 复原 2^-126）；C=1 归零的环节归属=**待测**（旧稿"证实机制=flush 不是 decoder"属选机制，已撤回） |
| 0x7F | 1.0 | 1.0 | ✓ | 参考行 |
| 0x80 | 2.0 | 2.0 | ✓ | 字面（观测与参考一致） |
| 0xFD | v1 干净测量 **8.5070592e37 = 2^126 精确**；v2 放大后 inf（y=C·2^126 溢出 f32） | 2^126 | ✓（v1 口径） | **0xFD = 2^126 精确**（实测，保留） |
| 0xFF | **+inf**（v2 全非零 h fixture；w12 侧同为 +inf） | 2^128→f32 +inf | ✓（按 f32 可表达口径；v2 `matches_literal=true`） | **0xFF → +inf**（positive fixture 实测，保留）。措辞限定：**本实现的被测行为与 "OCP 0xFF=NaN" 读法不符**（不作"OCP 读法被否证"断言；NaN 只出现在 v1 的 0×inf 污染行） |

### 4.2 w12 侧次级探针（过 silu；判别行；v1=v2 一致）
`0x80` 观测 2.417 vs 预测 silu-侧 2.4096 / mult-侧 2.0 → **w12 row 0 = silu 侧**（与 §3.2 的 gate-first 判定互证）；`0xFD` 观测 1.1646e38 ≈ 预测 1.1637e38（字面）；`0xFF` = +inf。`0x00/0x01/0x7F` 标 不可判（两假设在该字节重合）。

## 5. 四项裁决
① **vendor op 在 Gaudi2、MiMo 真实形状上可用且数值正确**：compile 路径 0 失败；unique-id 契约下（**实测范围=unique 且升序 id fixture**；unsorted-unique=待测，见 §8 routing 条）Path A 对 CPU fp32 oracle max_rel_err 在 ~0.5% 量级（N1 0.35%、N3u 0.44%、F2 0.41%、harness packed 0.50%；`S`），64 专家 top-8 组合形态 PASS；对 **naive per-expert eager BF16 baseline** 的加速比**逐统计量分标**（N3u：min 5.35× / median 2.34× / mean 1.76×，valid samples 20/20，`S`/`R`；其余 case 见 §3.4）——"stable ~5×" 稳定口径 **RETRACTED**（N3u Path A 双稳态；稳定性=待测，P4）。契约（全部实测）：w12=[gate;up] 行拼接、silu 作用于 gate（前）半、router_weights 原样乘（caller 归一化）、routing 表 int32、**token 行内 expert id 必须唯一**（否则 last-slot-wins 坍缩）、只能走 torch.compile(hpu_backend)（eager blocklist）。
② **`weight_scale` 与 `weight_scale_inv` 是两种不同量化方案的伴生张量**：expert `weight_scale`=MXFP4 的 uint8 E8M0（解码乘子 `2^(b-127)`，源码+幅值锚双证）；backbone `weight_scale_inv`=FP8_E4M3 的 per-128×128-block **F32 解量化乘子**（`w = w_q × scale_inv` 乘法解释=**当前 WORKING hypothesis（可据以设计）**：乘法候选 0.0175/0.0064 对齐 BF16 锚 0.0132、除法候选 3.6e5/4.5e5 被幅值否证（`S`）——但幅值锚+自洽 fixture **不是语义证明**；正式 checkpoint consumer 出货前需 §6 L4 契约的独立语义确认=待测 gate）。值不是 2 的幂（frac_pow2=0.0，"2^-x" 说否证）、全正（"additive inverse" 说否证）——**"_inv" 只指"存储的是量化 scale 的倒数（=解量化乘子）"**（工作假设口径）。
③ **E8M0 解码观测（结论仅覆盖被测 6 字节 {0x00,0x01,0x7F,0x80,0xFD,0xFF}；其余 250 字节=待测，本回合不做全字节扫描）**：**0xFD = 2^126 精确**（实测，保留）；**0xFF → +inf**（positive fixture 实测，保留）——措辞限定为"本实现的被测行为与 OCP 0xFF=NaN 读法不符"（**不作"OCP 读法被否证"断言**；NaN 只出现在 v1 的 0×inf 污染行）；0x01 在 amplified-C 下精确命中 2^-126（实测），C=1 时归零的环节归属=待测；**0x00 → zero effective contribution in the tested path**（实测），机制=**待测**：reserved-zero-decoder 与 subnormal-flush-before-multiply 在现有观测下**不可区分**，不选机制。
④ **packed 常驻 vs 展开**：packed 64 专家常驻增量 1,283,457,024 B（=packed 字节精确值）vs BF16 展开 **expanded net delta 4,831,838,208 B（=理论精确值）**（行 Δ 4,832,232,960 B = 展开权重 + 输入 394,752 B；x/ids/rw 精确分解见 §3.5），=packed 的 3.77×；峰值口径（逐字）："in the packed-and-BF16-co-resident experiment, the HPU allocator reports +786,432 B peak for the packed call; no GB-scale full expansion was observed"（`S`）——"完全无临时解量化/不做临时展开"全称断言 **RETRACTED**（allocator 数字不外推到 op 内部 scratch；内部 TPC/MME 切分=未测）。全模型 BF16 2.05 TB（=1.024T 参数×2 B，`A`）> 768 GB（8×96 GB，`A`）的准入结论不变。

## 6. P3 建议（显式映射 + 可测契约，Gate-2 P1-C）

**主路径 = 直接构建在 vendor `mixture_of_experts.mxfp4_fused_weights` 上**（裁决 ①）：
- P3-3a（"TPC fused unpack+scale+layout primitive"）**范围缩减**：MoE 的 unpack+scale+GEMM 已由 vendor kernel 完成（实测正确；加速比分标见 §3.4 vs naive per-expert eager BF16 baseline；**vendor op 内部 TPC/MME 切分=UNMEASURED/未测，不得写成 measured fact**），**不再需要自研 MoE 解包 kernel**。3a 仅保留两个可能需求：(a) backbone FP8_E4M3+F32 block-scale 解量化/布局（若 `vllm_gaudi` 既有 FP8 路径不能直接消费 `weight_scale_inv` 布局——先查现有 FP8 ops 再决定）；(b) gate/up 按行 stack 的 loader 期拷贝（廉价，无需 kernel）。
- **L4 语义契约（FP8 `weight_scale_inv`）**：`w = w_q × scale_inv` 乘法解释 = 当前 **WORKING hypothesis**（允许据以设计）。**出货 gate（正式 checkpoint consumer 前 REQUIRED，现=待测）**：独立语义确认，二选一 —— (i) producer/exporter 文档明示 `w_q*scale_inv` + block size + 布局，并对真实 checkpoint 切片做独立 decode 交叉验证；或 (ii) known inputs 过真实 exporter，验证 block 边界 + reconstruction error。**自洽 fixture（本仓库 oracle 自产自解码）不构成证明**。FP8 有损：**不做 exact-round-trip 要求**。
- **L5 qkv 映射契约**：qkv scale 网格 `[216,48]` 的段映射=**待测**，作为 **PRE-req** 仅阻塞：qkv-scale remap/dequant/split/TP slicing、qkv-dependent attention、whole-model acceptance、（同布局时的）MTP。**不阻塞** dense-MLP / MoE-loader 工作。解析假设=**待测的竞争解释之一**（见 §8 L5 条）：TP8 per-shard Q/K/V 行数 3072/192/128 → 24+2+1=27 blocks，×8=216 恰合 scale 网格；per-shard K tail 64 行 ×8=512 恰合行超出——但 **in-shard 整体 padding 产生完全相同的观测**，**"512 per-segment alignment" 不得编码为事实**。解决条件=权威映射（producer 文档/源码）+ segment-boundary same-source 对比。
- **routing 契约（不变量）**：每次 vendor-op 调用处，每个 token 行的有效 expert id **唯一** 且 weighted-sum 语义保持。标准 top-k：天然成立（by construction）。可能产生重复 id 的自定义 routing：入口 **必须拒绝或合并重复**（w'_e = Σ_j w_j 对同一 e），**不是 loader 的职责**；**不强制 CPU 在热路径**（CPU 作 test oracle 或 host-side 输入边界即可）。zero-padding **禁止** 用重复 expert id + 零权重填充（op 是 last-slot-wins，会覆写有效贡献）——用真正唯一的合法零权重 id，或显式 effective-length 形式。验证契约见 §8 routing 条的 4 项必测。
- **fallback 规则（重写；取代旧"两触发条件"）**：**investigate-condition ≠ build-condition**（"vendor 性能不满意 / 语义不匹配"只是调查条件，不是自研条件）。仅当以下三条**同时**成立才自研 primitive：(1) vendor 路径在某项 **REQUIRED** 的 correctness/feature/memory/established-performance 需求上不达标；(2) config/上层适配无法补齐差距；(3) 证据指向自研 primitive 是保住 packed memory budget 的修法。
  - fallback A = 自研 TPC fused unpack+scale primitive —— 准入=上述三条件（非旧的"P4 profile 证明瓶颈 / 语义不支持"两触发条件）。
  - fallback B = BF16 展开 + 标准 GEMM（= **naive per-expert eager BF16 baseline**）—— 仅作正确性兜底；加速比分标（N3u min 5.35×/median 2.34×/mean 1.76×，见 §3.4）、4× HBM、且全模型 BF16 不可驻留（准入否决），MiMo 全量不可用，仅小模型/子图对照可用（P4 对照粒度条款一致）。
- P3-3b（MiMoV2 适配面）必须实现（从本判决推导）：loader remap（L1：gate/up 行拼接→w12，down→w3；scale 同规则）；`weight_scale_inv` FP8 解量化逻辑（按 **L4 契约** gate 推进）；qkv 面按 **L5 契约**的 blocking 集合排期（不阻塞 dense-MLP/MoE-loader）；routing 入口按 **routing 契约**实现（拒绝或合并重复 id；MiMo top-k 天然唯一；`e_score_correction_bias` 由 router 前端处理）；activation 无需适配（gate-first SwiGLU = 标准约定）。

## 7. harness device slots（任务 4；`S`: harness_dev.jsonl `0ab39ba9`）
`tools/bench/harness.py`：`bf16` + `packed_mxfp4` 槽已实现（同一 MoE workload、MiMo 真实 per-expert 形状、top-k 唯一 id 路由、每 run 对 CPU fp32 oracle 判定 `JUDGE: cpu_fp32`）；`fp8` 槽保持 `NotImplementedError`（P4 保留）。远端实测（harness_dev.jsonl，E16/T32/top8 unique，warmup 3）：bf16 judge max_rel_err 9.4e-3 ✓、packed_mxfp4 5.0e-3 ✓、fp8 抛 P4 ✓。**harness 计时口径 = single post-warmup latency（单次 post-warmup 时延），不是 stable throughput（Gate-2 P1-C）**：实测 packed 4.86 ms vs bf16 21.49 ms（单次比值 4.4×；**单样本口径**，非稳定吞吐、非稳定加速比）（旧稿 packed "16.4ms" 无任何存证工件支撑，**已废弃**，以 harness_dev.jsonl 实测 4.859516 ms 为准；bf16 21.488347 ms 与旧稿一致）。CPU 侧测试本地复跑：31 passed, 2 skipped（2 个 device 测试无 HPU skip）——`R` recomputed-local 2026-09-23；Gate-2 整改后（judge fail-open 修复 + hygiene）：**50 passed, 2 skipped**（`R` recomputed-local 2026-09-23，同 CPU 环境）。

## 8. 残余风险（保留 待测 标签；P3 契约的验收测试在此登记）
1. **L4 独立语义确认 gate=待测**：`w = w_q × scale_inv` 是 WORKING hypothesis（可据以设计）；正式 checkpoint consumer **出货前 REQUIRED**（§6 L4 契约，i/ii 二选一）：producer/exporter 文档明示乘法 + block size + 布局并做真实切片独立 decode 交叉验证，或 known inputs 过真实 exporter 验证 block 边界 + reconstruction error。**自洽 fixture 不构成证明**；FP8 有损，**不做 exact-round-trip 要求**。
2. **L5 qkv scale 网格 [216,48] → 27648 行块 vs weight 27136 行（+512 行）段映射=待测**（blocking 集合按 §6 L5 契约：qkv-scale remap/dequant/split/TP slicing + qkv-dependent attention + whole-model acceptance（+同布局 MTP）；**不阻塞** dense-MLP / MoE-loader；dense MLP 网格已实测干净）。竞争解释**均待测、不可区分**：per-segment padding（TP8 per-shard Q/K/V 行数 3072/192/128 → 24+2+1=27 blocks，×8=216 恰合网格；per-shard K tail 64 行 ×8=512 恰合行超出）**与** in-shard 整体 padding（per-shard 3392 行 → 27 blocks，超出 64 行/shard ×8=512）**产生完全相同观测**——**"512 per-segment alignment" 不得编码为事实**。解决=权威映射 + segment-boundary same-source 对比。
3. **routing 防御=待测（P3 实现时验证；契约见 §6 routing 条）**：重复 id 入口必须拒绝或合并（w'_e=Σ w_j），**不是 loader 职责**；**不强制 CPU 在热路径**（CPU 作 test oracle 或 host-side 输入边界即可）；zero-padding 不得用重复 expert id + 零权重（last-slot-wins 会覆写有效贡献）。必测 4 项：(1) 真实 router 输出唯一性（**含 TIED scores**）；(2) 重复输入被拒绝或合并后与 per-slot accumulation oracle 一致；(3) **unsorted unique ids**——现 make_case/F2/harness fixtures 全部排序 id，**"unique 即覆盖"外推不成立**，unsorted-unique=待测；(4) padding 永不覆写有效贡献。
4. bias 变体 `bias_mxfp4_fused_weights`（GPT-OSS SwiGLU-OAI，alpha/limit）= **待测**（MiMo 无 bias 未走该路径）。
5. **E8M0 机制=待测（结论仅覆盖被测 6 字节）**：`0x00 → zero effective contribution in the tested path`（实测），但 reserved-zero-decoder 与 subnormal-flush-before-multiply 在现有观测下**不可区分**——两者均待测，**不选机制**；0x01 的 C=1 归零环节归属=待测（amplified-C 复原 2^-126=实测）。其余 250 字节解码行为=待测（S1E8M0 负指数编码等解释均待测；本回合不做全字节扫描）。真实 checkpoint 实测 0 个 ≥250 字节（保留码邻域无真实数据）。
6. chunk_size/total_experts 调优参数全部按 0 传入（默认路径）；分块调优收益=待测（P4 profile）。
7. 单卡（device 0）单轮测量；跨卡一致性、lazy 模式行为=待测（eager 已测；lazy A/B 按 Gate-1 裁决不作 P4 前置）。
8. 动态 T（变 batch）下的 graph 重编译行为=待测（P4）。
9. **N3u Path A timing 双稳态/离群（9/20 样本 ≥40ms）=待测**：稳定性与双稳态归因未测；P4 条款：**alternating A/B order + repeats + device-state recording**；"stable ~5×" 稳定加速口径已 **RETRACTED**（分标见 §3.4）。
10. **hl-smi 口径=待测**：旧采集仅含 banner——`hpu_mem_snapshot` 只取 stdout 前 8 行的**采集截断缺陷（脚本侧，已修）**，**不是** SMI 环境故障的证据（ab_N1 的 `unavailable rc=1` 同理=旧查询观测，不作故障结论）；HBM 数字以 torch.hpu 两 API 为准=实测。
11. §3.3 复算流依赖 numpy PCG64 跨版本稳定=未逐字节校验（待测）；不影响 F3 机制结论（F3 为显式构造 fixture）。

## 9. Evidence-Loss Inventory（证据损失清单，2026-09-23 远端销毁事件）
| 项 | 状态 | 影响 / 复现方式 |
|---|---|---|
| 远端临时实测整机（/data/mxfp4/、/data/p2run/、/data/venv、/data/models） | **LOST**（云端实例回收；探测 timeout exit 255） | 不可原地复跑；脚本已在库内 branch `oracle-mxfp4-cpu`，重部署可复现 |
| 全部测量工件（sample_report / ab_full / ab_N3u / ab_N1 / combine / e8m0 v1+v2 / harness_dev + 各 .log） | **幸存**（`docs/evidence-p2/`；sha256 与 README 表逐一核对吻合=完整性实测） | 非损失；本判决书全部 `S` 数字可回溯 |
| device 原始输出张量（y_A/y_B/ycpu 本体） | **L**（从未持久化，仅误差统计入档） | 重部署重跑可复现；不影响裁决（误差统计即判定依据） |
| N2/N3 fixture 终态 ids（随机抽样） | **L**（原始流未持久化） | `R` 本地重放 seed 1234 得 N2 9/16、N3 17/32 含重复行（待测级替代）；旧稿"约 59%/52%"无存证、已废弃 |
| harness packed 稳态 "16.4ms" 数字 | **L**（无任何存证工件） | 已废弃；实测口径=harness_dev.jsonl 4.86 ms |
| `/root/.habana_logs` graph dumps | **L（LOST-待测，重部署可复现）** | 从未入档；graph-mode 证据不属本次 4 问判定链 |
| hl-smi `-q -d MEMORY` 明细行 | **L**（仅 banner 行入档——`hpu_mem_snapshot` 只取 stdout 前 8 行的**采集截断缺陷**，非 SMI 环境故障结论；截断已修） | 裁决 ④ 已改以 torch.hpu 双 API 口径（ab_full memory 行） |
| P2 原始任务清单（brief "任务 1–6" 原文） | **L** | 任务编号映射按存证重构（见 closeout report）；四项 binding questions（§0）为存证原文 |
| 本 lane 会话历史（"含测量输出"前提） | **不存在**（前提已作废：message 历史检索空、raw history 读取被限） | 从未依赖；全部 `S` 数字直取本地存证。并行 lane 从原 run session 提取的数字仅作交叉核对，冲突时以 evidence-p2 raw 工件为准 |
| `tools/bench/results.jsonl` | **从未存在**（实测文件系统+git 全历史 grep 均无） | 无需删除；从未提交 |
| `__pycache__/`、`.graph_dumps/` | 从未入 commit（实测 `git ls-files`/show --stat 无命中；`__pycache__` 在 .gitignore） | 无 |

（记录纪律：本表 FAIL/LOST 行保留可见，不做静默修补。）
