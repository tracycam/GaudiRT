# MANIFEST.md — Gaudi2 驱动、TPC 工具链与版本契约

> 本文仅维护本项目的驱动、TPC 编译工具链与基础依赖版本契约。现行全栈架构、阶段进度与验收标准见 [`ROADMAP.md`](ROADMAP.md)；实测硬件、编译器与运行时踩坑总台账见 [`docs/gaudi2-pitfalls.md`](docs/gaudi2-pitfalls.md)；2026-09 早期单算子与工具链探索原始封存位于 `instance-assets/`。
> **认识论标签**: `实测` = 本机/给定输入直接运行并看到输出; `待测` = 未验证(附条件与命令)。

---

## 1. 远端服务器 (版本契约基准)

| 项 | 值 | 标签 |
|---|---|---|
| Habana driver | `1.24.1-b336d5e` | 实测 (任务给定, 由先前会话测得) |
| dkms package | `1.24.1-482` | 实测 (同上) |
| kernel | `5.15.0-118-generic` | 实测 (同上) |
| OS | Ubuntu 22.04.4 | 实测 (同上) |
| 加速器 | 8× Gaudi2, PCI ID `1da3:1020` | 实测 (同上) |
| SPI preboot | `hl-gaudi2-1.22.0-fw-61.3.1` | 实测 (同上) |

## 2. 本地开发机 (本机实测)

| 项 | 值 | 标签 |
|---|---|---|
| OS | Ubuntu 22.04.4 LTS (jammy) — WSL2 | 实测 (`lsb_release`, `uname -r`) |
| kernel | `6.6.87.2-microsoft-standard-WSL2` | 实测 (`uname -r`;≠ 远端, 本地无 Gaudi 硬件) |
| sudo | password-less 可用 | 实测 (`sudo -n true`) |
| apt proxy 坑 | `/etc/apt/apt.conf.d/proxy.conf` 指向死端口 `127.0.0.1:10809`, 覆盖可用的 `99proxy`(10808); apt 须加 `-o Acquire::{http,https}::Proxy=http://127.0.0.1:10808`。系统配置未改动 | 实测 (curl 走 10808=200, 10809=超时) |

## 3. Habana Vault apt 仓库 + TPC 工具链 (本地实测)

| 项 | 值 | 标签 |
|---|---|---|
| GPG key | `curl .../api/gpg/key/public \| sudo apt-key add -` → `OK` | 实测 |
| repo 文件 | `/etc/apt/sources.list.d/habanalabs_synapseai.list`: `deb https://vault.habana.ai/artifactory/debian jammy main` | 实测 |
| 已安装包 | `ii habanalabs-tools 1.24.1-482 amd64` (唯一 habana 包) | 实测 (`dpkg -l \| grep habana`) |
| 版本 pin | `apt-get install habanalabs-tools=1.24.1-482` 成功; madison 首选即 `1.24.1-482` | 实测 |
| tpc-clang | `TPC C/C++ compiler version 2.7.33` / `Target: tpc` / `InstalledDir: /usr/bin` | 实测 (`tpc-clang --version`) |
| `tpc-clang --tpcversion` | `c6df8aef53a5c0f8283b4da297e66efb0d08e329` | 实测 |
| 附带 | `tpc-llvm-objdump`, `/usr/lib/habanatools/{include/*.h, libtpcsim_shared.so, libtpc_numerics.so, …}` | 实测 (`dpkg -L`) |
| 捆绑 example/tutorial | **FAIL: 无** (`dpkg -L habanalabs-tools \| grep -i example` → 空; 无 `/opt/habanalabs*`) | 实测 |
| 附带 man page | **FAIL(质量): `man tpc-clang` 称 `-march` 只支持 `dali`(Goya) — 与二进制实测矛盾, 已过期** | 实测 |

## 4. Gaudi2 TPC 编译目标 flags (取自安装的工具链自身, 非 Goya 旧文档)

| 项 | 值 | 标签 |
|---|---|---|
| 目标选择 | `-mcpu=gaudi2` (等价可用 `-march=gaudi2`) | 实测 |
| 合法 target 列表 | `error: unknown target CPU 'bogus'` → `valid target CPU values are: gaudi, gaudib, greco, gaudi2, gaudi3` | 实测 (探针) |
| 已验证 build 行 | `tpc-clang -c src/vector_add_smoke.c -mcpu=gaudi2 -O2 -o build/vector_add_smoke.o` → exit 0, 产物 7656 B | 实测 (log: `tpc-dev/logs/build_vector_add_smoke.log`) |
| 内建头 | `tpc-defs.h` + `tpc-special.h` 强制内含, 不可按名 include (无 `tpc-api.h`); 提取法 `tpc-clang -E` | 实测 |
| 核心类型 | `float64`=64-lane f32 向量(单寄存器), `float256`=4×float64 struct | 实测 (内建头) |
| 典型 intrinsic | `v_f32_add_b(a,b,switches,0,1,0)` 等 (命名 `_b`/`_vb` 后缀; 无 `v_add_f32`) | 实测 (内建头 + 编译诊断) |
| 优化档 | `-O2` 默认/最高, `-O0` 最低, `-O1` 关 HW loop (来自 man page) | 待测 (行为未单独验证) |

## 5. vLLM 版本 pin (只读 vllm-gaudi)

| 项 | 值 | 标签 |
|---|---|---|
| plugin 仓库 | `<workspace>/vllm-gaudi` (只读访问, 未执行任何 git) | — |
| plugin SHA | `e487250f16867a39064fb81bbf2f4eb6680f536f` | 实测 (任务给定) |
| `requirements.txt` 中 vllm pin | **无** — 该文件只含 HPU 依赖 (pandas>=2.3.3, numba>=0.67.0, numpy>=1.26.0, kaldi-native-fbank>=1.22.3, tblib==3.2.2, fastapi<0.137, transformers<5.17) | 实测 (全文读取) |
| `pyproject.toml` 中 vllm pin | **无** — `name = "vllm_gaudi"`, `dynamic = ["version","dependencies",...]`, 未列 `vllm==` | 实测 (全文读取) |
| 结论 | 仓库内不存在 `vllm==`/`vllm>=` pin (requirements*.txt + pyproject.toml 全文 grep 无命中); 实际兼容 vLLM 版本由外部环境决定。**本项目既定兼容契约**：上游 vLLM commit `4be3dcf0fc7a9086d978eaea3c5a4d78fb98e44a`（repo `VLLM_STABLE_COMMIT` 契约文件）→ 版本快照见 `docs/stack-versions-p1.txt` | 实测 (grep + P1 部署验证) |
| 其它实测 pin | `transformers<5.17`, `fastapi<0.137`, `tblib==3.2.2` | 实测 |

## 6. Habana PyTorch binding (CustomOp 前置)

| 项 | 值 | 标签 |
|---|---|---|
| apt 候选包 | **FAIL: 无 `habanalabs-pytorch` / `habanalabs-libs` / 任何 torch 包** — `apt-cache search habana` 仅列出: tools, thunk, dkms, graph, firmware*, qual*, rdma-core, hypervisor*, container-runtime, perf-test 等 | 实测 |
| `habanalabs-thunk` / `habanalabs-graph` | 有 `1.24.1-482` 可装 (与 driver 同版), 但均非 PyTorch binding, 未安装 | 实测 (`apt-cache policy`) |
| vault PyPI `gaudi-python` | 仅 `habana-horovod, habana-lightning-plugins, habana-media-loader, habana-pyhlml, habana-tensorflow, hpu-media-loader, pyhlml` — **无 torch** | 实测 (`.../api/pypi/gaudi-python/simple/`) |
| 结论 | 1.24.1 PyTorch headers/libs 不可作为本机 dev 包获得 → 官方路径为容器 | 实测 |
| 容器 fallback | `docker pull vault.habana.ai/gaudi-docker/1.24.1/ubuntu22.04/habanalabs/pytorch-installer-2.11.0:latest` | **待测** — tag 来自 docs.habana.ai 1.24.1 Docker Installation 页 (websearch 实测); vault docker repo listing (`api/docker/gaudi-docker/v2/.../tags/list` 与 `_catalog`) 匿名返回 `UNAUTHORIZED`, **未能按要求从 repo listing 核实 tag** |
| 容器 repo 存在性 | `https://vault.habana.ai/artifactory/api/repositories` 匿名列出 `gaudi-docker` (packageType Docker) | 实测 |
| fallback 运行命令 (待测) | `docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all --net=host --ipc=host --cap-add=sys_nice vault.habana.ai/gaudi-docker/1.24.1/ubuntu22.04/habanalabs/pytorch-installer-2.11.0:latest` | 待测 |

## 7. 本地产物 (tpc-dev/)

| 路径 | 内容 | 标签 |
|---|---|---|
| `tpc-dev/src/vector_add_smoke.c` | 冒烟 kernel 源 | 实测 |
| `tpc-dev/build/vector_add_smoke.o` | 7656 B 编译产物 | 实测 |
| `tpc-dev/logs/build_vector_add_smoke.log` | 完整 build log, exit 0 | 实测 |
| `tpc-dev/logs/objdump_vector_add_smoke.txt` | 反汇编输出 (注: objdump 对 TPC ELF 回落 x86 助记符) | 实测 |
| `tpc-dev/notes/BUILD_NOTES.md` | 工具链推导过程 + proxy 坑 + man page 过期警告 | 实测 |
| `tpc-dev/notes/toolchain_evidence.txt` | repo/dpkg/version/madison 探针原始输出 | 实测 |
| `tpc-dev/notes/builtin_headers_extract_*.txt` | `tpc-clang -E` 提取的 tpc-defs.h/tpc-special.h 全文 | 实测 |
