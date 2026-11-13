# GaudiRT

面向 Intel Gaudi2 的零历史包袱 PyTorch 2 原生 `PrivateUse1`（设备名 `gaudirt`）运行时与分段图编译执行器。

## 核心契约与架构边界

1. **零旧桥隔离**：全程要求 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`。C++ 源码与构建产物不引入任何 `gaudi-pytorch-bridge` 头文件或动态库，Python 侧禁止导入 `habana_frameworks` 或启用 `PT_HPU_LAZY_MODE`。
2. **Fail-Fast 与无Fallback**：越界形状、设备不匹配、不支持算子或非法通信组在入口处直接报错终止，不做静默回退。
3. **框架解耦**：`GaudiRT` 本身不引用任何上层推理框架（如 SGLang），上层自定义集合通信算子与通信组解析器通过 `gaudirt.register_collective_op` 与 `gaudirt.register_comm_group_resolver` 在插件初始化阶段注册。

## 目录结构

- `include/gaudirt/` & `src/`: 设备运行时（`DeviceRuntime`）、流序块分配器（`BlockCachingAllocator` / `GraphPool`）、`EventPool`、`RecipeCache`（多进程原子落盘）与本地 C++ 测试底座。
- `src/bridge/`: 统一构图与分段桥接层（`SynapseGraphBuilder`、`BucketProgram`、`SegmentedRunner`、统一带 `weight_offset` 的 `rms_norm` / `fused_add_rms_norm`、统一 `launch_cold_recipe` / `binary_ewise_op` Eager 算子、通用 `PagedAttention`、`sampler_vocab` 与 HCCL `ProcessGroupGaudiRT`），消费 `gaudi-kernels` `generic_dense` 40 个 GUID 契约（含统一 `gk_kv_store_head_major_bf16_v1`）。
- `python/gaudirt/`: PyTorch 2 `PrivateUse1` 后端与 `torch.compile` 编译前端（`compiler.py`、`fx_collectives.py`、`fx_canonicalize.py`、`fx_passes.py`、`op_lowering.py` / `op_lowering_helpers.py`、`partitioner.py`、`segment_binding.py`）。
- `tests/`: C++ 单元/真机验证测试与 Python 算子/FX/多卡集合通信冒烟测试。
- `docs/DESIGN.md`: 完整架构设计规范（顶层全栈路线见仓库外层 `ROADMAP.md`，硬件/SDK 踩坑总台账见 `docs/gaudi2-pitfalls.md`）。

## 构建与本地测试（Gaudi2 主机）

构建依赖环境已激活的 Synapse SDK、PyTorch 以及已编译的 `gaudi-kernels` 共享库（`libgaudi_kernels_bundle.so`）：

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
cmake -B build -DGAUDIRT_BUILD_TORCH=ON
cmake --build build -j$(nproc)
```

详细编译选项、C++ 单元测试与真机冒烟清单请参考 [CMakeLists.txt](CMakeLists.txt) 与 [docs/DESIGN.md](docs/DESIGN.md)。
