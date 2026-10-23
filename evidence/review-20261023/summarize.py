"""Summarize the captured review experiments; does not access an accelerator."""
from pathlib import Path
import hashlib
import json
import re

base = Path(__file__).resolve().parent
arms = {}
for name in ("baseline-a1", "instrumented-b1", "instrumented-b2", "baseline-a2"):
    log = (base / f"{name}.log").read_text()
    assert log.count("bitwise_mismatches=0 words=4096") == 3, name
    match = re.search(r"batch=256 trials=32 (.*)", log)
    assert match, name
    arms[name] = {key: float(value) for key, value in re.findall(r"(\w+)=([\d.]+)", match[1])}
baseline = sum(arms[f"baseline-a{i}"]["enqueue_mean_us"] for i in (1, 2)) / 2
instrumented = sum(arms[f"instrumented-b{i}"]["enqueue_mean_us"] for i in (1, 2)) / 2
graph = json.loads((base / "sram-mme-post.json").read_text())["graphs"][0]
nodes = [{key: node.get(key) for key in ("name", "engine", "guid")} for node in graph["nodes"]]
assert {n["engine"] for n in nodes} == {"TPC", "MME"}
tensor = next(t for t in graph["tensors"] if t["name"] == "sram_sum")
assert tensor["allocation"] == "SRAM"
assert tensor["offset"] == 0x1000FFFFFD080000
assert (base / "sram-mme.log").read_text().count("bitwise_mismatches=0") == 2
fused = json.loads((base / "rmw-post.json").read_text())["graphs"][0]
assert len(fused["nodes"]) == 1
assert len(fused["nodes"][0]["fused_node_graph"]["nodes"]) == 2

summary = {
    "scope": "Independent review of GaudiRT design; not a GaudiRT implementation or full-model benchmark",
    "host": "ai@192.168.100.153",
    "hardware_module_for_compute": 0,
    "sdk_package_version": "1.24.2-481; exact binary hashes are in environment.txt",
    "timing_method": {
        "program": "launch_probe.cpp, native C++ Synapse API, fixed input/output addresses",
        "graph": "FP32 Add followed by ReLU, 4096 elements; compiler fuses the two nodes",
        "warmup_launches": 256,
        "reported_batch_size": 256,
        "batches_per_arm": 32,
        "synchronization": "Completion wait after each batch, outside enqueue timer",
        "cpu_affinity": "CPU 2, NUMA node 0, local to module 0",
        "instrumentation": "Original first-principles-bridge-20261022/synapse_launch_breakdown.so, no bypass flags",
        "percentile_scope": "Percentiles of batch means, not individual launches",
        "validation": "4096 FP32 words bit-exact after warmup, after timing, and after input refresh in all arms",
    },
    "abba_arms": arms,
    "baseline_enqueue_mean_us": baseline,
    "instrumented_enqueue_mean_us": instrumented,
    "instrumentation_relative_increase_percent": (instrumented / baseline - 1) * 100,
    "sram_to_mme": {
        "program": "sram_mme_probe.cpp",
        "nodes": nodes,
        "intermediate_name": tensor["name"],
        "intermediate_allocation": tensor["allocation"],
        "intermediate_address": hex(tensor["offset"]),
        "intermediate_shape": tensor["max_shape"],
        "numerical_check": "Two changed-input fixtures, 1024 BF16 outputs each, identity GEMM, bit-exact",
        "limits": "Proves compiled placement and execution correctness, not SRAM bandwidth or physical transaction counts",
    },
    "fusion": {
        "requested_nodes": ["add_fwd_f32", "relu_fwd_f32"],
        "compiled_node_count": len(fused["nodes"]),
        "compiled_guid": fused["nodes"][0]["guid"],
    },
    "limitations": [
        "The documented 6401-launch/22.291-us original workload was not reproduced",
        "No independent GaudiRT SCAL replay implementation was built or timed",
        "No new full-model, multi-device, SRAM bandwidth, or EOS-loop performance claim is made",
        "Public and archived protocol headers require version matching to the installed firmware before use",
        "plain-a.log is a failed setup attempt without the required kernel library environment; excluded from measurements",
        "rmw-placement graph fuses away the intermediate; only sram-mme-post.json proves SRAM placement",
    ],
}
(base / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
root = base.parents[2]
sources = [
    "GaudiRT/docs/DESIGN.md", "GaudiRT/docs/software-autopsy.md",
    "gaudi-pytorch-bridge/python_packages/habana_frameworks/torch/hpu/graphs.py",
    "gaudi-pytorch-bridge/python_packages/habana_frameworks/torch/dynamo/compile_backend/recipe_compiler.py",
    "gaudi-pytorch-bridge/python_packages/habana_frameworks/torch/dynamo/compile_backend/recipe_compiler.cpp",
    "gaudi-pytorch-bridge/backend/synapse_helpers/device_memory.cpp",
    "gaudi-kernels-public/csrc/ops/mxfp4_compact.cpp",
    "instance-assets/device-autonomy-audit-20261011/archived-hcl-source/dependencies/qman_fw/engines-arc/include/gaudi2_arc_eng_packets.h",
    "instance-assets/device-autonomy-audit-20261011/archived-hcl-source/dependencies/qman_fw/engines-arc/include/gaudi2_arc_sched_packets.h",
    "instance-assets/numerics-contract-20260925/workspace/deps/HCL/hcl/src/platform/gen2_arch_common/hcl_graph_sync.cpp",
    "instance-assets/numerics-contract-20260925/workspace/deps/HCL/hcl/src/platform/gen2_arch_common/hcl_device_controller.cpp",
]
manifest = {
    "reviewed_source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources},
    "evidence_sha256": {str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(base.rglob("*")) if p.is_file() and p.name != "manifest.json"},
}
(base / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"baseline_us": baseline, "instrumented_us": instrumented,
                  "instrumentation_percent": summary["instrumentation_relative_increase_percent"],
                  "sram_address": hex(tensor["offset"]), "validation": "passed"}, indent=2))
