"""Device verification for ISSUE-04: 8192 buckets, Core ATen, N-D, Gemma2/Qwen3 ops, P-TPC-01 exp."""

import math
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt
import gaudirt.compiler as compiler


def main():
    assert gaudirt.MAX_BUCKET_ROWS == 8192, gaudirt.MAX_BUCKET_ROWS
    torch.set_num_threads(4)
    torch.manual_seed(2048)
    gaudirt.initialize(hbm_bytes=512 * 1024**2, pinned_bytes=8 * 1024**2)
    dev = torch.device("gaudirt:0")

    # 1. Eager & FX 3D QK-Norm, GemmaRMSNorm, GeluAndMul, LayerNorm, Softcap_, RoPE-positions
    x3 = torch.randn(7, 8, 128, dtype=torch.bfloat16)
    w_d = torch.randn(128, dtype=torch.bfloat16)
    res3 = torch.randn(7, 8, 128, dtype=torch.bfloat16)

    def ref_rms(t, w, eps=1e-6, offset=0.0):
        tf = t.float()
        var = tf.pow(2).mean(dim=-1, keepdim=True)
        normed = tf * torch.rsqrt(var + eps)
        return (normed * (offset + w.float())).to(torch.bfloat16)

    out_qk = torch.ops.gaudirt.rms_norm.default(x3.to(dev), w_d.to(dev), 1e-6).cpu()
    exp_qk = ref_rms(x3, w_d, 1e-6, 0.0)
    assert (out_qk.float() - exp_qk.float()).abs().max().item() <= 3e-2

    out_gemma = torch.ops.gaudirt.rms_norm.default(x3.to(dev), w_d.to(dev), 1e-6, 1.0).cpu()
    exp_gemma = ref_rms(x3, w_d, 1e-6, 1.0)
    assert (out_gemma.float() - exp_gemma.float()).abs().max().item() <= 3e-2

    out_fg, out_fres = torch.ops.gaudirt.fused_add_rms_norm.default(
        x3.to(dev), res3.to(dev), w_d.to(dev), 1e-6, 1.0
    )
    sum_ref = (x3.float() + res3.float()).to(torch.bfloat16)
    exp_fg = ref_rms(sum_ref, w_d, 1e-6, 1.0)
    assert (out_fres.cpu().float() - sum_ref.float()).abs().max().item() <= 2e-2
    assert (out_fg.cpu().float() - exp_fg.float()).abs().max().item() <= 3e-2

    gate_up = torch.randn(13, 512, dtype=torch.bfloat16)
    for approx in ("none", "tanh"):
        act = torch.ops.gaudirt.gelu_and_mul.default(gate_up.to(dev), approx).cpu()
        g, u = gate_up.float().chunk(2, dim=-1)
        exp_act = (torch.nn.functional.gelu(g, approximate=approx) * u).to(torch.bfloat16)
        assert (act.float() - exp_act.float()).abs().max().item() <= 4e-2, approx

    # Standalone rotary_embedding_positions (2D & 3D)
    cos_sin = torch.randn(64, 64, dtype=torch.bfloat16)
    pos = torch.tensor([0, 3, 7, 15, 31], dtype=torch.int32)
    q_r = torch.randn(5, 4 * 128, dtype=torch.bfloat16)
    k_r = torch.randn(5, 2 * 128, dtype=torch.bfloat16)
    qo, ko = torch.ops.gaudirt.rotary_embedding_positions.default(
        q_r.to(dev), k_r.to(dev), cos_sin.to(dev), pos.to(dev), 128, 64, True
    )
    assert qo.shape == q_r.shape and ko.shape == k_r.shape
    assert torch.isfinite(qo.cpu()).all() and torch.isfinite(ko.cpu()).all()

    # 2. P-TPC-01 safe exp around the v_exp_f32 boundary (-8.0 .. -16.0)
    exp_in = torch.tensor([-25.0, -15.5, -10.5, -8.1, -1.0, 0.0, 2.5], dtype=torch.float32)
    exp_dev = torch.exp(exp_in.to(dev)).cpu()
    assert torch.allclose(exp_dev, torch.exp(exp_in), rtol=2e-3, atol=1e-6), (exp_dev, torch.exp(exp_in))

    # 3. Compiled FX model exercising inline 0D scalar folding, Core ATen ops, softcap_, 3D norm, RoPE-pos
    norm_w = torch.randn(128, dtype=torch.bfloat16).to(dev)
    ln_w = torch.randn(128, dtype=torch.bfloat16).to(dev)
    ln_b = torch.randn(128, dtype=torch.bfloat16).to(dev)
    proj_w = (torch.randn(256, 128, dtype=torch.bfloat16) * 0.05).to(dev)
    cache_dev = cos_sin.to(dev)

    def core_model(hidden, q_in, k_in, pos_in, mask):
        inv_rms_norm = torch.tensor(math.sqrt(0.5), dtype=torch.bfloat16)
        h = hidden * inv_rms_norm
        h = torch.ops.gaudirt.rms_norm.default(h, norm_w, 1e-6, 1.0)
        h = torch.nn.functional.layer_norm(h, (128,), ln_w, ln_b, 1e-5)
        h3 = h.view(-1, 2, 64)
        h3 = torch.ops.gaudirt.rms_norm.default(h3, norm_w[:64], 1e-6)
        h = h3.reshape(-1, 128)
        q_out, k_out = torch.ops.gaudirt.rotary_embedding_positions.default(
            q_in, k_in, cache_dev, pos_in, 128, 64, True
        )
        probs = torch.softmax(torch.matmul(h, proj_w.t()), dim=-1)
        gated = torch.ops.gaudirt.gelu_and_mul.default(probs, "tanh")
        clamped = torch.clamp(gated, min=-10.0, max=10.0)
        capped = 30.0 * torch.tanh(clamped / 30.0)
        filtered = torch.where(mask, capped, torch.zeros_like(capped))
        torch.ops.gaudirt.softcap_.default(filtered, 25.0)
        summary = filtered.sum(dim=-1, keepdim=True) + torch.exp(filtered[:, :1] - 10.0)
        return summary, q_out, k_out

    compiled_model = compiler.compile(core_model, dynamic_arg_indices=(0, 1, 2, 3, 4))
    for rows in (5, 19):
        h_cpu = torch.randn(rows, 128, dtype=torch.bfloat16)
        q_cpu = torch.randn(rows, 256, dtype=torch.bfloat16)
        k_cpu = torch.randn(rows, 128, dtype=torch.bfloat16)
        p_cpu = torch.arange(rows, dtype=torch.int32)
        m_cpu = (torch.arange(rows * 128) % 3 != 0).reshape(rows, 128)

        s_dev, q_out_d, k_out_d = compiled_model(
            h_cpu.to(dev), q_cpu.to(dev), k_cpu.to(dev), p_cpu.to(dev), m_cpu.to(dev)
        )
        assert s_dev.shape == (rows, 1) and torch.isfinite(s_dev.cpu()).all()
        assert torch.isfinite(q_out_d.cpu()).all() and torch.isfinite(k_out_d.cpu()).all()

    # 3b. Qwen3 apply_qk_norm (qkv.split -> reshape(-1, head_dim) -> rms_norm -> view(q.size())) + Gemma2 *= scalar + pow(2)
    qk_norm_w = torch.randn(64, dtype=torch.bfloat16).to(dev)
    qkv_w = (torch.randn(512, 128, dtype=torch.bfloat16) * 0.05).to(dev)

    def qwen3_gemma2_patterns(hidden, qkv_weight, norm_w):
        normalizer = torch.tensor(128**0.5, dtype=torch.bfloat16)
        h = hidden.clone()
        h *= normalizer
        qkv = torch.nn.functional.linear(h, qkv_weight)
        q, k, v = qkv.split([256, 128, 128], dim=-1)
        qh = q.reshape(-1, 64)
        qh = torch.ops.gaudirt.rms_norm.default(qh, norm_w, 1e-6)
        q_normed = qh.view(q.size())
        kh = k.reshape(k.size(0), -1)
        v_normed = v * torch.rsqrt(v.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(v.dtype)
        return h, q_normed, kh, v_normed

    compiled_patterns = compiler.compile(qwen3_gemma2_patterns, dynamic_arg_indices=(0,))
    bf16_norm = torch.tensor(128**0.5, dtype=torch.bfloat16)
    qkv_w_cpu = qkv_w.cpu()
    qk_norm_w_cpu = qk_norm_w.cpu()
    for rows in (32, 1, 16):
        h_cpu = torch.randn(rows, 128, dtype=torch.bfloat16)
        h_dev, q_dev, k_dev, v_dev = compiled_patterns(h_cpu.to(dev), qkv_w, qk_norm_w)

        h_ref = (h_cpu * bf16_norm).to(torch.bfloat16)
        assert (h_dev.cpu().float() - h_ref.float()).abs().max().item() <= 5e-2
        qkv_ref = torch.nn.functional.linear(h_ref, qkv_w_cpu)
        q_ref_in, k_ref_in, v_ref_in = qkv_ref.split([256, 128, 128], dim=-1)
        q_ref = ref_rms(q_ref_in.contiguous().view(rows, 4, 64), qk_norm_w_cpu, 1e-6, 0.0).view(rows, 256)
        assert (q_dev.cpu().float() - q_ref.float()).abs().max().item() <= 4e-2
        assert (k_dev.cpu().float() - k_ref_in.float()).abs().max().item() <= 4e-2
        v_ref = v_ref_in * torch.rsqrt(v_ref_in.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(torch.bfloat16)
        assert (v_dev.cpu().float() - v_ref.float()).abs().max().item() <= 4e-2
    p_stats_before = gaudirt.recipe_cache_stats()
    _ = compiled_patterns(torch.randn(1, 128, dtype=torch.bfloat16).to(dev), qkv_w, qk_norm_w)
    _ = compiled_patterns(torch.randn(16, 128, dtype=torch.bfloat16).to(dev), qkv_w, qk_norm_w)
    p_stats_after = gaudirt.recipe_cache_stats()
    assert p_stats_after[1] == p_stats_before[1], (p_stats_before, p_stats_after)

    # 4. 8192-row dynamic bucket capacity & steady-state zero recompile
    def mlp_8k(x, w):
        return torch.nn.functional.linear(x, w)

    compiled_8k = compiler.compile(mlp_8k, dynamic_arg_indices=(0,))
    w_8k = (torch.randn(128, 128, dtype=torch.bfloat16) * 0.05).to(dev)
    x_8k_1 = torch.randn(5000, 128, dtype=torch.bfloat16).to(dev)
    y_8k_1 = compiled_8k(x_8k_1, w_8k)
    stats_before = gaudirt.recipe_cache_stats()
    x_8k_2 = torch.randn(8192, 128, dtype=torch.bfloat16).to(dev)
    y_8k_2 = compiled_8k(x_8k_2, w_8k)
    stats_after = gaudirt.recipe_cache_stats()
    assert y_8k_1.shape == (5000, 128) and y_8k_2.shape == (8192, 128)
    assert stats_after[1] == stats_before[1], (stats_before, stats_after)
    gaudirt.synchronize()
    print("fx_core_aten_convergence_smoke PASSED; recipe_stats=", stats_after, flush=True)


if __name__ == "__main__":
    main()
