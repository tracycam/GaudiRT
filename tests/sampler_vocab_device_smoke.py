"""Device verification for ISSUE-05: GaudiRT device-side stochastic + greedy vocabulary sampler."""

import math
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt


def main():
    torch.set_num_threads(4)
    torch.manual_seed(5501)
    gaudirt.initialize(hbm_bytes=512 * 1024**2, pinned_bytes=8 * 1024**2)
    dev = torch.device("gaudirt:0")

    for dtype in (torch.bfloat16, torch.float32):
        for vocab in (127, 4099, 131072):
            bs = 8
            logits = torch.full((bs, vocab), -20.0, dtype=dtype)
            # Row 0: pure greedy via temp=0
            logits[0, 5] = 4.0
            logits[0, 17] = 9.0
            # Row 1: tied greedy (lowest index wins)
            logits[1, 3] = 7.0
            logits[1, 100] = 7.0
            # Row 2: top_k=2 strictly restricts support to {11, 22} (11 and 22 tied-close, 33 slightly below)
            logits[2, 11] = 10.0
            logits[2, 22] = 10.0
            logits[2, 33] = 9.0
            # Row 3: top_p=0.5 with dominant weight at 42
            logits[3, 42] = 12.0
            logits[3, 43] = 2.0
            # Row 4: min_p=0.5 removes weaker tokens relative to prob_max (token 77 keeps only itself)
            logits[4, 77] = 10.0
            logits[4, 78] = 8.0
            # Row 5: NaN poisoning safety (NaN must never be selected)
            logits[5, 0] = float("nan")
            logits[5, 9] = float("nan")
            logits[5, 51] = 5.0
            # Row 6 & 7: general stochastic over first 8 tokens
            logits[6, :8] = torch.linspace(1.0, 2.0, 8, dtype=dtype)
            logits[7, :8] = torch.linspace(2.0, 1.0, 8, dtype=dtype)

            temps = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.9, 1.2], dtype=torch.float32, device=dev)
            top_ps = torch.tensor([1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 0.95, 1.0], dtype=torch.float32, device=dev)
            top_ks = torch.tensor([vocab, vocab, 2, vocab, vocab, vocab, 6, 4], dtype=torch.int32, device=dev)
            min_ps = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.1], dtype=torch.float32, device=dev)

            seeds_a = torch.full((bs,), 1234567, dtype=torch.int64, device=dev)
            pos_a = torch.arange(bs, dtype=torch.int64, device=dev)

            logits_dev = logits.to(dev)
            out_a1 = torch.ops.gaudirt.sample_from_logits.default(
                logits_dev, temps, top_ps, top_ks, min_ps, seeds_a, pos_a, True
            ).cpu()
            out_a2 = torch.ops.gaudirt.sample_from_logits.default(
                logits_dev, temps, top_ps, top_ks, min_ps, seeds_a, pos_a, True
            ).cpu()
            assert torch.equal(out_a1, out_a2), "Deterministic seed+position reproducibility failed"
            assert out_a1[0].item() == 17
            assert out_a1[1].item() == 3
            assert out_a1[2].item() in (11, 22)
            assert out_a1[3].item() == 42
            assert out_a1[4].item() == 77
            assert out_a1[5].item() == 51
            assert 2 <= out_a1[6].item() < 8
            assert 0 <= out_a1[7].item() < 4

            # Check support bounds across many rounds
            for step in range(40):
                pos_s = pos_a + (step + 1) * 10
                tok = torch.ops.gaudirt.sample_from_logits.default(
                    logits_dev, temps, top_ps, top_ks, min_ps, seeds_a, pos_s, True
                ).cpu()
                assert tok[2].item() in (11, 22), tok[2]
                assert tok[4].item() == 77, tok[4]
                assert tok[5].item() == 51, tok[5]
                assert 2 <= tok[6].item() < 8, tok[6]
                assert 0 <= tok[7].item() < 4, tok[7]

    # Empirical multinomial distribution test (2048 samples over 5 non-uniform outcomes)
    vocab_emp = 512
    row_logits = torch.full((vocab_emp,), -50.0, dtype=torch.float32)
    base_weights = torch.tensor([3.0, 2.2, 1.5, 0.8, 0.2], dtype=torch.float32)
    row_logits[:5] = base_weights
    expected_probs = torch.softmax(base_weights, dim=-1)

    B = 64
    batch_logits = row_logits.unsqueeze(0).expand(B, vocab_emp).contiguous().to(dev)
    t_ones = torch.ones((B,), dtype=torch.float32, device=dev)
    p_ones = torch.ones((B,), dtype=torch.float32, device=dev)
    k_all = torch.full((B,), vocab_emp, dtype=torch.int32, device=dev)
    m_zero = torch.zeros((B,), dtype=torch.float32, device=dev)

    counts = torch.zeros(5, dtype=torch.int64)
    total_samples = 0
    for r in range(32):
        seeds_r = torch.full((B,), 9000 + r, dtype=torch.int64, device=dev)
        pos_r = torch.arange(B, dtype=torch.int64, device=dev)
        sampled = torch.ops.gaudirt.sample_from_logits.default(
            batch_logits, t_ones, p_ones, k_all, m_zero, seeds_r, pos_r, True
        ).cpu()
        assert (sampled >= 0).all() and (sampled < 5).all(), sampled
        for idx in sampled.tolist():
            counts[idx] += 1
        total_samples += B

    empirical = counts.float() / float(total_samples)
    tvd = 0.5 * (empirical - expected_probs).abs().sum().item()
    assert tvd < 0.035, (empirical.tolist(), expected_probs.tolist(), tvd)
    gaudirt.synchronize()
    print(f"sampler_vocab_device_smoke PASSED; empirical={ [round(x, 4) for x in empirical.tolist()] } "
          f"expected={ [round(x, 4) for x in expected_probs.tolist()] } tvd={tvd:.4f}", flush=True)


if __name__ == "__main__":
    main()
