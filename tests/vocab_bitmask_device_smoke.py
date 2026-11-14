"""Device verification for Phase 5B: GaudiRT device-side vocabulary bitmask operator."""

import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt


def ref_apply_bitmask(logits: torch.Tensor, bitmask: torch.Tensor, indices=None, vocab_size: int = -1, mask_tail: bool = False):
    out = logits.clone()
    is_1d = out.dim() == 1
    work = out.unsqueeze(0) if is_1d else out
    bm = bitmask.unsqueeze(0) if bitmask.dim() == 1 else bitmask
    batch, cols = work.shape
    mask_bits = bm.shape[1] * 32
    active = min(cols if vocab_size < 0 else vocab_size, mask_bits)
    rows = list(range(batch)) if indices is None else [int(x) for x in indices.tolist()]
    for r in rows:
        w = bm[r]
        col_idx = torch.arange(cols, dtype=torch.int64)
        mask_active = col_idx < active
        word_idx = (col_idx.clamp(max=active - 1) >> 5).to(torch.int64)
        bit_off = (col_idx & 31).to(torch.int32)
        bits = (w[word_idx] >> bit_off) & 1
        disallowed = torch.where(mask_active, bits == 0, torch.full_like(mask_active, mask_tail))
        work[r, disallowed] = float("-inf")
    return out


def make_bitmask(rows: int, mask_words: int, allowed_lists):
    bm = torch.zeros((rows, mask_words), dtype=torch.int32)
    for r, allowed in enumerate(allowed_lists):
        if allowed is None:
            bm[r].fill_(-1)
        else:
            for tok in allowed:
                bm[r, tok >> 5] |= 1 << (tok & 31)
    return bm


def _run_checks(dev: torch.device) -> None:
    # 1. Mixed BF16 & FP32 across odd/padded/large vocabularies & non-power-of-2 batches.
    for dtype in (torch.bfloat16, torch.float32):
        for batch, vocab in ((1, 35), (3, 1025), (13, 49152), (5, 166912)):
            words = (vocab + 31) // 32
            logits_cpu = (torch.randn((batch, vocab), dtype=torch.float32) * 3.0).to(dtype)
            allowed = [
                None if r % 3 == 0 else [0, 31 % vocab, 32 % vocab, (vocab - 1)]
                for r in range(batch)
            ]
            bm_cpu = make_bitmask(batch, words, allowed)
            ref = ref_apply_bitmask(logits_cpu, bm_cpu)

            # Test with bitmask on device (pinned H2D) and directly on CPU.
            for bm_in in (bm_cpu.pin_memory().to(dev, non_blocking=True), bm_cpu):
                cand_dev = logits_cpu.clone().to(dev)
                ret = torch.ops.gaudirt.apply_token_bitmask_inplace_.default(cand_dev, bm_in)
                assert ret is cand_dev
                del ret
                cand = cand_dev.cpu()
                assert torch.equal(cand.view(torch.uint8), ref.view(torch.uint8)), (dtype, batch, vocab)
                del cand_dev
                if bm_in.device.type == "gaudirt":
                    del bm_in

    # 2. Padded tail semantics: mask_tail=False (xgrammar) vs mask_tail=True (llguidance).
    for mask_tail in (False, True):
        logits_cpu = torch.linspace(-2.0, 2.0, 80, dtype=torch.bfloat16).reshape(2, 40)
        bm_cpu = torch.tensor([[0b101], [-1]], dtype=torch.int32)  # covers 32 of 40 cols
        ref = ref_apply_bitmask(logits_cpu, bm_cpu, mask_tail=mask_tail)
        cand_dev = logits_cpu.to(dev)
        torch.ops.gaudirt.apply_token_bitmask_inplace_.default(cand_dev, bm_cpu.to(dev), None, -1, mask_tail)
        assert torch.equal(cand_dev.cpu().view(torch.uint8), ref.view(torch.uint8)), mask_tail
        del cand_dev

    # 3. 1D logits + strided view slice preserving neighboring sentinel columns.
    logits_1d = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
    bm_1d = torch.tensor([0b10101], dtype=torch.int32)
    dev_1d = logits_1d.to(dev)
    torch.ops.gaudirt.apply_token_bitmask_inplace_.default(dev_1d, bm_1d)
    assert torch.equal(dev_1d.cpu(), torch.tensor([1.0, float("-inf"), 3.0, float("-inf"), 5.0]))
    del dev_1d

    full_storage = torch.full((3, 192), 77.0, dtype=torch.bfloat16)
    full_storage[:, :130] = torch.randn((3, 130), dtype=torch.bfloat16)
    full_dev = full_storage.clone().to(dev)
    sliced_dev = full_dev[:, :130]  # non-contiguous stride
    bm_slice = make_bitmask(3, (130 + 31) // 32, [[1, 65], None, [129]])
    torch.ops.gaudirt.apply_token_bitmask_inplace_.default(sliced_dev, bm_slice.to(dev))
    full_after = full_dev.cpu()
    assert torch.equal(full_after[:, 130:], full_storage[:, 130:]), "corrupted neighboring sentinel columns"
    ref_slice = ref_apply_bitmask(full_storage[:, :130], bm_slice)
    assert torch.equal(full_after[:, :130].view(torch.uint8), ref_slice.view(torch.uint8))
    del sliced_dev, full_dev

    # 4. Indices subset masking (including duplicate index deduplication semantics matching xgrammar).
    logits_idx = torch.arange(4 * 64, dtype=torch.float32).reshape(4, 64)
    bm_idx = make_bitmask(4, 2, [None, [3], [5, 63], [1]])
    idx_t = torch.tensor([2, 0, 2], dtype=torch.int32)
    dev_idx = logits_idx.to(dev)
    torch.ops.gaudirt.apply_token_bitmask_inplace_.default(dev_idx, bm_idx, idx_t)
    ref_idx = ref_apply_bitmask(logits_idx, bm_idx, indices=idx_t)
    assert torch.equal(dev_idx.cpu(), ref_idx)
    del dev_idx

    # 5. End-to-end composition with sample_from_logits on masked distributions.
    logits_samp = torch.full((4, 1025), 5.0, dtype=torch.bfloat16)
    bm_samp = make_bitmask(4, (1025 + 31) // 32, [[17], [1024], [31, 32], None])
    dev_samp = logits_samp.to(dev)
    torch.ops.gaudirt.apply_token_bitmask_inplace_.default(dev_samp, bm_samp.to(dev))
    toks = torch.ops.gaudirt.sample_from_logits.default(
        dev_samp,
        torch.tensor([0.0, 0.8, 1.0, 0.0], dtype=torch.float32, device=dev),
        torch.tensor([1.0, 0.9, 0.95, 1.0], dtype=torch.float32, device=dev),
        torch.tensor([1025, 1025, 1025, 1025], dtype=torch.int32, device=dev),
        torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=dev),
        torch.tensor([42, 42, 42, 42], dtype=torch.int64, device=dev),
        torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=dev),
        True,
    ).cpu()
    assert toks[0].item() == 17
    assert toks[1].item() == 1024
    assert toks[2].item() in (31, 32)
    assert toks[3].item() == 0
    del dev_samp

    # Negative contract check (Fail-Fast).
    bad_dev = torch.zeros((2, 40), dtype=torch.bfloat16, device=dev)
    bad_bm = torch.full((2, 1), -1, dtype=torch.int32)
    raised = False
    try:
        torch.ops.gaudirt.apply_token_bitmask_inplace_.default(bad_dev, bad_bm, None, 40, False)
    except RuntimeError:
        raised = True
    assert raised, "expected RuntimeError when mask_tail=False and bitmask smaller than vocab_size"
    del bad_dev

    gaudirt.synchronize()
    print("vocab_bitmask_device_smoke PASSED", flush=True)


def main():
    torch.manual_seed(20261113)
    gaudirt.initialize(hbm_bytes=512 * 1024**2, pinned_bytes=16 * 1024**2)
    dev = torch.device("gaudirt:0")

    try:
        _run_checks(dev)
    finally:
        gaudirt.shutdown()


if __name__ == "__main__":
    main()
