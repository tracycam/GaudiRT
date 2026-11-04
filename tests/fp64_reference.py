"""Direct binary64 to BF16 round-to-nearest-even for CPU references."""

import numpy as np
import torch


def direct_bf16(values):
    data = np.asarray(values.detach().cpu().numpy(), dtype=np.float64)
    if not np.isfinite(data).all():
        raise ValueError("nonfinite FP64 reference")
    shape = data.shape
    data = np.ascontiguousarray(data).reshape(-1)
    bits = data.view(np.uint64)
    sign = ((bits >> np.uint64(48)) & np.uint64(0x8000)).astype(np.uint16)
    exponent = (bits >> np.uint64(52)) & np.uint64(0x7ff)
    fraction = bits & np.uint64((1 << 52) - 1)
    out = np.zeros(bits.shape, dtype=np.uint16)
    normal = (exponent >= 897) & (exponent <= 1150)
    payload = ((exponent[normal] - np.uint64(896)) << np.uint64(7)) | (fraction[normal] >> np.uint64(45))
    tail = fraction[normal] & np.uint64((1 << 45) - 1)
    halfway = np.uint64(1 << 44)
    payload += (tail > halfway) | ((tail == halfway) & ((payload & np.uint64(1)) != 0))
    out[normal] = payload.astype(np.uint16)
    small = exponent < 897
    out[small] = np.rint(np.ldexp(np.abs(data[small]), 133)).astype(np.uint16)
    out[exponent > 1150] = np.uint16(0x7f80)
    out |= sign
    rounded = (out.astype(np.uint32) << np.uint32(16)).view(np.float32).reshape(shape)
    return torch.from_numpy(rounded).double()
