# MiMo SWA normalization correction

The CPU comparison cited in commit `c7f8b5b` rounded the residual sum to BF16
before normalization. The full-model mathematical reference normalizes the
unrounded sum. Its reported `2.8e-5` attention error does not establish
accuracy against the full-model contract and is withdrawn.

On the same frozen single-rank input, the original fused grid24 kernel has
`0.001571` relative L2 error in the written key versus the corrected reference;
the earlier separate FP32 norm path produces the key bitwise. The reference
attention norm is approximately `3.87e-15`, so its relative error alone is not
a representative model-quality result. The written value remains above the
`0.001` local relative-L2 threshold on this input.

The corrected comparison receipt has SHA256
`1a53142c52377f3eb54f12452cfbb7c78f7ba43e1dff8700f8e9a3996611e2d1`.
The combined candidate's four-arm performance receipt has SHA256
`bf3392ea0c813cf8930a6d952de43a18fcd780d61dc50ec04665dac9391b8ec9`;
its first-step full-vocabulary relative L2 is `0.028297`, above the `0.02`
quality threshold. Performance alone does not qualify the candidate.
