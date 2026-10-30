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

The FP32-residual fused variant matches the separate FP32 norm path bitwise on
that single-rank input, including the written KV values. One unprofiled
70-layer run measured `11.368715 ms` p50 and `11.342459 ms` mean, with first-step
`max_kl=0.001260` and `relative_l2=0.024301`; neither the historical latency
target nor the quality threshold was met. This single run is not an ABBA
comparison. Its receipt SHA256 is
`df81e2a536ba4d2f30ed0d8f68f9c1f878505161d41d4979934ee51020af6583`.

A separate first-step diagnostic preserved the candidate logits bitwise and
checked only layers 6 and 7. Their selected experts matched both FP64 routing
on the candidate input and the full-model mathematical reference; this does
not establish the routing of other layers. The diagnostic receipt SHA256 is
`17ffe228509191bf9867cc8cfd5056231f7fd11afad5632acb9c5bd3a73e2048`.

Of eight later sampled route sets, only layer 30 differed from the full-model
reference: the candidate selected expert 127 where the reference selected 316.
FP64 routing on the candidate's own layer-30 input selected the same set as
the candidate. That input's relative L2 against the full mathematical
trajectory was `0.07207`, so the observed divergence precedes this router.
These samples do not establish which earlier layer caused it. Their receipts
have SHA256 `35fb0e3ed3fed62c67512f546644ee24e72ae663eb3ce6cdf61bb464a80591f5`
and `c4a990cc855b7d5dd208efb82068fa28ac9237e5165f980f1d8e0ad9df96323a`.
