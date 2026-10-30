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

Replacing only the MoE gate activation with an FP32 variant passed its
single-card arithmetic check, but the full-model first-step relative L2 rose
from `0.024301` to `0.025228`; the one-run p50 was `11.280414 ms`. This is not
an ABBA performance result and quality still failed, so the replacement was
reverted. Its receipt SHA256 is
`e2fd6a610a143e5f86ae5425dc27794b69e34a60c3183e84cb0dc56c3d643c9c`.

A 31-layer diagnostic prefix matched the complete candidate's layer-30
router input bitwise. Within sampled layers 21–30, the first expert-set
difference was at layer 25; FP64 routing on that same candidate input selected
the candidate's set. The layer-25 input differed from the rolling mathematical
reference by `0.016289` relative L2. The receipt SHA256 is
`53b3b4f15d6e957adc87863305362b9a66d3bb24ac0cb0bee0aba22c7664cb0f`.

A 22-layer prefix also matched the verified prefix's layer-21 router input
bitwise. All sampled layers 11–21 chose the reference expert sets, but input
relative L2 increased from `0.000560` at layer 17 to `0.012680` at layer 21.
This identifies an earlier numerical trajectory difference, not its cause.
The receipt SHA256 is
`9fc3fc867444b7014ed84c41b940576e449e7c47f10ce45d2df4a0b17637bb98`.

On the captured normal-context layer-18 router input, a streamed-weight
single-rank MoE probe reproduced the candidate input and selected experts
bitwise. Its local output had `0.004364` relative L2 versus FP64 gate math;
the FP64 reference with the current BF16 gate boundaries differed by
`1.27e-7`. This local result does not establish an end-to-end improvement.
The receipt SHA256 is
`68a8d47d7014b2e4f6c1cb5d22a9bc802ea84573bae06524607f9e6f1716475e`.
With FP32 gate arithmetic, the same frozen input and expert set produced a
rank-0 local MoE relative L2 of `1.26e-7` versus FP64. Its receipt SHA256 is
`8acc465d322029be1319bbcfbee5ab28b3467f3c43b28d04f9f4486af6439a6c`.

On a shared frozen input at normal-context layer 18, rank-0 SWA attention
had `0.001819` relative L2 versus the CPU mathematical result. Substituting
the device-written KV in that reference reduced the difference to `0.000052`;
the previous value write differed by `0.003873` relative L2. After selecting
FP32 value scaling, the local output, key and value matched that reference
bitwise, and untouched KV bytes were unchanged. These are single-layer
results, not whole-model quality or performance. The before and after receipt
SHA256 values are
`81b71b04ef0f4b48934dd554d9b0f4a795df86efc12bece9d1eab49716271a2b` and
`350e5e4306c15215aec7961a937dbb707b2dcb57f1d482a0781a675de2a59c1c`.

One unprofiled 70-layer run combining FP32 SWA normalization, value scaling,
and MoE gate arithmetic measured `11.0428975 ms` p50 and `11.063560 ms` mean.
All eight ranks agreed on 144 generated tokens and all untouched KV bytes
were unchanged after generation. First-step `max_kl=0.000976` and
`relative_l2=0.021370`; the latter still exceeds the unchanged `0.02` gate.
This single run is not an ABBA result and did not audit KV after each step.
Its receipt SHA256 is
`e89f66017d06c28ebb8788067f548e1a11efbb7792758687adc5c544ffe36754`.

A separate untimed run used the same graph, matched that run's first-step
logits bitwise and reproduced its first 128 tokens on all ranks. It checked
each step's key and value slot on all 70 layers, verified no later overwrite,
and found untouched KV bytes unchanged. The quality gate still failed.
The audit receipt SHA256 is
`cc41fcf0d5fabcc1394b370da9983f1995325f58271e52eae3d4e3b3ae1ed803`.

After those runs, fixed page-table checks were moved to model initialization.
The metadata unit test covers the 144 positions across a page boundary and
rejects a duplicate future page. The updated runtime extension compiles; this
subsequent change has not yet been measured in an end-to-end run.

An independent 8-rank, 70-layer teacher-forced check used the mathematical
trajectory's frozen inputs and standalone unfused layer graphs. Forty-five
layer outputs exceeded `0.001` relative L2; the largest was layer 44 at
`0.002572`. This check did not record candidate router IDs, so it does not
attribute any difference to expert selection. Its receipt SHA256 is
`d237fcc06255f75b35a02dd88b90db4680b9b07a9ec274e4adf1ecf8965e9cf6`.

A targeted layer-44 comparison verified the diagnostic output bitwise against
the plain layer output. All ranks selected the same expert set as the CPU
reference and as FP64 routing on the actual input. On that same actual input,
the MoE output still differed from the CPU reference by `0.001368` relative
L2, so this layer's excess error cannot be attributed to an expert flip.
Its receipt SHA256 is
`c5d1fcf279e60a10f62fdae407e571702f0fa5f36eccd5179ec9fe41b6495083`.
An independent FP64 calculation on the same input measured candidate relative
L2 `0.001364`; the CPU FP32 path differed from FP64 by `0.000288`. The FP64
receipt SHA256 is
`0549e0965d88d00ca6bc83f6c91d75c989b533b9327c67d913d7af3fa8bfee1f`.

On the frozen layer-44 input, rank 7's local MoE output had `0.001311`
relative L2 error against FP64 while ranks 0–6 were near FP64. The captured
partial projections and downstream projection were close to FP64; one gate
activation differed substantially. The diagnostic receipt SHA256 is
`b5464632811c0e7fdbfaac3349650a11482322dae4d42d0faacc6a7a17a25e99`.

A single-card arithmetic check identified range-boundary errors in Gaudi2's
FP32 exponential, including `exp(-ln(2))` returning `1` instead of `0.5`.
The corrected MoE gate uses a second exponential decomposition to recognize
this error. It passed 750 boundary and adjacent inputs against CPU math, and
on the same frozen rank-7 input its relative L2 error against FP64 fell to
`0.0000641`. The latter receipt SHA256 is
`e40cd3a74f785f4d5c5dfd0dd640fefd8b3ad2e6ff076aec18da48686461ba26`.
These checks do not establish full-model quality or performance.

An unprofiled full-model run with that correction and initialization-time
metadata validation measured `11.221187 ms` p50 and `11.219819 ms` mean.
The eight ranks agreed on 144 generated tokens, and untouched KV bytes were
unchanged at the end. The first-step `max_kl` was `0.0007722`, while
`relative_l2` and `max_row_relative_l2` were `0.0205696`: the unchanged
`0.02` quality gate still fails. This run is not final ABBA and did not check
KV after every step. Its receipt SHA256 is
`a233d8df875d0e950e7a1509a225d3e1bae1387e676519edcb555c664879ca0c`.

The layer-44 same-input follow-up found that post-attention normalization and
the corrected MoE output matched the then-current CPU reference bitwise on
their respective actual inputs. The reduced attention output had `0.000669`
relative L2 difference; summing the eight captured rank outputs reproduced
the device reduction bitwise. The receipts have SHA256
`176229b91432c9fd0b6c765e425b841276d84e3400d526cb17bc3179b1baaab0` and
`06b58cd8a67ed2046026c14ee21f0acf2a462673fefaee278ea9efa496d15ee3`.

A subsequent comparison exposed a reference rounding limitation. One QKV
value was within `1.28e-8` of a BF16 midpoint. PyTorch's binary64-to-BF16
conversion on this environment first rounds through binary32, selecting the
opposite neighbor from direct binary64 round-to-nearest-even. The device
selected the direct-rounding neighbor on that sample. The earlier statement
that this value was a device error is withdrawn; its receipt SHA256 is
`d30f789b4b234dc19c281c428ffcfdc472196775da1b5d3db08e833b9ab7efe3`.
A separate full-model FP64 reference with direct BF16 rounding is complete;
its logits SHA256 is
`042f2bcfe915660fefa65e1d38e4f259cf8a097db6a73b1374465f4f5d246691`.
The earlier full-model FP32 reference differs from it by `0.024443` relative
L2, with different selected expert sets on 16 layers. The reference comparison
receipt SHA256 is
`befaaaf5c5656ddd15adbe68a7b6a99a4d059b2034bb3f42a896b8a1f7055cf9`.
Previous results against the FP32 reference remain unchanged; they do not
establish the candidate result against the new reference.

An untimed run of the same candidate matched the unprofiled run's first
logits and its first 128 generated tokens on all eight ranks. Each step's KV
slot was checked for every layer and rank, and all untouched KV bytes were
unchanged. Against the independent FP64 reference, `max_kl` was `0.003756`
and `relative_l2` / `max_row_relative_l2` were `0.023605`; the `0.02` gate
still fails. The separate audit receipt SHA256 is
`abcdb5aa0e8bc460c81ca316d37ecf5fdb11922a81de4de6dcfc03b71cd79f2f`.
