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

The Full Attention MME softmax also used the affected exponential. In a
single-card masked-softmax boundary test, the original maximum absolute
probability error was `0.166667`; the corrected kernel measured `3.39e-6`.
On a separate frozen real Full Attention layer, output and KV were bitwise
unchanged on all eight ranks. The local receipt SHA256 is
`286c537925034796eeeb6ce159e58f7fa5264f0a5408af6fc97c8a05f9416fd4`.
Neither check establishes an end-to-end quality or performance improvement.

One unprofiled full-model run with the corrected Full Attention softmax
measured `11.250428 ms` p50 and `11.244075 ms` mean. All eight ranks agreed
on 144 generated tokens, and all untouched KV bytes were unchanged at the
end. First-step logits were bitwise identical to the previous candidate.
Against the full-model FP32 reference, `max_kl=0.0007722` and
`relative_l2=max_row_relative_l2=0.0205696`; against the full-model FP64
reference, these values were `0.0037558` and `0.0236053`, respectively.
Both remain above the unchanged `0.02` relative-error gate. The candidate
tokens first differed from the previous candidate at the seventh warm step;
the earlier per-step KV audit therefore does not establish this candidate's
per-step KV behavior. The one-run result is not an ABBA acceptance.

The full-run receipt SHA256 is
`6a7137e44dc299aafa16235921d32e471add0c19a1e5340d98a38939a49a5e2b`;
the two-reference comparison receipt SHA256 is
`dfc4b17144dc376aaa270c36a692f8d732f43486d27dddb6c67a7037947fbd84`.

An 8-rank same-input check of 70 standalone layers against the FP64
mathematical trajectory found 34 layer outputs above `0.001` relative L2;
the largest was layer 39 at `0.002438`. Candidate expert IDs were not
recorded, so the result does not attribute these differences to routing.
The receipt SHA256 is
`d4a9c2e88abd545dc9fe56be9ee3d1b7bb60dbc88b3b599afa875dca157c27da`.
Separately, on the same frozen layer-39 input the CPU FP32 and FP64
references had the same expert set but differed by `0.002891` relative L2.
That comparison does not establish candidate correctness. Its receipt SHA256
is `e3bf5bc7dadd7c03fc943adbabcd3649c792cd9493682a94f8d2c2b770d8b7ea`.

The SWA fast kernel also uses the affected exponential. A single-card
boundary test measured original maximum absolute error `0.166667` and
corrected error `0.0006511` in BF16 output. On one real frozen layer-18
input, the corrected candidate matched the CPU output and written KV bitwise;
untouched KV was unchanged. The corrected provider compiles grouped AV from
C instead of using the original hand-scheduled assembly. Whole-model quality
and performance remain to be measured. The single-card receipt SHA256 is
`a9e0f47f940ea70fdcdcf60cad5a854698e16a49e415d4baa75b09256e4cf86d`.

The SWA boundary test also reproduces the same errors with the model's
`1/sqrt(192)` attention scale. One unprofiled 70-layer run with the
corrected SWA provider measured `11.061298 ms` p50 and `11.113407 ms` mean;
all eight ranks agreed on 144 generated tokens, and final untouched KV was
bitwise unchanged. The first logits were unchanged, so relative L2 remained
`0.0205696` against FP32 and `0.0236053` against FP64: both exceed `0.02`.
The generated sequence later differs from the previously audited candidate,
so its per-step KV audit cannot be reused. This single run is not an ABBA
acceptance. Its receipt SHA256 is
`bb6c5be08e99ec5597feb26a970f90f467615cf5021329c1c84c52a4a3f2bcc8`.

On the frozen FP64 layer-39 input, the plain and diagnostic device outputs
matched bitwise on all ranks. The candidate selected the same eight experts
as both CPU references; the device input's eighth/ninth score gap was
`0.03417`. Device output relative L2 was `0.002903` against CPU FP32 and
`0.002438` against CPU FP64. With the device's actual normalized input,
MoE output relative L2 against FP64 was `1.92e-7`; its expert set was the
same. These receipts have SHA256
`0a3fe791d0e92f14856936302b2695578a5eb4d5182e933e4fe9faa50f859eff` and
`63d2757d468655dd09746010992c35eef5704133b600193ba7c9587062848066`.

On that layer's same input, the device Full Attention output relative L2
was `0.0007283` against FP64; CPU FP32 versus FP64 was `0.0007974`.
The rank reduction and post-attention normalization matched their CPU
checks bitwise on the actual device inputs. The resulting normalized input
difference against the FP64 trajectory was `0.0008517`; subsequent MoE
amplifies it in the layer output. This does not establish the cause of the
attention difference. The receipt SHA256 is
`e52e4214d1743e4ea5f62487060d5ca782f2a41dcff407a8700baa4f4861f3a5`.

For layer 39, the two BF16 probability parts reproduced the dominant
softmax error when evaluated independently on CPU. Adding a third BF16
residual reduced the synthetic maximum probability error from `3.39e-6`
to `6.97e-8`. On the same frozen TP8 layer, output relative L2 against
FP64 decreased from `0.002438` to `0.0000246`; the layer receipt SHA256 is
`8b27831edf97454b8274710fa42e24853e5fdde02afa0a9d7f3d2a0d30f86f21`.

The resulting unprofiled full-model run measured `11.261994 ms` p50 and
`11.283988 ms` mean. All ranks agreed on 144 generated tokens and untouched
KV was bitwise unchanged at the end; per-step KV was not audited. The first
logits failed the unchanged `0.02` relative L2 gate against both independent
references: FP32 `0.0306621` (`max_kl=0.0060240`) and FP64 `0.0303310`
(`max_kl=0.0020335`). This is an unqualified experiment, not an ABBA or
an end-to-end accuracy improvement. Its receipt SHA256 is
`b540498ffe290e7f3df8cf076f835bc96c826bb8f68469beaa1dafabca068018`.

An independent same-input check of all 70 standalone layers with this
candidate found 31 outputs above `0.001` relative L2. The largest was layer
55 at `0.0023714`; the reference eighth/ninth expert score gap there was
`0.04422`. The check did not record device expert IDs, so it cannot determine
whether routing caused any particular difference. Its receipt SHA256 is
`873b9bb777742c382e3cfd0693d79ddfd1bed9348edd97841fb7721fbf4bdaa4`.
In separate checks on layers 31 and 55, all eight ranks selected the same
experts as the FP64 reference, while diagnostic output hashes matched the
corresponding uninstrumented layers. Their output relative L2 values were
`0.0018340` and `0.0023714`; expert flips therefore do not explain these
two deviations. The route-check receipt SHA256 is
`bf2773a3a0ebcd8edf1b2e8e0bd73fe2055d24ebff3645b33ecf6c3d2d278152`.

For layer 55, FP64 MoE evaluated on the device's actual normalized input
matched the device numerically; its preceding Full Attention output had
`0.0007076` relative L2 error. A QKV value differed from direct FP64 BF16
rounding; an ideal FP32 intermediate lands on the BF16 midpoint and
reproduces the device value. The receipts are
`d4c4bc38e63f391cb276557e143ac6af5bf1be8348190687e27003966f4acaca`,
`4d4302899a8aa6a2b1426b5f8b90133c5780c8f3a5f3834312759bef72e3aef0`, and
`a494d88b488e88b565ae140d8af50601a7fe7c6f7c1b5210006a68b10749d4c7`.

A separate compensated QKV prototype reduced that layer's relative error
to `0.001368` but still missed `0.001`; a one-card mechanism timing measured
`18.867 us` versus `7.794 us` for the existing kernel. Neither is an
end-to-end result. The prototype was discarded. The three-part softmax was
also reverted because both full-model reference errors worsened. The active
code returns to the earlier unqualified candidate, whose FP32 and FP64
relative errors were `0.0205696` and `0.0236053`, respectively. The QKV
layer receipt is `5b5802ea844dd0e571c30d186f9e89471ca9991ca19d228fcad70a158f2dcf77`;
the one-card mechanism receipt is
`71f0ba11114e98ebfcb35d281b85ea4ba2469b26afb4c1d042c2b67b2094b3c3`.

An untimed audit of the restored candidate reproduced the unprofiled run's
first logits and first 128 tokens on all eight ranks. For every step, rank,
and layer, the expected key and value slot was written and remained unchanged
later; all untouched KV bytes remained bitwise identical. This does not
change the failed first-step accuracy gate. The audit receipt SHA256 is
`ff968f35a54adb5742e29136899ac17fcedc647c7d3314be2a65af62209572dc`.

An independent CPU head calculation used that candidate's actual first-step
final normalized input and checkpoint weights. Its FP32 output had relative
L2 `0.0204918` against the FP32 full-model reference and `0.0235504` against
the FP64 reference. Both exceed the unchanged `0.02` gate. This is a CPU
same-input calculation, not a measurement of an FP32 device head. Its receipt
SHA256 is `9fac64529bf150dbf505265eaa7f563216d2a660e011d4e2c16d258d9d8daf36`.

The `0.02` statements above describe the earlier cross-arithmetic gate. The
revised roadmap treats raw logits L2 as a diagnostic and requires finite
logits, `max_kl <= 0.01`, and the applicable Top-1/Top-5 check. Against the
independent FP32 full-model reference, the restored candidate has
`max_kl=0.0007722`, `relative_l2=0.0205696`, `calc_diff=0.0002108`, and
matching Top-1. The older quality reporter still records its original L2
gate as failed; the numerical values themselves have not been changed.

Under the revised frozen-input contract, all 70 standalone layers passed on
eight ranks against the independent FP64 trajectory. All 69 MoE layers chose
the reference expert sets, including all 51 noncritical layers. The largest
output `max_row_relative_l2` was `0.0024381` at layer 39; the largest element
`max_abs` was `0.03125` at layer 69. Every same-expert result met its per-layer
limit, derived from the FP32 linear / FP64 nonlinear reference on the same
frozen input, with the `0.01` absolute cap. The route-observation graph's
output hashes matched the uninstrumented standalone graph on all ranks and
layers. The receipt SHA256 is
`36e8472ddf295499288e01c2ce124ae3737ac50c828c4b925e2fdf71989ed6a5`.
This does not establish full-model cache equivalence or final ABBA performance.

For the fused 70-layer model, a fresh fingerprinted cache saved ten recipes
per rank. A new warm process loaded ten and saved none on every rank; recipe
contents and modification times were unchanged. Cold and warm first logits,
128 greedy tokens and the full final KV states matched bitwise, and the first
logits and tokens also matched the previously audited run. This check compared
every untouched KV byte to the common fixture and directly compared the
written bytes; `max_abs=0.0`. The full-model cold and warm runs did not expose
their internal router IDs; the separate frozen-input route check above covers
router IDs on standalone layers. The cache receipt SHA256 is
`a9790363dd1e1668e28f3c4cc775eab4d80eaaedf2f7793117ec1f1f7dd07d00`.

An unprofiled A1-B1-B2-A2 run produced p50 values of `11.3253`, `11.1807`,
`11.4139`, and `11.3795 ms`; the corresponding means were `11.3698`,
`11.2263`, `11.3904`, and `11.4100 ms`. Each arm recorded 128 decode steps.
The B2 rank-0 process aborted after writing its samples, and its runner
returned 1. B1 returned 0 but its rank-0 log contained the same native
termination message; the source of either message has not been established.
This is a failed final acceptance, including because B2 exceeded `11.284 ms`.
Production p50 drifted by `0.48%` and candidate p50 by `2.09%` between arms.

Both production arms generated identical token sequences, and both candidate
arms matched each other and the audited candidate. Production and candidate
differed at 130 of 144 aligned steps, first at zero-based step 6. Candidate
timing included the production client output processor; production timed the
complete engine step. Their engine scheduling and IPC implementations differ.
The receipt SHA256 is
`175f6c06c7bbd7f1bb404c431fbe6ed346579480be139f861c45ed48517b4b03`.

The vocabulary-sharded head was checked on one card using the checkpoint's
BF16 head weights. Concatenating all eight shard outputs reproduced the full
head's `152576` logits bitwise. Device-event p50 was `897.493 us` for the
full head and `214.124 us` for one shard; this excludes TP8 gathering and is
not an end-to-end speed result. The receipt SHA256 is
`2e3e4afc1efc040192cde0e0451e076eacbc9c587c324ab1ff05e1f7d726e242`.
An eight-rank BF16 vocabulary AllGather test had zero mismatches on every
rank; its receipt SHA256 is
`a2ad4ca77752008778e328f7330cc1034b60188d8b4ab080339c5241db2cf56c`.

In the sharded model's cold-cache 70-layer run, all eight ranks' first logits,
128 greedy tokens, and entire final KV were bitwise identical to the prior
audited candidate. Every rank saved ten self-compiled recipes. The cold
receipt SHA256 is
`d85a2d7a03d3691efe378ab7b7434796bd87c46f885b756b1fa6ed806ebc3498`.
In one warm-cache run, each rank loaded ten recipes without saving or
rewriting any. The first logits, greedy tokens, and final KV matched the
audited candidate; untouched KV had `max_abs=0.0`. The 128-step p50 was
`10.4727 ms` and mean `10.5245 ms`, including the production client output
processor. All processes exited successfully and the earlier native
termination message was absent after coordinating teardown and releasing
compiled graph references. One successful run does not establish the cause
of the earlier failure or establish paired performance. The receipt SHA256 is
`ab1d75721324c1564a6d45e2170738ff6c8631d3ceb44aa7fa9743cbc9cad826`.

An independent untimed full-model observer recorded the Top-8 router IDs for
all 69 MoE layers over 128 steps. Cold compilation and warm loading produced
identical ordered IDs on every rank, and the eight ranks agreed. Each cold
rank saved ten recipes; each warm rank loaded ten without saving. The
observer's first logits, token sequence and entire final KV matched the
uninstrumented sharded candidate bitwise. Exposing IDs makes the observer
graph different from the release graph, so this is not a direct observation
of the release graph's internal tensors. Its receipt SHA256 is
`3e86346d72bd1e7da5747fa8dddad56c7d56065e6e0c0756ec5de3c6b93331ba`.

The unprofiled A1-B1-B2-A2 run on the sharded candidate measured p50 values
`11.414471`, `10.511892`, `10.608441`, and `11.244826 ms`; means were
`11.465220`, `10.531679`, `10.622476`, and `11.290237 ms`. Each arm had
128 timed steps after 16 decode warmup steps from initial context 4096. Both
candidate arms were below `11.284 ms` and below both production arms. The
production p50 drift was `-1.49%`, and the candidate p50 drift was `+0.92%`.
All four main processes returned zero; both candidate arms loaded ten
recipes per rank, matched the audited token sequence on all ranks, and had
no native termination message. Both production arms matched each other,
while their 144 aligned tokens differed from the candidates at 130 positions.
The source of those cross-path token differences was not established.

Production timed the complete engine step; the candidate timed the direct
TP8 model step through token completion and production client output
processing. Their scheduler and IPC implementations differ, so this result
does not establish equivalent performance for a candidate integrated into
the complete production serving engine. On shutdown, each production engine's
process manager terminated its remaining child processes and the Python
resource tracker warned about one shared-memory object. The previous
candidate termination's cause remains unknown. The four-arm receipt SHA256
is `93c16cb993b1ff0c4f8242e1e70914045c19b1518c01c921e2bc9be0f270a5aa`.
After both candidate arms, all 80 release-cache files retained their cold
content, sizes, and modification times; the check receipt SHA256 is
`070b03c9a97ce8f01c0cc7fb81cc16beaa5eed814d6bb279b1fb511a45c0ea4b`.
The raw four-arm receipt's `performance_acceptance` field checks the timing,
exit, token, layout, and isolation conditions; it does not establish the
same complete serving path or directly expose release-graph router IDs.
Those limits are also recorded in qualification receipt SHA256
`3a1588c0e449fa8df2f3bacc2d860dced6de624322890e7c2308ed77df80d218`.
