# Better StaticMLP optimisation plan

Written for: the maintainer and whoever implements the steps. Follow
`AGENTS.md` for every step that adds or changes MLP code (core0 and core1
builds, `validate_memory_placement.py`, callee-closure check, `nm`
spot-checks).

**Baseline.** The starting point is commit `f1b9488` and the tooling it
contains: the callee-closure validator, the private `expf`/`logf`
(`mlp/FastMath.h`), the production-path `train.*` probe, the bus-contention
counters and the extended placement CSV. Step 0 (making the measurements
trustworthy) was implemented in `02bc8d3`; section 3 records what it achieved
and where it fell short. **Step 0.5 fixes those shortfalls.** The findings in
section 1 have been revised with the `02bc8d3` data.

This plan supersedes `plan-mlpOptimisationPrompt.md` (Optimisations 1-5). It is
based on the benchmark CSVs, the source, and the disassembly and symbol tables
of the ELFs in `build-*`. **None of the expected savings below has been
measured**: each is an estimate from instruction counts or from the contention
counters, and each step says how it will be confirmed.

**Goals, in order:**
1. Deterministic timing: the core0 and core1 builds take the same time within
   measurement error, and that time does not depend on unrelated edits.
2. A demonstrated 100% bus independence between core 0 and core 1.
3. Only then, making the MLP as fast as possible.

---

## 1. Findings

### 1.1 The problem and the baseline

| Build | Benchmark time (10 epochs, one session, dormant) |
|---|---|
| Legacy heap MLP (`0732282`, core0-baseline) | 45.64 s |
| First StaticMLP (`0e9e71b`) | 45.50 s |
| "Peak" StaticMLP (`8fb216e`) | 42.52 s |
| `f1b9488`: core0 / core1 / naive | 41.72 s / 41.18 s / 41.73 s |
| **`02bc8d3` (after Step 0)**: core0 / core1 / naive | **41.800 s / 41.245 s / 41.244 s** |

Removing the heap gained nothing, and the five earlier optimisations gained
about 6.5% in total. The legacy engine has exactly the same loop nest as
StaticMLP (`Layer.h`), and the heap is touched only at construction, so
removing it could not make the hot loops faster. StaticMLP's real value is
placement and independence.

**Independence between the cores is already demonstrated.** Pinned builds,
flooding against dormant: core0 41.8014 s vs 41.8004 s, core1 41.2449 s vs
41.2449 s. During a profiled epoch the *other* core's bank group sees 0-106
contested accesses. The remaining problem is that **core0 is 0.556 s (1.3%)
slower than core1 and naive**, with or without the other core running. Section
1.5 shows that this is not contention between the cores. It is the core
contending with itself, and it is decided by link addresses.

### 1.2 Where the time goes (`02bc8d3`, core0 profile)

One session is 10 epochs x 3823 samples = 38,230 training samples plus one
3823-sample evaluation pass, about 6.2 G cycles at 150 MHz (1 s of session
time is about 3.9k cycles per training sample). The per-sample cycles come from
the per-layer profile of the core0 build, whose layout happens to be free of
the forward collision described in 1.5:

| Stage | Cycles/sample | Share of pass |
|---|---|---|
| layer0.forward (64->64) | 25.3k | 17% |
| layer1.forward (64->32) | 12.0k | 8% |
| layer2.forward (32->10) | 1.7k | 1% |
| loss (softmax-CE) | 0.9k | 0.6% |
| layer2.AccumulateGradients | 6.0k | 4% |
| layer1.AccumulateGradients | 35.3k | 23% |
| **layer0.AccumulateGradients** | **70.2k** | **46%** |
| **Full pass** | **151.4k** | 100% |
| per-batch work (RMSProp apply, grad-norm, zeroing) | 4.2k | 2.8% of a pass |

- Forward, loss and backward are about 94% of the session. The final
  evaluation pass in `Train()` is about 2.5%, and the per-batch work is the
  rest. **The training loop itself is not the bottleneck**: its overhead is
  about 3%, and most of that is RMSProp (468k cycles per 128-sample batch,
  about 72 cycles per weight).
- Backward is **75%** of the pass, and layer-0 backward alone is **46%**.
- Production costs about 2.1k cycles per sample more than the per-layer
  harness on core0, 0.4k on core1 and 1.2k on naive
  (`train.unexplained_per_sample`). The difference is fully explained in 1.6
  (E2): the harness and production use different buffers.

The algorithmic floor, for context: 6464 weights give 6464 forward MACs plus
2 x 6464 backward MACs, about 19k MACs per sample. 151k cycles is about
**8 cycles per MAC**; a scalar Cortex-M33 should reach roughly 2-3.

### 1.3 What the disassembly shows (`StaticLayer.h`)

- **Forward inner loop** (156 bytes, at offset `+0x30` in every `forward`
  instantiation): per 4 MACs, 8 `vldr` + 4 `vmul` + 4 `vadd` + 4 integer
  instructions, about 22 instructions. At 24.6 cycles per iteration it is
  **bound by instruction and load count, at about one instruction per
  cycle**. The same input vector is reloaded for every output row, so half of
  all loads are redundant. The 4-accumulator unroll (old Opt 3) did not help,
  because the loop was never limited by FMA latency
  (`17bc32a` -> `665faab`: 42.93 -> 42.83 s).
- **Backward inner loop** (92 bytes, at `+0x32`): per `j`, 10 instructions for
  2 MACs (4 loads, 2 stores, 2 `vfma`, compare, branch). It runs at about
  17-18 cycles per iteration, roughly 9 cycles per MAC. `g[j]` and
  `delta_out[j]` make a load-modify-store round trip on every (i, j). The cost
  is in the loads and stores, not the FP arithmetic.
- **Dead work**: `layer0.AccumulateGradients` computes `delta_out` (4096 MACs),
  but nothing consumes it, because layer 0 has no earlier layer
  (`backprop_accumulate<0>`). That is about half of layer 0's backward cost.
- **RMSProp**: about 25 instructions per weight, including one `vsqrt` and one
  `vdiv`.

### 1.4 Why the earlier optimisations did not help

| Optimisation | Targeted | Share of pass | Result |
|---|---|---|---|
| 1 branchless ReLU | ~96 activations/sample | <0.3% | 45.50 -> 45.29 s |
| 2 `__restrict` | enabler | n/a | -> 44.68 s |
| 3 4-acc forward | forward (wrong premise: latency) | 25% | ~0 (42.93 -> 42.83 s) |
| fp-contract=off | forward | 25% | -3k forward cycles in profile; ~-0.3 s benchmark |
| 4 backward j-outer interchange | backward | 75% | **regressed** |
| 5 skip layer-0 cached-input copy | 256 B memcpy | 0.04% | noise / reverted |

The earlier work went after the 25% forward half and after tiny items; the one
change aimed at the 75% backward half regressed. On top of that, **every
earlier comparison carried about ±0.56 s of layout noise** (section 1.5). A
change that shifted the layer-0 forward loop by 4 bytes relative to the weights
could add or remove that much with nothing visible in the disassembly. That is
the likeliest explanation for the "trivial change, unexplained regression"
results. Any gain smaller than about 0.6 s measured before the layout is
pinned (Step 1) is not trustworthy in either direction.

### 1.5 Self-contention: the core's code fetch and its own data collide in one bank

**The mechanism, in plain terms.** RP2350 main SRAM is two groups of four banks
(group 0 = banks 0-3 at `0x20000000-0x2003FFFF`, group 1 = banks 4-7 at
`0x20040000-0x2007FFFF`). Inside a group, consecutive 32-bit words go to
consecutive banks, so the bank of an address is given by bits [3:2]: **the
address mod 16 tells you which bank it is in**. Each bank serves one access per
cycle. With `PICO_COPY_TO_RAM=1` the hot code also lives in SRAM, in the same
group as the data it reads. When the core's instruction fetch and its own data
load want the same bank in the same cycle, one of them waits one cycle, and the
bus fabric counts a **contested access** (`SRAMn_ACCESS_CONTESTED`).

The evidence that the core contends with itself: with the other core asleep in
WFE (the poller makes about 5k accesses per *epoch*, see E4), the probe's own
group still records 33.8M contested accesses per epoch on core0 and 49.4M on
core1. No other requester is active, so fetch and data must reach the bank
arbiter as separate requesters. Because a loop repeats exactly the same
pattern, a collision that happens once happens on every iteration. The cost is
deterministic, and it is set by the **relative phase** (address difference mod
16) between a loop's code and the arrays it streams.

**One contested access costs one cycle.** Profile, core1 minus core0:
15.50M cycles against 15.54M extra contested accesses per epoch, a ratio of
0.998 (eval pass alone: 0.999).

**The forward loop follows a simple rule.** Take the phase of the forward inner
loop's head relative to its weight array (the input array has had the same
phase in every case observed):

| Binary | Kernel | Relative phase (loop - weights) mod 16 | Contested per call | Cycles per call |
|---|---|---|---|---|
| profile core0 | layer0.forward | 8 | 66 | 25,345 |
| profile core1 | layer0.forward | **0** | **2,130** | **27,408** |
| profile naive | layer0.forward | 8 | 66 | 25,345 |
| profile core0 | layer1.forward | 12 | 42 | 12,046 |
| profile core1 | layer1.forward | 4 | 67 | 12,054 |
| profile naive | layer1.forward | **0** | **1,066** | **13,069** |

At phase 0 the loop pays about **0.52 contested accesses per MAC** (2,130 /
4096 and 1,066 / 2048). At phases 4, 8 and 12 it pays almost nothing. (The
profile rows above refer to the harness net `g_net`; see E2 and E3.)

**The same rule explains the benchmark's core0 gap.** The benchmark ELFs carry
no counters, but their phases can be computed from the symbol table: the
production instance `test::TestRAMIndependence::mlp_experiment_` (or
`TestNaiveMemoryLayout::mlp_experiment_`), plus the weight offset inside it
(`0x7548` for layer 0, which is the same type in every build), against the loop
head at `forward + 0x30`.

| Benchmark ELF | Instance | layer0.forward loop | layer0 weights | Relative phase | Measured |
|---|---|---|---|---|---|
| core0 | `0x2000bbc0` | `0x20007718` | `0x20013108` | **0** | **41.800 s** |
| core1 | `0x20042148` | `0x200403e8` | `0x20049690` | 8 | 41.245 s |
| naive | `0x20025134` | `0x20000d48` | `0x2002c67c` | 12 | 41.244 s |

Prediction: 2,064 cycles x 42,053 forward passes per session = 86.8M cycles =
0.579 s. Measured gap: 0.556 s (96%). Layers 1 and 2 are at clean phases in
all three benchmark ELFs.

**The profile binary and the benchmark binary disagree, in opposite
directions.** At `f1b9488` the production probe predicted the benchmark to
within 0.01 s. At `02bc8d3` it no longer does:

| | Predicted from profile `train.*` | Benchmark measured |
|---|---|---|
| core0 | 41.24 s (clean) | 41.80 s (phase 0) |
| core1 | 41.81 s (phase 0) | 41.24 s (clean) |

The two binaries link differently, so the profile measures a different layout
from the one the benchmark runs. Neither core is inherently slower; whichever
binary has the phase-0 loop is slower.

**Backward has a contention floor that does not follow this rule.**
`AccumulateGradients` pays about one contested access per weight in every
build (layer 0: 4,144 per call; layer 1: 2,072; layer 2: 324), at relative
phases 2, 6 and 10 alike. That is about 6.5k cycles per training sample, or
about **1.7 s per session**: larger than the core0 gap, and the largest single
contention item. The likely reason is that the 92-byte loop body sweeps all
four banks on every iteration, so some fetch always lands on the data's bank,
whatever the phase. Phase tuning cannot remove it. Moving the code out of the
data's group can (Step 1c tests this, and Step 5 does it).
Core1's profile pays an extra ~0.25 per MAC in backward (5,185 / 2,599 / 416)
because the harness's `delta` buffers are on its stack, which for core1 is in
the same group (E2).

**Ceiling.** At `02bc8d3`, contested accesses are 4.5% (core0, naive) to 6.4%
(core1) of profiled epoch cycles. About 0.56 s of that is the phase-0 forward
lottery, and about 1.7 s is the backward floor.

### 1.6 Measurement status after Step 0

**Defects D1-D7 (found at `f1b9488`), after Step 0:**

| # | Defect | Status at `02bc8d3` |
|---|---|---|
| D1 | Two copies of each kernel | **Fixed**: one symbol per shape (`noinline, noclone`), validator-checked. But the acceptance (unexplained gap near 0) failed, for a different reason: E2. |
| D2 | Poller not idle | **Fixed in effect**: during `train.epoch` on core1 the poller's side (bank 0) makes only 5,383 accesses per epoch. The `idle_window` row meant to show this measures something else (E4). |
| D3 | CSV not self-explanatory | **Fixed**: `group` / `other_group_total` rows, `contested_pct_of_cycles`, `benchmark_results/README.md`. |
| D4 | No eval-only contention | **Fixed**: `train.eval_pass` rows and `cycles_per_contested` (but see E5). |
| D5 | `compute_loss` cost unexplained | **Explained**: 887 cycles with labels in flash, 612 with labels in RAM; about 150 XIP stall cycles per call. It is flash latency, not contention (see E7). |
| D6 | No golden numerics | **Fixed**: `golden_results.csv` (profile `Train(1)`: loss 0.395219296, accuracy 0.883860826; identical in all builds and repeats) and `final_loss` / `final_accuracy` in the benchmark CSV (0.129690 / 0.958933). This is **G0**. |
| D7 | Stack in different places per build | **Documented, not fixed**, and now measurably relevant: the harness's stack-resident buffers contend on core1 only (E2). Production's own stack exposure is unmeasured. |

**New defects revealed by the `02bc8d3` data (fixed by Step 0.5):**

| # | Defect | Evidence | Consequence |
|---|---|---|---|
| E1 | **The profile binary is not the benchmark binary.** Different link layout, therefore different phases. | 1.5: the profile predicts core0 fast and core1 slow; the benchmark measures the reverse. | Profile contention and placement say nothing reliable about the benchmark. Any layout sweep run on the profile alone optimises the wrong binary. |
| E2 | **The per-layer harness measures a different object from `Train`.** The harness runs `g_net` and keeps `delta_a`, `delta_b`, `delta_layer0` and `deriv` on the **stack**; production runs `g_prod` and uses `StaticMLP::m_bp0` / `m_bp1` inside the bank. | Unexplained gap +2,123 (core0), +438 (core1), +1,163 (naive). On core0 the harness's stack is in SRAM9 (never contends) but production's buffers are in the group (+~0.25/MAC, about 1.6k per sample); on core1 both are in the group. Core0 minus core1 = 1,685, against about 1,616 predicted. Naive's harness additionally pays a phase-0 layer-1 forward (1,024) that its production object does not: 2,123 - 1,024 = 1,099, against 1,163 measured. About 450 cycles per sample remain in every build (E7). | Per-layer rows are not comparable between builds and do not add up to production. |
| E3 | **Placement rows describe `g_net`**, not the production instance and not the benchmark ELF. | `placement_results.csv` data rows | Phases could only be obtained by hand from the ELFs (1.5). |
| E4 | **`idle_window` measures the probe's own `busy_wait_ms`**, not the poller. | About 938k bank-0 accesses per 50 ms in every build, the same figure as the old "spinning poller". For core1 the probe is core 1 running SDK code in group 0. | The row cannot confirm what the README says it confirms. The original D2 diagnosis was probably this same artefact. |
| E5 | **`cycles_per_contested` is reported even when its denominator is tiny.** | core0-naive: 0.88 (epoch) and 1.36 (eval), from differences of about 120k cycles. | Noise that looks like a finding. |
| E6 | **The benchmark binary has no counters.** | core0 and core1 each gained about 0.07 s since `f1b9488` (about 275 cycles per sample); no data can say why (`noinline` call overhead is a candidate). | Benchmark changes can only be guessed at. |
| E7 | **The dataset is read from flash.** `dataset::features` (978 KB) and the labels live in XIP flash, behind a cache that **both cores share**. The flooder (`utils/RAMFlooder.hpp`) writes only to the other bank's SRAM and never touches flash. | `nm`; `compute_loss.flash_labels` vs `ram_labels`; XIP stall counts. | About 450 cycles per sample of production cost the harness doesn't see. The independence claim is proven for SRAM only; flash is an untested shared resource. |

### 1.7 Resolved earlier (kept for the record)

- The hot path of the pinned core1 build used to call `memcpy`, `memset`,
  `__wrap_expf` and `__wrap_logf`, which are located in core 0's bank. Fixed by
  `-fno-tree-loop-distribute-patterns` on the MLP TUs, the private
  `smlp::exp_f32` / `log_f32`, and `check_callee_closure` in the validator.
- `compute_loss` went 728 -> 2,315 cycles with the first Cephes versions. It is
  now 924 (887 / 612 with flash / RAM labels; see D5).

---

## 2. Step overview and expected cumulative effect

Cycle counts are per training sample. "Cum. pass" is the estimated full-pass
time after the step (151.4k at `02bc8d3`, measured in a clean-phase layout).
All savings are estimates.

| # | Step | Est. saving | Cum. pass | Risk |
|---|---|---|---|---|
| 0 | Make the measurements trustworthy (**done**, `02bc8d3`) | 0 | 151.4k | n/a |
| 0.5 | Profile what production runs, measure the benchmark binary, freeze B0 | 0 | 151.4k | low |
| 1 | Phase-pinned layout with a build-time phase guard; diagnose the backward floor; decide stack placement | core0: -0.56 s (all builds equal); removes ±0.56 s noise from every later step | 151.4k (core0 benchmark: -2.1k) | low-medium |
| 2 | Skip the layer-0 input-gradient | ~28-32k | ~120k | very low |
| 3 | Row-blocked forward | ~20-22k | ~100k | low |
| 4 | Row-blocked backward | ~28-32k | ~70k | medium |
| 5 | Hot kernels in the scratch banks (if Step 1c confirms) | up to the remaining contention floor (~1.7 s/session today; re-estimate after Step 4) | ~65-70k | medium |
| 6 | Micro-batch of 4 samples (optional) | ~8-12% of the original pass | ~55-60k | high |
| 7 | RMSProp / loop-level cleanup (optional) | ~1-1.5 s per session | n/a | low |

Steps 2-4 together: about **1.8-2.2x** end to end (about 19-23 s per session
against 41.2 s).

**Why this order.**
- Step 0.5 comes first, because without it the profile describes neither the
  benchmark binary (E1) nor production's buffers (E2).
- Step 1 must land **before** any optimisation. Until the phase guard exists,
  each step's measurement carries about ±0.56 s of layout noise, which is
  enough to hide or fake the gains of Steps 6-7 and to distort those of
  Steps 2-4.
- Step 1c is a throwaway *diagnostic*: it tells us now whether the backward
  floor is worth Step 5. The decision to commit scratch-bank code waits until
  after Step 4, because Step 4 rewrites the backward loop.

---

## 3. Steps

### Step 0 (done, `02bc8d3`) - Make the measurements trustworthy

Implemented as planned. Outcome per sub-item:

| Item | What was done | Acceptance | Result |
|---|---|---|---|
| 0a one kernel copy | `noinline, noclone` on `forward`, `AccumulateGradients`, the RMSProp apply; validator check | unexplained per sample near 0 | **Not met** (+2,123 / +438 / +1,163). The copies are gone; the remaining gap is E2. |
| 0b event-driven poller | WFE + 10 ms timer alarm, 64-bit totals | `idle_window` < 0.1% of old; naive contested within 5% of core0 | **Met in effect** (naive 33.95M vs core0 33.81M; poller about 5k accesses per epoch), but `idle_window` does not measure the poller (E4). |
| 0c tooling | group rows, `contested_pct_of_cycles`, README, self-test | n/a | Done. |
| 0d eval contention | `train.eval_pass` counters, `cycles_per_contested` | about 1.0 | 0.998 / 0.999 for core0-core1; meaningless where the denominator is small (E5). |
| 0e loss stage | flash vs RAM labels, XIP counters | explain the gap | Explained: flash latency (887 vs 612 cycles). |
| 0f golden | `golden_results.csv`, `final_loss` / `final_accuracy` | identical across builds | Met: **G0**. |
| 0g stacks | `stack_pointer` row, README note | n/a | Documented only; see D7 and Step 1d. |
| 0h freeze B0 | profile and benchmark | repeatable | Repeatable, but B0 is frozen on a profile layout that differs from the benchmark's (E1). It is superseded by B0.5. |

---

### Step 0.5 - Profile what production runs, measure the benchmark binary, re-freeze the baseline

**1. Rationale.** Step 0 made the instruments honest about *what they
measure*. The `02bc8d3` data shows that they still measure the wrong
*things*. The profile binary has a different layout from the benchmark binary
(E1). The per-layer harness uses a twin network with stack buffers (E2). The
placement rows describe that twin (E3). One sanity row measures the probe
instead of the poller (E4). A derived ratio is reported when it is noise (E5).
The binary that produces the headline number has no counters at all (E6). And
part of production's input comes from a shared flash cache that nobody tests
(E7). In plain terms: we have a good stopwatch, but we were timing a rehearsal
in a different room from the performance. This step points the instruments at
the code, data and binary that actually run, then freezes a baseline (B0.5)
that every later step is compared against.

**2. Implementation.**

*0.5a. The harness profiles the production object (E2).*
- Delete `g_net`. Run the per-layer stages on `g_prod`'s own network
  (`state_.net`) and its own back-propagation buffers (`m_bp0` / `m_bp1`) and
  loss-gradient buffer, in exactly the roles `backprop_accumulate` gives them
  (`(I & 1) ? m_bp1 : m_bp0`). Expose them through a profile-only accessor
  (for example a `friend` declaration of the `test::profile` namespace guarded
  by `MEML_RUN_PROFILE`), not by making members public.
- No stack arrays may be passed to a kernel in the harness. If a stage needs
  scratch, take it from `g_prod`'s `State`, as production does.
- Initialise through `g_prod.Initialise(kProbeSeed)` and feed the same
  sample production uses, read the same way (from flash, via the same index
  path), so the inputs come from the same memory.
- **Acceptance:** `train.unexplained_per_sample` within ±0.5% of the pass
  (about ±750 cycles) on all three builds, **and** equal between the three
  builds to within ±100 cycles. The remaining common part should match what
  0.5d attributes to flash.

*0.5b. Placement and phases from the ELFs, for both binaries (E1, E3).*
- The profile prints, once, `offsetof` of every streamed array within
  `MLPOpticalRecognition` (weights, grad accumulators, squared-gradient
  averages, biases, cached inputs, activations, inner products, `m_bp0`,
  `m_bp1`, input and output scratch). Commit this offset table as a
  generated file (for example `benchmark_results/stream_offsets.csv`). The
  validator in Step 1b replaces this with `static_assert`s.
- `RunAndSaveBenchmarks.py` then computes **statically, without running the
  device**, for the benchmark ELF *and* the profile ELF of each build:
  - each stream's absolute address = the production instance symbol
    (`*::mlp_experiment_`, `g_prod`) + offset;
  - each hot loop head (the existing `kind=loop` detection; the main forward
    loop is at `forward + 0x30` today, the backward one at
    `AccumulateGradients + 0x32`);
  - a `rel_phase = (loop_head - stream) mod 16` row for every (hot loop,
    stream it reads) pair.
  Add a `binary` column (`benchmark` / `profile`) to `placement_results.csv`.
- Cross-check: in the profile ELF, instance + offset must equal the runtime
  address that the profile prints. A mismatch fails the run.
- Self-test fixture: today's ELF values must reproduce the hand analysis of
  1.5 (benchmark layer0.forward phases core0 0, core1 8, naive 12).

*0.5c. Counters in the benchmark binary, dormant condition only (E6).*
- In the dormant condition, the other core's dormant task becomes the same
  WFE + 10 ms alarm poller as in the profile. Its code, data and stack stay in
  the other core's bank, as the dormant task's do today. It accumulates
  group and other-group contested counts over each session and prints them
  with the session time. In the flooding condition no counters are read (the
  other core is busy flooding).
- A build-time switch keeps the old pure-WFE dormant task available. Run it
  once to show that the poller does not change the dormant time (within
  0.01%).
- Derived rows: per session, `contested_pct_of_cycles` for the benchmark, and
  a benchmark `cycles_per_contested` between pairs of builds (see 0.5f).
- **Acceptance:** the measured contested difference between core0 and core1
  benchmark sessions accounts for the 0.556 s gap within 10%. 1.5 predicts
  about 86.8M contested accesses.

*0.5d. Flash and the independence claim (E7).*
- Bracket a production `Train(1)` with the XIP counters (`XIP_MAIN0_ACCESS`,
  `..._CONTESTED`, `..._STALL_UPSTREAM`, `..._STALL_DOWNSTREAM`) and report XIP
  stall cycles per sample. That number should account for the ~450-cycle
  common residual in 0.5a.
- Add a third benchmark condition, `xip-flooding`: the other core reads a
  large flash region (not the dataset; for example the end of the image)
  through the cached XIP alias in a loop, evicting the shared XIP cache. Its
  code, data and stack stay in its own SRAM bank. Compare the session time
  with dormant.
  - If the time moves, the independence claim must be stated as "SRAM bus
    independence; flash is shared". Record the size of the effect.
    (Mitigations are out of scope here: the dataset does not fit in a bank.)
  - If it does not move, record that, and the claim covers flash too.

*0.5e. Measure the poller, not the probe (E4).*
- Replace `idle_window`: the probe core sleeps in WFE on a 50 ms alarm
  (instead of `busy_wait_ms`) while the counters run, so the only traffic is
  the poller's. Report it as `poller_footprint`, with accesses and contested
  counts on all ten banks.
- Also report the poller's accesses over a whole `train.epoch` (bank 0 for the
  core1 build, bank 4 for core0) as an explicit row, instead of leaving it
  readable only from `access_avg`.

*0.5f. Only report ratios that mean something (E5).*
- Emit `cycles_per_contested` only when the cycle difference between the two
  builds is at least 1M cycles; otherwise write `n/a`. Note the rule in
  `benchmark_results/README.md`.

*0.5g. Stack traffic, measured rather than assumed (D7).*
- For every `train.*` row, add the contested and access counts of the bank
  that holds the probe's stack: SRAM9 for core0 and naive, or the core-1
  group for core1 (stack accesses there cannot be separated from data
  accesses; record the stack-pointer range so this is clear). The question to
  answer: does production touch the stack in the hot loops at all? After
  0.5a the harness no longer does by design. The answer feeds Step 1d.

*0.5h. Freeze B0.5.*
- With 0.5a-0.5g in, run each profile twice and the benchmark once (dormant,
  flooding, xip-flooding). Record in a `baseline_B0.5` block in this file:
  per-layer cycles; `train.*` per sample; the `rel_phase` table for both
  binaries; group contested totals and percentages for the profile and the
  benchmark; benchmark times; G0; the commit hash. Cycles must repeat within
  0.1% and contested counts within 1%.
- **Check the model**: the benchmark core0 time should exceed core1's by
  2,064 x 42,053 cycles ± 10%, *if* the phases are unchanged. 0.5c changes the
  benchmark code, so the phases may well move. In that case use the new
  `rel_phase` table to predict which build is slow, then check it.

**Status (implemented without a board attached; nothing below has been run on
hardware).** Done and build-verified (core0/core1/naive profile builds pass
`validate_memory_placement.py`; `RunAndSaveBenchmarks.py --self-test` passes):
0.5a (harness runs on `g_prod`'s own network, `m_buf_a/b`, `m_pred`, `m_deriv`,
`m_bp0/1`, through a profile-only `Access` friend), 0.5b (offset printing,
`stream_offsets.csv`, static `rel_phase` rows with a `binary` column, ELF
cross-check; run against the real benchmark ELFs with the 0x7548 weight offset
it reproduces the 0 / 8 / 12 of section 1.5), 0.5d first bullet (XIP counters
around `Train(1)` and `Train(0)`, derived per training sample), 0.5e
(`poller_footprint` with the probe asleep in WFE, contested and access on all
ten banks), 0.5f (`n/a` rule), 0.5g (stack-bank rows). **Not done:** 0.5c
(counters in the benchmark's dormant task), 0.5d's `xip-flooding` condition, and
0.5h (the B0.5 freeze and every acceptance check: they need the board). The
unit test comparing the offset table with `offsetof` was replaced by the
script's runtime check (instance symbol + offset == printed address). The
`mlp/` change (friend declaration in `StaticMLP.h`) lives in the submodule and
must be committed there.

**3. Expected saving.** None. The product is a profile that measures
production, counters in the binary that produces the headline number, an
explicit statement about flash, and a baseline B0.5.

**4. Independence impact and mitigation.**
- 0.5a changes only the profile TU. The accessor is profile-only and must not
  add untagged code to the hot path (it is a plain member access; check with
  `nm`).
- 0.5c adds a poller to the benchmark's dormant task on the *other* core. It
  must stay entirely in that core's bank: extend the validator's
  dormant-task checks to the poller's symbols. Confirm the check fails if the
  poller is deliberately placed in the MLP's bank, as `AGENTS.md` requires.
- 0.5d's xip-flooder runs on the other core and lives in its bank; same
  check.

**5. Further testing.**
- Parser self-test for every new row and column (`rel_phase`, `binary`,
  benchmark contention, `poller_footprint`, XIP rows).
- A unit test that the profile's offset table equals the source's `offsetof`
  values. This guards against the table going stale before Step 1b replaces
  it.
- Run the profile twice to show repeatability.
- G0 must be unchanged: no MLP arithmetic changes in this step.

---

### Step 1 - Phase-pinned layout, a build-time phase guard, and the backward-floor diagnosis

**1. Rationale.** Section 1.5 shows that the cost of one loop depends on one
number: its head address minus its data's address, mod 16. Today that number
is whatever the linker happens to produce, so it changes with unrelated edits
and differs between the profile and the benchmark binaries. This step makes it
a **build decision**, checked by the build, so that:
- core0, core1 and naive take the same time (goal 1);
- the profile and the benchmark have the same phases and therefore agree;
- no later step can win or lose ±0.56 s by accident.

In plain terms: the code and the arrays it reads march through the same four
banks in step. We choose where each starts, so that they never tread on each
other, and we make the build fail if a change moves them.

A warning about the obvious approach: aligning both the code and the data to
16 bytes puts them at relative phase **0, the measured worst case**. That
would make timing deterministic, but uniformly 0.56 s slower. Alignment is
needed to fix the phase, but the phase it fixes must be chosen.

**2. Implementation.**

*1a. Fix every phase, then choose it.*
- **Code.** Add `-falign-functions=16 -falign-loops=16` to the MLP TUs
  (`MEML_MLP_TUS` in `CMakeLists.txt`). Every kernel and loop head then sits
  at a known phase (0) regardless of the code before it. Verify in the
  disassembly that all hot loop heads are 16-byte aligned. GCC can skip
  alignment padding in some cases; 1b catches that.
- **Data.** `alignas(16)` on `MLPOpticalRecognition` (so the instance is at
  phase 0 in every build) and on every streamed array in `StaticLayer`,
  `StaticMLP` (`m_bp0`, `m_bp1`) and `State`. `static_assert(offsetof(...) %
  16 == intended)` for each one. Arrays whose size is not a multiple of 16
  bytes are followed by explicit padding.
- **Two knobs, not one.** Every observation so far had the weights and the
  input at the same phase, so we do not know which of them the forward loop
  collides with:
  - `MEML_MLP_WEIGHT_PHASE` in {0, 4, 8, 12}: an offset applied to the weight,
    gradient-accumulator and squared-gradient arrays;
  - `MEML_MLP_ACT_PHASE` in {0, 4, 8, 12}: an offset applied to the
    activation streams (`m_cached_input`, `m_act_output`, `m_inner_products`,
    `m_bp0`, `m_bp1`).
  Both are compile-time constants; the build fails on any other value.
- **Sweep.** Because the phases are now fixed by construction, they are the
  same in every build and in both binaries, so the sweep needs only one build.
  Use the core0 profile. Run the 4 weight phases with the activation phase at
  8, then the 4 activation phases at the best weight phase: 8 runs, a few
  minutes each. For each run, record the per-stage contested counts and cycles.
  Choose the combination with the lowest total over layers 0-1 forward and
  backward. The data predicts that the weight phase should be 4, 8 or 12 (not
  0) relative to the loop heads. Record the table in this file.
- **Confirm across builds:** build core0, core1 and naive at the chosen
  point, run the profile and the benchmark, and check that the `rel_phase`
  tables (0.5b) are identical across the six ELFs.

*1b. A phase guard in the validator.*
- Add `check_hot_loop_phase` to `validate_memory_placement.py` and register it
  for the benchmark and profile builds, **naive included** (naive is the
  counterfactual and must be comparable):
  - the production instance symbol is at address mod 16 = 0 (the data
    phases then follow from the `static_assert`s in 1a);
  - each hot kernel's loop heads (forward main loop, backward main loop,
    RMSProp loop) are at the phase listed in a committed table,
    `scripts/hot_loop_phases.json`, which records for each kernel symbol the
    expected head phase and the stream phases chosen in 1a;
  - any mismatch fails the build, naming the kernel and both phases.
- Follow the `AGENTS.md` rule for new checks: first run it against today's
  core0 benchmark ELF (layer-0 forward at relative phase 0) and confirm it
  fails, then against the fixed layout and confirm it passes.
- When a later step changes a kernel's loop structure (Steps 2, 3, 4), the
  table entry is updated **deliberately** after a small sweep of that kernel,
  not edited to make the build pass.

*1c. Diagnose the backward floor now (throwaway, not committed).*
- In a scratch branch, place only `layer0`'s `AccumulateGradients` in SRAM8
  (`.scratch_x`) in the **naive profile build**. Naive has no placement
  constraint, so the validator needs no change. Measure its contested count
  and cycles against B0.5.
- If the ~4.1k contested per call disappears and the cycles drop by about the
  same amount, the backward floor is fetch-against-data and Step 5 is worth
  up to ~1.7 s per session at today's kernels. If it does not disappear, the
  model in 1.5 is missing something, and Step 5 must not be attempted until
  that is understood.
- Record the result in this file and delete the branch.

*1d. Decide where the stacks and any scratch-bank code live.*
- Today core 0's stack is in SRAM9, and core 1's stack is in `.core1_stack`
  inside the core-1 group. That is asymmetric: anything production keeps on
  the stack contends on core1 and not on core0. 0.5g says whether production
  actually does so.
- Options, to be chosen with the 0.5g and 1c numbers in hand:
  1. **Keep as is.** Valid if 0.5g shows that production's hot loops make no
     stack accesses. Add a validator or disassembly check that the hot kernels
     contain no `sp`-relative loads or stores inside their loops, so that this
     stays true.
  2. **Each core owns one scratch bank** (core 0: SRAM9, core 1: SRAM8, the
     SDK default) for its stack **and** its hot kernels (Step 5). This is
     symmetric by construction. It needs the stack's high-water mark (measure
     it by painting the stack) plus about 2 KB of kernels to fit in 4 KB.
  3. **Stacks in their own group** (core 0's moved into the core-0 group),
     leaving both scratch banks free for code. This is symmetric, but it moves
     the production stack of the core0 build.
- The recommendation is option 1 if 0.5g allows it, and option 2 if Step 5 goes
  ahead. Record the decision and its numbers in this file.

**3. Expected saving.**
- core0 benchmark: about -0.56 s (all three builds at about 41.24 s at B0.5).
  Whether 1a can also take a little off the clean builds depends on the sweep.
- Every later step: its measurement is no longer ±0.56 s of luck.
- 1c and 1d save nothing themselves; they decide Step 5.

**4. Independence impact and mitigation.**
- 1a changes alignment and adds at most 12 bytes of padding per array, inside
  the pinned bank. Check that the bank-capacity `ASSERT`s still pass.
- 1b only adds a check.
- 1c is throwaway and naive-only.
- 1d option 2 or 3 moves a stack and needs its own validator windows (see
  Step 5).

**5. Further testing.**
- The sweep table.
- At the chosen point: profile twice and benchmark once, on all three builds.
  **Acceptance: core0, core1 and naive dormant benchmark times within 0.01%
  (about 4 ms) of each other**, with benchmark contested counts (0.5c) equal
  within 1%.
- A negative control: build once with the worst sweep point and confirm that
  the counters and cycles move as the table predicts.
- A compile-only test for the `static_assert`s.
- G0 unchanged (layout changes no arithmetic).

---

### Step 2 - Skip the layer-0 input-gradient

**1. Rationale.** In plain terms: when training, each layer works out two
things: how to adjust its own weights, and what error to hand back to the
layer before it. Layer 0 is the first layer, so there is nobody to hand the
error to. Today it computes that error anyway (4096 multiply-adds per sample,
about half of layer 0's backward cost) and throws it away. It is the largest
single item, and it has no downside.

**2. Implementation.**
- In `StaticLayer.h`, make the method
  `template<bool NeedInputGrad = true> AccumulateGradients(...)`. When `false`:
  no zeroing loop for `delta_out`, no `delta_out[j] += es*w[j]` update, and
  `delta_out` is never dereferenced (it may be `nullptr`). The `g` and bias
  accumulation are unchanged. Keep the `noinline, noclone` attributes; this
  yields one extra instantiation (`<false>`), used only by layer 0.
- In `StaticMLP.h`, `backprop_accumulate<I>` calls
  `layer.template AccumulateGradients<(I > 0)>(...)`. This covers
  `TrainBatch` (both overloads) and the RL `AccumulatePolicyGradient`, none of
  which needs the network-input gradient. Leave `CalcGradients()` (autograd)
  untouched: it does return the network-input gradient.
- Optionally do the same for `UpdateImmediate` at layer 0 via
  `backprop_immediate<0>`, which has the same dead `delta_out`.
- The harness (after 0.5a) calls exactly what `backprop_accumulate` calls, so
  it picks this up automatically. Check that it does.
- The `<false>` kernel has a new loop: add it to `hot_loop_phases.json` after
  a 4-point sweep of `MEML_MLP_WEIGHT_PHASE` for this kernel alone (1b). If
  its best phase differs from the other kernels', the weight phase becomes
  per-layer.

**3. Expected saving.** About 8 cycles per iteration x 4096 = about 30k cycles
per sample (about 20% of the pass; 28-32k against the current 70.2k layer-0
figure). `layer0.AccumulateGradients` should fall from 70k to roughly 38-42k.
Weights, biases and the training trajectory are **bit-identical** (only
discarded values are skipped).

**4. Independence impact and mitigation.** None negative.
`AccumulateGradients<false>` is an `smlp::` member tagged `SMLP_CODE_ATTR`, so
the prefix and callee-closure checks cover it. Build core0 and core1 and `nm`
it anyway (template instantiations have surprised us before). The phase guard
covers the new loop.

**5. Further testing.**
- Final loss and accuracy equal to G0 exactly.
- A unit test that `AccumulateGradients<false>` leaves `g` and the bias
  accumulators exactly equal to `<true>`.
- The profile row above.
- Phase guard passes; benchmark core0, core1 and naive within 0.01%.
- `train.unexplained_per_sample` still within tolerance.

---

### Step 3 - Row-blocked forward pass

**1. Rationale.** In plain terms: to compute one output of a layer, the code
walks the whole input vector, and it does that again from scratch for every
output. Of the roughly 22 instructions per 4 multiply-adds, half re-load the
same inputs. Computing four outputs together loads each input once and uses it
four times, and the four running sums are independent of each other. The
forward kernel is limited by how many instructions and loads it issues, not by
arithmetic latency, so cutting instructions is what helps.

**2. Implementation.** In `StaticLayer::forward` (float branch only; keep the
fixed-point branch):
- Process rows in blocks of `R = 4`: pointers `w0..w3 = w + r*NIn`, and
  accumulators `a0..a3` initialised to the **biases** of the 4 rows. For each
  `j`: `x = input[j]; a0 = fmaf(w0[j], x, a0); ... a3 = ...`. Unroll `j` by 2
  (`#pragma GCC unroll 2`); the register budget is about 14 S registers. Per
  `j`: 5 loads + 4 `vfma`, against about 22 instructions per 4 MACs today.
- Handle `NOut % 4` with `if constexpr` tails (`R = 2`, then `R = 1`). Layer 2
  has 10 rows (2 blocks + 2).
- **Remove the `fp-contract=off` pragma** around `forward` (it existed for the
  4-way accumulator split, which this step replaces). Each row keeps a single
  sequential chain from the bias with fused FMA, so the result is
  bit-identical to the pre-Opt-3 implementation (`17bc32a`), but **not** to
  the current unfused 4-accumulator loop. Re-baseline G0 once and record G1.
- Keep the `m_cached_input` copy as an inline loop (the MLP TUs are built with
  `-fno-tree-loop-distribute-patterns`), and verify with `objdump` that no
  `bl memcpy` appears.
- Confirm in the disassembly that GCC emits post-incremented `vldm`/`vldr` with
  offsets and not extra address arithmetic. If it does not, fall back to
  `R = 8` or hand-scheduled intrinsics. Do not accept the step on source
  appearance alone.
- **Phases.** The four row streams are 256 bytes apart (64 floats), which is a
  multiple of 16, so all four sit on the same bank at every `j`. That is
  harmless for the single data port, but it means all five data streams share
  one fetch-collision pattern. Re-sweep `MEML_MLP_WEIGHT_PHASE` and
  `MEML_MLP_ACT_PHASE` for the new loop (4 + 4 runs, as in 1a). If no point is
  clean, try a row pitch of `NIn + 1` floats (padding each weight row) as a
  third sweep variable, and update `hot_loop_phases.json`.

**3. Expected saving.** Forward from about 39k to about 17-20k cycles, saving
20-22k per sample (about 14% of the pass). Layer 0 (25.3k) is the main target.

**4. Independence impact and mitigation.** The kernel must remain a plain
`SMLP_CODE_ATTR` member (no lambdas or untagged helper templates; if a helper
is unavoidable, follow the `AGENTS.md` untagged-code rule). Unrolled bodies
increase code size slightly: check that it fits the core's window and, if
Step 5 is planned, the scratch budget (1d). The phase guard covers the new
loop.

**5. Further testing.**
- Extend the `StaticMLPForwardAccumulatorTest` shapes to `NOut` in
  {1, 2, 3, 4, 5, 10, 32, 64} x `NIn` in {7, 10, 64, 65}.
- Check that final loss and accuracy equal the `17bc32a`-era golden
  (G1 = that value).
- Profile all three forward rows on core0, core1 and naive.
- Phase guard passes; benchmark core0, core1 and naive within 0.01%.
- Callee-closure check passes.

---

### Step 4 - Row-blocked backward pass

**1. Rationale.** In plain terms: for every weight, backward reads the running
gradient, adds to it and writes it back, and does the same for the error being
passed to the previous layer. That is six memory operations for two
multiply-adds, which is why the loop takes about 17 cycles per iteration.
Handling four output rows together lets the "error to the previous layer" for
all four rows be added up in registers and stored once, and reuses the input
value. Memory operations drop from 6 per 2 MACs to roughly 3. It keeps
unit-stride row access, which is why it should not repeat the failure of the
old j-outer interchange (Opt 4). Fewer memory operations per loop iteration
also mean fewer chances for fetch to collide with data, so this step should
lower the backward contention floor as well.

**2. Implementation.** `AccumulateGradients<NeedInputGrad>`:
- Compute `es0..es3` for a block of 4 rows into registers (not an array, to
  avoid stack temporaries) and add them to `m_bias_grad_accum`.
- For each `j`, with `x = input[j]`: `g0[j] += x*es0; ... g3[j] += x*es3`
  (4 loads, 4 FMA, 4 stores, independent).
- If `NeedInputGrad`: update `delta[j]` with the 4 rows in the **same
  sequential order as today**:
  `d = fma(es3,w3[j], fma(es2,w2[j], fma(es1,w1[j], fma(es0,w0[j], d))))`,
  with one load and one store of `delta[j]` per 4 rows. For the first block
  `d` starts at 0, so no zeroing pass is needed. This keeps the result
  **bit-identical to Step 3's output**. The 4-deep chain per `j` needs `j`
  unrolled by at least 2 to hide FMA latency; check the schedule. If the
  profile shows latency stalls, a tree sum is the fallback (then re-baseline
  the golden).
- Instruction count per 4 rows x 1 `j`: with `NeedInputGrad`, about 23 (40
  now); without (layer 0 after Step 2), about 13 (about 24 now).
- Tails for `NOut % 4` as in Step 3.
- `UpdateImmediate` / `CalcGradients` can get the same treatment later (they are
  RL paths, not on the benchmark path).
- Re-sweep the phases for the new loops and update `hot_loop_phases.json`, as
  in Step 3.

**3. Expected saving.** Backward after Step 2 is about 80k cycles. Blocked, it
should be roughly 48-52k, saving about 28-32k per sample (layer 0: ~40k ->
~23-25k; layer 1: ~35k -> ~20k; layer 2: ~6k -> ~3.5k). This is the biggest
uncertainty in the plan: it depends on how the M33 pipelines loads and stores.
Measure Step 2's output first and re-estimate.

**4. Independence impact and mitigation.** No library calls and no stack
arrays, so no extra pressure on either core's stack. If 1d chose option 1,
confirm that the new loops still contain no `sp`-relative accesses. Re-run the
callee-closure check, the inlining canary and the phase guard.

**5. Further testing.**
- A reference test comparing blocked and scalar `AccumulateGradients` for `g`,
  bias and `delta_out` across shapes (exact equality while the sequential
  order is kept).
- Final loss and accuracy equal to G1 exactly.
- The three backward profile rows on all builds, with their contested counts:
  record the new backward floor, which is Step 5's input.
- The full benchmark: flooding vs dormant within the 1% tolerance on the
  pinned builds, and core0, core1 and naive within 0.01%.

---

### Step 5 - Hot kernels in the scratch banks (if Step 1c confirmed the cause)

**1. Rationale.** The phase guard removes collisions that a good phase can
avoid. The backward floor (1.5) does not depend on phase: a loop body long
enough to sweep all four banks will always meet its data somewhere. The cure
is to fetch the code from a bank that holds no streamed data. SRAM8 and SRAM9
are 4 KB, unstriped and outside both groups, so code there cannot collide with
data in a group. After Steps 3-4 the hot kernels are small out-of-line
functions (budget about 2 KB per core).

**2. Implementation.**
- **Condition to start:** Step 1c showed that the floor is fetch-against-data,
  *and* after Step 4 the group's contested count is still above about 2% of
  epoch cycles.
- **Where:** as decided in Step 1d. With option 2, core 0's kernels go to
  SRAM9 (beside core 0's stack) and core 1's to SRAM8 (beside core 1's
  stack, moved back to the SDK default). With option 3, both stacks are in
  their groups and SRAM8/SRAM9 hold only code. Confirm from the link map what
  else the SDK places there.
- **Hook:** add `MEML_MLP_HOT_CODE` in `MemoryDefs.hpp` (pinned builds: the
  chosen scratch section; unpinned: blank, so the naive counterfactual stays
  unplaced) and a blank-by-default `SMLP_HOT_CODE_ATTR` in `mlp/Placement.h`.
  Per-shape kernels must be separate non-template-named functions or follow
  the `AGENTS.md` untagged-code rule (linker redirect plus validator check).
- **Validator:** add explicit scratch windows (`SCRATCH_X = [0x20080000,
  0x20081000)`, `SCRATCH_Y = [0x20081000, 0x20082000)`). Add a rule that a
  pinned build's tagged hot kernels are in *its* scratch bank and in no other
  core-specific region, and a check that the other core's code, data and stack
  are not in that bank. In `check_callee_closure`, allow the pinned core's
  scratch bank as "inside the pinned window", for the pinned core only.
  Update `AGENTS.md` to the three-region model. Confirm that each new check
  fails before the fix.
- The phase guard no longer applies to kernels in a scratch bank (an unstriped
  bank has no phase); mark them as such in `hot_loop_phases.json`.

**3. Expected saving.** Up to the remaining group contested count after Step 4
(about 1.7 s per session at today's kernels; re-estimate from the Step 4
counters).

**4. Independence impact and mitigation.** This is the one step that changes
the memory model. Mitigation: the validator rules above, the `AGENTS.md`
update, and the flooding-vs-dormant benchmark (`timing_verdict`), which
remains the end-to-end proof and must PASS on both pinned builds. Naive is
unaffected.

**5. Further testing.** Profile and benchmark on all three builds. Group
contested count for the moved kernels near zero. Flooding vs dormant within
tolerance. Core0 and core1 within 0.01% of each other. The stack high-water
mark still fits beside the code (if option 2).

---

### Step 6 (optional) - Micro-batch 4 samples through forward/backward

**1. Rationale.** After Step 4, the remaining backward cost is dominated by
loading and storing the gradient accumulator `g` once per sample. Processing 4
samples per pass over the weights loads each weight (forward) and each `g` cell
(backward) once per 4 samples instead of once per sample. Decide only after
re-profiling Step 4: expect 8-12% of the original pass, with low confidence.

**2. Implementation.** In `TrainBatch`, process samples in groups of 4:
- `forward` computes 4 samples' accumulators from one weight load (the four
  accumulators are the four samples, which keeps each sample's sum in its
  original order);
- `AccumulateGradients` does `g[i][j] += sum_b x_b[j]*es_b[i]`.
It needs per-sample caches (`m_cached_input`, `m_act_output`,
`m_inner_products`, or precomputed `es`) x4, about 6 KB extra per core, inside
the single `State` object so that the existing `MEML_MLP_DATA` annotation
covers it. A batch size of 128 is divisible by 4; the last batch (3823 mod 128
= 79) is not, so keep the existing single-sample path for the tail. New loops
mean new phase sweeps and new `hot_loop_phases.json` entries.

**3. Expected saving.** About 10% of the original pass (about 15k cycles per
sample), uncertain.

**4. Independence impact and mitigation.** More core-local state, still in one
tagged object; add a `static_assert` or linker capacity check against the bank
size. Both paths (blocked and tail) must be in the bank. Numerics: the `g`
summation order changes (not bit-exact), so the loss and accuracy trajectories
shift slightly.

**5. Further testing.** A convergence regression: final loss and accuracy after
a full session must stay within a tolerance of the previous golden (suggest
±0.5% accuracy and ±2% loss over several seeds). Unit tests for the tail path
(batch sizes 1, 3, 4, 5, 79, 128). Phase guard and contested counts.

---

### Step 7 (optional) - RMSProp and loop-level cleanup

**1. Rationale.** After Steps 2-4, the per-batch work (about 3% today, mostly
RMSProp at about 72 cycles per weight) becomes roughly 10-12% of the run. Its
gain (about 1-1.5 s) is of the same size as the layout lottery, so it can only
be judged once the phase guard exists, which is why it comes last.

**2. Implementation.** In `ApplyAccumulatedGradients`:
- replace `lr / (sqrt(v) + eps)` with a fast reciprocal square root (integer
  seed plus one or two Newton iterations), removing one `vsqrt` and one `vdiv`
  per weight (about 28 cycles); keep the `maxLR` clamp;
- fuse the clip, EMA, update and accumulator-clear into one pass with fewer
  flag transfers (`vmrs`); use `vmax`/`vmin`-style selects.
Leave the final evaluation pass alone: it provides `Result` loss and accuracy
and the benchmark's correctness check.

**3. Expected saving.** About 1-1.5 s per session (about 5% of the faster run).
Confirm first with the `batch.rmsprop.*` rows (468k cycles per batch today).

**4. Independence impact and mitigation.** Pure arithmetic inside already
tagged functions; no new callees if the fast rsqrt is inline (the
callee-closure check verifies this). The RMSProp loop is in
`hot_loop_phases.json`; update its entry.

**5. Further testing.** Compare weights after a batch against the exact formula
(relative error bound), plus the convergence regression from Step 6.

---

## 4. Process for every step

1. Build all three configs exactly as `RunAndSaveBenchmarks.py` does
   (`build-core0`, `build-core1`, `build-naive`, and the `-profile` dirs). The
   POST_BUILD validator must pass, including the callee-closure check and,
   from Step 1 on, the phase guard.
2. Use `arm-none-eabi-nm` to spot-check that any new symbol's address is in the
   right window.
3. If the step changes a loop, run the per-kernel phase sweep, update
   `hot_loop_phases.json` deliberately, and record the sweep table here.
4. Run the profile build for all three: per-layer stages (on the production
   object, after 0.5a), the `train.*` rows and the contention rows. **A step is
   accepted on the production-path rows and the benchmark, not on per-layer
   rows alone**: they must move the same way, and `train.unexplained_per_sample`
   must stay within tolerance.
5. Compare every number with B0.5 (Step 0.5h), including the benchmark's own
   contention counters (0.5c). A step that lowers cycles but raises the group
   contested percentage has probably moved a phase; the guard should have
   caught that, so find out why it did not.
6. Run the unit suite (`RUN_TESTS_OR_BENCHMARKS=tests`).
7. Only then run the full benchmark (dormant, flooding, xip-flooding) and
   commit the CSV rows with the commit hash (the script requires a clean tree).
   Core0, core1 and naive dormant must agree within 0.01%.
8. Keep each step in its own commit, so a regression can be reverted without
   losing the others. Never merge two steps before both are measured.

## 5. Open questions to resolve early

- **Which stream does the forward loop collide with:** the weights or the
  input? They have always had the same phase. The two-knob sweep in 1a
  answers this.
- **Why is the backward floor phase-independent?** The hypothesis (the loop
  body sweeps all four banks) is untested. Step 1c tests the cure; the
  per-phase sweep in 1a should show that the floor stays flat.
- **Layer-2 forward does not fit the rule cleanly.** It is a 502-byte loop:
  phase 4 gave 155 contested accesses in one build and 66 in another. It is
  small (1.7k cycles), but it shows that the rule is a lookup table found by
  measurement, not a law. The guard records measured phases, not predicted
  ones.
- **Does `-falign-loops=16` align every hot loop head**, or does GCC skip
  padding when it would be large? The guard detects this; if it happens, use
  `-falign-loops=16:16` or explicit `.p2align` in the kernel.
- **What is the +0.07 s common to core0 and core1 since `f1b9488`?** (About
  275 cycles per sample.) The benchmark counters (0.5c) and the `noinline` call
  overhead are the first things to check.
- **Does the other core's flash traffic slow the MLP?** 0.5d answers this, and
  it decides the wording of the independence claim.
- **What else lives in SRAM8 and SRAM9 in each build**, and what is each
  stack's high-water mark? Read the link map and paint the stacks before
  Step 1d.
- **Does production make any stack accesses inside the hot loops?** 0.5g
  answers this.
- **Is `-ffp-contract=fast` really the GCC default** under `-std=gnu++20`
  here, so that pre-Opt-3 builds used fused FMA? The Step 3 bit-exactness
  claim depends on it; the `vfma` in the `build-core0` disassembly says yes.
- **What are the real M33 load/store costs** for single-register
  `vldm`/`vstm`? Settle this by measuring Step 2, not by assumption.
- **Is the 1-cycle cost per contested access universal?** It holds for the
  core0-core1 differences (0.998, 0.999). Check it again in the benchmark
  (0.5c) and after Step 4, when stores are denser.
