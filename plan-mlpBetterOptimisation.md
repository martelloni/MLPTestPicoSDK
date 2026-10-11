# Better StaticMLP optimisation plan

Written for: the maintainer and whoever implements the steps. Follow
`AGENTS.md` for every step that adds or changes MLP code (core0 and core1
builds, `validate_memory_placement.py`, `nm` spot-checks).

This plan supersedes the targeting of `plan-mlpOptimisationPrompt.md`
(Optimisations 1-5). It is based on the benchmark CSVs, the source and the
disassembly of the existing ELFs. **No hardware was available when it was
written**: every "expected saving" below is a model-based estimate, not a
measurement, and each step says how to confirm it.

---

## 1. Findings

### 1.1 The problem

| Build | Benchmark time (10 epochs, one session) |
|---|---|
| Legacy heap MLP (`0732282`, core0-baseline) | 45.64 s |
| First StaticMLP (`0e9e71b`) | 45.50 s |
| "Peak" StaticMLP (`8fb216e`) | 42.52 s |

Removing the heap gained nothing; five optimisations gained 6.5% in total.

### 1.2 Where the time actually goes

One session is 10 epochs x 3823 samples = 38,230 samples = about 6.38 G
cycles at 150 MHz. Per-sample cycles come from `profiling_results.csv`
(`8fb216e`, core0):

| Stage | Cycles/sample | Share of pass |
|---|---|---|
| layer0.forward (64->64) | 25.2k | 16% |
| layer1.forward (64->32) | 11.9k | 8% |
| layer2.forward (32->10) | 1.6k | 1% |
| loss (softmax-CE, `expf`/`logf`) | 0.7k | 0.5% |
| layer2.AccumulateGradients | 5.9k | 4% |
| layer1.AccumulateGradients | 37.3k | 24% |
| **layer0.AccumulateGradients** | **74.2k** | **47%** |
| **Full pass** | **156.8k** | 100% |

- Forward + loss + backward = 156.8k x 38,230 = 5.99 G cycles = **94% of the
  benchmark**. The profile and benchmark agree (e.g. `dbc17e9`: profile
  predicts 41.5 s, benchmark measures 42.83 s).
- Final evaluation pass in `Train()`: about 2.3%.
- Everything else in `TrainBatch` (RMSProp apply, grad-norm clip, gradient
  zeroing, shuffle): about 3.7%.
- **The training loop is not the bottleneck.** Loop-level overhead is about 6%
  in total and perhaps 1-2% of that is reachable.
- Backward is **75%** of the pass; layer-0 backward alone is **47%**.

Algorithmic floor for context: 6464 weights give 6464 forward MACs plus
2 x 6464 backward MACs, about 19k MACs per sample. 156.8k cycles is about
**8 cycles/MAC**. A scalar M33 should reach roughly 2-3.

### 1.3 What the disassembly shows (`StaticLayer.h`)

Both the profile ELFs and `build-core0` have the same loop shapes.

- **Forward inner loop** ([StaticLayer.h:179](mlp/StaticLayer.h#L179)): per
  4 MACs, 8 `vldr` + 4 `vmul` + 4 `vadd` + 4 integer instructions, about 22
  instructions. Measured 24.6 cycles/iteration, so it is **issue/load bound
  at about 1 IPC**. The same input vector is reloaded for every output row, so
  half of all loads are redundant. The 4-accumulator unroll (Opt 3) did not
  help because the loop was never FMA-latency bound (`17bc32a` -> `665faab`:
  42.93 -> 42.83 s).
- **Backward inner loop** ([StaticLayer.h:268](mlp/StaticLayer.h#L268)): per
  `j`, 10 instructions for 2 MACs: 4 loads (g, input, delta, w), 2 stores,
  2 `vfma`, `cmp`, `bne`. Measured **18.1 cycles/iteration** (74,195 / 4096),
  about 9 cycles/MAC. `g[j]` and `delta_out[j]` do a load-modify-store round
  trip on every (i,j). Single-register `vldmia`/`vstmia` are not free: this
  loop is dominated by memory-operation cost, not by FP work. (The M33 TRM
  figures have not been checked; the measured cycles are consistent with
  roughly 2 cycles per load/store.)
- **Dead work**: `layer0.AccumulateGradients` computes `delta_out` (4096 MACs)
  that nothing consumes; `backprop_accumulate<0>` has no earlier layer
  ([StaticMLP.h:621](mlp/StaticMLP.h#L621)). That is roughly half of the
  47% layer-0 backward cost.
- **RMSProp** ([StaticLayer.h:290](mlp/StaticLayer.h#L290)): about 25
  instructions per weight, including one `vsqrt` and one `vdiv`, for about
  2M weight updates per session. Small, but it becomes about 12% of the run
  once Steps 2-4 land.

### 1.4 Why the previous optimisations did not help

| Optimisation | Targeted | Share of pass | Result |
|---|---|---|---|
| 1 branchless ReLU | ~96 activations/sample | <0.3% | 45.50 -> 45.29 s |
| 2 `__restrict` | enabler | n/a | -> 44.68 s |
| 3 4-acc forward | forward (wrong premise: latency) | 25% | ~0 (42.93 -> 42.83 s) |
| fp-contract=off | forward | 25% | -3k cycles forward in profile; ~-0.3 s benchmark |
| 4 backward j-outer interchange | backward | 75% | **regressed** (likely strided scalar `w[i*NIn+j]` loads and a two-pass split; build not disassembled) |
| 5 skip layer-0 cached-input copy | 256 B memcpy | 0.04% | noise / reverted |

Nothing touched the layer-0 dead work or the backward load/store traffic.
The plan's assumption that strided access is free because there is no data
cache does not hold in practice.

### 1.5 Benchmark noise exceeds the gains: placement matters

Benchmark time for identical hot-loop source swings between 42.5 and 44.7 s
with unrelated changes:

- `87d6af3`: core0 42.93 s, core1 44.68 s (same source).
- `17bc32a` 42.93 s -> `41b62ad` ("noinline wrapper") 44.47 s.

The profile gives a clean fingerprint. The 64->32 `forward` kernel contains
the same 74-byte inner loop in all three profile builds:

| Profile build | `layer1.forward` cycles | Loop head addr mod 16 | Code vs data bank group |
|---|---|---|---|
| core0 | 11,855 | 8 | same 4-bank group (0x2000xxxx) |
| core1 | 11,847 | 0 | **different** (code 0x2004xxxx, `net` 0x2000xxxx) |
| naive | 12,879 | 0 | same 4-bank group |

Naive is exactly **1024 cycles** slower (2 cycles x 512 inner iterations).
Data pointers are at mod 8 in all three.

### 1.6 Hypothesis: code fetch and hot data collide in the SRAM bank stripe

> RP2350 main SRAM is 8 banks arranged as two word-striped groups of 4
> (`0x20000000-0x2003FFFF` and `0x20040000-0x2007FFFF`). Each bank serves one
> access per cycle. With `PICO_COPY_TO_RAM=1` all hot code is in SRAM. When a
> kernel's instruction fetches and its weight/input/gradient streams are in the
> **same 4-bank group**, they can hit the same bank on the same cycle and one
> stalls. Because the streams advance one word per access and the stripe is
> 16 bytes, whether they collide every iteration, never, or sometimes depends
> on the **relative address phase mod 16** of the code and the data. The phase
> is decided by link order, which is why unrelated edits shift timing by
> multiples of 2 cycles x iterations.

Evidence for it: the single table above (same instructions, same data phase,
only code phase/group differs, and the delta is an exact multiple of the
iteration count) and the quantised, source-unrelated swings in 1.5.
Evidence against / unknown: it is one data point; the backward loop's
1.8 cycles/instruction might be partly this and partly raw load/store cost.
**It has not been tested.** Step 1 below is the test.

If true, the fix is **not** to chase alignment but to remove the sharing:
put the few hot kernels in a bank no data stream or other core touches. The
RP2350 has two small unstriped banks, SRAM8 (`scratch_x`, 0x20080000) and
SRAM9 (`scratch_y`, 0x20081000), 4 KB each, outside both 256 KB windows. They
exist for this purpose in the SDK (`__scratch_x()`/`__scratch_y()`). One
per core preserves independence: core 0's hot code in one, core 1's in the
other, and each core's code bank is never touched by the other core.

### 1.7 Independence leak found along the way (not caught by the validator)

Today the hot path of the **pinned core1** benchmark build
(`build-core1/MLPTestPicoSDK.elf`) calls code that lives in core 0's bank:

| Callee | Address (core 0 bank) | Call sites |
|---|---|---|
| `memcpy` | 0x2000726c | 3 (one per `forward()`: the `m_cached_input` copy), every sample |
| `memset` | 0x20007190 | per backward layer (delta zeroing), `Initialise`, `Train` |
| `__wrap_expf` | 0x200026d8 | about 30 sites in `Train` (softmax-CE loss and eval) |
| `__wrap_logf` | 0x20002838 | `Train` (loss) |

`validate_memory_placement.py` only checks the addresses of `smlp`/
`MLPOpticalRecognition` symbols, not their callees, so it passes. These are
cheap and measured cross-core impact is tiny (0.09%), but they contradict the
"zero code/data touching core 0's bank" guarantee in `AGENTS.md`. The fix is
partly free (Steps 3 and 4 remove `memcpy` and `memset`) and partly new work
(`expf`/`logf`, Step 0).

### 1.8 Results after Step 0 (`070f5aa`) and what they changed

Dormant benchmark: core0 **43.52 s**, core1 **42.97 s**, naive **42.97 s**
(all three were 42.52 s at `8fb216e`). Per-layer profile, per sample:

| | core0 | core1 | naive |
|---|---|---|---|
| `layer1.forward` | 11,968 | 11,960 | **12,992** |
| `compute_loss` | 2,315 | 2,315 | 2,315 (was 728) |
| full pass | 155,034 | 154,904 | 156,027 |

- **Step 0 caused a regression:** `compute_loss` is 1.6k cycles/sample slower
  (private Cephes `expf`/`logf` versus pico's). That is about 0.4 s per
  session, exactly the rise in core1 and naive. Fixed in Step 0.5 (2d).
- **Core 0's extra 0.55 s (about 2.2k cycles/sample) is not visible in the
  profile**, which says core0 and core1 are within 0.1%. The profile harness
  calls the layer kernels directly and always executes on core 0; production
  runs the inlined backward in `Train` on the pinned core. So the profile is a
  diagnostic for the kernels it calls, not a proxy for the benchmark.
  (This matches the header comment of `ProfileForwardBackward.hpp`: an earlier
  regression was whole-function register allocation, invisible to kernel-level
  views.)
- **The layer-1 forward +1024 on naive reproduced exactly** with a different
  link layout, so it is deterministic placement, not noise.
- **The simple phase-mod-16 explanation in 1.6 is not supported.** Last time:
  naive loop head mod 16 = 0 (slow), core0 = 8 (fast). Now: naive = 8 (slow),
  core0 = 4 (fast). Data phases are identical (8) in both. Layer 0 is
  insensitive (loop heads 0/0/12, times equal). The placement logging also
  omitted streams the kernel touches. The bank-collision idea stays plausible,
  but the mechanism is unproven: Step 0.5 tests it with hardware contention
  counters rather than address arithmetic.
- Useful new data: per-batch RMSProp is 481k cycles (74 cycles/weight) and
  the whole per-batch overhead is 4.5k cycles/sample (2.9% of the pass), which
  supports Step 6.

---

## 2. Step overview and expected cumulative effect

Per-sample cycles are estimates from instruction counts; "Cum." is the
estimated full-pass time after the step.

| # | Step | Est. cycles saved/sample | Cum. pass | Risk |
|---|---|---|---|---|
| 0 | Measurement guard rails + callee-closure validator (done, `070f5aa`) | +1.6k (loss regression) | ~158k | none |
| 0.5 | Production-path probe, bus-contention counters, fix the `expf`/`logf` cost | -1.2 to -1.6k (loss) | ~157k | low |
| 1 | Test the placement hypothesis (alignment control, then scratch banks) | 0-12k (mostly variance removal) | ~145-157k | medium (linker) |
| 2 | Skip layer-0 input-gradient | ~33k | ~124k | very low |
| 3 | Row-blocked forward | ~19-21k | ~104k | low |
| 4 | Row-blocked backward | ~32-36k | ~70k | medium |
| 5 | Micro-batch of 4 samples (optional) | ~8-12% of original | ~55-60k | high |
| 6 | RMSProp / loop-level cleanup (optional) | ~1-2.5 s per session | n/a | low |

Steps 2-4 together: about **1.8-2.2x** end-to-end (about 19-24 s per session
against 42.5 s today). Step order is by value/risk, but **Steps 0 and 0.5
first, Step 1 early** so that later gains can be measured above the placement
noise.

---

## 3. Steps

### Step 0 - Guard rails: stable measurement and a callee-closure check

**1. Rationale.** Placement noise (up to 4%) is larger than the gains of the
smaller steps, and the validator cannot see cross-core callees (1.7). Without
fixing both, "faster" cannot be trusted and the independence claim has holes.

**2. Implementation.**
- **Profile as a kernel-level diagnostic.** The DWT profile
  (`ProfileForwardBackward.cpp`) is deterministic to about 0.1% for the
  kernels it calls, but **does not run the production `Train` path** (see
  1.8); Step 0.5 adds that. Treat the 40 s benchmark as final confirmation.
  Extend the profile to print and record:
  - stage rows for RMSProp apply (`ApplyAccumulatedGradients` per layer),
    gradient zeroing and the grad-norm step, so the 6% loop overhead is
    measured rather than inferred;
  - the address of each hot kernel and of each layer's `m_weights`,
    `m_grad_accum`, `m_cached_input` (log `addr % 16` too). Add them as extra
    CSV columns or a sibling CSV in `RunAndSaveBenchmarks.py`
    (`parse_profile_transcript`), so every profile row carries its placement.
- **Golden numerics.** Record, per commit, the session's final loss and
  accuracy for a fixed seed (the CSV already has `correctness_identical`;
  add the actual values). Bit-exact steps (2, 4) must reproduce them
  exactly; Step 3 re-baselines them once (see Step 3).
- **Reference tests.** `tests/unit/StaticMLPForwardAccumulatorTest.cpp`
  already compares `forward` with a scalar reference over NIn = 64/65/10/7.
  Add the same for `AccumulateGradients` (g, bias accumulators, `delta_out`)
  over shapes with `NOut % 4 != 0` and `NIn % 2 != 0`.
- **Callee-closure validator** (new check, per the `AGENTS.md` rule; register
  it in the pinned-build branch of `main()`): disassemble every symbol with the
  `_MLP_SYMBOL_PREFIXES` and fail if any `bl`/`b` target (and any literal-pool
  function pointer) lies outside the pinned core's window. Mirror
  `check_cmsisdsp_symbol_placement`. **Confirm it fails first** on current HEAD
  (it must report `memcpy`, `memset`, `__wrap_expf`, `__wrap_logf` in
  `build-core1`), then fix them (below and Steps 3-4) and confirm it passes.
- **`expf`/`logf`.** Add private `smlp::`-namespace, `SMLP_CODE_ATTR` versions
  used by `CategoricalCrossEntropyLogits` (and any other hot-path caller),
  so they are tagged and covered by the existing prefix check. Do **not**
  simply redirect `__wrap_expf` into core 1's bank: core 0 code (the flooder,
  stdio) also calls it, which would create the mirror-image leak. Numerics
  differ slightly from pico_float's `expf`/`logf`; either port that algorithm
  verbatim for bit-identity or accept a one-off golden re-baseline, and check
  the loss/accuracy stay within tolerance over a full session.

**3. Expected saving.** None; this step is about trust. It may add or remove
a few hundred cycles in the loss stage (0.5% of the pass).

**4. Independence impact and mitigation.** This step *improves* independence:
it closes the leak in 1.7 and makes it enforceable. Risk: the private
`expf`/`logf` must be tagged or they will land in the default bank; the new
validator check is exactly the safety net.

**5. Further testing.** Unit tests for the new `expf`/`logf` against
`std::expf`/`std::logf` over the logit range seen in training (including large
negatives and the 10-way log-sum-exp). Run the validator on `build-core0` and
`build-core1`; spot-check with `arm-none-eabi-nm` as `AGENTS.md` requires.

---

### Step 0.5 - Production-path measurement, bus-contention counters, and the `expf`/`logf` cost

Do this after Step 0 (commit `070f5aa`) and **before Step 1**. See 1.8 for what
the Step 0 results showed and why this step is needed.

**1. Rationale.** Step 0's results (1.8) exposed three gaps that make Step 1
impossible to judge:
1. The per-layer profile does not run the production code (backward is inlined
   into `MLPOpticalRecognition::Train`, with its own loop addresses) and it
   always runs on core 0, even for the "core1" build. It cannot see core 0's
   extra +0.55 s in the benchmark.
2. The hypothesis in 1.6 is currently supported by one data point and
   contradicted by another (loop phase mod 16 does not predict which build is
   slow). Instead of guessing from addresses, the RP2350 can **count contested
   SRAM accesses directly**. Bank collisions are testable, not inferable.
3. Step 0's private Cephes `expf`/`logf` made `compute_loss` 3x slower
   (728 -> 2,315 cycles/sample, about +0.4 s per session) and that must be
   removed before it hides the effect of later steps.

**2. Implementation.**

*2a. Production-path probe (new profile mode, same binary as the benchmark).*
- In `tests/ProfileForwardBackward.cpp` add `RunTrainPathProfile()`, called
  from `main.cpp` in `MEML_RUN_PROFILE` after the per-layer profile.
- It must call the **real facade** `MLPOpticalRecognition::Initialise(seed)`
  and `Train(epochs, lr)`, not a copy of the loop. Only then are the measured
  instructions the benchmark's instructions at the benchmark's addresses.
  Measure with the DWT `CYCCNT` (already enabled by `EnableCycleCounter()`):
  - `T0` = cycles for `Train(0, lr)` (a weight-less evaluation pass over all
    3823 samples: forward + loss only);
  - `T1` = cycles for `Train(1, lr)` right after `Initialise(seed)`;
  - report `T0`, `T1`, `(T1 - T0) / 3823` per sample (training cost without
    the evaluation pass), and the ratio to the sum of per-layer stage averages
    from the existing profile (the "unexplained" difference is the
    production-versus-harness gap). One epoch is about 4 s, well inside the
    28 s 32-bit `CYCCNT` wrap at 150 MHz. Repeat 3 times and print each.
  - Do not mask interrupts for the whole epoch (USB stdio would starve);
    the few USB ISR bursts are small and identical across builds. Print the
    min of the 3 repeats.
- **Run it on the pinned core.** For `MEML_MLP_RUNS_ON_CORE=1` launch the
  probe on core 1 (reuse the launch mechanism of `TestRAMIndependence`) with
  core 0 idle in WFE exactly as in the "dormant" benchmark run; for `0` and
  the unpinned build run on core 0. `CYCCNT` is per core: enable it on the
  core that runs the probe. Do the same for the existing per-layer profile so
  its "core1" rows really measure core 1.
- Add rows to `profiling_results.csv` with new stage names: `train.eval_pass`,
  `train.epoch`, `train.epoch_minus_eval_per_sample` (extend
  `parse_profile_transcript`/`RE_PROFILE_STAGE` accordingly).

*2b. Bus-contention counters (the direct test of 1.6).*
- RP2350 `busctrl_hw` has 4 saturating 24-bit counters
  (`busctrl_hw->counter[0..3]`, each with `perfsel`/`perfctr`, enabled by
  `busctrl_hw->perfctr_en`). Events per arbiter include `SRAMn_ACCESS` and
  `SRAMn_ACCESS_CONTESTED` for n = 0..9 (`hardware/regs/busctrl.h`,
  `BUSCTRL_PERFSELx_VALUE_SRAMn_ACCESS_CONTESTED`). A contested access means
  the arbiter delayed one master because another wanted the same slave in the
  same cycle. With the other core idle (dormant) and no DMA, contention can
  only be between this core's own fetch and data ports.
- In the per-layer profile, bracket each stage with counter snapshots in
  passes of 4 counters: pass 1 = `SRAM0..3_ACCESS_CONTESTED`, pass 2 =
  `SRAM4..7_ACCESS_CONTESTED`, pass 3 = `SRAM8, SRAM9` contested plus
  `SRAM0` and `SRAM4` `ACCESS` as sanity totals. Re-run the 100-iteration loop
  once per pass (the cycle numbers must reproduce between passes; if they
  drift, the counters are perturbing the measurement and the pass is invalid).
  Reading a counter is a 32-bit load from the APB-side peripheral bus: do it
  outside the timed bracket (read before `t0`, after `t1`) so the load itself
  does not add contention inside the bracket.
- Add a CSV, `benchmark_results/contention_results.csv`, with columns
  `timestamp, git_commit, build_config, stage, sram_bank, contested_avg,
  access_avg, log_file`, written by the same script path as
  `placement_results.csv`.
- Do the same bracket around the production-path probe of 2a (whole
  `Train(1)`): one total per SRAM bank per pass.

*2c. Placement data for everything the kernels touch.*
- Extend the `PrintPlacement` list with the streams currently missing:
  each layer's `m_inner_products`, `m_act_output`, `m_biases`,
  `m_bias_grad_accum`; the harness `buf_a`, `buf_b`, `pred`, `deriv`, plus the
  stack pointer at probe entry, and in the production probe the
  `MLPOpticalRecognition` instance (`state_`, `shuffle_indices`).
- Extend `elf_code_placement_rows` to emit, per hot symbol, the address of
  **every inner loop** (a backward conditional branch whose target is within
  the symbol) as `kind=loop`, with loop length and `addr % 16` of the head.
  Include `MLPOpticalRecognition::Train` (backward loops are inlined here) and
  the `for_each_layer` lambda that runs RMSProp. Use `objdump -d` as in the
  analysis snippet in this plan's history, run by the script after each
  build. These rows are the join key between the cycle/contention results
  and code position.

*2d. Remove the `expf`/`logf` regression.*
- Exit criterion: `compute_loss` stage in the profile back to **<= 1,000
  cycles** (the baseline with pico's routines was 728; each 100 cycles is
  about 0.1 s per session), with `FastMathTest` still green and no callee
  outside the pinned bank.
- Ways to get there, in order of simplicity (profile after each):
  1. Specialise for the **restricted domains the loss actually uses**, in
     `CategoricalCrossEntropyLogits`: after subtracting the max, `exp` is only
     needed for `x <= 0`; the log is only needed for a sum in `[1, n_classes]`.
     Add `exp_nonpos_f32` (no NaN/overflow/sign branches, clamp at -87) and
     `log_ge1_f32` (reduce by exponent bits, one short polynomial). Keep the
     general `exp_f32`/`log_f32` for other callers and tests.
  2. Round with `vcvtn`/`vrintn` instead of the `x < 0 ? -0.5f : 0.5f` branch,
     use Horner with `fmaf`, and build `2^n` by integer add into the exponent.
  3. If still above target: port pico_float's table-based `expf`/`logf`
     (BSD-3, in `pico_float`) into `smlp::` as tagged code. Higher effort
     because the SDK versions are assembly; only do this if 1 and 2 miss.
- Re-baseline the golden loss/accuracy once, as Step 0 did, and record that
  commit hash.

**3. Expected saving.** About 1.2-1.6k cycles/sample from 2d (up to about
0.4 s per session, recovering core1/naive to 42.5 s). The rest of the step
produces no saving but should decide Step 1: it identifies where core 0's
extra 2.2k cycles/sample comes from, and whether the layer-1 forward +1024
penalty is bus contention.

**4. Independence impact and mitigation.**
- Counters read `busctrl_hw` (an APB peripheral, not RAM) and only run in
  profile builds; no change to benchmark/test builds' memory layout. Keep
  every new probe/printing function in the profile TU (`MEML_RUN_PROFILE`
  only) so none lands in the pinned banks of the benchmark build.
- The probe on core 1 (2a) must keep core 0 idle in WFE, otherwise the
  "dormant" condition is violated and the numbers are not comparable. Report
  that the idle core issued no SRAM accesses by also printing core-0-side
  `SRAMx_ACCESS` totals for a sample window where nothing is expected to run.
- New `expf`/`logf` specialisations are `smlp::`, `SMLP_CODE_ATTR`: the
  existing prefix and callee-closure checks cover them. Confirm the callee
  check still passes on `build-core1` after 2d.
- Counter results are **read-only evidence**: Step 0.5 changes no placement.

**5. Further testing.**
- `FastMathTest`: ULP/relative-error bound vs `std::exp`/`std::log` over the
  restricted domains, including `x = 0`, `x = -87`, `x = -100` (must give 0,
  not denormal garbage), sum = 1 and sum = `n_classes`; a unit test that
  `CategoricalCrossEntropyLogits` loss and gradient stay within the previous
  tolerance over a full-range logit sweep.
- Self-test of the script parsers (`run_self_test`) for the new stage names,
  `contention_results.csv` rows and `kind=loop` rows.
- Reproducibility: run the profile twice per build and require cycle counts
  to agree within 0.1% and contested counts within 1%, or the counters are
  not usable.
- **Acceptance run:** full benchmark for core0/core1/naive. Pass if
  `compute_loss <= 1,000` and the three dormant times are within the
  expected spread; then record the decision below.

**Decision gate for Step 1** (write the outcome into this file before
starting Step 1):

| Observation | Meaning | Go to |
|---|---|---|
| `layer1.forward` contested count (SRAM0-3) is higher in naive (slow) than in core0 (fast) by about the same order as the extra 1024 cycles | Bank contention confirmed; the counters also say *which* bank, i.e. which stream collides with fetch | Step 1b (scratch banks), skip 1a |
| Contested counts equal, cycles differ | Not bus arbitration. Likely fetch alignment / prefetch / branch behaviour | Step 1a only; revise 1.6 |
| Counts differ but not in the code bank (e.g. data vs data) | Data layout collision, not code | Step 1b is the wrong fix; move or pad the hot arrays instead; revise Step 1 |
| `train.epoch_minus_eval_per_sample` differs core0 vs core1 by about 2.2k while the per-layer stages do not | The slowdown is in code only the production path has (inlined backward in `Train`, RMSProp lambda) | Look at the `kind=loop` rows for `Train`; apply 1a to `Train` itself; the kernels to move in 1b include the inlined loops, so Steps 2-4 should land first or the backward must be split into out-of-line kernels |
| Counts are all near zero everywhere | The counters are not seeing the CPU traffic (wrong events/passes) or the hypothesis is false | Re-check event selection against the datasheet before proceeding |

**Step 0.5 results and decision (uncommitted working tree on top of `070f5aa`;
profile builds, core 1 for the core1 build, two runs each, cycle counts identical
between runs, counter passes within 0.1% of the counter-free reference):**

| | core0 | core1 | naive |
|---|---|---|---|
| per-layer fwd+loss (cycles/sample) | 41,018 | 41,156 | 41,018 |
| per-layer full pass | 152,826 | 154,592 | 152,827 |
| `train.eval_pass` / sample (production) | 42,517 | 40,570 | 42,534 |
| `train.epoch_minus_eval_per_sample` (production) | 159,465 | 157,512 | 159,479 |
| harness estimate (full pass + per-batch) | 157,228 | 158,994 | 157,229 |
| **unexplained (production - harness)** | **+2,237** | 0 | **+2,250** |
| production contested accesses / epoch | 49.2M (banks 0-3) | 34.2M (banks 4-7) | 118.2M (banks 0-3) |
| `layer1.forward` per-layer cycles | 12,992 | 13,016 | 12,992 |
| `layer1.forward` contested accesses | 1,064 | 0 | 1,064 |
| `compute_loss` per-layer cycles | 1,137 | 1,137 | 1,137 |

- Row 4 of the gate matches: production is about 2.2k/sample slower than the
  harness on core0 and naive, but not on core 1, and the per-layer stages do not
  show it. The excess is in the **eval/forward half** (`train.eval_pass` is
  1.9k/sample slower on core 0), i.e. code or data only the production path has
  (the `State` ping-pong buffers, `GetOutput`, the argmax loop), not the layer
  kernels. Contested counts are consistent with it (core 0: 6.4% of cycles,
  core 1: 4.5%, naive: 15%) but the per-layer harness cannot localise it.
- Row 2 matches for the `layer1.forward` +1,024 (now present in **all three**
  builds, 11,960 -> 12,992/13,016): core1 has **zero** contested accesses there
  yet the same slowdown, so that penalty is not bus arbitration. Treat it as
  fetch/alignment: **go to Step 1a first**. Step 1b (scratch banks) is not
  supported by this evidence for the layer kernels; reconsider after 1a and
  after localising the production-path excess (apply 1a to `Train`, read the
  `kind=loop` rows of `MLPOpticalRecognition::Train` in `placement_results.csv`).
- The probe core's own `compute_loss` figure moved between 845 and 1,137 cycles
  when unrelated code was added ahead of it (no source change to the loss), and
  an isolated call measures 752 cycles. So the 2d exit criterion (<= 1,000) is
  met by the kernel (752) but not by the in-harness figure of the current
  layout (1,137); that residual is the same placement sensitivity as above and
  is expected to be resolved by Step 1a.
- Not done: the full benchmark acceptance run and the golden re-baseline of
  loss/accuracy for the new `expf`/`logf` (needs a clean tree and ~15 min per
  config).

---

### Step 1 - Test the bank-collision hypothesis and, if true, fix it

**Prerequisite:** Step 0.5 and its decision gate. Which of 1a/1b to run is
decided there, not here.

**1. Rationale.** See 1.6 and 1.8. If it holds, a large part of the observed
cycles/instruction (especially in backward) and all of the build-to-build
variance is avoidable, and Steps 2-4 become measurable. If it does not hold,
we lose about half a day and learn that alignment/placement noise has another
cause.

**2. Implementation.**

*1a. Alignment control (cheap).* Compile the MLP translation units with
`-falign-functions=16 -falign-loops=16` (use `set_source_files_properties`
for `experiments/MLPOpticalRecognition.cpp`, the profile TU and the unit test
that instantiates `StaticMLP`; do not touch SDK sources). Run the profile in
core0, core1 and naive. Interpret:
- `layer1.forward` equal across all three builds (about 11.85k): alignment was
  the cause. Keep the flags permanently and stop.
- Naive still about 1024 cycles slower, or core0/core1 differ for another
  reason: bank sharing is the cause. Go to 1b.

*1b. Move hot kernels to the unstriped scratch banks.*
- Add `MEML_MLP_HOT_CODE` in `MemoryDefs.hpp`, next to `MEML_MLP_CODE`:
  - core0-pinned: `__attribute__((section(".scratch_x.meml_hot")))` (SRAM8);
  - core1-pinned: `.scratch_y.meml_hot` (SRAM9);
  - unpinned: blank (the naive counterfactual must stay unplaced).
- Bind a generic hook `SMLP_HOT_CODE_ATTR` (blank by default in
  `mlp/Placement.h`, like `SMLP_CODE_ATTR`) and apply it **only** to the
  innermost kernels after Steps 2-4: forward per layer shape and
  `AccumulateGradients<NeedDelta>`. Budget: about 2 KB of code, because each
  4 KB bank is shared with a core's default stack. Check the link map for what
  else the SDK puts in `.scratch_x/.scratch_y` and which core's stack sits
  there.
- Choose the bank the **other** core never touches. In the core1-pinned
  build core 0's SDK stack sits in the bank core 0 uses; in the core0-pinned
  build core 1's stack is in `.core1_stack` inside CORE1_RAM, so confirm from
  the map and pick accordingly. Do this before writing code.
- Per-instantiation templates cannot carry an explicit `section` attribute
  (see the `for_each_layer` comment in `StaticMLP.h`). Either give the hot
  kernels non-template, per-shape names, or follow the `AGENTS.md` untagged
  rule: linker redirect in `section_copy_to_ram_text.incl` plus a validator
  check.
- Pinned builds copy `.scratch_*` from flash at boot via the SDK runtime;
  confirm that this happens for `.scratch_y` before core 1 starts the MLP.

**3. Expected saving.** Unknown until measured. Upper bound on forward from
the layer1 example: 1024 of 11,855 cycles (8.6%); backward may have more
hidden contention. Plan for **0-8% of the pass**, but the more important
output is that identical code gives identical timing in all three builds,
removing the 4% noise from every later comparison.

**4. Independence impact and mitigation.** SRAM8/9 are outside both 256 KB
windows, so `RAM0_BASE/END`, `RAM1_BASE/END` in `validate_memory_placement.py`
do not cover them. Required changes:
- add explicit windows `SCRATCH_X = [0x20080000, 0x20081000)` and
  `SCRATCH_Y = [0x20081000, 0x20082000)` and a rule that in a core-N build all
  tagged hot-kernel symbols are in that core's scratch bank (and **not** in the
  other's);
- a check that the other core's own code/stack/flooder objects are not in the
  chosen bank (otherwise independence regresses);
- update the callee-closure check from Step 0 to accept the scratch bank as
  "inside the pinned window" for that core only;
- update `AGENTS.md` memory constraint text to the new three-region model.
Mitigation: the benchmark's flooding vs dormant comparison (`timing_verdict`
within 1%) remains the end-to-end proof; it must still PASS.

**5. Further testing.** Run the full `RunAndSaveBenchmarks.py` for
core0/core1/naive after 1a and after 1b, compare `layer*.forward`/backward
profile rows across builds (the pass criterion is cross-build equality, not a
specific number), and record the addresses/phases from Step 0. Also test with
code in the scratch bank but data deliberately moved by +4/+8/+12 bytes to
confirm timing no longer depends on phase.

---

### Step 2 - Skip the layer-0 input-gradient

**1. Rationale.** 1.3: layer-0 `delta_out` is computed (4096 MACs, 4 of the
10 loop instructions) and discarded. It is the largest single, risk-free
item.

**2. Implementation.**
- In `StaticLayer.h`, make the method
  `template<bool NeedInputGrad = true> AccumulateGradients(...)`. When
  `false`: no zeroing loop for `delta_out`, no `delta_out[j] += es*w[j]`
  update, and `delta_out` is never dereferenced (may be `nullptr`). The `g`
  and bias accumulation are unchanged.
- In `StaticMLP.h`, `backprop_accumulate<I>` calls
  `layer.template AccumulateGradients<(I > 0)>(...)`. This covers all callers
  at lines 310, 412 and the RL `AccumulatePolicyGradient` at 484, none of which
  needs the network-input gradient. Leave `CalcGradients()` (autograd)
  untouched: it does return the network-input gradient.
- Optionally the same for `UpdateImmediate` at layer 0 via
  `backprop_immediate<0>`, which has the same dead `delta_out`.
- Update `tests/ProfileForwardBackward.cpp` to call
  `layer0.AccumulateGradients<false>(...)` (it currently passes a
  `delta_layer0` buffer).

**3. Expected saving.** About 8 cycles/iteration x 4096 = **~33k cycles per
sample (about 21% of the pass)**, about 8 s of the 42.5 s. Weights, biases and
training trajectory are **bit-identical** (only discarded values are skipped).
The `layer0.AccumulateGradients` profile row should fall from 74k to
roughly 40-45k.

**4. Independence impact and mitigation.** None negative. New instantiation
`AccumulateGradients<false>` is an `smlp::` member tagged `SMLP_CODE_ATTR`, so
`validate_memory_placement.py`'s prefix check covers it; still build core0 and
core1 and `nm` it, because template instantiations have surprised us before.
It also removes one `memset`-style zeroing call (relevant to 1.7).

**5. Further testing.** Golden loss/accuracy identical to the pre-change
build for the fixed seed; unit test that `AccumulateGradients<false>` leaves
`g` and the bias accumulators exactly equal to `<true>`; the profile row above.

---

### Step 3 - Row-blocked forward pass

**1. Rationale.** 1.3: the forward kernel reloads the input vector for every
output row and is bound by instruction/load count, not FMA latency. Computing
4 output rows per sweep over the input loads each input element once and gives
4 independent accumulator chains.

**2. Implementation.** In `StaticLayer::forward` (float branch only; keep the
fixed-point branch):
- Process rows in blocks of `R = 4`: pointers `w0..w3 = w + r*NIn`,
  accumulators `a0..a3` initialised to the **biases** of the 4 rows; for each
  `j`: `x = input[j]; a0 = fmaf(w0[j], x, a0); ... a3 = ...`. Unroll `j` by 2
  (`#pragma GCC unroll 2`) for loop overhead; register budget is about 14 S
  registers. Per `j`: 5 loads + 4 `vfma` versus about 22 instructions per 4
  MACs today.
- Handle `NOut % 4` with `if constexpr` tails (`R = 2`, then `R = 1`); layer 2
  has 10 rows (2 blocks + 2).
- **Remove the `fp-contract=off` pragma** around `forward` (the unfused
  VMUL+VADD was for the 4-way accumulator split, which this step replaces).
  Each row keeps a single sequential chain from the bias with fused FMA, so
  it is bit-identical to the pre-Opt-3 implementation (`17bc32a`). It is not
  bit-identical to HEAD's 4-accumulator unfused loop, so **re-baseline the
  golden once**.
- Replace the `m_cached_input[j] = input[j]` loop's `memcpy` libcall with an
  inline loop (`__attribute__((optimize("no-tree-loop-distribute-patterns")))`
  on `forward`, or an explicit copy loop). Verify with `objdump` that no `bl`
  to `memcpy` remains.
- Confirm in the disassembly that GCC emits post-incremented `vldm`/
  `vldr` with offsets and not extra address arithmetic. If not, fall back to
  `R = 8` or hand-scheduled intrinsics. Do not accept the step on source
  appearance alone.

**3. Expected saving.** Forward from 38.7k to about **17-20k cycles**,
saving about 19-21k per sample (about 13% of the pass). Layer-0 forward
(25.2k) is the main target.

**4. Independence impact and mitigation.** Removes the `memcpy` call from the
core-1 hot path (1.7). Must remain a plain `SMLP_CODE_ATTR` member (no lambdas
or untagged helper templates; if a helper is unavoidable, follow the
`AGENTS.md` untagged-code rule: linker redirect plus validator check).
Larger unrolled bodies increase code size slightly; check it still fits the
core's window and, if Step 1b is in use, the 4 KB scratch budget.

**5. Further testing.** Extend `StaticMLPForwardAccumulatorTest` shapes to
`NOut` in {1, 2, 3, 4, 5, 10, 32, 64} x `NIn` in {7, 10, 64, 65}; re-baseline
golden and check loss/accuracy equal the `17bc32a`-era golden; profile all
three forward rows on core0, core1 and naive; run the Step-0 callee check.

---

### Step 4 - Row-blocked backward pass

**1. Rationale.** 1.3: backward is 75% of the pass and pays about 9 cycles/MAC
because `g[j]` and `delta_out[j]` are loaded and stored once per (i, j).
Blocking rows keeps unit stride (unlike Opt 4) and reuses loads.

**2. Implementation.** `AccumulateGradients<NeedInputGrad>`:
- Compute `es0..es3` for a block of 4 rows into registers (not an array, to
  avoid stack temporaries) and add them to `m_bias_grad_accum`.
- For each `j`, with `x = input[j]`: `g0[j] += x*es0; ... g3[j] += x*es3`
  (4 loads, 4 FMA, 4 stores, independent).
- If `NeedInputGrad`: `delta[j]` is updated by the 4 rows in the **same
  sequential order as today**:
  `d = fma(es3,w3[j], fma(es2,w2[j], fma(es1,w1[j], fma(es0,w0[j], d))))`,
  one load and one store of `delta[j]` per 4 rows instead of per row. For
  the first block `d` starts at 0 and no zeroing pass is needed. This keeps
  the result **bit-identical to HEAD**. The 4-deep chain per `j` needs `j`
  unrolled by at least 2 to hide FMA latency; check the schedule. If the
  profile shows latency stalls, a tree sum is the fallback (then re-baseline
  golden).
- Instruction count per 4 rows x 1 `j`: with `NeedInputGrad` about 23
  (versus 40 now); without (layer 0 after Step 2) about 13 (versus about
  24).
- Tails for `NOut % 4` as in Step 3.
- `UpdateImmediate`/`CalcGradients` get the same treatment later if wanted
  (RL paths, not on the benchmark path).

**3. Expected saving.** Backward after Step 2 is about 84k; blocked it is
roughly **48-52k**, saving about 32-36k per sample. Estimated as layer 0
(42k -> ~23-25k), layer 1 (37k -> ~20k), layer 2 (6k -> ~3.5k). The largest
uncertainty in the plan: M33 store/load pipelining behaviour. Measure Step 2
output first and re-estimate.

**4. Independence impact and mitigation.** Removes the delta-zeroing `memset`
calls (1.7). Same rules as Step 3 for tagging; no stack arrays so no extra
stack pressure on either core. Backward code is currently inlined into
`MLPOpticalRecognition::Train` (3.5 KB); blocked kernels may become
out-of-line symbols. The existing inlining canary
(`check_mlp_inlining_canary`) must still pass, and any new out-of-line
symbol must be `smlp::`-prefixed and tagged.

**5. Further testing.** Reference test comparing blocked vs scalar
`AccumulateGradients` for `g`, bias and `delta_out` across shapes (exact
equality while sequential order is kept); golden loss/accuracy identical to
Step 3's; the three backward profile rows on all builds; and the full
benchmark, where flooding vs dormant must still be within the 1% tolerance
on pinned builds.

---

### Step 5 (optional) - Micro-batch 4 samples through forward/backward

**1. Rationale.** After Step 4 the remaining backward cost is dominated by
loading/storing `g` once per sample. Processing 4 samples per weight pass
loads each weight (forward) and each `g` cell (backward) once per 4 samples.
Decide only after re-profiling Step 4: expect 8-12% of the original pass, low
confidence.

**2. Implementation.** In `TrainBatch`, process samples in groups of 4:
`forward` computes 4 samples' accumulators from one weight load (4
accumulators = 4 samples, which also preserves per-sample sequential sum
order); `AccumulateGradients` does `g[i][j] += sum_b x_b[j]*es_b[i]`.
Needs per-sample caches (`m_cached_input`, `m_act_output`,
`m_inner_products` or precomputed `es`) x4, about 6 KB extra per core, inside
the single `State` object so the existing `MEML_MLP_DATA` annotation covers
it. Batch size 128 is divisible by 4; the last batch (3823 mod 128 = 79) is
not, so keep the existing single-sample path for the tail.

**3. Expected saving.** About 10% of the original pass (about 15k cycles per
sample), uncertain.

**4. Independence impact and mitigation.** More core-local state, still in one
tagged object; add a linker/`static_assert` capacity check against the bank
size. Both paths (blocked and tail) must be in-bank. Numerics: the `g`
summation order changes (not bit-exact), so loss/accuracy trajectories shift
slightly.

**5. Further testing.** Convergence regression: final loss and accuracy
after a full session must stay within a tolerance (suggest +/-0.5% accuracy
and +/-2% loss over several seeds) of the previous golden. Unit tests for the
tail path (batch sizes 1, 3, 4, 5, 79, 128).

---

### Step 6 (optional) - RMSProp and loop-level cleanup

**1. Rationale.** After Steps 2-4 the non-fwd/bwd 6% becomes roughly 12% of
the run (about 2.5 s of about 20 s).

**2. Implementation.** In `ApplyAccumulatedGradients`:
- replace `lr / (sqrt(v) + eps)` with a fast reciprocal-sqrt (integer-seed
  plus one or two Newton iterations), removing one `vsqrt` and one `vdiv` per
  weight (about 28 cycles); keep the `maxLR` clamp;
- fuse the clip, EMA, update and accumulator-clear into a single pass with
  fewer `vmrs` flag transfers (use `vmax`/`vmin` style selects).
Leave the final evaluation pass alone: it provides `Result` loss/accuracy and
the benchmark's correctness check.

**3. Expected saving.** About 1-1.5 s per session (about 5% of the faster
run), if the per-stage rows added in Step 0 confirm RMSProp is about 2-4% of
the original run.

**4. Independence impact and mitigation.** Pure arithmetic inside already
tagged functions; no new callees if the fast rsqrt is inline (check the
callee-closure validator).

**5. Further testing.** Compare weights after a batch against the exact
formula (relative error bound), plus the convergence regression from Step 5.

---

## 4. Process for every step

1. Build all three configs exactly as `RunAndSaveBenchmarks.py` does
   (`build-core0`, `build-core1`, `build-naive`); the POST_BUILD validator must
   pass, including the Step-0 callee check.
2. `arm-none-eabi-nm` spot-check any new symbol address is in the right window.
3. Run the profile build for all three (from Step 0.5 on: per-layer stages,
   the production-path probe `train.*`, and the contention counters); compare
   per-stage cycles and the placement columns before benchmarking. A step is
   not accepted on per-layer rows alone: the production-path rows must move
   the same way.
4. Run the unit suite (`RUN_TESTS_OR_BENCHMARKS=tests`).
5. Only then run the full benchmark and commit the CSV rows with the commit
   hash (the script requires a clean tree).
6. Keep each step in its own commit so a regression can be reverted without
   losing the others; never merge two steps before both are measured.

## 5. Open questions to resolve early

- Which core's stack and which SDK objects live in SRAM8/SRAM9 in each build
  (decides the bank choice in Step 1b).
- Whether `-ffp-contract=fast` is really the GCC default under `-std=gnu++20`
  here, i.e. that pre-Opt-3 builds used fused FMA (the Step 3 bit-exactness
  claim depends on this; the `vfma` in the `build-core0` disassembly says yes).
- Real M33 load/store costs for single-register `vldm/vstm`; settle by
  measuring Step 2, not by assumption.
- Whether an out-of-line blocked kernel loses more to call overhead than it
  gains (call per layer per sample is cheap; verify).
