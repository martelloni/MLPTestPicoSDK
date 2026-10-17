# Better StaticMLP optimisation plan

Written for: the maintainer and whoever implements the steps. Follow
`AGENTS.md` for every step that adds or changes MLP code (core0 and core1
builds, `validate_memory_placement.py`, callee-closure check, `nm`
spot-checks).

**Baseline.** This revision treats the repository at commit `f1b9488` (and the
tooling it contains) as the starting point: the callee-closure validator, the
private `expf`/`logf` (`mlp/FastMath.h`), the production-path `train.*` probe,
the bus-contention counters and the extended placement CSV already exist. The
earlier "Step 0" (guard rails) and "Step 0.5" (production probe and
contention counters) are done and have been removed from this document. What
they revealed is written up in section 1, and the new Step 0 fixes the
inconsistencies they left behind.

The plan supersedes `plan-mlpOptimisationPrompt.md` (Optimisations 1-5). It is
based on the benchmark CSVs, the source and the disassembly of the existing
ELFs. **No expected saving below has been measured**: each is an estimate from
instruction counts or from the contention counters, and each step says how it
will be confirmed.

---

## 1. Findings

### 1.1 The problem and the baseline

| Build | Benchmark time (10 epochs, one session, dormant) |
|---|---|
| Legacy heap MLP (`0732282`, core0-baseline) | 45.64 s |
| First StaticMLP (`0e9e71b`) | 45.50 s |
| "Peak" StaticMLP (`8fb216e`) | 42.52 s |
| **Current baseline (`f1b9488`)**: core0 / core1 / naive | **41.72 s / 41.18 s / 41.73 s** |

Removing the heap gained nothing; the five earlier optimisations gained about
6.5% in total. The legacy engine has exactly the same loop nest as StaticMLP
(`Layer.h`), and the heap is touched only at construction, so removing it could
not make the hot loops faster. StaticMLP's real value is placement and
independence, which the benchmarks do prove (pinned builds: flooding vs
dormant within 0.00%).

### 1.2 Where the time goes (baseline `f1b9488`)

One session is 10 epochs x 3823 samples = 38,230 samples, about 6.2 G cycles
at 150 MHz (1 s of session time = about 3.9k cycles per sample). Per-sample
cycles from the per-layer profile (core0 build):

| Stage | Cycles/sample | Share of pass |
|---|---|---|
| layer0.forward (64->64) | 25.3k | 17% |
| layer1.forward (64->32) | 13.0k | 8% |
| layer2.forward (32->10) | 1.6k | 1% |
| loss (softmax-CE) | 1.1k | 0.7% |
| layer2.AccumulateGradients | 6.0k | 4% |
| layer1.AccumulateGradients | 35.4k | 23% |
| **layer0.AccumulateGradients** | **70.4k** | **46%** |
| **Full pass** | **152.8k** | 100% |
| per-batch work (RMSProp apply, grad-norm, zeroing) | 4.4k | 2.9% of a pass |

- Forward + loss + backward are about 94% of the session; the final
  evaluation pass in `Train()` is about 2.5%; everything else is the per-batch
  work. **The training loop itself is not the bottleneck**: its overhead is
  about 3% and most of it is RMSProp (482k cycles per 128-sample batch, 74
  cycles per weight).
- Backward is **75%** of the pass; layer-0 backward alone is **46%**.
- The production-path probe agrees with the benchmark exactly:
  `10 x (epoch - eval) + eval` = 41.73 s (core0) and 41.18 s (core1), against
  41.72 s and 41.18 s measured. So the `train.*` rows are a faithful stand-in
  for the benchmark; the per-layer rows are not (see 1.6).

Algorithmic floor for context: 6464 weights give 6464 forward MACs plus
2 x 6464 backward MACs, about 19k MACs per sample. 152.8k cycles is about
**8 cycles per MAC**; a scalar Cortex-M33 should reach roughly 2-3.

### 1.3 What the disassembly shows (`StaticLayer.h`)

- **Forward inner loop**: per 4 MACs, 8 `vldr` + 4 `vmul` + 4 `vadd` + 4
  integer instructions, about 22 instructions. Measured 24.6 cycles per
  iteration, so it is **bound by instruction and load count, at about one
  instruction per cycle**. The same input vector is reloaded for every output
  row, so half of all loads are redundant. The 4-accumulator unroll (old Opt 3)
  did not help because the loop was never limited by FMA latency
  (`17bc32a` -> `665faab`: 42.93 -> 42.83 s).
- **Backward inner loop**: per `j`, 10 instructions for 2 MACs (4 loads, 2
  stores, 2 `vfma`, compare, branch). Measured about 17-18 cycles per
  iteration, roughly 9 cycles per MAC. `g[j]` and `delta_out[j]` make a
  load-modify-store round trip on every (i,j). Loads and stores are the cost,
  not the FP arithmetic. (The M33 reference-manual timings have not been
  checked; the measurement is consistent with about 2 cycles per memory
  operation.)
- **Dead work**: `layer0.AccumulateGradients` computes `delta_out` (4096 MACs)
  that nothing consumes, since layer 0 has no earlier layer
  (`backprop_accumulate<0>`). That is about half of the 46% layer-0 backward
  cost.
- **RMSProp**: about 25 instructions per weight including one `vsqrt` and one
  `vdiv`, 74 cycles per weight measured.

### 1.4 Why the earlier optimisations did not help

| Optimisation | Targeted | Share of pass | Result |
|---|---|---|---|
| 1 branchless ReLU | ~96 activations/sample | <0.3% | 45.50 -> 45.29 s |
| 2 `__restrict` | enabler | n/a | -> 44.68 s |
| 3 4-acc forward | forward (wrong premise: latency) | 25% | ~0 (42.93 -> 42.83 s) |
| fp-contract=off | forward | 25% | -3k forward cycles in profile; ~-0.3 s benchmark |
| 4 backward j-outer interchange | backward | 75% | **regressed** (likely strided scalar `w[i*NIn+j]` loads and a two-pass split; build not disassembled) |
| 5 skip layer-0 cached-input copy | 256 B memcpy | 0.04% | noise / reverted |

They went after the 25% forward half and tiny items; the one change aimed at
the 75% backward half regressed. Nothing touched the layer-0 dead work or the
backward load/store traffic. The assumption that strided access is free
because the RP2350 has no data cache does not hold in practice.

### 1.5 Fetch-versus-data bank contention is real, measurable and costs about one cycle per event

**The mechanism, in plain terms.** RP2350 main SRAM is two groups of four
banks (group 0 = banks 0-3 at `0x20000000-0x2003FFFF`, group 1 = banks 4-7 at
`0x20040000-0x2007FFFF`). Inside a group, consecutive 32-bit words go to
consecutive banks, so a sequential array is spread across all four. Each bank
can serve one access per cycle. A core fetches instructions through one port
and loads/stores data through another. If, in the same cycle, both ports want
the *same bank*, one waits a cycle. The bus fabric counts these as
**contested accesses** (`SRAMn_ACCESS_CONTESTED`). With `PICO_COPY_TO_RAM=1`
all hot code lives in SRAM, so a kernel whose code and data are in the same
group can collide. In a loop the access pattern repeats every iteration, so a
collision that happens once happens on every iteration: the cost is
deterministic and depends on the relative address phase (mod 16 bytes) of the
code stream and each data stream. That is why unrelated edits, which shift
addresses, changed timing by "random" multiples of the iteration count.

**What the counters now show** (`contention_results.csv`, `f1b9488`; "per
call" means per stage execution):

| Observation | Value |
|---|---|
| `layer1.forward` contested accesses per call | core0 1,064 (banks 0-3); core1 **1,089 (banks 4-7)**; naive 1,064 |
| `layer1.forward` extra cycles vs the contention-free value | +1,024 in all three builds (12,992 / 13,016 vs 11,960) |
| backward inner loops | about **one contested access per inner iteration**, about 6% of each backward layer |
| `layer0.forward` | 32 contested: clean, 25.3k cycles in every build |
| production epoch, core0 vs core1 | contested **49.19M vs 34.24M** (difference 14.96M); epoch cycles 772.17M vs 757.27M (difference **14.91M**) |

The last row is the key one: in the two builds where nothing else shares the
probe's bank group, the cycle difference equals the contested-count difference
to 0.4%. **One contested access costs about one cycle.** Contested accesses
are 6.4% of core0's epoch cycles and 4.5% of core1's, which is the ceiling for
what removing contention can save (about 3-6% of the session).

**What did not hold up.** The earlier idea that the loop-head phase mod 16
alone decides the outcome is false: loop heads at every phase (0, 4, 8, 12) are
slow in the current builds. The earlier "fast" cases were fast because the
harness kept its input/output buffers on the stack in SRAM9, or because the
"core1" profile ran on core 0 with code and data in different groups. The
contention depends on the *relative* phases of code and all the data streams,
and on whether they share a group.

**Where the contested events can come from.** With the other core idle, the
only masters are this core's fetch port, data port and system port, so the
events are fetch-vs-data. That is why moving the code out of the data group
(scratch banks, Step 1b) is a valid cure, and why aligning the data streams
(Step 1a) can be one too.

### 1.6 Measurement inconsistencies at the baseline

These are the defects Step 0 fixes. Each is a reason a number in the current
CSVs cannot be taken at face value.

| # | Defect | Evidence | Consequence |
|---|---|---|---|
| D1 | **Two copies of each kernel.** `forward()` for each layer shape exists twice in every build (e.g. 212 B at `0x2000863c` and 220 B at `0x20009c08` for 64x32); the harness and `Train` call different copies. Backward is inlined into `Train` but called as separate functions by the harness. | `placement_results.csv` code rows | Per-layer rows describe code the benchmark does not run. Production vs harness differs by +2.2k cycles per sample on core0/naive and 0 on core1, and the harness cannot say why. |
| D2 | **The counter poller is not idle.** The "service core" spins reading counters: 940k SRAM accesses per 50 ms (`idle_window`), about 95M per epoch. In the naive build it shares the probe's bank group. | naive epoch: 118M contested vs core0 49M with identical cycles | Contested counts for naive are polluted and can't be compared. The "dormant" condition is violated during profiling. |
| D3 | **The contention CSV is not self-explanatory.** `access_avg` is filled only on SRAM0 and SRAM4 rows; the group that matters differs by build (0-3 for core0/naive, 4-7 for core1). | A summary that sums banks 0-3 for every build reported "0 contested on core1" (it was 1,089 in banks 4-7) | A wrong conclusion was drawn from the data. The tooling must do the per-group totalling, not a reader. |
| D4 | **Eval-only contention not recorded.** `train.eval_pass` is 1.95k cycles per sample slower on core0/naive than core1, and so is the training part; backward is identical. | `profiling_results.csv` | The gap is in the forward path, but contested counts are only for the whole epoch, so it can't be localised. |
| D5 | **`compute_loss` is 1,137 cycles in the harness, 752 isolated, 728 before the Cephes change.** Only 4 contested accesses, so it isn't SRAM contention. | `contention_results.csv`, notes | Its remaining cost, and its variation (845-1,137) with unrelated code changes, is unexplained. It reads labels from flash through the XIP cache, a candidate. |
| D6 | **No golden for the new `expf`/`logf`.** The loss values changed slightly when the Cephes versions replaced pico's. | n/a | Later "bit-identical" claims have no reference. |
| D7 | **Stack is in different places per build.** The profile's stack pointer is in SRAM9 for core0/naive (`0x20081f50`) but in the core 1 bank for core1. | `placement_results.csv` `stack_pointer` | Spilled locals contend differently per build; the builds are not apples-to-apples. |

### 1.7 Resolved at the baseline (kept for the record)

- The hot path of the pinned core1 build used to call `memcpy`, `memset`,
  `__wrap_expf`, `__wrap_logf` located in core 0's bank. Fixed:
  `-fno-tree-loop-distribute-patterns` on the MLP TUs, private
  `smlp::exp_f32`/`log_f32`, and `check_callee_closure` in the validator.
- `compute_loss` regression from the Cephes versions (728 -> 2,315 cycles) was
  reduced to 1,137 (still above the old 728, see D5).

---

## 2. Step overview and expected cumulative effect

Cycle counts are per sample; "Cum. pass" is the estimated full-pass time
(baseline 152.8k) after the step. All savings are estimates.

| # | Step | Est. saving | Cum. pass | Risk |
|---|---|---|---|---|
| 0 | Make the measurements trustworthy (fix D1-D7, freeze baseline B0) | 0 (the loss stage may gain up to ~0.4k) | ~152k | low |
| 1 | Deterministic layout and contention removal (1a now; 1b after Step 4) | 3-6% of the session from removing contested accesses | ~145-148k | medium |
| 2 | Skip layer-0 input-gradient | ~28-32k | ~120k | very low |
| 3 | Row-blocked forward | ~20-22k | ~100k | low |
| 4 | Row-blocked backward | ~28-32k | ~70k | medium |
| 5 | Micro-batch of 4 samples (optional) | ~8-12% of the original pass | ~55-60k | high |
| 6 | RMSProp / loop-level cleanup (optional) | ~1-1.5 s per session | n/a | low |

Steps 2-4 together: about **1.8-2.2x** end-to-end (about 19-23 s per session
against 41.7 s). **Order:** Step 0 first. Then Step 1a's *mechanism* (so that
layout stops being luck), then Steps 2-4, re-running the contention check
after each (any code change moves addresses). Step 1b needs compact
out-of-line kernels, so it comes after Step 4.

---

## 3. Steps

### Step 0 - Make the measurements trustworthy and freeze baseline B0

**1. Rationale.** The current tooling measures the right things but has seven
defects (1.6) that either mislead or hide the cause of the remaining
core0/core1 gap. Every later step is judged by numbers from this tooling, and
the gain from several steps is smaller than the placement effect, so the
tooling must first be made consistent, then frozen as the baseline B0 against
which all later steps are compared. In plain terms: before optimising, make
sure the stopwatch times the code that actually runs, and that the person
holding the stopwatch is not getting in the way.

**2. Implementation.** One sub-item per defect.

*0a. One copy of each kernel (D1).*
- Mark `StaticLayer::forward` and `StaticLayer::AccumulateGradients` (and the
  RMSProp apply function) `__attribute__((noinline, noclone))` (keep
  `SMLP_CODE_ATTR`), so each layer shape has exactly one function that both
  `Train` and the profile harness call. `noclone` prevents the `.constprop`
  specialisation that produced the second copy.
- `StaticMLP::forward_layer<I>`/`backprop_accumulate<I>` stay as they are;
  they now make three real calls per direction per sample (a call costs a few
  cycles against thousands).
- Check with `nm`: exactly one symbol per kernel and shape, no
  `[clone .constprop]` for them. Add that as a check in the validator
  (`check_mlp_inlining_canary` is the place; it currently expects the opposite
  for some symbols, so adjust it deliberately).
- **Acceptance:** per-layer sum (full pass + per-batch) equals
  `train.epoch_minus_eval_per_sample` within 0.5% on all three builds (the
  `train.unexplained_per_sample` row, currently +2,237 / 0 / +2,251, should be
  near zero). If it is not, there is still a code path the harness doesn't
  run, and that is the thing to find before moving on.

*0b. A poller that does not disturb the measurement (D2).*
- Replace the spinning poller by an event-driven one: the non-probe core arms
  a hardware timer alarm every ~10 ms, sleeps in `WFE`, wakes, reads the four
  24-bit counters, accumulates them into 64-bit totals, clears them, and
  sleeps again. 10 ms is well under the time a counter takes to saturate
  (16.7M events, > 100 ms at 150 MHz). The poller's own SRAM footprint then
  drops from about 95M accesses per epoch to a few tens of thousands.
- Keep all of it in the profile TU (`MEML_RUN_PROFILE` only).
- **Acceptance:** `idle_window` shows < 0.1% of the old figure on both bank
  groups, and the **naive** epoch contested count falls to within 5% of
  core0's (it is 2.4x now). Cycles must not change (they didn't with the
  spinning poller either).

*0c. Contention tooling that does the arithmetic (D3).*
- In `RunAndSaveBenchmarks.py`, derive and append rows with
  `sram_bank = "group"`: the sum of contested counts over the **four banks of
  the group the probe core's data lives in** (0-3 for core0/naive, 4-7 for
  core1), plus `other_group_total` for the other four banks. A reader should
  never have to sum banks.
- Add a derived column `contested_pct_of_cycles` for the `train.*` rows
  (group total / cycles).
- Write `benchmark_results/README.md` defining every CSV column, the bank
  groups, and what "contested" means (the plain-language text of 1.5).
  `access_avg` stays only on the two sanity banks and the README says so.
- Make the `run_self_test` assert the derived rows on a fixture (including a
  core1-style transcript where the counts are in banks 4-7).

*0d. Eval-only contention (D4).*
- In the production probe, bracket `Train(0)` (evaluation only) with the
  counters as well as `Train(1)`, and write `train.eval_pass` contested counts
  into the CSV, so the forward path can be compared between builds in
  isolation. Add a derived row `cycles_per_contested` = (cycle difference
  between two builds) / (contested difference) for each pair, printed in the
  log; B0 expects about 1.0.

*0e. Understand the loss stage (D5).*
- Run the loss stage in the harness three ways and record each: (i) as now,
  labels in flash; (ii) with the label vector copied to RAM first; (iii) with
  the XIP counters (`XIP_MAIN0_ACCESS`, `..._CONTESTED`, `..._STALL_UPSTREAM`)
  around the call. If (ii) is close to 752, the extra ~400 cycles are
  flash/XIP latency, which is a real production cost and not a layout artefact;
  record that and stop (it is 0.3% of the pass). If it is not, bisect: the
  `exp_f32`/`log_f32` domain checks in `FastMath.h` are the next suspect.
- Optionally apply the domain-restricted `exp`/`log` (after subtracting the
  max, `exp` only needs `x <= 0`; the log only needs a sum in `[1, classes]`)
  to bring the stage back to about 750. Worth about 0.1 s per session; do it
  only if 0e shows the arithmetic, not the flash read, is the cost.

*0f. Golden numerics (D6).*
- Record in `ram_independence_results.csv` (or a sibling) the final loss and
  accuracy of the session for the fixed seed, per build. They must be
  identical across core0, core1 and naive (same arithmetic), which is a free
  consistency check. The value becomes **golden G0**. Steps that claim to be
  bit-exact must reproduce G0 exactly; steps that change summation order
  re-baseline it once and say so.

*0g. Comparable stacks (D7).*
- Report the stack pointer of the probe in every build (already done) and
  make the profile builds put the probe stack in the same *kind* of place, or,
  if that is not feasible, add a column documenting where it is. Do not move
  the stacks of the real benchmark builds in this step (that would change what
  is being measured).

*0h. Freeze B0.*
- With 0a-0g in, build core0/core1/naive in profile mode, run each twice,
  and run the full benchmark once. Record in a new `baseline_B0` block in
  this file: per-layer cycles, `train.*` cycles per sample, group contested
  totals and percentages for eval and train, benchmark times, G0, the commit
  hash. Cycle counts must repeat within 0.1% and contested counts within 1%
  or the counters are not usable.

**3. Expected saving.** None from the arithmetic, except the loss stage may
regain up to ~0.4k cycles (0.3 s per session) if 0e shows it is arithmetic.
The step's product is a baseline and a toolset that can be trusted.

**4. Independence impact and mitigation.** Everything here is profile-only
code in `tests/ProfileForwardBackward.cpp` and the scripts; the real
benchmark and test builds change only through `noinline, noclone` on
`smlp::` members (still tagged, still covered by the prefix and
callee-closure checks). The one risk is 0a changing code size/placement of
the pinned builds: rebuild core0 and core1, confirm the validator passes, and
`nm` the new single kernel symbols are inside the pinned window. The poller
(0b) runs on the *non-probe* core by design, but because it sleeps in WFE the
dormant condition is now honoured, which also makes the profile numbers more
representative of the benchmark's dormant run.

**5. Further testing.** Beyond the acceptance criteria above: the parser
self-test (`run_self_test`) for new rows and columns; a check that the
`train.unexplained_per_sample` row is reported and its absolute value is
printed in the log; a repeat of the whole profile twice to demonstrate
repeatability; and a unit test that two `Train` runs from the same seed give
identical final loss/accuracy (this is what G0 relies on).

---

### Step 1 - Make the memory layout deterministic and remove fetch-versus-data collisions

**1. Rationale.** Section 1.5 shows that about 4.5% (core1) to 6.4% (core0) of
the epoch is lost to contested SRAM accesses, one cycle each, and that the
amount changes whenever any code or data moves. This step does two things:
(i) it makes the layout a *build decision* instead of a link accident, so
timing stops depending on unrelated edits and later steps can be compared
fairly, and (ii) it removes as many collisions as possible. In plain terms:
the code and the arrays it reads march through the same four memory banks in
step, and sometimes bump into each other. We make where they start a
deliberate choice, measure which choice bumps least with the counters, and
fix it in the source with checks so it stays that way.

**2. Implementation.**

*1a. Make the layout explicit (do now, after Step 0).*
- **Code phase.** Add `-falign-loops=16 -falign-functions=16` to the MLP
  translation units (`MEML_MLP_TUS` in `CMakeLists.txt`, the list that
  already carries `-fno-tree-loop-distribute-patterns`). Every inner-loop head
  then starts at phase 0 and does not drift with the code before it.
- **Data phase.** Give every array that a hot loop streams a fixed, known
  start phase mod 16 bytes: `alignas(16)` on `m_weights`, `m_grad_accum`,
  `m_sq_grad_avg` and the first array of each group in `StaticLayer`, and on
  `State` members and `StaticMLP` ping-pong buffers. Add `static_assert`s
  (using `offsetof` and the array sizes) that each streamed array's start is
  at the intended phase. Arrays whose size is not a multiple of 16 bytes
  (for example a 10-element output) must be followed by explicit padding so
  the next array's phase does not depend on it.
- **One tuning knob.** Add `MEML_MLP_STREAM_SKEW` (0, 4, 8 or 12 bytes), a
  padding member placed immediately before the activation streams
  (`m_cached_input`, `m_act_output`, `m_inner_products`, the buffers) so they
  move relative to the weight streams while everything else stays put. The
  skew is a compile-time constant; the build must fail on any other value.
- **Sweep.** Build the profile for each skew in {0, 4, 8, 12} on core0, core1
  and naive (12 builds; each takes minutes to run). For each, read the group
  contested totals per stage (Step 0c) and per `train.*` row (including
  `train.eval_pass`). Choose the skew with the lowest contested count over the
  stages that cost the most (layer 0 and layer 1 forward and backward). Record
  the full table in this file.
- If one skew is best for all three builds, bake it in as the default. If
  builds disagree (they might, since core1's addresses differ), the knob can
  stay a per-build constant selected in CMake from the sweep result, with a
  comment pointing at the table.
- **Any code change reshuffles addresses**, so the sweep (or at least a
  single-point confirmation at the chosen skew) is repeated at the end of
  Steps 2, 3 and 4. The acceptance criterion in each of those steps includes
  "group contested percentage no worse than B0".

*1b. Move the hot kernels out of the data banks (do only after Step 4, if the
residual justifies it).*
- **Condition to start:** after Steps 2-4 and 1a, group contested is still
  above about 2% of epoch cycles *and* the remaining collisions are
  fetch-vs-data (confirmed by the counters moving when code is moved in an
  experiment).
- **What:** place the few hot kernels in the unstriped 4 KB bank SRAM8
  (`.scratch_x`, `0x20080000`): one forward kernel per layer shape and the
  blocked `AccumulateGradients`. After Steps 3-4 these are small, out-of-line
  functions (budget about 2 KB). Code in an unstriped bank cannot collide with
  data spread over the four striped banks of a group.
- **Which bank.** The measured stack pointer for core 0 and the naive build
  is `0x20081f50`, which is in SRAM9 (`.scratch_y`). So **SRAM9 is not
  independent**: core 0's stack lives there. Use SRAM8 for the pinned core's
  hot code in both the core0 and the core1 build. Confirm from the link map
  that nothing else is in SRAM8 (the SDK's default core 1 stack is there, but
  this project already moves core 1's stack to `.core1_stack`).
- **Hook.** Add `MEML_MLP_HOT_CODE` in `MemoryDefs.hpp` (core0 or core1:
  `section(".scratch_x.meml_hot")`; unpinned: blank so the naive counterfactual
  stays unplaced) and a blank-by-default `SMLP_HOT_CODE_ATTR` in
  `mlp/Placement.h`. Per-shape kernels must be separate non-template-named
  functions or follow the `AGENTS.md` untagged-code rule (linker redirect plus
  validator check).

**3. Expected saving.** Contested accesses are 6.4% of core0's and 4.5% of
core1's epoch cycles, so the ceiling is about 3-6% of the session (1.2-2.5 s
of 41.7 s), if every event were removed. Plan for roughly half to most of it
from 1a. The larger benefit is that identical code then gives identical
timing in all three builds, which removes the run-to-run layout noise from
every later comparison. A bonus: core0 and naive pay about 2.2k cycles per
sample more than core1 in the forward path today (D4); 1a is expected to close
that gap, and if it does not, that residual tells us something new.

**4. Independence impact and mitigation.**
- 1a changes only alignment and a few padding bytes inside the pinned bank.
  It does not move anything across banks. Check the existing bank-capacity
  `ASSERT`s still pass; padding is at most 12 bytes per array.
- 1b places code in SRAM8, outside both 256 KB windows. Required changes:
  add an explicit `SCRATCH_X = [0x20080000, 0x20081000)` window to
  `validate_memory_placement.py`; a rule that tagged hot kernels in a
  pinned build are in SRAM8 and in no other core-specific region; a check that
  the other core's objects and stack are not in SRAM8; allow SRAM8 in
  `check_callee_closure` as "inside the pinned window" for the pinned core only;
  update `AGENTS.md` to the three-region model. The benchmark's
  flooding-versus-dormant check (`timing_verdict`) remains the end-to-end proof
  and must still PASS.

**5. Further testing.**
- The sweep tables themselves (contested per stage per skew per build).
- After choosing the layout: run the profile twice, require unchanged cycle
  counts; run the benchmark; compare against B0. Success = group contested
  percentage down, session time down, and `train.unexplained_per_sample` and
  the core0-versus-core1 gap (`cycles_per_contested` about 1.0) consistent
  with the counters.
- A negative control: deliberately compile one build with the worst skew from
  the sweep and confirm the counters and cycles move the way the table
  predicts. If they do not, the model of 1.5 is wrong somewhere and the plan
  must be revisited before Step 1b is attempted.
- Unit tests for the `static_assert`ed offsets (a compile test is enough).

---

### Step 2 - Skip the layer-0 input-gradient

**1. Rationale.** In plain terms: when training, each layer works out two
things, how to adjust its own weights and what error to hand back to the layer
before it. Layer 0 is the first layer, so there is nobody to hand the error
to, and today it computes it anyway (4096 multiply-adds per sample, about
half of layer 0's backward cost) and throws it away. It is the largest
single item and has no downside.

**2. Implementation.**
- In `StaticLayer.h`, make the method
  `template<bool NeedInputGrad = true> AccumulateGradients(...)`. When `false`:
  no zeroing loop for `delta_out`, no `delta_out[j] += es*w[j]` update, and
  `delta_out` is never dereferenced (may be `nullptr`). The `g` and bias
  accumulation are unchanged. Keep the `noinline, noclone` attributes from
  Step 0a; this yields one extra instantiation (`<false>`) that only layer 0
  uses.
- In `StaticMLP.h`, `backprop_accumulate<I>` calls
  `layer.template AccumulateGradients<(I > 0)>(...)`. This covers
  `TrainBatch` (both overloads) and the RL `AccumulatePolicyGradient`, none of
  which needs the network-input gradient. Leave `CalcGradients()` (autograd)
  untouched: it does return the network-input gradient.
- Optionally the same for `UpdateImmediate` at layer 0 via
  `backprop_immediate<0>`, which has the same dead `delta_out`.
- Update `tests/ProfileForwardBackward.cpp` to call
  `layer0.AccumulateGradients<false>(...)`.

**3. Expected saving.** About 8 cycles per iteration x 4096 = about 30k
cycles per sample (about 20% of the pass; 28-32k with the current 70.4k
layer-0 figure). `layer0.AccumulateGradients` should fall from 70k to
roughly 38-42k. Weights, biases and the training trajectory are
**bit-identical** (only discarded values are skipped).

**4. Independence impact and mitigation.** None negative. `AccumulateGradients<false>`
is an `smlp::` member tagged `SMLP_CODE_ATTR`, so the prefix and
callee-closure checks cover it; build core0 and core1 and `nm` it anyway
(template instantiations have surprised us before). It removes the zeroing
loop for layer 0.

**5. Further testing.** Final loss/accuracy equal to G0 exactly; a unit test
that `AccumulateGradients<false>` leaves `g` and the bias accumulators
exactly equal to `<true>`; the profile row above; group contested percentage
no worse than B0 (re-run the Step 1 confirmation point); and
`train.unexplained_per_sample` still near zero.

---

### Step 3 - Row-blocked forward pass

**1. Rationale.** In plain terms: to compute one output of a layer the code
walks the whole input vector, and it does that again from scratch for every
output. Of the roughly 22 instructions per 4 multiply-adds, half are re-loading
the same inputs. Computing four outputs together loads each input once and
uses it four times, and the four running sums are independent of each other.
The forward kernel is limited by how many instructions and loads it issues,
not by arithmetic latency, so cutting instructions is what helps.

**2. Implementation.** In `StaticLayer::forward` (float branch only; keep the
fixed-point branch):
- Process rows in blocks of `R = 4`: pointers `w0..w3 = w + r*NIn`,
  accumulators `a0..a3` initialised to the **biases** of the 4 rows; for each
  `j`: `x = input[j]; a0 = fmaf(w0[j], x, a0); ... a3 = ...`. Unroll `j` by 2
  (`#pragma GCC unroll 2`); register budget is about 14 S registers. Per `j`:
  5 loads + 4 `vfma`, versus about 22 instructions per 4 MACs today.
- Handle `NOut % 4` with `if constexpr` tails (`R = 2`, then `R = 1`); layer 2
  has 10 rows (2 blocks + 2).
- **Remove the `fp-contract=off` pragma** around `forward` (it existed for the
  4-way accumulator split, which this step replaces). Each row keeps a single
  sequential chain from the bias with fused FMA, so it is bit-identical to the
  pre-Opt-3 implementation (`17bc32a`), but **not** to the current unfused
  4-accumulator loop, so re-baseline G0 once and record G1.
- The `m_cached_input` copy is already an inline loop (the MLP TUs are built
  with `-fno-tree-loop-distribute-patterns`); keep it that way and verify with
  `objdump` that no `bl memcpy` appears.
- Confirm in the disassembly that GCC emits post-incremented `vldm`/`vldr`
  with offsets and not extra address arithmetic. If not, fall back to `R = 8`
  or hand-scheduled intrinsics. Do not accept the step on source appearance
  alone.

**3. Expected saving.** Forward from about 39.9k to about 17-20k cycles,
saving 20-22k per sample (about 14% of the pass). Layer 0 (25.3k) is the main
target.

**4. Independence impact and mitigation.** Must remain a plain `SMLP_CODE_ATTR`
member (no lambdas or untagged helper templates; if a helper is unavoidable,
follow the `AGENTS.md` untagged-code rule). Unrolled bodies increase code size
slightly; check it fits the core's window and, if Step 1b is in use, the 2 KB
scratch budget. The streams' phases change (5 streams per `j` instead of 2):
the new kernel's contested count must be re-measured and the Step 1 skew
re-chosen if needed.

**5. Further testing.** Extend `StaticMLPForwardAccumulatorTest` shapes to
`NOut` in {1, 2, 3, 4, 5, 10, 32, 64} x `NIn` in {7, 10, 64, 65}; check the
final loss/accuracy equal the `17bc32a`-era golden (G1 = that value); profile
all three forward rows on core0, core1 and naive; group contested percentage
no worse than B0; callee-closure check passes.

---

### Step 4 - Row-blocked backward pass

**1. Rationale.** In plain terms: for every weight, backward reads the
running gradient, adds to it and writes it back, and does the same for the
error being passed to the previous layer: six memory operations for two
multiply-adds, which is why the loop takes about 17 cycles per iteration.
Handling four output rows together lets the "error to the previous layer" for
all four rows be added up in registers and stored once, and reuses the input
value. Memory operations drop from 6 per 2 MACs to roughly 3. It keeps
unit-stride row access, which is why it should not repeat the failure of the
old j-outer interchange (Opt 4).

**2. Implementation.** `AccumulateGradients<NeedInputGrad>`:
- Compute `es0..es3` for a block of 4 rows into registers (not an array, to
  avoid stack temporaries) and add them to `m_bias_grad_accum`.
- For each `j`, with `x = input[j]`: `g0[j] += x*es0; ... g3[j] += x*es3`
  (4 loads, 4 FMA, 4 stores, independent).
- If `NeedInputGrad`: update `delta[j]` with the 4 rows in the **same
  sequential order as today**:
  `d = fma(es3,w3[j], fma(es2,w2[j], fma(es1,w1[j], fma(es0,w0[j], d))))`,
  with one load and one store of `delta[j]` per 4 rows. For the first block
  `d` starts at 0 so no zeroing pass is needed. This keeps the result
  **bit-identical to Step 3's output**. The 4-deep chain per `j` needs `j`
  unrolled by at least 2 to hide FMA latency; check the schedule. If the
  profile shows latency stalls, a tree sum is the fallback (then re-baseline
  the golden).
- Instruction count per 4 rows x 1 `j`: with `NeedInputGrad` about 23 (versus
  40 now); without (layer 0 after Step 2) about 13 (versus about 24).
- Tails for `NOut % 4` as in Step 3.
- `UpdateImmediate`/`CalcGradients` get the same treatment later if wanted
  (RL paths, not on the benchmark path).

**3. Expected saving.** Backward after Step 2 is about 80k; blocked it is
roughly 48-52k, saving about 28-32k per sample (layer 0: ~40k -> ~23-25k,
layer 1: ~35k -> ~20k, layer 2: ~6k -> ~3.5k). This is the biggest uncertainty
in the plan: it depends on how the M33 pipelines loads and stores. Measure
Step 2's output first and re-estimate.

**4. Independence impact and mitigation.** No library calls, no stack arrays,
so no extra pressure on either core's stack. Backward code is currently
inlined into `MLPOpticalRecognition::Train`; after Step 0a the kernels are
out-of-line, `smlp::`-prefixed, tagged symbols. Re-run the callee-closure
check and the inlining canary. As in Step 3, re-measure contested counts and
re-confirm the Step 1 skew choice.

**5. Further testing.** Reference test comparing blocked vs scalar
`AccumulateGradients` for `g`, bias and `delta_out` across shapes (exact
equality while sequential order is kept); final loss/accuracy equal to G1
exactly; the three backward profile rows on all builds; the full benchmark,
where flooding vs dormant must stay within the 1% tolerance on the pinned
builds; group contested percentage no worse than B0.

---

### Step 5 (optional) - Micro-batch 4 samples through forward/backward

**1. Rationale.** After Step 4 the remaining backward cost is dominated by
loading and storing the gradient accumulator `g` once per sample. Processing 4
samples per pass over the weights loads each weight (forward) and each `g`
cell (backward) once per 4 samples instead of once per sample. Decide only
after re-profiling Step 4: expect 8-12% of the original pass, low confidence.

**2. Implementation.** In `TrainBatch`, process samples in groups of 4:
`forward` computes 4 samples' accumulators from one weight load (the four
accumulators are the four samples, which keeps each sample's sum in its
original order); `AccumulateGradients` does `g[i][j] += sum_b x_b[j]*es_b[i]`.
Needs per-sample caches (`m_cached_input`, `m_act_output`, `m_inner_products`
or precomputed `es`) x4, about 6 KB extra per core, inside the single `State`
object so the existing `MEML_MLP_DATA` annotation covers it. Batch size 128 is
divisible by 4; the last batch (3823 mod 128 = 79) is not, so keep the existing
single-sample path for the tail.

**3. Expected saving.** About 10% of the original pass (about 15k cycles per
sample), uncertain.

**4. Independence impact and mitigation.** More core-local state, still in one
tagged object; add a `static_assert`/linker capacity check against the bank
size. Both paths (blocked and tail) must be in-bank. Numerics: the `g`
summation order changes (not bit-exact), so loss/accuracy trajectories shift
slightly.

**5. Further testing.** Convergence regression: final loss and accuracy after
a full session must stay within a tolerance (suggest +/-0.5% accuracy and
+/-2% loss over several seeds) of the previous golden. Unit tests for the tail
path (batch sizes 1, 3, 4, 5, 79, 128). Re-measure contested counts.

---

### Step 6 (optional) - RMSProp and loop-level cleanup

**1. Rationale.** After Steps 2-4 the per-batch work (about 3% today, mostly
RMSProp at 74 cycles per weight) becomes roughly 10-12% of the run.

**2. Implementation.** In `ApplyAccumulatedGradients`:
- replace `lr / (sqrt(v) + eps)` with a fast reciprocal-sqrt (integer-seed
  plus one or two Newton iterations), removing one `vsqrt` and one `vdiv` per
  weight (about 28 cycles); keep the `maxLR` clamp;
- fuse the clip, EMA, update and accumulator-clear into one pass with fewer
  flag transfers (`vmrs`); use `vmax`/`vmin`-style selects.
Leave the final evaluation pass alone: it provides `Result` loss/accuracy and
the benchmark's correctness check.

**3. Expected saving.** About 1-1.5 s per session (about 5% of the faster
run). Confirm first with the `batch.rmsprop.*` rows (482k cycles per batch
today).

**4. Independence impact and mitigation.** Pure arithmetic inside already
tagged functions; no new callees if the fast rsqrt is inline (the
callee-closure check verifies this).

**5. Further testing.** Compare weights after a batch against the exact
formula (relative error bound), plus the convergence regression from Step 5.

---

## 4. Process for every step

1. Build all three configs exactly as `RunAndSaveBenchmarks.py` does
   (`build-core0`, `build-core1`, `build-naive`); the POST_BUILD validator
   must pass, including the callee-closure check.
2. `arm-none-eabi-nm` spot-check any new symbol address is in the right window.
3. Run the profile build for all three: per-layer stages, the production-path
   rows `train.*`, and the contention rows. **A step is accepted on the
   production-path rows, not on per-layer rows alone**: both must move the
   same way, and `train.unexplained_per_sample` must stay near zero.
4. Compare every number with B0 (Step 0h). A step that lowers cycles but raises
   the group contested percentage has probably just got lucky with the layout;
   confirm with the Step 1 sweep point before keeping it.
5. Run the unit suite (`RUN_TESTS_OR_BENCHMARKS=tests`).
6. Only then run the full benchmark and commit the CSV rows with the commit
   hash (the script requires a clean tree).
7. Keep each step in its own commit so a regression can be reverted without
   losing the others; never merge two steps before both are measured.

## 5. Open questions to resolve early

- Does the production `Train` really call the same kernel after Step 0a, or is
  something else (the facade's `GetOutput` path, the RMSProp lambdas) still a
  separate copy? The `train.unexplained_per_sample` acceptance in 0a answers
  this.
- Is a contested access exactly one lost cycle everywhere, or can it cost more
  in some loops (for example when a store is also pending)? Step 0d's
  `cycles_per_contested` row will show whether 1.0 holds beyond the two clean
  builds.
- Which objects other than core 0's stack live in SRAM8 and SRAM9 in each
  build (decides whether Step 1b is feasible). Read the link map.
- Whether `-ffp-contract=fast` is really the GCC default under
  `-std=gnu++20` here, i.e. that pre-Opt-3 builds used fused FMA (the Step 3
  bit-exactness claim depends on it; the `vfma` in the `build-core0`
  disassembly says yes).
- Real M33 load/store costs for single-register `vldm/vstm`; settle by
  measuring Step 2, not by assumption.
- Whether an out-of-line blocked kernel loses more to call overhead than it
  gains (three calls per direction per sample is cheap; verify).
- Whether the 24-bit counters are wide enough for the per-call brackets of the
  slowest stages (they are for the per-layer rows; the epoch totals rely on the
  poller of Step 0b).
