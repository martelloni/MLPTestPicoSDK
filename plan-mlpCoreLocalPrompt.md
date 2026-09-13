# Core-local optical-recognition MLP

## Steps

### Step 0 — Correct the notebook reference

- Fix `dataset/preprocess.ipynb` to use `64 → 64 → 32 → 10`: `Linear(64,64)`, ReLU, `Linear(64,32)`, ReLU, `Linear(32,10)`.
- Remove the final `Softmax`; PyTorch `CrossEntropyLoss` consumes raw logits and internally applies log-softmax.
- Keep normalized `[0,1]` features, integer class targets for PyTorch, shuffled batches of 128, SGD `lr=0.01`, and add a fixed Torch/DataLoader seed for reproducibility.
- Retain one-hot labels in generated C++ data, since categorical cross-entropy in the static MLP consumes them.
- Regenerate `Dataset.hpp`/`Dataset.cpp`, retaining the dataset in flash rather than copying its ~1.1 MB payload into SRAM.

#### Testing and validation

- Manually rerun the notebook after every notebook/data-generation change; confirm its architecture has no final Softmax, generated dimensions remain `3823 × 64` and `3823 × 10`, and its seeded training run produces a recorded loss/accuracy baseline.
- Compile the generated `Dataset.cpp` into the Pico target and use a Pico unit test to check the first/last feature rows are in `[0,1]` and every label row contains exactly one `1.0f`.

### Step 1 — Make memory placement correct and verifiable

- Add one project selector: `MEML_MLP_RUNS_ON_CORE=0|1`; undefined expands all MLP placement macros to blank, and any value other than 0 or 1 is a compile error.
- Derive `MEML_MLP_CODE` and `MEML_MLP_DATA` from that one selector, preventing accidental split placement of code and mutable model state.
- Add generic, neutral hook macros in `mlp/` such as `SMLP_CODE_ATTR`, defaulting to blank. `MemoryDefs.hpp` binds that hook to `MEML_MLP_CODE` before static-MLP headers are included; no RP2350 names or project headers enter `mlp/`.
- Apply the generic code hook to the static MLP hot path: `GetOutput`, pointer-based `TrainBatch`, forward propagation, loss/backpropagation, gradient accumulation, optimizer update, and RNG shuffle helpers.
- Place the experiment’s single static state object—not individual `StaticMLP` members—with `MEML_MLP_DATA`; this places all weights, biases, RMSProp state, buffers, and shuffle indices in the selected bank.
- Reserve `0x20000000–0x2003ffff` for default/core-0 RAM by setting `RAM_LENGTH=256k`, preventing normal allocations from overlapping core-1 SRAM.
- Extend the linker fragments so `.time_critical.core1.code` is flash-loadable into `CORE1_RAM` and `.core1.bank` is a zeroed no-load range there; export their start/end/load symbols.
- Add a Pico-SDK-compatible pre-initialization bootstrap: a normal C/C++ function in default RAM plus a `used` function-pointer entry in `.preinit_array`. Do not replace SDK reset code or add a hand-written reset handler.
- The SDK CRT copies normal RAM first, then invokes this `.preinit_array` entry before ordinary C++ constructors. The bootstrap copies `.time_critical.core1.code` from its flash load address, zeroes the core-1 mutable-state range, and then returns; this makes `MEML_RUNS_ON_CORE(1)` genuinely execute from core-1 SRAM before any selected-state constructor can run.
- Keep core-0 code/data in the default now-reserved 256 KB RAM region; its existing CRT copy path remains valid.
- Add an explicit flash linker output section for `MEML_DATA_ON_FLASH`, ensuring the generated dataset remains XIP-resident under `copy_to_ram`.
- Add linker `ASSERT`s for core-1 code/data/stack capacity and verify both core variants from the ELF map.

#### Testing and validation

- Add a minimal microunit-on-Pico smoke case first (`ASSERT_TRUE(true)`) and require it to compile, link, and print a result over USB before adding MLP tests; this validates the bundled header's `iostream`/static-registration requirements on the actual Pico toolchain.
- Build unpinned, core-0, and core-1 variants. Parse each ELF map in a host-side validation script: selected MLP code/state sections must be in the selected 256 KiB range, the opposite core-specific sections must be empty, and the flash dataset must not be in SRAM.
- Add a constructor-time sentinel in `MEML_MLP_DATA` state. The core-1 unit suite must prove the pre-init routine zeroed it before its constructor ran and that the copied selected function address lies in `[0x20040000, 0x20080000)`.

### Step 1.5 — Add the Pico unit-test harness and core-aware runner

- Add `tests/unit/UnitTestRunner.hpp/.cpp` and test translation units under `tests/unit/`; include `tests/unit/microunit/microunit.h` only in those test translation units.
- Register tests with microunit's `UNIT(...)` macro and execute the complete suite through `microunit::UnitTester::Run()`.
- Add `bool test::unit::RunAllOnSelectedCore()` as the only test-runner API used by `main.cpp`.
- For the unpinned and core-0 configurations, call `UnitTester::Run()` directly on core 0. For `MEML_MLP_RUNS_ON_CORE=1`, launch a dedicated core-1 trampoline; it runs the same registry, returns pass/fail through `multicore_fifo`, then `main` calls `multicore_reset_core1()` before any benchmark later launches core 1.
- Keep MLP test fixtures as static `MEML_MLP_DATA` objects, so their model/scratch storage follows the selected build. The runner, not individual tests, supplies CPU affinity; the same test source runs unchanged in all three configurations.
- Change `main.cpp` sequencing: retain the first USB/key prompt, run `RunAllOnSelectedCore()`, print the microunit result, stop in the failure loop if it fails, then print and wait for a second `Press any key to start benchmarks...` prompt before the existing RAM-independence benchmark.
- Add all unit-test runner/test sources to the Pico executable; do not put microunit or project-specific fixtures under `mlp/`.
- Move validation of the dataset from main.cpp to a dedicated unit test.

#### Testing and validation

- Verify the unpinned/core-0 binaries run the suite on core 0, while the core-1 binary reports `get_core_num() == 1` from an MLP-affine test and returns to core 0 cleanly.
- Verify the second prompt is reached only after a passing suite and that the existing `TestRAMIndependence` benchmark still launches core 1 successfully after the core-1 unit-test reset.

#### Implementation notes / deviations resolved

A review after the initial Step 0–1.5 implementation found the placement *infrastructure* (macros, linker fragments, boot sequence, test runner) was in place, but incompletely applied and unverified. The following were fixed:

- **`SMLP_CODE_ATTR` coverage gap.** The hook was only applied to `StaticMLP`'s wrapper methods (`GetOutput`, `TrainBatch`, `forward_layer`, `backprop_*`), not to the actual inner hot-path functions: `mlp/StaticLayer.h`'s `forward`, `UpdateImmediate`, `AccumulateGradients`, `InitGradientAccumulators`, `ApplyAccumulatedGradients`, `GetGradSumSquared`, `ScaleAccumulatedGradients`, `FastRNG::next_u32`; and `mlp/StaticMLP.h`'s free `compute_loss` and the `for_each_layer` tuple-iteration driver. All are now tagged. `for_each_layer` in particular was caught by the new ELF validator (below), not by inspection: the compiler emitted an out-of-line `.isra` clone of it (called directly from `TrainBatch`) that stayed in the default bank in a `MEML_MLP_RUNS_ON_CORE=1` build.
- **`mlp/StaticMLP.h` included the project's own `MemoryDefs.hpp` directly** — a project-specific header name reaching into the submodule, violating "no RP2350 names or project headers enter `mlp/`". It now includes only the neutral `mlp/Placement.h`. **Standing requirement for later steps:** any project translation unit that needs real (non-blank) placement must `#include "MemoryDefs.hpp"` before any `mlp/*.h` header in that same TU — already true for `tests/TestRAMIndependence.hpp`, and this must hold for Step 4's `MLPOpticalRecognition.hpp/.cpp` too. Getting the order wrong doesn't error; it silently compiles with blank placement, which is exactly what the ELF validator below is for.
- **`MEML_MLP_DATA` had zero use sites.** Added `tests/unit/MLPPlacementTest.cpp`, a placement self-test (not the real experiment) with a `MEML_MLP_DATA`-tagged fixture containing a small `StaticMLP`, a constructor-set sentinel, and a constructor-untouched raw byte array. It asserts: the raw bytes are zero at first read (proving the pre-init bootstrap zeroed the bank before construction — the SDK's normal `.bss` clear only covers `[__bss_start__, __bss_end__]` and never reaches our custom NOLOAD banks), the sentinel holds only the value the constructor writes, `&fixture` and a `MEML_MLP_CODE`-tagged function's address both fall in the selected core's 256 KiB range, and `get_core_num()` matches the selector. **Step 4 should retire or repurpose this fixture** once `MLPOpticalRecognition::State` exists, so the two don't both consume reserved-bank budget.
- **`.core0_bank` had no explicit linker section**, unlike the explicit, `ASSERT`-guarded `.core1_bank`; it only landed correctly via ld's default orphan-section placement. Added an explicit `.core0_bank (NOLOAD)` section in `linker/section_extra_post_data.incl`, symmetric with `.core1_bank`, with `__core0_bank_start__`/`__core0_bank_end__` and an `ASSERT` confining it to `[0x20000000, 0x20040000)`.
  - **This surfaced a real, previously-latent bug**: making `.core0_bank` an explicit NOLOAD section (correctly, since it must not consume flash image space) meant it was no longer zeroed by anything — `.bss`'s clear loop only covers its own fixed symbol range, and crt0's `data_cpy_table` only copies a fixed list of known regions, neither of which extends to a NOLOAD section declared after `.bss` in the script. `memory/CoreSectionInit.cpp`'s `meml_core1_preinit` now zeroes *both* `.core1_bank` and `.core0_bank` before `.init_array` constructors run (it already runs unconditionally on every boot, so this was a natural extension, not a new bootstrap phase).
- **No host-side ELF-map validation script existed.** Added `scripts/validate_memory_placement.py`, wired into `CMakeLists.txt` as a `POST_BUILD` step on every build of every configuration. It checks: `.time_critical.core1.code`/`.core1_bank`/`.core1_stack` and `.core0_bank` stay within their respective 256 KiB windows; `.flash` has no SRAM address; and — the check that catches include-order or missing-hook regressions like the one above — no `smlp::`-namespaced symbol appears in the *other* core's bank for the selected `MEML_MLP_RUNS_ON_CORE` value.
- **`TestRAMIndependence`'s guard was `MEML_MLP_RUNS_ON_CORE == 0`**, excluding it from both the core-1 *and* the unpinned build, though it's only unsafe for `==1` (the benchmark hardcodes its own network onto core 0 via `MEML_DATA_ON_CORE(0)`/`MEML_RUNS_ON_CORE(0)`, independent of the project-wide selector; once the selector is `1`, the now-complete `SMLP_CODE_ATTR` coverage would route that hardcoded core-0 network's code into the core-1-only bank while its state stays on core 0). `main.cpp`'s guard is now `!defined(MEML_MLP_RUNS_ON_CORE) || MEML_MLP_RUNS_ON_CORE == 0`, so the benchmark runs for unpinned and core-0 builds and is excluded only for `=1`. Re-enabling it for `=1` is future work, not part of this fix.

**Follow-up fixes found via a clean `-O0`/`-Og` rebuild** (the incremental builds used above missed these):
- `SMLP_CODE_ATTR`/`MEML_MLP_CODE` reused `MEML_RUNS_ON_CORE(n)`, which carries `__attribute__((used))`. On a class-template member function, `used` forces GCC to eagerly instantiate it (and everything it calls) the moment the enclosing template is instantiated, even if the member is never actually called — pulling every untagged callee (`activate`, `nn::exp`, `nn::sqrt`, `std::get`, ...) into the default bank regardless of selector. `MemoryDefs.hpp` now defines a separate `MEML_RUNS_ON_CORE_CODE(n)` (same section, no `used`) that `MEML_MLP_CODE` binds to; `MEML_RUNS_ON_CORE(n)` itself is untouched since its other use sites (address-taken entry points registered as raw function pointers) still need `used` for the compiler to see the reference.
- `mlp/StaticLayer.h`'s free `activate`, `activate_deriv`, `activate_deriv_cached` templates (called from the now-tagged `forward`/`UpdateImmediate`/`AccumulateGradients`) were untagged and, at low optimization, didn't get inlined into their tagged callers. Now tagged with `SMLP_CODE_ATTR`.
- `scripts/validate_memory_placement.py`'s "no smlp:: symbol leaks into the other bank" check used a substring match (`"4smlp" in name`), which also matched `std::get<I>(tuple<smlp::StaticLayer<...>>)` — a libstdc++ symbol that merely mentions an `smlp::` type in its template arguments, not something `SMLP_CODE_ATTR` can ever reach. Changed to a prefix check (`name.startswith("_ZN4smlp")`), which only matches symbols actually defined inside `namespace smlp`. The check also now excludes constructors/destructors (Itanium `C1E`/`C2E`/`D0E`/`D1E`/`D2E` markers): they run once during the single-core boot sequence before core 1 is ever launched and were never part of the plan's hot-path list.
- `tests/unit/MLPPlacementTest.cpp` originally only constructed its probe `StaticMLP` without calling any method on it; once the `used`-attribute bug above was fixed, `--gc-sections` correctly discarded the now-provably-dead hot-path code, leaving nothing for the validator to check. It now calls `net.GetOutput(...)` once so the hot path is genuinely exercised and its placement genuinely verified.

**Consequences for Steps 2–5:**
- Step 2's `compute_loss` refactor into shared `Loss.h`/`Utils.h` helpers must preserve the `SMLP_CODE_ATTR` tag on the replacement functions.
- Step 3's new pointer-based `TrainBatch` overload must follow the same tagging pattern — now established precedent, not a new mechanism to invent.
- Step 4's `MLPOpticalRecognition::State`/methods must observe the `MemoryDefs.hpp`-before-`mlp/*.h` include order, and should retire or repurpose `tests/unit/MLPPlacementTest.cpp` rather than let it and the real `State` object both occupy reserved-bank space simultaneously.
- Step 5's "host-side ELF-map validation script" testing requirement is already satisfied by `scripts/validate_memory_placement.py` — reuse it rather than writing a new one; extend its checks (e.g. dataset-address, stack-capacity) as those steps land.
- Tagging more of `StaticLayer`'s hot path with `SMLP_CODE_ATTR` (which implies `noinline`) removes some inlining opportunity the compiler previously had into `forward_layer`/`TrainBatch`. Expected to be negligible at RP2350 clock speeds and the per-sample/per-epoch call counts involved, but worth keeping an eye on during Step 5's 20/100-epoch smoke-test wall-clock timing.

### Step 2 — Complete generic loss and softmax support

- Add allocation-free pointer-and-length helpers in `mlp/Loss.h` for stable categorical cross-entropy and its logits gradient: `softmax(logits) - one_hot_target`.
- Calculate categorical loss with log-sum-exp and `-Σ target[i] * log_softmax[i]`; retain vector overloads by delegating to the pointer implementation.
- Add allocation-free, max-shifted `SoftmaxInPlace(T* values, std::size_t count)` in `mlp/Utils.h`; make existing vector softmax delegate to it.
- Make `StaticMLP` use these shared helpers rather than its duplicate softmax/loss implementation.
- Configure optical recognition with `Loss::LOSS_CATEGORICAL_CROSSENTROPY` and final `LINEAR` activation. Softmax is only used when probabilities are requested; class prediction and accuracy use raw-logit argmax.

#### Testing and validation

- Add microunit cases for known logits: stable softmax sums to one, is invariant to adding a common constant, preserves argmax, and remains finite for large positive/negative logits.
- Add categorical-cross-entropy fixtures with hand-calculated loss/gradient values, including a non-one-hot probability target to verify the full `-Σ target * log_softmax` contract.
- Run these same cases through `RunAllOnSelectedCore()` in all three build variants; the core-1 variant thereby exercises copied MLP code on core 1.

### Step 3 — Add a zero-copy static training interface

- Keep legacy vector-based `TrainBatch` unchanged for compatibility.
- Add a distinct `StaticMLP::TrainBatch` overload accepting:
  - feature base pointer and element stride;
  - label base pointer and element stride;
  - sample count;
  - learning rate, epoch count, batch size, and stopping threshold;
  - caller-provided `std::size_t*` shuffle-index storage and its capacity.
- Derive input/output widths from the `StaticMLP` template; validate non-null pointers, non-zero sample/batch counts, adequate strides, and shuffle capacity before training.
- Shuffle the supplied index buffer with the existing deterministic `FastRNG`; do not allocate vectors or convert arrays.
- Train directly from `dataset::features[i].data()` and `dataset::labels[i].data()`. Fine-tuning becomes another pointer/count/stride view plus a suitably sized caller-owned index buffer.
- Preserve RMSProp and gradient clipping for this path, as selected; the notebook remains the corrected SGD reference for architecture/data/loss, not an optimizer-identical benchmark.

#### Testing and validation

- Add microunit cases using small static, strided feature/label arrays; assert pointer-based training matches an independently calculated one-sample update and never modifies source data.
- Add validation cases for null pointers, zero counts/batch size, undersized strides, and insufficient shuffle capacity. Define the overload's failure contract as `false` with no model or shuffle-buffer mutation; return the epoch loss through an output reference only on success.
- Add deterministic-shuffle tests: two networks with the same seed and caller-owned index buffers must produce identical index order, loss, and weights.
- Keep the definitive allocation probe host-only because microunit itself allocates for registration/reporting. On Pico, run the identical hot path functionally with caller-owned buffers; on the host, enable the global allocation counter only around the pointer-based training/inference calls and require zero allocations.

### Step 4 — Implement the experiment facade

- Repair and complete `MLPOpticalRecognition.hpp`, with definitions in `MLPOpticalRecognition.cpp` so the selected template specialization and its code sections have one deterministic home.
- Define `Net` as `StaticMLP<float, Layout<64,64,32,10>, Activations<RELU,RELU,LINEAR>, LOSS_CATEGORICAL_CROSSENTROPY>`.
- Add a private core-local `State` containing `Net`, a `std::array<std::size_t, dataset::kNumExamples>` shuffle buffer, and fixed input/output scratch arrays.
- Expose `Initialise(uint32_t seed)`, `Train(uint32_t epochs, float learning_rate)`, and `Predict(...)`; initialization explicitly seeds and Xavier-initializes the network, while `Train` preserves state for fine-tuning.
- Make `Train` call the pointer-based batch trainer with the flash dataset and batch size 128, then calculate final mean categorical loss and training accuracy without allocating.
- Make `Predict` run inference through the fixed arrays and return the argmax class.
- Annotate experiment entry methods with `MEML_MLP_CODE` and state with `MEML_MLP_DATA`.
- The caller owns CPU scheduling: invoke these methods directly on core 0, or from its own `multicore_launch_core1` trampoline on core 1. The facade must not be concurrently accessed from both cores.

#### Testing and validation

- Add a microunit fixture with the selected-core static `State`; test deterministic `Initialise`, a finite initial loss, a successful short train call, and `Predict` returning an index below `dataset::kLabelSize`.
- In the core-1 build, execute this fixture only through the core-1 runner and assert both `get_core_num() == 1` and `&State` lies in `[0x20040000, 0x20080000)`. In core-0/unpinned builds, assert core 0 and the corresponding default/core-0 range.
- Keep the full-data smoke training out of the default unit suite; use a small deterministic dataset subset so USB-connected Pico unit runs remain short.

### Step 5 — Build integration and verification

- Add `dataset/Dataset.cpp` and `experiments/MLPOpticalRecognition.cpp` to the Pico target.
- Build three configurations: selector undefined, `MEML_MLP_RUNS_ON_CORE=0`, and `MEML_MLP_RUNS_ON_CORE=1`.
- Add host/static tests for stable softmax, cross-entropy values and gradients, raw-logit class selection, zero-copy strided training, invalid-view rejection, deterministic shuffling, and absence of allocations in the new path. The no-allocation test uses a host-only global `new`/`new[]`/sized-delete/`delete[]` counter, reset immediately before the hot call and required to remain zero afterward.
- Add Pico integration checks that initialize, train, and predict from the caller-selected core; use the linker map to assert selected code and state addresses lie in the intended 256 KB bank.
- Verify the core-1 image bootstraps its copied sections before constructors/use, training loss decreases over a short run, accuracy improves above an agreed smoke-test threshold, and no selected-core MLP data appears in the other core’s SRAM region.

#### Testing and validation

- Treat the Pico microunit suite as the functional gate run interactively after the first `main.cpp` prompt; a failure prevents benchmarks from starting.
- Treat host unit tests and ELF-map parsing as build/CI gates for all three selector configurations. Hardware validation is required before release: capture USB output showing unit success, selected CPU ID, section-address checks, the 20-epoch smoke result, and successful subsequent benchmark launch.
- Run the 20-epoch and 100-epoch full-dataset checks as named Pico integration tests, not microunit cases, using the fixed seed and thresholds already specified below.

All new or modified classes, structs, functions, overloads, macros, and linker bootstrap symbols receive Doxygen headers. Implementation logic branches and loops receive concise one-line comments, especially for section copying, validation, shuffling, loss reduction, and core hand-off boundaries.

## Design Decisions and Details

### Fixed configuration and ownership

- `MEML_MLP_RUNS_ON_CORE` is a compile-time selector, never a runtime switch.
  - Undefined: one unpinned binary; MLP placement annotations are blank.
  - `0`: a core-0-pinned binary.
  - `1`: a core-1-pinned binary.
  - Any other value: preprocessor error.
- CMake exposes the same cache variable and adds `MEML_MLP_RUNS_ON_CORE=<0|1>` only when selected. CI builds all three configurations.
- The caller schedules work. `MLPOpticalRecognition` is not a multicore dispatcher; core 0 calls it directly, while core 1 calls it from the caller’s `multicore_launch_core1` trampoline. Concurrent access is forbidden.

### Exact RP2350 memory and boot contract

- Reserve SRAM banks 0–3 for default/core-0 use: `0x20000000–0x2003ffff` (256 KiB).
- Reserve SRAM banks 4–7 for core-1-local code/data: `0x20040000–0x2007ffff` (256 KiB).
- Leave scratch SRAM 8–9 (`0x20080000–0x20081fff`) out of scope.
- Set the default linker `RAM_LENGTH=256k`; this prevents ordinary SDK allocations from overlapping core-1 memory.
- Core-0 build rule: selected MLP code and state addresses must be in `[0x20000000, 0x20040000)`.
- Core-1 build rule: selected MLP code, state, and reserved core-1 stack must be in `[0x20040000, 0x20080000)`.
- The linker must assert the core-1 region does not overflow after accounting for selected code, state, and the 4 KiB stack.
- `MEML_RUNS_ON_CORE(1)` is repaired to be meaningful: `.time_critical.core1.code` becomes a loadable flash-backed section whose execution address is in core-1 SRAM.
- Use the Pico SDK's existing CRT lifecycle; do not replace `_entry_point`, `crt0`, `runtime_init`, or the SDK data-copy table.
- Implement `meml_core1_preinit()` as an ordinary, default-RAM function and register its address with a `used` pointer in `.preinit_array`. GNU/Pico CRT invokes this entry after its normal RAM copy and before `.init_array` C++ constructors.
- Link `.time_critical.core1.code` with a flash load address plus SRAM execution address; the pre-init routine copies exactly `__core1_code_end__ - __core1_code_start__` bytes from `__core1_code_load__` to `__core1_code_start__`.
- Keep the core-local `State` zero-initialized. Link its `.core1.bank` range as no-load SRAM and zero exactly `__core1_bank_end__ - __core1_bank_start__` bytes in the same pre-init routine; its ordinary C++ constructor then runs against valid zeroed storage.
- Do not defer copying/zeroing to the core-1 trampoline: static constructors may access selected state first. The bootstrap is a no-op for unpinned/core-0 builds and is verified with a constructor-time sentinel test.

### Zero-copy static training API

- Keep legacy vector-based `TrainBatch` unchanged.
- Add this separate generic overload:

```cpp
bool TrainBatch(
    const T* features, std::size_t feature_stride,
    const T* labels, std::size_t label_stride,
    std::size_t sample_count,
    float learning_rate, uint32_t epochs,
    std::size_t batch_size,
    std::size_t* shuffle_indices,
    std::size_t shuffle_index_capacity,
    T& final_loss,
    float min_error_cost = 0.001f);
```

- Strides are measured in elements. `StaticMLP` derives required row widths from `kNumInputs` and `kNumOutputs`.
- Require non-null feature/label pointers, non-zero sample count and batch size, strides at least the compiled widths, and `shuffle_index_capacity >= sample_count`. On validation failure return `false` without changing the network, shuffle buffer, or `final_loss`; write `final_loss` only after a successful training run.
- The generic caller supplies a pointer-and-capacity pair; it may use any static buffer size appropriate to its training/fine-tuning dataset.
- `MLPOpticalRecognition` owns `std::array<std::size_t, dataset::kNumExamples>` in its core-local `State` and passes it to this API.
- Training reads `dataset::features[i].data()` and `dataset::labels[i].data()` directly from flash. There is no array-to-vector conversion, heap ownership, or duplicate dataset buffer.

### No-allocation and model behavior

- The new pointer overload must not construct or resize vectors, allocate index storage, or call `new`, `delete`, `malloc`, or `free`.
- The enforced test mechanism is a host-only global-allocation counter, not an allocator wrapper: one dedicated test translation unit overrides `operator new`, `operator new[]`, all matching sized deletes, and aligned overloads used by the selected C++ standard library.
- Each replacement delegates storage to `std::malloc`/`std::free` and increments atomic call/byte counters only while a scoped `AllocationProbe` is enabled. This avoids counting test/framework setup while still catching every C++ heap allocation made by the hot call.
- Construct and initialize the network before enabling the probe; enable it immediately before pointer-based `TrainBatch`, disable it immediately afterward, then assert zero allocation calls and zero requested bytes. Repeat around inference and evaluation helpers used by the facade.
- API design remains the first guarantee: shuffle storage is mandatory caller-owned memory. Hot-path review must contain no vector/string/container construction, but the allocation-counter assertion is the definitive automated enforcement.
- Add stable pointer-based softmax and categorical-cross-entropy helpers; loss uses log-sum-exp and the gradient is `softmax(logits) - target`.
- Correct `preprocess.ipynb`: remove its final Softmax, use `64→64→32→10`, retain PyTorch SGD/batch-128 as the corrected reference, and seed Torch/DataLoader.
- The Pico experiment intentionally keeps existing RMSProp plus clipping, batch size 128, and the corrected architecture/loss semantics.
- `MLPOpticalRecognition::Net` is `Layout<64,64,32,10>`, `RELU, RELU, LINEAR`, and categorical cross-entropy.
- Add explicit `Initialise(seed)`, state-preserving `Train(epochs, learning_rate)`, and `Predict(input)` methods. `Initialise(0xC0DEu)` seeds and Xavier-initializes once; `Train` supports later fine-tuning.

### Verification contract

- Host CI: run loss/softmax/gradient tests, zero-copy stride tests, invalid-view validation, deterministic-shuffle tests, and no-allocation instrumentation.
- Cross-build CI: compile unpinned, core-0, and core-1 variants; parse each ELF map and fail if any selected section violates the stated address range or bank capacity.
- Hardware release validation: run the selected-core trampoline and confirm the pre-init sentinel, MLP result, and addresses over serial.
- Deterministic smoke configuration: seed `0xC0DEu`, RMSProp, `lr=0.01`, batch size 128, 20 epochs. Require final mean loss ≤ 70% of initial mean loss and training accuracy ≥ 80%.
- Extended hardware acceptance: 100 epochs under the same configuration must reach ≥ 90% training accuracy. A 1000-epoch reference run is recorded for comparison, but is not a per-PR gate.
- All new/changed macros, linker symbols, types, and methods receive Doxygen headers; each validation, copy/zeroing, shuffle, and training-loop step receives a concise one-line implementation comment.


### Step 6 — Implement a true, core-agnostic RAM Independence Benchmark

**Preconditions:** this benchmark only builds for `MEML_MLP_RUNS_ON_CORE=0` or `=1`; it requires a defined "MLP core" vs. "other core" pairing, which the unpinned/blank-placement configuration does not have. Guard the whole test behind `#if defined(MEML_MLP_RUNS_ON_CORE)` and give a clear preprocessor `#error` (or a skipped/no-op translation unit) for the unpinned build rather than silently compiling something meaningless.

- Delete `TestRAMIndependence`'s hardcoded `TestNN`/`core0_nn_` fixture ([TestRAMIndependence.hpp:34-53](tests/TestRAMIndependence.hpp#L34-L53)); it duplicates `MLPOpticalRecognition` and was already flagged in Step 1's implementation notes as not core-agnostic (it always ran on core 0 regardless of the selector). Replace it with one `MEML_MLP_DATA`-tagged `MLPOpticalRecognition` instance, placed on whichever core `MEML_MLP_RUNS_ON_CORE` selects — this is a build-owned instance, separate from the unit-test suite's `g_experiment` (`tests/unit/MLPOpticalRecognitionTest.cpp:46`); they never coexist in one binary since `tests`/`benchmarks` are still separate `RUN_TESTS_OR_BENCHMARKS` build modes.
- Keep `utils::RAMFlooder<uint32_t, core1_ram_flooder_size_>` (240 KB) on the core `MEML_MLP_RUNS_ON_CORE` did *not* select — i.e. the flooder and the MLP experiment are always on opposite cores, driven by the same selector, not independently configurable. Its static footprint (240 KB) is fixed at compile time regardless of the `runOtherCoreTask` runtime flag below; disabling that flag stops it from doing memory traffic, it does not free the RAM it reserves.
- `CMakeLists.txt` currently only adds `dataset/Dataset.cpp` and `experiments/MLPOpticalRecognition.cpp` under `RUN_TESTS_OR_BENCHMARKS=tests` ([CMakeLists.txt:56-67](CMakeLists.txt#L56-L67)). Add both to the `benchmarks` branch too, since `TestRAMIndependence` now links against the real facade.
- Constructor parameters (replacing the current single-`clock_frequency_hz` constructor):
  - `clock_frequency_hz` (unchanged).
  - `training_repeats` (default 10): number of independent train-from-scratch sessions on the MLP core. Assign this to whichever of `TestConfig::core0_iterations`/`core1_iterations` corresponds to `MEML_MLP_RUNS_ON_CORE`.
  - `epochs_per_session` (default 10): epochs passed to each `Train()` call within a session.
  - `runOtherCoreTask` (bool): selects, at construction time via `ConfigureCoreWork`, which of **two distinct task functions** runs on the other core — this is a one-time selection of a function pointer, not a runtime branch inside either hot loop. `true` selects the flooding task (`RAMFlooder::FillOnce()`, guarded by the completion signal below). `false` selects a dormant-idle task that puts the core into an ARM `WFE` low-power wait and performs no memory traffic and no polling loop at all until woken — this is the condition modeling "the other core completely inactive," not a busy no-op.
- **Completion signal:** add one plain (not `MEML_MLP_DATA`/`MEML_MLP_CODE`-tagged — it is not `smlp::` state and must stay outside `validate_memory_placement.py`'s bank-exclusivity check) `inline static std::atomic<bool> training_done_{false}` member. The MLP-core task sets it to `true` (release ordering) immediately after its **last** training session completes, then issues one `__sev()` (ARM Send Event, exposed by the SDK's `hardware/sync.h`) to wake any core blocked in `WFE`. The flooding task (`runOtherCoreTask=true`) checks the flag (acquire ordering) at the top of every iteration before calling `FillOnce()` — negligible overhead next to the flood traffic itself. The dormant task (`runOtherCoreTask=false`) never polls: it executes `while (!training_done_.load(std::memory_order_acquire)) { __wfe(); }`, blocking in `WFE` the whole time and only re-checking the flag on a wake event (the `SEV` above, or a spurious wake, which `WFE` permits), so it generates zero memory-bus traffic in the interval between construction and the MLP core finishing. RP2350's Cortex-M33 cores have no data cache over SRAM, so this ordering is sufficient without extra barriers.
- **Re-seeding without changing `TestBase`'s callback signature:** `ConfigureCoreWork` callbacks remain no-argument function pointers ([TestBase.hpp:52-58](tests/TestBase.hpp#L52-L58)); do not change that contract. The MLP-core task instead keeps its own `inline static uint32_t session_index_{0}` counter, incremented on entry, and seeds each session with `kBaseSeed + session_index_` (`kBaseSeed = 0xC0DEu`, matching the existing convention in `MLPOpticalRecognitionTest.cpp`) — deterministic per session, varying across sessions, reproducible across runs.
- **Iteration counts for the other core:** `TestBase`'s per-core loop is a fixed iteration count known at construction time ([TestBase.hpp:143-160](tests/TestBase.hpp#L143-L160)), with no early-exit primitive, and this step must not modify that shared contract. For `runOtherCoreTask=true` (flooding), give the non-MLP core's `TestConfig` iteration count a fixed, generous safety cap (e.g. `training_repeats * epochs_per_session * 50`, documented inline with the sizing rationale) — large enough that `training_done_` is essentially guaranteed to already be `true` well before the cap is reached, after which every further iteration is just an atomic load and a return (cheap, sub-microsecond). For `runOtherCoreTask=false` (dormant), set the iteration count to exactly `1`: the single task call *is* the entire `WFE`-wait-until-signaled, so there is no cap to size and no repeated-loop overhead of any kind.
- **Recording results:** store each session's `{loss, accuracy}` in a fixed-size `std::array<MLPOpticalRecognition::Result, kMaxTrainingRepeats>` (`kMaxTrainingRepeats` a compile-time constant, e.g. 32) alongside the count actually used, and expose both via a getter (mirroring `GetResults()`). `TestBase::TestResultsPerCore` (avg/max/min time) already captures per-session MLP-core timing through the existing `StartMeasurementCore{0,1}`/`StopMeasurementCore{0,1}` calls placed around each `Train()` call — no new timing plumbing is needed there.
- Model construction/init sequencing after `MLPOpticalRecognitionTrainingSmokeTest` ([tests/unit/MLPOpticalRecognitionTest.cpp:81-112](tests/unit/MLPOpticalRecognitionTest.cpp#L81-L112)) for the seed/epoch/threshold conventions, but note it is a single-shot microunit case; this benchmark instead follows `TestRAMIndependence`'s own existing pattern of being driven directly from `main.cpp` under `MEML_ENABLE_RAM_INDEPENDENCE_BENCHMARK`, outside microunit, since it is long-running and multi-session by design.

#### Testing and validation

- Extend `scripts/validate_memory_placement.py` (reused per Step 1's implementation notes, not replaced) to also check the benchmark's `MLPOpticalRecognition` instance and `RAMFlooder` instance land in opposite banks matching `MEML_MLP_RUNS_ON_CORE`, exactly as the existing MLP-vs-other-core checks already do for the unit-test build.
- **Correctness independence:** run the benchmark twice with identical construction parameters, once with `runOtherCoreTask=true` and once `=false`. Assert the recorded `{loss, accuracy}` sequences are bit-identical between the two runs for every session index — RAM-bank independence means core-1 memory traffic must not perturb core-0's (or vice versa) computed results at all, not just approximately.
- **Timing independence:** from the same two runs, assert the MLP core's `TestResultsPerCore.time_us_avg` (and `time_us_max`) differ by no more than an agreed tolerance (e.g. 5%) between `runOtherCoreTask=true` and `=false`. A larger deviation indicates the two cores are not actually RAM-independent (e.g. shared bus/flash contention) and the test should fail.
- For the `runOtherCoreTask=true` run, confirm `training_done_` is observed `true` by the flooding core well before its safety-cap iteration count is exhausted (log or assert the actual iteration count consumed vs. the cap, so a too-small cap is caught rather than silently truncating the flood). For the `runOtherCoreTask=false` run, confirm the dormant core's reported iteration count is exactly `1` (its `TestResultsPerCore` timing is not meaningful and is not asserted on).
- Treat this as a hardware integration benchmark, not a microunit case, consistent with the existing `TestRAMIndependence` invocation from `main.cpp`: capture USB output showing both runs' loss/accuracy sequences, both timing summaries, and the pass/fail verdict for the two assertions above.
