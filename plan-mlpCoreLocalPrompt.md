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
