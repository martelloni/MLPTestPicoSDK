# AGENTS.md

## Three builds (`MEML_MLP_RUNS_ON_CORE`)

| Config | Value | Dir | MLP hot path lives in |
|---|---|---|---|
| core0-pinned | `0` | `build-core0` | core 0 bank `[0x20000000, 0x20040000)` |
| core1-pinned | `1` | `build-core1` | core 1 bank `[0x20040000, 0x20080000)` |
| unpinned | `` | `build-naive` | no constraint (naive counterfactual) |

Build each exactly like `RunAndSaveBenchmarks.py` (its `configure()`/`build()`
is canonical — if this file disagrees, trust the script):

```sh
cmake -S . -B build-<name> -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DRUN_TESTS_OR_BENCHMARKS=benchmarks -DMEML_MLP_RUNS_ON_CORE=<0|1|>
cmake --build build-<name>
```

Use `RUN_TESTS_OR_BENCHMARKS=tests` for the unit-test suite instead. Don't
improvise generator/build-type/flags — a mismatch has already caused a build
dir to silently drift from what the harness runs.

`cmake --build` auto-runs `scripts/validate_memory_placement.py` as
`POST_BUILD`; failure fails the build. Never bypass/weaken it — fix placement.

## Memory constraint

`build-core1`: MLP hot path (forward/backward/train + everything it calls)
must run entirely from `[0x20040000, 0x20080000)` — zero code/data touching
core 0's bank. `build-core0`: symmetric, zero straddling into
`[0x20040000, 0x20080000)`. `build-naive`: no constraint.

Why it matters: these builds exist to prove MLP timing is independent of the
other core. Any hot-path code crossing banks silently breaks that guarantee —
the benchmark still runs and prints numbers, but they no longer mean what's
claimed.

Already happened once: linking CMSIS-DSP's `arm_dot_prod_f32()` (used by
`mlp/StaticLayer.h`'s `ARM_MATH_CM33` branch) added untagged object code the
linker defaulted into core 0's bank regardless of pin. Fixed via
`linker/core1-only/section_copy_to_ram_text.incl`'s
`.time_critical.core1.untagged_hotpath` section.

**Any new MLP code → build both core0 and core1 → confirm
`validate_memory_placement.py` passes → spot-check with**
`arm-none-eabi-nm build-core1/MLPTestPicoSDK.elf | grep <symbol>` **(address
must fall in the right window) — don't trust a green build alone unless you
know the check covers it (see below).**

## Untagged code needs its own check

Most MLP code is placed via `SMLP_CODE_ATTR` and caught by
`validate_memory_placement.py`'s mangled-prefix match (`_MLP_SYMBOL_PREFIXES`:
`_ZN4smlp`, `_ZNK4smlp`, `_ZZN4smlp`, `_ZZNK4smlp`, `_ZN21MLPOpticalRecognition`).

Code that can't carry a `section` attribute (invisible to that check) needs
manual handling — today's two cases, both redirected in
`linker/core1-only/section_copy_to_ram_text.incl`'s
`.time_critical.core1.untagged_hotpath`, copied by
`memory/CoreSectionInit.cpp`'s `meml_core1_preinit()`:
- per-instantiation templates (e.g. `for_each_layer<F,I>`) — matched by
  mangled-name substring
- third-party calls (e.g. `arm_dot_prod_f32`) — matched by archive member

**Rule: any new hot-path addition that could leave untagged code (new
templated closure, new third-party call, compiler-generated
thunk/vtable/escaping-lambda) needs BOTH:**
1. A linker redirect in `section_copy_to_ram_text.incl` if it doesn't land
   in the right bank by default (don't assume existing wildcards catch it).
2. A new check in `validate_memory_placement.py`, registered in `main()`'s
   pinned-build branch — template: `check_cmsisdsp_symbol_placement`. Fail on
   landing in the *other* core's bank specifically, not just "unexpected".

When adding a check: confirm it fails against the unfixed placement first,
then fix and confirm it passes.
