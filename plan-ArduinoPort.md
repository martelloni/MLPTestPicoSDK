# Implementation plan: RP2350 core-memory-independence for Arduino-Pico

## 0. Scope and deliverables

Two repositories, both under `github.com/MusicallyEmbodiedML`:

1. **`MusicallyEmbodiedML/arduino-pico`** — a maintained fork of `earlephilhower/arduino-pico`,
   diffed minimally against upstream, extending the existing `solderparty_rp2350_stamp_xl` board
   entry with the SRAM bank-split linker plumbing. This is the **board definition** deliverable.
2. **`MusicallyEmbodiedML/MEMLCoreSplit`** — a normal Arduino library providing the placement
   macros, the preinit bootstrap, and an example sketch. This is the **companion library**
   deliverable.

The CMake project's "naive/unpinned layout" counterfactual (Step 7 of the original plan,
`TestNaiveMemoryLayout`, the flooder-size probe build) is **out of scope for this port**. It exists
in `MLPTestPicoSDK` only to quantify how much the placement scheme is worth; the Arduino package
ships the scheme itself, not the benchmark that justified building it. Nothing in the plan below
re-creates the two-pass probe-build trick.

Every phase below ends in a **test milestone** — concrete, checkable pass/fail criteria — rather
than deferring all testing to one end-of-plan pass. A phase is not done until its milestone is met.

### Forking policy

Per instruction, upstream `arduino-pico` is not edited directly. Everything that genuinely
requires `boards.txt`/`platform.txt` control (confirmed necessary below, §1.1) lives in the fork,
kept as small and rebase-friendly as possible: new files wherever feasible, additive lines in
`boards.txt`/`platform.txt` where not, and **zero changes to the shipped `memmap_default.ld`**
(the split is spliced in at build time by a hook, not hand-edited into the base script). Upstreaming
via PR is explicitly deferred, not attempted in this plan.

### Clarifying "either core" — this is a build-time choice, not a runtime one

SRAM-bank placement is a **link-time** fact: once a function or object is placed, its address — and
therefore its bank — is fixed for the life of that binary. This plan does **not** compile `mlp`
into both banks and switch between them at runtime — that would require the library to exist
simultaneously on both cores, which is not a configuration the original `MLPTestPicoSDK` benchmark
builds or verifies, and this port does not go further than what's actually tested there. Instead,
which core the MLP workload is pinned to is selected the same way `MLPTestPicoSDK`'s CMake build
selects it — a single build-time switch (there, `MEML_MLP_RUNS_ON_CORE`; here, an Arduino **Tools
menu** entry on the board) that produces one pinning per build. Changing which core the MLP runs
on means re-selecting the menu entry and re-flashing, exactly as it means re-running CMake
configure with a different cache value today. The example in §6 is built and tested this way, once
per menu value — not as one image that decides at runtime.

---

## 1. Findings this plan depends on (from the feasibility study)

Confirmed directly against the installed `arduino-pico` 5.4.2 core (not docs):

- `solderparty_rp2350_stamp_xl` already exists in `boards.txt`: plain RP2350, no PSRAM, 16 MB
  flash, `build.core=rp2040`, `build.chip=rp2350`, uses the stock `lib/rp2350/memmap_default.ld`.
  Its SRAM layout (`RAM` 0x20000000 + `SCRATCH_X`/`SCRATCH_Y` at 0x20080000) is identical to Pico 2,
  so the 256K/256K bank split from `MLPTestPicoSDK`'s linker fragments ports with **unchanged
  addresses**.
- `platform.txt`'s `recipe.hooks.linking.prelink.{1,2}.pattern` already does exactly what we need a
  `.3` step for: `prelink.1` runs `simplesub.py` to materialize `{build.path}/memmap_default.ld`
  from the template; `recipe.c.combine.pattern` links against that materialized file. A
  `recipe.hooks.linking.prelink.3.pattern` step, added in the fork, can textually splice our
  `MEMORY`/`SECTIONS` fragments into that already-materialized file before the link step runs —
  the same effect as the CMake project's `pico_add_linker_script_override_path`, achieved as a
  build hook instead of a CMake extension point. **This is why a plain library cannot do this work
  — `platform.txt` ownership is unavoidable**, confirming the board-definition layer is required
  (not just a style preference).
- `compiler.flags` already includes `-ffunction-sections -fdata-sections`, and `compiler.ldflags`
  already includes `-Wl,--gc-sections`. This is the exact precondition for the COMDAT-folding
  hazard `MEML_RUNS_ON_CORE(_CODE)`'s `__COUNTER__` trick exists to avoid in `MLPTestPicoSDK` — so
  that trick is **not optional cargo-culting**, it is independently necessary here too.
- `cores/rp2040/main.cpp`'s `main()` runs every `.preinit_array`/`.init_array` entry (global C++
  constructors included) once, on core 0, **before** `multicore_launch_core1(main1)` is ever
  called. This is the same single-core-boot ordering `CoreSectionInit.cpp` relies on, so the
  `.preinit_array.12000`-registered bootstrap ports with no ordering changes.
- Core 1's call-stack is **already** placed in `SCRATCH_X`, physically separate from core 0's stack
  in `SCRATCH_Y` (`pico-sdk`'s `multicore.c`, `PICO_CORE1_STACK_SIZE`, default 0x800) — this part of
  "core independence" is free today and the plan does not touch it.
- `setup1()`/`loop1()` are declared `__attribute__((weak))` in the core's `main.cpp` with no
  `section` attribute. Whether a re-declaration in a header (adding a `section` attribute ahead of
  the sketch's own definition) actually binds to the sketch's later definition under `-weak` is
  **not yet verified** — this is Phase 0's spike, not an assumption the plan builds on blindly.

---

## 2. Repository layout

```
MusicallyEmbodiedML/arduino-pico/            (fork of earlephilhower/arduino-pico, pinned branch per upstream release)
├── boards.txt                               # + menu.memlcoresplit on solderparty_rp2350_stamp_xl only
├── platform.txt                             # + recipe.hooks.linking.prelink.3
│                                             # + recipe.hooks.objcopy.postobjcopy.1
├── lib/rp2350/memlcoresplit/
│   ├── memory_extra.incl                    # MEMORY { CORE1_RAM ... }
│   ├── section_extra_post_text.incl         # .time_critical.core1.code
│   └── section_extra_post_data.incl         # .core1_bank / .core0_bank / .core1_stack
└── tools/memlcoresplit/
    ├── splice_memmap.py                     # the prelink.3 hook
    └── validate_corebanks.py                # the postobjcopy.1 hook

MusicallyEmbodiedML/MEMLCoreSplit/           (Arduino library)
├── library.properties
├── src/
│   ├── MEMLCoreSplit.h                      # MEML_RUNS_ON_CORE_CODE(n)/MEML_DATA_ON_CORE(n)/MEML_DATA_ON_FLASH
│   └── MEMLCoreSplitInit.cpp                # ported meml_core1_preinit(), .preinit_array.12000
├── examples/
│   └── XOR_EitherCore/
│       └── XOR_EitherCore.ino
└── docs/
    ├── README.md
    ├── porting-a-library.md                 # "how mlp (or any lib) adopts this" guide
    └── internals.md                         # linker/preinit mechanism, for maintainers
```

---

## 3. Phase 0 — De-risking spikes (no committed artifacts yet)

**Goal:** resolve the two assumptions the rest of the plan would otherwise build on blind.

Steps:
1. Toolchain probe for `setup1`/`loop1` attribute inheritance: in a scratch sketch, declare
   `extern void setup1() __attribute__((weak, section(".time_critical.core1.entry")));` in a
   header included ahead of the `.ino`, define `void setup1(){...}` in the `.ino` with no
   attribute, compile with the Stamp XL's real flags, `readelf -S`/`nm` the object.
2. Hook-numbering probe: add a throwaway `recipe.hooks.linking.prelink.3.pattern` to a **local
   copy** of `platform.txt` (not yet the fork) that just touches a marker file; build a stock
   sketch; confirm the marker exists and that it ran after `prelink.1`/`.2` (check the marker's
   mtime against the materialized `memmap_default.ld`'s mtime).
3. Confirm `arm-none-eabi-readelf`/`nm` paths resolve the same way inside an Arduino CLI build as
   they do in the existing `scripts/validate_memory_placement.py` (same binaries, different
   invocation context — `{compiler.path}` vs. a bare `arm-none-eabi-*` on `$PATH`).

**Test milestone (must pass before Phase 1 starts):**
- Probe 1's `readelf` output either confirms the attribute survived onto the definition, or
  conclusively shows it didn't. Either outcome is a pass — this milestone is "the answer is known
  and written down," not "the answer is yes." The answer gates whether §6.1's example can rely on
  bare `setup1`/`loop1` placement or must use the explicit-tagging fallback.
- Probe 2's marker file is created on every build and postdates the materialized linker script.
- Probe 3's tool invocations succeed from inside an `arduino-cli compile --verbose` log.

---

## 4. Phase 1 — Fork and linker split

**Goal:** `solderparty_rp2350_stamp_xl` gains a second 256 KiB SRAM region (`CORE1_RAM`,
`0x20040000`–`0x2007FFFF`) and the matching NOLOAD banks, with zero effect on any other board in
the fork and zero effect on this board when the new menu is left at its default.

Steps:
1. Fork `earlephilhower/arduino-pico` into the org at the commit matching the already-installed
   5.4.2 release tag, so the local toolchain and the fork start in lockstep.
2. Port `linker/memory_extra.incl`, `section_extra_post_text.incl`, `section_extra_post_data.incl`
   verbatim into `lib/rp2350/memlcoresplit/` — addresses unchanged (§1 confirmed the Stamp XL's
   SRAM map matches the CMake project's target exactly).
3. Write `tools/memlcoresplit/splice_memmap.py`: reads `{build.path}/memmap_default.ld` (already
   placeholder-substituted by `prelink.1`), inserts the `MEMORY` block and appends the two
   `SECTIONS` blocks, writes back in place. Takes a flag so it's a no-op (copies the file
   unchanged) when invoked for any board other than `solderparty_rp2350_stamp_xl`, or when the new
   menu is at `Off`.
4. Add to `boards.txt`, scoped to `solderparty_rp2350_stamp_xl` only — a single menu drives both
   the linker reservation and the pinning define, exactly mirroring how one CMake cache variable
   (`MEML_MLP_RUNS_ON_CORE`, blank/`0`/`1`) drives both `pico_add_linker_script_override_path`'s
   inclusion and the compile define in the original build:
   ```
   menu.memlcoresplit=MEML Core Split
   solderparty_rp2350_stamp_xl.menu.memlcoresplit.off=Off (stock memory map)
   solderparty_rp2350_stamp_xl.menu.memlcoresplit.off.build.memlcoresplit=0
   solderparty_rp2350_stamp_xl.menu.memlcoresplit.core0=Core 0 (MLP pinned to core 0's bank)
   solderparty_rp2350_stamp_xl.menu.memlcoresplit.core0.build.memlcoresplit=1
   solderparty_rp2350_stamp_xl.menu.memlcoresplit.core0.build.extra_flags=-DMEML_MLP_RUNS_ON_CORE=0
   solderparty_rp2350_stamp_xl.menu.memlcoresplit.core1=Core 1 (MLP pinned to core 1's bank)
   solderparty_rp2350_stamp_xl.menu.memlcoresplit.core1.build.memlcoresplit=1
   solderparty_rp2350_stamp_xl.menu.memlcoresplit.core1.build.extra_flags=-DMEML_MLP_RUNS_ON_CORE=1
   ```
   `build.memlcoresplit` (0 or 1) only tells the splice hook whether to reserve `CORE1_RAM` at all
   — on for both `Core 0` and `Core 1`, same as the original CMake build reserving the bank
   regardless of which value `MEML_MLP_RUNS_ON_CORE` takes. `MEML_MLP_RUNS_ON_CORE` (0 or 1) is the
   one define a sketch/library actually binds its placement macros to; a given build produces
   exactly one pinning, never both.
5. Add `recipe.hooks.linking.prelink.3.pattern` to `platform.txt`, invoking `splice_memmap.py`
   with `{build.memlcoresplit}` (defaults to `0` for every board that doesn't define the menu, via
   platform.txt's own fallback value).

**Test milestone:**
- Build a trivial blank sketch for `solderparty_rp2350_stamp_xl` with the menu at **Off**; diff the
  resulting `.elf` section table against a build from unmodified upstream `arduino-pico` at the
  same commit — must be byte-for-byte identical modulo build timestamps/paths embedded by the
  toolchain. (Equivalent to the CMake project's `check_scheme_absent`.)
- Build the same blank sketch with the menu at **Core 0** and separately at **Core 1**; in both
  cases `readelf -SW` confirms `CORE1_RAM` exists at `0x20040000`/`256K`, and
  `.core1_bank`/`.core0_bank`/`.core1_stack` exist and are empty (size 0 — nothing has claimed them
  yet; the two menu values only differ in which define a later TU sees, not in the reservation
  itself).
- Build a **different** Stamp-XL-adjacent board (e.g. `rpipico2`) with no menu changes at all;
  confirm its `.elf` is unaffected (proves the splice hook's no-op branch actually fires for boards
  that don't carry the new menu).
- Deliberately corrupt the splice step (temporarily make it run twice) and confirm the build
  **fails** via a duplicate-`MEMORY`-region linker error, rather than silently double-splicing —
  this is the regression test for Phase 1's own idempotency, written now while the failure mode is
  easy to reproduce on demand.

---

## 5. Phase 2 — Placement macros (companion library)

**Goal:** the `__attribute__((section(...)))` macros exist, are independent of any one host
library's naming, and are provably immune to the COMDAT-folding hazard confirmed applicable in §1.

Same semantics as the CMake project, deliberately: `MEML_MLP_CODE`/`MEML_MLP_DATA` resolve through
a *single* global `MEML_MLP_RUNS_ON_CORE` define (now set by the Phase 1 boards.txt menu instead of
a CMake cache variable), and one build targets exactly one pinning — `mlp` never exists on both
banks at once. `MEMLCoreSplit.h` exposes the underlying per-site macros
(`MEML_RUNS_ON_CORE_CODE(n)`, `MEML_DATA_ON_CORE(n)`, `MEML_DATA_ON_FLASH`) as building blocks, with
`MEML_MLP_CODE`/`MEML_MLP_DATA` as the convenience layer every consuming library (and the §9
example) actually uses — a direct, line-for-line port of `MemoryDefs.hpp`'s existing contract, so
`mlp`'s existing `Placement.h` (`SMLP_CODE_ATTR`/`SMLP_DATA_ATTR`) binds against it completely
unchanged.

Steps:
1. Port `MemoryDefs.hpp` into `src/MEMLCoreSplit.h`, keeping the `__COUNTER__` mechanism and its
   explanatory comment (the hazard is non-obvious and the comment is what prevents it from being
   "cleaned up" by a future contributor who doesn't know why it's there).
2. Keep the optional single-selector convenience block
   (`#if defined(MEML_MLP_RUNS_ON_CORE) ... #define MEML_MLP_CODE ...`) for drop-in compatibility
   with libraries (like `mlp`) already written against that contract.
3. Port the template-instantiation redirect mechanism (originally `for_each_layer`-specific) as a
   documented pattern rather than new infrastructure: `docs/porting-a-library.md` shows a library
   author how to add their own `tools/memlcoresplit/`-style wildcard fragment to the **fork**, for
   the rare templated-hot-path case a `section` attribute can't reach. Not built as a generic
   plugin/manifest system — nothing in this plan's actual deliverables needs more than one such
   redirect (mlp's `for_each_layer`), and a speculative registry would be unused abstraction.

**Test milestone:**
- Unit test (host-side, plain `arm-none-eabi-gcc -c`, no board needed): two weak functions tagged
  `MEML_RUNS_ON_CORE_CODE(1)` in the same TU produce two distinct `.time_critical.core1.code.N`
  sections (`nm`/`readelf` the `.o` — confirms `__COUNTER__` fired twice, not once).
- Same check **across two TUs** compiled into one archive — confirms no COMDAT collision between
  translation units (this is the actual historical bug class the macro exists to prevent; it fails
  silently rather than at link time if the fix regresses, so the test must inspect symbol survival,
  not just "the link succeeded").
- A fixture sketch with one `MEML_DATA_ON_CORE(0)` object and one `MEML_DATA_ON_CORE(1)` object in
  the same TU builds (Phase 1's menu On) and `readelf` confirms each lands in its respective bank.

---

## 6. Phase 3 — Preinit bootstrap

**Goal:** port `CoreSectionInit.cpp`'s copy/zero bootstrap, verify its ordering guarantee holds
under Arduino-Pico's boot sequence specifically (not just "should, by the §1 finding").

Steps:
1. Port `meml_core1_preinit()` into `src/MEMLCoreSplitInit.cpp` essentially unchanged — same
   symbols (`__core1_code_start__` etc.), same `.preinit_array.12000` registration.
2. Make the `for_each_layer`-style conditional block (the optional second copy loop) a
   compile-time-absent block unless the consuming sketch defines a library-provided marker, so a
   sketch using *only* `MEML_DATA_ON_CORE`/`MEML_RUNS_ON_CORE_CODE` (no templated-hot-path redirect)
   never links against symbols that don't exist for it — same guard structure as the original,
   generalized past the one `mlp`-specific name.

**Test milestone:**
- Hardware test on the actual Stamp XL: a static object with a non-trivial constructor (writes a
  sentinel value, say `0xA5A5A5A5`, into a field) placed via `MEML_DATA_ON_CORE(1)`; after boot,
  before `setup1()` ever runs, confirm the sentinel is intact (proves the preinit zero-pass ran
  *before* the constructor, not after — reordering this is the exact failure mode the original
  project's comments warn about).
- A pinned core-1 function's output is bit-identical to an unpinned twin compiled into the default
  bank (placement must not change behavior, only address).
- Soft-reset / watchdog-reboot test: trigger `rp2040.reboot()` or a watchdog reset from within
  `loop()`, confirm the preinit bootstrap re-runs cleanly on the subsequent boot with no corruption
  — the original CMake project's own test suite never exercised a warm reboot, so this is new
  coverage, not a straight port.

---

## 7. Phase 4 — Build-time validator

**Goal:** a sketch that misuses the placement macros fails the build, the same safety net
`validate_memory_placement.py` gives the CMake project.

Steps:
1. Port `validate_memory_placement.py` into `tools/memlcoresplit/validate_corebanks.py`, stripped
   of the `mlp`/`MLPOpticalRecognition`-specific checks (`check_mlp_symbol_placement`,
   `check_mlp_entry_points`, `check_benchmark_placement`, the inlining canary) — those were
   specific to one library's internals and don't belong in board-level tooling. Keep the
   structural checks: bank-bounds (`check_bank_bounds`), flash-outside-RAM
   (`check_flash_outside_ram`), and the Off-build absence check (`check_scheme_absent`).
2. Add `recipe.hooks.objcopy.postobjcopy.1.pattern` in the fork's `platform.txt`, invoking it
   against `{build.path}/{build.project_name}.elf` unconditionally; it reads `{build.memlcoresplit}`
   itself rather than needing a separate CLI flag plumbed through the recipe.
3. Document, in `docs/porting-a-library.md`, how a library adds its *own* symbol-prefix checks on
   top of the generic ones — as a small Python snippet the library ships in its own `examples/` or
   `extras/` and the user wires in manually (CI-style), not as something the fork's generic
   validator special-cases per library. This keeps the board-level validator genuinely
   library-agnostic, matching the "any library" framing in §2's layout.

**Test milestone:**
- A sketch whose pinned static data exceeds 256 KiB in one bank fails the build via the `ASSERT`
  inside the spliced linker fragment (link-time failure) — confirm the error message names the
  overflowing section, not a bare linker OOM.
- A sketch with a bank mismatch the linker *can't* catch on its own (e.g. a `.core1.bank`-tagged
  object that's fine size-wise but the sketch author forgot the menu was `Off`, so the section
  doesn't exist at all) — confirm `validate_corebanks.py` fails with a clear message rather than
  the object silently falling through to the default bank unflagged.
- A clean sketch (menu at `Core 0` or `Core 1`, correctly sized, correctly tagged) passes with exit
  code 0 and the `PASS` message on stdout, visible in `arduino-cli compile --verbose` output.

---

## 8. Phase 5 — Packaging and distribution

**Goal:** both repos are installable the way any third-party Arduino board/library is — Boards
Manager URL and Library Manager (or `.zip` import) — not just buildable from a cloned working tree.

Steps:
1. Fork repo: generate a `package_musicallyembodiedml_rp2040_index.json` pointing at a tagged
   release archive of the fork (mirroring the structure of the installed
   `package_rp2040_index.json`), hosted via GitHub Releases + Pages (or raw.githubusercontent.com
   for the index itself).
2. Companion library repo: `library.properties` with `architectures=rp2040` (so Library Manager
   correctly flags it as RP2040/RP2350-only), tagged releases so it's installable via Library
   Manager once submitted, and directly via "Add .ZIP Library" before that submission lands.
3. A `README.md` at each repo root stating the dependency direction explicitly: the library works
   without the fork (macros compile to nothing without the reserved bank, same as `mlp`'s
   `Placement.h` degrading gracefully today), but needs the fork's board with the menu set to
   **Core 0** or **Core 1** for the placement to be real rather than a no-op.

**Test milestone:**
- A clean machine (or CI runner) with no prior Arduino state: add the fork's Boards Manager URL,
  install the board, install the library via `.zip`, compile the Phase 6 example for
  `solderparty_rp2350_stamp_xl` with the menu set to **Core 1**, end to end, with no manual file
  copying into `~/Library/Arduino15` or equivalent.
- The same machine, menu left at its default (**Off**): the example sketch still compiles
  (degrades to "no placement", matching the README's stated contract) rather than failing to find
  symbols.

---

## 9. Phase 6 — Example: XOR, core selected at build time

**Goal:** the flagship demonstration from the brief — `mlp` learning XOR, pinned to whichever core
the Arduino **Tools → MEML Core Split** menu selects at build time, with that pinning verified
independent under simultaneous load on the other core. One build, one pinning — `mlp` is compiled
into exactly one bank, never both, matching exactly what `MLPTestPicoSDK`'s own benchmark builds
and tests today.

### 6.1 Design

A single translation unit, binding `mlp`'s existing `Placement.h` contract through the one global
`MEML_MLP_RUNS_ON_CORE` define the Phase 1 boards.txt menu sets (§4) — there is no second,
alternately-pinned copy anywhere in the image:

```
XORNet.cpp:  #include <MEMLCoreSplit.h>   // binds SMLP_CODE_ATTR/SMLP_DATA_ATTR to
             #include <mlp.h>             // MEML_MLP_RUNS_ON_CORE, as set by the Tools menu
             class XORNet { /* smlp::StaticMLP<float, Layout<2,4,1>, ...> member */ };
```

Which physical core actually executes `XORNet`'s training/inference calls is a **separate,
ordinary runtime fact** — same as in `MLPTestPicoSDK`: whichever core's `main()`/`main1()` calls
into it. The sketch:
- defines `setup1()`/`loop1()` only when the menu selected **Core 1**, running `XORNet` from
  there and leaving core 0 running a trivial RAM-touching busy task (reusing the spirit of
  `RAMFlooder.hpp`) so the demo still shows the *other* core active and non-interfering;
- otherwise (menu at **Core 0**) runs `XORNet` directly from `loop()`, and instead launches the
  busy task on core 1 via `multicore_launch_core1`.

This mirrors the CMake project's own `main.cpp` exactly: which branch compiles in is decided by
the same build-time define that decided the linker placement, not by anything read at runtime.

The XOR dataset needs no storage — four hardcoded `{0,0}→0, {0,1}→1, {1,0}→1, {1,1}→0` samples —
which is precisely why the brief asks for XOR rather than the `MLPOpticalRecognition`
digit-recognition task (that one depends on a flash-resident dataset irrelevant to this port).

### 6.2 Steps

1. Implement `XORNet` as described, `MEML_MLP_CODE`/`MEML_MLP_DATA`-tagged through `mlp`'s
   existing, unmodified `Placement.h`.
2. Implement the build-time core dispatch (`#if MEML_MLP_RUNS_ON_CORE == 1` selecting which of
   `setup1()`/`loop1()` vs. `loop()` carries `XORNet`, the other carrying the busy task) in
   `XOR_EitherCore.ino`. This exercises Phase 0's finding directly: if bare placement worked, no
   extra tagging is needed on `setup1`/`loop1` themselves beyond what's already on `XORNet`'s
   methods; if not, this is where the documented explicit-wrapper fallback gets used for real.
3. Print training loss/epoch and final accuracy over Serial, so a build with the menu at **Core 0**
   and a separate build with the menu at **Core 1** are observably running the same learning task
   to completion, not just structurally symmetric on paper.

### Test milestone (acceptance test for the whole package — this is the plan's D1/D2 equivalent):

- **Correctness:** a **Core 0** build and a separate **Core 1** build converge on XOR (final
  accuracy 4/4 on the fixed truth table) and produce bit-identical loss trajectories to each other
  given the same fixed weight-init seed — proves the pinning changes *only* placement, not
  behavior, exactly mirroring the original project's own "bit-identical loss/accuracy"
  correctness-independence check.
- **Independence under load:** for each of the two builds, compare per-epoch training time with
  the other core running the busy task vs. fully idle (`WFE`) — timing must not differ by more
  than an agreed tolerance (port the CMake project's `kTimingToleranceRatio = 0.01f` as the
  starting figure), on real Stamp XL hardware. This is the test that actually demonstrates the
  stated goal of the entire exercise — contention-free concurrent execution — not merely that the
  build succeeds.
- **Build-time selection correctness:** build both menu values and, for each, confirm via
  `readelf`/`nm` that `XORNet`'s symbols land in the selected bank only — the *other* bank's
  `.core{0,1}_bank` section is present (Phase 1 reserves it regardless of which core is selected)
  but contains nothing belonging to `XORNet`, confirming the unselected core's bank is genuinely
  empty rather than silently also holding a stray copy.
- **Documentation parity:** `docs/README.md`'s quick-start, followed verbatim on a clean machine
  (Phase 5's milestone machine), reproduces this example's output with no steps missing, for
  whichever menu value the quick-start tells the reader to pick.

---

## 10. Documentation deliverables (tracked here, not deferred to "later")

- `MEMLCoreSplit/docs/README.md` — quick start: install board + library, open the XOR example,
  select the Stamp XL board with `MEML Core Split: Core 0` (or `Core 1`), flash, observe Serial
  output.
- `MEMLCoreSplit/docs/porting-a-library.md` — written for a library author (like `mlp`'s
  maintainer) who wants their library's hot path pinnable: the `Placement.h`-style contract, an
  explicit statement that one build pins to exactly one core (never both), and how to add a
  templated-hot-path redirect fragment to the fork if `-ffunction-sections` alone isn't enough.
- `MEMLCoreSplit/docs/internals.md` — written for whoever maintains the fork: what the splice hook
  does, why the COMDAT `__COUNTER__` trick is load-bearing, why `memmap_default.ld` itself stays
  untouched, and the rebase procedure against a new upstream `arduino-pico` release.
- Each phase's test milestone (§3–§9) *is* the test plan — no separate document restates them.

---

## 11. Risk register

| Risk | Mitigation |
|---|---|
| Phase 0 finds `setup1`/`loop1` don't inherit a pre-declared `section` attribute under `weak` | §6.2 already has the explicit-tagging fallback path; §0 states this is a known open question, not a promise. |
| Upstream `arduino-pico` releases change `memmap_default.ld`'s structure (section names, order) | Splice hook (not a hand-edited copy) means most upstream changes just re-splice cleanly; `internals.md`'s rebase procedure documents the manual-diff fallback when they don't. |
| `mlp`'s `for_each_layer`-style templated hot path needs the wildcard-redirect trick for the XOR net too | Confirm during Phase 6 implementation whether `StaticMLP<float, Layout<2,4,1>, ...>`'s small size actually hits this (it may not — the original issue arose with the larger 64-64-32-10 network); only build the redirect fragment if the §7 validator actually catches a misplaced symbol. |
| Boards Manager JSON hosting (Phase 5) has no CI gate today | Add a GitHub Actions workflow in the fork that builds the Phase 6 example on every push, failing loudly if packaging drifts from what Arduino CLI can actually install. |
