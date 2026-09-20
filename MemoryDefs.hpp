/**
 * @file MemoryDefs.hpp
 * @author Andrea Martelloni
 * @brief Definitions for memory management and multi-core cooperation on RP2350
 *
 */

#ifndef __MEMORY_DEFS_HPP__
#define __MEMORY_DEFS_HPP__

#include "mlp/Placement.h"

#define MEML_STR2(x) #x
#define MEML_STR(x)  MEML_STR2(x)

/* The linker overrides in linker/*.incl expect section names starting with these
 * exact prefixes (matched via a wildcard, not an exact string) to pin the
 * corresponding functions/data into SRAM bank 0-3 for core 0 and SRAM bank 4-7
 * for core 1 on RP2350.
 *
 * Every function tagged with one of these attributes gets its OWN section, via
 * __COUNTER__, rather than sharing one literal section name. GCC places a
 * weak/vague-linkage (template-instantiation) function into a COMDAT group
 * named after whichever function first claims a given *explicit* section
 * name; every other weak function subsequently assigned that identical
 * section string is folded into that SAME group instead of getting its own.
 * If a TU-unique function (e.g. one member function only ever instantiated
 * in a single translation unit) ends up sharing a section -- and hence a
 * COMDAT group -- with a commonly-instantiated one (e.g. the PRNG's
 * next_u32(), pulled in by every net's constructor), the linker's normal
 * COMDAT deduplication keeps only ONE translation unit's copy of the *whole
 * merged section* whenever another TU also defines that common function's
 * group. Every other, otherwise-unrelated function bundled into the
 * "losing" TU's copy is silently discarded with it -- including genuinely
 * unique code that is never duplicated anywhere. The call site survives
 * (the compiler still emits a `bl` to the now-undefined weak symbol), but
 * the linker resolves a branch to an unresolved weak symbol by turning it
 * into a no-op, which leaves the return-value register holding whatever it
 * last held (typically `this`) -- so calls silently misbehave instead of
 * failing to link. __COUNTER__ gives every attribute use site, and hence
 * every function, its own section/COMDAT group, so unrelated functions can
 * never be folded together like this; the shared ".time_criticalN.codeNNN"
 * prefix is preserved so the existing wildcard section rules keep matching.
 */
#define MEML_RUNS_ON_CORE(n) \
    __attribute__((section(".time_critical.core" MEML_STR(n) ".code." MEML_STR(__COUNTER__)), used))

/* Same section prefix as MEML_RUNS_ON_CORE(n), deliberately without `used`. This
 * is bound to mlp/'s hot-path member function templates (SMLP_CODE_ATTR), which
 * are ordinary functions reached through normal calls -- a real call site is
 * enough to keep them alive under --gc-sections. `used` on a class-template
 * member function instead forces GCC to eagerly instantiate it (and every
 * function it calls) the moment the enclosing template is instantiated, even
 * if it is never actually called; every untagged callee an instantiated-but-
 * unused method drags in then lands in the default bank instead of the
 * selected one. MEML_RUNS_ON_CORE(n) keeps `used` for its own use sites
 * (address-taken entry points registered as raw function pointers, where the
 * compiler cannot otherwise see the reference).
 *
 * Neither macro forces `noinline`/`optimize("O2")` any more. Placement now
 * follows the outermost tagged caller: a tagged callee inlined into a tagged
 * caller stays in that caller's bank via the section attribute already
 * carried through inlining, so the section still ends up right without
 * paying for an out-of-line call on every hot-path step. Forcing `noinline`
 * previously turned every per-weight helper (nn::fmul, activate, ...) into a
 * real function call inside the innermost training loop, which measured as
 * a ~1.7x slowdown against the unpinned build -- the call overhead, not
 * memory-bank placement, was responsible. The trade-off: an *untagged*
 * caller of mlp code now silently pulls that code into its own (wrong) bank
 * with no out-of-line symbol left for the validator to catch, so every
 * function that enters the MLP from outside must stay tagged with
 * MEML_MLP_CODE (see check_mlp_symbol_placement's entry-point requirement in
 * scripts/validate_memory_placement.py).
 */
#define MEML_RUNS_ON_CORE_CODE(n) \
    __attribute__((section(".time_critical.core" MEML_STR(n) ".code." MEML_STR(__COUNTER__))))

#define MEML_DATA_ON_CORE(n) \
    __attribute__((section(".core" MEML_STR(n) ".bank"), used, aligned(8)))

#define MEML_DATA_ON_FLASH \
    __attribute__((section(".flash"), used, aligned(8)))

#if defined(MEML_MLP_RUNS_ON_CORE)
#if MEML_MLP_RUNS_ON_CORE == 0
#define MEML_MLP_CODE MEML_RUNS_ON_CORE_CODE(0)
#define MEML_MLP_DATA MEML_DATA_ON_CORE(0)
#elif MEML_MLP_RUNS_ON_CORE == 1
#define MEML_MLP_CODE MEML_RUNS_ON_CORE_CODE(1)
#define MEML_MLP_DATA MEML_DATA_ON_CORE(1)
#else
#error "MEML_MLP_RUNS_ON_CORE must be 0 or 1 when defined."
#endif
#else
#define MEML_MLP_CODE
#define MEML_MLP_DATA
#endif

/* SMLP_CODE_ATTR_MULTI: blank for the unpinned build and the core-0-pinned
 * build; `always_inline` ONLY for the core-1-pinned build. Two things this
 * used to try both turned out to be wrong for the general case:
 *
 * 1. An explicit `section(...)` attribute on for_each_layer<F, I> would be
 *    attached to its ONE textual definition (mlp/StaticMLP.h), so a
 *    __COUNTER__-based name there is fixed once per translation unit: every
 *    lambda closure F and recursion depth I that instantiates the template
 *    -- and, worse, the SAME instantiation re-emitted (as a weak/COMDAT
 *    symbol) from a DIFFERENT TU that consumed a different number of
 *    __COUNTER__ tokens earlier in its own translation -- would collide on
 *    one literal section string, reintroducing the COMDAT-group-per-
 *    section-name folding hazard described above for MEML_RUNS_ON_CORE(n).
 *
 * 2. Forcing `always_inline` everywhere sidesteps that hazard, but measured
 *    ~4% slower on hardware than leaving the attribute blank -- confirmed by
 *    comparing disassembly across commits: with `always_inline`, EVERY call
 *    site gets its own full copy of for_each_layer's 3-layer-unrolled body;
 *    left blank, -ffunction-sections (on by default in this SDK) + ordinary
 *    vague linkage lets the compiler keep ONE small (~2.5 KiB) out-of-line
 *    copy per (F, I), reused via `bl` -- smaller footprint, measurably
 *    faster, and exactly what the unpinned build already did originally.
 *    Left blank, each instantiation gets its own auto-generated
 *    `.text.<mangled-name>` section, already uniquely keyed to that one
 *    mangled symbol -- ordinary, safe COMDAT grouping, no risk of two
 *    different instantiations colliding on one name, since none share one.
 *
 * That auto-named `.text._ZN...` section carries no bank hint, though, and
 * simply falls through the SDK's own default `.text*` catch-all
 * (section_default_text.incl) into the ordinary default-RAM window. For the
 * unpinned build that is fine by construction, and for a core-0-pinned
 * build it is ALSO already correct, for a non-obvious reason: core 0's bank
 * *is* that same default RAM region (0x20000000-0x2003ffff), so leaving
 * for_each_layer untagged there is a free, correct no-op -- verified by
 * diffing the resulting disassembly against the unpinned build byte-for-
 * byte (address-normalized). Core 1's bank (CORE1_RAM, 0x20040000+) is a
 * genuinely separate region, so the same trick doesn't reach it.
 *
 * A linker-side fix for core 1 (redirect the auto-named section into
 * CORE1_RAM before the SDK's default catch-all claims it) was investigated
 * and rejected: the only injection point earlier than that catch-all
 * (linker/memory_extra.incl, processed before sections_default_text.incl)
 * would also place the redirected section's flash LOAD address before
 * section_flash_begin.incl/section_boot2.incl's content -- i.e. before the
 * vector table and boot2 image header, which RP2350 requires at the start
 * of flash. That risks an unbootable image, so core 1 keeps `always_inline`
 * and pays the ~4% cost; the unpinned and core-0 builds do not.
 */
#if defined(MEML_MLP_RUNS_ON_CORE) && MEML_MLP_RUNS_ON_CORE == 1
#define MEML_MLP_CODE_MULTI __attribute__((always_inline))
#else
#define MEML_MLP_CODE_MULTI
#endif

#undef SMLP_CODE_ATTR
#undef SMLP_CODE_ATTR_MULTI
#undef SMLP_DATA_ATTR
#define SMLP_CODE_ATTR MEML_MLP_CODE
// See MEML_MLP_CODE_MULTI's comment above: blank for the unpinned and
// core-0-pinned builds (ordinary vague linkage, core 0's bank already is the
// default RAM window), `always_inline` only for the core-1-pinned build
// (CORE1_RAM can't be reached by the same free ride).
#define SMLP_CODE_ATTR_MULTI MEML_MLP_CODE_MULTI
#define SMLP_DATA_ATTR MEML_MLP_DATA

#endif // __MEMORY_DEFS_HPP__
