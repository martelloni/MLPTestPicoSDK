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

/* SMLP_CODE_ATTR_MULTI is bound to `always_inline` -- never to a `section`
 * attribute -- in every configuration, including the unpinned one. A
 * `section` attribute on for_each_layer<F, I> would be attached to its ONE
 * textual definition (mlp/StaticMLP.h), so __COUNTER__ there is evaluated
 * exactly once per translation unit: every lambda closure F and recursion
 * depth I that instantiates the template -- and, worse, the SAME
 * instantiation re-emitted (as a weak/COMDAT symbol) from a DIFFERENT TU
 * that happened to consume a different number of __COUNTER__ tokens earlier
 * in its own translation -- would collide on one literal section string.
 * That reintroduces exactly the COMDAT-group-per-section-name folding
 * hazard described above for MEML_RUNS_ON_CORE(n): the linker keeps only
 * one TU's copy of the group and silently no-ops every call to whichever
 * instantiations got discarded with it.
 *
 * `always_inline` sidesteps the problem instead of working around it:
 * for_each_layer<F, I> only ever calls for_each_layer<F, I+1> (a distinct
 * instantiation, not a real recursive call) until I+1 == kNumLayers, so it
 * is a finite, compile-time-unrolled call chain that GCC can always inline
 * in full. Forcing that (rather than leaving it to -O2/-O3 heuristics, which
 * may or may not keep an out-of-line copy) guarantees for_each_layer never
 * exists as a standalone symbol needing its own bank placement -- its
 * machine code always lands inside whichever SMLP_CODE_ATTR-tagged caller
 * pulled it in, in every one of the three memory configurations alike.
 */
#define MEML_MLP_CODE_MULTI __attribute__((always_inline))

#undef SMLP_CODE_ATTR
#undef SMLP_CODE_ATTR_MULTI
#undef SMLP_DATA_ATTR
#define SMLP_CODE_ATTR MEML_MLP_CODE
// See MEML_MLP_CODE_MULTI's comment above for why this is `always_inline`
// rather than a `section` attribute, in every configuration.
#define SMLP_CODE_ATTR_MULTI MEML_MLP_CODE_MULTI
#define SMLP_DATA_ATTR MEML_MLP_DATA

#endif // __MEMORY_DEFS_HPP__
