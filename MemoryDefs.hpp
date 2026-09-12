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
    __attribute__((section(".time_critical.core" MEML_STR(n) ".code." MEML_STR(__COUNTER__)), used, noinline, optimize("O2")))

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
 */
#define MEML_RUNS_ON_CORE_CODE(n) \
    __attribute__((section(".time_critical.core" MEML_STR(n) ".code." MEML_STR(__COUNTER__)), noinline, optimize("O2")))

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

#undef SMLP_CODE_ATTR
#undef SMLP_DATA_ATTR
#define SMLP_CODE_ATTR MEML_MLP_CODE
#define SMLP_DATA_ATTR MEML_MLP_DATA

#endif // __MEMORY_DEFS_HPP__
