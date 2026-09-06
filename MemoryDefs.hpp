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

/* The linker overrides in linker/*.incl expect these exact section names.
 * They are used to pin the corresponding functions/data into SRAM bank 0-3 for core 0
 * and SRAM bank 4-7 for core 1 on RP2350.
 */
#define MEML_RUNS_ON_CORE(n) \
    __attribute__((section(".time_critical.core" MEML_STR(n) ".code"), used, noinline, optimize("O2")))

#define MEML_DATA_ON_CORE(n) \
    __attribute__((section(".core" MEML_STR(n) ".bank"), used, aligned(8)))

#define MEML_DATA_ON_FLASH \
    __attribute__((section(".flash"), used, aligned(8)))

#if defined(MEML_MLP_RUNS_ON_CORE)
#if MEML_MLP_RUNS_ON_CORE == 0
#define MEML_MLP_CODE MEML_RUNS_ON_CORE(0)
#define MEML_MLP_DATA MEML_DATA_ON_CORE(0)
#elif MEML_MLP_RUNS_ON_CORE == 1
#define MEML_MLP_CODE MEML_RUNS_ON_CORE(1)
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
