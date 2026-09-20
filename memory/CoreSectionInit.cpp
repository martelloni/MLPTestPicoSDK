/**
 * @file CoreSectionInit.cpp
 * @brief Initializes project-defined core-local SRAM sections before C++ constructors.
 */

#include <cstdint>

extern "C" {
extern uint32_t __core1_code_start__[];
extern uint32_t __core1_code_end__[];
extern const uint32_t __core1_code_load__[];
extern uint32_t __core1_bank_start__[];
extern uint32_t __core1_bank_end__[];
extern uint32_t __core0_bank_start__[];
extern uint32_t __core0_bank_end__[];
#if defined(MEML_MLP_RUNS_ON_CORE) && MEML_MLP_RUNS_ON_CORE == 1
// Only defined by linker/core1-only/section_copy_to_ram_text.incl, which is
// only added to the link for the core-1-pinned build (see CMakeLists.txt) --
// referencing these unconditionally would fail to link for core 0.
extern uint32_t __core1_for_each_layer_start__[];
extern uint32_t __core1_for_each_layer_end__[];
extern const uint32_t __core1_for_each_layer_load__[];
#endif
}

/**
 * @brief Copies core-1 code and clears both per-core mutable-state banks before
 * constructors run.
 *
 * The Pico SDK CRT invokes this after its normal RAM copies and before the
 * ordinary C++ init array. Its priority deliberately precedes the Pico SDK's
 * per-core-initializer marker, so Core 1 does not clear configured work data.
 *
 * Both .core1.bank and .core0.bank are custom NOLOAD sections placed after the
 * SDK's own .bss output section, so neither crt0's fixed data_cpy_table nor its
 * __bss_start__/__bss_end__ clear loop ever touches them; this routine is the
 * only thing that zeroes either bank before their objects' constructors run.
 */
extern "C" void meml_core1_preinit() {
    uint32_t *destination = __core1_code_start__;
    const uint32_t *source = __core1_code_load__;

    // Copy the flash-backed core-1 instructions to their execution address.
    while (destination != __core1_code_end__) {
        *destination++ = *source++;
    }

#if defined(MEML_MLP_RUNS_ON_CORE) && MEML_MLP_RUNS_ON_CORE == 1
    destination = __core1_for_each_layer_start__;
    source = __core1_for_each_layer_load__;

    // Same as above, for mlp/StaticMLP.h's for_each_layer<F, I> instantiations
    // (see linker/core1-only/section_copy_to_ram_text.incl): its VMA
    // (CORE1_RAM) differs from its LMA (flash), so the SDK's own startup copy
    // never touches it either.
    while (destination != __core1_for_each_layer_end__) {
        *destination++ = *source++;
    }
#endif

    destination = __core1_bank_start__;

    // Clear all core-1 mutable state before its C++ constructors execute.
    while (destination != __core1_bank_end__) {
        *destination++ = 0u;
    }

    destination = __core0_bank_start__;

    // Clear all core-0 mutable state before its C++ constructors execute.
    while (destination != __core0_bank_end__) {
        *destination++ = 0u;
    }
}

/**
 * @brief Registers the core-0-only section bootstrap with the Pico SDK CRT.
 *
 * The Pico SDK runs entries at and after .preinit_array.YYYYY on Core 1.
 * Keep this bootstrap before that marker because it clears Core-1 mutable RAM.
 */
extern "C" void (*const meml_core1_preinit_entry)(void)
    __attribute__((used, section(".preinit_array.12000"))) = meml_core1_preinit;
