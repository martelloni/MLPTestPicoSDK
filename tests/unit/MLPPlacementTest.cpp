/**
 * @file MLPPlacementTest.cpp
 * @brief Proves the MEML_MLP_CODE / MEML_MLP_DATA placement guarantees.
 *
 * This is a placement probe, not the experiment: it exercises the single
 * selector-driven code/data hook end-to-end ahead of the real experiment
 * facade (Step 4). Once MLPOpticalRecognition::State exists, this fixture
 * should be retired or repurposed rather than left to consume reserved-bank
 * budget alongside it.
 */

#include <cstdint>
#include <cstring>

#include "pico.h"

#include "MemoryDefs.hpp"
#include "mlp/StaticMLP.h"
#include "tests/unit/microunit/microunit.h"

namespace {

using PlacementNet = smlp::StaticMLP<
    float,
    smlp::Layout<2, 2, 1>,
    smlp::Activations<
        ACTIVATION_FUNCTIONS::RELU,
        ACTIVATION_FUNCTIONS::LINEAR>>;

struct PlacementProbe {
    PlacementNet net;

    // Left untouched by the constructor: at first read it reflects whatever
    // meml_core1_preinit's manual NOLOAD-zeroing left behind, since the SDK's
    // ordinary .bss clear does not cover this custom section.
    uint8_t raw_probe[8];

    // Set only by this constructor: proves the constructor ran, and (since it
    // runs strictly after the preinit bootstrap) that it ran after zeroing.
    uint32_t constructed_sentinel;

    PlacementProbe() : net(), constructed_sentinel(0xC0FFEEu) {}
};

MEML_MLP_DATA PlacementProbe g_placement_probe{};

#if defined(MEML_MLP_RUNS_ON_CORE) && MEML_MLP_RUNS_ON_CORE == 1
constexpr uint32_t kExpectedCore = 1u;
constexpr uintptr_t kBankBase = 0x20040000u;
#else
constexpr uint32_t kExpectedCore = 0u;
constexpr uintptr_t kBankBase = 0x20000000u;
#endif
constexpr uintptr_t kBankEnd = kBankBase + 0x40000u;

} // namespace

// External, inline (vague) linkage: the mlp/ template methods SMLP_CODE_ATTR
// already places in this section are weak/COMDAT, and mixing a plain strong-
// linkage symbol into the same named section is a GCC "section type conflict"
// at link time; `inline` gives this function the same vague linkage.
MEML_MLP_CODE inline uint32_t PlacementProbeCodeAddress() {
    return reinterpret_cast<uint32_t>(&PlacementProbeCodeAddress);
}

UNIT(MLPPlacement) {
    // Constructor-time sentinel: the field it never touches must have been
    // zeroed before construction, and the field it does set must hold the
    // value only the constructor writes.
    uint8_t zero_probe[8] = {0};
    ASSERT_TRUE(std::memcmp(g_placement_probe.raw_probe, zero_probe, sizeof(zero_probe)) == 0);
    ASSERT_TRUE(g_placement_probe.constructed_sentinel == 0xC0FFEEu);

    // MEML_MLP_DATA state lives in the selected core's 256 KiB bank.
    const uintptr_t probe_address = reinterpret_cast<uintptr_t>(&g_placement_probe);
    ASSERT_TRUE(probe_address >= kBankBase && probe_address < kBankEnd);

    // MEML_MLP_CODE-tagged code lives in the same bank.
    const uintptr_t code_address = static_cast<uintptr_t>(PlacementProbeCodeAddress());
    ASSERT_TRUE(code_address >= kBankBase && code_address < kBankEnd);

    // Actually exercise the tagged hot path (forward/activate/...), not just
    // construct the net: an unexercised StaticMLP is optimized away entirely
    // by --gc-sections, so nothing would be left for the ELF-map validator to
    // check the placement of.
    float probe_input[2] = {0.0f, 0.0f};
    float probe_output[1] = {0.0f};
    g_placement_probe.net.GetOutput(probe_input, probe_output);

    // MLP-affine: whichever core the suite runs the registered tests on must
    // match the selector (the core-1 runner executes this test from inside
    // its core-1 trampoline; unpinned/core-0 run it directly on core 0).
    ASSERT_TRUE(static_cast<uint32_t>(get_core_num()) == kExpectedCore);

    PASS();
}
