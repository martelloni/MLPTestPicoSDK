/**
 * @file UnitTestRunner.cpp
 * @brief Runs the registered Pico microunit suite on the selected core.
 */

#include <cstddef>
#include <cstdint>

#include "MemoryDefs.hpp"
#include "pico/multicore.h"
#include "tests/unit/UnitTestRunner.hpp"
#include "tests/unit/microunit/microunit.h"

extern "C" {
extern uint32_t __core1_stack_bottom__[];
extern uint32_t __core1_stack_top__[];
}

namespace {
#if defined(MEML_MLP_RUNS_ON_CORE) && MEML_MLP_RUNS_ON_CORE == 1
static MEML_RUNS_ON_CORE(1) void Core1UnitTestEntry() {
    const uint32_t start_token = multicore_fifo_pop_blocking();
    (void)start_token;
    const bool passed = microunit::UnitTester::Run();
    multicore_fifo_push_blocking(passed ? 1u : 0u);
}
#endif
} // namespace

namespace test {
namespace unit {

bool RunAllOnSelectedCore() {
#if defined(MEML_MLP_RUNS_ON_CORE)
#if MEML_MLP_RUNS_ON_CORE == 1
    multicore_fifo_drain();
    const auto entry = reinterpret_cast<void (*)()>(
        reinterpret_cast<uintptr_t>(&Core1UnitTestEntry) | 1u);
    const std::size_t stack_size = reinterpret_cast<uintptr_t>(__core1_stack_top__) -
        reinterpret_cast<uintptr_t>(__core1_stack_bottom__);
    multicore_launch_core1_with_stack(entry, __core1_stack_bottom__, stack_size);
    multicore_fifo_push_blocking(0xC1C00001u);
    const uint32_t result = multicore_fifo_pop_blocking();
    multicore_reset_core1();
    return result != 0u;
#elif MEML_MLP_RUNS_ON_CORE == 0
    return microunit::UnitTester::Run();
#else
#error "MEML_MLP_RUNS_ON_CORE must be 0 or 1 when defined."
#endif
#else
    return microunit::UnitTester::Run();
#endif
}

} // namespace unit
} // namespace test
