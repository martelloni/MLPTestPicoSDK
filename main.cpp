#include <stdio.h>
#include "MemoryDefs.hpp"
#include "pico/stdlib.h"
#include "hardware/clocks.h"
#include "hardware/vreg.h"

// Set by CMake's RUN_TESTS_OR_BENCHMARKS="tests": builds and runs the unit-test
// suite. Mutually exclusive with MEML_ENABLE_RAM_INDEPENDENCE_BENCHMARK below --
// this benchmark's own hardcoded core-0 network (~82 KiB via MEML_DATA_ON_CORE(0))
// and MLPOpticalRecognition's real 64-64-32-10 experiment state (~98 KiB, used by
// the unit-test suite from Step 4 onward) together overflow the fixed 256 KiB
// copy_to_ram RAM window -- a window whose size does not depend on
// MEML_MLP_RUNS_ON_CORE, so this conflict is not specific to any one selector
// value. It was previously also a problem specifically for MEML_MLP_RUNS_ON_CORE
// == 1, since the shared mlp/ hot-path code hook (SMLP_CODE_ATTR) would route
// this core-0-only network's code into the core-1 bank while its state stayed on
// core 0; that reason still applies whenever the benchmark shares a build with a
// core-1-pinned MLP.
#if defined(MEML_RUN_UNIT_TESTS)
#include "tests/unit/UnitTestRunner.hpp"
#endif

// Set by CMake's RUN_TESTS_OR_BENCHMARKS="benchmarks": skips unit tests and
// builds the benchmark suite (currently TestRAMIndependence only) instead.
#if defined(MEML_ENABLE_RAM_INDEPENDENCE_BENCHMARK)
#include "tests/TestRAMIndependence.hpp"
#endif

int main()
{
    stdio_init_all();

    static constexpr uint32_t clock_frequency_hz = 125000000u; // 125 MHz
    set_sys_clock_khz(clock_frequency_hz / 1000, true);

    while (stdio_usb_connected() == false) {
        tight_loop_contents();
    }
    printf("Press any key to continue...\n");
    while (getchar_timeout_us((int64_t)-1) == PICO_ERROR_TIMEOUT) {
        tight_loop_contents();
    }

#if defined(MEML_RUN_UNIT_TESTS)
    const bool unit_suite_passed = test::unit::RunAllOnSelectedCore();
    printf("Microunit suite result: %s\n", unit_suite_passed ? "PASS" : "FAIL");
    if (!unit_suite_passed) {
        printf("Microunit suite failed; halting before benchmarks. SAD TROMBONE!\n");
        while (true) {
            tight_loop_contents();
        }
    }
#endif

#if defined(MEML_ENABLE_RAM_INDEPENDENCE_BENCHMARK)
    printf("Press any key to start benchmarks...\n");
    while (stdio_usb_connected() == false) {
        tight_loop_contents();
    }
    while (getchar_timeout_us((int64_t)-1) == PICO_ERROR_TIMEOUT) {
        tight_loop_contents();
    }
    printf("\nStarting RAM independence test...\n");

    test::TestRAMIndependence test(clock_frequency_hz);

    printf("Running RAM independence test with clock frequency: %u MHz\n\n", clock_frequency_hz / 1000000u);

    test.RunTest();

    printf("Core 0: iterations=%u, avg_time=%.2f us, max_time=%.2f us, min_time=%.2f us\n",
           test.GetResults().core0.iterations,
           test.GetResults().core0.time_us_avg,
           test.GetResults().core0.time_us_max,
           test.GetResults().core0.time_us_min);
    printf("Core 1: iterations=%u, avg_time=%.2f us, max_time=%.2f us, min_time=%.2f us\n",
           test.GetResults().core1.iterations,
           test.GetResults().core1.time_us_avg,
           test.GetResults().core1.time_us_max,
           test.GetResults().core1.time_us_min);

    printf("\nTest completed.\n");
#endif

    while (true) {
        tight_loop_contents();
    }

    return 0;
}
