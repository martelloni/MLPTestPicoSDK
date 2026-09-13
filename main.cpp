#include <stdio.h>
#include <cmath>
#include <algorithm>
#include <array>
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
// builds the benchmark suite instead. Step 7: the pinned configurations
// (MEML_MLP_RUNS_ON_CORE=0/1) run TestRAMIndependence against Step 1's real
// per-core placement scheme; the unpinned configuration runs
// TestNaiveMemoryLayout, the honest counterfactual with no placement scheme
// at all. Both share the same public surface, so the rest of this file uses
// them through one alias rather than duplicating the comparison logic.
#if defined(MEML_ENABLE_RAM_INDEPENDENCE_BENCHMARK)
#if defined(MEML_MLP_RUNS_ON_CORE)
#include "tests/TestRAMIndependence.hpp"
using BenchmarkTest = test::TestRAMIndependence;
#else
#include "tests/TestNaiveMemoryLayout.hpp"
using BenchmarkTest = test::TestNaiveMemoryLayout;
#endif
#endif

int main()
{
    stdio_init_all();

    static constexpr uint32_t clock_frequency_hz = 150000000u; // 125 MHz
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
    printf("Running RAM independence test with clock frequency: %u MHz\n\n", clock_frequency_hz / 1000000u);

    // Run the real training workload twice with identical parameters, once
    // against a RAM-flooding other core and once against a fully dormant
    // (WFE-blocked) one, to verify the two cores' RAM banks are genuinely
    // independent: the MLP core's computed results must not change at all,
    // and its timing must not change by more than an agreed tolerance.
    constexpr uint32_t kTrainingRepeats = 10u;
    constexpr uint32_t kEpochsPerSession = 10u;
    constexpr float kTimingToleranceRatio = 0.01f;

    // TestRAMIndependence's training results/iteration-cap accounting are
    // static (mirroring TestBase's own static CoreCallback-driven state), so
    // a second construction/RunTest() overwrites the first's before it is
    // read. Snapshot each run's data into locals immediately after RunTest()
    // returns, before constructing the next run.
    struct RunSnapshot {
        std::array<MLPOpticalRecognition::Result, BenchmarkTest::kMaxTrainingRepeats> results;
        uint32_t result_count;
        test::TestBase::TestResultsPerCore mlp_timing;
        test::TestBase::TestResultsPerCore other_timing;
        uint32_t other_iteration_cap;
        uint64_t other_iterations_consumed_until_done;
        float other_fill_time_us_mean;
        float other_fill_time_us_max;
    };

    BenchmarkTest flooding_test(kTrainingRepeats, kEpochsPerSession, /*runOtherCoreTask=*/true);
    flooding_test.RunTest();
    RunSnapshot flood{};
    flood.results = flooding_test.GetTrainingResults();
    flood.result_count = flooding_test.GetTrainingResultCount();
// The MLP experiment runs on TestBase's core-0 slot for both the unpinned
// build (TestNaiveMemoryLayout's fixed assignment, since there is no
// selector to key off) and the MEML_MLP_RUNS_ON_CORE=0 pinned build; only
// the =1 pinned build puts it on core 1.
#if !defined(MEML_MLP_RUNS_ON_CORE) || MEML_MLP_RUNS_ON_CORE == 0
    flood.mlp_timing = flooding_test.GetResults().core0;
    flood.other_timing = flooding_test.GetResults().core1;
#else
    flood.mlp_timing = flooding_test.GetResults().core1;
    flood.other_timing = flooding_test.GetResults().core0;
#endif
    flood.other_iteration_cap = flooding_test.GetOtherCoreIterationCap();
    flood.other_iterations_consumed_until_done = flooding_test.GetOtherCoreIterationsConsumedUntilDone();
    flood.other_fill_time_us_mean = flooding_test.GetOtherCoreFillTimeUsMean();
    flood.other_fill_time_us_max = flooding_test.GetOtherCoreFillTimeUsMax();

    BenchmarkTest dormant_test(kTrainingRepeats, kEpochsPerSession, /*runOtherCoreTask=*/false);
    dormant_test.RunTest();
    RunSnapshot dormant{};
    dormant.results = dormant_test.GetTrainingResults();
    dormant.result_count = dormant_test.GetTrainingResultCount();
#if !defined(MEML_MLP_RUNS_ON_CORE) || MEML_MLP_RUNS_ON_CORE == 0
    dormant.mlp_timing = dormant_test.GetResults().core0;
    dormant.other_timing = dormant_test.GetResults().core1;
#else
    dormant.mlp_timing = dormant_test.GetResults().core1;
    dormant.other_timing = dormant_test.GetResults().core0;
#endif
    dormant.other_iteration_cap = dormant_test.GetOtherCoreIterationCap();
    dormant.other_iterations_consumed_until_done = dormant_test.GetOtherCoreIterationsConsumedUntilDone();

    printf("Flooding run (other core flooding RAM):\n");
    for (uint32_t i = 0; i < flood.result_count; ++i) {
        printf("  session %u: loss=%f accuracy=%f\n", i,
               static_cast<double>(flood.results[i].loss),
               static_cast<double>(flood.results[i].accuracy));
    }
    printf("  MLP core: avg_time=%.2f us, max_time=%.2f us\n",
           flood.mlp_timing.time_us_avg, flood.mlp_timing.time_us_max);
    // Printed in exponential notation, not as a raw decimal: the flood loop
    // is genuinely uncapped (see OtherCoreFloodTask), so the actual count
    // depends on FillOnce()'s real hardware speed and can be large enough
    // that %llu would be an unwieldy wall of digits even though it still fits
    // in the uint64_t counter itself.
    printf("  other core: flood iterations consumed until done=%.6e\n",
           static_cast<double>(flood.other_iterations_consumed_until_done));
    printf("  other core: fill_time avg=%.2f us, max=%.2f us\n\n",
           static_cast<double>(flood.other_fill_time_us_mean),
           static_cast<double>(flood.other_fill_time_us_max));

    printf("Dormant run (other core fully idle in WFE):\n");
    for (uint32_t i = 0; i < dormant.result_count; ++i) {
        printf("  session %u: loss=%f accuracy=%f\n", i,
               static_cast<double>(dormant.results[i].loss),
               static_cast<double>(dormant.results[i].accuracy));
    }
    printf("  MLP core: avg_time=%.2f us, max_time=%.2f us\n",
           dormant.mlp_timing.time_us_avg, dormant.mlp_timing.time_us_max);
    printf("  other core: iterations=%u (cap=%u, expected 1)\n\n",
           dormant.other_timing.iterations, dormant.other_iteration_cap);

    // Correctness independence: RAM-bank independence means core-1 memory
    // traffic must not perturb core-0's (or vice versa) computed results at
    // all -- bit-identical, not just approximately equal.
    bool results_identical = flood.result_count == dormant.result_count;
    for (uint32_t i = 0; results_identical && i < flood.result_count; ++i) {
        const auto & a = flood.results[i];
        const auto & b = dormant.results[i];
        if (a.loss != b.loss || a.accuracy != b.accuracy) {
            results_identical = false;
        }
    }

    // Timing independence: the MLP core's average/max timing must not differ
    // by more than the agreed tolerance between the two other-core conditions.
    const float avg_diff_ratio = std::fabs(flood.mlp_timing.time_us_avg - dormant.mlp_timing.time_us_avg) /
        std::max(dormant.mlp_timing.time_us_avg, 1.0f);
    const float max_diff_ratio = std::fabs(flood.mlp_timing.time_us_max - dormant.mlp_timing.time_us_max) /
        std::max(dormant.mlp_timing.time_us_max, 1.0f);
    const bool timing_independent = avg_diff_ratio <= kTimingToleranceRatio && max_diff_ratio <= kTimingToleranceRatio;

    printf("Correctness independence (bit-identical loss/accuracy): %s\n", results_identical ? "PASS" : "FAIL");
#if defined(MEML_MLP_RUNS_ON_CORE)
    printf("Timing independence (avg_diff=%.2f%%, max_diff=%.2f%%, tolerance=%.0f%%): %s\n",
           static_cast<double>(avg_diff_ratio * 100.0f), static_cast<double>(max_diff_ratio * 100.0f),
           static_cast<double>(kTimingToleranceRatio * 100.0f), timing_independent ? "PASS" : "FAIL");
#else
    // Unpinned/naive layout: a timing deviation beyond kTimingToleranceRatio
    // here is the EXPECTED, DESIRED outcome (this build has no per-core
    // placement scheme at all), so this is reported as a finding, not gated
    // as PASS/FAIL the way the pinned builds' comparison is.
    (void)timing_independent;
    printf("Timing interference observed without per-core placement (expected): avg_diff=%.2f%%, max_diff=%.2f%%, tolerance=%.0f%%\n",
           static_cast<double>(avg_diff_ratio * 100.0f), static_cast<double>(max_diff_ratio * 100.0f),
           static_cast<double>(kTimingToleranceRatio * 100.0f));
#endif

    printf("\nTest completed.\n");
#endif

    while (true) {
        tight_loop_contents();
    }

    return 0;
}
