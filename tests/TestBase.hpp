/**
 * @file TestBase.hpp
 * @author Andrea Martelloni
 * @brief Base class to measure execution time of a test on RP2350 on both cores
 * @date 2026-08-16
 */

#ifndef __TEST_BASE_HPP__
#define __TEST_BASE_HPP__

#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <limits>

#include "pico/multicore.h"
#include "pico/time.h"
#include "MemoryDefs.hpp"

namespace test {

// The custom linker fragments that define these symbols are only injected
// for a pinned build (Step 7 of plan-mlpCoreLocalPrompt.md): the unpinned
// build gets the SDK's stock linker script with no .core1_stack section at
// all. TestBase still needs *some* valid core-1 stack to launch core 1 with,
// pinned scheme or not, so the unpinned build falls back to a plain,
// untagged file-scope array -- the SDK provides no ready-made symbol for a
// stack outside its own single-core-1-launch helpers, and this fallback
// itself is naive/default placement, not a gap to patch.
#if defined(MEML_MLP_RUNS_ON_CORE)
extern "C" {
extern uint32_t __core1_stack_bottom__[];
extern uint32_t __core1_stack_top__[];
}
#else
inline static uint32_t s_core1_stack_naive[1024]; // 4 KiB, matching the pinned build's .core1_stack size
#endif

class TestBase {

public:

    struct TestConfig {
        uint32_t core0_iterations;
        uint32_t core1_iterations;
    };

    struct TestResultsPerCore {
        uint32_t iterations;
        float time_us_avg;
        float time_us_max;
        float time_us_min;
    };

    struct TestResults {
        TestResultsPerCore core0;
        TestResultsPerCore core1;
    };

    using CoreCallback = void (*)();

    struct CoreWork {
        uint32_t iterations;
        CoreCallback init;
        CoreCallback task;
    };

    TestBase(TestConfig config) : config_(config) {
        results_.core0 = {0, 0.0f, 0.0f, std::numeric_limits<float>::max()};
        results_.core1 = {0, 0.0f, 0.0f, std::numeric_limits<float>::max()};
        scratch_core0_ = results_.core0;
        scratch_core1_ = results_.core1;
        core0_work_ = {config_.core0_iterations, nullptr, nullptr};
        core1_work_ = {config_.core1_iterations, nullptr, nullptr};
    }
    virtual ~TestBase() = default;

    TestResults GetResults() {
        return results_;
    }

    void ConfigureCoreWork(CoreCallback core0_init,
                          CoreCallback core0_task,
                          CoreCallback core1_init,
                          CoreCallback core1_task) {
        core0_work_.iterations = config_.core0_iterations;
        core0_work_.init = core0_init;
        core0_work_.task = core0_task;
        core1_work_.iterations = config_.core1_iterations;
        core1_work_.init = core1_init;
        core1_work_.task = core1_task;
    }

    void RunTest() {
        if (core0_work_.init == nullptr || core0_work_.task == nullptr ||
            core1_work_.init == nullptr || core1_work_.task == nullptr) {
            return;
        }

        multicore_fifo_drain();

        // Cortex-M Thumb entry points must have bit 0 set. The linker symbol itself is
        // emitted as an even address in the ELF, so we have to force the Thumb bit before
        // handing the function pointer to the ROM launcher.
        const auto core1_entry = reinterpret_cast<void (*)()>(reinterpret_cast<uintptr_t>(&Core1Entry) | 1u);
#if defined(MEML_MLP_RUNS_ON_CORE)
        const std::size_t core1_stack_size =
            reinterpret_cast<uintptr_t>(__core1_stack_top__) -
            reinterpret_cast<uintptr_t>(__core1_stack_bottom__);
        multicore_launch_core1_with_stack(core1_entry, __core1_stack_bottom__, core1_stack_size);
#else
        multicore_launch_core1_with_stack(core1_entry, s_core1_stack_naive, sizeof(s_core1_stack_naive));
#endif

        // Synchronise the start as closely as possible without sharing `this`.
        multicore_fifo_push_blocking(0xC1C00001u);
        Core0Entry();

        // Core 1 will push a completion token when it is done.
        const uint32_t completion = multicore_fifo_pop_blocking();
        (void)completion;

        // Reset core 1 so a later RunTest() call on the same or another
        // instance can safely relaunch it (the SDK does not allow relaunching
        // a still-halted core 1 without this).
        multicore_reset_core1();

        results_.core0 = scratch_core0_;
        results_.core1 = scratch_core1_;
    }

    static void StartMeasurementCore0() {
        cached_time_us_core0_ = time_us_64();
    }
    static void StartMeasurementCore1() {
        cached_time_us_core1_ = time_us_64();
    }
    static void StopMeasurementCore0() {
        const uint64_t end_time_us = time_us_64();
        const uint64_t elapsed_time_us = end_time_us - cached_time_us_core0_;
        TestResultsPerCore &out = scratch_core0_;
        out.iterations++;
        calcMetrics(elapsed_time_us, out.iterations,
            out.time_us_avg, out.time_us_max, out.time_us_min);
    }
    static void StopMeasurementCore1() {
        const uint64_t end_time_us = time_us_64();
        const uint64_t elapsed_time_us = end_time_us - cached_time_us_core1_;
        TestResultsPerCore &out = scratch_core1_;
        out.iterations++;
        calcMetrics(elapsed_time_us, out.iterations,
            out.time_us_avg, out.time_us_max, out.time_us_min);
    }

private:
    void Core0Entry() {
        core0_work_.init();
        for (uint32_t i = 0; i < core0_work_.iterations; ++i) {
            core0_work_.task();
        }
    }

    static MEML_RUNS_ON_CORE(1) void Core1Entry() {
        const uint32_t start_token = multicore_fifo_pop_blocking();
        (void)start_token;
        core1_work_.init();
        for (uint32_t i = 0; i < core1_work_.iterations; ++i) {
            core1_work_.task();
        }
        multicore_fifo_push_blocking(0xDEADBEEF);
    }

    const TestConfig config_;
    TestResults results_;
    inline static MEML_DATA_ON_CORE(0) TestResultsPerCore scratch_core0_{0, 0.0f, 0.0f, std::numeric_limits<float>::max()};
    inline static MEML_DATA_ON_CORE(1) TestResultsPerCore scratch_core1_{0, 0.0f, 0.0f, std::numeric_limits<float>::max()};
    inline static MEML_DATA_ON_CORE(0) CoreWork core0_work_{0, nullptr, nullptr};
    inline static MEML_DATA_ON_CORE(1) CoreWork core1_work_{0, nullptr, nullptr};
    inline static MEML_DATA_ON_CORE(0) uint64_t cached_time_us_core0_ = 0u;
    inline static MEML_DATA_ON_CORE(1) uint64_t cached_time_us_core1_ = 0u;

    static inline void calcMetrics(const uint64_t elapsed_time_us_raw,
                     const uint32_t iterations,
                     float &avg_time_us,
                     float &max_time_us,
                     float &min_time_us) {
        float elapsed_time_us = static_cast<float>(elapsed_time_us_raw);
        avg_time_us += (elapsed_time_us - avg_time_us) / static_cast<float>(iterations);
        max_time_us = std::max(max_time_us, elapsed_time_us);
        min_time_us = std::min(min_time_us, elapsed_time_us);
    }
};

}

#endif // __TEST_BASE_HPP__
