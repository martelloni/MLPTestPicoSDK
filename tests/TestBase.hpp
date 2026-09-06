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
#include "MemoryDefs.hpp"

#define DWT_CTRL     (*(volatile uint32_t*)0xE0001000)
#define DWT_CYCCNT   (*(volatile uint32_t*)0xE0001004)
#define DEMCR        (*(volatile uint32_t*)0xE000EDFC)

namespace test {

extern "C" {
extern uint32_t __core1_stack_bottom__[];
extern uint32_t __core1_stack_top__[];
}

class TestBase {

public:

    struct TestConfig {
        uint32_t clock_frequency_hz;
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
        core0_clock_hz_ = config_.clock_frequency_hz;
        core1_clock_hz_ = config_.clock_frequency_hz;
        EnableDWT();
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

        EnableDWT();
        multicore_fifo_drain();

        // Cortex-M Thumb entry points must have bit 0 set. The linker symbol itself is
        // emitted as an even address in the ELF, so we have to force the Thumb bit before
        // handing the function pointer to the ROM launcher.
        const auto core1_entry = reinterpret_cast<void (*)()>(reinterpret_cast<uintptr_t>(&Core1Entry) | 1u);
        const std::size_t core1_stack_size =
            reinterpret_cast<uintptr_t>(__core1_stack_top__) -
            reinterpret_cast<uintptr_t>(__core1_stack_bottom__);
        multicore_launch_core1_with_stack(core1_entry, __core1_stack_bottom__, core1_stack_size);

        // Synchronise the start as closely as possible without sharing `this`.
        multicore_fifo_push_blocking(0xC1C00001u);
        Core0Entry();

        // Core 1 will push a completion token when it is done.
        const uint32_t completion = multicore_fifo_pop_blocking();
        (void)completion;

        results_.core0 = scratch_core0_;
        results_.core1 = scratch_core1_;
    }

    static void StartMeasurementCore0() {
        cached_count_core0_ = GetDWTCount();
    }
    static void StartMeasurementCore1() {
        cached_count_core1_ = GetDWTCount();
    }
    static void StopMeasurementCore0() {
        const uint32_t end_count = GetDWTCount();
        const uint32_t elapsed_cycles = end_count - cached_count_core0_;
        TestResultsPerCore &out = scratch_core0_;
        out.iterations++;
        calcMetrics(elapsed_cycles, out.iterations, core0_clock_hz_,
            out.time_us_avg, out.time_us_max, out.time_us_min);
    }
    static void StopMeasurementCore1() {
        const uint32_t end_count = GetDWTCount();
        const uint32_t elapsed_cycles = end_count - cached_count_core1_;
        TestResultsPerCore &out = scratch_core1_;
        out.iterations++;
        calcMetrics(elapsed_cycles, out.iterations, core1_clock_hz_,
            out.time_us_avg, out.time_us_max, out.time_us_min);
    }

private:
    void Core0Entry() {
        EnableDWT();
        core0_work_.init();
        for (uint32_t i = 0; i < core0_work_.iterations; ++i) {
            core0_work_.task();
        }
    }

    static MEML_RUNS_ON_CORE(1) void Core1Entry() {
        const uint32_t start_token = multicore_fifo_pop_blocking();
        (void)start_token;
        EnableDWT();
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
    inline static MEML_DATA_ON_CORE(0) uint32_t core0_clock_hz_ = 0u;
    inline static MEML_DATA_ON_CORE(1) uint32_t core1_clock_hz_ = 0u;
    inline static MEML_DATA_ON_CORE(0) uint32_t cached_count_core0_ = 0u;
    inline static MEML_DATA_ON_CORE(1) uint32_t cached_count_core1_ = 0u;

    static inline void calcMetrics(const uint32_t elapsed_cycles,
                     const uint32_t iterations,
                     const uint32_t clock_frequency_hz,
                     float &avg_time_us,
                     float &max_time_us,
                     float &min_time_us) {
        float elapsed_time_us = static_cast<float>(elapsed_cycles)
                              / static_cast<float>(clock_frequency_hz) * 1e6f;
        avg_time_us += (elapsed_time_us - avg_time_us) / static_cast<float>(iterations);
        max_time_us = std::max(max_time_us, elapsed_time_us);
        min_time_us = std::min(min_time_us, elapsed_time_us);
    }

    static inline void EnableDWT() {
        DEMCR |= (1 << 24);      // enable trace (TRCENA)
        DWT_CYCCNT = 0;
        DWT_CTRL |= 1;           // enable cycle counter
    }

    static inline void ResetDWT() {
        DWT_CYCCNT = 0;
    }

    static inline uint32_t GetDWTCount() {
        return DWT_CYCCNT;
    }
};

}

#endif // __TEST_BASE_HPP__
