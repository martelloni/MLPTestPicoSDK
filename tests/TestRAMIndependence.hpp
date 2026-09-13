/**
 * @file TestRAMIndependence.hpp
 * @author Andrea Martelloni
 * @brief Core-agnostic RAM-independence benchmark: the real MLPOpticalRecognition
 * experiment trained repeatedly on MEML_MLP_RUNS_ON_CORE's core, against either a
 * RAM-flooding or a fully dormant (WFE-blocked) task on the other core.
 * @date 2026-09-13
 */

#ifndef __TEST_RAM_INDEPENDENCE_HPP__
#define __TEST_RAM_INDEPENDENCE_HPP__

// This benchmark only makes sense once there is a defined "MLP core" vs.
// "other core" pairing; the unpinned/blank-placement configuration has none,
// so fail loudly at compile time rather than silently building something
// meaningless.
#if !defined(MEML_MLP_RUNS_ON_CORE)
#error "TestRAMIndependence requires MEML_MLP_RUNS_ON_CORE=0 or =1 (an unpinned build has no MLP-core/other-core pairing)."
#endif

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>

#include "hardware/sync.h"

#include "TestBase.hpp"
#include "MemoryDefs.hpp"
#include "experiments/MLPOpticalRecognition.hpp"

// The RAM flooder must always live on whichever core the MLP experiment did
// NOT get pinned to, driven by the same selector rather than independently
// configurable. Must be defined before including RAMFlooder.hpp, which only
// supplies a default.
#if MEML_MLP_RUNS_ON_CORE == 0
#define RAM_FLOODER_ON_CORE 1
#else
#define RAM_FLOODER_ON_CORE 0
#endif
#include "utils/RAMFlooder.hpp"

// Entry-point placement attributes for this test's own address-taken
// CoreCallback function pointers (mirrors MEML_RUNS_ON_CORE(n)'s `used`
// requirement for raw function-pointer registration, per MemoryDefs.hpp).
// Data placement for the flooder mirrors it onto the opposite bank.
#if MEML_MLP_RUNS_ON_CORE == 0
#define MEML_TEST_MLP_CORE_ATTR MEML_RUNS_ON_CORE(0)
#define MEML_TEST_OTHER_CORE_ATTR MEML_RUNS_ON_CORE(1)
#define MEML_TEST_OTHER_CORE_DATA MEML_DATA_ON_CORE(1)
#else
#define MEML_TEST_MLP_CORE_ATTR MEML_RUNS_ON_CORE(1)
#define MEML_TEST_OTHER_CORE_ATTR MEML_RUNS_ON_CORE(0)
#define MEML_TEST_OTHER_CORE_DATA MEML_DATA_ON_CORE(0)
#endif

namespace test {

class TestRAMIndependence : public TestBase {

public:

    static constexpr uint32_t kMaxTrainingRepeats = 32u;

    /**
     * @param clock_frequency_hz System clock, used for cycle-to-microsecond conversion.
     * @param training_repeats Independent train-from-scratch sessions run on the MLP core.
     * @param epochs_per_session Epochs passed to each session's Train() call.
     * @param runOtherCoreTask true pins the RAM-flooding task to the other core;
     * false pins a fully dormant (WFE-blocked, zero memory traffic) task instead.
     * This selects a function pointer once at construction time, not a runtime branch.
     */
    TestRAMIndependence(uint32_t clock_frequency_hz,
                         uint32_t training_repeats = 10u,
                         uint32_t epochs_per_session = 10u,
                         bool runOtherCoreTask = true)
        : TestBase(BuildConfig(clock_frequency_hz, training_repeats, epochs_per_session, runOtherCoreTask)) {
        training_repeats_ = training_repeats;
        epochs_per_session_ = epochs_per_session;
        run_other_core_task_ = runOtherCoreTask;

        const CoreCallback other_init = runOtherCoreTask ? &OtherCoreFloodInit : &OtherCoreDormantInit;
        const CoreCallback other_task = runOtherCoreTask ? &OtherCoreFloodTask : &OtherCoreDormantTask;

#if MEML_MLP_RUNS_ON_CORE == 0
        ConfigureCoreWork(&MlpCoreInit, &MlpCoreTask, other_init, other_task);
#else
        ConfigureCoreWork(other_init, other_task, &MlpCoreInit, &MlpCoreTask);
#endif
    }

    /// {loss, accuracy} for each completed training session, in order.
    const std::array<MLPOpticalRecognition::Result, kMaxTrainingRepeats> & GetTrainingResults() const {
        return training_results_;
    }
    uint32_t GetTrainingResultCount() const { return result_count_; }

    /// The other core's outer iteration cap as configured on TestBase: always
    /// exactly 1, since the flood/dormant task loops internally until
    /// training_done_ rather than being called a preset number of times.
    uint32_t GetOtherCoreIterationCap() const { return other_core_iteration_cap_; }

    /// Only meaningful when constructed with runOtherCoreTask=true: how many
    /// RAMFlooder::FillOnce() passes ran before training_done_ was observed
    /// true. 64-bit: the flood loop is genuinely uncapped (see
    /// OtherCoreFloodTask), so a long-enough run -- many training_repeats /
    /// epochs_per_session, or a very fast FillOnce() -- could exceed UINT32_MAX.
    uint64_t GetOtherCoreIterationsConsumedUntilDone() const { return flood_iterations_at_done_; }

private:

    // Deterministic per-session seed base, matching MLPOpticalRecognitionTest.cpp's convention.
    static constexpr uint32_t kBaseSeed = 0xC0DEu;
    static constexpr float kLearningRate = 0.01f;

    // The other core's outer iteration cap is always exactly 1: TestBase's
    // Core1Entry loop bound is fixed at construction time, before training
    // even starts, so any fixed guess at "how many flood passes outlast N
    // training sessions" is a race that can lose (a flood pass over 96-240 KB
    // is far cheaper than a real training epoch, so a guessed cap can run out
    // and leave the other core idle for the rest of the MLP core's run,
    // silently defeating the whole point of the benchmark). Instead, both
    // OtherCoreFloodTask and OtherCoreDormantTask loop internally until
    // training_done_ is observed, and are each called exactly once.
    static TestConfig BuildConfig(uint32_t clock_frequency_hz, uint32_t training_repeats,
                                   uint32_t epochs_per_session, bool runOtherCoreTask) {
        (void)epochs_per_session;
        (void)runOtherCoreTask;
        const uint32_t mlp_iterations = training_repeats;
        const uint32_t other_iterations = 1u;
        other_core_iteration_cap_ = other_iterations;
#if MEML_MLP_RUNS_ON_CORE == 0
        return TestConfig{clock_frequency_hz, mlp_iterations, other_iterations};
#else
        return TestConfig{clock_frequency_hz, other_iterations, mlp_iterations};
#endif
    }

    // One build-owned experiment instance, pinned to whichever core the selector
    // chose; separate from the unit-test suite's g_experiment (tests/benchmarks
    // are mutually exclusive build modes, so the two never coexist in one binary).
    inline static MEML_MLP_DATA MLPOpticalRecognition mlp_experiment_{};

    // The flooder's other core is the otherwise near-empty core-1 bank when the
    // MLP runs on core 0, so it can claim the full 240 KB. When the MLP instead
    // runs on core 1, the flooder lands on core 0's bank, which also hosts the
    // rest of the program's default runtime state (heap, main stack, .data/.bss);
    // it must fit alongside that fixed baseline footprint within the same 256 KiB
    // window rather than claim all of it, so its size is deliberately smaller here.
#if MEML_MLP_RUNS_ON_CORE == 0
    static constexpr uint32_t core_ram_flooder_size_ = 1024 * 60; // 240 KB, on core 1
#else
    static constexpr uint32_t core_ram_flooder_size_ = 1024 * 24; // 96 KB, on core 0
#endif
    inline static MEML_TEST_OTHER_CORE_DATA utils::RAMFlooder<uint32_t, core_ram_flooder_size_> ram_flooder_{};

    // Plain (untagged) completion signal: not smlp:: state, must stay outside
    // validate_memory_placement.py's bank-exclusivity check.
    inline static std::atomic<bool> training_done_{false};

    // Re-seeding counter without changing TestBase's no-argument CoreCallback contract.
    inline static uint32_t session_index_{0u};

    inline static uint32_t training_repeats_{0u};
    inline static uint32_t epochs_per_session_{0u};
    inline static bool run_other_core_task_{true};
    inline static uint32_t other_core_iteration_cap_{0u};

    inline static std::array<MLPOpticalRecognition::Result, kMaxTrainingRepeats> training_results_{};
    inline static uint32_t result_count_{0u};

    inline static uint64_t flood_iteration_counter_{0u};
    inline static uint64_t flood_iterations_at_done_{0u};

    static MEML_TEST_MLP_CORE_ATTR void MlpCoreInit() {
        // Reset per-run state so repeated RunTest() calls on the same instance
        // (e.g. the flooding-vs-dormant correctness-independence comparison) start clean.
        session_index_ = 0u;
        result_count_ = 0u;
        training_done_.store(false, std::memory_order_relaxed);
    }

    static MEML_TEST_MLP_CORE_ATTR void MlpCoreTask() {
        // Deterministic, session-varying seed: same base as the microunit smoke test,
        // offset per session so each of the training_repeats runs is an independent
        // train-from-scratch trial rather than a repeat of the same trajectory.
        const uint32_t seed = kBaseSeed + session_index_;
        mlp_experiment_.Initialise(seed);

#if MEML_MLP_RUNS_ON_CORE == 0
        StartMeasurementCore0();
        const MLPOpticalRecognition::Result result = mlp_experiment_.Train(epochs_per_session_, kLearningRate);
        StopMeasurementCore0();
#else
        StartMeasurementCore1();
        const MLPOpticalRecognition::Result result = mlp_experiment_.Train(epochs_per_session_, kLearningRate);
        StopMeasurementCore1();
#endif

        // Record this session's result; the array is fixed-size, so extra sessions
        // beyond kMaxTrainingRepeats are silently not recorded (config is caller-controlled).
        if (result_count_ < kMaxTrainingRepeats) {
            training_results_[result_count_] = result;
            ++result_count_;
        }
        ++session_index_;

        // Progress checkpoint over stdio, deliberately AFTER Stop*Measurement*()
        // above so it never adds into the DWT-cycle-counted training time; a
        // multi-session run at 100 epochs/session can otherwise sit silent for a
        // long time. pico_stdio_usb serialises output with its own internal mutex
        // (stdio_usb_mutex in stdio_usb.c), so this is safe whichever physical
        // core the MLP experiment (and hence this task) runs on.
        printf("[TestRAMIndependence] session %u/%u complete: loss=%f accuracy=%f\n",
               session_index_, training_repeats_,
               static_cast<double>(result.loss), static_cast<double>(result.accuracy));

        // Last session: publish completion (release) and wake any core blocked in WFE.
        if (session_index_ >= training_repeats_) {
            training_done_.store(true, std::memory_order_release);
            __sev();
        }
    }

    static MEML_TEST_OTHER_CORE_ATTR void OtherCoreFloodInit() {
        flood_iteration_counter_ = 0u;
        flood_iterations_at_done_ = 0u;
    }

    static MEML_TEST_OTHER_CORE_ATTR void OtherCoreFloodTask() {
        // Called exactly once (other_core_iteration_cap_ == 1); loops
        // internally, mirroring OtherCoreDormantTask below, so flooding
        // spans the MLP core's whole variable-length training run instead of
        // racing a fixed outer iteration count that could run out early.
        while (!training_done_.load(std::memory_order_acquire)) {
            ram_flooder_.FillOnce();
            ++flood_iteration_counter_;
        }
        flood_iterations_at_done_ = flood_iteration_counter_;
    }

    static MEML_TEST_OTHER_CORE_ATTR void OtherCoreDormantInit() {
        // Nothing to initialise: the dormant task never touches ram_flooder_.
    }

    static MEML_TEST_OTHER_CORE_ATTR void OtherCoreDormantTask() {
        // Blocks in WFE for the whole interval, generating zero memory-bus traffic;
        // only re-checks the flag on a wake event (the MLP core's __sev(), or a
        // permitted spurious wake). This is the single call for this core's
        // iteration count of exactly 1.
        while (!training_done_.load(std::memory_order_acquire)) {
            __wfe();
        }
    }

};

}


#endif // __TEST_RAM_INDEPENDENCE_HPP__
