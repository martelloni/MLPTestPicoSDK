/**
 * @file TestNaiveMemoryLayout.hpp
 * @author Andrea Martelloni
 * @brief Counterfactual to TestRAMIndependence: the same MLPOpticalRecognition
 * experiment trained repeatedly against either a RAM-flooding or a fully
 * dormant (WFE-blocked) other-core task, but with NO per-core placement
 * scheme at all -- everything falls wherever the compiler/linker's ordinary,
 * default section placement puts it inside one shared 256 KiB SRAM window.
 * @date 2026-09-13
 */

#ifndef __TEST_NAIVE_MEMORY_LAYOUT_HPP__
#define __TEST_NAIVE_MEMORY_LAYOUT_HPP__

// This benchmark is the unpinned-build-only counterfactual to
// TestRAMIndependence: it only makes sense once there is NO per-core
// placement scheme in effect, so fail loudly at compile time rather than
// silently building something meaningless (the exact inverse of
// TestRAMIndependence.hpp's own guard).
#if defined(MEML_MLP_RUNS_ON_CORE)
#error "TestNaiveMemoryLayout only builds for the unpinned configuration"
#endif

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>

#include "hardware/sync.h"

#include "TestBase.hpp"
#include "MemoryDefs.hpp"
#include "experiments/MLPOpticalRecognition.hpp"

// Sizing the flood buffer to fill whatever the shared 256 KiB window has
// left after every other static object in the program (Step 7's two-pass
// build: a throwaway probe measures "everything else", then
// scripts/compute_naive_flooder_size.py computes the remaining budget). The
// probe build only needs to measure that baseline footprint, not actually
// flood anything, so it uses a nominal 1-word buffer instead of waiting on
// its own not-yet-generated header.
#if defined(MEML_NAIVE_FLOODER_PROBE_BUILD)
static constexpr std::size_t kNaiveFlooderWords = 1u;
#else
#include "NaiveFlooderSize.hpp" // generated into ${CMAKE_BINARY_DIR}/generated, on MLPTestPicoSDK's include path
static constexpr std::size_t kNaiveFlooderWords = MEML_NAIVE_FLOODER_WORDS;
#endif
// utils/RAMFlooder.hpp's FillOnce() carries MEML_RUNS_ON_CORE(RAM_FLOODER_ON_CORE)
// unconditionally (left as-is, deliberately, per this file's header comment
// on mlp_experiment_/ram_flooder_ below); without this build's linker
// fragments that section name has no explicit output-section rule, so ld's
// standard orphan-section placement lands it adjacently to .text -- itself
// the naive/default layout under test, not a gap to patch. No
// RAM_FLOODER_ON_CORE override is needed here: it only affects the section
// name suffix, which is inert without the fragments that would otherwise key
// off it.
#include "utils/RAMFlooder.hpp"

namespace test {

class TestNaiveMemoryLayout : public TestBase {

public:

    static constexpr uint32_t kMaxTrainingRepeats = 32u;

    /**
     * @param training_repeats Independent train-from-scratch sessions run on the MLP core.
     * @param epochs_per_session Epochs passed to each session's Train() call.
     * @param runOtherCoreTask true pins the RAM-flooding task to the other core;
     * false pins a fully dormant (WFE-blocked, zero memory traffic) task instead.
     * This selects a function pointer once at construction time, not a runtime branch.
     */
    TestNaiveMemoryLayout(uint32_t training_repeats = 10u,
                           uint32_t epochs_per_session = 10u,
                           bool runOtherCoreTask = true)
        : TestBase(BuildConfig(training_repeats, epochs_per_session, runOtherCoreTask)) {
        training_repeats_ = training_repeats;
        epochs_per_session_ = epochs_per_session;
        run_other_core_task_ = runOtherCoreTask;

        const CoreCallback other_init = runOtherCoreTask ? &OtherCoreFloodInit : &OtherCoreDormantInit;
        const CoreCallback other_task = runOtherCoreTask ? &OtherCoreFloodTask : &OtherCoreDormantTask;

        // Fixed, arbitrary core assignment: there is no MEML_MLP_RUNS_ON_CORE
        // selector to key off in the unpinned build. The assignment only
        // needs to stay fixed for the two runOtherCoreTask runs to be
        // comparable with each other and with the pinned builds' own results.
        ConfigureCoreWork(&MlpCoreInit, &MlpCoreTask, other_init, other_task);
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
    /// true.
    uint64_t GetOtherCoreIterationsConsumedUntilDone() const { return flood_iterations_at_done_; }

private:

    // Deterministic per-session seed base, matching TestRAMIndependence's convention.
    static constexpr uint32_t kBaseSeed = 0xC0DEu;
    static constexpr float kLearningRate = 0.01f;

    // See TestRAMIndependence::BuildConfig for why the other core's cap is
    // always exactly 1 and both other-core tasks loop internally instead.
    static TestConfig BuildConfig(uint32_t training_repeats,
                                   uint32_t epochs_per_session, bool runOtherCoreTask) {
        (void)epochs_per_session;
        (void)runOtherCoreTask;
        const uint32_t mlp_iterations = training_repeats;
        const uint32_t other_iterations = 1u;
        other_core_iteration_cap_ = other_iterations;
        // Fixed assignment: MLP experiment on TestBase's core-0 slot, flood/
        // dormant task on its core-1 slot.
        return TestConfig{mlp_iterations, other_iterations};
    }

    // One build-owned experiment instance and one flood buffer, both PLAIN
    // and UNTAGGED -- no MEML_DATA_ON_CORE/MEML_RUNS_ON_CORE attribute of any
    // kind -- so both fall wherever the compiler/linker's ordinary allocation
    // order puts them inside the shared 256 KiB window. This is the entire
    // point of the counterfactual: Step 1's placement scheme is switched off
    // completely here, not just for this one instance.
    inline static MLPOpticalRecognition mlp_experiment_{};
    inline static utils::RAMFlooder<uint32_t, kNaiveFlooderWords> ram_flooder_{};

    // Plain (untagged) completion signal: not smlp:: state, must stay outside
    // validate_memory_placement.py's bank-exclusivity check (which is skipped
    // entirely for the unpinned build in any case).
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

    static void MlpCoreInit() {
        // Reset per-run state so repeated RunTest() calls on the same instance
        // (e.g. the flooding-vs-dormant correctness-independence comparison) start clean.
        session_index_ = 0u;
        result_count_ = 0u;
        training_done_.store(false, std::memory_order_relaxed);
    }

    static void MlpCoreTask() {
        // Deterministic, session-varying seed: same base/convention as
        // TestRAMIndependence, offset per session so each of the
        // training_repeats runs is an independent train-from-scratch trial.
        const uint32_t seed = kBaseSeed + session_index_;
        mlp_experiment_.Initialise(seed);

        StartMeasurementCore0();
        const MLPOpticalRecognition::Result result = mlp_experiment_.Train(epochs_per_session_, kLearningRate);
        StopMeasurementCore0();

        // Record this session's result; the array is fixed-size, so extra sessions
        // beyond kMaxTrainingRepeats are silently not recorded (config is caller-controlled).
        if (result_count_ < kMaxTrainingRepeats) {
            training_results_[result_count_] = result;
            ++result_count_;
        }
        ++session_index_;

        printf("[TestNaiveMemoryLayout] session %u/%u complete: loss=%f accuracy=%f\n",
               session_index_, training_repeats_,
               static_cast<double>(result.loss), static_cast<double>(result.accuracy));

        // Last session: publish completion (release) and wake any core blocked in WFE.
        if (session_index_ >= training_repeats_) {
            training_done_.store(true, std::memory_order_release);
            __sev();
        }
    }

    static void OtherCoreFloodInit() {
        flood_iteration_counter_ = 0u;
        flood_iterations_at_done_ = 0u;
    }

    static void OtherCoreFloodTask() {
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

    static void OtherCoreDormantInit() {
        // Nothing to initialise: the dormant task never touches ram_flooder_.
    }

    static void OtherCoreDormantTask() {
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

#endif // __TEST_NAIVE_MEMORY_LAYOUT_HPP__
