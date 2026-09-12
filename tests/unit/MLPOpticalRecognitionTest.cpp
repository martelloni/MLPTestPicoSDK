/**
 * @file MLPOpticalRecognitionTest.cpp
 * @brief Microunit coverage for the MLPOpticalRecognition experiment facade.
 *
 * Repurposes the placement guarantees MLPPlacementTest.cpp (Step 1.5) proved
 * with a standalone probe: the same core/address checks now run against the
 * real experiment's State instead of a throwaway fixture, so the two do not
 * both consume reserved-bank budget. Most cases keep training calls short
 * (epochs=0 and epochs=1) so interactive, USB-driven unit runs stay quick;
 * MLPOpticalRecognitionTrainingSmokeTest below is the exception, running the
 * full 20-epoch deterministic acceptance contract from
 * plan-mlpCoreLocalPrompt.md as part of this same suite, on whichever core it
 * is compiled/run for.
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "pico.h"

#include "MemoryDefs.hpp"
#include "experiments/MLPOpticalRecognition.hpp"
#include "dataset/Dataset.hpp"
#include "tests/unit/microunit/microunit.h"

namespace {

constexpr uint32_t kSeed = 0xC0DEu;

#if defined(MEML_MLP_RUNS_ON_CORE) && MEML_MLP_RUNS_ON_CORE == 1
constexpr uint32_t kExpectedCore = 1u;
constexpr uintptr_t kBankBase = 0x20040000u;
#else
constexpr uint32_t kExpectedCore = 0u;
constexpr uintptr_t kBankBase = 0x20000000u;
#endif
constexpr uintptr_t kBankEnd = kBankBase + 0x40000u;

// The experiment's single static state object, placed by the caller (not by
// MLPOpticalRecognition itself) so it lands in the selected core's bank; its
// State member is the class's only data member, so this instance's address
// doubles as State's address for the placement check below.
MEML_MLP_DATA MLPOpticalRecognition g_experiment{};

} // namespace

UNIT(MLPOpticalRecognitionInitialiseIsDeterministic) {
    // Same seed -> same initial weights -> identical (untrained, epochs=0)
    // loss/accuracy when re-initialised.
    g_experiment.Initialise(kSeed);
    const MLPOpticalRecognition::Result first = g_experiment.Train(/*epochs=*/0, 0.01f);

    g_experiment.Initialise(kSeed);
    const MLPOpticalRecognition::Result second = g_experiment.Train(/*epochs=*/0, 0.01f);

    ASSERT_TRUE(std::isfinite(first.loss));
    ASSERT_TRUE(first.loss == second.loss);
    ASSERT_TRUE(first.accuracy == second.accuracy);
    PASS();
}

UNIT(MLPOpticalRecognitionShortTrainCallSucceeds) {
    g_experiment.Initialise(kSeed);
    const MLPOpticalRecognition::Result result = g_experiment.Train(/*epochs=*/1, 0.01f);

    ASSERT_TRUE(std::isfinite(result.loss));
    ASSERT_TRUE(result.accuracy >= 0.0f && result.accuracy <= 1.0f);
    PASS();
}

UNIT(MLPOpticalRecognitionPredictReturnsValidClass) {
    g_experiment.Initialise(kSeed);
    const uint32_t predicted = g_experiment.Predict(dataset::features[0]);
    ASSERT_TRUE(predicted < dataset::kLabelSize);
    PASS();
}

UNIT(MLPOpticalRecognitionTrainingSmokeTest) {
    // Deterministic acceptance contract from plan-mlpCoreLocalPrompt.md's
    // "Verification contract": seed 0xC0DEu, RMSProp, lr=0.01, batch size 128
    // (fixed inside Train()), 20 epochs over the full flash-resident dataset.
    // Require final mean loss <= 70% of the untrained initial mean loss and
    // training accuracy >= 80%. Runs on whichever core the suite executes on
    // (core 0 for unpinned/core-0 builds, the core-1 trampoline otherwise),
    // since it reuses the already-core-pinned g_experiment fixture above.
    constexpr uint32_t kSmokeEpochs = 20u;
    constexpr float kLearningRate = 0.01f;
    constexpr float kMaxLossRatio = 0.70f;
    constexpr float kMinAccuracy = 0.80f;

    g_experiment.Initialise(kSeed);
    // epochs=0 performs no weight update, so this reports the untrained
    // initial loss/accuracy baseline without perturbing the seeded weights.
    const MLPOpticalRecognition::Result initial = g_experiment.Train(/*epochs=*/0, kLearningRate);
    const MLPOpticalRecognition::Result final_result = g_experiment.Train(kSmokeEpochs, kLearningRate);

    // Print loss/accuracy over stdio/UART at test end regardless of outcome,
    // so a hardware run always leaves a serial record even on failure.
    std::printf(
        "MLPOpticalRecognitionTrainingSmokeTest: initial_loss=%f final_loss=%f final_accuracy=%f\n",
        static_cast<double>(initial.loss), static_cast<double>(final_result.loss),
        static_cast<double>(final_result.accuracy));

    ASSERT_TRUE(std::isfinite(initial.loss) && initial.loss > 0.0f);
    ASSERT_TRUE(std::isfinite(final_result.loss));
    ASSERT_TRUE(final_result.loss <= kMaxLossRatio * initial.loss);
    ASSERT_TRUE(final_result.accuracy >= kMinAccuracy);
    PASS();
}

UNIT(MLPOpticalRecognitionStateIsPlacedInSelectedCoreBank) {
    // MEML_MLP_DATA state lives in the selected core's 256 KiB bank.
    const uintptr_t state_address = reinterpret_cast<uintptr_t>(&g_experiment);
    ASSERT_TRUE(state_address >= kBankBase && state_address < kBankEnd);

    // MLP-affine: whichever core the suite runs this test on must match the
    // selector (the core-1 runner executes it from inside its core-1
    // trampoline; unpinned/core-0 builds run it directly on core 0).
    ASSERT_TRUE(static_cast<uint32_t>(get_core_num()) == kExpectedCore);
    PASS();
}
