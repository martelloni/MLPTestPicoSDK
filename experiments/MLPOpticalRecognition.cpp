/**
 * @file MLPOpticalRecognition.cpp
 * @brief Definitions for the optical-digit-recognition experiment facade.
 *
 * A single translation unit for MLPOpticalRecognition's non-template entry
 * points, so the selected build's code-placement attribute (MEML_MLP_CODE)
 * applies to one deterministic definition of each rather than to whatever
 * copy the linker happens to keep across multiple translation units.
 */

#include "experiments/MLPOpticalRecognition.hpp"

#include "mlp/Loss.h"

void MLPOpticalRecognition::Initialise(uint32_t seed) {
    // Same seed -> same initial weights -> same subsequent training trajectory.
    state_.net.SetSeed(seed);
    state_.net.InitXavier();
}

MLPOpticalRecognition::Result MLPOpticalRecognition::Train(uint32_t epochs, float learning_rate) {
    dataset::DataType final_loss = dataset::DataType(0);
    // Zero-copy pointer-based training straight over the flash-resident
    // dataset; the shuffle buffer is caller-owned (this instance's State),
    // so no vector/array conversion or heap allocation occurs.
    const bool trained = state_.net.TrainBatch(
        dataset::features[0].data(), dataset::kFeatureSize,
        dataset::labels[0].data(), dataset::kLabelSize,
        dataset::kNumExamples,
        learning_rate, epochs, /*batch_size=*/128,
        state_.shuffle_indices.data(), dataset::kNumExamples,
        final_loss);
    if (!trained) return Result{ -1.0f, -1.0f };

    // Re-evaluate mean loss and accuracy over the full training set from the
    // (now-trained) weights; raw-logit argmax decides both the loss target
    // index and the predicted class, matching the categorical-crossentropy
    // convention that probabilities are only materialised on request.
    float total_loss = 0.0f;
    std::size_t correct = 0;
    dataset::DataType deriv_scratch[dataset::kLabelSize];
    for (std::size_t i = 0; i < dataset::kNumExamples; ++i) {
        state_.net.GetOutput(dataset::features[i].data(), state_.output_scratch.data(), /*for_inference=*/false);
        total_loss += loss::CategoricalCrossEntropyLogits(
            dataset::labels[i].data(), state_.output_scratch.data(), deriv_scratch,
            dataset::kLabelSize, dataset::DataType(1.0));

        std::size_t predicted = 0;
        state_.net.GetOutputClass(state_.output_scratch.data(), &predicted);
        std::size_t expected = 0;
        state_.net.GetOutputClass(dataset::labels[i].data(), &expected); // one-hot label's argmax is its class index
        if (predicted == expected) ++correct;
    }

    Result result;
    result.loss = total_loss / static_cast<float>(dataset::kNumExamples);
    result.accuracy = static_cast<float>(correct) / static_cast<float>(dataset::kNumExamples);
    return result;
}

uint32_t MLPOpticalRecognition::Predict(const std::array<dataset::DataType, dataset::kFeatureSize> & input) {
    state_.input_scratch = input; // stable, core-local copy before inference
    state_.net.GetOutput(state_.input_scratch.data(), state_.output_scratch.data(), /*for_inference=*/false);
    std::size_t predicted = 0;
    state_.net.GetOutputClass(state_.output_scratch.data(), &predicted);
    return static_cast<uint32_t>(predicted);
}
