/**
 * @file MLPOpticalRecognitionBaseline.cpp
 * @brief Definitions for the old-engine (mlp/MLP.h) optical-recognition facade.
 */

#include "experiments/MLPOpticalRecognitionBaseline.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>

#include "mlp/Loss.h"

MLPOpticalRecognitionBaseline::Engine::Engine()
    : MLP<dataset::DataType>(
          {64, 64, 32, 10},
          {ACTIVATION_FUNCTIONS::RELU, ACTIVATION_FUNCTIONS::RELU, ACTIVATION_FUNCTIONS::LINEAR},
          loss::LOSS_FUNCTIONS::LOSS_CATEGORICAL_CROSSENTROPY),
      sample_features_(dataset::kFeatureSize),
      sample_labels_(dataset::kLabelSize) {}

void MLPOpticalRecognitionBaseline::Engine::Seed(uint32_t seed) {
    g.seed(seed);          // std::shuffle in TrainStreaming
    std::srand(seed);      // Layer::InitXavier via utils::gen_rand
}

bool MLPOpticalRecognitionBaseline::Engine::TrainStreaming(
        float learning_rate, int epochs, std::size_t batch_size,
        std::size_t * indices, dataset::DataType & final_loss) {
    using T = dataset::DataType;
    const std::size_t n_samples = dataset::kNumExamples;
    const std::size_t n_batches = (n_samples + batch_size - 1) / batch_size;
    constexpr float kMinErrorCost = 0.001f;

    std::iota(indices, indices + n_samples, std::size_t{0});
    std::vector<std::vector<T>> all_layers_activations;

    T epoch_loss = 0;
    for (int iter = 0; iter < epochs; iter++) {
        epoch_loss = 0;
        std::shuffle(indices, indices + n_samples, g);

        std::size_t sample_idx = 0;
        for (std::size_t batch = 0; batch < n_batches; batch++) {
            const std::size_t current_batch_size = std::min(batch_size, n_samples - sample_idx);
            const T batch_size_reciprocal = (T)1.0 / static_cast<T>(current_batch_size);

            InitializeAllGradientAccumulators();
            T batch_loss = 0;

            for (std::size_t i = 0; i < current_batch_size; i++) {
                const std::size_t idx = indices[sample_idx++];
                std::copy(dataset::features[idx].begin(), dataset::features[idx].end(), sample_features_.begin());
                std::copy(dataset::labels[idx].begin(), dataset::labels[idx].end(), sample_labels_.begin());

                all_layers_activations.clear();
                GetOutput(sample_features_, &m_train_predicted, &all_layers_activations, false);

                m_train_deriv_error.resize(m_train_predicted.size());
                batch_loss += loss_fn_(sample_labels_, m_train_predicted, m_train_deriv_error, 1.0f);

                BackpropagateWithAccumulation(all_layers_activations, m_train_deriv_error, true);
            }

            T grad_sumsq = 0.0f;
            for (auto & layer : m_layers) {
                grad_sumsq += layer.GetGradSumSquared(batch_size_reciprocal);
            }
            const T grad_norm = std::sqrt(grad_sumsq);
            if (grad_norm > 5.0f) {
                const T clip_coef = 5.0f / grad_norm;
                for (auto & layer : m_layers) {
                    layer.ScaleAccumulatedGradients(clip_coef);
                }
            }

            ApplyAllAccumulatedGradients(learning_rate, batch_size_reciprocal);
            epoch_loss += batch_loss / current_batch_size;
        }
        epoch_loss /= n_batches;
        if (epoch_loss < kMinErrorCost) break;
    }
    final_loss = epoch_loss;
    return true;
}

void MLPOpticalRecognitionBaseline::Initialise(uint32_t seed) {
    state_.net.Seed(seed);
    state_.net.InitXavier();
    state_.net.ResetOptimizerState();
}

MLPOpticalRecognitionBaseline::Result MLPOpticalRecognitionBaseline::Train(uint32_t epochs, float learning_rate) {
    dataset::DataType final_loss = dataset::DataType(0);
    const bool trained = state_.net.TrainStreaming(
        learning_rate, static_cast<int>(epochs), /*batch_size=*/128,
        state_.shuffle_indices.data(), final_loss);
    if (!trained) return Result{ -1.0f, -1.0f };

    float total_loss = 0.0f;
    std::size_t correct = 0;
    dataset::DataType deriv_scratch[dataset::kLabelSize];
    for (std::size_t i = 0; i < dataset::kNumExamples; ++i) {
        std::copy(dataset::features[i].begin(), dataset::features[i].end(), state_.input_scratch.begin());
        state_.net.GetOutput(state_.input_scratch, &state_.output_scratch, nullptr, /*for_inference=*/false);
        total_loss += loss::CategoricalCrossEntropyLogits(
            dataset::labels[i].data(), state_.output_scratch.data(), deriv_scratch,
            dataset::kLabelSize, dataset::DataType(1.0));

        std::size_t predicted = 0;
        state_.net.GetOutputClass(state_.output_scratch, &predicted);
        std::size_t expected = static_cast<std::size_t>(
            std::max_element(dataset::labels[i].begin(), dataset::labels[i].end()) - dataset::labels[i].begin());
        if (predicted == expected) ++correct;
    }

    Result result;
    result.loss = total_loss / static_cast<float>(dataset::kNumExamples);
    result.accuracy = static_cast<float>(correct) / static_cast<float>(dataset::kNumExamples);
    return result;
}

uint32_t MLPOpticalRecognitionBaseline::Predict(const std::array<dataset::DataType, dataset::kFeatureSize> & input) {
    std::copy(input.begin(), input.end(), state_.input_scratch.begin());
    state_.net.GetOutput(state_.input_scratch, &state_.output_scratch, nullptr, /*for_inference=*/false);
    std::size_t predicted = 0;
    state_.net.GetOutputClass(state_.output_scratch, &predicted);
    return static_cast<uint32_t>(predicted);
}
