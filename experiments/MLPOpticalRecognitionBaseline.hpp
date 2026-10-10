/**
 * @file MLPOpticalRecognitionBaseline.hpp
 * @brief Same experiment as MLPOpticalRecognition (64-64-32-10 optical-digit
 * recognition), but run on the original heap-based, non-templated-layout MLP
 * engine (mlp/MLP.h) instead of the static zero-copy smlp::StaticMLP.
 *
 * Public API is identical to MLPOpticalRecognition (Result, Initialise, Train,
 * Predict), so the two are interchangeable behind the MLPExperiment alias in
 * experiments/MLPExperiment.hpp. Engine selection is build-time only.
 *
 * Deliberate differences, all forced by the old engine rather than chosen:
 *  - The old MLP owns its weights/buffers on the heap (std::vector), so
 *    MEML_MLP_DATA on an instance pins only the thin State shell, not the
 *    weights; MEML_MLP_CODE pins only this facade's entry points, not the
 *    MLP<float> member functions. This is exactly the placement gap the
 *    baseline exists to expose.
 *  - MLP<float> has no SetSeed(): its shuffle PRNG (std::mt19937 `g`) is
 *    seeded from std::random_device and Xavier init uses libc rand(). The
 *    private Engine subclass reseeds both from Initialise()'s seed so runs
 *    stay reproducible.
 *  - MLP<float>::TrainBatch() needs the whole dataset as vector<vector<float>>
 *    (~1.1 MB, far over the 256 KiB RAM window), so Engine::TrainStreaming()
 *    re-implements TrainBatch's loop line for line (per-sample forward with
 *    retained activations, accumulate-backprop, grad-norm clip at 5.0,
 *    RMSProp apply, std::shuffle) but copies one sample at a time out of the
 *    flash-resident dataset into reusable vectors.
 */

#ifndef __MLP_OPTICAL_RECOGNITION_BASELINE_HPP__
#define __MLP_OPTICAL_RECOGNITION_BASELINE_HPP__

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "MemoryDefs.hpp"
#include "mlp/MLP.h"
#include "dataset/Dataset.hpp"

class MLPOpticalRecognitionBaseline {
public:
    struct Result {
        float loss;
        float accuracy;
    };

    MLPOpticalRecognitionBaseline() = default;

    /// Seeds shuffle PRNG + libc rand(), Xavier-initialises, resets RMSProp state.
    MEML_MLP_CODE void Initialise(uint32_t seed);

    /// Same contract as MLPOpticalRecognition::Train (batch size 128).
    MEML_MLP_CODE Result Train(uint32_t epochs, float learning_rate);

    /// Raw-logit argmax class.
    MEML_MLP_CODE uint32_t Predict(const std::array<dataset::DataType, dataset::kFeatureSize> & input);

private:
    class Engine : public MLP<dataset::DataType> {
    public:
        Engine();
        void Seed(uint32_t seed);
        /// MLP::TrainBatch, streaming samples from flash. Returns last epoch loss.
        bool TrainStreaming(float learning_rate, int epochs, std::size_t batch_size,
                            std::size_t * indices, dataset::DataType & final_loss);
    private:
        std::vector<dataset::DataType> sample_features_;
        std::vector<dataset::DataType> sample_labels_;
    };

    struct State {
        Engine net{};
        std::array<std::size_t, dataset::kNumExamples> shuffle_indices{};
        std::vector<dataset::DataType> input_scratch = std::vector<dataset::DataType>(dataset::kFeatureSize);
        std::vector<dataset::DataType> output_scratch = std::vector<dataset::DataType>(dataset::kLabelSize);
    };

    State state_{};
};

#endif  // __MLP_OPTICAL_RECOGNITION_BASELINE_HPP__
