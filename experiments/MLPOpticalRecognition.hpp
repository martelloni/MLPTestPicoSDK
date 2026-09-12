/**
 * @file MLPOpticalRecognition.hpp
 * @brief Experiment facade for the 64-64-32-10 optical-digit-recognition MLP.
 *
 * Wraps the zero-copy static training/inference interface (mlp/StaticMLP.h)
 * around the flash-resident digits dataset (dataset/Dataset.hpp). All state
 * -- the network, its RMSProp buffers, the shuffle-index buffer, and I/O
 * scratch arrays -- lives in one private State member, so a caller-declared
 * instance of this class can be pinned entirely into one core's reserved
 * SRAM bank with a single MEML_MLP_DATA annotation at its declaration site
 * (see main.cpp / tests/unit/MLPOpticalRecognitionTest.cpp), rather than
 * tagging individual members.
 *
 * This project header must be included after MemoryDefs.hpp binds
 * SMLP_CODE_ATTR/SMLP_DATA_ATTR in any translation unit that needs real
 * (non-blank) placement; mlp/*.h headers never include MemoryDefs.hpp
 * themselves.
 */

#ifndef __MLP_OPTICAL_RECOGNITION_HPP__
#define __MLP_OPTICAL_RECOGNITION_HPP__

#include <array>
#include <cstddef>
#include <cstdint>

#include "MemoryDefs.hpp"
#include "mlp/StaticMLP.h"
#include "mlp/StaticLayer.h"
#include "dataset/Dataset.hpp"

/**
 * @brief Facade over a fixed-architecture StaticMLP trained on the optical
 * digit-recognition dataset.
 *
 * The caller owns CPU scheduling: call these methods directly on core 0, or
 * from the caller's own multicore_launch_core1() trampoline on core 1. This
 * class is not a multicore dispatcher and must not be accessed concurrently
 * from both cores.
 */
class MLPOpticalRecognition {
public:
    /// 64 -> 64 -> 32 -> 10, ReLU/ReLU/Linear, categorical cross-entropy;
    /// class prediction and accuracy use the raw-logit argmax, not softmax.
    using Net = smlp::StaticMLP<
        dataset::DataType,
        smlp::Layout<64, 64, 32, 10>,
        smlp::Activations<
            ACTIVATION_FUNCTIONS::RELU,
            ACTIVATION_FUNCTIONS::RELU,
            ACTIVATION_FUNCTIONS::LINEAR>,
        loss::LOSS_FUNCTIONS::LOSS_CATEGORICAL_CROSSENTROPY>;

    /// Mean categorical loss and classification accuracy over the full
    /// training set, measured after a Train() call.
    struct Result {
        float loss;
        float accuracy;
    };

    MLPOpticalRecognition() = default;

    /**
     * @brief Seeds the network's PRNG and Xavier-initializes its weights.
     * @param seed PRNG seed; the same seed always produces the same initial
     * weights and the same subsequent training trajectory.
     *
     * Call once before the first Train()/Predict(). Calling again reseeds
     * and re-initializes from scratch, discarding any prior training.
     */
    MEML_MLP_CODE void Initialise(uint32_t seed);

    /**
     * @brief Trains on the full flash-resident dataset and reports mean loss
     * and accuracy over it.
     * @param epochs Number of shuffled passes over the dataset; 0 performs
     * no weight update and reports the network's current loss/accuracy.
     * @param learning_rate RMSProp learning rate.
     * @return Mean categorical loss and training-set accuracy after training.
     *
     * Uses the zero-copy pointer-based StaticMLP::TrainBatch() overload
     * directly over `dataset::features`/`dataset::labels`, batch size 128,
     * with no array-to-vector conversion or heap allocation. State persists
     * across calls, so a later call fine-tunes rather than restarts.
     */
    MEML_MLP_CODE Result Train(uint32_t epochs, float learning_rate);

    /**
     * @brief Runs inference on a single sample.
     * @param input `dataset::kFeatureSize` normalized feature values, copied
     * into the fixed-size input scratch array before inference.
     * @return The predicted class: the raw-logit argmax, in
     * `[0, dataset::kLabelSize)`.
     */
    MEML_MLP_CODE uint32_t Predict(const std::array<dataset::DataType, dataset::kFeatureSize> & input);

private:
    /**
     * @brief All of this experiment's core-local storage: the network plus
     * every scratch buffer Train()/Predict() use.
     *
     * Grouped into one struct so a single MEML_MLP_DATA-tagged instance of
     * MLPOpticalRecognition places all of it in one core's SRAM bank.
     */
    struct State {
        Net net{};
        std::array<std::size_t, dataset::kNumExamples> shuffle_indices{};
        std::array<dataset::DataType, dataset::kFeatureSize> input_scratch{};
        std::array<dataset::DataType, dataset::kLabelSize> output_scratch{};
    };

    State state_{};
};

#endif  // __MLP_OPTICAL_RECOGNITION_HPP__
