#ifndef __MLP_OPTICAL_RECOGNITION_HPP__
#define __MLP_OPTICAL_RECOGNITION_HPP__

#include "mlp/StaticMLP.h"
#include "mlp/StaticLayer.h"
#include "MemoryDefs.hpp"
#include "dataset/Dataset.hpp"


class MLPOpticalRecognition {
public:

    using Net = smlp::StaticMLP<
        dataset::DataType,
        smlp::Layout<64, 32, 10>,
        smlp::Activations<
            ACTIVATION_FUNCTIONS::RELU,
            ACTIVATION_FUNCTIONS::RELU,
            ACTIVATION_FUNCTIONS::SIGMOID>>;

    struct Result {
        float loss;
        float accuracy;
    };

    MLPOpticalRecognition() {};
    Result Train(uint32_t epochs, float learning_rate);
    uint32_t Predict(const std::array<dataset::DataType, dataset::kFeatureSize>& input);


protected:


};


#endif  // __MLP_OPTICAL_RECOGNITION_HPP__
