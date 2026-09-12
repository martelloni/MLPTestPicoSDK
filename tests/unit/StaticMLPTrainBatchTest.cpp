/**
 * @file StaticMLPTrainBatchTest.cpp
 * @brief Microunit coverage for the zero-copy, pointer/stride StaticMLP::TrainBatch()
 * overload: correctness against a hand-calculated RMSProp step, its validation
 * contract, and deterministic shuffling with caller-owned index buffers.
 *
 * The allocation-free property itself is enforced by Step 5's host-only global
 * allocation counter, since microunit itself allocates for registration and
 * reporting; this suite exercises the identical hot path functionally with
 * caller-owned buffers, as the plan's Step 3 testing section specifies.
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "MemoryDefs.hpp"
#include "mlp/StaticMLP.h"
#include "tests/unit/microunit/microunit.h"

namespace {

constexpr float kTolerance = 1e-4f;

bool IsClose(float a, float b, float tolerance = kTolerance) {
    return std::fabs(a - b) <= tolerance;
}

// Smallest possible net (one weight, one bias, LINEAR) so a single RMSProp
// step can be reproduced by hand rather than merely re-deriving the
// implementation.
using TinyNet = smlp::StaticMLP<float, smlp::Layout<1, 1>,
                                 smlp::Activations<ACTIVATION_FUNCTIONS::LINEAR>>;

// Small multi-sample net used for the shuffle-determinism check.
using ShuffleNet = smlp::StaticMLP<float, smlp::Layout<2, 2, 1>,
                                    smlp::Activations<ACTIVATION_FUNCTIONS::RELU,
                                                      ACTIVATION_FUNCTIONS::LINEAR>>;

} // namespace

// Hand-calculated single-sample RMSProp step (w=b=0 initially; x=2, target=4):
//   pred = 0; diff = 4; deriv = -2*diff = -8 (ssr=1, MSE, n=1)
//   es = deriv * d/dx LINEAR = -8
//   grad_w = x*es = -16; grad_b = es = -8
//   ||grad||= 16 > clip(5) -> coef = 5/16 = 0.3125
//   scaled grad_w = -5.0; scaled grad_b = -2.5
//   sq_avg_w = 0.1*5^2 = 2.5;  adj_w = lr/(sqrt(2.5)+eps)
//   sq_avg_b = 0.1*2.5^2 = 0.625; adj_b = lr/(sqrt(0.625)+eps)
//   w_new = -adj_w*(-5.0); b_new = -adj_b*(-2.5)
UNIT(TrainBatchPointerMatchesHandCalculatedRMSPropStep) {
    TinyNet net;
    const float feature[1] = {2.0f};
    const float label[1]   = {4.0f};
    std::size_t shuffle[1] = {0};
    float final_loss = -1.0f;
    const float lr = 0.01f;

    const bool ok = net.TrainBatch(feature, 1, label, 1, /*sample_count=*/1,
                                    lr, /*epochs=*/1, /*batch_size=*/1,
                                    shuffle, /*capacity=*/1, final_loss);
    ASSERT_TRUE(ok);

    const double sq_avg_w = 0.1 * 5.0 * 5.0;
    const double sq_avg_b = 0.1 * 2.5 * 2.5;
    const double adj_w = lr / (std::sqrt(sq_avg_w) + 1e-6);
    const double adj_b = lr / (std::sqrt(sq_avg_b) + 1e-6);
    const float expected_w = static_cast<float>(adj_w * 5.0);
    const float expected_b = static_cast<float>(adj_b * 2.5);

    ASSERT_TRUE(IsClose(net.layer<0>().weight(0, 0), expected_w, 1e-3f));
    ASSERT_TRUE(IsClose(net.layer<0>().bias(0), expected_b, 1e-3f));
    ASSERT_TRUE(IsClose(final_loss, 16.0f, 1e-2f));

    // Source arrays are untouched (also enforced at compile time via `const T*`).
    ASSERT_TRUE(feature[0] == 2.0f);
    ASSERT_TRUE(label[0] == 4.0f);
    PASS();
}

UNIT(TrainBatchPointerRejectsNullFeatures) {
    TinyNet net;
    const float label[1] = {4.0f};
    std::size_t shuffle[1] = {0};
    float final_loss = -1.0f;
    const bool ok = net.TrainBatch(nullptr, 1, label, 1, 1, 0.01f, 1, 1, shuffle, 1, final_loss);
    ASSERT_TRUE(!ok);
    ASSERT_TRUE(final_loss == -1.0f);
    PASS();
}

UNIT(TrainBatchPointerRejectsNullLabels) {
    TinyNet net;
    const float feature[1] = {2.0f};
    std::size_t shuffle[1] = {0};
    float final_loss = -1.0f;
    const bool ok = net.TrainBatch(feature, 1, nullptr, 1, 1, 0.01f, 1, 1, shuffle, 1, final_loss);
    ASSERT_TRUE(!ok);
    ASSERT_TRUE(final_loss == -1.0f);
    PASS();
}

UNIT(TrainBatchPointerRejectsNullShuffleBuffer) {
    TinyNet net;
    const float feature[1] = {2.0f};
    const float label[1]   = {4.0f};
    float final_loss = -1.0f;
    const bool ok = net.TrainBatch(feature, 1, label, 1, 1, 0.01f, 1, 1, nullptr, 1, final_loss);
    ASSERT_TRUE(!ok);
    ASSERT_TRUE(final_loss == -1.0f);
    PASS();
}

UNIT(TrainBatchPointerRejectsZeroSampleCount) {
    TinyNet net;
    const float feature[1] = {2.0f};
    const float label[1]   = {4.0f};
    std::size_t shuffle[1] = {0};
    float final_loss = -1.0f;
    const bool ok = net.TrainBatch(feature, 1, label, 1, /*sample_count=*/0,
                                    0.01f, 1, 1, shuffle, 1, final_loss);
    ASSERT_TRUE(!ok);
    ASSERT_TRUE(final_loss == -1.0f);
    PASS();
}

UNIT(TrainBatchPointerRejectsZeroBatchSize) {
    TinyNet net;
    const float feature[1] = {2.0f};
    const float label[1]   = {4.0f};
    std::size_t shuffle[1] = {0};
    float final_loss = -1.0f;
    const bool ok = net.TrainBatch(feature, 1, label, 1, 1, 0.01f, 1,
                                    /*batch_size=*/0, shuffle, 1, final_loss);
    ASSERT_TRUE(!ok);
    ASSERT_TRUE(final_loss == -1.0f);
    PASS();
}

UNIT(TrainBatchPointerRejectsUndersizedFeatureStride) {
    // TinyNet needs kNumInputs=1; a stride of 0 is undersized.
    TinyNet net;
    const float feature[1] = {2.0f};
    const float label[1]   = {4.0f};
    std::size_t shuffle[1] = {0};
    float final_loss = -1.0f;
    const bool ok = net.TrainBatch(feature, /*feature_stride=*/0, label, 1, 1,
                                    0.01f, 1, 1, shuffle, 1, final_loss);
    ASSERT_TRUE(!ok);
    ASSERT_TRUE(final_loss == -1.0f);
    PASS();
}

UNIT(TrainBatchPointerRejectsUndersizedLabelStride) {
    TinyNet net;
    const float feature[1] = {2.0f};
    const float label[1]   = {4.0f};
    std::size_t shuffle[1] = {0};
    float final_loss = -1.0f;
    const bool ok = net.TrainBatch(feature, 1, label, /*label_stride=*/0, 1,
                                    0.01f, 1, 1, shuffle, 1, final_loss);
    ASSERT_TRUE(!ok);
    ASSERT_TRUE(final_loss == -1.0f);
    PASS();
}

UNIT(TrainBatchPointerRejectsInsufficientShuffleCapacity) {
    TinyNet net;
    const float feature[2] = {2.0f, 3.0f};
    const float label[2]   = {4.0f, 6.0f};
    std::size_t shuffle[1] = {0}; // capacity 1 < sample_count 2
    float final_loss = -1.0f;
    const bool ok = net.TrainBatch(feature, 1, label, 1, /*sample_count=*/2,
                                    0.01f, 1, 1, shuffle, /*capacity=*/1, final_loss);
    ASSERT_TRUE(!ok);
    ASSERT_TRUE(final_loss == -1.0f);
    PASS();
}

UNIT(TrainBatchPointerValidationFailureLeavesWeightsUnchanged) {
    TinyNet net;
    const float w0 = net.layer<0>().weight(0, 0);
    const float b0 = net.layer<0>().bias(0);
    std::size_t shuffle[1] = {0};
    float final_loss = -1.0f;
    const bool ok = net.TrainBatch(nullptr, 1, nullptr, 1, 0, 0.01f, 1, 0,
                                    nullptr, 0, final_loss);
    ASSERT_TRUE(!ok);
    ASSERT_TRUE(net.layer<0>().weight(0, 0) == w0);
    ASSERT_TRUE(net.layer<0>().bias(0) == b0);
    PASS();
}

UNIT(TrainBatchPointerDeterministicShuffleMatchesAcrossIdenticallySeededNets) {
    ShuffleNet net_a;
    ShuffleNet net_b;
    net_a.SetSeed(0xABCDEFu);
    net_b.SetSeed(0xABCDEFu);

    constexpr std::size_t kSamples = 6;
    const float features[kSamples][2] = {
        {0.1f, 0.2f}, {0.3f, 0.4f}, {0.5f, 0.6f},
        {0.7f, 0.8f}, {0.9f, 1.0f}, {1.1f, 1.2f},
    };
    const float labels[kSamples][1] = {
        {0.0f}, {1.0f}, {0.0f}, {1.0f}, {0.0f}, {1.0f},
    };

    std::size_t shuffle_a[kSamples];
    std::size_t shuffle_b[kSamples];
    float loss_a = -1.0f, loss_b = -1.0f;

    const bool ok_a = net_a.TrainBatch(&features[0][0], 2, &labels[0][0], 1, kSamples,
                                        0.01f, /*epochs=*/3, /*batch_size=*/2,
                                        shuffle_a, kSamples, loss_a);
    const bool ok_b = net_b.TrainBatch(&features[0][0], 2, &labels[0][0], 1, kSamples,
                                        0.01f, /*epochs=*/3, /*batch_size=*/2,
                                        shuffle_b, kSamples, loss_b);
    ASSERT_TRUE(ok_a);
    ASSERT_TRUE(ok_b);

    for (std::size_t i = 0; i < kSamples; ++i) ASSERT_TRUE(shuffle_a[i] == shuffle_b[i]);
    ASSERT_TRUE(IsClose(loss_a, loss_b, 1e-6f));

    const auto weights_a = net_a.GetAllWeights();
    const auto weights_b = net_b.GetAllWeights();
    for (std::size_t l = 0; l < weights_a.size(); ++l)
        for (std::size_t n = 0; n < weights_a[l].size(); ++n)
            for (std::size_t j = 0; j < weights_a[l][n].size(); ++j)
                ASSERT_TRUE(IsClose(weights_a[l][n][j], weights_b[l][n][j], 1e-6f));
    PASS();
}
