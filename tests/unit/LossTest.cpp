/**
 * @file LossTest.cpp
 * @brief Microunit coverage for the allocation-free loss::CategoricalCrossEntropyLogits().
 */

#include <cmath>
#include <cstddef>

#include "MemoryDefs.hpp"
#include "mlp/Loss.h"
#include "tests/unit/microunit/microunit.h"

namespace {

constexpr float kTolerance = 1e-4f;

bool IsClose(float a, float b, float tolerance = kTolerance) {
    return std::fabs(a - b) <= tolerance;
}

} // namespace

// Hand-calculated reference: logits = {1, 2, 3}, one-hot target on class 2.
// softmax = {0.090031, 0.244728, 0.665241}; loss = -log(0.665241) = 0.407606.
UNIT(CategoricalCrossEntropyOneHot) {
    const float target[3] = {0.0f, 0.0f, 1.0f};
    const float logits[3] = {1.0f, 2.0f, 3.0f};
    float grad[3] = {0.0f, 0.0f, 0.0f};
    const float loss_value =
        loss::CategoricalCrossEntropyLogits(target, logits, grad, std::size_t(3), 1.0f);

    ASSERT_TRUE(IsClose(loss_value, 0.407606f, 1e-3f));
    ASSERT_TRUE(IsClose(grad[0], 0.090031f, 1e-3f));  // softmax[0] - target[0]
    ASSERT_TRUE(IsClose(grad[1], 0.244728f, 1e-3f));  // softmax[1] - target[1]
    ASSERT_TRUE(IsClose(grad[2], -0.334759f, 1e-3f)); // softmax[2] - target[2]
    PASS();
}

// Non-one-hot target: an even probability split exercises the full
// -Sum(target[i] * log_softmax[i]) contract, not a single-class fast path.
// logits = {0, 0, 0} -> softmax = {1/3, 1/3, 1/3}; log_softmax[i] = -log(3).
// target = {0.5, 0.5, 0.0} -> loss = 0.5*log(3) + 0.5*log(3) + 0 = log(3).
UNIT(CategoricalCrossEntropyNonOneHot) {
    const float target[3] = {0.5f, 0.5f, 0.0f};
    const float logits[3] = {0.0f, 0.0f, 0.0f};
    float grad[3] = {0.0f, 0.0f, 0.0f};
    const float loss_value =
        loss::CategoricalCrossEntropyLogits(target, logits, grad, std::size_t(3), 1.0f);

    ASSERT_TRUE(IsClose(loss_value, std::log(3.0f)));
    ASSERT_TRUE(IsClose(grad[0], 1.0f / 3.0f - 0.5f));
    ASSERT_TRUE(IsClose(grad[1], 1.0f / 3.0f - 0.5f));
    ASSERT_TRUE(IsClose(grad[2], 1.0f / 3.0f));
    PASS();
}

UNIT(CategoricalCrossEntropySampleSizeReciprocalScalesLossAndGradient) {
    const float target[2] = {1.0f, 0.0f};
    const float logits[2] = {0.0f, 0.0f};
    float grad_full[2] = {0.0f, 0.0f};
    float grad_half[2] = {0.0f, 0.0f};
    const float loss_full =
        loss::CategoricalCrossEntropyLogits(target, logits, grad_full, std::size_t(2), 1.0f);
    const float loss_half =
        loss::CategoricalCrossEntropyLogits(target, logits, grad_half, std::size_t(2), 0.5f);

    ASSERT_TRUE(IsClose(loss_half, loss_full * 0.5f));
    ASSERT_TRUE(IsClose(grad_half[0], grad_full[0] * 0.5f));
    ASSERT_TRUE(IsClose(grad_half[1], grad_full[1] * 0.5f));
    PASS();
}
