/**
 * @file SoftmaxTest.cpp
 * @brief Microunit coverage for the stable, allocation-free utils::SoftmaxInPlace().
 */

#include <cmath>
#include <cstddef>

#include "MemoryDefs.hpp"
#include "mlp/Utils.h"
#include "tests/unit/microunit/microunit.h"

namespace {

constexpr float kTolerance = 1e-4f;

bool IsClose(float a, float b, float tolerance = kTolerance) {
    return std::fabs(a - b) <= tolerance;
}

std::size_t ArgMax(const float *values, std::size_t count) {
    std::size_t best = 0;
    for (std::size_t i = 1; i < count; ++i) {
        if (values[i] > values[best]) best = i;
    }
    return best;
}

} // namespace

UNIT(SoftmaxSumsToOne) {
    float logits[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    utils::SoftmaxInPlace(logits, 4);
    float total = 0.0f;
    for (float v : logits) total += v;
    ASSERT_TRUE(IsClose(total, 1.0f));
    PASS();
}

UNIT(SoftmaxInvariantToConstantShift) {
    float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[4] = {1001.0f, 1002.0f, 1003.0f, 1004.0f}; // a shifted by a common constant
    utils::SoftmaxInPlace(a, 4);
    utils::SoftmaxInPlace(b, 4);
    for (std::size_t i = 0; i < 4; ++i) ASSERT_TRUE(IsClose(a[i], b[i]));
    PASS();
}

UNIT(SoftmaxPreservesArgmax) {
    float logits[5] = {-3.0f, 7.0f, 0.5f, 7.0001f, -1.0f};
    const std::size_t before = ArgMax(logits, 5);
    utils::SoftmaxInPlace(logits, 5);
    const std::size_t after = ArgMax(logits, 5);
    ASSERT_TRUE(before == after);
    PASS();
}

UNIT(SoftmaxFiniteForExtremeLogits) {
    // Large positive logits: naive exp() would overflow without max-shifting.
    float large_pos[3] = {1.0e30f, 1.0e30f - 1.0f, -1.0e30f};
    utils::SoftmaxInPlace(large_pos, 3);
    for (float v : large_pos) {
        ASSERT_TRUE(std::isfinite(v));
        ASSERT_TRUE(v >= 0.0f && v <= 1.0f);
    }

    // Large negative logits: naive exp() would underflow every term to zero,
    // making the un-shifted sum zero and the result NaN.
    float large_neg[3] = {-1.0e30f, -1.0e30f, -1.0e30f + 1.0f};
    utils::SoftmaxInPlace(large_neg, 3);
    float total = 0.0f;
    for (float v : large_neg) {
        ASSERT_TRUE(std::isfinite(v));
        total += v;
    }
    ASSERT_TRUE(IsClose(total, 1.0f));
    PASS();
}
