/**
 * @file StaticLayerBackwardLoopOrderTest.cpp
 * @brief Tolerance coverage for the backward loop-interchange in
 * AccumulateGradients/UpdateImmediate/CalcGradients (Optimisation 4,
 * plan-mlpOptimisationPrompt.md).
 *
 * Each of these methods used to compute delta_out[j] with i outermost (a
 * NOut-way reduction round-tripping delta_out[j] through memory on every
 * (i,j)); they now compute it with j outermost so the reduction accumulates
 * in a register across the whole inner loop. The summation order of
 * es[0..NOut-1] into a fixed delta_out[j] is unchanged by this interchange,
 * so results are expected to stay extremely close to (and typically bit-
 * identical with) a double-precision reference -- this suite checks that
 * across non-power-of-two NIn/NOut shapes, so neither dimension happens to
 * hide a transposed-index bug.
 */

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "MemoryDefs.hpp"
#include "mlp/StaticLayer.h"
#include "tests/unit/microunit/microunit.h"

namespace {

constexpr float kTolerance = 1e-4f;

bool IsClose(float a, float b, float tolerance = kTolerance) {
    return std::fabs(a - b) <= tolerance;
}

// Deterministic pseudo-random float in [-1, 1) from a simple integer hash,
// so weight/error patterns aren't uniform (which could hide a transposed- or
// reordered-index bug) but stay reproducible across runs.
float PseudoRandom(std::size_t index, uint32_t salt) {
    uint32_t x = static_cast<uint32_t>(index) * 2654435761u + salt;
    x ^= x >> 15;
    x *= 0x85ebca6bu;
    x ^= x >> 13;
    return ((x >> 8) * (1.0f / 16777216.0f)) * 2.0f - 1.0f;
}

// NIn/NOut deliberately not multiples of each other or of 4, so a
// transposed-index or off-by-one bug in the interchanged loops can't hide
// behind a coincidentally-symmetric shape.
constexpr std::size_t kNIn  = 37;
constexpr std::size_t kNOut = 23;

using Layer = smlp::StaticLayer<float, kNIn, kNOut, ACTIVATION_FUNCTIONS::LINEAR, /*EnableTraining=*/true>;

// LINEAR's derivative is the constant 1, so es[i] == deriv_err[i] regardless
// of m_act_output/m_inner_products -- isolates the loop-order change from
// activation-derivative math.
void SetUp(Layer & layer, float deriv_err[kNOut]) {
    for (std::size_t i = 0; i < kNOut; ++i) {
        deriv_err[i] = PseudoRandom(i, 0xA5A5u);
        for (std::size_t j = 0; j < kNIn; ++j)
            layer.weight(i, j) = PseudoRandom(i * kNIn + j, 0x1234u);
    }
}

// delta_out[j] = sum_{i=0}^{NOut-1} es[i] * w[i,j], es[i] == deriv_err[i] here.
// Summed in the same i-ascending order the original i-outer loop used, so
// this is the same reduction, just computed independently in double.
float ReferenceDeltaOut(const Layer & layer, const float deriv_err[kNOut], std::size_t j) {
    double acc = 0.0;
    for (std::size_t i = 0; i < kNOut; ++i)
        acc += static_cast<double>(deriv_err[i]) * static_cast<double>(layer.weight(i, j));
    return static_cast<float>(acc);
}

} // namespace

UNIT(AccumulateGradientsDeltaOutMatchesReference) {
    Layer layer;
    float deriv_err[kNOut];
    SetUp(layer, deriv_err);

    float input[kNIn];
    for (std::size_t j = 0; j < kNIn; ++j) input[j] = PseudoRandom(j, 0x9999u);

    float delta_out[kNIn];
    layer.AccumulateGradients(input, deriv_err, delta_out);

    for (std::size_t j = 0; j < kNIn; ++j)
        ASSERT_TRUE(IsClose(delta_out[j], ReferenceDeltaOut(layer, deriv_err, j)));
    PASS();
}

UNIT(CalcGradientsDeltaOutMatchesReference) {
    Layer layer;
    float deriv_err[kNOut];
    SetUp(layer, deriv_err);

    float delta_out[kNIn];
    layer.CalcGradients(nullptr, deriv_err, delta_out);

    for (std::size_t j = 0; j < kNIn; ++j) {
        const float expected = ReferenceDeltaOut(layer, deriv_err, j);
        ASSERT_TRUE(IsClose(delta_out[j], expected));
        ASSERT_TRUE(IsClose(layer.GetGrads()[j], expected)); // m_grads mirrors delta_out
    }
    PASS();
}

UNIT(UpdateImmediateDeltaOutUsesPreUpdateWeights) {
    Layer layer;
    float deriv_err[kNOut];
    SetUp(layer, deriv_err);

    // Reference must be computed against the *pre-update* weights: the
    // interchanged delta_out loop runs before the weight-update loop for
    // exactly this reason.
    float expected[kNIn];
    for (std::size_t j = 0; j < kNIn; ++j) expected[j] = ReferenceDeltaOut(layer, deriv_err, j);

    float input[kNIn];
    for (std::size_t j = 0; j < kNIn; ++j) input[j] = PseudoRandom(j, 0x9999u);

    float delta_out[kNIn];
    layer.UpdateImmediate(input, deriv_err, /*lr=*/0.01f, delta_out);

    for (std::size_t j = 0; j < kNIn; ++j) ASSERT_TRUE(IsClose(delta_out[j], expected[j]));
    PASS();
}
