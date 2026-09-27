/**
 * @file StaticMLPForwardAccumulatorTest.cpp
 * @brief Tolerance coverage for StaticLayer::forward()'s 4-way accumulator
 * dot product (Optimisation 3, plan-mlpOptimisationPrompt.md).
 *
 * Splitting the single scalar accumulator into 4 independent partial sums
 * changes float summation order, so the result is no longer bit-identical to
 * a naive left-to-right reduction -- this suite checks it stays within a
 * tight tolerance of a double-precision reference across NIn values that
 * exercise every remainder-loop tail (0, 1, 2, 3 leftover elements after the
 * 4-wide unroll).
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "MemoryDefs.hpp"
#include "mlp/StaticMLP.h"
#include "tests/unit/microunit/microunit.h"

namespace {

constexpr float kTolerance = 1e-4f;

bool IsClose(float a, float b, float tolerance = kTolerance) {
    return std::fabs(a - b) <= tolerance;
}

// Single LINEAR node so the layer's output *is* the raw dot product plus
// bias -- isolates the accumulator change from activation math.
template<std::size_t NIn>
using DotProdNet = smlp::StaticMLP<float, smlp::Layout<NIn, 1>,
                                   smlp::Activations<ACTIVATION_FUNCTIONS::LINEAR>>;

// Deterministic pseudo-random float in [-1, 1) from a simple integer hash,
// so weight/input patterns aren't uniform (which could hide reassociation
// error) but stay reproducible across runs.
float PseudoRandom(std::size_t index, uint32_t salt) {
    uint32_t x = static_cast<uint32_t>(index) * 2654435761u + salt;
    x ^= x >> 15;
    x *= 0x85ebca6bu;
    x ^= x >> 13;
    return ((x >> 8) * (1.0f / 16777216.0f)) * 2.0f - 1.0f;
}

template<std::size_t NIn>
bool DotProductMatchesReference() {
    DotProdNet<NIn> net;
    std::array<float, NIn> input{};
    double reference = 0.5; // bias

    net.template layer<0>().bias(0) = 0.5f;
    for (std::size_t j = 0; j < NIn; ++j) {
        const float w = PseudoRandom(j, 0x1234u);
        const float x = PseudoRandom(j, 0x5678u);
        net.template layer<0>().weight(0, j) = w;
        input[j] = x;
        reference += static_cast<double>(w) * static_cast<double>(x);
    }

    std::array<float, 1> output{};
    net.GetOutput(input.data(), output.data());

    return std::isfinite(output[0]) && IsClose(output[0], static_cast<float>(reference));
}

} // namespace

// NIn % 4 == 0: exercises only the 4-wide unrolled loop, no scalar tail.
UNIT(ForwardAccumulatorMatchesReferenceNIn64) {
    ASSERT_TRUE(DotProductMatchesReference<64>());
    PASS();
}

// NIn % 4 == 1: 1-element scalar tail.
UNIT(ForwardAccumulatorMatchesReferenceNIn65) {
    ASSERT_TRUE(DotProductMatchesReference<65>());
    PASS();
}

// NIn % 4 == 2: 2-element scalar tail (also covers the network's NIn=10 layer shape).
UNIT(ForwardAccumulatorMatchesReferenceNIn10) {
    ASSERT_TRUE(DotProductMatchesReference<10>());
    PASS();
}

// NIn % 4 == 3: 3-element scalar tail.
UNIT(ForwardAccumulatorMatchesReferenceNIn7) {
    ASSERT_TRUE(DotProductMatchesReference<7>());
    PASS();
}

// NIn < 4: never enters the unrolled loop, scalar-only path.
UNIT(ForwardAccumulatorMatchesReferenceNIn3) {
    ASSERT_TRUE(DotProductMatchesReference<3>());
    PASS();
}
