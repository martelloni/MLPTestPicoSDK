/**
 * @file StaticLayerAccumulateGradientsTest.cpp
 * @brief Reference coverage for StaticLayer::AccumulateGradients (weight
 * gradient accumulator, bias gradient accumulator, back-propagated delta_out)
 * against a double-precision scalar reference, over shapes with
 * NOut % 4 != 0 and NIn % 2 != 0.
 *
 * Guard rail for the row-blocked backward rewrite (plan-mlpBetterOptimisation
 * Step 4): blocking changes the loop structure, so g/bias/delta_out must keep
 * matching this reference for every tail shape. The tolerance is loose enough
 * for float summation-order differences yet far tighter than any indexing bug.
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "MemoryDefs.hpp"
#include "mlp/StaticLayer.h"
#include "tests/unit/microunit/microunit.h"

namespace {

float PseudoRandom(std::size_t index, uint32_t salt) {
    uint32_t x = static_cast<uint32_t>(index) * 2654435761u + salt;
    x ^= x >> 15;
    x *= 0x85ebca6bu;
    x ^= x >> 13;
    return ((x >> 8) * (1.0f / 16777216.0f)) * 2.0f - 1.0f; // [-1, 1)
}

bool Close(double got, double want) {
    return std::fabs(got - want) <= 1e-5 * (1.0 + std::fabs(want));
}

// Runs `kSamples` forward+AccumulateGradients passes (so the accumulators
// really accumulate) and compares everything against a double reference that
// uses the layer's own cached pre-activations for the derivative.
template<std::size_t NIn, std::size_t NOut, ACTIVATION_FUNCTIONS Act>
bool AccumulateMatchesReference() {
    using Layer = smlp::StaticLayer<float, NIn, NOut, Act>;
    // Static, not a local: a layer carries ~3 copies of its weights, and the
    // whole unit suite shares the 256 KiB window with the experiment's State.
    // Not re-assigned from a temporary either (that would build one on the stack).
    static Layer layer;
    layer.m_grad_accum.fill(0.0f);
    layer.m_bias_grad_accum.fill(0.0f);
    for (std::size_t k = 0; k < Layer::kWeights; ++k) layer.m_weights[k] = 0.5f * PseudoRandom(k, 0x1u);
    for (std::size_t i = 0; i < NOut; ++i) layer.m_biases[i] = 0.25f * PseudoRandom(i, 0x2u);

    constexpr std::size_t kSamples = 3;
    std::array<double, NIn * NOut> ref_g{};
    std::array<double, NOut> ref_b{};

    for (std::size_t s = 0; s < kSamples; ++s) {
        std::array<float, NIn> input;
        std::array<float, NOut> out, err;
        std::array<float, NIn> delta;
        for (std::size_t j = 0; j < NIn; ++j) input[j] = PseudoRandom(j + 1000 * s, 0x3u);
        for (std::size_t i = 0; i < NOut; ++i) err[i] = PseudoRandom(i + 1000 * s, 0x4u);

        layer.forward(input.data(), out.data());
        // Poison delta so a missing zero-initialisation is caught.
        for (float & d : delta) d = 1e6f;
        layer.AccumulateGradients(input.data(), err.data(), delta.data());

        std::array<double, NIn> ref_delta{};
        for (std::size_t i = 0; i < NOut; ++i) {
            double deriv = 1.0;
            if constexpr (Act == ACTIVATION_FUNCTIONS::RELU)
                deriv = layer.m_inner_products[i] > 0.0f ? 1.0 : static_cast<double>(utils::kReLUSlope);
            const double es = static_cast<double>(err[i]) * deriv;
            ref_b[i] += es;
            for (std::size_t j = 0; j < NIn; ++j) {
                ref_g[i * NIn + j] += static_cast<double>(input[j]) * es;
                ref_delta[j] += es * static_cast<double>(layer.m_weights[i * NIn + j]);
            }
        }
        for (std::size_t j = 0; j < NIn; ++j)
            if (!Close(delta[j], ref_delta[j])) return false;
    }

    for (std::size_t k = 0; k < NIn * NOut; ++k)
        if (!Close(layer.m_grad_accum[k], ref_g[k])) return false;
    for (std::size_t i = 0; i < NOut; ++i)
        if (!Close(layer.m_bias_grad_accum[i], ref_b[i])) return false;
    return true;
}

constexpr auto kLin  = ACTIVATION_FUNCTIONS::LINEAR;
constexpr auto kRelu = ACTIVATION_FUNCTIONS::RELU;

template<std::size_t NIn, std::size_t NOut, ACTIVATION_FUNCTIONS Act>
bool CheckShape(const char* name) {
    const bool ok = AccumulateMatchesReference<NIn, NOut, Act>();
    if (!ok) printf("  AccumulateGradients mismatch for shape %s\n", name);
    return ok;
}

} // namespace

// One UNIT, not one per shape: microunit holds at most 32 tests
// (UnitTester::kMaximumTests) and silently drops the rest.
//
// Shapes are kept small (the unit suite shares RAM with the experiment's
// State); they cover every NOut % 4 tail and odd/even NIn, not the benchmark's
// exact 64x64 / 64x32 sizes, which exercise the same code paths.
UNIT(AccumulateGradientsMatchesReference) {
    bool ok = true;
    ok &= CheckShape<16, 8,  kRelu>("16x8 (NOut%4==0, full blocks)");
    ok &= CheckShape<32, 10, kLin >("32x10 (benchmark's last layer: 2 blocks + tail 2)");
    ok &= CheckShape<20, 12, kRelu>("20x12");
    ok &= CheckShape<7,  5,  kRelu>("7x5 (tail 1, odd NIn)");
    ok &= CheckShape<65, 3,  kLin >("65x3 (tail 3, odd NIn)");
    ok &= CheckShape<9,  2,  kRelu>("9x2 (tail 2)");
    ok &= CheckShape<5,  1,  kLin >("5x1");
    ok &= CheckShape<1,  1,  kRelu>("1x1");
    ok &= CheckShape<3,  7,  kRelu>("3x7");
    ASSERT_TRUE(ok);
    PASS();
}
