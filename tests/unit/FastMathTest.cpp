/**
 * @file FastMathTest.cpp
 * @brief Accuracy coverage for smlp::exp_f32/log_f32 (mlp/FastMath.h) against
 * double-precision std::exp/std::log, over the ranges the softmax
 * cross-entropy actually sees, plus the special values.
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>

#include "MemoryDefs.hpp"
#include "mlp/FastMath.h"
#include "mlp/Loss.h"
#include "tests/unit/microunit/microunit.h"

namespace {

constexpr float kRelTol = 5e-7f; // a few ulp (float eps is 1.19e-7)

float RelErr(float got, double want) {
    const double d = std::fabs(static_cast<double>(got) - want);
    return static_cast<float>(want != 0.0 ? d / std::fabs(want) : d);
}

float PseudoRandom(std::size_t index, uint32_t salt) {
    uint32_t x = static_cast<uint32_t>(index) * 2654435761u + salt;
    x ^= x >> 15;
    x *= 0x85ebca6bu;
    x ^= x >> 13;
    return ((x >> 8) * (1.0f / 16777216.0f)) * 2.0f - 1.0f; // [-1, 1)
}

bool Check(const char* what, bool ok) {
    if (!ok) printf("  FastMath check failed: %s\n", what);
    return ok;
}


// Softmax exponents are x - max <= 0; training logits stay within a few tens.
bool ExpMatchesStdOverSoftmaxRange() {
    float worst = 0.0f;
    for (int i = 0; i <= 4000; ++i) {
        const float x = -80.0f + 0.02f * static_cast<float>(i); // [-80, 0]
        const float e = RelErr(smlp::exp_f32(x), std::exp(static_cast<double>(x)));
        if (e > worst) worst = e;
    }
    if (!(worst < kRelTol)) return false;
    return true;
}

bool ExpMatchesStdOverPositiveRange() {
    float worst = 0.0f;
    for (int i = 0; i <= 1770; ++i) {
        const float x = 0.05f * static_cast<float>(i); // [0, 88.5]
        const float e = RelErr(smlp::exp_f32(x), std::exp(static_cast<double>(x)));
        if (e > worst) worst = e;
    }
    if (!(worst < kRelTol)) return false;
    return true;
}

bool ExpSpecialValues() {
    if (!(smlp::exp_f32(0.0f) == 1.0f)) return false;
    if (!(smlp::exp_f32(-200.0f) == 0.0f)) return false;
    if (!(std::isinf(smlp::exp_f32(100.0f)))) return false;
    if (!(std::isnan(smlp::exp_f32(std::numeric_limits<float>::quiet_NaN())))) return false;
    return true;
}

bool LogMatchesStdOverLogSumExpRange() {
    // sum of 10 shifted exponentials lies in [1, 10]
    float worst = 0.0f;
    for (int i = 0; i <= 900; ++i) {
        const float x = 1.0f + 0.01f * static_cast<float>(i);
        const double want = std::log(static_cast<double>(x));
        const double abs_err = std::fabs(static_cast<double>(smlp::log_f32(x)) - want);
        // log(1) = 0 makes a relative bound meaningless; bound absolute error instead.
        if (static_cast<float>(abs_err) > worst) worst = static_cast<float>(abs_err);
    }
    if (!(worst < 2e-7f)) return false;
    return true;
}

bool LogMatchesStdOverWideRange() {
    float worst = 0.0f;
    for (std::size_t i = 0; i < 2000; ++i) {
        // x = 2^(40*u): spans ~1e-12 .. ~1e12 including both sides of sqrt(2)
        const float x = std::exp2(40.0f * PseudoRandom(i, 0xABCDu));
        const double want = std::log(static_cast<double>(x));
        const double abs_err = std::fabs(static_cast<double>(smlp::log_f32(x)) - want);
        const float e = static_cast<float>(abs_err / (std::fabs(want) > 1.0 ? std::fabs(want) : 1.0));
        if (e > worst) worst = e;
    }
    if (!(worst < kRelTol)) return false;
    return true;
}

bool LogSpecialValues() {
    if (!(smlp::log_f32(1.0f) == 0.0f)) return false;
    if (!(std::isinf(smlp::log_f32(0.0f)) && smlp::log_f32(0.0f) < 0.0f)) return false;
    if (!(std::isnan(smlp::log_f32(-1.0f)))) return false;
    if (!(std::isinf(smlp::log_f32(std::numeric_limits<float>::infinity())))) return false;
    // denormal input: ln(2^-140)
    const float denorm = std::ldexp(1.0f, -140);
    if (!(std::fabs(smlp::log_f32(denorm) - static_cast<float>(std::log(static_cast<double>(denorm)))) < 1e-4f)) return false;
    return true;
}

// End to end on the real consumer: a 10-way cross-entropy with spread logits
// (including large negatives) against a double-precision reference.
bool CrossEntropyMatchesDoubleReference() {
    constexpr std::size_t kN = 10;
    for (std::size_t trial = 0; trial < 50; ++trial) {
        float logits[kN], target[kN] = {}, grad[kN];
        double lmax = -1e30;
        for (std::size_t i = 0; i < kN; ++i) {
            logits[i] = 15.0f * PseudoRandom(trial * kN + i, 0x1111u);
            if (logits[i] > lmax) lmax = logits[i];
        }
        const std::size_t cls = trial % kN;
        target[cls] = 1.0f;

        double sum = 0.0;
        for (std::size_t i = 0; i < kN; ++i) sum += std::exp(static_cast<double>(logits[i]) - lmax);
        const double lse = lmax + std::log(sum);
        const double want_loss = -(static_cast<double>(logits[cls]) - lse);

        const float got = loss::CategoricalCrossEntropyLogits(
            target, logits, grad, kN, 1.0f);
        if (!(std::fabs(static_cast<double>(got) - want_loss) < 1e-5 * (1.0 + std::fabs(want_loss)))) return false;
        for (std::size_t i = 0; i < kN; ++i) {
            const double want_g = std::exp(static_cast<double>(logits[i]) - lse) - target[i];
            if (!(std::fabs(static_cast<double>(grad[i]) - want_g) < 1e-5)) return false;
        }
    }
    return true;
}

// Restricted-domain variants used by the loss (Step 0.5, 2d).
bool ExpNonposMatchesStd() {
    // Relative error budget: 1.1e-7 polynomial + float rounding, plus the
    // rounded log2e constant's |x|*1.9e-8*ln2 contribution, which only matters
    // far below the max logit (e^-16 is already 1e-7 of the leading term).
    auto tol = [](float x) { return 3e-7f + 1.5e-8f * std::fabs(x); };
    for (int i = 0; i <= 8600; ++i) {
        const float x = -0.01f * static_cast<float>(i); // [-86, 0]
        if (!(RelErr(smlp::exp_nonpos_f32(x), std::exp(static_cast<double>(x))) < tol(x))) return false;
    }
    for (std::size_t i = 0; i < 2000; ++i) {
        const float x = -30.0f * (PseudoRandom(i, 0x77u) * 0.5f + 0.5f);
        if (!(RelErr(smlp::exp_nonpos_f32(x), std::exp(static_cast<double>(x))) < tol(x))) return false;
    }
    return true;
}

bool ExpNonposEdges() {
    if (smlp::exp_nonpos_f32(0.0f) != 1.0f) return false;
    if (smlp::exp_nonpos_f32(-87.0f) != 0.0f) return false;
    if (smlp::exp_nonpos_f32(-100.0f) != 0.0f) return false;
    if (smlp::exp_nonpos_f32(-1000.0f) != 0.0f) return false;
    return smlp::exp_nonpos_f32(-86.0f) > 0.0f; // still representable, not denormal garbage
}

bool LogGe1MatchesStd() {
    for (int i = 0; i <= 2000; ++i) {
        const float x = 1.0f + 0.005f * static_cast<float>(i); // [1, 11]
        const double want = std::log(static_cast<double>(x));
        if (!(std::fabs(static_cast<double>(smlp::log_ge1_f32(x)) - want) < 5e-7)) return false;
    }
    if (smlp::log_ge1_f32(1.0f) != 0.0f) return false;
    return std::fabs(smlp::log_ge1_f32(10.0f) - 2.302585093f) < 5e-7f; // sum == n_classes
}

// Full-range logit sweep through the real consumer: loss and gradient against
// a double reference, including very negative logits and a one-class spike.
bool CrossEntropySweep() {
    constexpr std::size_t kN = 10;
    for (std::size_t trial = 0; trial < 200; ++trial) {
        float logits[kN], target[kN] = {}, grad[kN];
        double lmax = -1e30;
        const float span = 1.0f + 0.5f * static_cast<float>(trial % 100); // up to ~50
        for (std::size_t i = 0; i < kN; ++i) {
            logits[i] = span * PseudoRandom(trial * kN + i, 0x2222u);
            if (logits[i] > lmax) lmax = logits[i];
        }
        target[trial % kN] = 1.0f;
        double sum = 0.0;
        for (std::size_t i = 0; i < kN; ++i) sum += std::exp(static_cast<double>(logits[i]) - lmax);
        const double lse = lmax + std::log(sum);
        const double want_loss = -(static_cast<double>(logits[trial % kN]) - lse);
        const float got = loss::CategoricalCrossEntropyLogits(target, logits, grad, kN, 1.0f);
        if (!(std::fabs(static_cast<double>(got) - want_loss) < 2e-5 * (1.0 + std::fabs(want_loss)))) return false;
        for (std::size_t i = 0; i < kN; ++i) {
            const double want_g = std::exp(static_cast<double>(logits[i]) - lse) - target[i];
            if (!(std::fabs(static_cast<double>(grad[i]) - want_g) < 2e-6)) return false;
        }
    }
    return true;
}

} // namespace

// One UNIT, not one per check: microunit holds at most 32 tests
// (UnitTester::kMaximumTests) and silently drops the rest.
UNIT(FastMathMatchesReference) {
    bool ok = true;
    ok &= Check("ExpMatchesStdOverSoftmaxRange", ExpMatchesStdOverSoftmaxRange());
    ok &= Check("ExpMatchesStdOverPositiveRange", ExpMatchesStdOverPositiveRange());
    ok &= Check("ExpSpecialValues", ExpSpecialValues());
    ok &= Check("LogMatchesStdOverLogSumExpRange", LogMatchesStdOverLogSumExpRange());
    ok &= Check("LogMatchesStdOverWideRange", LogMatchesStdOverWideRange());
    ok &= Check("LogSpecialValues", LogSpecialValues());
    ok &= Check("CrossEntropyMatchesDoubleReference", CrossEntropyMatchesDoubleReference());
    ok &= Check("ExpNonposMatchesStd", ExpNonposMatchesStd());
    ok &= Check("ExpNonposEdges", ExpNonposEdges());
    ok &= Check("LogGe1MatchesStd", LogGe1MatchesStd());
    ok &= Check("CrossEntropySweep", CrossEntropySweep());
    ASSERT_TRUE(ok);
    PASS();
}
