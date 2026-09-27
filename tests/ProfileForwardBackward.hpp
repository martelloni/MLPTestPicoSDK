/**
 * @file ProfileForwardBackward.hpp
 * @brief Standalone per-layer cycle-count profiler for one forward+backward
 * pass of the real MLPOpticalRecognition network (RUN_TESTS_OR_BENCHMARKS=profile).
 *
 * Built to answer a question static disassembly comparison couldn't settle:
 * after reverting Optimisation 4 (backprop loop interchange), a later
 * attempt at Optimisation 5 (layer-0 skipping its input copy) still measured
 * slower than the plain baseline, even though the "after" disassembly showed
 * no obviously-worse loop structure and no missing/added vfma.f32/vldr
 * instructions in the hot per-weight loops. This tool settled it: per-stage
 * DWT cycle timing showed layers whose logic Optimisation 5 never touched
 * still slowed down, proving the regression was the compiler's whole-function
 * register allocation reacting to a new accessor's mere presence, not any
 * instruction-level cost -- a class of regression static disassembly alone
 * cannot catch. Optimisation 5 was reverted as a result; this profiler stays,
 * at the same per-layer granularity StaticMLP's private
 * forward_layer<I>/backprop_accumulate<I> recursion uses internally --
 * reimplemented here one call at a time via the public layer<I>() accessor,
 * so each stage can be bracketed individually.
 */

#ifndef PROFILE_FORWARD_BACKWARD_HPP
#define PROFILE_FORWARD_BACKWARD_HPP

namespace test {
namespace profile {

/// Runs kIterations forward+backward passes over real dataset samples,
/// timing each layer's forward() and AccumulateGradients() call (plus the
/// loss/gradient computation between them) with the DWT cycle counter, and
/// prints avg/max cycles per stage over UART/USB stdout.
void RunForwardBackwardProfile();

} // namespace profile
} // namespace test

#endif // PROFILE_FORWARD_BACKWARD_HPP
