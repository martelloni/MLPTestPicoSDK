What I found before planning
I inspected the actual compiled RP2350 code, not just the source. Facts that drive every recommendation below:

Network under test: smlp::StaticMLP<float, Layout<64,64,32,10>, Activations<RELU,RELU,LINEAR>, LOSS_CATEGORICAL_CROSSENTROPY> (MLPOpticalRecognition.hpp:44-52) — 6,464 weights total (4096 + 2048 + 320).
Compile flags (from build-core0/compile_commands.json): -mcpu=cortex-m33 -march=armv8-m.main+fp+dsp -mfloat-abi=softfp -O3. Hardware single-precision FPU (FPv5-SP) is in use — vfma.f32, vsqrt.f32, vdiv.f32 all appear directly in the disassembly, no software float emulation.
The arm_dot_prod_f32 branch in StaticLayer.h:191-195 is dead code. ARM_MATH_CM33 is never defined anywhere in the build (no CMSIS-DSP is linked), so forward() always takes the plain scalar-loop #else branch. This matters: what's actually running is a naive scalar reduction, which is both bad news (no SIMD help from that path today) and good news (there's real headroom).
Because SMLP_CODE_ATTR no longer forces noinline/-O2 (fixed after the measured 1.7x regression recorded in plan-mlpCoreLocalPrompt.md), GCC now inlines the whole layer stack into MLPOpticalRecognition::Train, so I disassembled that function directly (build-core0/MLPTestPicoSDK.elf, symbol MLPOpticalRecognition::Train(unsigned long, float), 6.7 KB of code) rather than the individual (inlined-away) StaticLayer methods.
I'll present optimizations inside-out: innermost MAC loop first, then loop structure, then call/ABI boundaries. All code line references are to the current source.

Optimization 1 — Branchless ReLU via fmaxf (bit-exact, do this first)
Theory. utils::relu (Utils.h:146-150) is (x > 0) ? x : slope*x. The disassembly shows this compiles to a compare + flag move + conditional branch + multiply:


20007c56: vcmpe.f32 s15, #0.0
20007c5a: vmrs      APSR_nzcv, fpscr
20007c64: bgt.n     20007c6a
20007c66: vmul.f32  s15, s15, s16      @ only on the not-taken path
For 0 < slope < 1, max(x, slope*x) is exactly relu(x) for every real x: when x>0, x > slope*x so the max picks x; when x<0, slope*x is less negative than x so the max picks slope*x; at x=0 both are 0. No new rounding is introduced — it's a different way of selecting between the same two already-computed values, so this is bit-identical, not an approximation. deriv_relu similarly reduces from a compare+branch to a vcmpe+vsel selection between two constants.

Implementation guide. In mlp/Utils.h:


template<typename T>
MLP_ACTIVATION_FN
inline T relu(T x) {
    T scaled = kReLUSlope * x;
    return (x > scaled) ? x : scaled;   // lowers to vcmpe+vselgt, no branch
}
Leave deriv_relu as its existing ternary — GCC already lowers plain ternaries between two values/constants to vsel* when there's no side effect (you can see it does exactly this elsewhere, e.g. the RMSProp maxLR clamp at [train.dis:20008d52] uses vselgt.f32); the branchy codegen for relu specifically comes from the slope*x computation sitting on the not-taken path, which the rewrite above removes by computing it unconditionally. Rebuild and confirm with objdump -d that vselgt.f32/vselge.f32 replaces the bgt.n/vmul.f32 pair and no vmrs/branch remains in the ReLU site.

Expected saving. Per activation: removes 1 conditional branch and the vmrs flag transfer (M-profile flag-to-GPR moves cost a pipeline sync cycle), replacing with vmul.f32 (unconditional, was already computed on the branch anyway in the worst case) + vcmpe+vselgt (~2 cycles, no misprediction risk). Called NOut times per layer per sample for the two RELU layers = 64+32 = 96 times/sample. At an estimated 3-5 cycles saved per call (branch-not-taken is cheap on M33's branch predictor, but branch-taken/misprediction is not, and vmrs has a known pipeline-drain cost), this is roughly 300-500 cycles/sample, i.e. ~1.1M-1.9M cycles over one 3823-sample epoch — small next to the MAC loop below, but free (bit-exact, zero risk) and it also speeds up every future inference call, since ReLU dominates activation calls in GetOutput.

Optimization 2 — __restrict the hot-path pointer parameters
Theory. StaticLayer.h's forward, UpdateImmediate, AccumulateGradients, and CalcGradients all take raw const T*/T* parameters (input, deriv_err, delta_out, plus the member arrays m_weights/m_grad_accum accessed via .data()). Without restrict, the compiler must assume any of these pointers could alias each other (e.g. that writing delta_out[j] could change what w[j] reads next iteration), which blocks keeping values in registers across iterations and blocks the loop-interchange in Optimization 4 from being profitable/legal in the compiler's eyes. This is a zero-cost, zero-risk annotation — the call sites never actually pass aliasing pointers (m_buf_a/m_buf_b ping-pong buffers and m_weights/m_grad_accum are always distinct arrays).

Implementation guide. In mlp/StaticLayer.h:


SMLP_CODE_ATTR inline void forward(const T* __restrict input, T* __restrict output) { ... }

SMLP_CODE_ATTR inline void UpdateImmediate(const T* __restrict input, const T* __restrict deriv_err,
                            float lr, T* __restrict delta_out) { ... }

SMLP_CODE_ATTR inline void AccumulateGradients(const T* __restrict input, const T* __restrict deriv_err,
                                                 T* __restrict delta_out) { ... }

inline void CalcGradients(const T* __restrict /*input*/, const T* __restrict deriv_err, T* __restrict delta_out) { ... }
__restrict is a GCC/Clang extension; it's already implicitly relied upon elsewhere in this codebase's toolchain (arm-none-eabi-g++), so no portability concern. Also add it to the T* w/T* g locals derived from .data() inside the bodies, since those are the pointers actually walked in the loops.

Expected saving. This alone yields no direct cycle count — it's an enabler. It's listed here (not folded into Optimization 4) because it must land first: Optimization 4's loop interchange is only safe for the compiler to auto-vectorize/keep-in-register once it knows delta_out, w, input, g don't overlap. Expect this to make Optimizations 3 and 4 measurably more effective (fewer defensive reloads), typically worth 5-15% on top of them in my experience with GCC -O3 on similar reduction loops — but treat that figure as a hypothesis to confirm by diffing disassembly before/after, not a hard number.

Optimization 3 — Break the FMA dependency chain in the forward dot product
Theory. The actual inner loop for e.g. layer 0's 64-wide dot product (StaticLayer.h:196-198, i.e. for (j) sum += w[j]*input[j]) compiles to:


20007c46: vldmia ip!, {s12}      @ load w[j]
20007c4a: vldmia lr!, {s14}      @ load input[j]
20007c4e: cmp    ip, r0
20007c50: vfma.f32 s15, s12, s14 @ s15 += s12*s14   <- s15 carries across iterations
20007c54: bne.n  20007c46
s15 is a single accumulator with a genuine loop-carried dependency: iteration j+1's vfma cannot issue until iteration j's vfma has produced its result, not merely been issued. Cortex-M33's FPv5-SP VFMA.F32 has multi-cycle result latency (typically several cycles on this class of core, non-pipelined for the purposes of a single dependent chain — the pipeline can accept a new instruction each cycle, but this loop can't feed it one until the previous result is ready). So the loop's throughput is bound by FMA latency, not by instruction issue width or memory bandwidth — the vldmia/cmp/bne around it are "free" (they execute during the latency bubble) but the accumulate chain itself stalls the core every iteration.

The fix is the standard one for reduction loops on any pipelined FPU: split the single accumulator into k independent accumulators, so k FMAs with no data dependency between them can be in flight simultaneously, hiding the latency; sum the k partials once at the end.

Implementation guide. In StaticLayer.h:170-205, replace the float MAC body with a 4-way-unrolled, 4-accumulator version, keeping a scalar remainder loop for NIn % 4 != 0:


} else {
    T acc0 = m_biases[i], acc1 = T(0), acc2 = T(0), acc3 = T(0);
    std::size_t j = 0;
#if defined(ARM_MATH_CM33) && defined(__arm__)
    // (unreachable today -- ARM_MATH_CM33 is never defined in this build;
    // left here only if a future build wires up CMSIS-DSP)
#endif
    for (; j + 4 <= NIn; j += 4) {
        acc0 += w[j + 0] * input[j + 0];
        acc1 += w[j + 1] * input[j + 1];
        acc2 += w[j + 2] * input[j + 2];
        acc3 += w[j + 3] * input[j + 3];
    }
    T sum = (acc0 + acc1) + (acc2 + acc3);
    for (; j < NIn; ++j) sum += w[j] * input[j];
Note the accumulation-tree summation (acc0+acc1)+(acc2+acc3) rather than acc0+acc1+acc2+acc3 — same value, but expressed so the compiler doesn't need to reassociate under -ffast-math (which isn't enabled here, and shouldn't be, since the plan explicitly requires bit-reproducibility). Caveat: this changes the floating-point summation order, so results are no longer bit-identical to the current single-accumulator sum (floating-point addition isn't associative). This needs the tolerance-based unit test the task description calls for — see "Testing" below. Given NIn ∈ {64, 32, 10} here, 4-way unrolling with a scalar tail handles all three layers cleanly (10 % 4 = 2 tail iterations).

Apply the identical restructuring to the es-scaled accumulation pattern is not needed here — this optimization is specifically for the single-scalar reduction in forward(). The gradient loops have a different (fixable-differently) problem, addressed next.

Expected saving. With 4 independent chains, the loop's steady-state throughput moves from "1 MAC per FMA-latency cycles" to "1 MAC per ~1 cycle" (bound by the 2 loads + compare + branch now, which the core can substantially overlap with FMA issue across the 4 chains), i.e. roughly a 2-3x speedup on the pure MAC loop, tapering to ~1.5-2x once loop overhead (now amortized 4:1 instead of 1:1) and load bandwidth are accounted for. Over NIn*NOut = 6464 MACs/sample × 3823 samples/epoch × forward pass only, this loop is the single largest cost center in the whole training step; a conservative 1.6x on it is a substantial fraction of total Train() time. Verify empirically — the RP2350 has a DWT cycle counter (hw_set_bits/DWT->CYCCNT via the SDK, or time_us_64() bracketing) already implied by your benchmark harness; wrap a single forward_layer<0> call before/after this change and compare cycle counts directly rather than trusting the estimate above.

Optimization 4 — Loop-interchange the backward gradient-accumulation loops
Theory. This is the highest-confidence, highest-value finding from the disassembly. AccumulateGradients (StaticLayer.h:241-255) is:


for (i = 0; i < NOut; ++i) {
    T es = ...;
    for (j = 0; j < NIn; ++j) {
        g[j] += input[j] * es;          // g advances by NIn each outer i -> written once, not a reduction
        delta_out[j] += es * w[j];      // delta_out is REUSED across all i -> genuine NOut-way reduction
    }
    w += NIn; g += NIn;
}
The disassembly confirms exactly this, and shows the cost: because delta_out[j] is a true accumulator across the outer loop, and the current loop nesting puts i outermost, delta_out[j] cannot stay resident in a register across the whole reduction (it's touched, then the loop moves to j+1, j+2, ... and only comes back to the same delta_out[j] on the next outer iteration, NIn iterations later). So every single (i,j) pair does a full load-modify-store round trip through memory for delta_out[j]:


200081d2: vldmia r3!, {s12}   @ w[j]
200081d6: vldr   s15, [r1]    @ RELOAD delta_out[j] from memory
200081da: vfma.f32 s15, s12, s14
200081de: vstmia r1!, {s15}   @ STORE delta_out[j] back to memory
That's NOut * NIn = up to 4096 redundant load+store pairs per layer per sample, when algorithmically only NIn values ever need to leave a register.

The fix: swap the loop nest. Precompute all es[i] values first (cheap, NOut ≤ 64 floats, fits easily on the stack), then loop j outermost and i innermost, accumulating delta_out[j] in a single register across the full inner loop and writing it to memory exactly once. The access pattern for w[i*NIn+j] becomes column-strided (stride NIn*4 bytes) instead of row-contiguous — normally a bad trade, but RP2350's SRAM has no data cache (flat, uniform-latency access across the whole address space on the Cortex-M33 AHB/AXI fabric), so this "worse locality" costs nothing on this specific target. This is exactly the kind of platform-specific reasoning the plan calls for.

The same restructuring applies verbatim to UpdateImmediate (used by ApplyLoss, i.e. the RL/ApplyLoss path) and CalcGradients (used by RL's CalcGradients/autograd path) — both have the identical delta_out[j] += es * w[j] reduction-over-i shape.

Implementation guide. In StaticLayer.h:241-255:


SMLP_CODE_ATTR inline void AccumulateGradients(const T* __restrict input, const T* __restrict deriv_err,
                                                 T* __restrict delta_out) {
    // Precompute es[i] once; NOut is small (<=64 here) so this is stack-cheap.
    T es[NOut];
    for (std::size_t i = 0; i < NOut; ++i) {
        es[i] = deriv_err[i] * activate_deriv_cached<Act>(m_act_output[i], m_inner_products[i]);
        m_bias_grad_accum[i] += es[i];
    }
    const T* w = m_weights.data();
    T* g = m_grad_accum.data();
    // g[i*NIn+j] is written exactly once per (i,j) -- unaffected by loop order,
    // keep the natural row-major walk for it.
    for (std::size_t i = 0; i < NOut; ++i) {
        for (std::size_t j = 0; j < NIn; ++j) g[i * NIn + j] += input[j] * es[i];
    }
    // delta_out[j] is a genuine NOut-way reduction: put j outermost so it
    // accumulates in a register across the whole inner loop instead of a
    // load-modify-store round trip through memory on every (i,j).
    for (std::size_t j = 0; j < NIn; ++j) {
        T acc = T(0);
        for (std::size_t i = 0; i < NOut; ++i) acc += es[i] * w[i * NIn + j];
        delta_out[j] = acc;
    }
}
Mirror the same es[]-precompute-then-j-outer structure in UpdateImmediate (there w[j] -= lr*es*input[j] stays in the natural i-outer walk since it's a per-cell, non-reduced write; only the delta_out loop gets interchanged) and in CalcGradients. Note delta_out[j] = acc (assignment, not the current += -after-zero-fill pattern) — this also removes the separate for (j) delta_out[j] = T(0); zeroing loop that currently precedes the accumulation in all three functions, since the new loop always fully initializes each delta_out[j] itself.

Expected saving. This is the biggest win in the plan. It converts NOut × NIn memory round trips (e.g. 4096 for layer 0) into NOut × NIn register-resident FMAs plus just NIn stores — i.e. it removes roughly 2 × NOut × NIn load/store instructions (each ~1-2 cycles on this core) network-wide, on top of also making the accumulation chain vectorizable/unrollable the same way as Optimization 3 (the inner i loop is now itself a single-accumulator reduction and should get the same 4-way-accumulator treatment). Combined, expect this loop to go from "dependency-chain-stalled with per-element memory traffic" (~6-8 cycles/element by the disassembly's instruction count) to "~2-3 cycles/element" once both changes land — call it a 2.5-3x reduction on the dominant cost of AccumulateGradients/UpdateImmediate/CalcGradients, which collectively are exercised by TrainBatch, ApplyLoss (RL), and CalcGradients (RL autograd) — i.e. every non-inference entry point in the library. This is the one to validate most carefully on-target given its size.

Optimization 5 — Skip the input-layer's memcpy cache-copy
Theory. StaticLayer::forward unconditionally copies its input into m_cached_input when training (StaticLayer.h:171-173):


if constexpr (EnableTraining) { for (j) m_cached_input[j] = input[j]; }
For layer 0 in the zero-copy TrainBatch overload, input is features + s * feature_stride — a pointer directly into the flash-resident dataset (StaticMLP.h:406-408). The disassembly confirms GCC recognized the 256-byte copy loop and replaced it with an actual bl memcpy call:


20007c12: ldr r3, [pc, #124]   @ = 0x10030658  (flash address of dataset::features)
...
20007c28: bl 2000724c <memcpy>
That flash row is read-only and doesn't change until the next sample is loaded (cursor advances only after this sample's forward+backward completes), so the copy is pure overhead for layer 0 specifically: the backward pass could read input[j] directly from the original flash pointer instead of from a RAM copy. Hidden layers (1, 2) genuinely need the copy, because their "input" is a ping-pong scratch buffer (m_buf_a/m_buf_b) that later layers overwrite before backprop runs.

Implementation guide. This needs a small, deliberate API change rather than a blind deletion, since StaticLayer::forward can't know at compile time whether its caller's buffer will outlive the backward pass. Add a bool CacheInputByCopy compile-time flag (or a same-named runtime bool defaulting true) to forward(), and have StaticMLP::forward_layer<0> — the only call site where the input pointer is caller-owned and guaranteed stable through backprop — pass false, storing the pointer instead:


// StaticLayer.h
std::array<T, EnableTraining ? NIn : 0> m_cached_input{};
const T* m_input_ptr = nullptr;   // used only when CacheInputByCopy == false

template<bool CacheInputByCopy = true>
SMLP_CODE_ATTR inline void forward(const T* __restrict input, T* __restrict output) {
    if constexpr (EnableTraining) {
        if constexpr (CacheInputByCopy) { for (j) m_cached_input[j] = input[j]; }
        else                             { m_input_ptr = input; }
    }
    ...
}
const T* cached_input() const { return CacheInputByCopy ? m_cached_input.data() : m_input_ptr; }
Then in StaticMLP.h:661-667, call std::get<0>(m_layers).forward<false>(in, out) for layer 0 specifically (the recursive template already distinguishes I==0), and update backprop_accumulate<0>/backprop_immediate<0>/calc_grad_impl<0> to read layer.cached_input() instead of layer.m_cached_input.data() for that layer. This is a source change confined to the I==0 path in StaticMLP.h, and it also directly benefits Train() (the per-sample overload) and ApplyLoss/CalcGradients (RL), which all call forward_layer<0>(feat) with a caller-owned, backprop-stable pointer.

Expected saving. One bl memcpy call (with its own prologue/epilogue and byte/word-copy loop overhead) plus 256 bytes of flash→RAM traffic, eliminated per sample for layer 0. A memcpy of this size typically costs on the order of a few dozen cycles including call overhead even when word-aligned; over 3823 samples/epoch this is a modest few-tens-of-thousands of cycles/epoch — small next to Optimizations 3-4, but essentially free once the plumbing exists, and it removes flash-XIP read traffic that (unlike SRAM) is not necessarily single-cycle on this part.

Optimization 6 — Replace RMSProp's per-weight vsqrt+vdiv with a Newton-Raphson rsqrt (training only, needs a tolerance test)
Theory. ApplyAccumulatedGradients (StaticLayer.h:283-290, 302-306) computes, for every one of the 6,464 weights (plus biases) in the network, once per batch:


adj = lr / (nn::sqrt(m_sq_grad_avg[k]) + rmsPropEpsilon);
The disassembly shows this as a direct hardware pair — vsqrt.f32 then vdiv.f32 — and I count roughly 20 distinct unrolled occurrences of this exact pair in the batch-update section of Train(), confirming GCC already partially loop-unrolled this reduction. VSQRT.F32 and VDIV.F32 on FPv5-SP are the two most expensive single-precision instructions on this FPU: unlike VADD/VMUL/VFMA, they are not pipelined — each one occupies the FPU divider/sqrt unit for its full multi-cycle latency (commonly cited as ~14 cycles each for FPv5-SP on this class of core) before the FPU can start the next one, so a vsqrt+vdiv pair costs roughly an order of magnitude more than a vfma. This runs kWeights + NOut times per layer, every batch (30 batches/epoch × 3823/128 ≈ 30), i.e. ~6,464 × 30 ≈ 194,000 sqrt+div pairs per epoch just for RMSProp.

The fixed-point path in this same codebase already solves exactly this problem with nn::rsqrt (FixedNN.h:138-159) — a 6-iteration Newton-Raphson reciprocal-square-root using only multiplies. For float, the classic approach (the famous "fast inverse square root") gets an initial estimate via a bit-level reinterpretation, then refines with 1-2 Newton iterations of y = y * (1.5 - 0.5*x*y*y), each iteration costing ~2 vmul/vfma (each ~1-4 cycles) instead of ~28 cycles for vsqrt+vdiv.

Implementation guide. Add a float-specialized rsqrt to nn:: in FixedNN.h (or a new small header, since this is a float-only helper — FixedNN.h's existing nn::sqrt already branches on is_fixed_point_v<T>, so this slots into the else arm):


template<typename T> SMLP_CODE_ATTR inline T rsqrt(T x) {
    if constexpr (is_fixed_point_v<T>) { /* existing fixed path, unchanged */ }
    else {
        static_assert(std::is_same_v<T, float>);
        float xhalf = 0.5f * x;
        int32_t i;
        std::memcpy(&i, &x, sizeof(i));
        i = 0x5f3759df - (i >> 1);
        float y;
        std::memcpy(&y, &i, sizeof(y));
        y = y * (1.5f - xhalf * y * y);   // Newton iteration 1
        y = y * (1.5f - xhalf * y * y);   // Newton iteration 2 (accuracy for RMSProp)
        return y;
    }
}
Then in StaticLayer.h:283-290, replace adj = lr / (nn::sqrt(v) + eps) with adj = lr * nn::rsqrt(v + eps*eps) — note this changes the epsilon's role slightly (added under the square root rather than after it, to avoid a second division); pick whichever epsilon placement you want to standardize on and apply it identically to the bias branch. This is not bit-exact — the two Newton iterations converge to within ~1e-6 relative error of the true 1/sqrt(x), not exactly. Since this feeds into an adaptive learning-rate multiplier that's already clamped (adj > maxLR) and epsilon-guarded, a ~1e-6 relative perturbation is very unlikely to change training trajectories meaningfully, but per the task's own instructions this must be validated, not assumed.

Testing. Add tests/unit/RMSPropRsqrtTest.cpp: compare nn::rsqrt(x) against 1.0f/std::sqrt(x) over a sweep of representative x values (RMSProp's m_sq_grad_avg range, i.e. small positive floats from squared clipped gradients, roughly [1e-8, 100]), asserting relative error < 1e-5. Separately, run a short deterministic MLPOpticalRecognition::Train (the existing 20-epoch smoke config, seed 0xC0DEu) with both the old and new ApplyAccumulatedGradients, and assert the final loss/accuracy are within a documented tolerance (e.g. loss within 1%, accuracy within 1 percentage point) rather than requiring bit-identity — this directly follows the task's guidance for when a numeric change is "OK" but must be tolerance-tested.

Expected saving. vsqrt.f32+vdiv.f32 (~28 cycles, non-pipelined) → 2 Newton iterations (~4 vfma/vmul each, pipelined, ~2-3 cycles/op with some latency overlap across the unrolled weights) ≈ 8-12 cycles. That's roughly a 2.5-3x reduction on this specific per-weight cost, applied 194,000 times/epoch — on the order of a few million cycles/epoch saved. Scope note: this only benefits TrainBatch's RMSProp step, not inference or ApplyLoss's immediate-SGD path (which has no sqrt/divide at all), so it's narrower in applicability than Optimizations 1-4, but it's the single most expensive individual instruction pair in the whole training loop, so it's included despite that.

Optimization 7 — Build-wide: -mfloat-abi=hard (broadest scope, higher risk, do last)
Theory. The build uses -mfloat-abi=softfp: the FPU is used for the actual arithmetic, but the calling convention still passes/returns float/double values in general-purpose registers (r0-r3), per the soft-float AAPCS variant — for ABI compatibility with any code that might not have an FPU. This means every function call or return crossing a translation-unit boundary (or any call GCC doesn't inline) that passes or returns a float value pays a VMOV round-trip between core and FPU registers on both sides. Today's fully-inlined Train() hides most of this for the hot loop itself, but it still hits: the MLPOpticalRecognition::Train/Predict/Initialise entry points themselves, any libm call (expf/logf for softmax and categorical cross-entropy — visible in the disassembly as bl __wrap_expf with a vmov s14, r0 right after, at [train.dis:20008108-20008110]), and critically, any future network using SIGMOID/TANH activations, which call std::exp/std::tanh once per node in the inner activation evaluation, each paying this tax. This is the change with the widest blast radius across "other applications" (inference, RL, any non-ReLU network) but also the one requiring the most care, since it changes the ABI for the whole link — the Pico SDK and any prebuilt libraries linked in must agree.

Implementation guide. In CMakeLists.txt, the Pico SDK's pico_stdlib/pico_float components support hard-float variants (pico_float_pico vs a hardfp alternative) — this needs verifying against SDK 2.3.0's supported float-ABI options before flipping the flag project-wide, since an ABI mismatch between your TU and a prebuilt SDK object is a silent-corruption risk (arguments in the wrong registers), not a compile error, if any prebuilt .a was built softfp-only. Concretely:

Check whether PICO_SDK's newlib/libc and any prebuilt archives are softfp-only or offer a hardfp build; RP2350 toolchains typically ship both.
If available, add target_compile_options(MLPTestPicoSDK PRIVATE -mfloat-abi=hard) and confirm every dependency (pico_stdlib, pico_multicore, newlib) is relinked/rebuilt consistently — CMake's Pico SDK integration rebuilds SDK sources from source per-target, so this is more tractable than it would be against a prebuilt vendor blob, but still needs a full clean rebuild and the validate_memory_placement.py/unit-test gate to pass.
Rebuild all three configs (build-core0, build-core1, build-naive) and re-run RunAndSaveBenchmarks.py — this is exactly the kind of change that could interact with the core-independence guarantees Steps 6-7 spent so much effort proving, so it should go through that same benchmark gate, not just a functional smoke test.
Expected saving. Each ABI-crossing float call/return avoids 1-2 vmov instructions (~1-2 cycles each) on both the caller and callee side. For the current ReLU-only network this is a small, diffuse win (mostly at the Train/Predict entry points, called rarely relative to the inner loop). For a SIGMOID/TANH network it would be much larger, since expf/tanhf calls happen NOut times per layer per sample and are not inlined. Given the narrow win for this specific network and the non-trivial verification burden (full SDK ABI consistency, re-running the core-independence benchmark suite), I'd sequence this last, and would frame it to you as optional/deferred unless a SIGMOID/TANH-activated network is on the near-term roadmap.

Suggested implementation order
Opt 1 (branchless ReLU) — bit-exact, trivial, do immediately.
Opt 2 (__restrict) — bit-exact, trivial, unblocks 3/4.
Opt 4 (loop interchange) — bit-exact-adjacent (associativity change, but no reordering of which terms sum, just the order libm sees them in — actually this one does reorder the NOut-way sum, so treat it with the same tolerance test as Opt 3/6), biggest single win, touches AccumulateGradients/UpdateImmediate/CalcGradients (training + RL).
Opt 3 (multi-accumulator forward) — second-biggest win, same reordering caveat, touches forward() (inference + everything downstream).
Opt 5 (skip layer-0 copy) — small, mostly-mechanical plumbing change.
Opt 6 (RMSProp rsqrt) — training-only, needs the explicit tolerance unit test called for in the task brief.
Opt 7 (hard-float ABI) — evaluate last; verify SDK compatibility before committing.
For 3, 4, and 6 — anything that changes floating-point summation/computation order — add the tolerance-based tests under tests/unit/ before or alongside the change, per your existing testing discipline, and re-run RunAndSaveBenchmarks.py to confirm the deterministic-seed accuracy/loss trajectory stays within an agreed tolerance rather than silently drifting.
