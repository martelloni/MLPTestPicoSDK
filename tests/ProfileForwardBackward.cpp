#include "tests/ProfileForwardBackward.hpp"

#include <array>
#include <cstdint>
#include <cstdio>

#include "pico/stdlib.h"
#include "hardware/sync.h" // save_and_disable_interrupts/restore_interrupts
#include "RP2350.h" // CMSIS DWT/DCB -- DWT->CYCCNT, DCB->DEMCR

#include "MemoryDefs.hpp"
#include "experiments/MLPOpticalRecognition.hpp" // MLPOpticalRecognition::Net (64-64-32-10)
#include "dataset/Dataset.hpp"

namespace test {
namespace profile {

namespace {

/// Enables the Cortex-M33 DWT cycle counter. DEMCR.TRCENA gates the whole
/// debug/trace unit (DWT included); without it CYCCNT silently stays frozen
/// at 0 (or its reset value) rather than faulting, which would look like
/// "every stage takes 0 cycles" rather than an obvious error.
void EnableCycleCounter() {
    DCB->DEMCR |= DCB_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

inline uint32_t Cycles() { return DWT->CYCCNT; }

/// Running avg/max over however many samples get add()ed; avg divides by
/// count directly rather than accumulating a running mean, since kIterations
/// is small (100) and sum fits comfortably in a uint64_t at these magnitudes.
struct Stat {
    uint64_t sum = 0;
    uint32_t max_val = 0;
    uint32_t count = 0;

    void add(uint32_t cycles) {
        sum += cycles;
        if (cycles > max_val) max_val = cycles;
        ++count;
    }
    float avg() const {
        return count ? static_cast<float>(static_cast<double>(sum) / static_cast<double>(count)) : 0.0f;
    }
};

/// Prints one stage's address for the placement record. The addr%16 column is
/// what plan-mlpBetterOptimisation.md section 1.6 needs to correlate cycle
/// counts with SRAM bank phase; RunAndSaveBenchmarks.py parses these lines
/// ("placement data <name> addr=0x... mod16=N") into placement_results.csv.
void PrintPlacement(const char* name, const void* p) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    printf("  placement data %-28s addr=0x%08lx mod16=%lu\n",
           name, static_cast<unsigned long>(a), static_cast<unsigned long>(a & 15u));
}

void PrintStage(const char* name, const Stat & s) {
    printf("  %-28s avg=%9.1f cycles   max=%8lu cycles\n",
           name, static_cast<double>(s.avg()), static_cast<unsigned long>(s.max_val));
}

} // namespace

void RunForwardBackwardProfile() {
    using Net = MLPOpticalRecognition::Net;

    // Network state alone is tens of KB (weights + RMSProp accumulators for
    // all three layers) -- far too large for the default stack, so this is a
    // function-local static (.bss), exactly like the real experiment's own
    // State member.
    static Net net{};
    net.SetSeed(0xC0DEu);
    net.InitXavier();

    if (DWT->CTRL & DWT_CTRL_NOCYCCNT_Msk) {
        printf("Profiling aborted: this core's DWT does not implement a cycle counter.\n");
        return;
    }

    constexpr int kIterations = 100;

    auto & layer0 = net.layer<0>();
    auto & layer1 = net.layer<1>();
    auto & layer2 = net.layer<2>();

    // Scratch buffers sized to the widest layer (64) so one set covers every
    // inter-layer handoff; only the first kIn/kOut elements of each are ever
    // read/written for a given layer.
    std::array<dataset::DataType, Net::kMaxWidth> buf_a{}, buf_b{};
    std::array<dataset::DataType, dataset::kLabelSize> pred{}, deriv{};
    std::array<dataset::DataType, Net::kMaxWidth> delta_a{}, delta_b{};
    // layer0.AccumulateGradients() always writes NIn (64) delta values even
    // though there is no earlier layer left to consume them.
    std::array<dataset::DataType, Net::kMaxWidth> delta_layer0{};

    Stat fwd0, fwd1, fwd2, fwd_total;
    Stat loss_stage;
    Stat bwd2, bwd1, bwd0, bwd_total;
    Stat full_total;

    EnableCycleCounter();

    for (int it = 0; it < kIterations; ++it) {
        const std::size_t s = static_cast<std::size_t>(it) % dataset::kNumExamples;
        const dataset::DataType* feat = dataset::features[s].data();
        const dataset::DataType* label = dataset::labels[s].data();

        // Mask interrupts for the timed region only: DWT->CYCCNT counts ISR
        // cycles too, and stdio_usb's periodic USB SOF handling landing
        // inside one particular bracket (rather than another) would show up
        // as a multi-thousand-cycle difference attributed to whichever
        // stage it happened to interrupt -- not a real difference in that
        // stage's own cost. ~1.1-1.2k cycles/iteration disabled, well under
        // anything that would stall the USB link.
        const uint32_t ints = save_and_disable_interrupts();

        uint32_t t0 = Cycles();
        layer0.forward(feat, buf_a.data());
        const uint32_t t_fwd0 = Cycles() - t0;

        t0 = Cycles();
        layer1.forward(buf_a.data(), buf_b.data());
        const uint32_t t_fwd1 = Cycles() - t0;

        t0 = Cycles();
        layer2.forward(buf_b.data(), pred.data());
        const uint32_t t_fwd2 = Cycles() - t0;

        t0 = Cycles();
        smlp::compute_loss<loss::LOSS_FUNCTIONS::LOSS_CATEGORICAL_CROSSENTROPY>(
            label, pred.data(), deriv.data(), dataset::kLabelSize, dataset::DataType(1));
        const uint32_t t_loss = Cycles() - t0;

        t0 = Cycles();
        layer2.AccumulateGradients(layer2.m_cached_input.data(), deriv.data(), delta_b.data());
        const uint32_t t_bwd2 = Cycles() - t0;

        t0 = Cycles();
        layer1.AccumulateGradients(layer1.m_cached_input.data(), delta_b.data(), delta_a.data());
        const uint32_t t_bwd1 = Cycles() - t0;

        t0 = Cycles();
        layer0.AccumulateGradients(layer0.m_cached_input.data(), delta_a.data(), delta_layer0.data());
        const uint32_t t_bwd0 = Cycles() - t0;

        restore_interrupts(ints);

        fwd0.add(t_fwd0);
        fwd1.add(t_fwd1);
        fwd2.add(t_fwd2);
        fwd_total.add(t_fwd0 + t_fwd1 + t_fwd2);
        loss_stage.add(t_loss);
        bwd2.add(t_bwd2);
        bwd1.add(t_bwd1);
        bwd0.add(t_bwd0);
        bwd_total.add(t_bwd2 + t_bwd1 + t_bwd0);
        full_total.add(t_fwd0 + t_fwd1 + t_fwd2 + t_loss + t_bwd2 + t_bwd1 + t_bwd0);
    }

    // ── Phase 2: per-batch stages (once per 128 samples in TrainBatch) ──
    // Measured separately from the per-sample loop above so the existing
    // rows keep their meaning. Each iteration: zero the accumulators (timed),
    // run one untimed forward+backward so the accumulators hold real
    // gradients, then time the grad-norm computation and each layer's RMSProp
    // apply. Weights drift over these iterations; timing is data-independent
    // for these stages. The conditional clip rescale (ScaleAccumulatedGradients)
    // is not measured: it runs only when the norm exceeds 5.0.
    constexpr int kBatchIterations = 20;
    constexpr std::size_t kBatchSize = 128; // matches MLPOpticalRecognition::Train
    constexpr float kLearningRate = 0.01f;  // matches TestRAMIndependence
    const dataset::DataType batch_inv = dataset::DataType(1) / static_cast<dataset::DataType>(kBatchSize);

    Stat zero_stage, norm_stage, rms0, rms1, rms2, rms_total, batch_total;

    for (int it = 0; it < kBatchIterations; ++it) {
        const std::size_t s = static_cast<std::size_t>(it) % dataset::kNumExamples;

        uint32_t ints = save_and_disable_interrupts();
        uint32_t t0 = Cycles();
        layer0.InitGradientAccumulators();
        layer1.InitGradientAccumulators();
        layer2.InitGradientAccumulators();
        const uint32_t t_zero = Cycles() - t0;
        restore_interrupts(ints);

        // Untimed: populate the accumulators with a real sample's gradients.
        layer0.forward(dataset::features[s].data(), buf_a.data());
        layer1.forward(buf_a.data(), buf_b.data());
        layer2.forward(buf_b.data(), pred.data());
        smlp::compute_loss<loss::LOSS_FUNCTIONS::LOSS_CATEGORICAL_CROSSENTROPY>(
            dataset::labels[s].data(), pred.data(), deriv.data(), dataset::kLabelSize, dataset::DataType(1));
        layer2.AccumulateGradients(layer2.m_cached_input.data(), deriv.data(), delta_b.data());
        layer1.AccumulateGradients(layer1.m_cached_input.data(), delta_b.data(), delta_a.data());
        layer0.AccumulateGradients(layer0.m_cached_input.data(), delta_a.data(), delta_layer0.data());

        ints = save_and_disable_interrupts();
        t0 = Cycles();
        dataset::DataType sumsq = dataset::DataType(0);
        sumsq += layer0.GetGradSumSquared(batch_inv);
        sumsq += layer1.GetGradSumSquared(batch_inv);
        sumsq += layer2.GetGradSumSquared(batch_inv);
        const dataset::DataType norm = smlp::nn::sqrt(sumsq);
        const uint32_t t_norm = Cycles() - t0;
        // Keep `norm` live so the compiler cannot discard the bracketed work.
        asm volatile("" : : "r"(norm) : "memory");

        t0 = Cycles();
        layer0.ApplyAccumulatedGradients(kLearningRate, batch_inv);
        const uint32_t t_rms0 = Cycles() - t0;
        t0 = Cycles();
        layer1.ApplyAccumulatedGradients(kLearningRate, batch_inv);
        const uint32_t t_rms1 = Cycles() - t0;
        t0 = Cycles();
        layer2.ApplyAccumulatedGradients(kLearningRate, batch_inv);
        const uint32_t t_rms2 = Cycles() - t0;
        restore_interrupts(ints);

        zero_stage.add(t_zero);
        norm_stage.add(t_norm);
        rms0.add(t_rms0);
        rms1.add(t_rms1);
        rms2.add(t_rms2);
        rms_total.add(t_rms0 + t_rms1 + t_rms2);
        batch_total.add(t_zero + t_norm + t_rms0 + t_rms1 + t_rms2);
    }

    printf("\nForward/backward cycle profile (%d iterations, network 64-64-32-10):\n", kIterations);
    printf("Forward:\n");
    PrintStage("layer0.forward", fwd0);
    PrintStage("layer1.forward", fwd1);
    PrintStage("layer2.forward", fwd2);
    PrintStage("forward total", fwd_total);
    printf("Loss:\n");
    PrintStage("compute_loss", loss_stage);
    printf("Backward:\n");
    PrintStage("layer2.AccumulateGradients", bwd2);
    PrintStage("layer1.AccumulateGradients", bwd1);
    PrintStage("layer0.AccumulateGradients", bwd0);
    PrintStage("backward total", bwd_total);
    printf("Full forward+loss+backward total:\n");
    PrintStage("full pass", full_total);

    printf("Per-batch stages (once per %u samples; %d iterations):\n",
           static_cast<unsigned>(kBatchSize), kBatchIterations);
    PrintStage("batch.zero_grads", zero_stage);
    PrintStage("batch.grad_norm", norm_stage);
    PrintStage("batch.rmsprop.layer0", rms0);
    PrintStage("batch.rmsprop.layer1", rms1);
    PrintStage("batch.rmsprop.layer2", rms2);
    PrintStage("batch.rmsprop total", rms_total);
    PrintStage("batch total", batch_total);
    // Amortised per sample so it compares directly with "full pass".
    {
        Stat per_sample;
        per_sample.add(static_cast<uint32_t>(batch_total.sum / batch_total.count / kBatchSize));
        PrintStage("batch total per sample", per_sample);
    }

    printf("Placement (net and kernel code addresses; code addresses are added by\n");
    printf("RunAndSaveBenchmarks.py from the ELF symbol table):\n");
    PrintPlacement("net", &net);
    PrintPlacement("layer0.m_weights", layer0.m_weights.data());
    PrintPlacement("layer0.m_grad_accum", layer0.m_grad_accum.data());
    PrintPlacement("layer0.m_cached_input", layer0.m_cached_input.data());
    PrintPlacement("layer1.m_weights", layer1.m_weights.data());
    PrintPlacement("layer1.m_grad_accum", layer1.m_grad_accum.data());
    PrintPlacement("layer1.m_cached_input", layer1.m_cached_input.data());
    PrintPlacement("layer2.m_weights", layer2.m_weights.data());
    PrintPlacement("layer2.m_grad_accum", layer2.m_grad_accum.data());
    PrintPlacement("layer2.m_cached_input", layer2.m_cached_input.data());
    PrintPlacement("delta_a", delta_a.data());
    PrintPlacement("delta_b", delta_b.data());
    PrintPlacement("delta_layer0", delta_layer0.data());
    printf("Profile complete.\n");
    printf("\n");
}

} // namespace profile
} // namespace test
