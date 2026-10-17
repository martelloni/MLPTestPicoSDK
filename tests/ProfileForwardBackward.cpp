#include "tests/ProfileForwardBackward.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "hardware/sync.h" // save_and_disable_interrupts/restore_interrupts
#include "hardware/structs/busctrl.h"
#include "hardware/structs/timer.h"
#include "RP2350.h" // CMSIS DWT/DCB -- DWT->CYCCNT, DCB->DEMCR

#include "MemoryDefs.hpp"
#include "experiments/MLPOpticalRecognition.hpp" // MLPOpticalRecognition::Net (64-64-32-10)
#include "dataset/Dataset.hpp"

#if defined(MEML_MLP_RUNS_ON_CORE) && MEML_MLP_RUNS_ON_CORE == 1
extern "C" {
extern uint32_t __core1_stack_bottom__[];
extern uint32_t __core1_stack_top__[];
}
#endif

// Which physical core runs the measurement ("probe") and which one hosts the
// bus-counter poller ("service"). The probe core is the one the MLP is pinned
// to, so the numbers are those of the production placement: core 1 for
// MEML_MLP_RUNS_ON_CORE=1, core 0 otherwise (pinned to 0, or unpinned).
#if defined(MEML_MLP_RUNS_ON_CORE) && MEML_MLP_RUNS_ON_CORE == 1
#define PROFILE_PROBE_ON_CORE1 1
#else
#define PROFILE_PROBE_ON_CORE1 0
#endif

// The service core's code must not sit in the probe core's bank, or its fetches
// would pollute the very contention being measured. Core 0's bank is the
// default RAM, so nothing to do when the service core is core 0; when it is
// core 1 in a core-0-pinned build, tag it into core 1's bank. (The unpinned
// build has no per-core sections at all.)
#if defined(MEML_MLP_RUNS_ON_CORE) && MEML_MLP_RUNS_ON_CORE == 0
#define PROFILE_SERVICE_ATTR MEML_RUNS_ON_CORE(1)
#else
#define PROFILE_SERVICE_ATTR
#endif

namespace test {
namespace profile {

namespace {

/// Enables the Cortex-M33 DWT cycle counter of the CALLING core. DEMCR.TRCENA
/// gates the whole debug/trace unit (DWT included); without it CYCCNT silently
/// stays frozen at 0 (or its reset value) rather than faulting, which would look
/// like "every stage takes 0 cycles" rather than an obvious error. CYCCNT is
/// per core, so this must run on the core that does the timing.
void EnableCycleCounter() {
    DCB->DEMCR |= DCB_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

inline uint32_t Cycles() { return DWT->CYCCNT; }

/// Running avg/max over however many samples get add()ed; avg divides by
/// count directly rather than accumulating a running mean, since the counts
/// are small and the sum fits comfortably in a uint64_t at these magnitudes.
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

/// Prints one stream's address for the placement record. The addr%16 column is
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

/// Same line format with a signed value (a difference that may be negative).
void PrintSigned(const char* name, int32_t value) {
    printf("  %-28s avg=%9.1f cycles   max=%8ld cycles\n",
           name, static_cast<double>(value), static_cast<long>(value < 0 ? -value : value));
}

/// A stage line carrying an explicit value (e.g. a min-of-repeats) in the avg slot.
void PrintValue(const char* name, uint32_t value, uint32_t max_value) {
    printf("  %-28s avg=%9.1f cycles   max=%8lu cycles\n",
           name, static_cast<double>(value), static_cast<unsigned long>(max_value));
}

// ════════════════════════════════════════════════════════════════════════
//  Bus-contention counters (RP2350 busctrl_hw)
// ════════════════════════════════════════════════════════════════════════
// Four saturating 24-bit counters. For SRAM bank n (0..9) the arbiter exposes
// ACCESS and ACCESS_CONTESTED events (SDK enum bus_ctrl_perf_counter_t:
// arbiter_sram<n>_perf_event_*). "Contested" = an access that previously
// stalled on another master.

struct CounterSel { uint8_t bank; bool access; };

// Three passes cover the ten SRAM banks (4 counters each); pass 3 also reads
// SRAM0/SRAM4 ACCESS as sanity totals. Pass numbering starts at 1 (0 = no
// counters, the pure cycle measurement).
constexpr CounterSel kPassSel[3][4] = {
    {{0, false}, {1, false}, {2, false}, {3, false}},
    {{4, false}, {5, false}, {6, false}, {7, false}},
    {{8, false}, {9, false}, {0, true},  {4, true}},
};

uint8_t EventFor(const CounterSel & s) {
    // perfsel encoding (hardware/regs/busctrl.h): SRAM9 ACCESS_CONTESTED = 0x12,
    // each lower bank sits 4 higher; ACCESS is the next code.
    const uint8_t contested = static_cast<uint8_t>(0x12 + 4 * (9 - s.bank));
    return s.access ? static_cast<uint8_t>(contested + 1) : contested;
}

void ProgramCounters(int pass) {
    busctrl_hw->perfctr_en = 0;
    for (int k = 0; k < 4; ++k) {
        busctrl_hw->counter[k].sel = EventFor(kPassSel[pass - 1][k]);
        busctrl_hw->counter[k].value = 0; // write any value to clear
    }
    busctrl_hw->perfctr_en = 1;
}

inline void ReadCounters(uint32_t (&v)[4]) {
    for (int k = 0; k < 4; ++k) v[k] = busctrl_hw->counter[k].value;
}

/// Format (parsed by RunAndSaveBenchmarks.py):
///   "  contention stage=<s> bank=SRAM<n> contested=<v|-> access=<v|->"
void PrintContention(const char* stage, int pass, const double (&per_counter)[4]) {
    for (int k = 0; k < 4; ++k) {
        const CounterSel & s = kPassSel[pass - 1][k];
        if (s.access)
            printf("  contention stage=%s bank=SRAM%d contested=- access=%.1f\n",
                   stage, s.bank, per_counter[k]);
        else
            printf("  contention stage=%s bank=SRAM%d contested=%.1f access=-\n",
                   stage, s.bank, per_counter[k]);
    }
}

// ── Poller: runs on the OTHER core while the probe core runs a long workload.
// The counters are 24-bit and saturate at ~16.7M events, which one training
// epoch (~600M cycles) overruns, so the service core drains them (read, then
// clear) into 64-bit totals. Step 0b: it does so EVENT-DRIVEN, not by spinning.
// A TIMER0 alarm fires every kPollPeriodUs; the service core sleeps in WFE in
// between and is woken by the alarm becoming pending (SCR.SEVONPEND, so the
// interrupt needs no NVIC enable, no handler and no vector fetch) or by the
// probe core's SEV on stop. The poller's own SRAM traffic is therefore a few
// accesses per wake-up instead of a continuous stream, which would otherwise
// pollute the very bank-contention counts it is collecting (and violate the
// "other core dormant" condition). A handful of events can fall between a read
// and its clear; that is far below the noise at these totals.
enum : uint32_t { kCmdPoll = 1, kCmdQuit = 2, kReplyDone = 0xD0DE };

constexpr uint32_t kPollAlarm = 2;            // TIMER0 alarm 2 (the SDK's default pool uses 3)
constexpr uint32_t kPollPeriodUs = 10000;     // 10 ms: counters saturate after >100 ms at 150 MHz

struct PollShared {
    std::atomic<uint32_t> ready{0};
    std::atomic<uint32_t> stop{0};
    uint64_t total[4] = {0, 0, 0, 0};
};
PollShared g_poll;

/// Service loop, run on the core that is NOT the probe core. Only inline code
/// is used inside the polling loop (no calls), so with PROFILE_SERVICE_ATTR its
/// whole fetch stream stays in the service core's own bank.
PROFILE_SERVICE_ATTR void ServiceLoop() {
    constexpr uint32_t kAlarmBit = 1u << kPollAlarm;
    for (;;) {
        const uint32_t cmd = multicore_fifo_pop_blocking_inline();
        if (cmd == kCmdQuit) return;
        if (cmd != kCmdPoll) continue;

        uint64_t t0 = 0, t1 = 0, t2 = 0, t3 = 0;
        SCB->SCR |= SCB_SCR_SEVONPEND_Msk;
        timer_hw->inte |= kAlarmBit; // IRQ line asserts; left disabled in this core's NVIC
        g_poll.ready.store(1, std::memory_order_release);
        while (!g_poll.stop.load(std::memory_order_acquire)) {
            t0 += busctrl_hw->counter[0].value; busctrl_hw->counter[0].value = 0;
            t1 += busctrl_hw->counter[1].value; busctrl_hw->counter[1].value = 0;
            t2 += busctrl_hw->counter[2].value; busctrl_hw->counter[2].value = 0;
            t3 += busctrl_hw->counter[3].value; busctrl_hw->counter[3].value = 0;
            timer_hw->intr = kAlarmBit;                    // clear the source, then the latched pending bit
            NVIC->ICPR[0] = 1u << TIMER0_IRQ_2;
            timer_hw->alarm[kPollAlarm] = timer_hw->timerawl + kPollPeriodUs;
            while (!g_poll.stop.load(std::memory_order_acquire) && !(timer_hw->ints & kAlarmBit))
                __wfe();
        }
        timer_hw->inte &= ~kAlarmBit;
        timer_hw->intr = kAlarmBit;
        NVIC->ICPR[0] = 1u << TIMER0_IRQ_2;
        t0 += busctrl_hw->counter[0].value;
        t1 += busctrl_hw->counter[1].value;
        t2 += busctrl_hw->counter[2].value;
        t3 += busctrl_hw->counter[3].value;
        g_poll.total[0] = t0; g_poll.total[1] = t1; g_poll.total[2] = t2; g_poll.total[3] = t3;
        multicore_fifo_push_blocking_inline(kReplyDone);
    }
}

// Probe-core side of the protocol.
void PollStart(int pass) {
    ProgramCounters(pass);
    g_poll.stop.store(0, std::memory_order_relaxed);
    g_poll.ready.store(0, std::memory_order_relaxed);
    multicore_fifo_push_blocking(kCmdPoll);
    while (!g_poll.ready.load(std::memory_order_acquire)) tight_loop_contents();
}

void PollStop(double (&totals)[4]) {
    g_poll.stop.store(1, std::memory_order_release);
    __sev(); // wake the service core out of its WFE
    (void)multicore_fifo_pop_blocking(); // kReplyDone
    for (int k = 0; k < 4; ++k) totals[k] = static_cast<double>(g_poll.total[k]);
}

// ════════════════════════════════════════════════════════════════════════
//  Per-layer profile
// ════════════════════════════════════════════════════════════════════════

constexpr int kIterations = 100;
constexpr int kStages = 7;
constexpr const char* kStageNames[kStages] = {
    "layer0.forward", "layer1.forward", "layer2.forward", "compute_loss",
    "layer2.AccumulateGradients", "layer1.AccumulateGradients", "layer0.AccumulateGradients",
};

/// Results of one 100-iteration pass over the per-layer stages.
struct LayerPass {
    Stat stage[kStages];
    Stat fwd_total, bwd_total, full_total;
    uint64_t contested_sum[kStages][4] = {};
};

using Net = MLPOpticalRecognition::Net;

// Same placement class as the production instance's network (MEML_MLP_DATA:
// the pinned core's bank), not core 0's default bank.
MEML_MLP_DATA Net g_net{};

/// One pass of kIterations forward+loss+backward over real samples. pass == 0
/// takes no counter snapshots (the reference cycle measurement); pass 1..3
/// brackets every stage with counter reads taken OUTSIDE the timed region
/// (read, then t0 ... t1, then read), so the APB loads do not add contention
/// inside the bracket. Tagged like the production hot path so the inlined
/// kernels execute from the pinned core's bank, as in Train().
MEML_MLP_CODE void RunLayerPass(int pass, LayerPass & out) {
    auto & layer0 = g_net.layer<0>();
    auto & layer1 = g_net.layer<1>();
    auto & layer2 = g_net.layer<2>();

    // Scratch buffers sized to the widest layer (64) so one set covers every
    // inter-layer handoff; only the first kIn/kOut elements of each are ever
    // read/written for a given layer.
    std::array<dataset::DataType, Net::kMaxWidth> buf_a{}, buf_b{};
    std::array<dataset::DataType, dataset::kLabelSize> pred{}, deriv{};
    std::array<dataset::DataType, Net::kMaxWidth> delta_a{}, delta_b{};
    // layer0.AccumulateGradients() always writes NIn (64) delta values even
    // though there is no earlier layer left to consume them.
    std::array<dataset::DataType, Net::kMaxWidth> delta_layer0{};

    if (pass > 0) ProgramCounters(pass);

    for (int it = 0; it < kIterations; ++it) {
        const std::size_t s = static_cast<std::size_t>(it) % dataset::kNumExamples;
        const dataset::DataType* feat = dataset::features[s].data();
        const dataset::DataType* label = dataset::labels[s].data();

        uint32_t t[kStages];
        uint32_t c_before[kStages][4];
        uint32_t c_after[kStages][4];

        // Mask interrupts for the timed region only: DWT->CYCCNT counts ISR
        // cycles too, and stdio_usb's periodic USB SOF handling landing
        // inside one particular bracket (rather than another) would show up
        // as a multi-thousand-cycle difference attributed to whichever
        // stage it happened to interrupt -- not a real difference in that
        // stage's own cost.
        const uint32_t ints = save_and_disable_interrupts();

#define PROFILE_STAGE(idx, ...)                                           \
        do {                                                              \
            if (pass > 0) ReadCounters(c_before[idx]);                    \
            const uint32_t t0 = Cycles();                                 \
            __VA_ARGS__;                                                  \
            t[idx] = Cycles() - t0;                                       \
            if (pass > 0) ReadCounters(c_after[idx]);                     \
        } while (0)

        PROFILE_STAGE(0, layer0.forward(feat, buf_a.data()));
        PROFILE_STAGE(1, layer1.forward(buf_a.data(), buf_b.data()));
        PROFILE_STAGE(2, layer2.forward(buf_b.data(), pred.data()));
        PROFILE_STAGE(3, smlp::compute_loss<loss::LOSS_FUNCTIONS::LOSS_CATEGORICAL_CROSSENTROPY>(
            label, pred.data(), deriv.data(), dataset::kLabelSize, dataset::DataType(1)));
        PROFILE_STAGE(4, layer2.AccumulateGradients(layer2.m_cached_input.data(), deriv.data(), delta_b.data()));
        PROFILE_STAGE(5, layer1.AccumulateGradients(layer1.m_cached_input.data(), delta_b.data(), delta_a.data()));
        PROFILE_STAGE(6, layer0.AccumulateGradients(layer0.m_cached_input.data(), delta_a.data(), delta_layer0.data()));
#undef PROFILE_STAGE

        restore_interrupts(ints);

        for (int i = 0; i < kStages; ++i) {
            out.stage[i].add(t[i]);
            if (pass > 0)
                for (int k = 0; k < 4; ++k)
                    out.contested_sum[i][k] += static_cast<uint32_t>(c_after[i][k] - c_before[i][k]);
        }
        out.fwd_total.add(t[0] + t[1] + t[2]);
        out.bwd_total.add(t[4] + t[5] + t[6]);
        out.full_total.add(t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6]);
    }
}

struct BatchStages {
    Stat zero, norm, rms0, rms1, rms2, rms_total, batch_total;
};
constexpr std::size_t kBatchSize = 128; // matches MLPOpticalRecognition::Train
constexpr int kBatchIterations = 20;

/// Per-batch stages (once per 128 samples in TrainBatch). Measured separately
/// from the per-sample loop so those rows keep their meaning. Each iteration:
/// zero the accumulators (timed), run one untimed forward+backward so the
/// accumulators hold real gradients, then time the grad-norm computation and
/// each layer's RMSProp apply. Weights drift over these iterations; timing is
/// data-independent for these stages. The conditional clip rescale
/// (ScaleAccumulatedGradients) is not measured: it runs only when the norm
/// exceeds 5.0.
MEML_MLP_CODE void RunBatchStages(BatchStages & b) {
    auto & layer0 = g_net.layer<0>();
    auto & layer1 = g_net.layer<1>();
    auto & layer2 = g_net.layer<2>();
    std::array<dataset::DataType, Net::kMaxWidth> buf_a{}, buf_b{};
    std::array<dataset::DataType, dataset::kLabelSize> pred{}, deriv{};
    std::array<dataset::DataType, Net::kMaxWidth> delta_a{}, delta_b{}, delta_layer0{};
    constexpr float kLearningRate = 0.01f; // matches TestRAMIndependence
    const dataset::DataType batch_inv = dataset::DataType(1) / static_cast<dataset::DataType>(kBatchSize);

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

        b.zero.add(t_zero);
        b.norm.add(t_norm);
        b.rms0.add(t_rms0);
        b.rms1.add(t_rms1);
        b.rms2.add(t_rms2);
        b.rms_total.add(t_rms0 + t_rms1 + t_rms2);
        b.batch_total.add(t_zero + t_norm + t_rms0 + t_rms1 + t_rms2);
    }
}

// ════════════════════════════════════════════════════════════════════════
//  Production-path probe
// ════════════════════════════════════════════════════════════════════════
// The REAL facade, placed exactly as in the benchmark (MEML_MLP_DATA in the
// pinned bank): Train()'s inlined backward/RMSProp code, at its own addresses.
MEML_MLP_DATA MLPOpticalRecognition g_prod{};

constexpr uint32_t kProbeSeed = 0xC0DEu;
constexpr float kProbeLR = 0.01f;
constexpr int kProbeRepeats = 3;

struct Reps {
    uint32_t v[kProbeRepeats] = {};
    uint32_t min() const { uint32_t m = v[0]; for (int i = 1; i < kProbeRepeats; ++i) if (v[i] < m) m = v[i]; return m; }
    uint32_t max() const { uint32_t m = v[0]; for (int i = 1; i < kProbeRepeats; ++i) if (v[i] > m) m = v[i]; return m; }
};

/// Final loss/accuracy of every Train() run from kProbeSeed, kept to prove the
/// runs are bit-identical (golden G0, Step 0f).
struct Golden {
    MLPOpticalRecognition::Result eval[kProbeRepeats] = {};
    MLPOpticalRecognition::Result epoch[kProbeRepeats] = {};
};

bool SameBits(const MLPOpticalRecognition::Result & a, const MLPOpticalRecognition::Result & b) {
    return std::memcmp(&a, &b, sizeof(a)) == 0;
}

/// Train(0): forward + loss over all samples, no weight update (the evaluation
/// pass). Train(1): one training epoch plus that same evaluation pass. Not
/// interrupt-masked (an epoch is seconds long; masking would starve USB); the
/// printed figure is the min of kProbeRepeats.
MEML_MLP_CODE void MeasureTrain(Reps & eval, Reps & epoch, Golden & golden) {
    for (int rep = 0; rep < kProbeRepeats; ++rep) {
        g_prod.Initialise(kProbeSeed);
        const uint32_t t0 = Cycles();
        golden.eval[rep] = g_prod.Train(0, kProbeLR);
        eval.v[rep] = Cycles() - t0;
    }
    for (int rep = 0; rep < kProbeRepeats; ++rep) {
        g_prod.Initialise(kProbeSeed);
        const uint32_t t0 = Cycles();
        golden.epoch[rep] = g_prod.Train(1, kProbeLR);
        epoch.v[rep] = Cycles() - t0;
    }
}

MEML_MLP_CODE void RunTrainOnce() {
    g_prod.Initialise(kProbeSeed);
    (void)g_prod.Train(1, kProbeLR);
}

/// The evaluation pass alone (Initialise + Train(0)), for its own counter
/// totals: compared between builds it localises a gap to the forward path.
MEML_MLP_CODE void RunEvalOnce() {
    g_prod.Initialise(kProbeSeed);
    (void)g_prod.Train(0, kProbeLR);
}

// ── Loss stage under three conditions (Step 0e) ──
// compute_loss reads its label vector from flash (XIP). To separate "the
// arithmetic" from "the flash read": (i) as production does, labels in flash;
// (ii) label vector copied to RAM first; (iii) the XIP bus counters around the
// flash-label call. Real softmax inputs come from a forward pass done untimed.
struct LossVariants {
    Stat flash, ram;
    uint64_t xip[4] = {};   // access, access_contested, stall_upstream, stall_downstream
};

constexpr uint8_t kXipSel[4] = {
    BUSCTRL_PERFSEL0_VALUE_XIP_MAIN0_ACCESS,
    BUSCTRL_PERFSEL0_VALUE_XIP_MAIN0_ACCESS_CONTESTED,
    BUSCTRL_PERFSEL0_VALUE_XIP_MAIN0_STALL_UPSTREAM,
    BUSCTRL_PERFSEL0_VALUE_XIP_MAIN0_STALL_DOWNSTREAM,
};

MEML_MLP_CODE void RunLossVariants(LossVariants & out) {
    auto & layer0 = g_net.layer<0>();
    auto & layer1 = g_net.layer<1>();
    auto & layer2 = g_net.layer<2>();
    std::array<dataset::DataType, Net::kMaxWidth> buf_a{}, buf_b{};
    std::array<dataset::DataType, dataset::kLabelSize> pred{}, deriv{}, label_ram{};

    for (int variant = 0; variant < 3; ++variant) {
        if (variant == 2) {
            busctrl_hw->perfctr_en = 0;
            for (int k = 0; k < 4; ++k) {
                busctrl_hw->counter[k].sel = kXipSel[k];
                busctrl_hw->counter[k].value = 0;
            }
            busctrl_hw->perfctr_en = 1;
        }
        for (int it = 0; it < kIterations; ++it) {
            const std::size_t s = static_cast<std::size_t>(it) % dataset::kNumExamples;
            layer0.forward(dataset::features[s].data(), buf_a.data());
            layer1.forward(buf_a.data(), buf_b.data());
            layer2.forward(buf_b.data(), pred.data());
            const dataset::DataType* label = dataset::labels[s].data();
            if (variant == 1) {
                for (std::size_t i = 0; i < dataset::kLabelSize; ++i) label_ram[i] = label[i];
                label = label_ram.data();
            }
            uint32_t before[4] = {};
            if (variant == 2) ReadCounters(before);
            const uint32_t ints = save_and_disable_interrupts();
            const uint32_t t0 = Cycles();
            smlp::compute_loss<loss::LOSS_FUNCTIONS::LOSS_CATEGORICAL_CROSSENTROPY>(
                label, pred.data(), deriv.data(), dataset::kLabelSize, dataset::DataType(1));
            const uint32_t dt = Cycles() - t0;
            restore_interrupts(ints);
            if (variant == 2) {
                uint32_t after[4];
                ReadCounters(after);
                for (int k = 0; k < 4; ++k) out.xip[k] += static_cast<uint32_t>(after[k] - before[k]);
            }
            if (variant == 1) out.ram.add(dt);
            else if (variant == 0) out.flash.add(dt);
        }
    }
}

/// Everything the probe core does. Runs on core 1 for the core-1-pinned build,
/// on core 0 otherwise. Tagged like the production hot path.
MEML_MLP_CODE void ProbeBody() {
    EnableCycleCounter();
    if (DWT->CTRL & DWT_CTRL_NOCYCCNT_Msk) {
        printf("Profiling aborted: this core's DWT does not implement a cycle counter.\n");
        printf("Profile complete.\n");
        return;
    }

    uintptr_t sp;
    asm volatile("mov %0, sp" : "=r"(sp));

    g_net.SetSeed(0xC0DEu);
    g_net.InitXavier();

    auto & layer0 = g_net.layer<0>();
    auto & layer1 = g_net.layer<1>();
    auto & layer2 = g_net.layer<2>();

    // ── Pass 0: pure cycles; passes 1..3: same loop with counter snapshots ──
    static LayerPass passes[4]; // static: the probe core's stack is only 4 KB
    for (int pass = 0; pass < 4; ++pass) RunLayerPass(pass, passes[pass]);
    const LayerPass & ref = passes[0];

    static BatchStages batch;
    RunBatchStages(batch);

    static LossVariants loss_var;
    RunLossVariants(loss_var);

    printf("\nForward/backward cycle profile (%d iterations, network 64-64-32-10):\n", kIterations);
    printf("Forward:\n");
    PrintStage("layer0.forward", ref.stage[0]);
    PrintStage("layer1.forward", ref.stage[1]);
    PrintStage("layer2.forward", ref.stage[2]);
    PrintStage("forward total", ref.fwd_total);
    printf("Loss:\n");
    PrintStage("compute_loss", ref.stage[3]);
    printf("Backward:\n");
    PrintStage("layer2.AccumulateGradients", ref.stage[4]);
    PrintStage("layer1.AccumulateGradients", ref.stage[5]);
    PrintStage("layer0.AccumulateGradients", ref.stage[6]);
    PrintStage("backward total", ref.bwd_total);
    printf("Full forward+loss+backward total:\n");
    PrintStage("full pass", ref.full_total);

    printf("Loss stage variants (Step 0e; %d iterations, real softmax inputs):\n", kIterations);
    PrintStage("compute_loss.flash_labels", loss_var.flash);
    PrintStage("compute_loss.ram_labels", loss_var.ram);
    {
        static const char* const kXipNames[4] = {"access", "access_contested", "stall_upstream", "stall_downstream"};
        for (int k = 0; k < 4; ++k)
            printf("  xip stage=compute_loss.flash_labels event=%s per_call=%.2f\n", kXipNames[k],
                   static_cast<double>(loss_var.xip[k]) / static_cast<double>(kIterations));
    }

    printf("Per-batch stages (once per %u samples; %d iterations):\n",
           static_cast<unsigned>(kBatchSize), kBatchIterations);
    PrintStage("batch.zero_grads", batch.zero);
    PrintStage("batch.grad_norm", batch.norm);
    PrintStage("batch.rmsprop.layer0", batch.rms0);
    PrintStage("batch.rmsprop.layer1", batch.rms1);
    PrintStage("batch.rmsprop.layer2", batch.rms2);
    PrintStage("batch.rmsprop total", batch.rms_total);
    PrintStage("batch total", batch.batch_total);
    const uint32_t batch_per_sample =
        static_cast<uint32_t>(batch.batch_total.sum / batch.batch_total.count / kBatchSize);
    PrintValue("batch total per sample", batch_per_sample, batch_per_sample);

    // ── Per-layer bus contention (counters read outside the timed brackets) ──
    printf("Bus contention per sample (per-layer harness; counters outside the timed bracket):\n");
    for (int pass = 1; pass <= 3; ++pass) {
        const double ref_avg = static_cast<double>(passes[0].full_total.avg());
        const double got = static_cast<double>(passes[pass].full_total.avg());
        const double drift = ref_avg > 0.0 ? (got - ref_avg) / ref_avg : 0.0;
        const bool valid = drift < 0.001 && drift > -0.001;
        printf("  contention pass=%d full_pass_cycles=%.1f reference=%.1f valid=%s\n",
               pass, got, ref_avg, valid ? "yes" : "NO");
        for (int st = 0; st < kStages; ++st) {
            double avg[4];
            for (int k = 0; k < 4; ++k)
                avg[k] = static_cast<double>(passes[pass].contested_sum[st][k]) / static_cast<double>(kIterations);
            PrintContention(kStageNames[st], pass, avg);
        }
    }

    // ── Production path: the real facade, on this core, at its real addresses ──
    static Reps eval_reps, epoch_reps;
    static Golden golden;
    MeasureTrain(eval_reps, epoch_reps, golden);

    const uint32_t eval_min = eval_reps.min();
    const uint32_t epoch_min = epoch_reps.min();
    const uint32_t train_per_sample =
        (epoch_min - eval_min) / static_cast<uint32_t>(dataset::kNumExamples);
    const uint32_t harness_per_sample =
        static_cast<uint32_t>(ref.full_total.avg()) + batch_per_sample;
    // Signed: positive means the production path costs more than the per-layer
    // harness explains (a code path the harness does not run); Step 0a wants
    // |gap| < 0.5% of train.epoch_minus_eval_per_sample.
    const int32_t gap = static_cast<int32_t>(train_per_sample) - static_cast<int32_t>(harness_per_sample);

    printf("Production path (MLPOpticalRecognition::Train on this core, min of %d):\n", kProbeRepeats);
    PrintValue("train.eval_pass", eval_min, eval_reps.max());
    PrintValue("train.epoch", epoch_min, epoch_reps.max());
    PrintValue("train.epoch_minus_eval_per_sample", train_per_sample, train_per_sample);
    PrintValue("train.harness_per_sample", harness_per_sample, harness_per_sample);
    PrintSigned("train.unexplained_per_sample", gap);
    for (int rep = 0; rep < kProbeRepeats; ++rep)
        printf("  train.repeat %d eval_pass=%lu epoch=%lu\n", rep,
               static_cast<unsigned long>(eval_reps.v[rep]), static_cast<unsigned long>(epoch_reps.v[rep]));

    // ── Golden numerics (G0): every run from the same seed must agree to the bit ──
    bool identical = true;
    for (int rep = 1; rep < kProbeRepeats; ++rep)
        identical = identical && SameBits(golden.eval[rep], golden.eval[0]) && SameBits(golden.epoch[rep], golden.epoch[0]);
    printf("Golden numerics (seed 0x%lx; bit-identical across %d runs: %s):\n",
           static_cast<unsigned long>(kProbeSeed), kProbeRepeats, identical ? "yes" : "NO");
    printf("  golden train0 loss=%.9g accuracy=%.9g\n",
           static_cast<double>(golden.eval[0].loss), static_cast<double>(golden.eval[0].accuracy));
    printf("  golden train1 loss=%.9g accuracy=%.9g\n",
           static_cast<double>(golden.epoch[0].loss), static_cast<double>(golden.epoch[0].accuracy));

    // ── Production-path contention: whole Train(1) per pass, drained by the service core ──
    printf("Bus contention, production path (totals over Initialise + Train(1); polled from the other core):\n");
    for (int pass = 1; pass <= 3; ++pass) {
        double totals[4];
        PollStart(pass);
        RunTrainOnce();
        PollStop(totals);
        PrintContention("train.epoch", pass, totals);
    }

    // ── Evaluation-only contention (Step 0d): Train(0), localises a gap to the forward path ──
    printf("Bus contention, evaluation pass only (Initialise + Train(0); polled from the other core):\n");
    for (int pass = 1; pass <= 3; ++pass) {
        double totals[4];
        PollStart(pass);
        RunEvalOnce();
        PollStop(totals);
        PrintContention("train.eval_pass", pass, totals);
    }

    // ── Poller footprint: counters while the probe core idles and the service core polls ──
    printf("Poller footprint (probe core idle for ~50 ms, service core polling):\n");
    {
        double totals[4];
        PollStart(3); // SRAM8/9 contested + SRAM0/4 access
        busy_wait_ms(50);
        PollStop(totals);
        PrintContention("idle_window", 3, totals);
    }

    // ── Placement of everything the kernels touch ──
    printf("Placement (net and kernel code addresses; code addresses are added by\n");
    printf("RunAndSaveBenchmarks.py from the ELF symbol table):\n");
    PrintPlacement("stack_pointer", reinterpret_cast<const void*>(sp));
    PrintPlacement("net", &g_net);
    PrintPlacement("prod_instance", &g_prod);
    printf("  prod_instance size=%lu bytes\n", static_cast<unsigned long>(sizeof(g_prod)));
    PrintPlacement("layer0.m_weights", layer0.m_weights.data());
    PrintPlacement("layer0.m_grad_accum", layer0.m_grad_accum.data());
    PrintPlacement("layer0.m_cached_input", layer0.m_cached_input.data());
    PrintPlacement("layer0.m_inner_products", layer0.m_inner_products.data());
    PrintPlacement("layer0.m_act_output", layer0.m_act_output.data());
    PrintPlacement("layer0.m_biases", layer0.m_biases.data());
    PrintPlacement("layer0.m_bias_grad_accum", layer0.m_bias_grad_accum.data());
    PrintPlacement("layer1.m_weights", layer1.m_weights.data());
    PrintPlacement("layer1.m_grad_accum", layer1.m_grad_accum.data());
    PrintPlacement("layer1.m_cached_input", layer1.m_cached_input.data());
    PrintPlacement("layer1.m_inner_products", layer1.m_inner_products.data());
    PrintPlacement("layer1.m_act_output", layer1.m_act_output.data());
    PrintPlacement("layer1.m_biases", layer1.m_biases.data());
    PrintPlacement("layer1.m_bias_grad_accum", layer1.m_bias_grad_accum.data());
    PrintPlacement("layer2.m_weights", layer2.m_weights.data());
    PrintPlacement("layer2.m_grad_accum", layer2.m_grad_accum.data());
    PrintPlacement("layer2.m_cached_input", layer2.m_cached_input.data());
    PrintPlacement("layer2.m_inner_products", layer2.m_inner_products.data());
    PrintPlacement("layer2.m_act_output", layer2.m_act_output.data());
    PrintPlacement("layer2.m_biases", layer2.m_biases.data());
    PrintPlacement("layer2.m_bias_grad_accum", layer2.m_bias_grad_accum.data());
    printf("Profile complete.\n");
    printf("\n");
}

// ── Launching the probe on the right core ──
#if PROFILE_PROBE_ON_CORE1
MEML_RUNS_ON_CORE(1) void Core1ProbeEntry() {
    ProbeBody();
    multicore_fifo_push_blocking_inline(kCmdQuit); // tells core 0's ServiceLoop to return
}
#else
PROFILE_SERVICE_ATTR void Core1ServiceEntry() {
    ServiceLoop();
}
#if !defined(MEML_MLP_RUNS_ON_CORE)
// Without a per-core linker section, a plain array stands in for the stack.
uint32_t s_service_stack[2048]; // 8 KiB
#endif
#endif

}  // namespace

void RunProfiles() {
    // Interrupt-driven USB stdio stays on core 0. The probe core runs the
    // measurement; the other core hosts the counter poller and is otherwise
    // blocked in WFE (multicore_fifo_pop_blocking), i.e. dormant, exactly as in
    // the benchmark's dormant condition.
    multicore_fifo_drain();
#if PROFILE_PROBE_ON_CORE1
    const std::size_t stack_size = reinterpret_cast<uintptr_t>(__core1_stack_top__) -
                                   reinterpret_cast<uintptr_t>(__core1_stack_bottom__);
    multicore_launch_core1_with_stack(
        reinterpret_cast<void (*)()>(reinterpret_cast<uintptr_t>(&Core1ProbeEntry) | 1u),
        __core1_stack_bottom__, stack_size);
    ServiceLoop(); // returns when core 1 pushes kCmdQuit
    multicore_reset_core1();
#else
#if defined(MEML_MLP_RUNS_ON_CORE)
    multicore_launch_core1(reinterpret_cast<void (*)()>(reinterpret_cast<uintptr_t>(&Core1ServiceEntry) | 1u));
#else
    multicore_launch_core1_with_stack(
        reinterpret_cast<void (*)()>(reinterpret_cast<uintptr_t>(&Core1ServiceEntry) | 1u),
        s_service_stack, sizeof(s_service_stack));
#endif
    ProbeBody();
    multicore_fifo_push_blocking(kCmdQuit);
    sleep_ms(10);
    multicore_reset_core1();
#endif
}

} // namespace profile
} // namespace test
