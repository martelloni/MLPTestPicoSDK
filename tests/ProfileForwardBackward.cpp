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

// The production instance (see "Production-path probe object" below); declared
// first because the placement printer takes offsets relative to it.
MEML_MLP_DATA MLPOpticalRecognition g_prod{};

/// Prints one stream's address for the placement record. The addr%16 column is
/// what plan-mlpBetterOptimisation.md section 1.5 needs to correlate cycle
/// counts with SRAM bank phase; `off` is the offset from the start of the
/// production instance g_prod (Step 0.5b: the benchmark ELF has the same type at
/// a different address, so absolute address = its instance symbol + off).
/// RunAndSaveBenchmarks.py parses these lines
/// ("placement data <name> addr=0x... mod16=N off=M") into placement_results.csv
/// and stream_offsets.csv. Streams outside g_prod (the stack) print off=-.
void PrintPlacement(const char* name, const void* p);

void PrintPlacement(const char* name, const void* p) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const uintptr_t base = reinterpret_cast<uintptr_t>(&g_prod);
    if (a >= base && a < base + sizeof(g_prod))
        printf("  placement data %-28s addr=0x%08lx mod16=%lu off=%lu\n", name, static_cast<unsigned long>(a),
               static_cast<unsigned long>(a & 15u), static_cast<unsigned long>(a - base));
    else
        printf("  placement data %-28s addr=0x%08lx mod16=%lu off=-\n", name, static_cast<unsigned long>(a),
               static_cast<unsigned long>(a & 15u));
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

// Passes 1-3 cover the contested counts of the ten SRAM banks (4 counters
// each); pass 3 also reads SRAM0/SRAM4 ACCESS as sanity totals. Passes 4-6
// (Step 0.5e/0.5g) add the ACCESS counts of all ten banks (4-6 are only used for
// whole-run totals, never by the per-layer harness). Pass numbering starts at 1
// (0 = no counters, the pure cycle measurement).
constexpr int kNumPasses = 6;
constexpr CounterSel kPassSel[kNumPasses][4] = {
    {{0, false}, {1, false}, {2, false}, {3, false}},
    {{4, false}, {5, false}, {6, false}, {7, false}},
    {{8, false}, {9, false}, {0, true},  {4, true}},
    {{0, true},  {1, true},  {2, true},  {3, true}},
    {{4, true},  {5, true},  {6, true},  {7, true}},
    {{8, true},  {9, true},  {0, true},  {4, true}},
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

constexpr uint8_t kXipSel[4] = {
    BUSCTRL_PERFSEL0_VALUE_XIP_MAIN0_ACCESS,
    BUSCTRL_PERFSEL0_VALUE_XIP_MAIN0_ACCESS_CONTESTED,
    BUSCTRL_PERFSEL0_VALUE_XIP_MAIN0_STALL_UPSTREAM,
    BUSCTRL_PERFSEL0_VALUE_XIP_MAIN0_STALL_DOWNSTREAM,
};
constexpr const char* kXipNames[4] = {"access", "access_contested", "stall_upstream", "stall_downstream"};

void ProgramXipCounters() {
    busctrl_hw->perfctr_en = 0;
    for (int k = 0; k < 4; ++k) {
        busctrl_hw->counter[k].sel = kXipSel[k];
        busctrl_hw->counter[k].value = 0;
    }
    busctrl_hw->perfctr_en = 1;
}

// Probe-core side of the protocol. pass == 0 programs the four XIP (flash)
// events instead of SRAM bank events.
void PollStart(int pass) {
    if (pass == 0) ProgramXipCounters(); else ProgramCounters(pass);
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

}  // namespace

// Production's own network and scratch buffers. The per-layer harness runs on
// THESE (Step 0.5a), not on a twin network with stack scratch, so the kernels see
// the same addresses, the same bank phases and the same back-propagation
// buffers as MLPOpticalRecognition::Train(). Befriended by StaticMLP and
// MLPOpticalRecognition under MEML_RUN_PROFILE; adds no code to the hot path.
struct Access {
    using T = dataset::DataType;
    static Net & net(MLPOpticalRecognition & m) { return m.state_.net; }
    static T* buf_a(Net & n) { return n.m_buf_a.data(); }
    static T* buf_b(Net & n) { return n.m_buf_b.data(); }
    static T* pred(Net & n) { return n.m_pred.data(); }
    static T* deriv(Net & n) { return n.m_deriv.data(); }
    static T* bp0(Net & n) { return n.m_bp0.data(); }
    static T* bp1(Net & n) { return n.m_bp1.data(); }
    static std::size_t* shuffle(MLPOpticalRecognition & m) { return m.state_.shuffle_indices.data(); }
    static T* input_scratch(MLPOpticalRecognition & m) { return m.state_.input_scratch.data(); }
    static T* output_scratch(MLPOpticalRecognition & m) { return m.state_.output_scratch.data(); }
};

namespace {

// ════════════════════════════════════════════════════════════════════════
//  Production-path probe object
// ════════════════════════════════════════════════════════════════════════
// The REAL facade, placed exactly as in the benchmark (MEML_MLP_DATA in the
// pinned bank): Train()'s inlined backward/RMSProp code, at its own addresses.
// (g_prod itself is declared at the top of this file's anonymous namespace.)

constexpr uint32_t kProbeSeed = 0xC0DEu;

/// Fisher-Yates over the production shuffle buffer with a local xorshift, so the
/// harness reads its samples through the same `features + s * stride` path, in a
/// scattered order like Train(), straight from flash.
void ShuffleProbeIndices() {
    std::size_t* idx = Access::shuffle(g_prod);
    const std::size_t n = dataset::kNumExamples;
    for (std::size_t i = 0; i < n; ++i) idx[i] = i;
    uint32_t x = kProbeSeed | 1u;
    for (std::size_t i = n; i > 1; --i) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        const std::size_t j = x % i;
        const std::size_t t = idx[i - 1]; idx[i - 1] = idx[j]; idx[j] = t;
    }
}

/// One pass of kIterations forward+loss+backward over real samples, on
/// g_prod's own network and buffers, each stage in exactly the role
/// StaticMLP::TrainBatch / backprop_accumulate gives it: forward ping-pongs
/// m_buf_a/m_buf_b (layer I writes `(I & 1) ? m_buf_b : m_buf_a`) and the last
/// output is copied into m_pred; the loss writes m_deriv; backward uses
/// `(I & 1) ? m_bp1 : m_bp0` as delta_out and the previous layer's delta (or
/// m_deriv) as the error. No stack arrays reach a kernel. pass == 0 takes no
/// counter snapshots (the reference cycle measurement); pass 1..3 brackets
/// every stage with counter reads taken OUTSIDE the timed region (read, then
/// t0 ... t1, then read), so the APB loads do not add contention inside the
/// bracket. Tagged like the production hot path so the inlined kernels execute
/// from the pinned core's bank, as in Train().
MEML_MLP_CODE void RunLayerPass(int pass, LayerPass & out) {
    Net & net = Access::net(g_prod);
    auto & layer0 = net.layer<0>();
    auto & layer1 = net.layer<1>();
    auto & layer2 = net.layer<2>();
    dataset::DataType* const buf_a = Access::buf_a(net);
    dataset::DataType* const buf_b = Access::buf_b(net);
    dataset::DataType* const pred = Access::pred(net);
    dataset::DataType* const deriv = Access::deriv(net);
    dataset::DataType* const bp0 = Access::bp0(net);
    dataset::DataType* const bp1 = Access::bp1(net);
    const std::size_t* const shuffle = Access::shuffle(g_prod);

    if (pass > 0) ProgramCounters(pass);

    for (int it = 0; it < kIterations; ++it) {
        const std::size_t s = shuffle[it];
        const dataset::DataType* feat = dataset::features[0].data() + s * dataset::kFeatureSize;
        const dataset::DataType* label = dataset::labels[0].data() + s * dataset::kLabelSize;

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

        PROFILE_STAGE(0, layer0.forward(feat, buf_a));
        PROFILE_STAGE(1, layer1.forward(buf_a, buf_b));
        // Layer 2 writes m_buf_a (parity), then Train() copies it to m_pred.
        PROFILE_STAGE(2, layer2.forward(buf_b, buf_a);
                         for (std::size_t o = 0; o < dataset::kLabelSize; ++o) pred[o] = buf_a[o]);
        PROFILE_STAGE(3, smlp::compute_loss<loss::LOSS_FUNCTIONS::LOSS_CATEGORICAL_CROSSENTROPY>(
            label, pred, deriv, dataset::kLabelSize, dataset::DataType(1)));
        PROFILE_STAGE(4, layer2.AccumulateGradients(layer2.m_cached_input.data(), deriv, bp0));
        PROFILE_STAGE(5, layer1.AccumulateGradients(layer1.m_cached_input.data(), bp0, bp1));
        PROFILE_STAGE(6, layer0.AccumulateGradients(layer0.m_cached_input.data(), bp1, bp0));
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
    Net & net = Access::net(g_prod);
    auto & layer0 = net.layer<0>();
    auto & layer1 = net.layer<1>();
    auto & layer2 = net.layer<2>();
    dataset::DataType* const buf_a = Access::buf_a(net);
    dataset::DataType* const buf_b = Access::buf_b(net);
    dataset::DataType* const pred = Access::pred(net);
    dataset::DataType* const deriv = Access::deriv(net);
    dataset::DataType* const bp0 = Access::bp0(net);
    dataset::DataType* const bp1 = Access::bp1(net);
    const std::size_t* const shuffle = Access::shuffle(g_prod);
    constexpr float kLearningRate = 0.01f; // matches TestRAMIndependence
    const dataset::DataType batch_inv = dataset::DataType(1) / static_cast<dataset::DataType>(kBatchSize);

    for (int it = 0; it < kBatchIterations; ++it) {
        const std::size_t s = shuffle[it];

        uint32_t ints = save_and_disable_interrupts();
        uint32_t t0 = Cycles();
        layer0.InitGradientAccumulators();
        layer1.InitGradientAccumulators();
        layer2.InitGradientAccumulators();
        const uint32_t t_zero = Cycles() - t0;
        restore_interrupts(ints);

        // Untimed: populate the accumulators with a real sample's gradients.
        layer0.forward(dataset::features[0].data() + s * dataset::kFeatureSize, buf_a);
        layer1.forward(buf_a, buf_b);
        layer2.forward(buf_b, buf_a);
        for (std::size_t o = 0; o < dataset::kLabelSize; ++o) pred[o] = buf_a[o];
        smlp::compute_loss<loss::LOSS_FUNCTIONS::LOSS_CATEGORICAL_CROSSENTROPY>(
            dataset::labels[0].data() + s * dataset::kLabelSize, pred, deriv, dataset::kLabelSize, dataset::DataType(1));
        layer2.AccumulateGradients(layer2.m_cached_input.data(), deriv, bp0);
        layer1.AccumulateGradients(layer1.m_cached_input.data(), bp0, bp1);
        layer0.AccumulateGradients(layer0.m_cached_input.data(), bp1, bp0);

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

MEML_MLP_CODE void RunLossVariants(LossVariants & out) {
    Net & net = Access::net(g_prod);
    auto & layer0 = net.layer<0>();
    auto & layer1 = net.layer<1>();
    auto & layer2 = net.layer<2>();
    dataset::DataType* const buf_a = Access::buf_a(net);
    dataset::DataType* const buf_b = Access::buf_b(net);
    dataset::DataType* const pred = Access::pred(net);
    dataset::DataType* const deriv = Access::deriv(net);
    // The RAM copy of the label lives in production's own output scratch.
    dataset::DataType* const label_ram = Access::output_scratch(g_prod);
    const std::size_t* const shuffle = Access::shuffle(g_prod);

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
            const std::size_t s = shuffle[it];
            layer0.forward(dataset::features[0].data() + s * dataset::kFeatureSize, buf_a);
            layer1.forward(buf_a, buf_b);
            layer2.forward(buf_b, buf_a);
            for (std::size_t o = 0; o < dataset::kLabelSize; ++o) pred[o] = buf_a[o];
            const dataset::DataType* label = dataset::labels[0].data() + s * dataset::kLabelSize;
            if (variant == 1) {
                for (std::size_t i = 0; i < dataset::kLabelSize; ++i) label_ram[i] = label[i];
                label = label_ram;
            }
            uint32_t before[4] = {};
            if (variant == 2) ReadCounters(before);
            const uint32_t ints = save_and_disable_interrupts();
            const uint32_t t0 = Cycles();
            smlp::compute_loss<loss::LOSS_FUNCTIONS::LOSS_CATEGORICAL_CROSSENTROPY>(
                label, pred, deriv, dataset::kLabelSize, dataset::DataType(1));
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

/// Sleeps this core in WFE until TIMER0 alarm 1 fires (same SEVONPEND scheme as
/// the poller: no NVIC enable, no handler), so that nothing but the poller moves
/// on the bus while the footprint is measured. busy_wait_ms() would instead
/// measure the probe's own spinning (E4).
MEML_MLP_CODE void SleepWfeUs(uint32_t us) {
    constexpr uint32_t kBit = 1u << 1;
    SCB->SCR |= SCB_SCR_SEVONPEND_Msk;
    timer_hw->intr = kBit;
    NVIC->ICPR[0] = 1u << TIMER0_IRQ_1;
    timer_hw->inte |= kBit;
    timer_hw->alarm[1] = timer_hw->timerawl + us;
    while (!(timer_hw->ints & kBit)) __wfe();
    timer_hw->inte &= ~kBit;
    timer_hw->intr = kBit;
    NVIC->ICPR[0] = 1u << TIMER0_IRQ_1;
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

    g_prod.Initialise(kProbeSeed);
    ShuffleProbeIndices();

    Net & net = Access::net(g_prod);
    auto & layer0 = net.layer<0>();
    auto & layer1 = net.layer<1>();
    auto & layer2 = net.layer<2>();

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
    for (int pass = 1; pass <= kNumPasses; ++pass) {
        double totals[4];
        PollStart(pass);
        RunTrainOnce();
        PollStop(totals);
        PrintContention("train.epoch", pass, totals);
    }

    // ── Evaluation-only contention (Step 0d): Train(0), localises a gap to the forward path ──
    printf("Bus contention, evaluation pass only (Initialise + Train(0); polled from the other core):\n");
    for (int pass = 1; pass <= kNumPasses; ++pass) {
        double totals[4];
        PollStart(pass);
        RunEvalOnce();
        PollStop(totals);
        PrintContention("train.eval_pass", pass, totals);
    }

    // ── Flash (XIP) traffic of the production path (Step 0.5d). The dataset lives
    // in flash behind a cache both cores share; these are whole-run totals, so the
    // script derives stall cycles per training sample from Train(1) minus Train(0).
    printf("XIP (flash) events, production path (whole-run totals; polled from the other core):\n");
    for (int run = 0; run < 2; ++run) {
        double totals[4];
        PollStart(0);
        if (run == 0) RunTrainOnce(); else RunEvalOnce();
        PollStop(totals);
        for (int k = 0; k < 4; ++k)
            printf("  xip stage=%s event=%s per_call=%.0f\n",
                   run == 0 ? "train.epoch" : "train.eval_pass", kXipNames[k], totals[k]);
    }

    // ── Poller footprint (Step 0.5e): the probe core sleeps in WFE on a 50 ms
    // alarm, so the only SRAM traffic in the window is the poller's own (plus the
    // other core's USB/SDK background). Contested AND access counts on all ten
    // banks, six 50 ms windows (one per counter pass). ──
    printf("Poller footprint (probe core asleep in WFE for 50 ms per pass, service core polling):\n");
    for (int pass = 1; pass <= kNumPasses; ++pass) {
        double totals[4];
        PollStart(pass);
        SleepWfeUs(50000);
        PollStop(totals);
        PrintContention("poller_footprint", pass, totals);
    }

    // ── Placement of everything the kernels touch ──
    printf("Placement (net and kernel code addresses; code addresses are added by\n");
    printf("RunAndSaveBenchmarks.py from the ELF symbol table):\n");
    PrintPlacement("stack_pointer", reinterpret_cast<const void*>(sp));
#if PROFILE_PROBE_ON_CORE1
    PrintPlacement("stack_bottom", __core1_stack_bottom__);
    PrintPlacement("stack_top", __core1_stack_top__);
#endif
    PrintPlacement("net", &net);
    PrintPlacement("prod_instance", &g_prod);
    printf("  prod_instance size=%lu bytes\n", static_cast<unsigned long>(sizeof(g_prod)));
    // Streams the hot loops touch that belong to the network, not to a layer.
    PrintPlacement("net.m_buf_a", Access::buf_a(net));
    PrintPlacement("net.m_buf_b", Access::buf_b(net));
    PrintPlacement("net.m_pred", Access::pred(net));
    PrintPlacement("net.m_deriv", Access::deriv(net));
    PrintPlacement("net.m_bp0", Access::bp0(net));
    PrintPlacement("net.m_bp1", Access::bp1(net));
    PrintPlacement("state.shuffle_indices", Access::shuffle(g_prod));
    PrintPlacement("state.input_scratch", Access::input_scratch(g_prod));
    PrintPlacement("state.output_scratch", Access::output_scratch(g_prod));
#define PRINT_LAYER_STREAMS(L, layer)                                                         \
    PrintPlacement("layer" #L ".m_weights", (layer).m_weights.data());                        \
    PrintPlacement("layer" #L ".m_grad_accum", (layer).m_grad_accum.data());                  \
    PrintPlacement("layer" #L ".m_sq_grad_avg", (layer).m_sq_grad_avg.data());                \
    PrintPlacement("layer" #L ".m_cached_input", (layer).m_cached_input.data());              \
    PrintPlacement("layer" #L ".m_inner_products", (layer).m_inner_products.data());          \
    PrintPlacement("layer" #L ".m_act_output", (layer).m_act_output.data());                  \
    PrintPlacement("layer" #L ".m_biases", (layer).m_biases.data());                          \
    PrintPlacement("layer" #L ".m_bias_grad_accum", (layer).m_bias_grad_accum.data())
    PRINT_LAYER_STREAMS(0, layer0);
    PRINT_LAYER_STREAMS(1, layer1);
    PRINT_LAYER_STREAMS(2, layer2);
#undef PRINT_LAYER_STREAMS
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
