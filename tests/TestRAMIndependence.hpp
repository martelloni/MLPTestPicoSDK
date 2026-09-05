/**
 * @file TestRAMIndependence.hpp
 * @author Andrea Martelloni
 * @brief Load test to verify deterministic timings between core 0 and core 1 on RP2350, with independent RAM banks
 * @date 2026-08-16
 */

#ifndef __TEST_RAM_INDEPENDENCE_HPP__
#define __TEST_RAM_INDEPENDENCE_HPP__

#include <array>

#include "TestBase.hpp"
#include "MemoryDefs.hpp"
#include "mlp/StaticLayer.h"
#include "mlp/StaticMLP.h"
#include "utils/RAMFlooder.hpp"

namespace test {

class TestRAMIndependence : public TestBase {

public:

    TestRAMIndependence(uint32_t clock_frequency_hz)
        : TestBase(TestConfig{clock_frequency_hz, 5u, 1u}) {
        ConfigureCoreWork(&Core0Init,
                         &Core0Task,
                         &Core1Init,
                         &Core1Task);
    }

private:
    struct TestNN {
        using Net = smlp::StaticMLP<
            float,
            smlp::Layout<64, 64, 32, 10, 2>,
            smlp::Activations<
                ACTIVATION_FUNCTIONS::RELU,
                ACTIVATION_FUNCTIONS::RELU,
                ACTIVATION_FUNCTIONS::RELU,
                ACTIVATION_FUNCTIONS::SIGMOID>>;

        Net net;
        std::array<float, 64> input;
        std::array<float, 2> output;

        TestNN() : net(), input{}, output{} {}
    };

    inline static MEML_DATA_ON_CORE(0) TestNN core0_nn_{};
    static constexpr uint32_t core1_ram_flooder_size_ = 1024 * 60; // 240 KB
    inline static MEML_DATA_ON_CORE(1) utils::RAMFlooder<uint32_t, core1_ram_flooder_size_> core1_ram_flooder_{};

    static MEML_RUNS_ON_CORE(0) void Core0Init() {
        core0_nn_.net.SetSeed(0xC0DEu);
        core0_nn_.net.InitXavier();
        core0_nn_.input.fill(0.0f);
        core0_nn_.output.fill(0.0f);
    }

    static MEML_RUNS_ON_CORE(0) void Core0Task() {
        StartMeasurementCore0();
        core0_nn_.net.GetOutput(core0_nn_.input.data(), core0_nn_.output.data());
        StopMeasurementCore0();
    }

    static MEML_RUNS_ON_CORE(1) void Core1Init() {
        // No init for nows
    }

    static MEML_RUNS_ON_CORE(1) void Core1Task() {
        StartMeasurementCore1();
        core1_ram_flooder_.FillOnce();
        StopMeasurementCore1();
    }

};

}


#endif // __TEST_RAM_INDEPENDENCE_HPP__
