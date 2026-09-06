#include <stdio.h>
#include "pico/stdlib.h"
#include "hardware/clocks.h"
#include "hardware/vreg.h"
#include "dataset/Dataset.hpp"
#include "tests/TestRAMIndependence.hpp"

static bool ValidateGeneratedDataset()
{
    constexpr std::size_t last_index = dataset::kNumExamples - 1u;

    for (std::size_t row = 0; row < dataset::kNumExamples; ++row) {
        const auto &sample = dataset::features[row];
        const bool is_boundary = (row == 0u) || (row == last_index);
        for (std::size_t col = 0; col < dataset::kFeatureSize; ++col) {
            const float value = sample[col];
            if (value < 0.0f || value > 1.0f) {
                printf("Dataset feature validation failed at row %zu, col %zu: %.6f\n", row, col, value);
                return false;
            }
            if (is_boundary && col == 0u) {
                // This is a boundary sanity check: the first and last rows remain normalized in [0,1].
            }
        }
    }

    for (std::size_t row = 0; row < dataset::kNumExamples; ++row) {
        std::size_t ones = 0u;
        for (std::size_t col = 0; col < dataset::kLabelSize; ++col) {
            const float value = dataset::labels[row][col];
            if (value != 0.0f && value != 1.0f) {
                printf("Dataset label validation failed at row %zu, col %zu: %.6f\n", row, col, value);
                return false;
            }
            if (value == 1.0f) {
                ++ones;
            }
        }
        if (ones != 1u) {
            printf("Dataset label row %zu contains %zu active values; expected exactly 1.\n", row, ones);
            return false;
        }
    }

    return true;
}

int main()
{
    stdio_init_all();

    static constexpr uint32_t clock_frequency_hz = 125000000u; // 125 MHz
    set_sys_clock_khz(clock_frequency_hz / 1000, true);

    if (!ValidateGeneratedDataset()) {
        printf("Generated dataset validation failed.\n");
        return 1;
    }
    printf("Generated dataset validated: first/last feature rows normalized and one-hot labels are valid.\n");

    printf("Press any key to continue...\n");
    while (stdio_usb_connected() == false) {
        tight_loop_contents();
    }
    while (getchar_timeout_us((int64_t)-1) == PICO_ERROR_TIMEOUT) {
        tight_loop_contents();
    }
    printf("\nStarting RAM independence test...\n");

    test::TestRAMIndependence test(clock_frequency_hz);

    printf("Running RAM independence test with clock frequency: %u MHz\n\n", clock_frequency_hz / 1000000u);

    test.RunTest();

    printf("Core 0: iterations=%u, avg_time=%.2f us, max_time=%.2f us, min_time=%.2f us\n",
           test.GetResults().core0.iterations,
           test.GetResults().core0.time_us_avg,
           test.GetResults().core0.time_us_max,
           test.GetResults().core0.time_us_min);
    printf("Core 1: iterations=%u, avg_time=%.2f us, max_time=%.2f us, min_time=%.2f us\n",
           test.GetResults().core1.iterations,
           test.GetResults().core1.time_us_avg,
           test.GetResults().core1.time_us_max,
           test.GetResults().core1.time_us_min);

    printf("\nTest completed.\n");

    while (true) {
        tight_loop_contents();
    }

    return 0;
}
