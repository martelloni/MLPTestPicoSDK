#include <stdio.h>
#include "pico/stdlib.h"
#include "hardware/clocks.h"
#include "hardware/vreg.h"
#include "tests/TestRAMIndependence.hpp"

int main()
{
    stdio_init_all();

    static constexpr uint32_t clock_frequency_hz = 125000000u; // 125 MHz
    set_sys_clock_khz(clock_frequency_hz / 1000, true);

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
