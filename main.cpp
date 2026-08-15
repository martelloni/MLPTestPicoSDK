#include <stdio.h>
#include "pico/stdlib.h"
#include "MemoryDefs.hpp"
#include "pico/multicore.h"

static void core0Task();
static void core1Task();

int main()
{
    // Launch core 1
    multicore_launch_core1(core1Task);
    // Start core 0 task
    core0Task();

    return 0;
}

MEML_RUNS_ON_CORE(0) void core0Task()
{
    while (true) {
        printf("Hello from core 0!\n");
        sleep_ms(1000);
    }
}

MEML_RUNS_ON_CORE(1) void core1Task()
{
    gpio_init(33);
    gpio_set_dir(33, true);

    while (true) {
        gpio_put(33, 1);
        sleep_ms(500);
        gpio_put(33, 0);
        sleep_ms(500);
    }
}
