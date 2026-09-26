# Generated Cmake Pico project file

cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Initialise pico_sdk from installed location
# (note this can come from environment, CMake cache etc)

# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.3.0)
set(toolchainVersion 15_2_Rel1)
set(picotoolVersion 2.3.0)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================
set(PICO_BOARD pico2 CACHE STRING "Board type")

# Pull in Raspberry Pi Pico SDK (must be before project)
include(pico_sdk_import.cmake)

project(MLPTestPicoSDK C CXX ASM)

# Needed early: the unpinned+benchmarks configuration below runs a Python
# helper script (scripts/compute_naive_flooder_size.py) as a build step for a
# probe executable, ahead of the main target's own POST_BUILD validation step.
find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(MEML_MLP_RUNS_ON_CORE "" CACHE STRING "Compile-time MLP placement selector: blank, 0, or 1")

# RAM is scarce on this target (a single 256 KiB copy_to_ram window shared by
# both a full microunit suite and the RAM-independence benchmark's own
# hardcoded network), so the two are mutually exclusive build modes rather
# than both always being linked in. "tests" compiles and runs the unit-test
# suite only; "benchmarks" skips every unit-test translation unit and builds
# the benchmark suite (currently TestRAMIndependence only) instead.
set(RUN_TESTS_OR_BENCHMARKS "tests" CACHE STRING "Build mode: 'tests' builds/runs the unit-test suite only; 'benchmarks' skips unit tests and builds the benchmark suite instead")
set_property(CACHE RUN_TESTS_OR_BENCHMARKS PROPERTY STRINGS tests benchmarks)
if(NOT RUN_TESTS_OR_BENCHMARKS STREQUAL "tests" AND NOT RUN_TESTS_OR_BENCHMARKS STREQUAL "benchmarks")
    message(FATAL_ERROR "RUN_TESTS_OR_BENCHMARKS must be 'tests' or 'benchmarks'.")
endif()

# Initialise the Raspberry Pi Pico SDK
pico_sdk_init()

# Optimisation-3 spike: pull in CMSIS-DSP so arm_dot_prod_f32() is available and
# the ARM_MATH_CM33 path in StaticLayer.h::forward() becomes reachable, so it
# can be benchmarked against the hand-optimised multi-accumulator loop. The
# Pico SDK itself ships only CMSIS-Core (register/intrinsic headers, via the
# cmsis_core/cmsis_core_headers targets below pico_stdlib) -- no CMSIS-DSP.
#
# We fetch the CMSIS-DSP sources but deliberately do NOT build them via the
# upstream Source/CMakeLists.txt target (which compiles every function group,
# including Q7/Q15/Q31/f64 kernels we don't need): with this toolchain
# (arm-none-eabi-gcc 15.2, -mcpu=cortex-m33 ... -mcmse -O3, as set by the Pico
# SDK), several of those unrelated q31 kernels (e.g. arm_abs_q31.c) trigger a
# compiler-internal segfault (ICE) at -O3. Since only the f32 dot product is
# needed for this spike, build just that one translation unit instead.
include(FetchContent)
FetchContent_Declare(
        CMSISDSP
        GIT_REPOSITORY https://github.com/ARM-software/CMSIS-DSP.git
        GIT_TAG v1.18.0
)
FetchContent_Populate(CMSISDSP)

set(PICO_CMSIS_CORE_INCLUDE ${PICO_SDK_PATH}/src/rp2_common/cmsis/stub/CMSIS/Core/Include)

add_library(CMSISDSP_dotprod STATIC
        ${cmsisdsp_SOURCE_DIR}/Source/BasicMathFunctions/arm_dot_prod_f32.c)
target_include_directories(CMSISDSP_dotprod PUBLIC
        ${cmsisdsp_SOURCE_DIR}/Include
        ${PICO_CMSIS_CORE_INCLUDE})
target_include_directories(CMSISDSP_dotprod PRIVATE
        ${cmsisdsp_SOURCE_DIR}/PrivateInclude)
target_compile_definitions(CMSISDSP_dotprod PUBLIC ARM_MATH_LOOPUNROLL)
add_library(CMSISDSP::CMSISDSP ALIAS CMSISDSP_dotprod)

# Add executable. Default name is the project name, version 0.1

set(MLPTestPicoSDK_SOURCES
        main.cpp)

# memory/CoreSectionInit.cpp's meml_core1_preinit() only exists to copy/zero
# the custom per-core SRAM sections the linker fragments below define; with
# those fragments skipped for the unpinned build (Step 7), there is nothing
# left for it to copy or zero, and the linker symbols it references would not
# exist at all.
if(NOT MEML_MLP_RUNS_ON_CORE STREQUAL "")
    list(APPEND MLPTestPicoSDK_SOURCES memory/CoreSectionInit.cpp)
endif()

if(RUN_TESTS_OR_BENCHMARKS STREQUAL "tests")
    list(APPEND MLPTestPicoSDK_SOURCES
        dataset/Dataset.cpp
        experiments/MLPOpticalRecognition.cpp
        tests/unit/UnitTestRunner.cpp
        tests/unit/SmokeTest.cpp
        tests/unit/DatasetValidationTest.cpp
        tests/unit/SoftmaxTest.cpp
        tests/unit/LossTest.cpp
        tests/unit/StaticMLPTrainBatchTest.cpp
        tests/unit/MLPOpticalRecognitionTest.cpp)
else()
    # TestRAMIndependence (pinned builds) and TestNaiveMemoryLayout (unpinned
    # build) both link against the real MLPOpticalRecognition facade rather
    # than a hardcoded network, so the benchmark build needs the same
    # dataset/experiment translation units the tests build does. Both
    # benchmark headers are header-only and included directly by main.cpp, so
    # no separate source file is added for either.
    list(APPEND MLPTestPicoSDK_SOURCES
        dataset/Dataset.cpp
        experiments/MLPOpticalRecognition.cpp)
endif()

add_executable(MLPTestPicoSDK ${MLPTestPicoSDK_SOURCES})

if(NOT MEML_MLP_RUNS_ON_CORE STREQUAL "")
    if(NOT MEML_MLP_RUNS_ON_CORE STREQUAL "0" AND NOT MEML_MLP_RUNS_ON_CORE STREQUAL "1")
        message(FATAL_ERROR "MEML_MLP_RUNS_ON_CORE must be blank, 0, or 1.")
    endif()
    target_compile_definitions(MLPTestPicoSDK PRIVATE MEML_MLP_RUNS_ON_CORE=${MEML_MLP_RUNS_ON_CORE})
endif()

if(RUN_TESTS_OR_BENCHMARKS STREQUAL "tests")
    target_compile_definitions(MLPTestPicoSDK PRIVATE MEML_RUN_UNIT_TESTS)
else()
    target_compile_definitions(MLPTestPicoSDK PRIVATE MEML_ENABLE_RAM_INDEPENDENCE_BENCHMARK)
endif()

# Keep the SDK-generated linker script and inject only the custom per-core SRAM sections.
# This is the supported Pico SDK extension point for custom linker fragments.
# Step 7: the unpinned (MEML_MLP_RUNS_ON_CORE blank) configuration must be a
# genuine naive-layout counterfactual, not the pinned layout with just the
# experiment's own tagging switched off, so it gets none of these custom
# fragments -- no CORE1_RAM region, no explicit .core0_bank/.core1_bank/
# .core1_stack/.time_critical.core1.code sections, none of their ASSERTs.
if(NOT MEML_MLP_RUNS_ON_CORE STREQUAL "")
    pico_add_linker_script_override_path(MLPTestPicoSDK
            ${CMAKE_CURRENT_SOURCE_DIR}/linker
            FILES memory_extra.incl section_extra_post_text.incl section_extra_post_data.incl)
endif()

# Core-1-pinned only: redirects mlp/StaticMLP.h's for_each_layer<F, I>
# instantiations (an untagged template, safe by construction via
# -ffunction-sections' own per-instantiation auto-naming -- see
# MemoryDefs.hpp's MEML_MLP_CODE_MULTI comment) into CORE1_RAM ahead of the
# SDK's own .text* catch-all, instead of paying a real call-overhead cost to
# place them via an ordinary tagged wrapper. Left out of the unpinned and
# core-0-pinned builds entirely: both already get this code into the right
# bank for free (core 0's bank IS the SDK's default RAM window), and
# including this override there would incorrectly redirect their
# for_each_layer instantiations into CORE1_RAM too. See
# linker/section_copy_to_ram_text.incl for the full reasoning.
#
# This lives in its own linker/core1-only subdirectory, not linker/ itself:
# pico_add_linker_script_override_path works by adding its path argument to
# the linker's INCLUDE search path wholesale, not by scoping visibility to
# just the FILES named in one particular call -- so a same-named override
# dropped into linker/ (already unconditionally on the search path for both
# pinned configs, via the call above) would resolve for core 0 too, wrongly
# redirecting ITS for_each_layer instantiations into CORE1_RAM as well.
if(MEML_MLP_RUNS_ON_CORE STREQUAL "1")
    pico_add_linker_script_override_path(MLPTestPicoSDK
            ${CMAKE_CURRENT_SOURCE_DIR}/linker/core1-only
            FILES section_copy_to_ram_text.incl)
endif()

# Core-0-pinned only: MEML_RUNS_ON_CORE_CODE(0)-tagged functions (e.g.
# MLPOpticalRecognition::Initialise/Train) get an explicit
# ".time_critical.core0.code.N" section that the SDK's own .text output
# section rule (`*(.text*)`) does not match, so without this override they
# fall through to the SDK's `.data` output section's later, more permissive
# `*(.time_critical*)` catch-all instead -- still correct (same core-0 SRAM
# window, still copied at boot via .data's own copy-table entry), but it
# means core 0's tagged hot path was never co-located with untagged code it
# calls into (e.g. CMSIS-DSP's arm_dot_prod_f32(), which lands in the
# ordinary `.text` section) the way core 1's build explicitly is. Measured on
# hardware as a build-core0-specific timing regression once CMSIS-DSP was
# added (RunAndSaveBenchmarks.py). See
# linker/core0-only/section_copy_to_ram_text.incl for the full writeup and
# the fix (folding ".time_critical.core0.code*" into `.text` itself).
if(MEML_MLP_RUNS_ON_CORE STREQUAL "0")
    pico_add_linker_script_override_path(MLPTestPicoSDK
            ${CMAKE_CURRENT_SOURCE_DIR}/linker/core0-only
            FILES section_copy_to_ram_text.incl)
endif()

pico_set_program_name(MLPTestPicoSDK "MLPTestPicoSDK")
pico_set_program_version(MLPTestPicoSDK "0.1")

# Reserve the default/core-0 SRAM region as 256 KiB so the remaining 256 KiB bank
# remains dedicated to the selected core-1 layout. Deliberately unconditional,
# including for the unpinned build: leaving that build at the SDK's ~512 KiB
# default would let the linker spread the naive benchmark's static objects far
# enough apart that they might land in different physical SRAM banks by
# chance, silently reproducing a pinned-like layout and defeating Step 7's
# naive-layout counterfactual. Confining it to the same 256 KiB window forces
# every static object -- MLP experiment, flood buffer, ordinary .data/.bss,
# stacks -- through the SDK's own default allocator into one physical region.
pico_set_linker_script_var(MLPTestPicoSDK RAM_LENGTH 256k)

# Keep the image flash-loadable as a standard UF2: the binary is stored in flash and then
# copied into RAM at reset, which is the supported Pico model for generated .uf2 images.
pico_set_binary_type(MLPTestPicoSDK copy_to_ram)

# Modify the below lines to enable/disable output over UART/USB
pico_enable_stdio_uart(MLPTestPicoSDK 0)
pico_enable_stdio_usb(MLPTestPicoSDK 1)

# Add the standard library to the build
target_link_libraries(MLPTestPicoSDK
        pico_stdlib
        pico_multicore
        CMSISDSP::CMSISDSP)

# Add the standard include files to the build
target_include_directories(MLPTestPicoSDK PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
)

# Optimisation-3 spike: define ARM_MATH_CM33 so StaticLayer.h::forward()'s
# existing (previously-dead) CMSIS branch calls arm_dot_prod_f32() instead of
# the scalar loop. __FPU_PRESENT=1 matches this target's hardware FPv5-SP
# (already relied on unconditionally elsewhere in the build via
# -mfloat-abi=softfp -mcpu=cortex-m33 -march=armv8-m.main+fp+dsp).
target_compile_definitions(MLPTestPicoSDK PRIVATE
        ARM_MATH_CM33
        __FPU_PRESENT=1)

pico_add_extra_outputs(MLPTestPicoSDK)

# Step 7: size the unpinned benchmark's RAM flooder to whatever the shared
# 256 KiB window has left after every other static object in the program,
# rather than a fixed guess -- otherwise leftover slack could let the
# compiler/linker keep mlp_experiment_ and ram_flooder_ comfortably apart,
# under-representing naive-layout contention. This needs a two-pass build: a
# throwaway probe with a nominal 1-word flooder measures "everything else",
# then a script computes the real flooder size from the probe's own ELF.
if(MEML_MLP_RUNS_ON_CORE STREQUAL "" AND RUN_TESTS_OR_BENCHMARKS STREQUAL "benchmarks")
    add_executable(MLPTestPicoSDK_naive_flooder_probe
            main.cpp
            dataset/Dataset.cpp
            experiments/MLPOpticalRecognition.cpp)
    target_compile_definitions(MLPTestPicoSDK_naive_flooder_probe PRIVATE
            MEML_ENABLE_RAM_INDEPENDENCE_BENCHMARK
            MEML_NAIVE_FLOODER_PROBE_BUILD)
    target_include_directories(MLPTestPicoSDK_naive_flooder_probe PRIVATE
            ${CMAKE_CURRENT_LIST_DIR})
    target_link_libraries(MLPTestPicoSDK_naive_flooder_probe
            pico_stdlib
            pico_multicore)
    pico_set_binary_type(MLPTestPicoSDK_naive_flooder_probe copy_to_ram)
    pico_set_linker_script_var(MLPTestPicoSDK_naive_flooder_probe RAM_LENGTH 256k)
    # Match the real target's stdio configuration so main.cpp (shared between
    # both) compiles identically; the probe is never actually flashed or run.
    pico_enable_stdio_uart(MLPTestPicoSDK_naive_flooder_probe 0)
    pico_enable_stdio_usb(MLPTestPicoSDK_naive_flooder_probe 1)
    pico_add_extra_outputs(MLPTestPicoSDK_naive_flooder_probe)

    set(MEML_NAIVE_FLOODER_SIZE_HEADER ${CMAKE_BINARY_DIR}/generated/NaiveFlooderSize.hpp)
    add_custom_command(
            OUTPUT ${MEML_NAIVE_FLOODER_SIZE_HEADER}
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/compute_naive_flooder_size.py
                    $<TARGET_FILE:MLPTestPicoSDK_naive_flooder_probe>
                    --readelf ${CMAKE_READELF}
                    --output ${MEML_NAIVE_FLOODER_SIZE_HEADER}
            DEPENDS MLPTestPicoSDK_naive_flooder_probe
            COMMENT "Computing naive-layout RAM flooder size from probe build"
            VERBATIM)
    add_custom_target(MLPTestPicoSDK_naive_flooder_size DEPENDS ${MEML_NAIVE_FLOODER_SIZE_HEADER})
    add_dependencies(MLPTestPicoSDK MLPTestPicoSDK_naive_flooder_size)
    target_include_directories(MLPTestPicoSDK PRIVATE ${CMAKE_BINARY_DIR}/generated)
endif()

# Build-framework gate for the RP2350 core-0/core-1 memory-placement guarantees:
# fails the build if any mlp/ hot-path symbol, or the core-0/core-1 SRAM banks
# themselves, land outside their selected 256 KiB window. See scripts/validate_memory_placement.py.
add_custom_command(TARGET MLPTestPicoSDK POST_BUILD
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/scripts/validate_memory_placement.py
                $<TARGET_FILE:MLPTestPicoSDK>
                --core "${MEML_MLP_RUNS_ON_CORE}"
                --readelf ${CMAKE_READELF}
                --nm ${CMAKE_NM}
                --mode ${RUN_TESTS_OR_BENCHMARKS}
        COMMENT "Validating core-0/core-1 memory placement"
        VERBATIM)
