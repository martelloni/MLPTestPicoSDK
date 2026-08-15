# Generated Cmake Pico project file

cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
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

# Initialise the Raspberry Pi Pico SDK
pico_sdk_init()

# Add executable. Default name is the project name, version 0.1

add_executable(MLPTestPicoSDK main.cpp )

# Keep the SDK-generated linker script and inject only the custom per-core SRAM sections.
# This is the supported Pico SDK extension point for custom linker fragments.
pico_add_linker_script_override_path(MLPTestPicoSDK
        ${CMAKE_CURRENT_SOURCE_DIR}/linker
        FILES memory_extra.incl section_extra_post_text.incl section_extra_post_data.incl)

pico_set_program_name(MLPTestPicoSDK "MLPTestPicoSDK")
pico_set_program_version(MLPTestPicoSDK "0.1")

# no_flash means the target is to run from RAM
pico_set_binary_type(MLPTestPicoSDK no_flash)

# Modify the below lines to enable/disable output over UART/USB
pico_enable_stdio_uart(MLPTestPicoSDK 0)
pico_enable_stdio_usb(MLPTestPicoSDK 1)

# Add the standard library to the build
target_link_libraries(MLPTestPicoSDK
        pico_stdlib
        pico_multicore)

# Add the standard include files to the build
target_include_directories(MLPTestPicoSDK PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
)

pico_add_extra_outputs(MLPTestPicoSDK)

