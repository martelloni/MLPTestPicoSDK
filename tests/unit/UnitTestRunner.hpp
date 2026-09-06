/**
 * @file UnitTestRunner.hpp
 * @brief Core-aware microunit runner for the Pico test suite.
 */

#ifndef __TESTS_UNIT_UNIT_TEST_RUNNER_HPP__
#define __TESTS_UNIT_UNIT_TEST_RUNNER_HPP__

namespace test {
namespace unit {

/**
 * @brief Runs the registered microunit suite on the selected core.
 *
 * For the default and core-0 builds, the suite executes on core 0 directly.
 * The core-1 branch launches a dedicated trampoline on core 1 and waits for a
 * pass/fail token over the multicore FIFO.
 */
bool RunAllOnSelectedCore();

} // namespace unit
} // namespace test

#endif // __TESTS_UNIT_UNIT_TEST_RUNNER_HPP__
