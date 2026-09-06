#include <cstddef>

#include "dataset/Dataset.hpp"
#include "tests/unit/microunit/microunit.h"

UNIT(DatasetValidation) {
    constexpr std::size_t last_index = dataset::kNumExamples - 1u;

    for (std::size_t row = 0; row < dataset::kNumExamples; ++row) {
        const auto& sample = dataset::features[row];
        const bool is_boundary = (row == 0u) || (row == last_index);
        for (std::size_t col = 0; col < dataset::kFeatureSize; ++col) {
            const float value = sample[col];
            ASSERT_TRUE(value >= 0.0f && value <= 1.0f);
            if (is_boundary && col == 0u) {
                // Boundary sanity check: the first and last feature rows remain normalized.
            }
        }
    }

    for (std::size_t row = 0; row < dataset::kNumExamples; ++row) {
        std::size_t ones = 0u;
        for (std::size_t col = 0; col < dataset::kLabelSize; ++col) {
            const float value = dataset::labels[row][col];
            ASSERT_TRUE(value == 0.0f || value == 1.0f);
            if (value == 1.0f) {
                ++ones;
            }
        }
        ASSERT_TRUE(ones == 1u);
    }

    PASS();
}
