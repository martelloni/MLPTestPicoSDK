#ifndef __DATASET_HPP__
#define __DATASET_HPP__

#include <array>
#include <cstddef>
#include "MemoryDefs.hpp"

namespace dataset {

constexpr std::size_t kFeatureSize = 64;
constexpr std::size_t kLabelSize = 10;
constexpr std::size_t kNumExamples = 3823;
using DataType = float;

extern const std::array<std::array<DataType, kFeatureSize>, kNumExamples> features;
extern const std::array<std::array<DataType, kLabelSize>, kNumExamples> labels;

};  // namespace dataset

#endif // __DATASET_HPP__
