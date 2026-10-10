/**
 * @file MLPExperiment.hpp
 * @brief Build-time engine selector for the optical-recognition experiment.
 *
 * MLPExperiment is MLPOpticalRecognition (static engine, the default, byte-for-
 * byte the pre-existing code path) unless the build defines
 * MEML_MLP_ENGINE_BASELINE (CMake: -DMEML_MLP_ENGINE=baseline), in which case
 * it is MLPOpticalRecognitionBaseline (old mlp/MLP.h engine). Never a runtime
 * choice; exactly one engine is compiled into any binary.
 */
#ifndef __MLP_EXPERIMENT_HPP__
#define __MLP_EXPERIMENT_HPP__

#if defined(MEML_MLP_ENGINE_BASELINE)
#include "experiments/MLPOpticalRecognitionBaseline.hpp"
using MLPExperiment = MLPOpticalRecognitionBaseline;
#else
#include "experiments/MLPOpticalRecognition.hpp"
using MLPExperiment = MLPOpticalRecognition;
#endif

#endif  // __MLP_EXPERIMENT_HPP__
