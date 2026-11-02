/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

/**
 * @file
 * @brief The NVRTC expression evaluator device runtime: the CUDA source pasted in front of every
 *        generated `ProjectSink` and `FilterSink` kernel.
 *
 * The runtime is split into fragments that are only meaningful in this order (each one uses the
 * definitions of the ones before it). The build concatenates the fragments in the same order into
 * `nvrtc_device_runtime_source()` (see `cmake/embed_nvrtc_device_runtime.cmake`), and this umbrella
 * header exists so the identical text can be compiled offline with nvcc (`device_runtime_check.cu`
 * does so on every build) and unit-tested as ordinary device code. A `.cuh` fragment must not
 * `#include` a sibling fragment: NVRTC resolves only the embedded `codegen/stdint_shim.hpp`.
 *
 * Many of these fragments reproduce library code found elsewhere, but the goal is to minimize
 * compilation latency and pre-compiled binary loading, which including library headers increases
 * appreciably.
 */

// clang-format off: the order below is the dependency order, not alphabetical.
#include "abi_constants.cuh"
#include "reg_val.cuh"
#include "arithmetic.cuh"
#include "decimal.cuh"
#include "string.cuh"
#include "temporal.cuh"
// clang-format on
