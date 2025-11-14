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

#include <cstdint>

namespace duckdb {

// If you are adding a new field to this struct, then you also need to make the following changes:
// * Specify the default value in config.cpp
// * Add a configuration field associated with Sirius (see InitialGPUConfigs in sirius_extension.cpp for examples)
struct Config {
  // For gpu buffer manager
  static bool USE_PIN_MEM_FOR_CPU_PROCESSING; // use_pin_memory

  // For expression executor
  static bool USE_CUDF_EXPR; // use_cudf_expr
  
  // For gpu physical top-N
  static bool USE_CUSTOM_TOP_N; // use_custom_top_n

  // For gpu physical table scan
  static bool USE_OPT_TABLE_SCAN; // use_opt_table_scan
  static int OPT_TABLE_SCAN_NUM_CUDA_STREAMS; // opt_table_scan_num_streams
  static uint64_t OPT_TABLE_SCAN_CUDA_MEMCPY_SIZE; // opt_table_scan_memcpy_size

  // For printing gpu table
  static uint64_t PRINT_GPU_TABLE_MAX_ROWS;

  // For checking whether to fall back to duckdb execution
  static bool ENABLE_FALLBACK_CHECK;

  // Whether to use special JIT implementation for particular regex evaluation
  static bool ENABLE_REGEX_JIT_IMPL;
};

}
