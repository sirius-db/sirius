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

struct Config {
  // For gpu buffer manager
  static constexpr bool USE_PIN_MEM_FOR_CPU_PROCESSING = true;

  // For expression executor
  static constexpr bool USE_CUDF_EXPR = true;
  
  // For gpu physical top-N
  static constexpr bool USE_CUSTOM_TOP_N = true;

  // For gpu physical table scan
  static constexpr bool USE_OPT_TABLE_SCAN = true;
  static constexpr int OPT_TABLE_SCAN_NUM_CUDA_STREAMS = 8;
  static constexpr uint64_t OPT_TABLE_SCAN_CUDA_MEMCPY_SIZE = 64UL * 1024 * 1024;  // 64 MB
};

}
