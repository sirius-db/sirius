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

#include "config.hpp"

namespace duckdb {

bool Config::USE_PIN_MEM_FOR_CPU_PROCESSING = true;

bool Config::USE_CUDF_EXPR = true;

bool Config::USE_CUSTOM_TOP_N = true;

bool Config::USE_OPT_TABLE_SCAN = true;
int Config::OPT_TABLE_SCAN_NUM_CUDA_STREAMS = 8;
uint64_t Config::OPT_TABLE_SCAN_CUDA_MEMCPY_SIZE = 64UL * 1024 * 1024;  // 64 MB

uint64_t Config::PRINT_GPU_TABLE_MAX_ROWS = 1000;

bool Config::ENABLE_FALLBACK_CHECK = false;

}