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

#include "log/logging.hpp"

namespace duckdb {

bool Config::USE_PIN_MEM_FOR_CPU_PROCESSING = true;
bool Config::USE_PIN_MEM_FOR_CACHING        = false;

bool Config::USE_CUDF_EXPR = true;
sirius::expression_evaluator_strategy Config::EXPRESSION_EVALUATOR_STRATEGY =
  sirius::expression_evaluator_strategy::AST_INTERPRET;

bool Config::FILTER_CASCADE_CHEAP_CONJUNCTS = true;
// Below ~1M rows a predicate kernel is launch-latency-bound (tens of microseconds) while the
// cascade adds a handful of launches plus a 4-byte device-to-host count sync of the same order,
// so the crossover sits near this size; above it the saving scales with rows and the overhead
// stays fixed.
uint64_t Config::FILTER_CASCADE_MIN_ROWS = 1ULL << 20;
// Break-even between gathering survivors before the residual (~55 ps per surviving row for a
// ~50 B row at the ~890 GB/s device-to-device gather rate) and the residual string evaluation
// the gather avoids (~180 ps per dropped row, measured JIT string-compare rate):
// 55*s = 180*(1-s) -> s ~ 0.77, rounded down.
double Config::FILTER_CASCADE_MAX_PASS_RATE = 0.75;

bool Config::USE_CUSTOM_TOP_N = true;

bool Config::USE_OPT_TABLE_SCAN                  = true;
int Config::OPT_TABLE_SCAN_NUM_CUDA_STREAMS      = 8;
uint64_t Config::OPT_TABLE_SCAN_CUDA_MEMCPY_SIZE = 64UL * 1024 * 1024;  // 64 MB

uint64_t Config::PRINT_GPU_TABLE_MAX_ROWS = 1000;

bool Config::ENABLE_FALLBACK_CHECK = false;

bool Config::ENABLE_REGEX_JIT_IMPL = true;

bool Config::MODIFIED_PIPELINE = false;

uint64_t Config::DEFAULT_SCAN_TASK_BATCH_SIZE = 512ULL * 1024 * 1024;  ///< 50 MB

uint64_t Config::MAX_SORT_PARTITION_BYTES = 0;  ///< 0 = auto (33% of available GPU memory)

std::string Config::LOG_BACKEND = "spdlog";
std::string Config::LOG_LEVEL   = "info";
std::string Config::LOG_DIR     = "log";
int Config::LOG_FLUSH_SECONDS   = 3;

}  // namespace duckdb
