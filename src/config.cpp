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

std::atomic<bool> Config::USE_PIN_MEM_FOR_CPU_PROCESSING = true;
std::atomic<bool> Config::USE_PIN_MEM_FOR_CACHING        = false;

std::atomic<bool> Config::USE_CUDF_EXPR = true;
std::atomic<sirius::expression_evaluator_strategy> Config::EXPRESSION_EVALUATOR_STRATEGY =
  sirius::expression_evaluator_strategy::AST_INTERPRET;

std::atomic<bool> Config::USE_CUSTOM_TOP_N = true;

std::atomic<bool> Config::USE_OPT_TABLE_SCAN                  = true;
std::atomic<int> Config::OPT_TABLE_SCAN_NUM_CUDA_STREAMS      = 8;
std::atomic<uint64_t> Config::OPT_TABLE_SCAN_CUDA_MEMCPY_SIZE = 64UL * 1024 * 1024;  // 64 MB

std::atomic<uint64_t> Config::PRINT_GPU_TABLE_MAX_ROWS = 1000;

std::atomic<bool> Config::ENABLE_FALLBACK_CHECK = false;

std::atomic<bool> Config::ENABLE_REGEX_JIT_IMPL = true;

std::atomic<bool> Config::MODIFIED_PIPELINE = false;

std::atomic<uint64_t> Config::DEFAULT_SCAN_TASK_BATCH_SIZE = 512ULL * 1024 * 1024;  ///< 50 MB

std::atomic<uint64_t> Config::MAX_SORT_PARTITION_BYTES =
  0;  ///< 0 = auto (33% of available GPU memory)

ConfigString Config::LOG_BACKEND{"spdlog"};
ConfigString Config::LOG_LEVEL{"info"};
ConfigString Config::LOG_DIR{"log"};
std::atomic<int> Config::LOG_FLUSH_SECONDS = 3;

}  // namespace duckdb
