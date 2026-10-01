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

sirius::expression_evaluator_strategy Config::EXPRESSION_EVALUATOR_STRATEGY =
  sirius::expression_evaluator_strategy::AST_INTERPRET;

bool Config::ENABLE_REGEX_JIT_IMPL = true;

uint64_t Config::DEFAULT_SCAN_TASK_BATCH_SIZE = 512ULL * 1024 * 1024;  ///< 50 MB

uint64_t Config::MAX_SORT_PARTITION_BYTES = 0;  ///< 0 = auto (33% of available GPU memory)

std::string Config::LOG_BACKEND = "spdlog";
std::string Config::LOG_LEVEL   = "info";
std::string Config::LOG_DIR     = "log";
int Config::LOG_FLUSH_SECONDS   = 3;

}  // namespace duckdb
