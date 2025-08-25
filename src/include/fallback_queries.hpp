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

#include "duckdb/common/exception.hpp"

#include <string>
#include <unordered_map>

namespace duckdb {

std::unordered_map<std::string, std::string> FALLBACK_QUERIES = {
};

void check_fallback_queries(const std::string& query) {
  auto it = FALLBACK_QUERIES.find(query);
  if (it != FALLBACK_QUERIES.end()) {
    throw InternalException("Fallback unsupported queries: %s", it->second);
  }
}

}
