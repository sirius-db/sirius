/*
 * Copyright (c) 2026, Alibaba Group Holding Limited
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "duckdb/function/table_function.hpp"
#include "paimon_read_bridge.h"

namespace duckdb {

// Discovery uses the matching DuckDB C++ ABI. Callbacks and descriptors do not
// expose paimon-cpp, Arrow C++, STL objects or exceptions across the boundary.
struct PaimonReadBridgeInfo final : public TableFunctionInfo {
  explicit PaimonReadBridgeInfo(const PaimonReadBridgeV1& api_p) : api(api_p) {}
  const PaimonReadBridgeV1& api;
};

}  // namespace duckdb
