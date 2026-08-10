/*
 * Copyright 2026, Sirius Contributors.
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

#include "vss/vector_join.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace duckdb {
class ClientContext;
class SiriusContext;
}  // namespace duckdb

namespace sirius::vss {

/// Resolve one join side (left or right) at bind time.
std::int64_t resolve_vector_join_side(duckdb::ClientContext& context,
                                      duckdb::SiriusContext& sirius_ctx,
                                      const std::string& label,
                                      const std::string& table_arg,
                                      const std::string& column_arg,
                                      const std::string& schema_name,
                                      const std::vector<std::string>& out_cols,
                                      vector_join_side& side,
                                      duckdb::vector<duckdb::LogicalType>& out_types,
                                      duckdb::vector<duckdb::string>& out_names);

/// Pull a LIST(VARCHAR) named parameter into a string vector; throws if empty.
std::vector<std::string> parse_output_columns(const duckdb::Value& v, const std::string& key);

}  // namespace sirius::vss
