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

#include "duckdb/function/function_set.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/database.hpp"
#include "exec/stream_bind_catalog.hpp"

namespace sirius::exec {

inline constexpr const char* kStreamSourceFunctionName = "sirius_stream_source";

/// Bind data for sirius_stream_source(stream_id). Schema comes from stream_bind_catalog.
struct stream_source_bind_data : public duckdb::FunctionData {
  explicit stream_source_bind_data(stream_id_t stream_id) : stream_id(stream_id) {}

  stream_id_t stream_id;

  duckdb::unique_ptr<duckdb::FunctionData> Copy() const override
  {
    return duckdb::make_uniq<stream_source_bind_data>(stream_id);
  }

  bool Equals(duckdb::FunctionData const& other_p) const override
  {
    auto const& other = other_p.Cast<stream_source_bind_data>();
    return stream_id == other.stream_id;
  }
};

/// Register sirius_stream_source(id). Bind reads stream_bind_catalog; body never runs
/// (replaced by STREAMING_SOURCE).
void register_stream_source_function(duckdb::DatabaseInstance& instance);

}  // namespace sirius::exec
