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

#include "sirius_registration.hpp"

#include <duckdb/common/multi_file/multi_file_states.hpp>
#include <duckdb/planner/operator/logical_get.hpp>

#include <string>

namespace sirius::planner {

/// The column names the table function bound, in LogicalGet order. LogicalGet::names carries the
/// query's column aliases (`AS t(b, a)`), which name no column of the source.
inline duckdb::vector<std::string> scan_source_names(duckdb::LogicalGet const& op)
{
  if (auto const* multi_file = dynamic_cast<duckdb::MultiFileBindData const*>(op.bind_data.get());
      multi_file && multi_file->names.size() == op.names.size()) {
    return duckdb::IdentifiersToStrings(multi_file->names);
  }
  if (auto const* s3 = dynamic_cast<duckdb::SiriusReadParquetBindData const*>(op.bind_data.get());
      s3 && s3->bound_names.size() == op.names.size()) {
    return s3->bound_names;
  }
  return duckdb::IdentifiersToStrings(op.names);
}

}  // namespace sirius::planner
