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

#include "duckdb.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/storage/statistics/node_statistics.hpp"

#include <cstddef>
#include <string>
#include <utility>

namespace duckdb {
struct DBConfig;

// Bind-time payload for the sirius_read_parquet table function. Carries the
// canonical URI (also passed in parameters[0] — the pipeline converter still
// reads it from there) plus the parquet's footer row count, so DuckDB's
// optimizer sees a real cardinality estimate via the registered cardinality
// callback instead of falling back to "unknown table function output".
struct SiriusReadParquetBindData : public FunctionData {
  SiriusReadParquetBindData(std::string uri, std::size_t total_num_rows)
    : uri(std::move(uri)), total_num_rows(total_num_rows)
  {
  }

  std::string uri;
  std::size_t total_num_rows{0};

  unique_ptr<FunctionData> Copy() const override
  {
    return make_uniq<SiriusReadParquetBindData>(uri, total_num_rows);
  }

  bool Equals(FunctionData const& other_p) const override
  {
    auto const& other = other_p.Cast<SiriusReadParquetBindData>();
    return uri == other.uri && total_num_rows == other.total_num_rows;
  }
};

// Cardinality callback for sirius_read_parquet. Returns the footer row count
// (exact value, doubles as both estimated and max cardinality). Returns
// nullptr on a null or wrong-type FunctionData so DuckDB falls back to its
// default behavior.
unique_ptr<NodeStatistics> SiriusReadParquetCardinality(ClientContext& context,
                                                        FunctionData const* bind_data);

class SiriusExtension : public Extension {
 public:
  void Load(ExtensionLoader& loader) override;
  std::string Name() override;
  std::string Version() const override;
  static void InitialGPUConfigs(DBConfig& db);
  static void RegisterGPUFunctions(DatabaseInstance& catalog);
  static void GPUExecutionFunction(ClientContext& context,
                                   TableFunctionInput& data_p,
                                   DataChunk& output);
  static unique_ptr<FunctionData> GPUExecutionBind(ClientContext& context,
                                                   TableFunctionBindInput& input,
                                                   vector<LogicalType>& return_types,
                                                   vector<string>& names);

  static void PinTableFunction(ClientContext& context,
                               TableFunctionInput& data_p,
                               DataChunk& output);
  static unique_ptr<FunctionData> PinTableBind(ClientContext& context,
                                               TableFunctionBindInput& input,
                                               vector<LogicalType>& return_types,
                                               vector<string>& names);

  static void UnpinTableFunction(ClientContext& context,
                                 TableFunctionInput& data_p,
                                 DataChunk& output);
  static unique_ptr<FunctionData> UnpinTableBind(ClientContext& context,
                                                 TableFunctionBindInput& input,
                                                 vector<LogicalType>& return_types,
                                                 vector<string>& names);
};

}  // namespace duckdb
