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
#include "duckdb/execution/execution_context.hpp"
#include "duckdb/main/client_context.hpp"
#include "gpu_pipeline.hpp"
#include "helper/helper.hpp"
#include "operator/gpu_physical_table_scan.hpp"

namespace sirius::op::scan {

class duckdb_scan_metadata {
 public:
  // constructor initializing function and op
  duckdb_scan_metadata(duckdb::ClientContext& client_context, duckdb::GPUPhysicalTableScan& op)
    : _client_context(client_context), _op(op)
  {
  }
  //
  ~duckdb_scan_metadata() = default;
  duckdb::ClientContext& _client_context;  // The client context for the scan operation
  duckdb::GPUPhysicalTableScan&
    _op;  // The GPU physical table scan operator associated with this executor
};

}  // namespace sirius::op::scan
