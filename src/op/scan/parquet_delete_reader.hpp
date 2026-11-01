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

#include <duckdb/common/open_file_info.hpp>
#include <duckdb/common/types.hpp>

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace duckdb {
class ClientContext;
class DataChunk;
class CachingFileHandle;
}  // namespace duckdb

namespace sirius::op::scan {

struct parquet_delete_column {
  std::string name;
  duckdb::LogicalType type;
};

struct parquet_delete_read_options {
  std::optional<uint64_t> expected_size;
  std::optional<uint64_t> expected_footer_size;
  // Throws on cancellation or expired caller-owned execution/transaction state.
  std::function<void()> check_stop;
  // Optional additional binding-identity check for remote inputs. The default
  // checks size, modification time and version tag exposed by the filesystem;
  // range-read consistency and credentials follow that filesystem's native
  // Parquet contract. Formats requiring strong identity must enforce it through
  // open options/backend policy and this callback. Called on every handle before
  // and after decoding; must not reopen/rebind the object.
  std::function<void(duckdb::CachingFileHandle&)> verify_remote;
};

// Called before decoding: validate format-specific schema, reserve output budget,
// and return physical column indices in the desired output order.
using parquet_delete_bind =
  std::function<std::vector<uint64_t>(std::vector<parquet_delete_column> const&, uint64_t)>;

// Synchronous, on the caller's thread and ClientContext (including credentials).
// The caller owns the active query/transaction needed for filesystem secret lookup.
// Batches are borrowed until consume returns. Callers must discard accumulated
// output on any exception; successful return certifies complete footer coverage.
// Snapshot selection, delete applicability and retained output budgets belong to
// the caller. DuckDB owns transient decode buffers; no whole-file materialization.
void read_parquet_delete_batches(duckdb::ClientContext& context,
                                 duckdb::OpenFileInfo file,
                                 parquet_delete_read_options const& options,
                                 parquet_delete_bind const& bind,
                                 std::function<void(duckdb::DataChunk&)> const& consume);

}  // namespace sirius::op::scan
