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

#include "catch.hpp"
#include "io/datasource_factory.hpp"
#include "sirius_config.hpp"

#include <cudf/io/datasource.hpp>

#include <duckdb.hpp>
#include <duckdb/main/connection.hpp>

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>

using sirius::sirius_config;
using sirius::io::datasource_factory;
using sirius::io::datasource_registry;

namespace {

// Build a small parquet file via DuckDB so the test stays CPU-only and does
// not depend on any cuda-side init.  Returns the path of the written file;
// caller removes it on scope exit via scoped_parquet_file.
std::filesystem::path write_tiny_parquet(std::string const& label, std::size_t num_rows)
{
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);

  auto create = con.Query("CREATE TABLE t(id INTEGER, v BIGINT, p DOUBLE)");
  REQUIRE(create);
  REQUIRE(!create->HasError());

  auto insert = con.Query("INSERT INTO t SELECT i, i*100, i*1.5 FROM range(" +
                          std::to_string(num_rows) + ") t(i)");
  REQUIRE(insert);
  REQUIRE(!insert->HasError());

  auto path =
    std::filesystem::temp_directory_path() / ("sirius_factory_scan_" + label + ".parquet");
  auto copy = con.Query("COPY t TO '" + path.string() + "' (FORMAT PARQUET, COMPRESSION snappy)");
  REQUIRE(copy);
  REQUIRE(!copy->HasError());
  return path;
}

struct scoped_parquet_file {
  std::filesystem::path path;
  ~scoped_parquet_file()
  {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
};

}  // namespace

TEST_CASE("scan_local_parquet_via_factory_equivalent_to_cudf_direct", "[parquet_scan]")
{
  scoped_parquet_file file{write_tiny_parquet("equivalence", /*num_rows=*/128)};

  datasource_registry reg;
  sirius_config cfg;

  // Reference path: cudf's default local-file datasource.
  auto ds_direct     = cudf::io::datasource::create(file.path.string());
  auto const n_bytes = ds_direct->size();
  REQUIRE(n_bytes > 0);

  // Path under test: local files bypass the registry and use cudf's default
  // local-file datasource.
  auto ds_factory = datasource_factory::create(file.path.string(), reg, cfg);

  REQUIRE(ds_factory != nullptr);
  REQUIRE(ds_factory->size() == n_bytes);

  // Whole-file read: the two datasources must yield identical bytes.  This is
  // the load-bearing guarantee for the scan task substitution — if the parquet
  // reader would observe different bytes via the two paths, the scan would
  // produce a different cudf::table.
  auto buf_direct  = ds_direct->host_read(0, n_bytes);
  auto buf_factory = ds_factory->host_read(0, n_bytes);
  REQUIRE(buf_direct != nullptr);
  REQUIRE(buf_factory != nullptr);
  REQUIRE(buf_factory->size() == n_bytes);
  REQUIRE(std::memcmp(buf_direct->data(), buf_factory->data(), n_bytes) == 0);

  // Range read covering only the parquet footer trailer (last 8 bytes) — the
  // shape cuDF actually uses when it probes for metadata.  Mirrors the
  // scan-task access pattern at parquet_scan_task.cpp's footer fetch.
  constexpr std::size_t tail = 8;
  REQUIRE(n_bytes >= tail);
  auto tail_direct  = ds_direct->host_read(n_bytes - tail, tail);
  auto tail_factory = ds_factory->host_read(n_bytes - tail, tail);
  REQUIRE(tail_direct->size() == tail);
  REQUIRE(tail_factory->size() == tail);
  CHECK(std::memcmp(tail_direct->data(), tail_factory->data(), tail) == 0);
}

TEST_CASE("scan_handles_missing_file_via_factory_error_path", "[parquet_scan]")
{
  datasource_registry reg;
  sirius_config cfg;

  auto missing =
    std::filesystem::temp_directory_path() / "sirius_factory_definitely_not_here.parquet";
  std::error_code ec;
  std::filesystem::remove(missing, ec);  // make doubly sure it doesn't exist

  CHECK_THROWS(datasource_factory::create(missing.string(), reg, cfg));
}
