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

#include <duckdb.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/duck_table_entry.hpp>
#include <duckdb/storage/data_table.hpp>
#include <duckdb_table_identity.hpp>

#include <stdexcept>
#include <string>

namespace sirius::test {

/// Owns real DuckDB storage for metadata tests; no GPU or disk required.
struct duckdb_identity_fixture {
  duckdb::DuckDB db{nullptr};
  duckdb::Connection con{db};

  duckdb_identity_fixture() { run("CREATE TABLE identity_t(a INTEGER);"); }

  void run(std::string const& sql)
  {
    auto result = con.Query(sql);
    if (result->HasError()) { throw std::runtime_error(result->GetError()); }
  }

  duckdb_table_identity current()
  {
    con.BeginTransaction();
    try {
      auto& table =
        duckdb::Catalog::GetEntry<duckdb::DuckTableEntry>(*con.context, "", "main", "identity_t");
      duckdb_table_identity result{table.oid, table.GetStorage().GetRowGroupCollection()};
      con.Rollback();
      return result;
    } catch (...) {
      con.Rollback();
      throw;
    }
  }
};

/// Share a live collection while letting metadata-only cases choose their own OID.
inline duckdb_table_identity test_table_identity(duckdb::idx_t oid)
{
  static duckdb_identity_fixture fixture;
  static auto const storage = fixture.current().row_groups;
  return {oid, storage};
}

}  // namespace sirius::test
