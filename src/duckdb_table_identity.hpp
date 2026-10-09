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

#include <duckdb/common/shared_ptr.hpp>
#include <duckdb/common/typedefs.hpp>

namespace duckdb {
class RowGroupCollection;
}

namespace sirius {

/// OID distinguishes DROP/CREATE. ALTER preserves OID, but a value/column-layout
/// rewrite replaces the RowGroupCollection, even for TYPE INTEGER USING a + 100.
/// Weak ownership rejects destroyed/reallocated storage without keeping old tables alive.
/// INSERT, DELETE and CHECKPOINT preserve the collection and retain their MVCC checks.
struct duckdb_table_identity {
  duckdb::idx_t oid{0};
  duckdb::weak_ptr<duckdb::RowGroupCollection> row_groups;

  [[nodiscard]] bool matches(duckdb_table_identity const& other) const
  {
    if (oid == 0 || oid != other.oid) { return false; }
    auto stored = row_groups.lock();
    return stored && stored == other.row_groups.lock();
  }
};

}  // namespace sirius
