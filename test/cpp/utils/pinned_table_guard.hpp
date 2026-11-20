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

#include <catch.hpp>
#include <duckdb.hpp>

#include <string>
#include <utility>

namespace sirius::test {

/** Pins a DuckDB table on the GPU tier; destruction unpins it or warns, even after a REQUIRE. */
class pinned_table_guard final {
 public:
  pinned_table_guard(duckdb::Connection& connection, std::string table)
    : con_(connection), table_(std::move(table))
  {
    auto pinned = con_.Query("CALL pin_table(format='duckdb', name='" + table_ + "', tier='gpu');");
    REQUIRE(pinned);
    REQUIRE_FALSE(pinned->HasError());
  }

  ~pinned_table_guard() noexcept
  {
    try {
      auto unpinned = con_.Query("CALL unpin_table('" + table_ + "');");
      if (unpinned && unpinned->HasError()) {
        WARN("pinned_table_guard: failed to unpin " << table_ << ": " << unpinned->GetError());
      }
    } catch (...) {
      // Destructors must not mask the failure that caused scope unwinding.
    }
  }

  pinned_table_guard(pinned_table_guard const&)            = delete;
  pinned_table_guard& operator=(pinned_table_guard const&) = delete;
  pinned_table_guard(pinned_table_guard&&)                 = delete;
  pinned_table_guard& operator=(pinned_table_guard&&)      = delete;

 private:
  duckdb::Connection& con_;
  std::string table_;
};

}  // namespace sirius::test
