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

/**
 * @file test_pin_registry_epoch_mutations.cpp
 * @brief Every exit from a registry-mutating member moves the pin-registry epoch.
 *
 * A finalize-validated plan is reused only while the epoch it was built against still
 * matches. The two exits that a per-site bump misses are pinned down here: the same-row-count
 * merge's early return, and a throw that lands after the stale entry was erased but before its
 * replacement was inserted. Either would let a stale plan run against a registry it was not
 * built for.
 */

#include "memory/topology_index.hpp"
#include "scan/test_utils.hpp"
#include "scan_manager/sirius_scan_manager.hpp"

#include <cudf/column/column_factories.hpp>

#include <catch.hpp>
#include <cucascade/memory/topology_discovery.hpp>
#include <duckdb/common/types.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using sirius::pinned_column_storage_matrix;
using sirius::pinned_column_storage_meta;
using sirius::scan_manager::cache_entry_info;
using sirius::scan_manager::pinned_entry;
using sirius::scan_manager::scan_manager_config;
using sirius::scan_manager::sirius_scan_manager;

constexpr char const* kTable = "epoch_t";
constexpr cudf::data_type kInt64{cudf::type_id::INT64};

std::shared_ptr<const sirius::memory::topology_index> single_gpu_index()
{
  cucascade::memory::system_topology_info topology;
  topology.num_gpus = 1;
  cucascade::memory::gpu_topology_info gpu;
  gpu.id        = 0;
  gpu.numa_node = 0;
  topology.gpus.push_back(std::move(gpu));
  return std::make_shared<sirius::memory::topology_index>(std::move(topology), std::vector<int>{0});
}

cache_entry_info make_cache_info(std::vector<std::size_t> const& primary_indices)
{
  cache_entry_info info;
  info.catalog_name = "memory";
  info.schema_name  = "main";
  info.table_name   = kTable;
  for (auto const idx : primary_indices) {
    info.column_ids.emplace_back(idx);
    info.names.push_back("c" + std::to_string(idx));
  }
  return info;
}

/// One chunk of @p n_columns INT64 columns with @p n_rows rows.
std::vector<std::unique_ptr<cudf::table>> one_chunk(std::size_t n_columns, cudf::size_type n_rows)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  for (std::size_t i = 0; i < n_columns; ++i) {
    columns.push_back(cudf::make_fixed_width_column(kInt64, n_rows));
  }
  std::vector<std::unique_ptr<cudf::table>> tables;
  tables.push_back(std::make_unique<cudf::table>(std::move(columns)));
  return tables;
}

pinned_column_storage_matrix storage_for(std::size_t n_columns)
{
  return pinned_column_storage_matrix{
    std::vector<pinned_column_storage_meta>(n_columns, pinned_column_storage_meta{kInt64, false})};
}

struct epoch_fixture {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> memory =
    initialize_memory_manager(1);
  sirius_scan_manager manager{scan_manager_config{}, *memory, single_gpu_index()};
  cucascade::memory::memory_space* space =
    sirius::scan_test_utils::get_space(*memory, cucascade::memory::Tier::GPU);

  /// GPU-tier insert of @p primary_indices over one chunk of @p n_rows rows; this is the path
  /// with the merge branch and the erase-then-rebuild branch.
  void insert(std::vector<std::size_t> const& primary_indices,
              cudf::size_type n_rows,
              std::size_t n_table_columns)
  {
    manager.insert_pinned_entry(kTable,
                                make_cache_info(primary_indices),
                                one_chunk(n_table_columns, n_rows),
                                std::vector<cucascade::memory::memory_space*>{space},
                                {},
                                {},
                                storage_for(primary_indices.size()));
  }

  [[nodiscard]] std::uint64_t epoch() const { return manager.pin_registry_epoch(); }

  [[nodiscard]] bool has_entry() const
  {
    bool found = false;
    manager.visit_pinned_entries([&](std::string_view name, pinned_entry const&) {
      if (name == kTable) { found = true; }
      return true;
    });
    return found;
  }

  [[nodiscard]] std::size_t entry_column_count() const
  {
    std::size_t count = 0;
    manager.visit_pinned_entries([&](std::string_view name, pinned_entry const& entry) {
      if (name == kTable) { count = entry.cache_info.column_ids.size(); }
      return true;
    });
    return count;
  }
};

}  // namespace

TEST_CASE("the same-row-count merge path moves the pin-registry epoch",
          "[scan_manager][pin_registry_epoch]")
{
  epoch_fixture f;
  f.insert({0}, 8, 1);
  REQUIRE(f.entry_column_count() == 1);

  auto const before_merge = f.epoch();
  // Same name, same row count, same chunk boundaries: takes the merge branch's early return.
  f.insert({1}, 8, 1);
  REQUIRE(f.entry_column_count() == 2);
  CHECK(f.epoch() > before_merge);
}

TEST_CASE("a throw after the stale entry is erased still moves the pin-registry epoch",
          "[scan_manager][pin_registry_epoch]")
{
  epoch_fixture f;
  f.insert({0}, 8, 1);
  REQUIRE(f.has_entry());

  auto const before_failed_repin = f.epoch();
  // A different row count skips the merge and erases the existing entry; the incoming table
  // then declares two columns but carries one, so the rebuild throws before re-inserting.
  // The registry has been mutated (the pin is gone) and the epoch must say so.
  REQUIRE_THROWS_AS(f.insert({0, 1}, 4, 1), std::runtime_error);
  CHECK_FALSE(f.has_entry());
  CHECK(f.epoch() > before_failed_repin);
}

TEST_CASE("attaching proven-unique columns moves the pin-registry epoch",
          "[scan_manager][pin_registry_epoch]")
{
  // The comparison-join planner reads proven uniqueness at plan time, so it is plan-visible
  // registry state like any other.
  epoch_fixture f;
  f.insert({0, 1}, 8, 2);
  auto const before_attach = f.epoch();
  f.manager.attach_proven_unique_columns(kTable, std::vector<std::string>{"c0"});
  CHECK(f.epoch() > before_attach);
}

TEST_CASE("removing a pinned entry moves the pin-registry epoch",
          "[scan_manager][pin_registry_epoch]")
{
  epoch_fixture f;
  f.insert({0}, 8, 1);
  auto const before_remove = f.epoch();
  f.manager.remove_pinned_entry(kTable);
  CHECK_FALSE(f.has_entry());
  CHECK(f.epoch() > before_remove);
}
