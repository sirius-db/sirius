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

// find_pinned_entry_for_duckdb_table over a split pin layout: one table pinned
// under two names with different column sets. The coverage and type cases are
// directed — a first-identity-match rule fails one of each pair for any
// unordered-map iteration order.

#include "memory/topology_index.hpp"
#include "scan/test_utils.hpp"
#include "scan_manager/sirius_scan_manager.hpp"

#include <cudf/column/column_factories.hpp>

#include <catch.hpp>
#include <cucascade/memory/topology_discovery.hpp>
#include <duckdb/common/column_index.hpp>
#include <duckdb/common/types.hpp>

#include <cstddef>
#include <memory>
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

constexpr char const* kCatalog = "memory";
constexpr char const* kSchema  = "main";
constexpr char const* kTable   = "orders";

cucascade::memory::system_topology_info single_gpu_topology()
{
  cucascade::memory::system_topology_info topology;
  topology.num_gpus = 1;
  cucascade::memory::gpu_topology_info gpu;
  gpu.id        = 0;
  gpu.numa_node = 0;
  topology.gpus.push_back(std::move(gpu));
  return topology;
}

std::shared_ptr<const sirius::memory::topology_index> single_gpu_index()
{
  return std::make_shared<sirius::memory::topology_index>(single_gpu_topology(),
                                                          std::vector<int>{0});
}

cache_entry_info make_cache_info(std::vector<std::size_t> const& primary_indices,
                                 std::string const& table = kTable)
{
  cache_entry_info info;
  info.catalog_name = kCatalog;
  info.schema_name  = kSchema;
  info.table_name   = table;
  for (auto const idx : primary_indices) {
    info.column_ids.emplace_back(idx);
    info.names.push_back("c" + std::to_string(idx));
  }
  return info;
}

duckdb::vector<duckdb::ColumnIndex> request(std::vector<std::size_t> const& primary_indices)
{
  duckdb::vector<duckdb::ColumnIndex> ids;
  for (auto const idx : primary_indices) {
    ids.emplace_back(idx);
  }
  return ids;
}

/// Zero-chunk pin: the lookup reads only cache_info.
void pin_metadata_entry(sirius_scan_manager& manager,
                        std::string const& name,
                        std::vector<std::size_t> const& primary_indices)
{
  manager.insert_pinned_entry(name, make_cache_info(primary_indices), {}, {}, {}, {}, {});
}

/// 'orders' covers every column but c8; the 'main.orders' split covers {0, 1, 8}.
struct split_pin_fixture {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> memory =
    initialize_memory_manager(1);
  sirius_scan_manager manager{scan_manager_config{}, *memory, single_gpu_index()};

  split_pin_fixture()
  {
    pin_metadata_entry(manager, "orders", {0, 1, 2, 3, 4, 5, 6, 7});
    pin_metadata_entry(manager, "main.orders", {0, 1, 8});
  }
};

bool is_wide_entry(pinned_entry const* entry)
{
  REQUIRE(entry != nullptr);
  return entry->cache_info.column_ids.size() == 8u;
}

constexpr cudf::data_type kInt64{cudf::type_id::INT64};
constexpr cudf::data_type kInt32{cudf::type_id::INT32};

/// One chunk of @p n_columns INT64 columns; insert validates carriers against these.
std::vector<std::unique_ptr<cudf::table>> one_int64_chunk(std::size_t n_columns)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  for (std::size_t i = 0; i < n_columns; ++i) {
    columns.push_back(cudf::make_fixed_width_column(kInt64, 4));
  }
  std::vector<std::unique_ptr<cudf::table>> tables;
  tables.push_back(std::make_unique<cudf::table>(std::move(columns)));
  return tables;
}

/// Pin one INT64 chunk recording @p native as its pin-time mapping. INT32 models
/// an entry no longer matching the INT64 the scan's BIGINT maps to.
void pin_typed_entry(sirius_scan_manager& manager,
                     sirius::memory::sirius_memory_reservation_manager& memory,
                     std::string const& name,
                     std::vector<std::size_t> const& primary_indices,
                     cudf::data_type native)
{
  pinned_column_storage_matrix storage{std::vector<pinned_column_storage_meta>(
    primary_indices.size(), pinned_column_storage_meta{kInt64, false, native})};
  manager.insert_pinned_entry(
    name,
    make_cache_info(primary_indices),
    one_int64_chunk(primary_indices.size()),
    std::vector<cucascade::memory::memory_space*>{
      sirius::scan_test_utils::get_space(memory, cucascade::memory::Tier::GPU)},
    {},
    {},
    std::move(storage));
}

/// Both entries cover {c0, c1}; only @p matching keeps the INT64 mapping. The
/// guard reads the other as unpinned, so preferring it loses a servable hit.
struct type_match_fixture {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> memory =
    initialize_memory_manager(1);
  sirius_scan_manager manager{scan_manager_config{}, *memory, single_gpu_index()};
  duckdb::vector<duckdb::LogicalType> returned_types{duckdb::LogicalType::BIGINT,
                                                     duckdb::LogicalType::BIGINT};
  std::string matching_name;

  explicit type_match_fixture(std::string matching) : matching_name(std::move(matching))
  {
    for (auto const* name : {"orders", "main.orders"}) {
      pin_typed_entry(manager, *memory, name, {0, 1}, name == matching_name ? kInt64 : kInt32);
    }
  }

  [[nodiscard]] static bool types_match(pinned_entry const* entry)
  {
    REQUIRE(entry != nullptr);
    REQUIRE_FALSE(entry->column_storage.empty());
    REQUIRE_FALSE(entry->column_storage.front().empty());
    return entry->column_storage.front().front().native == kInt64;
  }
};

}  // namespace

TEST_CASE("split pin: a request only the wide entry covers lands on the wide entry",
          "[pinned_lookup][scan_manager]")
{
  split_pin_fixture fixture;

  auto const wide_only = request({2, 3});
  auto const* chosen =
    fixture.manager.find_pinned_entry_for_duckdb_table(kCatalog, kSchema, kTable, &wide_only);
  REQUIRE(chosen != nullptr);
  REQUIRE(is_wide_entry(chosen));
  REQUIRE_FALSE(chosen->cache_info.column_projection_for(wide_only).empty());
}

TEST_CASE("split pin: a request only the split entry covers lands on the split entry",
          "[pinned_lookup][scan_manager]")
{
  split_pin_fixture fixture;

  auto const split_only = request({1, 8});
  auto const* chosen =
    fixture.manager.find_pinned_entry_for_duckdb_table(kCatalog, kSchema, kTable, &split_only);
  REQUIRE(chosen != nullptr);
  REQUIRE_FALSE(is_wide_entry(chosen));
  REQUIRE_FALSE(chosen->cache_info.column_projection_for(split_only).empty());
}

TEST_CASE("split pin: a request both entries cover returns a serving entry",
          "[pinned_lookup][scan_manager]")
{
  split_pin_fixture fixture;

  // {c0, c1} is a subset of both entries; either answer is valid if it serves.
  auto const both = request({0, 1});
  auto const* chosen =
    fixture.manager.find_pinned_entry_for_duckdb_table(kCatalog, kSchema, kTable, &both);
  REQUIRE(chosen != nullptr);
  REQUIRE_FALSE(chosen->cache_info.column_projection_for(both).empty());
}

TEST_CASE("split pin: no covering entry falls back to a non-null identity match",
          "[pinned_lookup][scan_manager]")
{
  split_pin_fixture fixture;

  // No entry covers {c1, c7, c8}; the guard declines the scan on the fallback.
  auto const uncovered = request({1, 7, 8});
  auto const* chosen =
    fixture.manager.find_pinned_entry_for_duckdb_table(kCatalog, kSchema, kTable, &uncovered);
  REQUIRE(chosen != nullptr);
  REQUIRE(chosen->cache_info.table_name == kTable);
  REQUIRE(chosen->cache_info.column_projection_for(uncovered).empty());

  REQUIRE(fixture.manager.find_pinned_entry_for_duckdb_table(
            kCatalog, kSchema, "lineitem", &uncovered) == nullptr);
}

TEST_CASE("split pin: null or empty requested ids keep the first-identity-match behavior",
          "[pinned_lookup][scan_manager]")
{
  split_pin_fixture fixture;

  auto const* chosen =
    fixture.manager.find_pinned_entry_for_duckdb_table(kCatalog, kSchema, kTable, nullptr);
  REQUIRE(chosen != nullptr);
  REQUIRE(chosen->cache_info.table_name == kTable);

  duckdb::vector<duckdb::ColumnIndex> const empty;
  chosen = fixture.manager.find_pinned_entry_for_duckdb_table(kCatalog, kSchema, kTable, &empty);
  REQUIRE(chosen != nullptr);
  REQUIRE(chosen->cache_info.table_name == kTable);

  // Defaulted three-argument form, as existing callers use it.
  chosen = fixture.manager.find_pinned_entry_for_duckdb_table(kCatalog, kSchema, kTable);
  REQUIRE(chosen != nullptr);
}

TEST_CASE("type match: a covering entry whose recorded types no longer match loses to one whose do",
          "[pinned_lookup][scan_manager]")
{
  // Both assignments run, so a coverage-only rule fails one of them for any iteration order.
  auto const matching_name = GENERATE(std::string{"orders"}, std::string{"main.orders"});
  type_match_fixture fixture{matching_name};

  auto const both    = request({0, 1});
  auto const* chosen = fixture.manager.find_pinned_entry_for_duckdb_table(
    kCatalog, kSchema, kTable, &both, &fixture.returned_types);
  REQUIRE(type_match_fixture::types_match(chosen));
}

TEST_CASE("type match: when no covering entry still matches, one is returned anyway",
          "[pinned_lookup][scan_manager]")
{
  auto memory = initialize_memory_manager(1);
  sirius_scan_manager manager{scan_manager_config{}, *memory, single_gpu_index()};
  pin_typed_entry(manager, *memory, "orders", {0, 1}, kInt32);
  duckdb::vector<duckdb::LogicalType> const returned_types{duckdb::LogicalType::BIGINT,
                                                           duckdb::LogicalType::BIGINT};

  // nullptr here would read as an unpinned table and route to the disk-native path.
  auto const both = request({0, 1});
  REQUIRE(manager.find_pinned_entry_for_duckdb_table(
            kCatalog, kSchema, kTable, &both, &returned_types) != nullptr);
}
