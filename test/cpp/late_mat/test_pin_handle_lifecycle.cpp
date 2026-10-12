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

// [late_mat][pin_lifecycle] — an origin follows the exact entry its query owns.
// GPU required.
//
// The benchmark harness pins the tables each query needs and unpins them after,
// so a long run is a stream of pin/unpin/re-pin against the same names. Every
// registry transition must hide the old entry from new queries without revoking
// a query already serving it. An origin therefore resolves the old exact object
// while that query owns it, never redirects through the reused name, and expires
// when the final owner releases it.
//
// These go through the scan manager's own insert and remove, because the bug
// they cover is one of sequencing inside those calls, not of the handle.

#include "operator/operator_test_utils.hpp"

#include <cudf/column/column_factories.hpp>

#include <catch.hpp>
#include <cucascade/memory/topology_discovery.hpp>
#include <late_mat/column_origin.hpp>
#include <memory/topology_index.hpp>
#include <scan_manager/sirius_scan_manager.hpp>
#include <utils/duckdb_table_identity.hpp>

#include <memory>
#include <string>
#include <vector>

using sirius::late_mat::column_origin;
using sirius::scan_manager::pinned_entry;
using sirius::scan_manager::scan_manager_config;
using sirius::scan_manager::sirius_scan_manager;

namespace {

constexpr char const* kTable = "lineitem";

/// One device chunk of `rows` INT32 values, all equal to `fill` — so a resolved
/// entry can be told apart from the one that replaced it.
sirius::device_pin_chunk make_chunk(cucascade::memory::memory_space& space,
                                    cudf::size_type rows,
                                    ::cuda::stream_ref stream)
{
  sirius::device_pin_chunk chunk;
  chunk.memory_space = &space;
  chunk.columns.push_back(
    std::shared_ptr<cudf::column>(cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                                            rows,
                                                            cudf::mask_state::UNALLOCATED,
                                                            stream,
                                                            space.get_default_allocator())));
  return chunk;
}

sirius::scan_manager::cache_entry_info make_cache_info()
{
  sirius::scan_manager::cache_entry_info info;
  info.table_name     = kTable;
  info.table_identity = sirius::test::test_table_identity(42);
  info.names          = {"l_quantity"};
  info.column_ids.emplace_back(0);  // aligned with names, as the insert requires
  return info;
}

void pin_once(sirius_scan_manager& manager,
              cucascade::memory::memory_space& space,
              cudf::size_type rows,
              ::cuda::stream_ref stream)
{
  std::vector<sirius::device_pin_chunk> chunks;
  chunks.push_back(make_chunk(space, rows, stream));
  sirius::pinned_column_storage_matrix storage{
    {sirius::pinned_column_storage_meta{cudf::data_type{cudf::type_id::INT32}, false}}};
  manager.insert_pinned_entry_device(
    kTable, make_cache_info(), std::move(chunks), space, std::move(storage));
}

/// An origin against whatever is pinned under kTable right now.
column_origin capture_origin(sirius_scan_manager const& manager)
{
  column_origin origin;
  manager.visit_pinned_entries([&](std::string_view name, pinned_entry const& entry) {
    if (name == kTable && entry.late_mat_handle) {
      origin.handle     = entry.late_mat_handle;
      origin.column_pos = 0;
      origin.generation = entry.late_mat_handle->generation();
    }
    return true;
  });
  return origin;
}

std::shared_ptr<pinned_entry const> capture_entry_owner(sirius_scan_manager const& manager)
{
  return manager.find_pinned_entry_for_duckdb_table(
    "", "", kTable, sirius::test::test_table_identity(42));
}

std::shared_ptr<const sirius::memory::topology_index> single_gpu_index()
{
  cucascade::memory::system_topology_info topology;
  topology.num_gpus = 1;
  cucascade::memory::gpu_topology_info gpu;
  topology.gpus.push_back(std::move(gpu));
  return std::make_shared<sirius::memory::topology_index>(topology, std::vector<int>{0});
}

struct manager_fixture {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> memory =
    sirius::test::operator_utils::initialize_memory_manager();
  std::shared_ptr<const sirius::memory::topology_index> topology = single_gpu_index();
};

}  // namespace

TEST_CASE("a replacing pin does not redirect an active query's origin", "[late_mat][pin_lifecycle]")
{
  // The gate decides whether handles are published at all.
  if (!sirius::late_mat::late_mat_enabled()) { return; }

  ::cuda::stream_ref const stream{cudaStream_t{}};
  manager_fixture fixture;
  auto* space = fixture.memory->get_memory_space(cucascade::memory::Tier::GPU, 0);
  sirius_scan_manager manager{scan_manager_config{}, *fixture.memory, fixture.topology};

  pin_once(manager, *space, 128, stream);
  auto const first = capture_origin(manager);
  auto first_owner = capture_entry_owner(manager);
  REQUIRE(first.has_origin());
  REQUIRE(first.resolve() == first_owner);

  // Re-pin the same name without unpinning. New lookups see the replacement,
  // while an already-running query continues against its old exact object.
  pin_once(manager, *space, 256, stream);
  REQUIRE(first.resolve() == first_owner);

  auto const second = capture_origin(manager);
  auto second_owner = capture_entry_owner(manager);
  REQUIRE(second.resolve() == second_owner);
  REQUIRE(second_owner != first_owner);
  REQUIRE(second.generation != first.generation);

  first_owner.reset();
  REQUIRE(first.resolve() == nullptr);
}

TEST_CASE("unpin keeps an active query's origin valid until its owner releases",
          "[late_mat][pin_lifecycle]")
{
  if (!sirius::late_mat::late_mat_enabled()) { return; }

  ::cuda::stream_ref const stream{cudaStream_t{}};
  manager_fixture fixture;
  auto* space = fixture.memory->get_memory_space(cucascade::memory::Tier::GPU, 0);
  sirius_scan_manager manager{scan_manager_config{}, *fixture.memory, fixture.topology};

  pin_once(manager, *space, 128, stream);
  auto const origin = capture_origin(manager);
  auto owner        = capture_entry_owner(manager);
  REQUIRE(origin.resolve() == owner);

  manager.remove_pinned_entry(kTable);
  REQUIRE(capture_entry_owner(manager) == nullptr);
  REQUIRE(origin.resolve() == owner);

  owner.reset();
  REQUIRE(origin.resolve() == nullptr);
}

TEST_CASE("pin, unpin, re-pin keeps active origins on distinct entries",
          "[late_mat][pin_lifecycle]")
{
  if (!sirius::late_mat::late_mat_enabled()) { return; }

  // The harness's actual rhythm: each query pins what it needs and unpins
  // after, so the same names cycle for the length of a run.
  ::cuda::stream_ref const stream{cudaStream_t{}};
  manager_fixture fixture;
  auto* space = fixture.memory->get_memory_space(cucascade::memory::Tier::GPU, 0);
  sirius_scan_manager manager{scan_manager_config{}, *fixture.memory, fixture.topology};

  pin_once(manager, *space, 128, stream);
  auto const from_first_query = capture_origin(manager);
  auto first_owner            = capture_entry_owner(manager);
  manager.remove_pinned_entry(kTable);

  pin_once(manager, *space, 128, stream);
  auto const from_second_query = capture_origin(manager);
  auto second_owner            = capture_entry_owner(manager);

  // Same name, same row count and same shape, but each origin is permanently
  // tied to the object its query acquired.
  REQUIRE(from_first_query.resolve() == first_owner);
  REQUIRE(from_second_query.resolve() == second_owner);
  REQUIRE(first_owner != second_owner);

  first_owner.reset();
  REQUIRE(from_first_query.resolve() == nullptr);
  REQUIRE(from_second_query.resolve() == second_owner);
}
