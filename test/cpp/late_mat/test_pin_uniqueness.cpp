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

// [late_mat][pin_uniqueness] — what the pin-time distinctness proof may claim.
// GPU required.
//
// The asymmetry is the whole point of these cases: a MISSED proof costs a
// group-by-rowid ride that could have happened, while a FALSE proof lets the
// aggregate group by a key that repeats, silently collapsing distinct groups
// into one. So most of what follows checks refusals, and the one case that
// checks a positive is the one our data actually hits — chunks that arrive out
// of value order, because `part.10` globs before `part.2`.

#include "operator/operator_test_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <cucascade/memory/topology_discovery.hpp>
#include <late_mat/pin_uniqueness.hpp>
#include <memory/topology_index.hpp>
#include <scan_manager/sirius_scan_manager.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using sirius::late_mat::pin_unique_probe_selection;
using sirius::late_mat::unique_probe;
using sirius::scan_manager::pinned_entry;
using sirius::scan_manager::scan_manager_config;
using sirius::scan_manager::sirius_scan_manager;

namespace {

/// One INT64 device column holding exactly @p values.
std::unique_ptr<cudf::column> int64_column(std::vector<std::int64_t> const& values,
                                           rmm::cuda_stream_view stream)
{
  auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                       static_cast<cudf::size_type>(values.size()),
                                       cudf::mask_state::UNALLOCATED,
                                       stream);
  cudaMemcpyAsync(col->mutable_view().data<std::int64_t>(),
                  values.data(),
                  values.size() * sizeof(std::int64_t),
                  cudaMemcpyHostToDevice,
                  stream.value());
  cudaStreamSynchronize(stream.value());
  return col;
}

/// Feed a one-column probe the given chunks and report whether it proved the column.
bool proves_single_column(std::vector<std::vector<std::int64_t>> const& chunks,
                          rmm::cuda_stream_view stream)
{
  unique_probe probe{std::vector<bool>{true}};
  std::vector<std::unique_ptr<cudf::column>> alive;
  for (auto const& values : chunks) {
    alive.push_back(int64_column(values, stream));
    std::vector<cudf::column_view> cols{alive.back()->view()};
    probe.observe(cudf::table_view{cols}, stream);
  }
  auto const proven = probe.proven();
  REQUIRE(proven.size() == 1);
  return proven[0];
}

/// RAII around one environment variable, so a case can set the gate and leave
/// the process as it found it (Catch runs every case in one process).
struct scoped_env {
  std::string name;
  bool had_previous;
  std::string previous;

  scoped_env(std::string var, char const* value) : name(std::move(var))
  {
    char const* old = std::getenv(name.c_str());
    had_previous    = old != nullptr;
    if (had_previous) { previous = old; }
    if (value == nullptr) {
      unsetenv(name.c_str());
    } else {
      setenv(name.c_str(), value, 1);
    }
  }
  ~scoped_env()
  {
    if (had_previous) {
      setenv(name.c_str(), previous.c_str(), 1);
    } else {
      unsetenv(name.c_str());
    }
  }
};

std::vector<std::string> const kNames{"c_custkey", "c_name", "c_nationkey"};

}  // namespace

TEST_CASE("the probe selection reads the gate", "[late_mat][pin_uniqueness]")
{
  {
    scoped_env off{"SIRIUS_EXP_LATE_MAT_PIN_UNIQUE_COLS", nullptr};
    auto const selected = pin_unique_probe_selection(kNames);
    REQUIRE(selected == std::vector<bool>{false, false, false});
  }
  {
    scoped_env none{"SIRIUS_EXP_LATE_MAT_PIN_UNIQUE_COLS", "none"};
    REQUIRE(pin_unique_probe_selection(kNames) == std::vector<bool>{false, false, false});
  }
  {
    scoped_env all{"SIRIUS_EXP_LATE_MAT_PIN_UNIQUE_COLS", "all"};
    REQUIRE(pin_unique_probe_selection(kNames) == std::vector<bool>{true, true, true});
  }
  {
    // A name list is the usual setting; spelling and spacing are forgiving, and
    // a name that matches nothing in THIS table is not an error, since one
    // setting covers a suite of pins over different tables.
    scoped_env named{"SIRIUS_EXP_LATE_MAT_PIN_UNIQUE_COLS", " C_CustKey , o_orderkey,c_nationkey"};
    REQUIRE(pin_unique_probe_selection(kNames) == std::vector<bool>{true, false, true});
  }
}

TEST_CASE("a key split across chunks in the wrong order is still proven",
          "[late_mat][pin_uniqueness]")
{
  rmm::cuda_stream_view const stream{};
  // Emission order is [100..102] then [0..2]: strictly-increasing-boundaries
  // would refuse this, and it is exactly what a lexicographic file glob gives us.
  REQUIRE(proves_single_column({{100, 101, 102}, {0, 1, 2}}, stream));

  // Order WITHIN a chunk is a different matter, and the cheap stage does care:
  // it counts distinct values off sortedness rather than off a hash set,
  // because a hash set over a gigabyte-sized pinned chunk overruns cuco's
  // representable extent and fails the pin. An unsorted chunk is therefore
  // UNDECIDED — the exact stage still settles it.
  unique_probe probe{std::vector<bool>{true}};
  auto const unsorted_a = int64_column({7, 5, 6}, stream);
  auto const unsorted_b = int64_column({1, 3, 2}, stream);
  for (auto const* column : {&unsorted_a, &unsorted_b}) {
    std::vector<cudf::column_view> cols{(*column)->view()};
    probe.observe(cudf::table_view{cols}, stream);
  }
  REQUIRE(probe.verdicts() == std::vector<sirius::late_mat::unique_verdict>{
                                sirius::late_mat::unique_verdict::undecided});
  std::vector<cudf::column_view> both{unsorted_a->view(), unsorted_b->view()};
  REQUIRE(sirius::late_mat::exact_distinct_over_chunks(both, stream) == std::optional<bool>{true});
}

TEST_CASE("a duplicate inside one chunk is refused", "[late_mat][pin_uniqueness]")
{
  rmm::cuda_stream_view const stream{};
  REQUIRE_FALSE(proves_single_column({{1, 2, 2}}, stream));
}

TEST_CASE("chunks that could share a value are refused", "[late_mat][pin_uniqueness]")
{
  rmm::cuda_stream_view const stream{};
  // Each chunk is internally distinct, so only the range check can catch this.
  REQUIRE_FALSE(proves_single_column({{1, 2, 3}, {3, 4, 5}}, stream));
  // Overlapping without a shared endpoint: [0,10] and [4,6] — the values happen
  // not to collide, but a range test cannot know that, and "unknown" must read
  // as not proven.
  REQUIRE_FALSE(proves_single_column({{0, 10}, {4, 6}}, stream));
}

TEST_CASE("a single-row chunk repeated is refused", "[late_mat][pin_uniqueness]")
{
  rmm::cuda_stream_view const stream{};
  // Degenerate ranges [5,5] and [5,5]: both chunks are trivially distinct on
  // their own, and the table is not.
  REQUIRE_FALSE(proves_single_column({{5}, {5}}, stream));
}

TEST_CASE("an empty chunk carries no evidence either way", "[late_mat][pin_uniqueness]")
{
  rmm::cuda_stream_view const stream{};
  REQUIRE(proves_single_column({{3, 4}, {}, {1, 2}}, stream));
}

TEST_CASE("nulls are refused; a type the cheap stage cannot judge is only undecided",
          "[late_mat][pin_uniqueness]")
{
  rmm::cuda_stream_view const stream{};
  using sirius::late_mat::unique_verdict;

  {
    // A nullable column is out of scope for both stages: two nulls are not a
    // value collision but are not distinct values either, and the consumer of
    // the fact treats it as a key.
    unique_probe probe{std::vector<bool>{true}};
    auto col = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT64}, 4, cudf::mask_state::ALL_NULL, stream);
    std::vector<cudf::column_view> cols{col->view()};
    probe.observe(cudf::table_view{cols}, stream);
    REQUIRE(probe.verdicts() == std::vector<unique_verdict>{unique_verdict::refused});
  }
  {
    // The cheap stage is integer-only, but a non-integer column is UNDECIDED,
    // not refused — the exact check sorts whatever cuDF can sort, and a
    // dimension's name column is exactly what a rider's proof rests on.
    unique_probe probe{std::vector<bool>{true}};
    auto col = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::FLOAT64}, 4, cudf::mask_state::UNALLOCATED, stream);
    std::vector<cudf::column_view> cols{col->view()};
    probe.observe(cudf::table_view{cols}, stream);
    REQUIRE(probe.verdicts() == std::vector<unique_verdict>{unique_verdict::undecided});
    REQUIRE(probe.proven() == std::vector<bool>{false});
  }
}

TEST_CASE("an unobserved column is never claimed", "[late_mat][pin_uniqueness]")
{
  rmm::cuda_stream_view const stream{};
  unique_probe probe{std::vector<bool>{true, false}};
  auto unique_col = int64_column({1, 2, 3}, stream);
  auto other_col  = int64_column({7, 8, 9}, stream);  // also unique, but not selected
  std::vector<cudf::column_view> cols{unique_col->view(), other_col->view()};
  probe.observe(cudf::table_view{cols}, stream);

  REQUIRE(probe.proven() == std::vector<bool>{true, false});
  REQUIRE(probe.proven_names(std::vector<std::string>{"a", "b"}) == std::vector<std::string>{"a"});
}

TEST_CASE("undecided is told apart from refused", "[late_mat][pin_uniqueness]")
{
  // The exact check runs on the undecided ones only, so conflating the two
  // either wastes a sort of a column known to repeat or skips the column the
  // check exists for.
  rmm::cuda_stream_view const stream{};
  using sirius::late_mat::unique_verdict;

  unique_probe probe{std::vector<bool>{true, true, false}};
  auto overlapping_a = int64_column({1, 5}, stream);  // ranges [1,5] and [3,9] overlap:
  auto repeating_a   = int64_column({1, 1}, stream);  // repeats inside the chunk
  auto unobserved    = int64_column({1, 2}, stream);
  std::vector<cudf::column_view> first{
    overlapping_a->view(), repeating_a->view(), unobserved->view()};
  probe.observe(cudf::table_view{first}, stream);

  auto overlapping_b = int64_column({3, 9}, stream);
  auto repeating_b   = int64_column({7, 8}, stream);
  auto unobserved_b  = int64_column({3, 4}, stream);
  std::vector<cudf::column_view> second{
    overlapping_b->view(), repeating_b->view(), unobserved_b->view()};
  probe.observe(cudf::table_view{second}, stream);

  auto const verdicts = probe.verdicts();
  REQUIRE(verdicts[0] == unique_verdict::undecided);
  REQUIRE(verdicts[1] == unique_verdict::refused);
  REQUIRE(verdicts[2] == unique_verdict::not_observed);
}

TEST_CASE("the exact check decides what the range test could not", "[late_mat][pin_uniqueness]")
{
  rmm::cuda_stream_view const stream{};
  using sirius::late_mat::exact_distinct_over_chunks;

  // Overlapping ranges, no shared value — the case a real multi-file pin
  // produces, and the reason the exact check exists.
  auto a = int64_column({1, 5, 9}, stream);
  auto b = int64_column({2, 6, 8}, stream);
  std::vector<cudf::column_view> disjoint_values{a->view(), b->view()};
  REQUIRE(exact_distinct_over_chunks(disjoint_values, stream) == std::optional<bool>{true});

  auto c = int64_column({1, 5, 9}, stream);
  auto d = int64_column({2, 5, 8}, stream);  // 5 appears in both chunks
  std::vector<cudf::column_view> shared_value{c->view(), d->view()};
  REQUIRE(exact_distinct_over_chunks(shared_value, stream) == std::optional<bool>{false});

  // Undecidable stays undecidable rather than becoming "not distinct": a
  // nullable column is a question the check does not answer.
  auto nullable = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, 4, cudf::mask_state::ALL_NULL, stream);
  std::vector<cudf::column_view> with_nulls{nullable->view()};
  REQUIRE_FALSE(exact_distinct_over_chunks(with_nulls, stream).has_value());
}

TEST_CASE("a chunk of the wrong width abandons the proof", "[late_mat][pin_uniqueness]")
{
  rmm::cuda_stream_view const stream{};
  unique_probe probe{std::vector<bool>{true, true}};
  auto col = int64_column({1, 2, 3}, stream);
  std::vector<cudf::column_view> narrow{col->view()};  // one column, selection covers two
  probe.observe(cudf::table_view{narrow}, stream);

  REQUIRE_FALSE(probe.active());
  REQUIRE(probe.proven() == std::vector<bool>{false, false});
}

namespace {

std::shared_ptr<const sirius::memory::topology_index> single_gpu_index()
{
  cucascade::memory::system_topology_info topology;
  topology.num_gpus = 1;
  topology.gpus.emplace_back();
  return std::make_shared<sirius::memory::topology_index>(topology, std::vector<int>{0});
}

sirius::device_pin_chunk make_chunk(cucascade::memory::memory_space& space,
                                    std::size_t num_columns,
                                    rmm::cuda_stream_view stream)
{
  sirius::device_pin_chunk chunk;
  chunk.memory_space = &space;
  for (std::size_t i = 0; i < num_columns; ++i) {
    chunk.columns.push_back(
      std::shared_ptr<cudf::column>(cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                                              8,
                                                              cudf::mask_state::UNALLOCATED,
                                                              stream,
                                                              space.get_default_allocator())));
  }
  return chunk;
}

/// Pin `names` under `table`, one INT32 column each.
void pin_columns(sirius_scan_manager& manager,
                 cucascade::memory::memory_space& space,
                 std::string const& table,
                 std::vector<std::string> const& names,
                 rmm::cuda_stream_view stream)
{
  sirius::scan_manager::cache_entry_info info;
  info.table_name = table;
  info.names      = names;
  for (std::size_t i = 0; i < names.size(); ++i) {
    info.column_ids.emplace_back(static_cast<duckdb::idx_t>(i));
  }
  std::vector<sirius::device_pin_chunk> chunks;
  chunks.push_back(make_chunk(space, names.size(), stream));
  sirius::pinned_column_storage_matrix storage{std::vector<sirius::pinned_column_storage_meta>(
    names.size(),
    sirius::pinned_column_storage_meta{cudf::data_type{cudf::type_id::INT32}, false})};
  manager.insert_pinned_entry_device(
    table, std::move(info), std::move(chunks), space, std::move(storage));
}

/// Pin `names` under `table` through the merge-capable GPU path (one INT32
/// column each, one chunk of 8 rows), and report which columns the insert
/// actually stored.
std::vector<std::string> pin_columns_mergeable(sirius_scan_manager& manager,
                                               cucascade::memory::memory_space& space,
                                               std::string const& table,
                                               std::vector<std::string> const& names,
                                               rmm::cuda_stream_view stream)
{
  sirius::scan_manager::cache_entry_info info;
  info.table_name = table;
  info.names      = names;
  for (std::size_t i = 0; i < names.size(); ++i) {
    info.column_ids.emplace_back(static_cast<duckdb::idx_t>(i));
  }

  std::vector<std::unique_ptr<cudf::column>> columns;
  for (std::size_t i = 0; i < names.size(); ++i) {
    columns.push_back(cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                                8,
                                                cudf::mask_state::UNALLOCATED,
                                                stream,
                                                space.get_default_allocator()));
  }
  std::vector<std::unique_ptr<cudf::table>> tables;
  tables.push_back(std::make_unique<cudf::table>(std::move(columns)));

  sirius::pinned_column_storage_matrix storage{std::vector<sirius::pinned_column_storage_meta>(
    names.size(),
    sirius::pinned_column_storage_meta{cudf::data_type{cudf::type_id::INT32}, false})};

  return manager.insert_pinned_entry(table,
                                     std::move(info),
                                     std::move(tables),
                                     {&space},
                                     duckdb::vector<duckdb::LogicalType>{},
                                     {},
                                     std::move(storage));
}

std::vector<bool> proven_of(sirius_scan_manager const& manager, std::string_view table)
{
  std::vector<bool> out;
  manager.visit_pinned_entries([&](std::string_view name, pinned_entry const& entry) {
    if (name == table) { out = entry.proven_unique_columns; }
    return true;
  });
  return out;
}

}  // namespace

TEST_CASE("the pinned entry records the proof by name", "[late_mat][pin_uniqueness]")
{
  rmm::cuda_stream_view const stream{};
  auto memory   = sirius::test::operator_utils::initialize_memory_manager();
  auto topology = single_gpu_index();
  auto* space   = memory->get_memory_space(cucascade::memory::Tier::GPU, 0);
  sirius_scan_manager manager{scan_manager_config{}, *memory, topology};

  pin_columns(manager, *space, "customer", {"c_name", "c_custkey"}, stream);
  REQUIRE(proven_of(manager, "customer").empty());  // no fact yet = unknown

  std::vector<std::string> const proven{"c_custkey"};
  manager.attach_proven_unique_columns("customer", proven);
  // By NAME: the proof lands on position 1, not on the first column.
  REQUIRE(proven_of(manager, "customer") == std::vector<bool>{false, true});

  // A name that matches no pinned column is ignored rather than mispositioned.
  std::vector<std::string> const stranger{"o_orderkey"};
  manager.attach_proven_unique_columns("customer", stranger);
  REQUIRE(proven_of(manager, "customer") == std::vector<bool>{false, true});
}

TEST_CASE("a replacing re-pin starts with no facts", "[late_mat][pin_uniqueness]")
{
  rmm::cuda_stream_view const stream{};
  auto memory   = sirius::test::operator_utils::initialize_memory_manager();
  auto topology = single_gpu_index();
  auto* space   = memory->get_memory_space(cucascade::memory::Tier::GPU, 0);
  sirius_scan_manager manager{scan_manager_config{}, *memory, topology};

  pin_columns(manager, *space, "customer", {"c_custkey"}, stream);
  std::vector<std::string> const proven{"c_custkey"};
  manager.attach_proven_unique_columns("customer", proven);
  REQUIRE(proven_of(manager, "customer") == std::vector<bool>{true});

  // Re-pinning the name with a different column set replaces the entry. The
  // fact must not survive into data it was never taken against.
  pin_columns(manager, *space, "customer", {"c_nationkey", "c_custkey"}, stream);
  auto const after = proven_of(manager, "customer");
  REQUIRE(std::count(after.begin(), after.end(), true) == 0);
}

TEST_CASE("a merge reports only the columns it actually stored", "[late_mat][pin_uniqueness]")
{
  // The merge path keeps an already-cached column's chunks and DROPS the
  // incoming ones. A uniqueness verdict describes the values the pin driver just
  // read, so attaching it to a retained column would assert distinctness about
  // bytes that never entered the cache — and this flag admits a group key. The
  // insert therefore reports what it stored, and the caller filters by it.
  rmm::cuda_stream_view const stream{};
  auto memory   = sirius::test::operator_utils::initialize_memory_manager();
  auto topology = single_gpu_index();
  auto* space   = memory->get_memory_space(cucascade::memory::Tier::GPU, 0);
  sirius_scan_manager manager{scan_manager_config{}, *memory, topology};

  // First pin stores both of its columns.
  auto const first =
    pin_columns_mergeable(manager, *space, "customer", {"c_custkey", "c_name"}, stream);
  REQUIRE(first == std::vector<std::string>{"c_custkey", "c_name"});

  // Same row count, one column already cached: only c_nationkey is stored, and
  // c_custkey keeps the chunks it was pinned with.
  auto const merged =
    pin_columns_mergeable(manager, *space, "customer", {"c_custkey", "c_nationkey"}, stream);
  REQUIRE(merged == std::vector<std::string>{"c_nationkey"});

  // What the caller does with that: a verdict for the retained column is not
  // attached, so the entry claims nothing about data this pin never stored.
  std::vector<std::string> const attachable{"c_nationkey"};
  manager.attach_proven_unique_columns("customer", attachable);
  auto const proven = proven_of(manager, "customer");
  REQUIRE(proven.size() == 3u);
  REQUIRE(proven[0] == false);  // c_custkey — retained, and unclaimed
  REQUIRE(proven[1] == false);  // c_name
  REQUIRE(proven[2] == true);   // c_nationkey — newly stored
}
