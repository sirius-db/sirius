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
#include "sirius_config.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/error.hpp>

#include <cuda/stream>
#include <cuda_runtime.h>

#include <catch.hpp>
#include <cucascade/memory/topology_discovery.hpp>
#include <late_mat/pin_uniqueness.hpp>
#include <memory/topology_index.hpp>
#include <scan_manager/sirius_scan_manager.hpp>
#include <utils/duckdb_table_identity.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using sirius::late_mat::unique_probe;
using sirius::scan_manager::pinned_entry;
using sirius::scan_manager::scan_manager_config;
using sirius::scan_manager::sirius_scan_manager;

namespace {

/// One INT64 device column holding exactly @p values.
std::unique_ptr<cudf::column> int64_column(std::vector<std::int64_t> const& values,
                                           ::cuda::stream_ref stream)
{
  auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                       static_cast<cudf::size_type>(values.size()),
                                       cudf::mask_state::UNALLOCATED,
                                       stream);
  cudaMemcpyAsync(col->mutable_view().data<std::int64_t>(),
                  values.data(),
                  values.size() * sizeof(std::int64_t),
                  cudaMemcpyHostToDevice,
                  stream.get());
  cudaStreamSynchronize(stream.get());
  return col;
}

/// Feed a one-column probe the given chunks and report whether it proved the column.
bool proves_single_column(std::vector<std::vector<std::int64_t>> const& chunks,
                          ::cuda::stream_ref stream)
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

}  // namespace

TEST_CASE("a key split across chunks in the wrong order is still proven",
          "[late_mat][pin_uniqueness]")
{
  ::cuda::stream_ref const stream{cudaStream_t{}};
  // Emission order is [100..102] then [0..2]: strictly-increasing-boundaries
  // would refuse this, and it is exactly what a lexicographic file glob gives us.
  REQUIRE(proves_single_column({{100, 101, 102}, {0, 1, 2}}, stream));

  // Order WITHIN a chunk is a different matter, and the cheap stage does care:
  // it counts distinct values off sortedness rather than off a hash set,
  // because a hash set over a gigabyte-sized pinned chunk overruns cuco's
  // representable extent and fails the pin. An unsorted chunk is therefore
  // sorted on the spot, and chunks whose ranges do not overlap still prove the column.
  unique_probe probe{std::vector<bool>{true}};
  auto const unsorted_a = int64_column({7, 5, 6}, stream);
  auto const unsorted_b = int64_column({1, 3, 2}, stream);
  for (auto const* column : {&unsorted_a, &unsorted_b}) {
    std::vector<cudf::column_view> cols{(*column)->view()};
    probe.observe(cudf::table_view{cols}, stream);
  }
  REQUIRE(probe.verdicts() ==
          std::vector<sirius::late_mat::unique_verdict>{sirius::late_mat::unique_verdict::proven});
  std::vector<cudf::column_view> both{unsorted_a->view(), unsorted_b->view()};
  REQUIRE(sirius::late_mat::exact_distinct_over_chunks(both, stream) == std::optional<bool>{true});
}

TEST_CASE("a duplicate inside one chunk is refused", "[late_mat][pin_uniqueness]")
{
  ::cuda::stream_ref const stream{cudaStream_t{}};
  REQUIRE_FALSE(proves_single_column({{1, 2, 2}}, stream));
}

TEST_CASE("chunks that could share a value are refused", "[late_mat][pin_uniqueness]")
{
  ::cuda::stream_ref const stream{cudaStream_t{}};
  // Each chunk is internally distinct, so only the range check can catch this.
  REQUIRE_FALSE(proves_single_column({{1, 2, 3}, {3, 4, 5}}, stream));
  // Overlapping without a shared endpoint: [0,10] and [4,6] — the values happen
  // not to collide, but a range test cannot know that, and "unknown" must read
  // as not proven.
  REQUIRE_FALSE(proves_single_column({{0, 10}, {4, 6}}, stream));
}

TEST_CASE("a single-row chunk repeated is refused", "[late_mat][pin_uniqueness]")
{
  ::cuda::stream_ref const stream{cudaStream_t{}};
  // Degenerate ranges [5,5] and [5,5]: both chunks are trivially distinct on
  // their own, and the table is not.
  REQUIRE_FALSE(proves_single_column({{5}, {5}}, stream));
}

TEST_CASE("an empty chunk carries no evidence either way", "[late_mat][pin_uniqueness]")
{
  ::cuda::stream_ref const stream{cudaStream_t{}};
  REQUIRE(proves_single_column({{3, 4}, {}, {1, 2}}, stream));
}

TEST_CASE("nulls are refused", "[late_mat][pin_uniqueness]")
{
  ::cuda::stream_ref const stream{cudaStream_t{}};
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
}

TEST_CASE("an unobserved column is never claimed", "[late_mat][pin_uniqueness]")
{
  ::cuda::stream_ref const stream{cudaStream_t{}};
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
  ::cuda::stream_ref const stream{cudaStream_t{}};
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
  ::cuda::stream_ref const stream{cudaStream_t{}};
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

TEST_CASE("a column is settled by its first chunk where it can be", "[late_mat][pin_uniqueness]")
{
  // Observing every column of a fact table must not hand each of them to the exact stage's sort,
  // so an unsorted chunk is sorted on the spot to find a repeat, and a non-integer column goes no
  // further than a small table (below).
  ::cuda::stream_ref const stream{cudaStream_t{}};
  using sirius::late_mat::unique_verdict;

  // Columns: unsorted with a repeat, sorted, unsorted but distinct.
  unique_probe probe{std::vector<bool>{true, true, true}};
  auto unsorted_repeat   = int64_column({7, 5, 7}, stream);
  auto sorted            = int64_column({10, 11, 12}, stream);
  auto unsorted_distinct = int64_column({12, 10, 11}, stream);
  std::vector<cudf::column_view> cols{
    unsorted_repeat->view(), sorted->view(), unsorted_distinct->view()};
  probe.observe(cudf::table_view{cols}, stream);

  REQUIRE(probe.verdicts() == std::vector<unique_verdict>{unique_verdict::refused,
                                                          unique_verdict::proven,
                                                          unique_verdict::proven});
  REQUIRE_FALSE(probe.candidate(0));
  REQUIRE(probe.candidate(1));
  REQUIRE(probe.candidate(2));
}

TEST_CASE("a non-integer column stays in the running only for a small table",
          "[late_mat][pin_uniqueness]")
{
  // A dimension's name column (nation.n_name) can prove a group-by-rowid ride, so a non-integer
  // column is left to the exact stage while its table is small, whatever its type. Past the row cap
  // it is refused at the chunk that crosses it.
  ::cuda::stream_ref const stream{cudaStream_t{}};
  using sirius::late_mat::unique_verdict;
  auto const mr = cudf::get_current_device_resource_ref();

  unique_probe probe{std::vector<bool>{true, true}};
  auto names =
    sirius::test::operator_utils::make_string_column({"ALGERIA", "BRAZIL", "CHINA"}, stream, mr);
  auto floats = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::FLOAT64}, 3, cudf::mask_state::UNALLOCATED, stream);
  std::vector<cudf::column_view> cols{names->view(), floats->view()};
  probe.observe(cudf::table_view{cols}, stream);

  // The cheap stage cannot judge them: undecided, still candidates for the exact stage.
  REQUIRE(probe.verdicts() ==
          std::vector<unique_verdict>{unique_verdict::undecided, unique_verdict::undecided});
  REQUIRE(probe.candidate(0));
  REQUIRE(probe.candidate(1));

  // At exactly the cap the column is still in the running; the next chunk pushes it over.
  std::vector<std::string> big(sirius::late_mat::small_table_max_rows, "x");
  auto big_names = sirius::test::operator_utils::make_string_column(big, stream, mr);
  unique_probe big_probe{std::vector<bool>{true}};
  std::vector<cudf::column_view> big_cols{big_names->view()};
  big_probe.observe(cudf::table_view{big_cols}, stream);
  REQUIRE(big_probe.candidate(0));
  big_probe.observe(cudf::table_view{big_cols}, stream);
  REQUIRE(big_probe.verdicts() == std::vector<unique_verdict>{unique_verdict::refused});
}

TEST_CASE("a column is dropped, not the pin, when its sort cannot be reserved",
          "[late_mat][pin_uniqueness]")
{
  ::cuda::stream_ref const stream{cudaStream_t{}};
  using sirius::late_mat::unique_verdict;

  // An unsorted column that is distinct, and a sorted one (needs no sort).
  auto const make_probe  = [] { return unique_probe{std::vector<bool>{true, true}}; };
  auto unsorted_distinct = int64_column({12, 10, 11}, stream);
  auto sorted            = int64_column({10, 11, 12}, stream);
  std::vector<cudf::column_view> cols{unsorted_distinct->view(), sorted->view()};

  SECTION("a granted reservation sorts the chunk and proves the column")
  {
    auto probe = make_probe();
    std::vector<std::size_t> asked;
    probe.observe(cudf::table_view{cols}, stream, [&](std::size_t bytes) {
      asked.push_back(bytes);
      return true;
    });
    REQUIRE(asked == std::vector<std::size_t>{3 * sizeof(std::int64_t) *
                                              sirius::late_mat::sort_scratch_factor});
    REQUIRE(probe.verdicts() ==
            std::vector<unique_verdict>{unique_verdict::proven, unique_verdict::proven});
  }

  SECTION("a declined reservation drops only the column that needed the sort")
  {
    auto probe = make_probe();
    probe.observe(cudf::table_view{cols}, stream, [](std::size_t) { return false; });
    REQUIRE(probe.verdicts() ==
            std::vector<unique_verdict>{unique_verdict::refused, unique_verdict::proven});
    REQUIRE_FALSE(probe.candidate(0));
    REQUIRE(probe.candidate(1));
  }

  SECTION("a gate that throws costs the column its proof, never the observation")
  {
    auto probe = make_probe();
    REQUIRE_NOTHROW(probe.observe(cudf::table_view{cols}, stream, [](std::size_t) -> bool {
      throw std::runtime_error("out of memory");
    }));
    REQUIRE(probe.verdicts() ==
            std::vector<unique_verdict>{unique_verdict::refused, unique_verdict::proven});
  }
}

TEST_CASE("a column the exact stage cannot settle is not sorted again once chunks overlap",
          "[late_mat][pin_uniqueness]")
{
  // Unsorted, distinct chunks: each one costs a sort. Once the exact stage is out of reach only
  // chunks that never overlap in range can prove the column, so an overlapping chunk is dropped
  // before it is sorted.
  ::cuda::stream_ref const stream{cudaStream_t{}};
  using sirius::late_mat::unique_verdict;

  auto const observe_two =
    [&](bool exact_available, std::vector<std::int64_t> const& second, std::size_t& sorts) {
      unique_probe probe{std::vector<bool>{true}};
      auto const count_sorts = [&](std::size_t) {
        ++sorts;
        return true;
      };
      auto first_chunk  = int64_column({12, 10, 11}, stream);
      auto second_chunk = int64_column(second, stream);
      {
        std::vector<cudf::column_view> cols{first_chunk->view()};
        probe.observe(cudf::table_view{cols}, stream, count_sorts);
      }
      if (!exact_available) { probe.exact_unavailable(0); }
      {
        std::vector<cudf::column_view> cols{second_chunk->view()};
        probe.observe(cudf::table_view{cols}, stream, count_sorts);
      }
      return probe;
    };

  SECTION("with the exact stage available an overlapping chunk is sorted and left undecided")
  {
    std::size_t sorts = 0;
    auto probe        = observe_two(/*exact_available=*/true, {11, 13, 9}, sorts);
    REQUIRE(sorts == 2);
    REQUIRE(probe.verdicts() == std::vector<unique_verdict>{unique_verdict::undecided});
  }

  SECTION("without it an overlapping chunk is dropped before its sort")
  {
    std::size_t sorts = 0;
    auto probe        = observe_two(/*exact_available=*/false, {11, 13, 9}, sorts);
    REQUIRE(sorts == 1);
    REQUIRE_FALSE(probe.candidate(0));
    REQUIRE(probe.verdicts() == std::vector<unique_verdict>{unique_verdict::refused});
  }

  SECTION("without it a chunk that does not overlap is still sorted and the column is proven")
  {
    std::size_t sorts = 0;
    auto probe        = observe_two(/*exact_available=*/false, {22, 20, 21}, sorts);
    REQUIRE(sorts == 2);
    REQUIRE(probe.verdicts() == std::vector<unique_verdict>{unique_verdict::proven});
  }
}

TEST_CASE("an optional retained copy yields to the batches the pin still has to store",
          "[late_mat][pin_uniqueness]")
{
  using sirius::late_mat::retention_has_headroom;
  using sirius::late_mat::retention_headroom_batches;
  constexpr std::size_t mib = 1024 * 1024;
  // The reviewer's example: 100 MiB of capacity, four 19 MiB batches stored and 8 MiB of key copies
  // retained leave 16 MiB free. The fifth batch needs 19 MiB, so no further copy may be kept, and
  // the 8 MiB copy itself would already have left too little.
  REQUIRE_FALSE(
    retention_has_headroom(/*available=*/16 * mib, /*copy=*/8 * mib, /*batch=*/19 * mib));
  // With room for a copy and the headroom beside it, the copy is kept.
  REQUIRE(retention_has_headroom(
    /*available=*/8 * mib + retention_headroom_batches * 19 * mib, 8 * mib, 19 * mib));
  // One byte short of that is not enough.
  REQUIRE_FALSE(
    retention_has_headroom(8 * mib + retention_headroom_batches * 19 * mib - 1, 8 * mib, 19 * mib));
  // Nothing to protect: an empty batch needs only the copy itself.
  REQUIRE(retention_has_headroom(8 * mib, 8 * mib, 0));
}

TEST_CASE("required work is retried once after the optional copies are given back",
          "[late_mat][pin_uniqueness]")
{
  using sirius::late_mat::retry_after_releasing_optional;
  int runs     = 0;
  int releases = 0;

  SECTION("work that succeeds is run once and nothing is released")
  {
    auto const result = retry_after_releasing_optional(
      [&] {
        ++runs;
        return 7;
      },
      [&] {
        ++releases;
        return true;
      });
    REQUIRE(result == 7);
    REQUIRE(runs == 1);
    REQUIRE(releases == 0);
  }

  SECTION("an out-of-memory failure releases the copies and runs the work again")
  {
    auto const result = retry_after_releasing_optional(
      [&] {
        if (++runs == 1) { throw rmm::out_of_memory("the pin's decode"); }
        return 7;
      },
      [&] {
        ++releases;
        return true;
      });
    REQUIRE(result == 7);
    REQUIRE(runs == 2);
    REQUIRE(releases == 1);
  }

  SECTION("with nothing to release the original failure propagates without a retry")
  {
    REQUIRE_THROWS_AS(retry_after_releasing_optional(
                        [&]() -> int {
                          ++runs;
                          throw std::bad_alloc();
                        },
                        [&] {
                          ++releases;
                          return false;
                        }),
                      std::bad_alloc);
    REQUIRE(runs == 1);
    REQUIRE(releases == 1);
  }

  SECTION("a second failure propagates and the copies are released only once")
  {
    REQUIRE_THROWS_AS(retry_after_releasing_optional(
                        [&]() -> int {
                          ++runs;
                          throw rmm::out_of_memory("the pin's decode");
                        },
                        [&] {
                          ++releases;
                          return true;
                        }),
                      rmm::out_of_memory);
    REQUIRE(runs == 2);
    REQUIRE(releases == 1);
  }

  SECTION("a failure that is not about memory is not retried")
  {
    REQUIRE_THROWS_AS(retry_after_releasing_optional(
                        [&]() -> int {
                          ++runs;
                          throw std::runtime_error("corrupt file");
                        },
                        [&] {
                          ++releases;
                          return true;
                        }),
                      std::runtime_error);
    REQUIRE(runs == 1);
    REQUIRE(releases == 0);
  }
}

TEST_CASE("a column with sorted, overlapping chunks is left to the exact stage",
          "[late_mat][pin_uniqueness]")
{
  // SF1000 `orders`: every chunk is sorted by o_orderkey, but the chunks overlap in range.
  ::cuda::stream_ref const stream{cudaStream_t{}};
  using sirius::late_mat::unique_verdict;
  unique_probe probe{std::vector<bool>{true}};
  auto first  = int64_column({1, 5, 9}, stream);
  auto second = int64_column({2, 6, 8}, stream);
  for (auto const* column : {&first, &second}) {
    std::vector<cudf::column_view> cols{(*column)->view()};
    probe.observe(cudf::table_view{cols}, stream);
  }
  REQUIRE(probe.verdicts() == std::vector<unique_verdict>{unique_verdict::undecided});
  REQUIRE(probe.candidate(0));
}

TEST_CASE("the exact-stage byte cap is a share of GPU memory unless set",
          "[late_mat][pin_uniqueness]")
{
  using sirius::late_mat::exact_uniqueness_byte_cap;
  constexpr std::uint64_t gib = 1ULL << 30;
  // An explicit cap wins over the derived one.
  REQUIRE(exact_uniqueness_byte_cap(1000, 100 * gib) == 1000u);
  // 0 derives the cap from the GPU memory space's capacity.
  REQUIRE(
    exact_uniqueness_byte_cap(0, 100 * gib) ==
    static_cast<std::size_t>(100 * gib * sirius::config::PIN_TABLE_UNIQUE_EXACT_MEMORY_FRACTION));
}

TEST_CASE("a chunk of the wrong width abandons the proof", "[late_mat][pin_uniqueness]")
{
  ::cuda::stream_ref const stream{cudaStream_t{}};
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
                                    ::cuda::stream_ref stream)
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
                 ::cuda::stream_ref stream,
                 sirius::scan_manager::pinned_entry_metadata metadata = {})
{
  sirius::scan_manager::cache_entry_info info;
  info.table_name     = table;
  info.table_identity = sirius::test::test_table_identity(42);
  info.names          = names;
  for (std::size_t i = 0; i < names.size(); ++i) {
    info.column_ids.emplace_back(static_cast<duckdb::idx_t>(i));
  }
  std::vector<sirius::device_pin_chunk> chunks;
  chunks.push_back(make_chunk(space, names.size(), stream));
  sirius::pinned_column_storage_matrix storage{std::vector<sirius::pinned_column_storage_meta>(
    names.size(),
    sirius::pinned_column_storage_meta{cudf::data_type{cudf::type_id::INT32}, false})};
  manager.insert_pinned_entry_device(
    table, std::move(info), std::move(chunks), space, std::move(storage), std::move(metadata));
}

/// Pin `names` under `table` through the merge-capable GPU path (one INT32
/// column each, one chunk of 8 rows), and report which columns the insert
/// actually stored.
std::vector<std::string> pin_columns_mergeable(
  sirius_scan_manager& manager,
  cucascade::memory::memory_space& space,
  std::string const& table,
  std::vector<std::string> const& names,
  ::cuda::stream_ref stream,
  sirius::scan_manager::pinned_entry_metadata metadata = {})
{
  sirius::scan_manager::cache_entry_info info;
  info.table_name     = table;
  info.table_identity = sirius::test::test_table_identity(42);
  info.names          = names;
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
                                     std::move(storage),
                                     std::move(metadata));
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
  ::cuda::stream_ref const stream{cudaStream_t{}};
  auto memory   = sirius::test::operator_utils::initialize_memory_manager();
  auto topology = single_gpu_index();
  auto* space   = memory->get_memory_space(cucascade::memory::Tier::GPU, 0);
  sirius::exec::query_lifecycle_registry scan_lifecycle;
  sirius_scan_manager manager{scan_lifecycle, scan_manager_config{}, *memory, topology};

  pin_columns(manager,
              *space,
              "customer",
              {"c_name", "c_custkey"},
              stream,
              {.proven_unique_columns = {"c_custkey", "o_orderkey"}});
  // The first visible generation includes the proof by name, at position 1.
  // Unknown names are ignored rather than mispositioned.
  REQUIRE(proven_of(manager, "customer") == std::vector<bool>{false, true});
}

TEST_CASE("a replacing re-pin starts with no facts", "[late_mat][pin_uniqueness]")
{
  ::cuda::stream_ref const stream{cudaStream_t{}};
  auto memory   = sirius::test::operator_utils::initialize_memory_manager();
  auto topology = single_gpu_index();
  auto* space   = memory->get_memory_space(cucascade::memory::Tier::GPU, 0);
  sirius::exec::query_lifecycle_registry scan_lifecycle;
  sirius_scan_manager manager{scan_lifecycle, scan_manager_config{}, *memory, topology};

  pin_columns(
    manager, *space, "customer", {"c_custkey"}, stream, {.proven_unique_columns = {"c_custkey"}});
  REQUIRE(proven_of(manager, "customer") == std::vector<bool>{true});

  // Re-pinning the name with a different column set replaces the entry. The
  // fact must not survive into data it was never taken against.
  pin_columns(manager, *space, "customer", {"c_nationkey", "c_custkey"}, stream);
  auto const after = proven_of(manager, "customer");
  REQUIRE(std::count(after.begin(), after.end(), true) == 0);
}

TEST_CASE("a merge publishes proofs only for columns it actually stored",
          "[late_mat][pin_uniqueness]")
{
  // The merge path keeps an already-cached column's chunks and DROPS the
  // incoming ones. A uniqueness verdict describes the values the pin driver just
  // read, so attaching it to a retained column would assert distinctness about
  // bytes that never entered the cache — and this flag admits a group key. The
  // insertion must filter the supplied proofs before publishing the new generation.
  ::cuda::stream_ref const stream{cudaStream_t{}};
  auto memory   = sirius::test::operator_utils::initialize_memory_manager();
  auto topology = single_gpu_index();
  auto* space   = memory->get_memory_space(cucascade::memory::Tier::GPU, 0);
  sirius::exec::query_lifecycle_registry scan_lifecycle;
  sirius_scan_manager manager{scan_lifecycle, scan_manager_config{}, *memory, topology};

  // First pin stores both of its columns.
  auto const first = pin_columns_mergeable(manager,
                                           *space,
                                           "customer",
                                           {"c_custkey", "c_name"},
                                           stream,
                                           {.proven_unique_columns = {"c_name"}});
  REQUIRE(first == std::vector<std::string>{"c_custkey", "c_name"});

  // Same row count, one column already cached: only c_nationkey is stored, and
  // c_custkey keeps the chunks it was pinned with.
  auto const merged =
    pin_columns_mergeable(manager,
                          *space,
                          "customer",
                          {"c_custkey", "c_nationkey"},
                          stream,
                          {.proven_unique_columns = {"c_custkey", "c_nationkey"}});
  REQUIRE(merged == std::vector<std::string>{"c_nationkey"});

  // A proof for discarded incoming c_custkey bytes cannot change the retained column.
  auto const proven = proven_of(manager, "customer");
  REQUIRE(proven.size() == 3u);
  REQUIRE(proven[0] == false);  // c_custkey — retained, and unclaimed
  REQUIRE(proven[1] == true);   // c_name keeps its existing proof
  REQUIRE(proven[2] == true);   // c_nationkey — newly stored
}
