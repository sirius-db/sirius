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

// scan_operator_input's prefetch predicates over the *metadata* split shape, plus the ladder
// counting both shapes share. There is deliberately no mock ioctx or mock datasource in this repo:
// the datasources below are real kvikio-backed ones over a real parquet file, which is exactly the
// shipped local-disk shape — no prefetching cache is wired, so no fadvise ever stores a handle and
// every state reads empty. The resident-split half of the predicate surface needs a GPU batch and
// lives in test_cached_serving_hardening.cpp.

#include <catch.hpp>
#include <cucascade/data/data_batch.hpp>
#include <io/kvikio/kvikio_context.hpp>
#include <io/sirius_datasource.hpp>
#include <op/scan/parquet_gpu_ingestible.hpp>
#include <op/scan/row_group_metadata.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <scan_manager/prefetching_state_manager.hpp>

#include <array>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace scan = sirius::op::scan;

using sirius::io::cache::prefetch_progress;
using sirius::io::cache::prefetching_stage;
using sirius::scan_manager::prefetching_state_manager;

/// The four rungs of the ladder, in the order a split climbs them.
constexpr std::array kLadder{prefetching_stage::metadata_created,
                             prefetching_stage::task_queued,
                             prefetching_stage::task_preprocessing,
                             prefetching_stage::disposable};

std::filesystem::path project_root()
{
#ifdef SIRIUS_PROJECT_ROOT
  return std::filesystem::path{SIRIUS_PROJECT_ROOT};
#else
  return std::filesystem::current_path();
#endif
}

std::string nation_parquet()
{
  return (project_root() / "test/cpp/integration/data/parquet/nation.parquet").string();
}

/// A metadata split whose three row-group slices carry a datasource, no datasource, and a
/// datasource — the null-skipping contract and the N-handle fold in one fixture.
///
/// The datasources are real (kvikio over nation.parquet) and are never fadvised, so they hold no
/// prefetching handle. That is not a shortcut around a missing mock: it is the state every
/// local-disk query is in, because both shipped local backends opt out of the prefetch ladder.
struct three_slice_split {
  std::shared_ptr<sirius::io::kvikio_context> ioctx =
    std::make_shared<sirius::io::kvikio_context>();
  std::unique_ptr<sirius::io::sirius_datasource> origin = ioctx->open_datasource(nation_parquet());
  std::unique_ptr<scan::scan_operator_input> input;

  three_slice_split()
  {
    auto split = std::make_unique<scan::parquet_split_info>();
    add_slice(*split, duplicate_datasource());
    add_slice(*split, nullptr);
    add_slice(*split, duplicate_datasource());
    input = std::make_unique<scan::scan_operator_input>(std::move(split));
  }

  /// One datasource per slice, as production does: io_objects are shareable, handles are not.
  [[nodiscard]] std::shared_ptr<sirius::io::sirius_datasource> duplicate_datasource() const
  {
    return {origin->duplicate()};
  }

  static void add_slice(scan::parquet_split_info& split,
                        std::shared_ptr<sirius::io::sirius_datasource> datasource)
  {
    split.rg_slices.emplace_back(/*file_metadata=*/nullptr,
                                 nation_parquet(),
                                 std::vector<cudf::size_type>{0},
                                 /*estimated_output_bytes=*/20,
                                 /*estimated_decode_working_bytes=*/60,
                                 /*reserved_compressed_bytes=*/10,
                                 std::move(datasource));
  }
};

}  // namespace

TEST_CASE("a metadata split with no datasource is not io-prefetchable",
          "[scan][prefetch_api][scan_input]")
{
  auto file = std::make_unique<scan::parquet_file_scan_info>();
  file->row_groups.push_back({0, 20, 60, 10, 1});
  scan::scan_operator_input input{std::move(file)};

  CHECK(input.datasource_count() == 0);
  CHECK_FALSE(input.is_io_prefetchable());
  CHECK(input.prefetch_state() == prefetch_progress::empty);
}

TEST_CASE("a metadata split counts one datasource per row-group slice",
          "[scan][prefetch_api][scan_input]")
{
  three_slice_split fixture;

  CHECK(fixture.input->datasource_count() == 2);
  CHECK(fixture.input->is_io_prefetchable());

  std::size_t visited = 0;
  fixture.input->for_each_datasource([&](sirius::io::sirius_datasource&) { ++visited; });
  CHECK(visited == 2);
}

TEST_CASE("a metadata split with no fadvise reports an empty prefetch state",
          "[scan][prefetch_api][scan_input]")
{
  three_slice_split fixture;

  CHECK(fixture.input->prefetch_state() == prefetch_progress::empty);
  CHECK(fixture.input->is_prefetched() == std::optional<bool>{false});
}

TEST_CASE("prefetch on a split with no fadvised datasource is a no-op",
          "[scan][prefetch_api][scan_input]")
{
  three_slice_split fixture;

  for (auto const site : kLadder) {
    REQUIRE_NOTHROW(fixture.input->prefetch(site));
    CHECK(fixture.input->prefetch_state() == prefetch_progress::empty);
  }
}

TEST_CASE("a metadata split is not memory-prefetchable", "[scan][prefetch_api][scan_input]")
{
  three_slice_split fixture;

  // Nothing resident to promote, and the split is not where the task wants it either: its IO has
  // not been reported complete, so is_prefetched() is false rather than the negation of the line
  // above. The two predicates only mirror each other on the resident path.
  CHECK(fixture.input->is_memory_prefetchable() == std::optional<bool>{false});
  CHECK(fixture.input->is_prefetched() == std::optional<bool>{false});
}

//===----------------------------------------------------------------------===//
// ladder counting
//===----------------------------------------------------------------------===//

TEST_CASE("a counted split reports every ladder rung it climbs", "[scan][prefetch_api][scan_input]")
{
  auto counters = std::make_shared<prefetching_state_manager>(
    prefetching_state_manager::config{.memory_threshold = 0, .prefetch_lookahead_window = 4});

  {
    auto split = std::make_unique<scan::parquet_file_scan_info>();
    split->row_groups.push_back({0, 20, 60, 10, 1});
    scan::scan_operator_input input{std::move(split), counters};
    CHECK(counters->snapshot().n_live == 1);

    for (auto const site : kLadder) {
      input.prefetch(site);
    }

    auto const climbed = counters->snapshot();
    CHECK(climbed.n_metadata_created == 1);
    CHECK(climbed.n_task_queued == 1);
    CHECK(climbed.n_task_prepared == 1);
    CHECK(climbed.n_task_completed == 1);
    CHECK(climbed.n_inputs_created == 1);
  }

  // The disposal counter is the reason scan_operator_input has a user-declared destructor at all.
  auto const disposed = counters->snapshot();
  CHECK(disposed.n_inputs_disposed == 1);
  CHECK(disposed.n_live == 0);
}

TEST_CASE("a resident split is counted even though it has no datasource",
          "[scan][prefetch_api][scan_input]")
{
  // The regression gate for the under-count: the rung is recorded above prefetch()'s metadata
  // check, so a pinned-cache split — which has no scan metadata and no datasource, and therefore
  // no IO to hint — still records the ladder it climbed. Counting below the check reported
  // 0/0/0/0 for a fully-pinned query. A null batch pointer is enough: is_resident() is true and
  // has_scan_metadata() is false, with no GPU involved.
  auto counters = std::make_shared<prefetching_state_manager>(
    prefetching_state_manager::config{.memory_threshold = 0, .prefetch_lookahead_window = 4});

  scan::scan_operator_input input{std::shared_ptr<cucascade::data_batch>{}, counters};
  REQUIRE(input.is_resident());
  REQUIRE_FALSE(input.has_scan_metadata());

  for (auto const site : kLadder) {
    input.prefetch(site);
  }

  auto const climbed = counters->snapshot();
  CHECK(climbed.n_metadata_created == 1);
  CHECK(climbed.n_task_queued == 1);
  CHECK(climbed.n_task_prepared == 1);
  CHECK(climbed.n_task_completed == 1);
}

TEST_CASE("a split with no state manager does not crash", "[scan][prefetch_api][scan_input]")
{
  // The defaulted null manager is what keeps every call site outside the scan manager — including
  // every other case in this file — compiling and running unchanged.
  three_slice_split fixture;

  for (auto const site : kLadder) {
    REQUIRE_NOTHROW(fixture.input->prefetch(site));
  }
}
