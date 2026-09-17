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

// Unit tests for build_batch_layout — the mapping from a decoded parquet batch's rows back to
// the file rows they came from. Iceberg positional deletes and deletion vectors are keyed on
// (data file, row position within that file), so this mapping is what decides WHICH rows a
// delete removes. Get it wrong and the scan deletes real rows and keeps deleted ones, with no
// error anywhere — which is why it is tested directly rather than only through a fixture.
//
// Pure metadata arithmetic over footer row counts: no GPU, no IO.

#include "op/scan/iceberg_gpu_ingestible.hpp"

#include <catch.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using namespace sirius::op::scan;

namespace {

/// A footer whose row groups hold the given row counts, in file order.
std::shared_ptr<cudf::io::parquet::FileMetaData const> footer_with(
  std::vector<int64_t> const& row_counts)
{
  auto meta = std::make_shared<cudf::io::parquet::FileMetaData>();
  for (auto const count : row_counts) {
    cudf::io::parquet::RowGroup rg;
    rg.num_rows = count;
    meta->row_groups.push_back(rg);
    meta->num_rows += count;
  }
  return meta;
}

row_group_slice slice_of(std::string path,
                         std::vector<int64_t> const& row_counts,
                         std::vector<cudf::size_type> selected)
{
  return row_group_slice{footer_with(row_counts),
                         std::move(path),
                         std::move(selected),
                         /*estimated_output_bytes=*/0,
                         /*estimated_decode_working_bytes=*/0,
                         /*reserved_compressed_bytes=*/0,
                         /*datasource=*/nullptr};
}

parquet_split_info split_of(std::vector<row_group_slice> slices)
{
  parquet_split_info split;
  split.rg_slices = std::move(slices);
  return split;
}

}  // namespace

TEST_CASE("build_batch_layout maps a single whole file", "[scan][iceberg]")
{
  auto const split  = split_of({slice_of("a.parquet", {3, 4}, {0, 1})});
  auto const layout = build_batch_layout(split);

  REQUIRE(layout.size() == 2);
  CHECK(layout[0].data_file_path == "a.parquet");
  CHECK(layout[0].file_row_offset == 0);
  CHECK(layout[0].batch_row_offset == 0);
  CHECK(layout[0].num_rows == 3);
  // The second row group starts at file row 3 and at batch row 3 — they agree only because
  // nothing before it was pruned.
  CHECK(layout[1].file_row_offset == 3);
  CHECK(layout[1].batch_row_offset == 3);
  CHECK(layout[1].num_rows == 4);
}

TEST_CASE("build_batch_layout keeps file offsets across a pruned row group", "[scan][iceberg]")
{
  // Row group 1 was pruned. Batch positions close up; FILE positions must not — a delete at
  // file row 9 belongs to row group 2, and treating batch row 9 as file row 9 would delete a
  // row from the wrong place.
  auto const split  = split_of({slice_of("a.parquet", {3, 5, 4}, {0, 2})});
  auto const layout = build_batch_layout(split);

  REQUIRE(layout.size() == 2);
  CHECK(layout[0].file_row_offset == 0);
  CHECK(layout[0].batch_row_offset == 0);
  CHECK(layout[0].num_rows == 3);
  CHECK(layout[1].file_row_offset == 8);   // 3 + 5, counting the pruned group
  CHECK(layout[1].batch_row_offset == 3);  // but only 3 rows precede it in the batch
  CHECK(layout[1].num_rows == 4);
}

TEST_CASE("build_batch_layout restarts file offsets per file", "[scan][iceberg]")
{
  // The parquet coalescer bundles small files into one batch. Each file's positions restart at
  // zero while batch positions keep running — the case a single (path, first_row) pair could
  // not express at all.
  auto const split =
    split_of({slice_of("a.parquet", {2, 2}, {0, 1}), slice_of("b.parquet", {5}, {0})});
  auto const layout = build_batch_layout(split);

  REQUIRE(layout.size() == 3);
  CHECK(layout[1].data_file_path == "a.parquet");
  CHECK(layout[1].file_row_offset == 2);
  CHECK(layout[1].batch_row_offset == 2);
  CHECK(layout[2].data_file_path == "b.parquet");
  CHECK(layout[2].file_row_offset == 0);
  CHECK(layout[2].batch_row_offset == 4);
  CHECK(layout[2].num_rows == 5);
}

TEST_CASE("build_batch_layout skips fully pruned files", "[scan][iceberg]")
{
  // A slice with no selected row groups is the coalescer's all-pruned fallback: it contributes
  // no rows, so it must contribute no runs either (an empty run would offset everything after).
  auto const split  = split_of({slice_of("a.parquet", {4}, {}), slice_of("b.parquet", {6}, {0})});
  auto const layout = build_batch_layout(split);

  REQUIRE(layout.size() == 1);
  CHECK(layout[0].data_file_path == "b.parquet");
  CHECK(layout[0].batch_row_offset == 0);
  CHECK(layout[0].num_rows == 6);
}

TEST_CASE("build_batch_layout of an entirely pruned split is empty", "[scan][iceberg]")
{
  auto const split = split_of({slice_of("a.parquet", {4}, {})});
  CHECK(build_batch_layout(split).empty());
}

TEST_CASE("build_batch_layout rejects a row group index outside the footer", "[scan][iceberg]")
{
  // Better to fail than to read past the row-group list and compute a nonsense file offset.
  auto const split = split_of({slice_of("a.parquet", {4}, {3})});
  CHECK_THROWS(build_batch_layout(split));
}
