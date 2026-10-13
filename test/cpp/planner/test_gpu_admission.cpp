/*
 * Copyright 2025, Sirius Contributors.
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

#include "catch.hpp"
#include "planner/gpu_admission.hpp"
#include "sirius_config.hpp"

#include <duckdb/storage/statistics/node_statistics.hpp>
#include <parquet_types.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

using sirius::logical_type;
using sirius::type_id;
using sirius::planner::accumulate_scan_bytes;
using sirius::planner::admission_row_count;
using sirius::planner::apply_gpu_cap;
using sirius::planner::estimate_bytes_per_row;
using sirius::planner::gpu_count_for_bytes;
using sirius::planner::scan_estimate;
using sirius::planner::total_scan_bytes;

namespace {
constexpr uint64_t k_avg_var = 32;
}

TEST_CASE("apply_gpu_cap with no cap admits every GPU", "[gpu_admission]")
{
  const std::vector<int> all{0, 1, 2, 3};
  REQUIRE(apply_gpu_cap(all, 0) == all);
}

TEST_CASE("apply_gpu_cap takes a prefix of the sorted list", "[gpu_admission]")
{
  const std::vector<int> all{0, 1, 2, 3};
  REQUIRE(apply_gpu_cap(all, 1) == std::vector<int>{0});
  REQUIRE(apply_gpu_cap(all, 2) == std::vector<int>{0, 1});
  REQUIRE(apply_gpu_cap(all, 3) == std::vector<int>{0, 1, 2});
}

TEST_CASE("apply_gpu_cap at the fleet size is a no-op", "[gpu_admission]")
{
  const std::vector<int> all{0, 1, 2, 3};
  REQUIRE(apply_gpu_cap(all, 4) == all);
}

TEST_CASE("apply_gpu_cap above the fleet size clamps to what exists", "[gpu_admission]")
{
  // Asking for more GPUs than the host has yields every GPU rather than failing.
  const std::vector<int> all{0, 1};
  REQUIRE(apply_gpu_cap(all, 8) == all);
}

TEST_CASE("apply_gpu_cap preserves non-contiguous device ids", "[gpu_admission]")
{
  // CUDA_VISIBLE_DEVICES / explicit gpu_ids can leave gaps; the cap is positional.
  const std::vector<int> sparse{2, 5, 7};
  REQUIRE(apply_gpu_cap(sparse, 2) == std::vector<int>{2, 5});
}

TEST_CASE("apply_gpu_cap treats a negative cap as no cap", "[gpu_admission]")
{
  // Config load rejects negatives; this is the defensive path.
  const std::vector<int> all{0, 1, 2, 3};
  REQUIRE(apply_gpu_cap(all, -1) == all);
}

TEST_CASE("apply_gpu_cap on an empty fleet stays empty", "[gpu_admission]")
{
  REQUIRE(apply_gpu_cap({}, 2).empty());
}

TEST_CASE("estimate_bytes_per_row sums fixed-width carrier widths", "[admission_estimator]")
{
  // 1 + 2 + 4 + 8 = 15
  const std::vector<logical_type> types{logical_type::make(type_id::TINYINT),
                                        logical_type::make(type_id::SMALLINT),
                                        logical_type::make(type_id::INTEGER),
                                        logical_type::make(type_id::BIGINT)};
  REQUIRE(estimate_bytes_per_row(types, k_avg_var) == 15);
}

TEST_CASE("estimate_bytes_per_row charges 128-bit integers their cuDF carrier width",
          "[admission_estimator]")
{
  // get_cudf_type has no 128-bit integer carrier and narrows both to a 64-bit one,
  // so they cost 8 bytes on device rather than their DuckDB width of 16.
  REQUIRE(estimate_bytes_per_row({logical_type::make(type_id::HUGEINT)}, k_avg_var) == 8);
  REQUIRE(estimate_bytes_per_row({logical_type::make(type_id::UHUGEINT)}, k_avg_var) == 8);
}

TEST_CASE("estimate_bytes_per_row treats unsigned types as their signed width",
          "[admission_estimator]")
{
  const std::vector<logical_type> signed_types{logical_type::make(type_id::TINYINT),
                                               logical_type::make(type_id::SMALLINT),
                                               logical_type::make(type_id::INTEGER),
                                               logical_type::make(type_id::BIGINT)};
  const std::vector<logical_type> unsigned_types{logical_type::make(type_id::UTINYINT),
                                                 logical_type::make(type_id::USMALLINT),
                                                 logical_type::make(type_id::UINTEGER),
                                                 logical_type::make(type_id::UBIGINT)};
  REQUIRE(estimate_bytes_per_row(signed_types, k_avg_var) ==
          estimate_bytes_per_row(unsigned_types, k_avg_var));
}

TEST_CASE("estimate_bytes_per_row sizes DECIMAL from its precision", "[admission_estimator]")
{
  // Precision selects the cuDF carrier: DECIMAL32/64/128. There is no DECIMAL16, so the
  // narrowest precisions still cost 4 bytes.
  REQUIRE(estimate_bytes_per_row({logical_type::make_decimal(4, 2)}, k_avg_var) == 4);
  REQUIRE(estimate_bytes_per_row({logical_type::make_decimal(9, 2)}, k_avg_var) == 4);
  REQUIRE(estimate_bytes_per_row({logical_type::make_decimal(18, 2)}, k_avg_var) == 8);
  REQUIRE(estimate_bytes_per_row({logical_type::make_decimal(38, 2)}, k_avg_var) == 16);
}

TEST_CASE("estimate_bytes_per_row charges variable-width columns the configured average",
          "[admission_estimator]")
{
  const std::vector<logical_type> types{logical_type::make(type_id::VARCHAR),
                                        logical_type::make(type_id::LIST)};
  REQUIRE(estimate_bytes_per_row(types, k_avg_var) == 2 * k_avg_var);
  // The fallback is configurable, so the same schema scales with it.
  REQUIRE(estimate_bytes_per_row(types, 8) == 16);
}

TEST_CASE("estimate_bytes_per_row mixes fixed and variable widths", "[admission_estimator]")
{
  // BIGINT(8) + VARCHAR(32) + DATE(4) = 44
  const std::vector<logical_type> types{logical_type::make(type_id::BIGINT),
                                        logical_type::make(type_id::VARCHAR),
                                        logical_type::make(type_id::DATE)};
  REQUIRE(estimate_bytes_per_row(types, k_avg_var) == 44);
}

TEST_CASE("estimate_bytes_per_row returns zero for an empty projection", "[admission_estimator]")
{
  REQUIRE(estimate_bytes_per_row({}, k_avg_var) == 0);
}

TEST_CASE("estimate_bytes_per_row takes a known width for a variable-width column",
          "[admission_estimator]")
{
  // VARCHAR with a source-reported width of 5 (a one-character string plus its offset word)
  // instead of the 32-byte fallback; BIGINT keeps its carrier width.
  const std::vector<logical_type> types{logical_type::make(type_id::VARCHAR),
                                        logical_type::make(type_id::BIGINT)};
  const std::vector<std::optional<uint64_t>> known{uint64_t{5}, std::nullopt};
  REQUIRE(estimate_bytes_per_row(types, k_avg_var, known) == 5 + 8);
}

TEST_CASE("estimate_bytes_per_row ignores a known width on a fixed-width column",
          "[admission_estimator]")
{
  // A Parquet footer's encoded size for an INT64 column can be far below 8 bytes per value
  // (dictionary, RLE); the cuDF carrier is 8 regardless.
  const std::vector<logical_type> types{logical_type::make(type_id::BIGINT)};
  const std::vector<std::optional<uint64_t>> known{uint64_t{1}};
  REQUIRE(estimate_bytes_per_row(types, k_avg_var, known) == 8);
}

TEST_CASE("estimate_bytes_per_row falls back where no width is known", "[admission_estimator]")
{
  const std::vector<logical_type> types{logical_type::make(type_id::VARCHAR),
                                        logical_type::make(type_id::VARCHAR),
                                        logical_type::make(type_id::VARCHAR)};
  // Shorter vector: the third column is covered by the fallback; unset entries too.
  const std::vector<std::optional<uint64_t>> known{uint64_t{10}, std::nullopt};
  REQUIRE(estimate_bytes_per_row(types, k_avg_var, known) == 10 + k_avg_var + k_avg_var);
  REQUIRE(estimate_bytes_per_row(types, k_avg_var, {}) == estimate_bytes_per_row(types, k_avg_var));
}

TEST_CASE("admission_row_count never goes below the planner estimate", "[admission_estimator]")
{
  // The unfiltered count raises an under-estimated filtered scan ...
  REQUIRE(admission_row_count(120'000'000, uint64_t{600'000'000}) == 600'000'000);
  // ... but never lowers one the planner already sized larger (appended rows, unions).
  REQUIRE(admission_row_count(700'000'000, uint64_t{600'000'000}) == 700'000'000);
  // Without a table count the planner estimate stands.
  REQUIRE(admission_row_count(120'000'000, std::nullopt) == 120'000'000);
  REQUIRE(admission_row_count(0, uint64_t{600'000'000}) == 600'000'000);
}

TEST_CASE("gpu_count_for_bytes rounds up to cover the total", "[admission_estimator]")
{
  constexpr uint64_t per_gpu = 100;
  REQUIRE(gpu_count_for_bytes(100, per_gpu, 4) == 1);  // exactly one GPU's worth
  REQUIRE(gpu_count_for_bytes(101, per_gpu, 4) == 2);  // one byte over spills to a second
  REQUIRE(gpu_count_for_bytes(250, per_gpu, 4) == 3);
  REQUIRE(gpu_count_for_bytes(400, per_gpu, 4) == 4);
}

TEST_CASE("gpu_count_for_bytes clamps to the available GPUs", "[admission_estimator]")
{
  // A query far larger than the fleet still only gets what exists.
  REQUIRE(gpu_count_for_bytes(1'000'000, 100, 4) == 4);
}

TEST_CASE("gpu_count_for_bytes always admits at least one GPU", "[admission_estimator]")
{
  REQUIRE(gpu_count_for_bytes(1, 1'000'000, 4) == 1);
}

TEST_CASE("gpu_count_for_bytes falls back to the full fleet when disabled", "[admission_estimator]")
{
  // bytes_per_gpu == 0 means estimation is off.
  REQUIRE(gpu_count_for_bytes(500, 0, 4) == 4);
  // Nothing to size against.
  REQUIRE(gpu_count_for_bytes(0, 100, 4) == 4);
}

TEST_CASE("gpu_count_for_bytes is a no-op on a single-GPU host", "[admission_estimator]")
{
  REQUIRE(gpu_count_for_bytes(1'000'000, 100, 1) == 1);
}

TEST_CASE("gpu_count_for_bytes survives a total near the 64-bit ceiling", "[admission_estimator]")
{
  // (total + per_gpu - 1) would wrap here, and a wrapped quotient narrowed to int can go
  // negative — clamping the largest possible query down to a single GPU.
  constexpr auto k_max = std::numeric_limits<uint64_t>::max();
  REQUIRE(gpu_count_for_bytes(k_max, 1, 8) == 8);
  REQUIRE(gpu_count_for_bytes(k_max, 1024, 4) == 4);
  REQUIRE(gpu_count_for_bytes(k_max - 1, 2, 2) == 2);
}

TEST_CASE("gpu_count_for_bytes clamps a quotient wider than int", "[admission_estimator]")
{
  // required is ~1.8e19, far past INT_MAX; the clamp must happen in 64-bit.
  REQUIRE(gpu_count_for_bytes(std::numeric_limits<uint64_t>::max(), 1, 3) == 3);
}

TEST_CASE("accumulate_scan_bytes sums the ordinary case", "[admission_estimator]")
{
  REQUIRE(accumulate_scan_bytes(0, 100, 8) == 800);
  REQUIRE(accumulate_scan_bytes(800, 50, 4) == 1000);
}

TEST_CASE("accumulate_scan_bytes ignores empty contributions", "[admission_estimator]")
{
  REQUIRE(accumulate_scan_bytes(500, 0, 8) == 500);
  REQUIRE(accumulate_scan_bytes(500, 100, 0) == 500);
}

TEST_CASE("accumulate_scan_bytes saturates instead of wrapping", "[admission_estimator]")
{
  constexpr auto k_max = std::numeric_limits<uint64_t>::max();
  // A bogus cardinality must not wrap the total into a small number, which would read as a
  // tiny query and admit it onto one GPU.
  REQUIRE(accumulate_scan_bytes(0, k_max, 64) == k_max);
  REQUIRE(accumulate_scan_bytes(k_max - 10, 100, 8) == k_max);
  // Saturated totals still resolve to the full fleet downstream.
  REQUIRE(gpu_count_for_bytes(accumulate_scan_bytes(0, k_max, 64), 1024, 4) == 4);
}

TEST_CASE("total_scan_bytes sums sized scans", "[admission_estimator]")
{
  const std::vector<scan_estimate> scans{{100, 8}, {50, 4}};
  auto const total = total_scan_bytes(scans);
  REQUIRE(total.has_value());
  CHECK(*total == 1000);
}

TEST_CASE("total_scan_bytes reports unsized when any scan has no row estimate",
          "[admission_estimator]")
{
  // Zero is ambiguous — provably empty, or simply unestimated — and Sirius reads it as
  // "cannot size" (sirius_physical_sort_sample gates on estimated_cardinality > 0). One
  // unsized scan makes the whole plan unsized, so a large query is never admitted onto a
  // small subset off the back of the scans that happened to carry estimates.
  CHECK_FALSE(total_scan_bytes({{0, 8}}).has_value());
  CHECK_FALSE(total_scan_bytes({{1'000'000, 8}, {0, 4}}).has_value());
  CHECK_FALSE(total_scan_bytes({{0, 4}, {1'000'000, 8}}).has_value());
}

TEST_CASE("total_scan_bytes reports unsized for a plan with no scans", "[admission_estimator]")
{
  CHECK_FALSE(total_scan_bytes({}).has_value());
}

TEST_CASE("total_scan_bytes saturates rather than wrapping", "[admission_estimator]")
{
  constexpr auto k_max = std::numeric_limits<uint64_t>::max();
  auto const total     = total_scan_bytes({{k_max, 64}, {10, 8}});
  REQUIRE(total.has_value());
  CHECK(*total == k_max);
}

TEST_CASE("estimate_bytes_per_row saturates instead of wrapping", "[admission_estimator]")
{
  constexpr auto k_max = std::numeric_limits<uint64_t>::max();
  const std::vector<logical_type> two_var{logical_type::make(type_id::VARCHAR),
                                          logical_type::make(type_id::VARCHAR)};

  // Two columns at 2^63+1 sum to 2^64+2, wrapping the row width to 2.
  CHECK(estimate_bytes_per_row(two_var, (k_max / 2) + 2) == k_max);

  const std::vector<logical_type> mixed{logical_type::make(type_id::BIGINT),
                                        logical_type::make(type_id::VARCHAR)};
  CHECK(estimate_bytes_per_row(mixed, k_max) == k_max);

  // A saturated width still resolves to the full fleet downstream, not to one GPU.
  auto const total = total_scan_bytes({{1000, k_max}});
  REQUIRE(total.has_value());
  CHECK(gpu_count_for_bytes(*total, 1024, 4) == 4);
}

TEST_CASE("admission default scales with GPU memory between 1 and 8 GiB", "[admission_estimator]")
{
  using sirius::config::admission_bytes_per_gpu_for_memory;
  constexpr uint64_t GiB = 1ULL << 30;
  constexpr uint64_t MiB = 1ULL << 20;
  // GB200: 189,471 MiB -> 7.9 GiB, under the cap.
  CHECK(admission_bytes_per_gpu_for_memory(189'471 * MiB) == (189'471 * MiB) / 24);
  // GB300 (256 GB) and anything larger hit the 8 GiB cap.
  CHECK(admission_bytes_per_gpu_for_memory(256 * GiB) == 8 * GiB);
  CHECK(admission_bytes_per_gpu_for_memory(192 * GiB) == 8 * GiB);
  // 80 GB H100 class lands in between.
  CHECK(admission_bytes_per_gpu_for_memory(80 * GiB) == (80 * GiB) / 24);
  // L4 class (24 GB) and smaller sit on the 1 GiB floor.
  CHECK(admission_bytes_per_gpu_for_memory(24 * GiB) == 1 * GiB);
  CHECK(admission_bytes_per_gpu_for_memory(16 * GiB) == 1 * GiB);
  CHECK(admission_bytes_per_gpu_for_memory(0) == 1 * GiB);
}

TEST_CASE("admission uses the largest available callback row count", "[gpu_admission]")
{
  using sirius::planner::detail::reported_row_count;
  CHECK_FALSE(reported_row_count(duckdb::NodeStatistics{}).has_value());
  CHECK(reported_row_count(duckdb::NodeStatistics{0}) == 0);
  CHECK(reported_row_count(duckdb::NodeStatistics{120}) == 120);
  // Pinned DuckDB reports stored rows as estimated and transaction-local rows in max.
  CHECK(reported_row_count(duckdb::NodeStatistics{100, 120}) == 120);
  // Other callbacks may report the two counts in the opposite order.
  CHECK(reported_row_count(duckdb::NodeStatistics{120, 100}) == 120);
  duckdb::NodeStatistics max_only;
  max_only.has_max_cardinality = true;
  max_only.max_cardinality     = 120;
  CHECK(reported_row_count(max_only) == 120);
}

namespace {
duckdb_parquet::RowGroup admission_string_group(int64_t rows,
                                                std::optional<int64_t> chars,
                                                duckdb_parquet::Encoding::type encoding)
{
  duckdb_parquet::ColumnMetaData meta;
  meta.__set_type(duckdb_parquet::Type::BYTE_ARRAY);
  meta.__set_num_values(rows);
  meta.__set_encodings({encoding});
  meta.__set_total_uncompressed_size(10);  // deliberately smaller than the decoded strings
  if (chars) {
    duckdb_parquet::SizeStatistics sizes;
    sizes.__set_unencoded_byte_array_data_bytes(*chars);
    meta.__set_size_statistics(sizes);
  }
  duckdb_parquet::ColumnChunk chunk;
  chunk.__set_meta_data(meta);
  duckdb_parquet::RowGroup group;
  group.__set_num_rows(rows);
  group.__set_columns({chunk});
  return group;
}
}  // namespace

TEST_CASE("Parquet admission requires decoded sizes for the entire scan", "[gpu_admission]")
{
  using duckdb_parquet::Encoding;
  using sirius::planner::detail::parquet_column_widths;
  duckdb_parquet::FileMetaData metadata;
  metadata.__set_row_groups({admission_string_group(10, 100, Encoding::PLAIN),
                             admission_string_group(30, 900, Encoding::RLE_DICTIONARY)});
  auto widths = parquet_column_widths(metadata, 1, true);
  REQUIRE(widths.size() == 1);
  CHECK(widths[0] == 29);  // (100 + 900) / 40 chars + four-byte offsets

  SECTION("first-file statistics never describe a multi-file scan")
  {
    CHECK(parquet_column_widths(metadata, 1, false).empty());
  }
  SECTION("one missing row group invalidates the column width")
  {
    metadata.row_groups[1] = admission_string_group(30, std::nullopt, Encoding::PLAIN);
    CHECK_FALSE(parquet_column_widths(metadata, 1, true)[0].has_value());
  }
  SECTION("encoded-only chunks retain the configured fallback")
  {
    for (auto encoding : {Encoding::PLAIN,
                          Encoding::DELTA_LENGTH_BYTE_ARRAY,
                          Encoding::DELTA_BYTE_ARRAY,
                          Encoding::RLE_DICTIONARY}) {
      metadata.__set_row_groups({admission_string_group(10, std::nullopt, encoding)});
      CHECK_FALSE(parquet_column_widths(metadata, 1, true)[0].has_value());
    }
  }
  SECTION("decoded statistics work for delta-encoded strings")
  {
    metadata.__set_row_groups({admission_string_group(10, 101, Encoding::DELTA_BYTE_ARRAY)});
    CHECK(parquet_column_widths(metadata, 1, true)[0] == 15);
  }
  SECTION("empty and all-null strings still require offsets")
  {
    metadata.__set_row_groups({admission_string_group(10, 0, Encoding::PLAIN)});
    CHECK(parquet_column_widths(metadata, 1, true)[0] == 4);
  }
  SECTION("malformed and nested metadata cannot supply a width")
  {
    CHECK(parquet_column_widths(metadata, 2, true).empty());
    metadata.row_groups[1].columns.clear();
    CHECK(parquet_column_widths(metadata, 1, true).empty());
  }
}
