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

#include <cstdint>
#include <limits>
#include <vector>

using sirius::logical_type;
using sirius::type_id;
using sirius::planner::accumulate_scan_bytes;
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
