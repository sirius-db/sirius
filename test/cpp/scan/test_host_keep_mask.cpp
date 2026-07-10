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

// Gates for the shared host-keep-mask GPU apply helpers:
// apply_host_keep_mask (byte mask — the positional-delete path) and
// apply_host_keep_bitmask (bit-packed, cuDF bitmask convention — the MVCC
// path) must drop exactly the flagged rows, agree with each other on every
// pattern (the bitmask goes through cudf::mask_to_bools, so this also locks
// the host bit-packing convention to what the device reads), and reject
// mis-sized masks loudly instead of filtering garbage.

#include "operator/operator_test_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <op/scan/host_keep_mask.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

struct test_env {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> mgr;
  cucascade::memory::memory_space* gpu_space;

  test_env()
    : mgr(sirius::test::operator_utils::initialize_memory_manager()),
      gpu_space(mgr->get_memory_space(cucascade::memory::Tier::GPU, 0))
  {
  }
};

test_env& env()
{
  static test_env e;
  return e;
}

/// One INT32 column with values 0..rows-1 (row identity), so surviving values
/// name the surviving row indices.
std::unique_ptr<cudf::column> make_iota_column(std::size_t rows)
{
  auto& e     = env();
  auto mr     = sirius::test::operator_utils::get_resource_ref(*e.gpu_space);
  auto stream = sirius::test::operator_utils::default_stream();
  std::vector<int32_t> values(rows);
  std::iota(values.begin(), values.end(), 0);
  auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                       static_cast<cudf::size_type>(rows),
                                       cudf::mask_state::UNALLOCATED,
                                       stream,
                                       mr);
  if (rows > 0) {
    cudaMemcpy(col->mutable_view().data<int32_t>(),
               values.data(),
               sizeof(int32_t) * rows,
               cudaMemcpyHostToDevice);
  }
  return col;
}

/// Pack a byte keep-mask into cuDF bitmask words (bit i%32 of word i/32).
std::vector<std::uint32_t> pack_bits(std::vector<std::uint8_t> const& keep)
{
  std::vector<std::uint32_t> words((keep.size() + 31) / 32, 0u);
  for (std::size_t i = 0; i < keep.size(); ++i) {
    if (keep[i]) { words[i / 32] |= 1u << (i % 32); }
  }
  return words;
}

std::vector<int32_t> expected_survivors(std::vector<std::uint8_t> const& keep)
{
  std::vector<int32_t> out;
  for (std::size_t i = 0; i < keep.size(); ++i) {
    if (keep[i]) { out.push_back(static_cast<int32_t>(i)); }
  }
  return out;
}

void require_survivors(std::unique_ptr<cudf::table> const& result,
                       std::vector<std::uint8_t> const& keep)
{
  sirius::test::operator_utils::default_stream().synchronize();
  auto const expected = expected_survivors(keep);
  REQUIRE(result->num_columns() == 1);
  REQUIRE(static_cast<std::size_t>(result->num_rows()) == expected.size());
  if (!expected.empty()) {
    auto actual =
      sirius::test::operator_utils::copy_column_to_host<int32_t>(result->view().column(0));
    REQUIRE(actual == expected);
  }
}

/// Deterministic irregular keep pattern (~50%) for equivalence sweeps.
std::vector<std::uint8_t> pseudo_random_keep(std::size_t rows)
{
  std::vector<std::uint8_t> keep(rows);
  for (std::size_t i = 0; i < rows; ++i) {
    keep[i] = static_cast<std::uint8_t>(((i * 2654435761u) >> 7) & 1u);
  }
  return keep;
}

}  // namespace

TEST_CASE("apply_host_keep_mask keeps exactly the flagged rows", "[host_keep_mask][scan]")
{
  auto& e     = env();
  auto mr     = sirius::test::operator_utils::get_resource_ref(*e.gpu_space);
  auto stream = sirius::test::operator_utils::default_stream();

  SECTION("sparse pattern")
  {
    auto col = make_iota_column(100);
    std::vector<std::uint8_t> keep(100, 1u);
    for (auto drop : {0, 31, 32, 63, 99}) {
      keep[static_cast<std::size_t>(drop)] = 0u;
    }
    auto result =
      sirius::op::scan::apply_host_keep_mask(cudf::table_view({col->view()}), keep, stream, mr);
    require_survivors(result, keep);
  }

  SECTION("all rows kept")
  {
    auto col = make_iota_column(64);
    std::vector<std::uint8_t> keep(64, 1u);
    auto result =
      sirius::op::scan::apply_host_keep_mask(cudf::table_view({col->view()}), keep, stream, mr);
    require_survivors(result, keep);
  }

  SECTION("all rows dropped keeps the schema")
  {
    auto col = make_iota_column(10);
    std::vector<std::uint8_t> keep(10, 0u);
    auto result =
      sirius::op::scan::apply_host_keep_mask(cudf::table_view({col->view()}), keep, stream, mr);
    REQUIRE(result->num_rows() == 0);
    REQUIRE(result->num_columns() == 1);
  }

  SECTION("empty table")
  {
    auto col    = make_iota_column(0);
    auto result = sirius::op::scan::apply_host_keep_mask(
      cudf::table_view({col->view()}), std::vector<std::uint8_t>{}, stream, mr);
    REQUIRE(result->num_rows() == 0);
  }
}

TEST_CASE("apply_host_keep_mask rejects a mis-sized mask", "[host_keep_mask][scan]")
{
  auto& e     = env();
  auto mr     = sirius::test::operator_utils::get_resource_ref(*e.gpu_space);
  auto stream = sirius::test::operator_utils::default_stream();
  auto col    = make_iota_column(4);
  std::vector<std::uint8_t> keep(5, 1u);
  REQUIRE_THROWS_AS(
    sirius::op::scan::apply_host_keep_mask(cudf::table_view({col->view()}), keep, stream, mr),
    std::invalid_argument);
}

TEST_CASE("apply_host_keep_bitmask matches the byte-mask reference", "[host_keep_mask][scan]")
{
  auto& e     = env();
  auto mr     = sirius::test::operator_utils::get_resource_ref(*e.gpu_space);
  auto stream = sirius::test::operator_utils::default_stream();

  SECTION("irregular patterns across word-count shapes")
  {
    // 1 (single partial word), 31/32/33 (word boundary +/- 1), 64 (exact two
    // words), 100 (partial final word), 2085 (many words + partial tail).
    for (std::size_t rows : {std::size_t{1},
                             std::size_t{31},
                             std::size_t{32},
                             std::size_t{33},
                             std::size_t{64},
                             std::size_t{100},
                             std::size_t{2085}}) {
      auto keep  = pseudo_random_keep(rows);
      auto words = pack_bits(keep);
      auto col   = make_iota_column(rows);
      cudf::table_view view({col->view()});

      auto via_bits = sirius::op::scan::apply_host_keep_bitmask(view, words, rows, stream, mr);
      require_survivors(via_bits, keep);

      auto via_bytes = sirius::op::scan::apply_host_keep_mask(view, keep, stream, mr);
      REQUIRE(via_bits->num_rows() == via_bytes->num_rows());
    }
  }

  SECTION("drop range straddling a word boundary")
  {
    auto const rows = std::size_t{64};
    std::vector<std::uint8_t> keep(rows, 1u);
    for (std::size_t i = 30; i <= 35; ++i) {
      keep[i] = 0u;  // crosses bit 31 -> 32 (word 0 -> word 1)
    }
    auto col    = make_iota_column(rows);
    auto result = sirius::op::scan::apply_host_keep_bitmask(
      cudf::table_view({col->view()}), pack_bits(keep), rows, stream, mr);
    require_survivors(result, keep);
  }

  SECTION("all dropped / all kept")
  {
    auto const rows = std::size_t{100};
    auto col        = make_iota_column(rows);
    cudf::table_view view({col->view()});

    std::vector<std::uint8_t> none(rows, 0u);
    auto dropped =
      sirius::op::scan::apply_host_keep_bitmask(view, pack_bits(none), rows, stream, mr);
    REQUIRE(dropped->num_rows() == 0);
    REQUIRE(dropped->num_columns() == 1);

    std::vector<std::uint8_t> all(rows, 1u);
    auto kept = sirius::op::scan::apply_host_keep_bitmask(view, pack_bits(all), rows, stream, mr);
    require_survivors(kept, all);
  }

  SECTION("empty table with an empty mask")
  {
    auto col    = make_iota_column(0);
    auto result = sirius::op::scan::apply_host_keep_bitmask(
      cudf::table_view({col->view()}), std::vector<std::uint32_t>{}, 0, stream, mr);
    REQUIRE(result->num_rows() == 0);
  }
}

TEST_CASE("apply_host_keep_bitmask rejects mis-sized masks", "[host_keep_mask][scan]")
{
  auto& e     = env();
  auto mr     = sirius::test::operator_utils::get_resource_ref(*e.gpu_space);
  auto stream = sirius::test::operator_utils::default_stream();
  auto col    = make_iota_column(40);
  cudf::table_view view({col->view()});

  SECTION("row count disagrees with the table")
  {
    std::vector<std::uint32_t> words(2, 0xFFFFFFFFu);
    REQUIRE_THROWS_AS(sirius::op::scan::apply_host_keep_bitmask(view, words, 39, stream, mr),
                      std::invalid_argument);
  }

  SECTION("word count disagrees with the row count")
  {
    std::vector<std::uint32_t> words(1, 0xFFFFFFFFu);  // 40 rows need 2 words
    REQUIRE_THROWS_AS(sirius::op::scan::apply_host_keep_bitmask(view, words, 40, stream, mr),
                      std::invalid_argument);
  }
}
