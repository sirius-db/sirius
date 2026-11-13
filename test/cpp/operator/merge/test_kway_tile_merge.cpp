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
#include "op/merge/detail/kway_merge_detail.hpp"
#include "operator/merge/merge_test_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sirius::op::merge::detail {

// Defined in kway_tile_merge.cu.
void tile_merge_with_sub_tiles_for_test(const kway_key_comparator& cmp,
                                        cudf::device_span<const cudf::size_type> lengths,
                                        cudf::device_span<const cudf::size_type> tile_coranks,
                                        cudf::size_type target,
                                        cudf::size_type tile_rows,
                                        cudf::device_span<std::int32_t> out_input,
                                        cudf::device_span<cudf::size_type> out_row,
                                        std::int32_t sub_tiles,
                                        rmm::cuda_stream_view stream);

}  // namespace sirius::op::merge::detail

using namespace sirius::op::merge::detail;
using namespace sirius::test;

namespace {

/// Marks output entries tile_merge must not write.
constexpr std::int32_t kUntouched = -7;
/// Entries past `target` in the output buffers; they must stay untouched.
constexpr std::size_t kGuardEntries = 16;

template <typename T>
rmm::device_uvector<T> to_device(const std::vector<T>& host)
{
  rmm::device_uvector<T> out(
    host.size(), cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  if (!host.empty()) {
    REQUIRE(cudaMemcpy(out.data(), host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice) ==
            cudaSuccess);
  }
  return out;
}

template <typename T>
std::vector<T> to_host(const rmm::device_uvector<T>& device)
{
  std::vector<T> out(device.size());
  if (!out.empty()) {
    cudf::get_default_stream().sync();
    REQUIRE(cudaMemcpy(out.data(), device.data(), out.size() * sizeof(T), cudaMemcpyDeviceToHost) ==
            cudaSuccess);
  }
  return out;
}

/// INT64 key column drawn from [0, domain); a positive @p null_probability adds NULLs.
std::unique_ptr<cudf::column> random_int_column(cudf::size_type rows,
                                                std::int64_t domain,
                                                double null_probability,
                                                std::mt19937& rng)
{
  if (rows == 0) { return cudf::make_empty_column(cudf::type_id::INT64); }
  std::vector<std::int64_t> values;
  std::vector<bool> valids;
  std::bernoulli_distribution is_null(null_probability);
  for (cudf::size_type i = 0; i < rows; ++i) {
    values.push_back(static_cast<std::int64_t>(rng() % static_cast<std::uint64_t>(domain)));
    if (null_probability > 0.0) { valids.push_back(!is_null(rng)); }
  }
  return merge_utils::make_column(values, cudf::type_id::INT64, valids);
}

/// STRING key column of short strings over a three-letter alphabet, so prefixes and ties are
/// frequent.
std::unique_ptr<cudf::column> random_string_column(cudf::size_type rows, std::mt19937& rng)
{
  if (rows == 0) { return cudf::make_empty_column(cudf::type_id::STRING); }
  std::vector<std::string> values;
  for (cudf::size_type i = 0; i < rows; ++i) {
    std::string value;
    const auto length = rng() % 5;
    for (std::size_t c = 0; c < length; ++c) {
      value.push_back(static_cast<char>('a' + rng() % 3));
    }
    values.push_back(std::move(value));
  }
  return merge_utils::make_string_column(values);
}

/// Random window lengths in [max_rows / 2, max_rows]. With more than two windows the second one
/// is empty, and so is about one in four of the others past the fifth.
std::vector<cudf::size_type> random_lengths(std::size_t k,
                                            cudf::size_type max_rows,
                                            std::mt19937& rng)
{
  const auto min_rows = max_rows / 2;
  std::vector<cudf::size_type> lengths;
  for (std::size_t i = 0; i < k; ++i) {
    const bool empty = (k > 2 && i == 1) || (i >= 5 && rng() % 4 == 0);
    lengths.push_back(empty ? 0
                            : min_rows + static_cast<cudf::size_type>(
                                           rng() % static_cast<unsigned>(max_rows - min_rows + 1)));
  }
  return lengths;
}

/// The k key windows of one merge: each window sorted by the key order.
struct merge_case {
  std::vector<std::unique_ptr<cudf::table>> windows;
  std::vector<cudf::table_view> views;
  std::vector<cudf::size_type> lengths;
  std::vector<cudf::order> orders;
  std::vector<cudf::null_order> null_orders;
  /// (input, row) of every row in merged order: the stable sort of the concatenated windows,
  /// which breaks ties by input, then by row, as kway_key_device_view::less does.
  std::vector<std::pair<std::int32_t, cudf::size_type>> merged;

  [[nodiscard]] cudf::size_type total_rows() const
  {
    return static_cast<cudf::size_type>(merged.size());
  }
};

merge_case make_case(std::vector<std::unique_ptr<cudf::table>> unsorted,
                     std::vector<cudf::order> orders,
                     std::vector<cudf::null_order> null_orders)
{
  merge_case out;
  out.orders      = std::move(orders);
  out.null_orders = std::move(null_orders);
  std::vector<std::pair<std::int32_t, cudf::size_type>> concatenated_rows;
  for (std::size_t i = 0; i < unsorted.size(); ++i) {
    auto& window = unsorted[i];
    out.windows.push_back(window->num_rows() > 0
                            ? cudf::sort(window->view(), out.orders, out.null_orders)
                            : std::move(window));
    out.views.push_back(out.windows.back()->view());
    out.lengths.push_back(out.views.back().num_rows());
    for (cudf::size_type r = 0; r < out.lengths.back(); ++r) {
      concatenated_rows.emplace_back(static_cast<std::int32_t>(i), r);
    }
  }
  if (!concatenated_rows.empty()) {
    auto concatenated = cudf::concatenate(out.views);
    auto order = cudf::stable_sorted_order(concatenated->view(), out.orders, out.null_orders);
    const auto host_order = operator_utils::copy_column_to_host<cudf::size_type>(order->view());
    for (const auto position : host_order) {
      out.merged.push_back(concatenated_rows[static_cast<std::size_t>(position)]);
    }
  }
  return out;
}

/// Host reference of the co-ranks: per-input counts of the first min(b * tile_rows, target)
/// merged rows, for every tile boundary b.
std::vector<cudf::size_type> reference_coranks(const merge_case& data,
                                               cudf::size_type target,
                                               cudf::size_type tile_rows)
{
  const auto k     = data.views.size();
  const auto tiles = target == 0 ? 0 : (target - 1) / tile_rows + 1;
  std::vector<cudf::size_type> coranks((static_cast<std::size_t>(tiles) + 1) * k, 0);
  std::vector<cudf::size_type> counts(k, 0);
  for (cudf::size_type pos = 0; pos < target; ++pos) {
    ++counts[static_cast<std::size_t>(data.merged[static_cast<std::size_t>(pos)].first)];
    if ((pos + 1) % tile_rows == 0 || pos + 1 == target) {
      const auto boundary = static_cast<std::size_t>(pos / tile_rows) + 1;
      std::copy(counts.begin(), counts.end(), coranks.begin() + boundary * k);
    }
  }
  return coranks;
}

kway_key_comparator make_comparator(const merge_case& data)
{
  return kway_key_comparator::create(data.views,
                                     data.orders,
                                     data.null_orders,
                                     cudf::get_default_stream(),
                                     cudf::get_current_device_resource_ref());
}

/// Runs tile_merge for the first @p target merged rows and checks the gather map against the
/// reference order; entries past @p target must stay untouched. @p sub_tiles > 0 forces that many
/// sub-tiles per tile; 0 runs tile_merge itself, which chooses the count from the device.
void check_tile_merge(const merge_case& data,
                      const kway_key_comparator& cmp,
                      cudf::size_type target,
                      cudf::size_type tile_rows,
                      std::int32_t sub_tiles = 0)
{
  CAPTURE(data.views.size(), data.total_rows(), target, tile_rows, sub_tiles);
  auto stream  = cudf::get_default_stream();
  auto lengths = to_device(data.lengths);
  auto coranks = to_device(reference_coranks(data, target, tile_rows));

  const auto out_size = static_cast<std::size_t>(target) + kGuardEntries;
  auto out_input      = to_device(std::vector<std::int32_t>(out_size, kUntouched));
  auto out_row        = to_device(std::vector<cudf::size_type>(out_size, kUntouched));

  const cudf::device_span<const cudf::size_type> lengths_span(lengths.data(), lengths.size());
  const cudf::device_span<const cudf::size_type> coranks_span(coranks.data(), coranks.size());
  const cudf::device_span<std::int32_t> input_span(out_input.data(), out_input.size());
  const cudf::device_span<cudf::size_type> row_span(out_row.data(), out_row.size());
  if (sub_tiles > 0) {
    tile_merge_with_sub_tiles_for_test(
      cmp, lengths_span, coranks_span, target, tile_rows, input_span, row_span, sub_tiles, stream);
  } else {
    tile_merge(cmp, lengths_span, coranks_span, target, tile_rows, input_span, row_span, stream);
  }

  const auto got_input   = to_host(out_input);
  const auto got_row     = to_host(out_row);
  std::size_t mismatches = 0;
  std::size_t first_bad  = out_size;
  for (std::size_t pos = 0; pos < out_size; ++pos) {
    const bool in_range   = pos < static_cast<std::size_t>(target);
    const auto want_input = in_range ? data.merged[pos].first : kUntouched;
    const auto want_row   = in_range ? data.merged[pos].second : kUntouched;
    if (got_input[pos] != want_input || got_row[pos] != want_row) {
      ++mismatches;
      first_bad = std::min(first_bad, pos);
    }
  }
  CAPTURE(first_bad);
  CHECK(mismatches == 0);
}

/// Checks target = Σ len and targets below it, for each of @p tile_rows_list: through tile_merge
/// (0) and with 1, 3, 16 and 64 sub-tiles per tile forced (a tile of fewer than 64 rows is never
/// split, and the fan-in caps the count, so the larger values also cover the capping).
void check_targets(const merge_case& data, const std::vector<cudf::size_type>& tile_rows_list)
{
  auto cmp         = make_comparator(data);
  const auto total = data.total_rows();
  for (const auto tile_rows : tile_rows_list) {
    for (const std::int32_t sub_tiles : {0, 1, 3, 16, 64}) {
      if (sub_tiles > 1 && tile_rows < 64) { continue; }
      check_tile_merge(data, cmp, total, tile_rows, sub_tiles);
      if (total > 1) {
        check_tile_merge(data, cmp, total * 2 / 3, tile_rows, sub_tiles);
        check_tile_merge(data, cmp, 1, tile_rows, sub_tiles);
      }
      if (total > tile_rows) {
        // A whole number of tiles: the last tile is full.
        check_tile_merge(data, cmp, total / tile_rows * tile_rows, tile_rows, sub_tiles);
      }
    }
  }
}

std::vector<std::unique_ptr<cudf::table>> int_windows(const std::vector<cudf::size_type>& lengths,
                                                      std::int64_t domain,
                                                      double null_probability,
                                                      std::mt19937& rng)
{
  std::vector<std::unique_ptr<cudf::table>> windows;
  for (const auto rows : lengths) {
    windows.push_back(
      merge_utils::make_table(random_int_column(rows, domain, null_probability, rng)));
  }
  return windows;
}

}  // namespace

TEST_CASE("tile merge matches the reference merged order", "[kway_tile_merge]")
{
  const auto k =
    GENERATE(std::size_t{1}, std::size_t{2}, std::size_t{5}, std::size_t{64}, std::size_t{200});
  std::mt19937 rng(4200 + static_cast<unsigned>(k));

  // About 6000 rows in total: several tiles of 2048 rows and hundreds of tiles of 7.
  const auto max_rows = static_cast<cudf::size_type>(2 * 6000 / k);
  auto lengths        = random_lengths(k, max_rows, rng);
  if (k > 5) {
    lengths.front() = 0;
    lengths.back()  = 0;
  }
  const auto data = make_case(
    int_windows(lengths, 1000, 0.0, rng), {cudf::order::ASCENDING}, {cudf::null_order::AFTER});
  REQUIRE(data.total_rows() > 2048);
  check_targets(data, {7, 100, 2048});
}

TEST_CASE("tile merge handles the largest fan-ins", "[kway_tile_merge]")
{
  // 1024 inputs leave room for only a few tiles per block; 4096 is the documented limit.
  const auto k = GENERATE(std::size_t{1024}, std::size_t{4096});
  std::mt19937 rng(77 + static_cast<unsigned>(k));
  const auto data = make_case(int_windows(random_lengths(k, 6, rng), 50, 0.0, rng),
                              {cudf::order::ASCENDING},
                              {cudf::null_order::AFTER});
  check_targets(data, {7, 100, 2048});
}

TEST_CASE("tile merge breaks heavy ties by input, then by row", "[kway_tile_merge]")
{
  std::mt19937 rng(11);

  SECTION("three distinct keys and NULLs")
  {
    const auto null_order = GENERATE(cudf::null_order::BEFORE, cudf::null_order::AFTER);
    const auto data       = make_case(int_windows(random_lengths(7, 900, rng), 3, 0.2, rng),
                                      {cudf::order::ASCENDING},
                                      {null_order});
    check_targets(data, {7, 100, 2048});
  }

  SECTION("every key equal")
  {
    const auto data = make_case(int_windows({500, 0, 1200, 3, 800, 800}, 1, 0.0, rng),
                                {cudf::order::ASCENDING},
                                {cudf::null_order::AFTER});
    // With equal keys the merged order is the inputs one after the other.
    REQUIRE(data.merged.front() == std::pair<std::int32_t, cudf::size_type>{0, 0});
    REQUIRE(data.merged[500] == std::pair<std::int32_t, cudf::size_type>{2, 0});
    check_targets(data, {7, 100, 2048});
  }
}

TEST_CASE("tile merge orders STRING keys", "[kway_tile_merge]")
{
  const auto order = GENERATE(cudf::order::ASCENDING, cudf::order::DESCENDING);
  std::mt19937 rng(23 + static_cast<unsigned>(order));
  std::vector<std::unique_ptr<cudf::table>> windows;
  for (const auto rows : random_lengths(5, 1500, rng)) {
    windows.push_back(merge_utils::make_table(random_string_column(rows, rng)));
  }
  const auto data = make_case(std::move(windows), {order}, {cudf::null_order::AFTER});
  check_targets(data, {7, 100, 2048});
}

TEST_CASE("tile merge orders descending and multi-column keys", "[kway_tile_merge]")
{
  std::mt19937 rng(31);

  SECTION("descending INT64 with NULLs")
  {
    const auto null_order = GENERATE(cudf::null_order::BEFORE, cudf::null_order::AFTER);
    const auto data       = make_case(int_windows(random_lengths(9, 700, rng), 400, 0.1, rng),
                                      {cudf::order::DESCENDING},
                                      {null_order});
    check_targets(data, {7, 100, 2048});
  }

  SECTION("descending INT64, then ascending STRING")
  {
    std::vector<std::unique_ptr<cudf::table>> windows;
    for (const auto rows : random_lengths(6, 900, rng)) {
      windows.push_back(merge_utils::make_table(random_int_column(rows, 4, 0.1, rng),
                                                random_string_column(rows, rng)));
    }
    const auto data = make_case(std::move(windows),
                                {cudf::order::DESCENDING, cudf::order::ASCENDING},
                                {cudf::null_order::BEFORE, cudf::null_order::AFTER});
    check_targets(data, {7, 100, 2048});
  }
}

TEST_CASE("tile merge handles one-row tiles and a tile larger than the target", "[kway_tile_merge]")
{
  std::mt19937 rng(17);
  const auto data = make_case(int_windows(random_lengths(4, 300, rng), 50, 0.05, rng),
                              {cudf::order::ASCENDING},
                              {cudf::null_order::BEFORE});
  check_targets(data, {1, data.total_rows() + 100});
}

TEST_CASE("tile merge ignores co-rank entries past the last boundary", "[kway_tile_merge]")
{
  std::mt19937 rng(19);
  auto stream            = cudf::get_default_stream();
  const auto data        = make_case(int_windows(random_lengths(3, 200, rng), 30, 0.0, rng),
                                     {cudf::order::ASCENDING},
                                     {cudf::null_order::AFTER});
  auto cmp               = make_comparator(data);
  const auto target      = data.total_rows() / 2;
  const auto tile_rows   = cudf::size_type{13};
  auto host_coranks      = reference_coranks(data, target, tile_rows);
  const auto exact_count = host_coranks.size();
  // Garbage boundaries past the last one: a kernel that read them would run out of range.
  host_coranks.resize(exact_count + 4 * data.views.size(), 1 << 30);
  auto lengths   = to_device(data.lengths);
  auto coranks   = to_device(host_coranks);
  auto out_input = to_device(std::vector<std::int32_t>(target, kUntouched));
  auto out_row   = to_device(std::vector<cudf::size_type>(target, kUntouched));

  tile_merge(cmp,
             cudf::device_span<const cudf::size_type>(lengths.data(), lengths.size()),
             cudf::device_span<const cudf::size_type>(coranks.data(), coranks.size()),
             target,
             tile_rows,
             cudf::device_span<std::int32_t>(out_input.data(), out_input.size()),
             cudf::device_span<cudf::size_type>(out_row.data(), out_row.size()),
             stream);

  const auto got_input = to_host(out_input);
  const auto got_row   = to_host(out_row);
  for (cudf::size_type pos = 0; pos < target; ++pos) {
    CAPTURE(pos);
    REQUIRE(got_input[pos] == data.merged[pos].first);
    REQUIRE(got_row[pos] == data.merged[pos].second);
  }
}

TEST_CASE("tile merge with target 0 writes nothing", "[kway_tile_merge]")
{
  std::mt19937 rng(5);
  auto stream = cudf::get_default_stream();

  SECTION("non-empty windows")
  {
    const auto data = make_case(
      int_windows({40, 0, 25}, 10, 0.0, rng), {cudf::order::ASCENDING}, {cudf::null_order::AFTER});
    auto cmp = make_comparator(data);
    check_tile_merge(data, cmp, 0, 7);
    check_tile_merge(data, cmp, 0, 2048);

    // No co-ranks and no output at all.
    auto lengths = to_device(data.lengths);
    CHECK_NOTHROW(
      tile_merge(cmp,
                 cudf::device_span<const cudf::size_type>(lengths.data(), lengths.size()),
                 cudf::device_span<const cudf::size_type>(),
                 0,
                 2048,
                 cudf::device_span<std::int32_t>(),
                 cudf::device_span<cudf::size_type>(),
                 stream));
  }

  SECTION("every window empty")
  {
    const auto data = make_case(
      int_windows({0, 0, 0}, 10, 0.0, rng), {cudf::order::ASCENDING}, {cudf::null_order::AFTER});
    auto cmp = make_comparator(data);
    check_tile_merge(data, cmp, 0, 7);
  }
}

TEST_CASE("tile merge rejects invalid arguments", "[kway_tile_merge]")
{
  std::mt19937 rng(9);
  auto stream = cudf::get_default_stream();

  auto run = [&](const merge_case& data,
                 const kway_key_comparator& cmp,
                 std::size_t num_lengths,
                 std::size_t num_coranks,
                 cudf::size_type target,
                 cudf::size_type tile_rows,
                 std::size_t out_entries) {
    auto lengths   = to_device(std::vector<cudf::size_type>(num_lengths, 0));
    auto coranks   = to_device(std::vector<cudf::size_type>(num_coranks, 0));
    auto out_input = to_device(std::vector<std::int32_t>(out_entries, kUntouched));
    auto out_row   = to_device(std::vector<cudf::size_type>(out_entries, kUntouched));
    tile_merge(cmp,
               cudf::device_span<const cudf::size_type>(lengths.data(), lengths.size()),
               cudf::device_span<const cudf::size_type>(coranks.data(), coranks.size()),
               target,
               tile_rows,
               cudf::device_span<std::int32_t>(out_input.data(), out_input.size()),
               cudf::device_span<cudf::size_type>(out_row.data(), out_row.size()),
               stream);
    stream.sync();
    (void)data;
  };

  SECTION("sizes that do not match")
  {
    const auto data = make_case(
      int_windows({20, 20}, 10, 0.0, rng), {cudf::order::ASCENDING}, {cudf::null_order::AFTER});
    auto cmp = make_comparator(data);
    // 40 rows in tiles of 7: 6 tiles, 7 boundaries, 14 co-ranks. All-zero co-ranks are valid
    // input sizes; the merge then emits nothing.
    CHECK_NOTHROW(run(data, cmp, 2, 14, 40, 7, 40));
    CHECK_THROWS_AS(run(data, cmp, 3, 14, 40, 7, 40), std::invalid_argument);
    CHECK_THROWS_AS(run(data, cmp, 2, 13, 40, 7, 40), std::invalid_argument);
    CHECK_THROWS_AS(run(data, cmp, 2, 14, 40, 7, 39), std::invalid_argument);
    CHECK_THROWS_AS(run(data, cmp, 2, 14, 40, 0, 40), std::invalid_argument);
    CHECK_THROWS_AS(run(data, cmp, 2, 14, 40, -1, 40), std::invalid_argument);
    CHECK_THROWS_AS(run(data, cmp, 2, 14, -1, 7, 40), std::invalid_argument);
  }

  SECTION("more inputs than the supported fan-in")
  {
    const std::size_t k = 4097;
    const auto data     = make_case(int_windows(std::vector<cudf::size_type>(k, 1), 10, 0.0, rng),
                                    {cudf::order::ASCENDING},
                                    {cudf::null_order::AFTER});
    auto cmp            = make_comparator(data);
    CHECK_THROWS_AS(run(data, cmp, k, 2 * k, 10, 2048, 10), std::invalid_argument);
  }
}
