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

#include "catch.hpp"
#include "op/order/gpu_sampled_merge.hpp"
#include "operator_test_utils.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/filling.hpp>
#include <cudf/merge.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/unary.hpp>

#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <type_traits>

using namespace sirius::op;
using namespace sirius::test::operator_utils;
namespace sm = sirius::op::sampled_merge;

namespace {
sort_comparison_config make_comparison(std::vector<int> const& keys,
                                       std::vector<duckdb::OrderType> const& directions,
                                       uint64_t width)
{
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  for (std::size_t i = 0; i < keys.size(); ++i) {
    orders.emplace_back(
      directions[i],
      duckdb::OrderByNullType::NULLS_LAST,
      duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::BIGINT, keys[i]));
  }
  return sort_comparison_config(orders, width);
}

// Deliberately complete SAMPLE items in reverse input order. Frozen input IDs
// must determine cuts regardless of assembly order or SQL peer merge order.
void verify_gpu_cuts(std::vector<std::shared_ptr<cucascade::data_batch>> const& batches,
                     sort_comparison_config const& comparison,
                     uint64_t stride,
                     uint64_t sample_step)
{
  auto* space = get_default_gpu_space();
  auto stream = default_stream();
  auto mr     = get_resource_ref(*space);
  uint64_t k  = 0;
  for (auto const& batch : batches) {
    k += sirius::get_cudf_table_view(*batch).num_rows() != 0;
  }
  auto limit = stride * sample_step + k * (stride - 1);
  std::vector<std::unique_ptr<cudf::table>> originals, inputs, samples, outputs;
  int64_t total_rows = 0;
  std::vector<cudf::table_view> sample_views;
  for (auto const& batch : batches) {
    auto ro      = batch->to_read_only();
    auto table   = sirius::get_cudf_table_view(ro);
    auto columns = std::make_unique<cudf::table>(table, stream, mr)->release();
    cudf::numeric_scalar<int64_t> start(total_rows, true, stream, mr);
    cudf::numeric_scalar<int64_t> step(1, true, stream, mr);
    columns.push_back(cudf::sequence(table.num_rows(), start, step, stream, mr));
    total_rows += table.num_rows();
    originals.push_back(std::make_unique<cudf::table>(std::move(columns)));
    table      = originals.back()->view();
    auto order = cudf::sorted_order(table.select(comparison.key_columns),
                                    comparison.directions,
                                    comparison.null_precedence,
                                    stream,
                                    mr);
    inputs.push_back(
      cudf::gather(table, order->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr));
    stream.sync();  // test input owner/lock retirement
  }
  for (std::size_t i = inputs.size(); i-- > 0;) {
    samples.push_back(sm::gather_samples(inputs[i]->view(), i, stride, comparison, stream, mr));
    sample_views.push_back(samples.back()->view());
  }
  auto splitters = sm::order_sample_splitters(sample_views, sample_step, comparison, stream, mr);
  std::vector<std::vector<cudf::size_type>> cuts(inputs.size());
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    auto device =
      sm::discover_cuts(inputs[i]->view(), i, splitters->view(), comparison, stream, mr);
    auto host = copy_column_to_host<cudf::size_type>(device->view(), stream);
    REQUIRE(host.size() == static_cast<std::size_t>(splitters->num_rows()));
    REQUIRE(std::is_sorted(host.begin(), host.end()));
    for (auto offset : host) {
      REQUIRE(offset >= 0);
      REQUIRE(offset <= inputs[i]->num_rows());
    }
    cuts[i].push_back(0);
    cuts[i].insert(cuts[i].end(), host.begin(), host.end());
    cuts[i].push_back(inputs[i]->num_rows());
    // Windowed searches preserve the same original row coordinates. Each
    // window has one bulk offset readback, never one key readback per splitter.
    for (cudf::size_type begin = 0; begin < splitters->num_rows(); begin += 3) {
      auto end         = std::min<cudf::size_type>(begin + 3, splitters->num_rows());
      auto window      = cudf::slice(splitters->view(), {begin, end}, stream)[0];
      auto window_cuts = sm::discover_cuts(inputs[i]->view(), i, window, comparison, stream, mr);
      CHECK(copy_column_to_host<cudf::size_type>(window_cuts->view(), stream) ==
            std::vector<cudf::size_type>(host.begin() + begin, host.begin() + end));
    }
  }
  // Exercise the n-ary library seam and owning piece extraction, without
  // claiming that this direct helper test qualifies task/storage integration.
  for (std::size_t t = 0; t <= static_cast<std::size_t>(splitters->num_rows()); ++t) {
    std::vector<std::unique_ptr<cudf::table>> pieces;
    std::vector<cudf::table_view> views;
    uint64_t row_count = 0;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      if (cuts[i][t] == cuts[i][t + 1]) { continue; }
      auto view = cudf::slice(inputs[i]->view(), {cuts[i][t], cuts[i][t + 1]}, stream)[0];
      pieces.push_back(std::make_unique<cudf::table>(view, stream, mr));
      views.push_back(pieces.back()->view());
      row_count += view.num_rows();
    }
    REQUIRE(row_count <= limit);
    if (views.empty()) { continue; }
    auto merged = cudf::merge(
      views, comparison.key_columns, comparison.directions, comparison.null_precedence, stream, mr);
    CHECK(merged->num_rows() == row_count);
    CHECK(cudf::is_sorted(merged->view().select(comparison.key_columns),
                          comparison.directions,
                          comparison.null_precedence,
                          stream));
    outputs.push_back(std::move(merged));
    stream.sync();
  }
  std::vector<cudf::table_view> output_views, original_views;
  for (auto const& output : outputs) {
    output_views.push_back(output->view());
  }
  for (auto const& original : originals) {
    original_views.push_back(original->view());
  }
  REQUIRE(!output_views.empty());
  auto actual   = cudf::concatenate(output_views, stream, mr);
  auto expected = cudf::concatenate(original_views, stream, mr);
  REQUIRE(actual->num_rows() == total_rows);
  // Check ordering across tile boundaries, not just inside each merge.
  CHECK(cudf::is_sorted(actual->view().select(comparison.key_columns),
                        comparison.directions,
                        comparison.null_precedence,
                        stream));
  auto id_column = actual->num_columns() - 1;
  auto order     = cudf::sorted_order(actual->view().select({id_column}), {}, {}, stream, mr);
  auto by_id =
    cudf::gather(actual->view(), order->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
  auto ids = copy_column_to_host<int64_t>(by_id->view().column(id_column), stream);
  for (int64_t row = 0; row < total_rows; ++row) {
    REQUIRE(ids[row] == row);
  }
  // Unique row IDs establish exact conservation. Compare every original column
  // in ID order to catch detached payloads without imposing an order on SQL peers.
  for (cudf::size_type col = 0; col < id_column; ++col) {
    auto a = by_id->view().column(col), e = expected->view().column(col);
    auto a_null = cudf::is_null(a, stream, mr);
    auto e_null = cudf::is_null(e, stream, mr);
    auto nulls  = copy_column_to_host<bool>(a_null->view(), stream);
    REQUIRE(nulls == copy_column_to_host<bool>(e_null->view(), stream));
    auto compare = [&]<typename T>() {
      stream.sync();
      auto av = copy_column_to_host<T>(a), ev = copy_column_to_host<T>(e);
      for (std::size_t row = 0; row < av.size(); ++row) {
        if (nulls[row]) { continue; }
        if constexpr (std::is_floating_point_v<T>) {
          CHECK((av[row] == ev[row] || (std::isnan(av[row]) && std::isnan(ev[row]))));
          if (av[row] == 0 && ev[row] == 0) {
            CHECK(std::signbit(av[row]) == std::signbit(ev[row]));
          }
        } else {
          CHECK(av[row] == ev[row]);
        }
      }
    };
    switch (a.type().id()) {
      case cudf::type_id::INT64: compare.template operator()<int64_t>(); break;
      case cudf::type_id::FLOAT64: compare.template operator()<double>(); break;
      case cudf::type_id::STRING: compare.template operator()<std::string>(); break;
      default: FAIL("Add a payload comparison for this test type");
    }
  }
}
}  // namespace

TEST_CASE("GPU merge tiles preserve global order and rows for unequal peer-heavy inputs",
          "[gpu_sampled_merge]")
{
  auto* space     = get_default_gpu_space();
  auto comparison = make_comparison({0}, {duckdb::OrderType::ASCENDING}, 1);
  std::mt19937_64 rng(90123);
  for (int trial = 0; trial < 20; ++trial) {
    std::vector<std::vector<int64_t>> keys(1 + trial % 7);
    std::vector<std::shared_ptr<cucascade::data_batch>> batches;
    for (auto& input : keys) {
      input.resize(17 + rng() % 41);
      for (auto& key : input) {
        key = trial % 2 ? 7 : rng() % 17;
      }
      batches.push_back(make_numeric_batch<int64_t>(*space, input, cudf::type_id::INT64));
      std::sort(input.begin(), input.end());
    }
    // Exercise full sampling and incomplete blocks independently of operator policy.
    verify_gpu_cuts(batches, comparison, 1 + trial % 4, 1 + trial % 5);
  }
}

TEST_CASE("GPU cut ties preserve null NaN and signed-zero SQL equivalence", "[gpu_sampled_merge]")
{
  auto* space     = get_default_gpu_space();
  auto comparison = make_comparison({0}, {duckdb::OrderType::ASCENDING}, 1);
  auto nan        = std::numeric_limits<double>::quiet_NaN();
  std::vector<std::vector<std::optional<double>>> keys{{nan, std::nullopt, -0.0, 0.0, 2.0, nan},
                                                       {nan, std::nullopt, 0.0, -1.0, 2.0},
                                                       {std::nullopt, nan, 0.0},
                                                       {}};
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  for (auto& input : keys) {
    std::vector<double> values;
    std::vector<bool> valid;
    for (auto const& key : input) {
      values.push_back(key.value_or(0.0));
      valid.push_back(key.has_value());
    }
    batches.push_back(
      make_numeric_batch_with_nulls<double>(*space, values, valid, cudf::type_id::FLOAT64));
  }
  verify_gpu_cuts(batches, comparison, 1, 3);
}

TEST_CASE("GPU samples select hidden string keys and use mixed key directions",
          "[gpu_sampled_merge]")
{
  auto* space = get_default_gpu_space();
  std::vector<std::vector<int64_t>> numbers{{2, 1, 1, 3, 2}, {2, 1, 2}, {1, 3, 1, 2}};
  std::vector<std::vector<std::string>> strings{
    {"bb", "z", "a", "", "bb"}, {"bb", "z", "aa"}, {"a", "", "z", "bb"}};
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  for (std::size_t i = 0; i < numbers.size(); ++i) {
    batches.push_back(make_two_column_batch<int64_t, std::string>(
      *space, numbers[i], strings[i], cudf::type_id::STRING));
  }
  auto hidden = make_comparison({1}, {duckdb::OrderType::DESCENDING}, 2);
  verify_gpu_cuts(batches, hidden, 1, 3);
  auto mixed =
    make_comparison({0, 1}, {duckdb::OrderType::ASCENDING, duckdb::OrderType::DESCENDING}, 2);
  verify_gpu_cuts(batches, mixed, 1, 3);
}

TEST_CASE("GPU samples retain 64-bit input IDs and omit incomplete blocks", "[gpu_sampled_merge]")
{
  auto* space     = get_default_gpu_space();
  auto comparison = make_comparison({0}, {duckdb::OrderType::ASCENDING}, 1);
  auto batch  = make_numeric_batch<int64_t>(*space, {0, 0, 0, 0, 0, 0, 0, 0}, cudf::type_id::INT64);
  auto ro     = batch->to_read_only();
  auto table  = sirius::get_cudf_table_view(ro);
  auto stream = default_stream();
  auto mr     = get_resource_ref(*space);
  auto ordinal = std::numeric_limits<uint64_t>::max() - 1;
  auto samples = sm::gather_samples(table, ordinal, 3, comparison, stream, mr);
  CHECK(copy_column_to_host<uint64_t>(samples->view().column(1), stream) ==
        std::vector<uint64_t>{ordinal, ordinal});
  CHECK(copy_column_to_host<uint64_t>(samples->view().column(2), stream) ==
        std::vector<uint64_t>{2, 5});
  auto splitters = sm::order_sample_splitters({samples->view()}, 1, comparison, stream, mr);
  for (auto id : {ordinal - 1, ordinal, ordinal + 1}) {
    auto cuts     = sm::discover_cuts(table, id, splitters->view(), comparison, stream, mr);
    auto expected = id < ordinal   ? std::vector<int32_t>{8, 8}
                    : id > ordinal ? std::vector<int32_t>{0, 0}
                                   : std::vector<int32_t>{3, 6};
    CHECK(copy_column_to_host<int32_t>(cuts->view(), stream) == expected);
  }
  auto empty = sm::gather_samples(table, ordinal, uint64_t{1} << 40, comparison, stream, mr);
  CHECK(empty->num_rows() == 0);
  auto no_splitters = sm::order_sample_splitters({empty->view()}, 10, comparison, stream, mr);
  auto no_cuts = sm::discover_cuts(table, ordinal, no_splitters->view(), comparison, stream, mr);
  CHECK(no_cuts->size() == 0);
  CHECK_THROWS_AS(sm::gather_samples(table, ordinal, 0, comparison, stream, mr),
                  std::invalid_argument);
  CHECK_THROWS_AS(sm::order_sample_splitters({samples->view()}, 0, comparison, stream, mr),
                  std::invalid_argument);
  stream.sync();
}

TEST_CASE("full-row samples fit the original reservation without a sorted-key copy",
          "[gpu_sampled_merge][sampled_merge_capacity_limit]")
{
  // Historical gate counterexample, unchanged row cap and reservation. Selecting
  // permutation entries first avoids the complete sorted-key allocation.
  auto* space = get_default_gpu_space();
  auto stream = default_stream();
  auto mr     = get_resource_ref(*space);
  auto* tracker =
    space->get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
  REQUIRE(tracker != nullptr);
  auto reservation_bytes = space->get_max_memory() * 3 / 4;
  REQUIRE(tracker->attach_reservation_to_tracker(
    stream,
    space->make_reservation(reservation_bytes),
    std::make_unique<cucascade::memory::fail_reservation_limit_policy>()));
  struct reservation_guard {
    cucascade::memory::reservation_aware_resource_adaptor* tracker;
    ::cuda::stream_ref stream;
    ~reservation_guard()
    {
      cudaStreamSynchronize(stream.get());
      tracker->reset_stream_reservation(stream);
    }
  } guard{tracker, stream};
  constexpr uint64_t k = 128, rows = 4096, width = 256, row_limit = 128;
  constexpr uint64_t stride = 1, sample_step = row_limit;
  auto comparison = make_comparison({0}, {duckdb::OrderType::ASCENDING}, 1);
  std::vector<std::unique_ptr<cudf::table>> samples;
  std::vector<cudf::table_view> views;
  uint64_t sample_bytes = 0;
  for (uint64_t i = 0; i < k; ++i) {
    auto batch = make_string_batch(*space, std::vector<std::string>(rows, std::string(width, 'x')));
    auto ro    = batch->to_read_only();
    samples.push_back(
      sm::gather_samples(sirius::get_cudf_table_view(ro), i, stride, comparison, stream, mr));
    views.push_back(samples.back()->view());
    sample_bytes += samples.back()->alloc_size();
    stream.sync();
  }
  tracker->reset_peak_allocated_bytes(stream);
  auto splitters = sm::order_sample_splitters(views, sample_step, comparison, stream, mr);
  std::vector<uint64_t> tile_rows(splitters->num_rows() + 1, 0);
  for (uint64_t i = 0; i < k; ++i) {
    auto batch = make_string_batch(*space, std::vector<std::string>(rows, std::string(width, 'x')));
    auto ro    = batch->to_read_only();
    auto cuts  = sm::discover_cuts(
      sirius::get_cudf_table_view(ro), i, splitters->view(), comparison, stream, mr);
    auto offsets = copy_column_to_host<cudf::size_type>(cuts->view(), stream);
    offsets.insert(offsets.begin(), 0);
    offsets.push_back(rows);
    REQUIRE(std::is_sorted(offsets.begin(), offsets.end()));
    REQUIRE(offsets.front() == 0);
    REQUIRE(offsets.back() == rows);
    for (std::size_t t = 0; t < tile_rows.size(); ++t) {
      tile_rows[t] += offsets[t + 1] - offsets[t];
    }
  }
  INFO("sampled_merge_capacity_limit stride="
       << stride << " fan_in=" << k << " source_rows=" << rows << " sample_rows=" << k * rows
       << " sample_bytes=" << sample_bytes << " row_limit=" << row_limit
       << " tile_key_bytes_at_most=" << row_limit * (width + 4) << " reservation_bytes="
       << reservation_bytes << " measured_order_peak=" << tracker->get_peak_allocated_bytes(stream)
       << " sample_order_admitted=true");
  CHECK(std::accumulate(tile_rows.begin(), tile_rows.end(), uint64_t{0}) == k * rows);
  CHECK(*std::max_element(tile_rows.begin(), tile_rows.end()) <= row_limit);
  stream.sync();
}
