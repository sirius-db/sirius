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

/*
 * Tests `exact_key_stager` and `bounds_accumulator` against single-batch oracles: keys staged from
 * any split of a build hold exactly the keys the IN-list column constructors prepare from the whole
 * build, and merged bounds equal one reduction over the whole build at the recorded type, across
 * key families and carriers. Neither accumulator waits on the host while contributing, except for
 * STRING bounds.
 */

#include "operator/dynamic_filter_accumulation_test_utils.hpp"
#include "operator/dynamic_filter_column_test_utils.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>
#include <helper/numeric_narrowing.hpp>
#include <op/dynamic_filter/detail/bounds_accumulator.hpp>
#include <op/dynamic_filter/detail/exact_key_stager.hpp>
#include <op/dynamic_filter/dynamic_filter_key_domain.hpp>
#include <op/dynamic_filter/prepared_build_keys.hpp>
#include <op/dynamic_filter/sirius_dynamic_filter.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <vector>

namespace {

namespace op  = sirius::op;
namespace acc = sirius::test::accumulation;
using op::detail::bounds_accumulator;
using op::detail::exact_key_stager;
using sirius::test::columns::families;
using sirius::test::columns::family;
using sirius::test::columns::k_rows;
using sirius::test::columns::make_column;
using sirius::test::columns::valid_row;
using sirius::test::columns::valid_rows;
using sirius::test::columns::values;

/// Uneven split bounds of a build of @p rows rows: @p splits pieces, one of them empty when 5.
std::vector<cudf::size_type> split_bounds(std::size_t splits, std::size_t rows)
{
  std::vector<cudf::size_type> bounds{0};
  if (splits == 2) { bounds.push_back(static_cast<cudf::size_type>(rows / 3)); }
  if (splits == 5) {
    bounds.insert(
      bounds.end(),
      {1, 1, static_cast<cudf::size_type>(rows / 2), static_cast<cudf::size_type>(rows / 2 + 1)});
  }
  bounds.push_back(static_cast<cudf::size_type>(rows));
  return bounds;
}

/// Valid rows of the test build in rows [begin, end).
std::size_t valid_between(cudf::size_type begin, cudf::size_type end)
{
  std::size_t count = 0;
  for (auto row = begin; row < end; ++row) {
    count += valid_row(static_cast<std::size_t>(row)) ? 1 : 0;
  }
  return count;
}

/// The bytes of @p column's rows at their element width.
std::vector<std::byte> bytes_of(cudf::column_view const& column, ::cuda::stream_ref stream)
{
  std::vector<std::byte> bytes(static_cast<std::size_t>(column.size()) *
                               cudf::size_of(column.type()));
  REQUIRE(cudaMemcpyAsync(bytes.data(),
                          column.head<std::byte>() + column.offset() * cudf::size_of(column.type()),
                          bytes.size(),
                          cudaMemcpyDeviceToHost,
                          stream.get()) == cudaSuccess);
  stream.sync();
  return bytes;
}

/// A scalar's value, as its storage bytes or its string, comparable across scalars of one type.
std::vector<std::byte> value_of(cudf::scalar const& bound, ::cuda::stream_ref stream)
{
  REQUIRE(bound.is_valid(stream));
  if (bound.type().id() == cudf::type_id::STRING) {
    auto const text = static_cast<cudf::string_scalar const&>(bound).to_string(stream);
    std::vector<std::byte> bytes(text.size());
    std::memcpy(bytes.data(), text.data(), text.size());
    return bytes;
  }
  auto const column = cudf::make_column_from_scalar(bound, 1, stream);
  return bytes_of(column->view(), stream);
}

/// @p bounds as device scalars, so they compare with the oracle's.
std::pair<std::unique_ptr<cudf::scalar>, std::unique_ptr<cudf::scalar>> scalars_of(
  bounds_accumulator::key_bounds const& bounds, ::cuda::stream_ref stream)
{
  CHECK(bounds.device_bytes() > 0);
  return bounds.to_scalars(stream, cudf::get_current_device_resource_ref());
}

/// The minimum and maximum one reduction over @p column gives.
std::pair<std::unique_ptr<cudf::scalar>, std::unique_ptr<cudf::scalar>> oracle_bounds(
  cudf::column_view const& column, ::cuda::stream_ref stream)
{
  return {
    cudf::reduce(
      column, *cudf::make_min_aggregation<cudf::reduce_aggregation>(), column.type(), stream),
    cudf::reduce(
      column, *cudf::make_max_aggregation<cudf::reduce_aggregation>(), column.type(), stream)};
}

void require_no_host_wait(::cuda::stream_ref stream, auto const& contribute)
{
  acc::stream_gate gate{stream};
  contribute();
  cudaEvent_t after{};
  REQUIRE(cudaEventCreateWithFlags(&after, cudaEventDisableTiming) == cudaSuccess);
  REQUIRE(cudaEventRecord(after, stream.get()) == cudaSuccess);
  CHECK(cudaEventQuery(after) == cudaErrorNotReady);
  gate.open();
  REQUIRE(cudaEventSynchronize(after) == cudaSuccess);
  REQUIRE(cudaEventDestroy(after) == cudaSuccess);
}

}  // namespace

//===----------------------------------------------------------------------===//
// exact_key_stager
//===----------------------------------------------------------------------===//

TEST_CASE("staged keys equal the column constructor's prepared keys for any split",
          "[dynamic_filter][construction][stager]")
{
  acc::fixture fixture;
  auto const stream = fixture.task_stream(0);
  auto const mr     = fixture.gpu(0).get_default_allocator();
  for (auto const& tested : families()) {
    for (std::size_t splits : {1, 2, 5}) {
      CAPTURE(tested.name, splits);
      auto const build    = tested.carrier(stream);
      auto const oracle   = tested.oracle(stream);
      auto const expected = op::prepare_build_column("[test]", oracle->view(), stream, mr);
      auto const valid    = valid_rows(k_rows);
      auto stager         = exact_key_stager::try_create(
        {{0, tested.construction, op::membership_filter_kind::hash_in_list, valid}},
        fixture.spaces);
      REQUIRE(stager);

      auto const bounds  = split_bounds(splits, k_rows);
      std::size_t offset = 0;
      for (std::size_t index = 0; index + 1 < bounds.size(); ++index) {
        auto const rows = valid_between(bounds[index], bounds[index + 1]);
        std::vector<exact_key_stager::slice> const slices{{offset, rows}};
        auto const slice = cudf::slice(build->view(), {bounds[index], bounds[index + 1]}).front();
        REQUIRE(stager->enqueue_add(
          slices, cudf::table_view{{slice}}, rmm::cuda_device_id{0}, stream, mr));
        offset += rows;
      }
      // An all-null batch stages nothing.
      std::vector<exact_key_stager::slice> const none{{offset, 0}};
      auto const all_null = cudf::slice(build->view(), {2, 3}).front();
      REQUIRE(stager->enqueue_add(
        none, cudf::table_view{{all_null}}, rmm::cuda_device_id{0}, stream, mr));
      stream.sync();

      rmm::cuda_set_device_raii guard{rmm::cuda_device_id{0}};
      auto const published =
        stager->finish(rmm::cuda_device_id{0}, fixture.publication_stream(0), mr);
      REQUIRE(published);
      CHECK(published->counts.filters_skipped_construction_invariant == 0);
      auto const staged = stager->inspect_buffer(0, rmm::cuda_device_id{0});
      REQUIRE(staged);
      CHECK(*staged == bytes_of(expected.keys->view(), stream));
      auto const* list =
        dynamic_cast<op::sirius_dynamic_in_list_filter const*>(published->filters.front().get());
      REQUIRE(list != nullptr);
      CHECK(list->size() == valid);
      CHECK(list->domain() == expected.domain);
    }
  }
}

TEST_CASE("staged keys build a small IN-list, and an all-null key the empty set",
          "[dynamic_filter][construction][stager]")
{
  acc::fixture fixture;
  auto const stream = fixture.task_stream(0);
  auto const mr     = fixture.gpu(0).get_default_allocator();
  auto const int64  = cudf::data_type{cudf::type_id::INT64};
  auto const keys =
    make_column(int64, values<std::int64_t>(10), stream, sirius::test::columns::validity(10));
  auto const nulls =
    make_column(int64, values<std::int64_t>(4), stream, std::vector<bool>(4, false));
  auto stager = exact_key_stager::try_create(
    {{0, int64, op::membership_filter_kind::small_in_list, valid_rows(10)},
     {1, int64, op::membership_filter_kind::hash_in_list, 0}},
    fixture.spaces);
  REQUIRE(stager);
  auto const first  = cudf::slice(keys->view(), {0, 4}).front();
  auto const second = cudf::slice(keys->view(), {4, 8}).front();
  std::vector<exact_key_stager::slice> const into_first{{0, valid_between(0, 4)}, {0, 0}};
  std::vector<exact_key_stager::slice> const into_second{{valid_between(0, 4), valid_between(4, 8)},
                                                         {0, 0}};
  REQUIRE(stager->enqueue_add(
    into_first, cudf::table_view{{first, nulls->view()}}, rmm::cuda_device_id{0}, stream, mr));
  REQUIRE(stager->enqueue_add(
    into_second, cudf::table_view{{second, nulls->view()}}, rmm::cuda_device_id{0}, stream, mr));
  auto const third = cudf::slice(keys->view(), {8, 10}).front();
  std::vector<exact_key_stager::slice> const into_third{{valid_between(0, 8), valid_between(8, 10)},
                                                        {0, 0}};
  auto const two_nulls = cudf::slice(nulls->view(), {0, 2}).front();
  REQUIRE(stager->enqueue_add(
    into_third, cudf::table_view{{third, two_nulls}}, rmm::cuda_device_id{0}, stream, mr));
  stream.sync();

  rmm::cuda_set_device_raii guard{rmm::cuda_device_id{0}};
  auto const published = stager->finish(rmm::cuda_device_id{0}, fixture.publication_stream(0), mr);
  REQUIRE(published);
  auto const expected = op::prepare_build_column("[test]", keys->view(), stream, mr);
  CHECK(*stager->inspect_buffer(0, rmm::cuda_device_id{0}) ==
        bytes_of(expected.keys->view(), stream));
  auto const* small =
    dynamic_cast<op::sirius_dynamic_small_in_list_filter const*>(published->filters[0].get());
  REQUIRE(small != nullptr);
  CHECK(small->size() == valid_rows(10));
  CHECK_FALSE(stager->inspect_buffer(1, rmm::cuda_device_id{0}));
  auto const* empty =
    dynamic_cast<op::sirius_dynamic_in_list_filter const*>(published->filters[1].get());
  REQUIRE(empty != nullptr);
  CHECK(empty->size() == 0);
}

TEST_CASE("a staged key with more valid rows than its slice is dropped",
          "[dynamic_filter][construction][stager]")
{
  acc::fixture fixture;
  auto const stream = fixture.task_stream(0);
  auto const mr     = fixture.gpu(0).get_default_allocator();
  auto const int32  = cudf::data_type{cudf::type_id::INT32};
  auto const keys   = make_column(int32, values<std::int32_t>(20), stream);
  auto stager =
    exact_key_stager::try_create({{0, int32, op::membership_filter_kind::hash_in_list, 20},
                                  {0, int32, op::membership_filter_kind::hash_in_list, 20}},
                                 fixture.spaces);
  REQUIRE(stager);
  // The second key's slice is one row short: that row is not written, and the key is dropped.
  std::vector<exact_key_stager::slice> const slices{{0, 20}, {0, 19}};
  REQUIRE(stager->enqueue_add(
    slices, cudf::table_view{{keys->view()}}, rmm::cuda_device_id{0}, stream, mr));
  stream.sync();
  rmm::cuda_set_device_raii guard{rmm::cuda_device_id{0}};
  auto const published = stager->finish(rmm::cuda_device_id{0}, fixture.publication_stream(0), mr);
  REQUIRE(published);
  CHECK(published->filters[0] != nullptr);
  CHECK(published->filters[1] == nullptr);
  CHECK(published->counts.filters_skipped_construction_invariant == 1);
  // The overflowing row stayed inside the slice: the last buffer row is still zero.
  auto const staged = stager->inspect_buffer(1, rmm::cuda_device_id{0});
  REQUIRE(staged);
  std::vector<std::byte> const zero(sizeof(std::int32_t), std::byte{0});
  CHECK(
    std::equal(zero.begin(), zero.end(), staged->end() - static_cast<std::ptrdiff_t>(zero.size())));
}

TEST_CASE("a staged contribution never waits on the host", "[dynamic_filter][construction][stager]")
{
  acc::fixture fixture;
  auto const stream = fixture.task_stream(0);
  rmm::mr::cuda_async_memory_resource async_resource;  // cudaFree would wait for the device.
  rmm::device_async_resource_ref const scratch{async_resource};
  for (auto const& tested : families()) {
    CAPTURE(tested.name);
    auto const build = tested.carrier(stream);
    auto const valid = valid_rows(k_rows);
    auto stager      = exact_key_stager::try_create(
      {{0, tested.construction, op::membership_filter_kind::hash_in_list, valid}}, fixture.spaces);
    REQUIRE(stager);
    std::vector<exact_key_stager::slice> const slices{{0, valid}};
    // A first contribution loads the kernels lazily; the second is gated.
    REQUIRE(stager->enqueue_add(
      slices, cudf::table_view{{build->view()}}, rmm::cuda_device_id{0}, stream, scratch));
    stream.sync();
    require_no_host_wait(stream, [&] {
      REQUIRE(stager->enqueue_add(
        slices, cudf::table_view{{build->view()}}, rmm::cuda_device_id{0}, stream, scratch));
    });
  }
}

//===----------------------------------------------------------------------===//
// bounds_accumulator
//===----------------------------------------------------------------------===//

TEST_CASE("merged bounds equal one reduction over the whole build for any split",
          "[dynamic_filter][construction][bounds]")
{
  acc::fixture fixture;
  auto const stream = fixture.task_stream(0);
  auto const mr     = fixture.gpu(0).get_default_allocator();
  for (auto const& tested : families()) {
    for (std::size_t splits : {1, 2, 5}) {
      CAPTURE(tested.name, splits);
      auto const build  = tested.carrier(stream);
      auto const oracle = tested.oracle(stream);
      auto const bounds = split_bounds(splits, k_rows);
      auto accumulator =
        bounds_accumulator::try_create({{0, tested.construction}}, bounds.size(), fixture.spaces);
      REQUIRE(accumulator);
      for (std::size_t index = 0; index + 1 < bounds.size(); ++index) {
        auto const slice = cudf::slice(build->view(), {bounds[index], bounds[index + 1]}).front();
        REQUIRE(accumulator->enqueue_add(
          index, cudf::table_view{{slice}}, rmm::cuda_device_id{0}, stream, mr));
      }
      // The last slot holds an all-null batch.
      auto const all_null = cudf::slice(build->view(), {2, 3}).front();
      REQUIRE(accumulator->enqueue_add(
        bounds.size() - 1, cudf::table_view{{all_null}}, rmm::cuda_device_id{0}, stream, mr));
      stream.sync();

      auto const merged = accumulator->finish(stream, mr);
      REQUIRE(merged.size() == 1);
      REQUIRE(merged.front());
      auto const [low, high]        = oracle_bounds(oracle->view(), stream);
      auto const [minimum, maximum] = scalars_of(*merged.front(), stream);
      CHECK(minimum->type() == tested.construction);
      CHECK(value_of(*minimum, stream) == value_of(*low, stream));
      CHECK(value_of(*maximum, stream) == value_of(*high, stream));
      CHECK(merged.front()->fits_membership_rep);
    }
  }
}

TEST_CASE("merged bounds restore each batch's carrier to the recorded type",
          "[dynamic_filter][construction][bounds]")
{
  acc::fixture fixture;
  auto const stream = fixture.task_stream(0);
  auto const mr     = fixture.gpu(0).get_default_allocator();
  using cudf::type_id;
  auto const recorded =
    GENERATE(cudf::data_type{type_id::INT64}, cudf::data_type{type_id::TIMESTAMP_DAYS});
  CAPTURE(static_cast<int>(recorded.id()));
  bool const date = recorded.id() == type_id::TIMESTAMP_DAYS;
  // Three batches at three carriers of the key: INT16, then INT32 (INT8 for a DATE), then native.
  std::vector<std::unique_ptr<cudf::column>> batches;
  batches.push_back(
    make_column(cudf::data_type{type_id::INT16}, std::vector<std::int16_t>{-300, 7, 25}, stream));
  batches.push_back(
    date ? make_column(cudf::data_type{type_id::INT8}, std::vector<std::int8_t>{-5, 100, 9}, stream)
         : make_column(
             cudf::data_type{type_id::INT32}, std::vector<std::int32_t>{-5, 100, 9}, stream));
  batches.push_back(date ? make_column(recorded, std::vector<std::int32_t>{40, 1, 20}, stream)
                         : make_column(recorded, std::vector<std::int64_t>{40, 1, 20}, stream));

  auto accumulator = bounds_accumulator::try_create({{0, recorded}}, 3, fixture.spaces);
  REQUIRE(accumulator);
  for (std::size_t batch = 0; batch < batches.size(); ++batch) {
    REQUIRE(accumulator->enqueue_add(
      batch, cudf::table_view{{batches[batch]->view()}}, rmm::cuda_device_id{0}, stream, mr));
  }
  stream.sync();
  auto const merged = accumulator->finish(stream, mr);
  REQUIRE(merged.front());
  auto const [minimum, maximum] = scalars_of(*merged.front(), stream);
  CHECK(minimum->type() == recorded);

  std::vector<std::unique_ptr<cudf::column>> restored;
  for (auto const& column : batches) {
    restored.push_back(sirius::cast_through_rep(column->view(), recorded, stream, mr));
  }
  auto const whole = cudf::concatenate(
    std::vector<cudf::column_view>{restored[0]->view(), restored[1]->view(), restored[2]->view()},
    stream);
  auto const [low, high] = oracle_bounds(whole->view(), stream);
  CHECK(value_of(*minimum, stream) == value_of(*low, stream));
  CHECK(value_of(*maximum, stream) == value_of(*high, stream));
}

TEST_CASE("DECIMAL128 bounds report whether the values fit the int64 rep",
          "[dynamic_filter][construction][bounds]")
{
  acc::fixture fixture;
  auto const stream  = fixture.task_stream(0);
  auto const mr      = fixture.gpu(0).get_default_allocator();
  auto const decimal = cudf::data_type{cudf::type_id::DECIMAL128, -2};
  auto const large   = GENERATE(false, true);
  CAPTURE(large);
  std::vector<__int128_t> values{-7, 12, 3};
  if (large) { values.push_back(__int128_t{1} << 70); }
  auto const column = make_column(decimal, values, stream);
  auto accumulator  = bounds_accumulator::try_create({{0, decimal}}, 1, fixture.spaces);
  REQUIRE(accumulator);
  REQUIRE(accumulator->enqueue_add(
    0, cudf::table_view{{column->view()}}, rmm::cuda_device_id{0}, stream, mr));
  stream.sync();
  auto const merged = accumulator->finish(stream, mr);
  REQUIRE(merged.front());
  CHECK(merged.front()->fits_membership_rep == !large);
  auto const [low, high] = oracle_bounds(column->view(), stream);
  CHECK(value_of(*scalars_of(*merged.front(), stream).second, stream) == value_of(*high, stream));
}

TEST_CASE("a key without valid rows has no bounds", "[dynamic_filter][construction][bounds]")
{
  acc::fixture fixture;
  auto const stream = fixture.task_stream(0);
  auto const mr     = fixture.gpu(0).get_default_allocator();
  auto const int32  = cudf::data_type{cudf::type_id::INT32};
  auto const string = cudf::data_type{cudf::type_id::STRING};
  auto const nulls =
    make_column(int32, std::vector<std::int32_t>{1, 2}, stream, std::vector<bool>(2, false));
  auto const strings = sirius::test::columns::string_family().carrier(stream);
  auto const empty   = cudf::slice(strings->view(), {0, 0}).front();
  auto accumulator   = bounds_accumulator::try_create({{0, int32}, {1, string}}, 2, fixture.spaces);
  REQUIRE(accumulator);
  REQUIRE(accumulator->enqueue_add(
    0,
    cudf::table_view{{nulls->view(), cudf::slice(strings->view(), {2, 4}).front()}},
    rmm::cuda_device_id{0},
    stream,
    mr));
  REQUIRE(
    accumulator->enqueue_add(1,
                             cudf::table_view{{cudf::slice(nulls->view(), {0, 0}).front(), empty}},
                             rmm::cuda_device_id{0},
                             stream,
                             mr));
  stream.sync();
  auto const merged = accumulator->finish(stream, mr);
  CHECK_FALSE(merged[0]);
  REQUIRE(merged[1]);  // Rows 2 and 3 of the STRING build: row 2 is null, row 3 is valid.
  auto const [minimum, maximum] = scalars_of(*merged[1], stream);
  CHECK(value_of(*minimum, stream) == value_of(*maximum, stream));
}

TEST_CASE("a numeric bounds contribution never waits on the host",
          "[dynamic_filter][construction][bounds]")
{
  acc::fixture fixture;
  auto const stream = fixture.task_stream(0);
  rmm::mr::cuda_async_memory_resource async_resource;  // cudaFree would wait for the device.
  rmm::device_async_resource_ref const scratch{async_resource};
  for (auto const& tested : families()) {
    if (tested.construction.id() == cudf::type_id::STRING) { continue; }  // The declared wait.
    CAPTURE(tested.name);
    auto const build = tested.carrier(stream);
    auto accumulator =
      bounds_accumulator::try_create({{0, tested.construction}}, 2, fixture.spaces);
    REQUIRE(accumulator);
    // A first contribution loads the kernels lazily; the second is gated.
    REQUIRE(accumulator->enqueue_add(
      0, cudf::table_view{{build->view()}}, rmm::cuda_device_id{0}, stream, scratch));
    stream.sync();
    require_no_host_wait(stream, [&] {
      REQUIRE(accumulator->enqueue_add(
        1, cudf::table_view{{build->view()}}, rmm::cuda_device_id{0}, stream, scratch));
    });
  }
}
