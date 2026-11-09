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
 * Tests `prepare_build_keys` and the membership filters' construction from prepared keys: the
 * prepared column holds exactly the valid rows at the rep (in order, widened, fingerprinted),
 * writes stay inside their output with an overflow word, nothing waits on the host, and a filter
 * built from prepared keys answers every probe like the filter built from the column.
 */

#include "operator/dynamic_filter_accumulation_test_utils.hpp"
#include "operator/operator_test_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/hashing.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>
#include <op/dynamic_filter/dynamic_filter_key_domain.hpp>
#include <op/dynamic_filter/prepared_build_keys.hpp>
#include <op/dynamic_filter/sirius_dynamic_filter.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

namespace {

namespace op  = sirius::op;
namespace acc = sirius::test::accumulation;

/// A column of @p type holding @p values bit-for-bit, with row `i` null iff `valid[i]` is false.
template <class T>
std::unique_ptr<cudf::column> make_column(cudf::data_type type,
                                          std::vector<T> const& values,
                                          ::cuda::stream_ref stream,
                                          std::vector<bool> const& valid = {})
{
  auto column = cudf::make_fixed_width_column(
    type, static_cast<cudf::size_type>(values.size()), cudf::mask_state::UNALLOCATED, stream);
  REQUIRE(cudaMemcpyAsync(column->mutable_view().head(),
                          values.data(),
                          values.size() * sizeof(T),
                          cudaMemcpyHostToDevice,
                          stream.get()) == cudaSuccess);
  if (!valid.empty()) {
    auto mask = cudf::create_null_mask(column->size(), cudf::mask_state::ALL_VALID, stream);
    cudf::size_type nulls = 0;
    for (std::size_t row = 0; row < valid.size(); ++row) {
      if (valid[row]) { continue; }
      cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask.data()),
                          static_cast<cudf::size_type>(row),
                          static_cast<cudf::size_type>(row + 1),
                          false,
                          stream);
      ++nulls;
    }
    column->set_null_mask(std::move(mask), nulls);
  }
  stream.sync();
  return column;
}

template <class T>
std::vector<T> to_host(cudf::column_view const& column, ::cuda::stream_ref stream)
{
  std::vector<T> values(static_cast<std::size_t>(column.size()));
  REQUIRE(cudaMemcpyAsync(values.data(),
                          column.head<std::byte>() + column.offset() * sizeof(T),
                          values.size() * sizeof(T),
                          cudaMemcpyDeviceToHost,
                          stream.get()) == cudaSuccess);
  stream.sync();
  return values;
}

/// Prepares @p keys at the domain of @p domain_type into an exactly sized column.
std::unique_ptr<cudf::column> prepare(cudf::data_type domain_type,
                                      cudf::column_view const& keys,
                                      ::cuda::stream_ref stream)
{
  auto const domain = op::classify_membership_key(domain_type);
  REQUIRE(domain);
  auto out = cudf::make_fixed_width_column(op::membership_rep_type(*domain),
                                           keys.size() - keys.null_count(),
                                           cudf::mask_state::UNALLOCATED,
                                           stream);
  op::prepare_build_keys(
    *domain, keys, out->mutable_view(), nullptr, stream, cudf::get_current_device_resource_ref());
  return out;
}

std::vector<bool> const k_valid{true, false, true, true, false, true, true};

}  // namespace

TEST_CASE("prepared keys are the valid rows in order at the rep", "[dynamic_filter][prepared_keys]")
{
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  using cudf::type_id;

  SECTION("an INT32 key with nulls")
  {
    auto const keys = make_column<std::int32_t>(
      cudf::data_type{type_id::INT32}, {7, 99, -3, 12, 99, 0, 7}, stream, k_valid);
    CHECK(to_host<std::int32_t>(prepare(keys->type(), keys->view(), stream)->view(), stream) ==
          std::vector<std::int32_t>{7, -3, 12, 0, 7});
  }
  SECTION("an INT16 carrier of an INT64 key widens per element")
  {
    auto const keys = make_column<std::int16_t>(
      cudf::data_type{type_id::INT16}, {-32768, 1, 2, 32767, 5, 6, 7}, stream, k_valid);
    CHECK(to_host<std::int64_t>(
            prepare(cudf::data_type{type_id::INT64}, keys->view(), stream)->view(), stream) ==
          std::vector<std::int64_t>{-32768, 2, 32767, 6, 7});
  }
  SECTION("an INT16 carrier of a DATE key reads epoch days")
  {
    auto const keys = make_column<std::int16_t>(
      cudf::data_type{type_id::INT16}, {19000, 1, -20, 3, 4, 5, 6}, stream, k_valid);
    CHECK(to_host<std::int32_t>(
            prepare(cudf::data_type{type_id::TIMESTAMP_DAYS}, keys->view(), stream)->view(),
            stream) == std::vector<std::int32_t>{19000, -20, 3, 5, 6});
  }
  SECTION("a DECIMAL32 carrier of a DECIMAL64 key keeps its unscaled values")
  {
    auto const keys = make_column<std::int32_t>(
      cudf::data_type{type_id::DECIMAL32, -2}, {150, 1, 275, -5, 0, 9, 10}, stream, k_valid);
    auto const prepared = prepare(cudf::data_type{type_id::DECIMAL64, -2}, keys->view(), stream);
    CHECK(prepared->type() == cudf::data_type(type_id::DECIMAL64, -2));
    CHECK(to_host<std::int64_t>(prepared->view(), stream) ==
          std::vector<std::int64_t>{150, 275, -5, 9, 10});
  }
  SECTION("a DECIMAL128 key whose values fit narrows to the int64 rep")
  {
    auto const keys = make_column<__int128_t>(cudf::data_type{type_id::DECIMAL128, -2},
                                              {__int128_t{1} << 40, 1, -7, 8, 9, 10, 11},
                                              stream,
                                              k_valid);
    CHECK(to_host<std::int64_t>(prepare(keys->type(), keys->view(), stream)->view(), stream) ==
          std::vector<std::int64_t>{std::int64_t{1} << 40, -7, 8, 10, 11});
  }
  SECTION("a sliced column reads from its offset")
  {
    auto const keys = make_column<std::int32_t>(
      cudf::data_type{type_id::INT32}, {7, 99, -3, 12, 99, 0, 7}, stream, k_valid);
    auto const slice = cudf::slice(keys->view(), {2, 7}).front();  // -3, 12, null, 0, 7
    CHECK(to_host<std::int32_t>(prepare(keys->type(), slice, stream)->view(), stream) ==
          std::vector<std::int32_t>{-3, 12, 0, 7});
  }
  SECTION("a column without nulls is copied whole")
  {
    auto const keys = make_column<std::int64_t>(cudf::data_type{type_id::INT64}, {4, 3, 2}, stream);
    CHECK(to_host<std::int64_t>(prepare(keys->type(), keys->view(), stream)->view(), stream) ==
          std::vector<std::int64_t>{4, 3, 2});
  }
}

TEST_CASE("prepared STRING keys are the probes' fingerprints of the valid rows",
          "[dynamic_filter][prepared_keys]")
{
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  auto const integers             = make_column<std::int32_t>(
    cudf::data_type{cudf::type_id::INT32}, {7, 99, -3, 12, 99, 0, 7}, stream, k_valid);
  auto const strings = cudf::strings::from_integers(integers->view(), stream);
  auto const sliced  = GENERATE(false, true);
  CAPTURE(sliced);
  auto const column   = sliced ? cudf::slice(strings->view(), {2, 7}).front() : strings->view();
  auto const prepared = prepare(column.type(), column, stream);
  auto const valid    = cudf::drop_nulls(cudf::table_view{{column}}, {0}, stream);
  auto const expected = cudf::hashing::xxhash_64(valid->view(), cudf::DEFAULT_HASH_SEED, stream);
  CHECK(prepared->type() == cudf::data_type{cudf::type_id::UINT64});
  CHECK(to_host<std::uint64_t>(prepared->view(), stream) ==
        to_host<std::uint64_t>(expected->view(), stream));
}

TEST_CASE("prepared writes stay inside their output and flag the overflow",
          "[dynamic_filter][prepared_keys]")
{
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  auto const mr                   = cudf::get_current_device_resource_ref();
  auto const domain   = *op::classify_membership_key(cudf::data_type{cudf::type_id::INT32});
  auto const nullable = GENERATE(true, false);
  // Six valid keys either way: seven rows with one null, or six rows without nulls.
  auto const keys     = nullable
                          ? make_column<std::int32_t>(cudf::data_type{cudf::type_id::INT32},
                                                      {1, 2, 3, 4, 5, 6, 7},
                                                  stream,
                                                      {true, true, false, true, true, true, true})
                          : make_column<std::int32_t>(
                          cudf::data_type{cudf::type_id::INT32}, {1, 2, 4, 5, 6, 7}, stream);
  auto const capacity = GENERATE(cudf::size_type{4}, cudf::size_type{6});
  CAPTURE(nullable, capacity);

  // An eight-slot buffer of sentinels; the output view covers its first `capacity` slots.
  constexpr std::int32_t sentinel = -1;
  auto buffer                     = make_column<std::int32_t>(
    cudf::data_type{cudf::type_id::INT32}, std::vector<std::int32_t>(8, sentinel), stream);
  cudf::mutable_column_view const out{
    cudf::data_type{cudf::type_id::INT32}, capacity, buffer->mutable_view().head(), nullptr, 0};
  rmm::device_buffer overflow{sizeof(std::uint32_t), stream, mr};
  REQUIRE(cudaMemsetAsync(overflow.data(), 0, overflow.size(), stream.get()) == cudaSuccess);

  op::prepare_build_keys(
    domain, keys->view(), out, static_cast<std::uint32_t*>(overflow.data()), stream, mr);
  std::uint32_t flagged = 0;
  REQUIRE(cudaMemcpyAsync(
            &flagged, overflow.data(), sizeof(flagged), cudaMemcpyDeviceToHost, stream.get()) ==
          cudaSuccess);
  auto const written = to_host<std::int32_t>(buffer->view(), stream);

  std::vector<std::int32_t> expected{1, 2, 4, 5, 6, 7, sentinel, sentinel};
  for (auto slot = static_cast<std::size_t>(capacity); slot < expected.size(); ++slot) {
    expected[slot] = sentinel;
  }
  CHECK(written == expected);
  CHECK((flagged != 0) == (capacity < 6));
}

TEST_CASE("preparing keys never waits on the host", "[dynamic_filter][prepared_keys]")
{
  rmm::cuda_stream owner{rmm::cuda_stream::flags::non_blocking};
  ::cuda::stream_ref const stream{owner.value()};
  // A stream-ordered resource: cudaFree would itself wait for the device.
  rmm::mr::cuda_async_memory_resource async_resource;
  rmm::device_async_resource_ref const mr{async_resource};
  auto const integers = make_column<std::int32_t>(
    cudf::data_type{cudf::type_id::INT32}, {7, 99, -3, 12, 99, 0, 7}, stream, k_valid);
  auto const strings = cudf::strings::from_integers(integers->view(), stream);
  stream.sync();
  auto const keys = GENERATE(0, 1);
  CAPTURE(keys);
  auto const& column = keys == 0 ? *integers : *strings;
  auto const domain  = *op::classify_membership_key(column.type());
  auto out           = cudf::make_fixed_width_column(
    op::membership_rep_type(domain), 5, cudf::mask_state::UNALLOCATED, stream, mr);
  // An ungated run first: a kernel's first, lazy load would otherwise wait behind the gate.
  op::prepare_build_keys(domain, column.view(), out->mutable_view(), nullptr, stream, mr);
  stream.sync();

  acc::stream_gate gate{stream};
  op::prepare_build_keys(domain, column.view(), out->mutable_view(), nullptr, stream, mr);
  cudaEvent_t after{};
  REQUIRE(cudaEventCreateWithFlags(&after, cudaEventDisableTiming) == cudaSuccess);
  REQUIRE(cudaEventRecord(after, stream.get()) == cudaSuccess);
  // prepare_build_keys returned while its work is still queued behind the gate.
  CHECK(cudaEventQuery(after) == cudaErrorNotReady);
  gate.open();
  REQUIRE(cudaEventSynchronize(after) == cudaSuccess);
  REQUIRE(cudaEventDestroy(after) == cudaSuccess);
}

TEST_CASE("filters built from prepared keys answer like filters built from the column",
          "[dynamic_filter][prepared_keys]")
{
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  auto const mr                   = cudf::get_current_device_resource_ref();
  int device                      = -1;
  REQUIRE(cudaGetDevice(&device) == cudaSuccess);

  auto const rows = GENERATE(std::size_t{7}, std::size_t{3000});
  std::vector<std::int16_t> values;
  std::vector<bool> valid;
  for (std::size_t row = 0; row < rows; ++row) {
    values.push_back(static_cast<std::int16_t>(row * 3));
    valid.push_back(row % 4 != 1);
  }
  // A narrowed INT16 carrier of the key, with nulls.
  auto const keys =
    make_column<std::int16_t>(cudf::data_type{cudf::type_id::INT16}, values, stream, valid);
  auto const domain   = *op::classify_membership_key(keys->type());
  auto const prepared = prepare(keys->type(), keys->view(), stream);

  // Members, non-members, values beyond INT16, and NULL probes.
  std::vector<std::int64_t> probe_values;
  std::vector<bool> probe_valid;
  for (std::int64_t value = -5; value < static_cast<std::int64_t>(rows * 3) + 40'000; value += 1) {
    probe_values.push_back(value);
    probe_valid.push_back(value % 97 != 0);
  }
  auto const probe = make_column<std::int64_t>(
    cudf::data_type{cudf::type_id::INT64}, probe_values, stream, probe_valid);
  auto const mask_of = [&](op::sirius_mask_applicable const& filter) {
    auto const mask = filter.compute_mask(probe->view(), device, stream, mr);
    REQUIRE(mask);
    return sirius::test::operator_utils::copy_column_to_host<bool>(mask->view(), stream);
  };

  op::sirius_dynamic_bloom_filter const bloom_column{keys->view(), stream, mr};
  op::sirius_dynamic_bloom_filter const bloom_prepared{domain, prepared->view(), stream, mr};
  CHECK(bloom_prepared.domain() == bloom_column.domain());
  CHECK(mask_of(bloom_prepared) == mask_of(bloom_column));

  op::sirius_dynamic_in_list_filter const set_column{keys->view(), stream, mr};
  op::sirius_dynamic_in_list_filter const set_prepared{domain, prepared->view(), stream, mr};
  CHECK(set_prepared.domain() == set_column.domain());
  CHECK(set_prepared.size() == set_column.size());
  CHECK(mask_of(set_prepared) == mask_of(set_column));

  if (prepared->size() <=
      static_cast<cudf::size_type>(op::sirius_dynamic_small_in_list_filter::k_max_keys)) {
    op::sirius_dynamic_small_in_list_filter const small_column{keys->view(), stream, mr};
    op::sirius_dynamic_small_in_list_filter const small_prepared{
      domain, prepared->view(), stream, mr};
    CHECK(small_prepared.domain() == small_column.domain());
    CHECK(small_prepared.size() == small_column.size());
    CHECK(mask_of(small_prepared) == mask_of(small_column));
  }
}

TEST_CASE("construction from prepared keys rejects anything but a non-null rep column",
          "[dynamic_filter][prepared_keys]")
{
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  auto const mr                   = cudf::get_current_device_resource_ref();
  auto const domain = *op::classify_membership_key(cudf::data_type{cudf::type_id::INT64});
  auto const narrow =
    make_column<std::int32_t>(cudf::data_type{cudf::type_id::INT32}, {1, 2}, stream);
  auto const nullable =
    make_column<std::int64_t>(cudf::data_type{cudf::type_id::INT64}, {1, 2}, stream, {true, false});
  for (auto const* prepared : {narrow.get(), nullable.get()}) {
    CHECK_THROWS_AS((op::sirius_dynamic_in_list_filter{domain, prepared->view(), stream, mr}),
                    std::invalid_argument);
    CHECK_THROWS_AS((op::sirius_dynamic_small_in_list_filter{domain, prepared->view(), stream, mr}),
                    std::invalid_argument);
    CHECK_THROWS_AS((op::sirius_dynamic_bloom_filter{domain, prepared->view(), stream, mr}),
                    std::invalid_argument);
  }
  auto const empty = make_column<std::int64_t>(cudf::data_type{cudf::type_id::INT64}, {}, stream);
  CHECK_THROWS_AS((op::sirius_dynamic_small_in_list_filter{domain, empty->view(), stream, mr}),
                  std::invalid_argument);
  CHECK_THROWS_AS(
    (op::prepare_build_keys(domain, narrow->view(), narrow->mutable_view(), nullptr, stream, mr)),
    std::invalid_argument);
}
