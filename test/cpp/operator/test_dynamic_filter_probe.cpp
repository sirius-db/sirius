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

/**
 * @file test_dynamic_filter_probe.cpp
 * @brief Single-GPU probe-kernel semantics of the membership dynamic filters (IN-list,
 *        small IN-list, Bloom): heterogeneous integer probe carriers (no materialized cast),
 *        the optional prior keep-mask (dead rows skip the lookup), sentinel conservation, and
 *        the refusal of non-integer probe types.
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <op/dynamic_filter/sirius_dynamic_filter.hpp>

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

using sirius::op::sirius_dynamic_bloom_filter;
using sirius::op::sirius_dynamic_in_list_filter;
using sirius::op::sirius_dynamic_small_in_list_filter;

namespace {

constexpr int kDevice = 0;  // build == probe device: the source replica answers directly

template <typename T>
std::unique_ptr<cudf::column> make_values(std::vector<T> const& values,
                                          cudf::data_type type,
                                          rmm::cuda_stream_view stream)
{
  auto col       = cudf::make_numeric_column(type,
                                       static_cast<cudf::size_type>(values.size()),
                                       cudf::mask_state::UNALLOCATED,
                                       stream,
                                       cudf::get_current_device_resource_ref());
  auto const err = cudaMemcpyAsync(col->mutable_view().data<T>(),
                                   values.data(),
                                   values.size() * sizeof(T),
                                   cudaMemcpyHostToDevice,
                                   stream.value());
  REQUIRE(err == cudaSuccess);
  stream.synchronize();  // callers pass temporaries; these tests do not benchmark ingestion
  return col;
}

std::unique_ptr<cudf::column> make_int32(std::vector<std::int32_t> const& v,
                                         rmm::cuda_stream_view stream)
{
  return make_values(v, cudf::data_type{cudf::type_id::INT32}, stream);
}

std::unique_ptr<cudf::column> make_int64(std::vector<std::int64_t> const& v,
                                         rmm::cuda_stream_view stream)
{
  return make_values(v, cudf::data_type{cudf::type_id::INT64}, stream);
}

std::vector<std::uint8_t> mask_to_host(cudf::column_view const& mask, rmm::cuda_stream_view stream)
{
  REQUIRE(mask.type().id() == cudf::type_id::BOOL8);
  std::vector<std::uint8_t> host(static_cast<std::size_t>(mask.size()));
  auto const err = cudaMemcpyAsync(host.data(),
                                   mask.data<bool>(),
                                   host.size() * sizeof(bool),
                                   cudaMemcpyDeviceToHost,
                                   stream.value());
  REQUIRE(err == cudaSuccess);
  stream.synchronize();
  return host;
}

/// Upload a packed 1-bit/row keep-mask (bit row%32 of word row/32, 1 = keep) built from @p keep.
rmm::device_buffer upload_prior_mask(std::vector<bool> const& keep, rmm::cuda_stream_view stream)
{
  std::vector<std::uint32_t> words((keep.size() + 31) / 32, 0U);
  for (std::size_t row = 0; row < keep.size(); ++row) {
    if (keep[row]) { words[row / 32] |= (1U << (row % 32)); }
  }
  rmm::device_buffer out{words.data(), words.size() * sizeof(std::uint32_t), stream};
  stream.synchronize();
  return out;
}

}  // namespace

//===----------------------------------------------------------------------===//
// Heterogeneous probe carriers (the killed probe-key cast)
//===----------------------------------------------------------------------===//

TEST_CASE("IN-list over INT64 keys probes an INT32 carrier without a cast",
          "[dynamic_filter][probe]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  auto keys = make_int64({10, 20, 30, 40}, stream);
  sirius_dynamic_in_list_filter filter{keys->view(), stream, mr};
  REQUIRE(filter.has_persistent_set());

  auto probe = make_int32({10, 15, 20, -3, 40}, stream);
  auto mask  = filter.compute_mask(probe->view(), kDevice, stream, mr);
  REQUIRE(mask != nullptr);
  auto const host = mask_to_host(mask->view(), stream);
  CHECK(host == std::vector<std::uint8_t>{1, 0, 1, 0, 1});
}

TEST_CASE("IN-list over INT32 keys drops out-of-range INT64 probe values",
          "[dynamic_filter][probe]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  auto keys = make_int32({1, 2, 3}, stream);
  sirius_dynamic_in_list_filter filter{keys->view(), stream, mr};

  auto probe = make_int64({1, 5'000'000'000LL, 3, -5'000'000'000LL}, stream);
  auto mask  = filter.compute_mask(probe->view(), kDevice, stream, mr);
  REQUIRE(mask != nullptr);
  auto const host = mask_to_host(mask->view(), stream);
  CHECK(host == std::vector<std::uint8_t>{1, 0, 1, 0});
}

TEST_CASE("IN-list sentinel semantics under heterogeneous probes", "[dynamic_filter][probe]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  SECTION("an INT64 probe equal to the INT32 set's empty sentinel is kept conservatively")
  {
    auto keys = make_int32({7}, stream);
    sirius_dynamic_in_list_filter filter{keys->view(), stream, mr};
    auto probe = make_int64(
      {static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::min()), 7, 8}, stream);
    auto mask = filter.compute_mask(probe->view(), kDevice, stream, mr);
    REQUIRE(mask != nullptr);
    auto const host = mask_to_host(mask->view(), stream);
    CHECK(host == std::vector<std::uint8_t>{1, 1, 0});  // sentinel keep stays conservative
  }

  SECTION("an INT32 probe against an INT64 set is exact: no widened value hits the sentinel")
  {
    auto keys = make_int64({7}, stream);
    sirius_dynamic_in_list_filter filter{keys->view(), stream, mr};
    auto probe = make_int32({std::numeric_limits<std::int32_t>::min(), 7}, stream);
    auto mask  = filter.compute_mask(probe->view(), kDevice, stream, mr);
    REQUIRE(mask != nullptr);
    auto const host = mask_to_host(mask->view(), stream);
    CHECK(host == std::vector<std::uint8_t>{0, 1});  // INT32_MIN widened != INT64 sentinel
  }

  SECTION("the homogeneous sentinel keep is unchanged")
  {
    auto keys = make_int64({5}, stream);
    sirius_dynamic_in_list_filter filter{keys->view(), stream, mr};
    auto probe = make_int64({std::numeric_limits<std::int64_t>::min(), 5, 6}, stream);
    auto mask  = filter.compute_mask(probe->view(), kDevice, stream, mr);
    REQUIRE(mask != nullptr);
    auto const host = mask_to_host(mask->view(), stream);
    CHECK(host == std::vector<std::uint8_t>{1, 1, 0});
  }
}

TEST_CASE("small IN-list probes heterogeneous integer carriers", "[dynamic_filter][probe]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  SECTION("INT64 needles, INT32 probe")
  {
    auto keys = make_int64({7, 1'000'000'000'000LL}, stream);
    sirius_dynamic_small_in_list_filter filter{keys->view(), stream, mr};
    auto probe = make_int32({7, -7, 0}, stream);
    auto mask  = filter.compute_mask(probe->view(), kDevice, stream, mr);
    REQUIRE(mask != nullptr);
    auto const host = mask_to_host(mask->view(), stream);
    CHECK(host == std::vector<std::uint8_t>{1, 0, 0});
  }

  SECTION("INT32 needles, INT64 probe with out-of-range values")
  {
    auto keys = make_int32({5, 6}, stream);
    sirius_dynamic_small_in_list_filter filter{keys->view(), stream, mr};
    auto probe = make_int64({5, 6'000'000'000LL, 6}, stream);
    auto mask  = filter.compute_mask(probe->view(), kDevice, stream, mr);
    REQUIRE(mask != nullptr);
    auto const host = mask_to_host(mask->view(), stream);
    CHECK(host == std::vector<std::uint8_t>{1, 0, 1});
  }
}

TEST_CASE("Bloom filter has no false negatives across probe carriers", "[dynamic_filter][probe]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  SECTION("INT64 build keys, INT32 probe")
  {
    auto keys = make_int64({100, 200, 300, 9'000'000'000LL}, stream);
    sirius_dynamic_bloom_filter filter{keys->view(), stream, mr};
    auto probe = make_int32({100, 200, 300}, stream);
    auto mask  = filter.compute_mask(probe->view(), kDevice, stream, mr);
    REQUIRE(mask != nullptr);
    auto const host = mask_to_host(mask->view(), stream);
    CHECK(host == std::vector<std::uint8_t>{1, 1, 1});  // every inserted key must test positive
  }

  SECTION("INT32 build keys, INT64 probe")
  {
    auto keys = make_int32({100, 200, 300}, stream);
    sirius_dynamic_bloom_filter filter{keys->view(), stream, mr};
    auto probe = make_int64({100, 300}, stream);
    auto mask  = filter.compute_mask(probe->view(), kDevice, stream, mr);
    REQUIRE(mask != nullptr);
    auto const host = mask_to_host(mask->view(), stream);
    CHECK(host == std::vector<std::uint8_t>{1, 1});
  }
}

TEST_CASE("membership filters refuse non-integer probe carriers", "[dynamic_filter][probe]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  auto const n      = cudf::size_type{4};

  auto keys = make_int64({1, 2, 3}, stream);
  sirius_dynamic_in_list_filter in_list{keys->view(), stream, mr};
  sirius_dynamic_small_in_list_filter small_list{keys->view(), stream, mr};
  sirius_dynamic_bloom_filter bloom{keys->view(), stream, mr};

  auto const decimal = cudf::make_fixed_point_column(
    cudf::data_type{cudf::type_id::DECIMAL64, -2}, n, cudf::mask_state::UNALLOCATED, stream, mr);
  auto const date = cudf::make_timestamp_column(
    cudf::data_type{cudf::type_id::TIMESTAMP_DAYS}, n, cudf::mask_state::UNALLOCATED, stream, mr);
  auto const fp = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::FLOAT64}, n, cudf::mask_state::UNALLOCATED, stream, mr);

  for (auto const* probe : {&decimal, &date, &fp}) {
    CHECK(in_list.compute_mask((*probe)->view(), kDevice, stream, mr) == nullptr);
    CHECK(small_list.compute_mask((*probe)->view(), kDevice, stream, mr) == nullptr);
    CHECK(bloom.compute_mask((*probe)->view(), kDevice, stream, mr) == nullptr);
  }
}

//===----------------------------------------------------------------------===//
// Prior keep-mask (mask-aware probing)
//===----------------------------------------------------------------------===//

TEST_CASE("prior keep-mask gates the membership probes", "[dynamic_filter][probe]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  // > 32 rows so the packed-word indexing crosses word boundaries. Membership = even values;
  // prior mask keeps rows divisible by 3.
  constexpr std::size_t n = 70;
  std::vector<std::int64_t> key_values;
  std::vector<std::int32_t> probe_values(n);
  std::vector<bool> keep(n);
  std::vector<std::uint8_t> expected_unmasked(n);
  std::vector<std::uint8_t> expected_masked(n);
  for (std::size_t i = 0; i < n; ++i) {
    if (i % 2 == 0) { key_values.push_back(static_cast<std::int64_t>(i)); }
    probe_values[i]      = static_cast<std::int32_t>(i);
    keep[i]              = (i % 3 == 0);
    expected_unmasked[i] = (i % 2 == 0) ? 1 : 0;
    expected_masked[i]   = (i % 2 == 0 && i % 3 == 0) ? 1 : 0;
  }
  auto keys               = make_int64(key_values, stream);
  auto probe              = make_int32(probe_values, stream);
  auto prior              = upload_prior_mask(keep, stream);
  auto const* prior_words = static_cast<std::uint32_t const*>(prior.data());

  auto all_dead = upload_prior_mask(std::vector<bool>(n, false), stream);
  auto all_live = upload_prior_mask(std::vector<bool>(n, true), stream);

  SECTION("IN-list")
  {
    sirius_dynamic_in_list_filter filter{keys->view(), stream, mr};

    auto masked = filter.compute_mask(probe->view(), prior_words, kDevice, stream, mr);
    REQUIRE(masked != nullptr);
    CHECK(mask_to_host(masked->view(), stream) == expected_masked);

    auto dead = filter.compute_mask(
      probe->view(), static_cast<std::uint32_t const*>(all_dead.data()), kDevice, stream, mr);
    REQUIRE(dead != nullptr);
    CHECK(mask_to_host(dead->view(), stream) == std::vector<std::uint8_t>(n, 0));

    auto live = filter.compute_mask(
      probe->view(), static_cast<std::uint32_t const*>(all_live.data()), kDevice, stream, mr);
    REQUIRE(live != nullptr);
    CHECK(mask_to_host(live->view(), stream) == expected_unmasked);

    auto unmasked = filter.compute_mask(probe->view(), nullptr, kDevice, stream, mr);
    REQUIRE(unmasked != nullptr);
    CHECK(mask_to_host(unmasked->view(), stream) == expected_unmasked);
  }

  SECTION("small IN-list")
  {
    // Needles capped at k_max_keys: membership = {0, 6, 12} over the same probe.
    auto small_keys = make_int64({0, 6, 12}, stream);
    sirius_dynamic_small_in_list_filter filter{small_keys->view(), stream, mr};
    std::vector<std::uint8_t> expected(n, 0);
    for (auto const v : {0, 6, 12}) {
      expected[static_cast<std::size_t>(v)] = keep[static_cast<std::size_t>(v)] ? 1 : 0;
    }

    auto masked = filter.compute_mask(probe->view(), prior_words, kDevice, stream, mr);
    REQUIRE(masked != nullptr);
    CHECK(mask_to_host(masked->view(), stream) == expected);

    auto dead = filter.compute_mask(
      probe->view(), static_cast<std::uint32_t const*>(all_dead.data()), kDevice, stream, mr);
    REQUIRE(dead != nullptr);
    CHECK(mask_to_host(dead->view(), stream) == std::vector<std::uint8_t>(n, 0));
  }

  SECTION("Bloom")
  {
    sirius_dynamic_bloom_filter filter{keys->view(), stream, mr};

    auto masked = filter.compute_mask(probe->view(), prior_words, kDevice, stream, mr);
    REQUIRE(masked != nullptr);
    auto const host = mask_to_host(masked->view(), stream);
    for (std::size_t i = 0; i < n; ++i) {
      if (!keep[i]) {
        CHECK(host[i] == 0);  // dead rows never pass, whatever the filter says
      } else if (i % 2 == 0) {
        CHECK(host[i] == 1);  // live in-set rows must pass (no false negatives)
      }
    }

    auto dead = filter.compute_mask(
      probe->view(), static_cast<std::uint32_t const*>(all_dead.data()), kDevice, stream, mr);
    REQUIRE(dead != nullptr);
    CHECK(mask_to_host(dead->view(), stream) == std::vector<std::uint8_t>(n, 0));
  }
}

TEST_CASE("prior-masked probe still propagates the probe's null mask", "[dynamic_filter][probe]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  auto keys = make_int64({1, 2, 3, 4}, stream);
  sirius_dynamic_in_list_filter filter{keys->view(), stream, mr};

  auto probe     = make_int32({1, 2, 9, 4}, stream);
  auto null_mask = cudf::create_null_mask(4, cudf::mask_state::ALL_VALID, stream, mr);
  cudf::set_null_mask(static_cast<cudf::bitmask_type*>(null_mask.data()), 1, 2, false, stream);
  probe->set_null_mask(std::move(null_mask), 1);

  auto prior = upload_prior_mask({true, true, true, false}, stream);
  auto mask  = filter.compute_mask(
    probe->view(), static_cast<std::uint32_t const*>(prior.data()), kDevice, stream, mr);
  REQUIRE(mask != nullptr);
  CHECK(mask->null_count() == 1);
  auto const host = mask_to_host(mask->view(), stream);
  CHECK(host[0] == 1);  // live, in set
  CHECK(host[2] == 0);  // live, not in set
  CHECK(host[3] == 0);  // dead row
}
