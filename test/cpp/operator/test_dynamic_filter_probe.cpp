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
 *        the optional prior keep-mask (dead rows skip the lookup), sentinel conservation, the
 *        refusal of non-integer probe types, INT8/INT16 and unsigned keys, carrier-typed build
 *        sets, and cross-carrier mask identity.
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <op/dynamic_filter/dynamic_filter_key_domain.hpp>
#include <op/dynamic_filter/sirius_dynamic_filter.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

using sirius::op::classify_membership_key;
using sirius::op::membership_key_domain;
using sirius::op::membership_key_family;
using sirius::op::membership_key_rep;
using sirius::op::membership_key_supported;
using sirius::op::membership_probe_compatible;
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

  // An unsigned carrier is a semantic mismatch for a signed set too: the planner's casts make a
  // signed<->unsigned key pair unreachable, so it declines rather than reinterpreting bits.
  auto const unsigned_probe = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::UINT32}, n, cudf::mask_state::UNALLOCATED, stream, mr);

  for (auto const* probe : {&decimal, &date, &fp, &unsigned_probe}) {
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

//===----------------------------------------------------------------------===//
// Key reps, carriers, and cross-carrier mask identity
//===----------------------------------------------------------------------===//

namespace {

template <typename T>
std::unique_ptr<cudf::column> make_typed(std::vector<std::int64_t> const& values,
                                         rmm::cuda_stream_view stream)
{
  std::vector<T> typed;
  typed.reserve(values.size());
  for (auto const v : values) {
    typed.push_back(static_cast<T>(v));
  }
  return make_values(typed, cudf::data_type{cudf::type_to_id<T>()}, stream);
}

template <typename T>
std::unique_ptr<cudf::column> make_unsigned(std::vector<std::uint64_t> const& values,
                                            rmm::cuda_stream_view stream)
{
  std::vector<T> typed;
  typed.reserve(values.size());
  for (auto const v : values) {
    typed.push_back(static_cast<T>(v));
  }
  return make_values(typed, cudf::data_type{cudf::type_to_id<T>()}, stream);
}

/// Calls fn(T{}) for each signed integer carrier.
template <class Fn>
void for_each_signed_carrier(Fn&& fn)
{
  fn(std::int8_t{});
  fn(std::int16_t{});
  fn(std::int32_t{});
  fn(std::int64_t{});
}

/// Calls fn(T{}) for each unsigned integer carrier.
template <class Fn>
void for_each_unsigned_carrier(Fn&& fn)
{
  fn(std::uint8_t{});
  fn(std::uint16_t{});
  fn(std::uint32_t{});
  fn(std::uint64_t{});
}

std::vector<std::uint8_t> and_with(std::vector<std::uint8_t> mask, std::vector<bool> const& keep)
{
  for (std::size_t i = 0; i < mask.size(); ++i) {
    mask[i] = (mask[i] != 0 && keep[i]) ? 1 : 0;
  }
  return mask;
}

template <class Filter>
std::vector<std::uint8_t> probe_mask(Filter const& filter,
                                     cudf::column_view const& probe,
                                     std::uint32_t const* prior_words,
                                     rmm::cuda_stream_view stream)
{
  auto mask = filter.compute_mask(
    probe, prior_words, kDevice, stream, cudf::get_current_device_resource_ref());
  REQUIRE(mask != nullptr);
  REQUIRE(mask->null_count() == 0);
  return mask_to_host(mask->view(), stream);
}

}  // namespace

TEST_CASE("membership key domain classifies integer types onto four reps",
          "[dynamic_filter][key_domain]")
{
  using id          = cudf::type_id;
  auto const expect = [](id t, membership_key_rep rep, membership_key_family family) {
    auto const domain = classify_membership_key(cudf::data_type{t});
    REQUIRE(domain.has_value());
    CHECK(domain->rep == rep);
    CHECK(domain->family == family);
    CHECK(domain->native == cudf::data_type{t});
    CHECK(membership_key_supported(cudf::data_type{t}));
  };
  expect(id::INT8, membership_key_rep::i32, membership_key_family::signed_int);
  expect(id::INT16, membership_key_rep::i32, membership_key_family::signed_int);
  expect(id::INT32, membership_key_rep::i32, membership_key_family::signed_int);
  expect(id::INT64, membership_key_rep::i64, membership_key_family::signed_int);
  expect(id::UINT8, membership_key_rep::u32, membership_key_family::unsigned_int);
  expect(id::UINT16, membership_key_rep::u32, membership_key_family::unsigned_int);
  expect(id::UINT32, membership_key_rep::u32, membership_key_family::unsigned_int);
  expect(id::UINT64, membership_key_rep::u64, membership_key_family::unsigned_int);

  for (auto const t : {id::EMPTY,
                       id::BOOL8,
                       id::FLOAT32,
                       id::FLOAT64,
                       id::TIMESTAMP_DAYS,
                       id::TIMESTAMP_MICROSECONDS,
                       id::DURATION_SECONDS,
                       id::DECIMAL32,
                       id::DECIMAL64,
                       id::DECIMAL128,
                       id::STRING,
                       id::DICTIONARY32,
                       id::LIST,
                       id::STRUCT}) {
    CHECK_FALSE(classify_membership_key(cudf::data_type{t}).has_value());
    CHECK_FALSE(membership_key_supported(cudf::data_type{t}));
  }

  // The host mirror of the probe dispatch: same-signedness integer carriers only.
  auto const signed_domain   = *classify_membership_key(cudf::data_type{id::INT64});
  auto const unsigned_domain = *classify_membership_key(cudf::data_type{id::UINT32});
  for (auto const t : {id::INT8, id::INT16, id::INT32, id::INT64}) {
    CHECK(membership_probe_compatible(signed_domain, cudf::data_type{t}));
    CHECK_FALSE(membership_probe_compatible(unsigned_domain, cudf::data_type{t}));
  }
  for (auto const t : {id::UINT8, id::UINT16, id::UINT32, id::UINT64}) {
    CHECK(membership_probe_compatible(unsigned_domain, cudf::data_type{t}));
    CHECK_FALSE(membership_probe_compatible(signed_domain, cudf::data_type{t}));
  }
  for (auto const t : {id::FLOAT64, id::TIMESTAMP_DAYS, id::DECIMAL64, id::STRING, id::BOOL8}) {
    CHECK_FALSE(membership_probe_compatible(signed_domain, cudf::data_type{t}));
    CHECK_FALSE(membership_probe_compatible(unsigned_domain, cudf::data_type{t}));
  }

  // The three filters' type gates are the same predicate.
  for (auto const t : {id::INT8, id::INT64, id::UINT16, id::UINT64, id::FLOAT64, id::STRING}) {
    CHECK(sirius_dynamic_bloom_filter::supports(cudf::data_type{t}) ==
          membership_key_supported(cudf::data_type{t}));
  }
}

TEST_CASE("hash IN-list set bytes are sized at the key rep, not the build carrier",
          "[dynamic_filter][key_domain]")
{
  using id            = cudf::type_id;
  auto const bytes_of = [](id t) {
    return sirius_dynamic_in_list_filter::estimated_set_bytes(1000, cudf::data_type{t});
  };
  CHECK(bytes_of(id::INT8) == bytes_of(id::INT32));
  CHECK(bytes_of(id::INT16) == bytes_of(id::INT32));
  CHECK(bytes_of(id::UINT8) == bytes_of(id::INT32));
  CHECK(bytes_of(id::UINT32) == bytes_of(id::INT32));
  CHECK(bytes_of(id::UINT64) == bytes_of(id::INT64));
  CHECK(bytes_of(id::INT64) == 2 * bytes_of(id::INT32));
}

// The three filters must answer identically whatever carrier the probe arrives at, with or
// without a prior keep-mask, for both signed reps. IN-lists are exact against a host oracle; Bloom
// is compared against its own widest-carrier answer, which must also contain every inserted key.
TEST_CASE("membership masks are identical across probe carriers and prior masks",
          "[dynamic_filter][probe][key_domain]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  // Every value fits INT8 so each carrier sees the same logical probe.
  std::vector<std::int64_t> const key_values{1, 2, 3, 50, 100, 127, -128, -7};
  std::vector<std::int64_t> probe_values;
  std::vector<std::uint8_t> expected;
  std::vector<bool> keep;
  for (std::int64_t v = -128; v <= 127; ++v) {  // 256 rows: crosses prior-mask word boundaries
    probe_values.push_back(v);
    expected.push_back(std::find(key_values.begin(), key_values.end(), v) != key_values.end() ? 1
                                                                                              : 0);
    keep.push_back((v % 3) == 0);
  }
  auto prior                 = upload_prior_mask(keep, stream);
  auto const* prior_words    = static_cast<std::uint32_t const*>(prior.data());
  auto const expected_masked = and_with(expected, keep);

  auto const run = [&](auto key_tag) {
    using key_type  = decltype(key_tag);
    auto const keys = make_typed<key_type>(key_values, stream);
    sirius_dynamic_in_list_filter in_list{keys->view(), stream, mr};
    sirius_dynamic_small_in_list_filter small_list{keys->view(), stream, mr};
    sirius_dynamic_bloom_filter bloom{keys->view(), stream, mr};
    CHECK(in_list.domain().native == keys->type());

    // Reference: the native (widest) carrier with no prior.
    auto const native          = make_typed<std::int64_t>(probe_values, stream);
    auto const bloom_reference = probe_mask(bloom, native->view(), nullptr, stream);
    for (std::size_t i = 0; i < expected.size(); ++i) {
      if (expected[i] != 0) { REQUIRE(bloom_reference[i] == 1); }  // no false negatives
    }
    auto const bloom_reference_masked = and_with(bloom_reference, keep);

    for_each_signed_carrier([&](auto carrier_tag) {
      using carrier_type = decltype(carrier_tag);
      auto const probe   = make_typed<carrier_type>(probe_values, stream);
      INFO("key=" << static_cast<int>(keys->type().id())
                  << " carrier=" << static_cast<int>(probe->type().id()));
      CHECK(probe_mask(in_list, probe->view(), nullptr, stream) == expected);
      CHECK(probe_mask(in_list, probe->view(), prior_words, stream) == expected_masked);
      CHECK(probe_mask(small_list, probe->view(), nullptr, stream) == expected);
      CHECK(probe_mask(small_list, probe->view(), prior_words, stream) == expected_masked);
      CHECK(probe_mask(bloom, probe->view(), nullptr, stream) == bloom_reference);
      CHECK(probe_mask(bloom, probe->view(), prior_words, stream) == bloom_reference_masked);
    });
  };
  run(std::int32_t{});
  run(std::int64_t{});
}

TEST_CASE("INT8 and INT16 build keys publish carrier-typed 32-bit sets",
          "[dynamic_filter][probe][key_domain]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  // Probe values: two hits, an INT8-range miss, and values only a wider carrier can hold (an
  // INT16 value, an INT32 value, and one beyond INT32 that must range-check to non-member).
  std::vector<std::int64_t> const probe_values{7, -5, 100, 0, 300, 70'000, 5'000'000'000LL};

  auto const run = [&](auto key_tag, std::vector<std::int64_t> key_values) {
    using key_type  = decltype(key_tag);
    auto const keys = make_typed<key_type>(key_values, stream);
    REQUIRE(sirius_dynamic_in_list_filter::supports(keys->view()));
    REQUIRE(sirius_dynamic_small_in_list_filter::supports(keys->view()));
    REQUIRE(sirius_dynamic_bloom_filter::supports(keys->type()));

    sirius_dynamic_in_list_filter in_list{keys->view(), stream, mr};
    sirius_dynamic_small_in_list_filter small_list{keys->view(), stream, mr};
    sirius_dynamic_bloom_filter bloom{keys->view(), stream, mr};
    for (auto const* domain : {&in_list.domain(), &small_list.domain(), &bloom.domain()}) {
      CHECK(domain->rep == membership_key_rep::i32);
      CHECK(domain->family == membership_key_family::signed_int);
      CHECK(domain->native == keys->type());
    }
    CHECK(in_list.has_persistent_set());

    for_each_signed_carrier([&](auto carrier_tag) {
      using carrier_type = decltype(carrier_tag);
      std::vector<std::int64_t> values;
      std::vector<std::uint8_t> expected;
      for (auto const v : probe_values) {
        if (v < std::numeric_limits<carrier_type>::min() ||
            v > std::numeric_limits<carrier_type>::max()) {
          continue;
        }
        values.push_back(v);
        expected.push_back(
          std::find(key_values.begin(), key_values.end(), v) != key_values.end() ? 1 : 0);
      }
      auto const probe = make_typed<carrier_type>(values, stream);
      INFO("key=" << static_cast<int>(keys->type().id())
                  << " carrier=" << static_cast<int>(probe->type().id()));
      CHECK(probe_mask(in_list, probe->view(), nullptr, stream) == expected);
      CHECK(probe_mask(small_list, probe->view(), nullptr, stream) == expected);
      auto const bloom_mask = probe_mask(bloom, probe->view(), nullptr, stream);
      for (std::size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != 0) { CHECK(bloom_mask[i] == 1); }
      }
      // Anything beyond INT32 can never be a member of a 32-bit set.
      for (std::size_t i = 0; i < values.size(); ++i) {
        if (values[i] > std::numeric_limits<std::int32_t>::max()) { CHECK(bloom_mask[i] == 0); }
      }
    });
  };
  run(std::int8_t{}, {7, -5, 100});
  run(std::int16_t{}, {7, -5, 300});
}

TEST_CASE("unsigned keys probe every unsigned carrier and decline signed ones",
          "[dynamic_filter][probe][key_domain]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  std::vector<std::uint64_t> const probe_values{
    0, 1, 200, 255, 60'000, 3'000'000'000ULL, 5'000'000'000ULL};

  auto const run = [&](
                     auto key_tag, std::vector<std::uint64_t> key_values, membership_key_rep rep) {
    using key_type  = decltype(key_tag);
    auto const keys = make_unsigned<key_type>(key_values, stream);
    REQUIRE(membership_key_supported(keys->type()));

    sirius_dynamic_in_list_filter in_list{keys->view(), stream, mr};
    sirius_dynamic_small_in_list_filter small_list{keys->view(), stream, mr};
    sirius_dynamic_bloom_filter bloom{keys->view(), stream, mr};
    for (auto const* domain : {&in_list.domain(), &small_list.domain(), &bloom.domain()}) {
      CHECK(domain->rep == rep);
      CHECK(domain->family == membership_key_family::unsigned_int);
    }

    for_each_unsigned_carrier([&](auto carrier_tag) {
      using carrier_type = decltype(carrier_tag);
      std::vector<std::uint64_t> values;
      std::vector<std::uint8_t> expected;
      for (auto const v : probe_values) {
        if (v > std::numeric_limits<carrier_type>::max()) { continue; }
        values.push_back(v);
        expected.push_back(
          std::find(key_values.begin(), key_values.end(), v) != key_values.end() ? 1 : 0);
      }
      auto const probe = make_unsigned<carrier_type>(values, stream);
      INFO("key=" << static_cast<int>(keys->type().id())
                  << " carrier=" << static_cast<int>(probe->type().id()));
      CHECK(probe_mask(in_list, probe->view(), nullptr, stream) == expected);
      CHECK(probe_mask(small_list, probe->view(), nullptr, stream) == expected);
      auto const bloom_mask = probe_mask(bloom, probe->view(), nullptr, stream);
      for (std::size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != 0) { CHECK(bloom_mask[i] == 1); }
      }
    });

    // Signed carriers are a semantic mismatch, not a width one.
    auto const signed_probe = make_typed<std::int32_t>({0, 1, 200}, stream);
    CHECK(in_list.compute_mask(signed_probe->view(), kDevice, stream, mr) == nullptr);
    CHECK(small_list.compute_mask(signed_probe->view(), kDevice, stream, mr) == nullptr);
    CHECK(bloom.compute_mask(signed_probe->view(), kDevice, stream, mr) == nullptr);
  };
  run(std::uint8_t{}, {1, 200, 255}, membership_key_rep::u32);
  run(std::uint16_t{}, {1, 200, 60'000}, membership_key_rep::u32);
  run(std::uint32_t{}, {0, 200, 3'000'000'000ULL}, membership_key_rep::u32);
  run(std::uint64_t{}, {0, 200, 5'000'000'000ULL}, membership_key_rep::u64);
}

TEST_CASE("unsigned hash IN-list reserves the maximum as its sentinel, so 0 is exact",
          "[dynamic_filter][probe][key_domain]")
{
  auto const stream      = cudf::get_default_stream();
  auto const mr          = cudf::get_current_device_resource_ref();
  constexpr auto u32_max = std::numeric_limits<std::uint32_t>::max();
  constexpr auto u64_max = std::numeric_limits<std::uint64_t>::max();

  SECTION("u32 set: 0 is a real key that can be absent; UINT32_MAX is kept conservatively")
  {
    auto const keys = make_unsigned<std::uint32_t>({5}, stream);
    sirius_dynamic_in_list_filter filter{keys->view(), stream, mr};
    auto const probe = make_unsigned<std::uint32_t>({0, 5, u32_max}, stream);
    CHECK(probe_mask(filter, probe->view(), nullptr, stream) == std::vector<std::uint8_t>{0, 1, 1});

    // A UINT64 probe equal to the u32 sentinel narrows onto it and is kept; one past the u32
    // range is a definite non-member.
    auto const wide =
      make_unsigned<std::uint64_t>({0, u32_max, std::uint64_t{u32_max} + 1}, stream);
    CHECK(probe_mask(filter, wide->view(), nullptr, stream) == std::vector<std::uint8_t>{0, 1, 0});
  }

  SECTION("u64 set: UINT64_MAX is kept conservatively, 0 is exact")
  {
    auto const keys = make_unsigned<std::uint64_t>({5}, stream);
    sirius_dynamic_in_list_filter filter{keys->view(), stream, mr};
    auto const probe = make_unsigned<std::uint64_t>({0, 5, u64_max}, stream);
    CHECK(probe_mask(filter, probe->view(), nullptr, stream) == std::vector<std::uint8_t>{0, 1, 1});
    // A UINT32 probe widened into a u64 set never lands on the sentinel.
    auto const narrow = make_unsigned<std::uint32_t>({u32_max, 5}, stream);
    CHECK(probe_mask(filter, narrow->view(), nullptr, stream) == std::vector<std::uint8_t>{0, 1});
  }

  SECTION("a build key equal to the unsigned sentinel is still kept on probe")
  {
    auto const keys = make_unsigned<std::uint32_t>({u32_max, 3}, stream);
    sirius_dynamic_in_list_filter filter{keys->view(), stream, mr};
    auto const probe = make_unsigned<std::uint32_t>({u32_max, 3, 4}, stream);
    CHECK(probe_mask(filter, probe->view(), nullptr, stream) == std::vector<std::uint8_t>{1, 1, 0});
  }
}
