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

#pragma once

/**
 * @file fused_scan_filter_test_utils.hpp
 * @brief Fixtures shared by the filtered-decode tests: the env-gate armer, device upload and
 * readback of plain columns, the synthetic source tables, and exact membership sets with the probe
 * directives that consult them.
 */

#include "codegen/selection/selection.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <op/dynamic_filter/sirius_dynamic_filter.hpp>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace sirius::test::fused_scan_filter {

/// Both knobs are function-local statistics latched on first use. Arm them at static-init time --
/// before any test can latch them -- without overriding an explicit setting (overwrite=0). The
/// filtered path's contract is byte-identical output whatever these say, so suite-wide arming is
/// behavior-neutral for every other test.
///
/// MAX_MEMBER defaults to 1, which makes the multi-probe cascade unreachable; raise it to 2 so the
/// fold-back (each probe's survivors become the next probe's prior) is actually covered.
struct fused_gate_armer {
  fused_gate_armer()
  {
    setenv("SIRIUS_EXP_FUSED_SCAN_FILTER", "1", /*overwrite=*/0);
    setenv("SIRIUS_EXP_FUSED_SCAN_MAX_MEMBER", "2", /*overwrite=*/0);
  }
};
inline fused_gate_armer const arm_fused_gate{};

inline constexpr cudf::size_type kRows = 4096;

template <typename T>
std::unique_ptr<cudf::column> upload_column(std::vector<T> const& host, ::cuda::stream_ref stream)
{
  auto const mr = cudf::get_current_device_resource_ref();
  auto col      = cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<T>()},
                                       static_cast<cudf::size_type>(host.size()),
                                       cudf::mask_state::UNALLOCATED,
                                       stream,
                                       mr);
  REQUIRE(cudaMemcpyAsync(col->mutable_view().data<T>(),
                          host.data(),
                          host.size() * sizeof(T),
                          cudaMemcpyHostToDevice,
                          stream.get()) == cudaSuccess);
  return col;
}

/// A STRING column of @p host; with a non-empty @p valid, row i is null wherever valid[i] is false.
inline std::unique_ptr<cudf::column> upload_strings(std::vector<std::string> const& host,
                                                    ::cuda::stream_ref stream,
                                                    std::vector<bool> const& valid = {})
{
  auto const mr = cudf::get_current_device_resource_ref();
  auto const n  = static_cast<cudf::size_type>(host.size());
  std::vector<cudf::size_type> offsets(static_cast<std::size_t>(n) + 1, 0);
  std::string chars;
  for (std::size_t i = 0; i < host.size(); ++i) {
    chars += host[i];
    offsets[i + 1] = static_cast<cudf::size_type>(chars.size());
  }
  auto offsets_col = upload_column(offsets, stream);
  rmm::device_buffer chars_buf{chars.data(), chars.size(), stream, mr};
  if (valid.empty()) {
    stream.sync();
    return cudf::make_strings_column(
      n,
      std::move(offsets_col),
      std::move(chars_buf),
      0,
      cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr));
  }
  REQUIRE(valid.size() == host.size());
  std::vector<cudf::bitmask_type> words(cudf::num_bitmask_words(n), cudf::bitmask_type{0});
  cudf::size_type null_count = 0;
  for (std::size_t i = 0; i < valid.size(); ++i) {
    if (valid[i]) {
      words[i / 32] |= cudf::bitmask_type{1} << (i % 32);
    } else {
      ++null_count;
    }
  }
  rmm::device_buffer mask{words.data(), words.size() * sizeof(cudf::bitmask_type), stream, mr};
  stream.sync();
  return cudf::make_strings_column(
    n, std::move(offsets_col), std::move(chars_buf), null_count, std::move(mask));
}

/// The STRING key of row @p i: "item_<i % 50>".
inline std::string key_string(cudf::size_type i) { return "item_" + std::to_string(i % 50); }

/// The values of STRING column @p col (a null row reads as its stored, normally empty, value).
inline std::vector<std::string> strings_to_host(cudf::column_view const& col,
                                                ::cuda::stream_ref stream)
{
  REQUIRE(col.type().id() == cudf::type_id::STRING);
  cudf::strings_column_view const sv{col};
  std::vector<std::string> out;
  if (col.size() == 0) { return out; }
  REQUIRE(sv.offsets().type().id() == cudf::type_id::INT32);
  std::vector<cudf::size_type> offsets(static_cast<std::size_t>(col.size()) + 1);
  REQUIRE(cudaMemcpyAsync(offsets.data(),
                          sv.offsets().data<cudf::size_type>() + col.offset(),
                          offsets.size() * sizeof(cudf::size_type),
                          cudaMemcpyDeviceToHost,
                          stream.get()) == cudaSuccess);
  stream.sync();
  auto const first = offsets.front();
  auto const bytes = static_cast<std::size_t>(offsets.back() - first);
  std::string chars(bytes, '\0');
  if (bytes > 0) {
    REQUIRE(cudaMemcpyAsync(chars.data(),
                            sv.chars_begin(stream) + first,
                            bytes,
                            cudaMemcpyDeviceToHost,
                            stream.get()) == cudaSuccess);
    stream.sync();
  }
  out.reserve(static_cast<std::size_t>(col.size()));
  for (cudf::size_type i = 0; i < col.size(); ++i) {
    auto const b = static_cast<std::size_t>(offsets[static_cast<std::size_t>(i)] - first);
    auto const e = static_cast<std::size_t>(offsets[static_cast<std::size_t>(i) + 1] - first);
    out.emplace_back(chars.substr(b, e - b));
  }
  return out;
}

/// col 0 ("v"): i % 100. col 1 ("k"): i % 50, stored at KeyT, or as the unscaled storage of @p
/// key_type when one is given (a fixed-point chunk). With @p filter_modulus, col 2 ("f"): i %
/// filter_modulus as INT32, a column only a filter reads.
template <typename KeyT = std::int32_t>
std::unique_ptr<cudf::table> make_source_table(
  ::cuda::stream_ref stream,
  cudf::data_type key_type                   = cudf::data_type{cudf::type_to_id<KeyT>()},
  std::optional<std::int32_t> filter_modulus = std::nullopt)
{
  std::vector<std::int32_t> v(kRows);
  std::vector<KeyT> k(kRows);
  for (cudf::size_type i = 0; i < kRows; ++i) {
    v[static_cast<std::size_t>(i)] = i % 100;
    k[static_cast<std::size_t>(i)] = static_cast<KeyT>(i % 50);
  }
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(upload_column(v, stream));
  auto key = upload_column(k, stream);
  if (key_type != key->type()) {
    // Re-tag the storage as the fixed-point type (same width, identical bits).
    REQUIRE(cudf::size_of(key_type) == sizeof(KeyT));
    auto contents = key->release();
    key           = std::make_unique<cudf::column>(
      key_type,
      kRows,
      std::move(*contents.data),
      cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream),
      0);
  }
  cols.push_back(std::move(key));
  if (filter_modulus) {
    std::vector<std::int32_t> f(kRows);
    for (cudf::size_type i = 0; i < kRows; ++i) {
      f[static_cast<std::size_t>(i)] = i % *filter_modulus;
    }
    cols.push_back(upload_column(f, stream));
  }
  stream.sync();
  return std::make_unique<cudf::table>(std::move(cols));
}

/// INT64-built exact membership set over the multiples of @p step in [0, 50) -- the probe column
/// decodes as INT32, so the probe exercises the heterogeneous (cast-free) path.
inline std::shared_ptr<sirius::op::sirius_dynamic_in_list_filter> make_key_set(
  ::cuda::stream_ref stream, std::int64_t step = 5)
{
  std::vector<std::int64_t> keys;
  for (std::int64_t k = 0; k < 50; k += step) {
    keys.push_back(k);
  }
  auto const mr = cudf::get_current_device_resource_ref();
  auto col      = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                       static_cast<cudf::size_type>(keys.size()),
                                       cudf::mask_state::UNALLOCATED,
                                       stream,
                                       mr);
  REQUIRE(cudaMemcpyAsync(col->mutable_view().data<std::int64_t>(),
                          keys.data(),
                          keys.size() * sizeof(std::int64_t),
                          cudaMemcpyHostToDevice,
                          stream.get()) == cudaSuccess);
  stream.sync();
  return std::make_shared<sirius::op::sirius_dynamic_in_list_filter>(col->view(), stream, mr);
}

inline sirius::codegen::membership_filter_directive make_probe_directive(
  std::size_t column,
  std::shared_ptr<sirius::op::sirius_dynamic_in_list_filter> filter,
  bool* saw_prior)
{
  return {column,
          [filter = std::move(filter), saw_prior](cudf::column_view keys,
                                                  std::uint32_t const* prior_mask_words,
                                                  ::cuda::stream_ref s,
                                                  rmm::device_async_resource_ref mr) {
            if (saw_prior != nullptr) { *saw_prior = prior_mask_words != nullptr; }
            return filter->compute_mask(keys, prior_mask_words, /*device_id=*/0, s, mr);
          }};
}

template <typename T = std::int32_t>
std::vector<T> column_to_host(cudf::column_view const& col,
                              ::cuda::stream_ref stream,
                              cudf::data_type expected_type = cudf::data_type{
                                cudf::type_to_id<T>()})
{
  REQUIRE(col.type() == expected_type);
  std::vector<T> host(static_cast<std::size_t>(col.size()));
  if (!host.empty()) {
    REQUIRE(cudaMemcpyAsync(host.data(),
                            col.data<T>(),
                            host.size() * sizeof(T),
                            cudaMemcpyDeviceToHost,
                            stream.get()) == cudaSuccess);
  }
  stream.sync();
  return host;
}

/// True when the filtered attempt was declined because the env gate stayed off (an explicit
/// SIRIUS_EXP_FUSED_SCAN_FILTER=0, or another TU latched the static before the armer ran).
inline bool gate_stayed_off(sirius::codegen::scan_filter_result const& result)
{
  return !result.applied && result.status == sirius::codegen::scan_filter_status::refused;
}

}  // namespace sirius::test::fused_scan_filter
