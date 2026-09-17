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
 * @file test_fused_membership_mask.cpp
 * @brief Mask-aware membership probing in the filtered-decode wave: with another mask source
 *        present, the membership probes run AFTER the partial combine and receive it as a prior
 *        mask; without one they stay concurrent and prior-free. The probe key stays at its narrow
 *        decoded carrier (INT32, or INT8 for a chunk stored that narrow) against an INT64-built
 *        set — the filtered decode must not require a materialized cast — and a nullable probe
 *        column yields a non-nullable mask that the filtered decode consumes.
 */

#include "api/simpatico_codegen.hpp"
#include "codegen/selection/selection.hpp"
#include "codegen/util/stream_pool.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <op/dynamic_filter/sirius_dynamic_filter.hpp>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

namespace {

// Both knobs are function-local statistics latched on first use. Arm them at static-init time —
// before any test can latch them — without overriding an explicit setting (overwrite=0). The
// filtered path's contract is byte-identical output whatever these say, so suite-wide arming is
// behavior-neutral for every other test.
//
// MAX_MEMBER defaults to 1, which makes the multi-probe cascade unreachable; raise it to 2 so the
// fold-back (each probe's survivors become the next probe's prior) is actually covered.
struct fused_gate_armer {
  fused_gate_armer()
  {
    setenv("SIRIUS_EXP_FUSED_SCAN_FILTER", "1", /*overwrite=*/0);
    setenv("SIRIUS_EXP_FUSED_SCAN_MAX_MEMBER", "2", /*overwrite=*/0);
  }
};
[[maybe_unused]] fused_gate_armer const arm_fused_gate{};

constexpr cudf::size_type kRows = 4096;

template <typename T>
std::unique_ptr<cudf::column> upload_column(std::vector<T> const& host,
                                            rmm::cuda_stream_view stream)
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
                          stream.value()) == cudaSuccess);
  return col;
}

// col 0 ("v", range-filtered): i % 100. col 1 ("k", membership key): i % 50, stored at KeyT.
template <typename KeyT = std::int32_t>
std::unique_ptr<cudf::table> make_source_table(rmm::cuda_stream_view stream)
{
  std::vector<std::int32_t> v(kRows);
  std::vector<KeyT> k(kRows);
  for (cudf::size_type i = 0; i < kRows; ++i) {
    v[static_cast<std::size_t>(i)] = i % 100;
    k[static_cast<std::size_t>(i)] = static_cast<KeyT>(i % 50);
  }
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(upload_column(v, stream));
  cols.push_back(upload_column(k, stream));
  stream.synchronize();
  return std::make_unique<cudf::table>(std::move(cols));
}

constexpr char const* kBitpackPlans =
  "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
  "---\n"
  "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";

// INT64-built exact membership set over the multiples of @p step in [0, 50) — the probe column
// decodes as INT32, so the probe exercises the heterogeneous (cast-free) path.
std::shared_ptr<sirius::op::sirius_dynamic_in_list_filter> make_key_set(
  rmm::cuda_stream_view stream, std::int64_t step = 5)
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
                          stream.value()) == cudaSuccess);
  stream.synchronize();
  return std::make_shared<sirius::op::sirius_dynamic_in_list_filter>(col->view(), stream, mr);
}

sirius::codegen::membership_filter_directive make_probe_directive(
  std::size_t column,
  std::shared_ptr<sirius::op::sirius_dynamic_in_list_filter> filter,
  bool* saw_prior)
{
  return {column,
          [filter = std::move(filter), saw_prior](cudf::column_view keys,
                                                  std::uint32_t const* prior_mask_words,
                                                  rmm::cuda_stream_view s,
                                                  rmm::device_async_resource_ref mr) {
            if (saw_prior != nullptr) { *saw_prior = prior_mask_words != nullptr; }
            return filter->compute_mask(keys, prior_mask_words, /*device_id=*/0, s, mr);
          }};
}

template <typename T = std::int32_t>
std::vector<T> column_to_host(cudf::column_view const& col, rmm::cuda_stream_view stream)
{
  REQUIRE(col.type().id() == cudf::type_to_id<T>());
  std::vector<T> host(static_cast<std::size_t>(col.size()));
  if (!host.empty()) {
    REQUIRE(cudaMemcpyAsync(host.data(),
                            col.data<T>(),
                            host.size() * sizeof(T),
                            cudaMemcpyDeviceToHost,
                            stream.value()) == cudaSuccess);
  }
  stream.synchronize();
  return host;
}

/// True when the filtered attempt was declined because the env gate stayed off (an explicit
/// SIRIUS_EXP_FUSED_SCAN_FILTER=0, or another TU latched the static before our armer ran).
bool gate_stayed_off(sirius::codegen::scan_filter_result const& result)
{
  return !result.applied && result.status == sirius::codegen::scan_filter_status::refused;
}

}  // namespace

TEST_CASE("filtered-decode membership probe consumes the static range mask as a prior",
          "[fused_scan_filter][dynamic_filter]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  auto table        = make_source_table(stream);
  auto const ct     = simpatico::compress_with_plan(table->view(), kBitpackPlans, stream, mr);

  simpatico::stream_pool pool;
  REQUIRE(pool.init(4));

  bool saw_prior = false;
  sirius::codegen::scan_filter_request request;
  request.filters.push_back({0, {0, 19}});  // v in [0, 19]: 20% static selectivity
  request.membership_filters.push_back(make_probe_directive(1, make_key_set(stream), &saw_prior));
  request.routes = {sirius::codegen::decode_route::bitpack_mask,
                    sirius::codegen::decode_route::bitpack_mask};

  std::vector<std::size_t> const selected{0, 1};
  sirius::codegen::scan_filter_result result;
  auto out = simpatico::decompress_scan_filter(ct, selected, request, result, pool, stream, mr);
  if (gate_stayed_off(result)) {
    WARN("filtered-decode env gate off in this process; skipping membership coverage");
    return;
  }

  REQUIRE(result.applied);
  CHECK(saw_prior);  // another mask source exists, so the probe must run mask-aware

  // Reference: (i % 100) <= 19 AND (i % 50) % 5 == 0.
  std::vector<std::int32_t> expect_v;
  std::vector<std::int32_t> expect_k;
  for (cudf::size_type i = 0; i < kRows; ++i) {
    if (i % 100 <= 19 && (i % 50) % 5 == 0) {
      expect_v.push_back(i % 100);
      expect_k.push_back(i % 50);
    }
  }
  REQUIRE(result.survivor_count == static_cast<std::int64_t>(expect_v.size()));
  REQUIRE(out->num_columns() == 2);
  CHECK(column_to_host(out->view().column(0), stream) == expect_v);
  CHECK(column_to_host(out->view().column(1), stream) == expect_k);
}

TEST_CASE("filtered-decode membership probe stays prior-free without another source",
          "[fused_scan_filter][dynamic_filter]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  auto table        = make_source_table(stream);
  auto const ct     = simpatico::compress_with_plan(table->view(), kBitpackPlans, stream, mr);

  simpatico::stream_pool pool;
  REQUIRE(pool.init(4));

  bool saw_prior = true;
  sirius::codegen::scan_filter_request request;
  request.membership_filters.push_back(make_probe_directive(1, make_key_set(stream), &saw_prior));
  request.routes = {sirius::codegen::decode_route::bitpack_mask,
                    sirius::codegen::decode_route::bitpack_mask};

  std::vector<std::size_t> const selected{0, 1};
  sirius::codegen::scan_filter_result result;
  auto out = simpatico::decompress_scan_filter(ct, selected, request, result, pool, stream, mr);
  if (gate_stayed_off(result)) {
    WARN("filtered-decode env gate off in this process; skipping membership coverage");
    return;
  }

  REQUIRE(result.applied);
  CHECK_FALSE(saw_prior);  // membership-only requests keep the concurrent, prior-free path

  std::vector<std::int32_t> expect_k;
  for (cudf::size_type i = 0; i < kRows; ++i) {
    if ((i % 50) % 5 == 0) { expect_k.push_back(i % 50); }
  }
  REQUIRE(result.survivor_count == static_cast<std::int64_t>(expect_k.size()));
  REQUIRE(out->num_columns() == 2);
  CHECK(column_to_host(out->view().column(1), stream) == expect_k);
}

TEST_CASE("filtered-decode membership probe takes a keep mask alone as its prior",
          "[fused_scan_filter][dynamic_filter]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  auto table        = make_source_table(stream);
  auto const ct     = simpatico::compress_with_plan(table->view(), kBitpackPlans, stream, mr);

  simpatico::stream_pool pool;
  REQUIRE(pool.init(4));

  // No static range: the positional keep mask (even rows only) is the only other source, so it
  // must prime the cascade's prior on its own and survive into the combined mask.
  std::vector<std::uint32_t> const keep_words(static_cast<std::size_t>((kRows + 31) / 32),
                                              0x55555555u);
  bool saw_prior = false;
  sirius::codegen::scan_filter_request request;
  request.membership_filters.push_back(make_probe_directive(1, make_key_set(stream), &saw_prior));
  request.routes          = {sirius::codegen::decode_route::bitpack_mask,
                             sirius::codegen::decode_route::bitpack_mask};
  request.keep_mask_words = keep_words.data();
  request.keep_mask_rows  = kRows;

  std::vector<std::size_t> const selected{0, 1};
  sirius::codegen::scan_filter_result result;
  auto out = simpatico::decompress_scan_filter(ct, selected, request, result, pool, stream, mr);
  if (gate_stayed_off(result)) {
    WARN("filtered-decode env gate off in this process; skipping membership coverage");
    return;
  }

  REQUIRE(result.applied);
  CHECK(result.keep_mask_applied);
  CHECK(saw_prior);  // the keep mask alone is enough to take the sequential, pruned path

  std::vector<std::int32_t> expect_k;
  for (cudf::size_type i = 0; i < kRows; ++i) {
    if (i % 2 == 0 && (i % 50) % 5 == 0) { expect_k.push_back(i % 50); }
  }
  REQUIRE(result.survivor_count == static_cast<std::int64_t>(expect_k.size()));
  REQUIRE(out->num_columns() == 2);
  CHECK(column_to_host(out->view().column(1), stream) == expect_k);
}

TEST_CASE("a membership cascade folds each probe's survivors into the next probe's prior",
          "[fused_scan_filter][dynamic_filter]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  auto table        = make_source_table(stream);
  auto const ct     = simpatico::compress_with_plan(table->view(), kBitpackPlans, stream, mr);

  simpatico::stream_pool pool;
  REQUIRE(pool.init(4));

  // Two membership sources on the same key column behind one static range. The second probe's
  // prior must already carry the first probe's result, so the conjunction is {multiples of 10}.
  bool saw_prior_a = false;
  bool saw_prior_b = false;
  sirius::codegen::scan_filter_request request;
  request.filters.push_back({0, {0, 49}});  // v in [0, 49]: keeps i % 100 < 50
  request.membership_filters.push_back(
    make_probe_directive(1, make_key_set(stream, /*step=*/5), &saw_prior_a));
  request.membership_filters.push_back(
    make_probe_directive(1, make_key_set(stream, /*step=*/2), &saw_prior_b));
  request.routes = {sirius::codegen::decode_route::bitpack_mask,
                    sirius::codegen::decode_route::bitpack_mask};

  std::vector<std::size_t> const selected{0, 1};
  sirius::codegen::scan_filter_result result;
  auto out = simpatico::decompress_scan_filter(ct, selected, request, result, pool, stream, mr);
  if (gate_stayed_off(result)) {
    WARN("filtered-decode env gate off in this process; skipping membership coverage");
    return;
  }

  REQUIRE(result.applied);
  CHECK(saw_prior_a);
  CHECK(saw_prior_b);

  // Reference: (i % 100) <= 49 AND (i % 50) % 5 == 0 AND (i % 50) % 2 == 0.
  std::vector<std::int32_t> expect_k;
  for (cudf::size_type i = 0; i < kRows; ++i) {
    if (i % 100 <= 49 && (i % 50) % 5 == 0 && (i % 50) % 2 == 0) { expect_k.push_back(i % 50); }
  }
  REQUIRE(result.survivor_count == static_cast<std::int64_t>(expect_k.size()));
  REQUIRE(out->num_columns() == 2);
  CHECK(column_to_host(out->view().column(1), stream) == expect_k);
}

TEST_CASE("filtered-decode membership probe on an all-dead prior stays mask-aware",
          "[fused_scan_filter][dynamic_filter]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  auto table        = make_source_table(stream);
  auto const ct     = simpatico::compress_with_plan(table->view(), kBitpackPlans, stream, mr);

  simpatico::stream_pool pool;
  REQUIRE(pool.init(4));

  bool saw_prior = false;
  sirius::codegen::scan_filter_request request;
  request.filters.push_back({0, {1000, 2000}});  // v max is 99: the range kills every row
  request.membership_filters.push_back(make_probe_directive(1, make_key_set(stream), &saw_prior));
  request.routes = {sirius::codegen::decode_route::bitpack_mask,
                    sirius::codegen::decode_route::bitpack_mask};

  std::vector<std::size_t> const selected{0, 1};
  sirius::codegen::scan_filter_result result;
  auto out = simpatico::decompress_scan_filter(ct, selected, request, result, pool, stream, mr);
  if (gate_stayed_off(result)) {
    WARN("filtered-decode env gate off in this process; skipping membership coverage");
    return;
  }

  // The probe must still run mask-aware (it precedes the survivor count), with every row dead in
  // its prior.
  CHECK(saw_prior);
  REQUIRE(out->num_columns() == 2);
  // Zero-survivor batches may be handled by either the compacted route (0-row columns) or the
  // full-width fallback, depending on the launcher's empty-batch support; both are correct.
  if (result.applied) {
    CHECK(result.survivor_count == 0);
    CHECK(out->view().column(0).size() == 0);
  } else {
    CHECK(out->view().column(0).size() == kRows);
  }
}

// A pinned chunk can store a BIGINT key as INT8 when its values fit; the fused decode re-tags the
// decoded column with that stored carrier, so the membership probe sees INT8 against an
// INT64-built set and must answer exactly what the native column would.
TEST_CASE("filtered-decode membership probe reads an INT8-carrier key chunk",
          "[fused_scan_filter][dynamic_filter]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  auto table        = make_source_table<std::int8_t>(stream);
  REQUIRE(table->view().column(1).type().id() == cudf::type_id::INT8);
  auto const ct = simpatico::compress_with_plan(table->view(), kBitpackPlans, stream, mr);

  simpatico::stream_pool pool;
  REQUIRE(pool.init(4));

  bool saw_prior = false;
  sirius::codegen::scan_filter_request request;
  request.filters.push_back({0, {0, 19}});
  request.membership_filters.push_back(make_probe_directive(1, make_key_set(stream), &saw_prior));
  request.routes = {sirius::codegen::decode_route::bitpack_mask,
                    sirius::codegen::decode_route::bitpack_mask};

  std::vector<std::size_t> const selected{0, 1};
  sirius::codegen::scan_filter_result result;
  auto out = simpatico::decompress_scan_filter(ct, selected, request, result, pool, stream, mr);
  if (gate_stayed_off(result)) {
    WARN("filtered-decode env gate off in this process; skipping membership coverage");
    return;
  }

  REQUIRE(result.applied);
  CHECK(saw_prior);

  std::vector<std::int32_t> expect_v;
  std::vector<std::int8_t> expect_k;
  for (cudf::size_type i = 0; i < kRows; ++i) {
    if (i % 100 <= 19 && (i % 50) % 5 == 0) {
      expect_v.push_back(i % 100);
      expect_k.push_back(static_cast<std::int8_t>(i % 50));
    }
  }
  REQUIRE(result.survivor_count == static_cast<std::int64_t>(expect_v.size()));
  REQUIRE(out->num_columns() == 2);
  CHECK(column_to_host(out->view().column(0), stream) == expect_v);
  CHECK(column_to_host<std::int8_t>(out->view().column(1), stream) == expect_k);
}

// The filtered decode requires a non-nullable BOOL8 from every membership probe (a null-masked
// result is a broken probe contract and throws). A nullable probe column must therefore come back
// with its null rows written as `false`, not with the probe's null mask copied onto the result.
//
// The compression planner fails closed on nullable input to the integer codecs, so a nullable key
// chunk cannot be produced through compress_with_plan today; the validity is attached to the
// decoded key column inside the probe directive instead, which is exactly the view the filter
// sees.
TEST_CASE("filtered-decode membership probe consumes a nullable probe column",
          "[fused_scan_filter][dynamic_filter]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  auto table        = make_source_table(stream);
  auto const ct     = simpatico::compress_with_plan(table->view(), kBitpackPlans, stream, mr);

  simpatico::stream_pool pool;
  REQUIRE(pool.init(4));

  // Rows whose key is 10 (a member of the set {0, 5, ..., 45}) are null: they must be dropped.
  std::vector<std::uint32_t> words((kRows + 31) / 32, 0U);
  cudf::size_type null_count = 0;
  for (cudf::size_type i = 0; i < kRows; ++i) {
    if (i % 50 == 10) {
      ++null_count;
      continue;
    }
    words[static_cast<std::size_t>(i) / 32] |= (1U << (static_cast<std::uint32_t>(i) % 32));
  }
  rmm::device_buffer validity{words.data(), words.size() * sizeof(std::uint32_t), stream};
  stream.synchronize();

  bool saw_prior         = false;
  bool saw_nullable_mask = false;
  bool shape_ok          = true;
  auto filter            = make_key_set(stream);
  sirius::codegen::scan_filter_request request;
  request.filters.push_back({0, {0, 19}});
  request.membership_filters.push_back(
    {1,
     [filter, &validity, null_count, &saw_prior, &saw_nullable_mask, &shape_ok](
       cudf::column_view keys,
       std::uint32_t const* prior_mask_words,
       rmm::cuda_stream_view s,
       rmm::device_async_resource_ref probe_mr) {
       // The validity words were laid out for an unsliced kRows-row decode; anything else is a
       // test-shape surprise, reported after the decode rather than thrown through the codegen.
       if (keys.offset() != 0 || keys.size() != kRows) {
         shape_ok = false;
         return filter->compute_mask(keys, prior_mask_words, /*device_id=*/0, s, probe_mr);
       }
       cudf::column_view const nullable{keys.type(),
                                        keys.size(),
                                        keys.head(),
                                        static_cast<cudf::bitmask_type const*>(validity.data()),
                                        null_count};
       saw_prior = prior_mask_words != nullptr;
       auto mask = filter->compute_mask(nullable, prior_mask_words, /*device_id=*/0, s, probe_mr);
       if (mask && mask->nullable()) { saw_nullable_mask = true; }
       return mask;
     }});
  request.routes = {sirius::codegen::decode_route::bitpack_mask,
                    sirius::codegen::decode_route::bitpack_mask};

  std::vector<std::size_t> const selected{0, 1};
  sirius::codegen::scan_filter_result result;
  auto out = simpatico::decompress_scan_filter(ct, selected, request, result, pool, stream, mr);
  if (gate_stayed_off(result)) {
    WARN("filtered-decode env gate off in this process; skipping membership coverage");
    return;
  }

  REQUIRE(result.applied);
  REQUIRE(shape_ok);
  CHECK(saw_prior);
  CHECK_FALSE(saw_nullable_mask);

  // Reference: (i % 100) <= 19 AND (i % 50) % 5 == 0 AND NOT null (i % 50 != 10).
  std::vector<std::int32_t> expect_v;
  std::vector<std::int32_t> expect_k;
  for (cudf::size_type i = 0; i < kRows; ++i) {
    if (i % 100 <= 19 && (i % 50) % 5 == 0 && i % 50 != 10) {
      expect_v.push_back(i % 100);
      expect_k.push_back(i % 50);
    }
  }
  REQUIRE_FALSE(expect_v.empty());
  REQUIRE(result.survivor_count == static_cast<std::int64_t>(expect_v.size()));
  REQUIRE(out->num_columns() == 2);
  CHECK(column_to_host(out->view().column(0), stream) == expect_v);
  CHECK(column_to_host(out->view().column(1), stream) == expect_k);
}
