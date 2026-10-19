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
 * @file test_fused_scan_filter_delivery.cpp
 * @brief Verify filtered output prefixes and retention of columns needed by residual filters.
 *
 * Codec cases check delivered_prefix; engine cases drop trailing filter-only columns only when the
 * scan's whole filter was applied. Join-filter acceptance or refusal does not complete a partially
 * applied scan filter. Nullable selected values decline compaction and preserve the original rows
 * and validity.
 */

#include "api/simpatico_codegen.hpp"
#include "codegen/selection/decompression_pushdown_policy.hpp"
#include "codegen/selection/selection.hpp"
#include "codegen/util/stream_pool.hpp"
#include "scan/fused_scan_filter_test_utils.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <compression/compressed_scan.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace sirius::test::fused_scan_filter;

namespace {

namespace sc = sirius::codegen;

constexpr char const* kEqualitySourcePlans =
  "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
  "---\n"
  "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
  "---\n"
  "input -> dictionary -> keys_offsets, keys_chars, indices\n"
  "dictionary.indices -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";

/// "n" is dictionary-encoded with bitpacked codes, which decodes on the compacted dict_codes route.
constexpr char const* kStringSourcePlans =
  "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
  "---\n"
  "input -> dictionary -> keys_offsets, keys_chars, indices\n"
  "dictionary.indices -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
  "---\n"
  "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";

/// The null_mask channel that carries the validity of "n" takes its dictionary off the compacted
/// route, so "n" decodes full width and is then gathered to the survivors.
constexpr char const* kNullableStringSourcePlans =
  "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
  "---\n"
  "input -> dictionary -> keys_offsets, keys_chars, indices, null_mask\n"
  "dictionary.indices -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
  "dictionary.null_mask -> identity\n"
  "---\n"
  "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";

bool every_row(cudf::size_type) { return true; }

bool null_in_nullable_source(cudf::size_type i) { return i % 7 == 0; }

simpatico::compressed_table compress_equality_source(::cuda::stream_ref stream)
{
  std::vector<std::int32_t> v(kRows);
  std::vector<std::int32_t> k(kRows);
  std::vector<std::string> s(kRows);
  for (cudf::size_type i = 0; i < kRows; ++i) {
    v[static_cast<std::size_t>(i)] = i % 100;
    k[static_cast<std::size_t>(i)] = i % 50;
    s[static_cast<std::size_t>(i)] = key_string(i);
  }
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(upload_column(v, stream));
  cols.push_back(upload_column(k, stream));
  cols.push_back(upload_strings(s, stream));
  stream.sync();
  cudf::table const table{std::move(cols)};
  return simpatico::compress_with_plan(
    table.view(), kEqualitySourcePlans, stream, cudf::get_current_device_resource_ref());
}

simpatico::compressed_table compress_string_source(bool nullable, ::cuda::stream_ref stream)
{
  std::vector<std::int32_t> v(kRows);
  std::vector<std::string> n(kRows);
  std::vector<std::int32_t> f(kRows);
  std::vector<bool> valid;
  if (nullable) { valid.resize(kRows); }
  for (cudf::size_type i = 0; i < kRows; ++i) {
    auto const row     = static_cast<std::size_t>(i);
    bool const is_null = nullable && null_in_nullable_source(i);
    v[row]             = i % 100;
    n[row]             = is_null ? std::string{} : key_string(i);
    f[row]             = i % kFilterModulus;
    if (nullable) { valid[row] = !is_null; }
  }
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(upload_column(v, stream));
  cols.push_back(upload_strings(n, stream, valid));
  cols.push_back(upload_column(f, stream));
  stream.sync();
  cudf::table const table{std::move(cols)};
  return simpatico::compress_with_plan(table.view(),
                                       nullable ? kNullableStringSourcePlans : kStringSourcePlans,
                                       stream,
                                       cudf::get_current_device_resource_ref());
}

/// The rows of a string source that @p keep selects, column by column ("n" is nullopt where null).
struct string_source_rows {
  std::vector<std::int32_t> v;
  std::vector<std::optional<std::string>> n;
  std::vector<std::int32_t> f;
};

string_source_rows string_rows_where(std::predicate<cudf::size_type> auto keep, bool nullable)
{
  string_source_rows rows;
  for (cudf::size_type i = 0; i < kRows; ++i) {
    if (keep(i)) {
      rows.v.push_back(i % 100);
      rows.n.push_back(nullable && null_in_nullable_source(i)
                         ? std::nullopt
                         : std::optional<std::string>{key_string(i)});
      rows.f.push_back(i % kFilterModulus);
    }
  }
  return rows;
}

/// The values of STRING column @p col, with nullopt for each null row.
std::vector<std::optional<std::string>> nullable_strings_to_host(cudf::column_view const& col,
                                                                 ::cuda::stream_ref stream)
{
  auto const values = strings_to_host(col, stream);
  std::vector<std::optional<std::string>> out(values.begin(), values.end());
  if (!col.nullable()) { return out; }
  std::vector<cudf::bitmask_type> words(
    static_cast<std::size_t>(cudf::num_bitmask_words(col.offset() + col.size())));
  REQUIRE(cudaMemcpyAsync(words.data(),
                          col.null_mask(),
                          words.size() * sizeof(cudf::bitmask_type),
                          cudaMemcpyDeviceToHost,
                          stream.get()) == cudaSuccess);
  stream.sync();
  for (cudf::size_type i = 0; i < col.size(); ++i) {
    if (!cudf::bit_is_set(words.data(), col.offset() + i)) {
      out[static_cast<std::size_t>(i)].reset();
    }
  }
  return out;
}

/// @p out holds exactly the first @p width columns of @p rows, values included.
void require_columns(cudf::table_view out,
                     source_rows const& rows,
                     std::size_t width,
                     ::cuda::stream_ref stream)
{
  REQUIRE(static_cast<std::size_t>(out.num_columns()) == width);
  CHECK(column_to_host(out.column(0), stream) == rows.v);
  CHECK(column_to_host(out.column(1), stream) == rows.k);
  if (width > kOutputWidth) { CHECK(column_to_host(out.column(2), stream) == rows.f); }
}

/// @p out holds exactly the first @p width columns of @p rows, values and nulls included.
void require_columns(cudf::table_view out,
                     string_source_rows const& rows,
                     std::size_t width,
                     ::cuda::stream_ref stream)
{
  REQUIRE(static_cast<std::size_t>(out.num_columns()) == width);
  CHECK(column_to_host(out.column(0), stream) == rows.v);
  CHECK(nullable_strings_to_host(out.column(1), stream) == rows.n);
  if (width > kOutputWidth) { CHECK(column_to_host(out.column(2), stream) == rows.f); }
}

/// A codec request whose only source is the range on "f", with every column on its bitpack route.
sc::scan_filter_request range_request(std::optional<std::size_t> delivered_prefix,
                                      sc::range_predicate range = {kFilterLo, kFilterHi})
{
  sc::scan_filter_request request;
  request.filters.push_back({2, range});
  request.routes = {
    sc::decode_route::bitpack_mask, sc::decode_route::bitpack_mask, sc::decode_route::bitpack_mask};
  request.delivered_prefix = delivered_prefix;
  return request;
}

/// A codec request whose only source is an equality on "s" over @p keys, delivering "v" and "k".
sc::scan_filter_request equality_request(std::vector<std::string> keys)
{
  sc::scan_filter_request request;
  request.bool8_filters.push_back({2, std::move(keys)});
  request.routes = {
    sc::decode_route::bitpack_mask, sc::decode_route::bitpack_mask, sc::decode_route::dict_codes};
  request.delivered_prefix = kOutputWidth;
  return request;
}

struct decode_environment {
  using tracking = cucascade::memory::reservation_aware_resource_adaptor::AllocationTrackingScope;

  static cucascade::memory::gpu_memory_space_config config(bool per_stream)
  {
    int device = -1;
    REQUIRE(cudaGetDevice(&device) == cudaSuccess);
    cucascade::memory::gpu_memory_space_config config;
    config.device_id              = device;
    config.memory_capacity        = 16U << 20;
    config.per_stream_reservation = per_stream;
    return config;
  }

  explicit decode_environment(::cuda::stream_ref stream, bool per_stream = false)
    : space{config(per_stream)},
      stream{stream},
      tracking_scope{per_stream ? tracking::PER_STREAM : tracking::PER_THREAD}
  {
    auto* allocator =
      space.get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
    REQUIRE(allocator != nullptr);
    auto reservation = space.make_reservation(8U << 20);
    REQUIRE(reservation != nullptr);
    REQUIRE(allocator->attach_reservation_to_tracker(stream, std::move(reservation)));
  }

  ~decode_environment()
  {
    space.get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>()
      ->reset_stream_reservation(stream);
  }

  cucascade::memory::memory_space space;
  ::cuda::stream_ref stream;
  tracking tracking_scope;
};

sirius::decompress_result decompress_with(decode_environment& environment,
                                          simpatico::compressed_table const& chunk,
                                          sirius::pushdown_request request,
                                          ::cuda::stream_ref stream)
{
  sirius::decompression_pushdown_scan const scan{std::move(request)};
  return sirius::decompress_chunk(chunk,
                                  kSelected,
                                  &scan,
                                  sirius::decode_visibility_mask{},
                                  environment.space,
                                  environment.tracking_scope,
                                  stream,
                                  environment.space.get_default_allocator());
}

/// Join filters per selected slot, as a scan attaches them to a batch.
using slot_probes = std::vector<std::vector<sirius::membership_probe>>;

/// A join filter backed by @p keys and ordered by @p selectivity_rank, which defaults to the rank a
/// snapshot gives an exact hash set.
sirius::membership_probe key_set_probe(
  std::shared_ptr<sirius::op::sirius_dynamic_in_list_filter> keys,
  std::uint8_t selectivity_rank = 1)
{
  auto const num_keys = static_cast<std::uint64_t>(keys->size());
  return {[keys = std::move(keys)](cudf::column_view const& column,
                                   std::uint32_t const* prior_mask_words,
                                   ::cuda::stream_ref s,
                                   rmm::device_async_resource_ref mr) {
            return keys->compute_mask(column, prior_mask_words, /*device_id=*/0, s, mr);
          },
          selectivity_rank,
          num_keys};
}

/// A join filter that declines every batch, as one does when the slot's type is outside its key
/// family.
sirius::membership_probe declining_probe()
{
  return {[](cudf::column_view const&,
             std::uint32_t const*,
             ::cuda::stream_ref,
             rmm::device_async_resource_ref) { return std::unique_ptr<cudf::column>{}; },
          /*selectivity_rank=*/1,
          /*num_keys=*/10};
}

/// @p probe as the only join filter, on @p slot.
slot_probes one_probe(std::size_t slot, sirius::membership_probe probe)
{
  slot_probes probes(kSelected.size());
  probes[slot].push_back(std::move(probe));
  return probes;
}

/// Decodes @p chunk under @p request with @p probes attached through
/// `decompression_pushdown_scan::with_membership_probes`, the way a scan refreshes a batch's join
/// filters.
sirius::decompress_result decompress_with_probes(
  decode_environment& environment,
  simpatico::compressed_table const& chunk,
  sirius::pushdown_request request,
  slot_probes probes,
  ::cuda::stream_ref stream,
  sirius::decode_visibility_mask const& keep_mask = {})
{
  auto const scan = sirius::decompression_pushdown_scan{std::move(request)}.with_membership_probes(
    std::move(probes), /*generation=*/1);
  return sirius::decompress_chunk(chunk,
                                  kSelected,
                                  scan.get(),
                                  keep_mask,
                                  environment.space,
                                  environment.tracking_scope,
                                  stream,
                                  environment.space.get_default_allocator());
}

/// The codec and the engine read the same env gate; an explicit SIRIUS_EXP_FUSED_SCAN_FILTER=0
/// turns every filtered decode into a plain one.
bool gate_off()
{
  if (sc::decompression_pushdown_enabled()) { return false; }
  WARN("filtered-decode env gate off in this process; skipping delivery coverage");
  return true;
}

}  // namespace

//===----------------------------------------------------------------------===//
// Codec: simpatico::decompress_scan_filter
//===----------------------------------------------------------------------===//

TEST_CASE("an applied filtered decode delivers only its prefix", "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  auto const mr                   = cudf::get_current_device_resource_ref();
  auto const ct                   = compress_range_source(stream);
  simpatico::stream_pool pool;
  REQUIRE(pool.init(4));

  sc::scan_filter_result result;
  auto out = simpatico::decompress_scan_filter(
    ct, kSelected, range_request(kOutputWidth), result, pool, stream, mr);

  REQUIRE(result.applied);
  auto const expected = rows_where(in_filter_range);
  REQUIRE(result.survivor_count == static_cast<std::int64_t>(expected.v.size()));
  CHECK(result.routes.size() == kOutputWidth);
  require_columns(out->view(), expected, kOutputWidth, stream);
}

TEST_CASE("a filtered decode with no survivors delivers empty prefix columns",
          "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  auto const mr                   = cudf::get_current_device_resource_ref();
  auto const ct                   = compress_range_source(stream);
  simpatico::stream_pool pool;
  REQUIRE(pool.init(4));

  // "f" never exceeds 999, so this range keeps nothing.
  sc::scan_filter_result result;
  auto out = simpatico::decompress_scan_filter(
    ct, kSelected, range_request(kOutputWidth, {2000, 3000}), result, pool, stream, mr);

  REQUIRE(result.applied);
  CHECK(result.survivor_count == 0);
  REQUIRE(static_cast<std::size_t>(out->num_columns()) == kOutputWidth);
  for (cudf::size_type c = 0; c < out->num_columns(); ++c) {
    CHECK(out->view().column(c).size() == 0);
    CHECK(out->view().column(c).type() == cudf::data_type{cudf::type_id::INT32});
  }
}

TEST_CASE("a nullable-selection refusal after filtering returns every selected column",
          "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  auto const mr                   = cudf::get_current_device_resource_ref();
  auto const ct                   = compress_string_source(/*nullable=*/true, stream);
  simpatico::stream_pool pool;
  REQUIRE(pool.init(4));

  // The filter selects 250 rows, then the gather that would compact the null-masked "n" refuses.
  auto request      = range_request(kOutputWidth);
  request.routes[1] = sc::decode_route::full;
  sc::scan_filter_result result;
  auto out = simpatico::decompress_scan_filter(ct, kSelected, request, result, pool, stream, mr);

  REQUIRE_FALSE(result.applied);
  REQUIRE(result.status == sc::scan_filter_status::refused);
  require_columns(
    out->view(), string_rows_where(every_row, /*nullable=*/true), kSelected.size(), stream);
}

TEST_CASE("a delivered prefix outside [1, selected] refuses the filtered decode",
          "[fused_scan_filter][delivery]")
{
  // With the gate off every request is refused, which would pass this case for the wrong reason.
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  auto const mr                   = cudf::get_current_device_resource_ref();
  auto const ct                   = compress_range_source(stream);
  simpatico::stream_pool pool;
  REQUIRE(pool.init(4));

  auto const prefix = GENERATE(std::size_t{0}, std::size_t{4});
  CAPTURE(prefix);
  sc::scan_filter_result result;
  auto out = simpatico::decompress_scan_filter(
    ct, kSelected, range_request(prefix), result, pool, stream, mr);

  REQUIRE_FALSE(result.applied);
  CHECK(result.status == sc::scan_filter_status::refused);
  require_columns(out->view(), rows_where(every_row), kSelected.size(), stream);
}

TEST_CASE("an equality source past the delivered prefix is applied but not delivered",
          "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  auto const mr                   = cudf::get_current_device_resource_ref();
  auto const ct                   = compress_equality_source(stream);
  simpatico::stream_pool pool;
  REQUIRE(pool.init(4));

  sc::scan_filter_result result;
  auto out = simpatico::decompress_scan_filter(
    ct, kSelected, equality_request({key_string(5)}), result, pool, stream, mr);

  REQUIRE(result.applied);
  auto const expected = rows_where([](cudf::size_type i) { return i % 50 == 5; });
  REQUIRE(result.survivor_count == static_cast<std::int64_t>(expected.v.size()));
  // The BOOL8 answer belongs to the dropped slot, so nothing delivered is BOOL8.
  require_columns(out->view(), expected, kOutputWidth, stream);
}

TEST_CASE("an applied decode builds no survivor map when only a dropped column would read it",
          "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  auto const mr                   = cudf::get_current_device_resource_ref();
  auto const ct                   = compress_equality_source(stream);
  simpatico::stream_pool pool;
  REQUIRE(pool.init(4));

  // Ten of the fifty keys keep 20% of the rows. Above the index-walk ceiling the delivered bitpack
  // columns walk the mask bits, which leaves the dropped BOOL8 answer as the map's only reader.
  REQUIRE(sc::decompression_pushdown_index_walk_max_selectivity() < 0.2);
  std::vector<std::string> keys;
  for (cudf::size_type k = 0; k < 10; ++k) {
    keys.push_back(key_string(k));
  }
  sc::scan_filter_result result;
  auto out = simpatico::decompress_scan_filter(
    ct, kSelected, equality_request(std::move(keys)), result, pool, stream, mr);

  REQUIRE(result.applied);
  CHECK(result.row_indices.size() == 0);
  auto const expected = rows_where([](cudf::size_type i) { return i % 50 < 10; });
  REQUIRE(result.survivor_count == static_cast<std::int64_t>(expected.v.size()));
  require_columns(out->view(), expected, kOutputWidth, stream);
}

TEST_CASE("a delivered prefix covering every selected column changes nothing",
          "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  auto const mr                   = cudf::get_current_device_resource_ref();
  auto const ct                   = compress_range_source(stream);
  simpatico::stream_pool pool;
  REQUIRE(pool.init(4));

  sc::scan_filter_result unset_result;
  auto unset = simpatico::decompress_scan_filter(
    ct, kSelected, range_request(std::nullopt), unset_result, pool, stream, mr);
  sc::scan_filter_result full_result;
  auto full = simpatico::decompress_scan_filter(
    ct, kSelected, range_request(kSelected.size()), full_result, pool, stream, mr);

  REQUIRE(unset_result.applied);
  REQUIRE(full_result.applied);
  CHECK(full_result.survivor_count == unset_result.survivor_count);
  CHECK(full_result.routes == unset_result.routes);
  auto const expected = rows_where(in_filter_range);
  require_columns(unset->view(), expected, kSelected.size(), stream);
  require_columns(full->view(), expected, kSelected.size(), stream);
}

//===----------------------------------------------------------------------===//
// Engine: sirius::decompress_chunk
//===----------------------------------------------------------------------===//

TEST_CASE("a decode carrying the whole filter drops the trailing pure-filter column",
          "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  auto const per_stream           = GENERATE(false, true);
  CAPTURE(per_stream);
  decode_environment environment{stream, per_stream};
  auto const ct = compress_range_source(stream);

  auto const decoded = decompress_with(environment, ct, covered_request(kOutputWidth), stream);

  CHECK(decoded.outcome.row_filtered);
  CHECK(decoded.outcome.filter_only_columns_dropped);
  CHECK(decoded.outcome.predicate_columns.empty());
  require_columns(decoded.table->view(), rows_where(in_filter_range), kOutputWidth, stream);
}

TEST_CASE("a decode with a join filter still drops the pure-filter column",
          "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  decode_environment environment{stream};
  auto const ct = compress_range_source(stream);

  // A join filter is not part of the scan's own filter, so the range alone decides coverage.
  auto const decoded = decompress_with_probes(environment,
                                              ct,
                                              covered_request(kOutputWidth),
                                              one_probe(0, key_set_probe(make_key_set(stream))),
                                              stream);

  CHECK(decoded.outcome.row_filtered);
  CHECK(decoded.outcome.filter_only_columns_dropped);
  CHECK(decoded.outcome.predicate_columns.empty());
  // The join filter keeps v = i % 100 in its key set {0, 5, ..., 45}.
  auto const expected =
    rows_where([](cudf::size_type i) { return in_filter_range(i) && in_key_set(i % 100); });
  require_columns(decoded.table->view(), expected, kOutputWidth, stream);
}

TEST_CASE("a join filter past the delivered prefix is applied but not delivered",
          "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  decode_environment environment{stream};
  auto const ct = compress_range_source(stream);

  // The join filter tests the pure-filter column "f" while the selection is built; "f" is still
  // dropped from the delivered columns.
  auto const decoded = decompress_with_probes(environment,
                                              ct,
                                              covered_request(kOutputWidth),
                                              one_probe(2, key_set_probe(make_key_set(stream))),
                                              stream);

  CHECK(decoded.outcome.row_filtered);
  CHECK(decoded.outcome.filter_only_columns_dropped);
  auto const expected = rows_where(
    [](cudf::size_type i) { return in_filter_range(i) && in_key_set(i % kFilterModulus); });
  require_columns(decoded.table->view(), expected, kOutputWidth, stream);
}

TEST_CASE("a join filter that declines leaves the range's survivors and the drop",
          "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  decode_environment environment{stream};
  auto const ct = compress_range_source(stream);

  auto const decoded = decompress_with_probes(
    environment, ct, covered_request(kOutputWidth), one_probe(0, declining_probe()), stream);

  // The declined join filter removed nothing, so every survivor is a row of the range.
  CHECK(decoded.outcome.row_filtered);
  CHECK(decoded.outcome.filter_only_columns_dropped);
  require_columns(decoded.table->view(), rows_where(in_filter_range), kOutputWidth, stream);
}

TEST_CASE("a decode whose only source is a join filter keeps every column",
          "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  decode_environment environment{stream};
  auto const ct = compress_range_source(stream);

  sirius::pushdown_request request;
  request.columns.resize(kSelected.size());
  request.output_prefix_width = kOutputWidth;
  SECTION("the request claims no coverage") { request.ranges_cover_whole_filter = false; }
  // Coverage needs a range the decode evaluates; a claim with none is not enough.
  SECTION("the request claims coverage with no range") { request.ranges_cover_whole_filter = true; }
  auto const decoded = decompress_with_probes(
    environment, ct, std::move(request), one_probe(0, key_set_probe(make_key_set(stream))), stream);

  CHECK_FALSE(decoded.outcome.row_filtered);
  CHECK_FALSE(decoded.outcome.filter_only_columns_dropped);
  auto const expected = rows_where([](cudf::size_type i) { return in_key_set(i % 100); });
  require_columns(decoded.table->view(), expected, kSelected.size(), stream);
}

TEST_CASE("a join filter does not complete a partly carried filter",
          "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  decode_environment environment{stream};
  auto const ct = compress_range_source(stream);

  auto const in_range_and_keys = [](cudf::size_type i) {
    return in_filter_range(i) && in_key_set(i % 100);
  };
  auto request = covered_request(kOutputWidth);
  SECTION("the ranges are not the scan's whole filter")
  {
    request.ranges_cover_whole_filter = false;
  }
  // A provably empty conjunct is dropped from the decode and left to the scan, which rejects
  // every row.
  SECTION("a conjunct is dropped") { request.columns[1].range = sirius::decode_range{5, 3}; }
  auto const decoded = decompress_with_probes(
    environment, ct, std::move(request), one_probe(0, key_set_probe(make_key_set(stream))), stream);

  CHECK_FALSE(decoded.outcome.row_filtered);
  CHECK_FALSE(decoded.outcome.filter_only_columns_dropped);
  require_columns(decoded.table->view(), rows_where(in_range_and_keys), kSelected.size(), stream);
}

TEST_CASE("a nullable-selection refusal with a join filter keeps every row and column",
          "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  decode_environment environment{stream};
  auto const ct = compress_string_source(/*nullable=*/true, stream);

  auto const decoded = decompress_with_probes(environment,
                                              ct,
                                              covered_request(kOutputWidth),
                                              one_probe(0, key_set_probe(make_key_set(stream))),
                                              stream);

  CHECK_FALSE(decoded.outcome.row_filtered);
  CHECK_FALSE(decoded.outcome.filter_only_columns_dropped);
  require_columns(decoded.table->view(),
                  string_rows_where(every_row, /*nullable=*/true),
                  kSelected.size(),
                  stream);
}

TEST_CASE("a decode with a join filter and a visibility mask consumes both",
          "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  decode_environment environment{stream};
  auto const ct = compress_range_source(stream);

  // Keep every even row.
  static_assert(kRows % 32 == 0);
  sirius::decode_visibility_mask keep_even;
  keep_even.words     = std::make_shared<std::uint32_t[]>(kRows / 32, 0x55555555U);
  keep_even.row_count = kRows;
  auto const decoded  = decompress_with_probes(environment,
                                              ct,
                                              covered_request(kOutputWidth),
                                              one_probe(0, key_set_probe(make_key_set(stream))),
                                              stream,
                                              keep_even);

  CHECK(decoded.outcome.row_filtered);
  CHECK(decoded.outcome.visibility_mask_applied);
  CHECK(decoded.outcome.filter_only_columns_dropped);
  auto const expected = rows_where(
    [](cudf::size_type i) { return i % 2 == 0 && in_filter_range(i) && in_key_set(i % 100); });
  require_columns(decoded.table->view(), expected, kOutputWidth, stream);
}

TEST_CASE("join filters past the decode's cap leave the drop in place",
          "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  INFO(
    "fused_gate_armer sets SIRIUS_EXP_FUSED_SCAN_MAX_MEMBER=2 unless the environment overrides it");
  REQUIRE(sc::decompression_pushdown_max_membership_sources() == 2);
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  decode_environment environment{stream};
  auto const ct = compress_range_source(stream);

  // Listed out of rank order: the decode sorts them strongest first and carries the first two.
  slot_probes probes(kSelected.size());
  probes[1].push_back(key_set_probe(make_key_set(stream, /*step=*/3), /*selectivity_rank=*/2));
  probes[0].push_back(key_set_probe(make_key_set(stream, /*step=*/5), /*selectivity_rank=*/0));
  probes[1].push_back(key_set_probe(make_key_set(stream, /*step=*/2), /*selectivity_rank=*/1));
  auto const decoded = decompress_with_probes(
    environment, ct, covered_request(kOutputWidth), std::move(probes), stream);

  CHECK(decoded.outcome.row_filtered);
  CHECK(decoded.outcome.filter_only_columns_dropped);
  // The capped multiples-of-3 filter on "k" did not run.
  auto const expected = rows_where([](cudf::size_type i) {
    return in_filter_range(i) && in_key_set(i % 100) && in_key_set(i % 50, 2);
  });
  require_columns(decoded.table->view(), expected, kOutputWidth, stream);
}

TEST_CASE("a decode carrying part of the filter keeps every column",
          "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  decode_environment environment{stream};
  auto const ct = compress_range_source(stream);

  auto request                      = covered_request(kOutputWidth);
  request.ranges_cover_whole_filter = false;
  auto const decoded                = decompress_with(environment, ct, std::move(request), stream);

  CHECK_FALSE(decoded.outcome.row_filtered);
  CHECK_FALSE(decoded.outcome.filter_only_columns_dropped);
  require_columns(decoded.table->view(), rows_where(in_filter_range), kSelected.size(), stream);
}

TEST_CASE("a decode without an output prefix keeps every column", "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  decode_environment environment{stream};
  auto const ct = compress_range_source(stream);

  auto const decoded = decompress_with(environment, ct, covered_request(std::nullopt), stream);

  CHECK(decoded.outcome.row_filtered);
  CHECK_FALSE(decoded.outcome.filter_only_columns_dropped);
  require_columns(decoded.table->view(), rows_where(in_filter_range), kSelected.size(), stream);
}

TEST_CASE("each chunk of one scan decides its own delivery", "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  decode_environment environment{stream};
  auto const mr       = environment.space.get_default_allocator();
  auto const clean    = compress_string_source(/*nullable=*/false, stream);
  auto const nullable = compress_string_source(/*nullable=*/true, stream);
  sirius::decompression_pushdown_scan const scan{covered_request(kOutputWidth)};

  auto const dropped = sirius::decompress_chunk(clean,
                                                kSelected,
                                                &scan,
                                                sirius::decode_visibility_mask{},
                                                environment.space,
                                                environment.tracking_scope,
                                                stream,
                                                mr);
  CHECK(dropped.outcome.row_filtered);
  CHECK(dropped.outcome.filter_only_columns_dropped);
  require_columns(dropped.table->view(),
                  string_rows_where(in_filter_range, /*nullable=*/false),
                  kOutputWidth,
                  stream);

  // The same request over a chunk whose "n" is null-masked refuses nullable selection after
  // filtering, so the scan gets the whole chunk back and filters it itself.
  auto const kept = sirius::decompress_chunk(nullable,
                                             kSelected,
                                             &scan,
                                             sirius::decode_visibility_mask{},
                                             environment.space,
                                             environment.tracking_scope,
                                             stream,
                                             mr);
  CHECK_FALSE(kept.outcome.row_filtered);
  CHECK_FALSE(kept.outcome.filter_only_columns_dropped);
  require_columns(
    kept.table->view(), string_rows_where(every_row, /*nullable=*/true), kSelected.size(), stream);
}

TEST_CASE("an equality-only pure-filter column past the prefix is dropped with its answer",
          "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  decode_environment environment{stream};
  auto const ct = compress_equality_source(stream);

  // The range sits on output column "v"; "s" is read only by its equality.
  sirius::pushdown_request request;
  request.columns.resize(kSelected.size());
  request.columns[0].range          = sirius::decode_range{0, 49};
  request.columns[2].equals_any     = {key_string(5)};
  request.ranges_cover_whole_filter = true;
  request.output_prefix_width       = kOutputWidth;
  auto const decoded                = decompress_with(environment, ct, std::move(request), stream);

  CHECK(decoded.outcome.row_filtered);
  CHECK(decoded.outcome.filter_only_columns_dropped);
  CHECK(decoded.outcome.predicate_columns.empty());
  CHECK(decoded.outcome.predicates_enforced);
  auto const expected = rows_where([](cudf::size_type i) { return i % 100 <= 49 && i % 50 == 5; });
  require_columns(decoded.table->view(), expected, kOutputWidth, stream);
}
