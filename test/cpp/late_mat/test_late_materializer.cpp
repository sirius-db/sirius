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

// Engine-level unit tests for the late-materialization v1 stack
// (SIRIUS_EXP_LATE_MAT): pin_entry_handle generation semantics (host-only),
// pinned_table_layout construction, prepare_selection /
// prepare_selection_from_batch edge geometries and refusal paths, and
// materialize() against host-computed gathers over UNCOMPRESSED origins
// (INT32, DOUBLE, STRING) — the q10-class path. Compressed-origin decode
// correctness is covered at the simpatico layer
// (test_late_mat_row_decode.cpp / test_late_mat_edge_geometry.cpp); this
// file covers the orchestration above it: batch splitting, assembly order,
// restore-rank semantics, empty/degenerate selections, and every documented
// v1 refusal (nullable source, generation mismatch, layout drift, >2^31).
//
// The refusal tests matter most: each must throw loudly (std::runtime_error)
// — the v1 contract is "refuse, never silently wrong".

#include <catch.hpp>

#include "late_mat/column_origin.hpp"
#include "late_mat/late_materializer.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

using sirius::late_mat::column_origin;
using sirius::late_mat::pin_entry_handle;
using sirius::late_mat::pinned_column_view;
using sirius::late_mat::pinned_table_layout;
using sirius::late_mat::row_id_list;
using sirius::late_mat::row_range;
using sirius::late_mat::row_selection;
using sirius::late_mat::row_selection_kind;

namespace {

rmm::cuda_stream_view test_stream() { return rmm::cuda_stream_view{}; }

// Column builders on the current device resource (plain rmm, no cucascade
// machinery needed — materialize() takes stream + mr explicitly).
template <typename T>
std::unique_ptr<cudf::column> make_numeric_column_from(std::vector<T> const& host,
                                                       cudf::type_id id)
{
  rmm::device_buffer buf(host.data(), host.size() * sizeof(T), test_stream());
  return std::make_unique<cudf::column>(cudf::data_type{id},
                                        static_cast<cudf::size_type>(host.size()),
                                        std::move(buf), rmm::device_buffer{}, 0);
}

std::unique_ptr<cudf::column> make_strings_column_from(std::vector<std::string> const& host)
{
  std::vector<char> chars;
  std::vector<cudf::size_type> offsets{0};
  for (auto const& s : host) {
    chars.insert(chars.end(), s.begin(), s.end());
    offsets.push_back(static_cast<cudf::size_type>(chars.size()));
  }
  auto offsets_col = make_numeric_column_from<cudf::size_type>(offsets, cudf::type_id::INT32);
  rmm::device_buffer char_buf(chars.data(), chars.size(), test_stream());
  return cudf::make_strings_column(static_cast<cudf::size_type>(host.size()),
                                   std::move(offsets_col), std::move(char_buf), 0,
                                   rmm::device_buffer{});
}

rmm::device_buffer upload_u64(std::vector<std::uint64_t> const& v)
{
  return rmm::device_buffer(v.data(), v.size() * sizeof(std::uint64_t), test_stream());
}

template <typename T>
std::vector<T> download_numeric(cudf::column_view const& col)
{
  std::vector<T> out(static_cast<std::size_t>(col.size()));
  cudaStreamSynchronize(test_stream().value());
  cudaMemcpy(out.data(), col.data<T>(), out.size() * sizeof(T), cudaMemcpyDeviceToHost);
  return out;
}

std::vector<std::string> download_strings(cudf::column_view const& col)
{
  cudaStreamSynchronize(test_stream().value());
  cudf::strings_column_view scv(col);
  std::vector<cudf::size_type> offsets(static_cast<std::size_t>(col.size()) + 1);
  cudaMemcpy(offsets.data(), scv.offsets().data<cudf::size_type>(),
             offsets.size() * sizeof(cudf::size_type), cudaMemcpyDeviceToHost);
  std::vector<char> chars(static_cast<std::size_t>(offsets.back()));
  if (!chars.empty()) {
    cudaMemcpy(chars.data(), scv.chars_begin(test_stream()), chars.size(),
               cudaMemcpyDeviceToHost);
  }
  std::vector<std::string> out;
  out.reserve(static_cast<std::size_t>(col.size()));
  for (std::size_t i = 0; i + 1 < offsets.size(); ++i) {
    out.emplace_back(chars.data() + offsets[i],
                     static_cast<std::size_t>(offsets[i + 1] - offsets[i]));
  }
  return out;
}

// A 3-batch uncompressed INT32 origin: 1000 + 0 + 1025 rows (empty batch mid
// layout, non-chunk-aligned tail), values = global row id * 3 + 7.
struct uncompressed_fixture {
  std::vector<std::unique_ptr<cudf::column>> storage;
  pinned_column_view view;
  pinned_table_layout layout;
  std::vector<std::int32_t> host_values;  // pin-order reference

  explicit uncompressed_fixture(std::uint64_t generation = 1)
  {
    const std::vector<std::int64_t> rows = {1000, 0, 1025};
    layout = pinned_table_layout::from_batch_rows(rows, generation);
    view.pin_generation = generation;
    view.dtype          = cudf::data_type{cudf::type_id::INT32};
    std::int64_t gid    = 0;
    for (auto const r : rows) {
      std::vector<std::int32_t> host(static_cast<std::size_t>(r));
      for (auto& v : host) {
        v = static_cast<std::int32_t>(gid * 3 + 7);
        host_values.push_back(v);
        ++gid;
      }
      storage.push_back(make_numeric_column_from<std::int32_t>(host, cudf::type_id::INT32));
      sirius::late_mat::batch_source src;
      src.uncompressed = storage.back()->view();
      src.num_rows     = r;
      view.batches.push_back(src);
    }
  }
};

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Host-only: pin_entry_handle / column_origin generation semantics
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("late_mat pin_entry_handle fails closed across lifecycle", "[late_mat]")
{
  // The entry pointer never has to be a real pinned_entry for these checks —
  // resolve() only gates and returns it.
  auto* fake_entry = reinterpret_cast<sirius::scan_manager::pinned_entry const*>(0x1234);

  pin_entry_handle handle("test_entry", /*generation=*/1);
  handle.set_entry(fake_entry);

  SECTION("happy path resolves at the captured generation")
  {
    CHECK(handle.resolve(1) == fake_entry);
    CHECK(handle.generation() == 1);
  }
  SECTION("expected generation 0 never resolves (zero-initialized origins)")
  {
    CHECK(handle.resolve(0) == nullptr);
  }
  SECTION("wrong generation never resolves")
  {
    CHECK(handle.resolve(2) == nullptr);
  }
  SECTION("invalidate (unpin / replacing re-pin) fails closed")
  {
    handle.invalidate();
    CHECK(handle.resolve(1) == nullptr);
    CHECK(handle.generation() == 0);
  }
  SECTION("bump_generation (in-place merge) fails pre-merge origins closed")
  {
    handle.bump_generation();
    CHECK(handle.resolve(1) == nullptr);   // stale origin
    CHECK(handle.resolve(2) == fake_entry);  // post-merge capture works
  }
  SECTION("column_origin with empty handle fails closed")
  {
    column_origin origin;
    CHECK_FALSE(origin.has_origin());
    CHECK(origin.resolve() == nullptr);
  }
  SECTION("column_origin resolve is generation-checked")
  {
    auto shared = std::make_shared<pin_entry_handle>("e", 3);
    shared->set_entry(fake_entry);
    column_origin origin{shared, 0, 3};
    CHECK(origin.resolve() == fake_entry);
    shared->bump_generation();
    CHECK(origin.resolve() == nullptr);
    shared->invalidate();
    CHECK(origin.resolve() == nullptr);
  }
}

TEST_CASE("late_mat pinned_table_layout::from_batch_rows", "[late_mat]")
{
  SECTION("prefix construction")
  {
    auto layout = pinned_table_layout::from_batch_rows({5, 0, 7}, 9);
    CHECK(layout.pin_generation == 9);
    CHECK(layout.batch_rows == std::vector<std::int64_t>{5, 0, 7});
    CHECK(layout.batch_row_start == std::vector<std::int64_t>{0, 5, 5, 12});
  }
  SECTION("empty layout")
  {
    auto layout = pinned_table_layout::from_batch_rows({}, 1);
    CHECK(layout.batch_row_start == std::vector<std::int64_t>{0});
  }
  SECTION("negative batch rows refuse")
  {
    CHECK_THROWS_AS(pinned_table_layout::from_batch_rows({5, -1}, 1), std::runtime_error);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// prepare_selection: id-list entry point
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("late_mat prepare_selection edge selections", "[late_mat]")
{
  auto mr     = rmm::mr::get_current_device_resource_ref();
  auto stream = test_stream();
  uncompressed_fixture fx;

  SECTION("empty id list prepares an all-empty selection")
  {
    row_id_list ids;  // count 0
    auto sel = sirius::late_mat::prepare_selection(fx.layout, ids, stream, mr);
    CHECK(sel->total_survivors == 0);
    CHECK_FALSE(sel->needs_restore());
    for (auto const& b : sel->batches) {
      CHECK(b.rows.num_survivors == 0);
    }
  }
  SECTION("single id in the tail batch")
  {
    const std::vector<std::uint64_t> host_ids = {2024};  // batch 2, local 1024 (last row)
    auto buf = upload_u64(host_ids);
    row_id_list ids{static_cast<std::uint64_t const*>(buf.data()), 1, true};
    auto sel = sirius::late_mat::prepare_selection(fx.layout, ids, stream, mr);
    REQUIRE(sel->total_survivors == 1);
    CHECK(sel->batches[0].rows.num_survivors == 0);
    CHECK(sel->batches[1].rows.num_survivors == 0);
    CHECK(sel->batches[2].rows.num_survivors == 1);
    CHECK(sel->out_base == std::vector<std::int64_t>{0, 0, 0, 1});
  }
  SECTION("id exactly at a batch boundary lands in the later batch")
  {
    const std::vector<std::uint64_t> host_ids = {999, 1000};
    auto buf = upload_u64(host_ids);
    row_id_list ids{static_cast<std::uint64_t const*>(buf.data()), 2, true};
    auto sel = sirius::late_mat::prepare_selection(fx.layout, ids, stream, mr);
    CHECK(sel->batches[0].rows.num_survivors == 1);  // 999
    CHECK(sel->batches[1].rows.num_survivors == 0);  // empty batch
    CHECK(sel->batches[2].rows.num_survivors == 1);  // 1000 = batch2 local 0
  }
  SECTION("out-of-range id refuses")
  {
    const std::vector<std::uint64_t> host_ids = {0, 2025};  // total rows = 2025
    auto buf = upload_u64(host_ids);
    row_id_list ids{static_cast<std::uint64_t const*>(buf.data()), 2, true};
    CHECK_THROWS_AS(sirius::late_mat::prepare_selection(fx.layout, ids, stream, mr),
                    std::runtime_error);
  }
  SECTION(">2^31-1 ids refuse before any device work")
  {
    rmm::device_buffer tiny(8, stream, mr);
    row_id_list ids{static_cast<std::uint64_t const*>(tiny.data()),
                    (std::int64_t{1} << 31), false};
    CHECK_THROWS_AS(sirius::late_mat::prepare_selection(fx.layout, ids, stream, mr),
                    std::runtime_error);
  }
  SECTION("negative count refuses")
  {
    row_id_list ids{nullptr, -1, false};
    CHECK_THROWS_AS(sirius::late_mat::prepare_selection(fx.layout, ids, stream, mr),
                    std::runtime_error);
  }
  SECTION("non-null count with null pointer refuses")
  {
    row_id_list ids{nullptr, 4, false};
    CHECK_THROWS_AS(sirius::late_mat::prepare_selection(fx.layout, ids, stream, mr),
                    std::runtime_error);
  }
  SECTION("inconsistent layout refuses")
  {
    pinned_table_layout broken = fx.layout;
    broken.batch_row_start.pop_back();
    row_id_list ids;
    CHECK_THROWS_AS(sirius::late_mat::prepare_selection(broken, ids, stream, mr),
                    std::runtime_error);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// prepare_selection_from_batch: annotation-contract forms
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("late_mat prepare_selection_from_batch forms", "[late_mat]")
{
  auto mr     = rmm::mr::get_current_device_resource_ref();
  auto stream = test_stream();
  uncompressed_fixture fx;

  SECTION("dense with unset ranges (annotation carrier default)")
  {
    std::vector<row_selection> sels(3);  // all dense, all-zero ranges
    auto sel =
      sirius::late_mat::prepare_selection_from_batch(fx.layout, sels, stream, mr);
    CHECK(sel->total_survivors == 2025);
    CHECK(sel->batches[0].dense);
    CHECK(sel->batches[2].dense);
    CHECK_FALSE(sel->needs_restore());
  }
  SECTION("dense with a filled range must agree with the layout")
  {
    std::vector<row_selection> sels(3);
    sels[0] = row_selection::make_dense(row_range{0, 999});  // wrong: batch 0 has 1000
    CHECK_THROWS_AS(
      sirius::late_mat::prepare_selection_from_batch(fx.layout, sels, stream, mr),
      std::runtime_error);
  }
  SECTION("span size must match the layout")
  {
    std::vector<row_selection> sels(2);
    CHECK_THROWS_AS(
      sirius::late_mat::prepare_selection_from_batch(fx.layout, sels, stream, mr),
      std::runtime_error);
  }
  SECTION("mask form: hand-built wave-1 geometry over batch 0")
  {
    // Batch 0 (1000 rows, 1 chunk): keep rows {0, 31, 999}.
    const std::int64_t n_rows = 1000;
    std::vector<std::uint32_t> words(32, 0u);
    for (std::uint32_t r : {0u, 31u, 999u}) words[r / 32] |= 1u << (r % 32);
    std::vector<std::uint32_t> choffs = {0u, 3u};
    row_selection mask_sel;
    mask_sel.kind           = row_selection_kind::mask;
    mask_sel.range          = row_range{0, n_rows};
    mask_sel.survivor_count = 3;
    mask_sel.mask_words     = std::make_shared<rmm::device_buffer>(
      words.data(), words.size() * 4, stream);
    mask_sel.chunk_offsets = std::make_shared<rmm::device_buffer>(
      choffs.data(), choffs.size() * 4, stream);

    std::vector<row_selection> sels(3);
    sels[0]        = mask_sel;
    // Batches 1/2: id_list of nothing is invalid; use dense-with-zero rows for
    // batch 1 (0-row batch) and an explicit empty mask is not allowed — leave
    // dense (batch1 contributes 0, batch2 contributes all rows).
    auto sel = sirius::late_mat::prepare_selection_from_batch(fx.layout, sels, stream, mr);
    CHECK(sel->total_survivors == 3 + 0 + 1025);
    CHECK(sel->batches[0].rows.num_survivors == 3);
    CHECK(sel->batches[0].mask_words != nullptr);
    // The prepare-built int32 expansion must equal the set bits.
    auto got = std::vector<std::int32_t>(3);
    cudaStreamSynchronize(stream.value());
    cudaMemcpy(got.data(), sel->batches[0].local_indices.data(), 3 * 4,
               cudaMemcpyDeviceToHost);
    CHECK(got == std::vector<std::int32_t>{0, 31, 999});
  }
  SECTION("mask form missing buffers refuses")
  {
    std::vector<row_selection> sels(3);
    sels[0].kind           = row_selection_kind::mask;
    sels[0].survivor_count = 3;  // no buffers
    CHECK_THROWS_AS(
      sirius::late_mat::prepare_selection_from_batch(fx.layout, sels, stream, mr),
      std::runtime_error);
  }
  SECTION("id_list form: batch-local ascending int32")
  {
    std::vector<std::int32_t> local = {0, 512, 1024};  // batch 2 rows (1025 rows)
    row_selection idl;
    idl.kind    = row_selection_kind::id_list;
    idl.range   = row_range{1000, 1025};
    idl.num_ids = 3;
    idl.row_ids = std::make_shared<rmm::device_buffer>(local.data(), local.size() * 4,
                                                       stream);
    std::vector<row_selection> sels(3);
    sels[2] = idl;
    // Batches 0/1 dense.
    auto sel = sirius::late_mat::prepare_selection_from_batch(fx.layout, sels, stream, mr);
    CHECK(sel->total_survivors == 1000 + 0 + 3);
    CHECK(sel->batches[2].rows.num_survivors == 3);
    CHECK(sel->batches[2].rows.num_touched == 2);  // chunks 0 and 1
  }
  SECTION("live rows beyond the batch refuse")
  {
    std::vector<row_selection> sels(3);
    sels[1].kind    = row_selection_kind::id_list;  // batch 1 has 0 rows
    sels[1].num_ids = 1;
    std::int32_t zero = 0;
    sels[1].row_ids = std::make_shared<rmm::device_buffer>(&zero, 4, stream);
    CHECK_THROWS_AS(
      sirius::late_mat::prepare_selection_from_batch(fx.layout, sels, stream, mr),
      std::runtime_error);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// materialize: uncompressed origins vs host gather
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("late_mat materialize uncompressed vs host reference", "[late_mat]")
{
  auto mr     = rmm::mr::get_current_device_resource_ref();
  auto stream = test_stream();
  uncompressed_fixture fx;

  auto materialize_ids = [&](std::vector<std::uint64_t> const& host_ids, bool sorted_unique) {
    auto buf = upload_u64(host_ids);
    row_id_list ids{static_cast<std::uint64_t const*>(buf.data()),
                    static_cast<std::int64_t>(host_ids.size()), sorted_unique};
    auto sel = sirius::late_mat::prepare_selection(fx.layout, ids, stream, mr);
    auto col = sirius::late_mat::materialize(fx.view, *sel, stream, mr);
    return download_numeric<std::int32_t>(col->view());
  };
  auto host_gather = [&](std::vector<std::uint64_t> const& host_ids) {
    std::vector<std::int32_t> out;
    out.reserve(host_ids.size());
    for (auto id : host_ids) out.push_back(fx.host_values[static_cast<std::size_t>(id)]);
    return out;
  };

  SECTION("empty selection materializes an empty column of the origin dtype")
  {
    row_id_list ids;
    auto sel = sirius::late_mat::prepare_selection(fx.layout, ids, stream, mr);
    auto col = sirius::late_mat::materialize(fx.view, *sel, stream, mr);
    CHECK(col->size() == 0);
    CHECK(col->type().id() == cudf::type_id::INT32);
  }
  SECTION("single row (first / last / boundary)")
  {
    for (std::uint64_t id : {std::uint64_t{0}, std::uint64_t{999}, std::uint64_t{1000},
                             std::uint64_t{2024}}) {
      CHECK(materialize_ids({id}, true) == host_gather({id}));
    }
  }
  SECTION("full density takes the deep-copy path and matches pin order")
  {
    std::vector<std::uint64_t> all(2025);
    std::iota(all.begin(), all.end(), 0);
    CHECK(materialize_ids(all, true) == host_gather(all));
  }
  SECTION("unsorted duplicate-carrying ids restore caller order (gather semantics)")
  {
    const std::vector<std::uint64_t> ids = {2024, 3, 1000, 3, 999, 2024, 0, 1500, 3};
    CHECK(materialize_ids(ids, false) == host_gather(ids));
  }
  SECTION("sparse spread across batches, ascending")
  {
    const std::vector<std::uint64_t> ids = {0, 31, 999, 1000, 1023, 1024, 2024};
    CHECK(materialize_ids(ids, true) == host_gather(ids));
  }
  SECTION("dense-from-batch equals the whole pin")
  {
    std::vector<row_selection> sels(3);
    auto sel = sirius::late_mat::prepare_selection_from_batch(fx.layout, sels, stream, mr);
    auto col = sirius::late_mat::materialize(fx.view, *sel, stream, mr);
    CHECK(download_numeric<std::int32_t>(col->view()) == fx.host_values);
  }
}

TEST_CASE("late_mat materialize strings origin", "[late_mat]")
{
  auto mr     = rmm::mr::get_current_device_resource_ref();
  auto stream = test_stream();

  // Two string batches (uneven lengths incl. empty strings) — the q10 shape.
  std::vector<std::string> b0 = {"alpha", "", "carol#0003", "d", "eeeeeeeeeeeeeeee"};
  std::vector<std::string> b1 = {"zulu", "yankee", "", "x-ray"};
  std::vector<std::string> all = b0;
  all.insert(all.end(), b1.begin(), b1.end());

  auto layout = pinned_table_layout::from_batch_rows(
    {static_cast<std::int64_t>(b0.size()), static_cast<std::int64_t>(b1.size())}, 5);
  pinned_column_view view;
  view.pin_generation = 5;
  view.dtype          = cudf::data_type{cudf::type_id::STRING};
  std::vector<std::unique_ptr<cudf::column>> storage;
  for (auto const* batch : {&b0, &b1}) {
    storage.push_back(make_strings_column_from(*batch));
    sirius::late_mat::batch_source src;
    src.uncompressed = storage.back()->view();
    src.num_rows     = static_cast<std::int64_t>(batch->size());
    view.batches.push_back(src);
  }

  const std::vector<std::uint64_t> ids = {8, 1, 4, 1, 5, 0};  // unsorted + dups + ""
  auto buf = upload_u64(ids);
  row_id_list rid{static_cast<std::uint64_t const*>(buf.data()),
                  static_cast<std::int64_t>(ids.size()), false};
  auto sel = sirius::late_mat::prepare_selection(layout, rid, stream, mr);
  auto col = sirius::late_mat::materialize(view, *sel, stream, mr);
  std::vector<std::string> expect;
  for (auto id : ids) expect.push_back(all[static_cast<std::size_t>(id)]);
  CHECK(download_strings(col->view()) == expect);
}

TEST_CASE("late_mat materialize refusal paths", "[late_mat]")
{
  auto mr     = rmm::mr::get_current_device_resource_ref();
  auto stream = test_stream();
  uncompressed_fixture fx;

  const std::vector<std::uint64_t> host_ids = {1};
  auto buf = upload_u64(host_ids);
  row_id_list ids{static_cast<std::uint64_t const*>(buf.data()), 1, true};
  auto sel = sirius::late_mat::prepare_selection(fx.layout, ids, stream, mr);

  SECTION("generation mismatch refuses (re-pin between prepare and materialize)")
  {
    pinned_column_view stale = fx.view;
    stale.pin_generation     = fx.view.pin_generation + 1;
    CHECK_THROWS_AS(sirius::late_mat::materialize(stale, *sel, stream, mr),
                    std::runtime_error);
  }
  SECTION("origin/selection batch count mismatch refuses")
  {
    pinned_column_view fewer = fx.view;
    fewer.batches.pop_back();
    CHECK_THROWS_AS(sirius::late_mat::materialize(fewer, *sel, stream, mr),
                    std::runtime_error);
  }
  SECTION("origin batch row-count drift refuses")
  {
    pinned_column_view drift = fx.view;
    drift.batches[0].num_rows -= 1;
    CHECK_THROWS_AS(sirius::late_mat::materialize(drift, *sel, stream, mr),
                    std::runtime_error);
  }
  SECTION("nullable source column refuses (v1 non-null contract)")
  {
    // Rebuild batch 0 with a null at row 1 and select that batch.
    std::vector<std::int32_t> host(1000, 3);
    auto nullable = make_numeric_column_from<std::int32_t>(host, cudf::type_id::INT32);
    auto mask     = cudf::create_null_mask(1000, cudf::mask_state::ALL_VALID);
    cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask.data()), 1, 2, false);
    nullable->set_null_mask(std::move(mask), 1);
    pinned_column_view withnull = fx.view;
    withnull.batches[0].uncompressed = nullable->view();
    CHECK_THROWS_AS(sirius::late_mat::materialize(withnull, *sel, stream, mr),
                    std::runtime_error);
  }
}
