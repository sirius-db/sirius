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

// [late_mat][materialize_compressed] — deferring a column out of a compressed
// pin, including a nullable one. GPU required.
//
// The pinned column holds row i at value i and is null on a fixed, irregular
// set of rows, so both halves of a materialized row name the row it was read
// from: the value IS the global id, and the validity is a function of that id.
// A route that decodes the right values but pairs them with another row's
// validity produces a column that is right in one half and wrong in the other,
// which is the failure this file exists to catch — the stored bitmask describes
// every row of a chunk, while a compacted decode returns only the selected ones.
//
// materialize_compressed picks between four routes and cannot be asked which one
// it took, so each case pins the choice from below: it calls the simpatico decode
// the route rests on and asserts it serves (or refuses) for that plan, then goes
// through materialize() for the values and validity. The plans are chosen to
// classify differently — a bitpack root takes the sparse walk, a delta root has
// no random access and falls to the mask route, and an ANS root has no compacted
// route at all and falls to the full decode.

#include "operator/operator_test_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>

#include <api/simpatico_codegen.hpp>
#include <catch.hpp>
#include <codegen/selection/chunk_row_set.hpp>
#include <codegen/selection/selection.hpp>
#include <compression/compressed_representation.hpp>
#include <compression/device_compressed_blob.hpp>
#include <late_mat/materialize.hpp>

#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

using sirius::late_mat::batch_source;
using sirius::late_mat::materialize;
using sirius::late_mat::pinned_column_view;
using sirius::late_mat::pinned_table_layout;
using sirius::late_mat::prepared_selection;
using sirius::late_mat::row_id_list;

namespace {

constexpr std::int64_t kChunk = 1024;

/// Which rows of the fixture are null: an irregular pattern, so a validity
/// gather that is off by one row, or that reads the selection's position
/// instead of the row it selected, cannot coincide with the right answer.
bool row_is_null(std::int64_t global_id) { return global_id % 7 == 3 || global_id % 11 == 0; }

/// One batch of the fixture as a plain cuDF column: row i holds i, null per
/// row_is_null. The values under a null row are still written, so a decode that
/// drops validity is caught by the validity check alone rather than by the
/// values changing too.
std::unique_ptr<cudf::column> make_batch(std::int64_t first_row,
                                         std::int64_t rows,
                                         bool nullable,
                                         rmm::cuda_stream_view stream)
{
  std::vector<std::int32_t> values(static_cast<std::size_t>(rows));
  std::iota(values.begin(), values.end(), static_cast<std::int32_t>(first_row));

  auto col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32},
    static_cast<cudf::size_type>(rows),
    nullable ? cudf::mask_state::UNINITIALIZED : cudf::mask_state::UNALLOCATED,
    stream);
  cudaMemcpyAsync(col->mutable_view().data<std::int32_t>(),
                  values.data(),
                  values.size() * sizeof(std::int32_t),
                  cudaMemcpyHostToDevice,
                  stream.value());

  if (nullable) {
    auto const words = static_cast<std::size_t>(cudf::bitmask_allocation_size_bytes(
                         static_cast<cudf::size_type>(rows))) /
                       sizeof(cudf::bitmask_type);
    std::vector<cudf::bitmask_type> host_mask(words, 0);
    cudf::size_type nulls = 0;
    for (std::int64_t r = 0; r < rows; ++r) {
      if (row_is_null(first_row + r)) {
        ++nulls;
      } else {
        host_mask[static_cast<std::size_t>(r) / 32] |= (1u << (r % 32));
      }
    }
    cudaMemcpyAsync(col->mutable_view().null_mask(),
                    host_mask.data(),
                    host_mask.size() * sizeof(cudf::bitmask_type),
                    cudaMemcpyHostToDevice,
                    stream.value());
    cudaStreamSynchronize(stream.value());
    col->set_null_count(nulls);
  }
  cudaStreamSynchronize(stream.value());
  return col;
}

/// A pinned column whose batches are Simpatico-compressed under one plan.
///
/// The blob carries only the compressed_table: its leaves are owned by the
/// plan's own representations rather than sliced out of a staged payload, which
/// is all a decode needs and all this fixture builds.
struct fake_compressed_pin {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> manager;
  std::vector<std::shared_ptr<sirius::compressed_device_representation>> chunks;
  pinned_column_view view;

  fake_compressed_pin(std::vector<std::int64_t> const& batch_rows,
                      std::string const& dsl,
                      bool nullable,
                      rmm::cuda_stream_view stream)
  {
    auto const mr   = rmm::mr::get_current_device_resource_ref();
    manager         = sirius::test::operator_utils::initialize_memory_manager();
    auto* gpu_space = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);

    std::int64_t next = 0;
    for (auto const rows : batch_rows) {
      auto source = make_batch(next, rows, nullable, stream);
      next += rows;

      auto blob   = std::make_shared<sirius::compressed_device_blob>();
      blob->table =
        simpatico::compress_with_plan(cudf::table_view{{source->view()}}, dsl, stream, mr);
      cudaStreamSynchronize(stream.value());

      auto rep = std::make_shared<sirius::compressed_device_representation>(
        *gpu_space,
        blob,
        std::vector<std::string>{"value"},
        /*compressed_bytes=*/64,
        /*uncompressed_bytes=*/static_cast<std::size_t>(rows) * sizeof(std::int32_t),
        rows);

      batch_source src;
      src.compressed   = rep.get();
      src.column_index = 0;
      src.num_rows     = rows;
      view.batches.push_back(src);
      chunks.push_back(std::move(rep));
    }
    view.dtype = cudf::data_type{cudf::type_id::INT32};
  }

  [[nodiscard]] simpatico::compressed_table const& table(std::size_t batch = 0) const
  {
    return chunks[batch]->table();
  }
};

std::vector<std::int32_t> read_back(cudf::column_view const& col)
{
  std::vector<std::int32_t> host(static_cast<std::size_t>(col.size()));
  if (!host.empty()) {
    cudaMemcpy(host.data(),
               col.data<std::int32_t>(),
               host.size() * sizeof(std::int32_t),
               cudaMemcpyDeviceToHost);
  }
  return host;
}

/// The column's per-row validity, as a plain vector. An unallocated mask reads
/// as all-valid, which is what cuDF means by it.
std::vector<bool> read_validity(cudf::column_view const& col)
{
  std::vector<bool> valid(static_cast<std::size_t>(col.size()), true);
  if (col.null_mask() == nullptr) { return valid; }
  auto const words =
    static_cast<std::size_t>(cudf::bitmask_allocation_size_bytes(col.size())) /
    sizeof(cudf::bitmask_type);
  std::vector<cudf::bitmask_type> host(words, 0);
  cudaMemcpy(
    host.data(), col.null_mask(), host.size() * sizeof(cudf::bitmask_type), cudaMemcpyDeviceToHost);
  for (cudf::size_type i = 0; i < col.size(); ++i) {
    valid[static_cast<std::size_t>(i)] =
      (host[static_cast<std::size_t>(i) / 32] >> (i % 32)) & 1u;
  }
  return valid;
}

rmm::device_buffer upload_ids(std::vector<std::uint64_t> const& host, rmm::cuda_stream_view stream)
{
  rmm::device_buffer buf(
    host.size() * sizeof(std::uint64_t), stream, rmm::mr::get_current_device_resource_ref());
  if (!host.empty()) {
    cudaMemcpyAsync(buf.data(),
                    host.data(),
                    host.size() * sizeof(std::uint64_t),
                    cudaMemcpyHostToDevice,
                    stream.value());
    cudaStreamSynchronize(stream.value());
  }
  return buf;
}

/// Materialize `ids` out of `pin` and check both halves against the fixture's
/// own rule: value i is i, and row i is null exactly when row_is_null says so.
void check_against_ids(pinned_column_view const& view,
                       std::vector<std::int64_t> const& batch_rows,
                       std::vector<std::uint64_t> const& ids,
                       bool nullable,
                       bool sorted_unique,
                       rmm::cuda_stream_view stream)
{
  auto const mr     = rmm::mr::get_current_device_resource_ref();
  auto const layout = pinned_table_layout::from_batch_rows(batch_rows);
  auto d_ids        = upload_ids(ids, stream);

  prepared_selection const prepared(layout,
                                    row_id_list{static_cast<std::uint64_t const*>(d_ids.data()),
                                                static_cast<std::int64_t>(ids.size()),
                                                sorted_unique});
  auto const column = materialize(view, prepared, stream, mr);
  cudaStreamSynchronize(stream.value());

  REQUIRE(column->size() == static_cast<cudf::size_type>(ids.size()));

  std::vector<std::int32_t> expect_values;
  std::vector<bool> expect_valid;
  for (auto const id : ids) {
    expect_values.push_back(static_cast<std::int32_t>(id));
    expect_valid.push_back(!nullable || !row_is_null(static_cast<std::int64_t>(id)));
  }
  REQUIRE(read_back(column->view()) == expect_values);
  REQUIRE(read_validity(column->view()) == expect_valid);
}

/// A chunk-CSR over `rows`, batch-local and ascending — the shape the sparse
/// walk indexes by.
struct device_row_set {
  rmm::device_buffer chunk_ids, block_offsets, in_chunk_rows;
  sirius::codegen::chunk_row_set view{};

  device_row_set(std::vector<std::int64_t> const& rows,
                 std::int64_t num_rows,
                 rmm::cuda_stream_view stream)
  {
    std::vector<std::uint32_t> chunks, offsets{0};
    std::vector<std::uint16_t> positions;
    for (auto const r : rows) {
      auto const c = static_cast<std::uint32_t>(r / kChunk);
      if (chunks.empty() || chunks.back() != c) {
        if (!chunks.empty()) { offsets.push_back(static_cast<std::uint32_t>(positions.size())); }
        chunks.push_back(c);
      }
      positions.push_back(static_cast<std::uint16_t>(r % kChunk));
    }
    offsets.push_back(static_cast<std::uint32_t>(positions.size()));

    auto const mr = rmm::mr::get_current_device_resource_ref();
    auto upload   = [&](void const* src, std::size_t bytes) {
      rmm::device_buffer buf(bytes, stream, mr);
      if (bytes != 0) {
        cudaMemcpyAsync(buf.data(), src, bytes, cudaMemcpyHostToDevice, stream.value());
      }
      return buf;
    };
    chunk_ids     = upload(chunks.data(), chunks.size() * sizeof(std::uint32_t));
    block_offsets = upload(offsets.data(), offsets.size() * sizeof(std::uint32_t));
    in_chunk_rows = upload(positions.data(), positions.size() * sizeof(std::uint16_t));
    cudaStreamSynchronize(stream.value());

    view.chunk_ids     = static_cast<std::uint32_t const*>(chunk_ids.data());
    view.block_offsets = static_cast<std::uint32_t const*>(block_offsets.data());
    view.in_chunk_rows = static_cast<std::uint16_t const*>(in_chunk_rows.data());
    view.num_touched   = static_cast<std::int64_t>(chunks.size());
    view.num_survivors = static_cast<std::int64_t>(positions.size());
    view.num_rows      = num_rows;
  }
};

/// The counted selection mask the mask route reads, derived from a row set the
/// same way materialize_compressed derives it.
struct device_mask {
  rmm::device_buffer words, chunk_offsets;
  sirius::codegen::selection_mask view{};
};

device_mask mask_over(device_row_set const& set,
                      std::int64_t num_rows,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr)
{
  device_mask out;
  auto const chunks = sirius::codegen::selection_mask::ChunksFor(num_rows);
  out.words         = rmm::device_buffer(
    static_cast<std::size_t>(sirius::codegen::selection_mask::WordsFor(num_rows)) *
      sizeof(std::uint32_t),
    stream,
    mr);
  out.chunk_offsets =
    rmm::device_buffer((static_cast<std::size_t>(chunks) + 1) * sizeof(std::uint32_t), stream, mr);
  sirius::codegen::row_set_to_mask(set.view,
                                   static_cast<std::uint32_t*>(out.words.data()),
                                   static_cast<std::uint32_t*>(out.chunk_offsets.data()),
                                   stream,
                                   mr);
  out.view = sirius::codegen::selection_mask{static_cast<std::uint32_t*>(out.words.data()),
                                             num_rows,
                                             set.view.num_survivors,
                                             static_cast<std::uint32_t*>(out.chunk_offsets.data())};
  return out;
}

// Plans that classify onto the three selective routes. A bitpack root has random
// access; a delta root does not, so it takes the mask kernels; an ANS root has no
// compacted route at all.
constexpr char const* kBitpackPlan = "input -> bitpack -> packed\n";
constexpr char const* kDeltaPlan =
  "input -> delta -> differences\n"
  "delta.differences -> bitpack\n";
constexpr char const* kFullPlan    = "input -> ans\n";

}  // namespace

TEST_CASE("a compressed chunk reports its null count without decoding",
          "[late_mat][materialize_compressed]")
{
  auto const stream = rmm::cuda_stream_view{};

  fake_compressed_pin nullable({4 * kChunk}, kBitpackPlan, /*nullable=*/true, stream);
  fake_compressed_pin plain({4 * kChunk}, kBitpackPlan, /*nullable=*/false, stream);

  std::int64_t expected = 0;
  for (std::int64_t r = 0; r < 4 * kChunk; ++r) {
    if (row_is_null(r)) { ++expected; }
  }
  REQUIRE(expected > 0);

  auto const counted = simpatico::column_null_count(nullable.table(), 0);
  REQUIRE(counted.has_value());
  REQUIRE(*counted == expected);

  auto const none = simpatico::column_null_count(plain.table(), 0);
  REQUIRE(none.has_value());
  REQUIRE(*none == 0);

  // A column the table does not have answers nothing, rather than answering
  // zero — the whole point of the count is that the gate can trust it.
  REQUIRE_FALSE(simpatico::column_null_count(nullable.table(), 9).has_value());
  REQUIRE(simpatico::column_validity(nullable.table(), 9) == nullptr);
}

TEST_CASE("the dense route decodes a whole nullable batch", "[late_mat][materialize_compressed]")
{
  auto const stream = rmm::cuda_stream_view{};
  std::vector<std::int64_t> const batch_rows{2 * kChunk};
  fake_compressed_pin pin(batch_rows, kBitpackPlan, /*nullable=*/true, stream);

  // Every row of the batch survives, so the selection is dense and the route is
  // an ordinary full decode, which reattaches validity itself.
  std::vector<std::uint64_t> ids(static_cast<std::size_t>(2 * kChunk));
  std::iota(ids.begin(), ids.end(), std::uint64_t{0});
  check_against_ids(pin.view, batch_rows, ids, /*nullable=*/true, /*sorted_unique=*/true, stream);
}

TEST_CASE("the sparse route carries validity for the rows it selected",
          "[late_mat][materialize_compressed]")
{
  auto const stream = rmm::cuda_stream_view{};
  auto const mr     = rmm::mr::get_current_device_resource_ref();
  std::vector<std::int64_t> const batch_rows{4 * kChunk};
  fake_compressed_pin pin(batch_rows, kBitpackPlan, /*nullable=*/true, stream);

  std::vector<std::int64_t> const rows{5, 1023, 1024, 2050, 3000, 4095};
  device_row_set const set(rows, 4 * kChunk, stream);

  // The route this case is about: a bitpack root serves the sparse walk, and
  // it declines the nullable column unless the caller takes the sidecar.
  {
    std::string err;
    auto refused = simpatico::decompress_column_rows(pin.table(), 0, set.view, stream, mr, &err);
    REQUIRE(refused == nullptr);
    REQUIRE(err.find("null-masked") != std::string::npos);
  }
  {
    std::string err;
    simpatico::validity_sidecar const* validity = nullptr;
    auto served =
      simpatico::decompress_column_rows(pin.table(), 0, set.view, stream, mr, &err, &validity);
    REQUIRE(served != nullptr);
    REQUIRE(validity != nullptr);
    REQUIRE(validity->kind == simpatico::validity_kind::mask);
    // Values only: the sidecar is the caller's to compact.
    REQUIRE(served->null_count() == 0);
  }

  std::vector<std::uint64_t> ids;
  for (auto const r : rows) {
    ids.push_back(static_cast<std::uint64_t>(r));
  }
  check_against_ids(pin.view, batch_rows, ids, /*nullable=*/true, /*sorted_unique=*/true, stream);
}

TEST_CASE("the mask route carries validity where the sparse walk cannot serve",
          "[late_mat][materialize_compressed]")
{
  auto const stream = rmm::cuda_stream_view{};
  auto const mr     = rmm::mr::get_current_device_resource_ref();
  std::vector<std::int64_t> const batch_rows{4 * kChunk};
  fake_compressed_pin pin(batch_rows, kDeltaPlan, /*nullable=*/true, stream);

  std::vector<std::int64_t> const rows{2, 900, 1500, 2600, 4000};
  device_row_set const set(rows, 4 * kChunk, stream);

  // A delta root has no random access, so the sparse walk refuses for a reason
  // that is not nullability, and the mask kernels are what serve this plan.
  std::string err;
  simpatico::validity_sidecar const* validity = nullptr;
  auto declined =
    simpatico::decompress_column_rows(pin.table(), 0, set.view, stream, mr, &err, &validity);
  REQUIRE(declined == nullptr);
  REQUIRE(err.find("random-access") != std::string::npos);

  // Well under the density at which materialize_compressed stops trying the
  // mask route, so this is the route the selection below actually takes.
  REQUIRE(static_cast<double>(rows.size()) / static_cast<double>(4 * kChunk) < 0.35);

  // And the mask kernels really do serve this plan — otherwise the cascade
  // would fall through to the full decode and this case would be testing that
  // route a second time.
  auto const mask = mask_over(set, 4 * kChunk, stream, mr);
  {
    std::string err2;
    simpatico::validity_sidecar const* mask_validity = nullptr;
    auto served                                      = simpatico::decompress_column_compacted(
      pin.table(), 0, mask.view, stream, mr, &err2, &mask_validity);
    REQUIRE(served != nullptr);
    REQUIRE(served->size() == static_cast<cudf::size_type>(rows.size()));
    REQUIRE(mask_validity != nullptr);
    REQUIRE(mask_validity->kind == simpatico::validity_kind::mask);
    REQUIRE(served->null_count() == 0);
  }

  std::vector<std::uint64_t> ids;
  for (auto const r : rows) {
    ids.push_back(static_cast<std::uint64_t>(r));
  }
  check_against_ids(pin.view, batch_rows, ids, /*nullable=*/true, /*sorted_unique=*/true, stream);
}

TEST_CASE("the full-decode fallback carries validity through its gather",
          "[late_mat][materialize_compressed]")
{
  auto const stream = rmm::cuda_stream_view{};
  auto const mr     = rmm::mr::get_current_device_resource_ref();
  std::vector<std::int64_t> const batch_rows{2 * kChunk};
  fake_compressed_pin pin(batch_rows, kFullPlan, /*nullable=*/true, stream);

  std::vector<std::int64_t> const rows{1, 40, 700, 1100, 2047};
  device_row_set const set(rows, 2 * kChunk, stream);

  // Neither compacting route classifies this plan, so the cascade ends at the
  // full decode and one gather.
  {
    std::string err;
    simpatico::validity_sidecar const* validity = nullptr;
    REQUIRE(simpatico::decompress_column_rows(
              pin.table(), 0, set.view, stream, mr, &err, &validity) == nullptr);
  }
  {
    std::string err;
    auto full = simpatico::decompress_column_full(pin.table(), 0, stream, mr, &err);
    REQUIRE(full != nullptr);
    // The full decode is the one route that reattaches validity on its own.
    REQUIRE(full->null_count() > 0);
  }

  std::vector<std::uint64_t> ids;
  for (auto const r : rows) {
    ids.push_back(static_cast<std::uint64_t>(r));
  }
  check_against_ids(pin.view, batch_rows, ids, /*nullable=*/true, /*sorted_unique=*/true, stream);
}

TEST_CASE("validity stays with its value under a repeated, unsorted selection",
          "[late_mat][materialize_compressed]")
{
  auto const stream = rmm::cuda_stream_view{};
  std::vector<std::int64_t> const batch_rows{3 * kChunk, 2 * kChunk};
  fake_compressed_pin pin(batch_rows, kBitpackPlan, /*nullable=*/true, stream);

  // What a join hands back: out of order, with repeats, spanning both batches.
  // The canonical form sorts and deduplicates it, materializes in table order
  // and gathers back — three places for validity to part company with its value
  // while the row count stays right.
  std::vector<std::uint64_t> const ids{
    4000, 3, 4000, 3072, 3, 12, 4999, 3072, 0, 4999, 4999, 2050, 1023, 1024, 4999, 7};
  check_against_ids(pin.view, batch_rows, ids, /*nullable=*/true, /*sorted_unique=*/false, stream);
}

TEST_CASE("a compressed origin with no nulls materializes unchanged",
          "[late_mat][materialize_compressed]")
{
  auto const stream = rmm::cuda_stream_view{};
  std::vector<std::int64_t> const batch_rows{3 * kChunk, 2 * kChunk};
  fake_compressed_pin pin(batch_rows, kBitpackPlan, /*nullable=*/false, stream);

  std::vector<std::uint64_t> const ids{4000, 3, 4000, 3072, 12, 4999, 0, 2050, 1023, 1024};
  check_against_ids(pin.view, batch_rows, ids, /*nullable=*/false, /*sorted_unique=*/false, stream);
}

TEST_CASE("an all-null compressed column materializes as all null",
          "[late_mat][materialize_compressed]")
{
  auto const stream = rmm::cuda_stream_view{};
  auto const mr     = rmm::mr::get_current_device_resource_ref();
  auto manager      = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space   = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);

  // The all-null shape stores no bitmask at all — only the kind and the count —
  // so it is the one case where nothing is gathered and the mask is regenerated.
  constexpr std::int64_t kRows = 2 * kChunk;
  auto source                  = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                          static_cast<cudf::size_type>(kRows),
                                          cudf::mask_state::ALL_NULL,
                                          stream);
  cudaStreamSynchronize(stream.value());

  auto blob   = std::make_shared<sirius::compressed_device_blob>();
  blob->table =
    simpatico::compress_with_plan(cudf::table_view{{source->view()}}, kBitpackPlan, stream, mr);
  cudaStreamSynchronize(stream.value());

  auto const counted = simpatico::column_null_count(blob->table, 0);
  REQUIRE(counted.has_value());
  REQUIRE(*counted == kRows);

  auto rep = std::make_shared<sirius::compressed_device_representation>(
    *gpu_space,
    blob,
    std::vector<std::string>{"value"},
    /*compressed_bytes=*/64,
    /*uncompressed_bytes=*/static_cast<std::size_t>(kRows) * sizeof(std::int32_t),
    kRows);

  pinned_column_view view;
  view.dtype = cudf::data_type{cudf::type_id::INT32};
  batch_source src;
  src.compressed   = rep.get();
  src.column_index = 0;
  src.num_rows     = kRows;
  view.batches.push_back(src);

  std::vector<std::uint64_t> const ids{5, 900, 900, 2047, 0};
  auto const layout = pinned_table_layout::from_batch_rows({kRows});
  auto d_ids        = upload_ids(ids, stream);
  prepared_selection const prepared(layout,
                                    row_id_list{static_cast<std::uint64_t const*>(d_ids.data()),
                                                static_cast<std::int64_t>(ids.size()),
                                                false});
  auto const column = materialize(view, prepared, stream, mr);
  cudaStreamSynchronize(stream.value());

  REQUIRE(column->size() == static_cast<cudf::size_type>(ids.size()));
  REQUIRE(column->null_count() == static_cast<cudf::size_type>(ids.size()));
}
