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

#include "late_mat/materialize.hpp"

#include "compression/compressed_representation.hpp"
#include "late_mat/multi_source_gather.hpp"
#include "log/logging.hpp"
#include "telemetry/nvtx.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda/buffer>
#include <cuda/utility>
#include <cuda_runtime.h>

#include <api/simpatico_codegen.hpp>
#include <codegen/selection/chunk_row_set.hpp>
#include <codegen/selection/selection.hpp>
#include <cucascade/cuda/event.hpp>
#include <cucascade/error.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace sirius::late_mat {

namespace {

std::atomic<std::uint64_t> g_remote_batches_materialized{0};

std::unique_ptr<cudf::column> gather_one(cudf::column_view const& source,
                                         cudf::column_view const& map,
                                         ::cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr)
{
  // DONT_CHECK: ids are pin-order positions the caller is responsible for, the
  // same contract cudf::gather itself takes. Checking would cost a pass to
  // re-establish what the addressing already guarantees.
  auto gathered = cudf::gather(
    cudf::table_view{{source}}, map, cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
  return std::move(gathered->release().front());
}

cudf::column_view int32_map(rmm::device_buffer const& buf, std::int64_t count)
{
  return cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                           static_cast<cudf::size_type>(count),
                           buf.data(),
                           nullptr,
                           0};
}

/// The device the consumer runs on. The pipeline task made it current before
/// the port ran, and every local gather below is enqueued against it.
int current_device()
{
  int device = 0;
  CUCASCADE_CUDA_TRY(cudaGetDevice(&device));
  return device;
}

/// A batch pinned on a GPU other than the consumer's.
bool is_remote(batch_source const& source, int consumer_device)
{
  return source.memory_space != nullptr &&
         source.memory_space->get_tier() == cucascade::memory::Tier::GPU &&
         source.memory_space->get_device_id() != consumer_device;
}

/// Can this column be produced straight from the raw ids, with no sort?
///
/// Three things have to hold. Nothing may be compressed, since a compressed row
/// is decoded rather than copied and producing it once per reference is the
/// waste deduplication exists to avoid. Every batch must be on the consumer's
/// GPU, because the one-pass gather dereferences every batch from there. And a
/// multi-batch column has to be fixed width, because the one-pass gather copies
/// a fixed element and a variable-width column would need its offsets rebuilt.
bool can_gather_raw(pinned_column_view const& column, int consumer_device)
{
  for (auto const& b : column.batches) {
    if (b.is_compressed() || is_remote(b, consumer_device)) { return false; }
  }
  if (column.batches.size() == 1) { return true; }
  return cudf::is_fixed_width(column.dtype);
}

/// The raw path: gather by global id, no canonical form, no restoring pass.
std::unique_ptr<cudf::column> materialize_raw(pinned_column_view const& column,
                                              prepared_selection const& selection,
                                              ::cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
{
  auto const& ids = selection.ids();

  // One batch starts at global row 0, so the ids ARE the batch-local map and
  // cudf gathers straight off the borrowed list. Any dtype, since cudf does the
  // element handling.
  if (column.batches.size() == 1) {
    auto const map = cudf::column_view{cudf::data_type{cudf::type_id::UINT64},
                                       static_cast<cudf::size_type>(ids.count),
                                       ids.ids,
                                       nullptr,
                                       0};
    return gather_one(column.batches.front().uncompressed, map, stream, mr);
  }

  auto const elem_size = cudf::size_of(column.dtype);
  bool const any_nullable =
    std::any_of(column.batches.begin(), column.batches.end(), [](batch_source const& b) {
      return b.uncompressed.nullable();
    });
  auto out = cudf::make_fixed_width_column(
    column.dtype,
    static_cast<cudf::size_type>(ids.count),
    any_nullable ? cudf::mask_state::UNINITIALIZED : cudf::mask_state::UNALLOCATED,
    stream,
    mr);

  std::vector<void const*> host_bases;
  std::vector<std::int64_t> host_starts;
  std::vector<cudf::bitmask_type const*> host_masks;
  host_bases.reserve(column.batches.size());
  host_starts.reserve(column.batches.size());
  if (any_nullable) { host_masks.reserve(column.batches.size()); }
  for (std::size_t b = 0; b < column.batches.size(); ++b) {
    host_bases.push_back(column.batches[b].uncompressed.head<void>());
    host_starts.push_back(selection.layout().batch_row_start[b]);
    if (any_nullable) { host_masks.push_back(column.batches[b].uncompressed.null_mask()); }
  }

  rmm::device_buffer bases(host_bases.size() * sizeof(void const*), stream, mr);
  rmm::device_buffer starts(host_starts.size() * sizeof(std::int64_t), stream, mr);
  cudaMemcpyAsync(bases.data(),
                  host_bases.data(),
                  host_bases.size() * sizeof(void const*),
                  cudaMemcpyHostToDevice,
                  stream.get());
  cudaMemcpyAsync(starts.data(),
                  host_starts.data(),
                  host_starts.size() * sizeof(std::int64_t),
                  cudaMemcpyHostToDevice,
                  stream.get());

  rmm::device_buffer masks;
  if (any_nullable) {
    masks = rmm::device_buffer(host_masks.size() * sizeof(cudf::bitmask_type const*), stream, mr);
    cudaMemcpyAsync(masks.data(),
                    host_masks.data(),
                    host_masks.size() * sizeof(cudf::bitmask_type const*),
                    cudaMemcpyHostToDevice,
                    stream.get());
  }

  multi_source_gather_fixed(
    static_cast<void const* const*>(bases.data()),
    static_cast<std::int64_t const*>(starts.data()),
    static_cast<int>(column.batches.size()),
    elem_size,
    ids.ids,
    ids.count,
    out->mutable_view().head<void>(),
    any_nullable ? static_cast<std::uint32_t const* const*>(masks.data()) : nullptr,
    any_nullable ? reinterpret_cast<std::uint32_t*>(out->mutable_view().null_mask()) : nullptr,
    stream);
  // The base, start and mask arrays are freed as this returns; they are stream-
  // ordered on the same stream the gather was enqueued on, so the deallocation
  // is already ordered after it.
  if (any_nullable) {
    out->set_null_count(cudf::null_count(
      out->view().null_mask(), 0, static_cast<cudf::size_type>(ids.count), stream));
  }
  return out;
}

/// Below this the mask route still beats decoding everything and gathering
/// once; above it, it does not. Measured on the original campaign's crossover
/// microbench.
constexpr double kMaskRouteMaxDensity = 0.35;

std::unique_ptr<cudf::column> require_non_null(std::unique_ptr<cudf::column> col)
{
  // None of these routes gathers a validity mask alongside the values, so a
  // nullable column would come back with someone else's nulls.
  if (col && col->null_count() != 0) {
    throw std::runtime_error("late_mat::materialize: nullable origin columns are not supported");
  }
  return col;
}

/// One compressed batch, by the cheapest route its plan can take.
///
/// A route returning nothing means this plan cannot take it, and the next one
/// is tried; the full decode ends every cascade. That is what lets the deferral
/// decision stay ignorant of decode internals — any column may be deferred, and
/// the worst case is the decode that would have happened anyway.
std::unique_ptr<cudf::column> materialize_compressed(batch_source const& source,
                                                     batch_selection const& selection,
                                                     ::cuda::stream_ref stream,
                                                     rmm::device_async_resource_ref mr)
{
  auto const& table = source.compressed->table();
  auto const idx    = source.column_index;

  // Whole batch live: no selection to express, so this is an ordinary decode.
  if (selection.dense) {
    std::string err;
    auto full = simpatico::decompress_column_full(table, idx, stream, mr, &err);
    if (!full) { throw std::runtime_error("late_mat::materialize: full decode failed: " + err); }
    return require_non_null(std::move(full));
  }

  auto const rows = selection.rows.view();

  // (a) Sparse walk: one block per touched chunk. Tried at every density,
  // because the mask route never won a density-based choice when it was
  // measured — it is the capability fallback below, not the faster one.
  {
    std::string err;
    auto sparse = simpatico::decompress_column_rows(table, idx, rows, stream, mr, &err);
    if (sparse) { return require_non_null(std::move(sparse)); }
  }

  // (b) Mask route: the shipped kernels, for the shapes with no random access
  // (dictionary, str_split, render rejections). The mask is derived from the
  // row set rather than rebuilt from the ids.
  if (selection.density < kMaskRouteMaxDensity) {
    auto const num_rows = source.num_rows;
    auto const chunks   = sirius::codegen::selection_mask::ChunksFor(num_rows);
    rmm::device_buffer mask_words(
      static_cast<std::size_t>(sirius::codegen::selection_mask::WordsFor(num_rows)) *
        sizeof(std::uint32_t),
      stream,
      mr);
    rmm::device_buffer chunk_offsets(
      (static_cast<std::size_t>(chunks) + 1) * sizeof(std::uint32_t), stream, mr);
    sirius::codegen::row_set_to_mask(rows,
                                     static_cast<std::uint32_t*>(mask_words.data()),
                                     static_cast<std::uint32_t*>(chunk_offsets.data()),
                                     stream,
                                     mr);
    sirius::codegen::selection_mask mask{static_cast<std::uint32_t*>(mask_words.data()),
                                         num_rows,
                                         rows.num_survivors,
                                         static_cast<std::uint32_t*>(chunk_offsets.data())};
    std::string err;
    auto compacted = simpatico::decompress_column_compacted(table, idx, mask, stream, mr, &err);
    if (compacted) { return require_non_null(std::move(compacted)); }
  }

  // (c) The route that always works: decode everything, gather once.
  std::string err;
  auto full = simpatico::decompress_column_full(table, idx, stream, mr, &err);
  if (!full) { throw std::runtime_error("late_mat::materialize: full decode failed: " + err); }
  auto checked = require_non_null(std::move(full));
  return gather_one(
    checked->view(), int32_map(selection.local_indices, selection.survivors), stream, mr);
}

/// Buffers released from a column on another GPU. They stay alive here until
/// the copies that read them have been ordered ahead of their frees.
using retired_buffer  = std::variant<std::unique_ptr<rmm::device_buffer>,
                                     std::unique_ptr<::cuda::device_buffer<std::byte>>>;
using retired_buffers = std::vector<retired_buffer>;

/// What a remote gather allocated on the owning GPU: the gather map and, once
/// handed to the peer copies, the gathered column's buffers.
///
/// The owner's stream reads the consumer's selection indices; the consumer's
/// stream reads the owner's gathered buffers; each buffer is freed on the
/// stream it was allocated on. So before anything is freed, on every exit path
/// including a throw mid-way, each stream is ordered behind the other's queued
/// reads. All no-throw; if an event fails, the stream waited for is
/// synchronized instead.
class owner_scratch {
 public:
  owner_scratch(int owner_device,
                ::cuda::stream_ref owner_stream,
                int consumer_device,
                ::cuda::stream_ref consumer_stream)
    : _owner_device{owner_device},
      _owner_stream{owner_stream},
      _consumer_device{consumer_device},
      _consumer_stream{consumer_stream}
  {
  }
  ~owner_scratch() { release(); }
  owner_scratch(owner_scratch const&)            = delete;
  owner_scratch& operator=(owner_scratch const&) = delete;

  rmm::device_buffer map;                  ///< batch-local indices, copied to the owner
  retired_buffers retired;                 ///< the gathered column's buffers, after release()
  std::unique_ptr<cudf::column> gathered;  ///< owner-device column, until its buffers are retired

  /// Order each stream behind the other, then free the owner's buffers. Idempotent.
  void release() noexcept
  {
    if (_released) { return; }
    _released = true;
    // Consumer behind owner: the owner's index copy reads `local_indices`.
    if (!order(_owner_stream, _owner_device, _consumer_stream, _consumer_device)) {
      rmm::cuda_set_device_raii const owner_guard{rmm::cuda_device_id{_owner_device}};
      (void)cudaStreamSynchronize(_owner_stream.get());
    }
    // Owner behind consumer: the consumer's copies read the gathered buffers.
    if (!order(_consumer_stream, _consumer_device, _owner_stream, _owner_device)) {
      (void)cudaStreamSynchronize(_consumer_stream.get());
    }
    (void)cudaGetLastError();
    rmm::cuda_set_device_raii const owner_guard{rmm::cuda_device_id{_owner_device}};
    gathered.reset();
    retired.clear();
    map = rmm::device_buffer{};
  }

 private:
  /// Make `waiter` wait for everything `producer` has queued so far.
  [[nodiscard]] static bool order(::cuda::stream_ref producer,
                                  int producer_device,
                                  ::cuda::stream_ref waiter,
                                  int waiter_device) noexcept
  {
    cudaEvent_t event{};
    bool ordered = false;
    {
      rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{producer_device}};
      if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) { return false; }
      ordered = cudaEventRecord(event, producer.get()) == cudaSuccess;
    }
    if (ordered) {
      rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{waiter_device}};
      ordered = cudaStreamWaitEvent(waiter.get(), event, 0) == cudaSuccess;
    }
    (void)cudaEventDestroy(event);
    return ordered;
  }

  int _owner_device;
  ::cuda::stream_ref _owner_stream;
  int _consumer_device;
  ::cuda::stream_ref _consumer_stream;
  bool _released = false;
};

template <typename Buffer>
std::unique_ptr<Buffer> peer_copy(std::unique_ptr<Buffer> source,
                                  int source_device,
                                  int destination_device,
                                  ::cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr,
                                  retired_buffers& retire)
{
  if constexpr (!std::is_same_v<Buffer, rmm::device_buffer>) {
    static_assert(std::is_same_v<typename Buffer::value_type, std::byte>,
                  "peer_copy only supports the cudf null-mask byte buffer type");
  }
  auto const bytes       = source ? source->size() : std::size_t{0};
  Buffer const* retained = nullptr;
  {
    rmm::cuda_set_device_raii const owner_guard{rmm::cuda_device_id{source_device}};
    retained = source.get();
    // Retire first so a later failure leaves the source alive until its readers are ordered.
    try {
      if (source) { retire.emplace_back(std::move(source)); }
    } catch (...) {
      source.reset();
      throw;
    }
  }

  // Keep a partially allocated consumer buffer inside its device guard so an
  // exception destroys it with the device that owns its allocation current.
  rmm::cuda_set_device_raii const destination_guard{rmm::cuda_device_id{destination_device}};
  auto copy = [&] {
    if constexpr (std::is_same_v<Buffer, rmm::device_buffer>) {
      return std::make_unique<Buffer>(bytes, stream, mr);
    } else {
      return std::make_unique<Buffer>(stream, mr, bytes, ::cuda::no_init);
    }
  }();
  if (bytes > 0) {
    CUCASCADE_CUDA_TRY(cudaMemcpyPeerAsync(
      copy->data(), destination_device, retained->data(), source_device, bytes, stream.get()));
  }
  return copy;
}

/// Rebuild @p column on the current device: every buffer and every child is
/// peer-copied as it is, so any dtype crosses. The column's own buffers end
/// up in @p retire.
std::unique_ptr<cudf::column> transfer_column(std::unique_ptr<cudf::column> column,
                                              int source_device,
                                              int destination_device,
                                              ::cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr,
                                              retired_buffers& retire)
{
  rmm::cuda_set_device_raii const destination_guard{rmm::cuda_device_id{destination_device}};
  cudf::data_type type{};
  cudf::size_type size{};
  cudf::size_type null_count{};
  cudf::column::contents contents;
  decltype(contents.null_mask) null_mask;
  {
    rmm::cuda_set_device_raii const owner_guard{rmm::cuda_device_id{source_device}};
    type       = column->type();
    size       = column->size();
    null_count = column->null_count();
    contents   = column->release();
  }

  std::vector<std::unique_ptr<cudf::column>> children;
  std::unique_ptr<rmm::device_buffer> data;
  try {
    children.reserve(contents.children.size());
    for (auto& child : contents.children) {
      children.push_back(
        transfer_column(std::move(child), source_device, destination_device, stream, mr, retire));
    }
    data =
      peer_copy(std::move(contents.data), source_device, destination_device, stream, mr, retire);
    null_mask = peer_copy(
      std::move(contents.null_mask), source_device, destination_device, stream, mr, retire);
    return std::make_unique<cudf::column>(
      type, size, std::move(*data), std::move(*null_mask), null_count, std::move(children));
  } catch (...) {
    children.clear();
    data.reset();
    null_mask.reset();
    rmm::cuda_set_device_raii const owner_guard{rmm::cuda_device_id{source_device}};
    contents.children.clear();
    contents.data.reset();
    contents.null_mask.reset();
    throw;
  }
}

/// A batch pinned on another GPU: gather its survivors where it lives, then
/// bring only those rows over.
///
/// Two transfers cross, both the size of the SELECTION rather than the batch:
/// the batch-local indices go to the owner, the gathered rows come back. Both
/// are cudaMemcpyPeerAsync, so a pair with working peer DMA takes it and a pair
/// without is staged through host by the driver. The owner's work runs on a
/// stream of its own space, ordered by events behind the consumer's selection
/// build and ahead of the consumer's copy.
std::unique_ptr<cudf::column> materialize_remote_batch(batch_source const& source,
                                                       batch_selection const& selection,
                                                       int consumer_device,
                                                       ::cuda::stream_ref stream,
                                                       rmm::device_async_resource_ref mr)
{
  if (source.is_compressed()) {
    // Its decode would run on the owning GPU against a selection that lives on
    // this one, which is why install_late_materialization declines such a pin.
    // A batch that reaches the port anyway cannot fall back: the scan has
    // already emitted rowids in place of values.
    throw std::runtime_error(
      "late_mat::materialize: internal error: a compressed batch pinned on another GPU reached "
      "the port; the installer declines such pins, and the scan has already emitted rowids in "
      "place of values so the query cannot fall back");
  }
  g_remote_batches_materialized.fetch_add(1, std::memory_order_relaxed);
  auto const& owner       = *source.memory_space;
  auto const owner_device = owner.get_device_id();
  auto const owner_stream = owner.acquire_stream();
  auto const owner_mr     = owner.get_default_allocator();

  // An event belongs to the device current when it is created: the consumer's
  // outside the owner guard, the owner's inside it.
  cucascade::cuda::cuda_event selection_ready;
  selection_ready.record(stream);

  // Owns the gathered column until its buffers have been retired safely.
  owner_scratch scratch{owner_device, owner_stream, consumer_device, stream};
  std::optional<cucascade::cuda::cuda_event> gathered_ready;
  {
    rmm::cuda_set_device_raii const owner_guard{rmm::cuda_device_id{owner_device}};
    selection_ready.wait(owner_stream);
    if (selection.dense) {
      scratch.gathered =
        std::make_unique<cudf::column>(source.uncompressed, owner_stream, owner_mr);
    } else {
      auto const map_bytes = static_cast<std::size_t>(selection.survivors) * sizeof(std::int32_t);
      scratch.map          = rmm::device_buffer(map_bytes, owner_stream, owner_mr);
      CUCASCADE_CUDA_TRY(cudaMemcpyPeerAsync(scratch.map.data(),
                                             owner_device,
                                             selection.local_indices.data(),
                                             consumer_device,
                                             map_bytes,
                                             owner_stream.get()));
      scratch.gathered = gather_one(
        source.uncompressed, int32_map(scratch.map, selection.survivors), owner_stream, owner_mr);
    }
    gathered_ready.emplace();
    gathered_ready->record(owner_stream);
  }

  gathered_ready->wait(stream);
  auto local = transfer_column(
    std::move(scratch.gathered), owner_device, consumer_device, stream, mr, scratch.retired);
  scratch.release();
  return local;
}

/// A batch's surviving rows, as a column of its own.
///
/// Dense batches are copied rather than gathered: the gather map would be the
/// identity, and building one to apply it is more work than the copy.
std::unique_ptr<cudf::column> materialize_batch(batch_source const& source,
                                                batch_selection const& selection,
                                                int consumer_device,
                                                ::cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr)
{
  if (is_remote(source, consumer_device)) {
    return materialize_remote_batch(source, selection, consumer_device, stream, mr);
  }
  if (source.is_compressed()) { return materialize_compressed(source, selection, stream, mr); }
  if (selection.dense) {
    auto copied =
      std::make_unique<cudf::table>(cudf::table_view{{source.uncompressed}}, stream, mr);
    return std::move(copied->release().front());
  }
  return gather_one(
    source.uncompressed, int32_map(selection.local_indices, selection.survivors), stream, mr);
}

}  // namespace

std::unique_ptr<cudf::column> materialize(pinned_column_view const& column,
                                          prepared_selection const& selection,
                                          ::cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  nvtx_scoped_range nvtx_range{"sirius::late_mat::materialize"};
  auto const& layout = selection.layout();
  if (column.batches.size() != layout.num_batches()) {
    throw std::runtime_error(
      "late_mat::materialize: the column's batches do not match the "
      "layout the selection was prepared against");
  }
  for (std::size_t b = 0; b < column.batches.size(); ++b) {
    if (column.batches[b].num_rows != layout.batch_rows[b]) {
      throw std::runtime_error("late_mat::materialize: batch " + std::to_string(b) +
                               " has a different row count than the prepared layout");
    }
  }

  if (selection.original_count() == 0) { return cudf::make_empty_column(column.dtype); }

  auto const consumer_device = current_device();

  // Nothing here needs the rows sorted or deduplicated, so do not build a
  // canonical form that a later compressed column may never ask for either.
  if (can_gather_raw(column, consumer_device)) {
    return materialize_raw(column, selection, stream, mr);
  }

  auto const& canonical = selection.canonical(stream, mr);
  if (canonical.total_survivors == 0) { return cudf::make_empty_column(column.dtype); }

  // In pin order, so the assembled column is in pinned-table order.
  std::vector<std::unique_ptr<cudf::column>> pieces;
  pieces.reserve(column.batches.size());
  std::size_t remote_batches = 0;
  std::int64_t remote_rows   = 0;
  for (std::size_t b = 0; b < column.batches.size(); ++b) {
    if (canonical.batches[b].survivors == 0) { continue; }
    if (is_remote(column.batches[b], consumer_device)) {
      ++remote_batches;
      remote_rows += canonical.batches[b].survivors;
    }
    pieces.push_back(
      materialize_batch(column.batches[b], canonical.batches[b], consumer_device, stream, mr));
  }
  if (remote_batches > 0) {
    SIRIUS_LOG_INFO("[late-mat] GPU {}: gathered {} row(s) from {} batch(es) pinned on other GPUs",
                    consumer_device,
                    remote_rows,
                    remote_batches);
  }

  std::unique_ptr<cudf::column> assembled;
  if (pieces.size() == 1) {
    assembled = std::move(pieces.front());
  } else {
    std::vector<cudf::column_view> views;
    views.reserve(pieces.size());
    for (auto const& p : pieces) {
      views.push_back(p->view());
    }
    assembled = cudf::concatenate(views, stream, mr);
  }

  if (!canonical.needs_restore()) { return assembled; }

  // Back into the caller's order, repeats included. This gather is over the
  // materialized column — narrow, and already reduced to the surviving rows —
  // which is why deduplicating was worth a sort for the consumers that need it.
  return gather_one(
    assembled->view(), int32_map(canonical.restore_rank, selection.original_count()), stream, mr);
}

std::uint64_t remote_batches_materialized() noexcept
{
  return g_remote_batches_materialized.load(std::memory_order_relaxed);
}

}  // namespace sirius::late_mat
