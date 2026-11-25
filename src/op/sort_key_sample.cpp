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

#include "op/sort_key_sample.hpp"

#include <cudf/column/column.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda_runtime.h>

#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <stdexcept>

namespace sirius::op {

struct sort_key_sample::piece {
  std::uint64_t batch_id = 0;
  std::size_t rows       = 0;  //!< Rows of the batch.
  std::size_t bytes      = 0;  //!< Bytes of the batch.
  std::size_t stride     = 1;  //!< Stride the key rows were taken at.
  std::size_t key_rows   = 0;
  //! The key rows as cudf::pack() wrote them, the data copied to the host.
  std::unique_ptr<std::vector<std::uint8_t>> metadata;
  std::vector<std::uint8_t> data;
};

namespace {

//! A device INT32 column holding first, first + step, ... below @p end.
std::unique_ptr<cudf::column> row_indices(std::size_t first,
                                          std::size_t step,
                                          std::size_t end,
                                          ::cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  auto const count = end > first ? 1 + (end - first - 1) / step : std::size_t{0};
  cudf::numeric_scalar<cudf::size_type> const init(
    static_cast<cudf::size_type>(first), true, stream, mr);
  cudf::numeric_scalar<cudf::size_type> const increment(
    static_cast<cudf::size_type>(step), true, stream, mr);
  return cudf::sequence(static_cast<cudf::size_type>(count), init, increment, stream, mr);
}

//! Of key rows taken at stride @p from, those kept at the larger stride @p to.
std::unique_ptr<cudf::table> thin(cudf::table_view keys,
                                  std::size_t from,
                                  std::size_t to,
                                  ::cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  // Kept row i is input row first_row(from) + i * from.
  auto const step  = to / from;
  auto const first = (sort_key_sample::first_row(to) - sort_key_sample::first_row(from)) / from;
  auto index = row_indices(first, step, static_cast<std::size_t>(keys.num_rows()), stream, mr);
  return cudf::gather(keys, index->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
}

//! @p keys as cudf::pack() writes them: the metadata, and the data copied to @p data.
std::unique_ptr<std::vector<std::uint8_t>> pack_to_host(cudf::table_view keys,
                                                        std::vector<std::uint8_t>& data,
                                                        ::cuda::stream_ref stream,
                                                        rmm::device_async_resource_ref mr)
{
  auto packed = cudf::pack(keys, stream, mr);
  data.resize(packed.gpu_data->size());
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    data.data(), packed.gpu_data->data(), data.size(), cudaMemcpyDeviceToHost, stream.get()));
  // The copy ends before its source is freed and before other tasks read the rows.
  stream.sync();
  return std::move(packed.metadata);
}

//! @p data copied to the GPU. The caller keeps @p data alive until the stream is synchronized.
std::unique_ptr<rmm::device_buffer> to_device(std::vector<std::uint8_t> const& data,
                                              ::cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
{
  auto copied = std::make_unique<rmm::device_buffer>(data.size(), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    copied->data(), data.data(), data.size(), cudaMemcpyHostToDevice, stream.get()));
  return copied;
}

}  // namespace

sort_key_sample::sort_key_sample(std::size_t max_key_rows) : _max_key_rows(max_key_rows)
{
  if (max_key_rows == 0) {
    throw std::invalid_argument("[sort_key_sample] max_key_rows must be positive");
  }
}

std::size_t sort_key_sample::kept_at(std::size_t stride, std::uint64_t const* skip_batch_id) const
{
  std::size_t kept = 0;
  for (auto const& [id, entry] : _pieces) {
    if (skip_batch_id != nullptr && id == *skip_batch_id) { continue; }
    kept += key_row_count(entry->rows, stride);
  }
  return kept;
}

void sort_key_sample::add(std::uint64_t batch_id,
                          cudf::table_view sorted_keys,
                          std::size_t batch_bytes,
                          cucascade::memory::memory_space& space,
                          ::cuda::stream_ref stream)
{
  if (_closed.load(std::memory_order_acquire)) { return; }
  auto const rows = static_cast<std::size_t>(sorted_keys.num_rows());

  auto made      = std::make_shared<piece>();
  made->batch_id = batch_id;
  made->rows     = rows;
  made->bytes    = batch_bytes;
  {
    std::lock_guard lock(_mutex);
    // Never past the final stride: the rows counted here are a part of the final input.
    made->stride = _stride;
    while (kept_at(made->stride, &batch_id) + key_row_count(rows, made->stride) > _max_key_rows) {
      made->stride *= 2;
    }
  }
  made->key_rows = key_row_count(rows, made->stride);

  if (made->key_rows != 0) {
    auto const mr = space.get_default_allocator();
    auto index    = row_indices(first_row(made->stride), made->stride, rows, stream, mr);
    auto keys =
      cudf::gather(sorted_keys, index->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
    made->metadata = pack_to_host(keys->view(), made->data, stream, mr);
  }

  {
    std::lock_guard lock(_mutex);
    if (_closed.load(std::memory_order_acquire)) { return; }
    auto& slot = _pieces[batch_id];
    if (slot) {
      --_seen.batches;
      _seen.rows -= slot->rows;
      _seen.bytes -= slot->bytes;
    }
    ++_seen.batches;
    _seen.rows += rows;
    _seen.bytes += batch_bytes;
    slot = std::move(made);
    while (kept_at(_stride, nullptr) > _max_key_rows) {
      _stride *= 2;
    }
  }
  thin_held_rows(space, stream);
}

std::size_t sort_key_sample::held_key_rows() const
{
  std::size_t held = 0;
  for (auto const& [id, entry] : _pieces) {
    held += entry->key_rows;
  }
  return held;
}

void sort_key_sample::thin_held_rows(cucascade::memory::memory_space& space,
                                     ::cuda::stream_ref stream)
{
  {
    std::lock_guard lock(_mutex);
    // The caller that is thinning checks the cap again before it stops, so it sees these rows.
    if (_thinning) { return; }
    _thinning = true;
  }
  try {
    auto const mr = space.get_default_allocator();
    while (true) {
      std::size_t stride = 1;
      std::vector<std::shared_ptr<piece const>> dense;
      {
        std::lock_guard lock(_mutex);
        // At the current stride the rows fit, so past the cap some piece is denser than that.
        if (!_closed.load(std::memory_order_acquire) && held_key_rows() > _max_key_rows) {
          stride = _stride;
          for (auto const& [id, entry] : _pieces) {
            if (entry->stride < stride) { dense.push_back(entry); }
          }
        }
        // Stops in the step that finds nothing to thin, so no rows added meanwhile are missed.
        if (dense.empty()) {
          _thinning = false;
          return;
        }
      }

      for (auto const& old : dense) {
        auto thinned      = std::make_shared<piece>();
        thinned->batch_id = old->batch_id;
        thinned->rows     = old->rows;
        thinned->bytes    = old->bytes;
        thinned->stride   = stride;
        thinned->key_rows = key_row_count(old->rows, stride);
        if (thinned->key_rows != 0) {
          auto copied = to_device(old->data, stream, mr);
          auto view =
            cudf::unpack(old->metadata->data(), static_cast<std::uint8_t const*>(copied->data()));
          auto keys         = thin(view, old->stride, stride, stream, mr);
          thinned->metadata = pack_to_host(keys->view(), thinned->data, stream, mr);
        }
        std::lock_guard lock(_mutex);
        // A retried task may have replaced the piece meanwhile.
        if (auto slot = _pieces.find(old->batch_id); slot != _pieces.end() && slot->second == old) {
          slot->second = std::move(thinned);
        }
      }
    }
  } catch (...) {
    std::lock_guard lock(_mutex);
    _thinning = false;
    throw;
  }
}

sort_key_sample::totals sort_key_sample::seen() const
{
  std::lock_guard lock(_mutex);
  return _seen;
}

std::size_t sort_key_sample::stride() const
{
  std::lock_guard lock(_mutex);
  return _stride;
}

std::size_t sort_key_sample::host_key_rows() const
{
  std::lock_guard lock(_mutex);
  return held_key_rows();
}

void sort_key_sample::clear()
{
  std::unordered_map<std::uint64_t, std::shared_ptr<piece const>> dropped;
  {
    std::lock_guard lock(_mutex);
    dropped.swap(_pieces);
  }
}

sort_key_sample::device_keys sort_key_sample::on_device(cucascade::memory::memory_space& space,
                                                        ::cuda::stream_ref stream) const
{
  device_keys out;
  {
    std::lock_guard lock(_mutex);
    out.stride = _stride;
    out.input  = _seen;
    out.pieces.reserve(_pieces.size());
    for (auto const& [id, entry] : _pieces) {
      out.pieces.push_back(entry);
    }
  }
  std::sort(out.pieces.begin(), out.pieces.end(), [](auto const& a, auto const& b) {
    return a->batch_id < b->batch_id;
  });

  auto const mr = space.get_default_allocator();
  for (auto const& entry : out.pieces) {
    if (key_row_count(entry->rows, out.stride) == 0) { continue; }
    // out.pieces keeps the host copy alive while the device copy is in flight.
    auto copied = to_device(entry->data, stream, mr);
    auto view =
      cudf::unpack(entry->metadata->data(), static_cast<std::uint8_t const*>(copied->data()));
    if (entry->stride != out.stride) {
      auto thinned = thin(view, entry->stride, out.stride, stream, mr);
      view         = thinned->view();
      out.thinned.push_back(std::move(thinned));
      // Only the thinned rows stay on the GPU, so the copies never add up to more than the cap.
      stream.sync();
      copied.reset();
    }
    if (copied) { out.buffers.push_back(std::move(copied)); }
    out.key_rows += static_cast<std::size_t>(view.num_rows());
    out.tables.push_back(view);
  }
  return out;
}

}  // namespace sirius::op
