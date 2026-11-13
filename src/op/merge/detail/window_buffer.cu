/*
 * Copyright 2025, Sirius Contributors.
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

#include "op/merge/detail/window_buffer.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda_runtime_api.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <cucascade/error.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace sirius::op::merge::detail {

namespace {

constexpr std::size_t max_int32_chars  = std::numeric_limits<int32_t>::max();
constexpr std::size_t min_chars_bytes  = 64;
constexpr std::size_t chars_copy_chunk = 16;

/// Device functor: word `j` of a mask whose bit 0 is bit `begin_bit` of `src`.
struct shift_mask_word {
  const cudf::bitmask_type* src;
  std::size_t begin_bit;
  std::size_t src_words;

  __device__ cudf::bitmask_type operator()(std::size_t j) const
  {
    constexpr std::size_t word_bits = 8 * sizeof(cudf::bitmask_type);
    auto const bit                  = begin_bit + j * word_bits;
    auto const word                 = bit / word_bits;
    auto const shift                = bit % word_bits;
    auto out                        = src[word] >> shift;
    if (shift != 0 && word + 1 < src_words) { out |= src[word + 1] << (word_bits - shift); }
    return out;
  }
};

/// Device functor: rebases an offset by the offset of the first kept row.
template <typename Offset>
struct rebase_offset {
  const Offset* base;

  __device__ Offset operator()(Offset offset) const { return offset - *base; }
};

/// Device functor: copies chunk `i` of the chars of rows [begin_row, end_row) to the front of
/// `dst`. The byte range is read from the offsets on the device, so the host needs no sync.
template <typename Offset>
struct copy_tail_chars {
  const char* src;
  char* dst;
  const Offset* offsets;
  std::size_t begin_row;
  std::size_t end_row;

  __device__ void operator()(std::size_t i) const
  {
    auto const first = static_cast<std::size_t>(offsets[begin_row]);
    auto const count = static_cast<std::size_t>(offsets[end_row]) - first;
    auto const begin = i * chars_copy_chunk;
    auto const end   = begin + chars_copy_chunk < count ? begin + chars_copy_chunk : count;
    for (auto b = begin; b < end; ++b) {
      dst[b] = src[first + b];
    }
  }
};

/// The buffers of one column in one slot.
struct column_slot {
  rmm::device_buffer data;     ///< Fixed-width values, or STRING chars.
  rmm::device_buffer mask;     ///< Validity bitmask.
  rmm::device_buffer offsets;  ///< STRING offsets; empty for fixed-width columns.
};

struct column_state {
  cudf::data_type type;
  std::size_t width{0};         ///< Bytes per value; 0 for STRING.
  std::size_t offset_width{0};  ///< Bytes per STRING offset (4 or 8); 0 for fixed-width columns.
  /// Chars bytes in the active slot. After a partial consume it is only an upper bound (the
  /// exact end is on the device) until the next refill reads it back.
  std::size_t chars_end{0};
  bool chars_end_exact{true};
  std::array<column_slot, 2> slots;

  [[nodiscard]] bool is_string() const noexcept { return type.id() == cudf::type_id::STRING; }
};

/// n * est * 1.25, saturating.
std::size_t chars_capacity(std::size_t rows, std::size_t est_bytes_per_row)
{
  constexpr auto max = std::numeric_limits<std::size_t>::max();
  if (est_bytes_per_row != 0 && rows > max / est_bytes_per_row) { return max; }
  auto const bytes = rows * est_bytes_per_row;
  auto const slack = bytes / 4;
  return bytes > max - slack ? max : bytes + slack;
}

template <typename Offset>
void carry_string_tail(const column_slot& src,
                       column_slot& dst,
                       std::size_t consumed,
                       std::size_t size,
                       std::size_t chars_bound,
                       ::cuda::stream_ref stream)
{
  auto const* src_offsets = static_cast<const Offset*>(src.offsets.data());
  thrust::transform(rmm::exec_policy_nosync(stream),
                    src_offsets + consumed,
                    src_offsets + size + 1,
                    static_cast<Offset*>(dst.offsets.data()),
                    rebase_offset<Offset>{src_offsets + consumed});
  auto const chunks = (chars_bound + chars_copy_chunk - 1) / chars_copy_chunk;
  thrust::for_each_n(rmm::exec_policy_nosync(stream),
                     thrust::counting_iterator<std::size_t>(0),
                     chunks,
                     copy_tail_chars<Offset>{static_cast<const char*>(src.data.data()),
                                             static_cast<char*>(dst.data.data()),
                                             src_offsets,
                                             consumed,
                                             size});
}

}  // anonymous namespace

struct window_buffer::impl {
  std::vector<column_state> columns;
  std::size_t capacity{0};
  std::size_t size{0};
  std::size_t active{0};
  rmm::device_async_resource_ref mr;
  /// Stream of the last device work on the slots; later work on another stream waits for it.
  cudaStream_t last_stream{nullptr};
  /// Null counts of the active slot, computed by view() when stale.
  mutable std::vector<cudf::size_type> null_counts;
  mutable bool null_counts_valid{true};

  impl(std::size_t capacity_rows, rmm::device_async_resource_ref resource, cudaStream_t stream)
    : capacity(capacity_rows), mr(resource), last_stream(stream)
  {
  }

  /// Orders work about to be enqueued on @p stream after the previous work on the slots, without
  /// blocking the host.
  void order_after_last(cudaStream_t stream)
  {
    if (stream == last_stream) { return; }
    cudaEvent_t event{};
    CUCASCADE_CUDA_TRY(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    auto const record_status = cudaEventRecord(event, last_stream);
    auto const wait_status =
      record_status == cudaSuccess ? cudaStreamWaitEvent(stream, event, 0) : record_status;
    // The event's resources are released once the event has completed.
    CUCASCADE_ASSERT_CUDA_SUCCESS(cudaEventDestroy(event));
    CUCASCADE_CUDA_TRY(wait_status);
    last_stream = stream;
  }

  /// Reads back the chars end of the STRING columns whose end only the device knows.
  void resolve_chars_ends(cudaStream_t stream)
  {
    std::vector<int64_t> ends(columns.size(), 0);
    bool pending = false;
    for (std::size_t i = 0; i < columns.size(); ++i) {
      auto& col = columns[i];
      if (!col.is_string() || col.chars_end_exact) { continue; }
      auto const* last_offset = static_cast<const std::uint8_t*>(col.slots[active].offsets.data()) +
                                size * col.offset_width;
      // Little-endian: an int32 offset fills the low half of the zeroed int64.
      CUCASCADE_CUDA_TRY(
        cudaMemcpyAsync(&ends[i], last_offset, col.offset_width, cudaMemcpyDeviceToHost, stream));
      pending = true;
    }
    if (!pending) { return; }
    CUCASCADE_CUDA_TRY(cudaStreamSynchronize(stream));
    for (std::size_t i = 0; i < columns.size(); ++i) {
      auto& col = columns[i];
      if (!col.is_string() || col.chars_end_exact) { continue; }
      col.chars_end       = static_cast<std::size_t>(ends[i]);
      col.chars_end_exact = true;
    }
  }

  [[nodiscard]] std::vector<cucascade::column_destination> destinations()
  {
    std::vector<cucascade::column_destination> dst(columns.size());
    for (std::size_t i = 0; i < columns.size(); ++i) {
      auto& col        = columns[i];
      auto& slot       = col.slots[active];
      auto& d          = dst[i];
      d.type_id        = static_cast<int32_t>(col.type.id());
      d.data           = {slot.data.data(), slot.data.size()};
      d.null_mask      = {slot.mask.data(), slot.mask.size()};
      d.dst_row_offset = size;
      if (col.is_string()) {
        d.offsets         = {slot.offsets.data(), slot.offsets.size()};
        d.offset_width    = static_cast<uint8_t>(col.offset_width);
        d.dst_byte_offset = col.chars_end;
      }
    }
    return dst;
  }

  /// Doubles the chars of every STRING column in both slots, keeping the active slot's bytes.
  /// Returns false if no column could grow.
  bool grow_chars(::cuda::stream_ref stream)
  {
    std::vector<std::array<rmm::device_buffer, 2>> grown(columns.size());
    bool any = false;
    for (std::size_t i = 0; i < columns.size(); ++i) {
      auto& col = columns[i];
      if (!col.is_string()) { continue; }
      auto const old_bytes = col.slots[active].data.size();
      auto new_bytes       = std::max(2 * old_bytes, min_chars_bytes);
      if (col.offset_width == sizeof(int32_t)) { new_bytes = std::min(new_bytes, max_int32_chars); }
      if (new_bytes <= old_bytes) { continue; }
      for (auto& buffer : grown[i]) {
        buffer = rmm::device_buffer(new_bytes, stream, mr);
      }
      if (col.chars_end > 0) {
        CUCASCADE_CUDA_TRY(cudaMemcpyAsync(grown[i][active].data(),
                                           col.slots[active].data.data(),
                                           col.chars_end,
                                           cudaMemcpyDeviceToDevice,
                                           stream.get()));
      }
      any = true;
    }
    if (!any) { return false; }
    // The old chars are freed below; the copies out of them must have run by then.
    CUCASCADE_CUDA_TRY(cudaStreamSynchronize(stream.get()));
    for (std::size_t i = 0; i < columns.size(); ++i) {
      if (grown[i][0].size() == 0) { continue; }
      columns[i].slots[0].data = std::move(grown[i][0]);
      columns[i].slots[1].data = std::move(grown[i][1]);
    }
    return true;
  }
};

window_buffer::window_buffer(std::vector<cudf::data_type> fetch_types,
                             std::size_t window_rows,
                             std::size_t est_string_bytes_per_row,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  if (window_rows > static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()) - 1) {
    throw std::invalid_argument("window_buffer: window_rows exceeds the cudf row limit");
  }
  for (auto const& type : fetch_types) {
    if (type.id() != cudf::type_id::STRING && !cudf::is_fixed_width(type)) {
      throw std::invalid_argument("window_buffer: unsupported column type id " +
                                  std::to_string(static_cast<int32_t>(type.id())));
    }
  }

  auto const sref = ::cuda::stream_ref{stream.value()};
  auto state      = std::make_unique<impl>(window_rows, mr, stream.value());
  state->columns.reserve(fetch_types.size());
  state->null_counts.assign(fetch_types.size(), 0);
  auto const mask_bytes =
    cudf::bitmask_allocation_size_bytes(static_cast<cudf::size_type>(window_rows));
  for (auto const& type : fetch_types) {
    auto& col = state->columns.emplace_back();
    col.type  = type;
    std::size_t data_bytes{0};
    std::size_t offsets_bytes{0};
    if (col.is_string()) {
      auto const est_chars = chars_capacity(window_rows, est_string_bytes_per_row);
      auto const int32_ok =
        est_string_bytes_per_row == 0 || window_rows <= max_int32_chars / est_string_bytes_per_row;
      col.offset_width = int32_ok ? sizeof(int32_t) : sizeof(int64_t);
      data_bytes       = std::max(est_chars, min_chars_bytes);
      if (int32_ok) { data_bytes = std::min(data_bytes, max_int32_chars); }
      offsets_bytes = (window_rows + 1) * col.offset_width;
    } else {
      col.width  = cudf::size_of(type);
      data_bytes = window_rows * col.width;
    }
    for (auto& slot : col.slots) {
      slot.data    = rmm::device_buffer(data_bytes, sref, mr);
      slot.mask    = rmm::device_buffer(mask_bytes, sref, mr);
      slot.offsets = rmm::device_buffer(offsets_bytes, sref, mr);
    }
  }
  _impl = std::move(state);
}

window_buffer::window_buffer(window_buffer&&) noexcept            = default;
window_buffer& window_buffer::operator=(window_buffer&&) noexcept = default;
window_buffer::~window_buffer()                                   = default;

cudf::table_view window_buffer::view() const
{
  auto const& state = *_impl;
  auto const rows   = static_cast<cudf::size_type>(state.size);
  if (!state.null_counts_valid) {
    // Counting reads the masks back, so it waits for the work enqueued on the slots.
    for (std::size_t i = 0; i < state.columns.size(); ++i) {
      auto const* mask =
        static_cast<const cudf::bitmask_type*>(state.columns[i].slots[state.active].mask.data());
      state.null_counts[i] =
        rows == 0 ? 0 : cudf::null_count(mask, 0, rows, ::cuda::stream_ref{state.last_stream});
    }
    state.null_counts_valid = true;
  }

  std::vector<cudf::column_view> views;
  views.reserve(state.columns.size());
  for (std::size_t i = 0; i < state.columns.size(); ++i) {
    auto const& col  = state.columns[i];
    auto const& slot = col.slots[state.active];
    if (rows == 0) {
      views.emplace_back(col.type, 0, nullptr, nullptr, 0);
      continue;
    }
    auto const null_count = state.null_counts[i];
    auto const* mask =
      null_count > 0 ? static_cast<const cudf::bitmask_type*>(slot.mask.data()) : nullptr;
    if (col.is_string()) {
      auto const offsets_type = cudf::data_type{
        col.offset_width == sizeof(int32_t) ? cudf::type_id::INT32 : cudf::type_id::INT64};
      cudf::column_view offsets(offsets_type, rows + 1, slot.offsets.data(), nullptr, 0);
      views.emplace_back(
        col.type, rows, slot.data.data(), mask, null_count, 0, std::vector{offsets});
    } else {
      views.emplace_back(col.type, rows, slot.data.data(), mask, null_count);
    }
  }
  return cudf::table_view{views};
}

std::size_t window_buffer::size() const noexcept { return _impl->size; }

std::size_t window_buffer::capacity_rows() const noexcept { return _impl->capacity; }

std::size_t window_buffer::device_bytes() const noexcept
{
  std::size_t bytes{0};
  for (auto const& col : _impl->columns) {
    for (auto const& slot : col.slots) {
      bytes += slot.data.size() + slot.mask.size() + slot.offsets.size();
    }
  }
  return bytes;
}

void window_buffer::consume(std::size_t consumed, rmm::cuda_stream_view stream)
{
  auto& state = *_impl;
  if (consumed > state.size) {
    throw std::out_of_range("window_buffer::consume: consumed exceeds the window size");
  }
  if (consumed == 0) { return; }
  if (consumed == state.size) {
    state.size = 0;
    for (auto& col : state.columns) {
      col.chars_end       = 0;
      col.chars_end_exact = true;
    }
    std::fill(state.null_counts.begin(), state.null_counts.end(), 0);
    state.null_counts_valid = true;
    return;
  }

  auto const sref = ::cuda::stream_ref{stream.value()};
  state.order_after_last(stream.value());
  auto const tail = state.size - consumed;
  auto const src_words =
    static_cast<std::size_t>(cudf::num_bitmask_words(static_cast<cudf::size_type>(state.size)));
  auto const dst_words =
    static_cast<std::size_t>(cudf::num_bitmask_words(static_cast<cudf::size_type>(tail)));
  for (auto& col : state.columns) {
    auto const& src = col.slots[state.active];
    auto& dst       = col.slots[1 - state.active];
    thrust::transform(
      rmm::exec_policy_nosync(sref),
      thrust::counting_iterator<std::size_t>(0),
      thrust::counting_iterator<std::size_t>(dst_words),
      static_cast<cudf::bitmask_type*>(dst.mask.data()),
      shift_mask_word{
        static_cast<const cudf::bitmask_type*>(src.mask.data()), consumed, src_words});
    if (!col.is_string()) {
      CUCASCADE_CUDA_TRY(
        cudaMemcpyAsync(dst.data.data(),
                        static_cast<const std::uint8_t*>(src.data.data()) + consumed * col.width,
                        tail * col.width,
                        cudaMemcpyDeviceToDevice,
                        stream.value()));
    } else if (col.offset_width == sizeof(int32_t)) {
      carry_string_tail<int32_t>(src, dst, consumed, state.size, col.chars_end, sref);
      col.chars_end_exact = false;
    } else {
      carry_string_tail<int64_t>(src, dst, consumed, state.size, col.chars_end, sref);
      col.chars_end_exact = false;
    }
  }
  state.active            = 1 - state.active;
  state.size              = tail;
  state.null_counts_valid = false;
}

cucascade::row_range_result window_buffer::refill(
  const cucascade::row_range_reader_registry& readers,
  const cucascade::read_only_data_batch& batch,
  std::span<const std::size_t> fetch_cols,
  std::size_t row_begin,
  std::unique_ptr<cucascade::row_range_reader_state>* state,
  cucascade::memory::reservation* scratch,
  rmm::cuda_stream_view stream)
{
  auto& self = *_impl;
  if (fetch_cols.size() != self.columns.size()) {
    throw std::invalid_argument("window_buffer::refill: fetch_cols does not match the columns");
  }
  auto const sref = ::cuda::stream_ref{stream.value()};
  self.order_after_last(stream.value());
  self.resolve_chars_ends(stream.value());

  cucascade::row_range_request request;
  request.columns   = fetch_cols;
  request.row_begin = row_begin;
  request.max_rows  = self.capacity - self.size;

  // The reader may write past size() even when it throws; the masks are stale from here on.
  self.null_counts_valid = false;
  auto dst               = self.destinations();
  auto result = cucascade::read_rows_into(readers, batch, request, dst, sref, scratch, state);
  if (result.rows_written == 0 && !result.source_exhausted && request.max_rows > 0) {
    // One row does not fit a STRING slot.
    if (self.grow_chars(sref)) {
      dst    = self.destinations();
      result = cucascade::read_rows_into(readers, batch, request, dst, sref, scratch, state);
    }
    if (result.rows_written == 0 && !result.source_exhausted) {
      throw std::runtime_error(
        "window_buffer::refill: a single row does not fit the window's STRING slot");
    }
  }

  self.size += result.rows_written;
  for (std::size_t i = 0; i < self.columns.size(); ++i) {
    if (self.columns[i].is_string()) { self.columns[i].chars_end += result.data_bytes_written[i]; }
  }
  return result;
}

}  // namespace sirius::op::merge::detail
