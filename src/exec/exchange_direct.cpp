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

#include "exec/exchange_direct.hpp"

#include "data/data_batch_utils.hpp"
#include "data/sirius_converter_registry.hpp"
#include "exec/exchange_staging.hpp"
#include "log/logging.hpp"
#include "sirius/exception.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/error.hpp>

#include <cuda_runtime_api.h>

#include <absl/cleanup/cleanup.h>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <format>
#include <functional>
#include <limits>
#include <numeric>
#include <string_view>

namespace sirius::exec {
namespace {

// "SXD1" | i32 rows | u32 ncols | per column { i32 type_id | i32 scale | i32 null_count |
// u8 has_mask | STRING only: i32 offsets type_id | u64 chars }, copied as native integers.
static_assert(std::endian::native == std::endian::little, "the direct layout is little-endian");

constexpr std::array<char, 4> magic{'S', 'X', 'D', '1'};
constexpr std::size_t min_column_bytes = 13;
constexpr std::size_t alignment        = 256;

void require(bool ok, std::string_view what)
{
  if (!ok) { throw sirius::invalid_input_exception("direct layout: {}", what); }
}

template <typename T>
void put(std::vector<std::uint8_t>& out, T value)
{
  auto const* bytes = reinterpret_cast<std::uint8_t const*>(&value);
  out.insert(out.end(), bytes, bytes + sizeof(T));
}

template <typename T>
T take(std::span<std::uint8_t const>& in)
{
  require(in.size() >= sizeof(T), "truncated");
  T value;
  std::memcpy(&value, in.data(), sizeof(T));
  in = in.subspan(sizeof(T));
  return value;
}

bool is_string(cudf::data_type type) { return type.id() == cudf::type_id::STRING; }

bool is_sendable(cudf::data_type type)
{
  // EMPTY is checked first: is_fixed_width throws on it, since the type dispatcher omits it.
  return is_string(type) || (type.id() != cudf::type_id::EMPTY && cudf::is_fixed_width(type));
}

bool is_sliced(cudf::table_view const& table)
{
  // One level of children is enough: the deepest sendable column is STRING (one offsets child).
  // A nested column may be checked here, but describe_table() refuses it whichever way this
  // answers, so a deeper slice is never sent.
  return std::any_of(table.begin(), table.end(), [](cudf::column_view const& c) {
    return c.offset() != 0 ||
           std::any_of(c.child_begin(), c.child_end(), [](cudf::column_view const& child) {
             return child.offset() != 0;
           });
  });
}

std::uintptr_t address(void const* p) { return reinterpret_cast<std::uintptr_t>(p); }

/// The [address, length] of each non-empty buffer of @p table, or nullopt if one is outside
/// @p region.
std::optional<std::vector<std::uint64_t>> sources(direct_export const& table,
                                                  memory::slab_region const& region)
{
  auto const plan = plan_buffers(table.layout, std::numeric_limits<std::size_t>::max());
  std::vector<std::uint64_t> src;
  for (std::size_t i = 0; i < plan.size(); ++i) {
    if (plan[i].wire == 0) { continue; }
    auto const at = address(table.buffers[i]);
    if (!region.covers(at, plan[i].wire)) { return std::nullopt; }
    src.push_back(at);
    src.push_back(plan[i].wire);
  }
  return src;
}

}  // namespace

std::vector<std::uint8_t> encode_layout(direct_layout const& layout)
{
  std::vector<std::uint8_t> out(magic.begin(), magic.end());
  put(out, layout.rows);
  put(out, static_cast<std::uint32_t>(layout.columns.size()));
  for (auto const& c : layout.columns) {
    put(out, static_cast<std::int32_t>(c.type.id()));
    put(out, c.type.scale());
    put(out, c.null_count);
    put(out, static_cast<std::uint8_t>(c.has_mask));
    if (is_string(c.type)) {
      put(out, static_cast<std::int32_t>(c.offsets));
      put(out, c.chars);
    }
  }
  return out;
}

direct_layout decode_layout(std::span<std::uint8_t const> in)
{
  require(take<std::array<char, 4>>(in) == magic, "bad magic");
  direct_layout layout{take<cudf::size_type>(in), {}};
  // rows + 1 string offsets must fit in a cudf::size_type.
  require(layout.rows >= 1 && layout.rows < std::numeric_limits<cudf::size_type>::max(),
          "row count out of range");
  auto const ncols = take<std::uint32_t>(in);
  // Bounded by the bytes left before reserving, so a forged count cannot force a huge allocation.
  require(ncols >= 1 && ncols <= in.size() / min_column_bytes, "column count out of range");
  layout.columns.reserve(ncols);
  for (std::uint32_t i = 0; i < ncols; ++i) {
    auto const id = take<std::int32_t>(in);
    require(id >= 0 && id < static_cast<std::int32_t>(cudf::type_id::NUM_TYPE_IDS),
            "type id out of range");
    cudf::data_type type{static_cast<cudf::type_id>(id)};
    require(is_sendable(type), "type is neither fixed-width nor STRING");
    auto const scale = take<std::int32_t>(in);
    if (cudf::is_fixed_point(type)) {
      type = cudf::data_type{type.id(), scale};
    } else {
      require(scale == 0, "scale on a type that is not fixed-point");
    }
    direct_column c{type, take<cudf::size_type>(in), false};
    auto const has_mask = take<std::uint8_t>(in);
    require(has_mask <= 1, "mask flag is not 0 or 1");
    c.has_mask = has_mask == 1;
    require(c.null_count >= 0 && c.null_count <= layout.rows, "null count out of range");
    require(c.null_count == 0 || c.has_mask, "nulls without a mask");
    if (is_string(type)) {
      c.offsets = static_cast<cudf::type_id>(take<std::int32_t>(in));
      c.chars   = take<std::uint64_t>(in);
      require(c.offsets == cudf::type_id::INT32 || c.offsets == cudf::type_id::INT64,
              "string offsets are not INT32 or INT64");
      require(c.offsets == cudf::type_id::INT64 ||
                c.chars <= static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()),
              "chars overflow INT32 offsets");
    }
    layout.columns.push_back(c);
  }
  require(in.empty(), "trailing bytes");
  return layout;
}

std::vector<direct_buffer> plan_buffers(direct_layout const& layout, std::size_t limit)
{
  std::vector<direct_buffer> plan;
  std::size_t total = 0;
  auto const add    = [&](std::size_t column, std::size_t wire, std::size_t alloc) {
    // Counted in whole alignment units, so neither rounding alloc up nor the sum can wrap.
    auto const units = alloc / alignment + (alloc % alignment != 0);
    if (units > (limit - total) / alignment) {
      throw sirius::invalid_input_exception("direct layout: buffers exceed {} bytes", limit);
    }
    total += units * alignment;
    plan.push_back({column, wire, alloc});
  };
  auto const rows = static_cast<std::size_t>(layout.rows);
  for (std::size_t i = 0; i < layout.columns.size(); ++i) {
    auto const& c = layout.columns[i];
    if (c.has_mask) {
      add(i,
          cudf::num_bitmask_words(layout.rows) * sizeof(cudf::bitmask_type),
          cudf::bitmask_allocation_size_bytes(layout.rows));
    }
    if (is_string(c.type)) {
      auto const offsets = (rows + 1) * cudf::size_of(cudf::data_type{c.offsets});
      add(i, c.chars, c.chars);
      add(i, offsets, offsets);
    } else {
      auto const data = rows * cudf::size_of(c.type);
      add(i, data, data);
    }
  }
  return plan;
}

direct_export describe_table(cudf::table_view const& table, rmm::cuda_stream_view stream)
{
  direct_export out{{table.num_rows(), {}}, {}};
  for (auto const& c : table) {
    if (!is_sendable(c.type())) {
      throw sirius::invalid_input_exception("direct exchange cannot send type id {}",
                                            static_cast<std::int32_t>(c.type().id()));
    }
    direct_column d{c.type(), c.null_count(), c.nullable()};
    if (d.has_mask) { out.buffers.push_back(c.null_mask()); }
    out.buffers.push_back(c.head());
    if (is_string(c.type())) {
      cudf::strings_column_view const strings{c};
      d.offsets = strings.offsets().type().id();
      d.chars   = strings.chars_size(stream);
      out.buffers.push_back(strings.offsets().head());
    }
    out.layout.columns.push_back(d);
  }
  return out;
}

direct_exchange::direct_exchange(cucascade::memory::memory_space& gpu, memory::slab_region region)
  : _gpu{gpu},
    _region{std::move(region)},
    _sealed{std::make_shared<cucascade::shared_data_repository>()}
{
}

std::optional<direct_exchange::exported> direct_exchange::export_batch(
  std::shared_ptr<cucascade::data_batch> batch)
{
  bring_to_gpu(*batch);
  std::lock_guard const lock{_mutex};
  rmm::cuda_set_device_raii const device{rmm::cuda_device_id{_region.device}};
  require_open();
  rmm::cuda_stream_view const stream{_gpu.acquire_stream()};
  direct_export described;
  std::optional<std::vector<std::uint64_t>> src;
  std::unique_ptr<cudf::table> copy;
  {
    // Dropped before returning: it holds the batch's shared lock, which only this thread may
    // unlock.
    auto const ro = batch->to_read_only();
    if (ro.get_current_tier() != cucascade::memory::Tier::GPU) {
      throw sirius::invalid_input_exception("direct exchange: the batch is not on the GPU");
    }
    auto const view = get_cudf_table_view(ro);
    if (view.num_rows() == 0) { return std::nullopt; }
    if (cudaEvent_t writer = ro.get_writer_event()) {
      RMM_CUDA_TRY(cudaStreamWaitEvent(stream.value(), writer, 0));
    }
    if (!is_sliced(view)) {
      described = describe_table(view, stream);
      src       = sources(described, _region);
    }
    if (!src) {
      copy      = std::make_unique<cudf::table>(view, stream, _gpu.get_default_allocator());
      described = describe_table(copy->view(), stream);
      src       = sources(described, _region);
    }
    // The NIC reads outside stream order, so the producer and the copy must be done.
    stream.synchronize();
  }
  if (!src) {
    throw sirius::internal_exception(
      "direct exchange: a batch copied for sending is not in the slab");
  }
  std::shared_ptr<void const> keepalive = std::move(batch);
  if (copy) { keepalive = std::move(copy); }
  auto const token = _next++;
  _entries.emplace(token, std::move(keepalive));
  return exported{token,
                  static_cast<std::uint64_t>(described.layout.rows),
                  encode_layout(described.layout),
                  std::move(*src)};
}

std::pair<std::uint64_t, std::vector<std::uint64_t>> direct_exchange::allocate(
  std::span<std::uint8_t const> bytes)
{
  auto layout     = decode_layout(bytes);
  auto const plan = plan_buffers(layout, _region.len);
  // The tracker charges each allocation rounded up on its own; plan_buffers bounded this sum.
  auto const total = std::transform_reduce(
    plan.begin(), plan.end(), std::size_t{0}, std::plus<>{}, [](direct_buffer const& b) {
      return rmm::align_up(b.alloc, alignment);
    });

  // Never the blocking make_reservation: a receiver waiting for memory could hold up the senders
  // whose batches would free it. Making room spills to host instead, which waits on no transfer,
  // and runs without the lock so other tokens move meanwhile.
  auto reservation = reserve(total);
  std::lock_guard const lock{_mutex};
  rmm::cuda_set_device_raii const device{rmm::cuda_device_id{_region.device}};
  require_open();
  if (!reservation) {
    throw rmm::out_of_memory(std::format(
      "direct exchange: {} bytes requested, {} available", total, _gpu.get_available_memory()));
  }
  auto const stream = _gpu.acquire_stream();
  auto* tracker     = _gpu.get_memory_resource_of<cucascade::memory::Tier::GPU>();
  if (!tracker->attach_reservation_to_tracker(
        stream,
        std::move(reservation),
        std::make_unique<cucascade::memory::fail_reservation_limit_policy>())) {
    throw sirius::internal_exception("direct exchange: this thread already tracks a reservation");
  }
  absl::Cleanup detach = [&] { tracker->reset_stream_reservation(stream); };
  received entry{std::move(layout), {}};
  entry.buffers.reserve(plan.size());
  for (auto const& b : plan) {
    entry.buffers.emplace_back(b.alloc, stream, _gpu.get_default_allocator());
  }
  // The pool reuses a block freed on another stream behind a stream wait, which the NIC ignores.
  rmm::cuda_stream_view{stream}.synchronize();

  std::vector<std::uint64_t> dst;
  for (std::size_t i = 0; i < plan.size(); ++i) {
    if (plan[i].wire == 0) { continue; }
    auto const at = address(entry.buffers[i].data());
    if (!_region.covers(at, plan[i].wire)) {
      throw sirius::internal_exception("direct exchange: a receive buffer is not in the slab");
    }
    dst.push_back(at);
    dst.push_back(plan[i].wire);
  }
  auto const token = _next++;
  _entries.emplace(token, std::move(entry));
  return {token, std::move(dst)};
}

std::unique_ptr<cudf::table> direct_exchange::take(std::uint64_t token)
{
  std::lock_guard const lock{_mutex};
  rmm::cuda_set_device_raii const device{rmm::cuda_device_id{_region.device}};
  require_open();
  auto const it     = _entries.find(token);
  auto* const entry = it == _entries.end() ? nullptr : std::get_if<received>(&it->second);
  if (entry == nullptr) {
    throw sirius::invalid_input_exception("direct exchange: token {} holds no received batch",
                                          token);
  }
  auto taken = std::move(*entry);
  _entries.erase(it);
  return to_table(std::move(taken));
}

std::unique_ptr<cudf::table> direct_exchange::to_table(received&& taken)
{
  auto const rows = taken.layout.rows;
  auto buffer     = taken.buffers.begin();
  std::vector<std::unique_ptr<cudf::column>> columns;
  for (auto const& c : taken.layout.columns) {
    rmm::device_buffer mask = c.has_mask ? std::move(*buffer++) : rmm::device_buffer{};
    rmm::device_buffer data = std::move(*buffer++);
    if (is_string(c.type)) {
      auto offsets = std::make_unique<cudf::column>(
        cudf::data_type{c.offsets}, rows + 1, std::move(*buffer++), rmm::device_buffer{}, 0);
      columns.push_back(cudf::make_strings_column(
        rows, std::move(offsets), std::move(data), c.null_count, std::move(mask)));
    } else {
      columns.push_back(std::make_unique<cudf::column>(
        c.type, rows, std::move(data), std::move(mask), c.null_count));
    }
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

void direct_exchange::release(std::uint64_t token)
{
  std::lock_guard const lock{_mutex};
  rmm::cuda_set_device_raii const device{rmm::cuda_device_id{_region.device}};
  require_open();
  _entries.erase(token);
  if (auto const sealed = _sealed_ids.find(token); sealed != _sealed_ids.end()) {
    (void)_sealed->pop_data_batch_by_id(sealed->second);
    _sealed_ids.erase(sealed);
  }
}

std::size_t direct_exchange::outstanding() const
{
  std::lock_guard const lock{_mutex};
  require_open();
  return _entries.size() + _sealed_ids.size();
}

void direct_exchange::close()
{
  std::lock_guard const lock{_mutex};
  rmm::cuda_set_device_raii const device{rmm::cuda_device_id{_region.device}};
  _entries.clear();
  for (auto const& [token, batch_id] : _sealed_ids) {
    (void)_sealed->pop_data_batch_by_id(batch_id);
  }
  _sealed_ids.clear();
  _closed = true;
}

void direct_exchange::enable_spill(exchange_staging& staging, make_room_fn make_room)
{
  std::lock_guard const lock{_mutex};
  _make_room = std::move(make_room);
  staging.track(_sealed);
}

void direct_exchange::seal(std::uint64_t token)
{
  std::lock_guard const lock{_mutex};
  rmm::cuda_set_device_raii const device{rmm::cuda_device_id{_region.device}};
  require_open();
  if (_sealed_ids.contains(token)) { return; }
  auto const it     = _entries.find(token);
  auto* const entry = it == _entries.end() ? nullptr : std::get_if<received>(&it->second);
  // A duplicate frame can name a token its first copy already delivered and its receiver
  // consumed; like release(), that is not an error.
  if (entry == nullptr) { return; }
  auto taken = std::move(*entry);
  _entries.erase(it);
  auto batch = sirius::make_data_batch(
    std::move(*to_table(std::move(taken))), _gpu, _gpu.acquire_stream(), {});
  _sealed_ids.emplace(token, batch->get_batch_id());
  _sealed->add_data_batch(std::move(batch));
}

std::shared_ptr<cucascade::data_batch> direct_exchange::take_batch(std::uint64_t token)
{
  {
    std::lock_guard const lock{_mutex};
    require_open();
    if (auto const sealed = _sealed_ids.find(token); sealed != _sealed_ids.end()) {
      auto batch = _sealed->pop_data_batch_by_id(sealed->second);
      _sealed_ids.erase(sealed);
      if (!batch) {
        throw sirius::internal_exception("direct exchange: sealed token {} lost its batch", token);
      }
      return batch;
    }
  }
  auto table = take(token);
  return sirius::make_data_batch(std::move(*table), _gpu, _gpu.acquire_stream(), {});
}

std::shared_ptr<cucascade::data_batch> direct_exchange::peek_batch(std::uint64_t token) const
{
  std::lock_guard const lock{_mutex};
  require_open();
  auto const sealed = _sealed_ids.find(token);
  auto batch =
    sealed == _sealed_ids.end() ? nullptr : _sealed->get_data_batch_by_id(sealed->second);
  if (!batch) {
    throw sirius::invalid_input_exception("direct exchange: token {} holds no sealed batch", token);
  }
  return batch;
}

namespace {

/// The column at @p column of a GPU-resident batch's view.
cudf::column_view key_column(cudf::table_view const& view, int column)
{
  if (column < 0 || column >= view.num_columns()) {
    throw sirius::invalid_input_exception(
      "direct exchange: key column {} is out of range for a {}-column batch",
      column,
      view.num_columns());
  }
  return view.column(column);
}

std::int64_t integer_value(cudf::scalar const& value, rmm::cuda_stream_view stream)
{
  switch (value.type().id()) {
    case cudf::type_id::INT8:
      return static_cast<cudf::numeric_scalar<std::int8_t> const&>(value).value(stream);
    case cudf::type_id::INT16:
      return static_cast<cudf::numeric_scalar<std::int16_t> const&>(value).value(stream);
    case cudf::type_id::INT32:
      return static_cast<cudf::numeric_scalar<std::int32_t> const&>(value).value(stream);
    case cudf::type_id::INT64:
      return static_cast<cudf::numeric_scalar<std::int64_t> const&>(value).value(stream);
    default:
      throw sirius::invalid_input_exception(
        "direct exchange: key column of type {} is not a "
        "signed integer",
        cudf::type_to_name(value.type()));
  }
}

}  // namespace

void direct_exchange::add_key_stats(cucascade::data_batch& batch, int column, key_stats& stats)
{
  bring_to_gpu(batch);
  rmm::cuda_set_device_raii const device{rmm::cuda_device_id{_region.device}};
  auto const ro = batch.to_read_only();
  if (ro.get_current_tier() != cucascade::memory::Tier::GPU) {
    throw sirius::invalid_input_exception("direct exchange: the batch is not on the GPU");
  }
  auto const col = key_column(get_cudf_table_view(ro), column);
  if (!cudf::is_integral(col.type()) || cudf::is_unsigned(col.type())) {
    throw sirius::invalid_input_exception(
      "direct exchange: key column of type {} is not a "
      "signed integer",
      cudf::type_to_name(col.type()));
  }
  auto const valid = col.size() - col.null_count();
  if (valid == 0) { return; }
  rmm::cuda_stream_view const stream{_gpu.acquire_stream()};
  if (cudaEvent_t writer = ro.get_writer_event()) {
    RMM_CUDA_TRY(cudaStreamWaitEvent(stream.value(), writer, 0));
  }
  auto const [lo, hi] = cudf::minmax(col, stream, _gpu.get_default_allocator());
  stats.rows += static_cast<std::uint64_t>(valid);
  stats.min = std::min(stats.min, integer_value(*lo, stream));
  stats.max = std::max(stats.max, integer_value(*hi, stream));
}

std::shared_ptr<cucascade::data_batch> direct_exchange::copy_column(cucascade::data_batch& batch,
                                                                    int column)
{
  bring_to_gpu(batch);
  rmm::cuda_set_device_raii const device{rmm::cuda_device_id{_region.device}};
  rmm::cuda_stream_view const stream{_gpu.acquire_stream()};
  std::unique_ptr<cudf::table> copy;
  {
    auto const ro = batch.to_read_only();
    if (ro.get_current_tier() != cucascade::memory::Tier::GPU) {
      throw sirius::invalid_input_exception("direct exchange: the batch is not on the GPU");
    }
    auto const col = key_column(get_cudf_table_view(ro), column);
    if (cudaEvent_t writer = ro.get_writer_event()) {
      RMM_CUDA_TRY(cudaStreamWaitEvent(stream.value(), writer, 0));
    }
    copy = std::make_unique<cudf::table>(
      cudf::table_view{std::vector<cudf::column_view>{col}}, stream, _gpu.get_default_allocator());
  }
  return sirius::make_data_batch(std::move(copy), _gpu, stream, {});
}

std::unique_ptr<cucascade::memory::reservation> direct_exchange::reserve(std::size_t bytes)
{
  auto reservation = _gpu.make_reservation_or_null(bytes);
  if (reservation || !_make_room) { return reservation; }
  // A request may fail without having run: closing any query window drains the downgrade
  // executor, which cancels what is queued. Only a request that ran and still left too little
  // is final.
  constexpr int attempts = 3;
  for (int attempt = 1; attempt <= attempts; ++attempt) {
    try {
      _make_room(bytes);
      break;
    } catch (std::exception const& e) {
      SIRIUS_LOG_WARN("direct exchange: making room for {} bytes failed (attempt {} of {}): {}",
                      bytes,
                      attempt,
                      attempts,
                      e.what());
    }
  }
  return _gpu.make_reservation_or_null(bytes);
}

void direct_exchange::bring_to_gpu(cucascade::data_batch& batch)
{
  // Blocking locks: the batch is about to be sent, and whoever holds it now (the downgrade
  // executor mid-spill) releases it soon.
  if (batch.to_read_only().get_current_tier() == cucascade::memory::Tier::GPU) { return; }
  auto mutable_batch = batch.to_mutable();
  if (mutable_batch.get_current_tier() == cucascade::memory::Tier::GPU) { return; }
  auto const bytes = sirius::peak_materialization_bytes(mutable_batch.get_data());
  auto reservation = reserve(bytes);
  if (!reservation) {
    throw rmm::out_of_memory(
      std::format("direct exchange: {} bytes to bring a spilled batch back, {} available",
                  bytes,
                  _gpu.get_available_memory()));
  }
  auto const stream = _gpu.acquire_stream();
  auto* tracker     = _gpu.get_memory_resource_of<cucascade::memory::Tier::GPU>();
  if (!tracker->attach_reservation_to_tracker(stream, std::move(reservation))) {
    throw sirius::internal_exception("direct exchange: this thread already tracks a reservation");
  }
  absl::Cleanup detach = [&] { tracker->reset_stream_reservation(stream); };
  mutable_batch.convert_to<cucascade::gpu_table_representation>(
    sirius::converter_registry::get(), &_gpu, stream);
}

void direct_exchange::require_open() const
{
  if (_closed) { throw sirius::invalid_input_exception("direct exchange: already closed"); }
}

}  // namespace sirius::exec
