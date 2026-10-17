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

#include "exec/exchange_ring_ends.hpp"

#include "compression/compressed_representation.hpp"
#include "data/data_batch_utils.hpp"
#include "exec/exchange_inbox.hpp"
#include "log/logging.hpp"
#include "sirius/exception.hpp"
#include "telemetry/data_batch_probe.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/error.hpp>

#include <cuda_runtime_api.h>

#include <absl/cleanup/cleanup.h>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/cudf/host_table.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/data/disk_io_backend.hpp>
#include <cucascade/memory/disk_table.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <format>
#include <utility>

namespace sirius::exec {

using cucascade::memory::column_metadata;
using cucascade::memory::Tier;
using ring::batch_kind;

namespace {

constexpr std::size_t tier_index(Tier tier)
{
  switch (tier) {
    case Tier::GPU: return 0;
    case Tier::HOST: return 1;
    default: return 2;
  }
}

constexpr std::size_t disk_alignment = 4096;

/// Queues device/host copies on @p stream, in one driver call where the runtime has it.
class copy_batch {
 public:
  void add(void* dst, void const* src, std::size_t size)
  {
    if (size == 0) { return; }
    _dsts.push_back(dst);
    _srcs.push_back(src);
    _sizes.push_back(size);
  }

  void flush(rmm::cuda_stream_view stream)
  {
    if (_dsts.empty()) { return; }
#if CUDART_VERSION >= 13000
    cudaMemcpyAttributes attr{};
    attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    attr.flags          = cudaMemcpyFlagDefault;
    RMM_CUDA_TRY(cudaMemcpyBatchAsync(
      _dsts.data(), _srcs.data(), _sizes.data(), _dsts.size(), attr, stream.value()));
#else
    for (std::size_t i = 0; i < _dsts.size(); ++i) {
      RMM_CUDA_TRY(
        cudaMemcpyAsync(_dsts[i], _srcs[i], _sizes[i], cudaMemcpyDefault, stream.value()));
    }
#endif
    _dsts.clear();
    _srcs.clear();
    _sizes.clear();
  }

 private:
  std::vector<void*> _dsts;
  std::vector<void const*> _srcs;
  std::vector<std::size_t> _sizes;
};

using blocks_allocation =
  cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation;

/// Copies between the device range [@p device, +@p size) and host blocks @p blocks at logical
/// offset @p at, one piece per block the range touches.
void add_block_copies(copy_batch& copies,
                      blocks_allocation const& blocks,
                      std::size_t at,
                      std::uint8_t* device,
                      std::size_t size,
                      bool to_device)
{
  auto const block_size = blocks.block_size();
  std::size_t done      = 0;
  while (done < size) {
    auto const block  = (at + done) / block_size;
    auto const within = (at + done) % block_size;
    auto const n      = std::min(size - done, block_size - within);
    auto* host        = reinterpret_cast<std::uint8_t*>(blocks.at(block).data()) + within;
    if (to_device) {
      copies.add(device + done, host, n);
    } else {
      copies.add(host, device + done, n);
    }
    done += n;
  }
}

/// Disk offsets, 4 KiB aligned like cuCascade's host-to-disk layout, for @p c's buffers.
column_metadata disk_layout(column_metadata const& c, std::size_t& cursor)
{
  column_metadata d = c;
  d.children.clear();
  if (c.has_null_mask && c.null_mask_size > 0) {
    cursor             = rmm::align_up(cursor, disk_alignment);
    d.null_mask_offset = cursor;
    cursor += c.null_mask_size;
  } else {
    d.null_mask_offset = 0;
  }
  if (c.has_data && c.data_size > 0) {
    cursor        = rmm::align_up(cursor, disk_alignment);
    d.data_offset = cursor;
    cursor += c.data_size;
  } else {
    d.data_offset = 0;
  }
  for (auto const& child : c.children) {
    d.children.push_back(disk_layout(child, cursor));
  }
  return d;
}

/// Rebuilds one column from its buffers, consumed in walk order from @p next.
std::unique_ptr<cudf::column> rebuild(column_metadata const& c,
                                      std::vector<rmm::device_buffer>::iterator& next)
{
  auto const id   = static_cast<cudf::type_id>(c.type_id);
  auto const rows = static_cast<cudf::size_type>(c.num_rows);
  rmm::device_buffer mask =
    c.has_null_mask && c.null_mask_size > 0 ? std::move(*next++) : rmm::device_buffer{};
  rmm::device_buffer data =
    c.has_data && c.data_size > 0 ? std::move(*next++) : rmm::device_buffer{};
  if (id == cudf::type_id::STRING) {
    auto const& o = c.children.front();
    rmm::device_buffer offsets_data =
      o.has_data && o.data_size > 0 ? std::move(*next++) : rmm::device_buffer{};
    auto offsets =
      std::make_unique<cudf::column>(cudf::data_type{static_cast<cudf::type_id>(o.type_id)},
                                     rows + 1,
                                     std::move(offsets_data),
                                     rmm::device_buffer{},
                                     0);
    return cudf::make_strings_column(
      rows, std::move(offsets), std::move(data), c.null_count, std::move(mask));
  }
  cudf::data_type const type =
    cudf::is_fixed_point(cudf::data_type{id}) ? cudf::data_type{id, c.scale} : cudf::data_type{id};
  return std::make_unique<cudf::column>(type, rows, std::move(data), std::move(mask), c.null_count);
}

std::string bytes_text(std::size_t bytes)
{
  return std::format("{:.2f} GiB", bytes / double(1 << 30));
}

}  // namespace

// ---- state ---------------------------------------------------------------------------------

struct ring_exchange::held {
  std::atomic<std::uint64_t> gpu{0};
  std::atomic<std::uint64_t> host{0};
};

struct ring_exchange::job {
  batch_kind kind;
  std::uint64_t stream_bytes;
  /// Subscribed while the job holds it: the job reads its buffers without a lock, so a spill
  /// candidate snapshotted before the batch left its repository must not convert it.
  std::shared_ptr<cucascade::data_batch> batch;
  bool subscribed{false};

  job()                      = default;
  job(job const&)            = delete;
  job& operator=(job const&) = delete;
  ~job()
  {
    if (subscribed && batch) { batch->unsubscribe(); }
  }
  /// A sliced batch's unsliced copy; sources point into it.
  std::unique_ptr<cudf::table> copy;
  /// gpu_columns: buffers and their device addresses, in walk order.
  std::vector<ring::stream_buffer> buffers;
  std::vector<void const*> sources;
  /// host_columns and compressed: the host blocks the stream is, at logical offset 0.
  blocks_allocation const* blocks{nullptr};
};

struct ring_exchange::landing_state {
  ring::wire_batch header;
  std::vector<ring::stream_buffer> buffers;
  Tier tier;
  std::uint32_t frames;
  std::vector<bool> copied;
  std::uint32_t copied_count{0};
  /// Tickets of this landing not yet seen done; a disk landing may not seal before they are.
  std::shared_ptr<std::atomic<int>> in_flight = std::make_shared<std::atomic<int>>(0);
  std::size_t held_bytes{0};
  // GPU
  std::vector<rmm::device_buffer> gpu_buffers;
  // host
  cucascade::memory::fixed_size_host_memory_resource::fixed_multiple_blocks_allocation host_blocks;
  // disk
  std::string path;
  std::vector<column_metadata> disk_columns;
  std::vector<std::size_t> disk_offsets;  ///< per stream buffer
  std::size_t file_bytes{0};
  bool sealed{false};
};

struct ring_exchange::disk_task {
  std::string path;
  std::uint8_t const* frame;
  /// (frame offset, file offset, bytes)
  std::vector<std::array<std::size_t, 3>> writes;
  /// Header bytes written at file offset 0 first (a compressed landing's first frame).
  std::vector<std::uint8_t> head;
  cucascade::memory::memory_space* space;
};

struct ring_exchange::ticket {
  cudaEvent_t event{nullptr};
  /// 0 running, 1 done, 2 failed: set by the disk thread.
  std::shared_ptr<std::atomic<int>> disk_state;
  std::shared_ptr<std::string> error;
  std::shared_ptr<std::atomic<int>> in_flight;
  std::shared_ptr<disk_task> task;
  int device{0};

  ~ticket()
  {
    if (event != nullptr) {
      rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device}};
      static_cast<void>(cudaEventDestroy(event));
    }
  }
};

// ---- lifetime ------------------------------------------------------------------------------

namespace {
rmm::cuda_stream stream_on(int device)
{
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device}};
  return rmm::cuda_stream{rmm::cuda_stream::flags::non_blocking};
}
}  // namespace

ring_exchange::ring_exchange(int device, options const& opts)
  : _device{device},
    _send{device, frame_ring::frames_in(opts.send_bytes, opts.frame_bytes), opts.frame_bytes},
    _recv{device, frame_ring::frames_in(opts.recv_bytes, opts.frame_bytes), opts.frame_bytes},
    _send_stream{stream_on(device)},
    _recv_stream{stream_on(device)},
    _held{std::make_shared<held>()}
{
  _disk_thread = std::thread([this] { run_disk_writes(); });
  SIRIUS_LOG_INFO(
    "ring exchange on device {}: send ring {} x {} bytes at {:#x}, receive ring {} x {} bytes at "
    "{:#x}",
    device,
    _send.frames(),
    _send.frame_bytes(),
    _send.base(),
    _recv.frames(),
    _recv.frame_bytes(),
    _recv.base());
}

ring_exchange::~ring_exchange()
{
  {
    std::lock_guard lock{_disk_mutex};
    _disk_stop = true;
  }
  _disk_cv.notify_all();
  if (_disk_thread.joinable()) { _disk_thread.join(); }
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{_device}};
  // Copies in flight read or write the rings, which go with this object.
  _send_stream.synchronize_no_throw();
  _recv_stream.synchronize_no_throw();
  std::lock_guard lock{_mutex};
  _tickets.clear();
  _landings.clear();
  _jobs.clear();
}

void ring_exchange::attach(std::shared_ptr<exchange_inbox> exchange, landing_tiers tiers)
{
  std::lock_guard lock{_mutex};
  _exchange = std::move(exchange);
  _tiers    = tiers;
  SIRIUS_LOG_INFO("ring exchange landing: GPU up to {} bytes, host up to {} bytes ({}), disk {}",
                  tiers.gpu_bytes,
                  tiers.host_bytes,
                  tiers.host ? "on" : "none",
                  tiers.disk ? std::string{tiers.disk->get_disk_mount_path()} : "none");
}

// ---- tickets -------------------------------------------------------------------------------

std::shared_ptr<ring_exchange::ticket> ring_exchange::record(rmm::cuda_stream_view stream)
{
  auto t    = std::make_shared<ticket>();
  t->device = _device;
  RMM_CUDA_TRY(cudaEventCreateWithFlags(&t->event, cudaEventDisableTiming));
  RMM_CUDA_TRY(cudaEventRecord(t->event, stream.value()));
  return t;
}

std::uint64_t ring_exchange::new_ticket(std::shared_ptr<ticket> t)
{
  std::lock_guard lock{_mutex};
  auto const id = _next++;
  if (t->in_flight) { t->in_flight->fetch_add(1); }
  _tickets.emplace(id, std::move(t));
  return id;
}

bool ring_exchange::done(std::uint64_t id)
{
  std::shared_ptr<ticket> t;
  {
    std::lock_guard lock{_mutex};
    auto const it = _tickets.find(id);
    if (it == _tickets.end()) {
      throw sirius::invalid_input_exception("ring exchange: unknown ticket {}", id);
    }
    t = it->second;
  }
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{_device}};
  std::string failure;
  if (t->disk_state) {
    auto const state = t->disk_state->load(std::memory_order_acquire);
    if (state == 0) { return false; }
    if (state == 2) { failure = *t->error; }
  } else {
    auto const status = cudaEventQuery(t->event);
    if (status == cudaErrorNotReady) {
      static_cast<void>(cudaGetLastError());
      return false;
    }
    if (status != cudaSuccess) {
      static_cast<void>(cudaGetLastError());
      failure = std::format("a ring exchange copy failed: {}", cudaGetErrorString(status));
    }
  }
  {
    std::lock_guard lock{_mutex};
    _tickets.erase(id);
  }
  if (t->in_flight) { t->in_flight->fetch_sub(1); }
  if (!failure.empty()) { throw std::runtime_error(failure); }
  return true;
}

void ring_exchange::wait(std::uint64_t id)
{
  while (!done(id)) {
    std::this_thread::yield();
  }
}

// ---- sender --------------------------------------------------------------------------------

std::optional<ring_exchange::send_job> ring_exchange::export_batch(
  std::shared_ptr<cucascade::data_batch> batch)
{
  std::shared_ptr<exchange_inbox> exchange;
  {
    std::lock_guard lock{_mutex};
    exchange = _exchange;
  }
  if (!exchange) {
    throw sirius::invalid_input_exception("ring exchange: not attached, or closed");
  }
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{_device}};
  auto const stream = _send_stream.view();

  auto out = std::make_shared<job>();
  send_job described{};
  ring::wire_batch wire;

  // Under a read-only lock no conversion is running; once subscribed, none may start.
  auto const hold = [&] {
    if (out->subscribed) { return; }
    batch->subscribe();
    out->subscribed = true;
    out->batch      = batch;
  };
  auto const describe_host = [&]() -> bool {
    auto const ro    = batch->to_read_only();
    auto const* data = ro.get_data();
    hold();
    if (auto const* host = dynamic_cast<cucascade::host_data_representation const*>(data)) {
      auto const& table = host->get_host_table();
      if (!table || table->columns.empty()) { return false; }
      wire.kind         = batch_kind::host_columns;
      wire.columns      = table->columns;
      wire.rows         = static_cast<std::uint64_t>(std::max(0, table->columns.front().num_rows));
      wire.stream_bytes = table->data_size;
      if (wire.rows == 0) { return true; }
      // Refuse here what the receiver would: a column the ring cannot rebuild goes through the
      // GPU instead, which names it.
      try {
        static_cast<void>(ring::decode_header(ring::encode_header(wire)));
      } catch (sirius::invalid_input_exception const&) {
        return false;
      }
      out->blocks = table->allocation.get();
      return true;
    }
    if (auto const* packed = dynamic_cast<sirius::compressed_host_representation const*>(data)) {
      if (packed->selected_indices().has_value() || packed->pushdown_scan()) { return false; }
      wire.kind               = batch_kind::compressed;
      wire.rows               = static_cast<std::uint64_t>(packed->num_rows());
      wire.stream_bytes       = packed->get_size_in_bytes();
      wire.uncompressed_bytes = packed->get_uncompressed_data_size_in_bytes();
      wire.blob_header.assign(packed->header().begin(), packed->header().end());
      wire.blob_columns = packed->column_names();
      out->blocks       = &packed->payload();
      return true;
    }
    return false;
  };

  bool described_host = false;
  if (batch->to_read_only().get_current_tier() == Tier::HOST) { described_host = describe_host(); }
  if (!described_host) {
    // Anything else travels from the GPU: bring it back first (it may wait for memory).
    exchange->bring_to_gpu(*batch);
    auto const ro = batch->to_read_only();
    hold();
    if (ro.get_current_tier() != Tier::GPU) {
      throw sirius::invalid_input_exception("ring exchange: the batch is not on the GPU");
    }
    auto view = get_cudf_table_view(ro);
    if (view.num_rows() == 0) { return std::nullopt; }
    if (cudaEvent_t writer = ro.get_writer_event()) {
      RMM_CUDA_TRY(cudaStreamWaitEvent(stream.value(), writer, 0));
    }
    if (ring::is_sliced(view)) {
      out->copy =
        std::make_unique<cudf::table>(view, stream, exchange->space().get_default_allocator());
      view = out->copy->view();
    }
    std::size_t stream_bytes = 0;
    wire.kind                = batch_kind::gpu_columns;
    wire.columns             = ring::plan_stream(view, stream, stream_bytes);
    wire.rows                = static_cast<std::uint64_t>(view.num_rows());
    wire.stream_bytes        = stream_bytes;
    out->buffers             = ring::stream_map(wire.columns);
    out->sources             = ring::stream_sources(view, wire.columns);
    if (out->sources.size() != out->buffers.size()) {
      throw sirius::internal_exception("ring exchange: {} sources for {} stream buffers",
                                       out->sources.size(),
                                       out->buffers.size());
    }
  }
  if (wire.rows == 0) { return std::nullopt; }

  out->kind              = wire.kind;
  out->stream_bytes      = wire.stream_bytes;
  out->batch             = std::move(batch);
  described.kind         = wire.kind;
  described.rows         = wire.rows;
  described.stream_bytes = wire.stream_bytes;
  described.header       = ring::encode_header(wire);
  std::lock_guard lock{_mutex};
  described.token = _next++;
  _jobs.emplace(described.token, std::move(out));
  auto const kind = static_cast<std::size_t>(described.kind);
  ++_counters.sent_batches[kind];
  _counters.sent_bytes[kind] += described.stream_bytes;
  return described;
}

std::uint64_t ring_exchange::stage(std::uint64_t token, std::uint32_t k, std::uint32_t frame)
{
  std::shared_ptr<job> j;
  {
    std::lock_guard lock{_mutex};
    auto const it = _jobs.find(token);
    if (it == _jobs.end()) {
      throw sirius::invalid_input_exception("ring exchange: unknown send job {}", token);
    }
    j = it->second;
  }
  auto const frame_bytes = _send.frame_bytes();
  auto const begin       = static_cast<std::size_t>(k) * frame_bytes;
  if (begin >= j->stream_bytes) {
    throw sirius::invalid_input_exception(
      "ring exchange: frame {} is past a {}-byte stream", k, j->stream_bytes);
  }
  auto const end = std::min<std::size_t>(begin + frame_bytes, j->stream_bytes);
  auto* dst      = reinterpret_cast<std::uint8_t*>(_send.frame_address(frame));
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{_device}};
  copy_batch copies;
  if (j->kind == batch_kind::gpu_columns) {
    for (auto const& piece : ring::pieces_in(j->buffers, begin, end)) {
      copies.add(dst + piece.range_offset,
                 static_cast<std::uint8_t const*>(j->sources[piece.buffer]) + piece.buffer_offset,
                 piece.size);
    }
  } else {
    add_block_copies(copies, *j->blocks, begin, dst, end - begin, /*to_device=*/true);
  }
  copies.flush(_send_stream.view());
  return new_ticket(record(_send_stream.view()));
}

void ring_exchange::release_job(std::uint64_t token)
{
  std::shared_ptr<job> j;
  {
    std::lock_guard lock{_mutex};
    auto const it = _jobs.find(token);
    if (it == _jobs.end()) { return; }
    j = std::move(it->second);
    _jobs.erase(it);
  }
  // A staged copy may still read the batch: free it behind the send stream.
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{_device}};
  _send_stream.synchronize();
}

// ---- receiver ------------------------------------------------------------------------------

ring_exchange::landing ring_exchange::begin(std::span<std::uint8_t const> header_bytes)
{
  auto st          = std::make_shared<landing_state>();
  st->header       = ring::decode_header(header_bytes);
  auto const& wire = st->header;
  std::shared_ptr<exchange_inbox> exchange;
  landing_tiers tiers;
  {
    std::lock_guard lock{_mutex};
    exchange = _exchange;
    tiers    = _tiers;
  }
  if (!exchange) {
    throw sirius::invalid_input_exception("ring exchange: not attached, or closed");
  }
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{_device}};
  auto const frame_bytes = _recv.frame_bytes();
  st->frames = static_cast<std::uint32_t>((wire.stream_bytes + frame_bytes - 1) / frame_bytes);
  st->copied.assign(st->frames, false);
  bool const columns = wire.kind != batch_kind::compressed;
  if (columns) { st->buffers = ring::stream_map(wire.columns); }

  // 1. GPU, within the landing budget, never waiting (I1): a non-blocking reservation, and an
  //    allocation that fails over to host instead of retrying.
  std::size_t gpu_need = 0;
  for (auto const& b : st->buffers) {
    gpu_need += rmm::align_up(b.size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  }
  auto& gpu = exchange->space();
  if (columns && _held->gpu.load() + gpu_need <= tiers.gpu_bytes) {
    if (auto reservation = gpu.make_reservation_or_null(gpu_need)) {
      auto const stream = _recv_stream.view();
      auto* tracker     = gpu.get_memory_resource_of<Tier::GPU>();
      if (tracker->attach_reservation_to_tracker(
            stream,
            std::move(reservation),
            std::make_unique<cucascade::memory::fail_reservation_limit_policy>())) {
        absl::Cleanup detach = [&] { tracker->reset_stream_reservation(stream); };
        try {
          st->gpu_buffers.reserve(st->buffers.size());
          for (auto const& b : st->buffers) {
            st->gpu_buffers.emplace_back(b.size, stream, gpu.get_default_allocator());
          }
          st->tier       = Tier::GPU;
          st->held_bytes = gpu_need;
        } catch (rmm::out_of_memory const&) {
          st->gpu_buffers.clear();
        }
      }
    }
  }
  // 2. Host, within its share.
  if (st->held_bytes == 0 && tiers.host != nullptr &&
      _held->host.load() + wire.stream_bytes <= tiers.host_bytes) {
    if (auto reservation = tiers.host->make_reservation_or_null(wire.stream_bytes)) {
      try {
        auto* mr =
          tiers.host->get_memory_resource_as<cucascade::memory::fixed_size_host_memory_resource>();
        if (mr != nullptr) {
          st->host_blocks = mr->allocate_multiple_blocks(wire.stream_bytes, reservation.get());
          st->tier        = Tier::HOST;
          st->held_bytes  = wire.stream_bytes;
        }
      } catch (rmm::out_of_memory const&) {
        st->host_blocks.reset();
      }
    }
  }
  // 3. Disk.
  if (st->held_bytes == 0 && tiers.disk != nullptr) {
    try {
      st->path = cucascade::memory::generate_disk_file_path(tiers.disk->get_disk_mount_path());
      if (columns) {
        std::size_t cursor = 0;
        for (auto const& c : wire.columns) {
          st->disk_columns.push_back(disk_layout(c, cursor));
        }
        st->file_bytes = cursor;
        for (auto const& b : ring::stream_map(st->disk_columns)) {
          st->disk_offsets.push_back(b.offset);
        }
      } else {
        st->file_bytes = wire.blob_header.size() + wire.stream_bytes;
      }
      st->tier       = Tier::DISK;
      st->held_bytes = 1;  // a disk landing holds no budget
    } catch (std::exception const& e) {
      SIRIUS_LOG_WARN("ring exchange: cannot open a disk landing: {}", e.what());
      st->path.clear();
    }
  }
  if (st->held_bytes == 0) {
    std::lock_guard lock{_mutex};
    ++_counters.refused;
    throw ring_no_room(std::format(
      "ring exchange: no tier has room for a {}-byte batch of {} rows: GPU needs {} bytes, {} "
      "available, {} of {} held by landed batches; host {}; disk {}",
      wire.stream_bytes,
      wire.rows,
      gpu_need,
      gpu.get_available_memory(),
      _held->gpu.load(),
      tiers.gpu_bytes,
      tiers.host ? std::format("{} available, {} of {} held",
                               tiers.host->get_available_memory(),
                               _held->host.load(),
                               tiers.host_bytes)
                 : std::string{"none"},
      tiers.disk ? std::format("{} available", tiers.disk->get_available_memory())
                 : std::string{"none"}));
  }
  if (st->tier == Tier::GPU) { _held->gpu.fetch_add(st->held_bytes); }
  if (st->tier == Tier::HOST) { _held->host.fetch_add(st->held_bytes); }
  if (st->tier == Tier::DISK) { st->held_bytes = 0; }

  landing out{0, st->tier, wire.rows, wire.stream_bytes, st->frames};
  std::lock_guard lock{_mutex};
  out.id = _next++;
  _landings.emplace(out.id, std::move(st));
  return out;
}

std::uint64_t ring_exchange::copy_out(std::uint64_t id,
                                      std::uint32_t k,
                                      std::uint32_t frame,
                                      std::uint64_t len)
{
  std::shared_ptr<landing_state> st;
  {
    std::lock_guard lock{_mutex};
    auto const it = _landings.find(id);
    if (it == _landings.end()) {
      throw sirius::invalid_input_exception("ring exchange: unknown landing {}", id);
    }
    st = it->second;
    if (k >= st->frames || st->copied[k]) {
      throw sirius::invalid_input_exception(
        "ring exchange: frame {} of a {}-frame landing is out of range or already copied",
        k,
        st->frames);
    }
  }
  auto const frame_bytes = _recv.frame_bytes();
  auto const begin       = static_cast<std::size_t>(k) * frame_bytes;
  auto const expect      = std::min<std::size_t>(frame_bytes, st->header.stream_bytes - begin);
  if (len != expect) {
    throw sirius::invalid_input_exception(
      "ring exchange: frame {} carries {} bytes, the stream needs {}", k, len, expect);
  }
  auto const end = begin + expect;
  auto* src      = reinterpret_cast<std::uint8_t*>(_recv.frame_address(frame));
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{_device}};

  std::shared_ptr<ticket> t;
  if (st->tier == Tier::DISK) {
    auto task   = std::make_shared<disk_task>();
    task->path  = st->path;
    task->frame = src;
    task->space = _tiers.disk;
    if (st->header.kind == batch_kind::compressed) {
      if (k == 0) { task->head = st->header.blob_header; }
      task->writes.push_back({0, st->header.blob_header.size() + begin, expect});
    } else {
      for (auto const& piece : ring::pieces_in(st->buffers, begin, end)) {
        task->writes.push_back(
          {piece.range_offset, st->disk_offsets[piece.buffer] + piece.buffer_offset, piece.size});
      }
    }
    t             = std::make_shared<ticket>();
    t->device     = _device;
    t->disk_state = std::make_shared<std::atomic<int>>(0);
    t->error      = std::make_shared<std::string>();
    t->task       = task;
  } else {
    copy_batch copies;
    if (st->tier == Tier::GPU) {
      for (auto const& piece : ring::pieces_in(st->buffers, begin, end)) {
        copies.add(
          static_cast<std::uint8_t*>(st->gpu_buffers[piece.buffer].data()) + piece.buffer_offset,
          src + piece.range_offset,
          piece.size);
      }
    } else {
      add_block_copies(copies, *st->host_blocks, begin, src, expect, /*to_device=*/false);
    }
    copies.flush(_recv_stream.view());
    t = record(_recv_stream.view());
  }
  t->in_flight = st->in_flight;
  {
    std::lock_guard lock{_mutex};
    st->copied[k] = true;
    ++st->copied_count;
  }
  auto const ticket_id = new_ticket(t);
  if (t->task) {
    {
      std::lock_guard lock{_disk_mutex};
      _disk_queue.push_back(t->task);
    }
    _disk_cv.notify_one();
  }
  return ticket_id;
}

void ring_exchange::run_disk_writes()
{
  // Each task writes one received frame; its ticket is found by task.
  for (;;) {
    std::shared_ptr<disk_task> task;
    {
      std::unique_lock lock{_disk_mutex};
      _disk_cv.wait(lock, [&] { return _disk_stop || !_disk_queue.empty(); });
      if (_disk_queue.empty()) { return; }
      task = std::move(_disk_queue.front());
      _disk_queue.pop_front();
    }
    std::shared_ptr<ticket> t;
    {
      std::lock_guard lock{_mutex};
      for (auto const& [id, candidate] : _tickets) {
        if (candidate->task == task) {
          t = candidate;
          break;
        }
      }
    }
    if (!t) { continue; }  // aborted
    try {
      rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{_device}};
      auto& backend = task->space->get_io_backend();
      if (!task->head.empty()) {
        backend.write(task->path, task->head.data(), task->head.size(), 0);
      }
      for (auto const& [frame_offset, file_offset, size] : task->writes) {
        backend.write(
          task->path, task->frame + frame_offset, size, file_offset, _recv_stream.view());
      }
      _recv_stream.synchronize();
      t->disk_state->store(1, std::memory_order_release);
    } catch (std::exception const& e) {
      *t->error = std::format(
        "ring exchange: writing a landed frame to '{}' failed: {}", task->path, e.what());
      t->disk_state->store(2, std::memory_order_release);
    }
  }
}

std::uint64_t ring_exchange::seal(std::uint64_t id)
{
  std::shared_ptr<landing_state> st;
  std::shared_ptr<exchange_inbox> exchange;
  {
    std::lock_guard lock{_mutex};
    auto const it = _landings.find(id);
    if (it == _landings.end()) {
      throw sirius::invalid_input_exception("ring exchange: unknown landing {}", id);
    }
    st = it->second;
    if (st->copied_count != st->frames) {
      throw sirius::invalid_input_exception(
        "ring exchange: landing {} has {} of {} frames", id, st->copied_count, st->frames);
    }
    if (st->in_flight->load() != 0) {
      throw sirius::invalid_input_exception("ring exchange: landing {} still has copies in flight",
                                            id);
    }
    _landings.erase(it);
    exchange = _exchange;
  }
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{_device}};
  auto const& wire = st->header;
  auto& gpu        = exchange->space();
  std::shared_ptr<cucascade::data_batch> batch;
  auto const batch_id = sirius::get_next_batch_id();
  auto probe          = telemetry::quent_data_batch_probe::create({}, batch_id);
  switch (st->tier) {
    case Tier::GPU: {
      // Freed later on one of the space's streams; the copies are ordered before the batch's
      // writer event, recorded on the receive stream.
      for (auto& buffer : st->gpu_buffers) {
        buffer.set_stream(gpu.acquire_stream());
      }
      auto next = st->gpu_buffers.begin();
      std::vector<std::unique_ptr<cudf::column>> columns;
      for (auto const& c : wire.columns) {
        columns.push_back(rebuild(c, next));
      }
      auto repr = std::make_unique<cucascade::gpu_table_representation>(
        std::make_unique<cudf::table>(std::move(columns)), gpu, _recv_stream.view());
      if (repr->get_writer_event() == nullptr) { repr->record_writer_event(_recv_stream.view()); }
      batch = cucascade::data_batch::make(batch_id, std::move(repr), std::move(probe));
      break;
    }
    case Tier::HOST: {
      auto& host = *_tiers.host;
      if (wire.kind == batch_kind::compressed) {
        auto blob           = std::make_shared<sirius::pinned_compressed_blob>();
        blob->header        = wire.blob_header;
        blob->payload       = std::move(st->host_blocks);
        blob->payload_bytes = wire.stream_bytes;
        batch =
          cucascade::data_batch::make(batch_id,
                                      std::make_unique<sirius::compressed_host_representation>(
                                        host,
                                        std::move(blob),
                                        wire.blob_columns,
                                        wire.stream_bytes,
                                        wire.uncompressed_bytes,
                                        static_cast<std::int64_t>(wire.rows)),
                                      std::move(probe));
      } else {
        auto table = cucascade::memory::host_table_allocation::create(
          std::move(st->host_blocks), wire.columns, wire.stream_bytes);
        batch = cucascade::data_batch::make(
          batch_id,
          std::make_unique<cucascade::host_data_representation>(std::move(table), &host),
          std::move(probe));
      }
      break;
    }
    default: {
      auto& disk = *_tiers.disk;
      if (wire.kind == batch_kind::compressed) {
        // Landing a compressed batch on disk needs compressed_disk_representation, which spill
        // compression (#2204) adds. Until then only a compressed pin ships compressed, and its
        // batches land on the GPU or in host memory.
        throw sirius::invalid_input_exception(
          "ring exchange: a compressed batch cannot land on disk without spill compression");
      } else {
        auto table = std::make_unique<cucascade::memory::disk_table_allocation>(
          st->path, st->disk_columns, st->file_bytes);
        batch = cucascade::data_batch::make(
          batch_id,
          std::make_unique<cucascade::disk_data_representation>(std::move(table), disk),
          std::move(probe));
      }
      break;
    }
  }
  auto const tier  = st->tier;
  auto const bytes = st->held_bytes;
  {
    std::lock_guard lock{_mutex};
    auto const at = tier_index(tier);
    ++_counters.landed_batches[at];
    _counters.landed_bytes[at] += wire.stream_bytes;
  }
  st->sealed = true;
  std::function<void()> on_gone;
  if (bytes > 0) {
    on_gone = [held = _held, tier, bytes] {
      (tier == Tier::GPU ? held->gpu : held->host).fetch_sub(bytes);
    };
  }
  return exchange->adopt(std::move(batch), std::move(on_gone));
}

void ring_exchange::abort(std::uint64_t id)
{
  std::shared_ptr<landing_state> st;
  {
    std::lock_guard lock{_mutex};
    auto const it = _landings.find(id);
    if (it == _landings.end()) { return; }
    st = std::move(it->second);
    _landings.erase(it);
  }
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{_device}};
  // Copies in flight write into what the landing holds: wait them out and drop their tickets.
  _recv_stream.synchronize();
  for (;;) {
    bool pending = false;
    {
      std::lock_guard lock{_mutex};
      for (auto it = _tickets.begin(); it != _tickets.end();) {
        auto const& t = it->second;
        if (t->in_flight != st->in_flight) {
          ++it;
          continue;
        }
        if (t->disk_state && t->disk_state->load(std::memory_order_acquire) == 0) {
          pending = true;
          ++it;
          continue;
        }
        t->in_flight->fetch_sub(1);
        it = _tickets.erase(it);
      }
    }
    if (!pending) { break; }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  if (st->tier == Tier::GPU) { _held->gpu.fetch_sub(st->held_bytes); }
  if (st->tier == Tier::HOST) { _held->host.fetch_sub(st->held_bytes); }
  if (st->tier == Tier::DISK && !st->path.empty()) {
    std::error_code ignored;
    std::filesystem::remove(st->path, ignored);
  }
}

void ring_exchange::close()
{
  std::vector<std::uint64_t> landings;
  {
    std::lock_guard lock{_mutex};
    if (_closed) { return; }
    for (auto const& [id, st] : _landings) {
      landings.push_back(id);
    }
  }
  for (auto const id : landings) {
    abort(id);
  }
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{_device}};
  _send_stream.synchronize_no_throw();
  _recv_stream.synchronize_no_throw();
  std::lock_guard lock{_mutex};
  _jobs.clear();
  _tickets.clear();
  _exchange.reset();
  _closed = true;
}

// ---- stats ---------------------------------------------------------------------------------

ring_exchange::counters ring_exchange::stats() const
{
  std::lock_guard lock{_mutex};
  auto out      = _counters;
  out.gpu_held  = _held->gpu.load();
  out.host_held = _held->host.load();
  return out;
}

std::string ring_exchange::stats_line() const
{
  auto const c = stats();
  return std::format(
    "sent gpu_columns {} ({}), host_columns {} ({}), compressed {} ({}); landed gpu {} ({}), "
    "host {} ({}), disk {} ({}); refused {}; held gpu {}, host {}; send ring peak {}/{}, receive "
    "ring peak {}/{} frames",
    c.sent_batches[1],
    bytes_text(c.sent_bytes[1]),
    c.sent_batches[2],
    bytes_text(c.sent_bytes[2]),
    c.sent_batches[3],
    bytes_text(c.sent_bytes[3]),
    c.landed_batches[0],
    bytes_text(c.landed_bytes[0]),
    c.landed_batches[1],
    bytes_text(c.landed_bytes[1]),
    c.landed_batches[2],
    bytes_text(c.landed_bytes[2]),
    c.refused,
    bytes_text(c.gpu_held),
    bytes_text(c.host_held),
    _send.peak_leased(),
    _send.frames(),
    _recv.peak_leased(),
    _recv.frames());
}

}  // namespace sirius::exec
