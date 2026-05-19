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

#include "io/uring/uring_reactor.hpp"

#include "driver_types.h"
#include "io/types.hpp"

#include <fcntl.h>
#include <spdlog/spdlog.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstring>
#include <deque>
#include <filesystem>
#include <ranges>
#include <stdexcept>

namespace sirius::io {

// ---------------------------------------------------------------------------
// uring_reactor
// ---------------------------------------------------------------------------

std::unique_ptr<uring_io_object> uring_reactor::create_io_object(std::string path)
{
  if (!supports(path))
    throw std::runtime_error("uring_reactor::create_io_object: unsupported path: " + path);

  file_descriptor fd{::open(path.c_str(), O_RDONLY)};
  if (!fd)
    throw std::runtime_error("uring_reactor::create_io_object: open failed: " + path + ": " +
                             strerror(errno));

  file_descriptor fd_direct{::open(path.c_str(), O_RDONLY | O_DIRECT)};
  if (!fd_direct)
    throw std::runtime_error("uring_reactor::create_io_object: O_DIRECT open failed: " + path +
                             ": " + strerror(errno));

  auto file_size = size(fd.get());
  return std::make_unique<uring_io_object>(
    std::move(path), std::move(fd), std::move(fd_direct), file_size);
}

size_t uring_reactor::size(int fd)
{
  struct stat st{};
  if (::fstat(fd, &st) != 0)
    throw std::runtime_error("uring_reactor::size: fstat failed: " + std::string(strerror(errno)));
  return static_cast<size_t>(st.st_size);
}

uring_reactor::uring_reactor(cucascade::memory::fixed_size_host_memory_resource& mr,
                             unsigned ring_entries)
  : _bounce_slot_size(mr.get_block_size()), _ring_entries(ring_entries)
{
  _bounce_storage = mr.allocate_multiple_blocks(NUM_CHUNKS * _bounce_slot_size);
  auto blocks     = _bounce_storage->get_blocks();
  if (blocks.size() < NUM_CHUNKS) {
    throw std::runtime_error(
      "uring_reactor: fixed_size_host_memory_resource returned fewer blocks (" +
      std::to_string(blocks.size()) + ") than required (" + std::to_string(NUM_CHUNKS) + ")");
  }
  for (int i = 0; i < static_cast<int>(NUM_CHUNKS); ++i) {
    _bounce[i].buf = blocks[i];
    _cb_args[i]    = {this, i};
  }

  _worker = std::thread([this] { worker_loop(); });
}

uring_reactor::~uring_reactor() { shutdown(); }

void uring_reactor::interrupt()
{
  _wake_seq.fetch_add(1, std::memory_order_release);
  _wake_seq.notify_one();
}

void uring_reactor::shutdown()
{
  if (_worker.joinable()) {
    _stop.store(true, std::memory_order_release);
    interrupt();
    _worker.join();
  }
}

void uring_reactor::cuda_copy_cb(cudaStream_t /*stream*/, cudaError_t status, void* p) noexcept
{
  auto* arg = static_cast<cb_arg*>(p);
  // Write status BEFORE the release-store on cuda_done so a reader that
  // observes cuda_done == true via acquire also observes this status.
  arg->self->_bounce[arg->slot].cuda_status = status;
  arg->self->_bounce[arg->slot].cuda_done.store(true, std::memory_order_release);
  // Bump the unified wake atomic — the same one queue enqueues and
  // interrupt() bump.  The reactor's single park point on _wake_seq wakes
  // on any event, so a stalled CUDA callback no longer blocks the reactor
  // from observing new host reads or shutdown.
  arg->self->_wake_seq.fetch_add(1, std::memory_order_release);
  arg->self->_wake_seq.notify_one();
}

cudf::io::text::byte_range_info uring_reactor::align_to_physical(
  cudf::io::text::byte_range_info logical, size_t file_size)
{
  auto offset    = static_cast<size_t>(logical.offset());
  auto size      = static_cast<size_t>(logical.size());
  size_t a_start = offset & ~(IO_BLOCK_SIZE - 1);
  size_t a_end   = std::min((offset + size + IO_BLOCK_SIZE - 1) & ~(IO_BLOCK_SIZE - 1),
                          (file_size + IO_BLOCK_SIZE - 1) & ~(IO_BLOCK_SIZE - 1));
  return {static_cast<int64_t>(a_start), static_cast<int64_t>(a_end - a_start)};
}

bool uring_reactor::supports(std::string_view path)
{
  std::error_code ec;
  std::filesystem::path p{path};
  return std::filesystem::is_regular_file(p, ec) && !ec;
}

void uring_reactor::enqueue_bulk(std::span<device_read_req_type> batch)
{
  if (batch.empty()) return;

  // Capture ctxs before bulk-enqueue: on failure (OOM in moodycamel) items
  // may already be moved-from, so batch[i].ctx could be null.  The captured
  // shared_ptrs let us drain ctx->pending via chunk_failed so the
  // completion handler still fires instead of stranding readers in
  // wait_while_loading().
  std::vector<std::shared_ptr<request_context>> ctxs;
  ctxs.reserve(batch.size());
  for (auto& r : batch)
    ctxs.push_back(r.ctx);

  if (!_queue.enqueue_bulk(std::make_move_iterator(batch.begin()), batch.size())) {
    auto e = std::make_exception_ptr(
      std::runtime_error("uring_reactor::enqueue_bulk: queue enqueue failed"));
    for (auto& ctx : ctxs)
      ctx->chunk_failed(e);
    return;
  }

  _wake_seq.fetch_add(1, std::memory_order_release);
  _wake_seq.notify_one();
}

size_t uring_reactor::host_read(int fd, size_t offset, size_t size, uint8_t* dst)
{
  if (size == 0) return 0;
  // Loop until either the full requested size is read, EOF (n == 0), or a
  // real error. pread on a regular file should only return short on EOF, but
  // we retry defensively against EINTR and any unexpected short-read paths
  // so callers don't have to.
  size_t total = 0;
  while (total < size) {
    ssize_t n = ::pread(fd, dst + total, size - total, static_cast<off_t>(offset + total));
    if (n < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("uring_reactor::host_read pread: " + std::string(strerror(errno)));
    }
    if (n == 0) break;  // EOF
    total += static_cast<size_t>(n);
  }
  return total;
}

void uring_reactor::host_read_async(host_read_req_type req)
{
  // Hold the ctx separately so we can drain pending via chunk_failed if the
  // enqueue itself fails (otherwise req.ctx may be moved-from and lost).
  auto ctx = req.ctx;
  if (!_host_queue.enqueue(std::move(req))) {
    ctx->chunk_failed(std::make_exception_ptr(
      std::runtime_error("uring_reactor::host_read_async: queue enqueue failed")));
    return;
  }
  _wake_seq.fetch_add(1, std::memory_order_release);
  _wake_seq.notify_one();
}

void uring_reactor::host_enqueue_bulk(std::span<host_read_req_type> batch)
{
  if (batch.empty()) return;

  // See enqueue_bulk() above for the rationale: snapshot ctxs first so a
  // partial-move failure on the moodycamel queue still drains ctx->pending
  // and fires the completion handler instead of deadlocking readers.
  std::vector<std::shared_ptr<request_context>> ctxs;
  ctxs.reserve(batch.size());
  for (auto& r : batch)
    ctxs.push_back(r.ctx);

  if (!_host_queue.enqueue_bulk(std::make_move_iterator(batch.begin()), batch.size())) {
    auto e = std::make_exception_ptr(
      std::runtime_error("uring_reactor::host_enqueue_bulk: queue enqueue failed"));
    for (auto& ctx : ctxs)
      ctx->chunk_failed(e);
    return;
  }

  _wake_seq.fetch_add(1, std::memory_order_release);
  _wake_seq.notify_one();
}

void uring_reactor::worker_loop()
{
  unique_ring ring = [this]() -> unique_ring {
#if defined(IORING_SETUP_SINGLE_ISSUER) && defined(IORING_SETUP_DEFER_TASKRUN)
    auto r                   = std::make_unique<io_uring>();
    struct io_uring_params p = {0};
    // Disable kernel locks (assuming 1 thread per ring)
    p.flags |= IORING_SETUP_SINGLE_ISSUER;

    // Keep completions on the current CPU, avoid cache thrashing
    p.flags |= IORING_SETUP_COOP_TASKRUN | IORING_SETUP_DEFER_TASKRUN;
    int rc = io_uring_queue_init_params(_ring_entries, r.get(), &p);
    if (rc == 0) {
      spdlog::debug("uring_device_reactor: ring using SINGLE_ISSUER|DEFER_TASKRUN");
      return unique_ring{r.release()};
    }
    spdlog::debug(
      "uring_device_reactor: SINGLE_ISSUER|DEFER_TASKRUN unsupported "
      "({}), falling back to plain flags",
      strerror(-rc));
#endif
    auto r2 = std::make_unique<io_uring>();
    int rc2 = io_uring_queue_init(_ring_entries, r2.get(), 0);
    if (rc2 < 0)
      throw std::runtime_error("uring_reactor: ring init: " + std::string(strerror(-rc2)));
    spdlog::debug("uring_reactor: ring using plain flags");
    return unique_ring{r2.release()};
  }();

  std::array<iovec, NUM_CHUNKS> iovecs{};
  for (size_t i = 0; i < NUM_CHUNKS; ++i)
    iovecs[i] = {_bounce[i].buf, _bounce_slot_size};
  if (int rc = io_uring_register_buffers(ring.get(), iovecs.data(), NUM_CHUNKS); rc < 0)
    throw std::runtime_error("uring_device_reactor: io_uring_register_buffers: " +
                             std::string(strerror(-rc)));

  static constexpr uint64_t HOST_TAG = 1ULL << 63;

  enum class slot_state : uint8_t { FREE, READING, COPYING };
  struct slot_info {
    slot_state state{slot_state::FREE};
    device_read_req_type req{};
    // Cumulative bytes read for the current request across short-read retries.
    // Reset to 0 on slot acquisition; on each completion we add `cqe->res` and
    // either re-submit the remainder or proceed to the H2D path.
    size_t bytes_read{0};
  };
  // Heap-allocated wrapper for in-flight host reads. Mirrors the device
  // slot's bytes_read tracking so we can retry short reads at the same
  // user_data identity across multiple SQE submissions.
  struct host_in_flight {
    host_read_req_type req;
    size_t bytes_read{0};
  };
  std::array<slot_info, NUM_CHUNKS> slots{};
  std::deque<device_read_req_type> pending;
  std::deque<host_read_req_type> pending_host;
  int inflight = 0;

  auto find_free = [&]() -> int {
    auto it = std::ranges::find_if(slots, [](auto& s) { return s.state == slot_state::FREE; });
    return it != slots.end() ? static_cast<int>(it - slots.begin()) : -1;
  };
  auto has_active = [&]() {
    return inflight > 0 || !pending.empty() || !pending_host.empty() ||
           std::ranges::any_of(slots, [](auto& s) { return s.state == slot_state::COPYING; });
  };
  auto poll_cuda = [&]() {
    for (auto i : std::views::iota(size_t{0}, NUM_CHUNKS)) {
      if (slots[i].state == slot_state::COPYING &&
          _bounce[i].cuda_done.load(std::memory_order_acquire)) {
        // cuda_copy_cb stored the stream's status at callback time, before
        // the release-store on cuda_done.  Read it directly — this is the
        // state of the stream when our copy finished, not the current state
        // (which may also include later work the consumer queued).
        cudaError_t err = _bounce[i].cuda_status;
        if (err != cudaSuccess) {
          slots[i].req.ctx->chunk_failed(std::make_exception_ptr(std::runtime_error(
            std::string("uring_reactor: H2D copy failed: ") + cudaGetErrorString(err))));
        } else {
          slots[i].req.ctx->chunk_done();
        }
        slots[i] = slot_info{};
      }
    }
  };
  auto drain_queue = [&]() {
    device_read_req_type r;
    while (_queue.try_dequeue(r))
      pending.push_back(std::move(r));
    host_read_req_type hr;
    while (_host_queue.try_dequeue(hr))
      pending_host.push_back(std::move(hr));
  };
  auto submit_pending = [&]() {
    int added = 0;
    // Device reads consume a bounce slot.
    while (!pending.empty()) {
      int si = find_free();
      if (si < 0) break;
      io_uring_sqe* sqe = io_uring_get_sqe(ring.get());
      if (!sqe) break;
      auto& req = pending.front();
      // Bounce slots are pre-registered via io_uring_register_buffers, so
      // submit through the fixed-buffer fast path — skips per-IO buffer
      // pin/unmap overhead in the kernel.
      io_uring_prep_read_fixed(sqe,
                               req.handle,
                               _bounce[si].buf,
                               (unsigned)req.io_size,
                               (unsigned long long)req.file_off,
                               si);
      io_uring_sqe_set_data64(sqe, (uint64_t)si);
      slots[si].state      = slot_state::READING;
      slots[si].bytes_read = 0;  // fresh request → reset retry accumulator
      slots[si].req        = std::move(req);
      pending.pop_front();
      ++inflight;
      ++added;
    }
    // Host reads: heap-allocate a host_in_flight wrapper so we can carry
    // cumulative bytes across short-read retries via the same user_data tag.
    while (!pending_host.empty()) {
      io_uring_sqe* sqe = io_uring_get_sqe(ring.get());
      if (!sqe) break;
      auto* hf = new host_in_flight{std::move(pending_host.front()), 0};
      pending_host.pop_front();
      io_uring_prep_read(
        sqe, hf->req.handle, hf->req.dst, (unsigned)hf->req.size, (__u64)hf->req.offset);
      io_uring_sqe_set_data64(sqe, reinterpret_cast<uint64_t>(hf) | HOST_TAG);
      ++inflight;
      ++added;
    }
    if (added > 0) io_uring_submit(ring.get());
  };
  auto reap_cqes = [&]() {
    io_uring_cqe* cqes[NUM_CHUNKS];
    unsigned n         = io_uring_peek_batch_cqe(ring.get(), cqes, NUM_CHUNKS);
    bool need_resubmit = false;  // any retry SQE prepared in this pass?
    for (auto* cqe : std::span{cqes, n}) {
      uint64_t data = io_uring_cqe_get_data64(cqe);
      int res       = cqe->res;
      io_uring_cqe_seen(ring.get(), cqe);
      --inflight;

      if (data & HOST_TAG) {
        // Host read completion.
        auto* hf = reinterpret_cast<host_in_flight*>(data & ~HOST_TAG);
        if (res < 0) {
          hf->req.ctx->chunk_failed(std::make_exception_ptr(
            std::runtime_error("reactor host read: " + std::string(strerror(-res)))));
          delete hf;
          continue;
        }
        size_t rd = static_cast<size_t>(res);
        hf->bytes_read += rd;
        bool const fully_read = hf->bytes_read >= hf->req.size;
        bool const eof        = (rd == 0);
        if (!fully_read && !eof) {
          // Short read mid-file: queue a follow-up SQE for the unread tail.
          // We reuse the same host_in_flight identity so subsequent CQEs
          // continue to accumulate bytes_read against this request.
          io_uring_sqe* sqe = io_uring_get_sqe(ring.get());
          if (sqe) {
            io_uring_prep_read(sqe,
                               hf->req.handle,
                               hf->req.dst + hf->bytes_read,
                               (unsigned)(hf->req.size - hf->bytes_read),
                               (__u64)(hf->req.offset + hf->bytes_read));
            io_uring_sqe_set_data64(sqe, reinterpret_cast<uint64_t>(hf) | HOST_TAG);
            ++inflight;
            need_resubmit = true;
            spdlog::warn(
              "uring_reactor: host short read, retrying tail. fd={} offset={} size={} "
              "bytes_read={} this_rd={}",
              hf->req.handle,
              hf->req.offset,
              hf->req.size,
              hf->bytes_read,
              rd);
            continue;
          }
          // SQE exhaustion is unexpected (we have _ring_entries SQEs and
          // never more than NUM_CHUNKS in flight); fall through and complete
          // with whatever we have rather than spin.
          spdlog::warn("uring_reactor: SQE exhausted on host short-read retry");
        }
        hf->req.ctx->chunk_done();
        delete hf;
      } else {
        // Device read completion.
        int si      = static_cast<int>(data);
        auto& sinfo = slots[si];
        auto& req   = sinfo.req;
        if (res < 0) {
          req.ctx->chunk_failed(std::make_exception_ptr(
            std::runtime_error("reactor read: " + std::string(strerror(-res)))));
          sinfo = slot_info{};
          continue;
        }
        size_t rd = static_cast<size_t>(res);
        sinfo.bytes_read += rd;
        bool const fully_read = sinfo.bytes_read >= req.io_size;
        bool const eof        = (rd == 0);
        if (!fully_read && !eof) {
          // Short read mid-file: queue a follow-up SQE for the unread tail
          // into the same bounce slot at the next-byte offset. The slot
          // stays in READING; once we get the full io_size (or EOF), we
          // proceed to the H2D path below using sinfo.bytes_read.
          io_uring_sqe* sqe = io_uring_get_sqe(ring.get());
          if (sqe) {
            io_uring_prep_read_fixed(sqe,
                                     req.handle,
                                     (uint8_t*)_bounce[si].buf + sinfo.bytes_read,
                                     (unsigned)(req.io_size - sinfo.bytes_read),
                                     (__u64)(req.file_off + sinfo.bytes_read),
                                     si);
            io_uring_sqe_set_data64(sqe, (uint64_t)si);
            ++inflight;
            need_resubmit = true;
            spdlog::warn(
              "uring_reactor: device short read, retrying tail. slot={} file_off={} "
              "io_size={} bytes_read={} this_rd={}",
              si,
              req.file_off,
              req.io_size,
              sinfo.bytes_read,
              rd);
            continue;
          }
          spdlog::warn("uring_reactor: SQE exhausted on device short-read retry");
        }
        // Fully read or true EOF: compute user-visible bytes from the
        // accumulated bytes_read, not just this CQE's `rd`.
        size_t actual = sinfo.bytes_read > req.data_off
                          ? std::min(req.data_size, sinfo.bytes_read - req.data_off)
                          : 0;
        if (actual > 0 && !req.ctx->failed.load(std::memory_order_relaxed)) {
          _bounce[si].cuda_done.store(false, std::memory_order_relaxed);
          _bounce[si].cuda_status = cudaSuccess;
          if (req.device_id >= 0) cudaSetDevice(req.device_id);
          cudaError_t cpy_err = cudaMemcpyAsync(req.dst,
                                                (uint8_t*)_bounce[si].buf + req.data_off,
                                                actual,
                                                cudaMemcpyHostToDevice,
                                                req.stream);
          if (cpy_err != cudaSuccess) {
            // Synchronous failure (e.g., invalid pointer / context).  Drain
            // the sticky error so unrelated runtime calls don't observe it,
            // fail the slot, and skip the callback registration — there's
            // no stream work to attach a callback to.
            cudaGetLastError();
            req.ctx->chunk_failed(std::make_exception_ptr(
              std::runtime_error(std::string("uring_reactor: cudaMemcpyAsync failed: ") +
                                 cudaGetErrorString(cpy_err))));
            sinfo = slot_info{};
            continue;
          }
          // cudaStreamAddCallback (deprecated but used deliberately): unlike
          // cudaLaunchHostFunc, the callback fires even if the stream is in
          // an error state — guaranteeing we never strand the slot in
          // COPYING waiting for a callback that won't come.
          cudaStreamAddCallback(req.stream, cuda_copy_cb, &_cb_args[si], 0);
          sinfo.state = slot_state::COPYING;
        } else {
          // After retry-to-completion this only fires when the file truly
          // ends inside the alignment prefix (actual == 0) or when another
          // chunk of the same request already failed. The handler still
          // resolves with total_bytes, so anything in `req.dst` past what
          // we could read is left untouched — log so it surfaces.
          spdlog::warn(
            "uring_reactor: chunk completed with no H2D after retry. "
            "slot={} file_off={} io_size={} data_off={} data_size={} bytes_read={} actual={} "
            "ctx_failed={}",
            si,
            req.file_off,
            req.io_size,
            req.data_off,
            req.data_size,
            sinfo.bytes_read,
            actual,
            req.ctx->failed.load(std::memory_order_relaxed));
          req.ctx->chunk_done();
          sinfo = slot_info{};
        }
      }
    }
    if (need_resubmit) io_uring_submit(ring.get());
  };

  // Single park atomic: _wake_seq is bumped by every event source —
  // queue enqueues, interrupt(), and cuda_copy_cb.  CQE delivery is the
  // only exception: those are signalled by the kernel directly, so we
  // wait on io_uring_wait_cqe_timeout when (and only when) there is
  // in-flight IO that can produce one.
  auto any_copying = [&]() {
    return std::ranges::any_of(slots, [](auto& s) { return s.state == slot_state::COPYING; });
  };

  while (true) {
    drain_queue();
    poll_cuda();  // retire COPYING slots whose callback has fired

    if (_stop.load(std::memory_order_acquire)) break;

    bool work_to_submit = !pending.empty() || !pending_host.empty();

    if (!work_to_submit && inflight == 0 && !any_copying()) {
      // Fully idle.  Park on _wake_seq with the standard race-guard:
      // load the seq, re-drain, and only park if still idle.
      uint64_t seq = _wake_seq.load(std::memory_order_acquire);
      drain_queue();
      poll_cuda();
      if (pending.empty() && pending_host.empty() && inflight == 0 && !any_copying() &&
          !_stop.load(std::memory_order_acquire)) {
        _wake_seq.wait(seq, std::memory_order_relaxed);
      }
      continue;
    }

    submit_pending();

    if (inflight > 0) {
      io_uring_cqe* tmp = nullptr;
      // Bounded wait so the top-of-loop _stop check is reachable even
      // when no CQE arrives.  SINGLE_ISSUER means we can't post a NOP
      // SQE from interrupt() to unblock a plain wait_cqe; the timeout
      // bounds shutdown latency to SHUTDOWN_POLL_MS instead.
      static constexpr long SHUTDOWN_POLL_MS = 100;
      __kernel_timespec ts{};
      ts.tv_sec  = SHUTDOWN_POLL_MS / 1000;
      ts.tv_nsec = (SHUTDOWN_POLL_MS % 1000) * 1'000'000L;
      int rc     = io_uring_wait_cqe_timeout(ring.get(), &tmp, &ts);
      if (rc < 0 && rc != -EINTR && rc != -ETIME) {
        spdlog::error("uring_reactor: io_uring_wait_cqe_timeout failed: {}", strerror(-rc));
        break;
      }
      reap_cqes();
      continue;  // loop top: drain_queue + poll_cuda handle any state change
    }

    // Here: nothing to submit, no in-flight IO, but has_active is true
    // because of COPYING slots.  Park on _wake_seq until cuda_copy_cb
    // (or new queue work / interrupt) bumps it.
    uint64_t seq = _wake_seq.load(std::memory_order_acquire);
    poll_cuda();  // race-guard: a callback may have fired since the top
    drain_queue();
    if (!pending.empty() || !pending_host.empty() || !any_copying() ||
        _stop.load(std::memory_order_acquire))
      continue;
    _wake_seq.wait(seq, std::memory_order_relaxed);
  }
}

}  // namespace sirius::io
