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

#include <fcntl.h>
#include <spdlog/spdlog.h>
#include <sys/stat.h>

#include <algorithm>
#include <deque>
#include <ranges>
#include <stdexcept>

namespace sirius::io {

// ---------------------------------------------------------------------------
// uring_io_object
// ---------------------------------------------------------------------------

uring_io_object::uring_io_object(std::string path) : _path(std::move(path))
{
  _fd = file_descriptor(::open(_path.c_str(), O_RDONLY));
  if (!_fd)
    throw std::runtime_error("uring_io_object: open failed: " + _path + ": " + strerror(errno));

  struct stat st{};
  if (::fstat(_fd.get(), &st) < 0)
    throw std::runtime_error("uring_io_object: fstat failed: " + std::string(strerror(errno)));
  _file_size = static_cast<size_t>(st.st_size);

  _fd_direct = file_descriptor(::open(_path.c_str(), O_RDONLY | O_DIRECT));
  if (!_fd_direct)
    throw std::runtime_error("uring_io_object: O_DIRECT open failed: " + _path + ": " +
                             strerror(errno));
}

// ---------------------------------------------------------------------------
// uring_reactor
// ---------------------------------------------------------------------------

uring_reactor::uring_reactor(unsigned ring_entries, size_t bounce_slot_size)
  : _ring_entries(ring_entries)
{
  for (int i = 0; i < static_cast<int>(NUM_CHUNKS); ++i) {
    void* raw = nullptr;
    // Portable flag so these buffers are reachable from CUDA contexts
    // other than the one current at ioctx construction (multi-GPU usage).
    CUDA_CHECK(cudaHostAlloc(&raw, bounce_slot_size, cudaHostAllocPortable));
    _bounce[i].buf.reset(raw);
    _cb_args[i] = {this, i};
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

void uring_reactor::cuda_copy_cb(void* p) noexcept
{
  auto* arg = static_cast<cb_arg*>(p);
  arg->self->_bounce[arg->slot].cuda_done.store(true, std::memory_order_release);
  arg->self->_cuda_seq.fetch_add(1, std::memory_order_release);
  arg->self->_cuda_seq.notify_one();
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

void uring_reactor::enqueue_bulk(std::span<device_read_req_type> batch)
{
  if (batch.empty()) return;
  _queue.enqueue_bulk(std::make_move_iterator(batch.begin()), batch.size());
  _wake_seq.fetch_add(1, std::memory_order_release);
  _wake_seq.notify_one();
}

size_t uring_reactor::host_read(int fd, size_t offset, size_t size, uint8_t* dst)
{
  if (size == 0) return 0;
  ssize_t n = ::pread(fd, dst, size, static_cast<off_t>(offset));
  if (n < 0)
    throw std::runtime_error("uring_reactor::host_read pread: " + std::string(strerror(errno)));
  return static_cast<size_t>(n);
}

void uring_reactor::host_read_async(host_read_req_type req)
{
  _host_queue.enqueue(std::move(req));
  _wake_seq.fetch_add(1, std::memory_order_release);
  _wake_seq.notify_one();
}

void uring_reactor::host_enqueue_bulk(std::span<host_read_req_type> batch)
{
  if (batch.empty()) return;
  _host_queue.enqueue_bulk(std::make_move_iterator(batch.begin()), batch.size());
  _wake_seq.fetch_add(1, std::memory_order_release);
  _wake_seq.notify_one();
}

void uring_reactor::worker_loop()
{
  unique_ring ring = [this]() -> unique_ring {
#if defined(IORING_SETUP_SINGLE_ISSUER) && defined(IORING_SETUP_DEFER_TASKRUN)
    auto r = std::make_unique<io_uring>();
    int rc = io_uring_queue_init(
      _ring_entries, r.get(), IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_DEFER_TASKRUN);
    if (rc == 0) {
      spdlog::debug("uring_reactor: ring using SINGLE_ISSUER|DEFER_TASKRUN");
      return unique_ring{r.release()};
    }
    spdlog::debug(
      "uring_reactor: SINGLE_ISSUER|DEFER_TASKRUN unsupported "
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

  static constexpr uint64_t HOST_TAG = 1ULL << 63;

  enum class slot_state : uint8_t { FREE, READING, COPYING };
  struct slot_info {
    slot_state state{slot_state::FREE};
    device_read_req_type req{};
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
        slots[i].req.ctx->chunk_done();
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
      spdlog::trace("reactor submit slot={} fd={} file_off={} io_size={:.2f}MB",
                    si,
                    req.handle,
                    req.file_off,
                    to_mb(req.io_size));
      io_uring_prep_read(
        sqe, req.handle, _bounce[si].buf.get(), (unsigned)req.io_size, (__u64)req.file_off);
      io_uring_sqe_set_data64(sqe, (uint64_t)si);
      slots[si].state = slot_state::READING;
      slots[si].req   = std::move(req);
      pending.pop_front();
      ++inflight;
      ++added;
    }
    // Host reads: heap-allocate the req and tag the pointer in user_data.
    while (!pending_host.empty()) {
      io_uring_sqe* sqe = io_uring_get_sqe(ring.get());
      if (!sqe) break;
      auto* req = new host_read_req_type(std::move(pending_host.front()));
      pending_host.pop_front();
      spdlog::trace("reactor host submit fd={} offset={} size={:.2f}MB",
                    req->handle,
                    req->offset,
                    to_mb(req->size));
      io_uring_prep_read(sqe, req->handle, req->dst, (unsigned)req->size, (__u64)req->offset);
      io_uring_sqe_set_data64(sqe, reinterpret_cast<uint64_t>(req) | HOST_TAG);
      ++inflight;
      ++added;
    }
    if (added > 0) io_uring_submit(ring.get());
  };
  auto reap_cqes = [&]() {
    io_uring_cqe* cqes[NUM_CHUNKS];
    unsigned n = io_uring_peek_batch_cqe(ring.get(), cqes, NUM_CHUNKS);
    for (auto* cqe : std::span{cqes, n}) {
      uint64_t data = io_uring_cqe_get_data64(cqe);
      int res       = cqe->res;
      io_uring_cqe_seen(ring.get(), cqe);
      --inflight;

      if (data & HOST_TAG) {
        // Host read completion.
        auto* req = reinterpret_cast<host_read_req_type*>(data & ~HOST_TAG);
        if (res < 0) {
          req->ctx->chunk_failed(std::make_exception_ptr(
            std::runtime_error("reactor host read: " + std::string(strerror(-res)))));
        } else {
          spdlog::trace("reactor host done fd={} offset={} rd={:.2f}MB",
                        req->handle,
                        req->offset,
                        to_mb((size_t)res));
          req->ctx->chunk_done();
        }
        delete req;
      } else {
        // Device read completion.
        int si      = static_cast<int>(data);
        auto& sinfo = slots[si];
        auto& req   = sinfo.req;
        if (res < 0) {
          req.ctx->chunk_failed(std::make_exception_ptr(
            std::runtime_error("reactor read: " + std::string(strerror(-res)))));
          sinfo = slot_info{};
        } else {
          size_t rd     = (size_t)res;
          size_t actual = rd > req.data_off ? std::min(req.data_size, rd - req.data_off) : 0;
          if (actual > 0 && !req.ctx->failed.load(std::memory_order_relaxed)) {
            _bounce[si].cuda_done.store(false, std::memory_order_relaxed);
            if (req.device_id >= 0) cudaSetDevice(req.device_id);
            cudaMemcpyAsync(req.dst,
                            (uint8_t*)_bounce[si].buf.get() + req.data_off,
                            actual,
                            cudaMemcpyHostToDevice,
                            req.stream);
            cudaLaunchHostFunc(req.stream, cuda_copy_cb, &_cb_args[si]);
            sinfo.state = slot_state::COPYING;
          } else {
            req.ctx->chunk_done();
            sinfo = slot_info{};
          }
        }
      }
    }
  };

  while (true) {
    // Always drain the moodycamel queues into local dequeues BEFORE checking
    // has_active() — the dequeues are the only state has_active() looks at.
    // Without this, a producer that enqueued + bumped wake_seq before our
    // seq.load() will leave work sitting invisible in the moodycamel queue,
    // and we'll park on seq=current_value and never wake.
    drain_queue();

    if (!has_active()) {
      if (_stop.load(std::memory_order_acquire)) break;
      uint64_t seq = _wake_seq.load(std::memory_order_acquire);
      // Re-drain AFTER the seq load.  Any producer whose enqueue is now
      // observable either (a) landed before our load and is pulled in here,
      // making has_active() true, or (b) landed after our load, in which
      // case their fetch_add(1) made wake_seq != seq so the wait below
      // returns immediately.  Either way, no lost wake-up.
      drain_queue();
      if (!has_active()) { _wake_seq.wait(seq, std::memory_order_relaxed); }
      if (_stop.load(std::memory_order_acquire)) break;
      // Pull whatever work woke us up before the main body runs.
      drain_queue();
    }

    submit_pending();

    if (inflight > 0) {
      io_uring_cqe* tmp = nullptr;
      io_uring_wait_cqe(ring.get(), &tmp);
      reap_cqes();
    }

    uint64_t cuda_seq = _cuda_seq.load(std::memory_order_acquire);
    poll_cuda();

    if (inflight == 0 && pending.empty() && has_active()) {
      _cuda_seq.wait(cuda_seq, std::memory_order_acquire);
    }
  }
}

}  // namespace sirius::io
