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

#include "cucascade/cuda/event.hpp"
#include "driver_types.h"
#include "io/details/slot_pool.hpp"
#include "io/types.hpp"
#include "io/uring/types.hpp"
#include "util/error_utils.hpp"

#include <rmm/cuda_device.hpp>

#include <absl/cleanup/cleanup.h>
#include <fcntl.h>
#include <log/logging.hpp>
#include <numa.h>
#include <sys/stat.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <vector>

namespace sirius::io::uring {

using request_type_ptr             = std::unique_ptr<rx_request>;
using chunk_io_request_type_ptr    = std::unique_ptr<chunked_rx_request>;
static constexpr size_t NUM_CHUNKS = 64;  // max concurrent device reads, i.e. ring size / 2

namespace {

/// True iff @p v is a multiple of IO_BLOCK_SIZE (O_DIRECT page size).
[[maybe_unused]] [[nodiscard]] constexpr bool is_block_aligned(size_t v) noexcept
{
  return (v & (static_cast<size_t>(IO_BLOCK_SIZE) - 1)) == 0;
}

/// True iff @p errc (a positive errno) means the kernel could not serve a
/// fixed-buffer (IORING_OP_READ_FIXED) read because the registered-buffer table
/// is missing or incompatible — so the slot should retry as a plain read.
/// Other errno values are genuine I/O failures and must be reported, not masked
/// by a silent fallback.
[[nodiscard]] constexpr bool is_fixed_buffer_error(int errc) noexcept
{
  return errc == EOPNOTSUPP || errc == EINVAL || errc == EFAULT || errc == ENOBUFS ||
         errc == ENOMEM;
}

struct io_slot {
  enum class h2d_sync_hint {
    h2d_failed,       // no host-to-device copy needed for this request
    h2d_not_needed,   // host-to-device copy has been issued and is in-flight
    h2d_detached,     // host-to-device copy has completed, slot can be released
    h2d_event_based,  // host-to-device copy is in-flight and should be synchronized through the
                      // event
  };

  using slot_token = slot_pool::token;
  explicit io_slot(int slot_index, uint8_t* internal_buffer, bool support_fixed_buffers = true)
    : slot_index(slot_index),
      internal_buffer(internal_buffer),
      support_fixed_buffers(support_fixed_buffers)
  {
    assert(internal_buffer && "io_slot: internal_buffer must not be null");
  }

  void register_sqe(io_uring_sqe* sqe)
  {
    assert(sqe);
    if (req->is_vectored()) {
      // Rebuild the remaining iovec list from the running byte cursor so a
      // resubmitted short read picks up where it left off.  resume_iov is a
      // slot member so the array outlives the SQE until its CQE is reaped, and
      // is filled in place to reuse its capacity across resubmissions.
      req->fill_remaining_iovecs(bytes_read, resume_iov);
      assert(!resume_iov.empty() && "vectored resume produced an empty iovec list");
      io_uring_prep_readv(sqe,
                          req->fd,
                          resume_iov.data(),
                          static_cast<unsigned>(resume_iov.size()),
                          static_cast<__u64>(req->chunk.offset + bytes_read));
    } else {
      register_host_buffer(sqe);
    }
    io_uring_sqe_set_data64(sqe, static_cast<uint64_t>(slot_index));
  }

  /// Prepare the non-vectored read for this slot's remaining range, choosing the
  /// fixed-buffer (pre-registered) path when it is available.  Fixed buffers are
  /// only the reactor's internal bounce slots — caller-owned buffers are never
  /// registered — so prep_read_fixed is used only when the read stages through
  /// the internal buffer @e and fixed-buffer support has not been disabled.  All
  /// other reads take the plain prep_read path.
  ///
  /// io_uring_prep_read_fixed cannot fail synchronously; a fixed read that the
  /// kernel rejects surfaces as a CQE error, at which point the reap loop clears
  /// @c support_fixed_buffers on this slot and resubmits — this method then
  /// re-preps it as a plain read.  @c used_fixed_buffer records which path was
  /// taken so the reap loop can tell a fixed-read failure apart.
  void register_host_buffer(io_uring_sqe* sqe)
  {
    auto segment = req->get_remaining_chunk(bytes_read);
    if (use_internal_buffer && support_fixed_buffers) {
      io_uring_prep_read_fixed(sqe,
                               req->fd,
                               segment.data(),
                               static_cast<unsigned>(segment.size),
                               static_cast<__u64>(segment.offset),
                               slot_index);
      used_fixed_buffer = true;
    } else {
      io_uring_prep_read(sqe,
                         req->fd,
                         segment.data(),
                         static_cast<unsigned>(segment.size),
                         static_cast<__u64>(segment.offset));
      used_fixed_buffer = false;
    }
  }

  void on_request(chunk_io_request_type_ptr r,
                  slot_token token,
                  cucascade::cuda::cuda_event* cu_event = nullptr)
  {
    req = std::move(r);
    // Vectored (readv) requests always carry caller-owned buffers, so they
    // never borrow the internal bounce slot and never use the fixed-buffer path.
    use_internal_buffer = !req->is_vectored() && req->chunk.data() == nullptr;
    assert(!(req->is_vectored() && req->chunk.data() == nullptr) &&
           "vectored request must carry caller-owned buffers");
    if (use_internal_buffer) { req->chunk.set_data(internal_buffer); }
    bytes_read       = 0;
    this->pool_token = std::move(token);
    event            = cu_event;
    resume_iov.clear();
  }

  void on_error(const typename request_manager::error_type& error,
                std::source_location loc = std::source_location::current())
  {
    req->manager->report_error(error, loc);
    reset();
  }

  void on_complete(std::size_t n_bytes)
  {
    req->manager->chunk_complete(n_bytes);
    reset();
  }

  h2d_sync_hint copy_h2d_async(cudaError_t& err)
  {
    err = cudaSuccess;
    if (!req->cpy_req) { return h2d_sync_hint::h2d_not_needed; }
    cudaEvent_t copy_event = event ? (use_internal_buffer ? event->get() : nullptr) : nullptr;
    err                    = req->copy_h2d_async(copy_event);
    if (err != cudaSuccess) { return h2d_sync_hint::h2d_failed; }
    return event ? h2d_sync_hint::h2d_event_based : h2d_sync_hint::h2d_detached;
  }

  void reset()
  {
    req.reset();
    pool_token.reset();
    bytes_read = 0;
  }

  slot_token release_slot() noexcept { return std::exchange(pool_token, {}); }

  int slot_index;
  uint8_t* const internal_buffer;
  std::unique_ptr<chunked_rx_request> req;
  bool use_internal_buffer{false};
  // Whether this slot may submit reads against pre-registered (fixed) buffers.
  // Set at construction from the ring's buffer-registration result; cleared by
  // the reap loop if a fixed read fails so the slot falls back to plain reads.
  bool support_fixed_buffers{true};
  // Records whether the in-flight read used the fixed-buffer path, so the reap
  // loop can distinguish a fixed-read failure (which triggers fallback) from an
  // ordinary I/O error.
  bool used_fixed_buffer{false};
  size_t bytes_read{0};
  cucascade::cuda::cuda_event* event;
  slot_token pool_token;
  // Scratch iovec list backing an in-flight readv SQE.  Rebuilt by
  // register_sqe from the byte cursor; must outlive the SQE until its CQE is
  // reaped.  Empty for non-vectored requests.
  std::vector<iovec> resume_iov;
};

/// Build one device-read chunk for the O_DIRECT-aligned file window
/// [@p window_off, @p window_off + @p read_size) (read_size <= _bounce_slot_size, with
/// both ends page aligned).
///
/// The reactor reads that whole window into @p host_buf — when @p host_buf is
/// null it stages the read through one of its own internal bounce slots
/// instead.  Once the data lands, the worker H2D-copies just the part of this
/// window that overlaps the request [@p req_offset, @p req_offset + @p req_size)
/// into @p dst at its position within that request.  For the first window the
/// copy offset is the alignment overhang (req_offset - window_off); for the
/// rest it is zero and the dst position advances by the bytes already copied.
[[nodiscard]] chunk_io_request_type_ptr make_device_chunk(int fd,
                                                          size_t window_off,
                                                          size_t read_size,
                                                          uint8_t* host_buf,
                                                          size_t req_offset,
                                                          size_t req_size,
                                                          uint8_t* dst,
                                                          rmm::cuda_stream_view stream,
                                                          int device_id,
                                                          size_t file_size,
                                                          std::shared_ptr<request_manager> manager)
{
  size_t const req_end = req_offset + req_size;
  size_t const data_lo = std::max(req_offset, window_off);
  size_t const data_hi = std::min(req_end, window_off + read_size);
  assert(data_lo < data_hi &&
         "make_device_chunk: window does not overlap the request — caller must filter "
         "non-overlapping segments before building a device copy");

  auto req       = std::make_unique<chunked_rx_request>();
  req->fd        = fd;
  req->chunk     = io_object_segment{window_off, read_size, host_buf};
  req->file_size = file_size;

  auto cpy       = std::make_unique<device_cpy_request>();
  cpy->stream    = stream;
  cpy->device_id = device_id;
  cpy->copies.push_back(device_cpy_request::copy{
    /*dst=*/dst + (data_lo - req_offset),  // where this window lands in dst
    /*src=*/nullptr,                       // resolved late to the (bounce or caller) host buffer
    /*src_off=*/data_lo - window_off,      // offset of the wanted data within host_buf
    /*size=*/data_hi - data_lo});
  req->cpy_req = std::move(cpy);

  req->manager = std::move(manager);
  return req;
}

/// Build one device-read chunk for a merged group of contiguous host buffers
/// (@p seg, whose buffer lengths are already O_DIRECT-clamped to the file end).
/// The whole group is read in a single readv (or a plain read when it carries
/// one buffer), then each buffer's overlap with the request
/// [@p req_offset, @p req_offset + @p req_size) is H2D-copied into @p dst at its
/// position within that request — a batch of copies issued together.
///
/// A multi-buffer group always carries real (caller-owned) host buffers — the
/// merge step never fuses a null-buffer segment into a readv — so those copies
/// hold absolute src pointers into separate host allocations.  A single-buffer
/// group may instead be a null-buffer (internal bounce slot) segment whose host
/// buffer is assigned late once a slot is acquired; for it the copy's src is
/// left null and the within-window offset goes in src_off, so copy_async
/// resolves it against the bounce buffer (mirrors make_device_chunk).
[[nodiscard]] chunk_io_request_type_ptr make_device_chunk_vectored(
  int fd,
  io_object_segment seg,
  size_t req_offset,
  size_t req_size,
  uint8_t* dst,
  rmm::cuda_stream_view stream,
  int device_id,
  size_t file_size,
  std::shared_ptr<request_manager> manager)
{
  size_t const req_end = req_offset + req_size;

  auto cpy       = std::make_unique<device_cpy_request>();
  cpy->stream    = stream;
  cpy->device_id = device_id;
  cpy->copies.reserve(seg.n_chunks());

  // Walk the buffers in file order, accumulating each buffer's file range from
  // the segment base so the request-overlap clip can be computed per buffer.
  size_t file_lo = seg.offset;
  for (auto const& b : seg.buffers) {
    size_t const file_hi = file_lo + b.iov_len;
    size_t const data_lo = std::max(req_offset, file_lo);
    size_t const data_hi = std::min(req_end, file_hi);
    assert(data_lo < data_hi &&
           "make_device_chunk_vectored: buffer does not overlap the request — caller must filter "
           "non-overlapping segments before building a device copy");
    // A real host buffer carries an absolute src into that allocation; a null
    // (bounce-slot) buffer leaves src null and puts the within-window offset in
    // src_off so copy_async resolves it against the late-assigned bounce buffer.
    auto* const base = static_cast<uint8_t*>(b.iov_base);
    cpy->copies.push_back(device_cpy_request::copy{
      /*dst=*/dst + (data_lo - req_offset),  // where this buffer lands in dst
      /*src=*/base != nullptr ? base + (data_lo - file_lo) : nullptr,  // wanted data in buffer
      /*src_off=*/base != nullptr ? 0 : (data_lo - file_lo),
      /*size=*/data_hi - data_lo});
    file_lo = file_hi;
  }

  auto req       = std::make_unique<chunked_rx_request>();
  req->fd        = fd;
  req->chunk     = std::move(seg);
  req->file_size = file_size;
  req->cpy_req   = std::move(cpy);
  req->manager   = std::move(manager);
  return req;
}

/// A merged read range paired with the backing fd it must be submitted on.
struct merged_segment {
  io_object_segment seg;
  int fd;
};

/// Fuse neighboring segments into vectored reads during request preparation.
/// A group extends while the next segment is contiguous in the file, shares the
/// same backing fd (@p fd_for), and the fused buffer count stays within
/// @p max_n_chunks.  Each group becomes one merged segment whose @c buffers are
/// the concatenated destination iovecs; a 1-buffer group is a plain read, a
/// multi-buffer group is a readv.  Input @p segments are consumed by move.
///
/// Only segments whose destination buffer is already allocated can be fused: a
/// null-buffer segment must read through one of the reactor's internal bounce
/// slots (a fixed-buffer single read), which the readv path cannot serve, so it
/// is always emitted as a standalone group and never fused into or onto a
/// neighbor.  A run of contiguous segments split by a null-buffer segment in the
/// middle therefore yields three groups: a vectored head, the single null
/// segment, and a vectored tail.
template <typename FdFor>
[[nodiscard]] std::vector<merged_segment> merge_contiguous(std::span<io_object_segment> segments,
                                                           size_t max_n_chunks,
                                                           FdFor&& fd_for)
{
  std::vector<merged_segment> merged;
  for (size_t i = 0; i < segments.size();) {
    int const group_fd = fd_for(segments[i]);
    io_object_segment seg{std::move(segments[i])};
    size_t j = i + 1;
    // A null-buffer segment needs an internal bounce slot and cannot be part of
    // a readv, so only grow the group when this segment carries a real buffer
    // and the next one does too.
    if (seg.is_buffer_allocated()) {
      // Test contiguity against the running group (@c seg) rather than
      // segments[j - 1]: each append keeps seg.offset + seg.size equal to the
      // end of the last fused segment, so this stays correct without relying on
      // segments[i] retaining its scalar fields after being moved-from.
      while (j < segments.size() && segments[j].is_buffer_allocated() &&
             seg.n_chunks() + segments[j].n_chunks() <= max_n_chunks &&
             contiguous(seg, segments[j]) && fd_for(segments[j]) == group_fd) {
        for (auto const& b : segments[j].buffers) {
          seg.append(b);
        }
        ++j;
      }
    }
    merged.push_back({std::move(seg), group_fd});
    i = j;
  }
  return merged;
}

/**
 * @brief Custom deleter for @c unique_ring: calls @c io_uring_queue_exit
 *        before freeing the allocation.
 */
struct ring_deleter {
  void operator()(io_uring* r) const noexcept
  {
    io_uring_queue_exit(r);
    delete r;
  }
};
using unique_ring_ptr = std::unique_ptr<io_uring, ring_deleter>;

unique_ring_ptr make_ring(unsigned depth)
{
#if defined(IORING_SETUP_SINGLE_ISSUER) && defined(IORING_SETUP_DEFER_TASKRUN)
  auto r                   = std::make_unique<io_uring>();
  struct io_uring_params p = {0};
  p.flags |= IORING_SETUP_SINGLE_ISSUER;
  p.flags |= IORING_SETUP_COOP_TASKRUN | IORING_SETUP_DEFER_TASKRUN;
  int rc = io_uring_queue_init_params(depth, r.get(), &p);
  if (rc == 0) {
    SIRIUS_LOG_TRACE("uring_device_reactor: ring using SINGLE_ISSUER|DEFER_TASKRUN, entries={}",
                     depth);
    return unique_ring_ptr{r.release()};
  }
  SIRIUS_LOG_TRACE(
    "uring_device_reactor: SINGLE_ISSUER|DEFER_TASKRUN unsupported "
    "({}), falling back to plain flags",
    strerror(-rc));
#endif
  auto r2 = std::make_unique<io_uring>();
  int rc2 = io_uring_queue_init(depth, r2.get(), 0);
  if (rc2 < 0) throw std::runtime_error("uring_reactor: ring init: " + std::string(strerror(-rc2)));
  SIRIUS_LOG_TRACE("uring_reactor: ring using plain flags, entries={}", depth);
  return unique_ring_ptr{r2.release()};
}

struct unique_ring {
  explicit unique_ring(unsigned depth) : ring(make_ring(depth)) {}
  ~unique_ring() noexcept = default;

  [[nodiscard]] io_uring* native_handle() const noexcept { return ring.get(); }

  /// Register @p iovecs as fixed buffers for the ring.  Returns true on success;
  /// on failure returns false (rather than throwing) so the reactor can fall
  /// back to plain, unregistered reads instead of aborting startup.
  [[nodiscard]] bool register_buffers(std::span<iovec> iovecs)
  {
    if (int rc = io_uring_register_buffers(ring.get(), iovecs.data(), iovecs.size()); rc < 0) {
      SIRIUS_LOG_WARN(
        "uring_reactor: io_uring_register_buffers failed ({}); fixed buffers disabled",
        strerror(-rc));
      return false;
    }
    return true;
  }

  [[nodiscard]] int wait_for(std::chrono::milliseconds timeout) const
  {
    io_uring_cqe* tmp = nullptr;
    // Bounded wait so the top-of-loop _stop check is reachable even when
    // no CQE arrives.  SINGLE_ISSUER means we can't post a NOP SQE from
    // interrupt() to unblock a plain wait_cqe; the timeout bounds shutdown
    // latency to SHUTDOWN_POLL_MS.
    __kernel_timespec ts{};
    ts.tv_sec  = timeout.count() / 1000;
    ts.tv_nsec = (timeout.count() % 1000) * 1'000'000L;
    int rc     = io_uring_wait_cqe_timeout(ring.get(), &tmp, &ts);
    if (rc < 0 && rc != -EINTR && rc != -ETIME) { return -rc; }
    return 0;
  }

  [[nodiscard]] int wait() const
  {
    io_uring_cqe* tmp = nullptr;
    int rc            = io_uring_wait_cqe(ring.get(), &tmp);
    if (rc < 0) { return -1; }
    return 0;
  }

  [[nodiscard]] io_uring_sqe* get_sqe() const { return io_uring_get_sqe(ring.get()); }

  [[nodiscard]] io_uring_sqe* get_sqe_with_drain() const { return io_uring_get_sqe(ring.get()); }

  [[nodiscard]] unsigned peek_cqe_batch(std::span<io_uring_cqe*> cqes) const
  {
    return io_uring_peek_batch_cqe(ring.get(), cqes.data(), cqes.size());
  }

  [[nodiscard]] void mark_cqe_seen(io_uring_cqe* cqe) const { io_uring_cqe_seen(ring.get(), cqe); }

  void submit([[maybe_unused]] std::size_t n_added)
  {
    if (int rc = io_uring_submit(ring.get()); rc < 0) {
      throw std::runtime_error("uring_reactor: io_uring_submit: " + std::string(strerror(-rc)));
    }
  }

 private:
  unique_ring_ptr ring;
};

}  // namespace

// ---------------------------------------------------------------------------
// uring_reactor
// ---------------------------------------------------------------------------

uring_reactor::uring_reactor(std::shared_ptr<reactor_context> ctx, std::string_view tname)
  : _ctx(std::move(ctx)), _tname(tname)
{
  if (!_ctx) { throw std::invalid_argument("uring_reactor: reactor_context must be non-null"); }
  if (_ctx->host_memory_resource() == nullptr) {
    throw std::invalid_argument("uring_reactor: context host_memory_resource must be non-null");
  }
  // Constructor only captures config — no pinned-memory allocation and no worker
  // thread until start().  Keeps a parked (unused) reactor cheap.
  _config           = _ctx->cfg();
  _bounce_slot_size = _ctx->host_memory_resource()->get_block_size();
}

void uring_reactor::start()
{
  if (_worker.joinable()) { return; }  // already started
  _bounce_storage =
    _ctx->host_memory_resource()->allocate_multiple_blocks(NUM_CHUNKS * _bounce_slot_size);
  _worker = std::jthread([this](const std::stop_token& stop_token) { worker_loop(stop_token); },
                         _stop_source.get_token());
  if (!_tname.empty()) {
    std::string full_name = _tname + "_worker";
    pthread_setname_np(_worker.native_handle(), full_name.c_str());
  }
}

uring_reactor::~uring_reactor() { shutdown(); }

std::unique_ptr<local_io_object> uring_reactor::create_io_object(std::string path)
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

  auto file_size = size(fd.native_handle());
  return std::make_unique<local_io_object>(
    std::move(path), std::move(fd), std::move(fd_direct), file_size);
}

size_t uring_reactor::size(int fd)
{
  struct stat st{};
  if (::fstat(fd, &st) != 0)
    throw std::runtime_error("uring_reactor::size: fstat failed: " + std::string(strerror(errno)));
  return static_cast<size_t>(st.st_size);
}

request_type_ptr uring_reactor::prep_host_rx_request(const reactor_config_type& cfg,
                                                     const io_object_type& file,
                                                     const io_object_segment& segment)
{
  if (segment.size == 0) { return rx_request::create({}); }

  int const fd = (cfg.use_odirect && segment.is_odirect_compatible()) ? file.odirect_handle()
                                                                      : file.buffered_handle();

  // The read lands directly in the caller's buffer (no bounce, no H2D copy).
  // io_uring_prep_read/readv encode the length in a 32-bit unsigned, so a single
  // submitted read cannot exceed 4 GiB - 1.  Split the range into
  // <= max_host_read_chunk pieces that land contiguously in the caller's buffer.
  // The split size is a power-of-two multiple of the largest IO block size, so
  // each piece of an O_DIRECT-aligned segment stays block-aligned.
  constexpr size_t max_host_read_chunk = size_t{1} << 30;  // 1 GiB
  size_t const n_chunks = (segment.size + max_host_read_chunk - 1) / max_host_read_chunk;
  auto manager          = std::make_shared<request_manager>(segment.size, n_chunks);

  std::vector<chunk_io_request_type_ptr> chunks;
  chunks.reserve(n_chunks);
  for (size_t done = 0; done < segment.size; done += max_host_read_chunk) {
    size_t const read_size = std::min(max_host_read_chunk, segment.size - done);
    auto req               = std::make_unique<chunked_rx_request>();
    req->fd                = fd;
    req->chunk     = io_object_segment{segment.offset + done, read_size, segment.data() + done};
    req->file_size = file.size();
    req->manager   = manager;
    chunks.push_back(std::move(req));
  }
  return rx_request::create(std::move(chunks));
}

request_type_ptr uring_reactor::prep_device_rx_request(const reactor_config_type& cfg,
                                                       const io_object_type& file,
                                                       uint8_t* dst,
                                                       size_t offset,
                                                       size_t size,
                                                       rmm::cuda_stream_view stream,
                                                       int device_id)
{
  if (size == 0) { return rx_request::create({}); }

  int const fd = cfg.use_odirect ? file.odirect_handle() : file.buffered_handle();
  // align_to_physical aligns the offset down and the end up to IO_BLOCK_SIZE
  // (clamped to the file), giving an O_DIRECT-compliant span.
  auto const phys =
    align_to_physical({static_cast<int64_t>(offset), static_cast<int64_t>(size)}, file.size());
  auto const a_start  = static_cast<size_t>(phys.offset());
  auto const a_end    = a_start + static_cast<size_t>(phys.size());
  size_t alinged_size = phys.size();
  auto manager =
    std::make_shared<request_manager>(size, (alinged_size + cfg.bounce_size - 1) / cfg.bounce_size);

  std::vector<chunk_io_request_type_ptr> chunks;
  for (size_t w = a_start; w < a_end; w += cfg.bounce_size) {
    size_t const read_size = std::min<size_t>(cfg.bounce_size, a_end - w);
    chunks.push_back(make_device_chunk(fd,
                                       w,
                                       read_size,
                                       /*host_buf=*/nullptr,
                                       offset,
                                       size,
                                       dst,
                                       stream,
                                       device_id,
                                       file.size(),
                                       manager));
  }
  return rx_request::create(std::move(chunks));
}

request_type_ptr uring_reactor::prep_host_to_device_rx_request(
  const reactor_config_type& cfg,
  const io_object_type& file,
  std::span<io_object_segment> segments,
  uint8_t* dst,
  size_t offset,
  size_t size,
  rmm::cuda_stream_view stream,
  int device_id)
{
  // Device read staged through caller-supplied pinned host buffers.  The
  // provider hands back one chunk-aligned segment per buffer it owns; the
  // reactor reads each chunk into that buffer and H2D-copies only the part that
  // overlaps the request into dst.
  if (size == 0 || segments.empty()) { return rx_request::create({}); }

  int const fd         = cfg.use_odirect ? file.odirect_handle() : file.buffered_handle();
  size_t const req_end = offset + size;

  // Every segment must overlap the device destination window [offset, req_end).
  // A segment that covers no part of it is a caller error: it would read into a
  // host buffer whose bytes never land in dst, and it would inflate the
  // request_manager's chunk count past the copies that actually fill dst.  Sum
  // the device-buffer bytes each segment contributes — that covered total (not
  // the chunk-aligned host read size, which over-reads whole O_DIRECT blocks)
  // is what the future reports back to the caller.
  size_t bytes_covered = 0;
  for (auto const& s : segments) {
    size_t const lo = std::max(offset, s.offset);
    size_t const hi = std::min(req_end, s.offset + s.size);
    if (lo >= hi) {
      throw std::runtime_error("prep_host_to_device_rx_request: segment [" +
                               std::to_string(s.offset) + ", " + std::to_string(s.offset + s.size) +
                               ") does not overlap the requested device range [" +
                               std::to_string(offset) + ", " + std::to_string(req_end) + ")");
    }
    bytes_covered += hi - lo;
  }

  // O_DIRECT requires the read length to stay block-aligned, so each buffer's
  // read is clamped to the block-rounded file end rather than the raw file
  // size: the file's final partial block is read in full and the bytes past
  // EOF are simply never copied into dst (the copy is clipped to the request,
  // which never exceeds file_size).  Without this clamp a chunk at the tail of
  // the file short-reads and the worker resubmits the remainder at a
  // non-block-aligned offset, which O_DIRECT rejects with EINVAL.  Only the
  // segment straddling the file end is clamped, and it is necessarily the last
  // one, so clamping in place before merging cannot break contiguity.
  size_t const file_end_aligned =
    (file.size() + IO_BLOCK_SIZE - 1) & ~(static_cast<size_t>(IO_BLOCK_SIZE) - 1);
  for (auto& s : segments) {
    size_t const read_size = std::min(s.size, file_end_aligned - s.offset);
    s                      = io_object_segment{s.offset, read_size, s.data()};
  }

  // Fuse contiguous bounce buffers into one readv per group (1 buffer => plain
  // read), capped at cfg.max_n_chunks; every group becomes one device chunk
  // that batch-copies its buffers' request-overlaps into dst.  All segments
  // share the same backing fd, so fd_for is constant.
  auto merged =
    merge_contiguous(segments, cfg.max_n_chunks, [fd](const io_object_segment&) { return fd; });
  auto manager = std::make_shared<request_manager>(bytes_covered, merged.size());

  std::vector<chunk_io_request_type_ptr> chunks;
  chunks.reserve(merged.size());
  for (auto& m : merged) {
    chunks.push_back(make_device_chunk_vectored(
      fd, std::move(m.seg), offset, size, dst, stream, device_id, file.size(), manager));
  }
  return rx_request::create(std::move(chunks));
}

request_type_ptr uring_reactor::prep_host_rxv_request(const reactor_config_type& cfg,
                                                      const io_object_type& file,
                                                      std::span<io_object_segment> segments)
{
  if (segments.empty()) { return rx_request::create({}); }

  size_t const fsize = file.size();

  // Per-segment backing fd: O_DIRECT when enabled and the segment is aligned,
  // buffered otherwise.  Segments only fuse into one readv if they share an fd.
  auto fd_for = [&](const io_object_segment& s) {
    return (cfg.use_odirect && s.is_odirect_compatible()) ? file.odirect_handle()
                                                          : file.buffered_handle();
  };

  // Requested bytes = sum of per-segment sizes clamped to the file end (a
  // segment at the tail reads fewer bytes than its size).  This is what the
  // future returns, never the over-read amount.  Merging does not change it.
  size_t bytes_requested = 0;
  for (auto const& s : segments) {
    bytes_requested += s.offset < fsize ? std::min(s.size, fsize - s.offset) : 0;
  }

  // Fuse contiguous, same-fd segments into vectored reads (1 buffer => plain
  // read, >1 => readv).  total_chunks == number of merged segments: each emitted
  // chunked_rx_request calls chunk_complete exactly once.
  auto merged  = merge_contiguous(segments, cfg.max_n_chunks, fd_for);
  auto manager = std::make_shared<request_manager>(bytes_requested, merged.size());

  std::vector<chunk_io_request_type_ptr> chunks;
  chunks.reserve(merged.size());
  for (auto& m : merged) {
    auto req       = std::make_unique<chunked_rx_request>();
    req->fd        = m.fd;
    req->chunk     = std::move(m.seg);
    req->file_size = fsize;
    req->manager   = manager;
    chunks.push_back(std::move(req));
  }
  return rx_request::create(std::move(chunks));
}

void uring_reactor::interrupt() {}

void uring_reactor::shutdown()
{
  if (_worker.joinable()) {
    _stop_source.request_stop();
    _worker.join();
  }
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

std::vector<cudf::io::text::byte_range_info> uring_reactor::align_and_coalesce(
  std::span<const cudf::io::text::byte_range_info> ranges, std::optional<size_t> alignment)
{
  // O_DIRECT mandates IO_BLOCK_SIZE alignment, so it is the floor: honor a
  // larger caller request, ignore anything smaller (including an unset value).
  size_t const align = std::max<size_t>(alignment.value_or(IO_BLOCK_SIZE), IO_BLOCK_SIZE);

  // Round each range's ends outward to `align`; drop empty ranges.  Integer
  // (not bitmask) rounding so a non-power-of-two caller alignment still works.
  std::vector<cudf::io::text::byte_range_info> aligned;
  aligned.reserve(ranges.size());
  for (auto const& r : ranges) {
    if (r.size() <= 0) { continue; }
    auto const offset  = static_cast<size_t>(r.offset());
    auto const end     = offset + static_cast<size_t>(r.size());
    size_t const start = (offset / align) * align;
    size_t const stop  = ((end + align - 1) / align) * align;
    aligned.emplace_back(static_cast<int64_t>(start), static_cast<int64_t>(stop - start));
  }
  if (aligned.empty()) { return aligned; }

  // Sort by offset so one forward pass can fuse overlapping/adjacent ranges.
  std::sort(aligned.begin(), aligned.end(), [](auto const& a, auto const& b) {
    return a.offset() < b.offset();
  });

  std::vector<cudf::io::text::byte_range_info> coalesced;
  coalesced.reserve(aligned.size());
  coalesced.push_back(aligned.front());
  for (size_t i = 1; i < aligned.size(); ++i) {
    auto& last            = coalesced.back();
    auto const last_start = static_cast<size_t>(last.offset());
    auto const last_end   = last_start + static_cast<size_t>(last.size());
    auto const cur_start  = static_cast<size_t>(aligned[i].offset());
    auto const cur_end    = cur_start + static_cast<size_t>(aligned[i].size());
    if (cur_start <= last_end) {  // overlap or adjacency (ends are aligned)
      size_t const new_end = std::max(last_end, cur_end);
      last                 = {last.offset(), static_cast<int64_t>(new_end - last_start)};
    } else {
      coalesced.push_back(aligned[i]);
    }
  }
  return coalesced;
}

bool uring_reactor::supports(std::string_view path)
{
  std::error_code ec;
  std::filesystem::path p{path};
  return std::filesystem::is_regular_file(p, ec) && !ec;
}

size_t uring_reactor::host_read(const io_object_type& file,
                                size_t offset,
                                size_t size,
                                uint8_t* dst)
{
  if (size == 0) return 0;
  // Loop until either the full requested size is read, EOF (n == 0), or a
  // real error. pread on a regular file should only return short on EOF, but
  // we retry defensively against EINTR and any unexpected short-read paths
  // so callers don't have to.
  size_t total = 0;
  while (total < size) {
    ssize_t n = ::pread(
      file.buffered_handle(), dst + total, size - total, static_cast<off_t>(offset + total));
    if (n < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("uring_reactor::host_read pread: " + std::string(strerror(errno)));
    }
    if (n == 0) break;  // EOF
    total += static_cast<size_t>(n);
  }
  return total;
}

void uring_reactor::enqueue(request_type_ptr req)
{
  auto chunks = req->get_all_chunks();
  enqueue_chunks(chunks);
}

void uring_reactor::enqueue_chunks(std::span<chunk_io_request_type_ptr> batch)
{
  bool success = _requests.enqueue_bulk(std::make_move_iterator(batch.data()), batch.size());
  if (!success) {
    throw std::runtime_error("uring_reactor::enqueue_chunks: failed to enqueue bulk requests");
  }
}

void uring_reactor::enqueue_chunk(chunk_io_request_type_ptr request)
{
  bool success = _requests.enqueue(std::move(request));
  if (!success) {
    throw std::runtime_error("uring_reactor::enqueue_chunk: failed to enqueue request");
  }
}

void uring_reactor::worker_loop(const std::stop_token& stop_token)
{
  static constexpr std::chrono::milliseconds SHUTDOWN_POLL_MS{100};

  std::stop_callback cb(stop_token, [this] {
    SIRIUS_LOG_TRACE("uring_reactor worker_loop: stop requested");
    _requests.enqueue(nullptr);  // unblock the worker if it's waiting on an empty queue
  });

  using slot_token = slot_pool::token;

  unique_ring ring(2 * NUM_CHUNKS);

  auto blocks = _bounce_storage->get_blocks();
  std::vector<iovec> iovecs;
  iovecs.reserve(blocks.size());
  std::ranges::transform(
    blocks, std::back_inserter(iovecs), [len = _bounce_slot_size](auto* b) mutable {
      return iovec{.iov_base = b, .iov_len = len};
    });

  // Register the bounce buffers up front so slots know whether the fixed-buffer
  // read path is available.  If registration fails the reactor still works —
  // every slot just falls back to plain (unregistered) reads.
  bool const support_fixed_buffers = ring.register_buffers(iovecs);

  slot_pool slot_pool{NUM_CHUNKS};
  std::vector<io_slot> slots;
  slots.reserve(NUM_CHUNKS);
  std::ranges::transform(
    iovecs, std::back_inserter(slots), [i = 0, support_fixed_buffers](auto& b) mutable {
      return io_slot(i++, reinterpret_cast<uint8_t*>(b.iov_base), support_fixed_buffers);
    });

  std::array<io_uring_cqe*, NUM_CHUNKS> cqes;
  std::vector<int> incomplete_requests;
  incomplete_requests.reserve(NUM_CHUNKS);
  std::vector<slot_token> copying_slots;
  copying_slots.reserve(NUM_CHUNKS);
  std::unordered_map<int, std::vector<cucascade::cuda::cuda_event>> per_device_copy_events;
  auto n_devices = rmm::get_num_cuda_devices();
  for (int device_id = 0; device_id < n_devices; ++device_id) {
    rmm::cuda_set_device_raii device_guard(rmm::cuda_device_id{device_id});
    auto& events = per_device_copy_events[device_id];
    events.reserve(NUM_CHUNKS);
    std::generate_n(std::back_inserter(events), NUM_CHUNKS, []() {
      return cucascade::cuda::cuda_event{cudaEventDisableTiming};
    });
  }

  int inflight = 0;

  auto poll_copy_completions = [&]() {
    using query_status = cucascade::cuda::event::query_result;
    copying_slots.erase(std::remove_if(copying_slots.begin(),
                                       copying_slots.end(),
                                       [&](slot_token const& token) {
                                         int si         = token.slot_index();
                                         auto& s        = slots[si];
                                         auto ev_status = s.event->query();
                                         return !(ev_status == query_status::in_progress);
                                       }),
                        copying_slots.end());
  };

  auto drain_and_submit = [&]() {
    int added          = 0;
    bool wait_for_copy = false;
    while (true) {
      auto slot = slot_pool.try_acquire_token();
      if (!slot) {
        if (inflight == 0 && !copying_slots.empty() && !std::exchange(wait_for_copy, true)) {
          SIRIUS_TRY_AND_LOG_EXCEPTION(
            slots[copying_slots.back().slot_index()].event->synchronize(),
            "uring_reactor: failed to synchronize copy event for slot {}",
            copying_slots.back().slot_index());
          poll_copy_completions();
          continue;
        }
        break;
      }

      auto& s                      = slots[slot.slot_index()];
      chunk_io_request_type_ptr dr = nullptr;
      while (dr == nullptr) {
        if (!_requests.try_dequeue(dr) && inflight == 0) { _requests.wait_dequeue(dr); }
        if (dr && dr->manager->has_error()) {
          // If the request is already in error state, skip it.
          dr.reset(nullptr);
          continue;
        }
        break;
      }
      if (dr == nullptr) {
        break;  // queue empty
      }
      cucascade::cuda::cuda_event* cu_event = nullptr;
      if (dr->needs_event_for_synchronization()) {
        cu_event = std::addressof(per_device_copy_events[dr->cpy_req->device_id][s.slot_index]);
      }

      s.on_request(std::move(dr), std::move(slot), cu_event);

      auto* sqe = ring.get_sqe();
      if (!sqe) {
        incomplete_requests.push_back(s.slot_index);
        break;
      }
      s.register_sqe(sqe);
      ++inflight;
      ++added;
    }
    if (added > 0) { ring.submit(added); }
  };

  auto reap_cqes = [&]() {
    unsigned n = ring.peek_cqe_batch(cqes);
    for (auto* cqe : std::span{cqes.data(), n}) {
      uint64_t raw  = io_uring_cqe_get_data64(cqe);
      int si        = static_cast<int>(raw);
      int cqe_bytes = cqe->res;
      ring.mark_cqe_seen(cqe);
      --inflight;

      auto& s = slots[si];

      if (cqe_bytes < 0) {
        int const errc = -cqe_bytes;
        // A fixed-buffer read the kernel can't serve (registered-buffer table
        // missing/incompatible): disable the fixed path on this slot and
        // resubmit — register_bound_buffer re-preps it as a plain read.  No
        // bytes landed, so the resubmit reads the whole range from scratch.
        if (s.used_fixed_buffer && is_fixed_buffer_error(errc)) {
          SIRIUS_LOG_WARN(
            "uring_reactor: fixed-buffer read failed on slot {} ({}); "
            "falling back to plain read",
            si,
            strerror(errc));
          s.support_fixed_buffers = false;
          incomplete_requests.push_back(si);
          continue;
        }
        s.on_error(std::error_code(errc, std::generic_category()));
        continue;
      }

      // For readv, cqe_bytes is the total read across all iovecs; the EOF/
      // short-read arithmetic below is offset-based and works for both modes.
      s.bytes_read += static_cast<size_t>(cqe_bytes);
      bool const fully_read = s.bytes_read >= s.req->chunk.size;
      bool const eof = cqe_bytes == 0 || s.req->chunk.offset + s.bytes_read >= s.req->file_size;

      if (!fully_read && !eof) {
        incomplete_requests.push_back(si);
        continue;
      }

      cudaError_t err = cudaSuccess;
      auto hint       = s.copy_h2d_async(err);
      if (hint == io_slot::h2d_sync_hint::h2d_failed) {
        s.on_error(err);
        continue;
      } else if (hint == io_slot::h2d_sync_hint::h2d_event_based) {
        copying_slots.push_back(s.release_slot());
      }
      s.on_complete(s.bytes_read);
    }
  };

  auto resubmit_incomplete_requests = [&]() {
    size_t any_added = 0;
    while (!incomplete_requests.empty()) {
      int si    = incomplete_requests.back();
      auto& s   = slots[si];
      auto* sqe = ring.get_sqe_with_drain();
      if (!sqe) {
        // This should be very unlikely since we reserved enough SQEs for
        // every slot to be re-submitted once, but if it happens we can just
        // wait for the next CQE batch to drain some SQEs and try again then.
        break;
      }
      incomplete_requests.pop_back();
      s.register_sqe(sqe);
      ++inflight;
      ++any_added;
    }
    if (any_added > 0) { ring.submit(any_added); }
  };

  auto clean_up_and_shutdown = [&]() {
    // wait for all in-flight requests to complete so we don't report spurious errors on shutdown
    while (inflight > 0) {
      auto s = ring.wait_for(SHUTDOWN_POLL_MS);
      if (s) {
        SIRIUS_LOG_ERROR("uring_reactor: io_uring_wait_cqe failed during shutdown: {}",
                         strerror(s));
        break;
      }
      reap_cqes();
    }

    // wait for any in-flight copies to complete so we don't introduce illegal accesses when we
    // release the bounce buffers back to the memory resource
    std::for_each(copying_slots.begin(), copying_slots.end(), [&](auto& s) {
      int si     = s.slot_index();
      auto& slot = slots[si];
      if (slot.event) {
        SIRIUS_TRY_AND_LOG_EXCEPTION(slot.event->synchronize(),
                                     "uring_reactor: failed to synchronize copy event for slot {}",
                                     si);
      }
    });

    // Mark all pending requests as canceled so their managers don't wait indefinitely for
    // completion
    chunk_io_request_type_ptr dr = nullptr;
    while (_requests.try_dequeue(dr)) {
      if (dr) {
        dr->manager->report_error(std::make_error_code(std::errc::operation_canceled));
        dr.reset(nullptr);
      } else {
        break;  // queue empty
      }
    }
  };

  // The main loop: drain the request queue and submit new SQEs, wait for completions and reap
  {
    auto cleanup = absl::MakeCleanup([&]() { clean_up_and_shutdown(); });

    try {
      while (!stop_token.stop_requested()) {
        drain_and_submit();

        if (inflight > 0) {
          auto s = ring.wait_for(SHUTDOWN_POLL_MS);
          if (s) {
            SIRIUS_LOG_ERROR("uring_reactor: io_uring_wait_cqe_timeout failed: {}", strerror(s));
            break;
          }
          reap_cqes();
        }

        resubmit_incomplete_requests();

        poll_copy_completions();
      }
    } catch (const std::exception& e) {
      SIRIUS_LOG_ERROR("uring_reactor: exception: {}", e.what());
    }
  }
}

}  // namespace sirius::io::uring
