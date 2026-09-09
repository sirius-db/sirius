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
#include "io/cache/types.hpp"
#include "io/details/slot_pool.hpp"
#include "io/io_request.hpp"
#include "io/types.hpp"
#include "io/uring/types.hpp"

#include <rmm/cuda_device.hpp>

#include <fcntl.h>
#include <log/logging.hpp>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/uio.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace sirius::io::uring {

namespace {

constexpr std::size_t NUM_SLOTS           = 64;
constexpr std::size_t MAX_PLAIN_READ_SIZE = 1UL << 30;
constexpr std::chrono::milliseconds POLL_INTERVAL{20};
constexpr auto POLL_INTERVAL_US =
  std::chrono::duration_cast<std::chrono::microseconds>(POLL_INTERVAL).count();

[[nodiscard]] constexpr std::size_t saturating_add(std::size_t lhs, std::size_t rhs) noexcept
{
  return rhs > std::numeric_limits<std::size_t>::max() - lhs
           ? std::numeric_limits<std::size_t>::max()
           : lhs + rhs;
}

[[nodiscard]] constexpr std::size_t align_down(std::size_t value, std::size_t alignment) noexcept
{
  return alignment == 0 ? value : value - value % alignment;
}

[[nodiscard]] constexpr std::size_t align_up(std::size_t value, std::size_t alignment) noexcept
{
  if (alignment == 0) return value;
  auto const remainder = value % alignment;
  if (remainder == 0) return value;
  return saturating_add(value, alignment - remainder);
}

[[nodiscard]] constexpr bool is_fixed_buffer_error(int errc) noexcept
{
  return errc == EOPNOTSUPP || errc == EINVAL || errc == EFAULT || errc == ENOBUFS ||
         errc == ENOMEM;
}

[[nodiscard]] std::error_code canceled_error() noexcept
{
  return std::make_error_code(std::errc::operation_canceled);
}

struct ring_deleter {
  void operator()(io_uring* ring) const noexcept
  {
    if (ring != nullptr) {
      io_uring_queue_exit(ring);
      delete ring;
    }
  }
};

using unique_ring_ptr = std::unique_ptr<io_uring, ring_deleter>;

[[nodiscard]] unique_ring_ptr make_ring(unsigned depth)
{
#if defined(IORING_SETUP_SINGLE_ISSUER) && defined(IORING_SETUP_DEFER_TASKRUN)
  auto preferred = std::make_unique<io_uring>();
  io_uring_params params{};
  params.flags =
    IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_COOP_TASKRUN | IORING_SETUP_DEFER_TASKRUN;
  if (auto const rc = io_uring_queue_init_params(depth, preferred.get(), &params); rc == 0) {
    return unique_ring_ptr{preferred.release()};
  }
#endif

  auto fallback = std::make_unique<io_uring>();
  auto const rc = io_uring_queue_init(depth, fallback.get(), 0);
  if (rc < 0) {
    throw std::system_error(std::error_code{-rc, std::generic_category()},
                            "uring_reactor: io_uring_queue_init");
  }
  return unique_ring_ptr{fallback.release()};
}

class unique_ring {
 public:
  explicit unique_ring(unsigned depth) : _ring(make_ring(depth)) {}

  [[nodiscard]] io_uring_sqe* get_sqe() const noexcept { return io_uring_get_sqe(_ring.get()); }

  [[nodiscard]] unsigned peek(std::span<io_uring_cqe*> cqes) const noexcept
  {
    return io_uring_peek_batch_cqe(_ring.get(), cqes.data(), cqes.size());
  }

  void seen(io_uring_cqe* cqe) const noexcept { io_uring_cqe_seen(_ring.get(), cqe); }

  void submit(std::size_t expected, std::size_t& inflight)
  {
    std::size_t submitted = 0;
    while (submitted < expected) {
      auto const rc = io_uring_submit(_ring.get());
      if (rc <= 0) {
        auto const error = rc < 0 ? -rc : EIO;
        throw std::system_error(std::error_code{error, std::generic_category()},
                                "uring_reactor: io_uring_submit");
      }
      submitted += static_cast<std::size_t>(rc);
      inflight += static_cast<std::size_t>(rc);
    }
  }

  [[nodiscard]] int cancel_all_sync() const noexcept
  {
    io_uring_sync_cancel_reg cancel{};
    cancel.fd              = -1;
    cancel.flags           = IORING_ASYNC_CANCEL_ANY | IORING_ASYNC_CANCEL_ALL;
    cancel.timeout.tv_sec  = -1;
    cancel.timeout.tv_nsec = -1;
    return io_uring_register_sync_cancel(_ring.get(), &cancel);
  }

  [[nodiscard]] int wait_for(std::chrono::milliseconds timeout) const noexcept
  {
    io_uring_cqe* cqe = nullptr;
    __kernel_timespec ts{};
    ts.tv_sec     = timeout.count() / 1000;
    ts.tv_nsec    = (timeout.count() % 1000) * 1'000'000L;
    auto const rc = io_uring_wait_cqe_timeout(_ring.get(), &cqe, &ts);
    return rc < 0 && rc != -EINTR && rc != -ETIME ? -rc : 0;
  }

  [[nodiscard]] int run_deferred_taskwork() const noexcept
  {
    // Unlike io_uring_submit_and_wait(), io_uring_get_events() enters the
    // kernel with to_submit=0.  This is important on the terminal path: a
    // failed partial submission can leave prepared SQEs in the userspace SQ,
    // and those entries must not be published after their slots have been
    // classified as unsubmitted.  GETEVENTS still runs deferred task work and
    // flushes CQ overflow without consuming any such SQEs.
    auto const rc = io_uring_get_events(_ring.get());
    return rc < 0 && rc != -EINTR ? -rc : 0;
  }

  [[nodiscard]] bool register_buffers(std::span<iovec> buffers) const noexcept
  {
    auto const rc = io_uring_register_buffers(_ring.get(), buffers.data(), buffers.size());
    if (rc < 0) {
      SIRIUS_LOG_WARN("uring_reactor: fixed buffers disabled: {}", strerror(-rc));
      return false;
    }
    return true;
  }

 private:
  unique_ring_ptr _ring;
};

struct staging_lease {
  std::vector<slot_pool::token> tokens;
};

enum class slot_state { idle, reading, copying };

struct io_slot {
  explicit io_slot(int index, bool fixed_supported) : index(index), fixed_supported(fixed_supported)
  {
  }

  int index;
  bool fixed_supported;
  bool used_fixed{false};
  std::size_t bytes_read{0};
  slot_state state{slot_state::idle};
  std::unique_ptr<uring_io_op> op;
  std::vector<iovec> resume_iovecs;
  std::unique_ptr<cucascade::cuda::cuda_event> copy_event;
  int event_device{-1};

  void reset() noexcept
  {
    op.reset();
    resume_iovecs.clear();
    bytes_read = 0;
    used_fixed = false;
    state      = slot_state::idle;
  }

  void prepare_remaining_iovecs()
  {
    assert(op != nullptr && state == slot_state::reading);
    auto& request = op->request;
    detail::fill_remaining_iovecs(request.iovecs, bytes_read, resume_iovecs);
    if (resume_iovecs.empty()) {
      throw std::logic_error("uring_reactor: no buffers remain for an unfinished operation");
    }
  }

  void prepare_sqe(io_uring_sqe* sqe) noexcept
  {
    assert(sqe != nullptr && op != nullptr && state == slot_state::reading &&
           !resume_iovecs.empty());
    auto& request = op->request;

    auto const offset        = request.io_rng.offset + bytes_read;
    bool const can_use_fixed = op->needs_staging() && op->staging_blocks == 1 &&
                               resume_iovecs.size() == 1 && bytes_read == 0 && fixed_supported;
    if (can_use_fixed) {
      auto const& iov = resume_iovecs.front();
      io_uring_prep_read_fixed(sqe,
                               op->fd,
                               iov.iov_base,
                               static_cast<unsigned>(iov.iov_len),
                               static_cast<__u64>(offset),
                               index);
      used_fixed = true;
    } else if (resume_iovecs.size() == 1) {
      auto const& iov = resume_iovecs.front();
      io_uring_prep_read(
        sqe, op->fd, iov.iov_base, static_cast<unsigned>(iov.iov_len), static_cast<__u64>(offset));
      used_fixed = false;
    } else {
      io_uring_prep_readv(sqe,
                          op->fd,
                          resume_iovecs.data(),
                          static_cast<unsigned>(resume_iovecs.size()),
                          static_cast<__u64>(offset));
      used_fixed = false;
    }
    io_uring_sqe_set_data64(sqe, static_cast<std::uint64_t>(index));
  }
};

[[nodiscard]] range staged_physical_range(range logical,
                                          std::size_t file_size,
                                          bool use_odirect) noexcept
{
  if (!use_odirect) return logical;

  auto const start = align_down(logical.offset, IO_BLOCK_SIZE);
  auto const end =
    std::min(align_up(logical.end(), IO_BLOCK_SIZE), align_up(file_size, IO_BLOCK_SIZE));
  return end > start ? range{start, end - start} : range{start, 0};
}

void attach_common(uring_io_op& op,
                   std::shared_ptr<const io_object> const& object,
                   prepared_io_slice const& slice,
                   std::shared_ptr<grouped_coordinator> const& coordinator)
{
  op.request.obj         = object;
  op.request.coordinator = coordinator;
  op.request.on_complete = slice.on_complete;
  if (slice.has_device_request()) {
    op.request.device_copy            = std::make_unique<device_cpy_request>();
    op.request.device_copy->req_rng   = slice.rng;
    op.request.device_copy->d_buffer  = slice.d_buffer;
    op.request.device_copy->device_id = slice.d_buffer.device_id;
  }
}

[[nodiscard]] std::unique_ptr<uring_io_op> make_op(
  std::shared_ptr<const io_object> const& object,
  prepared_io_slice const& slice,
  std::shared_ptr<grouped_coordinator> const& coordinator,
  local_io_object const& file,
  range physical,
  bool use_odirect)
{
  auto op            = std::make_unique<uring_io_op>();
  op->fd             = use_odirect ? file.odirect_handle() : file.buffered_handle();
  op->file_size      = file.size();
  op->use_odirect    = use_odirect;
  op->request.io_rng = physical;
  attach_common(*op, object, slice, coordinator);
  return op;
}

[[nodiscard]] std::vector<std::unique_ptr<uring_io_op>> plan_slice(
  std::shared_ptr<const io_object> const& object,
  prepared_io_slice const& slice,
  std::shared_ptr<grouped_coordinator> const& coordinator,
  config const& cfg,
  std::size_t block_size,
  std::size_t backlog_bytes,
  std::size_t free_slots)
{
  auto const file = std::dynamic_pointer_cast<local_io_object const>(object);
  if (file == nullptr) {
    throw std::invalid_argument("uring_reactor: grouped request contains a foreign io_object");
  }
  if (slice.rng.empty()) {
    throw std::invalid_argument("uring_reactor: zero-sized prepared slice");
  }

  std::vector<std::unique_ptr<uring_io_op>> result;

  if (slice.is_staged()) {
    if (!slice.has_device_request()) {
      throw std::invalid_argument("uring_reactor: staging requires a device destination");
    }
    if (block_size == 0) {
      throw std::invalid_argument("uring_reactor: staging block size is zero");
    }

    bool const direct = detail::odirect_available(cfg.use_odirect, file->odirect_handle()) &&
                        block_size % IO_BLOCK_SIZE == 0;
    auto const physical = staged_physical_range(slice.rng, file->size(), direct);
    if (physical.empty()) {
      throw std::out_of_range("uring_reactor: staged range is outside the object");
    }

    auto target = detail::dynamic_io_target(backlog_bytes, free_slots, block_size);
    if (target == 0) target = std::min(block_size, max_dynamic_io_size);
    std::size_t consumed = 0;
    while (consumed < physical.size) {
      auto const bytes = std::min(target, physical.size - consumed);
      auto op          = make_op(
        object, slice, coordinator, *file, range{physical.offset + consumed, bytes}, direct);
      op->staging_blocks = bytes / block_size + (bytes % block_size != 0);
      result.push_back(std::move(op));
      consumed += bytes;
    }
    return result;
  }

  if (slice.is_contiguous()) {
    auto* const base = std::get<std::uint8_t*>(slice.h_buffer.buffer);
    if (base == nullptr) {
      throw std::invalid_argument("uring_reactor: contiguous buffer is null");
    }

    std::size_t consumed = 0;
    while (consumed < slice.rng.size) {
      auto const bytes    = std::min(MAX_PLAIN_READ_SIZE, slice.rng.size - consumed);
      auto const physical = range{slice.rng.offset + consumed, bytes};
      iovec const buffer{base + consumed, bytes};
      bool const direct =
        detail::odirect_available(cfg.use_odirect, file->odirect_handle()) &&
        detail::is_odirect_compatible(physical, std::span<iovec const>{&buffer, 1});
      auto op = make_op(object, slice, coordinator, *file, physical, direct);
      op->request.iovecs.push_back(buffer);
      result.push_back(std::move(op));
      consumed += bytes;
    }
    return result;
  }

  auto const chunks = slice.h_buffer.fragments();
  if (chunks.empty()) {
    throw std::invalid_argument("uring_reactor: fragmented buffer has no chunks");
  }
  if (block_size == 0) { throw std::invalid_argument("uring_reactor: cache block size is zero"); }

  auto target =
    detail::dynamic_io_target(backlog_bytes, std::max<std::size_t>(1, free_slots), block_size);
  if (target == 0) target = std::min(block_size, max_dynamic_io_size);
  auto const max_iovecs = static_cast<std::size_t>(IOV_MAX);

  std::unique_ptr<uring_io_op> current;
  std::size_t current_end = 0;

  auto flush = [&]() {
    if (current == nullptr) return;
    bool const direct =
      detail::odirect_available(cfg.use_odirect, file->odirect_handle()) &&
      detail::is_odirect_compatible(current->request.io_rng, current->request.iovecs);
    current->use_odirect = direct;
    current->fd          = direct ? file->odirect_handle() : file->buffered_handle();
    result.push_back(std::move(current));
  };

  for (auto* chunk : chunks) {
    if (chunk == nullptr || chunk->data == nullptr) {
      throw std::invalid_argument("uring_reactor: cache fragment is not allocated");
    }

    auto const [fill_begin, fill_end] =
      cache::fill_span(chunk->state.get_fill(), chunk->offset, block_size);
    if (fill_end <= fill_begin) {
      throw std::invalid_argument("uring_reactor: cache fragment has an empty fill span");
    }
    auto const bytes      = fill_end - fill_begin;
    bool const contiguous = current != nullptr && current_end == fill_begin;
    bool const fits_bytes =
      current != nullptr && bytes <= target - std::min(target, current->request.io_rng.size);
    bool const fits_iovecs = current != nullptr && current->request.iovecs.size() < max_iovecs;

    if (!contiguous || !fits_bytes || !fits_iovecs) {
      flush();
      current     = make_op(object, slice, coordinator, *file, range{fill_begin, 0}, false);
      current_end = fill_begin;
    }

    current->request.iovecs.push_back(iovec{chunk->data + (fill_begin - chunk->offset), bytes});
    current->request.completion_chunks.push_back(chunk);
    current->request.io_rng.size += bytes;
    current_end += bytes;
  }
  flush();
  return result;
}

}  // namespace

uring_reactor::uring_reactor(std::shared_ptr<reactor_context> ctx, std::string_view tname)
  : _ctx(std::move(ctx)), _tname(tname)
{
  if (_ctx == nullptr) {
    throw std::invalid_argument("uring_reactor: reactor_context must be non-null");
  }
  if (_ctx->host_memory_resource() == nullptr) {
    throw std::invalid_argument("uring_reactor: host memory resource must be non-null");
  }
  _config           = _ctx->cfg();
  _bounce_slot_size = _ctx->host_memory_resource()->get_block_size();
}

uring_reactor::~uring_reactor() { shutdown(); }

void uring_reactor::start()
{
  if (_worker.joinable()) return;
  if (_bounce_slot_size == 0) {
    throw std::invalid_argument("uring_reactor: staging block size must be non-zero");
  }

  _bounce_storage =
    _ctx->host_memory_resource()->allocate_multiple_blocks(NUM_SLOTS * _bounce_slot_size);
  if (_bounce_storage == nullptr || _bounce_storage->get_blocks().size() < NUM_SLOTS) {
    _bounce_storage.reset();
    throw std::runtime_error("uring_reactor: failed to allocate all staging slots");
  }

  std::lock_guard lock(_enqueue_mutex);
  try {
    _accepting.store(true, std::memory_order_release);
    _worker = std::jthread([this](std::stop_token stop_token) { worker_loop(stop_token); },
                           _stop_source.get_token());
  } catch (...) {
    _accepting.store(false, std::memory_order_release);
    _bounce_storage.reset();
    throw;
  }
  if (!_tname.empty()) {
    auto const name = _tname + "_worker";
    pthread_setname_np(_worker.native_handle(), name.c_str());
  }
}

void uring_reactor::interrupt() {}

void uring_reactor::shutdown()
{
  {
    std::lock_guard lock(_enqueue_mutex);
    _accepting.store(false, std::memory_order_release);
  }

  if (_worker.joinable()) {
    _stop_source.request_stop();
    _worker.join();
    return;
  }

  std::unique_ptr<grouped_io_request> request;
  while (_requests.try_dequeue(request)) {
    if (request == nullptr) continue;
    _queued_bytes.fetch_sub(request->remaining_bytes(), std::memory_order_relaxed);
    request->cancel_remaining(canceled_error());
  }
}

void uring_reactor::enqueue(std::unique_ptr<grouped_io_request> request) noexcept
{
  if (request == nullptr) return;

  std::lock_guard lock(_enqueue_mutex);
  if (!_accepting.load(std::memory_order_acquire)) {
    request->cancel_remaining(canceled_error());
    return;
  }

  auto const bytes = request->remaining_bytes();
  _queued_bytes.fetch_add(bytes, std::memory_order_relaxed);
  try {
    if (!_requests.enqueue(std::move(request))) {
      _queued_bytes.fetch_sub(bytes, std::memory_order_relaxed);
      if (request != nullptr) {
        request->cancel_remaining(std::make_error_code(std::errc::no_buffer_space));
      }
    }
  } catch (...) {
    _queued_bytes.fetch_sub(bytes, std::memory_order_relaxed);
    if (request != nullptr) request->cancel_remaining(std::current_exception());
  }
}

bool uring_reactor::supports(std::string_view path)
{
  std::error_code ec;
  return std::filesystem::is_regular_file(std::filesystem::path{path}, ec) && !ec;
}

std::unique_ptr<local_io_object> uring_reactor::create_io_object(std::string path)
{
  if (!supports(path)) {
    throw std::runtime_error("uring_reactor::create_io_object: unsupported path: " + path);
  }

  file_descriptor buffered{::open(path.c_str(), O_RDONLY)};
  if (!buffered) {
    throw std::system_error(
      errno, std::generic_category(), "uring_reactor::create_io_object: buffered open");
  }

  file_descriptor direct{::open(path.c_str(), O_RDONLY | O_DIRECT)};
  if (!direct) {
    SIRIUS_LOG_WARN("uring_reactor: O_DIRECT unavailable for '{}': {}; using buffered I/O",
                    path,
                    strerror(errno));
  }

  auto const file_size = size(buffered.get());
  return std::make_unique<local_io_object>(
    std::move(path), std::move(buffered), std::move(direct), file_size);
}

std::size_t uring_reactor::size(int native_handle)
{
  struct stat stat_buffer{};
  if (::fstat(native_handle, &stat_buffer) != 0) {
    throw std::system_error(errno, std::generic_category(), "uring_reactor::size");
  }
  return static_cast<std::size_t>(stat_buffer.st_size);
}

std::size_t uring_reactor::host_read(local_io_object const& file,
                                     std::size_t offset,
                                     std::size_t bytes,
                                     std::uint8_t* destination)
{
  std::size_t completed = 0;
  while (completed < bytes) {
    auto const result = ::pread(file.buffered_handle(),
                                destination + completed,
                                bytes - completed,
                                static_cast<off_t>(offset + completed));
    if (result < 0) {
      if (errno == EINTR) continue;
      throw std::system_error(errno, std::generic_category(), "uring_reactor::host_read");
    }
    if (result == 0) break;
    completed += static_cast<std::size_t>(result);
  }
  return completed;
}

cudf::io::text::byte_range_info uring_reactor::align_to_physical(
  cudf::io::text::byte_range_info logical, std::size_t file_size)
{
  if (logical.offset() < 0 || logical.size() <= 0) return {0, 0};

  auto const offset = static_cast<std::size_t>(logical.offset());
  auto const bytes  = static_cast<std::size_t>(logical.size());
  auto const begin  = align_down(offset, IO_BLOCK_SIZE);
  auto const end    = std::min(align_up(saturating_add(offset, bytes), IO_BLOCK_SIZE),
                            align_up(file_size, IO_BLOCK_SIZE));
  return end > begin ? cudf::io::text::byte_range_info{static_cast<std::int64_t>(begin),
                                                       static_cast<std::int64_t>(end - begin)}
                     : cudf::io::text::byte_range_info{static_cast<std::int64_t>(begin), 0};
}

std::vector<cudf::io::text::byte_range_info> uring_reactor::align_and_coalesce(
  std::span<cudf::io::text::byte_range_info const> ranges,
  std::optional<std::size_t> alignment) noexcept
{
  try {
    auto const requested = alignment.value_or(IO_BLOCK_SIZE);
    auto const effective = std::max<std::size_t>(requested, IO_BLOCK_SIZE);

    std::vector<cudf::io::text::byte_range_info> aligned;
    aligned.reserve(ranges.size());
    for (auto const& input : ranges) {
      if (input.offset() < 0 || input.size() <= 0) continue;
      auto const offset = static_cast<std::size_t>(input.offset());
      auto const end =
        align_up(saturating_add(offset, static_cast<std::size_t>(input.size())), effective);
      auto const begin = align_down(offset, effective);
      aligned.emplace_back(static_cast<std::int64_t>(begin),
                           static_cast<std::int64_t>(end - begin));
    }

    std::sort(aligned.begin(), aligned.end(), [](auto const& lhs, auto const& rhs) {
      return lhs.offset() < rhs.offset();
    });

    std::vector<cudf::io::text::byte_range_info> merged;
    merged.reserve(aligned.size());
    for (auto const& input : aligned) {
      if (merged.empty()) {
        merged.push_back(input);
        continue;
      }
      auto& previous            = merged.back();
      auto const previous_begin = static_cast<std::size_t>(previous.offset());
      auto const previous_end =
        saturating_add(previous_begin, static_cast<std::size_t>(previous.size()));
      auto const input_begin = static_cast<std::size_t>(input.offset());
      auto const input_end   = saturating_add(input_begin, static_cast<std::size_t>(input.size()));
      if (input_begin <= previous_end) {
        previous = {previous.offset(),
                    static_cast<std::int64_t>(std::max(previous_end, input_end) - previous_begin)};
      } else {
        merged.push_back(input);
      }
    }
    return merged;
  } catch (...) {
    return {};
  }
}

void uring_reactor::worker_loop(std::stop_token const& stop_token)
{
  auto cancel_queued = [&](grouped_coordinator::error_type const& error) noexcept {
    std::unique_ptr<grouped_io_request> request;
    while (_requests.try_dequeue(request)) {
      if (request == nullptr) continue;
      auto const bytes = request->remaining_bytes();
      _queued_bytes.fetch_sub(bytes, std::memory_order_relaxed);
      request->cancel_remaining(error);
    }
  };

  try {
    unique_ring ring{2 * NUM_SLOTS};
    auto const blocks = _bounce_storage->get_blocks();

    std::vector<iovec> registered_buffers;
    registered_buffers.reserve(blocks.size());
    for (auto* block : blocks) {
      registered_buffers.push_back(iovec{block, _bounce_slot_size});
    }
    bool const fixed_supported = ring.register_buffers(registered_buffers);

    slot_pool available_slots{NUM_SLOTS};
    std::vector<io_slot> slots;
    slots.reserve(NUM_SLOTS);
    for (std::size_t index = 0; index < NUM_SLOTS; ++index) {
      slots.emplace_back(static_cast<int>(index), fixed_supported);
    }

    std::array<io_uring_cqe*, NUM_SLOTS> cqes{};
    std::vector<int> incomplete;
    incomplete.reserve(NUM_SLOTS);
    std::vector<int> copying;
    copying.reserve(NUM_SLOTS);
    std::vector<std::unique_ptr<uring_io_op>> pending;
    std::unique_ptr<grouped_io_request> active;
    std::size_t inflight = 0;

    auto reset_slot = [&](io_slot& slot) noexcept { slot.reset(); };

    auto settle_slot_error = [&](io_slot& slot,
                                 grouped_coordinator::error_type const& error,
                                 bool host_data_valid = false) noexcept {
      if (slot.op != nullptr) slot.op->request.finish_error(error, host_data_valid);
      reset_slot(slot);
    };

    auto start_device_copy = [&](io_slot& slot) noexcept {
      auto& copy = *slot.op->request.device_copy;
      int device = copy.device_id >= 0 ? copy.device_id : copy.d_buffer.device_id;
      if (device < 0) {
        auto const status = cudaGetDevice(&device);
        if (status != cudaSuccess) {
          settle_slot_error(slot, status, true);
          return;
        }
      }

      try {
        rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device}};
        if (slot.copy_event == nullptr || slot.event_device != device) {
          slot.copy_event   = std::make_unique<cucascade::cuda::cuda_event>(cudaEventDisableTiming);
          slot.event_device = device;
        }
        auto const status =
          copy.copy_async(slot.op->request.io_rng, slot.op->request.iovecs, slot.copy_event->get());
        if (status != cudaSuccess) {
          settle_slot_error(slot, status, true);
          return;
        }
        slot.state = slot_state::copying;
        copying.push_back(slot.index);
      } catch (...) {
        settle_slot_error(slot, std::current_exception(), true);
      }
    };

    auto finish_host_io = [&](io_slot& slot) noexcept {
      if (slot.op->request.device_copy != nullptr) {
        start_device_copy(slot);
      } else {
        slot.op->request.finish_success();
        reset_slot(slot);
      }
    };

    auto poll_copy_completions = [&]() noexcept {
      auto output = copying.begin();
      for (auto it = copying.begin(); it != copying.end(); ++it) {
        auto& slot        = slots[*it];
        auto const status = cudaEventQuery(slot.copy_event->get());
        if (status == cudaErrorNotReady) {
          *output++ = *it;
          continue;
        }
        if (status == cudaSuccess) {
          slot.op->request.finish_success();
          reset_slot(slot);
        } else {
          settle_slot_error(slot, status, true);
        }
      }
      copying.erase(output, copying.end());
    };

    auto fallback_to_buffered = [](io_slot& slot) noexcept {
      auto const file = std::dynamic_pointer_cast<local_io_object const>(slot.op->request.obj);
      if (file == nullptr) return false;
      slot.op->fd          = file->buffered_handle();
      slot.op->use_odirect = false;
      slot.used_fixed      = false;
      return true;
    };

    auto reap_completions = [&]() {
      auto const count = ring.peek(cqes);
      for (auto* cqe : std::span{cqes.data(), count}) {
        auto const user_data = io_uring_cqe_get_data64(cqe);
        auto const result    = cqe->res;
        ring.seen(cqe);

        // Kernels without IORING_FEAT_EXT_ARG implement liburing's timed wait
        // with an internal timeout SQE.  It is not one of our published read
        // operations and therefore owns no `inflight` credit.
        if (user_data == LIBURING_UDATA_TIMEOUT) continue;

        auto const index = static_cast<int>(user_data);
        if (inflight != 0) --inflight;

        if (index < 0 || static_cast<std::size_t>(index) >= slots.size()) continue;
        auto& slot = slots[index];
        if (slot.op == nullptr || slot.state != slot_state::reading) continue;

        if (result < 0) {
          auto const errc = -result;
          if (slot.op->use_odirect && detail::is_odirect_runtime_error(errc)) {
            if (slot.used_fixed) slot.fixed_supported = false;
            if (!fallback_to_buffered(slot)) {
              settle_slot_error(slot, std::make_error_code(std::errc::bad_file_descriptor));
              continue;
            }
            incomplete.push_back(index);
          } else if (slot.used_fixed && is_fixed_buffer_error(errc)) {
            slot.fixed_supported = false;
            if (slot.op->use_odirect && !fallback_to_buffered(slot)) {
              settle_slot_error(slot, std::make_error_code(std::errc::bad_file_descriptor));
              continue;
            }
            slot.used_fixed = false;
            incomplete.push_back(index);
          } else {
            settle_slot_error(slot, std::error_code{errc, std::generic_category()});
          }
          continue;
        }

        auto const completed = static_cast<std::size_t>(result);
        auto const remaining = slot.op->request.io_rng.size - slot.bytes_read;
        if (completed > remaining) {
          settle_slot_error(slot, std::make_error_code(std::errc::io_error));
          continue;
        }
        slot.bytes_read += completed;

        auto const& io_range = slot.op->request.io_rng;
        auto const available = io_range.offset < slot.op->file_size
                                 ? std::min(io_range.size, slot.op->file_size - io_range.offset)
                                 : std::size_t{0};
        if (slot.bytes_read >= available) {
          finish_host_io(slot);
          continue;
        }
        if (completed == 0) {
          settle_slot_error(slot, std::make_error_code(std::errc::io_error));
          continue;
        }

        if (slot.op->use_odirect) {
          std::vector<iovec> remaining_buffers;
          detail::fill_remaining_iovecs(
            slot.op->request.iovecs, slot.bytes_read, remaining_buffers);
          auto const remaining_range =
            range{io_range.offset + slot.bytes_read, io_range.size - slot.bytes_read};
          if (!detail::is_odirect_compatible(remaining_range, remaining_buffers)) {
            if (!fallback_to_buffered(slot)) {
              settle_slot_error(slot, std::make_error_code(std::errc::bad_file_descriptor));
              continue;
            }
          }
        }
        incomplete.push_back(index);
      }
    };

    auto submit_slots = [&](std::vector<int> const& indexes) {
      if (indexes.empty()) return;
      ring.submit(indexes.size(), inflight);
    };

    auto resubmit_incomplete = [&]() {
      std::vector<int> submitted;
      submitted.reserve(incomplete.size());
      auto input = incomplete.begin();
      while (input != incomplete.end()) {
        auto& slot = slots[*input];
        try {
          slot.prepare_remaining_iovecs();
          auto* sqe = ring.get_sqe();
          if (sqe == nullptr) break;
          slot.prepare_sqe(sqe);
          submitted.push_back(*input);
        } catch (...) {
          settle_slot_error(slot, std::current_exception());
        }
        ++input;
      }
      incomplete.erase(incomplete.begin(), input);
      submit_slots(submitted);
    };

    auto dispatch_pending = [&]() {
      std::vector<int> submitted;
      submitted.reserve(NUM_SLOTS);
      while (!pending.empty()) {
        auto& candidate = pending.back();
        if (!candidate->request.coordinator->should_continue()) {
          candidate->request.finish_error(canceled_error());
          pending.pop_back();
          continue;
        }

        auto const needed = std::max<std::size_t>(1, candidate->staging_blocks);
        if (available_slots.approx_free() < needed) break;

        auto lease = std::make_shared<staging_lease>();
        lease->tokens.reserve(needed);
        for (std::size_t i = 0; i < needed; ++i) {
          auto token = available_slots.try_acquire_token(static_cast<unsigned>(i));
          if (!token) throw std::logic_error("uring_reactor: slot reservation lost");
          lease->tokens.push_back(std::move(token));
        }

        auto op = std::move(candidate);
        pending.pop_back();
        auto const leader = lease->tokens.front().slot_index();

        try {
          if (op->needs_staging()) {
            op->request.iovecs.clear();
            op->request.iovecs.reserve(needed);
            std::size_t remaining = op->request.io_rng.size;
            for (auto const& token : lease->tokens) {
              auto const bytes = std::min(remaining, _bounce_slot_size);
              op->request.iovecs.push_back(iovec{blocks[token.slot_index()], bytes});
              remaining -= bytes;
            }
            if (remaining != 0) {
              throw std::logic_error("uring_reactor: insufficient staging blocks");
            }
          }
          op->request.staging_owner = lease;

          auto& slot = slots[leader];
          assert(slot.state == slot_state::idle && slot.op == nullptr);
          slot.op         = std::move(op);
          slot.state      = slot_state::reading;
          slot.bytes_read = 0;

          slot.prepare_remaining_iovecs();
          auto* sqe = ring.get_sqe();
          if (sqe == nullptr) {
            incomplete.push_back(leader);
            break;
          }
          slot.prepare_sqe(sqe);
          submitted.push_back(leader);
        } catch (...) {
          if (slots[leader].op != nullptr) {
            settle_slot_error(slots[leader], std::current_exception());
          } else if (op != nullptr) {
            op->request.finish_error(std::current_exception());
          }
        }
      }
      submit_slots(submitted);
    };

    auto cancel_pending = [&](grouped_coordinator::error_type const& error) noexcept {
      while (!pending.empty()) {
        pending.back()->request.finish_error(error);
        pending.pop_back();
      }
    };

    auto cancel_active = [&](grouped_coordinator::error_type const& error) noexcept {
      if (active == nullptr) return;
      auto const bytes = active->remaining_bytes();
      _queued_bytes.fetch_sub(bytes, std::memory_order_relaxed);
      active->cancel_remaining(error);
      active.reset();
    };

    std::exception_ptr fatal_error;
    try {
      while (!stop_token.stop_requested()) {
        poll_copy_completions();
        reap_completions();
        resubmit_incomplete();

        if (!pending.empty() && !pending.back()->request.coordinator->should_continue()) {
          cancel_pending(canceled_error());
          cancel_active(canceled_error());
        }

        dispatch_pending();

        if (pending.empty() && active != nullptr) {
          if (!active->coordinator->should_continue()) {
            cancel_active(canceled_error());
          } else if (!active->empty()) {
            auto const backlog =
              std::max(_queued_bytes.load(std::memory_order_relaxed), active->remaining_bytes());
            auto slice = active->take_front();
            _queued_bytes.fetch_sub(slice.size(), std::memory_order_relaxed);

            try {
              auto planned = plan_slice(active->obj,
                                        slice,
                                        active->coordinator,
                                        _config,
                                        _bounce_slot_size,
                                        backlog,
                                        available_slots.approx_free());
              if (planned.empty()) {
                throw std::logic_error("uring_reactor: slice produced no physical operations");
              }

              pending.reserve(pending.size() + planned.size());
              for (auto it = planned.rbegin(); it != planned.rend(); ++it) {
                pending.push_back(std::move(*it));
              }
              active->coordinator->add_tasks(planned.size() - 1);
            } catch (...) {
              if (slice.on_complete != nullptr) {
                (*slice.on_complete)(slice.h_buffer.fragments(), false);
              }
              active->coordinator->report_error(std::current_exception());
              cancel_active(canceled_error());
            }

            if (active != nullptr && active->empty()) active.reset();
            dispatch_pending();
          } else {
            active.reset();
          }
        }

        if (active == nullptr && pending.empty()) {
          std::unique_ptr<grouped_io_request> next;
          if (_requests.try_dequeue(next)) {
            if (next == nullptr) break;
            active = std::move(next);
            continue;
          }
        }

        if (inflight != 0) {
          if (auto const error = ring.wait_for(POLL_INTERVAL); error != 0) {
            throw std::system_error(std::error_code{error, std::generic_category()},
                                    "uring_reactor: io_uring_wait_cqe_timeout");
          }
          reap_completions();
        } else if (!copying.empty() && pending.empty() && active == nullptr) {
          auto& slot        = slots[copying.back()];
          auto const status = slot.copy_event->synchronize_no_throw();
          if (status != cudaSuccess) settle_slot_error(slot, status, true);
          poll_copy_completions();
        } else if (pending.empty() && active == nullptr) {
          std::unique_ptr<grouped_io_request> next;
          if (!_requests.wait_dequeue_timed(next, POLL_INTERVAL_US)) continue;
          if (next == nullptr) break;
          active = std::move(next);
        }
      }
    } catch (...) {
      fatal_error = std::current_exception();
    }

    // Close admission before draining the queue.  Taking the same mutex as
    // enqueue() ensures every request that observed _accepting=true has
    // finished publishing its queue entry before cancel_queued() runs.
    {
      std::lock_guard lock(_enqueue_mutex);
      _accepting.store(false, std::memory_order_release);
    }

    auto const cancellation = canceled_error();
    grouped_coordinator::error_type terminal_error =
      fatal_error != nullptr ? grouped_coordinator::error_type{fatal_error}
                             : grouped_coordinator::error_type{cancellation};
    cancel_pending(terminal_error);
    cancel_active(terminal_error);
    cancel_queued(terminal_error);

    bool sync_cancel_available = true;
    auto sync_cancel_inflight  = [&]() noexcept {
      if (inflight == 0) return true;
      if (!sync_cancel_available) return false;
      auto const rc = ring.cancel_all_sync();
      if (rc >= 0 || rc == -ENOENT) {
        inflight = 0;
        return true;
      }
      sync_cancel_available = false;
      SIRIUS_LOG_WARN("uring_reactor: synchronous cancel-all failed: {}", strerror(-rc));
      return false;
    };

    if (fatal_error != nullptr) sync_cancel_inflight();

    if (fatal_error == nullptr) {
      try {
        while (inflight != 0 || !incomplete.empty()) {
          resubmit_incomplete();
          if (inflight == 0) break;
          if (auto const error = ring.wait_for(POLL_INTERVAL); error != 0) {
            throw std::system_error(std::error_code{error, std::generic_category()},
                                    "uring_reactor: drain failed");
          }
          reap_completions();
        }
      } catch (...) {
        fatal_error    = std::current_exception();
        terminal_error = fatal_error;
        sync_cancel_inflight();
      }
    }

    // A fatal submission can leave additional prepared SQEs in the userspace
    // ring. Keep their operation storage parked, and use only zero-submit
    // GETEVENTS enters until every operation already visible to the kernel is
    // quiescent. A timed wait or submit-and-wait here could publish those
    // untracked entries and make releasing their buffers unsafe.
    bool terminal_enter_error_logged = false;
    while (inflight != 0) {
      auto const before = inflight;
      try {
        reap_completions();
      } catch (...) {
        fatal_error    = std::current_exception();
        terminal_error = fatal_error;
      }
      if (inflight == 0 || sync_cancel_inflight()) break;

      // In addition to flushing CQ overflow, this enter is required to run
      // deferred task work when the preferred DEFER_TASKRUN ring is active.
      if (auto const enter_error = ring.run_deferred_taskwork(); enter_error != 0) {
        if (!terminal_enter_error_logged) {
          SIRIUS_LOG_WARN("uring_reactor: terminal GETEVENTS failed: {}", strerror(enter_error));
          terminal_enter_error_logged = true;
        }
        std::this_thread::sleep_for(POLL_INTERVAL);
        continue;
      }
      try {
        reap_completions();
      } catch (...) {
        fatal_error    = std::current_exception();
        terminal_error = fatal_error;
      }
      if (inflight == before) std::this_thread::sleep_for(POLL_INTERVAL);
    }

    for (auto const index : copying) {
      auto& slot        = slots[index];
      auto const status = slot.copy_event->synchronize_no_throw();
      if (status == cudaSuccess) {
        slot.op->request.finish_success();
        reset_slot(slot);
      } else {
        settle_slot_error(slot, status, true);
      }
    }
    copying.clear();

    for (auto& slot : slots) {
      if (slot.op != nullptr) settle_slot_error(slot, terminal_error);
    }
  } catch (...) {
    auto const error = std::current_exception();
    {
      std::lock_guard lock(_enqueue_mutex);
      _accepting.store(false, std::memory_order_release);
    }
    cancel_queued(error);
  }
}

}  // namespace sirius::io::uring
