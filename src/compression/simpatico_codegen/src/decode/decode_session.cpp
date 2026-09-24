// SPDX-License-Identifier: Apache-2.0
#include "decode_session.hpp"

#include <cudf/utilities/traits.hpp>

#include <algorithm>
#include <cstdio>
#include <exception>
#include <list>
#include <stdexcept>
#include <thread>
#include <utility>

namespace simpatico {
namespace {

void check_cuda(cudaError_t status)
{
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

// Provisional retained-temporary estimate limits, separate from input/output and MR peaks.
struct retirement_window {
  static constexpr std::size_t device_bytes = 64 * 1024 * 1024;
  static constexpr std::size_t host_bytes   = 8 * 1024 * 1024;
  static constexpr std::size_t frames       = 64;
};

std::unique_ptr<cudf::column> restore_type(std::unique_ptr<cudf::column> column,
                                           cudf::data_type stored)
{
  if (column->type() == stored || !cudf::is_fixed_width(column->type()) ||
      !cudf::is_fixed_width(stored) || cudf::size_of(column->type()) != cudf::size_of(stored)) {
    return column;
  }
  auto const rows  = column->size();
  auto const nulls = column->null_count();
  auto contents    = column->release();
  auto mask        = contents.null_mask ? std::move(*contents.null_mask) : rmm::device_buffer{};
  return std::make_unique<cudf::column>(
    stored, rows, std::move(*contents.data), std::move(mask), nulls, std::move(contents.children));
}

}  // namespace

void decode_column_slot::adopt(std::unique_ptr<cudf::column> column) const noexcept
{
  if (!owner_ || *owner_) std::terminate();
  *owner_ = std::move(column);
}

cudf::column& decode_column_slot::get() const
{
  if (!*this) throw std::logic_error("decode column slot is empty");
  return **owner_;
}

decode_frame::decode_frame(rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr,
                           decode_session& session)
  : stream_(stream), mr_(mr), session_(session)
{
}

decode_frame::~decode_frame() = default;

decode_column_slot decode_frame::make_column()
{
  session_.allocation_checkpoint(0, 0);
  columns_.emplace_back();
  return decode_column_slot{columns_.back()};
}

decode_column_slot decode_frame::keep_column(std::unique_ptr<cudf::column> column)
{
  try {
    auto slot = make_column();
    slot.adopt(std::move(column));
    session_.allocation_checkpoint(0, 0);
    return slot;
  } catch (...) {
    stream_.synchronize_no_throw();
    throw;
  }
}

decode_column_slot decode_frame::memo_column(std::uint64_t key)
{
  session_.allocation_checkpoint(0, 0);
  return decode_column_slot{memo_.try_emplace(key).first->second};
}

cudf::column* decode_frame::find_memo(std::uint64_t key) const
{
  auto it = memo_.find(key);
  return it == memo_.end() ? nullptr : it->second.get();
}

bool decode_frame::contains_memo(std::uint64_t key) const { return memo_.contains(key); }

std::unique_ptr<cudf::column> decode_frame::release(decode_column_slot slot) noexcept
{
  return std::move(*slot.owner_);
}

std::unique_ptr<cudf::column> decode_frame::release_memo(std::uint64_t key)
{
  return std::move(memo_.at(key));
}

rmm::device_buffer& decode_frame::allocate_buffer(std::size_t bytes)
{
  session_.allocation_checkpoint(bytes, 0);
  buffers_.emplace_back(bytes, stream_, mr_);
  return buffers_.back();
}

rmm::device_buffer& decode_frame::allocate_output_buffer(std::size_t bytes)
{
  session_.allocation_checkpoint(0, 0);
  output_buffers_.emplace_back(bytes, stream_, mr_);
  return output_buffers_.back();
}

rmm::device_buffer& decode_frame::keep_buffer(rmm::device_buffer buffer)
{
  try {
    session_.allocation_checkpoint(buffer.capacity(), 0);
    buffers_.push_back(std::move(buffer));
    return buffers_.back();
  } catch (...) {
    stream_.synchronize_no_throw();
    throw;
  }
}

compressed_representation& decode_frame::keep_representation(
  std::unique_ptr<compressed_representation> rep)
{
  try {
    representations_.push_back(std::move(rep));
    session_.allocation_checkpoint(0, 0);
    return *representations_.back();
  } catch (...) {
    stream_.synchronize_no_throw();
    throw;
  }
}

cudf::scalar& decode_frame::keep_scalar(std::unique_ptr<cudf::scalar> scalar,
                                        std::size_t device_bytes)
{
  try {
    scalars_.push_back(std::move(scalar));
    scalar_bytes_ += device_bytes;
    session_.allocation_checkpoint(0, 0);
    return *scalars_.back();
  } catch (...) {
    stream_.synchronize_no_throw();
    throw;
  }
}

void decode_frame::keep_kernel(std::shared_ptr<codegen::jit::CompiledKernel const> kernel)
{
  session_.keep_kernel(std::move(kernel));
}

std::span<std::byte> decode_frame::host_bytes(std::size_t bytes)
{
  session_.allocation_checkpoint(0, bytes);
  // Byte-array allocation supplies fundamental alignment and implicit-lifetime storage for T.
  uploads_.push_back({std::make_unique_for_overwrite<std::byte[]>(bytes), bytes});
  return {uploads_.back().bytes.get(), bytes};
}

void decode_frame::read_bytes(void* destination, void const* source, std::size_t bytes)
{
  try {
    check_cuda(
      cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToHost, stream_.value()));
    // Host observation: the caller needs these bytes to establish a shape or branch.
    stream_.synchronize();
  } catch (...) {
    stream_.synchronize_no_throw();
    throw;
  }
}

std::size_t decode_frame::retained_device_bytes() const
{
  std::size_t bytes = scalar_bytes_;
  for (auto const& buffer : buffers_)
    bytes += buffer.capacity();
  for (auto const& column : columns_)
    if (column) bytes += column->alloc_size();
  for (auto const& [key, column] : memo_)
    if (column && (!terminal_memo_ || key != *terminal_memo_)) bytes += column->alloc_size();
  for (auto const& rep : representations_)
    bytes += rep->owned_device_bytes_estimate();
  return bytes;
}

std::size_t decode_frame::retained_host_bytes() const noexcept
{
  std::size_t bytes = 0;
  for (auto const& block : uploads_)
    bytes += block.capacity;
  return bytes;
}

//===----------decode_session::impl----------===//
struct decode_session::impl {
  enum class phase { OPEN, FAILED, FINISHED };
  struct result_slot {
    std::unique_ptr<cudf::column> column;
    std::optional<cudf::data_type> stored_type;
  };
  struct retained_totals {
    std::size_t device      = 0;
    std::size_t host        = 0;
    std::size_t frame_count = 0;
  };
  struct submitted_frame {
    decode_frame frame;
    std::variant<std::monostate, column_decode_request, mask_decode_request> request;
    std::size_t lane            = 0;
    bool sealed                 = false;
    std::size_t retained_device = 0;
    std::size_t retained_host   = 0;
    submitted_frame(rmm::cuda_stream_view stream,
                    rmm::device_async_resource_ref mr,
                    decode_session& session,
                    std::variant<std::monostate, column_decode_request, mask_decode_request> value,
                    std::size_t assigned_lane)
      : frame(stream, mr, session), request(std::move(value)), lane(assigned_lane)
    {
    }
    void seal()
    {
      // Output ownership has already left the frame; sealing fixes ownership, not completion.
      retained_device = frame.retained_device_bytes();
      retained_host   = frame.retained_host_bytes();
      sealed          = true;
    }
  };
  static_assert(!std::is_copy_constructible_v<submitted_frame> &&
                !std::is_move_constructible_v<submitted_frame>);

  std::vector<rmm::cuda_stream_view> streams;
  rmm::device_async_resource_ref mr;
  std::thread::id thread = std::this_thread::get_id();
  std::list<submitted_frame> frames;
  std::vector<result_slot> results;
  // Dropping the final module handle can synchronize the CUDA context. Keep one pin per
  // loaded kernel until the final stream drain, not merely its frame's earlier retirement.
  std::vector<std::shared_ptr<codegen::jit::CompiledKernel const>> kernels;
  decode_session_stats stats;
  phase state           = phase::OPEN;
  std::size_t next_lane = 0;
  bool submitted        = false;
  bool drained          = false;

  impl(std::span<rmm::cuda_stream_view const> supplied, rmm::device_async_resource_ref resource)
    : streams(supplied.begin(), supplied.end()), mr(resource)
  {
    if (streams.empty()) throw std::invalid_argument("decode_session requires a stream");
  }

  void check_open() const
  {
    if (std::this_thread::get_id() != thread)
      throw std::logic_error("decode_session thread changed");
    if (state != phase::OPEN) throw std::logic_error("decode_session is no longer open");
  }

  [[nodiscard]] bool first_stream_handle(std::size_t lane) const noexcept
  {
    for (std::size_t earlier = 0; earlier < lane; ++earlier)
      if (streams[earlier].value() == streams[lane].value()) return false;
    return true;
  }

  cudaError_t drain() noexcept
  {
    cudaError_t first = cudaSuccess;
    if (!submitted || drained) return first;
    // Earlier retirement does not complete later submissions, frees, or external phase work.
    for (std::size_t lane = 0; lane < streams.size(); ++lane) {
      if (!first_stream_handle(lane)) continue;
      auto const stream = streams[lane];
      auto status       = cudaStreamQuery(stream.value());
      if (status != cudaSuccess) {
        if (status != cudaErrorNotReady && first == cudaSuccess) first = status;
        status = cudaStreamSynchronize(stream.value());
        if (status != cudaSuccess && first == cudaSuccess) first = status;
      }
    }
    drained = first == cudaSuccess;
    return first;
  }

  void abort() noexcept
  {
    state             = phase::FAILED;
    auto const status = drain();
    if (status != cudaSuccess) {
      std::fprintf(stderr, "simpatico decode cleanup failed: %s\n", cudaGetErrorString(status));
    }
  }

  [[nodiscard]] retained_totals retained_snapshot() const
  {
    retained_totals totals;
    totals.frame_count = frames.size();
    for (auto const& item : frames) {
      totals.device += item.sealed ? item.retained_device : item.frame.retained_device_bytes();
      totals.host += item.sealed ? item.retained_host : item.frame.retained_host_bytes();
    }
    return totals;
  }

  void update_stats(retained_totals const& totals)
  {
    stats.peak_frames                = std::max(stats.peak_frames, totals.frame_count);
    stats.peak_retained_device_bytes = std::max(stats.peak_retained_device_bytes, totals.device);
    stats.peak_retained_host_bytes   = std::max(stats.peak_retained_host_bytes, totals.host);
  }

  void retire_sealed(cudaStream_t stream)
  {
    std::erase_if(frames, [&](auto const& item) {
      return item.sealed && streams[item.lane].value() == stream;
    });
  }

  bool retire_ready()
  {
    auto const previous_size = frames.size();
    for (std::size_t lane = 0; lane < streams.size(); ++lane) {
      if (!first_stream_handle(lane)) continue;
      auto const stream     = streams[lane].value();
      auto const has_sealed = std::any_of(frames.begin(), frames.end(), [&](auto const& item) {
        return item.sealed && streams[item.lane].value() == stream;
      });
      if (!has_sealed) continue;
      ++stats.retirement_stream_queries;
      auto const status = cudaStreamQuery(stream);
      if (status == cudaErrorNotReady) continue;
      check_cuda(status);
      retire_sealed(stream);
    }
    return frames.size() != previous_size;
  }

  void pressure(std::size_t extra_device, std::size_t extra_host, std::size_t extra_frames)
  {
    auto totals = retained_snapshot();
    update_stats(totals);
    auto exceeds = [&] {
      return totals.device + extra_device > retirement_window::device_bytes ||
             totals.host + extra_host > retirement_window::host_bytes ||
             totals.frame_count + extra_frames > retirement_window::frames;
    };
    if (!exceeds()) return;
    if (retire_ready()) totals = retained_snapshot();
    while (exceeds()) {
      auto oldest =
        std::find_if(frames.begin(), frames.end(), [](auto const& item) { return item.sealed; });
      if (oldest == frames.end())
        break;  // One intrinsic active-frame working set may exceed limits.
      ++stats.pressure_waits;
      auto const stream = streams[oldest->lane].value();
      check_cuda(cudaStreamSynchronize(stream));
      retire_sealed(stream);
      totals = retained_snapshot();
      if (exceeds() && retire_ready()) totals = retained_snapshot();
    }
  }
};

//===----------decode_session----------===//
decode_session::decode_session(std::span<rmm::cuda_stream_view const> streams,
                               rmm::device_async_resource_ref mr)
  : state_(std::make_unique<impl>(streams, mr))
{
}

decode_session::~decode_session() noexcept
{
  auto const status = state_->drain();
  if (status != cudaSuccess) {
    std::fprintf(stderr, "simpatico decode destruction failed: %s\n", cudaGetErrorString(status));
  }
}

void decode_session::allocation_checkpoint(std::size_t device_bytes, std::size_t host_bytes)
{
  state_->pressure(device_bytes, host_bytes, 0);
}

void decode_session::keep_kernel(std::shared_ptr<codegen::jit::CompiledKernel const> kernel)
{
  if (std::find(state_->kernels.begin(), state_->kernels.end(), kernel) == state_->kernels.end()) {
    state_->kernels.push_back(std::move(kernel));
  }
}

decode_frame& decode_session::register_test_frame()
{
  state_->check_open();
  state_->pressure(0, 0, 1);
  auto const lane = state_->next_lane++ % state_->streams.size();
  auto stream     = state_->streams[lane];
  state_->frames.emplace_back(stream, state_->mr, *this, std::monostate{}, lane);
  state_->submitted = true;
  state_->drained   = false;
  return state_->frames.back().frame;
}

void decode_session::append(column_decode_request const& request)
{
  state_->check_open();
  try {
    state_->pressure(0, 0, 1);
    auto const slot = state_->results.size();
    state_->results.emplace_back();
    auto& result = state_->results.back();
    if (auto const* values = std::get_if<value_result>(&request.result))
      result.stored_type = values->stored_type;
    auto const lane = state_->next_lane++ % state_->streams.size();
    auto stream     = state_->streams[lane];
    state_->frames.emplace_back(stream, state_->mr, *this, request, lane);
    auto& item        = state_->frames.back();
    state_->submitted = true;
    state_->drained   = false;
    decode_request(std::get<column_decode_request>(item.request), item.frame);
    if (!item.frame.output()) throw std::runtime_error("decode produced no output column");
    state_->results[slot].column = item.frame.release(item.frame.output());
    item.seal();
    state_->update_stats(state_->retained_snapshot());
  } catch (...) {
    state_->abort();
    throw;
  }
}

mask_source_status decode_session::append(mask_decode_request const& request)
{
  state_->check_open();
  try {
    state_->pressure(0, 0, 1);
    auto const lane = state_->next_lane++ % state_->streams.size();
    auto stream     = state_->streams[lane];
    state_->frames.emplace_back(stream, state_->mr, *this, request, lane);
    auto& item        = state_->frames.back();
    state_->submitted = true;
    state_->drained   = false;
    auto result       = decode_request(std::get<mask_decode_request>(item.request), item.frame);
    item.seal();
    state_->update_stats(state_->retained_snapshot());
    return result;
  } catch (...) {
    state_->abort();
    throw;
  }
}

std::vector<std::unique_ptr<cudf::column>> decode_session::finish()
{
  state_->check_open();
  try {
    check_cuda(state_->drain());
    state_->frames.clear();
    state_->kernels.clear();
    std::vector<std::unique_ptr<cudf::column>> outputs;
    outputs.reserve(state_->results.size());
    for (auto& result : state_->results) {
      if (!result.column) throw std::runtime_error("decode result is missing");
      if (result.stored_type)
        result.column = restore_type(std::move(result.column), *result.stored_type);
      outputs.push_back(std::move(result.column));
    }
    state_->state = impl::phase::FINISHED;
    return outputs;
  } catch (...) {
    state_->abort();
    throw;
  }
}

decode_session_stats const& decode_session::stats() const noexcept { return state_->stats; }

}  // namespace simpatico
