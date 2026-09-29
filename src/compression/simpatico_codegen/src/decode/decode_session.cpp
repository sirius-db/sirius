// SPDX-License-Identifier: Apache-2.0
#include "decode_session.hpp"

#include "util/host_observation.hpp"

#include <cudf/utilities/traits.hpp>

#include <algorithm>
#include <cstdio>
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

//===----------------------------------------------------------------------===//
// decode_frame
//===----------------------------------------------------------------------===//
decode_frame::decode_frame(::cuda::stream_ref stream, rmm::device_async_resource_ref mr) noexcept
  : stream_(stream), mr_(mr)
{
}

void decode_frame::keep_kernel(std::shared_ptr<codegen::jit::CompiledKernel const> kernel)
{
  if (std::find(kernels_.begin(), kernels_.end(), kernel) == kernels_.end()) {
    kernels_.push_back(std::move(kernel));
  }
}

std::span<std::byte> decode_frame::host_bytes(std::size_t bytes)
{
  // Byte-array allocation supplies fundamental alignment and implicit-lifetime storage for T.
  auto const& upload = uploads_.emplace_back(std::make_unique_for_overwrite<std::byte[]>(bytes));
  return {upload.get(), bytes};
}

void decode_frame::read_bytes(void* destination, void const* source, std::size_t bytes)
{
  read_device_bytes_completed(destination, source, bytes, stream_);
}

//===----------------------------------------------------------------------===//
// decode_session::impl
//===----------------------------------------------------------------------===//
struct decode_session::impl {
  /// OPEN -> FINISHED or FAILED; no further append() or finish() calls allowed
  enum class phase { OPEN, FAILED, FINISHED };
  struct result_slot {
    std::unique_ptr<cudf::column> column;  ///< The output column for this request
    std::optional<cudf::data_type> stored_type;
  };
  // The request copy keeps predicate strings, descriptors, and probe captures alive for queued
  // work.
  struct submitted_request {
    decode_frame frame;
    std::variant<std::monostate, column_decode_request, mask_decode_request> request;
    template <typename Request>
    submitted_request(::cuda::stream_ref stream,
                      rmm::device_async_resource_ref mr,
                      Request const& value)
      : frame(stream, mr), request(std::in_place_type<Request>, value)
    {
    }
  };
  static_assert(!std::is_copy_constructible_v<submitted_request> &&
                !std::is_move_constructible_v<submitted_request>);

  std::vector<::cuda::stream_ref> streams;
  rmm::device_async_resource_ref mr;
  std::thread::id thread = std::this_thread::get_id();
  // List nodes keep each frame at a stable address while its decoder or a test refers to it.
  std::list<submitted_request> requests;
  std::vector<result_slot> results;
  phase state           = phase::OPEN;
  std::size_t next_lane = 0;
  bool submitted        = false;
  bool drained          = false;

  impl(std::span<const ::cuda::stream_ref> supplied, rmm::device_async_resource_ref resource)
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
      if (streams[earlier].get() == streams[lane].get()) return false;
    return true;
  }

  /** Copy the request into a new frame on the next stream in rotation. */
  template <typename Request>
  submitted_request& submit(Request const& request)
  {
    auto const lane = next_lane++ % streams.size();
    auto& item      = requests.emplace_back(streams[lane], mr, request);
    submitted       = true;
    drained         = false;
    return item;
  }

  cudaError_t drain() noexcept
  {
    cudaError_t first = cudaSuccess;
    if (!submitted || drained) return first;
    // Every distinct stream is observed now: earlier host observations do not cover later
    // submissions, stream-ordered frees, or external phase work.
    for (std::size_t lane = 0; lane < streams.size(); ++lane) {
      if (!first_stream_handle(lane)) continue;
      auto const stream = streams[lane];
      auto status       = cudaStreamQuery(stream.get());
      if (status != cudaSuccess) {
        if (status != cudaErrorNotReady && first == cudaSuccess) first = status;
        status = cudaStreamSynchronize(stream.get());
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
};

//===----------------------------------------------------------------------===//
// decode_session
//===----------------------------------------------------------------------===//
decode_session::decode_session(std::span<const ::cuda::stream_ref> streams,
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

decode_frame& decode_session::register_test_frame()
{
  state_->check_open();
  return state_->submit(std::monostate{}).frame;
}

std::size_t decode_session::retained_host_uploads() const noexcept
{
  std::size_t uploads = 0;
  for (auto const& item : state_->requests)
    uploads += item.frame.uploads_.size();
  return uploads;
}

void decode_session::append(column_decode_request const& request)
{
  state_->check_open();
  try {
    auto& result = state_->results.emplace_back();
    if (auto const* values = std::get_if<value_result>(&request.result))
      result.stored_type = values->stored_type;
    auto& item    = state_->submit(request);
    result.column = decode_request(std::get<column_decode_request>(item.request), item.frame);
    if (!result.column) throw std::runtime_error("decode produced no output column");
  } catch (...) {
    state_->abort();
    throw;
  }
}

mask_source_status decode_session::append(mask_decode_request const& request)
{
  state_->check_open();
  try {
    auto& item = state_->submit(request);
    return decode_request(std::get<mask_decode_request>(item.request), item.frame);
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
    state_->requests.clear();
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

}  // namespace simpatico
