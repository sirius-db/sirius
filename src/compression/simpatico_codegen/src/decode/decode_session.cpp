// SPDX-License-Identifier: Apache-2.0
#include "decode_session.hpp"

#include "codegen/util/cuda_check.hpp"
#include "codegen/util/stream_pool.hpp"
#include "util/host_observation.hpp"

#include <cudf/utilities/traits.hpp>

#include <algorithm>
#include <cstdio>
#include <functional>
#include <list>
#include <stdexcept>
#include <thread>
#include <utility>

namespace simpatico {
namespace {

// Retag a decoded column with its stored logical type when only the interpretation of identical
// bits differs, for example the INT64 storage of a DECIMAL64 column. No bytes change.
std::unique_ptr<cudf::column> restore_type(std::unique_ptr<cudf::column> column,
                                           cudf::data_type stored)
{
  if (column->type() == stored || !cudf::is_bit_castable(column->type(), stored)) return column;
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
  // The request copy keeps predicate strings, descriptors, and probe captures alive for queued
  // work. A test frame carries no request.
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
  std::vector<std::unique_ptr<cudf::column>> results;
  std::size_t next_lane = 0;
  bool open             = true;   ///< false once finish() succeeded or any call failed
  bool pending          = false;  ///< work was submitted since the last successful drain

  impl(std::span<const ::cuda::stream_ref> supplied, rmm::device_async_resource_ref resource)
    : streams(supplied.begin(), supplied.end()), mr(resource)
  {
    if (streams.empty()) throw std::invalid_argument("decode_session requires a stream");
  }

  void check_open() const
  {
    if (std::this_thread::get_id() != thread)
      throw std::logic_error("decode_session thread changed");
    if (!open) throw std::logic_error("decode_session is no longer open");
  }

  /** Copy the request into a new frame on the next stream in rotation. */
  template <typename Request>
  submitted_request& submit(Request const& request)
  {
    auto const lane = next_lane++ % streams.size();
    auto& item      = requests.emplace_back(streams[lane], mr, request);
    pending         = true;
    return item;
  }

  // Every distinct stream is observed now: earlier host observations do not cover later
  // submissions, stream-ordered frees, or external phase work.
  cudaError_t drain() noexcept
  {
    if (!pending) return cudaSuccess;
    auto const status = synchronize_distinct(streams);
    pending           = status != cudaSuccess;
    return status;
  }

  /**
   * Submit one request and pass its result to `publish`, closing the session and draining before
   * any failure propagates.
   */
  template <typename Request, typename Publish>
  auto append(Request const& request, Publish publish)
  {
    check_open();
    try {
      auto& item = submit(request);
      return publish(decode_request(std::get<Request>(item.request), item.frame));
    } catch (...) {
      open = false;
      if (auto const status = drain(); status != cudaSuccess) {
        std::fprintf(stderr, "simpatico decode cleanup failed: %s\n", cudaGetErrorString(status));
      }
      throw;
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
  if (auto const status = state_->drain(); status != cudaSuccess) {
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
  state_->append(request, [&](std::unique_ptr<cudf::column> column) {
    if (!column) throw std::runtime_error("decode produced no output column");
    if (auto const* values = std::get_if<value_result>(&request.result);
        values && values->stored_type) {
      column = restore_type(std::move(column), *values->stored_type);
    }
    state_->results.push_back(std::move(column));
  });
}

mask_source_status decode_session::append(mask_decode_request const& request)
{
  return state_->append(request, std::identity{});
}

std::vector<std::unique_ptr<cudf::column>> decode_session::finish()
{
  state_->check_open();
  state_->open = false;
  throw_if_cuda_error(state_->drain(), "simpatico decode completion");
  state_->requests.clear();
  return std::move(state_->results);
}

std::unique_ptr<cudf::column> decode_one(column_decode_request const& request,
                                         ::cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr)
{
  decode_session session{{&stream, 1}, mr};
  session.append(request);
  return std::move(session.finish().front());
}

mask_source_status decode_one(mask_decode_request const& request,
                              ::cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr)
{
  decode_session session{{&stream, 1}, mr};
  auto const status = session.append(request);
  session.finish();
  return status;
}

}  // namespace simpatico
