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

#include "io/s3/s3_rdma_ioctx.hpp"

#include "io/uri_parser.hpp"

#include <stdexcept>
#include <string>

namespace sirius::io::s3 {

namespace {

std::runtime_error not_implemented(std::string_view entry_point)
{
  return std::runtime_error("s3_rdma_ioctx::" + std::string(entry_point) +
                            ": the S3 RDMA transport is not implemented yet");
}

}  // namespace

s3_rdma_ioctx::s3_rdma_ioctx(object_store_config cfg) : _config(std::move(cfg)) {}

s3_rdma_ioctx::~s3_rdma_ioctx() { pre_destroy(); }

bool s3_rdma_ioctx::supports(std::string_view path) const noexcept
{
  try {
    return parse(path).scheme == "s3";
  } catch (...) {
    return false;
  }
}

std::vector<cudf::io::text::byte_range_info> s3_rdma_ioctx::align_and_coalesce(
  std::span<const cudf::io::text::byte_range_info> ranges,
  std::optional<size_t> /*alignment*/) const noexcept
{
  return {ranges.begin(), ranges.end()};
}

size_t s3_rdma_ioctx::host_read_io(const sirius_io_object& /*obj*/,
                                   size_t /*offset*/,
                                   size_t /*size*/,
                                   uint8_t* /*dst*/)
{
  throw not_implemented("host_read_io");
}

exec::semi_future<size_t> s3_rdma_ioctx::host_read_async_io(const sirius_io_object& /*obj*/,
                                                            size_t /*offset*/,
                                                            size_t /*size*/,
                                                            uint8_t* /*dst*/) noexcept
{
  return exec::make_semi_future<size_t>(
    std::make_exception_ptr(not_implemented("host_read_async_io")));
}

exec::semi_future<size_t> s3_rdma_ioctx::device_read_async_io(
  const sirius_io_object& /*obj*/,
  size_t /*offset*/,
  size_t /*size*/,
  uint8_t* /*dst*/,
  rmm::cuda_stream_view /*stream*/) noexcept
{
  return exec::make_semi_future<size_t>(
    std::make_exception_ptr(not_implemented("device_read_async_io")));
}

exec::semi_future<size_t> s3_rdma_ioctx::host_to_device_read_async_io(
  const sirius_io_object& /*obj*/,
  std::span<io_object_segment> /*slices*/,
  size_t /*offset*/,
  size_t /*size*/,
  uint8_t* /*device_dst*/,
  rmm::cuda_stream_view /*stream*/) noexcept
{
  return exec::make_semi_future<size_t>(
    std::make_exception_ptr(not_implemented("host_to_device_read_async_io")));
}

exec::semi_future<size_t> s3_rdma_ioctx::host_read_ranges_async_io(
  const sirius_io_object& /*obj*/, std::span<io_object_segment> /*segments*/) noexcept
{
  return exec::make_semi_future<size_t>(
    std::make_exception_ptr(not_implemented("host_read_ranges_async_io")));
}

std::shared_ptr<sirius_io_object> s3_rdma_ioctx::create_io_object(std::string /*path*/)
{
  throw not_implemented("create_io_object");
}

}  // namespace sirius::io::s3
