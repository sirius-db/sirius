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

#include "compression/compressed_payload_sources.hpp"

#include "compression/device_compressed_blob.hpp"
#include "compression/spill_io.hpp"
#include "log/logging.hpp"

#include <cudf/utilities/default_stream.hpp>

#include <cuda_runtime.h>

#include <api/compressed_table_io.hpp>
#include <cucascade/data/disk_file_format.hpp>
#include <cucascade/error.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sirius::compression {

namespace {

/// A share of @p representation's backing store (blob or file), which keeps the payload alive
/// for as long as a source reads from it.
template <typename Representation>
std::unique_ptr<cucascade::idata_representation> share(const Representation& representation)
{
  // Compressed representations clone by sharing; none of them uses the stream.
  return representation.clone(cudf::get_default_stream());
}

std::shared_ptr<const simpatico::hpln_description> parse_header(
  std::span<const std::uint8_t> header)
{
  return std::make_shared<const simpatico::hpln_description>(
    simpatico::describe_hpln_header(std::as_bytes(header)));
}

// ── Device ───────────────────────────────────────────────────────────────────

/// Where a device-resident batch's payload buffers are.
///
/// A device batch has no stored header and its buffers are not laid out at the header's dense
/// offsets (the pin path aligns each one; a freshly encoded table owns one allocation per
/// buffer). The header is therefore rebuilt from the cached table, and a buffer's payload
/// offset is its distance from the lowest buffer address, so that base + offset is the buffer
/// itself and nothing is copied.
struct device_payload_layout {
  std::vector<std::uint8_t> header;
  /// Dense header offset -> device address, for every non-empty buffer.
  std::unordered_map<std::uint64_t, std::uintptr_t> address_of;
  std::uintptr_t base   = 0;
  std::uint64_t extent  = 0;  ///< bytes from base to the end of the last buffer
  std::uint64_t buffers = 0;
};

device_payload_layout device_layout(const compressed_device_representation& representation,
                                    ::cuda::stream_ref stream)
{
  if (!representation.has_table()) {
    throw std::invalid_argument(
      "[compressed_payload_sources] the compressed device batch has no table");
  }
  device_payload_layout layout;
  std::vector<simpatico::payload_buffer_ref> buffers;
  std::uint64_t dense_bytes = 0;
  if (auto error = simpatico::build_compressed_table_header(
        representation.table(), layout.header, buffers, dense_bytes, stream);
      !error.empty()) {
    throw std::runtime_error("[compressed_payload_sources] " + error);
  }

  std::uintptr_t lowest = UINTPTR_MAX;
  std::uintptr_t end    = 0;
  for (auto const& buffer : buffers) {
    if (buffer.size_bytes == 0) { continue; }
    if (buffer.device_ptr == nullptr) {
      throw std::runtime_error(
        "[compressed_payload_sources] a compressed device buffer has no device memory");
    }
    auto const address = reinterpret_cast<std::uintptr_t>(buffer.device_ptr);
    layout.address_of.emplace(buffer.offset, address);
    lowest = std::min(lowest, address);
    end    = std::max(end, address + static_cast<std::uintptr_t>(buffer.size_bytes));
  }
  if (!layout.address_of.empty()) {
    layout.base   = lowest;
    layout.extent = end - lowest;
  }
  return layout;
}

class device_payload_source final : public simpatico::payload_source {
 public:
  device_payload_source(const compressed_device_representation& representation,
                        const device_payload_layout& layout)
    : _owner(share(representation)), _base(layout.base)
  {
  }

  void read_to_device(std::uint64_t offset,
                      std::uint64_t size,
                      void* dst,
                      rmm::cuda_stream_view stream) override
  {
    if (size == 0) { return; }
    CUCASCADE_CUDA_TRY(cudaMemcpyAsync(dst,
                                       reinterpret_cast<const void*>(_base + offset),
                                       static_cast<std::size_t>(size),
                                       cudaMemcpyDeviceToDevice,
                                       stream.value()));
  }

  [[nodiscard]] const std::byte* device_base() const noexcept override
  {
    return reinterpret_cast<const std::byte*>(_base);
  }

 private:
  std::unique_ptr<cucascade::idata_representation> _owner;
  std::uintptr_t _base;
};

/// The header of @p layout, with every payload offset relative to the layout's base.
std::shared_ptr<const simpatico::hpln_description> describe_device(
  const device_payload_layout& layout)
{
  auto description =
    simpatico::describe_hpln_header(std::as_bytes(std::span<const std::uint8_t>{layout.header}));

  // Replace each dense offset by the buffer's offset from the device base.
  auto const relocate = [&layout](std::uint64_t size, std::uint64_t& payload_offset) {
    if (size == 0) {
      payload_offset = 0;
      return;
    }
    auto const it = layout.address_of.find(payload_offset);
    if (it == layout.address_of.end()) {
      throw std::runtime_error(
        "[compressed_payload_sources] the device batch's header names a buffer its table lacks");
    }
    payload_offset = it->second - layout.base;
  };
  for (auto& column : description.columns) {
    if (column.validity == simpatico::hpln_validity::mask) {
      relocate(column.validity_size, column.validity_payload_offset);
    }
    for (auto& leaf : column.leaves) {
      for (auto& buffer : leaf.buffers) {
        relocate(buffer.size, buffer.payload_offset);
      }
    }
  }
  description.payload_bytes = layout.extent;
  return std::make_shared<const simpatico::hpln_description>(std::move(description));
}

// ── Host ─────────────────────────────────────────────────────────────────────

class host_payload_source final : public simpatico::payload_source {
 public:
  explicit host_payload_source(const compressed_host_representation& representation)
    : _owner(share(representation))
  {
  }

  void read_to_device(std::uint64_t offset,
                      std::uint64_t size,
                      void* dst,
                      rmm::cuda_stream_view stream) override
  {
    copy_pinned_blocks_to_device(_owner->cast<compressed_host_representation>().payload(),
                                 offset,
                                 dst,
                                 static_cast<std::size_t>(size),
                                 stream);
  }

 private:
  std::unique_ptr<cucascade::idata_representation> _owner;
};

// ── Disk ─────────────────────────────────────────────────────────────────────

/// Frees the host bytes of a copy once the stream has run it. Runs on a CUDA-owned thread and
/// makes no CUDA call.
void CUDART_CB free_host_bytes(void* bytes) { delete static_cast<std::vector<std::byte>*>(bytes); }

/// Enqueues a copy of @p bytes (pageable host memory) to @p dst on @p stream and keeps them
/// alive until the stream has run it: CUDA may stage such a copy after cudaMemcpyAsync returns.
/// cuCascade's disk reader guards the same hazard with a stream wait (host_copy_guard in
/// cucascade/src/cudf/row_range_reader_disk.cpp); here a host function on the stream frees the
/// bytes instead, so that a read of many buffers does not wait once per buffer.
void copy_host_bytes_to_device(std::unique_ptr<std::vector<std::byte>> bytes,
                               void* dst,
                               rmm::cuda_stream_view stream)
{
  CUCASCADE_CUDA_TRY(
    cudaMemcpyAsync(dst, bytes->data(), bytes->size(), cudaMemcpyHostToDevice, stream.value()));
  auto const status = cudaLaunchHostFunc(stream.value(), free_host_bytes, bytes.get());
  if (status == cudaSuccess) {
    static_cast<void>(bytes.release());
    return;
  }
  // No host function: wait for the copy, then `bytes` goes with this scope. The failure is
  // handled here, so it must not stay behind as the thread's last error for an unrelated check.
  static_cast<void>(cudaGetLastError());
  SIRIUS_LOG_DEBUG(
    "[compressed_payload_sources] cudaLaunchHostFunc failed ({}); waiting for the "
    "copy instead",
    cudaGetErrorString(status));
  CUCASCADE_CUDA_TRY(cudaStreamSynchronize(stream.value()));
}

class disk_payload_source final : public simpatico::payload_source {
 public:
  // The context and the one opened file are kept for the life of the source, so every read of
  // a window refill reuses them. Reads are cache-unaware: a spill file must never enter (or be
  // served from) a scan cache.
  explicit disk_payload_source(const compressed_disk_representation& representation)
    : _owner(share(representation)),
      _ctx(detail::spill_context(representation.get_memory_space())),
      _object(_ctx->open_io_object(representation.path()))
  {
    if (!_ctx->supports_device_read()) {
      throw std::logic_error("compressed spill: DISK context cannot read device memory");
    }
  }

  void read_to_device(std::uint64_t offset,
                      std::uint64_t size,
                      void* dst,
                      rmm::cuda_stream_view stream) override
  {
    if (size == 0) { return; }
    auto const& file  = _owner->cast<compressed_disk_representation>();
    auto file_offset  = file.header_bytes() + static_cast<std::size_t>(offset);
    auto remaining    = static_cast<std::size_t>(size);
    auto* destination = static_cast<std::byte*>(dst);

    // A backend that reads with O_DIRECT needs an aligned file offset, and payload ranges start
    // anywhere. The bytes up to the next aligned offset go through the host read, which takes
    // any offset; the device read then starts aligned.
    if (auto const misaligned = file_offset % cucascade::DISK_FILE_ALIGNMENT; misaligned != 0) {
      auto const head = std::min(remaining, cucascade::DISK_FILE_ALIGNMENT - misaligned);
      auto staged     = std::make_unique<std::vector<std::byte>>(head);
      detail::check_spill_bytes(
        _ctx->host_read_io(
          *_object, file_offset, head, reinterpret_cast<std::uint8_t*>(staged->data())),
        head);
      copy_host_bytes_to_device(std::move(staged), destination, stream);
      file_offset += head;
      destination += head;
      remaining -= head;
    }
    if (remaining > 0) {
      // Resolves once the bytes are in device memory, ordered after the work on `stream`.
      detail::spill_request request{_ctx->device_read_async_io(
        *_object, file_offset, remaining, reinterpret_cast<std::uint8_t*>(destination), stream)};
      detail::check_spill_bytes(request.get(), remaining);
    }
  }

 private:
  std::unique_ptr<cucascade::idata_representation> _owner;
  std::shared_ptr<cucascade::io::ioctx> _ctx;
  std::shared_ptr<cucascade::io::io_object> _object;
};

std::shared_ptr<const simpatico::hpln_description> describe_disk(
  const compressed_disk_representation& representation)
{
  std::vector<std::uint8_t> header(representation.header_bytes());
  auto const ctx    = detail::spill_context(representation.get_memory_space());
  auto const object = ctx->open_io_object(representation.path());
  detail::check_spill_bytes(ctx->host_read_io(*object, 0, header.size(), header.data()),
                            header.size());
  return parse_header(header);
}

}  // namespace

std::shared_ptr<simpatico::payload_source> make_payload_source(
  const compressed_device_representation& representation, ::cuda::stream_ref stream)
{
  return std::make_shared<device_payload_source>(representation,
                                                 device_layout(representation, stream));
}

std::shared_ptr<simpatico::payload_source> make_payload_source(
  const compressed_host_representation& representation)
{
  return std::make_shared<host_payload_source>(representation);
}

std::shared_ptr<simpatico::payload_source> make_payload_source(
  const compressed_disk_representation& representation)
{
  return std::make_shared<disk_payload_source>(representation);
}

std::shared_ptr<const simpatico::hpln_description> describe(
  const simpatico_compressed_representation& representation, ::cuda::stream_ref stream)
{
  if (auto const* host = dynamic_cast<const compressed_host_representation*>(&representation)) {
    return parse_header(host->header());
  }
  if (auto const* disk = dynamic_cast<const compressed_disk_representation*>(&representation)) {
    return describe_disk(*disk);
  }
  if (auto const* device = dynamic_cast<const compressed_device_representation*>(&representation)) {
    return describe_device(device_layout(*device, stream));
  }
  throw std::invalid_argument(
    "[compressed_payload_sources] describe: not a compressed device, host or disk batch");
}

compressed_payload open_compressed_payload(
  const simpatico_compressed_representation& representation, ::cuda::stream_ref stream)
{
  if (auto const* host = dynamic_cast<const compressed_host_representation*>(&representation)) {
    return {.description = parse_header(host->header()), .source = make_payload_source(*host)};
  }
  if (auto const* disk = dynamic_cast<const compressed_disk_representation*>(&representation)) {
    return {.description = describe_disk(*disk), .source = make_payload_source(*disk)};
  }
  if (auto const* device = dynamic_cast<const compressed_device_representation*>(&representation)) {
    // One layout for both: the description's offsets are relative to the source's base.
    auto const layout = device_layout(*device, stream);
    return {.description = describe_device(layout),
            .source      = std::make_shared<device_payload_source>(*device, layout)};
  }
  throw std::invalid_argument(
    "[compressed_payload_sources] open_compressed_payload: not a compressed device, host or disk "
    "batch");
}

}  // namespace sirius::compression
