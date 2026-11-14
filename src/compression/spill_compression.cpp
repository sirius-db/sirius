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

#include "spill_compression.hpp"

#include "compressed_disk_representation.hpp"
#include "compressed_representation.hpp"
#include "sirius_config.hpp"
#include "spill_io.hpp"
#include "telemetry/nvtx.hpp"

#include <cudf/copying.hpp>
#include <cudf/reduction/distinct_count.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/resource_ref.hpp>

#include <absl/cleanup/cleanup.h>
#include <api/compressed_table_io.hpp>
#include <api/simpatico_codegen.hpp>
#include <codegen/util/stream_pool.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/error.hpp>
#include <cucascade/memory/disk_table.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>
#include <log/logging.hpp>
#include <memory/sirius_memory_reservation_manager.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sirius::compression {

namespace {

std::mutex g_settings_mutex;
spill_compression_settings g_settings;
// Read on every spill; the struct behind it changes only on (re)configuration.
std::atomic<bool> g_enabled{false};

}  // namespace

void set_spill_compression_settings(spill_compression_settings const& settings)
{
  std::lock_guard lock(g_settings_mutex);
  g_settings = settings;
  g_enabled.store(settings.enabled, std::memory_order_release);
}

void apply_spill_compression_config(compression_config const& config)
{
  set_spill_compression_settings({
    .enabled                 = config.enable_spill_compression,
    .min_batch_bytes         = config.spill_min_batch_bytes,
    .max_compressed_fraction = config.spill_max_compressed_fraction,
    .encode_reserve_fraction = config.spill_encode_reserve_fraction,
  });
}

spill_compression_settings get_spill_compression_settings()
{
  std::lock_guard lock(g_settings_mutex);
  return g_settings;
}

bool spill_compression_applies(std::size_t bytes)
{
  if (!g_enabled.load(std::memory_order_acquire)) { return false; }
  return bytes >= get_spill_compression_settings().min_batch_bytes;
}

namespace {

constexpr char const* kBitpackPlan =
  "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";
constexpr char const* kDeltaBitpackPlan =
  "input -> delta -> differences\n"
  "delta.differences -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";
constexpr char const* kDeltaRleBitpackPlan =
  "input -> delta -> differences\n"
  "delta.differences -> rle -> runs, values\n"
  "delta.differences.runs -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
  "delta.differences.values -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";
constexpr char const* kDictionaryPlan =
  "input -> dictionary -> keys_offsets, keys_chars, indices\n"
  "dictionary.indices -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";
constexpr char const* kStringSplitPlan =
  "input -> str_split -> offsets, chars\n"
  "str_split.chars -> ans\n"
  "str_split.offsets -> delta -> differences\n"
  "str_split.offsets.differences -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";
constexpr char const* kStoredPlan = "input -> identity\n";

/// Rows a string column needs before a dictionary can pay for its key table.
constexpr cudf::size_type kMinDictionaryRows = 4096;
/// Rows sampled to estimate a string column's cardinality.
constexpr cudf::size_type kDictionarySampleRows = 64 * 1024;
/// Dictionary-encode when at most this fraction of the sampled values are distinct.
constexpr double kMaxDictionaryDistinctFraction = 0.1;
/// Largest fixed-width column bitpack accepts. Its encode scratch holds the column at
/// full width as 32-bit words in a cudf column, which caps it at 2^31 words.
constexpr std::size_t kMaxBitpackColumnBytes = std::size_t{1} << 33;  // 8 GiB
/// Rows sampled to choose an integer column's plan.
constexpr cudf::size_type kIntegerSampleRows = 64 * 1024;
/// Values per bitpack chunk; each chunk is packed at the bit width of its own range.
constexpr std::size_t kBitpackChunk = 1024;
/// A delta plan replaces plain bitpack only when it is estimated this much smaller.
constexpr double kMinDeltaSaving = 0.9;

/// Estimated bitpack size of a stream of values pushed one at a time: each chunk at the
/// bit width of its range, plus its min/count/width header.
class bitpack_size_estimate {
 public:
  void push(__int128 value)
  {
    lo_ = count_ == 0 ? value : std::min(lo_, value);
    hi_ = count_ == 0 ? value : std::max(hi_, value);
    if (++count_ == kBitpackChunk) { close_chunk(); }
  }

  /// The estimate, counting a partial last chunk.
  double finish()
  {
    close_chunk();
    return bytes_;
  }

 private:
  void close_chunk()
  {
    if (count_ == 0) { return; }
    auto range = static_cast<unsigned __int128>(hi_ - lo_);
    int bits   = 0;
    for (; range != 0; range >>= 1) {
      ++bits;
    }
    bytes_ += static_cast<double>(count_) * bits / 8 + 16;
    count_ = 0;
  }

  __int128 lo_{0};
  __int128 hi_{0};
  std::size_t count_{0};
  double bytes_{0};
};

/// Integer-valued columns (integers, decimals up to 64 bits, dates and times) choose
/// between bitpack, delta -> bitpack and delta -> rle -> bitpack by estimating each one's
/// size on a prefix sample. Sorted keys, which often reach a spill still in order, pack
/// far smaller as differences, and runs of equal keys smaller still as runs. The sample
/// is bounded (kIntegerSampleRows), so estimating it on the host costs little next to
/// the GPU encode that follows.
std::string integer_spill_plan(cudf::column_view const& column, ::cuda::stream_ref stream)
{
  auto const width = cudf::size_of(column.type());
  auto const rows  = static_cast<std::size_t>(std::min(column.size(), kIntegerSampleRows));
  std::vector<std::uint8_t> raw(rows * width);
  CUDF_CUDA_TRY(cudaMemcpyAsync(raw.data(),
                                column.head<std::uint8_t>() + column.offset() * width,
                                raw.size(),
                                cudaMemcpyDeviceToHost,
                                stream.get()));
  stream.sync();

  bool const is_unsigned = cudf::is_unsigned(column.type());
  auto const shift       = 64 - 8 * static_cast<int>(width);
  bitpack_size_estimate values;
  bitpack_size_estimate deltas;
  bitpack_size_estimate run_lengths;
  bitpack_size_estimate run_values;
  __int128 previous      = 0;
  __int128 run_value     = 0;
  std::size_t run_length = 0;
  for (std::size_t i = 0; i < rows; ++i) {
    // Widened to 128 bits so differences cannot overflow.
    std::uint64_t bits = 0;
    std::memcpy(&bits, raw.data() + i * width, width);
    __int128 const value =
      is_unsigned ? static_cast<__int128>(bits) : static_cast<std::int64_t>(bits << shift) >> shift;
    // The first value is its own difference, as delta stores it.
    __int128 const delta = i == 0 ? value : value - previous;
    previous             = value;
    values.push(value);
    deltas.push(delta);
    if (run_length > 0 && delta == run_value) {
      ++run_length;
      continue;
    }
    if (run_length > 0) {
      run_lengths.push(static_cast<__int128>(run_length));
      run_values.push(run_value);
    }
    run_value  = delta;
    run_length = 1;
  }
  run_lengths.push(static_cast<__int128>(run_length));
  run_values.push(run_value);

  auto const bitpack = values.finish();
  auto const delta   = deltas.finish();
  auto const rle     = run_lengths.finish() + run_values.finish();
  if (rle < kMinDeltaSaving * std::min(bitpack, delta)) { return kDeltaRleBitpackPlan; }
  if (delta < kMinDeltaSaving * bitpack) { return kDeltaBitpackPlan; }
  return kBitpackPlan;
}

}  // namespace

std::string default_spill_plan(cudf::column_view const& column, ::cuda::stream_ref stream)
{
  if (column.type().id() == cudf::type_id::STRING) {
    if (column.size() < kMinDictionaryRows) { return kStoredPlan; }
    // A prefix is a cheap and, for batches built from scans and joins, representative
    // sample; a wrong guess costs ratio, never correctness.
    auto const sample_rows = std::min(column.size(), kDictionarySampleRows);
    auto const sample      = cudf::slice(column, {0, sample_rows}, stream).front();
    auto const distinct    = cudf::distinct_count(
      sample, cudf::null_policy::INCLUDE, cudf::nan_policy::NAN_IS_VALID, stream);
    return static_cast<double>(distinct) <= kMaxDictionaryDistinctFraction * sample_rows
             ? kDictionaryPlan
             : kStringSplitPlan;
  }
  if (!cudf::is_fixed_width(column.type())) { return kStoredPlan; }
  auto const type  = column.type();
  auto const bytes = static_cast<std::size_t>(column.size()) * cudf::size_of(type);
  if (bytes > kMaxBitpackColumnBytes) { return kStoredPlan; }
  bool const integer_valued =
    cudf::is_integral_not_bool(type) || cudf::is_chrono(type) || cudf::is_fixed_point(type);
  if (integer_valued && cudf::size_of(type) <= sizeof(std::int64_t) && column.size() > 0) {
    return integer_spill_plan(column, stream);
  }
  return kBitpackPlan;
}

}  // namespace sirius::compression

namespace sirius {

namespace {

using compression::spill_compression_declined;

/// Streams the column-parallel encode runs on.
constexpr std::size_t kEncodeStreams = 4;

/// A batch encoded on the device, ready to be staged to host memory or a file. The
/// payload buffers belong to @c table and stay valid while it lives.
struct encoded_batch {
  simpatico::compressed_table table;
  std::vector<std::uint8_t> header;
  std::vector<simpatico::payload_buffer_ref> buffers;
  std::uint64_t payload_bytes{0};
  std::size_t uncompressed_bytes{0};
  std::int64_t num_rows{0};
};

std::vector<std::string> column_names(std::size_t num_columns)
{
  std::vector<std::string> names;
  names.reserve(num_columns);
  for (std::size_t i = 0; i < num_columns; ++i) {
    names.push_back("c" + std::to_string(i));
  }
  return names;
}

/// Encode @p source with its default plans. Throws spill_compression_declined when the
/// encode cannot get its headroom or does not save enough; any other exception is a
/// failure the caller treats the same way.
encoded_batch encode_for_spill(const cucascade::gpu_table_representation& source,
                               ::cuda::stream_ref stream)
{
  nvtx_scoped_range nvtx_range{"sirius::compression::spill_encode"};
  source.wait_for_writer(stream);
  auto const settings = compression::get_spill_compression_settings();
  auto const view     = source.get_table_view();

  encoded_batch out;
  out.uncompressed_bytes = source.get_size_in_bytes();
  out.num_rows           = view.num_rows();

  auto& space        = source.get_memory_space();
  auto* const gpu_mr = space.get_memory_resource_of<cucascade::memory::Tier::GPU>();
  if (gpu_mr == nullptr) { throw std::logic_error("spill encode: GPU space has no allocator"); }
  // A reservation covers the allocations of the stream it is attached to, or of the
  // calling thread under per-thread tracking. Only in the latter case does it also
  // cover the stream pool the column-parallel encode runs on, so per-stream tracking
  // encodes on `stream` alone.
  auto const per_thread =
    memory::sirius_memory_reservation_manager::gpu_allocation_tracking_scope(space) ==
    cucascade::memory::reservation_aware_resource_adaptor::AllocationTrackingScope::PER_THREAD;

  // The encoder allocates from the same GPU pool the batch is being evicted from, so
  // it runs only under a reservation of its headroom, which keeps it from taking
  // memory that other tasks are about to reserve.
  bool reserved = false;
  if (settings.encode_reserve_fraction > 0.0) {
    auto const bytes = static_cast<std::size_t>(settings.encode_reserve_fraction *
                                                static_cast<double>(out.uncompressed_bytes));
    auto reservation = space.make_reservation_or_null(bytes);
    if (!reservation) {
      throw spill_compression_declined("no GPU headroom for a " + std::to_string(bytes) +
                                       " byte encode");
    }
    if (!gpu_mr->attach_reservation_to_tracker(stream, std::move(reservation))) {
      throw spill_compression_declined("the allocation tracker already holds a reservation");
    }
    reserved = true;
  }
  absl::Cleanup detach = [&] {
    if (reserved) { gpu_mr->reset_stream_reservation(stream); }
  };

  std::string plan;
  bool all_stored = true;
  for (cudf::size_type i = 0; i < view.num_columns(); ++i) {
    if (i > 0) { plan += "---\n"; }
    auto column_plan = compression::default_spill_plan(view.column(i), stream);
    all_stored       = all_stored && column_plan == compression::kStoredPlan;
    plan += column_plan;
  }
  // Storing every column as is saves nothing, and the encode would still copy the batch.
  if (all_stored) { throw spill_compression_declined("no column of the batch compresses"); }

  auto const mr = rmm::device_async_resource_ref{*gpu_mr};
  if (per_thread) {
    // The pool streams are not ordered after `stream`, which the batch and the plan
    // sampling ran on.
    stream.sync();
    out.table = simpatico::compress_with_plan(
      view, plan, simpatico::thread_device_stream_pool(kEncodeStreams), mr, {});
  } else {
    out.table = simpatico::compress_with_plan(view, plan, stream, mr, {});
  }

  auto const error = simpatico::build_compressed_table_header(
    out.table, out.header, out.buffers, out.payload_bytes, stream);
  if (!error.empty()) { throw std::runtime_error("build_compressed_table_header: " + error); }

  auto const compressed_bytes = out.header.size() + out.payload_bytes;
  if (static_cast<double>(compressed_bytes) >
      settings.max_compressed_fraction * static_cast<double>(out.uncompressed_bytes)) {
    throw spill_compression_declined("compressed " + std::to_string(compressed_bytes) +
                                     " bytes of " + std::to_string(out.uncompressed_bytes));
  }
  SIRIUS_LOG_DEBUG("[spill_compression] compressed {} bytes to {} ({} rows, {} columns)",
                   out.uncompressed_bytes,
                   compressed_bytes,
                   out.num_rows,
                   view.num_columns());
  return out;
}

/// A file under the disk space's mount directory, deleted unless released.
class spill_file {
 public:
  explicit spill_file(cucascade::memory::memory_space const& disk_space)
    : _path(cucascade::memory::generate_disk_file_path(disk_space.get_disk_mount_path()))
  {
  }
  ~spill_file()
  {
    if (!_path.empty()) {
      std::error_code ec;
      std::filesystem::remove(_path, ec);
    }
  }
  spill_file(const spill_file&)            = delete;
  spill_file& operator=(const spill_file&) = delete;

  [[nodiscard]] std::string const& path() const noexcept { return _path; }
  [[nodiscard]] std::string release() noexcept { return std::exchange(_path, {}); }

 private:
  std::string _path;
};

/// gpu_table to compressed_host: encode on the device, stage the payload in pinned
/// blocks of the target host space.
std::unique_ptr<cucascade::idata_representation> compress_gpu_to_host(
  const cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  ::cuda::stream_ref stream,
  cucascade::memory::reservation* reservation)
{
  nvtx_scoped_range nvtx_range{"sirius::compression::gpu_to_compressed_host"};
  auto const& gpu  = source.cast<cucascade::gpu_table_representation>();
  auto encoded     = encode_for_spill(gpu, stream);
  auto* host_space = const_cast<cucascade::memory::memory_space*>(target_memory_space);
  auto* host_mr    = host_space->get_memory_resource_of<cucascade::memory::Tier::HOST>();
  if (host_mr == nullptr) {
    throw std::runtime_error("host space has no fixed_size_host_memory_resource");
  }

  auto blob           = std::make_shared<pinned_compressed_blob>();
  blob->payload       = host_mr->allocate_multiple_blocks(encoded.payload_bytes, reservation);
  blob->payload_bytes = encoded.payload_bytes;
  for (auto const& buffer : encoded.buffers) {
    if (buffer.size_bytes > 0 && buffer.device_ptr != nullptr) {
      copy_device_to_pinned_blocks(buffer.device_ptr,
                                   *blob->payload,
                                   buffer.offset,
                                   static_cast<std::size_t>(buffer.size_bytes),
                                   stream);
    }
  }
  // The device buffers go with `encoded` at return.
  stream.sync();
  blob->header = std::move(encoded.header);

  auto const num_columns = encoded.table.num_columns();
  return std::make_unique<compressed_host_representation>(
    *host_space,
    std::move(blob),
    column_names(num_columns),
    static_cast<std::size_t>(encoded.payload_bytes),
    encoded.uncompressed_bytes,
    encoded.num_rows);
}

/// gpu_table to compressed_disk: encode on the device, write the .hpln file through the
/// disk space's I/O context.
std::unique_ptr<cucascade::idata_representation> compress_gpu_to_disk(
  const cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  ::cuda::stream_ref stream,
  [[maybe_unused]] cucascade::memory::reservation* reservation)
{
  nvtx_scoped_range nvtx_range{"sirius::compression::gpu_to_compressed_disk"};
  auto const& gpu          = source.cast<cucascade::gpu_table_representation>();
  auto const source_device = gpu.get_memory_space().get_device_id();
  rmm::cuda_set_device_raii const device_guard{rmm::cuda_device_id{source_device}};
  int stream_device = -1;
  CUCASCADE_CUDA_TRY(cudaStreamGetDevice(stream.get(), &stream_device));
  if (stream_device != source_device) {
    throw std::invalid_argument("compressed spill: stream belongs to a different GPU");
  }
  auto encoded = encode_for_spill(gpu, stream);
  auto ctx     = compression::detail::spill_context(*target_memory_space);
  if (!ctx->supports_write() || !ctx->supports_device_write()) {
    throw std::logic_error("compressed spill: DISK context cannot write host and device memory");
  }
  auto const header = encoded.header.size();
  auto const bytes  = header + static_cast<std::size_t>(encoded.payload_bytes);
  std::vector<cucascade::io::write_slice> host_slices{{0, header, encoded.header.data()}};
  std::vector<cucascade::io::write_slice> device_slices;
  device_slices.reserve(encoded.buffers.size());
  for (auto const& buffer : encoded.buffers) {
    if (buffer.size_bytes > 0 && buffer.device_ptr != nullptr) {
      device_slices.emplace_back(header + static_cast<std::size_t>(buffer.offset),
                                 static_cast<std::size_t>(buffer.size_bytes),
                                 static_cast<std::uint8_t const*>(buffer.device_ptr));
    }
  }
  spill_file file(*target_memory_space);
  auto object =
    ctx->open_io_object_for_write(file.path(), compression::detail::spill_open_options(bytes));
  compression::detail::spill_request request{
    ctx->mixed_writev_async(*object, host_slices, device_slices, stream)};
  compression::detail::check_spill_bytes(request.get(), bytes);
  // Completion is host-observed and drains device transfers before encoded buffers die.

  auto result = std::make_unique<compressed_disk_representation>(
    *const_cast<cucascade::memory::memory_space*>(target_memory_space),
    file.path(),
    header,
    bytes,
    encoded.uncompressed_bytes,
    encoded.num_rows);
  (void)file.release();
  return result;
}

/// compressed_host to compressed_disk: write the header and pinned payload as an .hpln
/// file through the disk space's I/O context, with no decode and no device work.
std::unique_ptr<cucascade::idata_representation> flush_host_to_disk(
  const cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  [[maybe_unused]] ::cuda::stream_ref stream,
  [[maybe_unused]] cucascade::memory::reservation* reservation)
{
  nvtx_scoped_range nvtx_range{"sirius::compression::compressed_host_to_disk"};
  auto const& host = source.cast<compressed_host_representation>();
  if (host.selected_indices().has_value()) {
    // A projection is a view of a pinned table, never a spilled batch.
    throw std::runtime_error("cannot flush a column projection to disk");
  }

  auto const header        = host.header();
  auto const& payload      = host.payload();
  auto const payload_bytes = host.get_size_in_bytes();
  auto const block_size    = payload.block_size();
  auto ctx                 = compression::detail::spill_context(*target_memory_space);
  if (!ctx->supports_write()) {
    throw std::logic_error("compressed spill: DISK context cannot write host memory");
  }
  std::vector<cucascade::io::write_slice> slices{{0, header.size(), header.data()}};
  std::size_t written = 0;
  for (std::size_t block = 0; written < payload_bytes; ++block) {
    auto const n = std::min(block_size, payload_bytes - written);
    slices.emplace_back(
      header.size() + written, n, reinterpret_cast<std::uint8_t const*>(payload.at(block).data()));
    written += n;
  }
  spill_file file(*target_memory_space);
  auto object = ctx->open_io_object_for_write(
    file.path(), compression::detail::spill_open_options(header.size() + payload_bytes));
  compression::detail::spill_request request{ctx->host_writev_async(*object, slices)};
  compression::detail::check_spill_bytes(request.get(), header.size() + payload_bytes);

  auto result = std::make_unique<compressed_disk_representation>(
    *const_cast<cucascade::memory::memory_space*>(target_memory_space),
    file.path(),
    header.size(),
    header.size() + payload_bytes,
    host.get_uncompressed_data_size_in_bytes(),
    host.num_rows());
  (void)file.release();
  return result;
}

}  // namespace

void register_spill_compression_converters(cucascade::representation_converter_registry& registry)
{
  if (!registry
         .has_converter<cucascade::gpu_table_representation, compressed_host_representation>()) {
    registry
      .register_converter<cucascade::gpu_table_representation, compressed_host_representation>(
        compress_gpu_to_host);
  }
  if (!registry
         .has_converter<cucascade::gpu_table_representation, compressed_disk_representation>()) {
    registry
      .register_converter<cucascade::gpu_table_representation, compressed_disk_representation>(
        compress_gpu_to_disk);
  }
  if (!registry.has_converter<compressed_host_representation, compressed_disk_representation>()) {
    registry.register_converter<compressed_host_representation, compressed_disk_representation>(
      flush_host_to_disk);
  }
}

}  // namespace sirius
