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

#include "memory/sirius_memory_reservation_manager.hpp"
#include "operator/operator_test_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <api/compressed_table_io.hpp>
#include <catch.hpp>
#include <compression/compressed_disk_representation.hpp>
#include <compression/compressed_payload_sources.hpp>
#include <compression/compressed_representation.hpp>
#include <compression/device_compressed_blob.hpp>
#include <compression/spill_compression.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/disk_file_format.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <data/convertible_data_batch.hpp>
#include <data/data_batch_utils.hpp>
#include <data/sirius_converter_registry.hpp>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

namespace utils = sirius::test::operator_utils;
using sirius::compression::describe;
using sirius::compression::make_payload_source;
using sirius::compression::open_compressed_payload;

bool has_gpu()
{
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

/// GPU, host and disk spaces, with the converters registered.
struct source_env {
  fs::path dir =
    fs::temp_directory_path() / ("sirius_payload_sources_test_" + std::to_string(::getpid()));
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> mgr;
  cucascade::memory::memory_space* gpu  = nullptr;
  cucascade::memory::memory_space* host = nullptr;
  cucascade::memory::memory_space* disk = nullptr;
  rmm::cuda_stream stream;

  source_env()
  {
    fs::create_directories(dir);
    sirius::converter_registry::reset_for_testing();
    cucascade::memory::reservation_manager_configurator builder;
    builder.set_number_of_gpus(1)
      .track_reservation_per_stream(false)
      .set_gpu_usage_limit(1ull << 30)
      .set_reservation_fraction_per_gpu(0.75)
      .set_per_numa_region_capacity(1ull << 30)
      .use_gpu_id_as_host_id()
      .set_reservation_fraction_per_numa_region(0.75)
      .set_disk_mounting_point(0, 4ull << 30, dir.string());
    mgr = std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());
    sirius::converter_registry::initialize();
    gpu  = mgr->get_memory_space(cucascade::memory::Tier::GPU, 0);
    host = mgr->get_memory_space(cucascade::memory::Tier::HOST, 0);
    disk = const_cast<cucascade::memory::memory_space*>(
      mgr->get_memory_spaces_for_tier(cucascade::memory::Tier::DISK).front());
  }
  ~source_env() { fs::remove_all(dir); }
};

source_env& env()
{
  static source_env e;
  return e;
}

template <typename T>
std::unique_ptr<cudf::column> numeric_column(cudf::type_id type,
                                             std::vector<T> const& values,
                                             bool nullable)
{
  auto stream = utils::default_stream();
  auto mr     = utils::get_resource_ref(*env().gpu);
  auto rows   = static_cast<cudf::size_type>(values.size());
  auto column = cudf::make_fixed_width_column(
    cudf::data_type{type}, rows, cudf::mask_state::UNALLOCATED, stream, mr);
  REQUIRE(cudaMemcpy(column->mutable_view().head(),
                     values.data(),
                     values.size() * sizeof(T),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  if (nullable) {
    // Every seventh row null. cudf sizes a mask to its padded allocation.
    std::vector<std::uint32_t> words(
      cudf::bitmask_allocation_size_bytes(rows) / sizeof(std::uint32_t), 0);
    cudf::size_type nulls = 0;
    for (cudf::size_type i = 0; i < rows; ++i) {
      if (i % 7 == 0) {
        ++nulls;
      } else {
        words[static_cast<std::size_t>(i) / 32] |= 1u << (i % 32);
      }
    }
    auto mask = cudf::create_null_mask(rows, cudf::mask_state::UNINITIALIZED, stream, mr);
    REQUIRE(cudaMemcpy(mask.data(),
                       words.data(),
                       words.size() * sizeof(std::uint32_t),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    column->set_null_mask(std::move(mask), nulls);
  }
  return column;
}

/// Sorted keys, 20-bit random values (over a megabyte once bit-packed, so the pinned payload
/// spans several blocks), nullable small values and a low-cardinality string column.
std::unique_ptr<cudf::table> source_table()
{
  constexpr std::size_t rows = 1'200'000;
  std::mt19937 rng(20261112);
  std::vector<std::int64_t> keys(rows);
  std::vector<std::int32_t> noise(rows);
  std::vector<std::int32_t> small(rows);
  std::vector<std::string> flags(rows);
  for (std::size_t i = 0; i < rows; ++i) {
    keys[i]  = static_cast<std::int64_t>(i * 100 + i % 7);
    noise[i] = static_cast<std::int32_t>(rng() & 0xFFFFF);
    small[i] = static_cast<std::int32_t>(i % 1000);
    flags[i] = (i % 3 == 0) ? "RETURNED" : (i % 3 == 1 ? "NONE" : "DELIVER IN PERSON");
  }
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(numeric_column(cudf::type_id::INT64, keys, false));
  columns.push_back(numeric_column(cudf::type_id::INT32, noise, false));
  columns.push_back(numeric_column(cudf::type_id::INT32, small, true));
  columns.push_back(
    utils::make_string_column(flags, utils::default_stream(), utils::get_resource_ref(*env().gpu)));
  return std::make_unique<cudf::table>(std::move(columns));
}

/// The same compressed batch in the three tiers, and its payload bytes on the host.
struct compressed_tiers {
  std::unique_ptr<cucascade::idata_representation> host_owner;
  std::unique_ptr<cucascade::idata_representation> disk_owner;
  std::unique_ptr<sirius::compressed_device_representation> device;
  std::vector<std::byte> payload;

  [[nodiscard]] sirius::compressed_host_representation const& host() const
  {
    return host_owner->cast<sirius::compressed_host_representation>();
  }
  [[nodiscard]] sirius::compressed_disk_representation const& disk() const
  {
    return disk_owner->cast<sirius::compressed_disk_representation>();
  }
};

void move_to(std::shared_ptr<cucascade::data_batch> const& batch,
             cucascade::memory::memory_space* space)
{
  sirius::convertible_data_batch convertible(batch);
  REQUIRE(convertible.convert({space}, env().stream, *env().mgr, true).has_value());
}

compressed_tiers make_tiers()
{
  sirius::compression::spill_compression_settings settings;
  settings.enabled                 = true;
  settings.min_batch_bytes         = 0;
  settings.max_compressed_fraction = 1.0;
  sirius::compression::set_spill_compression_settings(settings);

  auto stream = utils::default_stream();
  auto batch  = sirius::make_data_batch(
    source_table(), *env().gpu, stream, sirius::telemetry::batch_telemetry_info{});
  batch->to_mutable().get_data()->record_writer_event(stream);

  compressed_tiers tiers;
  // Clones share the pinned blob and the file, so both outlive the batch's own moves.
  move_to(batch, env().host);
  {
    auto ro = batch->to_read_only();
    REQUIRE(dynamic_cast<sirius::compressed_host_representation const*>(ro.get_data()) != nullptr);
    tiers.host_owner = ro.get_data()->clone(stream);
  }
  move_to(batch, env().disk);
  {
    auto ro = batch->to_read_only();
    REQUIRE(dynamic_cast<sirius::compressed_disk_representation const*>(ro.get_data()) != nullptr);
    tiers.disk_owner = ro.get_data()->clone(stream);
  }
  sirius::compression::set_spill_compression_settings({});

  auto const& host    = tiers.host();
  auto const& blocks  = host.payload();
  auto const bytes    = host.get_size_in_bytes();
  auto const block_sz = blocks.block_size();
  tiers.payload.resize(bytes);
  for (std::size_t done = 0; done < bytes; done += block_sz) {
    std::memcpy(tiers.payload.data() + done,
                blocks.at(done / block_sz).data(),
                std::min(block_sz, bytes - done));
  }

  // The device tier: the table rebuilt on the device from the pinned payload.
  auto blob = std::make_shared<sirius::compressed_device_blob>();
  std::string error;
  simpatico::payload_fetch_fn fetch =
    [&blocks](std::uint64_t off, std::size_t sz, void* dst, ::cuda::stream_ref s) {
      sirius::copy_pinned_blocks_to_device(blocks, off, dst, sz, s);
    };
  blob->table = simpatico::read_compressed_table_from_memory(
    host.header(), fetch, stream, utils::get_resource_ref(*env().gpu), &error);
  REQUIRE(error.empty());
  stream.sync();
  tiers.device = std::make_unique<sirius::compressed_device_representation>(
    *env().gpu,
    std::move(blob),
    host.column_names(),
    bytes,
    host.get_uncompressed_data_size_in_bytes(),
    host.num_rows());
  return tiers;
}

compressed_tiers& tiers()
{
  static compressed_tiers t = make_tiers();
  return t;
}

/// Payload bytes [offset, offset + size) as @p source delivers them to the device.
std::vector<std::byte> read_range(simpatico::payload_source& source,
                                  std::uint64_t offset,
                                  std::uint64_t size)
{
  auto stream = utils::default_stream();
  // A byte of slack on each side shows a read that strays outside its range.
  rmm::device_buffer device(size + 2, stream, utils::get_resource_ref(*env().gpu));
  REQUIRE(cudaMemsetAsync(device.data(), 0xA5, size + 2, stream.get()) == cudaSuccess);
  source.read_to_device(offset, size, static_cast<std::byte*>(device.data()) + 1, stream);
  std::vector<std::byte> host(size + 2);
  REQUIRE(
    cudaMemcpyAsync(host.data(), device.data(), size + 2, cudaMemcpyDeviceToHost, stream.get()) ==
    cudaSuccess);
  stream.sync();
  REQUIRE(host.front() == std::byte{0xA5});
  REQUIRE(host.back() == std::byte{0xA5});
  return {host.begin() + 1, host.end() - 1};
}

std::vector<std::byte> expected_range(std::uint64_t offset, std::uint64_t size)
{
  auto const& payload = tiers().payload;
  REQUIRE(offset + size <= payload.size());
  return {payload.begin() + static_cast<std::ptrdiff_t>(offset),
          payload.begin() + static_cast<std::ptrdiff_t>(offset + size)};
}

/// Ranges at the payload's ends, around every multiple of @p boundary, and at random.
std::vector<std::pair<std::uint64_t, std::uint64_t>> probe_ranges(std::uint64_t payload_bytes,
                                                                  std::uint64_t boundary)
{
  std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges{
    {0, 1}, {0, 0}, {1, 4095}, {payload_bytes - 1, 1}, {payload_bytes - 5000, 5000}};
  std::uint64_t crossings = 0;
  for (std::uint64_t edge = boundary; edge < payload_bytes && crossings < 4; edge += boundary) {
    ++crossings;
    ranges.emplace_back(edge - 1, 2);
    ranges.emplace_back(edge - 777, std::min<std::uint64_t>(3001, payload_bytes - (edge - 777)));
    ranges.emplace_back(edge, std::min<std::uint64_t>(13, payload_bytes - edge));
  }
  std::mt19937_64 rng(7);
  for (int i = 0; i < 24; ++i) {
    auto const offset = rng() % payload_bytes;
    auto const limit  = std::min<std::uint64_t>(payload_bytes - offset, 3 * boundary);
    ranges.emplace_back(offset, 1 + rng() % limit);
  }
  return ranges;
}

/// Every payload range a description names: validity masks and leaf buffers, in order.
std::vector<std::pair<std::uint64_t, std::uint64_t>> buffer_ranges(
  simpatico::hpln_description const& description)
{
  std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
  for (auto const& column : description.columns) {
    if (column.validity == simpatico::hpln_validity::mask) {
      ranges.emplace_back(column.validity_payload_offset, column.validity_size);
    }
    for (auto const& leaf : column.leaves) {
      for (auto const& buffer : leaf.buffers) {
        ranges.emplace_back(buffer.payload_offset, buffer.size);
      }
    }
  }
  return ranges;
}

/// Checks that two descriptions agree on everything but where the buffers are.
void check_same_structure(simpatico::hpln_description const& a,
                          simpatico::hpln_description const& b)
{
  CHECK(a.version == b.version);
  REQUIRE(a.columns.size() == b.columns.size());
  for (std::size_t c = 0; c < a.columns.size(); ++c) {
    auto const& x = a.columns[c];
    auto const& y = b.columns[c];
    CAPTURE(c);
    CHECK(x.name == y.name);
    CHECK(x.dtype == y.dtype);
    CHECK(x.num_rows == y.num_rows);
    CHECK(x.validity == y.validity);
    CHECK(x.null_count == y.null_count);
    CHECK(x.validity_size == y.validity_size);
    REQUIRE(x.nodes.size() == y.nodes.size());
    for (std::size_t n = 0; n < x.nodes.size(); ++n) {
      CHECK(x.nodes[n].op == y.nodes[n].op);
      CHECK(x.nodes[n].is_bitjoin == y.nodes[n].is_bitjoin);
      CHECK(x.nodes[n].edges == y.nodes[n].edges);
      CHECK(x.nodes[n].outputs == y.nodes[n].outputs);
    }
    REQUIRE(x.leaves.size() == y.leaves.size());
    for (std::size_t l = 0; l < x.leaves.size(); ++l) {
      auto const& p = x.leaves[l];
      auto const& q = y.leaves[l];
      CHECK(p.node_index == q.node_index);
      CHECK(p.slot == q.slot);
      CHECK(p.kind == q.kind);
      CHECK(p.type_tag == q.type_tag);
      CHECK(p.meta.index() == q.meta.index());
      REQUIRE(p.buffers.size() == q.buffers.size());
      for (std::size_t i = 0; i < p.buffers.size(); ++i) {
        CHECK(p.buffers[i].name == q.buffers[i].name);
        CHECK(p.buffers[i].type_tag == q.buffers[i].type_tag);
        CHECK(p.buffers[i].size == q.buffers[i].size);
      }
    }
  }
}

}  // namespace

TEST_CASE("compressed payload sources: a host source reads across pinned block boundaries",
          "[compressed_payload_sources]")
{
  if (!has_gpu()) { return; }
  auto const& host       = tiers().host();
  auto const bytes       = tiers().payload.size();
  auto const block_bytes = host.payload().block_size();
  // The payload has to span blocks for the ranges below to cross one.
  REQUIRE(bytes > 2 * block_bytes);

  auto source = make_payload_source(host);
  CHECK(source->device_base() == nullptr);
  for (auto const& [offset, size] : probe_ranges(bytes, block_bytes)) {
    CAPTURE(offset, size);
    CHECK(read_range(*source, offset, size) == expected_range(offset, size));
  }
  CHECK(read_range(*source, 0, bytes) == tiers().payload);
}

TEST_CASE("compressed payload sources: a disk source reads unaligned ranges of the file",
          "[compressed_payload_sources]")
{
  if (!has_gpu()) { return; }
  auto const& disk = tiers().disk();
  auto const bytes = tiers().payload.size();
  REQUIRE(disk.get_size_in_bytes() == disk.header_bytes() + bytes);

  auto source = make_payload_source(disk);
  CHECK(source->device_base() == nullptr);
  for (auto const& [offset, size] : probe_ranges(bytes, cucascade::DISK_FILE_ALIGNMENT)) {
    CAPTURE(offset, size);
    CHECK(read_range(*source, offset, size) == expected_range(offset, size));
  }
  // Ranges that start on, and just off, an aligned file offset.
  auto const aligned =
    cucascade::DISK_FILE_ALIGNMENT - disk.header_bytes() % cucascade::DISK_FILE_ALIGNMENT;
  for (std::uint64_t offset : {aligned, aligned + 1, aligned - 1}) {
    for (std::uint64_t size : {1, 4096, 4097, 10000}) {
      CAPTURE(offset, size);
      CHECK(read_range(*source, offset, size) == expected_range(offset, size));
    }
  }
  CHECK(read_range(*source, 0, bytes) == tiers().payload);
}

TEST_CASE("compressed payload sources: a disk source keeps its file",
          "[compressed_payload_sources]")
{
  if (!has_gpu()) { return; }
  // A second owner of the file, dropped while the source is still in use.
  auto owner       = tiers().disk().clone(utils::default_stream());
  auto const& disk = owner->cast<sirius::compressed_disk_representation>();
  auto const path  = disk.path();
  auto source      = make_payload_source(disk);
  owner.reset();
  CHECK(fs::exists(path));
  CHECK(read_range(*source, 100, 9000) == expected_range(100, 9000));
}

TEST_CASE("compressed payload sources: a host source keeps its pinned payload",
          "[compressed_payload_sources]")
{
  if (!has_gpu()) { return; }
  auto owner  = tiers().host().clone(utils::default_stream());
  auto source = make_payload_source(owner->cast<sirius::compressed_host_representation>());
  owner.reset();
  CHECK(read_range(*source, 100, 9000) == expected_range(100, 9000));
}

TEST_CASE("compressed payload sources: describe agrees across the three tiers",
          "[compressed_payload_sources]")
{
  if (!has_gpu()) { return; }
  auto const host   = describe(tiers().host());
  auto const disk   = describe(tiers().disk());
  auto const device = describe(*tiers().device);
  REQUIRE(host != nullptr);
  REQUIRE(disk != nullptr);
  REQUIRE(device != nullptr);

  CHECK(host->columns.size() == 4);
  CHECK(host->payload_bytes == tiers().payload.size());
  CHECK(disk->payload_bytes == host->payload_bytes);
  for (auto const& column : host->columns) {
    CHECK(column.num_rows == tiers().host().num_rows());
  }
  CHECK(host->columns[2].validity == simpatico::hpln_validity::mask);

  check_same_structure(*host, *disk);
  check_same_structure(*host, *device);
  // Host and disk hold the same image, so their buffers are at the same offsets.
  CHECK(buffer_ranges(*host) == buffer_ranges(*disk));

  for (auto const* description : {host.get(), disk.get(), device.get()}) {
    std::uint64_t total = 0;
    for (auto const& [offset, size] : buffer_ranges(*description)) {
      CAPTURE(offset, size);
      CHECK(offset + size <= description->payload_bytes);
      total += size;
    }
    // Every payload byte belongs to a buffer; a device payload may have gaps between them.
    CHECK(total == host->payload_bytes);
  }
}

TEST_CASE("compressed payload sources: every buffer reads the same from the three tiers",
          "[compressed_payload_sources]")
{
  if (!has_gpu()) { return; }
  auto const host_ranges   = buffer_ranges(*describe(tiers().host()));
  auto const disk_ranges   = buffer_ranges(*describe(tiers().disk()));
  auto const device_ranges = buffer_ranges(*describe(*tiers().device));
  REQUIRE(host_ranges.size() == device_ranges.size());
  REQUIRE(host_ranges.size() == disk_ranges.size());

  auto host_source   = make_payload_source(tiers().host());
  auto disk_source   = make_payload_source(tiers().disk());
  auto device_source = make_payload_source(*tiers().device);
  // The device payload is read in place.
  auto const* base = device_source->device_base();
  REQUIRE(base != nullptr);

  auto stream = utils::default_stream();
  for (std::size_t i = 0; i < host_ranges.size(); ++i) {
    auto const [offset, size] = host_ranges[i];
    CAPTURE(i, offset, size);
    REQUIRE(device_ranges[i].second == size);
    auto const expected = expected_range(offset, size);
    CHECK(read_range(*host_source, offset, size) == expected);
    CHECK(read_range(*disk_source, disk_ranges[i].first, size) == expected);
    CHECK(read_range(*device_source, device_ranges[i].first, size) == expected);

    std::vector<std::byte> in_place(size);
    if (size > 0) {
      REQUIRE(cudaMemcpyAsync(in_place.data(),
                              base + device_ranges[i].first,
                              size,
                              cudaMemcpyDeviceToHost,
                              stream.get()) == cudaSuccess);
      stream.sync();
    }
    CHECK(in_place == expected);
  }
}

TEST_CASE("compressed payload sources: a device batch without a table is rejected",
          "[compressed_payload_sources]")
{
  if (!has_gpu()) { return; }
  sirius::compressed_device_representation empty(*env().gpu, nullptr, {"a"}, 0, 0, 0);
  CHECK_THROWS_AS(make_payload_source(empty), std::invalid_argument);
  CHECK_THROWS_AS(describe(empty), std::invalid_argument);
  CHECK_THROWS_AS(open_compressed_payload(empty, ::cuda::stream_ref{env().stream.value()}),
                  std::invalid_argument);
}

TEST_CASE("compressed payload sources: a batch opens as a matching description and source",
          "[compressed_payload_sources]")
{
  if (!has_gpu()) { return; }
  const sirius::simpatico_compressed_representation* const batches[] = {
    &tiers().host(), &tiers().disk(), tiers().device.get()};
  // Dense offsets of the host image, in the same buffer order as every description.
  auto const host_ranges = buffer_ranges(*describe(tiers().host()));
  for (auto const* batch : batches) {
    CAPTURE(typeid(*batch).name());
    auto const opened = open_compressed_payload(*batch, ::cuda::stream_ref{env().stream.value()});
    REQUIRE(opened.description != nullptr);
    REQUIRE(opened.source != nullptr);
    auto const separate = describe(*batch);
    check_same_structure(*opened.description, *separate);
    CHECK(opened.description->payload_bytes == separate->payload_bytes);
    // The description's offsets are the ones its own source reads the buffers at.
    auto const ranges = buffer_ranges(*opened.description);
    CHECK(ranges == buffer_ranges(*separate));
    REQUIRE(ranges.size() == host_ranges.size());
    for (std::size_t i = 0; i < ranges.size(); ++i) {
      CAPTURE(i);
      REQUIRE(ranges[i].second == host_ranges[i].second);
      CHECK(read_range(*opened.source, ranges[i].first, ranges[i].second) ==
            expected_range(host_ranges[i].first, host_ranges[i].second));
    }
  }
}
