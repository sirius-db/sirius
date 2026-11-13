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
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cuda_runtime.h>

#include <api/chunk_decode.hpp>
#include <api/compressed_table_io.hpp>
#include <api/simpatico_codegen.hpp>
#include <catch.hpp>
#include <compression/compressed_disk_representation.hpp>
#include <compression/compressed_payload_sources.hpp>
#include <compression/compressed_representation.hpp>
#include <compression/compressed_row_range_readers.hpp>
#include <compression/device_compressed_blob.hpp>
#include <compression/spill_compression.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/row_range_read.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <data/convertible_data_batch.hpp>
#include <data/data_batch_utils.hpp>
#include <data/sirius_converter_registry.hpp>
#include <data/sirius_row_range_reader_registry.hpp>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

namespace utils = sirius::test::operator_utils;

using reader_state = std::unique_ptr<cucascade::row_range_reader_state>;

/// Byte the destination buffers hold before a read; untouched bytes must still hold it.
constexpr std::uint8_t kFill = 0xA5;
/// Longest string of any source column.
constexpr std::size_t kMaxStringBytes = 39;

// Columns of the main source (20,500 rows): one per spill plan, plus nullable variants.
constexpr std::size_t kSorted         = 0;  // INT64 sorted, irregular steps -> delta
constexpr std::size_t kRuns           = 1;  // INT32 long runs -> delta -> rle
constexpr std::size_t kRandom         = 2;  // INT64 random -> bitpack
constexpr std::size_t kDict           = 3;  // STRING, 8 distinct values -> dictionary
constexpr std::size_t kSplit          = 4;  // STRING, all distinct, some empty -> str_split
constexpr std::size_t kRandomNulls    = 5;  // INT32 random, nullable -> bitpack
constexpr std::size_t kDictNulls      = 6;  // STRING low cardinality, nullable -> dictionary
constexpr std::size_t kSplitNulls     = 7;  // STRING high cardinality, nullable -> str_split
constexpr std::size_t kSortedAllValid = 8;  // INT64 sorted, has a mask but no nulls -> delta
constexpr std::size_t kMainRows       = 20'500;

// Columns of the small source (3,000 rows): STRING columns this short are stored as is.
constexpr std::size_t kSmallString      = 0;  // STRING -> stored
constexpr std::size_t kSmallStringNulls = 1;  // STRING, nullable -> stored
constexpr std::size_t kSmallSorted      = 2;  // INT64 sorted -> delta
constexpr std::size_t kSmallRows        = 3'000;

bool has_gpu()
{
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

/// GPU, host and disk spaces, with the converters registered.
struct reader_env {
  fs::path dir =
    fs::temp_directory_path() / ("sirius_compressed_readers_test_" + std::to_string(::getpid()));
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> mgr;
  cucascade::memory::memory_space* gpu  = nullptr;
  cucascade::memory::memory_space* host = nullptr;
  cucascade::memory::memory_space* disk = nullptr;
  rmm::cuda_stream stream;

  reader_env()
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
  ~reader_env() { fs::remove_all(dir); }
};

reader_env& env()
{
  static reader_env e;
  return e;
}

cucascade::row_range_reader_registry& registry()
{
  if (!sirius::row_range_reader_registry::is_initialized()) {
    sirius::row_range_reader_registry::initialize();
  }
  return sirius::row_range_reader_registry::get();
}

// ── Source data and its host oracle ──────────────────────────────────────────

/// Host copy of one source column: what a read must deliver.
struct host_column {
  cudf::data_type type{cudf::type_id::EMPTY};
  /// Bytes per row; 0 for STRING.
  std::size_t width{0};
  /// Fixed-width values, `width` bytes per row.
  std::vector<std::uint8_t> bytes;
  /// STRING values (empty for a null row).
  std::vector<std::string> strings;
  /// STRING: prefix[r] = chars bytes of rows [0, r).
  std::vector<std::size_t> prefix;
  /// Validity per row; empty when the column has no null mask.
  std::vector<bool> valid;

  [[nodiscard]] bool is_string() const { return type.id() == cudf::type_id::STRING; }
  [[nodiscard]] bool is_valid(std::size_t row) const { return valid.empty() || valid[row]; }
  [[nodiscard]] bool has_nulls() const
  {
    return std::find(valid.begin(), valid.end(), false) != valid.end();
  }
  /// Chars bytes of rows [begin, begin + rows).
  [[nodiscard]] std::size_t chars_bytes(std::size_t begin, std::size_t rows) const
  {
    return prefix[begin + rows] - prefix[begin];
  }
};

struct source_data {
  std::size_t rows{0};
  std::vector<host_column> columns;
  std::vector<std::unique_ptr<cudf::column>> device_columns;
};

void set_validity(cudf::column& column, std::vector<bool> const& valid)
{
  if (valid.empty()) { return; }
  auto stream = utils::default_stream();
  auto rows   = static_cast<cudf::size_type>(valid.size());
  // cudf sizes a mask to its padded allocation.
  std::vector<std::uint32_t> words(
    cudf::bitmask_allocation_size_bytes(rows) / sizeof(std::uint32_t), 0);
  cudf::size_type nulls = 0;
  for (std::size_t i = 0; i < valid.size(); ++i) {
    if (valid[i]) {
      words[i / 32] |= 1u << (i % 32);
    } else {
      ++nulls;
    }
  }
  auto mask = cudf::create_null_mask(
    rows, cudf::mask_state::UNINITIALIZED, stream, utils::get_resource_ref(*env().gpu));
  REQUIRE(cudaMemcpy(mask.data(),
                     words.data(),
                     words.size() * sizeof(std::uint32_t),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  column.set_null_mask(std::move(mask), nulls);
}

template <typename T>
void add_fixed(source_data& source,
               cudf::type_id type,
               std::vector<T> const& values,
               std::vector<bool> valid = {})
{
  auto stream = utils::default_stream();
  auto column = cudf::make_fixed_width_column(cudf::data_type{type},
                                              static_cast<cudf::size_type>(values.size()),
                                              cudf::mask_state::UNALLOCATED,
                                              stream,
                                              utils::get_resource_ref(*env().gpu));
  REQUIRE(cudaMemcpy(column->mutable_view().head(),
                     values.data(),
                     values.size() * sizeof(T),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  set_validity(*column, valid);
  host_column host;
  host.type  = cudf::data_type{type};
  host.width = sizeof(T);
  host.bytes.resize(values.size() * sizeof(T));
  std::memcpy(host.bytes.data(), values.data(), host.bytes.size());
  host.valid = std::move(valid);
  source.columns.push_back(std::move(host));
  source.device_columns.push_back(std::move(column));
}

/// Null rows must hold an empty string.
void add_strings(source_data& source,
                 std::vector<std::string> strings,
                 std::vector<bool> valid = {})
{
  auto column = utils::make_string_column(
    strings, utils::default_stream(), utils::get_resource_ref(*env().gpu));
  set_validity(*column, valid);
  host_column host;
  host.type = cudf::data_type{cudf::type_id::STRING};
  host.prefix.assign(strings.size() + 1, 0);
  for (std::size_t r = 0; r < strings.size(); ++r) {
    host.prefix[r + 1] = host.prefix[r] + strings[r].size();
  }
  host.strings = std::move(strings);
  host.valid   = std::move(valid);
  source.columns.push_back(std::move(host));
  source.device_columns.push_back(std::move(column));
}

std::vector<bool> random_validity(std::mt19937& rng, std::size_t rows)
{
  std::vector<bool> valid(rows);
  for (std::size_t r = 0; r < rows; ++r) {
    valid[r] = rng() % 5 != 0;
  }
  return valid;
}

/// Sorted values with irregular steps: differences pack small and do not form runs.
std::vector<std::int64_t> sorted_values(std::mt19937& rng, std::size_t rows)
{
  std::vector<std::int64_t> values(rows);
  std::int64_t value = 1'000'000;
  for (auto& v : values) {
    value += 1 + static_cast<std::int64_t>(rng() % 16);
    v = value;
  }
  return values;
}

std::vector<std::string> low_cardinality_strings(std::mt19937& rng,
                                                 std::size_t rows,
                                                 std::vector<bool> const& valid)
{
  static const char* const kValues[] = {
    "RETURNED", "NONE", "DELIVER IN PERSON", "", "COLLECT COD", "TAKE BACK RETURN", "AIR", "x"};
  std::vector<std::string> strings(rows);
  for (std::size_t r = 0; r < rows; ++r) {
    if (!valid.empty() && !valid[r]) { continue; }
    strings[r] = kValues[rng() % 8];
  }
  return strings;
}

/// Random letters of 0..kMaxStringBytes bytes: nearly all distinct, about 1 in 40 empty.
std::vector<std::string> high_cardinality_strings(std::mt19937& rng,
                                                  std::size_t rows,
                                                  std::vector<bool> const& valid)
{
  std::vector<std::string> strings(rows);
  for (std::size_t r = 0; r < rows; ++r) {
    if (!valid.empty() && !valid[r]) { continue; }
    strings[r].resize(rng() % (kMaxStringBytes + 1));
    for (auto& c : strings[r]) {
      c = static_cast<char>('a' + rng() % 26);
    }
  }
  return strings;
}

source_data main_source()
{
  std::mt19937 rng(20261112);
  source_data source;
  source.rows = kMainRows;
  add_fixed(source, cudf::type_id::INT64, sorted_values(rng, kMainRows));
  {
    std::vector<std::int32_t> runs(kMainRows);
    for (std::size_t i = 0; i < kMainRows; ++i) {
      runs[i] = static_cast<std::int32_t>(i / 500) * 3;
    }
    add_fixed(source, cudf::type_id::INT32, runs);
  }
  {
    std::vector<std::int64_t> random(kMainRows);
    for (auto& v : random) {
      v = static_cast<std::int64_t>((static_cast<std::uint64_t>(rng()) << 9) ^ rng()) - 12345;
    }
    add_fixed(source, cudf::type_id::INT64, random);
  }
  add_strings(source, low_cardinality_strings(rng, kMainRows, {}));
  add_strings(source, high_cardinality_strings(rng, kMainRows, {}));
  {
    std::vector<std::int32_t> random(kMainRows);
    for (auto& v : random) {
      v = static_cast<std::int32_t>(rng() & 0xFFFFF);
    }
    add_fixed(source, cudf::type_id::INT32, random, random_validity(rng, kMainRows));
  }
  {
    auto valid = random_validity(rng, kMainRows);
    add_strings(source, low_cardinality_strings(rng, kMainRows, valid), valid);
  }
  {
    auto valid = random_validity(rng, kMainRows);
    add_strings(source, high_cardinality_strings(rng, kMainRows, valid), valid);
  }
  add_fixed(source,
            cudf::type_id::INT64,
            sorted_values(rng, kMainRows),
            std::vector<bool>(kMainRows, true));
  return source;
}

source_data small_source()
{
  std::mt19937 rng(7);
  source_data source;
  source.rows = kSmallRows;
  add_strings(source, high_cardinality_strings(rng, kSmallRows, {}));
  {
    auto valid = random_validity(rng, kSmallRows);
    add_strings(source, high_cardinality_strings(rng, kSmallRows, valid), valid);
  }
  add_fixed(source, cudf::type_id::INT64, sorted_values(rng, kSmallRows));
  return source;
}

// ── The same compressed batch in the three tiers ─────────────────────────────

struct tier {
  std::string name;
  std::shared_ptr<cucascade::data_batch> batch;
};

struct compressed_source {
  source_data data;
  /// device, host, disk.
  std::vector<tier> tiers;
};

void move_to(std::shared_ptr<cucascade::data_batch> const& batch,
             cucascade::memory::memory_space* space)
{
  sirius::convertible_data_batch convertible(batch);
  REQUIRE(convertible.convert({space}, env().stream, *env().mgr, true).has_value());
}

std::shared_ptr<cucascade::data_batch> as_batch(
  std::unique_ptr<cucascade::idata_representation> representation)
{
  return cucascade::data_batch::make(sirius::get_next_batch_id(), std::move(representation));
}

compressed_source make_compressed(source_data data)
{
  sirius::compression::spill_compression_settings settings;
  settings.enabled                 = true;
  settings.min_batch_bytes         = 0;
  settings.max_compressed_fraction = 2.0;
  sirius::compression::set_spill_compression_settings(settings);

  auto stream = utils::default_stream();
  auto batch =
    sirius::make_data_batch(std::make_unique<cudf::table>(std::move(data.device_columns)),
                            *env().gpu,
                            stream,
                            sirius::telemetry::batch_telemetry_info{});
  data.device_columns.clear();
  batch->to_mutable().get_data()->record_writer_event(stream);

  // Clones share the pinned blob and the file, so both outlive the batch's own moves.
  std::unique_ptr<cucascade::idata_representation> host_owner;
  std::unique_ptr<cucascade::idata_representation> disk_owner;
  move_to(batch, env().host);
  {
    auto ro = batch->to_read_only();
    REQUIRE(dynamic_cast<sirius::compressed_host_representation const*>(ro.get_data()) != nullptr);
    host_owner = ro.get_data()->clone(stream);
  }
  move_to(batch, env().disk);
  {
    auto ro = batch->to_read_only();
    REQUIRE(dynamic_cast<sirius::compressed_disk_representation const*>(ro.get_data()) != nullptr);
    disk_owner = ro.get_data()->clone(stream);
  }
  sirius::compression::set_spill_compression_settings({});

  // The device tier: the table rebuilt on the device from the pinned payload.
  auto const& host   = host_owner->cast<sirius::compressed_host_representation>();
  auto const& blocks = host.payload();
  auto blob          = std::make_shared<sirius::compressed_device_blob>();
  std::string error;
  simpatico::payload_fetch_fn fetch =
    [&blocks](std::uint64_t off, std::size_t sz, void* dst, ::cuda::stream_ref s) {
      sirius::copy_pinned_blocks_to_device(blocks, off, dst, sz, s);
    };
  blob->table = simpatico::read_compressed_table_from_memory(
    host.header(), fetch, stream, utils::get_resource_ref(*env().gpu), &error);
  REQUIRE(error.empty());
  stream.sync();
  auto device = std::make_unique<sirius::compressed_device_representation>(
    *env().gpu,
    std::move(blob),
    host.column_names(),
    host.get_size_in_bytes(),
    host.get_uncompressed_data_size_in_bytes(),
    host.num_rows());

  compressed_source out;
  out.data = std::move(data);
  out.tiers.push_back({"device", as_batch(std::move(device))});
  out.tiers.push_back({"host", as_batch(std::move(host_owner))});
  out.tiers.push_back({"disk", as_batch(std::move(disk_owner))});
  return out;
}

compressed_source& main_batch()
{
  static compressed_source s = make_compressed(main_source());
  return s;
}

compressed_source& small_batch()
{
  static compressed_source s = make_compressed(small_source());
  return s;
}

/// Whether the plan of @p column has a node running @p op.
bool has_op(simpatico::hpln_column_description const& column, std::string const& op)
{
  return std::any_of(
    column.nodes.begin(), column.nodes.end(), [&](auto const& node) { return node.op == op; });
}

std::string plan_ops(simpatico::hpln_column_description const& column)
{
  std::string ops;
  for (auto const& node : column.nodes) {
    ops += node.op + " ";
  }
  return ops;
}

std::shared_ptr<const simpatico::hpln_description> describe_tier(tier const& t)
{
  auto ro = t.batch->to_read_only();
  auto const* data =
    dynamic_cast<sirius::simpatico_compressed_representation const*>(ro.get_data());
  REQUIRE(data != nullptr);
  return sirius::compression::describe(*data);
}

// ── Destinations ─────────────────────────────────────────────────────────────

/// Device buffers of one destination column, plus the column_destination describing them.
struct dst_column {
  rmm::device_buffer data;
  rmm::device_buffer mask;
  rmm::device_buffer offsets;
  cucascade::column_destination dest;
};

rmm::device_buffer filled_buffer(std::size_t bytes)
{
  auto stream = utils::default_stream();
  rmm::device_buffer buffer(bytes, stream, utils::get_resource_ref(*env().gpu));
  if (bytes > 0) {
    REQUIRE(cudaMemsetAsync(buffer.data(), kFill, bytes, stream.get()) == cudaSuccess);
  }
  return buffer;
}

/// Capacities of one destination column. A zero capacity leaves that slot absent.
struct dst_spec {
  std::size_t data_bytes{0};
  std::size_t mask_bytes{0};
  std::size_t offsets_bytes{0};
  std::uint8_t offset_width{4};
  std::size_t dst_row_offset{0};
  std::size_t dst_byte_offset{0};
};

dst_column make_dst(host_column const& src, dst_spec const& spec)
{
  dst_column d;
  d.data                 = filled_buffer(spec.data_bytes);
  d.mask                 = filled_buffer(spec.mask_bytes);
  d.offsets              = filled_buffer(spec.offsets_bytes);
  d.dest.type_id         = static_cast<std::int32_t>(src.type.id());
  d.dest.data            = {d.data.data(), d.data.size()};
  d.dest.null_mask       = {d.mask.data(), d.mask.size()};
  d.dest.offsets         = {d.offsets.data(), d.offsets.size()};
  d.dest.offset_width    = spec.offset_width;
  d.dest.dst_row_offset  = spec.dst_row_offset;
  d.dest.dst_byte_offset = spec.dst_byte_offset;
  return d;
}

/// Capacities with room for @p rows rows appended at the given position, and some to spare.
dst_spec roomy_spec(host_column const& src,
                    std::size_t rows,
                    bool with_mask              = true,
                    std::uint8_t offset_width   = 4,
                    std::size_t dst_row_offset  = 0,
                    std::size_t dst_byte_offset = 0)
{
  dst_spec spec;
  spec.offset_width    = offset_width;
  spec.dst_row_offset  = dst_row_offset;
  spec.dst_byte_offset = dst_byte_offset;
  auto const dst_rows  = dst_row_offset + rows;
  if (src.is_string()) {
    spec.data_bytes    = dst_byte_offset + rows * kMaxStringBytes + 16;
    spec.offsets_bytes = (dst_rows + 1) * offset_width + 16;
  } else {
    spec.data_bytes = dst_rows * src.width + 16;
  }
  if (with_mask) { spec.mask_bytes = (dst_rows + 31) / 32 * 4 + 8; }
  return spec;
}

std::vector<dst_column> roomy_dsts(source_data const& source,
                                   std::span<const std::size_t> columns,
                                   std::size_t rows,
                                   std::size_t dst_row_offset  = 0,
                                   std::size_t dst_byte_offset = 0)
{
  std::vector<dst_column> dsts;
  for (auto c : columns) {
    auto const& src = source.columns[c];
    dsts.push_back(make_dst(src, roomy_spec(src, rows, true, 4, dst_row_offset, dst_byte_offset)));
  }
  return dsts;
}

std::vector<std::uint8_t> to_host(rmm::device_buffer const& buffer)
{
  std::vector<std::uint8_t> host(buffer.size());
  if (!host.empty()) {
    REQUIRE(cudaMemcpy(host.data(), buffer.data(), host.size(), cudaMemcpyDeviceToHost) ==
            cudaSuccess);
  }
  return host;
}

bool all_fill(std::vector<std::uint8_t> const& bytes, std::size_t begin, std::size_t end)
{
  for (std::size_t i = begin; i < std::min(end, bytes.size()); ++i) {
    if (bytes[i] != kFill) { return false; }
  }
  return true;
}

std::int64_t offset_at(std::vector<std::uint8_t> const& offsets, std::uint8_t width, std::size_t i)
{
  if (width == 4) {
    std::int32_t v;
    std::memcpy(&v, offsets.data() + i * 4, 4);
    return v;
  }
  std::int64_t v;
  std::memcpy(&v, offsets.data() + i * 8, 8);
  return v;
}

bool mask_bit(std::vector<std::uint8_t> const& mask, std::size_t bit)
{
  return ((mask[bit / 8] >> (bit % 8)) & 1u) != 0;
}

// ── Reading and checking ─────────────────────────────────────────────────────

cucascade::row_range_result read_rows(tier const& t,
                                      std::span<const std::size_t> columns,
                                      std::size_t row_begin,
                                      std::size_t max_rows,
                                      std::vector<dst_column> const& dsts,
                                      reader_state* state)
{
  std::vector<cucascade::column_destination> dest;
  dest.reserve(dsts.size());
  for (auto const& d : dsts) {
    dest.push_back(d.dest);
  }
  cucascade::row_range_request request;
  request.columns   = columns;
  request.row_begin = row_begin;
  request.max_rows  = max_rows;
  auto ro           = t.batch->to_read_only();
  return cucascade::read_rows_into(
    registry(), ro, request, dest, utils::default_stream(), nullptr, state);
}

/// Reader-contract rule 1, from the oracle: the largest k every destination slot holds.
std::size_t expected_rows(source_data const& source,
                          std::span<const std::size_t> columns,
                          std::vector<dst_column> const& dsts,
                          std::size_t row_begin,
                          std::size_t max_rows)
{
  std::size_t k = std::min(max_rows, source.rows - row_begin);
  auto lower    = [&k](std::size_t capacity_rows, std::size_t used_rows) {
    k = std::min(k, capacity_rows > used_rows ? capacity_rows - used_rows : 0);
  };
  for (std::size_t i = 0; i < columns.size(); ++i) {
    auto const& src = source.columns[columns[i]];
    auto const& d   = dsts[i].dest;
    if (src.is_string()) {
      lower(d.offsets.capacity_bytes / d.offset_width, d.dst_row_offset + 1);
      while (k > 0 && d.dst_byte_offset + src.chars_bytes(row_begin, k) > d.data.capacity_bytes) {
        --k;
      }
    } else {
      lower(d.data.capacity_bytes / src.width, d.dst_row_offset);
    }
    if (d.null_mask.data != nullptr) {
      lower(d.null_mask.capacity_bytes / 4 * 32, d.dst_row_offset);
    }
  }
  return k;
}

/**
 * @brief Check that @p d holds source rows [row_begin, row_begin + k) appended at its append
 * position, and that the bytes and bits before that position still hold the fill pattern. With
 * @p check_tail the same is required of everything after the written range.
 */
void verify_read(host_column const& src,
                 dst_column const& d,
                 std::size_t row_begin,
                 std::size_t k,
                 bool check_tail)
{
  utils::default_stream().sync();
  auto const data    = to_host(d.data);
  auto const mask    = to_host(d.mask);
  auto const offsets = to_host(d.offsets);
  auto const dro     = d.dest.dst_row_offset;
  auto const dbo     = d.dest.dst_byte_offset;

  if (src.is_string()) {
    auto const width = d.dest.offset_width;
    auto const bytes = src.chars_bytes(row_begin, k);
    if (k > 0) {
      bool offsets_ok = true;
      for (std::size_t i = 0; i <= k; ++i) {
        offsets_ok = offsets_ok && offset_at(offsets, width, dro + i) ==
                                     static_cast<std::int64_t>(dbo + src.chars_bytes(row_begin, i));
      }
      CHECK(offsets_ok);
      std::string expected;
      for (std::size_t i = 0; i < k; ++i) {
        expected += src.strings[row_begin + i];
      }
      REQUIRE(dbo + bytes <= data.size());
      CHECK(std::memcmp(data.data() + dbo, expected.data(), bytes) == 0);
    }
    CHECK(all_fill(offsets, 0, dro * width));
    CHECK(all_fill(data, 0, dbo));
    if (check_tail) {
      CHECK(all_fill(offsets, (dro + (k > 0 ? k + 1 : 0)) * width, offsets.size()));
      CHECK(all_fill(data, dbo + bytes, data.size()));
    }
  } else {
    bool values_ok = true;
    for (std::size_t i = 0; i < k; ++i) {
      if (!src.is_valid(row_begin + i)) { continue; }
      values_ok = values_ok && std::memcmp(data.data() + (dro + i) * src.width,
                                           src.bytes.data() + (row_begin + i) * src.width,
                                           src.width) == 0;
    }
    CHECK(values_ok);
    CHECK(all_fill(data, 0, dro * src.width));
    CHECK(all_fill(data, (dro + k) * src.width, data.size()));
  }

  bool bits_ok = true;
  bool kept_ok = true;
  for (std::size_t bit = 0; bit < mask.size() * 8; ++bit) {
    if (bit >= dro && bit < dro + k) {
      bits_ok = bits_ok && mask_bit(mask, bit) == src.is_valid(row_begin + (bit - dro));
    } else if (bit < dro || check_tail) {
      kept_ok = kept_ok && mask_bit(mask, bit) == (((kFill >> (bit % 8)) & 1u) != 0);
    }
  }
  CHECK(bits_ok);
  CHECK(kept_ok);
}

/// Check a read's result against the oracle, and every destination's contents.
void check_read(source_data const& source,
                std::span<const std::size_t> columns,
                std::vector<dst_column> const& dsts,
                std::size_t row_begin,
                std::size_t max_rows,
                cucascade::row_range_result const& result)
{
  auto const k = expected_rows(source, columns, dsts, row_begin, max_rows);
  REQUIRE(result.rows_written == k);
  REQUIRE(result.source_exhausted == (row_begin + k == source.rows));
  REQUIRE(result.data_bytes_written.size() == columns.size());
  auto const strings = static_cast<std::size_t>(std::count_if(
    columns.begin(), columns.end(), [&](std::size_t c) { return source.columns[c].is_string(); }));
  for (std::size_t i = 0; i < columns.size(); ++i) {
    INFO("column " << columns[i]);
    auto const& src = source.columns[columns[i]];
    CHECK(result.data_bytes_written[i] ==
          (src.is_string() ? src.chars_bytes(row_begin, k) : k * src.width));
    // When a later STRING column lowers k, the STRING columns decoded before it have already
    // written past the final range (documented reader behaviour): check their prefix only.
    verify_read(src, dsts[i], row_begin, k, !src.is_string() || strings <= 1);
  }
}

cucascade::row_range_result read_and_check(compressed_source const& source,
                                           tier const& t,
                                           std::span<const std::size_t> columns,
                                           std::vector<dst_column> const& dsts,
                                           std::size_t row_begin,
                                           std::size_t max_rows,
                                           reader_state* state)
{
  auto result = read_rows(t, columns, row_begin, max_rows, dsts, state);
  check_read(source.data, columns, dsts, row_begin, max_rows, result);
  return result;
}

/**
 * @brief Read the whole batch window by window, appending to small destinations and flushing
 * them to the host when they are full, and compare the concatenation with the oracle.
 */
void read_whole_batch(compressed_source const& source,
                      tier const& t,
                      std::vector<std::size_t> const& columns,
                      std::size_t capacity_rows,
                      std::size_t chars_capacity)
{
  constexpr std::size_t kWindows[] = {1, 37, 1023, 1024, 1025, 300, 2500, 5000, 64, 777};
  auto const& data                 = source.data;

  std::vector<dst_column> dsts;
  for (std::size_t i = 0; i < columns.size(); ++i) {
    auto const& src = data.columns[columns[i]];
    dst_spec spec;
    spec.offset_width = i % 2 == 0 ? 4 : 8;
    spec.mask_bytes   = (capacity_rows + 31) / 32 * 4;
    if (src.is_string()) {
      spec.data_bytes    = chars_capacity;
      spec.offsets_bytes = (capacity_rows + 1) * spec.offset_width;
    } else {
      spec.data_bytes = capacity_rows * src.width;
    }
    dsts.push_back(make_dst(src, spec));
  }

  // What the caller collected: per requested column, values / strings and validity.
  std::vector<std::vector<std::uint8_t>> got_bytes(columns.size());
  std::vector<std::vector<std::string>> got_strings(columns.size());
  std::vector<std::vector<bool>> got_valid(columns.size());
  std::size_t filled = 0;
  auto flush         = [&] {
    utils::default_stream().sync();
    for (std::size_t i = 0; i < columns.size(); ++i) {
      auto const& src    = data.columns[columns[i]];
      auto const host    = to_host(dsts[i].data);
      auto const mask    = to_host(dsts[i].mask);
      auto const offsets = to_host(dsts[i].offsets);
      for (std::size_t r = 0; r < filled; ++r) {
        got_valid[i].push_back(mask_bit(mask, r));
        if (src.is_string()) {
          auto const begin = offset_at(offsets, dsts[i].dest.offset_width, r);
          auto const end   = offset_at(offsets, dsts[i].dest.offset_width, r + 1);
          REQUIRE(begin >= 0);
          REQUIRE(begin <= end);
          REQUIRE(static_cast<std::size_t>(end) <= host.size());
          got_strings[i].emplace_back(host.begin() + begin, host.begin() + end);
        }
      }
      if (!src.is_string()) {
        got_bytes[i].insert(got_bytes[i].end(), host.begin(), host.begin() + filled * src.width);
      }
      dsts[i].dest.dst_row_offset  = 0;
      dsts[i].dest.dst_byte_offset = 0;
    }
    filled = 0;
  };

  reader_state state;
  std::size_t row   = 0;
  std::size_t reads = 0;
  while (true) {
    auto const max_rows = kWindows[reads++ % std::size(kWindows)];
    auto const expected = expected_rows(data, columns, dsts, row, max_rows);
    auto const result   = read_rows(t, columns, row, max_rows, dsts, &state);
    REQUIRE(result.rows_written == expected);
    REQUIRE(result.source_exhausted == (row + expected == data.rows));
    REQUIRE(result.data_bytes_written.size() == columns.size());
    for (std::size_t i = 0; i < columns.size(); ++i) {
      auto const& src = data.columns[columns[i]];
      REQUIRE(result.data_bytes_written[i] ==
              (src.is_string() ? src.chars_bytes(row, expected) : expected * src.width));
      dsts[i].dest.dst_row_offset += expected;
      dsts[i].dest.dst_byte_offset += src.is_string() ? result.data_bytes_written[i] : 0;
    }
    row += expected;
    filled += expected;
    if (result.source_exhausted) { break; }
    if (expected == 0 || filled == capacity_rows) {
      // An empty destination always holds at least one row.
      REQUIRE(filled > 0);
      flush();
    }
  }
  flush();
  REQUIRE(row == data.rows);
  REQUIRE(state != nullptr);
  CHECK(dynamic_cast<sirius::compression::compressed_reader_state*>(state.get()) != nullptr);

  for (std::size_t i = 0; i < columns.size(); ++i) {
    INFO("column " << columns[i]);
    auto const& src = data.columns[columns[i]];
    REQUIRE(got_valid[i].size() == data.rows);
    bool valid_ok  = true;
    bool values_ok = true;
    for (std::size_t r = 0; r < data.rows; ++r) {
      valid_ok = valid_ok && got_valid[i][r] == src.is_valid(r);
      if (!src.is_valid(r)) { continue; }
      if (src.is_string()) {
        values_ok = values_ok && got_strings[i][r] == src.strings[r];
      } else {
        values_ok = values_ok && std::memcmp(got_bytes[i].data() + r * src.width,
                                             src.bytes.data() + r * src.width,
                                             src.width) == 0;
      }
    }
    CHECK(valid_ok);
    CHECK(values_ok);
  }
}

std::vector<std::size_t> all_columns(source_data const& source)
{
  std::vector<std::size_t> columns(source.columns.size());
  for (std::size_t i = 0; i < columns.size(); ++i) {
    columns[i] = i;
  }
  return columns;
}

}  // namespace

TEST_CASE("compressed readers are registered and the batch uses every spill plan",
          "[compressed_row_range_readers]")
{
  if (!has_gpu()) { return; }
  auto const& readers = registry();
  REQUIRE(readers.has_reader<sirius::compressed_device_representation>());
  REQUIRE(readers.has_reader<sirius::compressed_host_representation>());
  REQUIRE(readers.has_reader<sirius::compressed_disk_representation>());

  for (auto const* source : {&main_batch(), &small_batch()}) {
    for (auto const& t : source->tiers) {
      INFO(t.name);
      auto ro = t.batch->to_read_only();
      REQUIRE(readers.has_reader(*ro.get_data()));
    }
  }

  // The plans the batches were really spilled with.
  for (auto const& t : main_batch().tiers) {
    INFO(t.name);
    auto const description = describe_tier(t);
    auto const& columns    = description->columns;
    REQUIRE(columns.size() == main_batch().data.columns.size());
    auto ops = [&](std::size_t c) { return plan_ops(columns[c]); };
    for (auto c : {kSorted, kSortedAllValid}) {
      INFO("column " << c << ": " << ops(c));
      CHECK(has_op(columns[c], "delta"));
      CHECK_FALSE(has_op(columns[c], "rle"));
    }
    INFO("runs: " << ops(kRuns));
    CHECK(has_op(columns[kRuns], "delta"));
    CHECK(has_op(columns[kRuns], "rle"));
    for (auto c : {kRandom, kRandomNulls}) {
      INFO("column " << c << ": " << ops(c));
      CHECK(has_op(columns[c], "bitpack"));
      CHECK_FALSE(has_op(columns[c], "delta"));
    }
    for (auto c : {kDict, kDictNulls}) {
      INFO("column " << c << ": " << ops(c));
      CHECK(has_op(columns[c], "dictionary"));
    }
    for (auto c : {kSplit, kSplitNulls}) {
      INFO("column " << c << ": " << ops(c));
      CHECK(has_op(columns[c], "str_split"));
    }
    for (std::size_t c = 0; c < columns.size(); ++c) {
      CHECK(columns[c].num_rows == static_cast<std::int64_t>(kMainRows));
      CHECK((columns[c].null_count > 0) == main_batch().data.columns[c].has_nulls());
    }
  }
  for (auto const& t : small_batch().tiers) {
    INFO(t.name);
    auto const description = describe_tier(t);
    auto const& columns    = description->columns;
    REQUIRE(columns.size() == small_batch().data.columns.size());
    for (auto c : {kSmallString, kSmallStringNulls}) {
      INFO("column " << c << ": " << plan_ops(columns[c]));
      CHECK(has_op(columns[c], "identity"));
      CHECK_FALSE(has_op(columns[c], "dictionary"));
      CHECK_FALSE(has_op(columns[c], "str_split"));
    }
    CHECK(has_op(columns[kSmallSorted], "delta"));
  }
}

TEST_CASE("compressed readers deliver the whole batch window by window",
          "[compressed_row_range_readers]")
{
  if (!has_gpu()) { return; }
  for (auto const& t : main_batch().tiers) {
    INFO(t.name);
    read_whole_batch(main_batch(), t, all_columns(main_batch().data), 2600, 3000);
  }
  for (auto const& t : small_batch().tiers) {
    INFO("small batch, " << t.name);
    read_whole_batch(small_batch(), t, all_columns(small_batch().data), 700, 1500);
  }
}

TEST_CASE("compressed readers deliver a column subset in the requested order",
          "[compressed_row_range_readers]")
{
  if (!has_gpu()) { return; }
  for (auto const& t : main_batch().tiers) {
    INFO(t.name);
    read_whole_batch(main_batch(), t, {kDict, kSorted}, 4000, 20000);
    read_whole_batch(main_batch(), t, {kSplitNulls, kRandomNulls, kRuns}, 1500, 2000);
  }
  for (auto const& t : small_batch().tiers) {
    INFO("small batch, " << t.name);
    read_whole_batch(small_batch(), t, {kSmallSorted, kSmallString}, 1000, 900);
  }
}

TEST_CASE("compressed readers truncate to the destination budget", "[compressed_row_range_readers]")
{
  if (!has_gpu()) { return; }
  auto const& source = main_batch();
  auto const& data   = source.data;
  for (auto const& t : source.tiers) {
    INFO(t.name);
    reader_state state;

    SECTION("fixed-width data capacity")
    {
      const std::size_t columns[] = {kSorted, kRandomNulls};
      auto dsts                   = roomy_dsts(data, columns, 3000);
      auto spec                   = roomy_spec(data.columns[kSorted], 3000);
      spec.data_bytes             = 100 * 8 + 7;
      dsts[0]                     = make_dst(data.columns[kSorted], spec);
      auto result                 = read_and_check(source, t, columns, dsts, 1000, 3000, &state);
      CHECK(result.rows_written == 100);
    }
    SECTION("null mask capacity")
    {
      const std::size_t columns[] = {kRandomNulls, kRuns};
      auto dsts                   = roomy_dsts(data, columns, 3000, 3);
      auto spec                   = roomy_spec(data.columns[kRandomNulls], 3000, true, 4, 3);
      spec.mask_bytes             = 8;  // 64 bits, 3 of them already used
      dsts[0]                     = make_dst(data.columns[kRandomNulls], spec);
      auto result                 = read_and_check(source, t, columns, dsts, 2040, 3000, &state);
      CHECK(result.rows_written == 61);
    }
    SECTION("STRING offsets capacity")
    {
      for (std::uint8_t width : {std::uint8_t{4}, std::uint8_t{8}}) {
        const std::size_t columns[] = {kDictNulls, kSorted};
        auto dsts                   = roomy_dsts(data, columns, 3000, 2);
        auto spec                   = roomy_spec(data.columns[kDictNulls], 3000, true, width, 2);
        spec.offsets_bytes = 53 * width + (width - 1);  // offsets of rows 2..51, plus the end
        dsts[0]            = make_dst(data.columns[kDictNulls], spec);
        auto result        = read_and_check(source, t, columns, dsts, 4090, 3000, &state);
        CHECK(result.rows_written == 50);
      }
    }
    SECTION("STRING chars budget")
    {
      for (auto c : {kSplit, kDict, kSplitNulls, kDictNulls}) {
        INFO("column " << c);
        for (std::size_t row_begin : {std::size_t{0}, std::size_t{1000}, std::size_t{5119}}) {
          // Room for 40 rows and all but the last byte of the next non-empty one.
          std::size_t rows = 40;
          while (data.columns[c].chars_bytes(row_begin + rows, 1) == 0) {
            ++rows;
          }
          const std::size_t columns[] = {kRandom, c, kRandomNulls};
          auto dsts                   = roomy_dsts(data, columns, 3000, 5, 11);
          auto spec                   = roomy_spec(data.columns[c], 3000, true, 4, 5, 11);
          spec.data_bytes             = 11 + data.columns[c].chars_bytes(row_begin, rows + 1) - 1;
          dsts[1]                     = make_dst(data.columns[c], spec);
          auto result = read_and_check(source, t, columns, dsts, row_begin, 3000, &state);
          CHECK(result.rows_written == rows);
          CHECK_FALSE(result.source_exhausted);
        }
      }
    }
    SECTION("several STRING columns agree on the smallest budget")
    {
      // Each order: the tight column first, then last (it lowers k after the others decoded).
      for (auto const& order : {std::vector<std::size_t>{kSplit, kDict, kSorted, kSplitNulls},
                                std::vector<std::size_t>{kDict, kSplitNulls, kSorted, kSplit}}) {
        auto dsts = roomy_dsts(data, order, 3000, 1, 4);
        auto tight =
          static_cast<std::size_t>(std::find(order.begin(), order.end(), kSplit) - order.begin());
        auto spec       = roomy_spec(data.columns[kSplit], 3000, true, 8, 1, 4);
        spec.data_bytes = 4 + data.columns[kSplit].chars_bytes(900, 300) + 3;
        dsts[tight]     = make_dst(data.columns[kSplit], spec);
        auto result     = read_and_check(source, t, order, dsts, 900, 2500, &state);
        CHECK(result.rows_written >= 300);
        CHECK(result.rows_written < 310);
      }
    }
    SECTION("a single row that does not fit")
    {
      {
        const std::size_t columns[] = {kSorted};
        auto spec                   = roomy_spec(data.columns[kSorted], 10, true, 4, 4);
        spec.data_bytes             = 4 * 8 + 7;
        std::vector<dst_column> dsts;
        dsts.push_back(make_dst(data.columns[kSorted], spec));
        auto result = read_and_check(source, t, columns, dsts, 10, 100, &state);
        CHECK(result.rows_written == 0);
        CHECK_FALSE(result.source_exhausted);
      }
      for (auto c : {kSplit, kDict}) {
        INFO("column " << c);
        std::size_t row_begin = 2000;
        while (data.columns[c].chars_bytes(row_begin, 1) < 2) {
          ++row_begin;
        }
        const std::size_t columns[] = {c, kRandomNulls};
        auto dsts                   = roomy_dsts(data, columns, 100, 0, 6);
        auto spec                   = roomy_spec(data.columns[c], 100, true, 4, 0, 6);
        spec.data_bytes             = 6 + data.columns[c].chars_bytes(row_begin, 1) - 1;
        dsts[0]                     = make_dst(data.columns[c], spec);
        auto result = read_and_check(source, t, columns, dsts, row_begin, 100, &state);
        CHECK(result.rows_written == 0);
        CHECK_FALSE(result.source_exhausted);
      }
    }
  }
}

TEST_CASE("compressed readers report an exhausted source", "[compressed_row_range_readers]")
{
  if (!has_gpu()) { return; }
  for (auto const* source : {&main_batch(), &small_batch()}) {
    auto const& data   = source->data;
    auto const columns = all_columns(data);
    for (auto const& t : source->tiers) {
      INFO(t.name << ", " << data.rows << " rows");
      reader_state state;
      {
        // The last rows: fewer than asked for.
        auto dsts   = roomy_dsts(data, columns, 200);
        auto result = read_and_check(*source, t, columns, dsts, data.rows - 130, 200, &state);
        CHECK(result.rows_written == 130);
        CHECK(result.source_exhausted);
      }
      {
        // Exactly the last rows.
        auto dsts   = roomy_dsts(data, columns, 200);
        auto result = read_and_check(*source, t, columns, dsts, data.rows - 1, 1, &state);
        CHECK(result.rows_written == 1);
        CHECK(result.source_exhausted);
      }
      {
        // Nothing left: nothing is written.
        auto dsts   = roomy_dsts(data, columns, 200, 3, 5);
        auto result = read_and_check(*source, t, columns, dsts, data.rows, 200, &state);
        CHECK(result.rows_written == 0);
        CHECK(result.source_exhausted);
      }
      {
        // max_rows == 0 is not exhaustion.
        auto dsts   = roomy_dsts(data, columns, 200);
        auto result = read_and_check(*source, t, columns, dsts, 17, 0, &state);
        CHECK(result.rows_written == 0);
        CHECK_FALSE(result.source_exhausted);
      }
      {
        auto dsts = roomy_dsts(data, columns, 200);
        CHECK_THROWS_AS(read_rows(t, columns, data.rows + 1, 10, dsts, &state), std::out_of_range);
        for (std::size_t i = 0; i < columns.size(); ++i) {
          verify_read(data.columns[columns[i]], dsts[i], 0, 0, true);
        }
      }
    }
  }
}

TEST_CASE("compressed readers rebase STRING offsets at the append position",
          "[compressed_row_range_readers]")
{
  if (!has_gpu()) { return; }
  struct probe {
    std::size_t row_begin, rows, dst_row_offset, dst_byte_offset;
  };
  // Mid-chunk starts, ranges crossing 1024-row chunks, and several chunks at once.
  const probe probes[] = {
    {0, 10, 0, 0}, {1000, 100, 5, 13}, {1023, 2, 31, 1}, {2048, 1024, 1, 777}, {3333, 2600, 64, 9}};
  auto run = [&](compressed_source const& source, std::initializer_list<std::size_t> strings) {
    auto const& data = source.data;
    for (auto const& t : source.tiers) {
      reader_state state;
      for (auto c : strings) {
        for (std::uint8_t width : {std::uint8_t{4}, std::uint8_t{8}}) {
          for (auto const& p : probes) {
            if (p.row_begin + p.rows > data.rows) { continue; }
            INFO(t.name << ", column " << c << ", offset width " << int{width} << ", rows ["
                        << p.row_begin << ", +" << p.rows << ")");
            const std::size_t columns[] = {c};
            std::vector<dst_column> dsts;
            dsts.push_back(make_dst(
              data.columns[c],
              roomy_spec(
                data.columns[c], p.rows, true, width, p.dst_row_offset, p.dst_byte_offset)));
            auto result = read_and_check(source, t, columns, dsts, p.row_begin, p.rows, &state);
            CHECK(result.rows_written == p.rows);
          }
        }
      }
    }
  };
  run(main_batch(), {kDict, kSplit, kDictNulls, kSplitNulls});
  run(small_batch(), {kSmallString, kSmallStringNulls});
}

TEST_CASE("compressed readers write null bits at unaligned positions",
          "[compressed_row_range_readers]")
{
  if (!has_gpu()) { return; }
  auto run = [&](compressed_source const& source, std::vector<std::size_t> const& columns) {
    auto const& data = source.data;
    for (auto const& t : source.tiers) {
      reader_state state;
      for (std::size_t dst_row_offset : {0, 1, 31, 33, 45}) {
        for (std::size_t row_begin : {0, 7, 1019, 2047}) {
          for (std::size_t rows : {1, 3, 64, 70, 900}) {
            INFO(t.name << ", rows [" << row_begin << ", +" << rows << ") at destination row "
                        << dst_row_offset);
            // One STRING column per read, so everything past the range must be untouched.
            for (auto c : columns) {
              const std::size_t one[] = {c, columns.front()};
              auto count   = data.columns[c].is_string() || c == columns.front() ? 1u : 2u;
              auto request = std::span<const std::size_t>(one, count);
              auto dsts    = roomy_dsts(data, request, rows, dst_row_offset, 3);
              auto result  = read_and_check(source, t, request, dsts, row_begin, rows, &state);
              CHECK(result.rows_written == rows);
            }
          }
        }
      }
    }
  };
  run(main_batch(), {kRandomNulls, kDictNulls, kSplitNulls, kSortedAllValid, kSorted, kDict});
  run(small_batch(), {kSmallSorted, kSmallStringNulls, kSmallString});
}

TEST_CASE("compressed readers null-mask slot rules", "[compressed_row_range_readers]")
{
  if (!has_gpu()) { return; }
  auto const& source = main_batch();
  auto const& data   = source.data;
  for (auto const& t : source.tiers) {
    INFO(t.name);
    reader_state state;
    // No mask slot: fine for a source without nulls, mask or not.
    for (auto c : {kSorted, kSortedAllValid, kDict, kSplit}) {
      INFO("column " << c);
      const std::size_t columns[] = {c};
      std::vector<dst_column> dsts;
      dsts.push_back(make_dst(data.columns[c], roomy_spec(data.columns[c], 1500, false, 4, 2, 5)));
      auto result = read_and_check(source, t, columns, dsts, 700, 1500, &state);
      CHECK(result.rows_written == 1500);
    }
    // No mask slot and a source with nulls: refused, and nothing is written.
    for (auto c : {kRandomNulls, kDictNulls, kSplitNulls}) {
      INFO("column " << c);
      const std::size_t columns[] = {kSorted, c};
      auto dsts                   = roomy_dsts(data, columns, 100);
      dsts[1] = make_dst(data.columns[c], roomy_spec(data.columns[c], 100, false));
      CHECK_THROWS_AS(read_rows(t, columns, 0, 100, dsts, &state),
                      cucascade::row_range_unsupported_error);
      verify_read(data.columns[kSorted], dsts[0], 0, 0, true);
      verify_read(data.columns[c], dsts[1], 0, 0, true);
    }
    // A mask slot for a source without a mask is filled with 1s (checked by verify_read).
    {
      const std::size_t columns[] = {kRuns, kDict};
      auto dsts                   = roomy_dsts(data, columns, 2000, 13);
      auto result                 = read_and_check(source, t, columns, dsts, 1500, 2000, &state);
      CHECK(result.rows_written == 2000);
    }
  }
}

TEST_CASE("compressed readers validate the request", "[compressed_row_range_readers]")
{
  if (!has_gpu()) { return; }
  auto const& source = main_batch();
  auto const& data   = source.data;
  for (auto const& t : source.tiers) {
    INFO(t.name);
    reader_state state;
    auto untouched = [&](std::span<const std::size_t> columns,
                         std::vector<dst_column> const& dsts) {
      for (std::size_t i = 0; i < columns.size(); ++i) {
        if (columns[i] < data.columns.size()) {
          verify_read(data.columns[columns[i]], dsts[i], 0, 0, true);
        }
      }
    };
    {
      // Source column index out of range.
      const std::size_t columns[] = {kSorted, data.columns.size()};
      const std::size_t shape[]   = {kSorted, kSorted};
      auto dsts                   = roomy_dsts(data, shape, 10);
      CHECK_THROWS_AS(read_rows(t, columns, 0, 10, dsts, &state), std::out_of_range);
      untouched(shape, dsts);
    }
    {
      // Destination type differs from the source column's.
      const std::size_t columns[] = {kSorted, kRuns};
      auto dsts                   = roomy_dsts(data, columns, 10);
      dsts[1].dest.type_id        = static_cast<std::int32_t>(cudf::type_id::INT64);
      CHECK_THROWS_AS(read_rows(t, columns, 0, 10, dsts, &state), std::invalid_argument);
      untouched(columns, dsts);
      const std::size_t strings[] = {kDict};
      auto string_dsts            = roomy_dsts(data, strings, 10);
      string_dsts[0].dest.type_id = static_cast<std::int32_t>(cudf::type_id::INT32);
      CHECK_THROWS_AS(read_rows(t, strings, 0, 10, string_dsts, &state), std::invalid_argument);
      untouched(strings, string_dsts);
    }
    {
      // STRING destination without an offsets slot, or with a bad offset width.
      const std::size_t columns[] = {kSorted, kSplit};
      auto dsts                   = roomy_dsts(data, columns, 10);
      auto spec                   = roomy_spec(data.columns[kSplit], 10);
      spec.offsets_bytes          = 0;
      dsts[1]                     = make_dst(data.columns[kSplit], spec);
      CHECK_THROWS_AS(read_rows(t, columns, 0, 10, dsts, &state), std::invalid_argument);
      untouched(columns, dsts);

      dsts[1] = make_dst(data.columns[kSplit], roomy_spec(data.columns[kSplit], 10));
      dsts[1].dest.offset_width = 2;
      CHECK_THROWS_AS(read_rows(t, columns, 0, 10, dsts, &state), std::invalid_argument);
      untouched(columns, dsts);
    }
    {
      // Fixed-width destination without a data slot.
      const std::size_t columns[] = {kDict, kRandom};
      auto dsts                   = roomy_dsts(data, columns, 10);
      auto spec                   = roomy_spec(data.columns[kRandom], 10);
      spec.data_bytes             = 0;
      dsts[1]                     = make_dst(data.columns[kRandom], spec);
      CHECK_THROWS_AS(read_rows(t, columns, 0, 10, dsts, &state), std::invalid_argument);
      untouched(columns, dsts);
    }
    {
      // No requested columns: rows are still counted against the source.
      std::vector<dst_column> dsts;
      auto result = read_rows(t, {}, data.rows - 5, 100, dsts, &state);
      CHECK(result.rows_written == 5);
      CHECK(result.source_exhausted);
      CHECK(result.data_bytes_written.empty());
    }
    // The state survives refused requests: a good read still works.
    const std::size_t columns[] = {kSplitNulls, kSorted, kRandomNulls};
    auto dsts                   = roomy_dsts(data, columns, 1200, 9, 2);
    read_and_check(source, t, columns, dsts, 5000, 1200, &state);
  }
}

TEST_CASE("compressed readers keep, rebuild or do without the reader state",
          "[compressed_row_range_readers]")
{
  if (!has_gpu()) { return; }
  auto const& main            = main_batch();
  auto const& small           = small_batch();
  const std::size_t columns[] = {kSplitNulls, kSorted, kDict, kRandomNulls};
  const std::size_t few[]     = {kSmallString, kSmallSorted, kSmallStringNulls};

  SECTION("no state slot")
  {
    for (auto const& t : main.tiers) {
      INFO(t.name);
      for (std::size_t row_begin : {std::size_t{0}, std::size_t{4000}}) {
        auto dsts = roomy_dsts(main.data, columns, 1500, 7, 3);
        read_and_check(main, t, columns, dsts, row_begin, 1500, nullptr);
      }
    }
  }
  SECTION("one state slot across reads of one batch")
  {
    for (auto const& t : main.tiers) {
      INFO(t.name);
      reader_state state;
      // Forwards, backwards and overlapping: the state is not a forward-only cursor.
      for (std::size_t row_begin : {0, 1500, 3000, 100, 19000, 1499, 0}) {
        auto dsts = roomy_dsts(main.data, columns, 1500, 2, 1);
        read_and_check(main, t, columns, dsts, row_begin, 1500, &state);
        REQUIRE(state != nullptr);
        CHECK(dynamic_cast<sirius::compression::compressed_reader_state*>(state.get()) != nullptr);
      }
    }
  }
  SECTION("one state slot across representations and batches")
  {
    reader_state state;
    for (int round = 0; round < 2; ++round) {
      for (std::size_t i = 0; i < main.tiers.size(); ++i) {
        INFO("round " << round << ", " << main.tiers[i].name);
        auto dsts = roomy_dsts(main.data, columns, 900, 1);
        read_and_check(main, main.tiers[i], columns, dsts, 800 + 300 * i, 900, &state);
        auto small_dsts = roomy_dsts(small.data, few, 900, 1);
        read_and_check(small, small.tiers[i], few, small_dsts, 100 + 300 * i, 900, &state);
      }
    }
  }
}

/// Reads on @p stream, bypassing the helpers that use the default stream.
cucascade::row_range_result read_rows_on(tier const& t,
                                         std::span<const std::size_t> columns,
                                         std::size_t row_begin,
                                         std::size_t max_rows,
                                         std::vector<dst_column> const& dsts,
                                         ::cuda::stream_ref stream,
                                         reader_state* state)
{
  std::vector<cucascade::column_destination> dest;
  for (auto const& d : dsts) {
    dest.push_back(d.dest);
  }
  cucascade::row_range_request request;
  request.columns   = columns;
  request.row_begin = row_begin;
  request.max_rows  = max_rows;
  auto ro           = t.batch->to_read_only();
  return cucascade::read_rows_into(registry(), ro, request, dest, stream, nullptr, state);
}

TEST_CASE("compressed readers follow the caller from one stream to another",
          "[compressed_row_range_readers]")
{
  if (!has_gpu()) { return; }
  auto const& main            = main_batch();
  const std::size_t columns[] = {kSplitNulls, kSorted, kDict, kRandomNulls, kRuns};
  rmm::cuda_stream streams[2];

  SECTION("one state, alternating streams")
  {
    for (auto const& t : main.tiers) {
      INFO(t.name);
      reader_state state;
      std::size_t row_begin = 0;
      for (int read = 0; read < 12; ++read) {
        auto dsts = roomy_dsts(main.data, columns, 1700, 5, 2);
        utils::default_stream().sync();  // the destinations were filled on the default stream
        // Two reads per stream, then the other one.
        auto& stream = streams[(read / 2) % 2];
        auto result  = read_rows_on(t, columns, row_begin, 1700, dsts, stream, &state);
        stream.synchronize();
        check_read(main.data, columns, dsts, row_begin, 1700, result);
        row_begin += result.rows_written;
      }
    }
  }
  SECTION("the state is dropped right after a read on another stream")
  {
    // Nothing waits for the stream between the read and the drop of the state (and, with no
    // state slot, the end of the call): the reader keeps what the queued work reads alive.
    const std::size_t fixed[] = {kSorted, kRandom, kRuns};
    for (auto const& t : main.tiers) {
      INFO(t.name);
      for (bool keep_state : {true, false}) {
        auto dsts = roomy_dsts(main.data, fixed, 9000, 3);
        utils::default_stream().sync();
        cucascade::row_range_result result;
        {
          reader_state state;
          result =
            read_rows_on(t, fixed, 1234, 9000, dsts, streams[0], keep_state ? &state : nullptr);
        }
        streams[0].synchronize();
        check_read(main.data, fixed, dsts, 1234, 9000, result);
      }
    }
  }
}

TEST_CASE("compressed readers refuse a column whose plan is not chunk-decodable",
          "[compressed_row_range_readers]")
{
  if (!has_gpu()) { return; }
  constexpr std::size_t kRows = 5000;
  std::mt19937 rng(11);
  compressed_source source;
  source.data.rows = kRows;
  {
    std::vector<std::int64_t> random(kRows);
    for (auto& v : random) {
      v = static_cast<std::int64_t>(rng() & 0xFFFFFF);
    }
    add_fixed(source.data, cudf::type_id::INT64, random);
    std::vector<std::int32_t> other(kRows);
    for (auto& v : other) {
      v = static_cast<std::int32_t>(rng() & 0xFFFF);
    }
    add_fixed(source.data, cudf::type_id::INT32, other);
  }
  auto stream = utils::default_stream();
  cudf::table table(std::move(source.data.device_columns));
  source.data.device_columns.clear();

  // Column 0 with a spill plan, column 1 with a plan the spill never picks.
  const std::string plan =
    "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
    "---\n"
    "input -> bitcomp\n";
  auto blob   = std::make_shared<sirius::compressed_device_blob>();
  blob->table = simpatico::compress_with_plan(
    table.view(), plan, stream, utils::get_resource_ref(*env().gpu), {});
  stream.sync();
  source.tiers.push_back({"device",
                          as_batch(std::make_unique<sirius::compressed_device_representation>(
                            *env().gpu,
                            std::move(blob),
                            std::vector<std::string>{"c0", "c1"},
                            kRows * 8,
                            kRows * 12,
                            static_cast<std::int64_t>(kRows)))});
  auto const& t = source.tiers.front();

  reader_state state;
  const std::size_t refused[] = {1};
  const std::size_t both[]    = {0, 1};
  const std::size_t decoded[] = {0};
  {
    auto dsts = roomy_dsts(source.data, refused, 100);
    CHECK_THROWS_AS(read_rows(t, refused, 0, 100, dsts, &state),
                    cucascade::row_range_unsupported_error);
    CHECK_THROWS_AS(read_rows(t, refused, 0, 100, dsts, nullptr),
                    cucascade::row_range_unsupported_error);
  }
  {
    auto dsts = roomy_dsts(source.data, both, 100);
    CHECK_THROWS_AS(read_rows(t, both, 1000, 100, dsts, &state),
                    cucascade::row_range_unsupported_error);
  }
  // The batch's other column still reads, with the same state.
  for (std::size_t row_begin : {std::size_t{0}, std::size_t{1000}, std::size_t{4900}}) {
    auto dsts = roomy_dsts(source.data, decoded, 1500, 3);
    read_and_check(source, t, decoded, dsts, row_begin, 1500, &state);
  }
}

TEST_CASE("compressed readers decode with a caller's scratch reservation",
          "[compressed_row_range_readers]")
{
  if (!has_gpu()) { return; }
  auto const& main            = main_batch();
  const std::size_t columns[] = {kSorted, kRuns, kRandom, kDict, kSplitNulls, kRandomNulls};
  for (auto const& t : main.tiers) {
    INFO(t.name);
    auto scratch = env().gpu->make_reservation(64ull << 20);
    REQUIRE(scratch != nullptr);
    // Cursors are created on the first read with the scratch's resource and kept for the state.
    reader_state state;
    for (std::size_t row_begin : {std::size_t{0}, std::size_t{4095}, std::size_t{19500}}) {
      auto dsts = roomy_dsts(main.data, columns, 1200, 5, 2);
      std::vector<cucascade::column_destination> dest;
      for (auto const& d : dsts) {
        dest.push_back(d.dest);
      }
      cucascade::row_range_request request;
      request.columns   = columns;
      request.row_begin = row_begin;
      request.max_rows  = 1200;
      auto ro           = t.batch->to_read_only();
      auto const result = cucascade::read_rows_into(
        registry(), ro, request, dest, utils::default_stream(), scratch.get(), &state);
      check_read(main.data, columns, dsts, row_begin, 1200, result);
    }
    state.reset();
  }
}

TEST_CASE("compressed readers refuse a column projection of the stored chunk",
          "[compressed_row_range_readers]")
{
  if (!has_gpu()) { return; }
  auto const& main            = main_batch();
  const std::size_t picked[]  = {kRandom, kDict};
  const std::size_t columns[] = {0};

  auto const& device_tier = main.tiers[0];
  auto const& host_tier   = main.tiers[1];
  std::vector<tier> projections;
  {
    auto ro            = device_tier.batch->to_read_only();
    auto const& device = ro.get_data()->cast<sirius::compressed_device_representation>();
    projections.push_back({"device projection", as_batch(device.select_columns(picked))});
  }
  {
    auto ro          = host_tier.batch->to_read_only();
    auto const& host = ro.get_data()->cast<sirius::compressed_host_representation>();
    projections.push_back({"host projection", as_batch(host.select_columns(picked))});
  }

  for (auto const& p : projections) {
    INFO(p.name);
    reader_state state;
    auto dsts = roomy_dsts(main.data, std::span<const std::size_t>(picked, 1), 100);
    CHECK_THROWS_AS(read_rows(p, columns, 0, 100, dsts, &state),
                    cucascade::row_range_unsupported_error);
    CHECK_THROWS_AS(read_rows(p, columns, 0, 100, dsts, nullptr),
                    cucascade::row_range_unsupported_error);
  }

  // A state built for the whole batch is not reused for its projection, nor the other way round.
  reader_state state;
  const std::size_t whole[] = {kRandom};
  {
    auto dsts = roomy_dsts(main.data, whole, 100);
    read_and_check(main, device_tier, whole, dsts, 0, 100, &state);
  }
  {
    auto dsts = roomy_dsts(main.data, whole, 100);
    CHECK_THROWS_AS(read_rows(projections[0], columns, 0, 100, dsts, &state),
                    cucascade::row_range_unsupported_error);
  }
  {
    auto dsts = roomy_dsts(main.data, whole, 100);
    read_and_check(main, device_tier, whole, dsts, 300, 100, &state);
  }
}

namespace {

/// A GPU memory space whose allocator sits on a counting resource, so a test sees what the
/// reservations of this space, and nothing else, allocate.
struct counted_gpu_space {
  rmm::mr::statistics_resource_adaptor allocations{
    cuda::mr::any_resource<cuda::mr::device_accessible>{env().gpu->get_default_allocator()}};
  cucascade::memory::memory_space space{cucascade::memory::gpu_memory_space_config{
    .device_id       = env().gpu->get_device_id(),
    .memory_capacity = env().gpu->get_max_memory(),
    .mr_factory_fn   = [this](int, std::size_t) {
      return cuda::mr::any_resource<cuda::mr::device_accessible>{allocations};
    }}};

  [[nodiscard]] std::int64_t bytes_held() const { return allocations.get_bytes_counter().value; }
  [[nodiscard]] std::int64_t allocations_made() const
  {
    return allocations.get_allocations_counter().total;
  }
};

/// Reads with @p scratch on the default stream and checks the rows against the source.
void read_with_scratch(compressed_source const& source,
                       tier const& t,
                       std::span<const std::size_t> columns,
                       std::size_t row_begin,
                       std::size_t max_rows,
                       cucascade::memory::reservation* scratch,
                       reader_state* state)
{
  auto dsts = roomy_dsts(source.data, columns, max_rows, 5, 2);
  std::vector<cucascade::column_destination> dest;
  for (auto const& d : dsts) {
    dest.push_back(d.dest);
  }
  cucascade::row_range_request request;
  request.columns   = columns;
  request.row_begin = row_begin;
  request.max_rows  = max_rows;
  auto ro           = t.batch->to_read_only();
  auto const result = cucascade::read_rows_into(
    registry(), ro, request, dest, utils::default_stream(), scratch, state);
  check_read(source.data, columns, dsts, row_begin, max_rows, result);
}

std::vector<const simpatico::compressed_column_cursor*> cursors_of(reader_state const& state)
{
  auto const* compressed =
    dynamic_cast<const sirius::compression::compressed_reader_state*>(state.get());
  REQUIRE(compressed != nullptr);
  std::vector<const simpatico::compressed_column_cursor*> cursors;
  for (auto const& cursor : compressed->cursors) {
    cursors.push_back(cursor.get());
  }
  return cursors;
}

}  // namespace

TEST_CASE("compressed readers allocate from the scratch reservation's resource of each read",
          "[compressed_row_range_readers]")
{
  if (!has_gpu()) { return; }
  auto const& main = main_batch();
  // A dictionary column keeps its keys on the device for the cursor's life, so a state that has
  // read it holds device memory between reads.
  const std::size_t columns[] = {kDict, kRandom, kSorted, kSplitNulls, kRandomNulls};
  for (auto const& t : main.tiers) {
    INFO(t.name);
    // The spaces outlive the state: its cursors free their memory through them.
    counted_gpu_space first;
    counted_gpu_space second;
    {
      reader_state state;
      {
        auto scratch = first.space.make_reservation_upto(0);
        REQUIRE(scratch != nullptr);
        read_with_scratch(main, t, columns, 0, 1200, scratch.get(), &state);
        // The reservation goes; its space, whose allocator the cursors use, stays.
      }
      utils::default_stream().sync();
      CHECK(first.allocations_made() > 0);
      CHECK(first.bytes_held() > 0);
      CHECK(second.allocations_made() == 0);

      // A later reservation of the same space names the same resource: the cursors stay.
      auto const kept = cursors_of(state);
      {
        auto scratch = first.space.make_reservation_upto(0);
        REQUIRE(scratch != nullptr);
        read_with_scratch(main, t, columns, 1200, 1200, scratch.get(), &state);
      }
      CHECK(cursors_of(state) == kept);
      CHECK(second.allocations_made() == 0);

      // Another resource on the same stream: the cursors are rebuilt with it, and nothing is
      // allocated from, or left in, the first one.
      utils::default_stream().sync();
      auto const made_in_first = first.allocations_made();
      {
        auto scratch = second.space.make_reservation_upto(0);
        REQUIRE(scratch != nullptr);
        read_with_scratch(main, t, columns, 4095, 1200, scratch.get(), &state);
        CHECK(first.allocations_made() == made_in_first);
        CHECK(first.bytes_held() == 0);
        read_with_scratch(main, t, columns, 19500, 1200, scratch.get(), &state);
      }
      utils::default_stream().sync();
      CHECK(first.allocations_made() == made_in_first);
      CHECK(first.bytes_held() == 0);
      CHECK(second.bytes_held() > 0);

      // No scratch: the current device resource, again another one.
      auto const made_in_second = second.allocations_made();
      read_with_scratch(main, t, columns, 300, 1200, nullptr, &state);
      CHECK(second.allocations_made() == made_in_second);
      CHECK(second.bytes_held() == 0);
    }
    CHECK(first.bytes_held() == 0);
    CHECK(second.bytes_held() == 0);
  }
}
