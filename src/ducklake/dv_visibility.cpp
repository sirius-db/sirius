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

#include "ducklake/dv_visibility.hpp"

#include "io/io_errors.hpp"
#include "io/sirius_datasource.hpp"
#include "io/uri_parser.hpp"

#include <algorithm>
#include <charconv>
#include <cstring>
#include <system_error>

namespace sirius::ducklake {
namespace scan = op::scan;
namespace sm   = scan_manager;
bool parquet_unit_dependencies_supported(unit_descriptor const& unit)
{
  auto gate = assess_local_scope(unit);
  return unit.kind == unit_kind::parquet && !gate.outside_local_scope &&
         std::all_of(gate.reasons.begin(),
                     gate.reasons.end(),
                     [](auto reason) {
                       return reason == unit_gate_reason::dv_metadata_required ||
                              reason == unit_gate_reason::parquet_delete_semantics_required;
                     }) &&
         !unit.mapping_id;
}
sm::unit_envelope parquet_unit_envelope(unit_descriptor const& unit)
{
  if (!parquet_unit_dependencies_supported(unit))
    throw std::invalid_argument("DuckLake Parquet unit has unqualified dependencies");
  if (unit.inline_delete_count > dv_max_positions)
    throw sm::preparation_resource_error("DuckLake inline delete positions exceed limit", true);
  if (unit.deletes == delete_format::none)
    return {{1}, unit.inline_delete_count ? 3 * dv_max_positions * sizeof(int64_t) : 0};
  if (!unit.delete_file_size || unit.delete_file_size > dv_max_file_bytes)
    throw sm::preparation_resource_error("DuckLake delete file exceeds the bounded profile", true);
  if (unit.deletes == delete_format::parquet) return {{1}, 4 * dv_max_positions * sizeof(int64_t)};
  // JSON text/parser, previous union, decoded blob and replacement union can overlap.
  return {{unit.delete_file_size},
          14 * std::min<uint64_t>(unit.delete_file_size, dv_max_footer_bytes) + 512 +
            4 * dv_max_positions * sizeof(int64_t)};
}
namespace {
void check_stop(sm::preparation_stop_token stop)
{
  if (stop.stop_requested())
    throw std::system_error(std::make_error_code(std::errc::operation_canceled));
}
void read_exact(scan::puffin_byte_source& source,
                int64_t offset,
                std::span<std::byte> bytes,
                sm::preparation_stop_token stop)
{
  check_stop(stop);
  if (!source.seek(offset) || !source.read(bytes))
    throw transparent::classified_execution_error(transparent::late_failure_cause::reader_io,
                                                  "DuckLake DV range could not be read");
  check_stop(stop);
}
uint64_t cardinality(std::string_view value)
{
  uint64_t result = 0;
  auto parsed     = std::from_chars(value.data(), value.data() + value.size(), result);
  if (value.empty() || parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
    throw std::runtime_error("DuckLake DV cardinality is not a nonnegative integer");
  return result;
}
}  // namespace
std::shared_ptr<dv_visibility const> prepare_dv_visibility(unit_descriptor const& unit,
                                                           int64_t file_rows,
                                                           std::string data_identity,
                                                           scan::puffin_byte_source& source,
                                                           sm::charging_allocator& allocator,
                                                           sm::preparation_stop_token stop,
                                                           std::string delete_identity)
{
  (void)parquet_unit_envelope(unit);
  if (unit.deletes != delete_format::puffin || file_rows < 0 || data_identity.empty())
    throw std::invalid_argument("DuckLake DV requires original data evidence");
  check_stop(stop);
  if (source.size() < 8 || static_cast<uint64_t>(source.size()) != unit.delete_file_size)
    throw io::descriptor_mismatch_error("DuckLake DV object size differs from binding");
  std::array<std::byte, 4> magic;
  read_exact(source, 0, magic, stop);
  bool container = std::memcmp(magic.data(), "PFA1", 4) == 0;
  std::shared_ptr<scan::puffin_container_metadata const> metadata;
  size_t blobs = 1;
  if (container) {
    metadata = scan::read_puffin_metadata(source, allocator, {dv_max_footer_bytes, dv_max_blobs});
    if (unit.delete_footer_size &&
        *unit.delete_footer_size != static_cast<uint64_t>(source.size() - metadata->footer_start()))
      throw std::runtime_error("DuckLake DV footer size differs from binding");
    blobs = 0;
    for (size_t i = 0; i < metadata->blob_count(); ++i) {
      auto blob = metadata->blob(i);
      if (blob.type != "deletion-vector-v1") continue;
      ++blobs;
      if (blob.compression_codec || blob.snapshot_id != -1 || blob.sequence_number != -1)
        throw std::runtime_error("DuckLake DV blob profile is unqualified");
      if (metadata->property(i, "ducklake-snapshot-id"))
        throw std::runtime_error("DuckLake snapshot-bearing DV is outside the local scan profile");
      if (auto path = metadata->property(i, "referenced-data-file"))
        if (io::strip_file_scheme(std::string(*path)) != io::strip_file_scheme(unit.input.path))
          throw std::runtime_error("DuckLake DV referenced data file differs from original unit");
    }
    if (!blobs) throw std::runtime_error("DuckLake container has no relevant deletion vectors");
  } else if (unit.delete_footer_size && *unit.delete_footer_size)
    throw std::runtime_error("DuckLake container magic is missing");
  auto result             = std::make_shared<dv_visibility>();
  result->data_identity   = std::move(data_identity);
  result->delete_identity = std::move(delete_identity);
  size_t cumulative       = 0;
  for (size_t i = 0; i < (metadata ? metadata->blob_count() : 1); ++i) {
    int64_t offset = 0, length = source.size();
    if (metadata) {
      auto blob = metadata->blob(i);
      if (blob.type != "deletion-vector-v1") continue;
      offset = blob.offset;
      length = blob.length;
    }
    if (length <= 0 || static_cast<uint64_t>(length) > dv_max_file_bytes)
      throw sm::preparation_resource_error("DuckLake DV blob exceeds byte limit", true);
    auto payload = allocator.allocate(length);
    read_exact(source, offset, {payload.data(), static_cast<size_t>(length)}, stop);
    auto decoded = scan::decode_deletion_vector_payload(
      {reinterpret_cast<uint8_t const*>(payload.data()), static_cast<size_t>(length)},
      allocator,
      {dv_max_file_bytes, dv_max_positions});
    auto positions = decoded->positions();
    if (!positions.empty() && positions.back() >= file_rows)
      throw std::runtime_error("DuckLake DV position exceeds original file row domain");
    if (metadata)
      if (auto declared = metadata->property(i, "cardinality"))
        if (cardinality(*declared) != positions.size())
          throw std::runtime_error("DuckLake DV declared cardinality differs from decoded count");
    if (positions.size() > dv_max_positions - cumulative)
      throw sm::preparation_resource_error("DuckLake DV aggregate positions exceed limit", true);
    cumulative += positions.size();
    if (!result->validated_blobs) {
      result->decoded   = std::move(decoded);
      result->positions = result->decoded->positions();
    } else if (!positions.empty()) {
      auto merged    = allocator.allocate_retained((result->positions.size() + positions.size()) *
                                                sizeof(int64_t));
      auto* output   = reinterpret_cast<int64_t*>(merged.data());
      auto* end      = std::set_union(result->positions.begin(),
                                 result->positions.end(),
                                 positions.begin(),
                                 positions.end(),
                                 output);
      auto count     = static_cast<size_t>(end - output);
      result->merged = std::move(merged);
      result->decoded.reset();
      result->positions = {output, count};
    }
    ++result->validated_blobs;
    check_stop(stop);
  }
  if (result->validated_blobs != blobs)
    throw std::logic_error("DuckLake DV validation coverage incomplete");
  return result;
}
void validate_delete_position(int64_t position, int64_t& previous)
{
  if (position < 0 || position <= previous)
    throw std::runtime_error(
      "DuckLake Parquet delete positions must be nonnegative and strictly increasing");
  previous = position;
}
std::shared_ptr<dv_visibility const> bind_parquet_visibility(
  std::shared_ptr<dv_visibility const> source, std::string data_identity, int64_t file_rows)
{
  if (!source || !source->parquet_validated || source->delete_identity.empty() ||
      data_identity.empty() || file_rows < 0)
    throw std::invalid_argument("DuckLake Parquet deletes lack validated file evidence");
  if (!source->positions.empty() && source->positions.back() >= file_rows)
    throw std::runtime_error("DuckLake Parquet delete position exceeds original file row domain");
  auto result               = std::make_shared<dv_visibility>();
  result->data_identity     = std::move(data_identity);
  result->delete_identity   = source->delete_identity;
  result->parquet_validated = true;
  result->positions         = source->positions;
  result->predecessor       = std::move(source);
  return result;
}
std::shared_ptr<dv_visibility const> combine_inline_visibility(
  std::shared_ptr<dv_visibility const> base,
  std::span<uint64_t const> inlined,
  int64_t file_rows,
  std::optional<uint64_t> max_row_count,
  sm::charging_allocator& allocator)
{
  if (!base || file_rows < 0)
    throw std::invalid_argument("DuckLake visibility lacks original rows");
  if (base->positions.size() > dv_max_positions ||
      inlined.size() > dv_max_positions - base->positions.size())
    throw sm::preparation_resource_error("DuckLake combined delete positions exceed limit", true);
  for (size_t i = 0; i < inlined.size(); ++i)
    if (inlined[i] >= static_cast<uint64_t>(file_rows) || (i && inlined[i] <= inlined[i - 1]))
      throw std::runtime_error(
        "DuckLake inline delete positions are outside the ordered file domain");
  auto result               = std::make_shared<dv_visibility>();
  result->data_identity     = base->data_identity;
  result->delete_identity   = base->delete_identity;
  result->validated_blobs   = base->validated_blobs;
  result->parquet_validated = base->parquet_validated;
  result->inline_count      = inlined.size();
  result->max_row_count     = max_row_count;
  auto count                = base->positions.size() + inlined.size();
  if (count) {
    result->merged = allocator.allocate_retained(count * sizeof(int64_t));
    auto* out      = reinterpret_cast<int64_t*>(result->merged.data());
    auto* end      = std::set_union(
      base->positions.begin(), base->positions.end(), inlined.begin(), inlined.end(), out);
    result->positions = {out, static_cast<size_t>(end - out)};
  }
  return result;
}
std::shared_ptr<dv_visibility const> read_dv_visibility(unit_descriptor const& unit,
                                                        int64_t file_rows,
                                                        std::string data_identity,
                                                        io::ioctx_resolver const& resolve,
                                                        sm::charging_allocator& allocator,
                                                        sm::preparation_stop_token stop)
{
  check_stop(stop);
  std::unique_ptr<io::sirius_datasource> datasource;
  try {
    auto context = resolve(unit.delete_path);
    if (!context) throw std::runtime_error("DuckLake DV has no I/O context");
    datasource = context->open_datasource(unit.delete_path, io::open_hint::generic);
    if (!datasource) throw std::runtime_error("DuckLake DV has no datasource");
  } catch (transparent::classified_execution_error const&) {
    throw;
  } catch (std::bad_alloc const&) {
    throw;
  } catch (std::system_error const& e) {
    if (e.code() == std::errc::operation_canceled) throw;
    throw transparent::classified_execution_error(transparent::late_failure_cause::reader_io,
                                                  e.what());
  } catch (std::exception const& e) {
    throw transparent::classified_execution_error(transparent::late_failure_cause::reader_io,
                                                  e.what());
  }
  auto identity = datasource->get_io_object().identity_cache_key();
  auto source   = scan::make_puffin_byte_source(std::move(datasource));
  return prepare_dv_visibility(
    unit, file_rows, std::move(data_identity), *source, allocator, stop, std::move(identity));
}
}  // namespace sirius::ducklake
