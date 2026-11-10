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

#include "op/dynamic_filter/complete_build_inventory.hpp"

#include "compression/simpatico_compressed_representation.hpp"
#include "data/data_batch_utils.hpp"
#include "op/dynamic_filter/bloom_sizing.hpp"

#include <rmm/aligned.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/memory/column_metadata.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/disk_table.hpp>

#include <algorithm>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>

namespace sirius::op {

//===--------------------------------------------------------------------===//
// complete_build_inventory
//===--------------------------------------------------------------------===//
std::optional<complete_build_inventory> complete_build_inventory::try_create(
  std::vector<recorded_batch> batches)
{
  if (batches.empty()) { return std::nullopt; }
  std::ranges::sort(batches, {}, [](recorded_batch const& batch) { return batch.entry.batch_id; });
  std::size_t total_rows = 0;
  std::optional<std::uint64_t> previous;
  std::size_t width = 0;
  for (auto const& batch : batches) {
    auto const& entry = batch.entry;
    if (previous == entry.batch_id || !std::in_range<std::size_t>(entry.rows) ||
        entry.rows > std::numeric_limits<std::size_t>::max() - total_rows) {
      return std::nullopt;
    }
    total_rows += static_cast<std::size_t>(entry.rows);
    previous = entry.batch_id;
    width    = std::max(width, batch.columns.size());
  }
  // An ordinal keeps its type only when every batch carries it at that same type.
  std::vector<std::optional<cudf::data_type>> schema(width);
  for (std::size_t ordinal = 0; ordinal < width; ++ordinal) {
    auto const& first = batches.front().columns;
    if (ordinal >= first.size()) { continue; }
    auto const carrier = first[ordinal].carrier;
    if (std::ranges::all_of(batches, [&](recorded_batch const& batch) {
          return ordinal < batch.columns.size() && batch.columns[ordinal].carrier == carrier;
        })) {
      schema[ordinal] = carrier;
    }
  }
  std::vector<batch_entry> entries;
  std::vector<std::vector<column_facts>> columns;
  entries.reserve(batches.size());
  columns.reserve(batches.size());
  for (auto& batch : batches) {
    entries.push_back(batch.entry);
    columns.push_back(std::move(batch.columns));
  }
  return complete_build_inventory{
    std::move(entries), std::move(columns), std::move(schema), total_rows};
}

complete_build_inventory::complete_build_inventory(
  std::vector<batch_entry> batches,
  std::vector<std::vector<column_facts>> batch_columns,
  std::vector<std::optional<cudf::data_type>> schema,
  std::size_t total_rows)
  : _batches(std::move(batches)),
    _batch_columns(std::move(batch_columns)),
    _schema(std::move(schema)),
    _total_rows(total_rows)
{
}

std::span<column_facts const> complete_build_inventory::batch_columns(
  std::size_t batch_index) const noexcept
{
  return batch_index < _batch_columns.size()
           ? std::span<column_facts const>{_batch_columns[batch_index]}
           : std::span<column_facts const>{};
}

build_input_facts complete_build_inventory::facts() const
{
  build_input_facts facts{.total_rows = _total_rows, .max_batch_rows = 0, .columns = {}};
  for (auto const& batch : _batches) {
    facts.max_batch_rows = std::max(facts.max_batch_rows, static_cast<std::size_t>(batch.rows));
  }
  facts.columns.resize(_schema.size());
  for (std::size_t ordinal = 0; ordinal < facts.columns.size(); ++ordinal) {
    auto& column = facts.columns[ordinal];
    bool absent  = false;
    std::optional<std::size_t> null_rows{0};
    for (auto const& batch : _batch_columns) {
      if (ordinal >= batch.size()) {
        absent = true;
        break;
      }
      auto const& observed = batch[ordinal];
      if (std::ranges::find(column.carriers, observed.carrier) == column.carriers.end()) {
        column.carriers.push_back(observed.carrier);
      }
      if (null_rows && observed.null_count) {
        *null_rows += *observed.null_count;
      } else {
        null_rows.reset();
      }
    }
    if (absent) {
      column.carriers.clear();
      null_rows.reset();
    }
    column.null_rows = null_rows;
  }
  return facts;
}

complete_build_inventory::batch_entry const* complete_build_inventory::find(
  std::uint64_t batch_id) const noexcept
{
  auto const found = std::ranges::lower_bound(_batches, batch_id, {}, &batch_entry::batch_id);
  return found != _batches.end() && found->batch_id == batch_id ? &*found : nullptr;
}

std::optional<cudf::data_type> complete_build_inventory::consistent_type_at(
  std::size_t ordinal) const noexcept
{
  return ordinal < _schema.size() ? _schema[ordinal] : std::nullopt;
}

//===--------------------------------------------------------------------===//
// read_arrival
//===--------------------------------------------------------------------===//
namespace {

[[nodiscard]] std::optional<std::size_t> recorded_null_count(std::int32_t count) noexcept
{
  return count < 0 ? std::nullopt : std::optional<std::size_t>{static_cast<std::size_t>(count)};
}

/**
 * @brief The facts of a host or disk table described by @p columns
 */
[[nodiscard]] arrival_facts metadata_facts(
  std::vector<cucascade::memory::column_metadata> const& columns, cucascade::memory::Tier tier)
{
  arrival_facts facts{.rows = 0, .columns = {}, .tier = tier};
  facts.columns.reserve(columns.size());
  for (auto const& column : columns) {
    // Only decimal types carry a scale; any other id, including EMPTY, is taken as is.
    auto const id      = static_cast<cudf::type_id>(column.type_id);
    bool const decimal = id == cudf::type_id::DECIMAL32 || id == cudf::type_id::DECIMAL64 ||
                         id == cudf::type_id::DECIMAL128;
    auto const type = decimal ? cudf::data_type{id, column.scale} : cudf::data_type{id};
    facts.columns.push_back({type, recorded_null_count(column.null_count)});
  }
  if (!columns.empty()) {
    facts.rows = static_cast<std::uint64_t>(std::max<std::int32_t>(columns.front().num_rows, 0));
  }
  return facts;
}

}  // namespace

std::variant<arrival_facts, construction_decline> read_arrival(cucascade::data_batch& batch)
{
  auto source = batch.try_to_read_only();
  if (!source || source->get_data() == nullptr) { return construction_decline::unreadable_arrival; }
  auto const* data = source->get_data();
  auto const tier  = source->get_current_tier();
  if (dynamic_cast<sirius::simpatico_compressed_representation const*>(data) != nullptr) {
    return construction_decline::compressed_arrival;
  }
  if (dynamic_cast<cucascade::gpu_table_representation const*>(data) != nullptr) {
    auto const table = sirius::get_cudf_table_view(*source);
    arrival_facts facts{
      .rows = static_cast<std::uint64_t>(table.num_rows()), .columns = {}, .tier = tier};
    facts.columns.reserve(static_cast<std::size_t>(table.num_columns()));
    for (auto const& column : table) {
      facts.columns.push_back({column.type(), static_cast<std::size_t>(column.null_count())});
    }
    return facts;
  }
  if (auto const* host = dynamic_cast<cucascade::host_data_representation const*>(data)) {
    return metadata_facts(host->get_host_table()->columns, tier);
  }
  if (auto const* disk = dynamic_cast<cucascade::disk_data_representation const*>(data)) {
    return metadata_facts(disk->get_disk_table().columns, tier);
  }
  return construction_decline::unreadable_arrival;
}

namespace {

//===--------------------------------------------------------------------===//
// build_arrival_ledger
//===--------------------------------------------------------------------===//

/**
 * @brief Whether @p batch is readable without blocking and holds a GPU, host, or disk table with no
 * rows.
 */
[[nodiscard]] bool holds_no_rows(cucascade::data_batch& batch) noexcept
{
  try {
    auto const arrival = read_arrival(batch);
    auto const* facts  = std::get_if<arrival_facts>(&arrival);
    return facts != nullptr && facts->rows == 0;
  } catch (...) {
    return false;
  }
}

}  // namespace

void build_arrival_ledger::record(cucascade::data_batch& batch)
{
  std::scoped_lock lock(_mutex);
  if (_status == status::CERTIFIED) {
    // A late batch without rows cannot add a key, so it leaves the certified inventory complete.
    if (holds_no_rows(batch)) { return; }
    throw std::logic_error(
      "[build_arrival_ledger::record] a FULL build input received a batch after its source "
      "pipeline finished");
  }
  if (_status != status::RECORDING) { return; }
  try {
    auto arrival = read_arrival(batch);
    if (auto const* reason = std::get_if<construction_decline>(&arrival)) {
      close(*reason);
      return;
    }
    auto& facts = std::get<arrival_facts>(arrival);
    _entries.push_back(
      {.entry = {batch.get_batch_id(), facts.rows}, .columns = std::move(facts.columns)});
  } catch (std::exception const&) {
    // Host allocation or metadata access failed: the ledger can no longer prove completeness.
    close(construction_decline::unreadable_arrival);
  }
}

certified_build_input build_arrival_ledger::certify(certification facts) noexcept
{
  std::scoped_lock lock(_mutex);
  if (_status != status::RECORDING) {
    return _closed_by.value_or(construction_decline::inventory_mismatch);
  }
  _status = status::CLOSED;
  if (_entries.size() != facts.repository_batches) {
    return construction_decline::inventory_mismatch;
  }
  try {
    auto inventory = complete_build_inventory::try_create(std::move(_entries));
    if (!inventory) { return construction_decline::inventory_mismatch; }
    _status = status::CERTIFIED;
    return std::move(*inventory);
  } catch (std::bad_alloc const&) {
    return construction_decline::inventory_mismatch;
  }
}

void build_arrival_ledger::abandon() noexcept
{
  std::scoped_lock lock(_mutex);
  close();
}

void build_arrival_ledger::close() noexcept
{
  _status = status::CLOSED;
  std::vector<complete_build_inventory::recorded_batch>{}.swap(_entries);
}

void build_arrival_ledger::close(construction_decline reason) noexcept
{
  _closed_by = reason;
  close();
}

//===--------------------------------------------------------------------===//
// accumulated_bloom_geometry
//===--------------------------------------------------------------------===//
std::optional<detail::accumulated_bloom_geometry> detail::accumulated_bloom_geometry::try_create(
  std::size_t total_rows, std::size_t active_keys, std::uint64_t cap) noexcept
{
  auto constexpr alignment = rmm::CUDA_ALLOCATION_ALIGNMENT;
  auto constexpr maximum   = std::numeric_limits<std::size_t>::max();
  if (active_keys == 0) { return std::nullopt; }
  auto const blocks = bloom_blocks_for(total_rows);
  if (blocks > maximum / bloom_bytes_per_block) { return std::nullopt; }
  auto const raw_bytes = blocks * bloom_bytes_per_block;
  if (raw_bytes > maximum - (alignment - 1)) { return std::nullopt; }
  auto const aligned_bytes = rmm::align_up(raw_bytes, alignment);
  if (active_keys > maximum / aligned_bytes) { return std::nullopt; }
  auto const arrays_bytes = aligned_bytes * active_keys;
  if (arrays_bytes > cap) { return std::nullopt; }
  return accumulated_bloom_geometry{
    blocks, raw_bytes, arrays_bytes, accumulated_bloom_chunk_bytes(raw_bytes)};
}

std::size_t detail::accumulated_bloom_chunk_bytes(std::size_t raw_bytes) noexcept
{
  auto constexpr alignment = rmm::CUDA_ALLOCATION_ALIGNMENT;
  return std::min(rmm::align_up(std::clamp(raw_bytes / 8,
                                           accumulated_bloom_geometry::k_min_transfer_chunk_bytes,
                                           accumulated_bloom_geometry::k_max_transfer_chunk_bytes),
                                alignment),
                  rmm::align_up(raw_bytes, alignment));
}

}  // namespace sirius::op
