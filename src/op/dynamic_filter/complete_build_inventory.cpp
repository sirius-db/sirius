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

#include "data/data_batch_utils.hpp"
#include "op/dynamic_filter/bloom_sizing.hpp"

#include <rmm/aligned.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/common.hpp>

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
  std::vector<batch_entry> batches,
  std::vector<std::optional<cudf::data_type>> schema,
  std::size_t partition_count)
{
  if (partition_count <= 1 || batches.empty()) { return std::nullopt; }
  std::ranges::sort(batches, {}, &batch_entry::batch_id);
  std::size_t total_rows = 0;
  std::optional<std::uint64_t> previous;
  for (auto const& batch : batches) {
    if (previous == batch.batch_id || !std::in_range<std::size_t>(batch.rows) ||
        batch.rows > std::numeric_limits<std::size_t>::max() - total_rows) {
      return std::nullopt;
    }
    total_rows += static_cast<std::size_t>(batch.rows);
    previous = batch.batch_id;
  }
  return complete_build_inventory{std::move(batches), std::move(schema), total_rows};
}

complete_build_inventory::complete_build_inventory(
  std::vector<batch_entry> batches,
  std::vector<std::optional<cudf::data_type>> schema,
  std::size_t total_rows)
  : _batches(std::move(batches)), _schema(std::move(schema)), _total_rows(total_rows)
{
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

namespace {

//===--------------------------------------------------------------------===//
// build_arrival_ledger
//===--------------------------------------------------------------------===//

/**
 * @brief Whether @p batch is readable without blocking and holds a GPU table with no rows.
 */
[[nodiscard]] bool holds_no_rows(cucascade::data_batch const& batch) noexcept
{
  try {
    auto source = batch.try_to_read_only();
    return source && source->get_data() != nullptr &&
           source->get_current_tier() == cucascade::memory::Tier::GPU &&
           sirius::get_cudf_table_view(*source).num_rows() == 0;
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
    auto source = batch.try_to_read_only();
    if (!source || source->get_data() == nullptr ||
        source->get_current_tier() != cucascade::memory::Tier::GPU) {
      close();
      return;
    }
    auto const table = sirius::get_cudf_table_view(*source);
    if (!_schema) {
      std::vector<std::optional<cudf::data_type>> schema;
      schema.reserve(static_cast<std::size_t>(table.num_columns()));
      for (auto const& column : table) {
        schema.emplace_back(column.type());
      }
      _schema = std::move(schema);
    } else {
      for (std::size_t ordinal = 0; ordinal < _schema->size(); ++ordinal) {
        auto& observed = (*_schema)[ordinal];
        if (std::cmp_greater_equal(ordinal, table.num_columns()) ||
            observed != table.column(static_cast<cudf::size_type>(ordinal)).type()) {
          observed.reset();
        }
      }
    }
    _entries.push_back({batch.get_batch_id(), static_cast<std::uint64_t>(table.num_rows())});
  } catch (std::exception const&) {
    // Host allocation or metadata access failed: the ledger can no longer prove completeness.
    close();
  }
}

std::optional<complete_build_inventory> build_arrival_ledger::certify(certification facts) noexcept
{
  std::scoped_lock lock(_mutex);
  if (_status != status::RECORDING) { return std::nullopt; }
  _status = status::CLOSED;
  if (!_schema || _entries.size() != facts.repository_batches) { return std::nullopt; }
  try {
    auto inventory = complete_build_inventory::try_create(
      std::move(_entries), std::move(*_schema), facts.partition_count);
    if (inventory) { _status = status::CERTIFIED; }
    return inventory;
  } catch (std::bad_alloc const&) {
    return std::nullopt;
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
  std::vector<complete_build_inventory::batch_entry>{}.swap(_entries);
  _schema.reset();
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
  auto const chunk_bytes =
    std::min(rmm::align_up(std::clamp(raw_bytes / 8,
                                      accumulated_bloom_geometry::k_min_transfer_chunk_bytes,
                                      accumulated_bloom_geometry::k_max_transfer_chunk_bytes),
                           alignment),
             aligned_bytes);
  return accumulated_bloom_geometry{blocks, raw_bytes, arrays_bytes, chunk_bytes};
}

}  // namespace sirius::op
