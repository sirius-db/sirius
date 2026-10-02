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

#include <rmm/aligned.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/common.hpp>

#include <algorithm>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>

namespace sirius::op {

std::optional<complete_build_inventory> complete_build_inventory::try_create(
  std::vector<batch_entry> batches, std::vector<column_schema> schema, std::size_t partition_count)
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
  return complete_build_inventory{
    std::move(batches), std::move(schema), total_rows, partition_count};
}

complete_build_inventory::complete_build_inventory(std::vector<batch_entry> batches,
                                                   std::vector<column_schema> schema,
                                                   std::size_t total_rows,
                                                   std::size_t partition_count)
  : _batches(std::move(batches)),
    _schema(std::move(schema)),
    _total_rows(total_rows),
    _partition_count(partition_count)
{
}

complete_build_inventory::complete_build_inventory(complete_build_inventory&& other) noexcept
  : _batches(std::move(other._batches)),
    _schema(std::move(other._schema)),
    _total_rows(std::exchange(other._total_rows, 0)),
    _partition_count(std::exchange(other._partition_count, 0))
{
}

complete_build_inventory& complete_build_inventory::operator=(
  complete_build_inventory&& other) noexcept
{
  if (this != &other) {
    _batches         = std::move(other._batches);
    _schema          = std::move(other._schema);
    _total_rows      = std::exchange(other._total_rows, 0);
    _partition_count = std::exchange(other._partition_count, 0);
  }
  return *this;
}

complete_build_inventory::batch_entry const* complete_build_inventory::find(
  std::uint64_t batch_id) const noexcept
{
  auto const found = std::ranges::lower_bound(_batches, batch_id, {}, &batch_entry::batch_id);
  return found != _batches.end() && found->batch_id == batch_id ? &*found : nullptr;
}

complete_build_inventory::column_schema complete_build_inventory::column_schema::describe(
  cudf::column_view const& column)
{
  column_schema result{column.type(), {}};
  result.children.reserve(column.num_children());
  for (cudf::size_type index = 0; index < column.num_children(); ++index) {
    result.children.push_back(describe(column.child(index)));
  }
  return result;
}

bool complete_build_inventory::column_schema::matches(
  cudf::column_view const& column) const noexcept
{
  if (type != column.type() || children.size() != static_cast<std::size_t>(column.num_children())) {
    return false;
  }
  for (std::size_t index = 0; index < children.size(); ++index) {
    if (!children[index].matches(column.child(static_cast<cudf::size_type>(index)))) {
      return false;
    }
  }
  return true;
}

bool complete_build_inventory::schema_matches(cudf::table_view const& table) const noexcept
{
  if (_schema.size() != static_cast<std::size_t>(table.num_columns())) { return false; }
  for (std::size_t index = 0; index < _schema.size(); ++index) {
    if (!_schema[index].matches(table.column(static_cast<cudf::size_type>(index)))) {
      return false;
    }
  }
  return true;
}

namespace {

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
  if (_status == status::certified) {
    // A late batch without rows cannot add a key, so it leaves the certified inventory complete.
    if (holds_no_rows(batch)) { return; }
    throw std::logic_error(
      "[build_arrival_ledger::record] a FULL build input received a batch after its source "
      "pipeline finished");
  }
  if (_status != status::recording) { return; }
  auto const poison = [this] {
    _status = status::poisoned;
    _entries.clear();
    _entries.shrink_to_fit();
    _schema.reset();
  };
  try {
    auto source = batch.try_to_read_only();
    if (!source || source->get_data() == nullptr ||
        source->get_current_tier() != cucascade::memory::Tier::GPU) {
      poison();
      return;
    }
    auto const table = sirius::get_cudf_table_view(*source);
    if (!_schema) {
      std::vector<complete_build_inventory::column_schema> schema;
      schema.reserve(static_cast<std::size_t>(table.num_columns()));
      for (auto const& column : table) {
        schema.push_back(complete_build_inventory::column_schema::describe(column));
      }
      _schema = std::move(schema);
    } else if (_schema->size() != static_cast<std::size_t>(table.num_columns()) ||
               !std::ranges::equal(*_schema, table, [](auto const& expected, auto const& column) {
                 return expected.matches(column);
               })) {
      poison();
      return;
    }
    _entries.push_back({batch.get_batch_id(), static_cast<std::uint64_t>(table.num_rows())});
  } catch (std::exception const&) {
    // Host allocation or metadata access failed: the ledger can no longer prove completeness.
    poison();
  }
}

std::optional<complete_build_inventory> build_arrival_ledger::certify(certification facts) noexcept
{
  std::scoped_lock lock(_mutex);
  if (_status != status::recording) { return std::nullopt; }
  _status = status::closed;
  if (!_schema || _entries.size() != facts.repository_batches) { return std::nullopt; }
  try {
    auto inventory = complete_build_inventory::try_create(
      std::move(_entries), std::move(*_schema), facts.partition_count);
    if (inventory) { _status = status::certified; }
    return inventory;
  } catch (std::bad_alloc const&) {
    return std::nullopt;
  }
}

void build_arrival_ledger::abandon() noexcept
{
  std::scoped_lock lock(_mutex);
  _status = status::closed;
  _entries.clear();
  _entries.shrink_to_fit();
  _schema.reset();
}

std::optional<detail::accumulated_bloom_geometry> detail::accumulated_bloom_geometry::try_create(
  std::size_t total_rows, std::size_t active_keys, std::uint64_t cap) noexcept
{
  constexpr std::size_t bytes_per_block = 32;
  constexpr std::size_t keys_per_block  = 16;
  constexpr std::size_t alignment       = rmm::CUDA_ALLOCATION_ALIGNMENT;
  auto const maximum                    = std::numeric_limits<std::size_t>::max();
  if (active_keys == 0) { return std::nullopt; }
  auto const blocks = total_rows == 0 ? 1 : 1 + (total_rows - 1) / keys_per_block;
  if (blocks > maximum / bytes_per_block) { return std::nullopt; }
  auto const raw_bytes = blocks * bytes_per_block;
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
  return accumulated_bloom_geometry{blocks, raw_bytes, aligned_bytes, arrays_bytes, chunk_bytes};
}

}  // namespace sirius::op
