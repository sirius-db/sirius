/*
 * Copyright 2025, Sirius Contributors.
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

#include "op/groupby_surrogate_deferral.hpp"

#include "cudf/cudf_utils.hpp"
#include "sirius/exception.hpp"

#include <algorithm>
#include <utility>

namespace sirius::op {

surrogate_emit_plan::side_plan::side_plan(cudf::size_type rowid_out_pos,
                                          std::vector<cudf::size_type> dummy_out_pos)
  : _rowid_out_pos{rowid_out_pos}, _dummy_out_pos{std::move(dummy_out_pos)}
{
  if (_rowid_out_pos < 0) {
    throw sirius::internal_exception(
      "surrogate_emit_plan::side_plan: negative rowid output position {}", _rowid_out_pos);
  }
  for (auto const pos : _dummy_out_pos) {
    if (pos < 0 || pos == _rowid_out_pos || std::ranges::count(_dummy_out_pos, pos) != 1) {
      throw sirius::internal_exception(
        "surrogate_emit_plan::side_plan: invalid dummy output position {} (rowid position {})",
        pos,
        _rowid_out_pos);
    }
  }
}

surrogate_emit_plan::surrogate_emit_plan(std::optional<side_plan> left,
                                         std::optional<side_plan> right,
                                         std::shared_ptr<surrogate_deferral_store> store)
  : _left{std::move(left)}, _right{std::move(right)}, _store{std::move(store)}
{
  if (_store == nullptr) {
    throw sirius::internal_exception("surrogate_emit_plan: the deferral store must not be null");
  }
  if (!_left && !_right) {
    throw sirius::internal_exception("surrogate_emit_plan: at least one side must defer");
  }
}

surrogate_restore_plan::restore_group::restore_group(join_side side,
                                                     int rowid_key_slot,
                                                     std::vector<restored_key> keys)
  : _side{side}, _rowid_key_slot{rowid_key_slot}, _keys{std::move(keys)}
{
  if (_keys.empty()) {
    throw sirius::internal_exception(
      "surrogate_restore_plan::restore_group: no keys to restore on the {} side", to_string(_side));
  }
  if (std::ranges::find(_keys, _rowid_key_slot, &restored_key::key_slot) == _keys.end()) {
    throw sirius::internal_exception(
      "surrogate_restore_plan::restore_group: rowid key slot {} is not among the restored key "
      "slots on the {} side",
      _rowid_key_slot,
      to_string(_side));
  }
}

surrogate_restore_plan::surrogate_restore_plan(
  std::shared_ptr<surrogate_deferral_store> store,
  std::vector<restore_group> groups,
  std::vector<int> real_key_slots,
  duckdb::vector<sirius::logical_type> original_output_types,
  bool allow_unique_fastpath)
  : _store{std::move(store)},
    _groups{std::move(groups)},
    _real_key_slots{std::move(real_key_slots)},
    _original_output_types{std::move(original_output_types)},
    _allow_unique_fastpath{allow_unique_fastpath}
{
  if (_store == nullptr) {
    throw sirius::internal_exception("surrogate_restore_plan: the deferral store must not be null");
  }
  if (_groups.empty()) {
    throw sirius::internal_exception("surrogate_restore_plan: no restore groups");
  }
  if (_real_key_slots.empty()) {
    throw sirius::internal_exception(
      "surrogate_restore_plan: at least one real (non-deferred) key slot is required for "
      "partition hashing and the distinct proof");
  }
  auto const num_types = static_cast<int>(_original_output_types.size());
  auto const in_range  = [num_types](int slot) { return slot >= 0 && slot < num_types; };
  for (auto const& group : _groups) {
    for (auto const& key : group.keys()) {
      if (!in_range(key.key_slot)) {
        throw sirius::internal_exception(
          "surrogate_restore_plan: restored key slot {} outside the {} declared output types",
          key.key_slot,
          num_types);
      }
      if (std::ranges::find(_real_key_slots, key.key_slot) != _real_key_slots.end()) {
        throw sirius::internal_exception(
          "surrogate_restore_plan: key slot {} is both restored and real", key.key_slot);
      }
    }
  }
  for (auto const slot : _real_key_slots) {
    if (!in_range(slot)) {
      throw sirius::internal_exception(
        "surrogate_restore_plan: real key slot {} outside the {} declared output types",
        slot,
        num_types);
    }
  }
}

void restore_deferred_carriers(surrogate_restore_plan const& plan,
                               std::vector<cudf::data_type>& physical_types)
{
  for (auto const& group : plan.groups()) {
    for (auto const& key : group.keys()) {
      auto const slot = static_cast<std::size_t>(key.key_slot);
      if (slot < physical_types.size()) {
        physical_types[slot] = sirius::get_cudf_type(key.original_type);
      }
    }
  }
}

}  // namespace sirius::op
