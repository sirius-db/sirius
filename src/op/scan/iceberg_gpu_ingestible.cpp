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

#include "op/scan/iceberg_dv_preparation.hpp"
#include "op/scan/puffin_reader.hpp"
#include "sirius/exception.hpp"

#include <cudf/table/table.hpp>

#include <cucascade/memory/memory_space.hpp>
#include <duckdb/common/exception.hpp>
#include <io/uri_parser.hpp>
#include <log/logging.hpp>
#include <op/scan/iceberg_gpu_ingestible.hpp>
#include <op/scan/parquet_batch_layout.hpp>

#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sirius::op::scan {

std::shared_ptr<iceberg_gpu_ingestible> make_ingestible(
  std::unique_ptr<iceberg_ingestible_table_info> info)
{
  return std::make_shared<iceberg_gpu_ingestible>(std::move(info));
}

iceberg_gpu_ingestible::iceberg_gpu_ingestible(std::unique_ptr<iceberg_ingestible_table_info> info)
  : parquet_gpu_ingestible(std::move(info))
{
  auto const& bind = static_cast<iceberg_ingestible_table_info const&>(table_info());
  _table_path      = bind.table_path;
  _deferred        = bind.deferred;
  if (_deferred) {
    if (bind.delete_sets || bind.delete_data)
      throw sirius::internal_exception(
        "[iceberg_gpu_ingestible] deferred and eager delete inputs were supplied");
    if (!bind.partition_indices.empty())
      throw duckdb::NotImplementedException(
        "iceberg table '{}' combines hive partition columns with positional deletes, which the "
        "GPU scan path cannot order correctly",
        _table_path);
    return;
  }
  if (bind.delete_sets && bind.delete_data) {
    throw sirius::internal_exception(
      "[iceberg_gpu_ingestible] both legacy and per-file delete inputs were supplied");
  }
  auto const* legacy = bind.delete_sets ? nullptr : bind.delete_data.get();
  if (!bind.delete_sets && !legacy) {
    throw sirius::internal_exception(
      "[iceberg_gpu_ingestible] no delete data for '" + _table_path +
      "'; the planner must resolve it (or decline the scan) before building the ingestible");
  }

  iceberg_delete_sets sets;
  if (bind.delete_sets) {
    sets = *bind.delete_sets;
    for (auto const& [path, set] : sets) {
      if (!set || set->data_file != path) {
        throw sirius::internal_exception(
          "[iceberg_gpu_ingestible] incomplete or misbound delete set for '" + path + "'");
      }
    }
  } else {
    for (auto const& [path, positions] : legacy->positional_deletes) {
      auto owner = std::shared_ptr<std::vector<int64_t> const>(bind.delete_data, &positions);
      sets.emplace(path, std::make_shared<iceberg_delete_set const>(path, std::move(owner)));
    }
  }

  bool const has_deletes = legacy ? !legacy->positional_deletes.empty()
                                  : std::any_of(sets.begin(), sets.end(), [](auto const& entry) {
                                      return !entry.second->positions.empty();
                                    });
  // On the hive-partition path the reader's predicate is applied before this class sees the
  // table, breaking the row-position mapping. Unreachable for iceberg; guarded anyway.
  if (!bind.partition_indices.empty() && has_deletes) {
    throw duckdb::NotImplementedException(
      "iceberg table '{}' combines hive partition columns with positional deletes, which the "
      "GPU scan path cannot order correctly",
      _table_path);
  }

  if (has_deletes || !legacy) build_delete_key_map(bind.resolved_file_paths, legacy, sets);
  if (legacy && has_deletes) {
    SIRIUS_LOG_DEBUG("[iceberg_gpu_ingestible] '{}': positional deletes for {} data file(s)",
                     _table_path,
                     legacy->positional_deletes.size());
  }
  if (legacy && !legacy->equality_delete_groups.empty()) {
    throw duckdb::NotImplementedException(
      "iceberg table '{}' carries equality deletes, which the GPU scan path does not apply yet",
      _table_path);
  }

  iceberg_delete_sets resolved;
  for (auto const& path : bind.resolved_file_paths) {
    auto const& key = delete_key_for(path);
    auto it         = sets.find(key);
    if (it == sets.end()) {
      if (!legacy) {
        throw sirius::internal_exception("[iceberg_gpu_ingestible] no complete delete set for '" +
                                         path + "'");
      }
      resolved.emplace(path, std::make_shared<iceberg_delete_set const>(key));
    } else {
      resolved.emplace(path, it->second);
    }
  }
  _delete_sets = std::make_shared<iceberg_delete_sets const>(std::move(resolved));
}

namespace {
std::shared_ptr<iceberg_delete_set const> read_prepared_delete_set(
  iceberg_dv_preparation const& prepared,
  iceberg_dv_preparation::file const& descriptor,
  scan_manager::w_permit const& permit,
  std::shared_ptr<physical_check_counters> const& counters)
{
  auto const* dv = descriptor.dv;
  if (!dv)
    return std::make_shared<iceberg_delete_set const>(
      descriptor.path, scan_manager::charged_block{}, 0, 0, prepared.path_owner());

  std::shared_ptr<iceberg_delete_set const> set;
  auto allocator = prepared.ledger->allocator(permit);
  if (counters) {
    ++counters->iceberg_delete_payload_loads;
    if (counters->track_units && counters->iceberg_dv_phase_for_testing)
      counters->iceberg_dv_phase_for_testing(dv->file_path, true);
  }
  try {
    set = read_deletion_vector_charged(DeletionVectorRefView{dv->file_path,
                                                             dv->content_offset,
                                                             dv->content_size_in_bytes,
                                                             dv->referenced_data_file,
                                                             dv->file_size_in_bytes,
                                                             dv->record_count},
                                       allocator,
                                       descriptor.path,
                                       prepared.path_owner(),
                                       prepared.contract);
  } catch (unsupported_physical_input const& error) {
    if (counters) counters->record(error.reason);
    throw;
  } catch (std::bad_alloc const&) {
    throw scan_manager::preparation_resource_error(
      "DV preparation control allocation failed", false, std::current_exception());
  }
  if (counters && counters->track_units && counters->iceberg_dv_phase_for_testing)
    counters->iceberg_dv_phase_for_testing(dv->file_path, false);
  return set;
}

void attach_delete_set(parquet_file_scan_info& file, std::shared_ptr<iceberg_delete_set const> set)
{
  auto deps =
    std::vector<split_dependencies>(file.dependencies().begin(), file.dependencies().end());
  if (deps.size() != 1)
    throw sirius::internal_exception(
      "[iceberg_gpu_ingestible] parquet file requires one delete-set dependency");
  deps.front().delete_set = std::move(set);
  file.disable_filter_pushdown |= !deps.front().delete_set->positions.empty();
  file.set_contract_payload(file.contract_id(),
                            std::vector<split_materializer_certificate>(file.certificates().begin(),
                                                                        file.certificates().end()),
                            std::move(deps));
}
}  // namespace

gpu_ingestible::metadata_scan_task_t iceberg_gpu_ingestible::next_split_provider(
  io::ioctx_resolver resolve)
{
  auto work = parquet_gpu_ingestible::next_split_provider(std::move(resolve));
  if (!work) return {};
  if (_deferred) {
    auto index = _next_preparation;
    struct preparation_inputs {
      scan_manager::w_permit permit;
      std::shared_ptr<scan_manager::preparation_ledger> ledger;
      scan_manager::unit_key key;
      ~preparation_inputs()
      {
        permit.release();
        if (ledger) ledger->retire_unit(key);
      }
    };
    auto held    = std::make_shared<preparation_inputs>();
    held->permit = std::move(_next_permit);
    held->key    = {_deferred->contract, index + 1};
    if (_next_registered) held->ledger = _deferred->ledger;
    _next_registered = false;
    ++_next_preparation;
    auto const& profiles = static_cast<iceberg_ingestible_table_info const&>(table_info()).profiles;
    auto counters        = profiles ? profiles->counters : nullptr;
    return
      [work = std::move(work), prepared = _deferred, held = std::move(held), index, counters]() {
        auto metadata     = work();
        auto& file        = dynamic_cast<parquet_file_scan_info&>(*metadata);
        auto& descriptor  = prepared->files[index];
        auto set          = read_prepared_delete_set(*prepared, descriptor, held->permit, counters);
        descriptor.result = set;
        attach_delete_set(file, std::move(set));
        return metadata;
      };
  }
  return [work = std::move(work), sets = _delete_sets]() {
    auto metadata = work();
    auto& file    = dynamic_cast<parquet_file_scan_info&>(*metadata);
    attach_delete_set(file, sets->at(file.file_path));
    return metadata;
  };
}

scan_manager::unit_key iceberg_gpu_ingestible::next_preparation_unit() const
{
  return {_deferred->contract, _next_preparation + 1};
}
bool iceberg_gpu_ingestible::can_claim_preparation()
{
  if (!_deferred || _next_preparation >= _deferred->files.size()) return true;
  if (!_deferred->files[_next_preparation].dv) return true;
  if (!_deferred->ledger) throw std::logic_error("DV preparation has no attempt ledger");
  if (!_next_registered) {
    _deferred->ledger->register_unit(
      {_deferred->contract, _next_preparation + 1},
      _deferred->files[_next_preparation].dv->record_count * sizeof(int64_t));
    _next_registered = true;
  }
  if (!_next_permit)
    _next_permit = _deferred->ledger->acquire_permit({_deferred->contract, _next_preparation + 1});
  return bool(_next_permit);
}
void iceberg_gpu_ingestible::stop_preparation() noexcept
{
  _next_permit.release();
  if (_next_registered && _deferred && _deferred->ledger)
    _deferred->ledger->retire_unit({_deferred->contract, _next_preparation + 1});
  _next_registered = false;
  if (_deferred) _deferred->ledger.reset();
}
void iceberg_gpu_ingestible::finish_preparation() noexcept
{
  stop_preparation();
  if (_deferred) _deferred->finish();
}

void iceberg_gpu_ingestible::build_delete_key_map(
  std::vector<std::string> const& resolved_file_paths,
  IcebergDeleteData const* legacy,
  iceberg_delete_sets const& sets)
{
  // The delete map is keyed on the path the manifest wrote; the scan reads the path DuckDB's
  // binder resolved. A key that fails to match finds no deletes and returns deleted rows while
  // looking healthy, so the correspondence is established once here and ambiguity declines the
  // scan. Two paths name one file when one is a suffix of the other on a component boundary.
  auto same_file = [](std::string_view a, std::string_view b) {
    if (a == b) { return true; }
    std::string_view longer  = a.size() >= b.size() ? a : b;
    std::string_view shorter = a.size() >= b.size() ? b : a;
    if (shorter.empty() || !longer.ends_with(shorter)) { return false; }
    return longer[longer.size() - shorter.size() - 1] == '/';
  };

  auto basename = [](std::string_view path) {
    auto const slash = path.find_last_of('/');
    return slash == std::string_view::npos ? path : path.substr(slash + 1);
  };

  // Matching implies an identical final path component, so indexing the scanned paths by that
  // component finds every candidate without comparing against all of them. Comparing each key
  // against every file would be O(keys × files) — quadratic in the number of data files, on the
  // planning thread — and would re-strip both paths on every one of those comparisons.
  std::vector<std::string> stripped;  // owns the views held by the index below
  stripped.reserve(resolved_file_paths.size());
  for (auto const& resolved : resolved_file_paths) {
    stripped.push_back(sirius::io::strip_file_scheme(resolved));
  }
  std::unordered_map<std::string_view, std::vector<std::size_t>> scanned_by_basename;
  for (std::size_t i = 0; i < stripped.size(); ++i) {
    scanned_by_basename[basename(stripped[i])].push_back(i);
  }

  // Both manifest-keyed maps need translating: a table with only equality deletes has no
  // positional entries, so positional_deletes alone would leave this empty.
  std::vector<std::string const*> manifest_keys;
  if (legacy) {
    manifest_keys.reserve(legacy->positional_deletes.size() +
                          legacy->data_file_manifest_sequence_numbers.size());
    for (auto const& [delete_key, positions] : legacy->positional_deletes) {
      if (!positions.empty()) { manifest_keys.push_back(&delete_key); }
    }
    for (auto const& entry : legacy->data_file_manifest_sequence_numbers) {
      manifest_keys.push_back(&entry.first);
    }
  } else {
    manifest_keys.reserve(sets.size());
    for (auto const& [key, _] : sets)
      manifest_keys.push_back(&key);
  }

  // Collects EVERY matching key, including one that already spells the file exactly as the scan
  // does. Skipping that one early would leave a second, differently-spelled entry looking like the
  // only claimant (a bare v2 path vs a `file://` v3 one), so the guard below would not fire and
  // the other entry's deleted rows would come back.
  std::unordered_map<std::string, std::vector<std::string const*>> keys_by_scan_path;

  for (auto const* delete_key_ptr : manifest_keys) {
    auto const& delete_key = *delete_key_ptr;

    auto const key_stripped  = sirius::io::strip_file_scheme(delete_key);
    std::string const* match = nullptr;
    if (auto const bucket = scanned_by_basename.find(basename(key_stripped));
        bucket != scanned_by_basename.end()) {
      for (auto const index : bucket->second) {
        if (!same_file(key_stripped, stripped[index])) { continue; }
        if (match != nullptr) {
          throw duckdb::NotImplementedException(
            "iceberg table '{}': delete file entry '{}' matches more than one scanned data file, "
            "so its deleted rows cannot be attributed",
            _table_path,
            delete_key);
        }
        match = &resolved_file_paths[index];
      }
    }

    if (match == nullptr) {
      // Not in this scan's file list — deletes for unscanned files are irrelevant.
      SIRIUS_LOG_DEBUG(
        "[iceberg_gpu_ingestible] '{}': delete entry '{}' names a file this scan does not read",
        _table_path,
        delete_key);
      continue;
    }
    keys_by_scan_path[*match].push_back(&delete_key);
  }

  for (auto const& [scan_path, keys] : keys_by_scan_path) {
    if (keys.size() > 1) {
      throw duckdb::NotImplementedException(
        "iceberg table '{}': scanned data file '{}' is named by {} different manifest entries "
        "(including '{}' and '{}'), so its deletes cannot be attributed to one of them",
        _table_path,
        scan_path,
        keys.size(),
        *keys[0],
        *keys[1]);
    }
    if (*keys[0] == scan_path) { continue; }
    _delete_key_by_scan_path.emplace(scan_path, *keys[0]);
  }
}

std::string const& iceberg_gpu_ingestible::delete_key_for(std::string const& scan_path) const
{
  auto it = _delete_key_by_scan_path.find(scan_path);
  return it == _delete_key_by_scan_path.end() ? scan_path : it->second;
}

filtered_table iceberg_gpu_ingestible::materialize_metadata_to_table(
  scan_info const& info,
  const cucascade::memory::memory_space& mem_space,
  ::cuda::stream_ref stream,
  bool like_swar_fastpath,
  std::shared_ptr<const sirius::like_multiliteral_cache> like_cache)
{
  auto const& split = dynamic_cast<parquet_split_info const&>(info);
  if (split.dependencies().size() != split.rg_slices.size()) {
    throw sirius::internal_exception(
      "[iceberg_gpu_ingestible] split requires one complete delete set per file slice");
  }
  iceberg_delete_sets sets;
  bool has_deletes = false;
  for (size_t i = 0; i < split.rg_slices.size(); ++i) {
    auto const& slice = split.rg_slices[i];
    auto const& set   = split.dependencies()[i].delete_set;
    std::shared_ptr<iceberg_delete_set const> expected;
    if (_deferred) {
      if (slice.file_index < _deferred->files.size()) {
        auto const& f = _deferred->files[slice.file_index];
        if (f.path == slice.file_path) expected = f.result.lock();
      }
    } else {
      auto entry = _delete_sets->find(slice.file_path);
      if (entry != _delete_sets->end()) expected = entry->second;
    }
    if (!set || set != expected) {
      throw sirius::internal_exception(
        "[iceberg_gpu_ingestible] incomplete or misbound delete set for '" + slice.file_path + "'");
    }
    sets.emplace(set->data_file, set);
    has_deletes |= !set->positions.empty();
  }
  if (has_deletes && !split.disable_filter_pushdown) {
    throw sirius::internal_exception(
      "[iceberg_gpu_ingestible] positional deletes require pushdown suppression");
  }
  // Forwarded untouched: these steer the base decode's LIKE evaluation and have nothing to say
  // about deletes. The check below is what guards the case that matters -- if they cause the
  // reader to filter rows, positions stop identifying file rows and we refuse rather than
  // delete the wrong ones.
  auto base = parquet_gpu_ingestible::materialize_metadata_to_table(
    info, mem_space, stream, like_swar_fastpath, std::move(like_cache));

  if (!has_deletes) { return base; }

  // Position-matched deletes require the decoded rows in file order; anything else means the
  // reader filtered, and the mapping below would delete the wrong rows.
  if (base.state != filter_state::UNFILTERED) {
    throw sirius::internal_exception(
      "[iceberg_gpu_ingestible] '" + _table_path +
      "': decode returned a filtered table, so row positions no longer identify file rows; "
      "reader-side pushdown must be suppressed for iceberg splits carrying deletes");
  }

  auto layout = build_batch_layout(split);
  // Runs carry the scan's path; the delete map is keyed on the manifest's.
  for (auto& run : layout) {
    run.data_file_path = delete_key_for(run.data_file_path);
  }

  auto const expected_rows = std::accumulate(
    layout.begin(), layout.end(), int64_t{0}, [](int64_t acc, batch_row_run const& run) {
      return acc + run.num_rows;
    });
  if (expected_rows != static_cast<int64_t>(base.table.num_rows())) {
    throw sirius::internal_exception(
      "[iceberg_gpu_ingestible] '" + _table_path + "': decoded " +
      std::to_string(base.table.num_rows()) + " rows but the split's row groups describe " +
      std::to_string(expected_rows) + "; iceberg delete positions cannot be mapped");
  }

  // release() on an empty handle would hand back a null table.
  if (expected_rows == 0) { return base; }

  rmm::device_async_resource_ref mr_ref(mem_space.get_default_allocator());
  auto table = base.table.release(stream, mr_ref);
  if (!table) {
    throw sirius::internal_exception("[iceberg_gpu_ingestible] '" + _table_path +
                                     "': decoded table has " + std::to_string(expected_rows) +
                                     " rows but no owned state to filter");
  }

  positional_delete_filter filter(std::move(sets));
  auto filtered = filter.apply(std::move(table), layout, stream, mr_ref);
  // State is unchanged: deletes are not the query predicate, which post_filter_and_project must
  // still apply — after the deletes, as Iceberg requires.
  return filtered_table{owning_table_view{std::move(filtered)}, filter_state::UNFILTERED};
}

}  // namespace sirius::op::scan
