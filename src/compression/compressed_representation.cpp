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

#include "compressed_representation.hpp"

#include "device_compressed_blob.hpp"

#include <cuda_runtime.h>

#include <cucascade/error.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace sirius {

// ── Block-aware pinned copies ────────────────────────────────────────────────

void copy_device_to_pinned_blocks(
  const void* src_device,
  cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation& dst,
  std::uint64_t dst_offset,
  std::size_t size,
  rmm::cuda_stream_view stream)
{
  if (size == 0) return;
  const std::size_t bs = dst.block_size();
  std::size_t d_idx    = dst_offset / bs;
  std::size_t d_off    = dst_offset % bs;
  const auto* src      = static_cast<const std::byte*>(src_device);
  std::size_t copied   = 0;
  while (copied < size) {
    const std::size_t chunk = std::min(size - copied, bs - d_off);
    CUCASCADE_CUDA_TRY(cudaMemcpyAsync(
      dst.at(d_idx).data() + d_off, src + copied, chunk, cudaMemcpyDeviceToHost, stream.value()));
    copied += chunk;
    d_off += chunk;
    if (d_off == bs) {
      ++d_idx;
      d_off = 0;
    }
  }
}

void copy_pinned_blocks_to_device(
  const cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation& src,
  std::uint64_t src_offset,
  void* dst_device,
  std::size_t size,
  rmm::cuda_stream_view stream)
{
  if (size == 0) return;
  const std::size_t bs = src.block_size();
  std::size_t s_idx    = src_offset / bs;
  std::size_t s_off    = src_offset % bs;
  auto* dst            = static_cast<std::byte*>(dst_device);
  std::size_t copied   = 0;
  while (copied < size) {
    const std::size_t chunk = std::min(size - copied, bs - s_off);
    CUCASCADE_CUDA_TRY(cudaMemcpyAsync(
      dst + copied, src.at(s_idx).data() + s_off, chunk, cudaMemcpyHostToDevice, stream.value()));
    copied += chunk;
    s_off += chunk;
    if (s_off == bs) {
      ++s_idx;
      s_off = 0;
    }
  }
}

// ── Owning constructor ───────────────────────────────────────────────────────

compressed_host_representation::compressed_host_representation(
  cucascade::memory::memory_space& memory_space,
  std::shared_ptr<pinned_compressed_blob> blob,
  std::vector<std::string> column_names,
  std::size_t compressed_bytes,
  std::size_t uncompressed_bytes,
  std::int64_t num_rows)
  : cucascade::idata_representation(memory_space),
    _blob(std::move(blob)),
    _column_names(std::move(column_names)),
    _compressed_bytes(compressed_bytes),
    _uncompressed_bytes(uncompressed_bytes),
    _num_rows(num_rows)
{
}

// ── Sharing constructor (private) ────────────────────────────────────────────

compressed_host_representation::compressed_host_representation(
  cucascade::memory::memory_space& memory_space,
  std::shared_ptr<pinned_compressed_blob> blob,
  std::vector<std::string> column_names,
  std::size_t compressed_bytes,
  std::size_t uncompressed_bytes,
  std::int64_t num_rows,
  std::optional<std::vector<std::size_t>> selected_indices)
  : cucascade::idata_representation(memory_space),
    _blob(std::move(blob)),
    _column_names(std::move(column_names)),
    _compressed_bytes(compressed_bytes),
    _uncompressed_bytes(uncompressed_bytes),
    _num_rows(num_rows),
    _selected_indices(std::move(selected_indices))
{
}

// ── idata_representation interface ───────────────────────────────────────────

std::unique_ptr<cucascade::idata_representation> compressed_host_representation::clone(
  rmm::cuda_stream_view /*stream*/)
{
  // Share the same backing blob — no byte copy needed.
  auto copy = std::unique_ptr<compressed_host_representation>(
    new compressed_host_representation(get_memory_space(),
                                       _blob,
                                       _column_names,
                                       _compressed_bytes,
                                       _uncompressed_bytes,
                                       _num_rows,
                                       _selected_indices));
  // The pushdown is indexed by the selected column list, which the clone shares.
  copy->set_equality_pushdown(_equality_pushdown);
  copy->set_range_pushdown(_range_pushdown, _range_conjuncts_convertible);
  return copy;
}

// ── Projection ───────────────────────────────────────────────────────────────

std::unique_ptr<compressed_host_representation> compressed_host_representation::select_columns(
  std::span<const std::size_t> indices) const
{
  // Build absolute indices into _column_names, respecting any existing projection.
  std::vector<std::size_t> absolute;
  absolute.reserve(indices.size());
  for (auto idx : indices) {
    if (_selected_indices.has_value()) {
      if (idx >= _selected_indices->size()) {
        throw std::out_of_range(
          "[compressed_host_representation::select_columns] index out of range");
      }
      absolute.push_back((*_selected_indices)[idx]);
    } else {
      if (idx >= _column_names.size()) {
        throw std::out_of_range(
          "[compressed_host_representation::select_columns] index out of range");
      }
      absolute.push_back(idx);
    }
  }

  // const_cast is safe: select_columns is logically const (it creates a
  // projection sharing the same blob) but the base-class constructor requires
  // a non-const memory_space& — the underlying object is non-const.
  return std::unique_ptr<compressed_host_representation>(new compressed_host_representation(
    const_cast<cucascade::memory::memory_space&>(get_memory_space()),
    _blob,
    _column_names,
    _compressed_bytes,
    _uncompressed_bytes,
    _num_rows,
    std::move(absolute)));
}

// ── compressed_device_representation ─────────────────────────────────────────

namespace {

// Resolve caller-relative column indices to absolute indices into the chunk's
// full column list, honoring any projection already applied. Shared by the host
// and device select_columns() implementations.
std::vector<std::size_t> resolve_absolute_indices(
  std::span<const std::size_t> indices,
  const std::optional<std::vector<std::size_t>>& existing_selection,
  std::size_t num_all_columns)
{
  std::vector<std::size_t> absolute;
  absolute.reserve(indices.size());
  for (auto idx : indices) {
    if (existing_selection.has_value()) {
      if (idx >= existing_selection->size()) {
        throw std::out_of_range(
          "[compressed_device_representation::select_columns] index out of range");
      }
      absolute.push_back((*existing_selection)[idx]);
    } else {
      if (idx >= num_all_columns) {
        throw std::out_of_range(
          "[compressed_device_representation::select_columns] index out of range");
      }
      absolute.push_back(idx);
    }
  }
  return absolute;
}

}  // namespace

compressed_device_representation::compressed_device_representation(
  cucascade::memory::memory_space& memory_space,
  std::shared_ptr<compressed_device_blob> blob,
  std::vector<std::string> column_names,
  std::size_t compressed_bytes,
  std::size_t uncompressed_bytes,
  std::int64_t num_rows)
  : cucascade::idata_representation(memory_space),
    _blob(std::move(blob)),
    _column_names(std::move(column_names)),
    _compressed_bytes(compressed_bytes),
    _uncompressed_bytes(uncompressed_bytes),
    _num_rows(num_rows)
{
}

compressed_device_representation::compressed_device_representation(
  cucascade::memory::memory_space& memory_space,
  std::shared_ptr<compressed_device_blob> blob,
  std::vector<std::string> column_names,
  std::size_t compressed_bytes,
  std::size_t uncompressed_bytes,
  std::int64_t num_rows,
  std::optional<std::vector<std::size_t>> selected_indices)
  : cucascade::idata_representation(memory_space),
    _blob(std::move(blob)),
    _column_names(std::move(column_names)),
    _compressed_bytes(compressed_bytes),
    _uncompressed_bytes(uncompressed_bytes),
    _num_rows(num_rows),
    _selected_indices(std::move(selected_indices))
{
}

const simpatico::compressed_table& compressed_device_representation::table() const noexcept
{
  return _blob->table;
}

std::unique_ptr<cucascade::idata_representation> compressed_device_representation::clone(
  rmm::cuda_stream_view /*stream*/)
{
  // Share the same cached blob — no byte copy needed.
  auto copy = std::unique_ptr<compressed_device_representation>(
    new compressed_device_representation(get_memory_space(),
                                         _blob,
                                         _column_names,
                                         _compressed_bytes,
                                         _uncompressed_bytes,
                                         _num_rows,
                                         _selected_indices));
  // The pushdown is indexed by the selected column list, which the clone shares.
  copy->set_equality_pushdown(_equality_pushdown);
  copy->set_range_pushdown(_range_pushdown, _range_conjuncts_convertible);
  return copy;
}

std::unique_ptr<compressed_device_representation> compressed_device_representation::select_columns(
  std::span<const std::size_t> indices) const
{
  auto absolute = resolve_absolute_indices(indices, _selected_indices, _column_names.size());

  // const_cast is safe: select_columns is logically const (it creates a
  // projection sharing the same blob) but the base-class constructor requires
  // a non-const memory_space& — the underlying object is non-const.
  return std::unique_ptr<compressed_device_representation>(new compressed_device_representation(
    const_cast<cucascade::memory::memory_space&>(get_memory_space()),
    _blob,
    _column_names,
    _compressed_bytes,
    _uncompressed_bytes,
    _num_rows,
    std::move(absolute)));
}

compressed_device_representation::fused_scan_reservation_probe
compressed_device_representation::probe_fused_scan_reservation() const
{
  fused_scan_reservation_probe probe{};
  // Mirrors the env gate read in simpatico's wave orchestrator (the converter
  // path); duplicated here because the estimator runs before any simpatico
  // call. Cached — the gate is process-lifetime constant.
  static bool const gate = [] {
    char const* v = std::getenv("SIRIUS_EXP_FUSED_SCAN_FILTER");
    return v != nullptr && std::string_view{v} == "1";
  }();
  if (!gate) { return probe; }
  if (!_range_conjuncts_convertible) { return probe; }
  bool any_active = false;
  for (auto const& entry : _range_pushdown) {
    if (entry.active) {
      any_active = true;
      break;
    }
  }
  if (!any_active) { return probe; }

  // Reservation-time mirror of the converter's RULE 1: every column this
  // projection decodes must come back survivor-compacted (tier_a /
  // tier_a_delta / tier_dict_k5), else the pipeline runs classic and the
  // caller must keep the classic envelope. A tier_dict_k5 column also lifts
  // the RULE-2 selectivity bound (dict batches skip the bail). Pure host
  // metadata (plan-tree walks), no device work.
  auto const& ct     = _blob->table;
  bool any_unbounded = false;
  auto probes_fusable = [&](std::size_t idx) {
    if (idx >= ct.columns.size()) { return false; }
    auto const& plan = ct.columns[idx].plan_tree;
    if (!plan) { return false; }
    if (!simpatico::plan_supports_selection_decode(*plan)) { return false; }
    // Tiers exempt from the RULE-2 selectivity bail (their masked routes win
    // at every selectivity): dict-K5, and K6 strings once its classifier arm
    // flips (the str probe is checked directly so this line needs no change
    // at flip time; pre-flip the umbrella excludes str plans anyway).
    any_unbounded = any_unbounded ||
                    simpatico::plan_selection_tier(*plan) ==
                      sirius::codegen::output_tier::tier_dict_k5 ||
                    simpatico::plan_supports_str_selection_decode(*plan);
    return true;
  };
  bool planned = false;
  if (_selected_indices.has_value()) {
    planned = !_selected_indices->empty();
    for (auto const idx : *_selected_indices) {
      if (!probes_fusable(idx)) { return probe; }
    }
  } else {
    planned = !ct.columns.empty();
    for (std::size_t i = 0; i < ct.columns.size(); ++i) {
      if (!probes_fusable(i)) { return probe; }
    }
  }
  probe.planned       = planned;
  probe.rule2_bounded = planned && !any_unbounded;
  return probe;
}

}  // namespace sirius
