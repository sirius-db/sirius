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

#include "compression_converters.hpp"

#include "compressed_representation.hpp"
#include "device_compressed_blob.hpp"

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <api/compressed_table_io.hpp>
#include <api/simpatico_codegen.hpp>
#include <codegen/selection/selection.hpp>
#include <codegen/util/stream_pool.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <log/logging.hpp>
#include <op/scan/row_filtered_table_representation.hpp>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sirius {

namespace {

// Thread-local pool of 4 CUDA streams for cross-column decode parallelism.
// Work is submitted from the calling thread so cuCascade memory-reservation
// tracking (attached to the calling thread) sees all allocations.
// 4 is not a configuration parameter — it matches the typical SM occupancy
// sweet spot for column-parallel decode without thread-spawn overhead.
simpatico::stream_pool& decode_pool()
{
  thread_local simpatico::stream_pool pool;
  if (pool.streams.empty()) {
    if (!pool.init(4)) throw std::runtime_error("[compression_converters] stream_pool init failed");
  }
  return pool;
}

// Rebind a column's buffers (recursively) to `s` for ordered teardown.
// Pool streams are long-lived (thread-local), but the caller's pipeline stream
// `s` is what orders the rest of the work downstream — re-pointing frees here
// ensures deallocation is not racing concurrent pipeline operations on `s`.
std::unique_ptr<cudf::column> rebind_column_stream(std::unique_ptr<cudf::column> col,
                                                   rmm::cuda_stream_view s)
{
  if (!col) { return col; }
  const auto type = col->type();
  const auto size = col->size();
  const auto nc   = col->null_count();
  auto contents   = col->release();
  if (contents.data) { contents.data->set_stream(s); }
  rmm::device_buffer null_mask =
    contents.null_mask ? std::move(*contents.null_mask) : rmm::device_buffer{};
  null_mask.set_stream(s);
  std::vector<std::unique_ptr<cudf::column>> children;
  children.reserve(contents.children.size());
  for (auto& ch : contents.children) {
    children.push_back(rebind_column_stream(std::move(ch), s));
  }
  return std::make_unique<cudf::column>(
    type, size, std::move(*contents.data), std::move(null_mask), nc, std::move(children));
}

// True when column `column_index`'s plan is a plain bitpack leaf over a lane
// type the fused decode variants handle: `input -> bitpack` and nothing else —
// the node consuming the column input is `bitpack` and no node consumes the
// bitpack's output. `dictionary -> bitpack` (indices) and `delta -> bitpack`
// (orderkeys) fail the first test; a hypothetical re-compressed
// `bitpack -> <codec>` fails the second. The lane gate mirrors what K1/K3
// decode into: int32/int64-backed values compared as signed int64 — uint64 is
// excluded because its upper half would misorder under a signed compare.
bool column_is_fusable_bitpack_leaf(simpatico::compressed_table const& table,
                                    std::size_t column_index)
{
  if (column_index >= table.columns.size()) { return false; }
  auto const& column = table.columns[column_index];
  switch (column.dtype.id()) {
    case cudf::type_id::INT8:
    case cudf::type_id::INT16:
    case cudf::type_id::INT32:
    case cudf::type_id::INT64:
    case cudf::type_id::UINT8:
    case cudf::type_id::UINT16:
    case cudf::type_id::UINT32:
    case cudf::type_id::TIMESTAMP_DAYS:
    case cudf::type_id::DECIMAL32:
    case cudf::type_id::DECIMAL64: break;
    default: return false;
  }
  auto const* tree = column.plan_tree.get();
  if (tree == nullptr || tree->nodes.empty() || tree->nodes[0].op != "input") { return false; }
  simpatico::NodeId bitpack_id = 0;
  for (simpatico::NodeId nid = 1; nid < tree->nodes.size(); ++nid) {
    auto const& sources = tree->nodes[nid].input_sources;
    if (!sources.empty() && sources.front() == simpatico::ValueId{0, 0}) {
      if (tree->nodes[nid].op != "bitpack") { return false; }
      bitpack_id = nid;
      break;
    }
  }
  if (bitpack_id == 0) { return false; }
  for (simpatico::NodeId nid = 1; nid < tree->nodes.size(); ++nid) {
    if (nid == bitpack_id) { continue; }
    for (auto const& src : tree->nodes[nid].input_sources) {
      if (src.node == bitpack_id) { return false; }  // bitpack output re-consumed: not a leaf
    }
  }
  return true;
}

// Translate the representation's string-only pushdown into simpatico's decode
// directives, padded to `count` so it lines up 1:1 with the columns being
// decompressed. Returns empty when nothing is pushed down, which lets callers
// stay on the plain decompress overload.
std::vector<simpatico::decode_predicate> to_decode_predicates(
  decode_equality_pushdown const& pushdown, std::size_t count)
{
  bool const any =
    std::any_of(pushdown.begin(), pushdown.end(), [](auto const& v) { return !v.empty(); });
  if (!any) { return {}; }
  if (pushdown.size() > count) {
    throw std::runtime_error(
      "[compression_converters] equality pushdown wider than the projection");
  }
  std::vector<simpatico::decode_predicate> predicates(count);
  for (std::size_t i = 0; i < pushdown.size(); ++i) {
    predicates[i].equals_any = pushdown[i];
  }
  return predicates;
}

// Attempt the fused scan-filter decompress (SIRIUS_EXP_FUSED_SCAN_FILTER):
// wave-1 K1 masks on the range columns, wave-2 decode against the combined
// mask, output assembled to one uniformly survivor-sized table. Returns
// nullptr when the attempt is declined here (directives not buildable for this
// chunk) or the assembly refuses (null-masked column, iteration-1 guard) — the
// caller then runs the classic path. `row_filtered` reports whether the fused
// pipeline actually applied the whole conjunction; false with a non-null
// return means the orchestrator fell back internally and the table is the
// classic full-width decode, usable as-is.
std::unique_ptr<cudf::table> try_decompress_scan_filter(
  simpatico::compressed_table const& ct,
  std::span<const std::size_t> selected,
  decode_range_pushdown const& attached_ranges,
  bool all_conjuncts_convertible,
  simpatico::stream_pool& pool,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  bool& row_filtered)
{
  auto const directives =
    build_fused_scan_directives(ct, selected, attached_ranges, all_conjuncts_convertible);
  if (!directives.enabled) { return nullptr; }

  std::vector<sirius::codegen::column_decode_directive> columns(selected.size());
  for (std::size_t i = 0; i < selected.size(); ++i) {
    auto const& entry         = directives.ranges[i];
    columns[i].has_range      = entry.active;
    columns[i].range          = {entry.lo, entry.hi};
    columns[i].in_scan_mask   = entry.participates_in_scan_mask;
    columns[i].compact_output = directives.output_tiers[i] == decode_output_tier::tier_a;
  }
  auto const request = sirius::codegen::make_scan_filter_request(columns);

  sirius::codegen::scan_filter_result result;
  auto decoded = simpatico::decompress_scan_filter(ct, selected, request, result, pool, mr);
  std::string error;
  auto table =
    simpatico::compact_scan_filter_output(std::move(decoded), result, stream, mr, &error);
  // compact_scan_filter_output synchronized `stream`; re-point the selection
  // buffers there anyway so their teardown follows the batch's ordering.
  result.set_stream(stream);
  if (!table) {
    SIRIUS_LOG_DEBUG(
      "[compression_converters] fused scan-filter assembly refused ({}); falling back to the "
      "classic decompress",
      error);
    return nullptr;
  }
  row_filtered = result.applied;
  SIRIUS_LOG_DEBUG(
    "[compression_converters] fused scan-filter {}: {} of {} row(s) survive, {} column(s)",
    result.applied ? "applied" : "not applied (classic output)",
    result.survivor_count,
    result.num_rows,
    table->num_columns());
  return table;
}

// Reconstruct + project + decompress a compressed_table into a GPU table
// representation. Shared by the host and device compression converters — only
// the byte transport (how `fetch` pulls the payload) differs between them.
std::unique_ptr<cucascade::idata_representation> reconstruct_and_decompress_to_gpu(
  std::span<const std::uint8_t> header,
  simpatico::payload_fetch_fn const& fetch,
  const std::optional<std::vector<std::size_t>>& selected_indices,
  decode_equality_pushdown const& equality_pushdown,
  decode_range_pushdown const& range_pushdown,
  bool range_conjuncts_convertible,
  cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream)
{
  // Reconstruct only the requested columns. read_compressed_table_subset_from_memory
  // fetches just those columns' payload buffers, so serving a projection of a wide
  // pin does not pull every column's compressed bytes onto the GPU — that over-fetch
  // both wasted device memory and drove concurrent decode workers into the memory
  // adaptor's over-reservation path.
  std::string read_error;
  simpatico::compressed_table subset =
    selected_indices.has_value()
      ? simpatico::read_compressed_table_subset_from_memory(
          header,
          fetch,
          *selected_indices,
          stream,
          rmm::mr::get_current_device_resource_ref(),
          &read_error)
      : simpatico::read_compressed_table_from_memory(
          header, fetch, stream, rmm::mr::get_current_device_resource_ref(), &read_error);
  if (!read_error.empty()) {
    throw std::runtime_error("[compression_converters] reconstruct failed: " + read_error);
  }

  // Decode across 4 pool streams, submitted from the calling thread — no worker
  // threads are spawned. The H2D fetch above ran on `stream`; sync it first so
  // pool-stream reads are ordered after all fetched bytes are resident.
  stream.synchronize();
  auto& pool    = decode_pool();
  auto const mr = rmm::mr::get_current_device_resource_ref();
  // `subset` already holds only the projected columns, so the pushdown — which
  // is indexed by projected position — lines up with 0..num_columns.
  auto const predicates =
    to_decode_predicates(equality_pushdown, static_cast<std::size_t>(subset.num_columns()));
  std::unique_ptr<cudf::table> decompressed;
  bool decode_row_filtered = false;
  // Fused scan-filter attempt — same contract as the device converter: only
  // without string-equality predicates, decline ⇒ classic path below.
  if (predicates.empty() && range_conjuncts_convertible && !range_pushdown.empty()) {
    // `subset` holds exactly the projected columns, so selection is identity.
    std::vector<std::size_t> identity_selection(subset.num_columns());
    std::iota(identity_selection.begin(), identity_selection.end(), std::size_t{0});
    decompressed = try_decompress_scan_filter(subset,
                                              identity_selection,
                                              range_pushdown,
                                              range_conjuncts_convertible,
                                              pool,
                                              stream,
                                              mr,
                                              decode_row_filtered);
  }
  if (!decompressed) {
    if (predicates.empty()) {
      decompressed = simpatico::decompress(subset, pool, mr);
    } else {
      std::vector<std::size_t> all(subset.num_columns());
      std::iota(all.begin(), all.end(), std::size_t{0});
      decompressed = simpatico::decompress(subset, all, predicates, pool, mr);
    }
  }
  // Re-point decoded buffers onto `stream` so pipeline teardown is ordered.
  auto cols = decompressed->release();
  for (auto& c : cols)
    c = rebind_column_stream(std::move(c), stream);
  decompressed = std::make_unique<cudf::table>(std::move(cols));

  const cucascade::memory::memory_space* space =
    (target_memory_space != nullptr) ? target_memory_space : &source.get_memory_space();

  SIRIUS_LOG_DEBUG("[compression_converters] decompressed cols={} rows={} → GPU device={}",
                   decompressed->num_columns(),
                   decompressed->num_rows(),
                   space->get_device_id());

  // Tagged subclass ⇔ the fused pipeline applied the whole conjunction (see
  // row_filtered_table_representation.hpp).
  if (decode_row_filtered) {
    return std::make_unique<row_filtered_gpu_table_representation>(
      std::move(decompressed), *const_cast<cucascade::memory::memory_space*>(space), stream);
  }
  return std::make_unique<cucascade::gpu_table_representation>(
    std::move(decompressed), *const_cast<cucascade::memory::memory_space*>(space), stream);
}

// compressed_host_representation (pinned host) → GPU.
std::unique_ptr<cucascade::idata_representation> decompress_host_to_gpu(
  cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream,
  [[maybe_unused]] cucascade::memory::reservation* reservation)
{
  nvtx3::scoped_range nvtx_range{"sirius::compression::host_to_gpu"};
  auto& rep = source.cast<compressed_host_representation>();

  // Pull each compressed leaf buffer straight from the pinned host payload into
  // device memory (block-aware, since the payload is a multi-block allocation).
  auto const& payload = rep.payload();
  simpatico::payload_fetch_fn fetch =
    [&payload](std::uint64_t off, std::size_t sz, void* dst, rmm::cuda_stream_view s) {
      copy_pinned_blocks_to_device(payload, off, dst, sz, s);
    };

  return reconstruct_and_decompress_to_gpu(rep.header(),
                                           fetch,
                                           rep.selected_indices(),
                                           rep.equality_pushdown(),
                                           rep.range_pushdown(),
                                           rep.range_conjuncts_convertible(),
                                           source,
                                           target_memory_space,
                                           stream);
}

// compressed_device_representation (device memory) → GPU.
// The compressed_table is already cached on device; decompress directly with no
// re-fetch. When a column projection is set, only the selected columns are decoded.
std::unique_ptr<cucascade::idata_representation> decompress_device_to_gpu(
  cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream,
  [[maybe_unused]] cucascade::memory::reservation* reservation)
{
  nvtx3::scoped_range nvtx_range{"sirius::compression::device_to_gpu"};
  auto& rep           = source.cast<compressed_device_representation>();
  auto const& indices = rep.selected_indices();
  auto const& ct      = rep.table();
  auto const mr       = rmm::mr::get_current_device_resource_ref();
  auto& pool          = decode_pool();

  // Projected column count — what the pushdown is indexed by.
  auto const n_selected =
    indices.has_value() ? indices->size() : static_cast<std::size_t>(ct.num_columns());
  auto const predicates = to_decode_predicates(rep.equality_pushdown(), n_selected);

  std::unique_ptr<cudf::table> decompressed;
  bool decode_row_filtered = false;
  // Fused scan-filter attempt (SIRIUS_EXP_FUSED_SCAN_FILTER; the env gate and
  // every per-plan precondition are re-checked inside decompress_scan_filter).
  // Not composable with string-equality predicates (iteration 1) — those scans
  // keep the predicated overload below. Any decline falls through classically.
  std::vector<std::size_t> identity_selection;
  if (predicates.empty() && rep.range_conjuncts_convertible() && !rep.range_pushdown().empty()) {
    std::span<const std::size_t> selected;
    if (indices.has_value()) {
      selected = *indices;
    } else {
      identity_selection.resize(n_selected);
      std::iota(identity_selection.begin(), identity_selection.end(), std::size_t{0});
      selected = identity_selection;
    }
    decompressed = try_decompress_scan_filter(ct,
                                              selected,
                                              rep.range_pushdown(),
                                              rep.range_conjuncts_convertible(),
                                              pool,
                                              stream,
                                              mr,
                                              decode_row_filtered);
  }
  if (!decompressed) {
    if (predicates.empty()) {
      decompressed = indices.has_value() ? simpatico::decompress(ct, *indices, pool, mr)
                                         : simpatico::decompress(ct, pool, mr);
    } else if (indices.has_value()) {
      decompressed = simpatico::decompress(ct, *indices, predicates, pool, mr);
    } else {
      std::vector<std::size_t> all(n_selected);
      std::iota(all.begin(), all.end(), std::size_t{0});
      decompressed = simpatico::decompress(ct, all, predicates, pool, mr);
    }
  }
  auto cols = decompressed->release();
  for (auto& c : cols)
    c = rebind_column_stream(std::move(c), stream);
  decompressed = std::make_unique<cudf::table>(std::move(cols));

  const cucascade::memory::memory_space* space =
    (target_memory_space != nullptr) ? target_memory_space : &source.get_memory_space();

  SIRIUS_LOG_DEBUG("[compression_converters] decompressed cols={} rows={} → GPU device={}",
                   decompressed->num_columns(),
                   decompressed->num_rows(),
                   space->get_device_id());

  // The tagged subclass is the scan's hard promise that the WHOLE table-filter
  // conjunction was applied during decode (see row_filtered_table_representation.hpp);
  // it is only constructed when the fused pipeline reported applied=true.
  if (decode_row_filtered) {
    return std::make_unique<row_filtered_gpu_table_representation>(
      std::move(decompressed), *const_cast<cucascade::memory::memory_space*>(space), stream);
  }
  return std::make_unique<cucascade::gpu_table_representation>(
    std::move(decompressed), *const_cast<cucascade::memory::memory_space*>(space), stream);
}

}  // namespace

fused_scan_directives build_fused_scan_directives(const simpatico::compressed_table& table,
                                                  std::span<const std::size_t> selected_columns,
                                                  const decode_range_pushdown& attached_ranges,
                                                  bool all_conjuncts_convertible)
{
  fused_scan_directives out;  // disabled/empty unless everything below holds
  if (!all_conjuncts_convertible) { return out; }
  bool const any_active = std::any_of(
    attached_ranges.begin(), attached_ranges.end(), [](auto const& e) { return e.active; });
  if (!any_active) { return out; }
  if (attached_ranges.size() > selected_columns.size()) {
    throw std::runtime_error("[compression_converters] range pushdown wider than the projection");
  }

  auto const count = selected_columns.size();
  out.ranges       = attached_ranges;
  out.ranges.resize(count);  // pad the inactive tail
  out.output_tiers.assign(count, decode_output_tier::tier_b);

  std::size_t mask_columns   = 0;
  std::size_t tier_a_columns = 0;
  for (std::size_t i = 0; i < count; ++i) {
    bool const fusable = column_is_fusable_bitpack_leaf(table, selected_columns[i]);
    if (fusable) {
      out.output_tiers[i] = decode_output_tier::tier_a;
      ++tier_a_columns;
    }
    if (!out.ranges[i].active) { continue; }
    if (!fusable) {
      // A range conjunct this chunk cannot evaluate in-decode ⇒ the combined
      // mask would under-filter. Iteration 1 has no mixed-mask combine, so the
      // whole batch keeps today's decode-then-filter path.
      SIRIUS_LOG_DEBUG(
        "[compression_converters] fused scan-filter disabled for this batch: range column "
        "(selected pos {}, physical {}) is not a fusable bitpack leaf",
        i,
        selected_columns[i]);
      return {};
    }
    out.ranges[i].participates_in_scan_mask = true;
    ++mask_columns;
  }

  out.enabled = true;
  SIRIUS_LOG_DEBUG(
    "[compression_converters] fused scan-filter directives: {} mask column(s), {} tier-A / {} "
    "tier-B output column(s)",
    mask_columns,
    tier_a_columns,
    count - tier_a_columns);
  return out;
}

void register_compression_converters(cucascade::representation_converter_registry& registry)
{
  // Decompression paths used by prepare_for_processing / convert_to.
  if (!registry
         .has_converter<compressed_host_representation, cucascade::gpu_table_representation>()) {
    registry
      .register_converter<compressed_host_representation, cucascade::gpu_table_representation>(
        decompress_host_to_gpu);
  }
  if (!registry
         .has_converter<compressed_device_representation, cucascade::gpu_table_representation>()) {
    registry
      .register_converter<compressed_device_representation, cucascade::gpu_table_representation>(
        decompress_device_to_gpu);
  }
}

}  // namespace sirius
