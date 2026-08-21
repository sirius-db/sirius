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

#include "scan_manager/sirius_scan_manager.hpp"

#include "compression/decompression_pushdown_policy.hpp"
#include "compression/device_compressed_blob.hpp"
#include "cudf/cudf_utils.hpp"
#include "data/data_batch_utils.hpp"
#include "exec/thread_pool.hpp"
#include "helper/numeric_narrowing.hpp"
#include "io/cache/prefetching_cache.hpp"
#include "io/io_context.hpp"
#include "io/parquet_helpers.hpp"
#include "io/rest/rest_ioctx.hpp"
#include "io/sirius_datasource.hpp"
#include "log/logging.hpp"
#include "memory/topology_index.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"
#include "op/scan/duckdb_native_gpu_ingestible.hpp"
#include "op/scan/gpu_ingestible.hpp"
#include "op/scan/parquet_gpu_ingestible.hpp"
#include "op/scan/parquet_metadata.hpp"
#include "op/scan/scan_filter_analysis.hpp"
#include "op/scan/scan_utils.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "op/scan/sirius_gpu_scan_operator_data.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "planner/query.hpp"
#include "scan_manager/round_robin_strategy.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_device.hpp>

#include <api/simpatico_codegen.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/memory/column_metadata.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb/main/attached_database.hpp>
#include <duckdb/storage/data_table.hpp>
#include <duckdb/storage/single_file_block_manager.hpp>
#include <duckdb/storage/storage_manager.hpp>
#include <duckdb/transaction/duck_transaction_manager.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace sirius::scan_manager {

namespace {

using sirius::pinned_column_storage_matrix;
using sirius::pinned_column_storage_meta;

// Actual cuDF carrier of one column of an uncompressed pinned host chunk, rebuilt from the
// chunk's host column metadata. Keyed on the DECIMAL type ids, not on a nonzero scale: a
// DECIMAL(p,0) column has cuDF scale 0 and must still take the two-argument fixed-point
// constructor.
cudf::data_type host_column_carrier(cucascade::memory::column_metadata const& meta)
{
  auto const id         = static_cast<cudf::type_id>(meta.type_id);
  auto const is_decimal = id == cudf::type_id::DECIMAL32 || id == cudf::type_id::DECIMAL64 ||
                          id == cudf::type_id::DECIMAL128;
  return is_decimal ? cudf::data_type{id, meta.scale} : cudf::data_type{id};
}

struct cached_databatch_provider : public databatch_provider {
  cached_databatch_provider(pinned_entry const& entry,
                            std::span<std::size_t const> selected_columns,
                            cached_scan_plan plan,
                            const telemetry::batch_telemetry_info& telemetry_info,
                            mvcc_chunk_mask_set mvcc_masks,
                            std::vector<insert_delta_split> delta_splits,
                            std::vector<cudf::data_type> normalization_targets,
                            bool has_physical_overrides,
                            sirius::pushdown_request pushdown_req,
                            std::shared_ptr<sirius::op::sirius_dynamic_filter_set> dynamic_filters)
    : _plan(std::move(plan)),
      _entry(entry),
      _telemetry_info(telemetry_info),
      _mvcc_masks(std::move(mvcc_masks)),
      _delta_splits(std::move(delta_splits)),
      _normalization_targets(std::move(normalization_targets)),
      _has_physical_overrides(has_physical_overrides),
      _pushdown_req(std::move(pushdown_req)),
      _dynamic_filters(std::move(dynamic_filters))
  {
    auto const& entry_column_names = _entry.cache_info.column_names();
    std::ranges::for_each(selected_columns, [this, &entry_column_names](size_t idx) {
      _column_names.emplace_back(entry_column_names[idx]);
      _column_indices.push_back(idx);
    });
  }

  databatch_provider::batch get_next_batch() override
  {
    // The atomic cursor walks survivor positions, each mapping to a chunk
    // index; positions past the survivors yield the insert-delta splits.
    // The cursor hands out unique positions, so each split is moved out once.
    auto const cursor = _index.fetch_add(1);
    if (cursor >= _plan.survivor_chunk_indices.size()) {
      auto const delta_idx = cursor - _plan.survivor_chunk_indices.size();
      if (delta_idx >= _delta_splits.size()) { return {}; }
      auto& delta = _delta_splits[delta_idx];
      databatch_provider::batch out;
      out.mvcc_keep_mask   = std::move(delta.mask);
      out.scan_info        = std::move(delta.info);
      out.preferred_device = delta.preferred_device;
      return out;
    }
    auto const index = _plan.survivor_chunk_indices[cursor];
    std::shared_ptr<cucascade::data_batch> data;
    if (_entry.tier == cucascade::memory::Tier::GPU) {
      data = get_device_databatch(index);
    } else if (_entry.tier == cucascade::memory::Tier::HOST) {
      data = get_host_databatch(index);
    }
    if (!data) { return {}; }
    auto mask             = index < _mvcc_masks.size() ? _mvcc_masks[index] : mvcc_chunk_mask{};
    auto const converts   = chunk_needs_carrier_conversion(index);
    auto const dest_bytes = converts ? conversion_destination_bytes_for_chunk(index) : 0;
    return {std::move(data), std::move(mask), converts, dest_bytes};
  }

 private:
  std::shared_ptr<cucascade::data_batch> get_host_databatch(std::size_t index)
  {
    if (index >= _entry.host_chunks.size()) { return nullptr; }
    const auto& chunk = _entry.host_chunks.at(index);
    if (!chunk) { return nullptr; }
    if (auto* compressed = dynamic_cast<sirius::compressed_host_representation*>(chunk.get())) {
      auto projected = compressed->select_columns(_column_indices);
      return cucascade::data_batch::make(get_next_batch_id(), std::move(projected));
    }
    auto& host          = chunk->cast<cucascade::host_data_representation>();
    auto data_rep       = host.slice(_column_indices);
    const auto batch_id = get_next_batch_id();
    return cucascade::data_batch::make(
      batch_id,
      std::move(data_rep),
      telemetry::quent_data_batch_probe::create(_telemetry_info, batch_id));
  }

  std::shared_ptr<cucascade::data_batch> get_device_databatch(std::size_t index)
  {
    // GPU-tier compression-enabled pin: one device_pin_chunk per batch, in
    // emission order. Dispatch per chunk on the populated form — a single pin may
    // interleave compressed and uncompressed chunks.
    if (!_entry.device_chunks.empty()) {
      if (index >= _entry.device_chunks.size()) { return nullptr; }
      const auto& chunk = _entry.device_chunks.at(index);
      if (chunk.compressed) {
        // Hand out the projected compressed chunk; decompressed on demand by
        // scan_operator_input::prepare_for_processing.
        auto projected = chunk.compressed->select_columns(_column_indices);
        // Attach the scan's decode request to this projection only — never to
        // the shared pinned chunk, which other queries filter differently.
        // for_chunk narrows it to what this chunk's compression plans make
        // worth asking; row dropping is additionally skipped when this operator
        // carries mvcc keep-masks, since a decode-compacted batch no longer
        // lines up with a positional deleted-row mask.
        std::shared_ptr<const sirius::decompression_pushdown_scan> pushdown_scan;
        if (!_pushdown_req.empty()) {
          auto scan = std::make_shared<const sirius::decompression_pushdown_scan>(_pushdown_req);
          // Row dropping cannot compose with a positional deleted-row mask: a
          // compacted batch no longer lines up with it.
          if (!_mvcc_masks.empty()) { scan = scan->without_row_selection(); }
          if (scan) { pushdown_scan = scan->for_chunk(chunk.compressed->table(), _column_indices); }
        }
        // A PER-BATCH snapshot of the operator's dynamic-filter channel: join
        // builds publish mid-scan, so later batches legitimately carry more
        // filters, and `generation` (the channel's monotonic count, read BEFORE
        // the walk so it never claims filters the walk did not capture) says
        // which snapshot this batch used.
        //
        // THE MAPPING INVARIANT: provider slot order == the ingestible's
        // materialized_column_order == output columns FIRST, IN OUTPUT ORDER,
        // then pure-filter columns (gather_by_primary_index at construction),
        // while the filter set keys by the consumer's OUTPUT-COLUMN position
        // (parquet installs set_consumer_column_remap with
        // scan_plan::output_position_by_column_id). Slot i therefore maps to
        // output position i for every output column, so slot i's filters are
        // exactly filters_for_column(i); trailing pure-filter slots query keys
        // the set can never hold (push_filter rejects non-output columns) and
        // come back empty by construction — no output-arity knowledge is needed
        // here. Same mvcc guard as the row selection above.
        //
        // This drain runs on the metadata thread at query PREPARE, before any
        // join build has published, so this snapshot is almost always EMPTY. It
        // is kept as a free early base; the authoritative snapshot is taken at
        // decode time by scan_operator_input::prepare_for_processing (same
        // builder, same mapping invariant), which replaces this one.
        if (sirius::decompression_pushdown_enabled() && _dynamic_filters && _mvcc_masks.empty()) {
          if (_dynamic_filters->has_filters()) {
            auto snap = sirius::op::scan::snapshot_membership_probes(*_dynamic_filters,
                                                                     _column_indices.size());
            SIRIUS_DECOMPRESSION_PUSHDOWN_DIAG(
              "[decompression-pushdown] join filter attach (drain) channel={}: slots={} "
              "attached={} "
              "generation={} skipped_non_maskable={}",
              static_cast<void const*>(_dynamic_filters.get()),
              _column_indices.size(),
              snap.attached_probes,
              snap.generation,
              snap.skipped_non_mask);
            if (snap.attached_probes > 0) {
              auto const base = pushdown_scan
                                  ? pushdown_scan
                                  : std::make_shared<const sirius::decompression_pushdown_scan>(
                                      sirius::pushdown_request{});
              pushdown_scan = base->with_membership_probes(std::move(snap.probes), snap.generation);
            }
          } else {
            SIRIUS_DECOMPRESSION_PUSHDOWN_DIAG(
              "[decompression-pushdown] join filter attach (drain) channel={}: none published yet "
              "(expected — the drain precedes join publication; the decode-time snapshot "
              "retries)",
              static_cast<void const*>(_dynamic_filters.get()));
          }
        }
        if (pushdown_scan) { projected->set_pushdown_scan(std::move(pushdown_scan)); }
        return cucascade::data_batch::make(get_next_batch_id(), std::move(projected));
      }
      // Uncompressed chunk: project the requested columns (positions into the
      // pinned column set) and serve directly as a gpu_table_representation.
      std::vector<std::shared_ptr<cudf::column>> columns;
      std::vector<cudf::column_view> column_views;
      std::size_t alloc_size = 0;
      for (auto const& col_idx : _column_indices) {
        if (col_idx >= chunk.columns.size() || !chunk.columns[col_idx]) { return nullptr; }
        columns.push_back(chunk.columns[col_idx]);
        column_views.emplace_back(columns.back()->view());
        alloc_size += columns.back()->alloc_size();
      }
      cudf::table_view view(column_views);
      auto* chunk_space = chunk.memory_space ? chunk.memory_space : _entry.memory_space;
      auto gpu_repr     = std::make_unique<::cucascade::gpu_table_representation>(
        view, std::move(columns), alloc_size, *chunk_space, rmm::cuda_stream_view{});
      const auto batch_id = ::sirius::get_next_batch_id();
      return ::cucascade::data_batch::make(
        batch_id,
        std::move(gpu_repr),
        telemetry::quent_data_batch_probe::create(_telemetry_info, batch_id));
    }
    if (index >= _entry.chunk_memory_spaces.size()) { return nullptr; }
    std::vector<std::shared_ptr<cudf::column>> columns;
    std::vector<cudf::column_view> column_views;
    std::size_t alloc_size = 0;
    for (const auto& col_idx : _column_names) {
      const auto& col_chunks = _entry.data_batches_by_column.at(col_idx);
      if (index >= col_chunks.size()) { return nullptr; }
      columns.push_back(col_chunks.at(index));
      column_views.emplace_back(columns.back()->view());
      alloc_size += columns.back()->alloc_size();
    }
    cudf::table_view view(column_views);
    auto* chunk_space = !_entry.chunk_memory_spaces.empty() ? _entry.chunk_memory_spaces.at(index)
                                                            : _entry.memory_space;
    auto gpu_repr     = std::make_unique<::cucascade::gpu_table_representation>(
      view, std::move(columns), alloc_size, *chunk_space, rmm::cuda_stream_view{});
    const auto batch_id = ::sirius::get_next_batch_id();
    return ::cucascade::data_batch::make(
      batch_id,
      std::move(gpu_repr),
      telemetry::quent_data_batch_probe::create(_telemetry_info, batch_id));
  }

  // True when scan normalization will cast the recorded stored @p carrier to @p target: the same
  // predicate normalize_physical_schema applies per column.
  [[nodiscard]] bool will_convert(cudf::data_type carrier, cudf::data_type target) const
  {
    if (carrier == target) { return false; }
    return sirius::can_restore_to(carrier, target) ||
           (_has_physical_overrides && sirius::can_narrow_to(carrier, target));
  }

  // The plan carrier scan normalization holds selected slot @p sel to, or `std::nullopt` when the
  // column reaches no normalization target: a pure-filter column, which both ingestibles
  // materialize after every output column, or a scan that normalizes nothing.
  [[nodiscard]] std::optional<cudf::data_type> target_for(std::size_t sel) const
  {
    return sel < _normalization_targets.size()
             ? std::optional<cudf::data_type>{_normalization_targets[sel]}
             : std::nullopt;
  }

  // True when any selected column of chunk @p index converts. Decided from types alone, so a
  // chunk whose row count is unreadable still reports the conversion and the caller keeps its
  // conservative bound.
  [[nodiscard]] bool chunk_needs_carrier_conversion(std::size_t index) const
  {
    auto const& matrix = _entry.column_storage;
    if (matrix.empty() || index >= matrix.size()) { return false; }
    auto const& chunk_meta = matrix[index];
    for (std::size_t sel = 0; sel < _column_indices.size(); ++sel) {
      auto const entry_pos = _column_indices[sel];
      if (entry_pos >= chunk_meta.size()) { continue; }
      auto const target = target_for(sel);
      if (target && will_convert(chunk_meta[entry_pos].carrier, *target)) { return true; }
    }
    return false;
  }

  // Estimate the cast destinations for one chunk from the target width and row count. Include a
  // validity mask when the source is nullable. Compressed representations do not expose
  // per-column nullability, so their estimate always includes a mask and may overestimate a
  // non-nullable destination. Return zero when a destination cannot be sized; the caller then uses
  // the maximum-expansion fallback. The fixed-width guard protects cudf::size_of.
  [[nodiscard]] std::size_t conversion_destination_bytes_for_chunk(std::size_t index) const
  {
    auto const& matrix = _entry.column_storage;
    if (matrix.empty() || index >= matrix.size()) { return 0; }
    auto const& chunk_meta = matrix[index];
    std::size_t total      = 0;
    for (std::size_t sel = 0; sel < _column_indices.size(); ++sel) {
      auto const entry_pos = _column_indices[sel];
      if (entry_pos >= chunk_meta.size()) { continue; }
      auto const target = target_for(sel);
      if (!target || !will_convert(chunk_meta[entry_pos].carrier, *target)) { continue; }
      if (!cudf::is_fixed_width(*target)) { return 0; }
      std::size_t rows   = 0;
      bool has_null_mask = false;
      if (_entry.tier == cucascade::memory::Tier::GPU) {
        if (!_entry.device_chunks.empty()) {
          if (index >= _entry.device_chunks.size()) { return 0; }
          auto const& chunk = _entry.device_chunks[index];
          if (chunk.compressed) {
            rows          = static_cast<std::size_t>(chunk.compressed->num_rows());
            has_null_mask = true;
          } else {
            if (entry_pos >= chunk.columns.size() || !chunk.columns[entry_pos]) { return 0; }
            rows          = static_cast<std::size_t>(chunk.columns[entry_pos]->size());
            has_null_mask = chunk.columns[entry_pos]->nullable();
          }
        } else {
          auto const it = _entry.data_batches_by_column.find(_column_names[sel]);
          if (it == _entry.data_batches_by_column.end() || index >= it->second.size() ||
              !it->second[index]) {
            return 0;
          }
          auto const& column = *it->second[index];
          rows               = static_cast<std::size_t>(column.size());
          has_null_mask      = column.nullable();
        }
      } else {
        if (index >= _entry.host_chunks.size() || !_entry.host_chunks[index]) { return 0; }
        if (auto const* compressed = dynamic_cast<sirius::compressed_host_representation const*>(
              _entry.host_chunks[index].get())) {
          rows          = static_cast<std::size_t>(compressed->num_rows());
          has_null_mask = true;
        } else {
          auto const* host = dynamic_cast<cucascade::host_data_representation const*>(
            _entry.host_chunks[index].get());
          if (host == nullptr || !host->get_host_table()) { return 0; }
          auto const& host_columns = host->get_host_table()->columns;
          if (entry_pos >= host_columns.size()) { return 0; }
          rows          = static_cast<std::size_t>(host_columns[entry_pos].num_rows);
          has_null_mask = host_columns[entry_pos].has_null_mask;
        }
      }
      total += rows * cudf::size_of(*target);
      if (has_null_mask) {
        total += cudf::bitmask_allocation_size_bytes(static_cast<cudf::size_type>(rows));
      }
    }
    return total;
  }

  cached_scan_plan _plan;
  std::vector<std::string> _column_names;
  std::vector<size_t> _column_indices;
  /// The scan's filter as the decompressor can use it, parallel to
  /// @c _column_indices. Only the GPU-tier compressed path consumes it: the
  /// host path would have to parse the chunk header to know what its plans can
  /// exploit, and that tier is not where the decode costs anything.
  sirius::pushdown_request _pushdown_req;
  /// The operator's dynamic-filter channel (may be null). NOT a snapshot: the
  /// per-batch attach in get_device_databatch snapshots it at serve time so
  /// batches pick up join filters as they are published mid-scan.
  std::shared_ptr<sirius::op::sirius_dynamic_filter_set> _dynamic_filters;
  const pinned_entry& _entry;
  telemetry::batch_telemetry_info _telemetry_info;
  /// This provider's own copy of the entry's per-chunk keep-masks (empty for
  /// parquet pins); word storage is shared through each mask's owning pointer.
  mvcc_chunk_mask_set _mvcc_masks;
  /// This operator's insert-delta splits, yielded after the resident chunks
  /// (staging and mask words shared with sibling operators' cuts).
  std::vector<insert_delta_split> _delta_splits;
  // The scan's carrier targets in output order, which is also the order the served columns
  // arrive in (see the note at the make_provider_for_pinned_entry call site). Shorter than the
  // served column list exactly when the scan materializes trailing pure-filter columns.
  std::vector<cudf::data_type> _normalization_targets;
  // Whether the scan carries an explicit plan sidecar. Normalization narrows a stored carrier
  // only with one installed, so it belongs in the conversion predicate.
  bool _has_physical_overrides{false};
  std::atomic<std::size_t> _index{0};
};

/// The scan's column primary index per served slot: @p selected_columns are
/// positions in the pinned entry's column list, while a filter analysis is
/// keyed by primary index. Out-of-range slots map to a sentinel no filter can
/// match, so they simply carry no request.
std::vector<std::size_t> primary_index_by_slot(pinned_entry const& entry,
                                               std::span<std::size_t const> selected_columns)
{
  std::vector<std::size_t> mapping(selected_columns.size(),
                                   std::numeric_limits<std::size_t>::max());
  for (std::size_t i = 0; i < selected_columns.size(); ++i) {
    auto const entry_pos = selected_columns[i];
    if (entry_pos >= entry.cache_info.column_ids.size()) { continue; }
    mapping[i] = static_cast<std::size_t>(entry.cache_info.column_ids[entry_pos].GetPrimaryIndex());
  }
  return mapping;
}

/// Filter view extracted from the scan's ingestible info: the pushed-down TableFilterSet plus the
/// scan's column_ids its keys index into.
struct scan_filter_view {
  duckdb::TableFilterSet const* table_filters{nullptr};
  duckdb::vector<duckdb::ColumnIndex> const* column_ids{nullptr};
};

scan_filter_view extract_scan_filters(op::scan::ingestible_table_info const& info)
{
  if (auto const* p = dynamic_cast<op::scan::parquet_ingestible_table_info const*>(&info)) {
    return {p->table_filters.get(), &p->column_ids};
  } else if (auto const* p =
               dynamic_cast<op::scan::duckdb_native_ingestible_table_info const*>(&info)) {
    return {p->table_filters.get(), &p->column_ids};
  }
  return {};
}

/// Strip a leading "file://" scheme (case-insensitive) so the path can be
/// resolved by a local-file backend.
std::string normalize_path(std::string const& p)
{
  static constexpr std::string_view kFile = "file://";
  if (p.size() > kFile.size()) {
    bool is_file_uri = true;
    for (std::size_t i = 0; i < kFile.size(); ++i) {
      if (std::tolower(static_cast<unsigned char>(p[i])) != static_cast<unsigned char>(kFile[i])) {
        is_file_uri = false;
        break;
      }
    }
    if (is_file_uri) { return p.substr(kFile.size()); }
  }
  return p;
}

}  // namespace

sirius_scan_manager::sirius_scan_manager(
  const scan_manager_config& config,
  cucascade::memory::memory_reservation_manager& reservation_manager,
  std::shared_ptr<const sirius::memory::topology_index> topology_index)
  : _config(config),
    _reservation_manager(reservation_manager),
    _topology_index(std::move(topology_index)),
    _thread_pool(_config.thread_pool.num_threads + 1,
                 _config.thread_pool.thread_name_prefix,
                 _config.thread_pool.cpu_affinity_list),
    _dispatcher(
      std::make_unique<exec::scoped_dispatcher>(_thread_pool, _thread_pool.num_threads())),
    _ioctx_registry(config, reservation_manager)
{
  if (!_topology_index) {
    throw std::invalid_argument("[sirius_scan_manager] topology_index must be non-null");
  }

  // scan_manager always owns an io_ctx: sirius_datasource (uring) on the
  // fast path, kvikio_context as the universal fallback so the rest of the
  // scan path (split_provider, scan tasks) always has an ioctx to
  // talk to.  kvikio_context wraps cudf::io::datasource so the read path
  // is identical from the caller's point of view.  Both are built by the
  // ioctx registry, which sources the reactor staging resource from the
  // reservation manager it was constructed with.
  if (_config.use_sirius_datasource) {
    _io_ctx = _ioctx_registry.make_ioctx(sirius::io::io_context_type::uring);
    if (!_io_ctx) {
      throw std::runtime_error("[sirius_scan_manager] failed to create uring io_context");
    }
    SIRIUS_LOG_DEBUG("[sirius_scan_manager] sirius_datasource enabled (uring_ioctx n_reactors={})",
                     _config.uring_n_reactors);
  } else {
    if (_topology_index->gpu_ids().size() > 1) {
      throw std::runtime_error(
        "[sirius_scan_manager] kvikio_context fallback (use_sirius_datasource=false) "
        "does not support multi-GPU; topology reports " +
        std::to_string(_topology_index->gpu_ids().size()) +
        " GPUs.  Enable use_sirius_datasource for multi-GPU runs.");
    }
    _io_ctx = _ioctx_registry.make_ioctx(sirius::io::io_context_type::kvikio);
    if (!_io_ctx) {
      throw std::runtime_error("[sirius_scan_manager] failed to create kvikio io_context");
    }
    SIRIUS_LOG_DEBUG(
      "[sirius_scan_manager] sirius_datasource disabled — using kvikio_context fallback");
  }

  // Build the prefetching cache on the ioctx.  Budget=0 keeps the
  // cache unarmed (no background threads); we pass that whenever the
  // user has disabled prefetching so the construction is always
  // unconditional and there's no "is the cache present" branch to
  // worry about in callers.
  if (_config.enable_prefetch_cache && _io_ctx->can_use_prefetching_cache()) {
    _io_ctx->initialize_cache(reservation_manager, _config.cache, _topology_index);
  }

  // Reactors are built parked; start() launches their worker threads and
  // allocates per-reactor staging.  No-op for the kvikio fallback (no reactors).
  _io_ctx->start();
}

sirius_scan_manager::~sirius_scan_manager()
{
  if (_io_ctx && _io_ctx->cache()) {
    SIRIUS_LOG_INFO("[sirius_scan_manager] cache summary: {}", _io_ctx->cache()->summary());
  }
  // Drain the dispatcher (and the worker pool) first so no in-flight
  // metadata-scan / sequencer task can still be reaching into the
  // cache via _io_ctx when we tear it down below.
  stop();
  // Tear down the cache (which owns its buffer_pool).  shutdown_cache drains
  // in-flight IO before the pool is destroyed, so callbacks release their
  // chunks safely.
  if (_io_ctx) { _io_ctx->shutdown_cache(); }
  // Same drain for any path-routed ioctxs; their reactors stop when the
  // shared_ptrs in _routed_io_ctxs are released (member destruction below).
  std::lock_guard lk{_routed_io_ctxs_mtx};
  for (auto& [type, io_ctx] : _routed_io_ctxs) {
    if (io_ctx) { io_ctx->shutdown_cache(); }
  }
}

parquet_bind_result sirius_scan_manager::describe_parquet(std::string const& uri)
{
  // Footer-probe only when we will actually read + parse the footer.  On a warm
  // re-bind the metadata_store already holds the parsed footer, so a suffix GET
  // would download footer bytes we won't reuse — a plain HEAD resolves the size.
  auto const cache_key     = normalize_path(uri);
  auto const io_ctx        = ioctx_for_path(uri);
  bool const footer_cached = io_ctx && io_ctx->metadata_store().get_metadata(cache_key) != nullptr;
  auto const hint =
    footer_cached ? sirius::io::open_hint::generic : sirius::io::open_hint::parquet_footer_probe;

  auto datasource = create_datasource(uri, hint);
  if (!datasource) {
    throw std::runtime_error("[sirius_scan_manager::describe_parquet] no backend supports URI: " +
                             uri);
  }

  // Reuse a previously parsed footer when present — a prior bind or scan of the
  // same file parks it in the ioctx metadata store, which lives for the ioctx's
  // lifetime. On a miss, fetch + Thrift-parse the footer once and park it so the
  // subsequent scan reuses it. Mirrors parquet_gpu_ingestible::build_file_scan_info,
  // so the footer is parsed exactly once per file per process.
  std::shared_ptr<cudf::io::parquet::FileMetaData const> file_metadata;
  if (auto cached = datasource->metadata()) {
    if (auto pm = std::dynamic_pointer_cast<op::scan::parquet_metadata>(std::move(cached))) {
      file_metadata = pm->file_metadata();
    }
  }
  if (!file_metadata) {
    auto footer_buffer         = cudf::io::parquet::fetch_footer_to_host(*datasource);
    auto const footer_byte_len = footer_buffer->size();
    auto reader_options        = cudf::io::parquet_reader_options::builder().build();
    cudf::io::parquet::experimental::hybrid_scan_reader reader{
      cudf::host_span<std::uint8_t const>(footer_buffer->data(), footer_buffer->size()),
      reader_options};
    file_metadata =
      std::make_shared<cudf::io::parquet::FileMetaData const>(reader.parquet_metadata());
    [[maybe_unused]] auto const stored = datasource->store_metadata(
      std::make_shared<op::scan::parquet_metadata>(file_metadata, footer_byte_len));
  }

  auto schema = sirius::io::parquet_helpers::extract_schema(*file_metadata);

  parquet_bind_result result;
  result.return_types   = std::move(schema.types);
  result.names          = std::move(schema.names);
  result.object_size    = datasource->size();
  result.total_num_rows = static_cast<std::size_t>(file_metadata->num_rows);
  return result;
}

void sirius_scan_manager::prepare_for_query(const sirius::planner::query& query,
                                            bool enable_pinned_zone_map_pruning,
                                            const std::vector<int>& allocated_gpu_ids)
{
  _pruning_enabled = enable_pinned_zone_map_pruning;
  reset();

  if (_io_ctx && _io_ctx->cache()) {
    SIRIUS_LOG_INFO("[sirius_scan_manager] cache summary: {}", _io_ctx->cache()->summary());
    _io_ctx->cache()->prepare_for_query(query);
  }

  // Routed ioctxs (e.g. the restful context serving s3://) are built lazily and
  // reused across queries; advance their caches to this query too, or a routed
  // cache's epoch freezes at build time and a later query serves the prior
  // query's cached chunks as current.
  {
    std::lock_guard lk{_routed_io_ctxs_mtx};
    for (auto& [type, io_ctx] : _routed_io_ctxs) {
      if (io_ctx && io_ctx->cache()) { io_ctx->cache()->prepare_for_query(query); }
    }
  }

  auto round_robin = std::make_shared<round_robin_strategy>(allocated_gpu_ids);

  _metadata_processor = std::make_unique<load_balancing_scan_batch_coalescer>();

  std::vector<cached_assignment> cached_assignments;
  for (auto const& scan_op : query.get_scan_operators()) {
    if (scan_op->type != ::sirius::op::SiriusPhysicalOperatorType::GPU_SCAN) { continue; }
    auto* op = &scan_op->Cast<op::scan::sirius_gpu_scan_operator>();
    if (_providers_by_op.find(op) != _providers_by_op.end()) { continue; }
    _metadata_processor->register_pipeline(op, round_robin);
    // On a pinned-cache hit the coalescer serves this operator from a cached
    // batch_provider (process_cached_entries); skip the disk-reading
    // split_provider entirely so no read is issued for the cached scan. The
    // provider itself is built after the mask jobs run (below), so it can
    // take a copy of the entry's finished mask set.
    if (auto assignment = try_match_cached_entry(op)) {
      cached_assignments.push_back(std::move(*assignment));
      _scan_op_order.push_back(op);
      continue;
    }
    auto provider = std::make_unique<split_provider>(
      op->get_ingestible(),
      [this](std::string_view file_path) -> std::shared_ptr<io::sirius_ioctx> {
        auto io_ctx = ioctx_for_path(file_path);
        if (!io_ctx) {
          throw std::runtime_error("scan_manager: no backend supports path: " +
                                   std::string(file_path));
        }
        return io_ctx;
      });
    _providers_by_op.emplace(op, std::move(provider));
    _scan_op_order.push_back(op);
  }

  if (_scan_op_order.empty()) {
    SIRIUS_LOG_WARN(
      "[sirius_scan_manager::prepare_for_query] no GPU scan operators found in query");
    return;
  }

  _checkpoint_locks.reserve(_pending_mvcc_mask_jobs.size());
  for (auto const& request : _pending_mvcc_mask_jobs) {
    _checkpoint_locks.push_back(
      duckdb::DuckTransactionManager::Get(request.storage->GetAttached()).SharedCheckpointLock());
  }

  // A manual CHECKPOINT can replace DuckDB's on-disk base while the pinned
  // cache still holds the preceding image. Auto-checkpoint is suppressed for
  // pins, but explicit checkpoints remain possible; reject a changed database
  // generation before serving any cached rows.
  for (auto const& request : _pending_mvcc_mask_jobs) {
    auto const* block_manager = dynamic_cast<duckdb::SingleFileBlockManager const*>(
      &request.storage->GetAttached().GetStorageManager().GetBlockManager());
    if (block_manager == nullptr ||
        block_manager->GetCheckpointIteration() != request.metadata.checkpoint_iteration) {
      throw std::runtime_error(
        "Sirius cannot safely scan pinned DuckDB table '" + request.entry_name +
        "': its database was checkpointed after pin_table, so the pinned cache may be stale. "
        "Run CALL unpin_table('" +
        request.entry_name + "'), then pin_table again");
    }
  }

  // Compute this query's duckdb MVCC keep-masks BEFORE serving starts, so
  // masks are finished plain buffers when the sequencer runs (it walks all
  // ops' slots serially; a wait there would be head-of-line blocking across
  // every scan op). The dispatcher is fresh and otherwise idle here. Errors
  // are loud: transparent execution can replay its retained CPU plan, while
  // fallback-disabled callers receive the error instead of stale results.
  if (!_pending_mvcc_mask_jobs.empty()) {
    run_mvcc_mask_jobs(
      _pending_mvcc_mask_jobs, *_dispatcher, _reservation_manager, *_topology_index);
  }
  // Insert-delta jobs block in prepare for the same reason: staging and
  // masks must be finished before serving starts. No-op when no pinned
  // table has rows beyond its prefix.
  if (!_pending_insert_delta_jobs.empty()) {
    std::vector<int> const delta_gpu_ids(allocated_gpu_ids.begin(), allocated_gpu_ids.end());
    run_insert_delta_jobs(_pending_insert_delta_jobs,
                          *_dispatcher,
                          _reservation_manager,
                          *_topology_index,
                          delta_gpu_ids);
  }

  // Provider handoff, deferred to behind the mask run so ownership is a
  // sequential handoff instead of sharing: each matched op's provider takes
  // its own copy of its entry's completed set (slot-sized — word storage is
  // shared through the masks' owning pointers), then the requests release
  // theirs. The sequencer only spawns in start_metadata_processing, so
  // serving still starts with finished masks.
  for (auto& assignment : cached_assignments) {
    mvcc_chunk_mask_set masks;  // stays empty for parquet pins
    std::vector<insert_delta_split> delta_splits;
    if (assignment.entry->mvcc != nullptr) {
      auto request = std::ranges::find_if(
        _pending_mvcc_mask_jobs,
        [&](mvcc_mask_job_request const& r) { return r.entry_name == assignment.entry_name; });
      if (request == _pending_mvcc_mask_jobs.end()) {
        throw std::runtime_error(
          "[sirius_scan_manager::prepare_for_query] no mask job was queued for mvcc-pinned "
          "entry '" +
          assignment.entry_name + "'");
      }
      masks = request->masks;

      auto delta_request = std::ranges::find_if(
        _pending_insert_delta_jobs,
        [&](insert_delta_job_request const& r) { return r.entry_name == assignment.entry_name; });
      if (delta_request != _pending_insert_delta_jobs.end() && !delta_request->bundles.empty()) {
        auto const* duckdb_info =
          dynamic_cast<op::scan::duckdb_native_ingestible_table_info const*>(
            &assignment.op->get_ingestible().table_info());
        if (duckdb_info == nullptr) {
          throw std::runtime_error(
            "[sirius_scan_manager::prepare_for_query] delta bundles pending for a non-duckdb "
            "scan of entry '" +
            assignment.entry_name + "'");
        }
        auto io_ctx = ioctx_for_path(duckdb_info->db_path);
        if (!io_ctx) {
          throw std::runtime_error(
            "[sirius_scan_manager::prepare_for_query] no io backend supports the pinned "
            "database path '" +
            duckdb_info->db_path + "'");
        }
        auto const* sf_bm = dynamic_cast<duckdb::SingleFileBlockManager const*>(
          &duckdb_info->storage->GetAttached().GetStorageManager().GetBlockManager());
        delta_splits = cut_delta_splits_for_op(*delta_request,
                                               duckdb_info->projected_cols,
                                               io_ctx->open_datasource(duckdb_info->db_path),
                                               sf_bm);
        SIRIUS_LOG_INFO(
          "[sirius_scan_manager] operator '{}' serves {} insert-delta split(s) of pinned entry "
          "'{}' ({} delta row(s))",
          assignment.op->get_operator_id(),
          delta_splits.size(),
          assignment.entry_name,
          delta_request->plan.delta_rows());
      }
    }
    // The scan's filter as the decompressor can use it: columns the query only
    // ever tests for equality and never projects can come back as the boolean
    // answer instead of values, and range conjuncts can drop rows while the
    // batch decodes rather than after it. The ingestible analysed its filter
    // once at bind; here it is only mapped onto the slots this entry serves.
    auto const slot_map = primary_index_by_slot(*assignment.entry, assignment.columns);
    auto pushdown_req   = sirius::op::build_pushdown_request(
      assignment.op->get_ingestible().filter_analysis(), slot_map);
    // The provider captures the operator's dynamic-filter CHANNEL (not a
    // snapshot) so each compressed batch can pick up join-published filters at
    // serve time.
    std::shared_ptr<sirius::op::sirius_dynamic_filter_set> dynamic_filters;
    if (sirius::decompression_pushdown_enabled()) {
      auto const* pq = dynamic_cast<op::scan::parquet_ingestible_table_info const*>(
        &assignment.op->get_ingestible().table_info());
      if (pq != nullptr) { dynamic_filters = pq->sirius_dynamic_filters; }
      // Channel identity: this pointer must match the one the hash join
      // publishes into (both resolve through the generator's channel map, keyed
      // by duckdb's DynamicTableFilterSet pointer) and the one the decode-time
      // snapshot logs.
      SIRIUS_DECOMPRESSION_PUSHDOWN_DIAG(
        "[decompression-pushdown] entry '{}': join filter channel={} published_now={} decode "
        "request: "
        "{} slot(s), covers_whole_filter={}",
        assignment.entry_name,
        static_cast<void const*>(dynamic_filters.get()),
        dynamic_filters ? dynamic_filters->has_filters() : false,
        pushdown_req.columns.size(),
        pushdown_req.ranges_cover_whole_filter);
    } else {
      // Answering an equality off a dictionary is independent of the gate; row
      // dropping is not, so give it up when the gate is off.
      auto const unfiltered =
        sirius::decompression_pushdown_scan{std::move(pushdown_req)}.without_row_selection();
      pushdown_req = unfiltered ? unfiltered->request() : sirius::pushdown_request{};
    }
    // The provider charges a served column only for the cast scan normalization will make, so
    // it needs the scan's carrier targets. They are passed in output order, which is also the
    // order the cached chunks are served in: assignment.columns follows the ingestible's
    // materialized order, and both ingestibles materialize the output columns first (in output
    // order) and any pure-filter columns after them, so served slot k is output column k for
    // every k below the output arity. Only a hive-partition output column would decouple the
    // two -- it occupies an output position with no materialized column -- and such a scan
    // cannot serve from a pin, since a pin never captures a partition column and a scan
    // requesting one therefore misses the cache.
    auto provider = make_provider_for_pinned_entry(*assignment.entry,
                                                   assignment.columns,
                                                   std::move(assignment.plan),
                                                   assignment.op->batch_telemetry(),
                                                   std::move(masks),
                                                   std::move(delta_splits),
                                                   assignment.op->normalization_targets(),
                                                   assignment.op->has_physical_overrides(),
                                                   std::move(pushdown_req),
                                                   std::move(dynamic_filters));
    _metadata_processor->use_cached_entries_for_pipeline(assignment.op, std::move(provider));
  }
  _pending_mvcc_mask_jobs.clear();
  _pending_insert_delta_jobs.clear();

  start_metadata_processing();
}

void sirius_scan_manager::start_metadata_processing()
{
  _metadata_processor->spawn_workers(*_dispatcher);
  for (auto* op : _scan_op_order) {
    auto it = _providers_by_op.find(op);
    if (it == _providers_by_op.end()) { continue; }
    it->second->run(*_dispatcher, _metadata_processor->get_split_provider_bridge(op));
  }
  maybe_start_memory_prefetcher();
}

void sirius_scan_manager::maybe_start_memory_prefetcher()
{
  const auto& cfg = _config.memory_prefetcher;
  if (!cfg.enable) { return; }

  // Prototype scope: a single GPU space. Multi-GPU needs the task creator's
  // NUMA-locality derivation to pick the per-batch target device; converting
  // to the wrong space would strand data cross-device.
  auto gpu_spaces = _reservation_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  if (gpu_spaces.size() != 1) {
    SIRIUS_LOG_WARN(
      "[memory_prefetcher] disabled: prototype supports exactly 1 GPU space (found {})",
      gpu_spaces.size());
    return;
  }
  auto* gpu_space = _reservation_manager.get_memory_space(cucascade::memory::Tier::GPU,
                                                          gpu_spaces.front()->get_device_id());
  if (gpu_space == nullptr) { return; }

  std::vector<std::shared_ptr<split_connector>> connectors;
  connectors.reserve(_scan_op_order.size());
  for (auto* op : _scan_op_order) {
    connectors.push_back(op->get_shared_split_connector());
  }

  _prefetcher = std::make_unique<memory_prefetcher>(cfg, std::move(connectors), gpu_space);
}

std::shared_ptr<sirius::io::sirius_datasource> sirius_scan_manager::create_datasource(
  std::string_view path, sirius::io::open_hint hint)
{
  auto file_path = normalize_path(std::string(path));
  auto io_ctx    = ioctx_for_path(file_path);
  if (!io_ctx) { return nullptr; }  // no backend supports the path
  // Real I/O / HEAD / auth / missing-object errors propagate as exceptions;
  // only "no backend" is reported as nullptr (callers map it to that message).
  return io_ctx->open_datasource(file_path, hint);
}

void sirius_scan_manager::list_objects_paged(
  std::string const& s3_prefix_uri,
  std::size_t page_size,
  std::function<bool(sirius::io::s3::list_objects_v2_page const&)> const& sink,
  std::optional<std::size_t> max_scanned)
{
  // Hand-split rather than uri_parser::parse — a LIST prefix URI legitimately
  // has an EMPTY key part (bucket-root glob: "s3://bucket/"), which parse()
  // rejects as an object URI.
  constexpr std::string_view k_scheme = "s3://";
  if (s3_prefix_uri.size() <= k_scheme.size() ||
      s3_prefix_uri.compare(0, k_scheme.size(), k_scheme) != 0) {
    throw std::runtime_error("sirius_scan_manager::list_objects_paged: malformed prefix URI '" +
                             s3_prefix_uri + "'");
  }
  auto const rest_uri     = std::string_view{s3_prefix_uri}.substr(k_scheme.size());
  auto const bucket_slash = rest_uri.find('/');
  auto const bucket       = rest_uri.substr(0, bucket_slash);
  auto const prefix =
    bucket_slash == std::string_view::npos ? std::string_view{} : rest_uri.substr(bucket_slash + 1);
  if (bucket.empty()) {
    throw std::runtime_error("sirius_scan_manager::list_objects_paged: malformed prefix URI '" +
                             s3_prefix_uri + "'");
  }
  // Routing probe only (scheme dispatch, never touches the network): the
  // backend checkers parse() their input and reject an empty object key, so a
  // bucket-root prefix routes via a placeholder key.
  auto const route_probe =
    "s3://" + std::string(bucket) + "/" + (prefix.empty() ? "_" : std::string(prefix));
  auto io_ctx = ioctx_for_path(route_probe);
  auto* rest  = dynamic_cast<sirius::io::rest::rest_ioctx*>(io_ctx.get());
  if (rest == nullptr) {
    throw std::runtime_error("sirius_scan_manager::list_objects_paged: '" + s3_prefix_uri +
                             "' does not route to an object-store backend that supports LIST");
  }
  rest->list_objects_paged(bucket, prefix, page_size, sink, max_scanned);
}

std::size_t sirius_scan_manager::s3_list_max_matches(std::string const& s3_uri)
{
  // Route by scheme only (no LIST call) to read the backend's configured cap.
  // Same bucket-root-safe placeholder-key probe as list_objects_paged.
  constexpr std::string_view k_scheme = "s3://";
  if (s3_uri.size() <= k_scheme.size() || s3_uri.compare(0, k_scheme.size(), k_scheme) != 0) {
    throw std::runtime_error("sirius_scan_manager::s3_list_max_matches: malformed URI '" + s3_uri +
                             "'");
  }
  auto const rest_uri     = std::string_view{s3_uri}.substr(k_scheme.size());
  auto const bucket_slash = rest_uri.find('/');
  auto const bucket       = rest_uri.substr(0, bucket_slash);
  auto const prefix =
    bucket_slash == std::string_view::npos ? std::string_view{} : rest_uri.substr(bucket_slash + 1);
  auto const route_probe =
    "s3://" + std::string(bucket) + "/" + (prefix.empty() ? "_" : std::string(prefix));
  auto io_ctx = ioctx_for_path(route_probe);
  auto* rest  = dynamic_cast<sirius::io::rest::rest_ioctx*>(io_ctx.get());
  if (rest == nullptr) {
    throw std::runtime_error("sirius_scan_manager::s3_list_max_matches: '" + s3_uri +
                             "' does not route to an object-store backend that supports LIST");
  }
  return rest->list_max_matches();
}

std::shared_ptr<sirius::io::sirius_ioctx> sirius_scan_manager::ioctx_for_path(std::string_view path)
{
  // Normalize here so every caller (incl. the scan resolver, which forwards raw
  // ingestible paths) routes `file://` the same way create_datasource does.
  auto file_path = normalize_path(std::string(path));
  auto type      = _ioctx_registry.lookup_path(file_path);
  if (!type) { return nullptr; }
  // The local default `_io_ctx` already serves uring/kvikio; only an off-default
  // backend (e.g. s3:// -> restful) needs a separate, lazily-built context.
  if (_io_ctx && _io_ctx->type() == *type) { return _io_ctx; }

  {
    std::lock_guard lk{_routed_io_ctxs_mtx};
    if (auto it = _routed_io_ctxs.find(*type); it != _routed_io_ctxs.end()) { return it->second; }
  }
  // Build outside the map mutex: make_ioctx/start spawn reactor threads and
  // initialize_cache allocates, so holding _routed_io_ctxs_mtx across them would
  // park every concurrent lookup behind one long critical section. The build
  // mutex serializes builders instead, so two first-touches of the same type
  // never construct twice (a losing ioctx would need drain/stop teardown).
  std::lock_guard build_lk{_routed_io_ctxs_build_mtx};
  {
    std::lock_guard lk{_routed_io_ctxs_mtx};
    if (auto it = _routed_io_ctxs.find(*type); it != _routed_io_ctxs.end()) { return it->second; }
  }
  auto io_ctx = _ioctx_registry.make_ioctx(*type);
  if (!io_ctx) { return nullptr; }
  io_ctx->start();
  if (_config.enable_prefetch_cache && io_ctx->can_use_prefetching_cache()) {
    io_ctx->initialize_cache(_reservation_manager, _config.cache, _topology_index);
  }
  std::lock_guard lk{_routed_io_ctxs_mtx};
  auto [it, inserted] = _routed_io_ctxs.emplace(*type, std::move(io_ctx));
  return it->second;
}

void sirius_scan_manager::reset()
{
  // Stop the prefetcher first: it holds shared_ptrs to the operators'
  // connectors and must not convert batches while per-query state is torn down.
  _prefetcher.reset();
  _dispatcher->request_stop();
  _dispatcher->wait_for_all();
  _scan_op_order.clear();
  _providers_by_op.clear();
  _pending_mvcc_mask_jobs.clear();
  _pending_insert_delta_jobs.clear();
  _metadata_processor.reset();
  _checkpoint_locks.clear();
  _dispatcher = std::make_unique<exec::scoped_dispatcher>(_thread_pool, _thread_pool.num_threads());
}

void sirius_scan_manager::start() {}

void sirius_scan_manager::stop()
{
  reset();
  _thread_pool.stop();
}

namespace {

// Gather positions into @p cached_ids for each requested primary (storage) index, in the
// given order. Empty when any requested column is absent — i.e. the cache is not a superset.
std::vector<std::size_t> gather_by_primary_index(
  duckdb::vector<duckdb::ColumnIndex> const& cached_ids,
  std::vector<std::size_t> const& requested_primary_indices)
{
  std::unordered_map<duckdb::idx_t, std::size_t> pos;
  pos.reserve(cached_ids.size());
  for (std::size_t i = 0; i < cached_ids.size(); ++i) {
    pos.emplace(cached_ids[i].GetPrimaryIndex(), i);
  }
  std::vector<std::size_t> projection;
  projection.reserve(requested_primary_indices.size());
  for (auto const primary_idx : requested_primary_indices) {
    auto it = pos.find(primary_idx);
    if (it == pos.end()) { return {}; }  // cache lacks a requested column
    projection.push_back(it->second);
  }
  return projection;
}

// Gather projection that lets a cache holding @p cached_ids (by primary/storage
// index) serve a scan requesting @p requested_ids: for each requested column,
// its position within @p cached_ids, in the requested order. Empty when any
// requested column is absent — i.e. the cache is not a column superset.
std::vector<std::size_t> column_superset_projection(
  duckdb::vector<duckdb::ColumnIndex> const& cached_ids,
  duckdb::vector<duckdb::ColumnIndex> const& requested_ids)
{
  std::vector<std::size_t> requested_primary_indices;
  requested_primary_indices.reserve(requested_ids.size());
  for (auto const& c : requested_ids) {
    requested_primary_indices.push_back(c.GetPrimaryIndex());
  }
  return gather_by_primary_index(cached_ids, requested_primary_indices);
}

// column_ids-aligned names: for each column_ids[i], the full-schema name at its
// primary (storage) index — the keys data_batches_by_column / the gather use.
std::vector<std::string> aligned_column_names(duckdb::vector<std::string> const& full_names,
                                              duckdb::vector<duckdb::ColumnIndex> const& column_ids)
{
  std::vector<std::string> out;
  out.reserve(column_ids.size());
  for (auto const& c : column_ids) {
    auto const p = static_cast<std::size_t>(c.GetPrimaryIndex());
    out.push_back(p < full_names.size() ? full_names[p] : std::string{});
  }
  return out;
}

}  // namespace

cache_entry_info cache_entry_info::from(const op::scan::ingestible_table_info& info)
{
  cache_entry_info ci;
  if (auto const* p = dynamic_cast<op::scan::parquet_ingestible_table_info const*>(&info)) {
    // Cache identity is spelling-independent, so canonicalize here — at the
    // identity boundary — rather than in the bind info, whose resolved_file_paths
    // stay as bound so Hive partition parsing sees the original (un-symlink-
    // resolved) path.
    ci.resolved_file_paths = p->resolved_file_paths;
    op::scan::canonicalize_scan_file_paths(ci.resolved_file_paths);
    ci.column_ids = p->column_ids;
    ci.names      = aligned_column_names(p->names, p->column_ids);
  } else if (auto const* d =
               dynamic_cast<op::scan::duckdb_native_ingestible_table_info const*>(&info)) {
    ci.catalog_name = d->catalog_name;
    ci.schema_name  = d->schema_name;
    ci.table_name   = d->table_name;
    ci.column_ids   = d->column_ids;
    ci.names        = aligned_column_names(d->names, d->column_ids);
  }
  return ci;
}

std::vector<std::size_t> cache_entry_info::can_serve_with_columns(
  const op::scan::ingestible_table_info& other) const
{
  // A parquet pin serves a parquet scan over the same file set; a duckdb pin
  // serves a duckdb scan over the same catalog.schema.table. A cache of one format
  // never serves a scan of the other — the identity check below falls through (a
  // duckdb cache has empty resolved_file_paths; a parquet cache has an empty table_name).
  if (auto const* p = dynamic_cast<op::scan::parquet_ingestible_table_info const*>(&other)) {
    if (!matches_parquet_files(p->resolved_file_paths)) { return {}; }
    return column_projection_for(p->column_ids);
  }
  if (auto const* d = dynamic_cast<op::scan::duckdb_native_ingestible_table_info const*>(&other)) {
    if (!matches_duckdb_table(d->catalog_name, d->schema_name, d->table_name)) { return {}; }
    return column_projection_for(d->column_ids);
  }
  return {};
}

bool cache_entry_info::matches_duckdb_table(std::string_view catalog,
                                            std::string_view schema,
                                            std::string_view table) const
{
  // Same duckdb table by qualified name (catalog.schema.table), derived on both
  // pin and query sides from the resolved DuckTableEntry — so the stored casing is
  // the table's canonical (case-preserved) name on both sides and a byte-exact
  // compare is correct. (If a future site ever populates these from parsed input
  // rather than the resolved entry, switch to a case-insensitive compare.)
  // A parquet cache has an empty table_name, so it never matches a duckdb scan.
  if (table_name.empty()) { return false; }
  return catalog_name == catalog && schema_name == schema && table_name == table;
}

bool cache_entry_info::matches_parquet_files(std::span<std::string const> files) const
{
  if (files.empty() || resolved_file_paths.empty()) { return false; }
  if (files.size() != resolved_file_paths.size()) { return false; }
  // `this` is a stored entry (already canonical via cache_entry_info::from);
  // `files` arrives from a fresh bind whose paths are still as-bound, so
  // canonicalize the comparison copy to match. Every parquet identity check —
  // the serve-time cache-hit match and the plan-time residency-gate probe —
  // routes through here, so both see the same identity.
  auto these_files = resolved_file_paths;
  std::vector<std::string> those_files{files.begin(), files.end()};
  op::scan::canonicalize_scan_file_paths(those_files);
  std::ranges::sort(these_files);
  std::ranges::sort(those_files);
  return these_files == those_files;
}

std::vector<std::size_t> cache_entry_info::column_projection_for(
  duckdb::vector<duckdb::ColumnIndex> const& requested_ids) const
{
  for (auto const& c : requested_ids) {
    // rowid/virtual/empty/field-identifier columns are never cached; a
    // request for one is a miss. (GetPrimaryIndex on such an index is a
    // sentinel — or a throw for field identifiers — not a real column.)
    if (!c.HasPrimaryIndex() || c.IsRowIdColumn() || c.IsVirtualColumn() || c.IsEmptyColumn()) {
      return {};
    }
  }
  return column_superset_projection(column_ids, requested_ids);
}

void sirius_scan_manager::insert_pinned_entry(
  const std::string& name,
  cache_entry_info cache_info,
  std::vector<std::unique_ptr<cudf::table>> data_tables,
  std::vector<cucascade::memory::memory_space*> chunk_memory_spaces,
  duckdb::vector<duckdb::LogicalType> column_types,
  std::vector<std::vector<duckdb::unique_ptr<duckdb::BaseStatistics>>> chunk_stats,
  sirius::pinned_column_storage_matrix column_storage)
{
  // chunk_memory_spaces is parallel to data_tables — the caller
  // (PinTableFunction) emits one memory_space* per coalesced batch, and
  // there is exactly one
  // cudf::table per chunk in data_tables. Reject any misalignment loudly
  // rather than silently aliasing chunks to the wrong GPU.
  if (chunk_memory_spaces.size() != data_tables.size()) {
    throw std::invalid_argument(
      "[sirius_scan_manager::insert_pinned_entry] chunk_memory_spaces.size() (" +
      std::to_string(chunk_memory_spaces.size()) + ") must equal data_tables.size() (" +
      std::to_string(data_tables.size()) + ")");
  }

  // Compute the total row count of the incoming tables before releasing them
  // (release() empties the table; num_rows() would then return 0).
  std::size_t new_num_rows = 0;
  for (auto const& table : data_tables) {
    if (table) { new_num_rows += static_cast<std::size_t>(table->num_rows()); }
  }

  // Column names (aligned with the cached column_ids) key data_batches_by_column.
  // Copied out before cache_info is moved into the entry below.
  std::vector<std::string> column_names = cache_info.column_names();

  // column_ids and names within cache_info are built aligned 1:1 by
  // cache_entry_info::from; the merge path below indexes column_ids by the same
  // position as the column names, so reject any misalignment loudly rather than
  // risk an out-of-bounds access.
  if (cache_info.column_ids.size() != column_names.size()) {
    throw std::invalid_argument(
      "[sirius_scan_manager::insert_pinned_entry] cache_info.column_ids.size() (" +
      std::to_string(cache_info.column_ids.size()) + ") must equal column_names size (" +
      std::to_string(column_names.size()) + ")");
  }

  // The recorded storage metadata must cover the tables about to be stored, and a recorded
  // carrier that contradicts a stored type is a recorder bug that throws before anything is
  // cached.
  constexpr std::string_view kInsertContext = "[sirius_scan_manager::insert_pinned_entry]";
  validate_recorded_column_storage(
    column_storage,
    data_tables.size(),
    column_names.size(),
    kInsertContext,
    [&data_tables](std::size_t chunk, std::size_t column) -> std::optional<cudf::data_type> {
      if (!data_tables[chunk]) { return std::nullopt; }
      auto const& table = *data_tables[chunk];
      if (column >= static_cast<std::size_t>(table.num_columns())) { return std::nullopt; }
      return table.get_column(static_cast<cudf::size_type>(column)).type();
    });

  // Normalize the optional zone-map capture.
  bool const stats_supplied = !column_types.empty() && !chunk_stats.empty();
  auto pin_zone_maps        = pinned_zone_maps::from_capture(std::move(column_types),
                                                      std::move(chunk_stats),
                                                      cache_info.column_ids.size(),
                                                      data_tables.size());
  if (stats_supplied && !pin_zone_maps.has_stats()) {
    SIRIUS_LOG_WARN(
      "[sirius_scan_manager::insert_pinned_entry] zone-map capture failure; pinning '{}' without "
      "statistics",
      name);
  }

  auto existing_it = _pinned_entries.find(name);
  if (existing_it != _pinned_entries.end()) {
    // Same-row-count merge only applies when the completeness contracts match.
    // Mixing a full pin with a partial pin produces an entry whose columns came
    // from different row coverage — drop and rebuild instead.
    if (existing_it->second.num_rows == new_num_rows) {
      // A merge is only valid when the new pin reproduces the existing batch
      // boundaries and placement. The round-robin counter restarts at chunk 0
      // per pin_table call, and chunks at index i across all columns share a
      // memory_space because they came from the same coalesced batch. Batch
      // boundaries depend on the projected column set, so a re-pin can slice
      // the same files differently. Reject any mismatch loudly rather than
      // silently aliasing.
      auto& entry = existing_it->second;
      if (entry.chunk_memory_spaces.size() != chunk_memory_spaces.size()) {
        throw std::runtime_error(
          "[sirius_scan_manager::insert_pinned_entry] merge mismatch — "
          "existing.chunk_memory_spaces.size() (" +
          std::to_string(entry.chunk_memory_spaces.size()) +
          ") != new chunk_memory_spaces.size() (" + std::to_string(chunk_memory_spaces.size()) +
          ")");
      }
      for (std::size_t i = 0; i < chunk_memory_spaces.size(); ++i) {
        if (entry.chunk_memory_spaces[i] != chunk_memory_spaces[i]) {
          throw std::runtime_error(
            "[sirius_scan_manager::insert_pinned_entry] merge mismatch — "
            "chunk_memory_spaces[" +
            std::to_string(i) + "] differs between existing and new entry");
        }
      }
      // Per-chunk ROW counts must match too: the byte-budget coalescer can slice
      // the same table at different row-group boundaries for a different column
      // set (varchar byte budgets are data-dependent), which still yields equal
      // chunk counts and — because round-robin placement is a function of chunk
      // index alone — identical memory_spaces. Merging such chunks would corrupt
      // the entry positionally (columns disagreeing on chunk boundaries) and
      // silently invalidate the per-chunk MVCC row-count map a duckdb pin stamps
      // via attach_mvcc_metadata. Reject loudly instead.
      if (!entry.data_batches_by_column.empty()) {
        auto const& existing_chunks = entry.data_batches_by_column.begin()->second;
        if (existing_chunks.size() != data_tables.size()) {
          throw std::runtime_error(
            "[sirius_scan_manager::insert_pinned_entry] merge mismatch — existing entry has " +
            std::to_string(existing_chunks.size()) + " chunks but the new materialization has " +
            std::to_string(data_tables.size()));
        }
        for (std::size_t i = 0; i < data_tables.size(); ++i) {
          if (!data_tables[i]) { continue; }
          if (existing_chunks[i]->size() != data_tables[i]->num_rows()) {
            throw std::runtime_error(
              "[sirius_scan_manager::insert_pinned_entry] merge mismatch — chunk " +
              std::to_string(i) + " has " + std::to_string(existing_chunks[i]->size()) +
              " rows in the existing entry but " + std::to_string(data_tables[i]->num_rows()) +
              " in the new materialization (same total, different chunk boundaries)");
          }
        }
      }
      // Only now that the merge is known to be legal: the append below grows the existing
      // matrix row by row alongside cache_info, so an existing matrix that does not already
      // cover every chunk and column would desynchronize the two silently. Checked after the
      // guards above so a genuine boundary mismatch reports itself, not a shape error.
      validate_column_storage_shape(entry.column_storage,
                                    entry.chunk_memory_spaces.size(),
                                    entry.cache_info.column_ids.size(),
                                    "[sirius_scan_manager::insert_pinned_entry existing entry]",
                                    /*allow_empty=*/false);
      // Same row count → merge unique columns into the existing entry.
      // Decide which column INDICES are new BEFORE iterating chunks. Doing
      // the contains() check per-chunk would let chunk 0 install a new
      // column and then chunks 1..N-1 see contains()==true and skip — leaving
      // the new column with only chunk 0; the cached provider would then read
      // the short column vector as end-of-stream and silently truncate the scan.
      std::vector<bool> is_new_col(column_names.size(), false);
      for (std::size_t i = 0; i < column_names.size(); ++i) {
        is_new_col[i] = !entry.data_batches_by_column.contains(column_names[i]);
      }
      for (auto& table : data_tables) {
        if (!table) { continue; }
        auto cols = table->release();
        if (cols.size() != column_names.size()) {
          throw std::runtime_error(
            "[sirius_scan_manager::insert_pinned_entry] table column count " +
            std::to_string(cols.size()) + " does not match column_names size " +
            std::to_string(column_names.size()));
        }
        for (std::size_t i = 0; i < cols.size(); ++i) {
          if (!is_new_col[i]) {
            // Column was already cached before this merge call — drop the
            // duplicate chunk.
            continue;
          }
          entry.data_batches_by_column[std::string{column_names[i]}].emplace_back(
            std::move(cols[i]));
        }
      }
      // Reflect the merged columns in cache_info so can_serve_with_columns'
      // superset match — and the gather it drives — actually see them. Append
      // only columns that received data above (an empty data_tables call must
      // not list a column with no backing chunks in data_batches_by_column).
      // column_ids and names grow together and we only append, so the projection
      // positions already handed out for existing columns stay valid.
      bool const entry_had_stats = entry.zone_maps.has_stats();
      for (std::size_t i = 0; i < is_new_col.size(); ++i) {
        if (!is_new_col[i]) { continue; }
        if (!entry.data_batches_by_column.contains(column_names[i])) { continue; }
        for (std::size_t chunk_idx = 0; chunk_idx < entry.column_storage.size(); ++chunk_idx) {
          entry.column_storage[chunk_idx].push_back(column_storage[chunk_idx][i]);
        }
        entry.cache_info.column_ids.push_back(cache_info.column_ids[i]);
        entry.cache_info.names.push_back(column_names[i]);
        entry.zone_maps.append_column_from(pin_zone_maps, i);
      }
      if (entry_had_stats && !entry.zone_maps.has_stats()) {
        // append_from_column() cleared the sidecar stats because the new column's zone-map shape
        // disagreed with the existing entry's shape. Warn but still pin the entry.
        SIRIUS_LOG_WARN(
          "[sirius_scan_manager::insert_pinned_entry] merge appended columns without "
          "statistics; dropping zone-map statistics for entry '{}'",
          name);
      }
      // A statless entry adopts the incoming capture when it covers every entry column (by
      // primary index) — the re-pin recovery path for entries pinned while capture was off.
      // Safe because the merge guards above proved identical chunk boundaries.
      if (!entry.zone_maps.has_stats() && pin_zone_maps.has_stats()) {
        std::unordered_map<duckdb::idx_t, std::size_t> incoming_pos_by_primary;
        incoming_pos_by_primary.reserve(cache_info.column_ids.size());
        for (std::size_t i = 0; i < cache_info.column_ids.size(); ++i) {
          incoming_pos_by_primary.emplace(cache_info.column_ids[i].GetPrimaryIndex(), i);
        }
        std::vector<std::size_t> incoming_pos_by_entry_pos;
        incoming_pos_by_entry_pos.reserve(entry.cache_info.column_ids.size());
        for (auto const& col : entry.cache_info.column_ids) {
          auto it = incoming_pos_by_primary.find(col.GetPrimaryIndex());
          if (it == incoming_pos_by_primary.end()) { break; }
          incoming_pos_by_entry_pos.push_back(it->second);
        }
        if (incoming_pos_by_entry_pos.size() == entry.cache_info.column_ids.size()) {
          entry.zone_maps =
            pinned_zone_maps::remap(std::move(pin_zone_maps), incoming_pos_by_entry_pos);
        }
      }
      return;
    }
    // Row count or completeness contract differs → drop the stale entry and rebuild below.
    _pinned_entries.erase(existing_it);
  }

  pinned_entry entry;
  entry.cache_info          = std::move(cache_info);
  entry.chunk_memory_spaces = std::move(chunk_memory_spaces);
  entry.tier                = cucascade::memory::Tier::GPU;
  entry.column_storage      = std::move(column_storage);
  entry.num_rows            = new_num_rows;
  entry.zone_maps           = std::move(pin_zone_maps);

  for (auto& table : data_tables) {
    if (!table) { continue; }
    auto cols = table->release();
    if (cols.size() != column_names.size()) {
      throw std::runtime_error("[sirius_scan_manager::insert_pinned_entry] table column count " +
                               std::to_string(cols.size()) + " does not match column_names size " +
                               std::to_string(column_names.size()));
    }
    for (std::size_t i = 0; i < cols.size(); ++i) {
      entry.data_batches_by_column[std::string{column_names[i]}].emplace_back(std::move(cols[i]));
    }
  }

  _pinned_entries[name] = std::move(entry);
}

namespace {

/// Row count of one pinned host chunk, dispatching on its concrete
/// representation (a host pin may mix compressed and uncompressed chunks).
std::size_t pinned_host_chunk_rows(cucascade::idata_representation const& chunk)
{
  if (auto const* compressed =
        dynamic_cast<sirius::compressed_host_representation const*>(&chunk)) {
    return static_cast<std::size_t>(compressed->num_rows());
  }
  auto const& host_table = chunk.cast<cucascade::host_data_representation>().get_host_table();
  if (host_table && !host_table->columns.empty()) {
    return static_cast<std::size_t>(host_table->columns.front().num_rows);
  }
  return 0;
}

}  // namespace

void sirius_scan_manager::insert_pinned_entry_host(
  const std::string& name,
  cache_entry_info cache_info,
  std::vector<std::shared_ptr<cucascade::idata_representation>> host_chunks,
  cucascade::memory::memory_space& memory_space,
  duckdb::vector<duckdb::LogicalType> column_types,
  std::vector<std::vector<duckdb::unique_ptr<duckdb::BaseStatistics>>> chunk_stats,
  sirius::pinned_column_storage_matrix column_storage)
{
  // The host-tier path captures one chunk per emitted batch; each chunk holds every
  // pinned column (compressed or uncompressed). Re-insert always replaces — there is
  // no per-column merge analog to the GPU path because the chunk-vs-column dimensions
  // are flipped.
  std::size_t new_num_rows = 0;
  for (auto const& chunk : host_chunks) {
    if (!chunk) { continue; }
    new_num_rows += pinned_host_chunk_rows(*chunk);
  }

  auto const column_count = cache_info.column_names().size();
  if (cache_info.column_ids.size() != column_count) {
    throw std::invalid_argument(
      "[sirius_scan_manager::insert_pinned_entry_host] cache_info column shape mismatch");
  }
  constexpr std::string_view kInsertContext = "[sirius_scan_manager::insert_pinned_entry_host]";
  for (auto const& chunk : host_chunks) {
    // Column-shape cross-check on the uncompressed form only: a compressed chunk
    // holds every pinned column by construction (compress_with_plan consumed the
    // whole batch) and exposes no per-column host metadata to count.
    auto const* host =
      chunk ? dynamic_cast<cucascade::host_data_representation const*>(chunk.get()) : nullptr;
    if (host && host->get_host_table() && host->get_host_table()->columns.size() != column_count) {
      throw std::invalid_argument(
        "[sirius_scan_manager::insert_pinned_entry_host] host chunk column shape mismatch");
    }
  }
  validate_recorded_column_storage(
    column_storage,
    host_chunks.size(),
    column_count,
    kInsertContext,
    [&host_chunks](std::size_t chunk, std::size_t column) -> std::optional<cudf::data_type> {
      auto const* host =
        host_chunks[chunk]
          ? dynamic_cast<cucascade::host_data_representation const*>(host_chunks[chunk].get())
          : nullptr;
      if (host == nullptr || !host->get_host_table()) { return std::nullopt; }
      auto const& host_columns = host->get_host_table()->columns;
      if (column >= host_columns.size()) { return std::nullopt; }
      return host_column_carrier(host_columns[column]);
    });
  // Normalize the optional zone-map capture
  bool const stats_supplied = !column_types.empty() && !chunk_stats.empty();
  auto pin_zone_maps        = pinned_zone_maps::from_capture(std::move(column_types),
                                                      std::move(chunk_stats),
                                                      cache_info.column_ids.size(),
                                                      host_chunks.size());
  if (stats_supplied && !pin_zone_maps.has_stats()) {
    // Only reason stats failed to append is a shape mismatch between the captured stats and the
    // pinned entry; warn but still pin the entry.
    SIRIUS_LOG_WARN(
      "[sirius_scan_manager::insert_pinned_entry_host] zone-map capture shape mismatch; "
      "pinning '{}' without statistics",
      name);
  }

  pinned_entry entry;
  entry.cache_info     = std::move(cache_info);
  entry.tier           = cucascade::memory::Tier::HOST;
  entry.memory_space   = &memory_space;
  entry.num_rows       = new_num_rows;
  entry.host_chunks    = std::move(host_chunks);
  entry.column_storage = std::move(column_storage);
  entry.zone_maps      = std::move(pin_zone_maps);

  _pinned_entries[name] = std::move(entry);
}

void sirius_scan_manager::insert_pinned_entry_device(
  const std::string& name,
  cache_entry_info cache_info,
  std::vector<sirius::device_pin_chunk> chunks,
  cucascade::memory::memory_space& memory_space,
  sirius::pinned_column_storage_matrix column_storage)
{
  std::size_t new_num_rows = 0;
  for (auto const& chunk : chunks) {
    if (chunk.compressed) {
      new_num_rows += static_cast<std::size_t>(chunk.compressed->num_rows());
    } else if (!chunk.columns.empty() && chunk.columns.front()) {
      new_num_rows += static_cast<std::size_t>(chunk.columns.front()->size());
    }
  }

  auto const column_count = cache_info.column_names().size();
  if (cache_info.column_ids.size() != column_count) {
    throw std::invalid_argument(
      "[sirius_scan_manager::insert_pinned_entry_device] cache_info column shape mismatch");
  }
  constexpr std::string_view kInsertContext = "[sirius_scan_manager::insert_pinned_entry_device]";
  validate_recorded_column_storage(
    column_storage,
    chunks.size(),
    column_count,
    kInsertContext,
    [&chunks](std::size_t chunk, std::size_t column) -> std::optional<cudf::data_type> {
      if (chunks[chunk].compressed) { return std::nullopt; }
      if (column >= chunks[chunk].columns.size() || !chunks[chunk].columns[column]) {
        return std::nullopt;
      }
      return chunks[chunk].columns[column]->type();
    });

  pinned_entry entry;
  entry.cache_info     = std::move(cache_info);
  entry.tier           = cucascade::memory::Tier::GPU;
  entry.memory_space   = &memory_space;
  entry.num_rows       = new_num_rows;
  entry.device_chunks  = std::move(chunks);
  entry.column_storage = std::move(column_storage);

  SIRIUS_LOG_DEBUG("[sirius_scan_manager::insert_pinned_entry_device] '{}' chunks={} rows={}",
                   name,
                   entry.device_chunks.size(),
                   new_num_rows);

  _pinned_entries[name] = std::move(entry);
}

void sirius_scan_manager::attach_mvcc_metadata(const std::string& name,
                                               duckdb_mvcc_metadata metadata)
{
  auto it = _pinned_entries.find(name);
  if (it == _pinned_entries.end()) {
    throw std::invalid_argument("[attach_mvcc_metadata] no pinned entry named '" + name + "'");
  }
  it->second.mvcc = std::make_unique<duckdb_mvcc_metadata>(std::move(metadata));
}

void sirius_scan_manager::remove_pinned_entry(const std::string& name)
{
  _pinned_entries.erase(name);
}

void sirius_scan_manager::visit_pinned_entries(
  const std::function<bool(std::string_view, const pinned_entry&)>& visitor) const
{
  for (auto const& [name, entry] : _pinned_entries) {
    if (!visitor(name, entry)) { break; }
  }
}

pinned_entry const* sirius_scan_manager::find_pinned_entry_for_duckdb_table(
  std::string_view catalog_name, std::string_view schema_name, std::string_view table_name) const
{
  for (auto const& [name, entry] : _pinned_entries) {
    if (entry.cache_info.matches_duckdb_table(catalog_name, schema_name, table_name)) {
      return &entry;
    }
  }
  return nullptr;
}

pinned_entry const* sirius_scan_manager::find_pinned_entry_for_parquet_files(
  std::span<std::string const> resolved_file_paths) const
{
  for (auto const& [name, entry] : _pinned_entries) {
    if (entry.cache_info.matches_parquet_files(resolved_file_paths)) { return &entry; }
  }
  return nullptr;
}

void validate_column_storage_shape(pinned_column_storage_matrix const& matrix,
                                   std::size_t expected_chunks,
                                   std::size_t expected_columns,
                                   std::string_view context,
                                   bool allow_empty)
{
  if (matrix.empty() && allow_empty) { return; }
  if (matrix.size() != expected_chunks) {
    throw std::invalid_argument(std::string{context} + ": column_storage has " +
                                std::to_string(matrix.size()) + " chunks; expected " +
                                std::to_string(expected_chunks));
  }
  for (std::size_t chunk_idx = 0; chunk_idx < matrix.size(); ++chunk_idx) {
    if (matrix[chunk_idx].size() != expected_columns) {
      throw std::invalid_argument(std::string{context} + ": column_storage[" +
                                  std::to_string(chunk_idx) + "] has " +
                                  std::to_string(matrix[chunk_idx].size()) + " columns; expected " +
                                  std::to_string(expected_columns));
    }
  }
}

void throw_recorded_carrier_mismatch(std::string_view context,
                                     std::size_t chunk_idx,
                                     std::size_t column_idx,
                                     cudf::data_type recorded,
                                     cudf::data_type stored)
{
  throw std::invalid_argument(
    std::string{context} + ": recorded carrier " + cudf::type_to_name(recorded) + " for column " +
    std::to_string(column_idx) + " of chunk " + std::to_string(chunk_idx) +
    " contradicts the stored type " + cudf::type_to_name(stored));
}

void validate_pinned_entry_for_serving(pinned_entry const& entry,
                                       std::span<std::size_t const> selected_columns)
{
  auto const& entry_column_names = entry.cache_info.column_names();

  if (entry.tier == cucascade::memory::Tier::GPU) {
    // Compression-enabled pin: one device_pin_chunk per batch, each holding every
    // pinned column. A compressed chunk projects all columns internally; an
    // uncompressed chunk must carry every selected column as a non-null device
    // column (the serving loop reads a nullptr batch as end-of-stream).
    if (!entry.device_chunks.empty()) {
      validate_column_storage_shape(entry.column_storage,
                                    entry.device_chunks.size(),
                                    entry_column_names.size(),
                                    "pinned device entry",
                                    /*allow_empty=*/true);
      for (auto const& chunk : entry.device_chunks) {
        if (chunk.compressed) { continue; }
        for (auto idx : selected_columns) {
          if (idx >= chunk.columns.size() || !chunk.columns[idx]) {
            throw std::runtime_error("pinned device chunk is missing selected column " +
                                     std::to_string(idx));
          }
        }
      }
      return;
    }
    if (entry.data_batches_by_column.empty()) {
      validate_column_storage_shape(entry.column_storage,
                                    0,
                                    entry_column_names.size(),
                                    "pinned GPU entry",
                                    /*allow_empty=*/true);
      return;  // legitimate zero-chunk entry
    }
    auto const n_chunks = entry.data_batches_by_column.begin()->second.size();
    validate_column_storage_shape(entry.column_storage,
                                  n_chunks,
                                  entry_column_names.size(),
                                  "pinned GPU entry",
                                  /*allow_empty=*/true);
    if (entry.chunk_memory_spaces.size() != n_chunks) {
      throw std::runtime_error(
        "pinned entry's chunk_memory_spaces does not cover every chunk of the entry");
    }
    for (auto const* space : entry.chunk_memory_spaces) {
      if (!space) { throw std::runtime_error("pinned entry has a null chunk memory_space"); }
    }
    for (auto idx : selected_columns) {
      auto const& name = entry_column_names.at(idx);
      auto it          = entry.data_batches_by_column.find(name);
      if (it == entry.data_batches_by_column.end() || it->second.size() != n_chunks) {
        throw std::runtime_error("pinned column '" + name +
                                 "' does not cover every chunk of the entry");
      }
      for (auto const& chunk : it->second) {
        if (!chunk) { throw std::runtime_error("pinned column '" + name + "' has a null chunk"); }
      }
    }
    return;
  }

  if (entry.tier == cucascade::memory::Tier::HOST) {
    validate_column_storage_shape(entry.column_storage,
                                  entry.host_chunks.size(),
                                  entry_column_names.size(),
                                  "pinned HOST entry",
                                  /*allow_empty=*/true);
    for (auto const& chunk : entry.host_chunks) {
      if (!chunk) { throw std::runtime_error("pinned entry has a null host chunk"); }
    }
    return;
  }

  throw std::runtime_error("pinned entry has an unsupported tier");
}

bool pinned_column_narrowed_in_all_chunks(pinned_entry const& entry, std::size_t entry_position)
{
  if (entry.column_storage.empty()) { return false; }
  return std::ranges::all_of(entry.column_storage,
                             [&](std::vector<pinned_column_storage_meta> const& row) {
                               return entry_position < row.size() && row[entry_position].narrowed;
                             });
}

std::optional<cudf::data_type> pinned_column_narrow_carrier(pinned_entry const& entry,
                                                            std::size_t entry_position,
                                                            cudf::data_type native_type)
{
  if (!pinned_column_narrowed_in_all_chunks(entry, entry_position)) { return std::nullopt; }

  // The recorded matrix is pinned to the cached column count at insertion; refuse to answer from
  // a position the entry's own column list does not cover. This runs at plan time, before any
  // serving validator has inspected the entry this query.
  if (entry_position >= entry.cache_info.column_ids.size()) { return std::nullopt; }

  // Native equality preserves the logical domain: width and family alone cannot distinguish a
  // narrowed DATE from a narrowed INTEGER. Valid carriers then share a family and decimal scale,
  // making the widest carrier well-defined.
  std::optional<cudf::data_type> widest;
  for (auto const& row : entry.column_storage) {
    if (row[entry_position].native != native_type) { return std::nullopt; }
    auto const carrier = row[entry_position].carrier;
    if (!sirius::can_narrow_to(native_type, carrier)) { return std::nullopt; }
    if (!widest || cudf::size_of(carrier) > cudf::size_of(*widest)) { widest = carrier; }
  }
  return widest;
}

std::unique_ptr<databatch_provider> make_provider_for_pinned_entry(
  pinned_entry const& entry,
  std::span<std::size_t const> selected_columns,
  cached_scan_plan plan,
  const telemetry::batch_telemetry_info& telemetry_info,
  mvcc_chunk_mask_set mvcc_masks,
  std::vector<insert_delta_split> delta_splits,
  std::vector<cudf::data_type> normalization_targets,
  bool has_physical_overrides,
  sirius::pushdown_request pushdown_req,
  std::shared_ptr<sirius::op::sirius_dynamic_filter_set> dynamic_filters)
{
  return std::make_unique<cached_databatch_provider>(entry,
                                                     selected_columns,
                                                     std::move(plan),
                                                     telemetry_info,
                                                     std::move(mvcc_masks),
                                                     std::move(delta_splits),
                                                     std::move(normalization_targets),
                                                     has_physical_overrides,
                                                     std::move(pushdown_req),
                                                     std::move(dynamic_filters));
}

cached_scan_plan build_cached_scan_plan(pinned_entry const& entry,
                                        duckdb::TableFilterSet const* table_filters,
                                        duckdb::vector<duckdb::ColumnIndex> const* column_ids)
{
  std::size_t n_chunks = 0;
  if (entry.tier == cucascade::memory::Tier::GPU) {
    // A compression-enabled pin serves from device_chunks (one per batch); a plain
    // GPU pin serves the column-major data_batches_by_column. device_chunks takes
    // priority, matching the serving provider.
    if (!entry.device_chunks.empty()) {
      n_chunks = entry.device_chunks.size();
    } else {
      n_chunks = entry.data_batches_by_column.empty()
                   ? 0
                   : entry.data_batches_by_column.begin()->second.size();
    }
  } else if (entry.tier == cucascade::memory::Tier::HOST) {
    n_chunks = entry.host_chunks.size();
  }

  // Identity plan (serve everything) until proven otherwise.
  cached_scan_plan plan;
  plan.survivor_chunk_indices.reserve(n_chunks);
  for (std::size_t c = 0; c < n_chunks; ++c) {
    plan.survivor_chunk_indices.push_back(c);
  }
  if (n_chunks == 0 || !entry.zone_maps.has_stats() || table_filters == nullptr ||
      table_filters->filters.empty() || column_ids == nullptr) {
    return plan;
  }

  // Filter keys are positions into the QUERY's column_ids (remapped at plan time by
  // create_table_filter_set); map each to its primary/storage index, then to the entry's positional
  // column.
  std::unordered_map<duckdb::idx_t, std::size_t> entry_pos_by_primary;
  entry_pos_by_primary.reserve(entry.cache_info.column_ids.size());
  for (std::size_t i = 0; i < entry.cache_info.column_ids.size(); ++i) {
    entry_pos_by_primary.emplace(entry.cache_info.column_ids[i].GetPrimaryIndex(), i);
  }

  struct usable_filter {
    duckdb::TableFilter const* filter;
    std::size_t entry_pos;
  };
  std::vector<usable_filter> usable;
  for (auto const& [col_idx, filter] : table_filters->filters) {
    if (!filter) { continue; }
    if (col_idx >= column_ids->size()) { continue; }  // defensive
    auto const& column_id = (*column_ids)[col_idx];
    // rowid / empty / virtual sentinels have no storage stats.
    if (!column_id.HasPrimaryIndex() || column_id.IsRowIdColumn() || column_id.IsEmptyColumn() ||
        column_id.IsVirtualColumn()) {
      continue;
    }
    auto it = entry_pos_by_primary.find(column_id.GetPrimaryIndex());
    if (it == entry_pos_by_primary.end()) { continue; }
    auto const pos = it->second;
    if (pos >= entry.zone_maps.column_count()) { continue; }  // absent for this column
    if (!filter_safe_for_stats(*filter, entry.zone_maps.column_type(pos))) { continue; }
    usable.push_back({filter.get(), pos});
  }
  if (usable.empty()) { return plan; }

  plan.survivor_chunk_indices.clear();
  for (std::size_t c = 0; c < n_chunks; ++c) {
    bool pruned = false;
    for (auto const& uf : usable) {
      auto const* cell = entry.zone_maps.cell(uf.entry_pos, c);
      if (cell != nullptr && chunk_provably_empty(*uf.filter, *cell)) {
        pruned = true;
        break;
      }
    }
    if (!pruned) { plan.survivor_chunk_indices.push_back(c); }
  }
  plan.pruned = n_chunks - plan.survivor_chunk_indices.size();

  // Sentinel chunk: an all-pruned scan must not become a zero-batch scan (zero
  // splits => zero tasks => pipeline completion never fires — the hazard both
  // disk paths guard against). Keep chunk 0; the GPU filter empties it.
  if (plan.survivor_chunk_indices.empty()) {
    plan.survivor_chunk_indices.push_back(0);
    plan.pruned = n_chunks - 1;
  }
  return plan;
}

namespace {

// DuckDB cache-serve native-type gate. Every non-rowid projection must be present and retain its
// pin-time native mapping across all chunks. False is a clean cache miss; an empty matrix passes
// for zero-chunk and hand-built entry compatibility.
bool pinned_native_types_match_scan(pinned_entry const& entry,
                                    op::scan::duckdb_native_ingestible_table_info const& info)
{
  if (entry.column_storage.empty()) { return true; }

  std::unordered_map<duckdb::idx_t, std::size_t> entry_pos_by_primary;
  entry_pos_by_primary.reserve(entry.cache_info.column_ids.size());
  for (std::size_t i = 0; i < entry.cache_info.column_ids.size(); ++i) {
    entry_pos_by_primary.emplace(entry.cache_info.column_ids[i].GetPrimaryIndex(), i);
  }

  for (std::size_t ci = 0; ci < info.projected_cols.size(); ++ci) {
    auto const& pc = info.projected_cols[ci];
    if (pc.is_rowid) { continue; }
    auto const it = entry_pos_by_primary.find(pc.storage_idx.GetPrimaryIndex());
    if (it == entry_pos_by_primary.end()) { return false; }
    std::optional<cudf::data_type> native;
    if (ci < info.projected_types.size()) {
      native = sirius::try_get_cudf_type(info.projected_types[ci]);
    }
    if (!native) { return false; }
    for (auto const& row : entry.column_storage) {
      if (it->second >= row.size() || row[it->second].native != *native) { return false; }
    }
  }
  return true;
}

}  // namespace

std::optional<sirius_scan_manager::cached_assignment> sirius_scan_manager::try_match_cached_entry(
  op::scan::sirius_gpu_scan_operator* op)
{
  const auto& table_info = op->get_ingestible().table_info();

  for (auto const& [pinned_name, entry] : _pinned_entries) {
    // Identity + serviceability gate: empty when this cache cannot serve the scan
    // (wrong format / file-set / table, or missing a requested column).
    if (entry.cache_info.can_serve_with_columns(table_info).empty()) { continue; }
    // Check native types before strict MVCC handling so type drift is a clean cache miss rather
    // than an MVCC error or a cache hit under the wrong type.
    if (auto const* native_info =
          dynamic_cast<op::scan::duckdb_native_ingestible_table_info const*>(&table_info);
        native_info != nullptr && !pinned_native_types_match_scan(entry, *native_info)) {
      SIRIUS_LOG_INFO(
        "[sirius_scan_manager] pinned entry '{}' matches operator '{}' by table identity but its "
        "pin-time native column types differ from the scan's; treating as a cache miss",
        pinned_name,
        op->get_operator_id());
      continue;
    }
    // After identity, serviceability, and native-type checks, an MVCC pin must serve from cache.
    // The disk path is MVCC-blind under checkpoint suppression, so later failures are errors.
    bool const mvcc_strict = entry.mvcc != nullptr;
    try {
      // Serve cached columns in the ingestible's materialized (disk-decode) order rather
      // than raw column_ids order, so post_filter_and_project's index-based filter and
      // projection bind to the same columns they would on the disk read path.
      auto cols = gather_by_primary_index(entry.cache_info.column_ids,
                                          op->get_ingestible().materialized_column_order());
      if (cols.empty()) {
        throw std::runtime_error("materialized column set is not a subset of the cached columns");
      }
      // Serve-time defense: a malformed entry would end the cached batch stream
      // early (nullptr mid-stream reads as end-of-stream) and silently truncate
      // the scan; validate up front so it throws here and the catch below falls
      // back to the disk read instead.
      validate_pinned_entry_for_serving(entry, cols);
      // Zone-map survivor plan
      auto const filter_view =
        _pruning_enabled ? extract_scan_filters(table_info) : scan_filter_view{};
      auto plan = build_cached_scan_plan(entry, filter_view.table_filters, filter_view.column_ids);
      auto const total_chunks = plan.survivor_chunk_indices.size() + plan.pruned;
      if (plan.pruned > 0) {
        SIRIUS_LOG_INFO(
          "[sirius_scan_manager] zone-map pruning for pinned entry '{}' ({} tier): {}/{} chunks "
          "pruned, serving {}",
          pinned_name,
          entry.tier == cucascade::memory::Tier::GPU ? "GPU" : "HOST",
          plan.pruned,
          total_chunks,
          plan.survivor_chunk_indices.size());
      } else {
        SIRIUS_LOG_DEBUG(
          "[sirius_scan_manager] zone-map pruning for pinned entry '{}': nothing pruned ({} "
          "chunks served)",
          pinned_name,
          total_chunks);
      }

      if (entry.mvcc) {
        auto const* duckdb_info =
          dynamic_cast<op::scan::duckdb_native_ingestible_table_info const*>(&table_info);
        if (duckdb_info == nullptr || duckdb_info->storage == nullptr ||
            duckdb_info->context == nullptr) {
          throw std::runtime_error("mvcc-pinned entry matched a scan without duckdb table state");
        }
        auto const n_chunks = entry.mvcc->base_row_count_per_chunk.size();
        // A compression-enabled DuckDB GPU pin stores its chunks in
        // device_chunks (one entry per batch); the older column-major GPU path
        // uses data_batches_by_column. Derive the count from whichever backs this
        // entry, mirroring get_device_databatch's dispatch — otherwise a
        // device_chunks pin reads as 0 chunks and fails validation below.
        std::size_t stored_chunks = 0;
        if (entry.tier == cucascade::memory::Tier::GPU) {
          if (!entry.device_chunks.empty()) {
            stored_chunks = entry.device_chunks.size();
          } else if (!entry.data_batches_by_column.empty()) {
            stored_chunks = entry.data_batches_by_column.begin()->second.size();
          }
        } else {
          stored_chunks = entry.host_chunks.size();
        }
        if (stored_chunks != n_chunks) {
          // Pin-time metadata and entry chunks are kept in lock-step (the
          // insert merge-path guard); a mismatch means masks would bind to
          // the wrong rows.
          throw std::runtime_error("mvcc metadata covers " + std::to_string(n_chunks) +
                                   " chunk(s) but the entry stores " +
                                   std::to_string(stored_chunks));
        }
        // One mask job per entry per query: a second scan op over the same
        // entry (e.g. a self-join) shares the pending job's completed set at
        // handoff instead of re-running the capture+fill walk.
        auto const already_pending = std::ranges::any_of(
          _pending_mvcc_mask_jobs,
          [&](mvcc_mask_job_request const& r) { return r.entry_name == pinned_name; });
        if (already_pending) {
          SIRIUS_LOG_INFO(
            "[sirius_scan_manager] operator '{}' shares pinned entry '{}''s pending mvcc mask "
            "job",
            op->get_operator_id(),
            pinned_name);
        } else {
          mvcc_mask_job_request request;
          request.masks    = mvcc_chunk_mask_set(n_chunks);
          request.metadata = *entry.mvcc;
          request.storage  = duckdb_info->storage;
          request.context  = duckdb_info->context;
          // Per-chunk memory spaces, sourced from the same storage that backs
          // the chunk count above so device_chunks pins get real spaces instead
          // of an empty vector.
          std::vector<cucascade::memory::memory_space*> chunk_spaces;
          if (entry.tier == cucascade::memory::Tier::GPU) {
            if (!entry.device_chunks.empty()) {
              chunk_spaces.reserve(entry.device_chunks.size());
              for (auto const& chunk : entry.device_chunks) {
                chunk_spaces.push_back(chunk.memory_space ? chunk.memory_space
                                                          : entry.memory_space);
              }
            } else {
              chunk_spaces = entry.chunk_memory_spaces;
            }
          } else {
            chunk_spaces.assign(n_chunks, entry.memory_space);
          }
          request.chunk_spaces = std::move(chunk_spaces);
          request.entry_name   = pinned_name;
          _pending_mvcc_mask_jobs.push_back(std::move(request));
        }

        // One insert-delta job per entry per query, deduped like the mask
        // jobs. The job no-ops when no rows are beyond the pinned prefix, so
        // queuing unconditionally is safe. Later operators union their
        // columns in so the staging covers every requester; per-operator
        // splits are cut at handoff.
        auto delta_request = std::ranges::find_if(
          _pending_insert_delta_jobs,
          [&](insert_delta_job_request const& r) { return r.entry_name == pinned_name; });
        if (delta_request == _pending_insert_delta_jobs.end()) {
          insert_delta_job_request request;
          request.storage                = duckdb_info->storage;
          request.context                = duckdb_info->context;
          request.n_cache                = entry.mvcc->n_cache();
          request.approximate_batch_size = duckdb_info->approximate_batch_size;
          request.entry_name             = pinned_name;
          _pending_insert_delta_jobs.push_back(std::move(request));
          delta_request = std::prev(_pending_insert_delta_jobs.end());
        }
        for (std::size_t ci = 0; ci < duckdb_info->projected_cols.size(); ++ci) {
          auto const& pc = duckdb_info->projected_cols[ci];
          if (pc.is_rowid) { continue; }
          auto const sidx = pc.storage_idx.GetPrimaryIndex();
          if (std::find(delta_request->union_cols.begin(), delta_request->union_cols.end(), sidx) ==
              delta_request->union_cols.end()) {
            delta_request->union_cols.push_back(sidx);
            delta_request->union_types.push_back(duckdb_info->projected_types[ci]);
          }
        }
      }

      SIRIUS_LOG_INFO("[sirius_scan_manager] assigned pinned entry '{}' to operator '{}'",
                      pinned_name,
                      op->get_operator_id());
      return cached_assignment{op, &entry, std::move(cols), pinned_name, std::move(plan)};
    } catch (std::exception const& e) {
      if (mvcc_strict) {
        throw std::runtime_error("[sirius_scan_manager] mvcc-pinned entry '" + pinned_name +
                                 "' matched operator '" + std::to_string(op->get_operator_id()) +
                                 "' but cannot serve from the cache: " + e.what());
      }
      SIRIUS_LOG_ERROR(
        "[sirius_scan_manager] error while trying to assign cached entries to "
        "operator '{}': {}",
        op->get_operator_id(),
        e.what());
      return std::nullopt;
    } catch (...) {
      if (mvcc_strict) { throw; }
      SIRIUS_LOG_ERROR(
        "[sirius_scan_manager] error while trying to assign cached entries to "
        "operator '{}'",
        op->get_operator_id());
      return std::nullopt;
    }
  }
  return std::nullopt;
}

}  // namespace sirius::scan_manager
