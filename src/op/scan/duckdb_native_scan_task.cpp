/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#include "op/scan/duckdb_native_scan_task.hpp"

#include "cuda/scan/gpu_native_decode.cuh"
#include "data/data_batch_utils.hpp"
#include "helper/type_conversions.hpp"
#include "log/logging.hpp"
#include "op/scan/duckdb_scan_executor.hpp"
#include "pipeline/task_scheduler.hpp"
#include "sirius_context.hpp"

#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb/catalog/catalog_entry/duck_table_entry.hpp>
#include <duckdb/execution/adaptive_filter.hpp>
#include <duckdb/function/table/table_scan.hpp>
#include <duckdb/storage/buffer_manager.hpp>
#include <duckdb/storage/statistics/string_stats.hpp>
#include <duckdb/storage/table/column_data.hpp>
#include <duckdb/storage/table/column_segment.hpp>
#include <duckdb/storage/table/row_group.hpp>
#include <duckdb/storage/table/row_group_collection.hpp>
#include <duckdb/storage/table/scan_state.hpp>
#include <duckdb/storage/table/segment_tree.hpp>
#include <duckdb/storage/table/standard_column_data.hpp>

#include <algorithm>
#include <chrono>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// walk_duckdb_native_metadata — Level-1 free function (consumed below)
//
// Pure metadata extraction: walks the segment trees of every projected
// column across all `row_groups` and decides per-segment compression
// viability (data + validity). Operator-level escape gates
// (`dynamic_filters`, `extra_info.sample_options`, virtual columns,
// `type_pushdown`) are checked by the caller BEFORE invoking this — the
// walk assumes every entry in `projected_cols` corresponds to a real
// `StandardColumnData`.
//
// PR H promotes this body into a real metadata-scan operator without
// changing callers.
//===----------------------------------------------------------------------===//

namespace {

bool segment_data_compression_viable(duckdb::ColumnSegment const& segment, bool is_varchar)
{
  auto compression = segment.GetCompressionType();
  switch (compression) {
    case duckdb::CompressionType::COMPRESSION_BITPACKING:
    case duckdb::CompressionType::COMPRESSION_CONSTANT:
    case duckdb::CompressionType::COMPRESSION_UNCOMPRESSED:
    case duckdb::CompressionType::COMPRESSION_RLE: return true;

    case duckdb::CompressionType::COMPRESSION_DICTIONARY:
    case duckdb::CompressionType::COMPRESSION_FSST:
      // String codecs require a max_string_length stat for buffer sizing.
      // Without it, we'd under-allocate in pass 2 and OOB.
      return !is_varchar || duckdb::StringStats::HasMaxStringLength(segment.stats.statistics);

    default: return false;
  }
}

bool segment_validity_compression_viable(duckdb::ColumnSegment const& segment)
{
  auto compression = segment.GetCompressionType();
  switch (compression) {
    // UNCOMPRESSED: existing direct-memcpy validity path.
    case duckdb::CompressionType::COMPRESSION_UNCOMPRESSED:
    // EMPTY: no nulls in this range; the decoder's all-valid pre-fill is
    // exactly correct, no overlay needed.
    case duckdb::CompressionType::COMPRESSION_EMPTY:
    // CONSTANT: in practice the all-valid case (genuinely all-null columns
    // get EMPTY with appropriate stats). The decoder's all-valid pre-fill
    // handles this, preserving pre-validity-walk behavior on TPC-H schemas
    // where nullable columns happen to be all-valid.
    case duckdb::CompressionType::COMPRESSION_CONSTANT:
    // ROARING: host-decoded into an owned bitmap inside direct_block_scan
    // and handed to the decoder as UNCOMPRESSED bytes.
    case duckdb::CompressionType::COMPRESSION_ROARING: return true;

    default: return false;
  }
}

}  // namespace

partitioned_duckdb_native_metadata walk_duckdb_native_metadata(
  duckdb::DataTable& storage,
  std::vector<duckdb::StorageIndex> const& projected_cols,
  std::vector<sirius::logical_type> const& projected_types,
  std::vector<duckdb::RowGroup*> const& row_groups,
  duckdb::ClientContext& context)
{
  partitioned_duckdb_native_metadata md;

  if (row_groups.empty() || projected_cols.empty()) {
    md.viable                   = false;
    md.viability_failure_reason = "no row groups or no projected columns";
    return md;
  }

  // Per-segment viability + per-RG decode-byte budget. Single pass over the
  // segment tree per (rg, col); the column_scans population happens after
  // viability succeeds via direct_block_scan_column_range.
  md.rg_decoded_bytes.assign(row_groups.size(), 0);

  for (std::size_t rg_idx = 0; rg_idx < row_groups.size(); ++rg_idx) {
    auto* rg = row_groups[rg_idx];
    for (std::size_t ci = 0; ci < projected_cols.size(); ++ci) {
      auto& col_data  = rg->GetColumnDirect(projected_cols[ci]);
      auto& seg_tree  = col_data.GetSegmentTree();
      auto seg_node   = seg_tree.GetRootSegment();
      bool is_varchar = projected_types[ci].is_varchar();

      while (seg_node) {
        auto& segment  = seg_node->GetNode();
        auto row_count = segment.count.load();

        if (!segment_data_compression_viable(segment, is_varchar)) {
          md.viable                   = false;
          md.viability_failure_reason = "unsupported data compression on column " +
                                        std::to_string(projected_cols[ci].GetPrimaryIndex());
          SIRIUS_LOG_INFO("[duckdb_native_scan] not viable: {}", md.viability_failure_reason);
          return md;
        }

        // Decoded size budget — per-RG, summed across columns.
        if (is_varchar) {
          std::uint32_t max_len = 0;
          if (duckdb::StringStats::HasMaxStringLength(segment.stats.statistics)) {
            max_len = duckdb::StringStats::MaxStringLength(segment.stats.statistics);
          }
          md.rg_decoded_bytes[rg_idx] += row_count * (4 + max_len);
        } else {
          md.rg_decoded_bytes[rg_idx] += row_count * projected_types[ci].fixed_width_byte_size();
        }

        seg_node = seg_tree.GetNextSegment(*seg_node);
      }

      // Validity-compression viability. Only StandardColumnData exposes a
      // validity tree; non-standard columns (struct/list/map) must be
      // filtered upstream — Cast<> throws if not standard.
      auto& std_col  = col_data.Cast<duckdb::StandardColumnData>();
      auto& val_data = std_col.GetValidityData();
      auto& vtree    = val_data.GetSegmentTree();
      auto vnode     = vtree.GetRootSegment();
      while (vnode) {
        if (!segment_validity_compression_viable(vnode->GetNode())) {
          md.viable                   = false;
          md.viability_failure_reason = "unsupported validity compression on column " +
                                        std::to_string(projected_cols[ci].GetPrimaryIndex());
          SIRIUS_LOG_INFO("[duckdb_native_scan] not viable: {}", md.viability_failure_reason);
          return md;
        }
        vnode = vtree.GetNextSegment(*vnode);
      }
    }
  }

  // Now pin the segments — column-major. direct_block_scan_column_range
  // walks data + validity for one column across the row-group range.
  md.column_scans.reserve(projected_cols.size());
  for (auto const& storage_idx : projected_cols) {
    md.column_scans.push_back(
      direct_block_scan_column_range(storage, storage_idx, context, row_groups));
  }
  md.row_groups = row_groups;
  md.viable     = true;
  return md;
}

//===----------------------------------------------------------------------===//
// duckdb_native_scan_global_state
//===----------------------------------------------------------------------===//

duckdb_native_scan_global_state::duckdb_native_scan_global_state(
  duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline,
  pipeline::task_scheduler& task_scheduler,
  duckdb::ClientContext& client_ctx,
  sirius_physical_duckdb_scan* scan_op,
  partitioned_duckdb_native_metadata metadata)
  : sirius_pipeline_task_global_state(std::move(pipeline)),
    storage_(nullptr),
    client_ctx_(client_ctx),
    sirius_ctx_(client_ctx.registered_state->Get<duckdb::SiriusContext>("sirius_state").get()),
    op_(*scan_op),
    task_scheduler_(task_scheduler)
{
  auto& bind_data = op_.bind_data->Cast<duckdb::TableScanBindData>();
  auto& table     = bind_data.table.Cast<duckdb::DuckTableEntry>();
  storage_        = &table.GetStorage();

  // Project column indices and types from the scan operator. Caller has
  // already filtered virtual columns (rowid + others) via the operator-level
  // escape gates, so every entry here is a real storage column.
  if (!op_.projection_ids.empty()) {
    for (std::size_t i = 0; i < op_.projection_ids.size(); ++i) {
      auto pid = op_.projection_ids[i];
      col_indices_.emplace_back(op_.column_ids[pid].GetPrimaryIndex());
      col_types_.push_back(op_.scanned_types[i]);
    }
  } else {
    for (std::size_t ci = 0; ci < op_.column_ids.size(); ++ci) {
      col_indices_.emplace_back(op_.column_ids[ci].GetPrimaryIndex());
      col_types_.push_back(op_.scanned_types[ci]);
    }
  }

  if (!metadata.viable) {
    viable_                   = false;
    viability_failure_reason_ = std::move(metadata.viability_failure_reason);
    return;
  }

  // Resolve GPU memory space
  auto& mem_mgr   = sirius_ctx_->get_memory_manager();
  auto gpu_spaces = mem_mgr.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  gpu_space_      = const_cast<cucascade::memory::memory_space*>(gpu_spaces[0]);

  row_groups_       = std::move(metadata.row_groups);
  rg_decoded_bytes_ = std::move(metadata.rg_decoded_bytes);
  viable_           = true;

  prune_row_groups();
  compute_batch_size();
  SIRIUS_LOG_INFO("[duckdb_native_scan] viable: {} row groups, {} cols, batch_size={}",
                  row_groups_.size(),
                  col_types_.size(),
                  row_groups_per_batch_);
}

void duckdb_native_scan_global_state::prune_row_groups()
{
  std::size_t original_count = row_groups_.size();

  if (op_.table_filters && !op_.table_filters->filters.empty()) {
    duckdb::vector<duckdb::StorageIndex> all_col_ids;
    all_col_ids.reserve(op_.column_ids.size());
    for (auto& col_id : op_.column_ids) {
      all_col_ids.emplace_back(col_id.GetPrimaryIndex());
    }

    duckdb::ScanFilterInfo filter_info;
    filter_info.Initialize(client_ctx_, *op_.table_filters, all_col_ids);

    std::vector<duckdb::RowGroup*> surviving_rgs;
    std::vector<std::size_t> surviving_bytes;
    surviving_rgs.reserve(row_groups_.size());
    surviving_bytes.reserve(row_groups_.size());

    for (std::size_t i = 0; i < row_groups_.size(); ++i) {
      if (!row_groups_[i]->CheckZonemap(filter_info)) { continue; }
      surviving_rgs.push_back(row_groups_[i]);
      surviving_bytes.push_back(rg_decoded_bytes_[i]);
    }

    row_groups_       = std::move(surviving_rgs);
    rg_decoded_bytes_ = std::move(surviving_bytes);
  }

  std::size_t total_bytes = 0;
  for (auto bytes : rg_decoded_bytes_) {
    total_bytes += bytes;
  }
  decoded_bytes_per_rg_ = row_groups_.empty() ? 0 : total_bytes / row_groups_.size();

  rg_decoded_bytes_.clear();
  rg_decoded_bytes_.shrink_to_fit();

  if (row_groups_.size() < original_count) {
    SIRIUS_LOG_INFO("[duckdb_native_scan] row group pruning: {}/{} pruned ({} remaining)",
                    original_count - row_groups_.size(),
                    original_count,
                    row_groups_.size());
  }
}

void duckdb_native_scan_global_state::compute_batch_size()
{
  auto target_bytes = sirius_ctx_->get_config().get_operator_params().scan_task_batch_size;

  if (decoded_bytes_per_rg_ == 0) {
    row_groups_per_batch_ = row_groups_.size();
    return;
  }

  std::size_t rgs_per_batch = target_bytes / decoded_bytes_per_rg_;
  if (rgs_per_batch == 0) { rgs_per_batch = 1; }

  row_groups_per_batch_ = std::min(rgs_per_batch, row_groups_.size());
}

std::optional<duckdb_native_scan_global_state::row_group_range>
duckdb_native_scan_global_state::claim_next_batch()
{
  std::size_t batch_size = row_groups_per_batch_ > 0 ? row_groups_per_batch_ : row_groups_.size();
  std::size_t start      = next_claim_idx_.fetch_add(batch_size, std::memory_order_acq_rel);

  if (start >= row_groups_.size()) { return std::nullopt; }

  std::size_t count = std::min(batch_size, row_groups_.size() - start);
  return row_group_range{start, count};
}

void duckdb_native_scan_global_state::decrement_tasks()
{
  auto remaining = active_tasks_.fetch_sub(1, std::memory_order_acq_rel) - 1;
  if (remaining == 0 && all_claimed()) { op_.exhausted.store(true, std::memory_order_release); }
}

std::vector<sirius_physical_operator*> duckdb_native_scan_global_state::get_output_consumers()
  const noexcept
{
  std::vector<sirius_physical_operator*> consumers;
  auto ports = op_.get_next_port_after_sink();
  for (auto& next_port : ports) {
    consumers.push_back(next_port.next_operator);
  }
  return consumers;
}

//===----------------------------------------------------------------------===//
// duckdb_native_scan_task
//===----------------------------------------------------------------------===//

duckdb_native_scan_task::duckdb_native_scan_task(
  std::uint64_t task_id,
  shared_data_repository* data_repo,
  std::shared_ptr<duckdb_native_scan_global_state> global_state)
  : sirius_pipeline_itask(std::make_unique<duckdb_native_scan_task_local_state>(),
                          std::move(global_state)),
    data_repo_(data_repo),
    task_id_(task_id)
{
  _global_state->cast<duckdb_native_scan_global_state>().increment_tasks();
}

duckdb_native_scan_task::~duckdb_native_scan_task()
{
  auto& g = _global_state->cast<duckdb_native_scan_global_state>();
  if (auto* pipeline = g.get_pipeline()) { pipeline->mark_task_completed(); }
}

std::unique_ptr<op::operator_data> duckdb_native_scan_task::compute_task(
  rmm::cuda_stream_view stream)
{
  auto& g = _global_state->cast<duckdb_native_scan_global_state>();

  auto& row_groups     = g.row_groups();
  auto& col_indices    = g.col_indices();
  auto& col_types      = g.col_types();
  std::size_t num_cols = col_indices.size();
  auto mr              = g.gpu_space()->get_default_allocator();

  using clock = std::chrono::steady_clock;
  auto us     = [](clock::time_point a, clock::time_point b) {
    return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
  };

  // Process up to MAX_BATCHES_PER_TASK before returning and self-scheduling
  // a continuation. Multiple initial tasks (N = scan threads) run concurrently,
  // each replacing itself when done. Bounds peak memory to ~N × MAX_BATCHES.
  //   1: lowest memory, highest scheduling overhead (bad for TPC-H joins)
  //   ALL: minimum overhead, highest memory (OOMs on ClickBench single-session)
  //   4: bounded memory, amortizes scheduling — works for both workloads
  constexpr std::size_t MAX_BATCHES_PER_TASK = 4;

  std::vector<std::shared_ptr<cucascade::data_batch>> batches;

  auto first_range = g.claim_next_batch();
  if (!first_range) {
    g.decrement_tasks();
    return std::make_unique<op::pipelineable_operator_data>(std::move(batches));
  }

  std::optional<decltype(first_range)::value_type> current_range = first_range;
  std::size_t batches_this_task                                  = 0;
  while (current_range && batches_this_task < MAX_BATCHES_PER_TASK) {
    auto& range = current_range;

    auto t0 = clock::now();

    std::vector<column_scan_result> col_scans(num_cols);
    for (std::size_t ci = 0; ci < num_cols; ++ci) {
      std::vector<duckdb::RowGroup*> batch_rgs(
        row_groups.begin() + range->start_idx,
        row_groups.begin() + range->start_idx + range->count);
      col_scans[ci] =
        direct_block_scan_column_range(g.storage(), col_indices[ci], g.context(), batch_rgs);
    }

    auto t1_pin = clock::now();

    // Decoder takes duckdb::LogicalType for cuDF type construction; convert
    // at the boundary. col_types_ is sirius::logical_type per project policy.
    auto duckdb_col_types_owned = sirius::to_duckdb_vec(
      duckdb::vector<sirius::logical_type>(col_types.begin(), col_types.end()));
    std::vector<duckdb::LogicalType> duckdb_col_types(duckdb_col_types_owned.begin(),
                                                      duckdb_col_types_owned.end());
    auto gpu_table = sirius::cuda::scan::gpu_decode_table(col_scans, duckdb_col_types, stream, mr);

    auto t2_decode = clock::now();

    std::size_t total_rows = col_scans.empty() ? 0 : col_scans[0].data.total_rows;
    batches.push_back(sirius::make_data_batch(std::move(gpu_table), *g.gpu_space()));

    SIRIUS_LOG_INFO(
      "[duckdb_native_scan] task {}: {} rows, {} cols | "
      "pin={:.1f}ms decode={:.1f}ms total={:.1f}ms (rg {}-{})",
      task_id_,
      total_rows,
      num_cols,
      us(t0, t1_pin) / 1000.0,
      us(t1_pin, t2_decode) / 1000.0,
      us(t0, t2_decode) / 1000.0,
      range->start_idx,
      range->start_idx + range->count - 1);

    ++batches_this_task;
    if (batches_this_task < MAX_BATCHES_PER_TASK) {
      current_range = g.claim_next_batch();
    } else {
      current_range.reset();
    }
  }

  // Self-continuation: schedule a replacement when more batches remain.
  // Atomic counter balance keeps active_tasks_ stable across replacement
  // (continuation's ctor +1, this task's decrement_tasks() -1). The last
  // task (no continuation) is the only one that drives `remaining == 0`,
  // which is correct shutdown detection.
  if (!g.all_claimed()) {
    auto global_state_shared = _global_state;
    try {
      auto continuation = std::make_unique<duckdb_native_scan_task>(
        task_id_ + 1,
        data_repo_,
        std::static_pointer_cast<duckdb_native_scan_global_state>(global_state_shared));
      if (auto* pipeline = g.get_pipeline()) { pipeline->mark_task_created(); }
      g.scheduler().schedule(std::move(continuation));
    } catch (std::exception const& e) {
      SIRIUS_LOG_ERROR(
        "[duckdb_native_scan] failed to schedule continuation task: {}. "
        "Remaining batches will be picked up by other in-flight tasks.",
        e.what());
    }
  }

  g.decrement_tasks();

  return std::make_unique<op::pipelineable_operator_data>(std::move(batches));
}

void duckdb_native_scan_task::publish_output(op::operator_data& output_data,
                                             rmm::cuda_stream_view /*stream*/)
{
  auto& pipelineable_output = dynamic_cast<op::pipelineable_operator_data&>(output_data);
  for (auto& batch : pipelineable_output.release_data_batches()) {
    data_repo_->add_data_batch(std::move(batch));
  }
}

}  // namespace sirius::op::scan
