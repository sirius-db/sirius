/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#include "op/scan/gpu_native_scan_task.hpp"

#include <cuda/scan/gpu_native_decode.cuh>
#include <log/logging.hpp>

#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <duckdb/storage/buffer_manager.hpp>
#include <duckdb/storage/table/column_data.hpp>
#include <duckdb/storage/table/column_segment.hpp>
#include <duckdb/storage/table/row_group.hpp>
#include <duckdb/storage/table/row_group_collection.hpp>
#include <duckdb/storage/table/segment_tree.hpp>
#include <duckdb/storage/table/standard_column_data.hpp>
#include <duckdb/storage/statistics/string_stats.hpp>
#include <duckdb/catalog/catalog_entry/duck_table_entry.hpp>
#include <duckdb/function/table/table_scan.hpp>

#include <chrono>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// gpu_native_scan_global_state
//===----------------------------------------------------------------------===//

gpu_native_scan_global_state::gpu_native_scan_global_state(
  duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline,
  pipeline::pipeline_executor& pipeline_exec,
  duckdb::ClientContext& client_ctx,
  sirius_physical_duckdb_scan* scan_op)
  : sirius_pipeline_task_global_state(std::move(pipeline)),
    storage_(nullptr),
    client_ctx_(client_ctx),
    sirius_ctx_(client_ctx.registered_state->Get<duckdb::SiriusContext>("sirius_state").get()),
    op_(*scan_op),
    pipeline_executor_(pipeline_exec)
{
  // Get the DataTable from bind data
  auto& bind_data = op_.bind_data->Cast<duckdb::TableScanBindData>();
  auto& table     = bind_data.table.Cast<duckdb::DuckTableEntry>();
  storage_        = &table.GetStorage();

  // Build column indices and types from the scan operator's projected columns
  for (size_t ci = 0; ci < op_.scanned_types.size(); ++ci) {
    col_indices_.emplace_back(op_.column_ids[ci].GetPrimaryIndex());
    col_types_.push_back(op_.scanned_types[ci]);
  }

  // Resolve GPU memory space
  auto& mem_mgr  = sirius_ctx_->get_memory_manager();
  auto gpu_spaces = mem_mgr.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  gpu_space_      = const_cast<cucascade::memory::memory_space*>(gpu_spaces[0]);

  // Cache row group pointers first (viability check needs them)
  cache_row_groups();

  // Check if all segments are GPU-decodable
  check_viability();

  if (viable_) {
    // Phase 1: all row groups in one batch
    row_groups_per_batch_ = row_groups_.size();

    SIRIUS_LOG_INFO(
      "[gpu_native_scan] viable: {} row groups, {} cols, batch_size={}",
      row_groups_.size(), col_types_.size(), row_groups_per_batch_);
  } else {
    SIRIUS_LOG_INFO("[gpu_native_scan] not viable — will fall back to duckdb_scan_task");
  }
}

void gpu_native_scan_global_state::cache_row_groups()
{
  auto& rg_collection = *storage_->GetRowGroupCollectionRef();
  auto rg_tree        = rg_collection.GetRowGroupsDirect();
  auto rg_node        = rg_tree->GetRootSegment();

  while (rg_node) {
    row_groups_.push_back(&rg_node->GetNode());
    rg_node = rg_tree->GetNextSegment(*rg_node);
  }
}

void gpu_native_scan_global_state::check_viability()
{
  if (row_groups_.empty() || col_indices_.empty()) {
    viable_ = false;
    return;
  }

  auto& buffer_manager = duckdb::BufferManager::GetBufferManager(client_ctx_);

  for (auto* rg : row_groups_) {
    for (size_t ci = 0; ci < col_indices_.size(); ++ci) {
      auto& col_data = rg->GetColumnDirect(col_indices_[ci]);
      auto& seg_tree = col_data.GetSegmentTree();
      auto seg_node  = seg_tree.GetRootSegment();

      while (seg_node) {
        auto& segment     = seg_node->GetNode();
        auto compression  = segment.GetCompressionType();

        switch (compression) {
          case duckdb::CompressionType::COMPRESSION_BITPACKING:
          case duckdb::CompressionType::COMPRESSION_CONSTANT:
          case duckdb::CompressionType::COMPRESSION_UNCOMPRESSED:
            break;  // Always supported

          case duckdb::CompressionType::COMPRESSION_DICTIONARY:
            // Dictionary needs max_string_length for GPU concat path
            if (col_types_[ci].id() == duckdb::LogicalTypeId::VARCHAR) {
              if (!duckdb::StringStats::HasMaxStringLength(segment.stats.statistics)) {
                SIRIUS_LOG_INFO(
                  "[gpu_native_scan] not viable: col {} DICTIONARY segment missing max_string_length",
                  col_indices_[ci].GetPrimaryIndex());
                viable_ = false;
                return;
              }
            }
            break;

          default:
            SIRIUS_LOG_INFO(
              "[gpu_native_scan] not viable: col {} has unsupported compression {}",
              col_indices_[ci].GetPrimaryIndex(), static_cast<int>(compression));
            viable_ = false;
            return;
        }

        seg_node = seg_tree.GetNextSegment(*seg_node);
      }
    }
  }

  viable_ = true;
}

std::optional<gpu_native_scan_global_state::row_group_range>
gpu_native_scan_global_state::claim_next_batch()
{
  size_t batch_size = row_groups_per_batch_ > 0 ? row_groups_per_batch_ : row_groups_.size();
  size_t start      = next_claim_idx_.fetch_add(batch_size, std::memory_order_acq_rel);

  if (start >= row_groups_.size()) { return std::nullopt; }

  size_t count = std::min(batch_size, row_groups_.size() - start);
  return row_group_range{start, count};
}

void gpu_native_scan_global_state::decrement_tasks()
{
  auto remaining = active_tasks_.fetch_sub(1, std::memory_order_acq_rel) - 1;
  if (remaining == 0 && all_claimed()) {
    // All tasks done and all row groups consumed — set exhausted flag.
    // Pipeline completion is triggered later by mark_task_completed() in the destructor,
    // AFTER publish_output() has pushed data to the repo.
    op_.exhausted.store(true, std::memory_order_release);
  }
}

std::vector<sirius_physical_operator*>
gpu_native_scan_global_state::get_output_consumers() const noexcept
{
  std::vector<sirius_physical_operator*> consumers;
  auto ports = op_.get_next_port_after_sink();
  for (auto& next_port : ports) {
    consumers.push_back(next_port.next_operator);
  }
  return consumers;
}

//===----------------------------------------------------------------------===//
// gpu_native_scan_task
//===----------------------------------------------------------------------===//

gpu_native_scan_task::gpu_native_scan_task(
  uint64_t task_id,
  shared_data_repository* data_repo,
  std::shared_ptr<gpu_native_scan_global_state> global_state)
  : sirius_pipeline_itask(
      std::make_unique<gpu_native_scan_task_local_state>(),
      global_state),
    data_repo_(data_repo),
    task_id_(task_id)
{
  global_state->increment_tasks();
}

gpu_native_scan_task::~gpu_native_scan_task()
{
  auto& g = _global_state->cast<gpu_native_scan_global_state>();
  if (auto* pipeline = g.get_pipeline()) {
    pipeline->mark_task_completed();
  }
}

std::unique_ptr<op::operator_data> gpu_native_scan_task::compute_task(rmm::cuda_stream_view stream)
{
  auto& g = _global_state->cast<gpu_native_scan_global_state>();

  // 1. Claim next batch of row groups
  auto range = g.claim_next_batch();
  if (!range) {
    g.decrement_tasks();
    return std::make_unique<op::pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{});
  }

  using clock = std::chrono::steady_clock;
  auto t0     = clock::now();

  // 2. Pin segments for the claimed row groups
  auto& row_groups  = g.row_groups();
  auto& col_indices = g.col_indices();
  auto& col_types   = g.col_types();
  size_t num_cols   = col_indices.size();

  std::vector<column_scan_result> col_scans(num_cols);
  for (size_t ci = 0; ci < num_cols; ++ci) {
    std::vector<duckdb::RowGroup*> batch_rgs(
      row_groups.begin() + range->start_idx,
      row_groups.begin() + range->start_idx + range->count);
    col_scans[ci] = direct_block_scan_column_range(
      g.storage(), col_indices[ci], g.context(), batch_rgs);
  }

  auto t1_pin = clock::now();

  // 3. GPU decode -> cudf::table
  auto mr        = g.gpu_space()->get_default_allocator();
  auto gpu_table = sirius::cuda::scan::gpu_decode_table(col_scans, col_types, stream, mr);

  auto t2_decode = clock::now();

  // 4. Wrap in data_batch
  auto batch = sirius::make_data_batch(std::move(gpu_table), *g.gpu_space());

  auto us = [](clock::time_point a, clock::time_point b) {
    return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
  };

  size_t total_rows = col_scans.empty() ? 0 : col_scans[0].data.total_rows;
  size_t total_segs = 0, total_pinned_bytes = 0;
  for (auto& cs : col_scans) {
    total_segs += cs.data.segments.size();
    total_pinned_bytes += cs.data.total_pinned_bytes;
  }

  SIRIUS_LOG_INFO(
    "[gpu_native_scan] task {}: {} rows, {} cols, {} segs, {:.1f}MB pinned | "
    "pin={:.1f}ms decode={:.1f}ms total={:.1f}ms (rg {}-{})",
    task_id_, total_rows, num_cols, total_segs,
    total_pinned_bytes / (1024.0 * 1024.0),
    us(t0, t1_pin) / 1000.0,
    us(t1_pin, t2_decode) / 1000.0,
    us(t0, t2_decode) / 1000.0,
    range->start_idx, range->start_idx + range->count - 1);

  // 5. Schedule continuation if more row groups remain
  if (!g.all_claimed()) {
    auto next_task = std::make_unique<gpu_native_scan_task>(
      g.sirius_ctx()->get_task_creator().get_next_task_id(),
      data_repo_,
      std::static_pointer_cast<gpu_native_scan_global_state>(_global_state));
    g.get_pipeline()->mark_task_created();
    g.executor().schedule(std::move(next_task));
  }

  g.decrement_tasks();

  return std::make_unique<op::pipelineable_operator_data>(
    std::vector<std::shared_ptr<cucascade::data_batch>>{std::move(batch)});
}

void gpu_native_scan_task::publish_output(op::operator_data& output_data,
                                          rmm::cuda_stream_view /*stream*/)
{
  auto& pipelineable_output = dynamic_cast<op::pipelineable_operator_data&>(output_data);
  for (auto& batch : pipelineable_output.release_data_batches()) {
    data_repo_->add_data_batch(std::move(batch));
  }
}

}  // namespace sirius::op::scan
