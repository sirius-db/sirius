/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#include "op/scan/gpu_native_scan_task.hpp"

#include <cuda/scan/gpu_native_decode.cuh>
#include <log/logging.hpp>
#include <sirius_config.hpp>

#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <cstring>
#include <unordered_set>

#include <duckdb/storage/buffer_manager.hpp>
#include <duckdb/storage/buffer_manager.hpp>
#include <duckdb/storage/table/column_data.hpp>
#include <duckdb/storage/table/column_segment.hpp>
#include <duckdb/storage/table/row_group.hpp>
#include <duckdb/storage/table/row_group_collection.hpp>
#include <duckdb/storage/table/scan_state.hpp>
#include <duckdb/storage/table/segment_tree.hpp>
#include <duckdb/storage/table/standard_column_data.hpp>
#include <duckdb/storage/statistics/string_stats.hpp>
#include <duckdb/catalog/catalog_entry/duck_table_entry.hpp>
#include <duckdb/execution/adaptive_filter.hpp>
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

  // Build column indices and types from the scan operator's projected columns.
  // scanned_types is built using projection_ids (sorted), so col_indices_ must
  // use the same mapping: col_indices_[i] = column_ids[projection_ids[i]].
  // Without this, filter-only columns (in WHERE but not SELECT) cause a mismatch
  // where we scan the wrong storage column for the declared type.
  if (!op_.projection_ids.empty()) {
    for (size_t i = 0; i < op_.projection_ids.size(); ++i) {
      auto pid = op_.projection_ids[i];
      col_indices_.emplace_back(op_.column_ids[pid].GetPrimaryIndex());
      col_types_.push_back(op_.scanned_types[i]);
    }
  } else {
    for (size_t ci = 0; ci < op_.scanned_types.size(); ++ci) {
      col_indices_.emplace_back(op_.column_ids[ci].GetPrimaryIndex());
      col_types_.push_back(op_.scanned_types[ci]);
    }
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
    prune_row_groups();
    compute_batch_size();
    initialize_pipeline();

    SIRIUS_LOG_INFO(
      "[gpu_native_scan] viable: {} row groups, {} cols, batch_size={}, pipeline={}",
      row_groups_.size(), col_types_.size(), row_groups_per_batch_,
      pipeline_enabled() ? std::to_string(pipeline_chunk_rgs_) + "rg/chunk" : "off");
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

  // Walk all row groups → all segments for each projected column.
  // Two goals in one pass:
  //   1. Verify every segment has GPU-decodable compression
  //   2. Measure decoded GPU size per row group (for batch sizing after pruning)
  //
  // For fixed-width columns: decoded size = row_count × type_size
  // For VARCHAR columns: decoded size = row_count × (4 bytes offsets + max_string_length chars)
  //   max_string_length comes from segment stats — we already check it for DICTIONARY viability.

  rg_decoded_bytes_.resize(row_groups_.size(), 0);

  for (size_t rg_idx = 0; rg_idx < row_groups_.size(); ++rg_idx) {
    auto* rg = row_groups_[rg_idx];
    for (size_t ci = 0; ci < col_indices_.size(); ++ci) {
      auto& col_data = rg->GetColumnDirect(col_indices_[ci]);
      auto& seg_tree = col_data.GetSegmentTree();
      auto seg_node  = seg_tree.GetRootSegment();
      bool is_varchar = col_types_[ci].id() == duckdb::LogicalTypeId::VARCHAR;

      while (seg_node) {
        auto& segment    = seg_node->GetNode();
        auto compression = segment.GetCompressionType();
        auto row_count   = segment.count.load();

        // --- Viability check ---
        switch (compression) {
          case duckdb::CompressionType::COMPRESSION_BITPACKING:
          case duckdb::CompressionType::COMPRESSION_CONSTANT:
          case duckdb::CompressionType::COMPRESSION_UNCOMPRESSED:
          case duckdb::CompressionType::COMPRESSION_RLE:
            break;

          case duckdb::CompressionType::COMPRESSION_DICTIONARY:
          case duckdb::CompressionType::COMPRESSION_FSST:
            if (is_varchar &&
                !duckdb::StringStats::HasMaxStringLength(segment.stats.statistics)) {
              SIRIUS_LOG_INFO(
                "[gpu_native_scan] not viable: col {} segment missing max_string_length",
                col_indices_[ci].GetPrimaryIndex());
              viable_ = false;
              return;
            }
            break;

          default:
            SIRIUS_LOG_INFO(
              "[gpu_native_scan] not viable: col {} has unsupported compression {}",
              col_indices_[ci].GetPrimaryIndex(), static_cast<int>(compression));
            viable_ = false;
            return;
        }

        // --- Decoded size calculation (per row group) ---
        if (is_varchar) {
          uint32_t max_len = 0;
          if (duckdb::StringStats::HasMaxStringLength(segment.stats.statistics)) {
            max_len = duckdb::StringStats::MaxStringLength(segment.stats.statistics);
          }
          rg_decoded_bytes_[rg_idx] += row_count * (4 + max_len);
        } else {
          rg_decoded_bytes_[rg_idx] += row_count * duckdb::GetTypeIdSize(col_types_[ci].InternalType());
        }

        seg_node = seg_tree.GetNextSegment(*seg_node);
      }
    }
  }

  viable_ = true;
}

void gpu_native_scan_global_state::prune_row_groups()
{
  size_t original_count = row_groups_.size();

  // Prune row groups whose zonemaps prove no rows match the filter predicates
  if (op_.table_filters && !op_.table_filters->filters.empty()) {
    // Build StorageIndex vector for ALL column_ids (filter keys are positions in this vector)
    duckdb::vector<duckdb::StorageIndex> all_col_ids;
    all_col_ids.reserve(op_.column_ids.size());
    for (auto& col_id : op_.column_ids) {
      all_col_ids.emplace_back(col_id.GetPrimaryIndex());
    }

    duckdb::ScanFilterInfo filter_info;
    filter_info.Initialize(client_ctx_, *op_.table_filters, all_col_ids);

    std::vector<duckdb::RowGroup*> surviving_rgs;
    std::vector<size_t> surviving_bytes;
    surviving_rgs.reserve(row_groups_.size());
    surviving_bytes.reserve(row_groups_.size());

    for (size_t i = 0; i < row_groups_.size(); ++i) {
      if (!row_groups_[i]->CheckZonemap(filter_info)) {
        continue;
      }
      surviving_rgs.push_back(row_groups_[i]);
      surviving_bytes.push_back(rg_decoded_bytes_[i]);
    }

    row_groups_ = std::move(surviving_rgs);
    rg_decoded_bytes_ = std::move(surviving_bytes);
  }

  // Compute decoded_bytes_per_rg_ from surviving row groups
  size_t total_bytes = 0;
  for (auto bytes : rg_decoded_bytes_) {
    total_bytes += bytes;
  }
  decoded_bytes_per_rg_ = row_groups_.empty() ? 0 : total_bytes / row_groups_.size();

  // Free construction-time data
  rg_decoded_bytes_.clear();
  rg_decoded_bytes_.shrink_to_fit();

  if (row_groups_.size() < original_count) {
    SIRIUS_LOG_INFO(
      "[gpu_native_scan] row group pruning: {}/{} pruned ({} remaining)",
      original_count - row_groups_.size(), original_count, row_groups_.size());
  }
}

void gpu_native_scan_global_state::compute_batch_size()
{
  auto target_bytes = sirius_ctx_->get_config().get_operator_params().scan_task_batch_size;

  if (decoded_bytes_per_rg_ == 0) {
    row_groups_per_batch_ = row_groups_.size();
    return;
  }

  size_t rgs_per_batch  = target_bytes / decoded_bytes_per_rg_;
  if (rgs_per_batch == 0) { rgs_per_batch = 1; }

  row_groups_per_batch_ = std::min(rgs_per_batch, row_groups_.size());
}

void gpu_native_scan_global_state::initialize_pipeline()
{
  // Need at least 3 row groups per batch for pipelining to help:
  // 2 slots pre-filled + 1 decoding = minimum for overlap.
  if (row_groups_per_batch_ < 3) {
    pipeline_chunk_rgs_ = 0;
    return;
  }

  // Target 4-8 chunks per batch for good H2D/compute overlap.
  pipeline_chunk_rgs_ = std::max<size_t>(1, row_groups_per_batch_ / 4);

  // Slot size: enough for all unique 256KB blocks in one chunk.
  // Each column has data + validity segments, each potentially backed by
  // a separate block.  Use 2× col count as conservative per-RG estimate,
  // plus 20% headroom for boundary blocks shared across row groups.
  size_t blocks_per_chunk = pipeline_chunk_rgs_ * col_types_.size() * 2;
  size_t slot_bytes = static_cast<size_t>(blocks_per_chunk * 262144 * 1.2);
  slot_bytes = std::max<size_t>(slot_bytes, 16UL * 1024 * 1024);  // min 16MB
  slot_bytes = std::min<size_t>(slot_bytes, 128UL * 1024 * 1024); // max 128MB

  constexpr size_t NUM_SLOTS = 2;

  try {
    ring_buffer_ = std::make_unique<scan_ring_buffer>(NUM_SLOTS, slot_bytes);
    copy_stream_ = std::make_unique<rmm::cuda_stream>();
  } catch (const std::exception& e) {
    SIRIUS_LOG_WARN(
        "[gpu_native_scan] pipeline init failed: {} — falling back to serial",
        e.what());
    ring_buffer_.reset();
    copy_stream_.reset();
    pipeline_chunk_rgs_ = 0;
  }
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

//===----------------------------------------------------------------------===//
// Helper: gather unique blocks from col_scans into a pinned host buffer
// and build a device_block_map with offsets into that buffer.
//===----------------------------------------------------------------------===//

static void gather_blocks_to_pinned(
    const std::vector<column_scan_result>& col_scans,
    void* host_pinned, size_t slot_bytes,
    sirius::cuda::scan::device_block_map& block_map)
{
  block_map.offsets.clear();
  auto* host_buf = static_cast<uint8_t*>(host_pinned);
  size_t offset = 0;

  for (auto& cs : col_scans) {
    for (auto& seg : cs.data.segments) {
      if (!seg.persistent || !seg.data_ptr || seg.row_count == 0 || seg.block_id < 0)
        continue;
      if (block_map.offsets.count(seg.block_id))
        continue;
      if (offset + 262144 > slot_bytes) {
        SIRIUS_LOG_WARN("[pipeline] chunk exceeds slot size ({} + 256K > {})",
                        offset, slot_bytes);
        break;
      }
      const uint8_t* block_base = seg.data_ptr - seg.block_offset;
      std::memcpy(host_buf + offset, block_base, 262144);
      block_map.offsets[seg.block_id] = offset;
      offset += 262144;
    }
  }
  block_map.total_bytes = offset;
}

//===----------------------------------------------------------------------===//
// gpu_native_scan_task::compute_task
//===----------------------------------------------------------------------===//

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
  auto t0 = clock::now();

  auto& row_groups  = g.row_groups();
  auto& col_indices = g.col_indices();
  auto& col_types   = g.col_types();
  size_t num_cols   = col_indices.size();
  auto mr = g.gpu_space()->get_default_allocator();

  // Decide pipelined vs serial path
  bool use_pipeline = g.pipeline_enabled() && range->count >= 3;

  std::vector<std::shared_ptr<cucascade::data_batch>> output_batches;
  size_t total_rows = 0;
  size_t total_blocks = 0;

  if (use_pipeline) {
    //=== PIPELINED PATH: overlap H2D with decode via dual streams ===
    std::lock_guard<std::mutex> lk(g.pipeline_mutex());

    auto& ring = g.ring_buffer();
    auto copy_stream = g.copy_stream();
    auto compute_stream = stream;  // caller's stream = our compute stream

    size_t chunk_rgs = g.pipeline_chunk_rgs();
    size_t num_chunks = (range->count + chunk_rgs - 1) / chunk_rgs;

    // Per-chunk state: col_scans, ring slot, device block map
    struct chunk_state {
      std::vector<column_scan_result> col_scans;
      scan_ring_buffer::slot* slot = nullptr;
      sirius::cuda::scan::device_block_map block_map;
    };
    std::vector<chunk_state> chunks(num_chunks);

    auto pin_and_enqueue = [&](size_t ci) {
      auto& chunk = chunks[ci];
      size_t rg_start = range->start_idx + ci * chunk_rgs;
      size_t rg_count = std::min(chunk_rgs, range->count - ci * chunk_rgs);

      // Pin segments for this chunk's row groups
      chunk.col_scans.resize(num_cols);
      for (size_t col = 0; col < num_cols; ++col) {
        std::vector<duckdb::RowGroup*> rgs(
            row_groups.begin() + rg_start,
            row_groups.begin() + rg_start + rg_count);
        chunk.col_scans[col] = direct_block_scan_column_range(
            g.storage(), col_indices[col], g.context(), rgs);
      }

      // Gather blocks to pinned host buf
      chunk.slot = &ring.acquire();
      gather_blocks_to_pinned(
          chunk.col_scans, chunk.slot->host_pinned,
          ring.slot_bytes(), chunk.block_map);
      chunk.slot->used_bytes = chunk.block_map.total_bytes;

      // Async H2D on copy_stream (truly async — source is page-locked)
      if (chunk.slot->used_bytes > 0) {
        cudaMemcpyAsync(
            chunk.slot->device_staging, chunk.slot->host_pinned,
            chunk.slot->used_bytes, cudaMemcpyHostToDevice,
            copy_stream.value());
      }
      cudaEventRecord(chunk.slot->h2d_done, copy_stream.value());
      chunk.slot->state = scan_ring_buffer::slot_state::TRANSFERRING;
    };

    // Pre-fill: enqueue H2D for first num_slots chunks
    size_t enqueued = 0;
    for (size_t ci = 0; ci < std::min(num_chunks, ring.num_slots()); ++ci) {
      pin_and_enqueue(ci);
      enqueued++;
    }

    // Steady state: decode chunk, then enqueue next H2D
    for (size_t ci = 0; ci < num_chunks; ++ci) {
      auto& chunk = chunks[ci];

      // GPU-side wait: compute_stream waits for this chunk's H2D
      cudaStreamWaitEvent(compute_stream.value(), chunk.slot->h2d_done);

      // Decode from device staging
      auto gpu_table = sirius::cuda::scan::gpu_decode_table_pipelined(
          chunk.col_scans, col_types, chunk.block_map,
          chunk.slot->device_staging, compute_stream, mr);

      // Record compute completion for backpressure
      cudaEventRecord(chunk.slot->compute_done, compute_stream.value());
      chunk.slot->state = scan_ring_buffer::slot_state::COMPUTING;

      // Track stats
      size_t chunk_rows = chunk.col_scans.empty()
                              ? 0 : chunk.col_scans[0].data.total_rows;
      total_rows += chunk_rows;
      total_blocks += chunk.block_map.offsets.size();

      // Wrap in data_batch
      output_batches.push_back(
          sirius::make_data_batch(std::move(gpu_table), *g.gpu_space()));

      // Enqueue next chunk's H2D (overlaps with current decode)
      if (enqueued < num_chunks) {
        pin_and_enqueue(enqueued);
        enqueued++;
      }
    }

    // Drain: wait for last compute to finish
    cudaEventSynchronize(
        ring.get_slot(ring.last_acquired_idx()).compute_done);

  } else {
    //=== SERIAL PATH: pin all → bulk H2D → decode (existing behavior) ===
    std::vector<column_scan_result> col_scans(num_cols);
    for (size_t ci = 0; ci < num_cols; ++ci) {
      std::vector<duckdb::RowGroup*> batch_rgs(
          row_groups.begin() + range->start_idx,
          row_groups.begin() + range->start_idx + range->count);
      col_scans[ci] = direct_block_scan_column_range(
          g.storage(), col_indices[ci], g.context(), batch_rgs);
    }

    std::unordered_set<int64_t> unique_blocks;
    for (auto& cs : col_scans) {
      for (auto& seg : cs.data.segments) {
        if (seg.persistent && seg.data_ptr && seg.row_count > 0 && seg.block_id >= 0)
          unique_blocks.insert(seg.block_id);
      }
    }

    std::unique_ptr<cudf::table> gpu_table;

    if (unique_blocks.size() >= 4) {
      size_t buf_bytes = unique_blocks.size() * 262144;
      void* d_staging = nullptr;
      cudaMallocAsync(&d_staging, buf_bytes, stream.value());

      sirius::cuda::scan::device_block_map block_map;
      size_t offset = 0;
      for (auto& cs : col_scans) {
        for (auto& seg : cs.data.segments) {
          if (!seg.persistent || !seg.data_ptr || seg.row_count == 0 || seg.block_id < 0) continue;
          if (block_map.offsets.count(seg.block_id)) continue;
          cudaMemcpyAsync(static_cast<uint8_t*>(d_staging) + offset,
                          seg.data_ptr - seg.block_offset, 262144,
                          cudaMemcpyHostToDevice, stream.value());
          block_map.offsets[seg.block_id] = offset;
          offset += 262144;
        }
      }
      block_map.total_bytes = offset;
      total_blocks = block_map.offsets.size();

      gpu_table = sirius::cuda::scan::gpu_decode_table_pipelined(
          col_scans, col_types, block_map, d_staging, stream, mr);
      cudaFreeAsync(d_staging, stream.value());
    } else {
      gpu_table = sirius::cuda::scan::gpu_decode_table(
          col_scans, col_types, stream, mr);
    }

    total_rows = col_scans.empty() ? 0 : col_scans[0].data.total_rows;
    output_batches.push_back(
        sirius::make_data_batch(std::move(gpu_table), *g.gpu_space()));
  }

  auto t_end = clock::now();
  auto us = [](clock::time_point a, clock::time_point b) {
    return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
  };
  SIRIUS_LOG_INFO(
      "[gpu_native_scan] task {}: {} rows, {} cols, {} blocks, {} batches | "
      "total={:.1f}ms {} (rg {}-{})",
      task_id_, total_rows, num_cols, total_blocks, output_batches.size(),
      us(t0, t_end) / 1000.0,
      use_pipeline ? "PIPELINED" : "serial",
      range->start_idx, range->start_idx + range->count - 1);

  // Schedule continuation if more row groups remain
  if (!g.all_claimed()) {
    auto next_task = std::make_unique<gpu_native_scan_task>(
        g.sirius_ctx()->get_task_creator().get_next_task_id(),
        data_repo_,
        std::static_pointer_cast<gpu_native_scan_global_state>(_global_state));
    g.get_pipeline()->mark_task_created();
    g.executor().schedule(std::move(next_task));
  }

  g.decrement_tasks();

  return std::make_unique<op::pipelineable_operator_data>(std::move(output_batches));
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
