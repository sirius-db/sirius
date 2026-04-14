/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include <config.hpp>
#include <data/data_batch_utils.hpp>
#include <op/scan/direct_block_scan.hpp>
#include <op/sirius_physical_duckdb_scan.hpp>
#include <pipeline/pipeline_executor.hpp>
#include <pipeline/sirius_pipeline_itask.hpp>
#include <pipeline/sirius_pipeline_task_states.hpp>
#include <sirius_context.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>

#include <duckdb/common/types.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/storage/data_table.hpp>
#include <duckdb/storage/storage_index.hpp>
#include <duckdb/storage/table/row_group.hpp>

#include <atomic>
#include <cstddef>
#include <optional>
#include <vector>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// GPU Native Scan Global State
//===----------------------------------------------------------------------===//

/// @brief Global state for gpu_native_scan_task. Created once per DuckDB table scan operator.
///
/// Bypasses DuckDB's table function API entirely. Walks DuckDB storage directly,
/// checks GPU-decode viability, caches RowGroup pointers, and provides atomic
/// batch claiming for concurrent scan tasks.
class gpu_native_scan_global_state : public pipeline::sirius_pipeline_task_global_state {
 public:
  /// @brief Range of row groups assigned to one task.
  struct row_group_range {
    size_t start_idx;  ///< Index into active_row_groups_
    size_t count;      ///< Number of row groups in this batch
  };

  gpu_native_scan_global_state(duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline,
                               pipeline::pipeline_executor& pipeline_exec,
                               duckdb::ClientContext& client_ctx,
                               sirius_physical_duckdb_scan* scan_op);

  /// @brief Whether all projected columns are GPU-decodable.
  [[nodiscard]] bool viable() const noexcept { return viable_; }

  /// @brief Number of row groups that passed viability (all of them for Phase 1).
  [[nodiscard]] size_t total_row_groups() const noexcept { return row_groups_.size(); }

  /// @brief Atomically claim the next batch of row groups. Returns nullopt when exhausted.
  std::optional<row_group_range> claim_next_batch();

  /// @brief True when all row groups have been claimed (no more work).
  [[nodiscard]] bool all_claimed() const noexcept
  {
    return next_claim_idx_.load(std::memory_order_acquire) >= row_groups_.size();
  }

  /// @brief True once at least one task has been created (scan is self-managing).
  [[nodiscard]] bool has_started() const noexcept
  {
    return active_tasks_.load(std::memory_order_acquire) > 0 || all_claimed();
  }

  /// @brief Accessors for tasks
  [[nodiscard]] duckdb::DataTable& storage() { return *storage_; }
  [[nodiscard]] duckdb::ClientContext& context() { return client_ctx_; }
  [[nodiscard]] const std::vector<duckdb::StorageIndex>& col_indices() const { return col_indices_; }
  [[nodiscard]] const std::vector<duckdb::LogicalType>& col_types() const { return col_types_; }
  [[nodiscard]] const std::vector<duckdb::RowGroup*>& row_groups() const { return row_groups_; }
  [[nodiscard]] cucascade::memory::memory_space* gpu_space() { return gpu_space_; }
  [[nodiscard]] pipeline::pipeline_executor& executor() { return pipeline_executor_; }
  [[nodiscard]] sirius_physical_duckdb_scan& scan_op() { return op_; }
  [[nodiscard]] duckdb::SiriusContext* sirius_ctx() { return sirius_ctx_; }

  /// @brief Increment active task count. Call when scheduling a new task.
  void increment_tasks() { active_tasks_.fetch_add(1, std::memory_order_relaxed); }

  /// @brief Decrement active task count. When it hits 0 and all claimed, signals exhausted.
  void decrement_tasks();

  /// @brief Output consumers for downstream scheduling.
  [[nodiscard]] std::vector<sirius_physical_operator*> get_output_consumers() const noexcept;

 private:
  /// @brief Walk all segments: verify GPU-decodable + measure decoded size per row group.
  /// Sets viable_ and fills rg_decoded_bytes_.
  void check_viability();

  /// @brief Walk row group tree, cache RowGroup pointers.
  void cache_row_groups();

  /// @brief Prune row groups whose zonemaps prove no rows match filter predicates.
  /// Computes decoded_bytes_per_rg_ from surviving row groups.
  void prune_row_groups();

  /// @brief Compute row_groups_per_batch_ from decoded_bytes_per_rg_ vs config batch size.
  void compute_batch_size();

  // --- Fields ---
  duckdb::DataTable* storage_;
  duckdb::ClientContext& client_ctx_;
  duckdb::SiriusContext* sirius_ctx_;
  sirius_physical_duckdb_scan& op_;
  pipeline::pipeline_executor& pipeline_executor_;

  bool viable_ = false;
  std::vector<duckdb::StorageIndex> col_indices_;
  std::vector<duckdb::LogicalType> col_types_;

  // Row group management — pointers cached during construction
  std::vector<duckdb::RowGroup*> row_groups_;
  std::vector<size_t> rg_decoded_bytes_;  // per-RG decoded bytes (construction-time only)
  std::atomic<size_t> next_claim_idx_{0};
  size_t row_groups_per_batch_ = 0;     // 0 = all in one batch
  size_t decoded_bytes_per_rg_ = 0;     // average over surviving row groups

  // GPU memory
  cucascade::memory::memory_space* gpu_space_ = nullptr;

  // Lifecycle
  std::atomic<int64_t> active_tasks_{0};
};

//===----------------------------------------------------------------------===//
// GPU Native Scan Task Local State
//===----------------------------------------------------------------------===//

/// @brief Minimal local state for gpu_native_scan_task (no DuckDB TF state needed).
class gpu_native_scan_task_local_state : public pipeline::sirius_pipeline_task_local_state {
 public:
  gpu_native_scan_task_local_state() = default;

  [[nodiscard]] std::size_t get_task_consumption_basis() const override { return 0; }
};

//===----------------------------------------------------------------------===//
// GPU Native Scan Task
//===----------------------------------------------------------------------===//

/// @brief Scan task that bypasses DuckDB's table function API.
///
/// Claims a batch of row groups, pins segments directly from DuckDB storage,
/// decodes via GPU kernels, and returns a cudf::table wrapped in a data_batch.
class gpu_native_scan_task : public pipeline::sirius_pipeline_itask {
  using shared_data_repository = cucascade::shared_data_repository;

 public:
  gpu_native_scan_task(uint64_t task_id,
                       shared_data_repository* data_repo,
                       std::shared_ptr<gpu_native_scan_global_state> global_state);

  ~gpu_native_scan_task() override;

  std::unique_ptr<op::operator_data> compute_task(rmm::cuda_stream_view stream) override;

  void publish_output(op::operator_data& output_data, rmm::cuda_stream_view stream) override;

  [[nodiscard]] std::size_t get_estimated_reservation_size() const override { return 0; }

  std::vector<op::sirius_physical_operator*> get_output_consumers() override
  {
    return _global_state->cast<gpu_native_scan_global_state>().get_output_consumers();
  }

 private:
  shared_data_repository* data_repo_;
  uint64_t task_id_;
};

}  // namespace sirius::op::scan
