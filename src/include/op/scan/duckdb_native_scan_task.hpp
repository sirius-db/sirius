/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include "op/scan/direct_block_scan.hpp"
#include "op/scan/duckdb_native_metadata.hpp"
#include "op/sirius_physical_duckdb_scan.hpp"
#include "pipeline/sirius_pipeline_task_states.hpp"
#include "sirius_context.hpp"

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/storage/data_table.hpp>
#include <duckdb/storage/storage_index.hpp>
#include <duckdb/storage/table/row_group.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace sirius::pipeline {
class task_scheduler;
}  // namespace sirius::pipeline

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// duckdb_native_scan_global_state
//
// Bypasses DuckDB's table function API entirely. Walks DuckDB storage
// directly via the segment-tree primitive in `direct_block_scan`, hands
// per-segment byte ranges to GPU decode kernels.
//
// Construction is a two-phase split (Level-1 metadata-operator decomposition,
// see design_metadata_operator_split.md):
//
//   1. Caller computes `partitioned_duckdb_native_metadata` via
//      `walk_duckdb_native_metadata(...)`. That function does the segment-tree
//      walk + per-segment compression viability check, and is callable
//      independently of any task context.
//
//   2. This ctor consumes the metadata. No walking happens here.
//
// PR H promotes the free walk function into a real metadata-scan operator
// without changing this ctor's signature.
//===----------------------------------------------------------------------===//

class duckdb_native_scan_global_state : public pipeline::sirius_pipeline_task_global_state {
 public:
  /// @brief Range of row groups assigned to one task.
  struct row_group_range {
    std::size_t start_idx;  ///< Index into surviving-row-groups list
    std::size_t count;      ///< Number of row groups in this batch
  };

  duckdb_native_scan_global_state(duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline,
                                  pipeline::task_scheduler& task_scheduler,
                                  duckdb::ClientContext& client_ctx,
                                  sirius_physical_duckdb_scan* scan_op,
                                  partitioned_duckdb_native_metadata metadata);

  /// @brief Whether the scan is GPU-decodable end-to-end. False = fall back.
  [[nodiscard]] bool viable() const noexcept { return viable_; }

  /// @brief Diagnostic string set when `viable()` is false. Empty when viable.
  [[nodiscard]] std::string const& viability_failure_reason() const noexcept
  {
    return viability_failure_reason_;
  }

  /// @brief Number of row groups that passed viability + zonemap pruning.
  [[nodiscard]] std::size_t total_row_groups() const noexcept { return row_groups_.size(); }

  /// @brief Atomically claim the next batch of row groups; nullopt when exhausted.
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

  // Accessors for tasks
  [[nodiscard]] duckdb::DataTable& storage() { return *storage_; }
  [[nodiscard]] duckdb::ClientContext& context() { return client_ctx_; }
  [[nodiscard]] std::vector<duckdb::StorageIndex> const& col_indices() const
  {
    return col_indices_;
  }
  [[nodiscard]] std::vector<sirius::logical_type> const& col_types() const { return col_types_; }
  [[nodiscard]] std::vector<duckdb::RowGroup*> const& row_groups() const { return row_groups_; }
  [[nodiscard]] cucascade::memory::memory_space* gpu_space() { return gpu_space_; }
  [[nodiscard]] pipeline::task_scheduler& scheduler() { return task_scheduler_; }
  [[nodiscard]] sirius_physical_duckdb_scan& scan_op() { return op_; }
  [[nodiscard]] duckdb::SiriusContext* sirius_ctx() { return sirius_ctx_; }

  void increment_tasks() { active_tasks_.fetch_add(1, std::memory_order_relaxed); }
  void decrement_tasks();

  [[nodiscard]] std::vector<sirius_physical_operator*> get_output_consumers() const noexcept;

 private:
  /// @brief Apply zonemap pruning using `op_.table_filters`; updates `row_groups_`.
  void prune_row_groups();

  /// @brief Compute `row_groups_per_batch_` from per-RG decode bytes vs config target.
  void compute_batch_size();

  // --- Fields ---
  duckdb::DataTable* storage_;
  duckdb::ClientContext& client_ctx_;
  duckdb::SiriusContext* sirius_ctx_;
  sirius_physical_duckdb_scan& op_;
  pipeline::task_scheduler& task_scheduler_;

  bool viable_ = false;
  std::string viability_failure_reason_;
  std::vector<duckdb::StorageIndex> col_indices_;
  std::vector<sirius::logical_type> col_types_;

  // Row group management — pointers + per-RG decode bytes from metadata
  std::vector<duckdb::RowGroup*> row_groups_;
  std::vector<std::size_t> rg_decoded_bytes_;
  std::atomic<std::size_t> next_claim_idx_{0};
  std::size_t row_groups_per_batch_ = 0;
  std::size_t decoded_bytes_per_rg_ = 0;

  cucascade::memory::memory_space* gpu_space_ = nullptr;

  std::atomic<std::int64_t> active_tasks_{0};
};

//===----------------------------------------------------------------------===//
// duckdb_native_scan_task_local_state
//===----------------------------------------------------------------------===//

class duckdb_native_scan_task_local_state : public pipeline::sirius_pipeline_task_local_state {
 public:
  duckdb_native_scan_task_local_state() = default;

  [[nodiscard]] std::size_t get_task_consumption_basis() const override { return 0; }
};

//===----------------------------------------------------------------------===//
// duckdb_native_scan_task
//
// Claims a batch of row groups, pins segments via direct_block_scan, decodes
// via GPU kernels, and returns a cudf::table wrapped in a data_batch.
//
// Self-continuation: when more batches remain after a task processes
// MAX_BATCHES_PER_TASK, it schedules a replacement before returning. Initial
// task fan-out is N (= scan executor thread count); replacements maintain
// N-way parallelism while bounding peak memory to ~N × MAX_BATCHES_PER_TASK.
//===----------------------------------------------------------------------===//

class duckdb_native_scan_task : public pipeline::sirius_pipeline_itask {
  using shared_data_repository = cucascade::shared_data_repository;

 public:
  duckdb_native_scan_task(std::uint64_t task_id,
                          shared_data_repository* data_repo,
                          std::shared_ptr<duckdb_native_scan_global_state> global_state);

  ~duckdb_native_scan_task() override;

  std::unique_ptr<op::operator_data> compute_task(rmm::cuda_stream_view stream) override;

  void publish_output(op::operator_data& output_data, rmm::cuda_stream_view stream) override;

  [[nodiscard]] std::size_t get_estimated_reservation_size() const override { return 0; }

  std::vector<op::sirius_physical_operator*> get_output_consumers() override
  {
    return _global_state->cast<duckdb_native_scan_global_state>().get_output_consumers();
  }

 private:
  shared_data_repository* data_repo_;
  std::uint64_t task_id_;
};

}  // namespace sirius::op::scan
