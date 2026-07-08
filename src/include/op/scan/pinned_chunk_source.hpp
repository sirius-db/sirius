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

#pragma once

#include <op/scan/batch_coalescer.hpp>
#include <op/scan/gpu_ingestible_types.hpp>
#include <telemetry/data_batch_probe.hpp>

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace cucascade {
class data_batch;
class host_data_representation;
}  // namespace cucascade

namespace cucascade::memory {
class memory_space;
}  // namespace cucascade::memory

namespace cudf {
class column;
}  // namespace cudf

namespace sirius::op::scan {

/**
 * @brief Per-chunk split descriptor for a pinned (cached) table.
 *
 * Carries the already-resident data_batch a pinned-mode work item assembled.
 * The sequencer's emit path unwraps it (via @ref take_batch) into a resident
 * scan_operator_input, skipping balancer stamping and fadvise/prefetch —
 * task_creator routes the task off the batch's memory_space instead.
 * @c estimated_bytes / @c fadvise_entries keep the base defaults: the split is
 * resident (no IO to advise), and the resident scan_operator_input computes
 * its size from the batch itself.
 *
 * Format-neutral by design: parquet and duckdb pins serve identical resident
 * chunks. MVCC serving (issue #819, duckdb-only) subclasses this in
 * duckdb-native land to add visibility state; the emit path dynamic_casts to
 * THIS base, so derived splits flow through it unchanged.
 */
class cached_scan_info : public scan_info {
 public:
  cached_scan_info(std::shared_ptr<cucascade::data_batch> batch, std::size_t chunk_index);
  ~cached_scan_info() override;

  /// Surrender the resident batch to the emit path. Call once.
  [[nodiscard]] std::shared_ptr<cucascade::data_batch> take_batch() noexcept;

  [[nodiscard]] std::size_t chunk_index() const noexcept { return _chunk_index; }

 private:
  std::shared_ptr<cucascade::data_batch> _batch;
  std::size_t _chunk_index{0};
};

/**
 * @brief Pass-through coalescer for pinned (cached) pipeline slots.
 *
 * Pinned chunks were already coalesced to batch size at pin time, so there is
 * nothing to bundle: @c push emits each cached split as-is (foreign split
 * types are dropped, mirroring the disk-format coalescers) and @c flush never
 * emits — unlike the disk coalescers there is no template-split fallback, so
 * a zero-chunk pin closes its connector with zero splits, exactly as the
 * direct cached serving path did. MVCC serving (duckdb-only) replaces this
 * with its own join-coalescer variant; parquet keeps this one forever.
 */
class cached_batch_coalescer final : public batch_coalescer {
 public:
  std::vector<std::unique_ptr<scan_info>> push(std::unique_ptr<scan_info> split) override;

  std::vector<std::unique_ptr<scan_info>> flush() override;
};

/**
 * @brief The pinned chunks of one cache-hit scan, served as claimable work
 *        items.
 *
 * Built by the scan manager from a matched pinned entry (gathered down to the
 * scan's requested columns, in materialized order) and handed to the
 * operator's format ingestible via gpu_ingestible::serve_from_pinned_chunks.
 * The ingestible's pinned-mode serving methods delegate here: work items
 * claim chunk indices lock-free and assemble each chunk's resident
 * data_batch on a scan-manager dispatcher thread.
 *
 * Self-contained: holds shared_ptr copies of the pinned chunk data (columns /
 * host representations), not a reference into the scan manager's
 * pinned-entries map — the data stays alive for the query even if the entry
 * is unpinned concurrently, and nothing scan_manager-shaped leaks into the
 * op layer.
 */
class pinned_chunk_source {
 public:
  /// One GPU-tier chunk: the selected columns (serve order) + the chunk's
  /// pinned memory space (its home GPU — the placement signal task_creator
  /// routes by).
  struct gpu_chunk {
    std::vector<std::shared_ptr<cudf::column>> columns;
    cucascade::memory::memory_space* memory_space{nullptr};
  };

  /// One HOST-tier chunk: the pinned host representation holding every pinned
  /// column; the serve-time column subset is applied by slice().
  struct host_chunk {
    std::shared_ptr<cucascade::host_data_representation> data;
  };

  /// GPU-tier source. Every chunk must carry a non-null memory_space.
  /// @p telemetry_info attributes the assembled batches to the producing
  /// pipeline (a null context yields no-op probes, e.g. in unit tests).
  explicit pinned_chunk_source(std::vector<gpu_chunk> chunks,
                               telemetry::batch_telemetry_info telemetry_info = {});

  /// HOST-tier source. @p column_indices selects the served columns (in serve
  /// order) out of each chunk's representation. Every chunk must be non-null.
  pinned_chunk_source(std::vector<host_chunk> chunks,
                      std::vector<std::size_t> column_indices,
                      telemetry::batch_telemetry_info telemetry_info = {});

  [[nodiscard]] bool has_more() const noexcept
  {
    return _next_chunk.load(std::memory_order_relaxed) < _n_chunks;
  }

  [[nodiscard]] std::size_t num_chunks() const noexcept { return _n_chunks; }

  /// Claim the next chunk and return a callable that assembles its resident
  /// batch wrapped in a cached_scan_info. Null when all chunks are claimed
  /// (lost the race). Thread-safe; the callable runs on a dispatcher thread.
  [[nodiscard]] std::function<std::unique_ptr<scan_info>()> next_work_item();

 private:
  [[nodiscard]] std::shared_ptr<cucascade::data_batch> make_batch(std::size_t index) const;

  std::vector<gpu_chunk> _gpu_chunks;
  std::vector<host_chunk> _host_chunks;
  std::vector<std::size_t> _host_column_indices;
  telemetry::batch_telemetry_info _telemetry_info{};
  std::size_t _n_chunks{0};
  std::atomic<std::size_t> _next_chunk{0};
};

}  // namespace sirius::op::scan
