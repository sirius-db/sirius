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

#pragma once

#include "duckdb/execution/operator/join/join_filter_pushdown.hpp"
#include "op/dynamic_filter_publish_plan.hpp"
#include "op/sirius_physical_hash_join.hpp"  // sirius_physical_hash_join::key_cast_info

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace sirius::op {

class sirius_dynamic_filter;
class sirius_dynamic_bloom_filter;

//===----------------------------------------------------------------------===//
// dynamic_filter_publisher
//===----------------------------------------------------------------------===//
/// @brief Builds, replicates, and fans out one immutable filter snapshot from a complete join
/// build batch.
///
/// It borrows the join's plan and key metadata by reference; its caller (@ref
/// sirius_physical_hash_join::publish_dynamic_filters) owns source readiness and the exactly-once
/// arbitration. A publisher instance is single-use and does not outlive the referenced metadata.
class dynamic_filter_publisher final {
 public:
  dynamic_filter_publisher(duckdb::JoinFilterPushdownInfo const& filter_pushdown,
                           dynamic_filter_publish_plan const& plan,
                           std::vector<sirius_physical_hash_join::key_cast_info> const& key_casts,
                           std::vector<cudf::size_type> const& right_key_col_indices)
    : _filter_pushdown(filter_pushdown),
      _plan(plan),
      _key_casts(key_casts),
      _right_key_col_indices(right_key_col_indices)
  {
  }

  /// Apply publication gates, materialize device replicas, then publish to accepting targets.
  void publish(cudf::table_view const& build_view, rmm::cuda_stream_view stream) const;

 private:
  duckdb::JoinFilterPushdownInfo const& _filter_pushdown;
  dynamic_filter_publish_plan const& _plan;
  std::vector<sirius_physical_hash_join::key_cast_info> const& _key_casts;
  std::vector<cudf::size_type> const& _right_key_col_indices;
};

//===----------------------------------------------------------------------===//
// dynamic_filter_accumulator
//===----------------------------------------------------------------------===//
/// @brief Builds one filter snapshot *incrementally* from a build side that never arrives as a
/// single batch (a hash-partitioned, non-broadcast multi-partition build).
///
/// The one-shot @ref dynamic_filter_publisher needs the whole build in one folded batch; a
/// multi-partition build has no such batch. Instead, the build-side PARTITION arms an accumulator
/// at sizing time — when its FULL input barrier guarantees the complete build is resident, so the
/// expected batch count and a row estimate are exact/cheap — and each partition task calls
/// @ref add with its input batch on the task stream. The call that delivers the final expected
/// batch finalizes: it orders the stream after every contributing stream (per-batch CUDA events),
/// applies the exact-row domain-coverage gate, replicates, and fans out through the same channel
/// machinery as the one-shot path. Consumers therefore never observe a partially filled filter.
///
/// Membership filters are Bloom-only: Bloom capacity is an estimate-tolerant *sizing* input
/// (underestimate degrades FPR, never correctness), whereas the IN-list `static_set` is
/// capacity-rigid and the small IN-list needs the whole key set — neither fits incremental
/// construction, and multi-partition builds are far past their size regimes anyway. Zone maps are
/// not emitted (off by default; a per-batch min/max merge is a straightforward later extension).
///
/// Thread-safe: adds from concurrent partition tasks serialize on an internal mutex (enqueue-only,
/// the GPU work is async). Insertion is idempotent, so a retried batch cannot corrupt the filter.
/// Borrows the join's plan and key metadata by reference and must not outlive the join.
class dynamic_filter_accumulator final {
 public:
  dynamic_filter_accumulator(duckdb::JoinFilterPushdownInfo const& filter_pushdown,
                             dynamic_filter_publish_plan const& plan,
                             std::vector<sirius_physical_hash_join::key_cast_info> const& key_casts,
                             std::vector<cudf::size_type> const& right_key_col_indices,
                             std::size_t estimated_build_rows,
                             std::size_t expected_batches);
  ~dynamic_filter_accumulator();

  dynamic_filter_accumulator(dynamic_filter_accumulator const&)            = delete;
  dynamic_filter_accumulator& operator=(dynamic_filter_accumulator const&) = delete;

  /// Insert @p build_view's key columns on @p stream. The final expected call publishes before
  /// returning. @return true when this call performed the publication.
  bool add(cudf::table_view const& build_view, rmm::cuda_stream_view stream);

  [[nodiscard]] bool finished() const noexcept;

  /// The filters that survived the publication gates, as (key index into the pushdown's
  /// join_condition, filter) pairs. Valid only after the publishing add() returned true; empty when
  /// nothing was published.
  [[nodiscard]] std::vector<std::pair<std::size_t, std::shared_ptr<sirius_dynamic_filter>>> const&
  published_membership() const noexcept
  {
    return _published_membership;
  }

 private:
  struct key_slot {
    std::shared_ptr<sirius_dynamic_bloom_filter> bloom;  ///< null while ineligible / not begun
    cudf::data_type build_type{cudf::type_id::EMPTY};
  };

  /// First-batch initialization: per-key eligibility gates + empty Bloom creation on @p stream.
  void begin(cudf::table_view const& build_view, rmm::cuda_stream_view stream);
  /// Final-batch publication; @p stream is ordered after every recorded per-batch event.
  void finish(rmm::cuda_stream_view stream);
  void record_event(rmm::cuda_stream_view stream);

  duckdb::JoinFilterPushdownInfo const& _filter_pushdown;
  dynamic_filter_publish_plan const& _plan;
  std::vector<sirius_physical_hash_join::key_cast_info> const& _key_casts;
  std::vector<cudf::size_type> const& _right_key_col_indices;
  std::size_t const _estimated_build_rows;
  std::size_t const _expected_batches;
  int const _source_device;

  std::mutex _mutex;
  bool _begun    = false;
  bool _finished = false;
  std::vector<key_slot> _keys;           ///< aligned with _filter_pushdown.join_condition
  std::size_t _accumulated_rows    = 0;  ///< exact rows inserted so far
  std::size_t _accumulated_batches = 0;
  std::vector<cudaEvent_t> _events;  ///< one per contributing add(), for finish ordering
  /// Survivors of the publication gates, recorded by finish() for probe-side reuse.
  std::vector<std::pair<std::size_t, std::shared_ptr<sirius_dynamic_filter>>> _published_membership;
};

}  // namespace sirius::op
