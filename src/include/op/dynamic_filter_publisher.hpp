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

#include <vector>

namespace sirius::op {

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

}  // namespace sirius::op
