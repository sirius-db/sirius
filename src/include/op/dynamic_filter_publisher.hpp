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

#include "op/dynamic_filter_publish_plan.hpp"

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <utility>

namespace sirius::op {

//===----------------------------------------------------------------------===//
// dynamic_filter_publisher
//===----------------------------------------------------------------------===//
/// @brief Builds, replicates, and fans out one immutable filter snapshot from a complete join
/// build batch.
///
/// This is step 4 of the story in dynamic_filter_publish_plan.hpp: by the time a publisher runs,
/// planning has already decided everything — which keys are admitted, which build column each one
/// reads, which scans receive the filters, and on which devices replicas live. The publisher's
/// only inputs are the frozen plan and the build table itself; it reads no planner state and no
/// DuckDB metadata.
///
/// Its caller (@ref sirius_physical_hash_join::publish_dynamic_filters) owns source readiness and
/// the exactly-once arbitration. A publisher instance is single-use; sharing the plan's ownership
/// keeps every referenced value alive for the duration of the publish call.
class dynamic_filter_publisher final {
 public:
  explicit dynamic_filter_publisher(std::shared_ptr<dynamic_filter_publish_plan const> plan)
    : _plan(std::move(plan))
  {
  }

  /// Apply publication gates, materialize device replicas, then publish to accepting targets.
  void publish(cudf::table_view const& build_view, rmm::cuda_stream_view stream) const;

 private:
  std::shared_ptr<dynamic_filter_publish_plan const> _plan;
};

}  // namespace sirius::op
