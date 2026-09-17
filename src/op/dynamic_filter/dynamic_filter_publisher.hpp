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

#include "op/dynamic_filter/dynamic_filter_publish_plan.hpp"

#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>

namespace sirius::op {

struct dynamic_filter_publication_outcome {
  std::size_t keys_considered            = 0;
  std::size_t keys_with_known_domain     = 0;
  std::size_t keys_build_exceeded_domain = 0;
  std::size_t skipped_targets_drained    = 0;
  std::size_t keys_skipped_domain_gate   = 0;
  std::size_t keys_skipped_type_mismatch = 0;
  std::size_t membership_filters_built   = 0;
  std::size_t zone_map_filters_built     = 0;
  std::size_t active_targets             = 0;
  std::size_t filters_pushed             = 0;
};

/**
 * @brief Builds and publishes filters from a complete hash-join build table
 *
 * Replicas are ready before filters reach bound channels. The function retains no inputs; the
 * caller owns source readiness and one-shot arbitration. Type-mismatched keys are skipped.
 *
 * @pre @p plan is enabled
 * @throw std::runtime_error if the source GPU cannot be identified
 * @throw std::logic_error for inconsistent plan or filter metadata
 */
[[nodiscard]] dynamic_filter_publication_outcome publish_dynamic_filters(
  dynamic_filter_publish_plan const& plan,
  cudf::table_view const& build_view,
  rmm::cuda_stream_view stream);

}  // namespace sirius::op
