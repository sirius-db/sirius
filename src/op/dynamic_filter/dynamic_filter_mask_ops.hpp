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

#include <cudf/column/column_view.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

#include <optional>
#include <span>

namespace sirius::op {

class sirius_mask_applicable;

/**
 * @brief A single membership test input, connecting a filter to its probe column
 */
struct fused_membership_input {
  sirius_mask_applicable const* filter;
  cudf::column_view probe;
};

/**
 * @brief Prepares all membership steps and submits their ordered conjunction in rounds of four
 *
 * The first round starts with the residual's valid true rows, or all rows without a residual. Later
 * rounds use the running conjunction. `running` may alias the residual's values; its validity
 * bitmask remains untouched. The caller retains the filters and every device buffer until
 * completion on `stream`. This operation allocates only host descriptors and performs no gather or
 * readback.
 *
 * @pre `device_id` names the current device and every input buffer is readable on that device.
 * `running` is nonnullable BOOL8. Each probe and the optional BOOL8 residual have `running.size()`
 * rows.
 * @pre `counts` is empty, or holds one zero-initialized device counter per step plus a leading
 * post-residual counter when a residual exists. An empty span disables counting.
 * @throw std::invalid_argument if buffer types, row counts, counter extent or a filter pointer are
 * invalid
 * @param steps Membership inputs in execution order
 * @param device_id Resolved device holding the filter replicas
 * @param running Destination for one result byte per row
 * @param residual Optional initial keep mask, including null validity
 * @param counts Optional conditional survivor counts in execution order
 * @param[out] submitted_rounds Set to zero on entry and incremented as each round is submitted
 * @param stream Stream ordering every round
 */
void try_fused_membership_mask(std::span<fused_membership_input const> steps,
                               int device_id,
                               cudf::mutable_column_view running,
                               std::optional<cudf::column_view> residual,
                               cudf::device_span<cudf::size_type> counts,
                               std::size_t& submitted_rounds,
                               ::cuda::stream_ref stream);

/**
 * @brief Folds an optional BOOL8 mask into a nonnullable conjunction and counts survivors
 *
 * With `initialize`, the prior conjunction is all true. Otherwise `running` contains the preceding
 * conjunction. Null incoming values reject rows; offsets are honored. `running` may alias
 * `incoming`; the fold is elementwise. A null `count` skips counting; otherwise the caller zeros
 * `count` on `stream`. The caller retains every buffer until completion. No host readback is
 * performed.
 */
void fold_dynamic_filter_mask(cudf::mutable_column_view running,
                              std::optional<cudf::column_view> incoming,
                              cudf::size_type* count,
                              bool initialize,
                              ::cuda::stream_ref stream);

}  // namespace sirius::op
