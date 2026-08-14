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

// Late-mat port materialization (SIRIUS_EXP_LATE_MAT): the consuming-port
// prepare-time transform that turns a deferred batch (UINT64 pin-order rowid
// column + INT8 placeholders) back into its real columns via the late materializer's
// prepare_selection + materialize. Called from
// pipelineable_operator_data::prepare_for_processing for every batch of a
// task whose operator carries a port directive; batches that do not match
// the directive's placeholder signature (arity + exact placeholder types)
// pass through untouched, which is how mixed join task inputs (probe+build)
// self-select.

#include "late_mat/defer_directive.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>

namespace sirius::op {

/// Transform @p ro in place when its table matches @p directive's placeholder
/// signature; returns the (possibly re-acquired) read-only accessor either
/// way. Synchronizes @p stream once before releasing the placeholder columns
/// (their buffers are stream-ordered on the producer's stream). Throws on
/// stale origins (generation), null rowids, or materializer refusals — a
/// deferred batch that cannot be materialized must fail loudly, never flow on
/// with placeholder data.
[[nodiscard]] ::cucascade::read_only_data_batch late_mat_apply_port_directive(
  ::cucascade::read_only_data_batch ro,
  late_mat::port_materialize_directive const& directive,
  rmm::cuda_stream_view stream);

}  // namespace sirius::op
