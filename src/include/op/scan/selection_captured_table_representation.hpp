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

// Neutral wave-selection carrier for late materialization
// (SIRIUS_EXP_LATE_MAT) — the capture slot for fused batches that are
// compacted (status == applied) but UNTAGGED: membership-compacted and
// partial-coverage batches, whose row_filtered tag is deliberately withheld
// so the residual filter machinery still runs.
//
// Mirrors the row_filtered tag pattern (row_filtered_table_representation.hpp)
// but carries NO filter semantics whatsoever: the scan's
// row_filtered/rule2_bailed dynamic_casts both miss this type, so
// filter_state, post_filter behavior and RULE-2 latching are byte-identical
// to a plain gpu_table_representation — the capture is metadata-only.
// Downstream code that knows nothing about it sees the base type through
// every existing upcast.
//
// Lifetime is the same transient conversion -> same-thread harvest window as
// the tag types (scan_operator_input::prepare_for_processing); clone()
// intentionally degrades to the base representation, dropping the capture.
// The converter registry dispatches on exact typeid, so the delegating
// converter pairs for this type must be registered alongside the harvest
// cast (scan-manager side) — until then the type is only ever constructed
// when a capture was explicitly requested on an origin-stamped split.

#include "late_mat/column_origin.hpp"

#include <cucascade/cudf/gpu_data_representation.hpp>

#include <memory>
#include <utility>

namespace sirius {

class selection_captured_gpu_table_representation final
  : public ::cucascade::gpu_table_representation {
 public:
  selection_captured_gpu_table_representation(std::unique_ptr<cudf::table> table,
                                              ::cucascade::memory::memory_space& memory_space,
                                              rmm::cuda_stream_view writer_stream)
    : ::cucascade::gpu_table_representation(std::move(table), memory_space, writer_stream)
  {
  }

  /// The wave-1 selection moved out of the fused decode (kind=mask, geometry
  /// over the chunk's FULL row count, survivor_count == this batch's emitted
  /// row count; `range` zeroed — the scan side fills it from the split's
  /// origin at harvest). NEVER null on this type: the sole reason this
  /// subclass exists is to carry it. Set right after construction by the
  /// compression converters (capture requested AND status == applied AND the
  /// batch is untagged).
  std::shared_ptr<const late_mat::row_selection> captured_selection;
};

}  // namespace sirius
