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

#include <cucascade/cudf/gpu_data_representation.hpp>

#include <memory>
#include <utility>

namespace sirius {

/**
 * @brief GPU table representation whose rows were already filtered at decode.
 *
 * Constructed by the compression converters (decompress_host_to_gpu /
 * decompress_device_to_gpu) INSTEAD of the plain gpu_table_representation when
 * the fused scan-filter pipeline (`SIRIUS_EXP_FUSED_SCAN_FILTER`) applied the
 * scan's ENTIRE table-filter conjunction during decompression. The tag is a
 * hard promise:
 *
 *  - every column of the table is compacted to the same survivor row count;
 *  - every conjunct of the split's pushed-down table filter was applied
 *    (nothing is left for post_filter_and_project to evaluate);
 *  - no decode-time BOOL8 predicate-substitution columns are present — all
 *    columns carry real values in the batch's materialized column order.
 *
 * scan_operator_input::prepare_for_processing detects this type right after
 * convert_to and stamps the split `decode_row_filtered`, which
 * gpu_ingestible::materialize_table translates to filter_state::ROW_FILTERED —
 * post_filter_and_project then skips filter evaluation and only applies the
 * projection/layout assembly.
 *
 * When the feature gate is off the converters always construct the base class,
 * so this type is never observed and the scan path is byte-identical.
 *
 * NOTE: clone() intentionally degrades to the base representation. The tag is
 * only meaningful between the conversion and the scan's capture of the flag,
 * which happen back to back on the same thread in prepare_for_processing.
 */
class row_filtered_gpu_table_representation final : public ::cucascade::gpu_table_representation {
 public:
  row_filtered_gpu_table_representation(std::unique_ptr<cudf::table> table,
                                        ::cucascade::memory::memory_space& memory_space,
                                        rmm::cuda_stream_view writer_stream)
    : ::cucascade::gpu_table_representation(std::move(table), memory_space, writer_stream)
  {
  }
};

/**
 * @brief Classic full-width decode whose fused scan-filter attempt BAILED on
 *        RULE 2 (post-CNT survivors above SIRIUS_EXP_FUSED_SCAN_MAX_SEL).
 *
 * Constructed by the compression converters instead of the plain
 * gpu_table_representation when decompress_scan_filter reports
 * scan_filter_status::bailed_high_selectivity. The table content is exactly
 * the classic decode (NOT row-filtered — post_filter_and_project must still
 * run); the tag only carries the bail signal so the scan can memoize it:
 * per-batch selectivity is uniform across a scan's batches (unclustered
 * chunks), so one bail predicts all remaining batches. On seeing this type,
 * scan_operator_input::prepare_for_processing latches the operator-shared
 * bail flag; later splits of the same scan then strip the attached range
 * pushdown before conversion, skipping the wave-1 + CNT insurance cost. The
 * flag is per-operator: another query's scan of the same pinned entry
 * decides fresh.
 *
 * Same clone()/lifetime notes as the row-filtered tag above.
 */
class rule2_bailed_gpu_table_representation final : public ::cucascade::gpu_table_representation {
 public:
  rule2_bailed_gpu_table_representation(std::unique_ptr<cudf::table> table,
                                        ::cucascade::memory::memory_space& memory_space,
                                        rmm::cuda_stream_view writer_stream)
    : ::cucascade::gpu_table_representation(std::move(table), memory_space, writer_stream)
  {
  }
};

}  // namespace sirius
