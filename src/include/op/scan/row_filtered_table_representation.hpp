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
 * @brief What the decode did to a batch, beyond producing its columns.
 *
 * Facts the converter knows exactly and the scan would otherwise have to infer.
 * Carried as a value on @ref decoded_batch_representation rather than encoded in
 * the representation's dynamic type, so it survives copying and can grow a field
 * without growing a class.
 */
struct decode_outcome {
  /// The fused scan-filter decode applied the split's ENTIRE table-filter
  /// conjunction and every column is compacted to the survivor rows.
  /// materialize_table maps this to filter_state::ROW_FILTERED, so
  /// post_filter_and_project skips filter evaluation and only projects.
  ///
  /// A partial mask must leave this false: the residual conjuncts still have to
  /// run, and re-checking already-applied ones on the compacted rows is
  /// idempotent.
  bool row_filtered = false;

  /// The fused attempt hit the RULE-2 selectivity bail: the columns are the
  /// ordinary full-width decode (NOT row-filtered), and the only thing worth
  /// reporting is that the attempt did not pay off.
  ///
  /// Per-batch selectivity is uniform across a scan's batches (unclustered
  /// chunks), so one bail predicts the rest: the scan latches an operator-shared
  /// flag and later splits strip the attached range pushdown before conversion,
  /// dropping the wave-1 + CNT insurance cost from every-batch to once-per-scan.
  /// Per-operator by construction — another query's scan decides fresh.
  bool rule2_bailed = false;

  [[nodiscard]] bool any() const noexcept { return row_filtered || rule2_bailed; }
};

/**
 * @brief A GPU table representation that reports what its decode did.
 *
 * Constructed by the compression converters (decompress_host_to_gpu /
 * decompress_device_to_gpu) in place of the plain gpu_table_representation
 * whenever the decode has something to report; the plain type is still used
 * when it does not, so nothing changes on the ordinary path and the feature
 * gate being off is byte-identical to before.
 *
 * scan_operator_input::prepare_for_processing reads @ref outcome right after
 * convert_to and stamps the split from it.
 *
 * The outcome is a property of THIS decode, so a copy that shares the decoded
 * columns legitimately shares it too — unlike the dynamic-type encoding this
 * replaced, where clone() had to decide what the copy "is" and settled for
 * dropping the information.
 */
class decoded_batch_representation final : public ::cucascade::gpu_table_representation {
 public:
  decoded_batch_representation(std::unique_ptr<cudf::table> table,
                               ::cucascade::memory::memory_space& memory_space,
                               rmm::cuda_stream_view writer_stream,
                               decode_outcome outcome)
    : ::cucascade::gpu_table_representation(std::move(table), memory_space, writer_stream),
      _outcome(outcome)
  {
  }

  [[nodiscard]] const decode_outcome& outcome() const noexcept { return _outcome; }

 private:
  decode_outcome _outcome;
};

}  // namespace sirius
