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

#include "op/replicate/gpu_replicate_impl.hpp"
#include "op/sirius_physical_operator.hpp"

#include <cudf/types.hpp>

#include <memory>
#include <string>

namespace sirius {
namespace op {

/**
 * @brief A physical operator that replicates rows based on a count column, dropping that column in
 *        the process.
 *
 * The input batches are the output columns followed by the count column, which must be last. Row
 * `i` of an input batch appears `count[i]` times in the output, with the copies adjacent and in
 * input order. Each input batch becomes one or more output batches within @c
 * gpu_replicate_impl::limits. One @c execute holds the whole expansion of its input batches in
 * device memory, so the limits shape output batches but do not bound peak memory.
 */
class sirius_physical_replicate : public sirius_physical_operator {
 public:
  static constexpr SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::REPLICATE;

  /**
   * @brief Construct a new sirius_physical_replicate object.
   *
   * @param types The types of the output columns (the input schema without the count column).
   * @param output_limits The limits for the output batches.
   * @param estimated_cardinality The estimated cardinality of the output.
   * @throws sirius::internal_exception if @p types is empty or a limit in @p output_limits is not
   *         positive.
   */
  sirius_physical_replicate(duckdb::vector<sirius::logical_type> types,
                            gpu_replicate_impl::limits output_limits,
                            std::size_t estimated_cardinality);

  /**
   * @brief Get the output limits (maximum rows, maximum bytes) set for this operator.
   *
   * @return The output limits.
   */
  [[nodiscard]] gpu_replicate_impl::limits const& output_limits() const noexcept
  {
    return _output_limits;
  }

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         ::cuda::stream_ref stream) override;

 private:
  gpu_replicate_impl::limits _output_limits;  ///< Caps on each output batch.
};

}  // namespace op
}  // namespace sirius
