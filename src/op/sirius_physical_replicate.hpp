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

//! Repeats each input row by a count column and drops that column. Input batches are the output
//! columns followed by the count column, which must be last. Row `i` of an input batch appears
//! `count[i]` times, copies adjacent, in input order. Streaming and stateless, like `FILTER`: it
//! knows nothing of what the count means. A null or negative count is a planner bug and throws
//! `sirius::internal_exception`. Each input batch becomes one or more output batches within
//! `gpu_replicate_impl::limits`; one with no copies becomes one empty batch.
//!
//! One `execute` holds the whole expansion of its input batches in device memory, so the limits
//! shape output batches but do not bound peak memory.
class sirius_physical_replicate : public sirius_physical_operator {
 public:
  static constexpr SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::REPLICATE;

  //! @throws sirius::internal_exception if @p types is empty or a limit in @p output_limits is not
  //! positive.
  //!
  //! @param types  Output schema: the input schema without the count column.
  sirius_physical_replicate(duckdb::vector<sirius::logical_type> types,
                            gpu_replicate_impl::limits output_limits,
                            std::size_t estimated_cardinality);

  //! Caps on each output batch.
  [[nodiscard]] gpu_replicate_impl::limits const& output_limits() const noexcept
  {
    return _output_limits;
  }

  std::string params_to_string() const override;

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         ::cuda::stream_ref stream) override;

 private:
  gpu_replicate_impl::limits _output_limits;
};

}  // namespace op
}  // namespace sirius
