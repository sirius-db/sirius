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

#include "op/sirius_physical_dummy_scan.hpp"

namespace sirius {
namespace op {

std::unique_ptr<operator_data> sirius_physical_dummy_scan::execute(const operator_data& input_data,
                                                                   rmm::cuda_stream_view /*stream*/)
{
  /// DUMMY_SCAN is the source of its pipeline. The upstream CPU_SOURCE produces
  /// a single 1-row sentinel batch into this operator's input port; execute()
  /// must forward that batch unchanged so downstream operators (e.g. a
  /// PROJECTION of constant expressions) see one input row and produce one
  /// output row. The base execute() returns an empty batch, which would drop
  /// the row and yield zero output rows.

  // Forward the upstream CPU_SOURCE's single 1-row sentinel batch unchanged.
  auto& input = dynamic_cast<const pipelineable_operator_data&>(input_data);
  return std::make_unique<pipelineable_operator_data>(input.get_data_batches());
}

}  // namespace op
}  // namespace sirius
