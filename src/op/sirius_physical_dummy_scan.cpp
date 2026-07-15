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

#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"

namespace sirius {
namespace op {

void sirius_physical_dummy_scan::build_pipelines(pipeline::sirius_pipeline& current,
                                                 pipeline::sirius_meta_pipeline& meta_pipeline)
{
  // The RIGHT_DELIM_JOIN build placeholder carries no runtime data — the delim join fans its
  // input directly to PARTITION_build — so it contributes no pipeline operator. Real DUMMY_SCANs
  // (constant-row subqueries) fall back to the base source-leaf behavior.
  if (is_delim_join_placeholder()) { return; }
  sirius_physical_operator::build_pipelines(current, meta_pipeline);
}

}  // namespace op
}  // namespace sirius
