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

#include "duckdb/planner/operator/logical_column_data_get.hpp"
#include "planner/sirius_physical_plan_generator.hpp"

namespace sirius::planner {

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalColumnDataGet& op)
{
  // LogicalColumnDataGet (LOGICAL_CHUNK_GET) appears at the scan leaf of
  // DESCRIBE TABLE / SHOW queries. The IN_CLAUSE optimizer (which also
  // produces these nodes) is disabled for transparent execution, so this
  // path is only reached for catalog metadata queries. Those contain nullable
  // VARCHAR columns whose GPU memory transfer fails with cudaErrorInvalidValue.
  // Fall back to CPU for all standalone catalog scans.
  throw duckdb::NotImplementedException(
    "Catalog metadata scan (LOGICAL_CHUNK_GET) is not supported in Sirius GPU execution");
}

}  // namespace sirius::planner
