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

namespace sirius::op {

/// Which read of a delim CTE a `CTE_SCAN` is, once the delim-to-CTE rewrite has been proven safe to
/// restrict (see `planner/delim_cte_proof.hpp`).
enum class cte_reader_role : unsigned char {
  unknown,    ///< no proof: treat the read like any other
  outer_row,  ///< reads the outer query's rows
  key,        ///< reads only their distinct correlation keys
};

}  // namespace sirius::op
