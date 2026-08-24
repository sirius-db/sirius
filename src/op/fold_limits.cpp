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

#include "op/fold_limits.hpp"

#include <stdexcept>
#include <string>

namespace sirius::op {

void check_fold_row_limit(uint64_t total_rows, std::size_t num_batches, uint64_t limit)
{
  if (total_rows <= limit) { return; }
  throw std::runtime_error("[fold_limit] cannot fold " + std::to_string(num_batches) +
                           " batches holding " + std::to_string(total_rows) +
                           " rows into one cuDF table: the limit is " + std::to_string(limit) +
                           " rows");
}

}  // namespace sirius::op
