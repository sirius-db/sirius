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

#include "op/sirius_physical_operator.hpp"

#include <cstddef>

namespace duckdb {
struct DBConfig;
}  // namespace duckdb

namespace sirius::meld {

/// Adds the meld_units setting, off by default, which has the engine run place_meld_units.
void add_meld_units_option(duckdb::DBConfig& config);

/**
 * @brief Hands to Meld each ungrouped aggregate whose input is filters and projections over a
 * table scan, where Meld takes the plan.
 *
 * The aggregate keeps a Meld unit computing its per-batch partial, and the scan becomes its child
 * in place of the operators the unit computes. What Meld refuses stays as it was. Runs before the
 * plan's parents are set. Returns how many aggregates Meld was given.
 */
std::size_t place_meld_units(op::sirius_physical_operator& plan);

}  // namespace sirius::meld
