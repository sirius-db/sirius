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

#include <cstdint>
#include <memory>

namespace cucascade::memory {
class memory_space;
}

namespace sirius::exchange {
class exchange_service;
}

namespace sirius::planner {

/**
 * @brief Rewrites a plan, after insert_gpu_pipeline_operators, to run on every rank of
 * @p service, each over its own share of the scanned files.
 *
 * Exchanges go where the single-process plan already partitions:
 * - a hash join: both sides hash-exchanged on the join keys, below each side's PARTITION;
 * - a grouped aggregate: partial results hash-exchanged on the group keys, before the merge;
 * - an ungrouped aggregate: partial results gathered to rank 0, before the merge;
 * - an ORDER BY: rows range-exchanged after SORT_SAMPLE, whose boundaries every rank agrees on;
 * - the result: gathered to rank 0, ordered by rank after an ORDER BY.
 * Rank 0's result is the query's; the other ranks' are empty, or meaningless after a gather.
 *
 * Every rank must plan the same query: the plan's shape is compared across ranks, and a
 * mismatch fails the query on every rank.
 *
 * @param query This query's number, the same on every rank: its exchanges' stream ids.
 * @throws sirius::not_implemented_exception on an operator distributed execution does not
 *         support, rather than a wrong answer.
 */
void distribute_plan(duckdb::unique_ptr<op::sirius_physical_operator>& plan,
                     std::shared_ptr<exchange::exchange_service> service,
                     cucascade::memory::memory_space& gpu,
                     std::uint64_t query);

}  // namespace sirius::planner
