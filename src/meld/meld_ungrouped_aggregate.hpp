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

#include "op/sirius_physical_ungrouped_aggregate.hpp"

#include <cuda/stream>

#include <memory>
#include <string>

namespace sirius::meld {

class meld_unit;

/**
 * @brief An ungrouped aggregate whose per-batch partial a Meld unit computes, from the batches of
 * the scan below in place of the filters and projections the unit took in.
 *
 * It is an UNGROUPED_AGGREGATE to the rest of Sirius, so the merge above it and the pipelines
 * around it are those of the aggregate it stands in for.
 */
class meld_ungrouped_aggregate final : public op::sirius_physical_ungrouped_aggregate {
 public:
  /// Takes @p aggregate's aggregates and children. Only before the plan's parents and operator ids
  /// are set, and for an aggregate whose columns are in their native carriers.
  meld_ungrouped_aggregate(op::sirius_physical_ungrouped_aggregate&& aggregate,
                           std::shared_ptr<meld_unit const> unit);

  std::string get_name() const override;

  std::unique_ptr<op::operator_data> execute(const op::operator_data& input_data,
                                             ::cuda::stream_ref stream) override;

 private:
  std::shared_ptr<meld_unit const> _unit;
};

}  // namespace sirius::meld
