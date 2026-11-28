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

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sirius::meld {

/**
 * @brief A Meld plan that stands in for a run of Sirius operators: one root reading one table,
 * run on the GPU over each batch of it.
 *
 * The plan names the batch's columns, and a batch is bound to them in the order @p column_names
 * gives at check(). Meld compiles a plan once and keeps the code for every run of it after.
 */
class meld_unit {
 public:
  /**
   * @brief Compiles @p plan for the GPU and runs it over no rows, which is where Meld refuses a
   * plan it cannot run, and where the answer's types are learned.
   *
   * @p column_names names the columns of the batches the unit will be run over, in order, and has
   * to name every column the plan reads. Returns nullptr and says why in @p why when Meld refuses.
   */
  static std::unique_ptr<meld_unit> check(std::string plan,
                                          std::vector<std::string> column_names,
                                          std::string& why);

  /// The types of the answer's columns, one row of them per batch for an ungrouped aggregate.
  [[nodiscard]] std::vector<cudf::data_type> const& answer_types() const { return _answer_types; }

  [[nodiscard]] std::string const& plan() const { return _plan; }

  /**
   * @brief Runs the plan over @p batch, which Meld reads where it is, on the current device.
   *
   * Synchronizes @p stream first, since Meld runs on a stream of its own. The device memory Meld
   * takes for the run comes from @p mr, and is given back by the time the call returns. The answer
   * comes back from Meld on the host and is copied to a table allocated from @p mr too. Throws
   * sirius::internal_exception when Meld fails.
   */
  [[nodiscard]] std::unique_ptr<cudf::table> run(cudf::table_view const& batch,
                                                 ::cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref const& mr) const;

  /// How many batches every unit in this process has run over.
  static std::uint64_t batches_run();

 private:
  meld_unit(std::string plan,
            std::vector<std::string> column_names,
            std::vector<cudf::data_type> answer_types)
    : _plan(std::move(plan)),
      _column_names(std::move(column_names)),
      _answer_types(std::move(answer_types))
  {
  }

  std::string _plan;
  std::vector<std::string> _column_names;
  std::vector<cudf::data_type> _answer_types;
};

}  // namespace sirius::meld
