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

#include <op/scan/dynamic_filter_merge.hpp>
#include <op/sirius_physical_operator.hpp>

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sirius::op::scan {

/**
 * @brief Carries one scan batch and its evidence to sirius_physical_dynamic_filter.
 */
class scan_output_operator_data final : public pipelineable_operator_data {
 public:
  /**
   * @brief Constructs the complete payload before receipt installation
   *
   * @throw std::invalid_argument if batch is null
   */
  explicit scan_output_operator_data(std::shared_ptr<cucascade::data_batch> batch)
    : pipelineable_operator_data(single_batch(std::move(batch)))
  {
  }

  [[nodiscard]] batch_receipt const& receipt() const noexcept { return _receipt; }

  void install_receipt(batch_receipt&& receipt) noexcept { _receipt = std::move(receipt); }

  /**
   * @brief Keeps a copy of the receipt; see
   * `pipelineable_operator_data::with_batches_preserving_rows`.
   *
   * The receipt describes the scan's rows, which a row-preserving replacement does not change.
   */
  [[nodiscard]] std::unique_ptr<pipelineable_operator_data> with_batches_preserving_rows(
    std::vector<std::shared_ptr<cucascade::data_batch>> batches) const override
  {
    if (batches.size() != 1) {
      throw std::invalid_argument("Scan output is replaced by exactly one batch");
    }
    auto replacement = std::make_unique<scan_output_operator_data>(std::move(batches.front()));
    replacement->adopt_row_identity(*this);
    replacement->_receipt = _receipt;
    return replacement;
  }

 private:
  static std::vector<std::shared_ptr<cucascade::data_batch>> single_batch(
    std::shared_ptr<cucascade::data_batch> batch)
  {
    if (!batch) { throw std::invalid_argument("Scan output requires one nonnull batch"); }
    return {std::move(batch)};
  }

  batch_receipt _receipt;
};

}  // namespace sirius::op::scan
