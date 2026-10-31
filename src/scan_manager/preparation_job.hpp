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
#include "exec/invocable.hpp"
#include "op/scan/gpu_ingestible_types.hpp"
#include "scan_manager/preparation.hpp"
#include "scan_manager/preparation_stop_token.hpp"

#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace sirius::scan_manager {
struct preparation_unit_requirement {
  unit_key key;
  required_input_set required;
};
// A claimed operation owns its captures until execution or abandonment, never both.
class preparation_job {
 public:
  preparation_job() = default;
  template <typename F>
    requires(!std::is_same_v<std::remove_cvref_t<F>, preparation_job>)
  preparation_job(F&& work, std::shared_ptr<preparation_unit> readiness = {})
    : unit(std::move(readiness))
  {
    operation_ = [fn = std::forward<F>(work)](
                   preparation_stop_token stop) mutable -> std::unique_ptr<op::scan::scan_info> {
      if constexpr (std::is_invocable_v<F&, preparation_stop_token>)
        return fn(stop);
      else
        return fn();
    };
  }
  preparation_job(preparation_job&&)            = default;
  preparation_job& operator=(preparation_job&&) = default;
  template <typename F, typename Requirement>
    requires(std::is_same_v<Requirement, preparation_unit_requirement> ||
             std::is_same_v<Requirement, std::optional<preparation_unit_requirement>>)
  preparation_job(F&& work, Requirement requirement) : preparation_job(std::forward<F>(work))
  {
    this->requirement = requirement;
  }
  // Transform the raw result before run() settles the admitted unit. Keep admission,
  // dependency completion and failure reporting on this job, including mapper failures.
  template <typename F>
  preparation_job map_output(F&& transform) &&
  {
    if (!operation_) throw std::logic_error("preparation job already consumed");
    operation_ = [operation = std::move(operation_), fn = std::forward<F>(transform)](
                   preparation_stop_token stop) mutable -> std::unique_ptr<op::scan::scan_info> {
      return fn(operation(stop));
    };
    return std::move(*this);
  }
  std::unique_ptr<op::scan::scan_info> run(preparation_stop_token stop);
  std::optional<preparation_unit_requirement> requirement;
  std::shared_ptr<preparation_unit> unit;

 private:
  exec::invocable<std::unique_ptr<op::scan::scan_info>(preparation_stop_token)> operation_;
};

}  // namespace sirius::scan_manager
