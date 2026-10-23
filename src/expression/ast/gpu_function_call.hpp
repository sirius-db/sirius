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

#include "helper/logical_type.hpp"

#include <sirius/udf.hpp>

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace sirius::ast {

struct node;

/// A resolved GPU call. Runtime state and children are entirely Sirius-native;
/// DuckDB is only involved when the implementation is selected at plan time.
class gpu_function_call {
 public:
  gpu_function_call(std::shared_ptr<gpu_scalar_function const> implementation,
                    std::vector<std::unique_ptr<node>> arguments,
                    sirius::logical_type return_type,
                    bool propagate_nulls = true)
    : implementation_(std::move(implementation)),
      arguments_(std::move(arguments)),
      return_type_(std::move(return_type)),
      propagate_nulls_(propagate_nulls)
  {
  }

  [[nodiscard]] auto const& implementation() const noexcept { return implementation_; }
  [[nodiscard]] auto const& arguments() const noexcept { return arguments_; }
  [[nodiscard]] sirius::logical_type const& return_type() const noexcept { return return_type_; }
  [[nodiscard]] bool propagate_nulls() const noexcept { return propagate_nulls_; }
  [[nodiscard]] std::size_t cudf_ast_op_count() const noexcept { return 0; }

 private:
  std::shared_ptr<gpu_scalar_function const> implementation_;
  std::vector<std::unique_ptr<node>> arguments_;
  sirius::logical_type return_type_;
  bool propagate_nulls_;
};

}  // namespace sirius::ast
