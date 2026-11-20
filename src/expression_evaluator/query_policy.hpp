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

#pragma once

#include "expression_evaluator/expression_evaluator_strategy.hpp"

namespace sirius {

/// Expression policy copied from the query or pin operation's settings snapshot.
/// Evaluators retain a const copy; standalone callers may use these fixed defaults.
struct expression_query_policy {
  expression_evaluator_strategy strategy{expression_evaluator_strategy::AST_INTERPRET};
  bool regex_jit{true};
  /// Whether NVRTC may fuse a floating-point multiplication with an addition or subtraction of its
  /// product in `NVRTC_JIT` kernels. The fused operation does not round the product first, so
  /// multiply-add chains run faster and their results differ from DuckDB's and the other
  /// strategies' by up to one rounding of the product. Where the terms cancel, that is the whole
  /// result, so whatever is computed from it (a comparison, a cast, a key) can differ as well.
  bool nvrtc_jit_fuse_multiply_add{false};
};

}  // namespace sirius
