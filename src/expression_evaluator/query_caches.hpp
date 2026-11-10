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

#include "expression_evaluator/like_multiliteral.hpp"
#include "expression_evaluator/nvrtc/string_constant_cache.hpp"

namespace sirius {

/**
 * @brief The caches one query's expression evaluation shares, derived from the query's constants.
 *
 * Each query's `pipeline::pipeline_build_context` creates one and hands it, through its pipelines
 * and operators, to every task-local `expression_evaluator`, so work keyed by a constant is done
 * once per query rather than once per evaluator or batch. Every cache is thread-safe through
 * `const` access and lives exactly as long as the query.
 */
struct expression_query_caches {
  /** @brief Classified multi-literal LIKE patterns. */
  like_multiliteral_cache like_patterns;
  /** @brief Device copies of the string constants of `NVRTC_JIT` kernels. */
  nvrtc::string_constant_cache string_constants;
};

}  // namespace sirius
