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

#include <memory>
#include <utility>
#include <vector>

namespace sirius {

/**
 * @brief The caches one query's expression evaluation shares, derived from the query's constants.
 *
 * `sirius_engine` creates one per query over the query's GPU memory spaces and hands it to the
 * query's `pipeline::pipeline_build_context`, which passes it through the pipelines and operators
 * to every task-local `expression_evaluator`, so work keyed by a constant is done once per query
 * rather than once per evaluator or batch. Every cache is thread-safe through `const` access and
 * lives exactly as long as the query.
 */
struct expression_query_caches {
  /** @brief Caches for code without Sirius memory spaces; see `nvrtc::string_constant_cache`. */
  expression_query_caches() = default;

  /** @brief Caches whose device copies are charged to @p gpu_spaces, which must outlive them. */
  explicit expression_query_caches(std::vector<cucascade::memory::memory_space const*> gpu_spaces)
    : string_constants(std::move(gpu_spaces))
  {
  }

  /** @brief Classified multi-literal LIKE patterns. */
  like_multiliteral_cache like_patterns;
  /** @brief Device copies of the string constants of `NVRTC_JIT` kernels. */
  nvrtc::string_constant_cache string_constants;
};

/**
 * @brief Returns @p caches, or new caches private to the caller when @p caches is null.
 *
 * Only code outside a query, such as a unit test, passes null: private caches cost nothing until
 * a kernel acquires a string constant, and then upload it once per owner instead of once per query.
 */
[[nodiscard]] inline std::shared_ptr<expression_query_caches const> query_caches_or_private(
  std::shared_ptr<expression_query_caches const> caches)
{
  return caches ? std::move(caches) : std::make_shared<expression_query_caches>();
}

}  // namespace sirius
