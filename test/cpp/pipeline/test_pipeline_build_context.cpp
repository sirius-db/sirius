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

#include "catch.hpp"
#include "pipeline/pipeline_build_context.hpp"

#include <memory>
#include <stdexcept>
#include <vector>

TEST_CASE("Engine pipeline context derives GPU count from configured device ids",
          "[sirius][pipeline][config]")
{
  std::vector<int> configured_gpu_ids{1, 3};
  sirius::pipeline::pipeline_build_context context{nullptr, true, std::move(configured_gpu_ids)};

  REQUIRE(context.num_gpus() == 2);
  REQUIRE(context.active_gpu_ids() == std::vector<int>{1, 3});
}

TEST_CASE("Engine pipeline context rejects an empty configured GPU set",
          "[sirius][pipeline][config]")
{
  REQUIRE_THROWS_AS(sirius::pipeline::pipeline_build_context(nullptr, true, std::vector<int>{}),
                    std::invalid_argument);
}

TEST_CASE("Pipeline context retains explicit GPU counts for engine-free tests",
          "[sirius][pipeline][config]")
{
  sirius::pipeline::pipeline_build_context context{nullptr, true, 3};

  REQUIRE(context.num_gpus() == 3);
  REQUIRE(context.active_gpu_ids().empty());
}

TEST_CASE("Pipeline context copies share immutable query policy and LIKE cache",
          "[sirius][pipeline][config][like_multiliteral]")
{
  auto operator_params = std::make_shared<sirius::operator_params>();
  auto like_cache      = std::make_shared<sirius::like_multiliteral_cache>();
  sirius::pipeline::pipeline_build_context context{nullptr, true, 1, operator_params, like_cache};

  auto context_copy = context;

  REQUIRE(&context.get_operator_params() == operator_params.get());
  REQUIRE(&context_copy.get_operator_params() == operator_params.get());
  REQUIRE(context.get_like_multiliteral_cache().get() == like_cache.get());
  REQUIRE(context_copy.get_like_multiliteral_cache().get() == like_cache.get());
}
