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
#include "planner/gpu_admission.hpp"

#include <vector>

using sirius::planner::apply_gpu_cap;

TEST_CASE("apply_gpu_cap with no cap admits every GPU", "[gpu_admission]")
{
  const std::vector<int> all{0, 1, 2, 3};
  REQUIRE(apply_gpu_cap(all, 0) == all);
}

TEST_CASE("apply_gpu_cap takes a prefix of the sorted list", "[gpu_admission]")
{
  const std::vector<int> all{0, 1, 2, 3};
  REQUIRE(apply_gpu_cap(all, 1) == std::vector<int>{0});
  REQUIRE(apply_gpu_cap(all, 2) == std::vector<int>{0, 1});
  REQUIRE(apply_gpu_cap(all, 3) == std::vector<int>{0, 1, 2});
}

TEST_CASE("apply_gpu_cap at the fleet size is a no-op", "[gpu_admission]")
{
  const std::vector<int> all{0, 1, 2, 3};
  REQUIRE(apply_gpu_cap(all, 4) == all);
}

TEST_CASE("apply_gpu_cap above the fleet size clamps to what exists", "[gpu_admission]")
{
  // Asking for more GPUs than the host has yields every GPU rather than failing.
  const std::vector<int> all{0, 1};
  REQUIRE(apply_gpu_cap(all, 8) == all);
}

TEST_CASE("apply_gpu_cap preserves non-contiguous device ids", "[gpu_admission]")
{
  // CUDA_VISIBLE_DEVICES / explicit gpu_ids can leave gaps; the cap is positional.
  const std::vector<int> sparse{2, 5, 7};
  REQUIRE(apply_gpu_cap(sparse, 2) == std::vector<int>{2, 5});
}

TEST_CASE("apply_gpu_cap treats a negative cap as no cap", "[gpu_admission]")
{
  // Config load rejects negatives; this is the defensive path.
  const std::vector<int> all{0, 1, 2, 3};
  REQUIRE(apply_gpu_cap(all, -1) == all);
}

TEST_CASE("apply_gpu_cap on an empty fleet stays empty", "[gpu_admission]")
{
  REQUIRE(apply_gpu_cap({}, 2).empty());
}
