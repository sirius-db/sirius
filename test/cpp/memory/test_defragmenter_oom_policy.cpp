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

#include "memory/defragmenter_oom_policy.hpp"

#include <catch.hpp>

using sirius::memory::detail::kRelaxTrimAfterFailures;
using sirius::memory::detail::should_trim;

TEST_CASE("defragmenter: the 10x free-space rule holds for isolated failures",
          "[memory][defragmenter]")
{
  constexpr std::size_t request = 100;
  CHECK(should_trim(/*free=*/1000, request, /*factor=*/10.0, /*failures=*/1));
  CHECK_FALSE(should_trim(/*free=*/999, request, 10.0, 1));
  // Free memory at the request size is not enough while failures are isolated.
  CHECK_FALSE(should_trim(/*free=*/request, request, 10.0, kRelaxTrimAfterFailures - 1));
}

TEST_CASE("defragmenter: repeated physical failures relax the rule to 1x the request",
          "[memory][defragmenter]")
{
  constexpr std::size_t request = 100;
  CHECK(should_trim(/*free=*/request, request, 10.0, kRelaxTrimAfterFailures));
  CHECK(should_trim(/*free=*/5 * request, request, 10.0, kRelaxTrimAfterFailures + 100));
  // Never below the request itself: a trim cannot produce a block the pool does
  // not hold free, however often the allocation has failed.
  CHECK_FALSE(should_trim(/*free=*/request - 1, request, 10.0, kRelaxTrimAfterFailures + 100));
}

TEST_CASE("defragmenter: a non-positive factor always trims", "[memory][defragmenter]")
{
  CHECK(should_trim(/*free=*/0, /*bytes=*/100, /*factor=*/0.0, /*failures=*/0));
}
