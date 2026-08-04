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

// TODO(phase4): two things change here once the implementations land — drop the [.] tag, and
// restore the broad [io] tag alongside [prefetch_api].
//
// The prefetch-scan APIs currently ship as declarations whose bodies throw. Most of them are
// noexcept, so a throw from one aborts the whole test binary instead of failing a case; the
// hidden-test tag keeps the default suite runnable until the implementations arrive. A Catch2
// test spec *includes* hidden cases, so the broad tag stays off until then: with it, running
// `sirius_unittest "[io]"` would pull these in and abort.
//
// backgrounds_prefetch is the only consumer-visible policy on the prefetching_stage ladder: it
// decides whether prefetching_cache::prepare_loop hands a hint to its own background thread or
// lets the read path do the IO inline. The assertions below are STATIC_REQUIREs, so they are
// enforced at compile time whether or not the case is allowed to run.

#include <catch.hpp>
#include <io/cache/types.hpp>

TEST_CASE("backgrounds_prefetch is false exactly for none and task_preprocessing",
          "[.][prefetch_api][prefetch_stage]")
{
  using sirius::io::cache::backgrounds_prefetch;
  using sirius::io::cache::prefetching_stage;

  // One assertion per enumerator, so adding a rung to the ladder forces a conscious decision
  // about which side of the policy it falls on instead of silently inheriting "backgrounds it".
  STATIC_REQUIRE_FALSE(backgrounds_prefetch(prefetching_stage::none));
  STATIC_REQUIRE(backgrounds_prefetch(prefetching_stage::metadata_created));
  STATIC_REQUIRE(backgrounds_prefetch(prefetching_stage::task_queued));
  STATIC_REQUIRE_FALSE(backgrounds_prefetch(prefetching_stage::task_preprocessing));
  STATIC_REQUIRE(backgrounds_prefetch(prefetching_stage::disposable));
}
