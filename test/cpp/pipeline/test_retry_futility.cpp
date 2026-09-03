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
#include "pipeline/retry_futility.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

using Catch::Contains;
using sirius::pipeline::assess_retry_futility;
using sirius::pipeline::retry_futility_input;
using sirius::pipeline::retry_gate_observation;

namespace {

// The q05 chain of original task 2567 on cn1 at SF1000 (100 GiB pool): every retry gate granted
// 3,076,488,960 of the 6,637,897,728-byte floor, the downgrade freed 0, and the OOM needed
// live + requested = 3,376,381,184 bytes. 104,297,693,440 bytes (97.1% of the space) were held
// outside any task reservation (parked fragment output).
constexpr std::size_t kSpaceMax  = 107374182400ULL;
constexpr std::size_t kRequested = 6637897728ULL;
constexpr std::size_t kGranted   = 3076488960ULL;
constexpr std::size_t kRequired  = 3376381184ULL;
constexpr std::uint64_t kEpoch   = 41;

retry_gate_observation q05_gate()
{
  return retry_gate_observation{.requested_bytes         = kRequested,
                                .granted_bytes           = kGranted,
                                .freed_by_downgrade      = 0,
                                .downgrade_requested     = true,
                                .disk_tier_configured    = false,
                                .space_max_bytes         = kSpaceMax,
                                .completed_epoch_at_gate = kEpoch};
}

retry_futility_input q05_futile()
{
  return retry_futility_input{.is_oom                      = true,
                              .retry_count                 = 1,
                              .oom_required_bytes          = kRequired,
                              .gate                        = q05_gate(),
                              .completed_epoch_now         = kEpoch,
                              .inflight_first_attempts_now = 0};
}

}  // namespace

TEST_CASE("assess_retry_futility decides from the gate observation and progress signals",
          "[retry_futility]")
{
  SECTION("the q05 shape is futile and the reason names the held bytes")
  {
    auto const reason = assess_retry_futility(q05_futile());
    REQUIRE(reason.has_value());
    CHECK_THAT(*reason, Contains("held outside any task reservation"));
    CHECK_THAT(*reason, Contains("104297693440"));
    CHECK_THAT(*reason, Contains("97.1%"));
    CHECK_THAT(*reason, Contains(std::to_string(kRequired)));
    CHECK_THAT(*reason, Contains(std::to_string(kGranted)));
    CHECK_THAT(*reason, Contains(std::to_string(kRequested)));
    CHECK_THAT(*reason, Contains("freed 0 bytes"));
    CHECK_THAT(*reason, Contains("disk tier not configured"));
    CHECK_THAT(*reason, Contains("retry cap 100"));
  }

  SECTION("not an OOM: CUDA-launch reschedules keep the plain retry budget")
  {
    auto in   = q05_futile();
    in.is_oom = false;
    REQUIRE_FALSE(assess_retry_futility(in).has_value());
  }

  SECTION("first attempt never fails fast")
  {
    auto in        = q05_futile();
    in.retry_count = 0;
    REQUIRE_FALSE(assess_retry_futility(in).has_value());
  }

  SECTION("no gate observation: defensive nullopt")
  {
    auto in = q05_futile();
    in.gate = std::nullopt;
    REQUIRE_FALSE(assess_retry_futility(in).has_value());
  }

  SECTION("full grant: the gate was not the constraint (#732 contention shape)")
  {
    auto in                = q05_futile();
    in.gate->granted_bytes = in.gate->requested_bytes;
    in.oom_required_bytes  = in.gate->requested_bytes + 1;
    REQUIRE_FALSE(assess_retry_futility(in).has_value());
  }

  SECTION("spilling progressed: the downgrade freed something")
  {
    auto in                     = q05_futile();
    in.gate->freed_by_downgrade = 1;
    REQUIRE_FALSE(assess_retry_futility(in).has_value());
  }

  SECTION("need unknown: no requirement was recorded by the OOM handler")
  {
    auto in               = q05_futile();
    in.oom_required_bytes = 0;
    REQUIRE_FALSE(assess_retry_futility(in).has_value());
  }

  SECTION("need fits in the grant")
  {
    auto in               = q05_futile();
    in.oom_required_bytes = in.gate->granted_bytes;
    REQUIRE_FALSE(assess_retry_futility(in).has_value());
  }

  SECTION("something completed since the gate")
  {
    auto in                = q05_futile();
    in.completed_epoch_now = in.gate->completed_epoch_at_gate + 1;
    REQUIRE_FALSE(assess_retry_futility(in).has_value());
  }

  SECTION("a first attempt is running somewhere")
  {
    auto in                        = q05_futile();
    in.inflight_first_attempts_now = 1;
    REQUIRE_FALSE(assess_retry_futility(in).has_value());
  }

  SECTION("clamped floor: requested == space max is still short when granted < space max")
  {
    auto in                  = q05_futile();
    in.gate->requested_bytes = kSpaceMax;
    auto const reason        = assess_retry_futility(in);
    REQUIRE(reason.has_value());
    CHECK_THAT(*reason, Contains("held outside any task reservation"));
  }

  SECTION("no downgrade executor attached")
  {
    auto in                      = q05_futile();
    in.gate->downgrade_requested = false;
    auto const reason            = assess_retry_futility(in);
    REQUIRE(reason.has_value());
    CHECK_THAT(*reason, Contains("no downgrade executor"));
    CHECK_THAT(*reason, Contains("held outside any task reservation"));
  }

  SECTION("disk tier present but the downgrade still freed 0")
  {
    auto in                       = q05_futile();
    in.gate->disk_tier_configured = true;
    auto const reason             = assess_retry_futility(in);
    REQUIRE(reason.has_value());
    CHECK_THAT(*reason, Contains("freed 0 bytes"));
    CHECK_THAT(*reason, Contains("disk tier configured"));
    CHECK_THAT(*reason, !Contains("disk tier not configured"));
  }

  SECTION("a later retry with the same shape is also futile")
  {
    auto in        = q05_futile();
    in.retry_count = 7;
    REQUIRE(assess_retry_futility(in).has_value());
  }
}
