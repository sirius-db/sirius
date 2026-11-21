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

#include <catch.hpp>

#include <cstdlib>
#include <string_view>

namespace sirius::test {

/**
 * @brief Fails the current test with @p reason when `SIRIUS_TEST_REQUIRE_POOL_PEER_ACCESS` is set.
 *
 * Tests of direct peer copies skip when a replica pool does not grant another GPU access, so a run
 * on GPUs without that access passes without covering them. A runner whose GPUs must support direct
 * copies sets the variable, and a test calls this just before such a skip so that it fails instead.
 * Header-only, because `sirius_cuda_fault_test` does not link `sirius_test_env.cpp`.
 */
inline void fail_if_pool_peer_access_required(std::string_view reason)
{
  if (std::getenv("SIRIUS_TEST_REQUIRE_POOL_PEER_ACCESS") != nullptr) {
    FAIL(reason << "; SIRIUS_TEST_REQUIRE_POOL_PEER_ACCESS is set, so the test cannot skip");
  }
}

}  // namespace sirius::test
