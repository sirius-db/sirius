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

#include <cstdint>
#include <span>
#include <string_view>

namespace sirius::test {

/**
 * @brief Lazily bring up the SeaweedFS test backend for the [s3] integration tests.
 *
 * On the first call it:
 *   1. starts a local SeaweedFS process serving HTTP and self-signed TLS,
 *   2. generates and uploads fixtures using cuCascade's SigV4 signer and libcurl,
 *   3. publishes the SIRIUS_TEST_S3_* environment variables.
 * The process is terminated by shutdown_s3_test_env(); Linux also sends it
 * SIGKILL if the test process dies. The weed binary comes from PATH (Pixi),
 * or SIRIUS_TEST_WEED when set.
 *
 * The call is idempotent and cheap after the first success; main calls it once
 * before the tests run.
 *
 * Bring-up is **opt-in** via the @c SIRIUS_TEST_S3_AUTO env var so the default
 * `make test` suite never starts a server. Behavior:
 *   - If @c SIRIUS_TEST_S3_ENDPOINT is already set (manual run / real AWS), it is
 *     used as-is and no server is started → returns true.
 *   - Else if @c SIRIUS_TEST_S3_AUTO is not truthy, returns false. Callers use
 *     skip_or_fail_unless() to report a skip or fail under STRICT.
 *   - Else the server is brought up. On success returns true; on failure it
 *     returns false (skip) unless @c SIRIUS_TEST_S3_STRICT is truthy, in which
 *     case it throws std::runtime_error so the job goes red.
 *
 * @return true if the [s3] tests should run (env is ready), false to skip.
 */
bool ensure_s3_test_env();

/**
 * @brief PUT a small object into the managed HTTP SeaweedFS instance.
 *
 * @return false when the S3 environment is externally managed; throws on an
 * upload failure.
 */
bool put_s3_test_object(std::string_view key, std::span<std::uint8_t const> bytes);

/**
 * @brief Terminate the SeaweedFS process started by @ref ensure_s3_test_env.
 *
 * Safe to call when nothing was started and safe to call more than once. Invoked
 * once from unittest.cpp's main() before exit.
 */
void shutdown_s3_test_env();

}  // namespace sirius::test
