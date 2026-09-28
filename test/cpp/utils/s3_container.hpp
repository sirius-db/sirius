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
 * @brief Lazily bring up the MinIO test backend for the [s3] integration tests.
 *
 * Replaces the old out-of-band `make s3-up` flow (docker-compose + fixtures.sh +
 * env.sh). On the first call it:
 *   1. starts two MinIO containers (HTTP + self-signed-TLS) via the vendored
 *      testcontainers-native bridge, on dynamically-mapped host ports,
 *   2. generates the local fixtures (`generate_fixtures.py`) and uploads them to
 *      both endpoints from the host using Sirius's own SigV4 signer + libcurl
 *      (no `mc` container, no docker network), and
 *   3. publishes the `SIRIUS_TEST_S3_*` env vars the [s3] tests already consume.
 * Containers are torn down at process exit (see @ref shutdown_s3_container_env);
 * testcontainers' Ryuk sidecar guarantees cleanup even on a crash.
 *
 * The call is idempotent and cheap after the first success — safe to invoke
 * before every [s3] test from the Catch2 listener.
 *
 * Bring-up is **opt-in** via the @c SIRIUS_TEST_S3_AUTO env var so the default
 * `make test` suite never needs Docker. Behavior:
 *   - If @c SIRIUS_TEST_S3_ENDPOINT is already set (manual run / real AWS), it is
 *     used as-is and no container is started → returns true.
 *   - Else if @c SIRIUS_TEST_S3_AUTO is not truthy, returns false (tests skip,
 *     exactly as before).
 *   - Else containers are brought up. On success returns true; on failure it
 *     returns false (skip) unless @c SIRIUS_TEST_S3_STRICT is truthy, in which
 *     case it throws std::runtime_error so the job goes red.
 *
 * @return true if the [s3] tests should run (env is ready), false to skip.
 */
bool ensure_s3_container_env();

/**
 * @brief PUT a small object into the managed HTTP MinIO instance.
 *
 * @return false when the S3 environment is externally managed; throws on an
 * upload failure.
 */
bool put_s3_container_object(std::string_view key, std::span<std::uint8_t const> bytes);

/**
 * @brief Terminate the MinIO containers started by @ref ensure_s3_container_env.
 *
 * Safe to call when nothing was started and safe to call more than once. Invoked
 * once from unittest.cpp's main() before exit.
 */
void shutdown_s3_container_env();

}  // namespace sirius::test
