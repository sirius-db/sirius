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

namespace sirius::test {

/**
 * @brief Lazily bring up the S3 test backend for the [s3] integration tests.
 *
 * On the first call it:
 *   1. spawns a single SeaweedFS `weed server` process (no Docker) that serves
 *      the S3 API over both plain HTTP and self-signed TLS at once — both
 *      listeners share one filer backend — on dynamically-chosen free ports,
 *   2. generates the local fixtures (`generate_fixtures.py`) and uploads them
 *      once (over HTTP) using Sirius's own SigV4 signer + libcurl, and
 *   3. publishes the `SIRIUS_TEST_S3_*` env vars the [s3] tests consume.
 * The `weed` process is torn down at process exit (see @ref
 * shutdown_s3_test_env); on Linux it also inherits a parent-death signal so a
 * crashed test binary takes it down too.
 *
 * The call is idempotent and cheap after the first success — safe to invoke
 * before every [s3] test from the Catch2 listener.
 *
 * Bring-up is **opt-in** via the @c SIRIUS_TEST_S3_AUTO env var so the default
 * `make test` suite never spawns a server. Behavior:
 *   - If @c SIRIUS_TEST_S3_ENDPOINT is already set (manual run / real AWS), it is
 *     used as-is and no server is started → returns true.
 *   - Else if @c SIRIUS_TEST_S3_AUTO is not truthy, returns false (tests skip,
 *     exactly as before).
 *   - Else the server is brought up. On success returns true; on failure it
 *     returns false (skip) unless @c SIRIUS_TEST_S3_STRICT is truthy, in which
 *     case it throws std::runtime_error so the job goes red.
 *
 * The `weed` binary is resolved from @c PATH (provided by the pixi env's
 * `seaweedfs` package); override with the @c SIRIUS_TEST_WEED env var.
 *
 * @return true if the [s3] tests should run (env is ready), false to skip.
 */
bool ensure_s3_test_env();

/**
 * @brief Terminate the `weed` server started by @ref ensure_s3_test_env.
 *
 * Safe to call when nothing was started and safe to call more than once. Invoked
 * once from unittest.cpp's main() before exit.
 */
void shutdown_s3_test_env();

}  // namespace sirius::test
