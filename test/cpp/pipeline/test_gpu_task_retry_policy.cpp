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
#include "pipeline/gpu_task_retry_policy.hpp"
#include "sirius_config.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

using namespace std::chrono_literals;
using sirius::exec::gpu_task_retry_config;
using sirius::pipeline::gpu_task_retry_state;
using sirius::pipeline::gpu_task_retry_verdict;
using sirius::pipeline::memory_progress_signal;
using sirius::pipeline::on_gpu_task_retry;

namespace {

gpu_task_retry_config test_config()
{
  gpu_task_retry_config c;
  c.max_retries         = 5;
  c.no_progress_timeout = 1000ms;
  c.initial_backoff     = 10ms;
  c.max_backoff         = 80ms;
  return c;
}

}  // namespace

TEST_CASE("retry policy charges only full-reservation attempts", "[gpu_task_retry][pipeline]")
{
  auto const cfg = test_config();
  gpu_task_retry_state state;
  auto const t0 = std::chrono::steady_clock::now();

  // Partial-reservation attempts never exhaust the count, however many there are.
  for (int i = 0; i < 50; ++i) {
    // The epoch moves each time so the no-progress clock never trips.
    auto d = on_gpu_task_retry(state, cfg, /*full_reservation=*/false, i, false, t0 + i * 1ms);
    REQUIRE(d.verdict == gpu_task_retry_verdict::retry);
  }
  CHECK(state.total_retries == 50);
  CHECK(state.charged_retries == 0);

  for (int i = 0; i < 5; ++i) {
    auto d = on_gpu_task_retry(state, cfg, true, 100 + i, false, t0 + 100ms);
    REQUIRE(d.verdict == gpu_task_retry_verdict::retry);
  }
  CHECK(state.charged_retries == 5);
  auto d = on_gpu_task_retry(state, cfg, true, 200, false, t0 + 100ms);
  CHECK(d.verdict == gpu_task_retry_verdict::fail_max_retries);
}

TEST_CASE("retry policy max_retries 0 disables the count", "[gpu_task_retry][pipeline]")
{
  auto cfg        = test_config();
  cfg.max_retries = 0;
  gpu_task_retry_state state;
  auto const t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < 1000; ++i) {
    REQUIRE(on_gpu_task_retry(state, cfg, true, i, false, t0).verdict ==
            gpu_task_retry_verdict::retry);
  }
}

TEST_CASE("retry policy backoff doubles without progress and resets on progress",
          "[gpu_task_retry][pipeline]")
{
  auto const cfg = test_config();
  gpu_task_retry_state state;
  auto const t0 = std::chrono::steady_clock::now();

  // Same epoch every time: no progress, so the wait doubles up to max_backoff.
  std::chrono::milliseconds expected[] = {10ms, 20ms, 40ms, 80ms, 80ms};
  for (auto e : expected) {
    auto d = on_gpu_task_retry(state, cfg, false, 7, false, t0);
    REQUIRE(d.verdict == gpu_task_retry_verdict::retry);
    CHECK(d.min_wait == 10ms);
    CHECK(d.max_wait == e);
  }
  CHECK(state.retries_since_progress == 5);

  // The epoch moved: progress, so the backoff starts over.
  auto d = on_gpu_task_retry(state, cfg, false, 8, false, t0);
  CHECK(state.retries_since_progress == 1);
  CHECK(d.max_wait == 10ms);
  CHECK(state.last_seen_epoch == 8);
}

TEST_CASE("retry policy backoff exponent saturates", "[gpu_task_retry][pipeline]")
{
  auto cfg        = test_config();
  cfg.max_retries = 0;
  cfg.max_backoff = std::chrono::hours{1000};
  gpu_task_retry_state state;
  auto const t0 = std::chrono::steady_clock::now();
  std::chrono::milliseconds last{0};
  for (int i = 0; i < 100; ++i) {
    auto d = on_gpu_task_retry(state, cfg, false, 1, false, t0);
    REQUIRE(d.max_wait >= last);
    last = d.max_wait;
  }
  CHECK(last == cfg.initial_backoff * (1 << 20));
}

TEST_CASE("retry policy fails only after the no-progress timeout", "[gpu_task_retry][pipeline]")
{
  auto const cfg = test_config();
  gpu_task_retry_state state;
  auto const t0 = std::chrono::steady_clock::now();

  REQUIRE(on_gpu_task_retry(state, cfg, false, 3, false, t0).verdict ==
          gpu_task_retry_verdict::retry);
  // Within the timeout and no progress: still retrying.
  REQUIRE(on_gpu_task_retry(state, cfg, false, 3, false, t0 + 900ms).verdict ==
          gpu_task_retry_verdict::retry);
  // Progress at 900 ms+ restarts the clock ...
  REQUIRE(on_gpu_task_retry(state, cfg, false, 4, false, t0 + 950ms).verdict ==
          gpu_task_retry_verdict::retry);
  REQUIRE(on_gpu_task_retry(state, cfg, false, 4, false, t0 + 1900ms).verdict ==
          gpu_task_retry_verdict::retry);
  // ... and a full timeout without it fails the task.
  CHECK(on_gpu_task_retry(state, cfg, false, 4, false, t0 + 2000ms).verdict ==
        gpu_task_retry_verdict::fail_no_progress);
}

TEST_CASE("retry policy fails fast when the engine is idle", "[gpu_task_retry][pipeline]")
{
  auto cfg             = test_config();
  cfg.max_retries      = 0;
  cfg.max_idle_retries = 3;
  gpu_task_retry_state state;
  auto const t0 = std::chrono::steady_clock::now();

  for (int i = 0; i < 3; ++i) {
    REQUIRE(on_gpu_task_retry(state, cfg, false, 1, true, t0).verdict ==
            gpu_task_retry_verdict::retry);
  }
  CHECK(state.idle_retries == 3);
  // Progress since the previous retry: even an idle engine gets a fresh budget.
  REQUIRE(on_gpu_task_retry(state, cfg, false, 2, true, t0).verdict ==
          gpu_task_retry_verdict::retry);
  CHECK(state.idle_retries == 0);
  // Work in flight resets the idle streak as well.
  REQUIRE(on_gpu_task_retry(state, cfg, false, 2, true, t0).verdict ==
          gpu_task_retry_verdict::retry);
  REQUIRE(on_gpu_task_retry(state, cfg, false, 2, false, t0).verdict ==
          gpu_task_retry_verdict::retry);
  CHECK(state.idle_retries == 0);
  for (int i = 0; i < 3; ++i) {
    REQUIRE(on_gpu_task_retry(state, cfg, false, 2, true, t0).verdict ==
            gpu_task_retry_verdict::retry);
  }
  CHECK(on_gpu_task_retry(state, cfg, false, 2, true, t0).verdict ==
        gpu_task_retry_verdict::fail_idle);
}

TEST_CASE("memory progress signal tracks in-flight activity", "[gpu_task_retry][pipeline]")
{
  memory_progress_signal signal;
  CHECK(signal.idle());
  {
    auto a = signal.track_activity();
    CHECK_FALSE(signal.idle());
    {
      auto b = signal.track_activity();
      CHECK_FALSE(signal.idle());
    }
    CHECK_FALSE(signal.idle());
  }
  CHECK(signal.idle());
}

TEST_CASE("memory progress signal wakes a waiter early", "[gpu_task_retry][pipeline]")
{
  memory_progress_signal signal;
  auto const seen = signal.epoch();

  std::thread poster([&] {
    std::this_thread::sleep_for(50ms);
    signal.notify();
  });
  auto const start = std::chrono::steady_clock::now();
  bool const woke  = signal.wait_for_progress(seen, 1ms, 10s);
  auto const took  = std::chrono::steady_clock::now() - start;
  poster.join();

  CHECK(woke);
  CHECK(took < 5s);
  CHECK(signal.epoch() == seen + 1);
}

TEST_CASE("memory progress signal honors min and max wait", "[gpu_task_retry][pipeline]")
{
  memory_progress_signal signal;
  auto const seen = signal.epoch();

  // Times out without an event.
  auto start = std::chrono::steady_clock::now();
  CHECK_FALSE(signal.wait_for_progress(seen, 1ms, 30ms));
  CHECK(std::chrono::steady_clock::now() - start >= 30ms);

  // An event already posted still waits out min_wait.
  signal.notify();
  start = std::chrono::steady_clock::now();
  CHECK(signal.wait_for_progress(seen, 40ms, 10s));
  auto const took = std::chrono::steady_clock::now() - start;
  CHECK(took >= 40ms);
  CHECK(took < 5s);
}

TEST_CASE("sirius_config reads sirius.executor.pipeline.retry", "[gpu_task_retry][config]")
{
  auto const path = std::filesystem::temp_directory_path() / "sirius_gpu_retry_config.yaml";
  {
    std::ofstream out(path);
    out << "sirius:\n"
           "  executor:\n"
           "    pipeline:\n"
           "      num_threads: 3\n"
           "      retry:\n"
           "        max_retries: 42\n"
           "        max_idle_retries: 7\n"
           "        no_progress_timeout: 60000\n"
           "        initial_backoff: 5\n"
           "        max_backoff: 500\n";
  }
  sirius::sirius_config cfg;
  cfg.load_from_file(path);
  std::filesystem::remove(path);

  CHECK(cfg.get_gpu_pipeline_executor_config().num_threads == 3);
  auto const& retry = cfg.get_gpu_task_retry_config();
  CHECK(retry.max_retries == 42);
  CHECK(retry.max_idle_retries == 7);
  CHECK(retry.no_progress_timeout == 60000ms);
  CHECK(retry.initial_backoff == 5ms);
  CHECK(retry.max_backoff == 500ms);

  sirius::sirius_config defaults;
  defaults.apply_defaults();
  CHECK(defaults.get_gpu_task_retry_config().max_retries == 10000);
  CHECK(defaults.get_gpu_task_retry_config().no_progress_timeout == std::chrono::minutes{10});
}
