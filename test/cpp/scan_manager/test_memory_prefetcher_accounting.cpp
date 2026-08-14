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

// Admission-accounting gates for the memory prefetcher's reserve-then-gate
// admission path (fix 0727f820). Three defects used to be silent:
//
//   - gate-then-reserve overshoot: the min_free_fraction headroom gate ran
//     BEFORE make_reservation_or_null, so num_threads workers could pass the
//     gate concurrently against the same headroom and overshoot the admission
//     budget by up to num_threads x (breaching the free floor that protects
//     pipeline task reservations from unreclaimable prefetched bytes);
//   - decorative reservation: the conversion allocated through the space's
//     default allocator while the admission reservation sat unused, so the
//     space was transiently charged reserved-peak + actual-allocation
//     (double count) for every in-flight prefetch;
//   - phantom exclusive stream pool: per-worker streams from a dedicated
//     exclusive_stream_pool carried zero copy traffic (covered by review;
//     the header no longer declares _stream_pool).
//
// Gates here: admission keeps the floor including concurrent workers'
// in-flight reservations; the attached reservation genuinely draws down
// during conversion (space charged the peak ONCE, never peak + actual); the
// converted table round-trips through the per-worker-stream + tracker-attach
// path bit-exactly; and a non-admittable reservation backs the sweep off
// cleanly (_stops_reservation) without leaking arenas.
//
// The environment mirrors production accounting: Sirius configures per-THREAD
// allocation tracking (sirius_config.cpp: per_stream_reservation = false), so
// the builder here calls track_reservation_per_stream(false). The prefetcher's
// tracker attach keys on the worker's private stream but the tracker resolves
// per thread — with the cucascade-default per-STREAM tracking the conversion's
// allocations (made on a round-robin pool stream) would bypass the arena
// entirely.
//
// The diagnostic gate counters (_stops_headroom / _stops_reservation) have no
// accessors; they are logged once by stop(), so the tests parse them out of a
// recording log sink.

#include "utils/log_test_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <data/data_batch_utils.hpp>
#include <data/sirius_converter_registry.hpp>
#include <memory/sirius_memory_reservation_manager.hpp>
#include <scan_manager/load_balancing_scan_batch_coalescer.hpp>
#include <scan_manager/memory_prefetcher.hpp>
#include <scan_manager/split_connector.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <regex>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

using sirius::scan_manager::databatch_provider;
using sirius::scan_manager::load_balancing_scan_batch_coalescer;
using sirius::scan_manager::memory_prefetcher;
using sirius::scan_manager::memory_prefetcher_config;
using sirius::scan_manager::split_connector;

namespace {

constexpr std::size_t MiB = 1ull << 20;

/// Per-test environment: a fresh reservation manager over a 512MB GPU space
/// (reservation limit 384MB = 0.75 x capacity) with per-THREAD allocation
/// tracking, matching the production Sirius configuration the prefetcher's
/// tracker attach relies on. Fresh per test case so every case starts from an
/// empty space and exact availability arithmetic.
struct prefetcher_env {
  static constexpr std::size_t gpu_capacity = 512 * MiB;
  static constexpr double limit_ratio       = 0.75;

  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> mgr;
  cucascade::memory::memory_space* gpu_space;
  cucascade::memory::memory_space* host_space;
  rmm::cuda_stream conv_stream;  // converters need a non-default stream

  prefetcher_env()
  {
    sirius::converter_registry::reset_for_testing();

    cucascade::memory::reservation_manager_configurator builder;
    builder.set_number_of_gpus(1)
      .set_gpu_usage_limit(gpu_capacity)
      .set_reservation_fraction_per_gpu(limit_ratio)
      .set_per_numa_region_capacity(1ull << 30)
      .use_gpu_id_as_host_id()
      .set_reservation_fraction_per_numa_region(limit_ratio)
      .track_reservation_per_stream(false);  // production default: per-thread tracking

    auto space_configs = builder.build();
    mgr =
      std::make_unique<sirius::memory::sirius_memory_reservation_manager>(std::move(space_configs));
    sirius::converter_registry::initialize();

    gpu_space  = mgr->get_memory_space(cucascade::memory::Tier::GPU, 0);
    host_space = mgr->get_memory_space(cucascade::memory::Tier::HOST, 0);
  }

  rmm::cuda_stream_view stream() { return conv_stream.view(); }
};

/// HOST-resident single-INT32-column batch of @p n_rows rows with the
/// deterministic per-row pattern seed + 7*i — the shape a pinned scan's
/// per-query split arrives in, which the prefetcher upgrades to GPU tier.
/// Built GPU-first then converted down, like the pin path; the GPU staging
/// is fully released before returning so it does not perturb availability.
std::shared_ptr<cucascade::data_batch> make_host_batch(prefetcher_env& e,
                                                       std::size_t n_rows,
                                                       int32_t seed)
{
  auto mr     = e.gpu_space->get_default_allocator();
  auto stream = e.stream();

  std::vector<int32_t> values(n_rows);
  for (std::size_t i = 0; i < n_rows; ++i) {
    values[i] = seed + static_cast<int32_t>(7 * i);
  }

  auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                       static_cast<cudf::size_type>(n_rows),
                                       cudf::mask_state::UNALLOCATED,
                                       stream,
                                       mr);
  cudaMemcpyAsync(col->mutable_view().data<int32_t>(),
                  values.data(),
                  sizeof(int32_t) * n_rows,
                  cudaMemcpyHostToDevice,
                  stream.value());
  stream.synchronize();

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  cucascade::gpu_table_representation gpu_repr(
    std::make_unique<cudf::table>(std::move(cols)), *e.gpu_space, stream);
  auto host_repr = sirius::converter_registry::get().convert<cucascade::host_data_representation>(
    gpu_repr, e.host_space, stream);
  stream.synchronize();
  return cucascade::data_batch::make(sirius::get_next_batch_id(), std::move(host_repr));
}

/// Serves its scripted batches in order, then ends the stream — the minimal
/// provider drain_cached_provider needs to load a connector.
struct scripted_provider final : databatch_provider {
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  std::size_t served{0};

  databatch_provider::batch get_next_batch() override
  {
    databatch_provider::batch out;
    if (served < batches.size()) { out.data = batches[served++]; }
    return out;
  }
};

/// Connector pre-loaded with resident splits over @p batches (and closed —
/// still sweepable: the prefetcher only exits once closed AND drained).
std::shared_ptr<split_connector> make_loaded_connector(
  const std::vector<std::shared_ptr<cucascade::data_batch>>& batches)
{
  auto connector = std::make_shared<split_connector>();
  scripted_provider provider;
  provider.batches = batches;
  std::stop_source stop;
  load_balancing_scan_batch_coalescer::drain_cached_provider(
    provider, *connector, stop.get_token(), /*row_filter_pending=*/false);
  return connector;
}

std::size_t batch_size_bytes(cucascade::data_batch& batch)
{
  auto ro = batch.try_to_read_only();
  REQUIRE(ro);
  REQUIRE(ro->get_data() != nullptr);
  return ro->get_data()->get_size_in_bytes();
}

cucascade::memory::Tier batch_tier(cucascade::data_batch& batch)
{
  auto ro = batch.try_to_read_only();
  REQUIRE(ro);
  return ro->get_current_tier();
}

template <class Pred>
bool wait_until(Pred&& pred, std::chrono::milliseconds timeout)
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (pred()) { return true; }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return pred();
}

/// The gate counters logged once by memory_prefetcher::stop() — the only
/// surface they are readable through (no accessors on the class).
struct stop_counters {
  bool found{false};
  std::size_t headroom{0};
  std::size_t reservation{0};
};

stop_counters parse_stop_counters(const std::vector<sirius::test::recording_log_sink::record>& recs)
{
  static const std::regex kStops(R"(stops: headroom=(\d+) reservation=(\d+))");
  stop_counters out;
  for (auto it = recs.rbegin(); it != recs.rend(); ++it) {
    if (it->message.find("[memory_prefetcher] stopping") == std::string::npos) { continue; }
    std::smatch match;
    if (std::regex_search(it->message, match, kStops)) {
      out.found       = true;
      out.headroom    = std::stoul(match[1]);
      out.reservation = std::stoul(match[2]);
      return out;
    }
  }
  return out;
}

}  // namespace

//===----------------------------------------------------------------------===//
// 1. Admission floor invariant under concurrent workers
//===----------------------------------------------------------------------===//

TEST_CASE("prefetcher admission floor holds against concurrent workers",
          "[memory_prefetcher][scan_manager]")
{
  prefetcher_env e;
  sirius::test::scoped_recording_log_sink log{"info"};

  // Geometry: three 90MB host batches into a 512MB space with a 0.9 floor
  // over the 384MB reservation limit (min_free_bytes = 345.6MB). ONE
  // admission clears the floor (512-90 = 422 >= 345.6); TWO do not
  // (512-180 = 332 < 345.6). The old gate-then-reserve order let all three
  // workers pass the same headroom check concurrently (available 512 >=
  // 90+345.6 for each of them) and convert up to 3 batches; reserve-first
  // makes every worker's floor check see the others' in-flight reservations,
  // so exactly one batch may ever be admitted, on every interleaving.
  // (90MB, not more: the worst concurrent charge — one converted batch plus
  // all three workers' probing reservations, 4 x 90 = 360MB — must stay
  // clearly under the 384MB reservation limit so refusals are always the
  // floor's, never the reservation's.)
  constexpr std::size_t batch_rows = (90ull << 20) / sizeof(int32_t);  // 90MB of int32
  constexpr double min_free_fraction{0.9};

  std::vector<std::shared_ptr<cucascade::data_batch>> batches{make_host_batch(e, batch_rows, 1),
                                                              make_host_batch(e, batch_rows, 2),
                                                              make_host_batch(e, batch_rows, 3)};
  auto connector = make_loaded_connector(batches);

  auto const floor_bytes =
    static_cast<std::size_t>(min_free_fraction * e.gpu_space->get_max_memory());
  auto const avail_before = e.gpu_space->get_available_memory();
  auto const batch_bytes  = batch_size_bytes(*batches[0]);

  // Geometry sanity: this test only means something if exactly one admission
  // fits above the floor.
  REQUIRE(avail_before - batch_bytes >= floor_bytes);
  REQUIRE(avail_before - 2 * batch_bytes < floor_bytes);

  memory_prefetcher_config cfg;
  cfg.enable            = true;
  cfg.num_threads       = 3;  // one worker per queued batch: maximal race exposure
  cfg.min_free_fraction = min_free_fraction;
  cfg.poll_interval_ms  = 1;
  cfg.drain_quiet_ms    = 50;

  stop_counters counters;
  {
    memory_prefetcher prefetcher(cfg, {connector}, e.gpu_space);
    REQUIRE(
      wait_until([&] { return prefetcher.batches_prefetched() >= 1; }, std::chrono::seconds(60)));
    // Grace period: give the other two workers every chance to (incorrectly)
    // admit on the same headroom before we look.
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    prefetcher.stop();

    // The floor admits exactly one batch, no matter how the workers interleave.
    REQUIRE(prefetcher.batches_prefetched() == 1);
    REQUIRE(prefetcher.bytes_prefetched() <= avail_before - floor_bytes);
    counters = parse_stop_counters(log.records());
  }

  // Steady state: the floor held, nothing reserved is left behind.
  REQUIRE(e.gpu_space->get_available_memory() >= floor_bytes);
  REQUIRE(e.gpu_space->get_active_reservation_count() == 0);

  // Exactly one batch reached GPU tier; the other two are still host-resident.
  std::size_t gpu_resident = 0;
  for (auto const& batch : batches) {
    if (batch_tier(*batch) == cucascade::memory::Tier::GPU) { ++gpu_resident; }
  }
  REQUIRE(gpu_resident == 1);

  // Refusals took the headroom path: the peak reservation itself is always
  // admittable in this geometry (worst concurrent charge 4x90 = 360MB, under
  // the 384MB limit), so the workers that lost the race charged a
  // reservation, saw the floor fail post-charge, and released it
  // (_stops_headroom) rather than failing the reservation.
  REQUIRE(counters.found);
  REQUIRE(counters.headroom >= 1);
  REQUIRE(counters.reservation == 0);
}

//===----------------------------------------------------------------------===//
// 2. Reservation draw-down: conversion is charged once, never peak + actual
//===----------------------------------------------------------------------===//

TEST_CASE("prefetcher conversion draws down its reservation instead of double-counting",
          "[memory_prefetcher][scan_manager]")
{
  prefetcher_env e;

  // One 128MB batch, floor low enough to never interfere. With the
  // reservation attached to the worker's allocation tracker, the conversion's
  // device allocations land inside the reserved arena: total charged stays at
  // the 128MB peak throughout. The old decorative reservation charged
  // peak + actual (~256MB) for the whole copy — far below the threshold the
  // sampler enforces here.
  constexpr std::size_t batch_rows = 32ull << 20;  // 128MB of int32
  constexpr std::size_t slack      = 16 * MiB;     // alignment + converter temps

  std::vector<std::shared_ptr<cucascade::data_batch>> batches{make_host_batch(e, batch_rows, 11)};
  auto connector = make_loaded_connector(batches);

  auto const batch_bytes  = batch_size_bytes(*batches[0]);
  auto const avail_before = e.gpu_space->get_available_memory();

  memory_prefetcher_config cfg;
  cfg.enable            = true;
  cfg.num_threads       = 1;
  cfg.min_free_fraction = 0.05;
  cfg.poll_interval_ms  = 1;
  cfg.drain_quiet_ms    = 50;

  // Availability sampler: runs for the entire prefetcher lifetime and records
  // the minimum. The double-count window under the old code spanned the whole
  // H2D copy (milliseconds), so a tight loop cannot miss it; under the fix the
  // minimum provably never goes below avail_before - peak - slack.
  std::atomic<bool> sampling{true};
  std::atomic<std::size_t> min_available{avail_before};
  std::thread sampler([&] {
    while (sampling.load(std::memory_order_relaxed)) {
      auto const now = e.gpu_space->get_available_memory();
      auto prev      = min_available.load(std::memory_order_relaxed);
      while (now < prev &&
             !min_available.compare_exchange_weak(prev, now, std::memory_order_relaxed)) {}
    }
  });

  {
    memory_prefetcher prefetcher(cfg, {connector}, e.gpu_space);
    REQUIRE(
      wait_until([&] { return prefetcher.batches_prefetched() >= 1; }, std::chrono::seconds(60)));
    prefetcher.stop();
    REQUIRE(prefetcher.batches_prefetched() == 1);
    REQUIRE(prefetcher.bytes_prefetched() == batch_bytes);
  }
  sampling.store(false, std::memory_order_relaxed);
  sampler.join();

  // No transient double count: at no point was the space charged more than
  // the admission peak (plus alignment/temp slack). Pre-fix this dipped to
  // roughly avail_before - 2*batch_bytes for the whole conversion.
  REQUIRE(min_available.load() + slack >= avail_before - batch_bytes);

  // Steady state: the space is charged for the converted batch exactly once,
  // and the reservation arena was fully detached and released.
  auto const avail_after = e.gpu_space->get_available_memory();
  REQUIRE(avail_before - avail_after <= batch_bytes + slack);
  REQUIRE(avail_before - avail_after + slack >= batch_bytes);
  REQUIRE(e.gpu_space->get_active_reservation_count() == 0);
  REQUIRE(e.gpu_space->get_total_reserved_memory() == 0);
}

//===----------------------------------------------------------------------===//
// 3. Data-path correctness through the per-worker-stream + tracker-attach path
//===----------------------------------------------------------------------===//

TEST_CASE("prefetched batch converts to GPU tier bit-exactly and reads back on another stream",
          "[memory_prefetcher][scan_manager]")
{
  prefetcher_env e;

  constexpr std::size_t n_rows = 100'000;
  constexpr int32_t seed       = 42;
  std::vector<std::shared_ptr<cucascade::data_batch>> batches{make_host_batch(e, n_rows, seed)};
  auto connector = make_loaded_connector(batches);
  REQUIRE(batch_tier(*batches[0]) == cucascade::memory::Tier::HOST);

  memory_prefetcher_config cfg;
  cfg.enable            = true;
  cfg.num_threads       = 1;
  cfg.min_free_fraction = 0.05;
  cfg.poll_interval_ms  = 1;
  cfg.drain_quiet_ms    = 50;

  {
    memory_prefetcher prefetcher(cfg, {connector}, e.gpu_space);
    REQUIRE(
      wait_until([&] { return prefetcher.batches_prefetched() >= 1; }, std::chrono::seconds(60)));
    prefetcher.stop();
    REQUIRE(prefetcher.batches_prefetched() == 1);
  }

  // The batch was upgraded in place: GPU tier, placed in the target space.
  auto ro = batches[0]->try_to_read_only();
  REQUIRE(ro);
  REQUIRE(ro->get_current_tier() == cucascade::memory::Tier::GPU);
  REQUIRE(ro->get_data() != nullptr);

  // Read the converted table back on a stream the conversion never saw —
  // the conversion synchronized before releasing the batch lock, so the
  // contents must be stable and bit-exact from any stream.
  auto const view = ro->get_data()->cast<cucascade::gpu_table_representation>().get_table_view();
  REQUIRE(view.num_columns() == 1);
  REQUIRE(static_cast<std::size_t>(view.num_rows()) == n_rows);

  rmm::cuda_stream reader_stream;
  std::vector<int32_t> host_values(n_rows);
  cudaMemcpyAsync(host_values.data(),
                  view.column(0).data<int32_t>(),
                  sizeof(int32_t) * n_rows,
                  cudaMemcpyDeviceToHost,
                  reader_stream.value());
  reader_stream.synchronize();

  for (std::size_t i = 0; i < n_rows; ++i) {
    if (host_values[i] != seed + static_cast<int32_t>(7 * i)) {
      FAIL("row " << i << ": expected " << seed + static_cast<int32_t>(7 * i) << ", got "
                  << host_values[i]);
    }
  }
}

//===----------------------------------------------------------------------===//
// 4. Reservation-refused back-off (_stops_reservation), no crash, no leak
//===----------------------------------------------------------------------===//

TEST_CASE("prefetcher backs off cleanly when the peak reservation cannot be admitted",
          "[memory_prefetcher][scan_manager]")
{
  prefetcher_env e;
  sirius::test::scoped_recording_log_sink log{"info"};

  // 360MB of ballast against the 384MB reservation limit: a 32MB peak
  // reservation can never be admitted (360+32 > 384), so every sweep stops on
  // the make_reservation_or_null path (_stops_reservation) before the floor
  // is even consulted. The sweep must keep backing off without converting,
  // crashing, or leaking a reservation arena.
  constexpr std::size_t batch_rows   = 8ull << 20;  // 32MB of int32
  constexpr std::size_t ballast_size = 360 * MiB;

  std::vector<std::shared_ptr<cucascade::data_batch>> batches{make_host_batch(e, batch_rows, 5)};
  auto connector = make_loaded_connector(batches);

  rmm::device_buffer ballast(ballast_size, e.stream(), e.gpu_space->get_default_allocator());
  e.stream().synchronize();
  auto const avail_before = e.gpu_space->get_available_memory();

  memory_prefetcher_config cfg;
  cfg.enable            = true;
  cfg.num_threads       = 1;
  cfg.min_free_fraction = 0.05;
  cfg.poll_interval_ms  = 1;
  cfg.drain_quiet_ms    = 50;

  stop_counters counters;
  {
    memory_prefetcher prefetcher(cfg, {connector}, e.gpu_space);
    // Let it sweep (and be refused) repeatedly.
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    prefetcher.stop();
    REQUIRE(prefetcher.batches_prefetched() == 0);
    REQUIRE(prefetcher.bytes_prefetched() == 0);
    counters = parse_stop_counters(log.records());
  }

  // Refusals took the reservation path, never the (post-charge) headroom path.
  REQUIRE(counters.found);
  REQUIRE(counters.reservation >= 1);
  REQUIRE(counters.headroom == 0);

  // Nothing converted, nothing charged, nothing leaked.
  REQUIRE(batch_tier(*batches[0]) == cucascade::memory::Tier::HOST);
  REQUIRE(e.gpu_space->get_available_memory() == avail_before);
  REQUIRE(e.gpu_space->get_active_reservation_count() == 0);
}
