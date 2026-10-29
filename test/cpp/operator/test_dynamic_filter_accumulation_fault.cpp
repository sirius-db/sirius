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

#include "dynamic_filter_accumulation_test_utils.hpp"
#include "op/dynamic_filter/dynamic_filter_replica_reservation.hpp"
#include "utils/cuda_status_fault.hpp"

#include <cucascade/error.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <array>
#include <chrono>

namespace {
namespace acc = sirius::test::accumulation;
using fault   = sirius::test::scoped_cuda_status_fault;

bool has_two_gpus()
{
  int count = 0;
  REQUIRE(cudaGetDeviceCount(&count) == cudaSuccess);
  if (count < 2) {
    WARN("accumulation publication test needs 2 GPUs, " << count << " visible; skipping");
    return false;
  }
  return true;
}

struct failure_point {
  fault::operation operation;
  std::size_t skip;
  bool staged;
};
}  // namespace

TEST_CASE("accumulated publication faults drain all registered streams and retain the CUDA cause",
          "[dynamic_filter][accumulation_fault][multi_gpu]")
{
  if (!has_two_gpus()) { return; }
  auto const point = GENERATE(failure_point{fault::copy_peer, 1, false},
                              failure_point{fault::copy_peer, 4, false},
                              failure_point{fault::copy_peer, 6, false},
                              failure_point{fault::copy_peer, 11, false},
                              failure_point{fault::record_event, 4, false},
                              failure_point{fault::synchronize_stream_unjoined, 0, false},
                              failure_point{fault::enqueue_copy_batch, 1, true},
                              failure_point{fault::enqueue_copy_batch, 3, true},
                              failure_point{fault::enqueue_copy_batch, 12, true},
                              failure_point{fault::synchronize_stream_unjoined, 5, true});
  CAPTURE(point.operation, point.skip, point.staged);
  acc::fixture fixture({.devices = 2});
  if (!point.staged && !acc::has_pool_access(fixture.spaces)) { return; }
  auto const geometry = acc::bloom_geometry(2'097'153);
  auto builder        = acc::make_builder(fixture, geometry);
  auto first          = fixture.make_batch(0, 0, 100);
  auto second         = fixture.make_batch(1, 1000, 100);
  acc::add_to_builder(builder, fixture, first);
  acc::add_to_builder(builder, fixture, second);
  auto const before = fixture.allocated_bytes();
  auto* host        = fixture.spaces[0]
                 .get_host_staging_space()
                 .get_memory_resource_of<cucascade::memory::Tier::HOST>();
  auto const host_before = host->get_total_allocated_bytes();
  auto const start       = std::chrono::steady_clock::now();
  {
    fault injection{
      point.operation, cudaErrorInvalidValue, point.skip, nullptr, std::chrono::milliseconds{80}};
    try {
      (void)acc::builder_access::finish(builder,
                                        rmm::cuda_device_id{0},
                                        fixture.publication_stream(0),
                                        fixture.gpu(0).get_default_allocator(),
                                        point.staged);
      FAIL("the publication CUDA failure must escape");
    } catch (sirius::op::detail::accumulation_cuda_error const& error) {
      REQUIRE(error.code() == cudaErrorInvalidValue);
      REQUIRE_FALSE(error.transient_launch_failure());
    }
    REQUIRE(injection.consumed());
    REQUIRE(injection.observed_stream_count() > 0);
    REQUIRE_FALSE(injection.invalid_join());
    REQUIRE(injection.all_submitted_streams_joined());
  }
  REQUIRE(std::chrono::steady_clock::now() - start >= std::chrono::milliseconds{70});
  REQUIRE(host->get_total_allocated_bytes() == host_before);
  REQUIRE(fixture.allocated_bytes() == before);
  for (int device = 0; device < 2; ++device) {
    auto const partial = builder.inspect_partial(0, rmm::cuda_device_id{device});
    REQUIRE(partial);
    REQUIRE(partial->stream_idle);
  }
}

TEST_CASE("accumulation preparation event failure preserves partials and retires scratch",
          "[dynamic_filter][accumulation_fault][multi_gpu]")
{
  if (!has_two_gpus()) { return; }
  auto const skip = GENERATE(std::size_t{0}, std::size_t{6});
  acc::fixture fixture({.devices = 2});
  if (!acc::has_pool_access(fixture.spaces)) { return; }
  auto builder = acc::make_builder(fixture, acc::bloom_geometry(8000));
  auto batch   = fixture.make_batch(1, 0, 100);
  acc::add_to_builder(builder, fixture, batch);
  auto const before   = fixture.allocated_bytes();
  auto const expected = acc::partial_union(builder, 2, 2);
  {
    // Six prior events place the late failure at the last target-completion event.
    fault injection{fault::create_event, cudaErrorInvalidValue, skip};
    REQUIRE_THROWS_AS(builder.finish(rmm::cuda_device_id{0},
                                     fixture.publication_stream(0),
                                     fixture.gpu(0).get_default_allocator()),
                      cucascade::cuda_error);
    REQUIRE(injection.consumed());
    REQUIRE(injection.observed_stream_count() == 0);
    REQUIRE_FALSE(injection.invalid_join());
    REQUIRE(injection.all_submitted_streams_joined());
  }
  REQUIRE(fixture.allocated_bytes() == before);
  REQUIRE(acc::partial_union(builder, 2, 2) == expected);
}

TEST_CASE(
  "delayed accumulation ingress merge and egress preserve every bitmap byte across halves and keys",
  "[dynamic_filter][accumulation_fault][multi_gpu]")
{
  if (!has_two_gpus()) { return; }
  auto const point = GENERATE(failure_point{fault::copy_peer, 0, false},
                              failure_point{fault::copy_peer, 1, false},
                              failure_point{fault::copy_peer, 4, false},
                              failure_point{fault::copy_peer, 6, false},
                              failure_point{fault::record_event, 2, false},
                              failure_point{fault::enqueue_copy_batch, 1, true},
                              failure_point{fault::enqueue_copy_batch, 3, true});
  CAPTURE(point.operation, point.skip, point.staged);
  acc::fixture fixture({.devices = 2});
  if (!point.staged && !acc::has_pool_access(fixture.spaces)) { return; }
  auto builder = acc::make_builder(fixture, acc::bloom_geometry(2'097'153));
  auto first   = fixture.make_batch(0, 0, 100);
  auto second  = fixture.make_batch(1, 1000, 100);
  acc::add_to_builder(builder, fixture, first);
  acc::add_to_builder(builder, fixture, second);
  auto const expected = acc::partial_union(builder, 2, 2);
  std::optional<sirius::op::detail::accumulated_bloom_builder::publication> published;
  {
    fault injection{
      point.operation, cudaSuccess, point.skip, nullptr, std::chrono::milliseconds{80}};
    published = acc::builder_access::finish(builder,
                                            rmm::cuda_device_id{0},
                                            fixture.publication_stream(0),
                                            fixture.gpu(0).get_default_allocator(),
                                            point.staged);
    REQUIRE(injection.consumed());
    REQUIRE_FALSE(injection.invalid_join());
  }
  REQUIRE(published);
  using transport = sirius::op::detail::accumulated_bloom_builder::publication_transport;
  REQUIRE(published->transport == (point.staged ? transport::host_staged : transport::peer_copy));
  acc::require_exact_union(published->filters, expected, 2);
}

TEST_CASE("HOST exhaustion refuses preparation without changing any partial",
          "[dynamic_filter][multi_partition][accumulation_fault][multi_gpu]")
{
  if (!has_two_gpus()) { return; }
  acc::fixture fixture({.devices = 2});
  auto const geometry = acc::bloom_geometry(8000);
  auto builder        = acc::make_builder(fixture, geometry);
  auto batch          = fixture.make_batch(1, 0, 100);
  acc::add_to_builder(builder, fixture, batch);
  std::array<std::array<std::vector<std::byte>, 2>, 2> before;
  for (std::size_t key = 0; key < 2; ++key) {
    for (int device = 0; device < 2; ++device) {
      before[key][device] = builder.inspect_partial(key, rmm::cuda_device_id{device})->bytes;
    }
  }
  auto* host = fixture.spaces[0]
                 .get_host_staging_space()
                 .get_memory_resource_of<cucascade::memory::Tier::HOST>();
  auto blocker = host->reserve_upto(host->get_available_memory());
  REQUIRE(blocker);
  auto remaining         = host->allocate_multiple_blocks(host->get_available_memory());
  auto const baseline    = fixture.allocated_bytes();
  auto const host_before = host->get_total_allocated_bytes();
  acc::builder_access::workspace_observation workspace;
  {
    fault injection{fault::enqueue_copy_batch, cudaErrorInvalidValue};
    auto result = acc::builder_access::finish(builder,
                                              rmm::cuda_device_id{0},
                                              fixture.publication_stream(0),
                                              fixture.gpu(0).get_default_allocator(),
                                              true,
                                              &workspace);
    REQUIRE_FALSE(result);
    REQUIRE_FALSE(injection.consumed());
    REQUIRE(injection.observed_stream_count() == 0);
    REQUIRE_FALSE(injection.invalid_join());
  }
  REQUIRE(workspace.host_staging_bytes == 0);
  REQUIRE(workspace.device_scratch_bytes ==
          sirius::op::detail::tracked_replica_allocation_bytes(geometry.chunk_bytes));
  REQUIRE(host->get_total_allocated_bytes() == host_before);
  REQUIRE(fixture.allocated_bytes() == baseline);
  REQUIRE(cudaStreamQuery(fixture.publication_stream(0).get()) == cudaSuccess);
  for (std::size_t key = 0; key < 2; ++key) {
    for (int device = 0; device < 2; ++device) {
      auto const partial = builder.inspect_partial(key, rmm::cuda_device_id{device});
      REQUIRE(partial);
      REQUIRE(partial->stream_idle);
      REQUIRE(partial->bytes == before[key][device]);
    }
  }
}
