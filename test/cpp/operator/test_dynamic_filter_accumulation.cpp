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
#include "op/dynamic_filter/detail/accumulated_bloom_builder.hpp"
#include "op/dynamic_filter/dynamic_filter_replica_reservation.hpp"
#include "utils/sirius_test_env.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/device_buffer.hpp>

#include <catch.hpp>
#include <cucascade/memory/memory_reservation.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

namespace acc      = sirius::test::accumulation;
namespace op       = sirius::op;
using completion   = op::sirius_dynamic_filter_set::completion;
using stats_view   = op::dynamic_filter_stats_snapshot;
constexpr auto mib = acc::mib;

/// Bytes one GPU's partials occupy for @p keys keys over @p rows total rows.
std::size_t partial_bytes(std::size_t rows, std::size_t keys)
{
  auto const shape =
    op::detail::accumulated_bloom_geometry::try_create(rows, keys, ~std::uint64_t{0});
  REQUIRE(shape);
  return shape->arrays_bytes;
}

void require_no_tracker(acc::fixture const& fixture)
{
  for (std::size_t device = 0; device < fixture.devices; ++device) {
    auto& allocator = fixture.allocator(static_cast<int>(device));
    REQUIRE_FALSE(allocator.is_stream_tracked(fixture.task_stream(static_cast<int>(device))));
    REQUIRE(allocator.get_active_reservation_count() == 0);
  }
}

/**
 * @brief A GPU 0 batch whose first column is the fixture's INT32 key reinterpreted as @p first (a
 * `TIMESTAMP_DAYS` bit cast or a `STRING` rendering) and whose second is the fixture's INT64 key.
 */
acc::batch_ptr make_mixed_batch(acc::fixture const& fixture,
                                cudf::type_id first,
                                std::int64_t start,
                                cudf::size_type rows)
{
  auto const base   = fixture.make_batch(0, start, rows);
  auto const view   = sirius::get_cudf_table_view(*base);
  auto& space       = fixture.gpu(0);
  auto const stream = space.acquire_stream();
  auto const mr     = space.get_default_allocator();
  std::vector<std::unique_ptr<cudf::column>> columns;
  if (first == cudf::type_id::STRING) {
    columns.push_back(cudf::strings::from_integers(view.column(0), stream, mr));
  } else {
    columns.push_back(std::make_unique<cudf::column>(
      cudf::bit_cast(view.column(0), cudf::data_type{first}), stream, mr));
  }
  columns.push_back(std::make_unique<cudf::column>(view.column(1), stream, mr));
  auto table = std::make_unique<cudf::table>(std::move(columns));
  stream.sync();
  return sirius::make_data_batch(
    std::move(table), space, stream, sirius::telemetry::batch_telemetry_info{});
}

/// The accumulated Bloom filter @p fixture published for key @p key.
op::sirius_dynamic_bloom_filter const& published_bloom(acc::fixture const& fixture, std::size_t key)
{
  auto const filters = acc::filters_on_column(*fixture.channel, key);
  REQUIRE(filters.size() == 1);
  auto const* bloom = dynamic_cast<op::sirius_dynamic_bloom_filter const*>(filters.front().get());
  REQUIRE(bloom != nullptr);
  return *bloom;
}

/// A column of @p values on the current GPU, with row `i` null iff `valid[i]` is false.
template <class T>
std::unique_ptr<cudf::column> make_probe(std::vector<T> const& values,
                                         ::cuda::stream_ref stream,
                                         std::vector<bool> const& valid = {})
{
  auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<T>()},
                                          static_cast<cudf::size_type>(values.size()),
                                          cudf::mask_state::UNALLOCATED,
                                          stream);
  REQUIRE(cudaMemcpyAsync(column->mutable_view().data<T>(),
                          values.data(),
                          values.size() * sizeof(T),
                          cudaMemcpyHostToDevice,
                          stream.get()) == cudaSuccess);
  if (!valid.empty()) {
    std::vector<cudf::bitmask_type> words(
      cudf::bitmask_allocation_size_bytes(column->size()) / sizeof(cudf::bitmask_type), 0);
    for (std::size_t row = 0; row < valid.size(); ++row) {
      if (valid[row]) { words[row / 32] |= cudf::bitmask_type{1} << (row % 32); }
    }
    auto const nulls = static_cast<cudf::size_type>(std::ranges::count(valid, false));
    column->set_null_mask(
      rmm::device_buffer{words.data(), words.size() * sizeof(cudf::bitmask_type), stream}, nulls);
  }
  stream.sync();
  return column;
}

/// Runs @p bloom's GPU 0 probe over @p probe and returns the mask; the mask must have no nulls.
std::vector<bool> probe_mask(op::sirius_dynamic_bloom_filter const& bloom,
                             cudf::column_view const& probe,
                             ::cuda::stream_ref stream,
                             std::uint32_t const* prior_mask_words = nullptr)
{
  auto const mask =
    bloom.compute_mask(probe, prior_mask_words, 0, stream, cudf::get_current_device_resource_ref());
  REQUIRE(mask);
  REQUIRE(mask->type().id() == cudf::type_id::BOOL8);
  REQUIRE_FALSE(mask->nullable());
  std::vector<std::uint8_t> host(static_cast<std::size_t>(mask->size()));
  REQUIRE(
    cudaMemcpyAsync(
      host.data(), mask->view().data<bool>(), host.size(), cudaMemcpyDeviceToHost, stream.get()) ==
    cudaSuccess);
  stream.sync();
  return std::vector<bool>(host.begin(), host.end());
}

}  // namespace

TEST_CASE("join_on_failure joins listed streams only when the body throws",
          "[dynamic_filter][multi_partition][join_on_failure]")
{
  rmm::cuda_set_device_raii device{rmm::cuda_device_id{0}};
  rmm::cuda_stream first{rmm::cuda_stream::flags::non_blocking};
  rmm::cuda_stream second{rmm::cuda_stream::flags::non_blocking};
  std::array const streams{::cuda::stream_ref{first.value()}, ::cuda::stream_ref{second.value()}};

  SECTION("a throwing body leaves every listed stream idle at rethrow")
  {
    REQUIRE_THROWS_AS(
      op::detail::join_on_failure(streams,
                                  [&] {
                                    acc::enqueue_delay(streams[0], std::chrono::milliseconds{200});
                                    acc::enqueue_delay(streams[1], std::chrono::milliseconds{100});
                                    throw std::runtime_error{"enqueue failed"};
                                  }),
      std::runtime_error);
    REQUIRE(cudaStreamQuery(first.value()) == cudaSuccess);
    REQUIRE(cudaStreamQuery(second.value()) == cudaSuccess);
  }
  SECTION("a successful body returns while its work still runs")
  {
    acc::stream_gate gate{streams[0]};
    auto const value = op::detail::join_on_failure(streams, [] { return 7; });
    REQUIRE(value == 7);
    REQUIRE(cudaStreamQuery(first.value()) == cudaErrorNotReady);
    gate.open();
    first.synchronize();
  }
}

TEST_CASE("begin allocates zeroed partials under exact per-GPU leases",
          "[dynamic_filter][multi_partition]")
{
  acc::fixture fixture;
  auto const batch  = fixture.make_batch(0, 0, 1000);
  auto const other  = fixture.make_batch(0, 5000, 1000);
  auto const before = fixture.allocated_bytes();
  REQUIRE(fixture.begin({batch, other}));
  REQUIRE(fixture.allocated_bytes()[0] == before[0] + partial_bytes(2000, 2));
  require_no_tracker(fixture);
  REQUIRE(fixture.stats.snapshot().accumulations_started == 1);
  REQUIRE(fixture.stats.snapshot().accumulation_expected_contributions == 2);

  REQUIRE_FALSE(fixture.contribute(batch));
  fixture.run(fixture.contribute(other), 0);
  REQUIRE(fixture.stats.snapshot().accumulation_publications_finished == 1);
  // Only the contributed keys were inserted: a zeroed array rejects almost every other key.
  auto const absent = fixture.make_batch(0, 1'000'000, 4096);
  REQUIRE(fixture.possible_rows(0, absent, 0) < 64);
  REQUIRE(fixture.possible_rows(1, absent, 0) < 64);
  fixture.require_members({batch, other});
  REQUIRE(fixture.stats.snapshot().accumulations_skipped_error == 0);
}

TEST_CASE("accumulation never starts without a usable geometry or inventory",
          "[dynamic_filter][multi_partition]")
{
  acc::fixture fixture;
  auto const batch  = fixture.make_batch(0, 0, 1000);
  auto const before = fixture.allocated_bytes();
  SECTION("a zero cap")
  {
    REQUIRE_FALSE(fixture.begin({batch}, 0));
    REQUIRE(fixture.stats.snapshot().keys_skipped_bloom_size_gate == 2);
  }
  SECTION("a geometry above the cap")
  {
    REQUIRE_FALSE(fixture.begin({batch}, 256));
    REQUIRE(fixture.stats.snapshot().keys_skipped_bloom_size_gate == 2);
  }
  SECTION("no active key")
  {
    REQUIRE_FALSE(fixture.begin({batch},
                                fixture.make_plan(256 * mib,
                                                  {cudf::data_type{cudf::type_id::INT64},
                                                   cudf::data_type{cudf::type_id::INT32}})));
    REQUIRE(fixture.stats.snapshot().keys_skipped_type_mismatch == 2);
  }
  SECTION("zero rows") { REQUIRE_FALSE(fixture.begin({fixture.make_batch(0, 0, 0)})); }
  SECTION("no inventory")
  {
    fixture.session = std::make_unique<op::dynamic_filter_publication_session>(
      fixture.make_plan(256 * mib), &fixture.stats);
    REQUIRE_FALSE(fixture.session->try_begin_accumulation(std::nullopt));
    REQUIRE(fixture.stats.snapshot().accumulations_skipped_inventory == 1);
  }
  REQUIRE_FALSE(fixture.session->accumulation_claimed());
  REQUIRE(fixture.stats.snapshot().accumulations_started == 0);
  REQUIRE(fixture.allocated_bytes() == before);
  REQUIRE_FALSE(fixture.contribute(batch));
  fixture.session->finish_input();
  auto const snapshot = fixture.channel->snapshot();
  REQUIRE(snapshot.terminal());
  REQUIRE(snapshot.empty());
}

TEST_CASE("contributions enqueue without a host wait and publish from another stream",
          "[dynamic_filter][multi_partition]")
{
  acc::fixture fixture;
  auto const first  = fixture.make_batch(0, 0, 2000);
  auto const second = fixture.make_batch(0, 10'000, 2000);
  auto const third  = fixture.make_batch(0, 20'000, 2000);
  REQUIRE(fixture.begin({first, second, third}));
  // An ungated contribution first: lazily loaded insert kernels would otherwise wait behind the
  // gate.
  REQUIRE_FALSE(fixture.contribute(first));

  auto const stream = fixture.task_stream(0);
  acc::stream_gate gate{stream};
  auto source = second->to_read_only();
  REQUIRE_FALSE(fixture.session->contribute(second->get_batch_id(), source, stream));
  cudaEvent_t after{};
  REQUIRE(cudaEventCreateWithFlags(&after, cudaEventDisableTiming) == cudaSuccess);
  REQUIRE(cudaEventRecord(after, stream.get()) == cudaSuccess);
  // contribute() returned while the inserts are still queued behind the gate.
  REQUIRE(cudaEventQuery(after) == cudaErrorNotReady);
  gate.open();
  REQUIRE(cudaEventSynchronize(after) == cudaSuccess);
  REQUIRE(cudaEventDestroy(after) == cudaSuccess);

  auto publishing = fixture.contribute(third);
  REQUIRE(publishing);
  fixture.run(std::move(publishing), 0);
  fixture.require_members({first, second, third});
  auto const counters = fixture.stats.snapshot();
  REQUIRE(counters.accumulation_publications_finished == 1);
  REQUIRE(counters.accumulation_completed_contributions == 3);
  REQUIRE(counters.accumulations_skipped_error == 0);
  REQUIRE(fixture.channel->snapshot().terminal());
}

TEST_CASE("publication waits for contributions still queued on the root's task stream",
          "[dynamic_filter][multi_partition][in_flight]")
{
  acc::load_accumulation_kernels(1);
  acc::fixture fixture;
  auto const first  = fixture.make_batch(0, 0, 3000);
  auto const second = fixture.make_batch(0, 10'000, 3000);
  REQUIRE(fixture.begin({first, second}));
  // Both contributions stay queued behind the gate, as another task's inserts may when the final
  // contribution commits: nothing has host-synchronized them.
  std::vector<std::unique_ptr<acc::stream_gate>> gates;
  gates.push_back(std::make_unique<acc::stream_gate>(fixture.task_stream(0)));
  REQUIRE_FALSE(fixture.contribute_queued(first));
  auto publishing = fixture.contribute_queued(second);
  REQUIRE(publishing);

  REQUIRE(acc::publish_behind_gates(fixture, std::move(publishing), 0, gates));
  fixture.task_stream(0).sync();
  fixture.require_members({first, second});
  auto const counters = fixture.stats.snapshot();
  REQUIRE(counters.accumulation_publications_finished == 1);
  REQUIRE(counters.accumulations_skipped_error == 0);
}

TEST_CASE("publication waits for contributions still queued on every GPU",
          "[dynamic_filter][multi_partition][in_flight][mgpu][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { return; }
  auto const gpus = std::min(acc::visible_gpus(), 3);
  if (!acc::peer_dma_between_first(gpus)) {
    WARN("needs working peer DMA between every pair of GPUs");
    return;
  }
  // "sources": GPU 1 contributes, so the root merges it (fold, source-ready wait, merge, egress).
  // "root only": only the root contributes, so every other GPU copies the root's array directly
  // (fold, root-ready wait, egress). With three GPUs, GPU 2 is a target that contributes nothing.
  bool const with_sources = GENERATE(true, false);
  acc::load_accumulation_kernels(static_cast<std::size_t>(gpus));
  acc::fixture fixture(static_cast<std::size_t>(gpus));
  auto const settled = fixture.make_batch(with_sources ? 1 : 0, 0, 4000);
  auto const on_root = fixture.make_batch(0, 10'000, 4000);
  auto const queued  = fixture.make_batch(with_sources ? 1 : 0, 20'000, 4000);
  REQUIRE(fixture.begin({settled, on_root, queued}));
  REQUIRE_FALSE(fixture.contribute(settled));

  std::vector<std::unique_ptr<acc::stream_gate>> gates;
  gates.push_back(std::make_unique<acc::stream_gate>(fixture.task_stream(0)));
  if (with_sources) { gates.push_back(std::make_unique<acc::stream_gate>(fixture.task_stream(1))); }
  REQUIRE_FALSE(fixture.contribute_queued(on_root));
  auto publishing = fixture.contribute_queued(queued);
  REQUIRE(publishing);

  REQUIRE(acc::publish_behind_gates(fixture, std::move(publishing), 0, gates));
  for (int device = 0; device < gpus; ++device) {
    fixture.task_stream(device).sync();
  }
  fixture.require_members({settled, on_root, queued});
  auto const counters = fixture.stats.snapshot();
  REQUIRE(counters.accumulation_publications_finished == 1);
  REQUIRE(counters.accumulations_skipped_error == 0);
  REQUIRE(counters.accumulations_skipped_admission == 0);
}

TEST_CASE("the chunk pipeline publishes the exact union to every GPU",
          "[dynamic_filter][multi_partition][mgpu][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { return; }
  auto const gpus = std::min(acc::visible_gpus(), 4);
  if (!acc::peer_dma_between_first(gpus)) {
    WARN("needs working peer DMA between every pair of GPUs");
    return;
  }
  // 4'718'593 rows give 9'437'216-byte arrays: five 2 MiB chunks per key with a short last chunk,
  // so two keys cross both scratch buffers several times.
  constexpr cudf::size_type rows_per_batch = 1'179'649;
  auto const batch_rows = [](int batch) { return rows_per_batch - (batch == 3 ? 3 : 0); };

  // The oracle: the same batches accumulated on one GPU, where no transfer is involved. A Bloom
  // union does not depend on insertion order or placement, so every replica must equal it bit for
  // bit.
  std::array<std::vector<std::byte>, 2> expected;
  {
    acc::fixture oracle(1, false, 2048 * mib);
    std::vector<acc::batch_ptr> local;
    for (int batch = 0; batch < 4; ++batch) {
      local.push_back(
        oracle.make_batch(0, std::int64_t{batch} * rows_per_batch, batch_rows(batch)));
    }
    REQUIRE(oracle.begin(local));
    acc::job publishing;
    for (auto const& batch : local) {
      if (auto work = oracle.contribute(batch)) { publishing = std::move(work); }
    }
    REQUIRE(publishing);
    oracle.run(std::move(publishing), 0);
    for (std::size_t key = 0; key < expected.size(); ++key) {
      expected[key] = oracle.replica(key, 0).bytes;
    }
  }

  acc::fixture fixture(static_cast<std::size_t>(gpus), false, 2048 * mib);
  // GPU 0 holds two batches. With more than two GPUs the last GPU holds none, so it is a target
  // that is not a source.
  std::array<int, 4> const device_of{0, 0, 1, gpus > 2 ? 2 : 1};
  std::vector<acc::batch_ptr> batches;
  for (int batch = 0; batch < 4; ++batch) {
    batches.push_back(fixture.make_batch(device_of[static_cast<std::size_t>(batch)],
                                         std::int64_t{batch} * rows_per_batch,
                                         batch_rows(batch)));
  }
  REQUIRE(fixture.begin(batches));
  acc::job publishing;
  for (auto const& batch : batches) {
    auto work = fixture.contribute(batch);
    if (work) { publishing = std::move(work); }
  }
  REQUIRE(publishing);
  auto const root = device_of.back();
  fixture.run(std::move(publishing), root);

  // Visible only after every replica is ready.
  for (std::size_t key = 0; key < 2; ++key) {
    auto filters = acc::filters_on_column(*fixture.channel, key);
    REQUIRE(filters.size() == 1);
    REQUIRE(
      dynamic_cast<op::sirius_dynamic_bloom_filter const&>(*filters.front()).replica_count() ==
      static_cast<std::size_t>(gpus));
  }
  // Every replica equals the single-GPU union, and no partial stream still has work: the job's one
  // host wait covered every replica.
  for (std::size_t key = 0; key < 2; ++key) {
    for (int device = 0; device < gpus; ++device) {
      auto const replica = fixture.replica(key, device);
      REQUIRE(replica.stream_idle);
      REQUIRE(replica.bytes == expected[key]);
    }
  }
  fixture.require_members(batches);
  auto const counters = fixture.stats.snapshot();
  REQUIRE(counters.accumulation_publications_finished == 1);
  REQUIRE(counters.accumulations_skipped_error == 0);
  REQUIRE(counters.accumulations_skipped_admission == 0);
}

TEST_CASE("non-aligned geometries publish with exactly the scratch lease free",
          "[dynamic_filter][multi_partition][mgpu][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { return; }
  if (!acc::peer_dma_between_first(2)) {
    WARN("needs working peer DMA between two GPUs");
    return;
  }
  struct case_shape {
    cudf::size_type first_rows;
    cudf::size_type second_rows;
    cudf::size_type extra_rows;
    std::uint64_t cap;
  };
  // (a) 1'000'001 rows: one chunk of 2'000'128 bytes per key. (b) 20'971'536 rows: 41'943'072-byte
  // arrays in eight chunks with a partial last one, which needs a cap of at least 40 MiB per GPU.
  auto const shape_case = GENERATE(case_shape{500'000, 500'000, 1, 16 * mib},
                                   case_shape{10'485'768, 10'485'768, 0, 96 * mib});
  acc::fixture fixture(2, false, 4096 * mib);
  auto const first    = fixture.make_batch(0, 0, shape_case.first_rows);
  auto const second   = fixture.make_batch(1, 100'000'000, shape_case.second_rows);
  auto const extra    = fixture.make_batch(1, 200'000'000, shape_case.extra_rows);
  auto const baseline = fixture.allocated_bytes();
  REQUIRE(fixture.begin({first, second, extra}, shape_case.cap));
  auto const rows = static_cast<std::size_t>(shape_case.first_rows) + shape_case.second_rows +
                    shape_case.extra_rows;
  auto const shape = op::detail::accumulated_bloom_geometry::try_create(rows, 2, shape_case.cap);
  REQUIRE(shape);
  REQUIRE_FALSE(fixture.contribute(first));
  REQUIRE_FALSE(fixture.contribute(second));
  auto publishing = fixture.contribute(extra);
  REQUIRE(publishing);

  // The root (GPU 0) has one source; its scratch covers min(2, chunks) buffers of one chunk.
  auto const buffers = std::min<std::size_t>(2, 2 * shape->chunks_per_key());
  auto const scratch = buffers * shape->chunk_bytes;
  auto const spare   = fixture.reservable(0);
  REQUIRE(spare > scratch);
  auto blocker = fixture.gpu(0).make_reservation_or_null(spare - scratch);
  REQUIRE(blocker);
  REQUIRE(fixture.reservable(0) == scratch);
  fixture.run(std::move(publishing), 0);
  blocker.reset();

  // The scratch lease is gone with its allocation: the root holds no reservation, its publication
  // stream no tracker, and exactly the published arrays remain.
  REQUIRE(fixture.gpu(0).get_total_reserved_memory() == 0);
  REQUIRE_FALSE(fixture.allocator(0).is_stream_tracked(fixture.publication_stream(0)));
  require_no_tracker(fixture);
  REQUIRE(fixture.allocated_bytes()[0] == baseline[0] + shape->arrays_bytes);
  auto const counters = fixture.stats.snapshot();
  REQUIRE(counters.accumulations_skipped_admission == 0);
  REQUIRE(counters.accumulation_publications_finished == 1);
  REQUIRE(counters.accumulations_skipped_error == 0);
  fixture.require_members({first, second, extra});
}

TEST_CASE("duplicates and replays count once and never publish twice",
          "[dynamic_filter][multi_partition]")
{
  acc::fixture fixture;
  auto const first  = fixture.make_batch(0, 0, 500);
  auto const second = fixture.make_batch(0, 1000, 500);
  REQUIRE(fixture.begin({first, second}));
  REQUIRE_FALSE(fixture.contribute(first));
  REQUIRE_FALSE(fixture.contribute(first));
  auto publishing = fixture.contribute(second);
  REQUIRE(publishing);
  // A retry of the final contributor arrives while the publishing job is pending.
  REQUIRE_FALSE(fixture.contribute(second));
  fixture.run(std::move(publishing), 0);
  REQUIRE_FALSE(fixture.contribute(first));
  auto const counters = fixture.stats.snapshot();
  REQUIRE(counters.accumulation_duplicate_contributions == 3);
  REQUIRE(counters.accumulation_completed_contributions == 2);
  REQUIRE(counters.accumulation_publications_finished == 1);
  REQUIRE(counters.filters_pushed == 2);
  REQUIRE(fixture.channel->filter_count() == 2);
}

TEST_CASE("a refused scratch lease skips publication before enqueueing anything",
          "[dynamic_filter][multi_partition][mgpu][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { return; }
  if (!acc::peer_dma_between_first(2)) {
    WARN("needs working peer DMA between two GPUs");
    return;
  }
  acc::fixture fixture(2);
  auto const first  = fixture.make_batch(0, 0, 4000);
  auto const second = fixture.make_batch(1, 10'000, 4000);
  REQUIRE(fixture.begin({first, second}));
  REQUIRE_FALSE(fixture.contribute(second));
  auto publishing = fixture.contribute(first);
  REQUIRE(publishing);
  auto const shape = op::detail::accumulated_bloom_geometry::try_create(8000, 2, 256 * mib);
  REQUIRE(shape);
  // One contributing source: not even a single chunk of scratch fits.
  auto blocker =
    fixture.gpu(0).make_reservation_or_null(fixture.reservable(0) - shape->chunk_bytes + 1);
  REQUIRE(blocker);
  fixture.run(std::move(publishing), 0);
  REQUIRE(cudaStreamQuery(fixture.publication_stream(0).get()) == cudaSuccess);
  blocker.reset();
  auto const counters = fixture.stats.snapshot();
  REQUIRE(counters.accumulations_skipped_admission == 1);
  REQUIRE(counters.accumulation_publications_finished == 0);
  REQUIRE(counters.filters_pushed == 0);
  REQUIRE(fixture.channel->snapshot().terminal());
  REQUIRE(fixture.channel->snapshot().empty());
}

TEST_CASE("cancel while collecting never waits for queued inserts",
          "[dynamic_filter][multi_partition]")
{
  acc::fixture fixture;
  auto const first    = fixture.make_batch(0, 0, 2000);
  auto const second   = fixture.make_batch(0, 5000, 2000);
  auto const third    = fixture.make_batch(0, 9000, 2000);
  auto const baseline = fixture.allocated_bytes();
  REQUIRE(fixture.begin({first, second, third}));
  // An ungated contribution first: lazily loaded insert kernels would otherwise wait behind the
  // gate.
  REQUIRE_FALSE(fixture.contribute(first));

  auto const stream = fixture.task_stream(0);
  acc::stream_gate gate{stream};
  auto source = second->to_read_only();
  REQUIRE_FALSE(fixture.session->contribute(second->get_batch_id(), source, stream));
  cudaEvent_t after{};
  REQUIRE(cudaEventCreateWithFlags(&after, cudaEventDisableTiming) == cudaSuccess);
  REQUIRE(cudaEventRecord(after, stream.get()) == cudaSuccess);
  fixture.session->cancel();
  // cancel() retired the partials (stream-ordered after the queued inserts) without a host wait.
  // Under compute-sanitizer cudaStreamDestroy waits for queued work, so cancel() returns only after
  // the gate times out; report that instead of hanging.
  REQUIRE_FALSE(gate.timed_out());
  REQUIRE(cudaEventQuery(after) == cudaErrorNotReady);
  REQUIRE(fixture.allocated_bytes() == baseline);
  gate.open();
  REQUIRE(cudaEventSynchronize(after) == cudaSuccess);
  REQUIRE(cudaEventDestroy(after) == cudaSuccess);

  // The released storage is reusable: a fresh accumulation over the same memory is exact.
  fixture.channel = std::make_shared<op::sirius_dynamic_filter_set>();
  REQUIRE(fixture.begin({first, second}));
  REQUIRE_FALSE(fixture.contribute(first));
  fixture.run(fixture.contribute(second), 0);
  fixture.require_members({first, second});
  auto const snapshot = fixture.stats.snapshot();
  REQUIRE(snapshot.publications_failed == 1);
  REQUIRE(snapshot.accumulation_publications_finished == 1);
}

TEST_CASE("input closure during publication does not retire the builder early",
          "[dynamic_filter][multi_partition]")
{
  acc::fixture fixture;
  auto const first  = fixture.make_batch(0, 0, 300);
  auto const second = fixture.make_batch(0, 900, 300);
  REQUIRE(fixture.begin({first, second}));
  REQUIRE_FALSE(fixture.contribute(first));
  auto publishing = fixture.contribute(second);
  REQUIRE(publishing);
  fixture.session->finish_input();
  REQUIRE_FALSE(fixture.channel->snapshot().terminal());
  fixture.run(std::move(publishing), 0);
  REQUIRE(fixture.channel->snapshot().terminal());
  fixture.require_members({first, second});
  REQUIRE(fixture.stats.snapshot().accumulation_publications_finished == 1);
}

TEST_CASE("input closed before every batch contributed ends without a filter",
          "[dynamic_filter][multi_partition]")
{
  acc::fixture fixture;
  auto const first    = fixture.make_batch(0, 0, 300);
  auto const second   = fixture.make_batch(0, 900, 300);
  auto const baseline = fixture.allocated_bytes();
  REQUIRE(fixture.begin({first, second}));
  REQUIRE_FALSE(fixture.contribute(first));
  fixture.session->finish_input();
  REQUIRE_FALSE(fixture.contribute(second));
  REQUIRE(fixture.channel->snapshot().terminal());
  REQUIRE(fixture.channel->snapshot().empty());
  REQUIRE(fixture.allocated_bytes() == baseline);
  REQUIRE(fixture.stats.snapshot().accumulations_incomplete == 1);
}

TEST_CASE("a contribution that does not match its partial's GPU fails the attempt",
          "[dynamic_filter][multi_partition][mgpu][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { return; }
  acc::fixture fixture(2);
  auto const first  = fixture.make_batch(0, 0, 300);
  auto const remote = fixture.make_batch(1, 900, 300);
  REQUIRE(fixture.begin({first, remote},
                        fixture.make_plan(256 * mib,
                                          {cudf::data_type{cudf::type_id::INT32},
                                           cudf::data_type{cudf::type_id::INT64}},
                                          {0})));
  SECTION("a batch from a GPU without a partial") { REQUIRE_FALSE(fixture.contribute(remote)); }
  SECTION("a batch whose stream belongs to another GPU")
  {
    auto const source = first->to_read_only();
    REQUIRE_FALSE(
      fixture.session->contribute(first->get_batch_id(), source, fixture.task_stream(1)));
    // The device check precedes every enqueue.
    REQUIRE(cudaStreamQuery(fixture.task_stream(1).get()) == cudaSuccess);
  }
  REQUIRE_FALSE(fixture.contribute(first));
  auto const counters = fixture.stats.snapshot();
  REQUIRE(counters.accumulations_skipped_error == 1);
  REQUIRE(counters.accumulation_publications_finished == 0);
  REQUIRE(fixture.channel->snapshot().terminal());
  REQUIRE(fixture.channel->snapshot().empty());
}

TEST_CASE("a late batch without rows is ignored by contributions",
          "[dynamic_filter][multi_partition]")
{
  acc::fixture fixture;
  auto const first  = fixture.make_batch(0, 0, 300);
  auto const second = fixture.make_batch(0, 900, 300);
  REQUIRE(fixture.begin({first, second}));
  // `build_arrival_ledger` admits a batch without rows after certification; it is not in the
  // inventory.
  REQUIRE_FALSE(fixture.contribute(fixture.make_batch(0, 5000, 0)));
  REQUIRE_FALSE(fixture.contribute(first));
  fixture.run(fixture.contribute(second), 0);
  fixture.require_members({first, second});
  auto const counters = fixture.stats.snapshot();
  REQUIRE(counters.accumulations_skipped_error == 0);
  REQUIRE(counters.accumulation_duplicate_contributions == 0);
  REQUIRE(counters.accumulation_publications_finished == 1);
}

TEST_CASE("only retryable kernel-launch failures are transient",
          "[dynamic_filter][multi_partition]")
{
  using error = op::detail::accumulation_cuda_error;
  for (auto const code : {cudaErrorLaunchOutOfResources, cudaErrorInvalidValue}) {
    CAPTURE(code);
    REQUIRE(error{code, true, "launch"}.transient_launch_failure());
    REQUIRE_FALSE(error{code, false, "call"}.transient_launch_failure());
  }
  REQUIRE_FALSE(error{cudaErrorMemoryAllocation, true, "launch"}.transient_launch_failure());
  REQUIRE_FALSE(error{cudaErrorIllegalAddress, true, "launch"}.transient_launch_failure());
  REQUIRE(error{cudaErrorInvalidValue, true, "launch"}.code() == cudaErrorInvalidValue);
}

TEST_CASE("a mismatched planned key type is omitted without losing its sibling",
          "[dynamic_filter][multi_partition]")
{
  acc::fixture fixture;
  auto const first  = fixture.make_batch(0, 0, 300);
  auto const second = fixture.make_batch(0, 900, 300);
  REQUIRE(fixture.begin(
    {first, second},
    fixture.make_plan(
      256 * mib, {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT64}})));
  REQUIRE_FALSE(fixture.contribute(first));
  fixture.run(fixture.contribute(second), 0);
  REQUIRE(fixture.stats.snapshot().keys_skipped_type_mismatch == 1);
  REQUIRE(fixture.channel->filter_count() == 1);
  REQUIRE(acc::filters_on_column(*fixture.channel, 0).empty());
  for (auto const& batch : {first, second}) {
    REQUIRE(fixture.possible_rows(1, batch, 0) == 300);
  }
}

TEST_CASE("a tracked thread never releases partial storage", "[dynamic_filter][multi_partition]")
{
  bool const per_stream = GENERATE(false, true);
  acc::fixture fixture(1, per_stream);
  auto const first    = fixture.make_batch(0, 0, 300);
  auto const second   = fixture.make_batch(0, 900, 300);
  auto const baseline = fixture.allocated_bytes();
  REQUIRE(fixture.begin({first, second}));
  auto const partials = fixture.allocated_bytes()[0];

  // Stand in for a task: a reservation attached to this thread (and to the task stream).
  auto const stream = fixture.task_stream(0);
  auto task_lease =
    op::detail::scoped_replica_reservation::try_acquire(fixture.gpu(0), mib, stream);
  REQUIRE(task_lease);
  auto& allocator    = fixture.allocator(0);
  auto const tracked = allocator.get_allocated_bytes(stream);
  auto settle =
    fixture.session->decline_accumulation(op::accumulation_decline::CONTRIBUTION_UNACCOUNTABLE);
  REQUIRE(allocator.get_allocated_bytes(stream) == tracked);
  if (per_stream) {
    // Per-stream tracking: the partial streams are untracked, so the release is immediate and
    // global.
    REQUIRE_FALSE(settle);
    REQUIRE(fixture.allocated_bytes()[0] == baseline[0] + mib);
  } else {
    // Per-thread tracking: the release waits for the settle job on an untracked thread.
    REQUIRE(settle);
    REQUIRE(fixture.allocated_bytes()[0] == partials + mib);
  }
  task_lease.reset();
  settle = {};
  REQUIRE(fixture.allocated_bytes() == baseline);
  REQUIRE(fixture.stats.snapshot().accumulations_skipped_inventory == 1);
}

TEST_CASE("a refused partial lease on a later GPU rolls back every partial",
          "[dynamic_filter][multi_partition][mgpu][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { return; }
  if (!acc::peer_dma_between_first(2)) {
    WARN("needs working peer DMA between two GPUs");
    return;
  }
  acc::fixture fixture(2);
  auto const first    = fixture.make_batch(0, 0, 300'000);
  auto const second   = fixture.make_batch(1, 900'000, 300'000);
  auto const baseline = fixture.allocated_bytes();
  auto blocker =
    fixture.gpu(1).make_reservation_or_null(fixture.reservable(1) - partial_bytes(600'000, 2) + 1);
  REQUIRE(blocker);
  REQUIRE_FALSE(fixture.begin({first, second}));
  blocker.reset();
  REQUIRE(fixture.allocated_bytes() == baseline);
  require_no_tracker(fixture);
  REQUIRE_FALSE(fixture.session->accumulation_claimed());
  REQUIRE(fixture.stats.snapshot().accumulations_skipped_admission == 1);
  fixture.session->finish_input();
  REQUIRE(fixture.channel->snapshot().terminal());
  REQUIRE(fixture.channel->snapshot().empty());
}

TEST_CASE("a failing publishing job is masked, counted, and settles",
          "[dynamic_filter][multi_partition][mgpu][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { return; }
  acc::fixture fixture(2);
  auto const first    = fixture.make_batch(0, 0, 300);
  auto const second   = fixture.make_batch(0, 900, 300);
  auto const baseline = fixture.allocated_bytes();
  REQUIRE(fixture.begin({first, second},
                        fixture.make_plan(256 * mib,
                                          {cudf::data_type{cudf::type_id::INT32},
                                           cudf::data_type{cudf::type_id::INT64}},
                                          {0})));
  REQUIRE_FALSE(fixture.contribute(first));
  auto publishing = fixture.contribute(second);
  REQUIRE(publishing);
  // GPU 1 holds no partial, so it cannot be the publishing root.
  fixture.run(std::move(publishing), 1);
  auto const counters = fixture.stats.snapshot();
  REQUIRE(counters.accumulations_skipped_error == 1);
  REQUIRE(counters.filters_pushed == 0);
  REQUIRE(fixture.channel->snapshot().terminal());
  REQUIRE(fixture.channel->snapshot().empty());
  REQUIRE(fixture.allocated_bytes() == baseline);
}

TEST_CASE("an uninvoked publishing job ends the attempt without a filter",
          "[dynamic_filter][multi_partition]")
{
  acc::fixture fixture;
  auto const first    = fixture.make_batch(0, 0, 300);
  auto const second   = fixture.make_batch(0, 900, 300);
  auto const baseline = fixture.allocated_bytes();
  REQUIRE(fixture.begin({first, second}));
  REQUIRE_FALSE(fixture.contribute(first));
  SECTION("in a session that was not cancelled")
  {
    {
      auto publishing = fixture.contribute(second);
      REQUIRE(publishing);
    }
    REQUIRE(fixture.stats.snapshot().accumulations_abandoned == 1);
  }
  SECTION("after the session was cancelled, the attempt counts as cancelled")
  {
    {
      auto publishing = fixture.contribute(second);
      REQUIRE(publishing);
      fixture.session->cancel();
    }
    auto const counters = fixture.stats.snapshot();
    REQUIRE(counters.accumulations_abandoned == 0);
    REQUIRE(counters.publications_failed == 1);
  }
  REQUIRE(fixture.channel->snapshot().terminal());
  REQUIRE(fixture.channel->snapshot().empty());
  REQUIRE(fixture.allocated_bytes() == baseline);
}

TEST_CASE("cancellation during the late wait prevents fan-out", "[dynamic_filter][multi_partition]")
{
  acc::fixture fixture;
  auto const first  = fixture.make_batch(0, 0, 3000);
  auto const second = fixture.make_batch(0, 9000, 3000);
  REQUIRE(fixture.begin({first, second}));
  REQUIRE_FALSE(fixture.contribute(first));

  // Keep the final contribution's inserts queued, so the job's one host wait blocks on them.
  auto const stream = fixture.task_stream(0);
  auto gate         = std::make_unique<acc::stream_gate>(stream);
  auto source       = second->to_read_only();
  auto publishing   = fixture.session->contribute(second->get_batch_id(), source, stream);
  REQUIRE(publishing);
  auto running = std::async(std::launch::async, [&] { fixture.run(std::move(publishing), 0); });
  // The publication stream is idle until finish() enqueues its wait for the queued inserts, which
  // happens after the job's first cancellation check; cancelling from then on hits the late wait.
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{30};
  while (cudaStreamQuery(fixture.publication_stream(0).get()) == cudaSuccess) {
    REQUIRE(std::chrono::steady_clock::now() < deadline);
    std::this_thread::yield();
  }
  REQUIRE(running.wait_for(std::chrono::seconds{0}) == std::future_status::timeout);
  fixture.session->cancel();
  gate->open();
  running.get();
  stream.sync();
  auto const counters = fixture.stats.snapshot();
  REQUIRE(counters.filters_pushed == 0);
  REQUIRE(counters.publications_failed == 1);
  REQUIRE(counters.accumulations_skipped_error == 0);
  REQUIRE(fixture.channel->snapshot().terminal());
  REQUIRE(fixture.channel->snapshot().empty());
}

TEST_CASE("accumulation declines admitted key types it cannot insert",
          "[dynamic_filter][multi_partition]")
{
  auto const key_type = GENERATE(cudf::type_id::TIMESTAMP_DAYS, cudf::type_id::STRING);
  CAPTURE(key_type);
  acc::fixture fixture;
  auto const first  = make_mixed_batch(fixture, key_type, 0, 300);
  auto const second = make_mixed_batch(fixture, key_type, 900, 300);
  auto const before = fixture.allocated_bytes();
  // The whole-build Bloom filter admits both types; accumulation inserts only INT32 and INT64.
  REQUIRE(op::sirius_dynamic_bloom_filter::supports(cudf::data_type{key_type}));
  REQUIRE_FALSE(op::detail::accumulated_bloom_builder::supports(cudf::data_type{key_type}));

  SECTION("as the only key, nothing starts or allocates")
  {
    REQUIRE_FALSE(
      fixture.begin({first, second}, fixture.make_plan(256 * mib, {cudf::data_type{key_type}})));
    REQUIRE_FALSE(fixture.session->accumulation_claimed());
    REQUIRE(fixture.allocated_bytes() == before);
    auto const counters = fixture.stats.snapshot();
    REQUIRE(counters.keys_skipped_bloom_unsupported == 1);
    REQUIRE(counters.accumulations_started == 0);
  }
  SECTION("beside an INT64 key, only the INT64 key accumulates")
  {
    REQUIRE(fixture.begin(
      {first, second},
      fixture.make_plan(256 * mib,
                        {cudf::data_type{key_type}, cudf::data_type{cudf::type_id::INT64}})));
    REQUIRE(fixture.allocated_bytes()[0] == before[0] + partial_bytes(600, 1));
    REQUIRE_FALSE(fixture.contribute(first));
    fixture.run(fixture.contribute(second), 0);
    auto const counters = fixture.stats.snapshot();
    REQUIRE(counters.keys_skipped_bloom_unsupported == 1);
    REQUIRE(counters.accumulation_publications_finished == 1);
    REQUIRE(acc::filters_on_column(*fixture.channel, 0).empty());
    for (auto const& batch : {first, second}) {
      REQUIRE(fixture.possible_rows(1, batch, 0) == 300);
    }
  }
  REQUIRE(fixture.stats.snapshot().accumulations_skipped_error == 0);
}

TEST_CASE("an accumulated filter probes like every other membership filter",
          "[dynamic_filter][multi_partition]")
{
  acc::fixture fixture;
  // Key 0 (INT32) holds 0..299 and 900..1199; key 1 (INT64) holds 3k + 1 for the same k.
  auto const first  = fixture.make_batch(0, 0, 300);
  auto const second = fixture.make_batch(0, 900, 300);
  REQUIRE(fixture.begin({first, second}));
  REQUIRE_FALSE(fixture.contribute(first));
  fixture.run(fixture.contribute(second), 0);
  REQUIRE(fixture.stats.snapshot().accumulation_publications_finished == 1);
  auto const& int32_key = published_bloom(fixture, 0);
  auto const& int64_key = published_bloom(fixture, 1);

  rmm::cuda_set_device_raii guard{rmm::cuda_device_id{0}};
  auto const stream = fixture.task_stream(0);

  // Each probe also holds absent values inside and outside the build's range, so a probe that keeps
  // every row fails. Bloom false positives are deterministic, so these stay absent.
  SECTION("an INT64 key reads an INT32 probe")
  {
    auto const probe =
      make_probe<std::int32_t>({1, 4, 898, 2701, 3598, 2, 901, 2000, -1, 1'000'000}, stream);
    REQUIRE(probe_mask(int64_key, probe->view(), stream) ==
            std::vector<bool>{true, true, true, true, true, false, false, false, false, false});
  }
  SECTION("an INT32 key reads an INT64 probe and rejects values outside INT32")
  {
    constexpr auto above = std::int64_t{std::numeric_limits<std::int32_t>::max()} + 1;
    constexpr auto below = std::int64_t{std::numeric_limits<std::int32_t>::min()} - 1;
    auto const probe     = make_probe<std::int64_t>(
      {0, 299, 900, 1199, above, below, std::int64_t{1} << 40, 300, 600, -1, 5000}, stream);
    REQUIRE(
      probe_mask(int32_key, probe->view(), stream) ==
      std::vector<bool>{true, true, true, true, false, false, false, false, false, false, false});
  }
  SECTION("rows the prior mask dropped stay dropped")
  {
    auto const probe         = make_probe<std::int32_t>({0, 1, 2, 3}, stream);
    std::uint32_t const keep = 0b0101;
    rmm::device_buffer const words{&keep, sizeof(keep), stream};
    REQUIRE(probe_mask(
              int32_key, probe->view(), stream, static_cast<std::uint32_t const*>(words.data())) ==
            std::vector<bool>{true, false, true, false});
  }
  SECTION("null probe rows are non-members in a mask without nulls")
  {
    auto const probe =
      make_probe<std::int32_t>({0, 1, 2, 3}, stream, std::vector<bool>{true, false, true, false});
    REQUIRE(probe_mask(int32_key, probe->view(), stream) ==
            std::vector<bool>{true, false, true, false});
  }
}

TEST_CASE("accumulation skips a key none of whose bindings can read its filter",
          "[dynamic_filter][multi_partition]")
{
  acc::fixture fixture;
  auto const first  = fixture.make_batch(0, 0, 300);
  auto const second = fixture.make_batch(0, 900, 300);
  auto const before = fixture.allocated_bytes();
  constexpr cudf::data_type int32{cudf::type_id::INT32};
  constexpr cudf::data_type int64{cudf::type_id::INT64};
  constexpr cudf::data_type float64{cudf::type_id::FLOAT64};

  SECTION("as the only key, nothing starts or allocates")
  {
    REQUIRE_FALSE(
      fixture.begin({first, second}, fixture.make_plan(256 * mib, {int32}, {}, {float64})));
    REQUIRE_FALSE(fixture.session->accumulation_claimed());
    REQUIRE(fixture.allocated_bytes() == before);
    auto const counters = fixture.stats.snapshot();
    REQUIRE(counters.bindings_skipped_incompatible_probe == 1);
    REQUIRE(counters.accumulations_started == 0);
  }
  SECTION("beside a readable key, only the readable key accumulates")
  {
    // Key 0 is probed at FLOAT64, which no integer key domain reads; key 1 is probed at a narrower
    // integer carrier, which its domain reads.
    REQUIRE(fixture.begin({first, second},
                          fixture.make_plan(256 * mib, {int32, int64}, {}, {float64, int32})));
    REQUIRE(fixture.allocated_bytes()[0] == before[0] + partial_bytes(600, 1));
    REQUIRE(fixture.stats.snapshot().bindings_skipped_incompatible_probe == 1);
    REQUIRE_FALSE(fixture.contribute(first));
    fixture.run(fixture.contribute(second), 0);
    auto const counters = fixture.stats.snapshot();
    REQUIRE(counters.bindings_skipped_incompatible_probe == 1);
    REQUIRE(counters.filters_pushed == 1);
    REQUIRE(counters.accumulation_publications_finished == 1);
    REQUIRE(acc::filters_on_column(*fixture.channel, 0).empty());
    for (auto const& batch : {first, second}) {
      REQUIRE(fixture.possible_rows(1, batch, 0) == 300);
    }
  }
  SECTION("a binding without a recorded probe type is left to the runtime")
  {
    REQUIRE(fixture.begin(
      {first, second},
      fixture.make_plan(256 * mib, {int32}, {}, {cudf::data_type{cudf::type_id::EMPTY}})));
    REQUIRE(fixture.allocated_bytes()[0] == before[0] + partial_bytes(600, 1));
    REQUIRE(fixture.stats.snapshot().bindings_skipped_incompatible_probe == 0);
  }
  REQUIRE(fixture.stats.snapshot().accumulations_skipped_error == 0);
}

TEST_CASE("an accumulated fan-out skips bindings whose probe type the key cannot read",
          "[dynamic_filter][multi_partition]")
{
  using plan_type = op::dynamic_filter_publish_plan;
  constexpr cudf::data_type int32{cudf::type_id::INT32};
  constexpr cudf::data_type int64{cudf::type_id::INT64};
  acc::fixture fixture;
  auto const first  = fixture.make_batch(0, 0, 300);
  auto const second = fixture.make_batch(0, 900, 300);
  auto const before = fixture.allocated_bytes();
  // Key 0 is readable on the fixture's channel, so it accumulates, but `other` probes it at
  // FLOAT64, which its domain cannot read; key 1 is readable on the fixture's channel only.
  auto const other = std::make_shared<op::sirius_dynamic_filter_set>();
  std::vector<plan_type::probe_target> targets;
  targets.push_back({.filter_set               = fixture.channel,
                     .route_class              = op::dynamic_filter_route_class::scan,
                     .accepts_zone_map_filters = false,
                     .key_bindings             = {{0, 0, int32}, {1, 1, int32}}});
  targets.push_back({.filter_set               = other,
                     .route_class              = op::dynamic_filter_route_class::scan,
                     .accepts_zone_map_filters = false,
                     .key_bindings = {{0, 0, cudf::data_type{cudf::type_id::FLOAT64}}}});
  plan_type plan{{{.planner_condition_index = 0, .build_key_ordinal = 0, .storage_type = int32},
                  {.planner_condition_index = 1, .build_key_ordinal = 1, .storage_type = int64}},
                 std::move(targets),
                 fixture.spaces,
                 {.enable_multi_partition = true, .max_bloom_bytes_per_gpu = 256 * mib}};
  REQUIRE(fixture.begin({first, second}, std::move(plan)));
  REQUIRE(fixture.allocated_bytes()[0] == before[0] + partial_bytes(600, 2));
  REQUIRE(fixture.stats.snapshot().bindings_skipped_incompatible_probe == 0);
  REQUIRE_FALSE(fixture.contribute(first));
  fixture.run(fixture.contribute(second), 0);
  auto const counters = fixture.stats.snapshot();
  REQUIRE(counters.bindings_skipped_incompatible_probe == 1);
  REQUIRE(counters.filters_pushed == 2);
  REQUIRE(counters.accumulation_publications_finished == 1);
  REQUIRE(counters.accumulations_skipped_error == 0);
  REQUIRE(acc::filters_on_column(*other, 0).empty());
  REQUIRE(other->snapshot().terminal());
  for (auto const& batch : {first, second}) {
    REQUIRE(fixture.possible_rows(0, batch, 0) == 300);
    REQUIRE(fixture.possible_rows(1, batch, 0) == 300);
  }
  REQUIRE(fixture.channel->snapshot().terminal());
}
