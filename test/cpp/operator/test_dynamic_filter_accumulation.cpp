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
#include "op/dynamic_filter/detail/accumulation_failure.hpp"
#include "op/dynamic_filter/dynamic_filter_replica_reservation.hpp"
#include "utils/host_allocation_fault.hpp"
#include "utils/sirius_test_env.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/device_buffer.hpp>

#include <catch.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/memory/memory_reservation.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <future>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {

namespace acc      = sirius::test::accumulation;
namespace op       = sirius::op;
using completion   = op::sirius_dynamic_filter_set::completion;
using stats_view   = op::dynamic_filter_stats_snapshot;
constexpr auto mib = acc::mib;

static_assert(std::is_nothrow_constructible_v<op::detail::accumulation_cuda_error,
                                              cudaError_t,
                                              bool,
                                              char const*>);
static_assert(std::is_nothrow_copy_constructible_v<op::detail::accumulation_cuda_error>);
static_assert(
  std::is_nothrow_constructible_v<op::detail::accumulation_invariant_error, char const*>);
static_assert(std::is_nothrow_default_constructible_v<op::detail::unjoined_gpu_work>);
static_assert(std::is_nothrow_copy_constructible_v<op::detail::unjoined_gpu_work>);

/**
 * @brief Bytes one GPU's partials occupy for @p keys keys over @p rows total rows.
 */
std::size_t partial_bytes(std::size_t rows, std::size_t keys)
{
  return acc::bloom_geometry(rows, keys, ~std::uint64_t{0}).arrays_bytes;
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

/**
 * @brief A column of @p values on the current GPU, with row `i` null iff `valid[i]` is false.
 */
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
    auto mask = cudf::create_null_mask(column->size(), cudf::mask_state::UNINITIALIZED, stream);
    REQUIRE(cudaMemcpyAsync(mask.data(),
                            words.data(),
                            words.size() * sizeof(cudf::bitmask_type),
                            cudaMemcpyHostToDevice,
                            stream.get()) == cudaSuccess);
    column->set_null_mask(std::move(mask), nulls);
  }
  stream.sync();
  return column;
}

/**
 * @brief Runs @p bloom's GPU 0 probe over @p probe and returns the mask; the mask must have no
 * nulls.
 */
std::vector<bool> probe_mask(op::sirius_dynamic_bloom_filter const& bloom,
                             cudf::column_view const& probe,
                             ::cuda::stream_ref stream,
                             std::uint32_t const* prior_mask_words = nullptr)
{
  auto mask = acc::probe_membership(
    bloom, probe, 0, stream, cudf::get_current_device_resource_ref(), prior_mask_words);
  REQUIRE(mask);
  return *std::move(mask);
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

  fixture.contribute(batch);
  fixture.contribute(other);
  fixture.publish(0, other->get_batch_id());
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
  SECTION("a zero cap refuses every geometry")
  {
    REQUIRE_FALSE(fixture.begin({batch}, 0));
    REQUIRE(fixture.session->plan().multi_partition_enabled());
    auto const counters = fixture.stats.snapshot();
    REQUIRE(counters.publication_attempts == 0);
    REQUIRE(counters.accumulations_skipped_inventory == 0);
    REQUIRE(counters.accumulations_skipped_admission == 0);
    REQUIRE(counters.keys_skipped_bloom_size_gate == 2);
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
  fixture.contribute(batch);
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
  fixture.contribute(first);

  auto const stream = fixture.task_stream(0);
  acc::stream_gate gate{stream};
  auto source = second->to_read_only();
  fixture.session->contribute(second->get_batch_id(), source, stream);
  cudaEvent_t after{};
  REQUIRE(cudaEventCreateWithFlags(&after, cudaEventDisableTiming) == cudaSuccess);
  REQUIRE(cudaEventRecord(after, stream.get()) == cudaSuccess);
  // contribute() returned while the inserts are still queued behind the gate.
  REQUIRE(cudaEventQuery(after) == cudaErrorNotReady);
  gate.open();
  REQUIRE(cudaEventSynchronize(after) == cudaSuccess);
  REQUIRE(cudaEventDestroy(after) == cudaSuccess);

  fixture.contribute(third);
  fixture.publish(0, third->get_batch_id());
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
  fixture.contribute_queued(first);
  fixture.contribute_queued(second);

  REQUIRE(acc::publish_behind_gates(fixture, 0, second->get_batch_id(), gates));
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
  if (!acc::has_peer_connected_gpus(gpus)) { return; }
  // "sources": GPU 1 contributes, so the root merges it (fold, source-ready wait, merge, egress).
  // "root only": only the root contributes, so every other GPU copies the root's array directly
  // (fold, root-ready wait, egress). With three GPUs, GPU 2 is a target that contributes nothing.
  bool const with_sources = GENERATE(true, false);
  acc::load_accumulation_kernels(static_cast<std::size_t>(gpus));
  acc::fixture fixture({.devices = static_cast<std::size_t>(gpus)});
  auto const settled = fixture.make_batch(with_sources ? 1 : 0, 0, 4000);
  auto const on_root = fixture.make_batch(0, 10'000, 4000);
  auto const queued  = fixture.make_batch(with_sources ? 1 : 0, 20'000, 4000);
  REQUIRE(fixture.begin({settled, on_root, queued}));
  fixture.contribute(settled);

  std::vector<std::unique_ptr<acc::stream_gate>> gates;
  gates.push_back(std::make_unique<acc::stream_gate>(fixture.task_stream(0)));
  if (with_sources) { gates.push_back(std::make_unique<acc::stream_gate>(fixture.task_stream(1))); }
  fixture.contribute_queued(on_root);
  fixture.contribute_queued(queued);

  REQUIRE(acc::publish_behind_gates(fixture, 0, queued->get_batch_id(), gates));
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
  if (!acc::has_peer_connected_gpus(gpus)) { return; }
  // 4'718'593 rows give 9'437'216-byte arrays: five 2 MiB chunks per key with a short last chunk,
  // so two keys cross both scratch buffers several times.
  constexpr cudf::size_type rows_per_batch = 1'179'649;
  auto const batch_rows = [](int batch) { return rows_per_batch - (batch == 3 ? 3 : 0); };

  // The oracle: the same batches accumulated on one GPU, where no transfer is involved. A Bloom
  // union does not depend on insertion order or placement, so every replica must equal it bit for
  // bit.
  std::array<std::vector<std::byte>, 2> expected;
  {
    acc::fixture oracle({.gpu_bytes = 2048 * mib});
    std::vector<acc::batch_ptr> local;
    for (int batch = 0; batch < 4; ++batch) {
      local.push_back(
        oracle.make_batch(0, std::int64_t{batch} * rows_per_batch, batch_rows(batch)));
    }
    REQUIRE(oracle.begin(local));
    for (auto const& batch : local) {
      oracle.contribute(batch);
    }
    oracle.publish(0, local.back()->get_batch_id());
    for (std::size_t key = 0; key < expected.size(); ++key) {
      expected[key] = oracle.replica(key, 0).bytes;
    }
  }

  acc::fixture fixture({.devices = static_cast<std::size_t>(gpus), .gpu_bytes = 2048 * mib});
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
  for (auto const& batch : batches) {
    fixture.contribute(batch);
  }
  auto const root = device_of.back();
  fixture.publish(root, batches.back()->get_batch_id());

  // Visible only after every replica is ready.
  for (std::size_t key = 0; key < 2; ++key) {
    auto filters = acc::filters_on_column(*fixture.channel, key);
    REQUIRE(filters.size() == 1);
    REQUIRE(
      dynamic_cast<op::sirius_dynamic_bloom_filter const&>(*filters.front()).replica_count() ==
      static_cast<std::size_t>(gpus));
  }
  // Every replica equals the single-GPU union, and no partial stream still has work: the
  // publication's one host wait covered every replica.
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
  if (!acc::has_peer_connected_gpus(2)) { return; }
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
  acc::fixture fixture({.devices = 2, .gpu_bytes = 4096 * mib});
  auto const first    = fixture.make_batch(0, 0, shape_case.first_rows);
  auto const second   = fixture.make_batch(1, 100'000'000, shape_case.second_rows);
  auto const extra    = fixture.make_batch(1, 200'000'000, shape_case.extra_rows);
  auto const baseline = fixture.allocated_bytes();
  REQUIRE(fixture.begin({first, second, extra}, shape_case.cap));
  auto const rows = static_cast<std::size_t>(shape_case.first_rows) + shape_case.second_rows +
                    shape_case.extra_rows;
  auto const shape = acc::bloom_geometry(rows, 2, shape_case.cap);
  fixture.contribute(first);
  fixture.contribute(second);
  fixture.contribute(extra);

  // The root (GPU 0) has one source, GPU 1.
  auto const scratch = acc::merge_scratch_bytes(shape);
  auto const spare   = fixture.reservable(0);
  REQUIRE(spare > scratch);
  auto blocker = fixture.gpu(0).make_reservation_or_null(spare - scratch);
  REQUIRE(blocker);
  REQUIRE(fixture.reservable(0) == scratch);
  fixture.publish(0, extra->get_batch_id());
  blocker.reset();

  // The scratch lease is gone with its allocation: the root holds no reservation, its publication
  // stream no tracker, and exactly the published arrays remain.
  REQUIRE(fixture.gpu(0).get_total_reserved_memory() == 0);
  REQUIRE_FALSE(fixture.allocator(0).is_stream_tracked(fixture.publication_stream(0)));
  require_no_tracker(fixture);
  REQUIRE(fixture.allocated_bytes()[0] == baseline[0] + shape.arrays_bytes);
  auto const counters = fixture.stats.snapshot();
  REQUIRE(counters.accumulations_skipped_admission == 0);
  REQUIRE(counters.accumulation_publications_finished == 1);
  REQUIRE(counters.accumulations_skipped_error == 0);
  fixture.require_members({first, second, extra});
}

TEST_CASE("duplicates and retries count once and only the final original ID publishes, once",
          "[dynamic_filter][multi_partition]")
{
  acc::fixture fixture;
  auto const first    = fixture.make_batch(0, 0, 500);
  auto const second   = fixture.make_batch(0, 1000, 500);
  auto const final_id = second->get_batch_id();
  REQUIRE(fixture.begin({first, second}));
  fixture.contribute(first);
  std::size_t duplicates = 0;
  SECTION("duplicates before and replays after publication count once and never publish twice")
  {
    fixture.contribute(first);
    fixture.contribute(second);
    // A retry of the final contributor arrives while its publication is owed: it must publish.
    fixture.contribute(second);
    fixture.publish(0, final_id);
    fixture.contribute(first);
    fixture.contribute(second);
    duplicates = 4;
  }
  SECTION("a rescheduled final contributor publishes once on its retry")
  {
    // The final contribution commits, then its task is rescheduled before it publishes.
    fixture.contribute(second);
    // A retry of another batch while the publication is owed leaves it to the final contributor.
    fixture.contribute(first);
    // Retrying the same original ID preserves its pending publication.
    fixture.contribute(second);
    fixture.publish(0, final_id);
    fixture.publish(0, final_id);
    duplicates = 2;
  }
  SECTION("only the final original ID can consume publication")
  {
    fixture.contribute(second);
    fixture.publish(0, first->get_batch_id());
    REQUIRE(fixture.channel->snapshot().empty());
    REQUIRE_FALSE(fixture.channel->snapshot().terminal());
    fixture.publish(0, final_id);
  }
  fixture.require_members({first, second});
  auto const counters = fixture.stats.snapshot();
  REQUIRE(counters.accumulation_duplicate_contributions == duplicates);
  REQUIRE(counters.accumulation_completed_contributions == 2);
  REQUIRE(counters.accumulation_publications_finished == 1);
  REQUIRE(counters.filters_pushed == 2);
  REQUIRE(fixture.channel->filter_count() == 2);
  REQUIRE(fixture.channel->snapshot().terminal());
}

TEST_CASE("a refused task scratch allocation skips publication before enqueueing anything",
          "[dynamic_filter][multi_partition][mgpu][multi_gpu]")
{
  if (!acc::has_peer_connected_gpus(2)) { return; }
  acc::fixture fixture({.devices = 2});
  auto const first  = fixture.make_batch(0, 0, 4000);
  auto const second = fixture.make_batch(1, 10'000, 4000);
  REQUIRE(fixture.begin({first, second}));
  fixture.contribute(second);
  fixture.contribute(first);
  // The refusing test policy admits one chunk; the two-key merge requires two.
  auto refusing_task = op::detail::scoped_replica_reservation::try_acquire(
    fixture.gpu(0), acc::bloom_geometry(8000).chunk_bytes, fixture.publication_stream(0));
  REQUIRE(refusing_task);
  fixture.publish(0, first->get_batch_id());
  REQUIRE(cudaStreamQuery(fixture.publication_stream(0).get()) == cudaSuccess);
  refusing_task.reset();
  auto const counters = fixture.stats.snapshot();
  REQUIRE(counters.accumulations_skipped_admission == 1);
  REQUIRE(counters.accumulation_publications_finished == 0);
  REQUIRE(counters.filters_pushed == 0);
  REQUIRE(fixture.channel->snapshot().terminal());
  REQUIRE(fixture.channel->snapshot().empty());
}

TEST_CASE("closing or cancelling collecting input never waits for queued inserts",
          "[dynamic_filter][multi_partition]")
{
  bool const cancel = GENERATE(true, false);
  acc::fixture fixture;
  auto const first    = fixture.make_batch(0, 0, 2000);
  auto const second   = fixture.make_batch(0, 5000, 2000);
  auto const third    = fixture.make_batch(0, 9000, 2000);
  auto const baseline = fixture.allocated_bytes();
  REQUIRE(fixture.begin({first, second, third}));
  // An ungated contribution first: lazily loaded insert kernels would otherwise wait behind the
  // gate.
  fixture.contribute(first);

  auto const stream = fixture.task_stream(0);
  acc::stream_gate gate{stream};
  auto source = second->to_read_only();
  fixture.session->contribute(second->get_batch_id(), source, stream);
  cudaEvent_t after{};
  REQUIRE(cudaEventCreateWithFlags(&after, cudaEventDisableTiming) == cudaSuccess);
  REQUIRE(cudaEventRecord(after, stream.get()) == cudaSuccess);
  if (cancel) {
    fixture.session->cancel();
  } else {
    fixture.session->finish_input();
  }
  // Cancellation settles the channel but keeps partials alive until an explicit release.
  REQUIRE_FALSE(gate.timed_out());
  REQUIRE(cudaEventQuery(after) == cudaErrorNotReady);
  REQUIRE(fixture.allocated_bytes()[0] > baseline[0]);
  gate.open();
  REQUIRE(cudaEventSynchronize(after) == cudaSuccess);
  REQUIRE(cudaEventDestroy(after) == cudaSuccess);
  fixture.session->release_retired_storage();
  REQUIRE(fixture.allocated_bytes() == baseline);

  // The released storage is reusable: a fresh accumulation over the same memory is exact.
  fixture.channel = std::make_shared<op::sirius_dynamic_filter_set>();
  REQUIRE(fixture.begin({first, second}));
  fixture.contribute(first);
  fixture.contribute(second);
  fixture.publish(0, second->get_batch_id());
  fixture.require_members({first, second});
  auto const snapshot = fixture.stats.snapshot();
  REQUIRE(snapshot.publications_failed == (cancel ? 1 : 0));
  REQUIRE(snapshot.accumulations_incomplete == (cancel ? 0 : 1));
  REQUIRE(snapshot.accumulation_publications_finished == 1);
}

TEST_CASE("input closure during publication does not retire the builder early",
          "[dynamic_filter][multi_partition][in_flight]")
{
  acc::load_accumulation_kernels(1);
  acc::fixture fixture;
  auto const first  = fixture.make_batch(0, 0, 300);
  auto const second = fixture.make_batch(0, 900, 300);
  REQUIRE(fixture.begin({first, second}));
  fixture.contribute(first);
  // The final inserts stay queued behind the gate, so the publication blocks in its host wait.
  std::vector<std::unique_ptr<acc::stream_gate>> gates;
  gates.push_back(std::make_unique<acc::stream_gate>(fixture.task_stream(0)));
  fixture.contribute_queued(second);
  std::atomic<bool> closed{false};
  auto running = std::async(std::launch::async, [&] {
    fixture.publish(0, second->get_batch_id());
    return closed.load();
  });
  (void)running.wait_for(std::chrono::milliseconds{300});
  REQUIRE(running.wait_for(std::chrono::seconds{0}) == std::future_status::timeout);
  fixture.session->finish_input();
  closed.store(true);
  REQUIRE_FALSE(fixture.channel->snapshot().terminal());
  gates.clear();
  REQUIRE(running.get());
  fixture.task_stream(0).sync();
  REQUIRE(fixture.channel->snapshot().terminal());
  fixture.require_members({first, second});
  REQUIRE(fixture.stats.snapshot().accumulation_publications_finished == 1);
  REQUIRE(fixture.stats.snapshot().accumulations_abandoned == 0);
}

TEST_CASE("a tracked task thread publishes without releasing partial storage",
          "[dynamic_filter][multi_partition]")
{
  acc::fixture fixture;
  auto const first    = fixture.make_batch(0, 0, 300);
  auto const second   = fixture.make_batch(0, 900, 300);
  auto const baseline = fixture.allocated_bytes();
  REQUIRE(fixture.begin({first, second}));
  fixture.contribute(first);
  // Stand in for the final contributor's task: a reservation attached to this thread.
  auto const stream = fixture.task_stream(0);
  auto refusing_task =
    op::detail::scoped_replica_reservation::try_acquire(fixture.gpu(0), mib, stream);
  REQUIRE(refusing_task);
  auto& allocator = fixture.allocator(0);
  REQUIRE(allocator.is_stream_tracked(stream));
  auto const tracked = allocator.get_allocated_bytes(stream);
  fixture.contribute(second);
  fixture.publish(0, second->get_batch_id(), stream);
  // One GPU: no source, no scratch, nothing charged to or credited from the task.
  REQUIRE(allocator.get_allocated_bytes(stream) == tracked);
  REQUIRE(fixture.stats.snapshot().accumulation_publications_finished == 1);
  REQUIRE(fixture.channel->snapshot().terminal());
  fixture.require_members({first, second});
  refusing_task.reset();
  fixture.session->finish_input();
  // The published replicas are the only storage left.
  fixture.require_storage_returned(baseline);
}

TEST_CASE("a tracked task thread charges the merge scratch to its own reservation",
          "[dynamic_filter][multi_partition][mgpu][multi_gpu]")
{
  if (!acc::has_peer_connected_gpus(2)) { return; }
  acc::fixture fixture({.devices = 2});
  auto const first    = fixture.make_batch(1, 0, 4000);
  auto const second   = fixture.make_batch(0, 10'000, 4000);
  auto const baseline = fixture.allocated_bytes();
  REQUIRE(fixture.begin({first, second}));
  fixture.contribute(first);
  // The root (GPU 0) has one source, GPU 1.
  auto const scratch = acc::merge_scratch_bytes(acc::bloom_geometry(8000));

  auto const stream = fixture.task_stream(0);
  auto refusing_task =
    op::detail::scoped_replica_reservation::try_acquire(fixture.gpu(0), mib, stream);
  REQUIRE(refusing_task);
  auto& allocator = fixture.allocator(0);
  auto const peak = allocator.get_peak_allocated_bytes(stream);
  auto const live = allocator.get_allocated_bytes(stream);
  fixture.contribute(second);
  fixture.publish(0, second->get_batch_id(), stream);
  // The scratch was allocated and freed on this thread: its peak is in the task's history and
  // nothing stays charged.
  REQUIRE(allocator.get_allocated_bytes(stream) == live);
  REQUIRE(allocator.get_peak_allocated_bytes(stream) >= peak + scratch);
  REQUIRE(fixture.gpu(0).get_total_reserved_memory() == mib);
  auto const counters = fixture.stats.snapshot();
  REQUIRE(counters.accumulations_skipped_admission == 0);
  REQUIRE(counters.accumulation_publications_finished == 1);
  REQUIRE(counters.accumulations_skipped_error == 0);
  fixture.require_members({first, second});
  refusing_task.reset();
  fixture.session->finish_input();
  fixture.require_storage_returned(baseline);
}

TEST_CASE(
  "a tracked task thread admits a merge scratch larger than its reservation as its task would",
  "[dynamic_filter][multi_partition][mgpu][multi_gpu]")
{
  if (!acc::has_peer_connected_gpus(2)) { return; }
  acc::fixture fixture({.devices = 2});
  // One million rows: one 2,000,128-byte chunk per key, so the root's double-buffered scratch
  // for its one source exceeds a 1 MiB task reservation.
  auto const first    = fixture.make_batch(1, 0, 500'000);
  auto const second   = fixture.make_batch(0, 1'000'000, 500'000);
  auto const baseline = fixture.allocated_bytes();
  REQUIRE(fixture.begin({first, second}));
  fixture.contribute(first);
  auto const scratch = acc::merge_scratch_bytes(acc::bloom_geometry(1'000'000));
  REQUIRE(scratch > mib);
  auto const stream = fixture.task_stream(0);
  auto& allocator   = fixture.allocator(0);

  SECTION("under the engine's limit policy the excess is charged to the GPU's capacity")
  {
    acc::task_reservation admitting_task{fixture.gpu(0), mib, stream};
    REQUIRE(allocator.is_stream_tracked(stream));
    auto const peak = allocator.get_peak_allocated_bytes(stream);
    auto const live = allocator.get_allocated_bytes(stream);
    fixture.contribute(second);
    fixture.publish(0, second->get_batch_id(), stream);
    REQUIRE(allocator.get_allocated_bytes(stream) == live);
    REQUIRE(allocator.get_peak_allocated_bytes(stream) >= peak + scratch);
    auto const counters = fixture.stats.snapshot();
    REQUIRE(counters.accumulations_skipped_admission == 0);
    REQUIRE(counters.accumulations_skipped_error == 0);
    REQUIRE(counters.accumulation_publications_finished == 1);
    fixture.require_members({first, second});
  }
  SECTION(
    "under a limit policy that refuses the excess the publication is skipped before anything is "
    "enqueued")
  {
    auto refusing_task =
      op::detail::scoped_replica_reservation::try_acquire(fixture.gpu(0), mib, stream);
    REQUIRE(refusing_task);
    auto const peak = allocator.get_peak_allocated_bytes(stream);
    auto const live = allocator.get_allocated_bytes(stream);
    fixture.contribute(second);
    fixture.publish(0, second->get_batch_id(), stream);
    REQUIRE(cudaStreamQuery(stream.get()) == cudaSuccess);
    REQUIRE(allocator.get_allocated_bytes(stream) == live);
    REQUIRE(allocator.get_peak_allocated_bytes(stream) == peak);
    auto const counters = fixture.stats.snapshot();
    REQUIRE(counters.accumulations_skipped_admission == 1);
    REQUIRE(counters.accumulations_skipped_error == 0);
    REQUIRE(counters.accumulation_publications_finished == 0);
    REQUIRE(counters.filters_pushed == 0);
    REQUIRE(fixture.channel->snapshot().terminal());
    REQUIRE(fixture.channel->snapshot().empty());
    // The partials stay retired while this thread tracks the root's allocator.
    REQUIRE(fixture.allocated_bytes()[0] > baseline[0] + mib);
  }
  // Once the task's reservation is gone, the untracked settle leaves only what the channel owns.
  fixture.session->finish_input();
  fixture.require_storage_returned(baseline);
}

TEST_CASE("input closed before every batch contributed ends without a filter",
          "[dynamic_filter][multi_partition]")
{
  acc::fixture fixture;
  auto const first    = fixture.make_batch(0, 0, 300);
  auto const second   = fixture.make_batch(0, 900, 300);
  auto const baseline = fixture.allocated_bytes();
  REQUIRE(fixture.begin({first, second}));
  fixture.contribute(first);
  fixture.session->finish_input();
  fixture.contribute(second);
  REQUIRE(fixture.channel->snapshot().terminal());
  REQUIRE(fixture.channel->snapshot().empty());
  fixture.session->release_retired_storage();
  REQUIRE(fixture.allocated_bytes() == baseline);
  REQUIRE(fixture.stats.snapshot().accumulations_incomplete == 1);
}

TEST_CASE("a contribution that does not match its partial's GPU fails the attempt",
          "[dynamic_filter][multi_partition][mgpu][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { return; }
  acc::fixture fixture({.devices = 2});
  auto const first  = fixture.make_batch(0, 0, 300);
  auto const remote = fixture.make_batch(1, 900, 300);
  REQUIRE(fixture.begin({first, remote},
                        fixture.make_plan(256 * mib,
                                          {cudf::data_type{cudf::type_id::INT32},
                                           cudf::data_type{cudf::type_id::INT64}},
                                          {0})));
  SECTION("a batch from a GPU without a partial") { fixture.contribute(remote); }
  SECTION("a batch whose stream belongs to another GPU")
  {
    auto const source = first->to_read_only();
    REQUIRE_THROWS_AS(
      fixture.session->contribute(first->get_batch_id(), source, fixture.task_stream(1)),
      op::detail::accumulation_invariant_error);
    // The device check precedes every enqueue.
    REQUIRE(cudaStreamQuery(fixture.task_stream(1).get()) == cudaSuccess);
  }
  fixture.contribute(first);
  auto const counters = fixture.stats.snapshot();
  REQUIRE(counters.accumulations_skipped_error + counters.accumulations_skipped_admission == 1);
  REQUIRE(counters.accumulations_skipped_inventory == 0);
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
  fixture.contribute(fixture.make_batch(0, 5000, 0));
  fixture.contribute(first);
  fixture.contribute(second);
  fixture.publish(0, second->get_batch_id());
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

TEST_CASE("each accumulation failure kind sets exactly its counters",
          "[dynamic_filter][multi_partition]")
{
  using op::detail::accumulation_failure;
  // The cleanup failure join_on_failure raises when it cannot join submitted GPU work.
  auto const unjoined = [] {
    try {
      throw op::detail::accumulation_cuda_error{cudaErrorIllegalAddress, false, "body"};
    } catch (...) {
      try {
        std::throw_with_nested(op::detail::unjoined_gpu_work{});
      } catch (...) {
        return std::current_exception();
      }
    }
  }();
  struct expectation {
    char const* name;
    std::exception_ptr error;
    accumulation_failure kind;
    bool recoverable;
    std::uint64_t stats_view::* counter;
  };
  auto const cases = std::array{
    expectation{"unjoined GPU work",
                unjoined,
                accumulation_failure::LEAK,
                false,
                &stats_view::accumulation_storage_leaks},
    expectation{"host allocation refusal",
                std::make_exception_ptr(std::bad_alloc{}),
                accumulation_failure::ADMISSION,
                true,
                &stats_view::accumulations_skipped_admission},
    expectation{"retryable kernel launch",
                std::make_exception_ptr(
                  op::detail::accumulation_cuda_error{cudaErrorLaunchOutOfResources, true, "add"}),
                accumulation_failure::TRANSIENT,
                true,
                &stats_view::accumulations_skipped_transient},
    expectation{"fatal CUDA call",
                std::make_exception_ptr(
                  op::detail::accumulation_cuda_error{cudaErrorIllegalAddress, false, "sync"}),
                accumulation_failure::ERROR,
                false,
                &stats_view::accumulations_skipped_error},
    expectation{"invariant violation",
                std::make_exception_ptr(op::detail::accumulation_invariant_error{"invariant"}),
                accumulation_failure::ERROR,
                false,
                &stats_view::accumulations_skipped_error}};
  for (auto const& expected : cases) {
    CAPTURE(expected.name);
    auto const kind = op::detail::classify_failure(expected.error);
    REQUIRE(kind == expected.kind);
    REQUIRE(op::detail::recoverable(kind) == expected.recoverable);
    stats_view counts;
    op::detail::count_failure(counts, kind);
    REQUIRE(counts.*expected.counter == 1);
    // A leak is also an error; every other kind sets only its own counter.
    std::uint64_t set = 0;
    for (auto const field : op::dynamic_filter_counter_fields<std::uint64_t>) {
      set += counts.*field;
    }
    REQUIRE(set == (kind == accumulation_failure::LEAK ? 2 : 1));
    if (kind == accumulation_failure::LEAK) { REQUIRE(counts.accumulations_skipped_error == 1); }
  }
}

namespace {

/**
 * @brief Runs a contribution, or with @p reduction a two-GPU publication, behind a pending CUDA
 * error, then a healthy accumulation on the same thread.
 */
void exercise_pending_cuda_error(bool per_stream, bool reduction)
{
  CAPTURE(per_stream, reduction);
  {
    acc::fixture fixture({.devices = reduction ? 2U : 1U, .per_stream_tracking = per_stream});
    auto const first    = fixture.make_batch(0, 0, 300);
    auto const second   = fixture.make_batch(reduction ? 1 : 0, 900, 300);
    auto const baseline = fixture.allocated_bytes();
    REQUIRE(fixture.begin({first, second}));
    if (reduction) {
      fixture.contribute(first);
      fixture.contribute(second);
    }
    rmm::cuda_set_device_raii device{rmm::cuda_device_id{0}};
    auto const stream = fixture.task_stream(0);
    auto source       = first->to_read_only();
    {
      acc::task_reservation admitting_task{fixture.gpu(0), 8 * mib, stream};
      REQUIRE(cudaGetLastError() == cudaSuccess);
      REQUIRE(cudaMemsetAsync(nullptr, 0, 1, stream.get()) == cudaErrorInvalidValue);
      REQUIRE(cudaPeekAtLastError() == cudaErrorInvalidValue);
      // InvalidValue is recoverable only when the optional kernel's own launch produced it; a
      // pending one belongs to earlier work and must propagate.
      bool propagated = false;
      try {
        if (reduction) {
          fixture.session->publish_if_final(second->get_batch_id(), fixture.gpu(0), stream);
        } else {
          fixture.session->contribute(first->get_batch_id(), source, stream);
        }
      } catch (op::detail::accumulation_cuda_error const& error) {
        propagated = true;
        REQUIRE(error.code() == cudaErrorInvalidValue);
        REQUIRE_FALSE(error.transient_launch_failure());
      }
      REQUIRE(propagated);
      REQUIRE(cudaGetLastError() == cudaSuccess);
      stream.sync();
    }
    auto const counters = fixture.stats.snapshot();
    REQUIRE(counters.accumulations_skipped_error == 1);
    REQUIRE(counters.accumulations_skipped_transient == 0);
    REQUIRE(counters.accumulations_skipped_admission == 0);
    REQUIRE(counters.publications_failed == 1);
    REQUIRE(counters.accumulation_publications_finished == 0);
    REQUIRE(fixture.channel->snapshot().terminal());
    REQUIRE(fixture.channel->snapshot().empty());
    fixture.session->finish_input();
    fixture.session->cancel();
    fixture.session->release_retired_storage();
    REQUIRE(fixture.stats.snapshot().publications_failed == 1);
    REQUIRE(fixture.allocated_bytes() == baseline);
  }
  // The same host thread can run another complete GPU operation after propagation and cleanup.
  acc::fixture healthy({.per_stream_tracking = per_stream});
  auto const first  = healthy.make_batch(0, 0, 300);
  auto const second = healthy.make_batch(0, 900, 300);
  REQUIRE(healthy.begin({first, second}));
  healthy.contribute(first);
  healthy.contribute(second);
  healthy.publish(0, second->get_batch_id());
  healthy.require_members({first, second});
}

}  // namespace

TEST_CASE("a pending CUDA error propagates from contribution", "[dynamic_filter][multi_partition]")
{
  exercise_pending_cuda_error(GENERATE(false, true), false);
}

TEST_CASE("a pending CUDA error propagates from reduction",
          "[dynamic_filter][multi_partition][mgpu][multi_gpu]")
{
  if (!acc::has_peer_connected_gpus(2)) { return; }
  exercise_pending_cuda_error(GENERATE(false, true), true);
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
  fixture.contribute(first);
  fixture.contribute(second);
  fixture.publish(0, second->get_batch_id());
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
  acc::fixture fixture({.per_stream_tracking = per_stream});
  auto const first    = fixture.make_batch(0, 0, 300);
  auto const second   = fixture.make_batch(0, 900, 300);
  auto const baseline = fixture.allocated_bytes();
  REQUIRE(fixture.begin({first, second}));
  auto const partials = fixture.allocated_bytes()[0];

  // Stand in for a task: a reservation attached to this thread (and to the task stream).
  auto const stream = fixture.task_stream(0);
  auto refusing_task =
    op::detail::scoped_replica_reservation::try_acquire(fixture.gpu(0), mib, stream);
  REQUIRE(refusing_task);
  auto& allocator    = fixture.allocator(0);
  auto const tracked = allocator.get_allocated_bytes(stream);
  fixture.session->decline_accumulation();
  REQUIRE(allocator.get_allocated_bytes(stream) == tracked);
  REQUIRE(fixture.allocated_bytes()[0] == partials + mib);
  fixture.session->release_retired_storage();
  REQUIRE(fixture.allocated_bytes()[0] == (per_stream ? baseline[0] : partials) + mib);
  refusing_task.reset();
  fixture.session->finish_input();
  fixture.session->release_retired_storage();
  REQUIRE(fixture.allocated_bytes() == baseline);
  REQUIRE(fixture.stats.snapshot().accumulations_skipped_inventory == 1);
}

TEST_CASE("a refused partial lease on a later GPU rolls back every partial",
          "[dynamic_filter][multi_partition][mgpu][multi_gpu]")
{
  if (!acc::has_peer_connected_gpus(2)) { return; }
  acc::fixture fixture({.devices = 2});
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

TEST_CASE("a publication root without a partial declines and settles",
          "[dynamic_filter][multi_partition][mgpu][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { return; }
  acc::fixture fixture({.devices = 2});
  auto const first    = fixture.make_batch(0, 0, 300);
  auto const second   = fixture.make_batch(0, 900, 300);
  auto const baseline = fixture.allocated_bytes();
  REQUIRE(fixture.begin({first, second},
                        fixture.make_plan(256 * mib,
                                          {cudf::data_type{cudf::type_id::INT32},
                                           cudf::data_type{cudf::type_id::INT64}},
                                          {0})));
  fixture.contribute(first);
  fixture.contribute(second);
  // GPU 1 holds no partial, so it cannot be the publishing root.
  fixture.publish(1, second->get_batch_id());
  auto const counters = fixture.stats.snapshot();
  REQUIRE(counters.accumulations_skipped_inventory == 0);
  REQUIRE(counters.accumulations_skipped_admission == 1);
  REQUIRE(counters.filters_pushed == 0);
  REQUIRE(fixture.channel->snapshot().terminal());
  REQUIRE(fixture.channel->snapshot().empty());
  fixture.session->release_retired_storage();
  REQUIRE(fixture.allocated_bytes() == baseline);
}

TEST_CASE("a publication on another GPU's stream fails before it enqueues anything",
          "[dynamic_filter][multi_partition][mgpu][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { return; }
  acc::fixture fixture({.devices = 2});
  auto const first    = fixture.make_batch(0, 0, 300);
  auto const second   = fixture.make_batch(0, 900, 300);
  auto const baseline = fixture.allocated_bytes();
  REQUIRE(fixture.begin({first, second},
                        fixture.make_plan(256 * mib,
                                          {cudf::data_type{cudf::type_id::INT32},
                                           cudf::data_type{cudf::type_id::INT64}},
                                          {0})));
  fixture.contribute(first);
  fixture.contribute(second);
  // GPU 0, the root, is current, but the stream belongs to GPU 1.
  REQUIRE_THROWS_AS(fixture.publish(0, second->get_batch_id(), fixture.publication_stream(1)),
                    op::detail::accumulation_invariant_error);
  REQUIRE(cudaStreamQuery(fixture.publication_stream(1).get()) == cudaSuccess);
  auto const counters = fixture.stats.snapshot();
  REQUIRE(counters.accumulations_skipped_error == 1);
  REQUIRE(counters.accumulations_skipped_admission == 0);
  REQUIRE(counters.filters_pushed == 0);
  REQUIRE(fixture.channel->snapshot().terminal());
  REQUIRE(fixture.channel->snapshot().empty());
  fixture.session->release_retired_storage();
  REQUIRE(fixture.allocated_bytes() == baseline);
}

TEST_CASE("an owed publication that never runs ends the attempt without a filter",
          "[dynamic_filter][multi_partition]")
{
  acc::fixture fixture;
  auto const first    = fixture.make_batch(0, 0, 300);
  auto const second   = fixture.make_batch(0, 900, 300);
  auto const baseline = fixture.allocated_bytes();
  REQUIRE(fixture.begin({first, second}));
  fixture.contribute(first);
  fixture.contribute(second);
  SECTION("input closes in a session that was not cancelled")
  {
    fixture.session->finish_input();
    REQUIRE(fixture.stats.snapshot().accumulations_abandoned == 1);
  }
  SECTION("the session is cancelled: the attempt counts as cancelled")
  {
    fixture.session->cancel();
    auto const counters = fixture.stats.snapshot();
    REQUIRE(counters.accumulations_abandoned == 0);
    REQUIRE(counters.publications_failed == 1);
  }
  // The owed publication was consumed with the attempt.
  fixture.publish(0, second->get_batch_id());
  REQUIRE(fixture.stats.snapshot().filters_pushed == 0);
  REQUIRE(fixture.channel->snapshot().terminal());
  REQUIRE(fixture.channel->snapshot().empty());
  fixture.session->release_retired_storage();
  REQUIRE(fixture.allocated_bytes() == baseline);
}

TEST_CASE("cancellation during the late wait prevents fan-out", "[dynamic_filter][multi_partition]")
{
  acc::fixture fixture;
  auto const first  = fixture.make_batch(0, 0, 3000);
  auto const second = fixture.make_batch(0, 9000, 3000);
  REQUIRE(fixture.begin({first, second}));
  fixture.contribute(first);

  // Keep the final contribution's inserts queued, so the publication's one host wait blocks on
  // them.
  auto const stream = fixture.task_stream(0);
  auto gate         = std::make_unique<acc::stream_gate>(stream);
  auto source       = second->to_read_only();
  fixture.session->contribute(second->get_batch_id(), source, stream);
  auto running =
    std::async(std::launch::async, [&] { fixture.publish(0, second->get_batch_id()); });
  // The publication stream is idle until finish() enqueues its wait for the queued inserts, which
  // happens after the publication's first cancellation check; cancelling from then on hits the late
  // wait.
  acc::wait_until(
    [&] { return cudaStreamQuery(fixture.publication_stream(0).get()) != cudaSuccess; });
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
    fixture.contribute(first);
    fixture.contribute(second);
    fixture.publish(0, second->get_batch_id());
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
  fixture.contribute(first);
  fixture.contribute(second);
  fixture.publish(0, second->get_batch_id());
  REQUIRE(fixture.stats.snapshot().accumulation_publications_finished == 1);
  auto const& int32_key = fixture.published_bloom(0);
  auto const& int64_key = fixture.published_bloom(1);

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
    fixture.contribute(first);
    fixture.contribute(second);
    fixture.publish(0, second->get_batch_id());
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
  fixture.contribute(first);
  fixture.contribute(second);
  fixture.publish(0, second->get_batch_id());
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

TEST_CASE("a certified build survives payload canonicalization during spill and re-upgrade",
          "[dynamic_filter][multi_partition][inventory][spill]")
{
  acc::fixture fixture;
  rmm::cuda_set_device_raii guard{rmm::cuda_device_id{0}};
  auto const keys     = fixture.make_batch(0, 0, 300);
  auto const key_view = sirius::get_cudf_table_view(*keys);
  auto const stream   = fixture.task_stream(0);
  auto const mr       = fixture.gpu(0).get_default_allocator();
  std::vector<std::unique_ptr<cudf::column>> columns;
  for (auto const& key : key_view) {
    columns.push_back(std::make_unique<cudf::column>(key, stream, mr));
  }
  columns.push_back(cudf::strings::from_integers(key_view.column(0), stream, mr));
  auto const first = sirius::make_data_batch(std::make_unique<cudf::table>(std::move(columns)),
                                             fixture.gpu(0),
                                             stream,
                                             sirius::telemetry::batch_telemetry_info{});
  auto const second_keys = fixture.make_batch(0, 900, 300);
  auto const second_view = sirius::get_cudf_table_view(*second_keys);
  std::vector<std::unique_ptr<cudf::column>> second_columns;
  for (auto const& key : second_view) {
    second_columns.push_back(std::make_unique<cudf::column>(key, stream, mr));
  }
  second_columns.push_back(cudf::strings::from_integers(second_view.column(0), stream, mr));
  auto const second =
    sirius::make_data_batch(std::make_unique<cudf::table>(std::move(second_columns)),
                            fixture.gpu(0),
                            stream,
                            sirius::telemetry::batch_telemetry_info{});
  stream.sync();
  REQUIRE(sirius::get_cudf_table_view(*first).column(2).child(0).type().id() ==
          cudf::type_id::INT32);
  bool const spill_before_certification = GENERATE(true, false);
  if (!spill_before_certification) { REQUIRE(fixture.begin({first, second})); }
  auto* host = fixture.manager->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST).front();
  {
    auto writable = first->to_mutable();
    writable.convert_to<cucascade::host_data_representation>(
      sirius::converter_registry::get(), host, stream);
  }
  stream.sync();
  {
    auto writable = first->to_mutable();
    writable.convert_to<cucascade::gpu_table_representation>(
      sirius::converter_registry::get(), &fixture.gpu(0), stream);
  }
  stream.sync();
  REQUIRE(sirius::get_cudf_table_view(*first).column(2).offset() == 0);
  REQUIRE(sirius::get_cudf_table_view(*first).column(2).child(0).type().id() ==
          cudf::type_id::INT64);
  REQUIRE(sirius::get_cudf_table_view(*second).column(2).child(0).type().id() ==
          cudf::type_id::INT32);
  if (spill_before_certification) { REQUIRE(fixture.begin({first, second})); }
  fixture.contribute(first);
  fixture.contribute(second);
  fixture.publish(0, second->get_batch_id());
  fixture.require_members({first, second});
  REQUIRE(fixture.stats.snapshot().accumulations_skipped_inventory == 0);
}

TEST_CASE("a completed original ID is deduplicated before its changed representation is inspected",
          "[dynamic_filter][multi_partition][inventory]")
{
  acc::fixture fixture;
  auto const first   = fixture.make_batch(0, 0, 300);
  auto const second  = fixture.make_batch(0, 900, 300);
  auto const changed = make_mixed_batch(fixture, cudf::type_id::STRING, 0, 2);
  REQUIRE(fixture.begin({first, second}));
  fixture.contribute(first);
  fixture.contribute(changed, first->get_batch_id());
  fixture.contribute(second);
  fixture.publish(0, second->get_batch_id());
  fixture.require_members({first, second});
  REQUIRE(fixture.stats.snapshot().accumulation_duplicate_contributions == 1);
  REQUIRE(fixture.stats.snapshot().accumulations_skipped_inventory == 0);
}

TEST_CASE("an unclaimed batch with a changed active key or row count declines",
          "[dynamic_filter][multi_partition][inventory]")
{
  acc::fixture fixture;
  auto const first   = fixture.make_batch(0, 0, 300);
  auto const second  = fixture.make_batch(0, 900, 300);
  auto const changed = GENERATE(true, false)
                         ? make_mixed_batch(fixture, cudf::type_id::STRING, 0, 300)
                         : fixture.make_batch(0, 0, 299);
  REQUIRE(fixture.begin({first, second}));
  fixture.contribute(changed, first->get_batch_id());
  REQUIRE(fixture.channel->snapshot().terminal());
  REQUIRE(fixture.channel->snapshot().empty());
  REQUIRE(fixture.stats.snapshot().accumulations_skipped_inventory == 1);
}

TEST_CASE("late cancellation or drain retains completed wrappers outside task accounting",
          "[dynamic_filter][multi_partition][in_flight]")
{
  acc::load_accumulation_kernels(1);
  bool const cancel = GENERATE(true, false);
  acc::fixture fixture;
  auto const first    = fixture.make_batch(0, 0, 300);
  auto const second   = fixture.make_batch(0, 900, 300);
  auto const baseline = fixture.allocated_bytes();
  REQUIRE(fixture.begin({first, second}));
  fixture.contribute(first);
  acc::stream_gate gate{fixture.task_stream(0)};
  fixture.contribute_queued(second);
  auto running = std::async(std::launch::async, [&] {
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{0}};
    auto const stream = fixture.publication_stream(0);
    acc::task_reservation admitting_task{fixture.gpu(0), mib, stream};
    auto& allocator   = fixture.allocator(0);
    auto const before = allocator.get_allocated_bytes(stream);
    fixture.publish(0, second->get_batch_id());
    fixture.session->release_retired_storage();
    return std::pair{before, allocator.get_allocated_bytes(stream)};
  });
  acc::wait_until(
    [&] { return cudaStreamQuery(fixture.publication_stream(0).get()) != cudaSuccess; },
    std::chrono::seconds{5});
  if (cancel) {
    fixture.session->cancel();
  } else {
    fixture.channel->close_for_new_filters();
  }
  gate.open();
  auto const [before, after] = running.get();
  REQUIRE_FALSE(gate.timed_out());
  REQUIRE(before == after);
  REQUIRE(fixture.allocated_bytes()[0] == baseline[0] + partial_bytes(600, 2));
  REQUIRE(fixture.channel->snapshot().terminal());
  REQUIRE(fixture.channel->snapshot().empty());
  fixture.session->release_retired_storage();
  REQUIRE(fixture.allocated_bytes() == baseline);
}

TEST_CASE("inconsistent arrivals omit one key without suppressing its valid sibling",
          "[dynamic_filter][multi_partition][inventory]")
{
  acc::fixture fixture;
  auto const first  = fixture.make_batch(0, 0, 300);
  auto const second = make_mixed_batch(fixture, cudf::type_id::STRING, 900, 300);
  REQUIRE(fixture.begin({first, second}));
  fixture.contribute(first);
  fixture.contribute(second);
  fixture.publish(0, second->get_batch_id());
  REQUIRE(fixture.channel->filter_count() == 1);
  REQUIRE(acc::filters_on_column(*fixture.channel, 0).empty());
  for (auto const& batch : {first, second}) {
    REQUIRE(fixture.possible_rows(1, batch, 0) == 300);
  }
  REQUIRE(fixture.stats.snapshot().keys_skipped_type_mismatch == 1);
  REQUIRE(fixture.stats.snapshot().accumulations_skipped_inventory == 0);
}

TEST_CASE("a key accepted by no channel stays owned until safe retirement",
          "[dynamic_filter][multi_partition][memory]")
{
  bool const per_stream = GENERATE(false, true);
  acc::fixture fixture({.per_stream_tracking = per_stream});
  auto const first    = fixture.make_batch(0, 0, 300);
  auto const second   = fixture.make_batch(0, 900, 300);
  auto const baseline = fixture.allocated_bytes();
  REQUIRE(fixture.begin({first, second}));
  fixture.channel->ignore_columns({1});
  auto const stream = fixture.task_stream(0);
  {
    acc::task_reservation admitting_task{fixture.gpu(0), mib, stream};
    auto& allocator   = fixture.allocator(0);
    auto const before = allocator.get_allocated_bytes(stream);
    fixture.contribute(first);
    fixture.contribute(second);
    fixture.publish(0, second->get_batch_id(), stream);
    REQUIRE(allocator.get_allocated_bytes(stream) == before);
    REQUIRE(fixture.channel->filter_count() == 1);
    REQUIRE(acc::filters_on_column(*fixture.channel, 1).empty());
    REQUIRE(fixture.allocated_bytes()[0] == baseline[0] + mib + partial_bytes(600, 2));
    fixture.session->release_retired_storage();
    REQUIRE(allocator.get_allocated_bytes(stream) == before);
    REQUIRE(fixture.allocated_bytes()[0] ==
            baseline[0] + mib + partial_bytes(600, per_stream ? 1 : 2));
  }
  fixture.session->release_retired_storage();
  REQUIRE(fixture.allocated_bytes()[0] == baseline[0] + partial_bytes(600, 1));
  REQUIRE(fixture.possible_rows(0, first, 0) == 300);
  REQUIRE(fixture.possible_rows(0, second, 0) == 300);
  fixture.require_storage_returned(baseline);
}

namespace {

using sirius::test::scoped_host_allocation_fault;

/**
 * @brief Publishes with filter shell allocation @p ordinal failing (zero: none) and checks every
 * accumulation owner.
 *
 * @return The filter shell allocations the publication made
 */
std::size_t exercise_host_allocation_failure(bool per_stream, std::size_t ordinal)
{
  acc::fixture fixture({.per_stream_tracking = per_stream});
  auto const first    = fixture.make_batch(0, 0, 300);
  auto const second   = fixture.make_batch(0, 900, 300);
  auto const baseline = fixture.allocated_bytes();
  REQUIRE(fixture.begin({first, second}));
  fixture.contribute(first);
  fixture.contribute(second);
  auto const stream    = fixture.task_stream(0);
  std::size_t observed = 0;
  auto const accepted  = ordinal == 0 ? 2U : 0U;
  {
    acc::task_reservation admitting_task{fixture.gpu(0), mib, stream};
    auto& allocator = fixture.allocator(0);
    auto const live = allocator.get_allocated_bytes(stream);
    {
      scoped_host_allocation_fault fault{scoped_host_allocation_fault::filter_shell_scope, ordinal};
      fixture.publish(0, second->get_batch_id(), stream);
      observed = fault.stop();
      REQUIRE(fault.fired() == (ordinal != 0));
    }
    REQUIRE(allocator.get_allocated_bytes(stream) == live);
    REQUIRE(fixture.allocated_bytes()[0] == baseline[0] + mib + partial_bytes(600, 2));
    REQUIRE(fixture.channel->filter_count() == accepted);
    REQUIRE(fixture.channel->snapshot().terminal());
    auto const counters = fixture.stats.snapshot();
    REQUIRE(counters.filters_pushed == accepted);
    REQUIRE(counters.accumulations_skipped_admission == (accepted == 0 ? 1 : 0));
    REQUIRE(counters.accumulations_skipped_error == 0);
    REQUIRE(counters.accumulation_publications_finished == (accepted == 0 ? 0 : 1));
    REQUIRE((counters.accumulation_publication_latency_ns > 0) == (accepted != 0));
    REQUIRE(counters.publications_finished == 1);
    fixture.publish(0, second->get_batch_id(), stream);
    fixture.session->finish_input();
    REQUIRE(fixture.stats.snapshot().publications_finished == 1);
    REQUIRE(fixture.stats.snapshot().accumulation_publications_finished ==
            counters.accumulation_publications_finished);
    REQUIRE(fixture.stats.snapshot().accumulation_publication_latency_ns ==
            counters.accumulation_publication_latency_ns);
    REQUIRE(fixture.channel->filter_count() == accepted);
  }
  fixture.session->release_retired_storage();
  REQUIRE(fixture.allocated_bytes()[0] ==
          baseline[0] + (accepted == 0 ? 0 : partial_bytes(600, accepted)));
  for (std::size_t key = 0; key < accepted; ++key) {
    REQUIRE(fixture.possible_rows(key, first, 0) == 300);
    REQUIRE(fixture.possible_rows(key, second, 0) == 300);
  }
  fixture.require_storage_returned(baseline);
  return observed;
}

}  // namespace

TEST_CASE("host allocation failures preserve every accumulation owner",
          "[.][host_allocation_fault]")
{
  REQUIRE(scoped_host_allocation_fault::available());
  acc::load_accumulation_kernels(1);
  bool const per_stream = GENERATE(false, true);
  SECTION("each filter shell and shared control block allocation")
  {
    auto const allocations = exercise_host_allocation_failure(per_stream, 0);
    REQUIRE(allocations > 2);
    for (std::size_t ordinal = 1; ordinal <= allocations; ++ordinal) {
      CAPTURE(ordinal);
      REQUIRE(exercise_host_allocation_failure(per_stream, ordinal) == ordinal);
    }
  }
}

TEST_CASE("fatal accumulation diagnostics do not allocate during host OOM",
          "[.][host_allocation_fault]")
{
  REQUIRE(scoped_host_allocation_fault::available());
  bool const unjoined     = GENERATE(false, true);
  bool preserved          = false;
  std::size_t allocations = 0;
  {
    scoped_host_allocation_fault fault{scoped_host_allocation_fault::every_allocation_scope, 1};
    try {
      try {
        throw std::bad_alloc{};
      } catch (...) {
        if (unjoined) { std::throw_with_nested(op::detail::unjoined_gpu_work{}); }
        std::throw_with_nested(
          op::detail::accumulation_cuda_error{cudaErrorIllegalAddress, false, "failure cleanup"});
      }
    } catch (op::detail::unjoined_gpu_work const&) {
      preserved = unjoined;
    } catch (op::detail::accumulation_cuda_error const& error) {
      preserved =
        !unjoined && error.code() == cudaErrorIllegalAddress && !error.transient_launch_failure();
    }
    allocations = fault.stop();
    REQUIRE_FALSE(fault.fired());
  }
  REQUIRE(preserved);
  REQUIRE(allocations == 0);
}
