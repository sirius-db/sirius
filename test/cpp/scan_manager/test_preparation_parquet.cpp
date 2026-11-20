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

#include "catch.hpp"
#include "op/scan/parquet_gpu_ingestible.hpp"
#include "scan_manager/preparation_fixture.hpp"

namespace {
namespace scan = sirius::op::scan;
using namespace sirius::scan_manager;
using namespace sirius::test::preparation;
using namespace std::chrono_literals;

constexpr size_t cap         = size_t{64} << 20;
constexpr size_t input_bytes = 26825000;

// Delegate all sizing and batching to production; replace only its retention clock.
struct clocked_parquet_coalescer : scan::batch_coalescer {
  std::unique_ptr<scan::batch_coalescer> inner;
  std::function<clock::time_point()> now;
  std::function<void()> after_advance;
  size_t target_batch_bytes() const noexcept override { return inner->target_batch_bytes(); }
  size_t retained_bytes() const noexcept override { return inner->retained_bytes(); }
  void update_time()
  {
    if (inner->first_retained_time())
      note_retained(now());
    else
      clear_retained();
  }
  cursor_step advance(scan::scan_info& input, size_t& cursor, size_t quantum) override
  {
    auto step = inner->advance(input, cursor, quantum);
    update_time();
    if (step.finished && after_advance) after_advance();
    return step;
  }
  std::unique_ptr<scan::scan_info> partial_emit() override
  {
    auto batch = inner->partial_emit();
    update_time();
    return batch;
  }
  std::vector<std::unique_ptr<scan::scan_info>> push(
    std::unique_ptr<scan::scan_info> input) override
  {
    auto batches = inner->push(std::move(input));
    update_time();
    return batches;
  }
  std::vector<std::unique_ptr<scan::scan_info>> flush() override
  {
    auto batches = inner->flush();
    update_time();
    return batches;
  }
};

struct parquet_preparation : fixture {
  struct observation {
    size_t preparation, execution, files;
    bool positions;
  };
  std::shared_ptr<sink> out = std::make_shared<sink>();
  std::vector<observation> observed;  // Protected by out->mutex.
  std::atomic<int> milliseconds{100};
  std::atomic<int> available{1};
  std::shared_ptr<hold> retained = make_hold();
  std::shared_ptr<clocked_parquet_coalescer> coalescer =
    std::make_shared<clocked_parquet_coalescer>();
  std::unique_ptr<scan::parquet_gpu_ingestible> reader;

  static preparation_options options(size_t quantum)
  {
    auto result              = preparation_config{}.resolve(1);
    result.lookahead_batches = 1;
    result.drain_quantum     = quantum;
    result.partial_batch     = partial_batch_options{};
    return result;
  }
  explicit parquet_preparation(size_t quantum)
    : fixture(1, 1, k_interrupt_check_interval, options(quantum))
  {
    auto info                    = std::make_unique<scan::parquet_ingestible_table_info>();
    info->names                  = {"x"};
    info->returned_types         = {sirius::logical_type::make(sirius::type_id::INTEGER)};
    info->column_ids             = {duckdb::ColumnIndex(0)};
    info->projected_output_count = 1;
    info->contract_id            = 1;
    info->approximate_batch_size = cap;
    reader                       = std::make_unique<scan::parquet_gpu_ingestible>(std::move(info),
                                                            scan::parquet_source_kind::ICEBERG);
    coalescer->inner             = reader->create_batch_coalescer();
    coalescer->now               = [this] {
      return scan::batch_coalescer::clock::time_point{} +
             std::chrono::milliseconds(milliseconds.load());
    };
    coordinator.clock_for_testing(coalescer->now);
    coalescer->after_advance = [this, first = true]() mutable {
      if (std::exchange(first, false)) retained->wait();
    };
    // The fixture's default observer expects tagged inputs, not Parquet splits.
    coordinator.after_publication_for_testing([this](size_t, auto const& metadata, size_t bytes) {
      auto const& split = dynamic_cast<scan::parquet_split_info const&>(*metadata);
      std::lock_guard lock(out->mutex);
      observed.push_back({bytes,
                          split.estimated_working_set_bytes(),
                          split.rg_slices.size(),
                          split.requires_row_positions});
      out->published.push_back(static_cast<int>(observed.size()));
      out->pending_bytes.push_back(bytes);
      out->cv.notify_all();
    });
    preparation_coordinator::source source;
    source.coalescer = coalescer;
    source.try_claim = [this, next = 0]() mutable -> preparation_coordinator::claim_result {
      if (next >= available.load()) return preparation_pending{};
      if (next == 2) return preparation_exhausted{};
      auto index = next++;
      return preparation_coordinator::job{
        [index] {
          auto file                     = std::make_unique<scan::parquet_file_scan_info>();
          file->file_index              = index;
          file->file_path               = std::to_string(index) + ".parquet";
          file->disable_filter_pushdown = true;
          file->set_contract_payload(
            1, {{.contract_id = 1, .split_id = 0, .input_identity = file->file_path}}, {{}});
          for (int group : {0, 1}) {
            file->row_groups.push_back({group, input_bytes / 2, input_bytes / 2, 10, 50000});
          }
          return file;
        },
        {}};
    };
    source.construct = [](auto batch) { return make_publication(std::move(batch)); };
    source.close     = [output = out] { output->close(); };
    register_source(std::move(source), out);
  }
};
}  // namespace

TEST_CASE("Iceberg interval deletion preserves preparation progress",
          "[scan][scan_preparation][coordinator][row_positions]")
{
  parquet_preparation f(GENERATE(1u, 100u));
  auto guard = f.shutdown_guard();
  f.start();
  REQUIRE(f.retained->await());
  f.milliseconds = 2100;  // Maximum residence flushes even a legacy-underfilled batch.
  f.retained->release();
  REQUIRE(f.out->await(1));
  {
    std::lock_guard lock(f.out->mutex);
    REQUIRE(f.observed.size() == 1);
    CHECK(f.observed[0].preparation == input_bytes);
    CHECK(f.observed[0].execution == input_bytes);
    CHECK_FALSE(f.observed[0].positions);
  }
  f.out->consume();
  f.available = 3;  // Permit the next file and EOS only after consuming the first split.
  f.coordinator.wake_for_testing();
  REQUIRE(f.out->await(2));
  REQUIRE(f.out->await_closed());
  f.out->consume();
  f.finish();
  CHECK_FALSE(f.completion.has_error());
  std::lock_guard lock(f.out->mutex);
  REQUIRE(f.observed.size() == 2);
  CHECK(f.observed[1].preparation == input_bytes);
  CHECK(f.observed[1].execution == input_bytes);
  CHECK_FALSE(f.observed[1].positions);
}

TEST_CASE("Delayed Iceberg metadata uses legacy fill for timed batching",
          "[scan][scan_preparation][coordinator][row_positions][partial_batch]")
{
  // Process each two-RG file together: a partial cursor can legitimately reach
  // half-cap and flush between RGs.
  parquet_preparation f(GENERATE(2u, 100u));
  std::promise<void> parked;
  auto reached = parked.get_future();
  std::atomic<bool> watch{false};
  f.coordinator.before_control_wait_for_testing([&] {
    if (watch.exchange(false)) parked.set_value();
  });
  auto guard = f.shutdown_guard();
  f.start();
  REQUIRE(f.retained->await());
  f.milliseconds = 150;  // Default 50 ms minimum: this batch remains below half-cap.
  watch          = true;
  f.retained->release();
  REQUIRE(reached.wait_for(1s) == std::future_status::ready);
  CHECK(f.out->count() == 0);
  CHECK(f.coordinator.snapshot().partial_emissions == 0);

  f.milliseconds = 200;  // Delayed metadata still arrives before the 2 s maximum.
  f.available    = 3;
  f.coordinator.wake_for_testing();
  REQUIRE(f.out->await(1));
  REQUIRE(f.out->await_closed());
  f.out->consume();
  f.finish();
  CHECK_FALSE(f.completion.has_error());
  std::lock_guard lock(f.out->mutex);
  REQUIRE(f.observed.size() == 1);
  CHECK(f.observed[0].files == 2);
  CHECK(f.observed[0].preparation == 2 * input_bytes);
  CHECK(f.observed[0].execution == 2 * input_bytes);
  CHECK_FALSE(f.observed[0].positions);
}
