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

// Hidden benchmarks of fused membership masks against the ordinary `compute_mask` and
// `fold_dynamic_filter_mask` chain. They need one GPU with no other GPU work running. Every
// benchmark runs 3 warmups and 15 measured repetitions per arm, alternates the arm order, and
// checks output, prefix and receipt parity. Run one tag at a time:
//   pixi run build/release/test/cpp/sirius_unittest '[fused_membership_bench]'
//   pixi run build/release/test/cpp/sirius_unittest '[fused_checkpoint_bench]'
//   pixi run build/release/test/cpp/sirius_unittest '[fused_scan_bench]'
// `[fused_membership_bench]` times `try_fused_membership_mask` on synthetic splits shaped like the
// cached TPC-H Q3/Q5/Q7/Q8 checkpoints, counts off then on. `FUSED_SAMPLE` rows hold one measured
// repetition. `FUSED_PREFLIGHT` rows hold the host cost per call of descriptor preparation for each
// shape, measured with zero-row views so that nothing is submitted. `FUSED_PREFLIGHT_SAMPLE` rows
// repeat that measurement for an eligible program and for programs that a STRING step makes fall
// back to the ordinary chain.
// `[fused_checkpoint_bench]` times `dynamic_filter_consumer::apply` on a prepared cached checkpoint
// with 1, 2, 5 and 9 filters, with its source lease, residual, counters, projected gather,
// positions and `finish` (`FUSED_CHECKPOINT` rows, with RMM peak and allocation counts).
// `[fused_scan_bench]` times a resident Parquet scan through its cached checkpoint, projection and
// carrier normalization (`FUSED_SCAN` rows).

#include "operator/operator_test_utils.hpp"
#include "scan/dynamic_filter_mask_test_utils.hpp"
#include "scan/resident_scan_test_utils.hpp"
#include "utils/test_validation_utility.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cuda/stream>
#include <cuda_runtime.h>

#include <catch.hpp>
#include <cucascade/cuda/event.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <data/data_batch_utils.hpp>
#include <helper/type_conversions.hpp>
#include <op/dynamic_filter/dynamic_filter_mask_ops.hpp>
#include <op/dynamic_filter/sirius_dynamic_filter.hpp>
#include <op/scan/dynamic_filter_merge.hpp>
#include <op/scan/scan_output_operator_data.hpp>
#include <op/scan/sirius_gpu_scan_operator.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace scan = sirius::op::scan;

namespace {
using sirius::op::fused_membership_input;
using sirius::op::sirius_mask_applicable;
using sirius::test::dynamic_filter_mask::copy_to_host;
using sirius::test::dynamic_filter_mask::fuse_membership;
using sirius::test::dynamic_filter_mask::make_device_column;
using sirius::test::dynamic_filter_mask::make_membership_filter;

/**
 * @brief Forms the ordered membership conjunction of @p steps in @p running
 *
 * This benchmark times the membership mask stage alone, so its ordinary arm repeats the per-filter
 * `compute_mask`/`compute_mask_if` and `fold_dynamic_filter_mask` sequence that `apply_cached`
 * runs, with the counter layout `try_fused_membership_mask` documents, rather than a whole consumer
 * checkpoint. `[fused_checkpoint_bench]` compares the production consumer paths end to end.
 *
 * @return Fused rounds submitted; always zero for the ordinary arm
 */
std::size_t submit_membership(bool fused,
                              std::span<sirius::op::fused_membership_input const> steps,
                              cudf::mutable_column_view running,
                              std::optional<cudf::column_view> residual,
                              cudf::device_span<cudf::size_type> counts,
                              ::cuda::stream_ref stream)
{
  auto const device = rmm::get_current_cuda_device().value();
  if (fused) { return fuse_membership(steps, running, residual, counts, stream); }
  auto const counter = [&](std::size_t index) {
    return counts.empty() ? nullptr : counts.data() + index;
  };
  bool initialize = !residual;
  if (residual) {
    sirius::op::fold_dynamic_filter_mask(running, residual, counter(0), true, stream);
  }
  for (std::size_t i = 0; i < steps.size(); ++i) {
    auto const& step = steps[i];
    auto mask        = initialize
                         ? step.filter->compute_mask(
                      step.probe, device, stream, cudf::get_current_device_resource_ref())
                         : step.filter->compute_mask_if(
                      step.probe, running, device, stream, cudf::get_current_device_resource_ref());
    sirius::op::fold_dynamic_filter_mask(running,
                                         mask ? std::optional{mask->view()} : std::nullopt,
                                         counter(i + static_cast<std::size_t>(bool(residual))),
                                         initialize,
                                         stream);
    initialize = false;
  }
  return 0;
}

constexpr int kHashSet = 0;
constexpr int kBloom   = 1;

uint64_t next_random(uint64_t& state)
{
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return state;
}

/// Two INT32 probe columns (order and part keys) and one INT64 payload column.
std::unique_ptr<cudf::table> make_split(cudf::size_type rows, ::cuda::stream_ref stream)
{
  std::vector<int32_t> orders(rows), parts(rows);
  uint64_t state = 0x1234567897654321ULL;
  for (cudf::size_type row = 0; row < rows; ++row) {
    auto const value = next_random(state);
    orders[row]      = static_cast<int32_t>(1 + value % 600'000'000u);
    parts[row]       = static_cast<int32_t>(1 + (value >> 20) % 20'000'000u);
  }
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(make_device_column(orders, cudf::data_type{cudf::type_id::INT32}, stream));
  columns.push_back(make_device_column(parts, cudf::data_type{cudf::type_id::INT32}, stream));
  columns.push_back(cudf::sequence(rows, cudf::numeric_scalar<int64_t>(0, true, stream), stream));
  return std::make_unique<cudf::table>(std::move(columns));
}

struct measurement {
  std::unique_ptr<cudf::table> output;
  std::vector<cudf::size_type> counts;
  std::size_t rounds;
  double wall_us;
  double cuda_us;
};

/**
 * @brief Times the mask and gather checkpoint with an already evaluated residual
 *
 * Wall time includes allocations, descriptor preparation, gather, count readback and the completion
 * wait. CUDA events bracket allocations through gather and exclude count readback. Input creation,
 * residual evaluation and consumer retirement are outside this benchmark, so it does not replace a
 * production checkpoint measurement. The lone-filter case uses the ordinary mask and derives its
 * count from the gathered row count in both arms.
 */
measurement measure_checkpoint(bool fused,
                               cudf::table_view input,
                               std::span<fused_membership_input const> steps,
                               std::optional<cudf::column_view> residual,
                               bool count,
                               ::cuda::stream_ref stream)
{
  auto const bypass = steps.size() == 1 && !residual;
  cucascade::cuda::cuda_event begin{cudaEventDefault}, end{cudaEventDefault};
  auto const start = std::chrono::steady_clock::now();
  begin.record(stream);
  auto running = bypass
                   ? steps.front().filter->compute_mask(steps.front().probe,
                                                        rmm::get_current_cuda_device().value(),
                                                        stream,
                                                        cudf::get_current_device_resource_ref())
                   : cudf::make_numeric_column(cudf::data_type{cudf::type_id::BOOL8},
                                               input.num_rows(),
                                               cudf::mask_state::UNALLOCATED,
                                               stream);
  rmm::device_uvector<cudf::size_type> counters(
    count && !bypass ? steps.size() + static_cast<std::size_t>(bool(residual)) : 0, stream);
  if (!counters.is_empty()) {
    RMM_CUDA_TRY(
      cudaMemsetAsync(counters.data(), 0, counters.size() * sizeof(cudf::size_type), stream.get()));
  }
  auto const rounds = bypass ? 0
                             : submit_membership(fused,
                                                 steps,
                                                 running->mutable_view(),
                                                 residual,
                                                 {counters.data(), counters.size()},
                                                 stream);
  auto output       = cudf::apply_boolean_mask(
    input, running->view(), stream, cudf::get_current_device_resource_ref());
  end.record(stream);
  end.synchronize();
  auto counts = copy_to_host(counters.data(), counters.size(), stream);
  if (count && bypass) { counts.push_back(output->num_rows()); }
  auto const stop = std::chrono::steady_clock::now();
  return {std::move(output),
          std::move(counts),
          rounds,
          std::chrono::duration<double, std::micro>(stop - start).count(),
          end.elapsed_time(begin).count() * 1000.0};
}

/**
 * @brief Measures descriptor preparation using zero-row views, so `try_fused_membership_mask`
 * submits no work
 *
 * Types, pointers and filter replicas match the timed checkpoint. The host cost per call includes
 * vector allocation, input validation and, for a program with an unsupported step, the descriptors
 * prepared before the refusal. It is reported separately, not subtracted from checkpoint wall time.
 */
double preflight_us(std::span<fused_membership_input const> steps,
                    std::optional<cudf::column_view> residual,
                    bool count,
                    ::cuda::stream_ref stream,
                    int repetitions = 1000)
{
  auto const empty = [](cudf::column_view column) {
    return cudf::column_view{
      column.type(), 0, column.head<void>(), column.null_mask(), 0, column.offset()};
  };
  std::vector<fused_membership_input> zero_steps;
  for (auto const& step : steps) {
    zero_steps.push_back({step.filter, empty(step.probe)});
  }
  cudf::mutable_column_view running{cudf::data_type{cudf::type_id::BOOL8}, 0, nullptr, nullptr, 0};
  std::optional<cudf::column_view> zero_residual;
  if (residual) { zero_residual = empty(*residual); }
  rmm::device_uvector<cudf::size_type> counters(
    count ? steps.size() + static_cast<std::size_t>(bool(residual)) : 0, stream);
  if (!counters.is_empty()) {
    RMM_CUDA_TRY(
      cudaMemsetAsync(counters.data(), 0, counters.size() * sizeof(cudf::size_type), stream.get()));
  }
  stream.sync();
  auto const device     = rmm::get_current_cuda_device().value();
  std::size_t submitted = 0;
  auto const start      = std::chrono::steady_clock::now();
  for (int i = 0; i < repetitions; ++i) {
    std::size_t rounds = 0;
    sirius::op::try_fused_membership_mask(zero_steps,
                                          device,
                                          running,
                                          zero_residual,
                                          {counters.data(), counters.size()},
                                          rounds,
                                          stream);
    submitted += rounds;
  }
  auto const stop = std::chrono::steady_clock::now();
  REQUIRE(submitted == 0);
  return std::chrono::duration<double, std::micro>(stop - start).count() / repetitions;
}

void benchmark_shape(char const* shape,
                     cudf::table_view input,
                     std::span<fused_membership_input const> steps,
                     std::optional<cudf::column_view> residual,
                     ::cuda::stream_ref stream)
{
  for (auto const count : {false, true}) {
    auto reference = measure_checkpoint(false, input, steps, residual, count, stream);
    auto candidate = measure_checkpoint(true, input, steps, residual, count, stream);
    REQUIRE(sirius::test::expect_tables_equivalent_impl(
      reference.output->view(), candidate.output->view(), stream));
    REQUIRE(reference.counts == candidate.counts);
    auto const bypass = steps.size() == 1 && !residual;
    REQUIRE(candidate.rounds == (bypass ? 0 : (steps.size() + 3) / 4));
    std::printf("FUSED_PREFLIGHT,%s,%d,%.3f\n",
                shape,
                count,
                bypass ? 0.0 : preflight_us(steps, residual, count, stream));
    auto const expected_rows   = reference.output->num_rows();
    auto const expected_counts = reference.counts;
    reference.output.reset();
    candidate.output.reset();
    // Alternate order per repetition; no explicit L2 flush, matching a hot pinned source.
    for (int repetition = -3; repetition < 15; ++repetition) {
      for (int arm = 0; arm < 2; ++arm) {
        bool const fused = (arm == 0) != ((repetition & 1) == 0);
        auto result      = measure_checkpoint(fused, input, steps, residual, count, stream);
        REQUIRE(result.output->num_rows() == expected_rows);
        REQUIRE(result.counts == expected_counts);
        REQUIRE(result.rounds == (fused && !bypass ? (steps.size() + 3) / 4 : 0));
        if (repetition < 0) { continue; }
        std::printf("FUSED_SAMPLE,%s,%d,%s,%d,%zu,%d,%.3f,%.3f",
                    shape,
                    count,
                    fused ? "fused" : "ordinary",
                    repetition,
                    result.rounds,
                    result.output->num_rows(),
                    result.wall_us,
                    result.cuda_us);
        for (auto prefix : result.counts) {
          std::printf(",%d", prefix);
        }
        std::printf("\n");
      }
    }
    std::fflush(stdout);
  }
}
}  // namespace

TEST_CASE("fused membership benchmark on reference q3 q5 q7 q8 shapes",
          "[.][fused_membership_bench]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  std::printf(
    "FUSED_SAMPLE,shape,counts,arm,repetition,rounds,output_rows,wall_us,cuda_us,prefixes...\n");
  std::printf("FUSED_PREFLIGHT,shape,counts,zero_row_host_us_per_call\n");
  for (bool const two_filters : {false, true}) {
    auto const rows = two_filters ? 29'800'000 : 34'000'000;
    auto input      = make_split(rows, stream);
    uint64_t state  = two_filters ? 0x999ULL : 0x777ULL;
    std::vector<int64_t> bloom_keys(two_filters ? 9'100'000 : 14'700'000);
    for (auto& key : bloom_keys) {
      key = static_cast<int64_t>(1 + next_random(state) % 600'000'000u);
    }
    auto build = make_device_column(bloom_keys, cudf::data_type{cudf::type_id::INT64}, stream);
    auto bloom = make_membership_filter(kBloom, build->view(), stream);
    std::unique_ptr<sirius_mask_applicable> set;
    std::vector<fused_membership_input> steps;
    if (two_filters) {
      std::vector<int32_t> set_keys(133'000);
      for (auto& key : set_keys) {
        key = static_cast<int32_t>(1 + next_random(state) % 20'000'000u);
      }
      auto set_build = make_device_column(set_keys, cudf::data_type{cudf::type_id::INT32}, stream);
      set            = make_membership_filter(kHashSet, set_build->view(), stream);
      steps.push_back({set.get(), input->view().column(1)});
    }
    steps.push_back({bloom.get(), input->view().column(0)});
    state = 0xabcdef1234567ULL;
    std::vector<uint8_t> keep(rows);
    for (auto& value : keep) {
      value = next_random(state) % 100 < (two_filters ? 30u : 54u);
    }
    auto residual = make_device_column(keep, cudf::data_type{cudf::type_id::BOOL8}, stream);
    stream.sync();
    benchmark_shape(two_filters ? "q7_29.8M_res30_set133K_bloom9.1M" : "q3_34M_res54_bloom14.7M",
                    input->view(),
                    steps,
                    residual->view(),
                    stream);
    benchmark_shape(two_filters ? "q8_29.8M_set133K_bloom9.1M" : "q5_34M_nores_bloom14.7M",
                    input->view(),
                    steps,
                    std::nullopt,
                    stream);
  }
}

TEST_CASE("fused membership preflight cost for eligible and string-refused programs",
          "[.][fused_membership_bench]")
{
  ::cuda::stream_ref stream      = cudf::get_default_stream();
  auto const mr                  = cudf::get_current_device_resource_ref();
  constexpr cudf::size_type rows = 4096;
  auto input                     = make_split(rows, stream);
  auto set_keys =
    make_device_column<int32_t>({1, 2, 3, 4}, cudf::data_type{cudf::type_id::INT32}, stream);
  auto bloom_keys =
    make_device_column<int64_t>({1, 2, 3, 4}, cudf::data_type{cudf::type_id::INT64}, stream);
  auto set   = make_membership_filter(kHashSet, set_keys->view(), stream);
  auto bloom = make_membership_filter(kBloom, bloom_keys->view(), stream);
  std::vector<std::string> names(rows);
  for (cudf::size_type row = 0; row < rows; ++row) {
    names[row] = "name_" + std::to_string(row % 1000);
  }
  auto strings = sirius::test::operator_utils::make_string_column(names, stream, mr);
  auto string_keys =
    sirius::test::operator_utils::make_string_column({"name_1", "name_2", "name_3"}, stream, mr);
  sirius::op::sirius_dynamic_in_list_filter string_set(string_keys->view(), stream, mr);
  auto residual = make_device_column(
    std::vector<uint8_t>(rows, 1), cudf::data_type{cudf::type_id::BOOL8}, stream);
  stream.sync();

  fused_membership_input const set_step{set.get(), input->view().column(1)};
  fused_membership_input const bloom_step{bloom.get(), input->view().column(0)};
  fused_membership_input const string_step{&string_set, strings->view()};
  struct program {
    char const* name;
    bool eligible;
    std::vector<fused_membership_input> steps;
  };
  std::vector<program> const programs{
    {"eligible_set_bloom", true, {set_step, bloom_step}},
    {"refused_set_bloom_string_last", false, {set_step, bloom_step, string_step}},
    {"refused_string_first_set_bloom", false, {string_step, set_step, bloom_step}}};
  // Any submission would make the zero-row proxy meaningless, so first prove each program's path.
  auto running = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::BOOL8}, rows, cudf::mask_state::UNALLOCATED, stream);
  for (auto const& [name, eligible, steps] : programs) {
    REQUIRE(fuse_membership(steps, running->mutable_view(), residual->view(), {}, stream) ==
            (eligible ? 1 : 0));
  }
  stream.sync();
  std::printf("FUSED_PREFLIGHT_SAMPLE,program,counts,sample,zero_row_host_us_per_call\n");
  for (auto const count : {false, true}) {
    for (auto const& [name, eligible, steps] : programs) {
      static_cast<void>(preflight_us(steps, residual->view(), count, stream, 10'000));
      for (int sample = 0; sample < 15; ++sample) {
        std::printf("FUSED_PREFLIGHT_SAMPLE,%s,%d,%d,%.4f\n",
                    name,
                    count,
                    sample,
                    preflight_us(steps, residual->view(), count, stream, 10'000));
      }
    }
  }
  std::fflush(stdout);
}

namespace {
using scan::application_program;
using scan::batch_receipt;
using scan::binding;
using scan::dynamic_filter_consumer;
using scan::owning_table_view;
using scan::detail::consumer_test_access;
using sirius::op::sirius_dynamic_filter;
using sirius::op::sirius_dynamic_filter_set;
using sirius::test::dynamic_filter_mask::cached_layout;
using sirius::test::dynamic_filter_mask::counts_of;
struct checkpoint_measurement {
  std::unique_ptr<cudf::table> output;
  std::unique_ptr<cudf::column> positions;
  std::vector<cudf::size_type> counts;
  batch_receipt receipt;
  std::size_t rounds;
  std::size_t peak_bytes;
  std::size_t allocations;
  double wall_us;
  double cuda_us;
};

/**
 * @brief Measures a prepared cached checkpoint, including its real source lease and reader event
 *
 * Persistent inputs, filters and policy preparation precede timing. Wall time includes residual
 * evaluation, descriptor preparation, counters/readback, projected gather, optional positions,
 * completion and finish. Normalization is measured separately by the resident-scan benchmark. The
 * allocation adaptor must outlive the returned output and positions.
 */
checkpoint_measurement measure_cached_checkpoint(
  dynamic_filter_consumer& consumer,
  application_program&& program,
  std::shared_ptr<cucascade::data_batch> const& source,
  cudf::table_view view,
  rmm::mr::statistics_resource_adaptor& allocations,
  ::cuda::stream_ref stream)
{
  cucascade::cuda::cuda_event begin{cudaEventDefault}, end{cudaEventDefault};
  auto const start = std::chrono::steady_clock::now();
  begin.record(stream);
  auto result = consumer.apply(std::move(program),
                               owning_table_view{source->to_read_only(), view},
                               stream,
                               rmm::device_async_resource_ref{allocations});
  end.record(stream);
  end.synchronize();
  auto const rounds = consumer_test_access::storage(result).fused_rounds;
  auto counts       = counts_of(result);
  auto output       = result.take_output();
  auto positions    = result.take_row_positions();
  auto receipt      = consumer.finish(std::move(result), 1);
  auto const stop   = std::chrono::steady_clock::now();
  return {std::move(output),
          std::move(positions),
          std::move(counts),
          std::move(receipt),
          rounds,
          static_cast<std::size_t>(allocations.get_bytes_counter().peak),
          static_cast<std::size_t>(allocations.get_allocations_counter().total),
          std::chrono::duration<double, std::micro>(stop - start).count(),
          end.elapsed_time(begin).count() * 1000.0};
}
}  // namespace

TEST_CASE("cached fused checkpoint benchmark with projected gather and source retirement",
          "[.][fused_checkpoint_bench]")
{
  constexpr cudf::size_type rows = 6'400'000;
  ::cuda::stream_ref stream      = cudf::get_default_stream();
  auto manager                   = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu =
    manager->get_memory_space(cucascade::memory::Tier::GPU, rmm::get_current_cuda_device().value());
  REQUIRE(gpu);
  std::vector<int32_t> keys(rows);
  std::vector<uint8_t> residual(rows);
  for (cudf::size_type row = 0; row < rows; ++row) {
    keys[row]     = row % 8;
    residual[row] = row % 10 < 7;
  }
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(make_device_column(keys, cudf::data_type{cudf::type_id::INT32}, stream));
  columns.push_back(make_device_column(residual, cudf::data_type{cudf::type_id::BOOL8}, stream));
  columns.push_back(cudf::sequence(rows, cudf::numeric_scalar<int64_t>(0, true, stream), stream));
  auto source =
    sirius::make_data_batch(std::make_unique<cudf::table>(std::move(columns)), *gpu, stream, {});
  auto const view = sirius::get_cudf_table_view(*source);
  struct shape {
    int filters;
    bool strong;
    bool counts;
    bool positions;
  };
  std::array cases{shape{1, false, false, false},
                   shape{1, true, true, false},
                   shape{2, false, true, false},
                   shape{2, true, false, true},
                   shape{5, false, false, false},
                   shape{5, true, true, true},
                   shape{9, false, true, true},
                   shape{9, true, false, false}};
  std::printf(
    "FUSED_CHECKPOINT,filters,strong,counts,positions,arm,rep,rounds,rows,wall_us,cuda_us,peak_"
    "bytes,"
    "allocations,prefixes...\n");
  for (auto const shape : cases) {
    auto channel  = std::make_shared<sirius_dynamic_filter_set>();
    auto producer = channel->register_producer({0}, shape.filters);
    std::vector<int64_t> build_values(shape.strong ? 1 : 7);
    std::iota(build_values.begin(), build_values.end(), 0);
    auto build = make_device_column(build_values, cudf::data_type{cudf::type_id::INT64}, stream);
    for (int i = 0; i < shape.filters; ++i) {
      std::shared_ptr<sirius_mask_applicable> mask =
        make_membership_filter(i % 3, build->view(), stream);
      auto filter = std::dynamic_pointer_cast<sirius_dynamic_filter>(mask);
      REQUIRE(filter);
      REQUIRE(producer.push_filter(0, std::move(filter)));
    }
    std::array bindings{binding{0, 0}};
    auto layout              = cached_layout(shape.filters == 1);
    layout.projected_columns = {2, 0, 2};
    layout.output_columns    = layout.projected_columns;
    std::vector<binding> decode(bindings.begin(), bindings.end());
    stream.sync();
    for (int repetition = -3; repetition < 15; ++repetition) {
      dynamic_filter_consumer ordinary(
        channel, {.force_counts = shape.counts, .fused_membership = false}, decode);
      dynamic_filter_consumer fused(
        channel, {.force_counts = shape.counts, .fused_membership = true}, decode);
      sirius::op::scan::application_input request{.source             = view,
                                                  .bindings           = bindings,
                                                  .cached_layout      = &layout,
                                                  .want_row_positions = shape.positions};
      auto a = ordinary.prepare(request);
      auto b = fused.prepare(request);
      REQUIRE(a);
      REQUIRE(b);
      REQUIRE(consumer_test_access::same_selection(*a, *b));
      auto const mr = cudf::get_current_device_resource_ref();
      rmm::mr::statistics_resource_adaptor stats_a{
        cuda::mr::any_resource<cuda::mr::device_accessible>{mr}};
      rmm::mr::statistics_resource_adaptor stats_b{
        cuda::mr::any_resource<cuda::mr::device_accessible>{mr}};
      std::array<std::optional<checkpoint_measurement>, 2> results;
      for (int arm = 0; arm < 2; ++arm) {
        bool const candidate = (arm == 0) != ((repetition & 1) == 0);
        auto& consumer       = candidate ? fused : ordinary;
        auto& program        = candidate ? b : a;
        auto& stats          = candidate ? stats_b : stats_a;
        results[candidate].emplace(
          measure_cached_checkpoint(consumer, std::move(*program), source, view, stats, stream));
      }
      auto const& left  = *results[0];
      auto const& right = *results[1];
      REQUIRE(left.output);
      REQUIRE(right.output);
      REQUIRE(left.rounds == 0);
      REQUIRE(right.rounds == static_cast<std::size_t>((shape.filters + 3) / 4));
      REQUIRE(left.output->num_rows() == right.output->num_rows());
      REQUIRE(left.counts == right.counts);
      REQUIRE(left.receipt.applied == right.receipt.applied);
      REQUIRE(left.counts.empty() == !shape.counts);
      if (repetition == -3) {
        REQUIRE(sirius::test::expect_tables_equivalent_impl(
          left.output->view(), right.output->view(), stream));
        if (shape.positions) {
          auto const& a = *left.positions;
          auto const& b = *right.positions;
          REQUIRE(copy_to_host(a.view().data<int32_t>(), a.size(), stream) ==
                  copy_to_host(b.view().data<int32_t>(), b.size(), stream));
        }
      }
      if (repetition < 0) { continue; }
      for (int arm = 0; arm < 2; ++arm) {
        auto const& result = *results[arm];
        std::printf("FUSED_CHECKPOINT,%d,%d,%d,%d,%s,%d,%zu,%d,%.3f,%.3f,%zu,%zu",
                    shape.filters,
                    shape.strong,
                    shape.counts,
                    shape.positions,
                    arm ? "fused" : "ordinary",
                    repetition,
                    result.rounds,
                    result.output->num_rows(),
                    result.wall_us,
                    result.cuda_us,
                    result.peak_bytes,
                    result.allocations);
        for (auto prefix : result.counts) {
          std::printf(",%d", prefix);
        }
        std::printf("\n");
      }
    }
    std::fflush(stdout);
  }
}

namespace {
using sirius::test::resident_scan::admit_fixture_resident_input;
using sirius::test::resident_scan::env;
using sirius::test::resident_scan::resident_column;
using sirius::test::resident_scan::resident_columns;
using sirius::test::resident_scan::resident_parquet_info;
}  // namespace

TEST_CASE("resident fused scan benchmark includes projection and carrier normalization",
          "[.][fused_scan_bench]")
{
  constexpr cudf::size_type rows = 6'400'000;
  auto& e                        = env();
  auto const stream              = e.stream();
  std::vector<int32_t> payload(rows), key(rows), keep(rows);
  for (cudf::size_type row = 0; row < rows; ++row) {
    payload[row] = row;
    key[row]     = row % 8;
    keep[row]    = row % 10 < 7;
  }
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(resident_column(payload, cudf::type_id::INT32));
  columns.push_back(resident_column(key, cudf::type_id::INT32));
  columns.push_back(resident_column(keep, cudf::type_id::INT32));
  auto source     = resident_columns(std::move(columns));
  auto ingestible = scan::make_ingestible(resident_parquet_info({1, 0, 1, 2}, 3, 3));
  duckdb::vector<duckdb::LogicalType> types(3, duckdb::LogicalType::BIGINT);
  std::printf("FUSED_SCAN,strong,counts,arm,rep,rows,wall_us,cuda_us,peak_bytes,allocations\n");
  for (bool strong : {false, true}) {
    auto channel  = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
    auto producer = channel->register_producer({1});
    std::vector<int64_t> keys(strong ? 1 : 7);
    std::iota(keys.begin(), keys.end(), 0);
    auto build = resident_column(keys, cudf::type_id::INT64);
    for (int i = 0; i < 2; ++i) {
      REQUIRE(producer.push_filter(1,
                                   std::make_shared<sirius::op::sirius_dynamic_in_list_filter>(
                                     build->view(), stream, e.gpu_space->get_default_allocator())));
    }
    auto const snapshot = channel->snapshot();
    for (int repetition = -3; repetition < 15; ++repetition) {
      std::array<std::shared_ptr<scan::dynamic_filter_consumer>, 2> consumers;
      std::array<std::unique_ptr<scan::sirius_gpu_scan_operator>, 2> scans;
      for (int arm = 0; arm < 2; ++arm) {
        consumers[arm] = std::make_shared<scan::dynamic_filter_consumer>(
          channel,
          scan::consumer_config{.force_counts = strong, .fused_membership = arm != 0},
          scan::scan_decode_bindings(*ingestible, types.size()));
        scans[arm] = std::make_unique<scan::sirius_gpu_scan_operator>(
          sirius::from_duckdb_vec(types), 0, ingestible, 1, nullptr, nullptr, consumers[arm]);
      }
      auto const upstream = e.gpu_space->get_default_allocator();
      rmm::mr::statistics_resource_adaptor stats_a{
        cuda::mr::any_resource<cuda::mr::device_accessible>{upstream}};
      rmm::mr::statistics_resource_adaptor stats_b{
        cuda::mr::any_resource<cuda::mr::device_accessible>{upstream}};
      auto make_space = [&](rmm::mr::statistics_resource_adaptor& stats) {
        return std::make_unique<cucascade::memory::memory_space>(
          cucascade::memory::gpu_memory_space_config{
            .device_id       = e.gpu_space->get_device_id(),
            .memory_capacity = e.gpu_space->get_max_memory(),
            .mr_factory_fn   = [&stats](int, std::size_t) {
              return cuda::mr::any_resource<cuda::mr::device_accessible>{stats};
            }});
      };
      std::array spaces{make_space(stats_a), make_space(stats_b)};
      std::array<std::unique_ptr<scan::scan_operator_input>, 2> inputs;
      for (int arm = 0; arm < 2; ++arm) {
        inputs[arm] = std::make_unique<scan::scan_operator_input>(source);
        auto& input = *inputs[arm];
        admit_fixture_resident_input(input);
        input.needs_carrier_conversion = true;
        input.row_filter_pending       = true;
        input.cached_view_possible     = scans[arm]->cached_view_possible();
        for (auto const& entry : snapshot.entries()) {
          input.decode_attached.push_back({entry.filter, entry.column_index});
        }
        input.prepare_for_processing(spaces[arm].get(), stream);
        REQUIRE_FALSE(input.stolen_table);
        REQUIRE_FALSE(input.converted_table_steal_pending);
        REQUIRE(input.gpu_memory_space->get_device_id() == spaces[arm]->get_device_id());
        // Preparation adopts the resident source's space; count execution allocations on the same
        // device.
        input.gpu_memory_space = spaces[arm].get();
      }
      stream.sync();
      struct scan_measurement {
        std::unique_ptr<sirius::op::operator_data> output;
        double wall_us;
        double cuda_us;
        std::size_t peak;
        std::size_t allocations;
      };
      std::array<std::optional<scan_measurement>, 2> results;
      for (int order = 0; order < 2; ++order) {
        int const arm = (repetition & 1) == 0 ? order : 1 - order;
        auto& stats   = arm ? stats_b : stats_a;
        cucascade::cuda::cuda_event begin{cudaEventDefault}, end{cudaEventDefault};
        auto const start = std::chrono::steady_clock::now();
        begin.record(stream);
        auto output = scans[arm]->execute(*inputs[arm], stream);
        end.record(stream);
        end.synchronize();
        auto const stop = std::chrono::steady_clock::now();
        results[arm].emplace(
          scan_measurement{std::move(output),
                           std::chrono::duration<double, std::micro>(stop - start).count(),
                           end.elapsed_time(begin).count() * 1000.0,
                           static_cast<std::size_t>(stats.get_bytes_counter().peak),
                           static_cast<std::size_t>(stats.get_allocations_counter().total)});
      }
      auto const& left  = dynamic_cast<scan::scan_output_operator_data const&>(*results[0]->output);
      auto const& right = dynamic_cast<scan::scan_output_operator_data const&>(*results[1]->output);
      REQUIRE(left.receipt().applied == right.receipt().applied);
      REQUIRE(right.receipt().applied.size() == 2);
      auto left_batches  = left.get_read_only_batches();
      auto right_batches = right.get_read_only_batches();
      REQUIRE(left_batches.size() == 1);
      REQUIRE(right_batches.size() == 1);
      auto a = sirius::get_cudf_table_view(left_batches.front());
      auto b = sirius::get_cudf_table_view(right_batches.front());
      REQUIRE(a.num_rows() == b.num_rows());
      REQUIRE(b.num_rows() == rows * (strong ? 4 : 25) / 40);
      for (auto column : b) {
        REQUIRE(column.type().id() == cudf::type_id::INT64);
      }
      auto const normalized_bytes =
        static_cast<std::size_t>(b.num_rows()) * b.num_columns() * sizeof(int64_t);
      for (auto const& result : results) {
        REQUIRE(result->allocations > 0);
        REQUIRE(result->peak >= normalized_bytes);
      }
      if (repetition == -3) { REQUIRE(sirius::test::expect_tables_equivalent_impl(a, b, stream)); }
      if (repetition < 0) { continue; }
      for (int arm = 0; arm < 2; ++arm) {
        auto const& result = *results[arm];
        std::printf("FUSED_SCAN,%d,%d,%s,%d,%d,%.3f,%.3f,%zu,%zu\n",
                    strong,
                    strong,
                    arm ? "fused" : "ordinary",
                    repetition,
                    b.num_rows(),
                    result.wall_us,
                    result.cuda_us,
                    result.peak,
                    result.allocations);
      }
    }
    std::fflush(stdout);
  }
}
