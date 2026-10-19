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

#include "api/simpatico_codegen.hpp"
#include "operator/operator_test_utils.hpp"
#include "utils/host_allocation_fault.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>

#include <catch.hpp>
#include <compression/compressed_representation.hpp>
#include <compression/decompression_pushdown_policy.hpp>
#include <compression/device_compressed_blob.hpp>
#include <late_mat/column_origin.hpp>
#include <late_mat/defer_directive.hpp>
#include <op/scan/decoded_batch_representation.hpp>
#include <op/scan/scan_output_operator_data.hpp>
#include <op/scan/sirius_gpu_scan_operator.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <op/scan/sirius_physical_dynamic_filter.hpp>
#include <planner/late_mat_plan_pass.hpp>
#include <scan_manager/load_balancing_scan_batch_coalescer.hpp>
#include <scan_manager/sirius_scan_manager.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace sirius::op::scan;
using sirius::op::pipelineable_operator_data;
using sirius::op::sirius_dynamic_filter_set;
using sirius::op::scan::detail::gate_test_access;

std::unique_ptr<cudf::table> sequence_table(cudf::size_type rows, ::cuda::stream_ref stream)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(cudf::sequence(rows,
                                   cudf::numeric_scalar<int64_t>(0, true, stream),
                                   cudf::numeric_scalar<int64_t>(1, true, stream),
                                   stream));
  return std::make_unique<cudf::table>(std::move(columns));
}

std::shared_ptr<cucascade::data_batch> sequence_batch(cucascade::memory::memory_space& space,
                                                      ::cuda::stream_ref stream)
{
  return sirius::make_data_batch(sequence_table(8, stream), space, stream, {});
}

class counting_filter final : public sirius::op::sirius_dynamic_filter,
                              public sirius::op::sirius_mask_applicable {
 public:
  explicit counting_filter(::cuda::stream_ref stream)
  {
    auto keys = sequence_table(2, stream);
    _filter   = std::make_shared<sirius::op::sirius_dynamic_in_list_filter>(
      keys->view().column(0), stream, cudf::get_current_device_resource_ref());
  }

  sirius::op::sirius_dynamic_filter_kind kind() const override
  {
    return sirius::op::sirius_dynamic_filter_kind::IN_LIST;
  }

  std::unique_ptr<cudf::column> compute_mask(cudf::column_view const& probe,
                                             std::uint32_t const* prior,
                                             int device,
                                             ::cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr) const override
  {
    ++calls;
    if (calls == fail_on_call) { throw std::bad_alloc{}; }
    return _filter->compute_mask(probe, prior, device, stream, mr);
  }

  mutable std::size_t calls{0};
  std::size_t fail_on_call{0};

 private:
  std::shared_ptr<sirius::op::sirius_dynamic_in_list_filter> _filter;
};

duckdb::vector<sirius::logical_type> output_types()
{
  return {sirius::logical_type::make(sirius::type_id::BIGINT)};
}

dynamic_filter_gate& gate(dynamic_filter_consumer& consumer)
{
  return sirius::op::scan::detail::consumer_test_access::gate(consumer);
}

void train_for_decode(dynamic_filter_consumer& consumer, counting_filter const& filter)
{
  gate(consumer).record_keep_ratio(8, 2, 1);
  gate(consumer).record_filter_keep_ratio({&filter, 0}, 0.25, 1);
}

class queued_provider final : public sirius::scan_manager::databatch_provider {
 public:
  std::vector<batch> batches;
  batch get_next_batch() override
  {
    if (_next == batches.size()) { return {}; }
    return std::move(batches[_next++]);
  }

 private:
  std::size_t _next{0};
};

template <typename Base>
class inspected_compressed_representation final : public Base {
 public:
  using Base::Base;
  bool refuse_conversion{true};
};

template <typename Representation>
std::unique_ptr<Representation> compressed_representation(cucascade::memory::memory_space& space)
{
  using blob =
    std::conditional_t<std::is_base_of_v<sirius::compressed_host_representation, Representation>,
                       sirius::pinned_compressed_blob,
                       sirius::compressed_device_blob>;
  return std::make_unique<Representation>(
    space, std::make_shared<blob>(), std::vector<std::string>{"key"}, 64, 64, 8);
}

sirius::scan_manager::mvcc_chunk_mask visibility_mask()
{
  auto words = std::make_shared<std::uint32_t[]>(1);
  words[0]   = 0x7f;
  return {std::move(words), 8};
}

class resident_table_info final : public ingestible_table_info {
 public:
  std::span<std::string const> column_names() const override { return {}; }
  std::span<std::string const> file_paths() const override { return {}; }
  std::string display_name() const override { return "resident decode outcome test"; }
};

class resident_ingestible final : public gpu_ingestible {
 public:
  std::unique_ptr<batch_coalescer> create_batch_coalescer() const override { return nullptr; }
  bool has_processed_all_metadata() const override { return true; }
  metadata_scan_task_t next_split_provider(sirius::io::ioctx_resolver) override { return {}; }
  ingestible_table_info const& table_info() const noexcept override { return _info; }
  std::vector<std::size_t> materialized_column_order() const override { return {0}; }

  filtered_table materialize_metadata_to_table(
    scan_info const&,
    cucascade::memory::memory_space const&,
    ::cuda::stream_ref,
    bool,
    std::shared_ptr<sirius::like_multiliteral_cache const>,
    sirius::expression_query_policy /*policy*/) override
  {
    throw std::logic_error("resident decode outcome test cannot read scan metadata");
  }

  std::unique_ptr<cudf::table> post_filter_and_project(
    filtered_table&&,
    cucascade::memory::memory_space const&,
    ::cuda::stream_ref,
    bool,
    std::shared_ptr<sirius::like_multiliteral_cache const>,
    std::unique_ptr<cudf::column>*,
    std::span<std::size_t const>,
    sirius::expression_query_policy /*policy*/) override
  {
    throw std::logic_error("resident decode outcome test only materializes cached rows");
  }

 private:
  resident_table_info _info;
};

/// Serves resident splits whose materialized columns are the output columns, in order.
class leading_identity_ingestible final : public gpu_ingestible {
 public:
  explicit leading_identity_ingestible(std::size_t width) : _width(width) {}

  std::unique_ptr<batch_coalescer> create_batch_coalescer() const override { return nullptr; }
  bool has_processed_all_metadata() const override { return true; }
  metadata_scan_task_t next_split_provider(sirius::io::ioctx_resolver) override { return {}; }
  ingestible_table_info const& table_info() const noexcept override { return _info; }
  bool output_assembly_is_leading_identity() const noexcept override { return true; }

  std::vector<std::size_t> materialized_column_order() const override
  {
    std::vector<std::size_t> order(_width);
    std::iota(order.begin(), order.end(), std::size_t{0});
    return order;
  }

  filtered_table materialize_metadata_to_table(
    scan_info const&,
    cucascade::memory::memory_space const&,
    ::cuda::stream_ref,
    bool,
    std::shared_ptr<sirius::like_multiliteral_cache const>,
    sirius::expression_query_policy /*policy*/) override
  {
    throw std::logic_error("leading identity test ingestible cannot read scan metadata");
  }

  /// Projects every column except the elided ones; there is no row filter.
  std::unique_ptr<cudf::table> post_filter_and_project(
    filtered_table&& input,
    cucascade::memory::memory_space const& space,
    ::cuda::stream_ref stream,
    bool,
    std::shared_ptr<sirius::like_multiliteral_cache const>,
    std::unique_ptr<cudf::column>*,
    std::span<std::size_t const> elided,
    sirius::expression_query_policy /*policy*/) override
  {
    auto columns = input.table.release(stream, space.get_default_allocator())->release();
    std::vector<std::unique_ptr<cudf::column>> projected;
    for (std::size_t position = 0; position < columns.size(); ++position) {
      if (std::ranges::find(elided, position) == elided.end()) {
        projected.push_back(std::move(columns[position]));
      }
    }
    return std::make_unique<cudf::table>(std::move(projected));
  }

 private:
  std::size_t _width;
  resident_table_info _info;
};

template <typename T>
std::vector<T> to_host(cudf::column_view const& column, ::cuda::stream_ref stream)
{
  std::vector<T> host(static_cast<std::size_t>(column.size()));
  cudaMemcpyAsync(
    host.data(), column.data<T>(), host.size() * sizeof(T), cudaMemcpyDeviceToHost, stream.get());
  stream.sync();
  return host;
}

template <typename Base>
class inspected_converter {
 public:
  using representation = inspected_compressed_representation<Base>;

  inspected_converter()
  {
    sirius::converter_registry::get()
      .register_converter<representation, cucascade::gpu_table_representation>(
        [](cucascade::idata_representation& source,
           cucascade::memory::memory_space const* target,
           ::cuda::stream_ref stream,
           cucascade::memory::reservation*) -> std::unique_ptr<cucascade::idata_representation> {
          auto& compressed = source.template cast<representation>();
          if (compressed.refuse_conversion) { throw std::bad_alloc{}; }
          return std::make_unique<cucascade::gpu_table_representation>(
            sequence_table(8, stream),
            *const_cast<cucascade::memory::memory_space*>(target),
            stream);
        });
  }

  ~inspected_converter()
  {
    sirius::converter_registry::get()
      .unregister_converter<representation, cucascade::gpu_table_representation>();
  }
};

}  // namespace

TEST_CASE("scan receipts survive preparation and do not train from decode survivors",
          "[dynamic_filter][consumer_integration][scan_receipt]")
{
  auto* space = sirius::test::operator_utils::get_default_gpu_space();
  rmm::cuda_stream stream;
  auto channel  = std::make_shared<sirius_dynamic_filter_set>();
  auto producer = channel->register_producer({0});
  auto filter   = std::make_shared<counting_filter>(stream);
  REQUIRE(producer.push_filter(0, filter));
  auto consumer =
    std::make_shared<dynamic_filter_consumer>(channel, consumer_config{}, std::vector<binding>{});
  auto batch             = sequence_batch(*space, stream);
  auto const original_id = batch->get_batch_id();
  scan_output_operator_data input(batch);
  input.install_receipt({channel, original_id, {}, {{filter, 0}}});
  input.prepare_for_processing(space, stream);
  input.prepare_for_processing(space, stream);
  REQUIRE(input.get_type() == sirius::op::operator_data_type::PIPELINEABLE);
  REQUIRE(input.original_batch_ids() == std::vector<std::uint64_t>{original_id});
  REQUIRE(input.receipt().decode_attached.size() == 1);

  sirius_physical_dynamic_filter endpoint(output_types(), 8, consumer);
  auto output = endpoint.execute(input, stream);
  REQUIRE(dynamic_cast<scan_output_operator_data*>(output.get()) == nullptr);
  auto const& batches = dynamic_cast<pipelineable_operator_data const&>(*output).get_data_batches();
  REQUIRE(batches.size() == 1);
  auto lease = batches.front()->to_read_only();
  REQUIRE(sirius::get_cudf_table_view(lease).num_rows() == 2);
  REQUIRE_FALSE(
    gate_test_access::filter_keep_ratio(gate(*consumer), {filter.get(), 0}, 1).has_value());
  REQUIRE(gate_test_access::needs_combined_sample(gate(*consumer), channel->snapshot()));
}

TEST_CASE("endpoint commits no batch when a later batch fails",
          "[dynamic_filter][consumer_integration][consumer_commit]")
{
  auto* space = sirius::test::operator_utils::get_default_gpu_space();
  rmm::cuda_stream stream;
  auto channel  = std::make_shared<sirius_dynamic_filter_set>();
  auto producer = channel->register_producer({0});
  auto filter   = std::make_shared<counting_filter>(stream);
  REQUIRE(producer.push_filter(0, filter));
  auto consumer =
    std::make_shared<dynamic_filter_consumer>(channel, consumer_config{}, std::vector<binding>{});
  sirius_physical_dynamic_filter endpoint(output_types(), 16, consumer);
  pipelineable_operator_data input(
    {sequence_batch(*space, stream), sequence_batch(*space, stream)});
  input.prepare_for_processing(space, stream);
  filter->fail_on_call = 2;
  REQUIRE_THROWS_AS(endpoint.execute(input, stream), std::bad_alloc);
  REQUIRE(filter->calls == 2);
  REQUIRE_FALSE(
    gate_test_access::filter_keep_ratio(gate(*consumer), {filter.get(), 0}, 1).has_value());
  REQUIRE(gate_test_access::needs_combined_sample(gate(*consumer), channel->snapshot()));
  filter->fail_on_call = 0;
  auto output          = endpoint.execute(input, stream);
  REQUIRE(dynamic_cast<pipelineable_operator_data const&>(*output).get_data_batches().size() == 2);
  auto const ratio = gate_test_access::filter_keep_ratio(gate(*consumer), {filter.get(), 0}, 1);
  REQUIRE(ratio.has_value());
  REQUIRE(*ratio == Approx(0.25));
}

TEMPLATE_TEST_CASE("cached provider histories remain local to each chunk through coalescing",
                   "[dynamic_filter][consumer_integration][decode_history][prefetch_outcome]",
                   sirius::compressed_device_representation,
                   sirius::compressed_host_representation)
{
  auto manager               = sirius::test::operator_utils::initialize_memory_manager();
  auto* space                = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  constexpr auto source_tier = std::is_same_v<TestType, sirius::compressed_host_representation>
                                 ? cucascade::memory::Tier::HOST
                                 : cucascade::memory::Tier::GPU;
  auto* source_space         = manager->get_memory_space(source_tier, 0);
  REQUIRE(space != nullptr);
  REQUIRE(source_space != nullptr);
  rmm::cuda_stream stream;
  auto channel  = std::make_shared<sirius_dynamic_filter_set>();
  auto producer = channel->register_producer({0});
  auto filter   = std::make_shared<counting_filter>(stream);
  REQUIRE(producer.push_filter(0, filter));
  auto consumer = std::make_shared<dynamic_filter_consumer>(
    channel, consumer_config{}, std::vector<binding>{{0, 0}});
  train_for_decode(*consumer, *filter);
  sirius::scan_manager::pinned_entry entry;
  entry.cache_info.column_ids.emplace_back(0);
  entry.cache_info.names.push_back("key");
  entry.memory_space = source_space;
  entry.tier         = source_tier;
  entry.num_rows     = 16;
  for (int i = 0; i < 2; ++i) {
    if constexpr (source_tier == cucascade::memory::Tier::HOST) {
      entry.host_chunks.emplace_back(compressed_representation<TestType>(*source_space));
    } else {
      sirius::device_pin_chunk chunk;
      chunk.memory_space = source_space;
      chunk.compressed   = compressed_representation<TestType>(*source_space);
      entry.device_chunks.push_back(std::move(chunk));
    }
  }
  std::array<std::size_t, 1> columns{0};
  auto const mask = visibility_mask();
  auto provider   = sirius::scan_manager::make_provider_for_pinned_entry(
    std::make_shared<sirius::scan_manager::pinned_entry const>(std::move(entry)),
    columns,
    sirius::scan_manager::cached_scan_plan{.survivor_chunk_indices = {0, 1}},
    {},
    {mask, {}},
    {},
    {},
    false,
    {},
    consumer);
  auto first          = provider->get_next_batch();
  auto const attached = sirius::decompression_pushdown_enabled() ? std::size_t{1} : std::size_t{0};
  REQUIRE(first.decode_attached.size() == attached);
  REQUIRE(first.mvcc_keep_mask.words == mask.words);
  {
    auto lease = first.data->to_read_only();
    REQUIRE(lease.get_current_tier() == source_tier);
    auto const* compressed = dynamic_cast<TestType const*>(lease.get_data());
    REQUIRE(compressed != nullptr);
    REQUIRE(compressed->visibility_mask().words == mask.words);
    REQUIRE(compressed->visibility_mask().row_count == 8);
    // A chunk with nothing to attach keeps its absent decode request.
    auto const& scan = compressed->pushdown_scan();
    REQUIRE((scan ? scan->request().columns[0].membership.size() : 0) == attached);
  }
  REQUIRE(producer.push_filter(0, std::make_shared<counting_filter>(stream)));
  auto second = provider->get_next_batch();
  REQUIRE(second.decode_attached.size() == 2 * attached);
  REQUIRE_FALSE(second.mvcc_keep_mask.has_mask());
  REQUIRE(first.decode_attached.size() == attached);
  queued_provider queued;
  queued.batches.push_back(std::move(first));
  queued.batches.push_back(std::move(second));
  sirius::scan_manager::split_connector connector;
  sirius::scan_manager::load_balancing_scan_batch_coalescer::drain_cached_provider(
    queued, connector, std::stop_token{}, false);
  auto split = connector.get_next_split();
  REQUIRE(split.has_value());
  auto& first_split = dynamic_cast<scan_operator_input&>(**split);
  REQUIRE(first_split.decode_attached.size() == attached);
  REQUIRE(first_split.mvcc_keep_mask.words == mask.words);
  // Prefetch may have decoded this batch before task preparation sees it.
  {
    auto mutable_batch = first_split.get_cached_batch()->to_mutable();
    mutable_batch.set_data(std::make_unique<sirius::decompression_pushdown_batch_representation>(
      sequence_table(2, stream),
      *space,
      stream,
      sirius::pushdown_outcome{.row_filtered = true, .visibility_mask_applied = true}));
  }
  first_split.consumer           = consumer;
  first_split.row_filter_pending = true;
  auto const history             = first_split.decode_attached;
  resident_ingestible ingestible;
  for (int attempt = 0; attempt < 2; ++attempt) {
    first_split.prepare_for_processing(space, stream);
    auto materialized = ingestible.materialize_table(first_split, stream);
    REQUIRE(sirius::test::operator_utils::copy_column_to_host<std::int64_t>(
              materialized.table.view().column(0), stream) == std::vector<std::int64_t>{0, 1});
    REQUIRE(materialized.state == filter_state::ROW_FILTERED);
    REQUIRE_FALSE(first_split.mvcc_keep_mask.has_mask());
    REQUIRE(first_split.decode_attached == history);
    REQUIRE_FALSE(first_split.stolen_table);
    REQUIRE_FALSE(first_split.converted_table_steal_pending);
  }
  auto next = connector.get_next_split();
  REQUIRE(next.has_value());
  REQUIRE(dynamic_cast<scan_operator_input&>(**next).decode_attached.size() == 2 * attached);
}

TEST_CASE("resident GPU tables retain ownership and unapplied visibility",
          "[dynamic_filter][consumer_integration][prefetch_outcome]")
{
  auto* space = sirius::test::operator_utils::get_default_gpu_space();
  rmm::cuda_stream stream;
  auto const needs_carrier_conversion = GENERATE(false, true);
  std::shared_ptr<cudf::table> owner  = sequence_table(8, stream);
  auto batch                          = sirius::make_data_batch_from_view(
    owner->view(), owner, 8 * sizeof(std::int64_t), *space, stream, {});
  scan_operator_input split(batch);
  auto filter                    = std::make_shared<counting_filter>(stream);
  split.decode_attached          = {{filter, 0}};
  split.needs_carrier_conversion = needs_carrier_conversion;
  auto const mask                = visibility_mask();
  bool masked                    = false;
  SECTION("unfiltered shared pin") {}
  SECTION("unconsumed mask")
  {
    masked               = true;
    split.mvcc_keep_mask = mask;
  }
  resident_ingestible ingestible;
  for (int attempt = 0; attempt < 2; ++attempt) {
    split.prepare_for_processing(space, stream);
    REQUIRE_FALSE(split.stolen_table);
    REQUIRE_FALSE(split.converted_table_steal_pending);
    REQUIRE(split.decode_attached == std::vector<applied_entry>{{filter, 0}});
    REQUIRE(split.mvcc_keep_mask.has_mask() == masked);
    if (masked) { REQUIRE(split.mvcc_keep_mask.words == mask.words); }
    {
      auto lease = batch->to_read_only();
      auto view  = sirius::get_cudf_table_view(lease);
      REQUIRE(view.num_rows() == 8);
      REQUIRE(view.column(0).data<std::int64_t>() == owner->view().column(0).data<std::int64_t>());
    }
    auto materialized   = ingestible.materialize_table(split, stream);
    auto const expected = masked ? std::vector<std::int64_t>{0, 1, 2, 3, 4, 5, 6}
                                 : std::vector<std::int64_t>{0, 1, 2, 3, 4, 5, 6, 7};
    REQUIRE(sirius::test::operator_utils::copy_column_to_host<std::int64_t>(
              materialized.table.view().column(0), stream) == expected);
    REQUIRE(materialized.state == filter_state::UNFILTERED);
  }
}

TEST_CASE("prefetched predicate outcomes and decode history survive preparation retries",
          "[dynamic_filter][consumer_integration][decode_history][prefetch_outcome]")
{
  auto* space = sirius::test::operator_utils::get_default_gpu_space();
  rmm::cuda_stream stream;
  auto const enforced                 = GENERATE(false, true);
  auto const rows                     = enforced ? 2 : 8;
  auto columns                        = sequence_table(rows, stream)->release();
  ::cuda::stream_ref const stream_ref = stream;
  columns.push_back(cudf::make_column_from_scalar(
    cudf::numeric_scalar<bool>(true, true, stream_ref), rows, stream_ref));
  auto batch = cucascade::data_batch::make(
    sirius::get_next_batch_id(),
    std::make_unique<sirius::decompression_pushdown_batch_representation>(
      std::make_unique<cudf::table>(std::move(columns)),
      *space,
      stream,
      sirius::pushdown_outcome{.row_filtered           = enforced,
                               .selection_unprofitable = !enforced,
                               .predicate_columns      = {1},
                               .predicates_enforced    = enforced}));
  auto channel  = std::make_shared<sirius_dynamic_filter_set>();
  auto producer = channel->register_producer({0});
  auto filter   = std::make_shared<counting_filter>(stream);
  REQUIRE(producer.push_filter(0, filter));
  auto consumer = std::make_shared<dynamic_filter_consumer>(
    channel, consumer_config{}, std::vector<binding>{{0, 0}});
  scan_operator_input split(batch);
  split.consumer                        = consumer;
  split.decode_attached                 = {{filter, 0}};
  split.needs_carrier_conversion        = true;
  split.pushdown_selection_unprofitable = std::make_shared<std::atomic<bool>>(false);
  resident_ingestible ingestible;
  for (int attempt = 0; attempt < 3; ++attempt) {
    if (attempt == 2) {
      // A representation change after adoption must not erase the split's decode facts.
      auto mut = batch->to_mutable();
      auto table =
        mut.get_data()->cast<cucascade::gpu_table_representation>().release_table(stream);
      mut.set_data(
        std::make_unique<cucascade::gpu_table_representation>(std::move(table), *space, stream));
    }
    split.prepare_for_processing(space, stream);
    auto materialized = ingestible.materialize_table(split, stream);
    REQUIRE(materialized.table.view().num_rows() == rows);
    REQUIRE(materialized.state ==
            (enforced ? filter_state::ROW_FILTERED : filter_state::UNFILTERED));
    REQUIRE(materialized.predicate_columns == std::vector<std::size_t>{1});
    REQUIRE(materialized.predicates_enforced == enforced);
    REQUIRE(split.pushdown_selection_unprofitable->load(std::memory_order_relaxed) == !enforced);
    REQUIRE(split.decode_attached == std::vector<applied_entry>{{filter, 0}});
    REQUIRE_FALSE(split.converted_table_steal_pending);
    REQUIRE_FALSE(split.stolen_table);
    REQUIRE(sirius::test::operator_utils::copy_column_to_host<bool>(
              materialized.table.view().column(1), stream) == std::vector<bool>(rows, true));
  }

  // A later conversion failure must not make already-decoded rows a fresh sample.
  inspected_converter<sirius::compressed_device_representation> converter;
  {
    auto mut = batch->to_mutable();
    mut.set_data(
      compressed_representation<
        inspected_compressed_representation<sirius::compressed_device_representation>>(*space));
  }
  for (int attempt = 0; attempt < 2; ++attempt) {
    REQUIRE_THROWS_AS(split.prepare_for_processing(space, stream), std::bad_alloc);
    REQUIRE(split.decode_attached == std::vector<applied_entry>{{filter, 0}});
    REQUIRE(split.pushdown_row_filtered == enforced);
    REQUIRE(split.pushdown_predicate_columns == std::vector<std::size_t>{1});
    REQUIRE(split.pushdown_predicates_enforced == enforced);
  }
}

TEST_CASE("prefetched row filtering rejects an unconsumed visibility mask on every preparation",
          "[dynamic_filter][consumer_integration][prefetch_outcome]")
{
  auto* space = sirius::test::operator_utils::get_default_gpu_space();
  rmm::cuda_stream stream;
  auto batch = cucascade::data_batch::make(
    sirius::get_next_batch_id(),
    std::make_unique<sirius::decompression_pushdown_batch_representation>(
      sequence_table(2, stream), *space, stream, sirius::pushdown_outcome{.row_filtered = true}));
  scan_operator_input split(batch);
  auto const mask      = visibility_mask();
  split.mvcc_keep_mask = mask;
  for (int attempt = 0; attempt < 2; ++attempt) {
    REQUIRE_THROWS_AS(split.prepare_for_processing(space, stream), std::runtime_error);
    REQUIRE(split.mvcc_keep_mask.words == mask.words);
    REQUIRE_FALSE(split.stolen_table);
    REQUIRE_FALSE(split.converted_table_steal_pending);
  }
}

TEMPLATE_TEST_CASE(
  "fresh compressed retries clear stale membership while preserving static selection",
  "[dynamic_filter][consumer_integration][decode_history]",
  sirius::compressed_device_representation,
  sirius::compressed_host_representation)
{
  using representation       = inspected_compressed_representation<TestType>;
  auto manager               = sirius::test::operator_utils::initialize_memory_manager();
  auto* space                = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  constexpr auto source_tier = std::is_same_v<TestType, sirius::compressed_host_representation>
                                 ? cucascade::memory::Tier::HOST
                                 : cucascade::memory::Tier::GPU;
  auto* source_space         = manager->get_memory_space(source_tier, 0);
  REQUIRE(space != nullptr);
  REQUIRE(source_space != nullptr);
  rmm::cuda_stream stream;
  inspected_converter<TestType> converter;
  auto channel  = std::make_shared<sirius_dynamic_filter_set>();
  auto producer = channel->register_producer({0});
  auto filter   = std::make_shared<counting_filter>(stream);
  REQUIRE(producer.push_filter(0, filter));
  auto consumer = std::make_shared<dynamic_filter_consumer>(
    channel, consumer_config{}, std::vector<binding>{{0, 0}});
  auto rep = compressed_representation<representation>(*source_space);
  sirius::pushdown_request request;
  request.columns.resize(1);
  request.columns[0].range = sirius::decode_range{.lo = 1, .hi = 7};
  auto stale =
    snapshot_membership_probes(decode_selection{channel, 1, {{{filter, 0}, 0, {{filter, 0}}}}}, 1);
  request.columns[0].membership     = std::move(stale.probes[0]);
  request.ranges_cover_whole_filter = true;
  rep->set_pushdown_scan(std::make_shared<sirius::decompression_pushdown_scan>(request));
  auto batch = cucascade::data_batch::make(sirius::get_next_batch_id(), std::move(rep));
  {
    auto lease = batch->to_read_only();
    REQUIRE(lease.get_current_tier() == source_tier);
    REQUIRE(dynamic_cast<TestType const*>(lease.get_data()) != nullptr);
  }
  scan_operator_input split(batch);
  split.consumer                 = consumer;
  split.decode_attached          = {{filter, 0}};
  split.needs_carrier_conversion = true;
  SECTION("compaction latch survives removal of the last row selection source")
  {
    train_for_decode(*consumer, *filter);
    split.pushdown_selection_unprofitable = std::make_shared<std::atomic<bool>>(true);
    // Each attempt sees the latch and attaches nothing, so dropping the request loses no state.
    for (int attempt = 0; attempt < 2; ++attempt) {
      REQUIRE_THROWS_AS(split.prepare_for_processing(space, stream), std::bad_alloc);
      REQUIRE(split.decode_attached.empty());
      REQUIRE(split.pushdown_selection_unprofitable->load());
      auto lease             = batch->to_read_only();
      auto const& compressed = lease.get_data()->template cast<representation>();
      // Without an equality to keep, no request remains, and the decode selects no rows.
      REQUIRE(compressed.pushdown_scan() == nullptr);
    }
  }
  SECTION("fresh sampling preserves static predicates and visibility")
  {
    auto const mask      = visibility_mask();
    split.mvcc_keep_mask = mask;
    {
      auto mutable_batch = batch->to_mutable();
      mutable_batch.get_data()->template cast<representation>().set_visibility_mask(
        {mask.words, 8});
    }
    REQUIRE_THROWS_AS(split.prepare_for_processing(space, stream), std::bad_alloc);
    REQUIRE(split.decode_attached.empty());
    auto lease             = batch->to_read_only();
    auto const& compressed = lease.get_data()->template cast<representation>();
    auto const& refreshed  = compressed.pushdown_scan()->request();
    REQUIRE(refreshed.columns[0].membership.empty());
    REQUIRE(refreshed.columns[0].range->lo == 1);
    REQUIRE(refreshed.columns[0].range->hi == 7);
    REQUIRE(refreshed.ranges_cover_whole_filter);
    REQUIRE_FALSE(refreshed.row_selection_disabled);
    REQUIRE(compressed.visibility_mask().words == mask.words);
    REQUIRE(compressed.visibility_mask().row_count == 8);
    REQUIRE(split.mvcc_keep_mask.words == mask.words);
  }
  SECTION("fresh retry refreshes while converted retry and steal retain history")
  {
    REQUIRE_THROWS_AS(split.prepare_for_processing(space, stream), std::bad_alloc);
    REQUIRE(split.decode_attached.empty());
    {
      auto lease             = batch->to_read_only();
      auto const& compressed = lease.get_data()->template cast<representation>();
      auto const& refreshed  = compressed.pushdown_scan()->request();
      REQUIRE(refreshed.columns[0].membership.empty());
      REQUIRE(refreshed.columns[0].range->lo == 1);
      REQUIRE(refreshed.columns[0].range->hi == 7);
      REQUIRE(refreshed.ranges_cover_whole_filter);
      REQUIRE_FALSE(refreshed.row_selection_disabled);
    }
    train_for_decode(*consumer, *filter);
    REQUIRE_THROWS_AS(split.prepare_for_processing(space, stream), std::bad_alloc);
    auto const attached =
      sirius::decompression_pushdown_enabled() ? std::size_t{1} : std::size_t{0};
    REQUIRE(split.decode_attached.size() == attached);
    {
      auto mutable_batch = batch->to_mutable();
      mutable_batch.get_data()->template cast<representation>().refuse_conversion = false;
    }
    split.prepare_for_processing(space, stream);
    REQUIRE(split.converted_table_steal_pending);
    REQUIRE(split.decode_attached.size() == attached);
    split.prepare_for_processing(space, stream);
    REQUIRE(split.decode_attached.size() == attached);
    REQUIRE_THROWS_AS(split.transactionally_steal_converted_table(
                        1,
                        [](cudf::table_view) -> scan_operator_input::converted_column_replacements {
                          throw rmm::out_of_memory{"injected carrier conversion failure"};
                        },
                        stream),
                      rmm::out_of_memory);
    REQUIRE(split.decode_attached.size() == attached);
    auto stolen = split.transactionally_steal_converted_table(
      1,
      [](cudf::table_view) { return scan_operator_input::converted_column_replacements(1); },
      stream);
    REQUIRE(stolen != nullptr);
    REQUIRE(split.decode_attached.size() == attached);
  }
}

TEST_CASE("endpoint host allocation failures discard pending observations",
          "[.][host_allocation_fault][consumer_integration][consumer_commit]")
{
  using fault = sirius::test::scoped_host_allocation_fault;
  REQUIRE(fault::available());
  auto* space = sirius::test::operator_utils::get_default_gpu_space();
  rmm::cuda_stream stream;
  std::size_t allocations          = 0;
  std::size_t failures_after_probe = 0;
  for (std::size_t ordinal = 0; ordinal <= allocations; ++ordinal) {
    auto channel  = std::make_shared<sirius_dynamic_filter_set>();
    auto producer = channel->register_producer({0});
    auto filter   = std::make_shared<counting_filter>(stream);
    REQUIRE(producer.push_filter(0, filter));
    auto consumer =
      std::make_shared<dynamic_filter_consumer>(channel, consumer_config{}, std::vector<binding>{});
    sirius_physical_dynamic_filter endpoint(output_types(), 16, consumer);
    pipelineable_operator_data input(
      {sequence_batch(*space, stream), sequence_batch(*space, stream)});
    input.prepare_for_processing(space, stream);
    bool failed = false;
    std::unique_ptr<sirius::op::operator_data> output;
    std::size_t count = 0;
    {
      fault injection(fault::every_allocation_scope, ordinal);
      try {
        output = endpoint.execute(input, stream);
      } catch (std::bad_alloc const&) {
        failed = true;
      }
      count = injection.stop();
    }
    if (ordinal == 0) { allocations = count; }
    if (failed) {
      failures_after_probe += filter->calls != 0;
      REQUIRE_FALSE(
        gate_test_access::filter_keep_ratio(gate(*consumer), {filter.get(), 0}, 1).has_value());
      REQUIRE(gate_test_access::needs_combined_sample(gate(*consumer), channel->snapshot()));
      REQUIRE_NOTHROW(endpoint.execute(input, stream));
    } else {
      REQUIRE(output != nullptr);
    }
  }
  REQUIRE(allocations > 0);
  REQUIRE(failures_after_probe > 0);
}

TEST_CASE("endpoint pairs each lease with its own batch when the input holds null batches",
          "[dynamic_filter][consumer_integration]")
{
  auto* space = sirius::test::operator_utils::get_default_gpu_space();
  rmm::cuda_stream stream;
  auto channel  = std::make_shared<sirius_dynamic_filter_set>();
  auto producer = channel->register_producer({0});
  auto filter   = std::make_shared<counting_filter>(stream);
  REQUIRE(producer.push_filter(0, filter));
  auto consumer =
    std::make_shared<dynamic_filter_consumer>(channel, consumer_config{}, std::vector<binding>{});
  sirius_physical_dynamic_filter endpoint(output_types(), 8, consumer);
  // The empty batch has no rows to filter, so the endpoint forwards that exact batch.
  auto empty = sirius::make_data_batch(sequence_table(0, stream), *space, stream, {});
  pipelineable_operator_data input({nullptr, empty, nullptr, sequence_batch(*space, stream)});
  auto output         = endpoint.execute(input, stream);
  auto const& batches = dynamic_cast<pipelineable_operator_data const&>(*output).get_data_batches();
  REQUIRE(batches.size() == 2);
  REQUIRE(batches[0] == empty);
  REQUIRE(batches[1] != nullptr);
  auto lease = batches[1]->to_read_only();
  REQUIRE(sirius::get_cudf_table_view(lease).num_rows() == 2);
  REQUIRE(filter->calls == 1);
}

TEST_CASE("application keeps the source batch from mutation until its submitted reads complete",
          "[dynamic_filter][consumer_integration][consumer_retirement]")
{
  using sirius::op::scan::detail::consumer_test_access;
  using sirius::op::scan::detail::execution_fault;
  auto* space = sirius::test::operator_utils::get_default_gpu_space();
  rmm::cuda_stream stream;
  auto channel  = std::make_shared<sirius_dynamic_filter_set>();
  auto producer = channel->register_producer({0});
  auto filter   = std::make_shared<counting_filter>(stream);
  REQUIRE(producer.push_filter(0, filter));
  auto consumer =
    std::make_shared<dynamic_filter_consumer>(channel, consumer_config{}, std::vector<binding>{});
  // Measured ratios leave nothing to sample, so the application reads no counts back.
  train_for_decode(*consumer, *filter);
  auto batch = sequence_batch(*space, stream);
  stream.synchronize();
  auto source                   = batch->to_read_only();
  std::vector<binding> bindings = identity_bindings(1);
  auto const view               = sirius::get_cudf_table_view(source);
  auto program =
    consumer->prepare({.source = view, .bindings = bindings, .device_id = space->get_device_id()});
  REQUIRE(program.has_value());

  // Releases the stalled stream even when a check below fails, so later work cannot hang on it.
  struct stall_release {
    stall_release()                                = default;
    stall_release(stall_release const&)            = delete;
    stall_release& operator=(stall_release const&) = delete;
    ~stall_release() { consumer_test_access::release_stalled_streams(); }
  } const release;
  auto result = consumer_test_access::apply(*consumer,
                                            std::move(*program),
                                            std::move(source),
                                            stream,
                                            space->get_default_allocator(),
                                            execution_fault::STALLED_COMPLETION);
  // A wait for the stream inside apply would have returned only once the stall's deadline passed.
  REQUIRE_FALSE(consumer_test_access::stalled_stream_timed_out());
  REQUIRE_FALSE(result.counts_read_back());
  REQUIRE_FALSE(consumer_test_access::storage(result).has_source);
  // The source lease is gone and the stream still holds the reads, so only a pending reader event
  // can make the non-blocking mutable path decline.
  REQUIRE_FALSE(batch->try_to_mutable().has_value());
  consumer_test_access::release_stalled_streams();
  stream.synchronize();
  REQUIRE_FALSE(consumer_test_access::stalled_stream_timed_out());
  REQUIRE(batch->try_to_mutable().has_value());
  REQUIRE(result.take_output()->num_rows() == 2);
  consumer->commit(std::move(result));
}

TEST_CASE("decode probe shaping rejects a selection step outside the decoded slots",
          "[dynamic_filter][consumer_integration][decode_history]")
{
  rmm::cuda_stream stream;
  auto channel = std::make_shared<sirius_dynamic_filter_set>();
  auto filter  = std::make_shared<counting_filter>(stream);
  REQUIRE_THROWS_AS(
    snapshot_membership_probes(decode_selection{channel, 1, {{{filter, 0}, 1, {{filter, 0}}}}}, 1),
    std::invalid_argument);
  auto const shaped =
    snapshot_membership_probes(decode_selection{channel, 1, {{{filter, 0}, 0, {{filter, 0}}}}}, 1);
  REQUIRE(shaped.probes.size() == 1);
  REQUIRE(shaped.probes[0].size() == 1);
}

TEST_CASE("a deferring scan decodes whole compressed chunks and its endpoint filters them",
          "[late_mat][dynamic_filter][consumer_integration][decode_history][.late_mat_fused]")
{
  if (!sirius::late_mat::late_mat_enabled() || !sirius::decompression_pushdown_enabled()) {
    SKIP("needs SIRIUS_EXP_LATE_MAT=1 and SIRIUS_EXP_FUSED_SCAN_FILTER=1 in the environment");
  }
  constexpr cudf::size_type kRows = 1024;
  auto* space                     = sirius::test::operator_utils::get_default_gpu_space();
  rmm::cuda_stream stream;
  auto const mr = space->get_default_allocator();

  // Column 0 is the join key (row % 100) and column 1 the deferred payload (row).
  std::vector<std::int32_t> keys(kRows);
  std::vector<std::int32_t> payload(kRows);
  for (cudf::size_type row = 0; row < kRows; ++row) {
    keys[static_cast<std::size_t>(row)]    = row % 100;
    payload[static_cast<std::size_t>(row)] = row;
  }
  auto const upload = [&](std::vector<std::int32_t> const& host) {
    auto column = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32}, kRows, cudf::mask_state::UNALLOCATED, stream, mr);
    cudaMemcpyAsync(column->mutable_view().data<std::int32_t>(),
                    host.data(),
                    host.size() * sizeof(std::int32_t),
                    cudaMemcpyHostToDevice,
                    stream.value());
    return column;
  };
  std::vector<std::unique_ptr<cudf::column>> source_columns;
  source_columns.push_back(upload(keys));
  source_columns.push_back(upload(payload));
  cudf::table const source(std::move(source_columns));
  auto blob   = std::make_shared<sirius::compressed_device_blob>();
  blob->table = simpatico::compress_with_plan(source.view(),
                                              "input -> bitpack -> chunk_min, chunk_count, "
                                              "chunk_bits, packed\n---\ninput -> bitpack -> "
                                              "chunk_min, chunk_count, chunk_bits, packed\n",
                                              stream,
                                              mr);
  stream.synchronize();

  auto handle = std::make_shared<sirius::late_mat::pin_entry_handle>("deferring_compressed", 1);
  auto entry  = std::make_shared<sirius::scan_manager::pinned_entry>();
  entry->cache_info.column_ids.emplace_back(0);
  entry->cache_info.column_ids.emplace_back(1);
  entry->cache_info.names = {"key", "payload"};
  entry->tier             = cucascade::memory::Tier::GPU;
  entry->memory_space     = space;
  entry->num_rows         = kRows;
  entry->late_mat_handle  = handle;
  sirius::device_pin_chunk chunk;
  chunk.memory_space = space;
  chunk.compressed   = std::make_shared<sirius::compressed_device_representation>(
    *space,
    blob,
    std::vector<std::string>{"key", "payload"},
    source.alloc_size(),
    source.alloc_size(),
    kRows);
  entry->device_chunks.push_back(std::move(chunk));
  std::shared_ptr<sirius::scan_manager::pinned_entry const> const pinned = entry;
  handle->set_entry(pinned);

  // F keeps keys below 10; the gate is ACTIVE, so an attaching consumer offers F to the decode.
  auto channel  = std::make_shared<sirius_dynamic_filter_set>();
  auto producer = channel->register_producer({0});
  std::vector<std::int32_t> build_keys(10);
  std::iota(build_keys.begin(), build_keys.end(), 0);
  auto const build = [&] {
    auto column = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32}, 10, cudf::mask_state::UNALLOCATED, stream, mr);
    cudaMemcpyAsync(column->mutable_view().data<std::int32_t>(),
                    build_keys.data(),
                    build_keys.size() * sizeof(std::int32_t),
                    cudaMemcpyHostToDevice,
                    stream.value());
    return column;
  }();
  auto const filter =
    std::make_shared<sirius::op::sirius_dynamic_in_list_filter>(build->view(), stream, mr);
  stream.synchronize();
  REQUIRE(producer.push_filter(0, filter));
  duckdb::vector<sirius::logical_type> const types(
    2, sirius::logical_type::make(sirius::type_id::INTEGER));
  auto const ingestible = std::make_shared<leading_identity_ingestible>(2);
  auto consumer         = std::make_shared<dynamic_filter_consumer>(
    channel, consumer_config{}, scan_decode_bindings(*ingestible, types.size()));
  sirius_gpu_scan_operator scan(types, kRows, ingestible, 1, nullptr, nullptr, consumer);
  REQUIRE(consumer->decode_bindings().size() == 2);
  gate(*consumer).record_keep_ratio(100, 10, 1);
  gate(*consumer).record_filter_keep_ratio({filter.get(), 0}, 0.1, 1);
  REQUIRE(consumer->select_for_decode(consumer->decode_bindings()).steps.size() == 1);

  sirius_physical_dynamic_filter endpoint(types, kRows, consumer);
  sirius::late_mat::column_origin origin;
  origin.handle     = handle;
  origin.column_pos = 1;
  origin.generation = handle->generation();
  std::vector<cudf::data_type> const schema(2, cudf::data_type{cudf::type_id::INT32});
  REQUIRE(sirius::planner::install_deferral(
    scan, endpoint, sirius::late_mat::make_defer_pair(schema, {1}, schema, {1}, {origin})));

  // Serve the chunk the way prepare_for_query does, with the given attachment consumer.
  std::array<std::size_t, 2> const columns{0, 1};
  auto const serve = [&](std::shared_ptr<dynamic_filter_consumer> attaching) {
    auto provider = sirius::scan_manager::make_provider_for_pinned_entry(
      pinned,
      columns,
      sirius::scan_manager::cached_scan_plan{.survivor_chunk_indices = {0}},
      {},
      {},
      {},
      {},
      false,
      {},
      std::move(attaching));
    // Admitted for the scan's contract, as prepare_for_query stamps a pin's provider.
    provider->contract_id            = 1;
    provider->validation.identity    = provider->validation.layout =
      provider->validation.structure = {true, true};
    sirius::scan_manager::load_balancing_scan_batch_coalescer::drain_cached_provider(
      *provider, scan.get_split_connector(), std::stop_token{}, false, 1, 0);
  };

  SECTION("an attaching consumer compacts the chunk, which the scan refuses to address")
  {
    // The wiring before decode_consumer(): the scan's own consumer reaches the decode.
    serve(scan.consumer());
    auto input     = scan.get_next_task_input_data();
    auto& split    = dynamic_cast<scan_operator_input&>(*input);
    split.consumer = scan.consumer();
    split.prepare_for_processing(space, stream);
    REQUIRE(split.decode_attached.size() == 1);
    // Membership compaction alone does not report pushdown_row_filtered.
    REQUIRE_FALSE(split.pushdown_row_filtered);
    auto const decoded_rows =
      split.stolen_table
        ? split.stolen_table->num_rows()
        : sirius::get_cudf_table_view(split.get_cached_batch()->to_read_only()).num_rows();
    REQUIRE(decoded_rows < kRows);
    REQUIRE_THROWS_WITH(
      scan.execute(split, stream),
      Catch::Matchers::ContainsSubstring(
        "the decode removed rows of a scan whose deferral addresses whole chunks"));
  }
  SECTION("the decode consumer leaves whole chunks for the endpoint")
  {
    REQUIRE(scan.decode_consumer() == nullptr);
    serve(scan.decode_consumer());
    auto input  = scan.get_next_task_input_data();
    auto& split = dynamic_cast<scan_operator_input&>(*input);
    REQUIRE(split.consumer == nullptr);
    REQUIRE(split.decode_attached.empty());
    split.prepare_for_processing(space, stream);
    REQUIRE(split.decode_attached.empty());
    REQUIRE_FALSE(split.pushdown_row_filtered);

    auto scanned         = scan.execute(split, stream);
    auto* receipt_output = dynamic_cast<scan_output_operator_data*>(scanned.get());
    REQUIRE(receipt_output != nullptr);
    REQUIRE(receipt_output->receipt().decode_attached.empty());
    std::vector<std::uint64_t> expected_rowids(kRows);
    std::iota(expected_rowids.begin(), expected_rowids.end(), std::uint64_t{0});
    {
      auto lease = receipt_output->get_data_batches().front()->to_read_only();
      auto view  = sirius::get_cudf_table_view(lease);
      REQUIRE(view.num_rows() == kRows);
      REQUIRE(to_host<std::uint64_t>(view.column(1), stream) == expected_rowids);
    }

    // The endpoint keeps exactly F's survivors, each beside its own rowid.
    receipt_output->prepare_for_processing(space, stream);
    auto filtered = endpoint.execute(*receipt_output, stream);
    auto const leases =
      dynamic_cast<pipelineable_operator_data const&>(*filtered).get_read_only_batches();
    REQUIRE(leases.size() == 1);
    auto const view = sirius::get_cudf_table_view(leases.front());
    std::vector<std::int32_t> expected_keys;
    std::vector<std::uint64_t> expected_survivors;
    for (cudf::size_type row = 0; row < kRows; ++row) {
      if (row % 100 < 10) {
        expected_keys.push_back(row % 100);
        expected_survivors.push_back(static_cast<std::uint64_t>(row));
      }
    }
    REQUIRE(to_host<std::int32_t>(view.column(0), stream) == expected_keys);
    REQUIRE(to_host<std::uint64_t>(view.column(1), stream) == expected_survivors);
  }
}
