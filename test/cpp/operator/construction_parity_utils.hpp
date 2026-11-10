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

/**
 * @file construction_parity_utils.hpp
 * @brief The parity harness for `build_filter_construction`: builds a join's publish plan over
 * chosen key types, constructs its filters from any split of a build (in any order, from any number
 * of threads), and compares the result with the same build as one batch (oracle O1) and with the
 * legacy one-shot builder `publish_dynamic_filters` (oracle O2).
 */

#pragma once

#include "data/data_batch_utils.hpp"
#include "operator/dynamic_filter_accumulation_test_utils.hpp"
#include "operator/dynamic_filter_column_test_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>
#include <op/dynamic_filter/build_filter_construction.hpp>
#include <op/dynamic_filter/construction_plan.hpp>
#include <op/dynamic_filter/detail/accumulated_bloom_builder.hpp>
#include <op/dynamic_filter/dynamic_filter_publish_plan.hpp>
#include <op/dynamic_filter/dynamic_filter_publisher.hpp>
#include <op/dynamic_filter/sirius_dynamic_filter.hpp>

#include <cstddef>
#include <cstring>
#include <future>
#include <memory>
#include <optional>
#include <vector>

namespace sirius::test::construction {

namespace op  = sirius::op;
namespace acc = sirius::test::accumulation;

/**
 * @brief Test access to `build_filter_construction`'s tickets.
 */
struct test_access {
  /// Shortens exact key @p key's slice by one row, so its last valid row overflows.
  static void shrink_slice(op::build_filter_construction::contribution_ticket& ticket,
                           std::size_t key)
  {
    REQUIRE(ticket.slices.at(key).rows > 0);
    --ticket.slices.at(key).rows;
  }

  /// Points the ticket at a bound slot past the end, so the bounds accumulator rejects it.
  static void corrupt_batch(op::build_filter_construction::contribution_ticket& ticket)
  {
    ticket.batch_index = static_cast<std::size_t>(1) << 40;
  }
};

/// The smallest L2 of GPU 0, as the one-shot builder reads it.
[[nodiscard]] inline std::size_t l2_bytes()
{
  int l2 = 0;
  REQUIRE(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0) == cudaSuccess);
  return static_cast<std::size_t>(l2);
}

/**
 * @brief One join: its publish plan over key `i` at build column `i`, and its probe channel.
 */
struct join {
  std::shared_ptr<op::sirius_dynamic_filter_set> channel =
    std::make_shared<op::sirius_dynamic_filter_set>();
  op::dynamic_filter_publish_plan plan;
  op::construction_limits limits;

  join(acc::fixture const& fixture,
       std::vector<cudf::data_type> const& recorded,
       std::size_t l2,
       bool zone_maps,
       double inlist_max_l2_fraction = 0.125)
  {
    using plan_type = op::dynamic_filter_publish_plan;
    std::vector<plan_type::admitted_key> keys;
    std::vector<plan_type::key_binding> bindings;
    for (std::size_t index = 0; index < recorded.size(); ++index) {
      keys.push_back({.planner_condition_index = index,
                      .build_key_ordinal       = static_cast<cudf::size_type>(index),
                      .probe_key_ordinal       = static_cast<cudf::size_type>(index),
                      .storage_type            = recorded[index],
                      .probe_storage_type      = recorded[index]});
      bindings.push_back({.admitted_key_index   = index,
                          .channel_push_ordinal = index,
                          .probe_storage_type   = recorded[index]});
    }
    plan = plan_type{
      std::move(keys),
      {{channel, op::dynamic_filter_route_class::scan, true, std::move(bindings)}},
      fixture.spaces,
      {.emit_zone_map_filters = zone_maps, .inlist_max_l2_fraction = inlist_max_l2_fraction}};
    limits = {.l2_bytes                  = l2,
              .inlist_max_l2_fraction    = inlist_max_l2_fraction,
              .emit_zone_map_filters     = zone_maps,
              .domain_coverage_threshold = plan.domain_coverage_threshold(),
              .max_bloom_bytes_per_gpu   = plan.max_bloom_bytes_per_gpu()};
  }
};

/// Build batches on GPU 0: each a table of the key columns, in key order.
[[nodiscard]] inline std::vector<acc::batch_ptr> make_batches(
  acc::fixture const& fixture, std::vector<std::vector<cudf::column_view>> const& tables)
{
  std::vector<acc::batch_ptr> batches;
  auto& space       = fixture.gpu(0);
  auto const stream = space.acquire_stream();
  for (auto const& columns : tables) {
    auto table = std::make_unique<cudf::table>(
      cudf::table_view{columns}, stream, space.get_default_allocator());
    batches.push_back(sirius::make_data_batch(
      std::move(*table), space, stream, sirius::telemetry::batch_telemetry_info{}));
  }
  stream.sync();
  return batches;
}

/**
 * @brief The plan and its constructed filters for one build.
 */
struct constructed {
  op::construction_plan plan;
  op::build_filter_construction::constructed_filters filters;
};

/**
 * @brief Constructs @p subject's filters from @p batches through `build_filter_construction`.
 *
 * @param order The order batches contribute in, as indices into the inventory; empty is inventory
 * order
 * @param threads Contributing threads, each with its own stream; batches are dealt round-robin
 */
[[nodiscard]] inline constructed construct(acc::fixture const& fixture,
                                           join const& subject,
                                           std::vector<acc::batch_ptr> const& batches,
                                           std::vector<std::size_t> order = {},
                                           std::size_t threads            = 1)
{
  auto certified = acc::fixture::certify(batches);
  REQUIRE(std::holds_alternative<op::complete_build_inventory>(certified));
  auto const& inventory = std::get<op::complete_build_inventory>(certified);
  auto plan             = op::plan_construction(inventory.facts(),
                                    subject.plan.admitted_keys(),
                                    op::binding_facts_of(subject.plan),
                                    subject.limits);
  auto construction = op::build_filter_construction::try_create(plan, inventory, fixture.spaces);
  REQUIRE(construction);
  if (order.empty()) {
    for (std::size_t index = 0; index < batches.size(); ++index) {
      order.push_back(index);
    }
  }
  // The inventory sorts batches by ID; contributions find their own index.
  auto const index_of = [&](acc::batch_ptr const& batch) {
    return static_cast<std::size_t>(inventory.find(batch->get_batch_id()) -
                                    inventory.batches().data());
  };
  std::vector<std::unique_ptr<rmm::cuda_stream>> streams;
  for (std::size_t thread = 0; thread < threads; ++thread) {
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{0}};
    streams.push_back(std::make_unique<rmm::cuda_stream>(rmm::cuda_stream::flags::non_blocking));
  }
  auto const contribute_on = [&](std::size_t thread) {
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{0}};
    ::cuda::stream_ref const stream{streams[thread]->value()};
    for (std::size_t position = thread; position < order.size(); position += threads) {
      auto const& batch = batches[order[position]];
      auto source       = batch->to_read_only();
      auto const view   = sirius::get_cudf_table_view(source);
      auto ticket       = construction->reserve(
        index_of(batch), rmm::cuda_device_id{0}, op::batch_key_facts::of(view));
      if (!ticket) { throw std::runtime_error{"claim rejected"}; }
      if (!construction->contribute(
            *ticket, view, stream, fixture.gpu(0).get_default_allocator())) {
        throw std::runtime_error{"no storage on GPU 0"};
      }
      stream.sync();
    }
  };
  std::vector<std::future<void>> running;
  for (std::size_t thread = 0; thread < threads; ++thread) {
    running.push_back(std::async(std::launch::async, contribute_on, thread));
  }
  for (auto& thread : running) {
    thread.get();
  }
  rmm::cuda_set_device_raii guard{rmm::cuda_device_id{0}};
  auto finished = construction->finish({.space   = fixture.gpu(0),
                                        .stream  = fixture.publication_stream(0),
                                        .scratch = fixture.gpu(0).get_default_allocator()});
  REQUIRE(finished);
  return {std::move(plan), std::move(*finished)};
}

/**
 * @brief The filters the legacy one-shot builder publishes for @p subject over @p whole, by key.
 */
struct one_shot {
  std::vector<std::shared_ptr<op::sirius_dynamic_filter const>> membership;
  std::vector<std::shared_ptr<op::sirius_dynamic_filter const>> zone_map;
  op::dynamic_filter_publication_outcome outcome;
};

[[nodiscard]] inline one_shot publish_one_shot(acc::fixture const& fixture,
                                               std::vector<cudf::data_type> const& recorded,
                                               cudf::table_view const& whole,
                                               std::size_t l2,
                                               bool zone_maps,
                                               double inlist_max_l2_fraction = 0.125)
{
  join const subject{fixture, recorded, l2, zone_maps, inlist_max_l2_fraction};
  std::vector<std::size_t> columns;
  for (std::size_t index = 0; index < recorded.size(); ++index) {
    columns.push_back(index);
  }
  std::vector<op::sirius_dynamic_filter_set::producer> producers;
  producers.push_back(subject.channel->register_producer(columns));
  subject.channel->freeze_registration();
  rmm::cuda_set_device_raii guard{rmm::cuda_device_id{0}};
  auto const stream = fixture.publication_stream(0);
  one_shot result;
  result.outcome = op::publish_dynamic_filters(subject.plan, whole, stream, producers);
  stream.sync();
  producers.front().finish(result.outcome.filters_pushed != 0
                             ? op::sirius_dynamic_filter_set::completion::PUBLISHED
                             : op::sirius_dynamic_filter_set::completion::SKIPPED);
  result.membership.resize(recorded.size());
  result.zone_map.resize(recorded.size());
  auto const published = subject.channel->snapshot();  // entries() views the snapshot
  for (auto const& entry : published.entries()) {
    auto& slot = entry.filter->kind() == op::sirius_dynamic_filter_kind::ZONE_MAP
                   ? result.zone_map[entry.column_index]
                   : result.membership[entry.column_index];
    slot       = entry.filter;
  }
  return result;
}

/// The bytes of a scalar's value: its storage, or its string.
[[nodiscard]] inline std::vector<std::byte> scalar_bytes(cudf::scalar const& bound,
                                                         ::cuda::stream_ref stream)
{
  REQUIRE(bound.is_valid(stream));
  if (bound.type().id() == cudf::type_id::STRING) {
    auto const text = static_cast<cudf::string_scalar const&>(bound).to_string(stream);
    std::vector<std::byte> bytes(text.size());
    std::memcpy(bytes.data(), text.data(), text.size());
    return bytes;
  }
  auto const column = cudf::make_column_from_scalar(bound, 1, stream);
  std::vector<std::byte> bytes(cudf::size_of(bound.type()));
  REQUIRE(
    cudaMemcpyAsync(
      bytes.data(), column->view().head(), bytes.size(), cudaMemcpyDeviceToHost, stream.get()) ==
    cudaSuccess);
  stream.sync();
  return bytes;
}

/**
 * @brief Requires two membership filters to be the same filter: kind, size, domain, Bloom bits, and
 * the mask over @p probe.
 *
 * @p stream is the stream both filters were built on, or follows it.
 */
inline void require_same_membership(op::sirius_dynamic_filter const* left,
                                    op::sirius_dynamic_filter const* right,
                                    cudf::column_view const& probe,
                                    ::cuda::stream_ref stream)
{
  REQUIRE((left == nullptr) == (right == nullptr));
  if (left == nullptr) { return; }
  REQUIRE(left->kind() == right->kind());
  auto const* left_small  = dynamic_cast<op::sirius_dynamic_small_in_list_filter const*>(left);
  auto const* right_small = dynamic_cast<op::sirius_dynamic_small_in_list_filter const*>(right);
  REQUIRE((left_small == nullptr) == (right_small == nullptr));
  if (left_small) {
    CHECK(left_small->size() == right_small->size());
    CHECK(left_small->domain() == right_small->domain());
  }
  auto const* left_set  = dynamic_cast<op::sirius_dynamic_in_list_filter const*>(left);
  auto const* right_set = dynamic_cast<op::sirius_dynamic_in_list_filter const*>(right);
  REQUIRE((left_set == nullptr) == (right_set == nullptr));
  if (left_set) {
    CHECK(left_set->size() == right_set->size());
    CHECK(left_set->domain() == right_set->domain());
  }
  auto const* left_bloom  = dynamic_cast<op::sirius_dynamic_bloom_filter const*>(left);
  auto const* right_bloom = dynamic_cast<op::sirius_dynamic_bloom_filter const*>(right);
  REQUIRE((left_bloom == nullptr) == (right_bloom == nullptr));
  if (left_bloom) {
    CHECK(left_bloom->domain() == right_bloom->domain());
    using builder         = op::detail::accumulated_bloom_builder;
    auto const left_bits  = builder::inspect_replica(*left_bloom, rmm::cuda_device_id{0}, stream);
    auto const right_bits = builder::inspect_replica(*right_bloom, rmm::cuda_device_id{0}, stream);
    REQUIRE(left_bits);
    REQUIRE(right_bits);
    CHECK(left_bits->bytes == right_bits->bytes);
  }
  auto const mask = [&](op::sirius_dynamic_filter const* filter) {
    auto const* applicable = dynamic_cast<op::sirius_mask_applicable const*>(filter);
    REQUIRE(applicable != nullptr);
    auto const column =
      applicable->compute_mask(probe, 0, stream, cudf::get_current_device_resource_ref());
    REQUIRE(column);
    return sirius::test::operator_utils::copy_column_to_host<bool>(column->view(), stream);
  };
  CHECK(mask(left) == mask(right));
}

/// Requires two zone maps to hold the same bounds.
inline void require_same_zone_map(op::sirius_dynamic_filter const* left,
                                  op::sirius_dynamic_filter const* right,
                                  ::cuda::stream_ref stream)
{
  REQUIRE((left == nullptr) == (right == nullptr));
  if (left == nullptr) { return; }
  auto const& left_zones  = dynamic_cast<op::sirius_dynamic_zone_map_filter const&>(*left).zones();
  auto const& right_zones = dynamic_cast<op::sirius_dynamic_zone_map_filter const&>(*right).zones();
  REQUIRE(left_zones.size() == 1);
  REQUIRE(right_zones.size() == 1);
  CHECK(left_zones.front().min->type() == right_zones.front().min->type());
  CHECK(scalar_bytes(*left_zones.front().min, stream) ==
        scalar_bytes(*right_zones.front().min, stream));
  CHECK(scalar_bytes(*left_zones.front().max, stream) ==
        scalar_bytes(*right_zones.front().max, stream));
}

/// The constructed entry for admitted key @p index, or null.
[[nodiscard]] inline op::build_filter_construction::constructed_key const* key_of(
  constructed const& result, std::size_t index)
{
  for (auto const& key : result.filters.keys) {
    if (key.admitted_key_index == index) { return &key; }
  }
  return nullptr;
}

}  // namespace sirius::test::construction
