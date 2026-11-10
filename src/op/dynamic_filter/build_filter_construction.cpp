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

#include "op/dynamic_filter/build_filter_construction.hpp"

#include "log/logging.hpp"
#include "memory/size_arithmetic.hpp"
#include "op/dynamic_filter/detail/bounds_accumulator.hpp"
#include "op/dynamic_filter/dynamic_filter_replica_reservation.hpp"

#include <rmm/error.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <array>
#include <concepts>
#include <limits>
#include <stdexcept>
#include <utility>

namespace sirius::op {

namespace {

/**
 * @brief The compile-time contract `contribute` and `releasable_here` rely on: each accumulator is
 * a move-only owner of per-GPU storage that a contribution adds to through `enqueue_add` and that
 * reports whether it may be released on the calling thread.
 *
 * Each accumulator finishes through its own call (`finish`, `gather`), with arguments only
 * `build_filter_construction::finish` knows how to supply, so finishing is not part of the
 * contract.
 */
template <class Accumulator>
concept build_key_accumulator = std::movable<Accumulator> && !std::copyable<Accumulator> &&
                                requires(Accumulator const& accumulator) {
                                  { accumulator.releasable_here() } noexcept -> std::same_as<bool>;
                                  &Accumulator::enqueue_add;
                                };

static_assert(build_key_accumulator<detail::accumulated_bloom_builder>);
static_assert(build_key_accumulator<detail::exact_key_stager>);
static_assert(build_key_accumulator<detail::bounds_accumulator>);

constexpr std::size_t k_none = std::numeric_limits<std::size_t>::max();

using memory::saturating_add;
using memory::saturating_mul;

}  // namespace

batch_key_facts batch_key_facts::of(cudf::table_view const& table)
{
  batch_key_facts facts{.rows = static_cast<std::uint64_t>(table.num_rows()), .columns = {}};
  facts.columns.reserve(static_cast<std::size_t>(table.num_columns()));
  for (auto const& column : table) {
    facts.columns.push_back({column.type(), static_cast<std::size_t>(column.null_count())});
  }
  return facts;
}

struct build_filter_construction::impl {
  construction_plan plan;
  // Per planned key: its index among each accumulator's keys, or k_none.
  std::vector<std::size_t> bloom_of;
  std::vector<std::size_t> exact_of;
  std::vector<std::size_t> bounds_of;
  std::optional<detail::accumulated_bloom_builder> bloom;
  std::optional<detail::exact_key_stager> stager;
  std::optional<detail::bounds_accumulator> bounds;
  // The inventory's facts, by batch: what each claim must match.
  std::vector<std::uint64_t> batch_rows;
  std::vector<std::vector<column_facts>> batch_columns;
  // Per exact key, per batch: the first row of the batch's slice.
  std::vector<std::vector<std::size_t>> exact_offsets;
  std::size_t task_scratch      = 0;
  std::size_t root_construction = 0;
};

build_filter_construction::build_filter_construction(std::unique_ptr<impl> state) noexcept
  : _impl{std::move(state)}
{
}

build_filter_construction::~build_filter_construction() = default;
build_filter_construction::build_filter_construction(build_filter_construction&&) noexcept =
  default;
build_filter_construction& build_filter_construction::operator=(
  build_filter_construction&&) noexcept = default;

std::optional<build_filter_construction> build_filter_construction::try_create(
  construction_plan plan,
  complete_build_inventory const& inventory,
  std::span<dynamic_filter_replica_space const> targets)
{
  auto state         = std::make_unique<impl>();
  auto const batches = inventory.batches().size();
  for (std::size_t batch = 0; batch < batches; ++batch) {
    state->batch_rows.push_back(inventory.batches()[batch].rows);
    auto const columns = inventory.batch_columns(batch);
    state->batch_columns.emplace_back(columns.begin(), columns.end());
  }
  std::vector<detail::accumulated_bloom_builder::key> bloom_keys;
  std::vector<detail::exact_key_stager::key> exact_keys;
  std::vector<detail::bounds_accumulator::key> bound_keys;
  std::size_t numeric_bounds    = 0;
  std::size_t narrowed_carriers = 0;
  std::size_t root_filters      = 0;
  for (auto const& planned : plan.keys) {
    auto const ordinal = static_cast<std::size_t>(planned.build_key_ordinal);
    if (planned.build_key_ordinal < 0 ||
        std::ranges::any_of(
          state->batch_columns, [&](auto const& columns) { return ordinal >= columns.size(); })) {
      throw std::logic_error(
        "[build_filter_construction] a planned key's column is absent from a build batch");
    }
    state->bloom_of.push_back(k_none);
    state->exact_of.push_back(k_none);
    state->bounds_of.push_back(k_none);
    if (auto const* bloom = std::get_if<bloom_membership>(&planned.membership)) {
      state->bloom_of.back() = bloom_keys.size();
      bloom_keys.push_back({planned.build_key_ordinal, planned.construction_type, bloom->blocks});
    }
    if (auto const* exact = std::get_if<exact_membership>(&planned.membership)) {
      // Slices follow the inventory's order, each sized by its batch's recorded valid rows.
      std::vector<std::size_t> offsets;
      std::size_t valid = 0;
      for (std::size_t batch = 0; batch < batches; ++batch) {
        auto const& column = state->batch_columns[batch][ordinal];
        if (!column.null_count || *column.null_count > state->batch_rows[batch]) {
          throw std::logic_error(
            "[build_filter_construction] an exact key needs every batch's null count");
        }
        offsets.push_back(valid);
        valid += static_cast<std::size_t>(state->batch_rows[batch]) - *column.null_count;
      }
      if (valid != exact->valid_rows) {
        throw std::logic_error(
          "[build_filter_construction] an exact key's plan disagrees with its inventory");
      }
      state->exact_of.back() = exact_keys.size();
      state->exact_offsets.push_back(std::move(offsets));
      exact_keys.push_back(
        {planned.build_key_ordinal, planned.construction_type, exact->kind, exact->valid_rows});
    }
    if (planned.needs_bounds()) {
      state->bounds_of.back() = bound_keys.size();
      bound_keys.push_back({planned.build_key_ordinal, planned.recorded_type});
      numeric_bounds += planned.recorded_type.id() == cudf::type_id::STRING ? 0 : 1;
      // Every carrier other than the recorded type restores its bounds at publish.
      std::vector<cudf::data_type> carriers;
      for (auto const& columns : state->batch_columns) {
        auto const carrier = columns[ordinal].carrier;
        if (carrier != planned.recorded_type &&
            std::ranges::find(carriers, carrier) == carriers.end()) {
          carriers.push_back(carrier);
        }
      }
      narrowed_carriers += carriers.size();
    }
    if (auto const* exact = std::get_if<exact_membership>(&planned.membership)) {
      root_filters = saturating_add(
        root_filters,
        in_list_storage_upper_bound(exact->valid_rows,
                                    classify_membership_key(planned.construction_type)->rep));
    }
    if (planned.zone_map) {
      // A STRING bound's characters are known only at publish; its share here is one byte each.
      root_filters = saturating_add(
        root_filters,
        planned.recorded_type.id() == cudf::type_id::STRING
          ? 2 * (detail::tracked_replica_allocation_bytes(1) +
                 detail::tracked_replica_allocation_bytes(sizeof(bool)))
          : detail::bounds_accumulator::fixed_width_scalar_bytes(planned.recorded_type));
    }
  }
  // Restoring narrower carriers' bounds ends before the first root filter is built.
  state->root_construction =
    std::max(root_filters, detail::bounds_accumulator::finish_scratch_bytes(narrowed_carriers));

  // Task scratch: a contribution's temporaries for the largest batch, then the Bloom merge.
  std::size_t largest = 0;
  for (auto const rows : state->batch_rows) {
    largest = std::max<std::size_t>(largest, rows);
  }
  std::vector<int> devices;
  for (auto const& target : targets) {
    auto const device = target.get_gpu_space().get_device_id();
    if (std::ranges::find(devices, device) == devices.end()) { devices.push_back(device); }
  }
  // STRING bounds take nothing from the task's scratch: their reductions return to pinned memory.
  std::size_t scratch = 0;
  if (!exact_keys.empty()) {
    scratch = saturating_mul(exact_keys.size(),
                             detail::exact_key_stager::contribution_scratch_bytes(largest));
  }
  if (numeric_bounds != 0) {
    scratch = saturating_add(
      scratch,
      saturating_mul(numeric_bounds,
                     detail::bounds_accumulator::contribution_scratch_bytes(largest)));
  }
  state->task_scratch = saturating_add(
    scratch,
    detail::accumulated_bloom_builder::publication_scratch_bytes(bloom_keys, devices.size()));

  if (!bloom_keys.empty()) {
    state->bloom = detail::accumulated_bloom_builder::try_create(std::move(bloom_keys), targets);
    if (!state->bloom) { return std::nullopt; }
  }
  if (!exact_keys.empty()) {
    state->stager = detail::exact_key_stager::try_create(std::move(exact_keys), targets);
    if (!state->stager) { return std::nullopt; }
  }
  if (!bound_keys.empty()) {
    state->bounds = detail::bounds_accumulator::try_create(std::move(bound_keys), batches, targets);
    if (!state->bounds) { return std::nullopt; }
  }
  state->plan = std::move(plan);
  return build_filter_construction{std::move(state)};
}

std::optional<build_filter_construction::contribution_ticket> build_filter_construction::reserve(
  std::size_t batch_index, rmm::cuda_device_id device, batch_key_facts const& actual) const
{
  if (batch_index >= _impl->batch_rows.size() || actual.rows != _impl->batch_rows[batch_index]) {
    return std::nullopt;
  }
  auto const& recorded = _impl->batch_columns[batch_index];
  for (auto const& planned : _impl->plan.keys) {
    auto const ordinal = static_cast<std::size_t>(planned.build_key_ordinal);
    if (ordinal >= actual.columns.size()) { return std::nullopt; }
    auto const& claimed  = actual.columns[ordinal];
    auto const& expected = recorded[ordinal];
    if (claimed.carrier != expected.carrier ||
        (expected.null_count && claimed.null_count != expected.null_count)) {
      return std::nullopt;
    }
  }
  contribution_ticket ticket{.batch_index = batch_index, .device = device, .slices = {}};
  for (std::size_t index = 0; index < _impl->plan.keys.size(); ++index) {
    auto const exact = _impl->exact_of[index];
    if (exact == k_none) { continue; }
    auto const& column =
      recorded[static_cast<std::size_t>(_impl->plan.keys[index].build_key_ordinal)];
    ticket.slices.push_back({_impl->exact_offsets[exact][batch_index],
                             static_cast<std::size_t>(actual.rows) - *column.null_count});
  }
  return ticket;
}

bool build_filter_construction::contribute(contribution_ticket const& ticket,
                                           cudf::table_view const& input,
                                           ::cuda::stream_ref stream,
                                           rmm::device_async_resource_ref scratch)
{
  std::array const joined{stream};
  bool accepted = true;
  // A failure in a later accumulator joins the work earlier ones already queued on the stream.
  detail::join_on_failure(joined, [&] {
    auto const require = [&](bool added) {
      if (!added) {
        throw detail::accumulation_invariant_error{
          "[build_filter_construction] accumulators disagree on the GPUs that hold storage"};
      }
    };
    bool first     = true;
    auto const add = [&](bool added) {
      if (first && !added) {
        accepted = false;
      } else {
        require(added);
      }
      first = false;
    };
    if (_impl->bloom) { add(_impl->bloom->enqueue_add(input, ticket.device, stream)); }
    if (accepted && _impl->stager) {
      add(_impl->stager->enqueue_add(ticket.slices, input, ticket.device, stream, scratch));
    }
    if (accepted && _impl->bounds) {
      add(_impl->bounds->enqueue_add(ticket.batch_index, input, ticket.device, stream, scratch));
    }
  });
  return accepted;
}

std::optional<build_filter_construction::constructed_filters> build_filter_construction::finish(
  publication_root const& root)
{
  rmm::cuda_device_id const device{root.space.get_device_id()};
  int current = -1;
  if (cudaGetDevice(&current) != cudaSuccess || current != device.value()) {
    throw detail::accumulation_invariant_error{
      "[build_filter_construction::finish] the root GPU is not the current device"};
  }
  auto const& keys = _impl->plan.keys;
  constructed_filters result;
  result.keys.reserve(keys.size());
  for (auto const& planned : keys) {
    result.keys.push_back({.admitted_key_index = planned.admitted_key_index,
                           .zone_map           = nullptr,
                           .membership         = nullptr,
                           .domain             = std::nullopt,
                           .dropped            = std::nullopt});
  }

  std::vector<std::optional<detail::bounds_accumulator::key_bounds>> merged;
  if (_impl->bounds) { merged = _impl->bounds->finish(root.stream, root.scratch); }
  std::optional<detail::accumulated_bloom_builder::publication> blooms;
  if (_impl->bloom) {
    blooms = _impl->bloom->finish(device, root.stream, root.scratch);
    if (!blooms) { return std::nullopt; }
    result.transport = blooms->transport;
  }
  std::optional<detail::exact_key_stager::gathered> staged;
  if (_impl->stager) {
    staged = _impl->stager->gather(device, root.stream);
    if (!staged) { return std::nullopt; }
  }

  // Root filters are charged to the root task's own allocator. Each is built only when the room
  // left on this thread's tracker covers its upper bound, and an allocation that fails anyway drops
  // only that filter: whatever it had queued frees in stream order.
  auto const build_within_room = [&](constructed_key& out, std::size_t bytes, auto&& build) {
    if (root.space.get_available_memory(root.stream) >= bytes) {
      try {
        build();
        return;
      } catch (rmm::out_of_memory const&) {
      }
    }
    ++result.counts.filters_skipped_construction_admission;
    out.dropped = dynamic_filter_key_reason::construction_admission;
  };

  for (std::size_t index = 0; index < keys.size(); ++index) {
    auto const& planned = keys[index];
    auto& out           = result.keys[index];
    auto const* bounds  = _impl->bounds_of[index] == k_none || !merged[_impl->bounds_of[index]]
                            ? nullptr
                            : &*merged[_impl->bounds_of[index]];
    // A DECIMAL128 key narrowed unchecked: it publishes membership only if its values fit.
    bool const fits = !planned.fit_pending || bounds == nullptr || bounds->fits_membership_rep;
    if (!fits) { out.dropped = dynamic_filter_key_reason::decimal128_unfit; }

    if (auto const bloom = _impl->bloom_of[index]; bloom != k_none && fits) {
      out.membership = blooms->filters[bloom];
      out.domain     = blooms->filters[bloom]->domain();
    }
    if (auto const exact = _impl->exact_of[index]; exact != k_none) {
      if (staged->overflowed(exact)) {
        // Claim validation makes this unreachable; an incomplete list would have false negatives.
        SIRIUS_LOG_ERROR(
          "[build_filter_construction] key {} staged more valid rows than its inventory recorded; "
          "its exact filter is dropped.",
          planned.admitted_key_index);
        ++result.counts.filters_skipped_construction_invariant;
        out.dropped = dynamic_filter_key_reason::construction_invariant;
      } else if (fits) {
        auto const domain     = *classify_membership_key(planned.construction_type);
        auto const valid_rows = std::get<exact_membership>(planned.membership).valid_rows;
        build_within_room(out, in_list_storage_upper_bound(valid_rows, domain.rep), [&] {
          out.membership = staged->build(exact, root.stream, root.scratch);
          out.domain     = domain;
        });
      }
    }
    if (planned.zone_map && bounds != nullptr) {
      build_within_room(out, bounds->device_bytes(), [&] {
        auto [minimum, maximum] = bounds->to_scalars(root.stream, root.scratch);
        std::vector<zone_map_entry> zones;
        zones.push_back({std::move(minimum), std::move(maximum)});
        out.zone_map =
          std::make_shared<sirius_dynamic_zone_map_filter>(std::move(zones), true, true);
      });
    }
  }
  // The construction sync: every root filter is complete before it is replicated or published, and
  // before the staged rows it read are released.
  root.stream.sync();
  return result;
}

std::size_t build_filter_construction::task_scratch_bytes() const noexcept
{
  return _impl->task_scratch;
}

std::size_t build_filter_construction::root_construction_bytes() const noexcept
{
  return _impl->root_construction;
}

bool build_filter_construction::releasable_here() const noexcept
{
  return (!_impl->bloom || _impl->bloom->releasable_here()) &&
         (!_impl->stager || _impl->stager->releasable_here()) &&
         (!_impl->bounds || _impl->bounds->releasable_here());
}

construction_plan const& build_filter_construction::plan() const noexcept { return _impl->plan; }

}  // namespace sirius::op
