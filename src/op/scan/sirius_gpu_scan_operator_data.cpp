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

// sirius
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <data/sirius_converter_registry.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <scan_manager/prefetching_state_manager.hpp>

#include <algorithm>
#include <array>
#include <span>
#include <stdexcept>

namespace sirius::op::scan {

bool batch_needs_gpu_upload(const ::cucascade::read_only_data_batch& ro) noexcept
{
  auto const* data = ro.get_data();
  if (data == nullptr) { return false; }
  // A compressed_device_representation sits on Tier::GPU but still has to be decompressed in
  // place, so the tier alone is not the test -- the representation type is the other half of it.
  const bool is_gpu_table =
    dynamic_cast<const ::cucascade::gpu_table_representation*>(data) != nullptr;
  return ro.get_current_tier() != ::cucascade::memory::Tier::GPU || !is_gpu_table;
}

// The three lifecycle members are out-of-line so scan_manager::prefetching_state_manager only has
// to be forward-declared in the header.
scan_operator_input::scan_operator_input(
  std::unique_ptr<scan_info> metadata,
  std::shared_ptr<scan_manager::prefetching_state_manager> prefetch_state)
  : materialization_info(std::move(metadata)), _prefetch_state(std::move(prefetch_state))
{
  if (_prefetch_state) { _prefetch_state->on_input_created(); }
}

scan_operator_input::scan_operator_input(
  std::shared_ptr<cucascade::data_batch> cached_batch,
  std::shared_ptr<scan_manager::prefetching_state_manager> prefetch_state)
  : materialization_info(std::move(cached_batch)), _prefetch_state(std::move(prefetch_state))
{
  if (_prefetch_state) { _prefetch_state->on_input_created(); }
}

scan_operator_input::~scan_operator_input()
{
  // The defaulted move operations leave the moved-from object's _prefetch_state null, so a split
  // that was moved on its way down the chain reports exactly one creation and one disposal.
  if (_prefetch_state) { _prefetch_state->on_input_disposed(); }
}

void scan_operator_input::prefetch(io::cache::prefetching_stage site) const
{
  // Above the metadata check on purpose (D11): a resident pinned-cache split has no scan metadata
  // and no datasource, but it climbs the same ladder. Recording the rung below the check reported
  // 0/0/0/0 for a fully-pinned query.
  if (_prefetch_state) { _prefetch_state->update(site); }
  // for_each_datasource, not get_fadvise_hints(): the latter walks the parquet footer once per
  // row-group slice to rebuild byte ranges this method does not use.
  for_each_datasource([site](io::sirius_datasource& datasource) { datasource.prefetch(site); });
}

void scan_operator_input::for_each_datasource(
  const std::function<void(io::sirius_datasource&)>& visit) const
{
  if (!has_scan_metadata()) { return; }
  std::get<std::unique_ptr<scan_info>>(materialization_info)->for_each_datasource(visit);
}

std::size_t scan_operator_input::datasource_count() const noexcept
{
  if (!has_scan_metadata()) { return 0; }
  return std::get<std::unique_ptr<scan_info>>(materialization_info)->datasource_count();
}

io::cache::prefetch_progress scan_operator_input::prefetch_state() const noexcept
{
  using io::cache::prefetch_progress;

  // Folded one datasource at a time through combine_prefetch_progress rather than gathered into a
  // vector first: this method is noexcept and is reached from split_connector::get_next_split with
  // the connector mutex held, so it must neither allocate nor throw.
  //
  // The running fold equals the batch fold. combine_prefetch_progress reduces its input to four
  // order-independent predicates -- any loading / all ready / any prepared / any cancelled -- and
  // each intermediate result carries every predicate that can still affect the outcome. The one
  // lossy case is `prepared` absorbing a `cancelled`, which cannot change the answer because
  // `prepared` already outranks `cancelled` and, once set, stays set.
  //
  // The visitor state is one struct so the lambda captures a single pointer: the visit API takes a
  // std::function, and a capture too wide for its small-object buffer would heap-allocate, putting
  // a possible std::bad_alloc on this non-throwing boundary.
  struct fold_state {
    prefetch_progress value{prefetch_progress::empty};
    bool seen{false};
  } state;

  for_each_datasource([&state](io::sirius_datasource& datasource) {
    std::array<prefetch_progress, 2> const pair{state.value, datasource.prefetch_state()};
    // The first datasource folds alone: seeding the accumulator with `empty` would drag an
    // otherwise-cached split down, because `empty` breaks the all-ready rule.
    auto const parts = state.seen ? std::span<const prefetch_progress>{pair}
                                  : std::span<const prefetch_progress>{pair.data() + 1, 1};
    state.value      = io::cache::combine_prefetch_progress(parts);
    state.seen       = true;
  });

  // No datasource at all (a resident split, or a metadata split whose datasources are null) folds
  // to `empty`, which is what state.value still holds.
  return state.value;
}

bool scan_operator_input::is_io_prefetchable() const noexcept { return datasource_count() > 0; }

std::optional<bool> scan_operator_input::is_memory_prefetchable() const noexcept
{
  if (!is_resident()) { return false; }
  auto const& batch = std::get<std::shared_ptr<::cucascade::data_batch>>(materialization_info);
  if (!batch) { return false; }
  try {
    // try_to_read_only, never the blocking to_read_only: this runs under split_connector's mutex
    // and a concurrent prepare_for_processing holds the batch exclusively while it converts.
    auto ro = batch->try_to_read_only();
    if (!ro) { return std::nullopt; }
    return batch_needs_gpu_upload(*ro);
  } catch (...) {
    // The lock accessor is library code; contain anything it throws rather than terminate on a
    // noexcept boundary. "Could not read the tier" is exactly what nullopt already means.
    return std::nullopt;
  }
}

std::optional<bool> scan_operator_input::is_prefetched() const noexcept
{
  // A metadata split is where the task wants it once every datasource reported its request
  // complete. Note this is not the negation of is_memory_prefetchable on this path: both are
  // false for a metadata split whose IO has not landed.
  if (!is_resident()) { return prefetch_state() == io::cache::prefetch_progress::cached; }
  auto const needs_upload = is_memory_prefetchable();
  if (!needs_upload.has_value()) { return std::nullopt; }
  return !*needs_upload;
}

void scan_operator_input::prepare_for_processing(
  const ::cucascade::memory::memory_space* requested_memory_space, rmm::cuda_stream_view stream)
{
  gpu_memory_space = const_cast<::cucascade::memory::memory_space*>(requested_memory_space);
  // Hoisted above the resident early-return: both split kinds reach task_preprocessing, and only
  // the metadata kind has datasources to hint. Left inside the branch, a fully-pinned query records
  // zero on this rung despite every one of its splits climbing it.
  prefetch(io::cache::prefetching_stage::task_preprocessing);
  if (!std::holds_alternative<std::shared_ptr<cucascade::data_batch>>(materialization_info)) {
    return;
  }
  auto batch = std::get<std::shared_ptr<cucascade::data_batch>>(materialization_info);

  if (batch && requested_memory_space) {
    bool needs_upload = false;
    {
      // Shared with is_memory_prefetchable, so what the scheduler was told about this split and
      // what actually happens to it cannot drift. The read lock is released before converting.
      auto ro      = batch->to_read_only();
      needs_upload = batch_needs_gpu_upload(ro);
    }
    if (needs_upload) {
      auto& registry = ::sirius::converter_registry::get();
      auto mut       = batch->to_mutable();
      mut.convert_to<::cucascade::gpu_table_representation>(
        registry, requested_memory_space, stream);
    }
  }

  if (batch) {
    auto ro          = batch->to_read_only();
    gpu_memory_space = ro.get_memory_space();
  }
}

std::size_t scan_operator_input::get_estimated_size_in_bytes() const
{
  if (std::holds_alternative<std::unique_ptr<scan_info>>(materialization_info)) {
    return std::get<std::unique_ptr<scan_info>>(materialization_info)->estimated_bytes();
  }
  if (std::holds_alternative<std::shared_ptr<cucascade::data_batch>>(materialization_info)) {
    auto batch = std::get<std::shared_ptr<cucascade::data_batch>>(materialization_info);

    auto ro          = batch->to_read_only();
    auto const* data = ro.get_data();
    if (!data) { return 0; }
    // prepare_for_processing decompresses a compressed cache batch, so the task's
    // working set (hence its reservation) is the UNCOMPRESSED footprint, not the
    // resident compressed payload. get_size_in_bytes() reports only the compressed
    // bytes for a compressed entry; using it under-reserves, and the on-demand
    // decompress then over-allocates into the downgrade path — which cannot evict
    // the pinned GPU entry, so the query deadlocks. Uncompressed batches report
    // equal sizes, so this is a no-op for them.
    return std::max(data->get_size_in_bytes(), data->get_uncompressed_data_size_in_bytes());
  }
  return 0;
}

std::size_t scan_operator_input::get_estimated_working_set_size_in_bytes() const
{
  if (std::holds_alternative<std::unique_ptr<scan_info>>(materialization_info)) {
    auto const decode_bytes =
      std::get<std::unique_ptr<scan_info>>(materialization_info)->estimated_working_set_bytes();
    if (mvcc_keep_mask.has_mask()) {
      // A partially visible insert-delta split is mask-filtered right after
      // decode: the decoded input and the compacted output (up to input-sized)
      // coexist at peak, alongside the BOOL8 expansion column (1 B/row) and
      // the uploaded bitmask words — the same envelope as the cached branch
      // below.
      return 2 * decode_bytes + mvcc_keep_mask.row_count + mvcc_keep_mask.view().size_bytes();
    }
    return decode_bytes;
  }
  auto const batch_bytes = get_estimated_size_in_bytes();
  if (mvcc_keep_mask.has_mask()) {
    // A masked resident chunk is filtered by copy at materialize: the input
    // batch and the compacted output (up to input-sized) coexist at peak,
    // alongside the BOOL8 expansion column (1 B/row) and the uploaded bitmask
    // words. A pending row filter needs no extra headroom: its phase peaks at
    // mask output + predicate + compacted output, inside the same envelope.
    return 2 * batch_bytes + mvcc_keep_mask.row_count + mvcc_keep_mask.view().size_bytes();
  }
  if (row_filter_pending) {
    // post_filter_and_project filters by copy: the materialized input and the
    // compacted output (up to input-sized) coexist at peak. The BOOL8
    // predicate column (1 B/row) hides inside the 2x conservatism (any
    // projected column is >= 4 B/row).
    return 2 * batch_bytes;
  }
  // An unmasked, unfiltered chunk serves a zero-copy view whose output copy is
  // at most batch-sized, so plain batch_bytes stays accurate.
  return batch_bytes;
}

}  // namespace sirius::op::scan
