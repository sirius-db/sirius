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
#include "compression/compressed_representation.hpp"

#include <cudf/table/table.hpp>

#include <compression/decompression_pushdown_policy.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <data/sirius_converter_registry.hpp>
#include <log/logging.hpp>
#include <op/scan/decoded_batch_representation.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <op/sirius_dynamic_filter.hpp>

#include <algorithm>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace sirius::op::scan {

namespace {

// What a decode of this resident split's compressed batch is expected to do to
// its size. All-false for anything else — including post-convert states, where
// the data is no longer a compressed representation.
sirius::compressed_scan::compaction_forecast decode_compaction_forecast(
  scan_operator_input const& split)
{
  if (!split.is_resident()) { return {}; }
  auto batch      = split.get_cached_batch();
  auto ro         = batch->to_read_only();
  auto const* rep = dynamic_cast<sirius::compressed_device_representation const*>(ro.get_data());
  if (rep == nullptr || !rep->decode_scan()) { return {}; }
  auto const& indices = rep->selected_indices();
  std::vector<std::size_t> identity;
  if (!indices.has_value()) {
    identity.resize(rep->column_names().size());
    std::iota(identity.begin(), identity.end(), std::size_t{0});
  }
  return rep->decode_scan()->forecast_compaction(rep->table(),
                                                 indices.has_value() ? *indices : identity);
}

}  // namespace

membership_snapshot snapshot_membership_probes(sirius::op::sirius_dynamic_filter_set const& set,
                                               std::size_t n_slots)
{
  membership_snapshot snap;
  // generation FIRST: it must never claim probes the walk below did not
  // capture (see the header doc).
  snap.generation = set.filter_count();
  snap.probes.resize(n_slots);
  for (std::size_t i = 0; i < n_slots; ++i) {
    auto filters = set.filters_for_column(i);
    for (auto& filter : filters) {
      // Only mask-capable kinds (in-list / small-in-list / Bloom) can probe
      // at decode; zone-map filters have no per-row form.
      auto const* applicable =
        dynamic_cast<sirius::op::sirius_mask_applicable const*>(filter.get());
      if (applicable == nullptr) {
        ++snap.skipped_non_mask;
        continue;
      }
      // Ordering signal (sirius::membership_probe doc): rank by ascending
      // expected keep-rate, num_keys where the concrete filter exposes it.
      // Bloom has no size accessor — the rank alone places it last.
      std::uint8_t kind_rank = 255;
      std::uint64_t num_keys = 0;
      if (auto const* small =
            dynamic_cast<sirius::op::sirius_dynamic_small_in_list_filter const*>(filter.get())) {
        kind_rank = 0;
        num_keys  = small->size();
      } else if (auto const* set =
                   dynamic_cast<sirius::op::sirius_dynamic_in_list_filter const*>(filter.get())) {
        kind_rank = 1;
        num_keys  = set->size();
      } else if (filter->kind() == sirius::op::sirius_dynamic_filter_kind::BLOOM) {
        kind_rank = 2;
      }
      // The closure co-owns the filter. It is snapshotted before the balancer
      // assigns this split's chunk to a GPU, so the device isn't known yet
      // here; pass -1 so compute_mask resolves it from the CURRENT CUDA
      // device at probe time, which the task scheduler has already set to
      // the chunk's assigned GPU by then.
      snap.probes[i].push_back(
        {[f = std::move(filter), applicable](cudf::column_view const& keys,
                                             rmm::cuda_stream_view s,
                                             rmm::device_async_resource_ref mr) {
           return applicable->compute_mask(keys, /*device_id=*/-1, s, mr);
         },
         kind_rank,
         num_keys});
      ++snap.attached_probes;
    }
  }
  return snap;
}

void scan_operator_input::prepare_for_processing(
  const ::cucascade::memory::memory_space* requested_memory_space, rmm::cuda_stream_view stream)
{
  gpu_memory_space = const_cast<::cucascade::memory::memory_space*>(requested_memory_space);
  if (!std::holds_alternative<std::shared_ptr<cucascade::data_batch>>(materialization_info)) {
    prefetch(io::cache::prefetching_stage::just_in_time);
    return;
  }
  auto batch = std::get<std::shared_ptr<cucascade::data_batch>>(materialization_info);

  if (batch && requested_memory_space && !stolen_table && !stolen_table_consumed) {
    bool needs_upload = false;
    {
      auto ro          = batch->to_read_only();
      auto const* data = ro.get_data();
      // Convert when the data is not on the GPU tier, OR when it is on the GPU
      // tier but not already a plain gpu_table_representation (e.g. a
      // compressed_device_representation, which must be decompressed in place).
      const bool is_gpu_table =
        dynamic_cast<const ::cucascade::gpu_table_representation*>(data) != nullptr;
      needs_upload = data != nullptr &&
                     (ro.get_current_tier() != ::cucascade::memory::Tier::GPU || !is_gpu_table);
    }
    if (needs_upload) {
      auto& registry = ::sirius::converter_registry::get();
      auto mut       = batch->to_mutable();
      if (decode_selection_unprofitable &&
          decode_selection_unprofitable->load(std::memory_order_relaxed)) {
        // An earlier batch of this scan reported that compacting during decode
        // does not pay off; selectivity is uniform across batches, so drop the
        // row selection and stop paying for the attempt. Only the per-query
        // projected clone is touched — never the shared pin — and only this
        // operator's splits: another query's scan decides fresh.
        auto drop_row_selection = [](auto* rep) {
          if (rep->decode_scan()) {
            rep->set_decode_scan(rep->decode_scan()->without_row_selection());
          }
        };
        if (auto* device_rep =
              dynamic_cast<::sirius::compressed_device_representation*>(mut.get_data())) {
          drop_row_selection(device_rep);
        } else if (auto* host_rep =
                     dynamic_cast<::sirius::compressed_host_representation*>(mut.get_data())) {
          drop_row_selection(host_rep);
        }
      }
      // Decode-time join filter snapshot: the scan-manager drain runs at query
      // PREPARE, before any join build has published, so a drain-time snapshot
      // is empty for the whole scan. Executor tasks prepare right before decode
      // — by then upstream builds have published — so refresh the projected rep
      // with a fresh per-batch snapshot here, replacing the (typically empty)
      // drain-time one. The mapping invariant lives in
      // snapshot_membership_probes; same mvcc guard as the row selection.
      if (sirius::decompression_pushdown_enabled() && dynamic_filters &&
          dynamic_filters->has_filters() && !mvcc_keep_mask.has_mask()) {
        auto snapshot_onto = [&](auto* rep) {
          std::size_t const n_slots = rep->selected_indices().has_value()
                                        ? rep->selected_indices()->size()
                                        : rep->column_names().size();
          auto snap                 = snapshot_membership_probes(*dynamic_filters, n_slots);
          SIRIUS_DECOMPRESSION_PUSHDOWN_DIAG(
            "[decode-filter] join filter attach (decode time) channel={}: slots={} attached={} "
            "generation={} skipped_non_maskable={}",
            static_cast<void const*>(dynamic_filters.get()),
            n_slots,
            snap.attached_probes,
            snap.generation,
            snap.skipped_non_mask);
          if (snap.attached_probes == 0) { return; }
          auto const base =
            rep->decode_scan()
              ? rep->decode_scan()
              : std::make_shared<const ::sirius::compressed_scan>(::sirius::scan_decode_request{});
          rep->set_decode_scan(
            base->with_membership_probes(std::move(snap.probes), snap.generation));
        };
        if (auto* device_rep =
              dynamic_cast<::sirius::compressed_device_representation*>(mut.get_data())) {
          snapshot_onto(device_rep);
        } else if (auto* host_rep =
                     dynamic_cast<::sirius::compressed_host_representation*>(mut.get_data())) {
          snapshot_onto(host_rep);
        }
      }
      mut.convert_to<::cucascade::gpu_table_representation>(
        registry, requested_memory_space, stream);
      // The converter reports what the decode did as a value on the
      // representation. row_filtered means the whole table-filter conjunction
      // was applied and every column is compacted to the surviving rows —
      // materialize_table maps it to filter_state::ROW_FILTERED so the filter
      // is not re-evaluated. selection_unprofitable means the attempt did not
      // pay off, so the scan's remaining splits skip it. Off-gate the
      // converters install the plain representation and both stay false.
      if (auto const* decoded =
            dynamic_cast<::sirius::decoded_batch_representation const*>(mut.get_data())) {
        auto const& outcome        = decoded->outcome();
        decode_row_filtered        = outcome.row_filtered;
        decode_predicate_columns   = outcome.predicate_columns;
        decode_predicates_enforced = outcome.predicates_enforced;
        if (decode_selection_unprofitable && outcome.selection_unprofitable) {
          decode_selection_unprofitable->store(true, std::memory_order_relaxed);
        }
      }
      if (decode_row_filtered && mvcc_keep_mask.has_mask()) {
        // The keep-mask is positional over the chunk's full row range; a
        // decode-compacted table no longer lines up with it. Row dropping must
        // never be requested for mvcc-masked chunks — fail loudly rather than
        // filter the wrong rows.
        throw std::runtime_error(
          "[scan_operator_input::prepare_for_processing] decode-time row filtering is "
          "incompatible with an mvcc keep-mask; the attach must exclude masked chunks");
      }
      // The conversion output is a fresh per-query table (raw GPU pins serve a
      // plain gpu_table_representation and never reach this branch), so an
      // unmasked, unfiltered scan may take ownership of it here instead of
      // deep-copying it at materialize. Masked / row-filtered splits keep the
      // view path: they filter by copy and need the source view alive — and
      // skipping them means the scan allocates nothing after the take, so an
      // OOM retry can never re-enter materialize on a consumed split. A
      // decode-row-filtered split has no filter copy left to make, so it
      // regains the zero-copy steal.
      if (!mvcc_keep_mask.has_mask() && (!row_filter_pending || decode_row_filtered)) {
        if (auto* gpu_rep = dynamic_cast<::cucascade::gpu_table_representation*>(mut.get_data())) {
          auto& space        = gpu_rep->get_memory_space();
          stolen_table_bytes = gpu_rep->get_size_in_bytes();
          stolen_table       = gpu_rep->release_table(stream);
          // The batch cannot hold null data and its size/view queries
          // dereference the table, so leave a valid empty placeholder.
          mut.set_data(std::make_unique<::cucascade::gpu_table_representation>(
            std::make_unique<cudf::table>(), space, rmm::cuda_stream_view{}));
        }
      }
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
    // Once prepare_for_processing has taken the wrapper's table the batch only
    // holds an empty placeholder; answer from the stolen table so OOM-retry
    // estimates keep covering the live data.
    if (stolen_table_bytes > 0) { return stolen_table_bytes; }
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
  if (decode_row_filtered) {
    // The decode already compacted this split to its surviving rows, and batch_bytes reports that
    // compacted footprint (the conversion replaced the compressed representation; a stolen split
    // answers from stolen_table_bytes). A stolen table is moved into the scan
    // output with no copy; the view path still copies once at materialize, so
    // input + output stay within 2x compacted either way — the pre-decode 2x
    // FULL-WIDTH envelope below no longer applies.
    bool const stolen = stolen_table != nullptr || stolen_table_consumed;
    return stolen ? batch_bytes : 2 * batch_bytes;
  }
  if (row_filter_pending) {
    // Once compaction has been measured unprofitable, later batches drop the
    // row selection and decode full width — keep the full-width envelope.
    bool const unprofitable = decode_selection_unprofitable &&
                              decode_selection_unprofitable->load(std::memory_order_relaxed);
    if (auto const forecast = decode_compaction_forecast(*this);
        forecast.compacts && !unprofitable) {
      // Reservation for a compacting decode: the selection mask (1 bit/row per
      // filtered column — <= batch/4 across the source limit at >= 4 B/row
      // columns) plus the compacted outputs, and such splits steal their table,
      // so there is no second output copy. The surviving row count is bounded
      // by the selectivity ceiling unless a column is exempt from it, in which
      // case size for up to full width. A decode that gives compaction up
      // re-runs the full-width path and over-allocates into the adaptor's
      // over-reservation handling; by policy that only happens where the
      // forecast was wrong. Replaces a ~5x over-reservation on
      // highly-selective batches.
      auto const cap =
        forecast.survivors_bounded ? sirius::decompression_pushdown_max_selectivity() : 1.0;
      return batch_bytes / 4 + static_cast<std::size_t>(static_cast<double>(batch_bytes) * cap);
    }
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
