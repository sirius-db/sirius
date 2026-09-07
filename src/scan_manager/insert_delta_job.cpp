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

#include "scan_manager/insert_delta_job.hpp"

#include "log/logging.hpp"
#include "memory/topology_index.hpp"
#include "op/scan/duckdb_native_gpu_ingestible.hpp"
#include "op/scan/duckdb_native_metadata.hpp"
#include "scan_manager/mvcc_mask_job.hpp"

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace sirius::scan_manager {

namespace {

namespace ccm = cucascade::memory;

constexpr std::size_t kCarveAlign = 64;  // cache-line-aligned carve within the bundle storage

/// Keeps a bundle's pinned staging alive while any split references it.
/// Member order matters: blocks must be destroyed before the reservation.
struct pinned_staging_bundle {
  std::unique_ptr<ccm::reservation> reservation;
  std::unique_ptr<ccm::fixed_size_host_memory_resource::multiple_blocks_allocation> blocks;
};

std::size_t align_up(std::size_t v, std::size_t a) { return (v + a - 1) / a * a; }

std::size_t numa_node_for_device(int device, sirius::memory::topology_index const& topology)
{
  int const numa = topology.numa_node_of(device);
  return numa < 0 ? 0 : static_cast<std::size_t>(numa);
}

}  // namespace

insert_delta_workset prepare_insert_delta_tasks(
  std::span<insert_delta_job_request> requests,
  exec::scoped_dispatcher& dispatcher,
  cucascade::memory::memory_reservation_manager& reservation_manager,
  sirius::memory::topology_index const& topology,
  std::span<int const> gpu_ids)
{
  insert_delta_workset workset;
  if (requests.empty()) { return workset; }

  auto host_spaces = reservation_manager.get_memory_spaces_for_tier(ccm::Tier::HOST);
  if (host_spaces.empty()) {
    throw std::runtime_error("[prepare_insert_delta_tasks] no HOST-tier memory space registered");
  }
  auto const* probe_fsmr =
    host_spaces.front()->get_memory_resource_as<ccm::fixed_size_host_memory_resource>();
  if (probe_fsmr == nullptr) {
    throw std::runtime_error(
      "[prepare_insert_delta_tasks] host memory space is not a fixed_size_host_memory_resource");
  }
  auto const block_size = probe_fsmr->get_block_size();

  // Capture phase 1, serial: the ClientContext-touching transaction/buffer-
  // manager capture plus the delta row-group skeleton walk, which starts at
  // the binary-searched boundary row group rather than scanning the cached
  // prefix.
  for (auto& request : requests) {
    if (!request.storage || !request.context) {
      throw std::runtime_error(
        "[prepare_insert_delta_tasks] malformed delta request for pinned entry '" +
        request.entry_name + "'");
    }
    try {
      request.plan =
        op::scan::prepare_insert_delta_capture(*request.storage, *request.context, request.n_cache);
    } catch (std::exception const& e) {
      throw std::runtime_error("[prepare_insert_delta_tasks] pinned entry '" + request.entry_name +
                               "': " + e.what());
    }
    request.bundles.clear();
  }

  // Capture phase 2, parallel: the column segment-tree walks, in row-group
  // ranges of the shared metadata parse granularity. Each task fills its own
  // output; the merge below reassembles row-group order deterministically.
  {
    struct capture_range_work {
      insert_delta_job_request* request{nullptr};
      std::size_t rg_begin{0};
      std::size_t rg_end{0};
      std::vector<op::scan::insert_delta_row_group> out;
    };
    std::vector<std::unique_ptr<capture_range_work>> range_works;
    auto const parse_chunk = op::scan::metadata_parse_chunk();
    for (auto& request : requests) {
      auto const n_rgs = request.plan.row_groups.size();
      for (std::size_t begin = 0; begin < n_rgs; begin += parse_chunk) {
        auto work      = std::make_unique<capture_range_work>();
        work->request  = &request;
        work->rg_begin = begin;
        work->rg_end   = std::min(begin + parse_chunk, n_rgs);
        range_works.push_back(std::move(work));
      }
    }
    std::vector<sirius::exec::invocable<void()>> range_tasks;
    range_tasks.reserve(range_works.size());
    for (auto& work : range_works) {
      range_tasks.push_back([work_ptr = work.get()] {
        auto* req = work_ptr->request;
        try {
          work_ptr->out = op::scan::capture_insert_delta_row_group_range(
            req->plan, work_ptr->rg_begin, work_ptr->rg_end, req->union_cols, req->union_types);
        } catch (std::exception const& e) {
          throw std::runtime_error("[prepare_insert_delta_tasks] pinned entry '" + req->entry_name +
                                   "': " + e.what());
        }
      });
    }
    fan_out_and_join(dispatcher, std::move(range_tasks), "insert-delta capture");
    for (auto& work : range_works) {
      for (std::size_t i = 0; i < work->out.size(); ++i) {
        work->request->plan.row_groups[work->rg_begin + i] = std::move(work->out[i]);
      }
    }
  }

  std::size_t next_device = 0;

  for (auto& request : requests) {
    if (request.plan.empty()) { continue; }

    // Close a bundle on the byte budget, the cudf int32 varchar threshold, or
    // a cumulative row count off a byte boundary: staged validity needs
    // byte-aligned row offsets.
    auto const& rgs = request.plan.row_groups;
    std::vector<std::vector<std::size_t>> bundles_rgs;
    {
      std::vector<std::size_t> cur;
      std::size_t cur_budget = 0;
      std::size_t cur_rows   = 0;
      std::vector<std::size_t> cur_varchar(request.union_cols.size(), 0);
      auto flush = [&] {
        if (!cur.empty()) { bundles_rgs.push_back(std::move(cur)); }
        cur.clear();
        cur_budget = 0;
        cur_rows   = 0;
        std::fill(cur_varchar.begin(), cur_varchar.end(), 0);
      };
      for (std::size_t i = 0; i < rgs.size(); ++i) {
        bool exceed = !cur.empty() &&
                      (cur_budget + rgs[i].decoded_bytes_budget > request.approximate_batch_size);
        if (!exceed && !cur.empty()) {
          for (std::size_t c = 0; c < cur_varchar.size(); ++c) {
            if (cur_varchar[c] + rgs[i].varchar_bytes_per_col[c] >=
                op::scan::kCudfInt32StringsThreshold) {
              exceed = true;
              break;
            }
          }
        }
        if (exceed) { flush(); }
        cur.push_back(i);
        cur_budget += rgs[i].decoded_bytes_budget;
        cur_rows += rgs[i].row_count;
        for (std::size_t c = 0; c < cur_varchar.size(); ++c) {
          cur_varchar[c] += rgs[i].varchar_bytes_per_col[c];
        }
        if (cur_rows % 8 != 0) { flush(); }  // validity byte-alignment cut
      }
      flush();
    }

    // Carve each bundle's staging slab and mask words, then build its fill
    // task. One task per bundle keeps the mask single-writer, so bit offsets
    // need no word alignment.
    for (auto& rg_list : bundles_rgs) {
      insert_delta_bundle bundle;
      bundle.rg_indices = std::move(rg_list);

      std::size_t slab_bytes = 0;
      for (auto rg_idx : bundle.rg_indices) {
        bundle.rg_bit_offset.push_back(bundle.total_rows);
        bundle.rg_slab_base.push_back(slab_bytes);
        bundle.total_rows += rgs[rg_idx].row_count;
        slab_bytes = align_up(slab_bytes + rgs[rg_idx].transient_staging_bytes, kCarveAlign);
      }
      auto const n_words    = (bundle.total_rows + 31) / 32;
      auto const mask_off   = align_up(slab_bytes, kCarveAlign);
      auto const total_size = mask_off + n_words * sizeof(std::uint32_t);

      bundle.preferred_device = gpu_ids.empty() ? -1 : gpu_ids[next_device++ % gpu_ids.size()];

      std::uint8_t* base = nullptr;
      if (total_size <= block_size) {
        ccm::any_memory_space_in_tier_with_preference host_req(
          ccm::Tier::HOST, numa_node_for_device(bundle.preferred_device, topology));
        auto reservation = reservation_manager.request_reservation(host_req, block_size);
        if (!reservation) {
          throw std::runtime_error(
            "[prepare_insert_delta_tasks] failed to reserve " + std::to_string(block_size) +
            " bytes of pinned host staging for pinned entry '" + request.entry_name + "'");
        }
        auto* fsmr = reservation->get_memory_space()
                       .get_memory_resource_as<ccm::fixed_size_host_memory_resource>();
        if (fsmr == nullptr || fsmr->get_block_size() != block_size) {
          throw std::runtime_error(
            "[prepare_insert_delta_tasks] reserved host space is not a "
            "fixed_size_host_memory_resource with the expected block size");
        }
        auto pinned         = std::make_shared<pinned_staging_bundle>();
        pinned->reservation = std::move(reservation);
        pinned->blocks      = fsmr->allocate_multiple_blocks(block_size, pinned->reservation.get());
        base                = reinterpret_cast<std::uint8_t*>(pinned->blocks->at(0).data());
        bundle.staging      = std::move(pinned);
      } else {
        // Staging blocks are not virtually contiguous, so oversized bundles
        // use pageable memory. cudaMemcpyAsync stages pageable bytes before
        // returning, so only the async-DMA benefit is lost.
        SIRIUS_LOG_INFO(
          "[prepare_insert_delta_tasks] pinned entry '{}': delta bundle needs {} bytes (> one "
          "{}-byte staging block); using pageable host memory for it",
          request.entry_name,
          total_size,
          block_size);
        auto storage   = std::make_shared<std::vector<std::uint8_t>>(total_size);
        base           = storage->data();
        bundle.staging = std::move(storage);
      }
      bundle.slab_base = base;

      auto* words = reinterpret_cast<std::uint32_t*>(base + mask_off);
      std::fill_n(words, n_words, 0u);  // fill tasks set visible bits only
      bundle.mask = mvcc_chunk_mask{
        std::shared_ptr<std::uint32_t[]>(std::shared_ptr<void>(bundle.staging), words),
        bundle.total_rows};

      auto work          = std::make_unique<insert_delta_work>();
      work->request      = &request;
      work->bundle_index = request.bundles.size();
      request.bundles.push_back(std::move(bundle));

      workset.fill_tasks.push_back([work_ptr = work.get()] {
        auto* req = work_ptr->request;
        auto& b   = req->bundles[work_ptr->bundle_index];
        std::span<std::uint32_t> words_span{b.mask.words.get(), (b.total_rows + 31) / 32};
        std::size_t visible = 0;
        for (std::size_t i = 0; i < b.rg_indices.size(); ++i) {
          auto const& rg = req->plan.row_groups[b.rg_indices[i]];
          visible += op::scan::count_visible_delta_rows(
            rg, req->plan.transaction, words_span, b.rg_bit_offset[i]);
          if (rg.transient_staging_bytes != 0) {
            op::scan::copy_delta_row_group(
              rg, *req->plan.buffer_manager, b.slab_base + b.rg_slab_base[i]);
          }
        }
        work_ptr->visible = visible;
      });
      workset.works.push_back(std::move(work));
    }
  }
  return workset;
}

void finalize_insert_delta_jobs(insert_delta_workset& workset)
{
  std::unordered_set<insert_delta_job_request*> touched;
  for (auto& work : workset.works) {
    auto& bundle = work->request->bundles[work->bundle_index];
    if (work->visible == bundle.total_rows) {
      // Every row visible: serve unmasked.
      bundle.mask = mvcc_chunk_mask{};
    } else if (work->visible == 0) {
      // Nothing visible under this snapshot: mark empty so the sweep below
      // drops the bundle.
      bundle.total_rows = 0;
    }
    touched.insert(work->request);
  }
  for (auto* request : touched) {
    std::erase_if(request->bundles, [](insert_delta_bundle const& b) { return b.total_rows == 0; });
  }
}

void run_insert_delta_jobs(std::span<insert_delta_job_request> requests,
                           exec::scoped_dispatcher& dispatcher,
                           cucascade::memory::memory_reservation_manager& reservation_manager,
                           sirius::memory::topology_index const& topology,
                           std::span<int const> gpu_ids)
{
  auto workset =
    prepare_insert_delta_tasks(requests, dispatcher, reservation_manager, topology, gpu_ids);
  if (workset.fill_tasks.empty()) { return; }
  SIRIUS_LOG_DEBUG("[run_insert_delta_jobs] {} bundle(s) across {} request(s)",
                   workset.works.size(),
                   requests.size());
  fan_out_and_join(dispatcher, std::move(workset.fill_tasks), "insert-delta fill");
  finalize_insert_delta_jobs(workset);
}

std::vector<insert_delta_split> cut_delta_splits_for_op(
  insert_delta_job_request const& request,
  std::span<op::scan::projected_column const> op_projected_cols,
  std::shared_ptr<sirius::io::sirius_datasource> datasource,
  duckdb::SingleFileBlockManager const* block_manager)
{
  std::vector<std::size_t> union_idx;
  union_idx.reserve(op_projected_cols.size());
  for (auto const& pc : op_projected_cols) {
    if (pc.is_rowid) {
      throw std::runtime_error(
        "[cut_delta_splits_for_op] rowid projections are not served from the pin (declined at "
        "plan time)");
    }
    auto const sidx = pc.storage_idx.GetPrimaryIndex();
    auto const it   = std::find(request.union_cols.begin(), request.union_cols.end(), sidx);
    if (it == request.union_cols.end()) {
      throw std::runtime_error("[cut_delta_splits_for_op] operator column " + std::to_string(sidx) +
                               " missing from pinned entry '" + request.entry_name +
                               "''s delta capture");
    }
    union_idx.push_back(static_cast<std::size_t>(it - request.union_cols.begin()));
  }

  auto to_descriptor =
    [](op::scan::insert_delta_segment const& s, std::uint8_t* rg_slab, bool& any_file_read) {
      op::scan::duckdb_segment_descriptor d{};
      d.segment_start     = s.segment_start;
      d.segment_count     = s.segment_count;
      d.compression       = s.compression;
      d.max_string_length = s.max_string_length;
      d.bytes_size        = s.bytes_size;
      d.all_null          = s.all_null;
      d.segment_stats     = s.segment_stats;
      if (s.is_transient) {
        d.block_id     = -1;
        d.block_offset = 0;
        d.host_ptr     = rg_slab + s.slab_offset;
      } else {
        d.block_id          = s.block_id;
        d.block_offset      = s.block_offset;
        d.additional_blocks = s.additional_blocks;
        // Blockless persistent segments (CONSTANT data, all-NULL validity)
        // decode from stats or synthesis and read no file.
        if (d.block_id >= 0) { any_file_read = true; }
      }
      return d;
    };

  std::vector<insert_delta_split> out;
  out.reserve(request.bundles.size());
  for (auto const& bundle : request.bundles) {
    std::vector<op::scan::duckdb_row_group_metadata> row_groups;
    row_groups.reserve(bundle.rg_indices.size());

    bool any_file_read = false;
    for (std::size_t bi = 0; bi < bundle.rg_indices.size(); ++bi) {
      auto const& rgp = request.plan.row_groups[bundle.rg_indices[bi]];
      auto* rg_slab =
        bundle.slab_base == nullptr ? nullptr : bundle.slab_base + bundle.rg_slab_base[bi];

      op::scan::duckdb_row_group_metadata md;
      md.row_group_index      = rgp.row_group_index;
      md.row_group_start      = rgp.row_group_start;
      md.row_count            = rgp.row_count;
      md.decoded_bytes_budget = rgp.decoded_bytes_budget;
      md.varchar_bytes_per_col.reserve(union_idx.size());
      md.columns.reserve(union_idx.size());
      for (auto ui : union_idx) {
        auto const& src = rgp.columns[ui];
        md.varchar_bytes_per_col.push_back(rgp.varchar_bytes_per_col[ui]);
        op::scan::duckdb_column_metadata cm;
        cm.column_id = src.column_id;
        cm.is_array  = src.is_array;
        cm.data_segments.reserve(src.data_segments.size());
        for (auto const& s : src.data_segments) {
          cm.data_segments.push_back(to_descriptor(s, rg_slab, any_file_read));
        }
        cm.validity_segments.reserve(src.validity_segments.size());
        for (auto const& s : src.validity_segments) {
          cm.validity_segments.push_back(to_descriptor(s, rg_slab, any_file_read));
        }
        cm.array_child_data_segments.reserve(src.array_child_data_segments.size());
        for (auto const& s : src.array_child_data_segments) {
          cm.array_child_data_segments.push_back(to_descriptor(s, rg_slab, any_file_read));
        }
        cm.array_child_validity_segments.reserve(src.array_child_validity_segments.size());
        for (auto const& s : src.array_child_validity_segments) {
          cm.array_child_validity_segments.push_back(to_descriptor(s, rg_slab, any_file_read));
        }
        md.columns.push_back(std::move(cm));
      }
      row_groups.push_back(std::move(md));
    }
    // The prefetch handle inside a datasource is per-scan mutable state, so
    // every file-backed split owns a fresh duplicate (matching the native-scan
    // coalescer); splits that read no file carry none.
    auto split_datasource = any_file_read && datasource ? datasource->duplicate()
                                                        : std::shared_ptr<io::sirius_datasource>{};
    auto info             = std::make_unique<op::scan::duckdb_native_scan_info>(
      std::move(row_groups), std::move(split_datasource), block_manager);
    info->host_backed_only = !any_file_read;
    if (bundle.staging) { info->staging_keepalive.push_back(bundle.staging); }

    out.push_back({std::move(info), bundle.mask, bundle.preferred_device});
  }
  return out;
}

}  // namespace sirius::scan_manager
