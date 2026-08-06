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

#include "scan_manager/mvcc_mask_job.hpp"

#include "exec/completion_controller.hpp"
#include "exec/scoped_dispatcher.hpp"
#include "log/logging.hpp"
#include "memory/topology_index.hpp"
#include "op/scan/duckdb_mvcc_visibility.hpp"
#include "op/scan/duckdb_native_metadata.hpp"

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb/transaction/transaction_data.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <future>
#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace sirius::scan_manager {

void fan_out_and_join(exec::scoped_dispatcher& dispatcher,
                      std::vector<sirius::exec::invocable<void()>> tasks,
                      std::string_view label)
{
  if (tasks.empty()) { return; }
  auto const n_tasks = tasks.size();

  struct join_state {
    std::atomic<std::size_t> completed{0};
    std::once_flag first_error_once;
    std::exception_ptr first_error;
    std::promise<void> done;
  };
  auto state       = std::make_shared<join_state>();
  auto done_future = state->done.get_future();

  exec::completion_controller controller;
  auto completion_token = controller.on_completion([state] { state->done.set_value(); });

  for (auto& task : tasks) {
    // Slot acquired BEFORE the enqueue and moved into the lambda: a stopping
    // dispatcher's silent enqueue drop (or a skipped-after-stop lambda)
    // destroys the lambda, releasing the slot — the join fires either way and
    // the count check below makes the drop loud.
    dispatcher.enqueue([state, task = std::move(task), slot = controller.acquire()]() mutable {
      try {
        task();
        state->completed.fetch_add(1, std::memory_order_release);
      } catch (...) {
        // An errored task deliberately does not count as completed; the first
        // error is what the join rethrows.
        std::call_once(state->first_error_once,
                       [&] { state->first_error = std::current_exception(); });
      }
    });
  }
  controller.close();
  done_future.wait();

  if (state->first_error) { std::rethrow_exception(state->first_error); }
  auto const completed = state->completed.load(std::memory_order_acquire);
  if (completed != n_tasks) {
    throw std::runtime_error("[fan_out_and_join] " + std::string(label) + ": only " +
                             std::to_string(completed) + " of " + std::to_string(n_tasks) +
                             " tasks ran — the dispatcher stopped mid-fan-out; refusing to "
                             "continue with incomplete results");
  }
}

namespace {

namespace ccm = cucascade::memory;

/// Keeps a NUMA node's pinned mask storage alive for as long as any published
/// mask references it. Member order matters: `blocks` returns its chunks to
/// the pool referencing `reservation` in its destructor, so the reservation
/// must be destroyed AFTER the blocks (declared first = destroyed last).
struct pinned_mask_bundle {
  std::unique_ptr<ccm::reservation> reservation;
  std::unique_ptr<ccm::fixed_size_host_memory_resource::multiple_blocks_allocation> blocks;
};

/// The host NUMA node a chunk's mask should stage on: HOST-tier spaces are
/// keyed by NUMA node directly; GPU-tier spaces map device -> node through
/// the topology (unknown -> node 0, the decoder's normalization).
std::size_t numa_node_for(ccm::memory_space const& space,
                          sirius::memory::topology_index const& topology)
{
  int numa = -1;
  if (space.get_tier() == ccm::Tier::HOST) {
    numa = space.get_device_id();
  } else {
    numa = topology.numa_node_of(space.get_device_id());
  }
  return numa < 0 ? 0 : static_cast<std::size_t>(numa);
}

}  // namespace

mvcc_mask_workset prepare_mvcc_mask_tasks(
  std::span<mvcc_mask_job_request> requests,
  cucascade::memory::memory_reservation_manager& reservation_manager,
  sirius::memory::topology_index const& topology)
{
  mvcc_mask_workset workset;
  if (requests.empty()) { return workset; }

  // Stage 1 — SERIAL captures (prepare thread; ClientContext discipline).
  // Plans land in the workset (they own the row-group slices the works
  // reference); plan_requests[i] pairs workset.plans[i] with its request.
  workset.plans.reserve(requests.size());
  std::vector<mvcc_mask_job_request*> plan_requests;
  plan_requests.reserve(requests.size());
  bool any_dirty = false;
  for (auto& request : requests) {
    if (!request.storage || !request.context) {
      throw std::runtime_error(
        "[prepare_mvcc_mask_tasks] malformed mask request for pinned entry '" + request.entry_name +
        "'");
    }
    op::scan::mvcc_visibility_plan plan = [&] {
      try {
        return op::scan::capture_mvcc_visibility_plan(
          *request.storage, *request.context, request.metadata);
      } catch (std::exception const& e) {
        throw std::runtime_error("[prepare_mvcc_mask_tasks] pinned entry '" + request.entry_name +
                                 "': " + e.what());
      }
    }();
    if (plan.mvcc_row_groups.size() != request.masks.size() ||
        plan.mvcc_row_groups.size() != request.chunk_spaces.size()) {
      throw std::runtime_error("[prepare_mvcc_mask_tasks] pinned entry '" + request.entry_name +
                               "': chunk count mismatch between the visibility plan (" +
                               std::to_string(plan.mvcc_row_groups.size()) + "), the mask set (" +
                               std::to_string(request.masks.size()) + ") and chunk_spaces (" +
                               std::to_string(request.chunk_spaces.size()) + ")");
    }
    any_dirty = any_dirty || plan.any_version_state();
    workset.plans.push_back(std::move(plan));
    plan_requests.push_back(&request);
  }
  if (!any_dirty) {
    // Every mask slot stays default — the unmasked fast path.
    workset.plans.clear();
    return workset;
  }

  // Stage 2 — pinned mask storage, reservation-first (the decoder's staging
  // pattern). Lay dirty chunks out per NUMA node first (block size is uniform
  // across the per-NUMA host spaces — one host config), then make ONE
  // consolidated reservation + multi-block allocation per node. Each dirty
  // chunk's carved mask is installed into its request's set right here — the
  // aliasing words pointer IS the storage owner — with the bits uninitialized
  // until the fill tasks run.
  auto host_spaces = reservation_manager.get_memory_spaces_for_tier(ccm::Tier::HOST);
  if (host_spaces.empty()) {
    throw std::runtime_error("[prepare_mvcc_mask_tasks] no HOST-tier memory space registered");
  }
  auto const* probe_fsmr =
    host_spaces.front()->get_memory_resource_as<ccm::fixed_size_host_memory_resource>();
  if (probe_fsmr == nullptr) {
    throw std::runtime_error(
      "[prepare_mvcc_mask_tasks] host memory space is not a fixed_size_host_memory_resource");
  }
  auto const block_size = probe_fsmr->get_block_size();

  struct mask_slot {
    std::size_t plan;
    std::size_t chunk;
    std::size_t block;
    std::size_t offset;
    std::size_t bytes;
    std::size_t rows;
  };
  struct node_layout {
    std::size_t blocks_used = 0;
    std::size_t cur_offset  = 0;
    std::vector<mask_slot> slots;
  };
  constexpr std::size_t kMaskAlign = 64;  // cache-line-aligned carve within blocks
  std::map<std::size_t, node_layout> nodes;

  auto make_work = [&workset, &plan_requests](std::size_t p, std::size_t c) {
    auto& plan        = workset.plans[p];
    auto work         = std::make_unique<mvcc_mask_work>();
    work->masks       = &plan_requests[p]->masks;
    work->chunk_index = c;
    work->slices = std::span<op::scan::mvcc_row_group_slice const>(plan.mvcc_row_groups[c].data(),
                                                                   plan.mvcc_row_groups[c].size());
    work->transaction = plan.transaction;
    return work;
  };

  for (std::size_t p = 0; p < workset.plans.size(); ++p) {
    auto& plan = workset.plans[p];
    for (std::size_t c = 0; c < plan.mvcc_row_groups.size(); ++c) {
      if (!plan.chunk_has_version_state[c] || plan.mvcc_row_groups[c].empty()) { continue; }
      std::size_t rows = 0;
      for (auto const& slice : plan.mvcc_row_groups[c]) {
        rows += slice.row_count;
      }
      auto const n_words = (rows + 31) / 32;
      auto const bytes   = n_words * sizeof(std::uint32_t);
      if (bytes > block_size) {
        // Staging blocks are not virtually contiguous, so a mask cannot span
        // two. A chunk this large (block_size x 8 rows and up — narrow
        // columns under a big batch budget) keeps a plain pageable mask
        // instead: cudaMemcpyAsync stages pageable memory out before
        // returning, so only the true-async-DMA benefit is lost, and only
        // for these oversized chunks.
        SIRIUS_LOG_INFO(
          "[prepare_mvcc_mask_tasks] pinned entry '{}': chunk {} mask ({} bytes) exceeds one "
          "staging block ({} bytes); using pageable host memory for it",
          plan_requests[p]->entry_name,
          c,
          bytes,
          block_size);
        auto storage = std::make_shared<std::vector<std::uint32_t>>(n_words, 0u);
        plan_requests[p]->masks[c] =
          mvcc_chunk_mask{std::shared_ptr<std::uint32_t[]>(storage, storage->data()), rows};
        workset.works.push_back(make_work(p, c));
        continue;
      }
      auto* space = plan_requests[p]->chunk_spaces[c];
      if (space == nullptr) {
        throw std::runtime_error("[prepare_mvcc_mask_tasks] pinned entry '" +
                                 plan_requests[p]->entry_name + "': chunk " + std::to_string(c) +
                                 " has no memory space");
      }
      auto& node      = nodes[numa_node_for(*space, topology)];
      node.cur_offset = (node.cur_offset + kMaskAlign - 1) / kMaskAlign * kMaskAlign;
      if (node.blocks_used == 0 || node.cur_offset + bytes > block_size) {
        ++node.blocks_used;
        node.cur_offset = 0;
      }
      node.slots.push_back({p, c, node.blocks_used - 1, node.cur_offset, bytes, rows});
      node.cur_offset += bytes;
    }
  }

  for (auto& [numa_node, layout] : nodes) {
    auto const total_bytes = layout.blocks_used * block_size;
    ccm::any_memory_space_in_tier_with_preference host_req(ccm::Tier::HOST, numa_node);
    auto reservation = reservation_manager.request_reservation(host_req, total_bytes);
    if (!reservation) {
      throw std::runtime_error("[prepare_mvcc_mask_tasks] failed to reserve " +
                               std::to_string(total_bytes) +
                               " bytes of pinned host memory for MVCC keep-masks (NUMA node " +
                               std::to_string(numa_node) + ")");
    }
    auto* fsmr = reservation->get_memory_space()
                   .get_memory_resource_as<ccm::fixed_size_host_memory_resource>();
    if (fsmr == nullptr || fsmr->get_block_size() != block_size) {
      // The strategy may fall back to another host space; the layout above
      // assumed one uniform block size, so a mismatch would mis-carve.
      throw std::runtime_error(
        "[prepare_mvcc_mask_tasks] reserved host space is not a fixed_size_host_memory_resource "
        "with the expected block size");
    }
    auto bundle         = std::make_shared<pinned_mask_bundle>();
    bundle->reservation = std::move(reservation);
    bundle->blocks      = fsmr->allocate_multiple_blocks(total_bytes, bundle->reservation.get());

    for (auto const& slot : layout.slots) {
      auto* base  = reinterpret_cast<std::uint8_t*>(bundle->blocks->at(slot.block).data());
      auto* words = reinterpret_cast<std::uint32_t*>(base + slot.offset);
      // Zero-initialize the carve: the fill writer is bit-exact at word edges
      // and must never read indeterminate memory through them.
      std::fill_n(words, slot.bytes / sizeof(std::uint32_t), 0u);
      plan_requests[slot.plan]->masks[slot.chunk] =
        mvcc_chunk_mask{std::shared_ptr<std::uint32_t[]>(bundle, words), slot.rows};
      workset.works.push_back(make_work(slot.plan, slot.chunk));
    }
  }

  // Stage 3 — build one fill task per <= metadata_parse_chunk() row groups (a
  // task never spans chunks: slices are word-disjoint across tasks only
  // because they cut on 32-row-aligned row-group boundaries within one
  // chunk's mask). Tasks write through their chunk's installed mask slot —
  // request sets are sized once at request creation and never resized, and
  // the workset's unique_ptrs keep the work addresses stable.
  auto const parse_chunk = op::scan::metadata_parse_chunk();
  for (auto& work_ptr : workset.works) {
    auto* work = work_ptr.get();
    // Parallel slicing requires every slice to start on a mask-word boundary
    // (concurrent tasks must never share a word). Checkpoint-grown tables can
    // have row groups of any size, so a chunk with unaligned offsets fills
    // through a single task; other chunks still fill in parallel.
    bool const word_aligned = std::all_of(work->slices.begin(),
                                          work->slices.end(),
                                          [](auto const& s) { return s.chunk_offset % 32 == 0; });
    auto const step         = word_aligned ? parse_chunk : work->slices.size();
    for (std::size_t begin = 0; begin < work->slices.size(); begin += step) {
      auto const count = std::min(step, work->slices.size() - begin);
      workset.fill_tasks.push_back([work, begin, count] {
        auto const& mask = (*work->masks)[work->chunk_index];
        std::span<std::uint32_t> words{mask.words.get(), (mask.row_count + 31) / 32};
        if (op::scan::fill_keep_mask_for_row_groups(
              work->slices.subspan(begin, count), work->transaction, words)) {
          work->any_deleted.store(true, std::memory_order_relaxed);
        }
      });
    }
  }
  return workset;
}

void finalize_mvcc_masks(mvcc_mask_workset& workset)
{
  // Masks whose fill dropped rows are already in place (prepare installed the
  // words, the fill tasks wrote the bits). Chunks that dropped nothing reset
  // to the default slot (served unmasked: no upload, no kernel), releasing
  // their storage ref — the node bundle frees once the last kept mask
  // releases, or right here when none survives.
  for (auto& work_ptr : workset.works) {
    auto& work = *work_ptr;
    if (work.any_deleted.load(std::memory_order_relaxed)) { continue; }
    (*work.masks)[work.chunk_index] = mvcc_chunk_mask{};
  }
}

void run_mvcc_mask_jobs(std::span<mvcc_mask_job_request> requests,
                        exec::scoped_dispatcher& dispatcher,
                        cucascade::memory::memory_reservation_manager& reservation_manager,
                        sirius::memory::topology_index const& topology)
{
  auto workset = prepare_mvcc_mask_tasks(requests, reservation_manager, topology);
  if (workset.fill_tasks.empty()) { return; }
  SIRIUS_LOG_DEBUG("[run_mvcc_mask_jobs] {} dirty chunk(s), {} fill task(s) across {} request(s)",
                   workset.works.size(),
                   workset.fill_tasks.size(),
                   requests.size());
  fan_out_and_join(dispatcher, std::move(workset.fill_tasks), "mvcc keep-mask fill");
  finalize_mvcc_masks(workset);
}

}  // namespace sirius::scan_manager
