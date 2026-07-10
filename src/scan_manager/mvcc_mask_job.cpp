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
                      std::vector<absl::AnyInvocable<void()>> tasks,
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

/// One dirty chunk's fill state; stable-address (tasks hold pointers).
struct chunk_work {
  mvcc_chunk_mask_set* masks{nullptr};
  std::size_t chunk_index{0};
  std::span<std::uint32_t> words;
  std::size_t row_count{0};
  std::shared_ptr<void> retention;
  std::span<op::scan::mvcc_row_group_slice const> slices;
  duckdb::TransactionData transaction{duckdb::TransactionData::Committed()};
  std::atomic<bool> any_deleted{false};
};

}  // namespace

void run_mvcc_mask_jobs(std::span<mvcc_mask_job_request> requests,
                        exec::scoped_dispatcher& dispatcher,
                        cucascade::memory::memory_reservation_manager& reservation_manager,
                        sirius::memory::topology_index const& topology)
{
  if (requests.empty()) { return; }

  // Phase 1 — SERIAL captures (prepare thread; ClientContext discipline).
  struct captured_job {
    mvcc_mask_job_request* request;
    op::scan::mvcc_visibility_plan plan;
  };
  std::vector<captured_job> jobs;
  jobs.reserve(requests.size());
  bool any_dirty = false;
  for (auto& request : requests) {
    if (!request.masks || !request.storage || !request.context) {
      throw std::runtime_error("[run_mvcc_mask_jobs] malformed mask request for pinned entry '" +
                               request.entry_name + "'");
    }
    op::scan::mvcc_visibility_plan plan = [&] {
      try {
        return op::scan::capture_mvcc_visibility_plan(
          *request.storage, *request.context, request.metadata);
      } catch (std::exception const& e) {
        throw std::runtime_error("[run_mvcc_mask_jobs] pinned entry '" + request.entry_name +
                                 "': " + e.what());
      }
    }();
    if (plan.chunks.size() != request.masks->size() ||
        plan.chunks.size() != request.chunk_spaces.size()) {
      throw std::runtime_error("[run_mvcc_mask_jobs] pinned entry '" + request.entry_name +
                               "': chunk count mismatch between the visibility plan (" +
                               std::to_string(plan.chunks.size()) + "), the mask set (" +
                               std::to_string(request.masks->size()) + ") and chunk_spaces (" +
                               std::to_string(request.chunk_spaces.size()) + ")");
    }
    any_dirty = any_dirty || plan.any_version_state();
    jobs.push_back({&request, std::move(plan)});
  }
  if (!any_dirty) { return; }  // every mask slot stays null — the unmasked fast path

  // Phase 2 — pinned mask storage, reservation-first (the decoder's staging
  // pattern). Lay dirty chunks out per NUMA node first (block size is uniform
  // across the per-NUMA host spaces — one host config), then make ONE
  // consolidated reservation + multi-block allocation per node.
  auto host_spaces = reservation_manager.get_memory_spaces_for_tier(ccm::Tier::HOST);
  if (host_spaces.empty()) {
    throw std::runtime_error("[run_mvcc_mask_jobs] no HOST-tier memory space registered");
  }
  auto const* probe_fsmr =
    host_spaces.front()->get_memory_resource_as<ccm::fixed_size_host_memory_resource>();
  if (probe_fsmr == nullptr) {
    throw std::runtime_error(
      "[run_mvcc_mask_jobs] host memory space is not a fixed_size_host_memory_resource");
  }
  auto const block_size = probe_fsmr->get_block_size();

  struct mask_slot {
    std::size_t job;
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
  std::vector<std::unique_ptr<chunk_work>> works;

  auto make_work = [&jobs](std::size_t j, std::size_t c, std::size_t rows) {
    auto& job         = jobs[j];
    auto work         = std::make_unique<chunk_work>();
    work->masks       = job.request->masks.get();
    work->chunk_index = c;
    work->row_count   = rows;
    work->slices      = std::span<op::scan::mvcc_row_group_slice const>(job.plan.chunks[c].data(),
                                                                   job.plan.chunks[c].size());
    work->transaction = job.plan.transaction;
    return work;
  };

  for (std::size_t j = 0; j < jobs.size(); ++j) {
    auto& job = jobs[j];
    for (std::size_t c = 0; c < job.plan.chunks.size(); ++c) {
      if (!job.plan.chunk_has_version_state[c] || job.plan.chunks[c].empty()) { continue; }
      std::size_t rows = 0;
      for (auto const& slice : job.plan.chunks[c]) {
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
          "[run_mvcc_mask_jobs] pinned entry '{}': chunk {} mask ({} bytes) exceeds one "
          "staging block ({} bytes); using pageable host memory for it",
          job.request->entry_name,
          c,
          bytes,
          block_size);
        auto storage    = std::make_shared<std::vector<std::uint32_t>>(n_words);
        auto work       = make_work(j, c, rows);
        work->words     = std::span<std::uint32_t>(storage->data(), storage->size());
        work->retention = std::move(storage);
        works.push_back(std::move(work));
        continue;
      }
      auto* space = job.request->chunk_spaces[c];
      if (space == nullptr) {
        throw std::runtime_error("[run_mvcc_mask_jobs] pinned entry '" + job.request->entry_name +
                                 "': chunk " + std::to_string(c) + " has no memory space");
      }
      auto& node      = nodes[numa_node_for(*space, topology)];
      node.cur_offset = (node.cur_offset + kMaskAlign - 1) / kMaskAlign * kMaskAlign;
      if (node.blocks_used == 0 || node.cur_offset + bytes > block_size) {
        ++node.blocks_used;
        node.cur_offset = 0;
      }
      node.slots.push_back({j, c, node.blocks_used - 1, node.cur_offset, bytes, rows});
      node.cur_offset += bytes;
    }
  }

  for (auto& [numa_node, layout] : nodes) {
    auto const total_bytes = layout.blocks_used * block_size;
    ccm::any_memory_space_in_tier_with_preference host_req(ccm::Tier::HOST, numa_node);
    auto reservation = reservation_manager.request_reservation(host_req, total_bytes);
    if (!reservation) {
      throw std::runtime_error("[run_mvcc_mask_jobs] failed to reserve " +
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
        "[run_mvcc_mask_jobs] reserved host space is not a fixed_size_host_memory_resource "
        "with the expected block size");
    }
    auto bundle         = std::make_shared<pinned_mask_bundle>();
    bundle->reservation = std::move(reservation);
    bundle->blocks      = fsmr->allocate_multiple_blocks(total_bytes, bundle->reservation.get());

    for (auto const& slot : layout.slots) {
      auto* base  = reinterpret_cast<std::uint8_t*>(bundle->blocks->at(slot.block).data());
      auto work   = make_work(slot.job, slot.chunk, slot.rows);
      work->words = std::span<std::uint32_t>(reinterpret_cast<std::uint32_t*>(base + slot.offset),
                                             slot.bytes / sizeof(std::uint32_t));
      work->retention = bundle;
      works.push_back(std::move(work));
    }
  }

  // Phase 3 — fan out one task per <= metadata_parse_chunk() row groups (a
  // task never spans chunks: slices are word-disjoint across tasks only
  // because they cut on 32-row-aligned row-group boundaries within one
  // chunk's mask), then block until every task ran.
  auto const parse_chunk = op::scan::metadata_parse_chunk();
  std::vector<absl::AnyInvocable<void()>> tasks;
  for (auto& work_ptr : works) {
    auto* work = work_ptr.get();
    for (std::size_t begin = 0; begin < work->slices.size(); begin += parse_chunk) {
      auto const count = std::min(parse_chunk, work->slices.size() - begin);
      tasks.push_back([work, begin, count] {
        if (op::scan::fill_keep_mask_for_row_groups(
              work->slices.subspan(begin, count), work->transaction, work->words)) {
          work->any_deleted.store(true, std::memory_order_relaxed);
        }
      });
    }
  }
  SIRIUS_LOG_DEBUG("[run_mvcc_mask_jobs] {} dirty chunk(s), {} fill task(s) across {} request(s)",
                   works.size(),
                   tasks.size(),
                   requests.size());
  fan_out_and_join(dispatcher, std::move(tasks), "mvcc keep-mask fill");

  // Phase 4 — publish. Chunks that dropped nothing keep a null slot (served
  // unmasked); their carved span is simply unused, and the node bundle frees
  // once the last published mask releases (or right here when none did).
  for (auto& work_ptr : works) {
    auto& work = *work_ptr;
    if (!work.any_deleted.load(std::memory_order_relaxed)) { continue; }
    auto mask                       = std::make_shared<mvcc_chunk_mask>();
    mask->words                     = work.words;
    mask->row_count                 = work.row_count;
    mask->retention                 = work.retention;
    (*work.masks)[work.chunk_index] = std::move(mask);
  }
}

}  // namespace sirius::scan_manager
