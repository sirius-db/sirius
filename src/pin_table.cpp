/*
 * Copyright 2025, Sirius Contributors.
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

#include "pin_table.hpp"

#include "io/io_context.hpp"
#include "op/scan/gpu_ingestible.hpp"
#include "scan_manager/round_robin_strategy.hpp"

#include <cudf/table/table.hpp>

#include <rmm/cuda_device.hpp>

#include <cucascade/memory/memory_space.hpp>

#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace duckdb {

namespace {

std::mutex& recorded_calls_mutex()
{
  static std::mutex m;
  return m;
}

std::vector<PinTableArgs>& recorded_calls_storage()
{
  static std::vector<PinTableArgs> calls;
  return calls;
}

std::vector<std::string>& recorded_unpin_calls_storage()
{
  static std::vector<std::string> calls;
  return calls;
}

}  // namespace

void pin_table_to(const PinTableArgs& args)
{
  std::lock_guard<std::mutex> lock(recorded_calls_mutex());
  recorded_calls_storage().push_back(args);
}

void unpin_table_to(const std::string& name)
{
  std::lock_guard<std::mutex> lock(recorded_calls_mutex());
  recorded_unpin_calls_storage().push_back(name);
}

namespace pin_table_testing {

const std::vector<PinTableArgs>& recorded_calls() { return recorded_calls_storage(); }

void clear_recorded_calls()
{
  std::lock_guard<std::mutex> lock(recorded_calls_mutex());
  recorded_calls_storage().clear();
}

const std::vector<std::string>& recorded_unpin_calls() { return recorded_unpin_calls_storage(); }

void clear_recorded_unpin_calls()
{
  std::lock_guard<std::mutex> lock(recorded_calls_mutex());
  recorded_unpin_calls_storage().clear();
}

}  // namespace pin_table_testing

}  // namespace duckdb

namespace sirius {

materialized_pin materialize_all_batches(
  op::scan::gpu_ingestible& ingestible,
  std::span<cucascade::memory::memory_space* const> gpu_spaces,
  io::sirius_ioctx& io_ctx)
{
  if (gpu_spaces.empty()) {
    throw std::invalid_argument("[materialize_all_batches] gpu_spaces must be non-empty");
  }

  // GPU placement via the same strategy the query path uses
  // (sirius_scan_manager::prepare_for_query). A fresh strategy per call starts its
  // cursor at 0, so re-pinning the same source yields identical placement — required
  // by insert_pinned_entry's merge path. The strategy hands out device ids; map each
  // back to the memory_space to materialize into.
  std::vector<int> device_ids;
  std::unordered_map<int, cucascade::memory::memory_space*> space_by_device;
  device_ids.reserve(gpu_spaces.size());
  for (auto* sp : gpu_spaces) {
    device_ids.push_back(sp->get_device_id());
    space_by_device.emplace(sp->get_device_id(), sp);
  }
  scan_manager::round_robin_strategy placement(std::move(device_ids));

  // next_split_provider takes the io_ctx by shared_ptr; sirius_ioctx derives
  // std::enable_shared_from_this and the scan manager owns it via a shared_ptr, so
  // this hands the metadata reads a valid owning reference for the read's duration.
  auto io_ctx_sp = io_ctx.shared_from_this();

  materialized_pin out;
  auto coalecer = ingestible.create_batch_coalecer();

  // Materialize one coalesced batch into a GPU-resident cudf::table and record its
  // GPU placement. Mirrors load_balancing_scan_batch_coalecer::process_provider_inputs,
  // minus the connector/balancer and the post-decode step: there is no downstream scan
  // operator at pin time, so the unfiltered table is collected here directly.
  auto handle_batch = [&](std::unique_ptr<op::scan::scan_info> batch) {
    if (!batch) { return; }
    // round_robin_strategy ignores pipeline_id/data; it returns the next device id
    // from its cursor. space_by_device round-trips it to the materialization target.
    int const gpu_id = placement.get_next_gpu(/*pipeline_id=*/0, /*data=*/nullptr, /*hint=*/{});
    auto* target     = space_by_device.at(gpu_id);
    rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{gpu_id}};
    // Borrow a real (non-default) stream from the target memory_space's pool: the
    // duckdb-native decoder's cudaMemcpyBatchAsync rejects the legacy default
    // stream. The pool owns the stream for the memory_space's lifetime (which
    // outlives the pinned data), so the materialized buffers' deallocation stream
    // stays valid after this call.
    auto stream = target->acquire_stream();

    // A pin caches the table UNFILTERED: it has no row filter (no WHERE), and the
    // ingestible's projection_ids are identity into column_ids, so the reader already
    // emits exactly the pinned columns in column_ids order. materialize_metadata_to_table
    // thus yields the cache-ready table directly — no scan_operator_input wrapper and no
    // post_filter_and_project (which would only apply a row filter or re-project to a
    // query's output layout, neither of which a pin needs). `batch` is borrowed by const
    // ref and outlives the call.
    auto materialized = ingestible.materialize_metadata_to_table(*batch, *target, stream);
    auto tbl          = materialized.table.release(stream, target->get_default_allocator());
    // Cached batches are built with a null writer stream, so the data must be
    // fully resident before it can be served or host-converted.
    stream.synchronize();
    out.tables.emplace_back(std::move(tbl));
    out.chunk_memory_spaces.push_back(target);
  };

  while (!ingestible.has_processed_all_metadata()) {
    auto task = ingestible.next_split_provider(io_ctx_sp);
    if (!task) { continue; }
    auto info = task();
    if (!info) { continue; }
    for (auto& b : coalecer->push(std::move(info))) {
      handle_batch(std::move(b));
    }
  }
  for (auto& b : coalecer->flush()) {
    handle_batch(std::move(b));
  }
  return out;
}

}  // namespace sirius
