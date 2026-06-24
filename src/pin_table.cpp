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

#include "data/sirius_converter_registry.hpp"
#include "io/io_context.hpp"
#include "op/scan/gpu_ingestible.hpp"
#include "scan_manager/round_robin_strategy.hpp"

#include <cudf/table/table.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <functional>
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

namespace {

/// Per-materialized-batch sink: receives one GPU-resident table, its GPU placement, and the
/// stream it was decoded on. The driver does NOT synchronize — the sink owns the sync, because
/// the host-streaming path must synchronize while the GPU table (and the gpu_table_representation
/// wrapping it) is still alive, after the D2H conversion has been enqueued on the same stream.
using pin_batch_sink = std::function<void(std::unique_ptr<cudf::table>,
                                          cucascade::memory::memory_space* target,
                                          rmm::cuda_stream_view stream)>;

/// Shared driver behind @ref materialize_all_batches and @ref materialize_pin_to_host: walk the
/// ingestible's metadata + batch coalescer to completion, materialize each emitted batch onto a
/// round-robin GPU, and hand it to @p on_batch. The single-threaded, deterministic round-robin
/// placement means re-pinning the same source yields identical placement (required by
/// insert_pinned_entry's merge path on the GPU tier) and bounds peak GPU residency to ~one batch
/// (the host tier frees each table in @p on_batch before the next is materialized).
void materialize_pin_batches(op::scan::gpu_ingestible& ingestible,
                             std::span<cucascade::memory::memory_space* const> gpu_spaces,
                             io::sirius_ioctx& io_ctx,
                             const pin_batch_sink& on_batch)
{
  if (gpu_spaces.empty()) {
    throw std::invalid_argument("[materialize_pin_batches] gpu_spaces must be non-empty");
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

  auto coalecer = ingestible.create_batch_coalecer();

  // Materialize one coalesced batch into a GPU-resident cudf::table and hand it to on_batch
  // together with its GPU placement + the decode stream. Mirrors
  // load_balancing_scan_batch_coalecer::process_provider_inputs, minus the connector/balancer
  // and the post-decode step: there is no downstream scan operator at pin time, so the
  // unfiltered table is delivered to the sink directly. The device guard stays in scope across
  // the on_batch call, so the sink runs on the correct device.
  auto handle_batch = [&](std::unique_ptr<op::scan::scan_info> batch) {
    if (!batch) { return; }
    // round_robin_strategy ignores pipeline_id/data; it returns the next device id
    // from its cursor. gpu_spaces is non-empty (checked above), so the strategy always
    // yields a device. space_by_device round-trips it to the materialization target.
    int const gpu_id =
      placement.get_next_gpu(/*pipeline_id=*/0, /*data=*/nullptr, /*hint=*/{}).value();
    auto* target = space_by_device.at(gpu_id);
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
    on_batch(std::move(tbl), target, stream);
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
}

}  // namespace

materialized_pin materialize_all_batches(
  op::scan::gpu_ingestible& ingestible,
  std::span<cucascade::memory::memory_space* const> gpu_spaces,
  io::sirius_ioctx& io_ctx)
{
  materialized_pin out;
  materialize_pin_batches(ingestible,
                          gpu_spaces,
                          io_ctx,
                          [&](std::unique_ptr<cudf::table> tbl,
                              cucascade::memory::memory_space* target,
                              rmm::cuda_stream_view stream) {
                            // Cached GPU batches are stored with a null writer stream, so the data
                            // must be fully resident before it can be served or host-converted.
                            stream.synchronize();
                            out.tables.emplace_back(std::move(tbl));
                            out.chunk_memory_spaces.push_back(target);
                          });
  return out;
}

std::vector<std::shared_ptr<cucascade::host_data_representation>> materialize_pin_to_host(
  op::scan::gpu_ingestible& ingestible,
  std::span<cucascade::memory::memory_space* const> gpu_spaces,
  const std::unordered_map<int, cucascade::memory::memory_space*>& host_space_by_gpu,
  io::sirius_ioctx& io_ctx)
{
  auto& registry = converter_registry::get();
  std::vector<std::shared_ptr<cucascade::host_data_representation>> host_chunks;

  materialize_pin_batches(
    ingestible,
    gpu_spaces,
    io_ctx,
    [&](std::unique_ptr<cudf::table> tbl,
        cucascade::memory::memory_space* src_space,
        rmm::cuda_stream_view stream) {
      // Stream this freshly-materialized batch straight to pinned host memory and let the GPU
      // table free before the next batch is materialized — so peak GPU residency stays at ~one
      // batch and the whole table never needs to fit in GPU memory. The chunk is pinned on the
      // source GPU's NUMA-local host space (host_space_by_gpu), so on multi-GPU systems the
      // chunks land round-robin across NUMA nodes. The conversion reuses the decode stream; the
      // GPU->HOST converter (convert_gpu_to_host_fast) synchronizes internally before returning,
      // so the host copy is complete once convert() returns. The explicit sync below is
      // belt-and-suspenders before gpu_repr (which owns the GPU table's buffers) leaves scope.
      auto* target_host_space = host_space_by_gpu.at(src_space->get_device_id());
      cucascade::gpu_table_representation gpu_repr(std::move(tbl), *src_space, stream);
      auto host_repr =
        registry.convert<cucascade::host_data_representation>(gpu_repr, target_host_space, stream);
      stream.synchronize();
      host_chunks.emplace_back(std::move(host_repr));
    });

  return host_chunks;
}

}  // namespace sirius
