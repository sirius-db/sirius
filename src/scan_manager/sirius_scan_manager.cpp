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

#include "scan_manager/sirius_scan_manager.hpp"

#include "exec/thread_pool.hpp"
#include "io/gpu_ingestible.hpp"
#include "io/parquet_helpers.hpp"
#include "io/prefetching_cache.hpp"
#include "io/s3/s3_blocking_ioctx.hpp"
#include "io/uring/uring_ioctx.hpp"
#include "log/logging.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "planner/query.hpp"
#include "scan_manager/parquet_metadata.hpp"
#include "scan_manager/split_connector.hpp"
#include "scan_manager/split_provider.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/utilities/span.hpp>

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <utility>

namespace sirius::scan_manager {

sirius_scan_manager::sirius_scan_manager(
  scan_manager_config config, std::vector<std::shared_ptr<sirius::io::sirius_ioctx>> io_ctxs)
  : _config(std::move(config)),
    _thread_pool(_config.thread_pool.num_threads,
                 _config.thread_pool.thread_name_prefix,
                 _config.thread_pool.cpu_affinity_list),
    _dispatcher(
      std::make_unique<exec::scoped_dispatcher>(_thread_pool, _config.thread_pool.num_threads)),
    _io_ctxs(std::move(io_ctxs)),
    _factory(_pinned_entries)
{
  // S6 (NUMA) increment 1: the scan_manager no longer constructs IO backends.
  // SiriusContext owns the uring(s) + s3_ioctx + S3 async thread pool + prefetch
  // buffer_pool/cache and passes the routing backends in as borrowed
  // shared_ptrs. io_ctx_for / io_ctx_shared_for dispatch over this borrowed
  // list; stop() and the destructor must NOT shut down or destroy them.
  SIRIUS_LOG_DEBUG("[sirius_scan_manager] constructed with {} borrowed IO backend(s)",
                   _io_ctxs.size());
}

sirius_scan_manager::~sirius_scan_manager()
{
  // S6: backends + their caches are owned by SiriusContext (which logs cache
  // summaries on its own teardown). Only stop our scan-orchestration pool here;
  // do not touch the borrowed backends.
  stop();
}

namespace {

// Walk @c ioctxs and return the first @c ctx whose @c supports(path) is
// true, or nullptr. Also tries with @c "file://" prefix stripped because
// @c uring_reactor::supports (from #740) calls @c is_regular_file on the
// raw input — so it accepts bare absolute paths but not @c file:// URIs.
// Stripping at the dispatch layer keeps #740's code untouched and works
// for the both-shape inputs Sirius's parquet plans can produce.
template <typename Container, typename Out>
Out lookup_supporting(Container const& ioctxs,
                      std::string_view path,
                      Out (*get_value)(typename Container::value_type const&))
{
  for (auto const& ctx : ioctxs) {
    if (ctx && ctx->supports(path)) return get_value(ctx);
  }
  constexpr std::string_view kFileScheme = "file://";
  if (path.size() > kFileScheme.size() && path.substr(0, kFileScheme.size()) == kFileScheme) {
    auto bare = path.substr(kFileScheme.size());
    for (auto const& ctx : ioctxs) {
      if (ctx && ctx->supports(bare)) return get_value(ctx);
    }
  }
  return Out{};
}

sirius::io::sirius_ioctx* raw_ptr(std::shared_ptr<sirius::io::sirius_ioctx> const& ctx)
{
  return ctx.get();
}

std::shared_ptr<sirius::io::sirius_ioctx> shared_copy(
  std::shared_ptr<sirius::io::sirius_ioctx> const& ctx)
{
  return ctx;
}

}  // namespace

sirius::io::sirius_ioctx* sirius_scan_manager::io_ctx_for(std::string_view path) const noexcept
{
  return lookup_supporting<decltype(_io_ctxs), sirius::io::sirius_ioctx*>(_io_ctxs, path, raw_ptr);
}

std::shared_ptr<sirius::io::sirius_ioctx> sirius_scan_manager::io_ctx_shared_for(
  std::string_view path) const noexcept
{
  return lookup_supporting<decltype(_io_ctxs), std::shared_ptr<sirius::io::sirius_ioctx>>(
    _io_ctxs, path, shared_copy);
}

parquet_bind_result sirius_scan_manager::describe_parquet(std::string const& uri)
{
  auto* io_ctx = io_ctx_for(uri);
  if (io_ctx == nullptr) {
    throw std::runtime_error("[sirius_scan_manager::describe_parquet] no backend supports URI: " +
                             uri);
  }

  // Footer-only fetch + Thrift parse — the same path parquet_split_provider's
  // run_batch takes on a metadata-cache miss, so bind and scan agree on how
  // the footer is read.
  auto io_object  = io_ctx->create_io_object(uri);
  auto datasource = io_ctx->make_datasource(io_object);

  auto footer_buffer         = cudf::io::parquet::fetch_footer_to_host(*datasource);
  auto const footer_byte_len = footer_buffer->size();
  auto reader_options        = cudf::io::parquet_reader_options::builder().build();
  cudf::io::parquet::experimental::hybrid_scan_reader reader{
    cudf::host_span<std::uint8_t const>(footer_buffer->data(), footer_buffer->size()),
    reader_options};
  auto file_metadata         = reader.parquet_metadata();
  auto const footer_num_rows = file_metadata.num_rows;

  auto schema = sirius::io::parquet_helpers::extract_schema(file_metadata);

  // Footer-parse reuse: a metadata-only insert (empty ranges => no chunk
  // prefetch) lets the subsequent scan's get_metadata hit, so the footer is
  // Thrift-parsed once instead of twice.
  if (auto* cache = io_ctx->cache(); cache != nullptr) {
    auto metadata = std::make_shared<parquet_metadata>(
      std::make_shared<cudf::io::parquet::FileMetaData const>(std::move(file_metadata)),
      footer_byte_len);
    cache->insert(*io_object, std::move(metadata), /*ranges=*/{});
  }

  parquet_bind_result result;
  result.return_types   = std::move(schema.types);
  result.names          = std::move(schema.names);
  result.object_size    = datasource->size();
  result.total_num_rows = static_cast<std::size_t>(footer_num_rows);
  return result;
}

void sirius_scan_manager::prepare_for_query(
  const sirius::planner::query& query,
  std::unordered_map<int, std::shared_ptr<sirius::io::sirius_ioctx>> const& gpu_ioctxs,
  std::unordered_map<int, cucascade::memory::memory_space*> const& gpu_memory_spaces)
{
  reset();

  // Advance the cache age so the evictor can score this query's inserts
  // against entries left over from prior queries. The buffer_pool (and the
  // prefetching_cache on each ioctx that draws from it) is shared across all
  // registered backends, so refreshing one cache covers all of them — but
  // calling refresh on each is safe and explicit; pick whichever has one.
  for (auto const& ctx : _io_ctxs) {
    if (ctx && ctx->cache()) { ctx->cache()->refresh_cache(); }
  }

  SIRIUS_LOG_DEBUG(
    "[sirius_scan_manager::prepare_for_query] pipelines={} gpu_ioctxs={} gpu_memory_spaces={}",
    query.get_pipelines().size(),
    gpu_ioctxs.size(),
    gpu_memory_spaces.size());

  for (auto const& pipeline : query.get_pipelines()) {
    if (!pipeline) { continue; }
    auto source = pipeline->get_source();
    if (!source) { continue; }
    if (source->type != ::sirius::op::SiriusPhysicalOperatorType::GPU_SCAN) { continue; }

    auto* op = &source->Cast<op::scan::sirius_gpu_scan_operator>();
    if (_providers_by_op.find(op) != _providers_by_op.end()) { continue; }

    auto table_info = op->take_table_info();
    if (!table_info) { continue; }

    // When use_sirius_datasource=false (single-GPU only — multi-GPU runs are
    // forced to true by sirius_config::enforce_sirius_datasource_for_multi_gpu()),
    // suppress the per-GPU uring map. Local files then route through
    // cudf::io::datasource::create (kvikio fallback, safe with one GPU); S3
    // paths are unaffected (they dispatch through the scan_manager's _io_ctxs
    // vector independently).
    std::unordered_map<int, std::shared_ptr<sirius::io::sirius_ioctx>> const empty_gpu_ioctxs;
    auto const& effective_gpu_ioctxs =
      _config.use_sirius_datasource ? gpu_ioctxs : empty_gpu_ioctxs;

    // Inject the per-GPU ioctx map into the operator before any provider returns
    // — read_table_from_metadata needs it before its first invocation, and
    // prepare_for_query runs before any execute().
    op->set_gpu_ioctxs(effective_gpu_ioctxs);

    auto ingestible = _factory.produce(std::move(table_info),
                                       *this,
                                       effective_gpu_ioctxs,
                                       gpu_memory_spaces,
                                       op->get_operator_id());
    if (!ingestible) { continue; }
    op->install_ingestible(ingestible);

    auto provider = std::make_unique<split_provider>(std::move(ingestible));
    op->set_split_connector(std::make_unique<split_connector>());
    _providers_by_op.emplace(op, std::move(provider));
    _scan_op_order.push_back(op);

    SIRIUS_LOG_DEBUG("[sirius_scan_manager::prepare_for_query] registered gpu scan op_id={}",
                     op->get_operator_id());
  }

  if (_scan_op_order.empty()) { return; }

  start_metadata_processing();
}

void sirius_scan_manager::start_metadata_processing()
{
  for (auto* op : _scan_op_order) {
    auto it = _providers_by_op.find(op);
    if (it == _providers_by_op.end()) { continue; }
    auto* connector = op->get_split_connector();
    if (connector == nullptr) { continue; }

    try {
      // run() is fire-and-forget: it enqueues workers and returns immediately.
      // Worker exceptions ride on connector.close(exception_ptr) and surface
      // when the consumer drains via get_next_split().
      it->second->run(*_dispatcher, *connector);
    } catch (const std::exception& e) {
      SIRIUS_LOG_ERROR("[sirius_scan_manager] driver: provider failed to start: {}", e.what());
      // Synchronous failure inside run() (e.g. scheduler.enqueue throwing)
      // bypasses the worker error path, so forward it through the connector
      // here. close() is idempotent and keeps the first stored exception.
      connector->close(std::current_exception());
    }
  }
}

void sirius_scan_manager::reset()
{
  _dispatcher->request_stop();
  _dispatcher->wait_for_all();
  _scan_op_order.clear();
  _providers_by_op.clear();
  _dispatcher =
    std::make_unique<exec::scoped_dispatcher>(_thread_pool, _config.thread_pool.num_threads);
}

void sirius_scan_manager::start() {}

void sirius_scan_manager::stop()
{
  reset();
  // S6: the IO backends + the S3 async pool are owned by SiriusContext, which
  // drains (shutdown) the s3_ioctx and stops the S3 pool during its own
  // teardown. The scan_manager only stops its scan-orchestration pool here and
  // must NOT shut down the borrowed backends.
  _thread_pool.stop();
}

void sirius_scan_manager::insert_pinned_entry(
  const std::string& name,
  std::vector<std::string> column_names,
  std::vector<std::string> file_paths,
  std::vector<std::unique_ptr<cudf::table>> data_tables,
  std::vector<cucascade::memory::memory_space*> chunk_memory_spaces,
  bool is_partial)
{
  // chunk_memory_spaces is parallel to data_tables — the caller
  // (PinTableFunction) emits one memory_space* per
  // chunked_parquet_reader::read_chunk() result, and there is exactly one
  // cudf::table per chunk in data_tables. Reject any misalignment loudly
  // rather than silently aliasing chunks to the wrong GPU.
  if (chunk_memory_spaces.size() != data_tables.size()) {
    throw std::invalid_argument(
      "[sirius_scan_manager::insert_pinned_entry] chunk_memory_spaces.size() (" +
      std::to_string(chunk_memory_spaces.size()) + ") must equal data_tables.size() (" +
      std::to_string(data_tables.size()) + ")");
  }

  // Compute the total row count of the incoming tables before releasing them
  // (release() empties the table; num_rows() would then return 0).
  std::size_t new_num_rows = 0;
  for (auto const& table : data_tables) {
    if (table) { new_num_rows += static_cast<std::size_t>(table->num_rows()); }
  }

  auto existing_it = _pinned_entries.find(name);
  if (existing_it != _pinned_entries.end()) {
    // Same-row-count merge only applies when the completeness contracts match.
    // Mixing a full pin with a partial pin produces an entry whose columns came
    // from different row coverage — drop and rebuild instead.
    if (existing_it->second.num_rows == new_num_rows &&
        existing_it->second.is_partial == is_partial) {
      // Same-row-count merge MUST preserve per-chunk memory_space alignment
      // between existing and new entry. The round-robin counter restarts at
      // chunk 0 → GPU 0 per pin_table call, and chunks at index i across all
      // columns share a memory_space because they came from the same
      // chunked_parquet_reader::read_chunk() call. Two pin_table calls of the
      // same file_paths with the same chunk_read_limit MUST therefore produce
      // identical chunk_memory_spaces vectors. Reject any mismatch loudly
      // rather than silently aliasing.
      auto& entry = existing_it->second;
      if (entry.chunk_memory_spaces.size() != chunk_memory_spaces.size()) {
        throw std::runtime_error(
          "[sirius_scan_manager::insert_pinned_entry] merge mismatch — "
          "existing.chunk_memory_spaces.size() (" +
          std::to_string(entry.chunk_memory_spaces.size()) +
          ") != new chunk_memory_spaces.size() (" + std::to_string(chunk_memory_spaces.size()) +
          ")");
      }
      for (std::size_t i = 0; i < chunk_memory_spaces.size(); ++i) {
        if (entry.chunk_memory_spaces[i] != chunk_memory_spaces[i]) {
          throw std::runtime_error(
            "[sirius_scan_manager::insert_pinned_entry] merge mismatch — "
            "chunk_memory_spaces[" +
            std::to_string(i) + "] differs between existing and new entry");
        }
      }
      // Same row count → merge unique columns into the existing entry.
      // Decide which column INDICES are new BEFORE iterating chunks. Doing
      // the contains() check per-chunk would let chunk 0 install a new
      // column and then chunks 1..N-1 see contains()==true and skip — leaving
      // the new column with only chunk 0 and tripping cached_split_provider's
      // "mismatched chunk count across requested columns" invariant.
      std::vector<bool> is_new_col(column_names.size(), false);
      for (std::size_t i = 0; i < column_names.size(); ++i) {
        is_new_col[i] = !entry.data_batches_by_column.contains(column_names[i]);
      }
      for (auto& table : data_tables) {
        if (!table) { continue; }
        auto cols = table->release();
        if (cols.size() != column_names.size()) {
          throw std::runtime_error(
            "[sirius_scan_manager::insert_pinned_entry] table column count " +
            std::to_string(cols.size()) + " does not match column_names size " +
            std::to_string(column_names.size()));
        }
        for (std::size_t i = 0; i < cols.size(); ++i) {
          if (!is_new_col[i]) {
            // Column was already cached before this merge call — drop the
            // duplicate chunk.
            continue;
          }
          entry.data_batches_by_column[column_names[i]].emplace_back(std::move(cols[i]));
        }
      }
      // Append any new column names to the entry's column_names list so its
      // metadata reflects the union of pinned columns.
      for (auto& cn : column_names) {
        if (std::find(entry.column_names.begin(), entry.column_names.end(), cn) ==
            entry.column_names.end()) {
          entry.column_names.push_back(std::move(cn));
        }
      }
      return;
    }
    // Row count or completeness contract differs → drop the stale entry and rebuild below.
    _pinned_entries.erase(existing_it);
  }

  pinned_entry entry;
  entry.column_names        = std::move(column_names);
  entry.file_paths          = std::move(file_paths);
  entry.chunk_memory_spaces = std::move(chunk_memory_spaces);
  entry.tier                = cucascade::memory::Tier::GPU;
  entry.num_rows            = new_num_rows;
  entry.is_partial          = is_partial;

  for (auto& table : data_tables) {
    if (!table) { continue; }
    auto cols = table->release();
    if (cols.size() != entry.column_names.size()) {
      throw std::runtime_error("[sirius_scan_manager::insert_pinned_entry] table column count " +
                               std::to_string(cols.size()) + " does not match column_names size " +
                               std::to_string(entry.column_names.size()));
    }
    for (std::size_t i = 0; i < cols.size(); ++i) {
      entry.data_batches_by_column[entry.column_names[i]].emplace_back(std::move(cols[i]));
    }
  }

  _pinned_entries[name] = std::move(entry);
}

void sirius_scan_manager::insert_pinned_entry_host(
  const std::string& name,
  std::vector<std::string> column_names,
  std::vector<std::string> file_paths,
  std::vector<std::shared_ptr<cucascade::host_data_representation>> host_chunks,
  cucascade::memory::memory_space& memory_space,
  bool is_partial)
{
  // The host-tier path captures one chunk per emitted batch; each chunk holds every
  // pinned column. Re-insert always replaces — there is no per-column merge analog
  // to the GPU path because the chunk-vs-column dimensions are flipped.
  std::size_t new_num_rows = 0;
  for (auto const& chunk : host_chunks) {
    if (!chunk) { continue; }
    auto const& host_table = chunk->get_host_table();
    if (host_table && !host_table->columns.empty()) {
      new_num_rows += static_cast<std::size_t>(host_table->columns.front().num_rows);
    }
  }

  pinned_entry entry;
  entry.column_names = std::move(column_names);
  entry.file_paths   = std::move(file_paths);
  entry.tier         = cucascade::memory::Tier::HOST;
  entry.memory_space = &memory_space;
  entry.num_rows     = new_num_rows;
  entry.host_chunks  = std::move(host_chunks);
  entry.is_partial   = is_partial;

  _pinned_entries[name] = std::move(entry);
}

void sirius_scan_manager::remove_pinned_entry(const std::string& name)
{
  _pinned_entries.erase(name);
}

}  // namespace sirius::scan_manager
