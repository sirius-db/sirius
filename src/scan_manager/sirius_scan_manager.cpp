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
#include "op/scan/parquet_gpu_ingestible.hpp"
#include "op/scan/pinned_table_gpu_ingestible.hpp"
#include "op/scan/scan_plan.hpp"
#include "op/scan/scan_utils.hpp"
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
    _io_ctxs(std::move(io_ctxs))
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

    auto ingestible = create_ingestible_for(op, gpu_ioctxs, gpu_memory_spaces);
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

std::shared_ptr<io::gpu_ingestible> sirius_scan_manager::create_ingestible_for(
  op::scan::sirius_gpu_scan_operator* op,
  std::unordered_map<int, std::shared_ptr<sirius::io::sirius_ioctx>> const& gpu_ioctxs,
  std::unordered_map<int, cucascade::memory::memory_space*> const& gpu_memory_spaces)
{
  auto table_info = op->take_table_info();
  if (!table_info) { return nullptr; }

  // When use_sirius_datasource=false (single-GPU only — multi-GPU runs are
  // forced to true by sirius_config::enforce_sirius_datasource_for_multi_gpu()),
  // suppress the per-GPU uring map. The provider then routes local files
  // through cudf::io::datasource::create (kvikio fallback, safe with one GPU),
  // and the operator reads slices without any sirius_ioctx. S3 paths are
  // unaffected — they dispatch through the scan_manager's _io_ctxs vector
  // independently.
  std::unordered_map<int, std::shared_ptr<sirius::io::sirius_ioctx>> const empty_gpu_ioctxs;
  auto const& effective_gpu_ioctxs = _config.use_sirius_datasource ? gpu_ioctxs : empty_gpu_ioctxs;

  // Inject the per-GPU ioctx map into the operator before any provider returns —
  // read_table_from_metadata needs it before its first invocation, and
  // prepare_for_query runs before any execute().
  op->set_gpu_ioctxs(effective_gpu_ioctxs);

  // Cache short-circuit: peeks file_paths(), and on a hit steals table_info into
  // the cached ingestible. On a miss, table_info stays valid for make_gpu_ingestible.
  if (auto cached =
        try_make_cached_ingestible(table_info, op->get_operator_id(), gpu_memory_spaces)) {
    return cached;
  }
  return io::make_gpu_ingestible(std::move(table_info), *this, effective_gpu_ioctxs);
}

std::shared_ptr<io::gpu_ingestible> sirius_scan_manager::try_make_cached_ingestible(
  std::unique_ptr<io::ingestible_table_info>& table_info,
  std::size_t op_id,
  std::unordered_map<int, cucascade::memory::memory_space*> const& gpu_memory_spaces)
{
  if (!table_info) { return nullptr; }

  // Cache is parquet-only today (no pin_duckdb_table path). Cast probe;
  // non-parquet table_info falls through.
  auto const* parquet_info =
    dynamic_cast<op::scan::parquet_ingestible_table_info const*>(table_info.get());
  if (parquet_info == nullptr) { return nullptr; }
  auto const& info = *parquet_info;  // alias so the original cache body reads unchanged

  // If a pinned entry's file paths match this table_info, build the same
  // scan_plan the parquet path would build and serve the scan from cache.
  auto matches_scan_info = [&info](const pinned_entry& entry) {
    if (entry.file_paths.size() != info.resolved_file_paths.size()) { return false; }
    auto sorted_a = entry.file_paths;
    auto sorted_b = info.resolved_file_paths;
    std::sort(sorted_a.begin(), sorted_a.end());
    std::sort(sorted_b.begin(), sorted_b.end());
    return sorted_a == sorted_b;
  };
  try {
    for (auto const& [pinned_name, entry] : _pinned_entries) {
      if (!matches_scan_info(entry)) { continue; }
      // A partial pin (pin_table(..., n_rows=N) capped below the full file
      // content) MUST NOT serve cached reads — the incoming scan_info carries
      // no n_rows budget, so a partial-entry hit would silently mask missing
      // rows. Fall through to the per-format path.
      if (entry.is_partial) {
        SIRIUS_LOG_DEBUG(
          "[sirius_scan_manager::try_make_cached_ingestible] pinned entry '{}' matches op_id={} "
          "but "
          "is partial (row-count budget at pin time); falling through to per-format split_provider",
          pinned_name,
          op_id);
        break;
      }

      // Build the canonical scan_plan once. Everything downstream — cached column
      // layout, filter pushdown indices, post-read assembly — reads from this.
      // Held by shared_ptr<const> so each emitted scan_cached_operator_data can
      // carry it to the GPU scan operator's per-task assembly check without copying.
      auto plan_shared = std::make_shared<op::scan::scan_plan const>(
        op::scan::build_scan_plan(info.column_ids,
                                  info.projection_ids,
                                  info.names,
                                  info.returned_types,
                                  info.scan_output_arity,
                                  info.partition_indices));
      auto const& plan = *plan_shared;

      // Hive partitions on a cached scan would require per-chunk file_path metadata
      // that pinned entries don't carry today. Fall through to the per-format path,
      // which extracts partition values per file at read time.
      if (plan.has_partitions()) {
        SIRIUS_LOG_DEBUG(
          "[sirius_scan_manager::try_make_cached_ingestible] pinned entry '{}' matches op_id={} "
          "but "
          "scan has hive partitions; falling through to per-format split_provider",
          pinned_name,
          op_id);
        break;
      }

      // Filter expression: BoundReferences are in D-space, via plan.batch_position_by_column_id.
      // Same recipe parquet_split_provider uses, so the filter evaluates correctly against
      // the cached batch (which is in D-order by construction above). Built before the
      // tier-specific assembly so both branches share the same filter.
      std::shared_ptr<duckdb::Expression> filter_expression;
      if (info.table_filters && !info.table_filters->filters.empty()) {
        auto duckdb_expression =
          op::convert_table_filters_to_expression(*info.table_filters,
                                                  info.column_ids,
                                                  info.returned_types,
                                                  plan.batch_position_by_column_id,
                                                  plan.partition_primary_indices);
        if (duckdb_expression) {
          filter_expression = std::shared_ptr<duckdb::Expression>(std::move(duckdb_expression));
        }
      }

      if (entry.tier == cucascade::memory::Tier::HOST) {
        // HOST-tier entries store one host_data_representation per chunk in
        // entry.host_chunks; chunk_memory_spaces is intentionally empty (see
        // pinned_entry doc comment + insert_pinned_entry_host). Validate the
        // host_chunks vector instead.
        if (entry.host_chunks.empty()) {
          throw std::runtime_error(
            "[sirius_scan_manager::try_make_cached_ingestible] pinned host entry '" + pinned_name +
            "' has no host_chunks");
        }
        for (std::size_t i = 0; i < entry.host_chunks.size(); ++i) {
          if (!entry.host_chunks[i]) {
            throw std::runtime_error(
              "[sirius_scan_manager::try_make_cached_ingestible] pinned host entry '" +
              pinned_name + "' host_chunks[" + std::to_string(i) + "] is null");
          }
        }
        // The HOST cached path materializes host chunks onto the executing
        // GPU via converter_registry.convert<gpu_table_representation>(...).
        // Without a GPU memory_space map there is no destination — fall
        // through to the per-format path so the query still succeeds.
        if (gpu_memory_spaces.empty()) {
          SIRIUS_LOG_DEBUG(
            "[sirius_scan_manager::try_make_cached_ingestible] pinned host entry '{}' matches "
            "op_id={} but no gpu_memory_spaces map was provided; falling through to per-format "
            "split_provider",
            pinned_name,
            op_id);
          break;
        }

        // Map each D-position to its index inside the captured host chunk. column_names
        // is in capture order, so we look up the requested data column by name. A missing
        // column means the user pinned a subset that doesn't cover this scan — fall back
        // to the per-format path so the query still succeeds.
        std::vector<std::size_t> column_indices;
        column_indices.reserve(plan.data_columns.size());
        for (auto const& dc : plan.data_columns) {
          auto it = std::find(entry.column_names.begin(), entry.column_names.end(), dc.name);
          if (it == entry.column_names.end()) {
            throw std::runtime_error(
              "[sirius_scan_manager::try_make_cached_ingestible] pinned entry '" + pinned_name +
              "' missing column '" + dc.name + "' required by scan op");
          }
          column_indices.push_back(
            static_cast<std::size_t>(std::distance(entry.column_names.begin(), it)));
        }

        SIRIUS_LOG_DEBUG(
          "[sirius_scan_manager::try_make_cached_ingestible] using host cached_split_provider for "
          "op_id={} (pinned='{}' data_cols={} chunks={} needs_assembly={})",
          op_id,
          pinned_name,
          column_indices.size(),
          entry.host_chunks.size(),
          op::scan::needs_output_assembly(plan));

        return std::make_shared<op::scan::pinned_table_gpu_ingestible>(std::move(table_info),
                                                                       entry.host_chunks,
                                                                       std::move(column_indices),
                                                                       *entry.memory_space,
                                                                       gpu_memory_spaces,
                                                                       std::move(filter_expression),
                                                                       std::move(plan_shared));
      }

      // GPU-tier validation: every cached chunk has an owning memory_space.
      // chunk_memory_spaces is parallel to the inner vectors of
      // data_batches_by_column; empty vector means no chunks; null entries
      // violate the chunks-at-index-i invariant.
      if (entry.chunk_memory_spaces.empty()) {
        throw std::runtime_error(
          "[sirius_scan_manager::try_make_cached_ingestible] pinned entry '" + pinned_name +
          "' has no chunk_memory_spaces");
      }
      for (std::size_t i = 0; i < entry.chunk_memory_spaces.size(); ++i) {
        if (entry.chunk_memory_spaces[i] == nullptr) {
          throw std::runtime_error(
            "[sirius_scan_manager::try_make_cached_ingestible] pinned entry '" + pinned_name +
            "' chunk_memory_spaces[" + std::to_string(i) + "] is null");
        }
      }

      // Look up the pinned chunks for each D-position by name. data_columns is in
      // D-order, so columns_per_request[d] is the chunk vector for D-position d.
      std::vector<std::vector<std::shared_ptr<cudf::column>>> columns_per_request;
      columns_per_request.reserve(plan.data_columns.size());
      for (auto const& dc : plan.data_columns) {
        auto it = entry.data_batches_by_column.find(dc.name);
        if (it == entry.data_batches_by_column.end()) {
          throw std::runtime_error(
            "[sirius_scan_manager::try_make_cached_ingestible] pinned entry '" + pinned_name +
            "' missing column '" + dc.name + "' required by scan op");
        }
        columns_per_request.push_back(it->second);
      }

      SIRIUS_LOG_DEBUG(
        "[sirius_scan_manager::try_make_cached_ingestible] using cached_split_provider for "
        "op_id={} "
        "(pinned='{}' data_cols={} needs_assembly={})",
        op_id,
        pinned_name,
        columns_per_request.size(),
        op::scan::needs_output_assembly(plan));

      // Each chunk's data_batch is tagged with its actual memory_space so
      // data-locality scheduling fans cached-scan tasks across GPUs.
      return std::make_shared<op::scan::pinned_table_gpu_ingestible>(std::move(table_info),
                                                                     std::move(columns_per_request),
                                                                     entry.chunk_memory_spaces,
                                                                     std::move(filter_expression),
                                                                     std::move(plan_shared));
    }
  } catch (...) {
    SIRIUS_LOG_TRACE("not all the columns are pinned for this query");
  }
  return nullptr;
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
