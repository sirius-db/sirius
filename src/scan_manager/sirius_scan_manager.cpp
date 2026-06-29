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
#include "io/s3/s3_ioctx.hpp"
#include "io/uring/uring_ioctx.hpp"
#include "log/logging.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "planner/query.hpp"
#include "scan_manager/parquet_metadata.hpp"
#include "scan_manager/round_robin_strategy.hpp"
#include "scan_manager/split_connector.hpp"
#include "scan_manager/split_provider.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/utilities/span.hpp>

#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <utility>

namespace sirius::scan_manager {

namespace {

/// Strip a leading "file://" scheme (case-insensitive) so the path can be
/// resolved by a local-file backend.
std::string normalize_path(std::string const& p)
{
  static constexpr std::string_view kFile = "file://";
  if (p.size() > kFile.size()) {
    bool is_file_uri = true;
    for (std::size_t i = 0; i < kFile.size(); ++i) {
      if (std::tolower(static_cast<unsigned char>(p[i])) != static_cast<unsigned char>(kFile[i])) {
        is_file_uri = false;
        break;
      }
    }
    if (is_file_uri) { return p.substr(kFile.size()); }
  }
  return p;
}

}  // namespace

sirius_scan_manager::sirius_scan_manager(
  scan_manager_config config, cucascade::memory::fixed_size_host_memory_resource* host_fsmr)
  : _config(std::move(config)),
    _thread_pool(_config.thread_pool.num_threads,
                 _config.thread_pool.thread_name_prefix,
                 _config.thread_pool.cpu_affinity_list),
    _dispatcher(
      std::make_unique<exec::scoped_dispatcher>(_thread_pool, _config.thread_pool.num_threads)),
    _factory(_pinned_entries)
{
  // The local uring backend is constructed unconditionally: the DuckDB-native
  // GPU scan needs it for host reads even when use_sirius_datasource=false.
  // That flag instead gates local-path CLAIMING in create_datasource (local
  // parquet falls back to cudf/KvikIO when false).
  auto ioctx = std::make_shared<sirius::io::uring_ioctx>(
    /*host_ring_depth=*/16u,
    /*ring_entries=*/_config.uring_ring_entries,
    /*n_reactors=*/_config.uring_n_reactors,
    /*bounce_slot_size=*/sirius::io::CHUNK_SIZE);
  _io_ctxs.push_back(std::move(ioctx));

  if (_config.s3_config) {
    auto s3_cfg                 = *_config.s3_config;
    s3_cfg.host_memory_resource = host_fsmr;
    std::shared_ptr<sirius::io::sirius_ioctx> s3_backend;
    if (_config.s3_use_async_backend) {
      // Async backend (default): the libcurl-multi reactor owns its own worker
      // thread, so no s3_thread_pool is created. Retry knobs are forwarded so
      // the async backend is config-equivalent to the blocking one.
      s3_backend = std::make_shared<sirius::io::s3::s3_ioctx>(std::move(s3_cfg.creds),
                                                              s3_cfg.request_timeout_s,
                                                              s3_cfg.ca_bundle_path,
                                                              s3_cfg.tls_verify,
                                                              s3_cfg.max_connections,
                                                              host_fsmr,
                                                              s3_cfg.max_retry_attempts,
                                                              s3_cfg.retry_backoff_base,
                                                              s3_cfg.retry_jitter,
                                                              s3_cfg.honor_retry_after);
    } else {
      // Blocking backend (fallback): fan async work out over a dedicated pool.
      if (s3_cfg.async_thread_pool == nullptr) {
        _s3_thread_pool = std::make_unique<sirius::exec::static_thread_pool>(
          _config.s3_thread_pool.num_threads,
          _config.s3_thread_pool.thread_name_prefix,
          _config.s3_thread_pool.cpu_affinity_list);
        s3_cfg.async_thread_pool = _s3_thread_pool.get();
      }
      s3_backend = std::make_shared<sirius::io::s3::s3_blocking_ioctx>(std::move(s3_cfg));
    }
    _io_ctxs.push_back(std::move(s3_backend));
  }

  if (_config.enable_prefetch_cache && host_fsmr != nullptr) {
    auto const slab_bytes = host_fsmr->get_block_size() *
                            static_cast<std::size_t>(sirius::io::buffer_pool::CHUNKS_PER_SLAB);
    auto const max_slabs =
      static_cast<uint32_t>((_config.prefetch_buffer_pool_bytes + slab_bytes - 1) / slab_bytes);
    _prefetch_buffer_pool = std::make_unique<sirius::io::buffer_pool>(*host_fsmr, max_slabs);
    for (auto& ctx : _io_ctxs) {
      if (ctx) {
        ctx->initialize_cache(*_prefetch_buffer_pool, _config.prefetch_inflight_budget_chunks);
      }
    }
  }

  SIRIUS_LOG_DEBUG("[sirius_scan_manager] constructed with {} IO backend(s)", _io_ctxs.size());
}

sirius_scan_manager::~sirius_scan_manager()
{
  stop();
  _io_ctxs.clear();
  if (_s3_thread_pool) { _s3_thread_pool->stop(); }
  _s3_thread_pool.reset();
  _prefetch_buffer_pool.reset();
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

}  // namespace

parquet_bind_result sirius_scan_manager::describe_parquet(std::string const& uri)
{
  auto datasource = create_datasource(uri);
  if (!datasource) {
    throw std::runtime_error("[sirius_scan_manager::describe_parquet] no backend supports URI: " +
                             uri);
  }

  // Footer-only fetch + Thrift parse — the same path parquet_split_provider's
  // run_batch takes on a metadata-cache miss, so bind and scan agree on how
  // the footer is read.
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
  if (auto* cache = datasource->io_ctx()->cache(); cache != nullptr) {
    auto metadata = std::make_shared<parquet_metadata>(
      std::make_shared<cudf::io::parquet::FileMetaData const>(std::move(file_metadata)),
      footer_byte_len);
    cache->insert(*datasource->io_object(), std::move(metadata), /*ranges=*/{});
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
  std::unordered_map<int, cucascade::memory::memory_space*> const& gpu_memory_spaces)
{
  reset();

  for (auto const& ctx : _io_ctxs) {
    if (ctx && ctx->cache()) { ctx->cache()->refresh_cache(); }
  }

  SIRIUS_LOG_DEBUG("[sirius_scan_manager::prepare_for_query] pipelines={} gpu_memory_spaces={}",
                   query.get_pipelines().size(),
                   gpu_memory_spaces.size());

  // Device placement for fresh-read scan splits. Snapshot the query's GPU set
  // in a stable (sorted) order and build one round-robin strategy shared across
  // every provider, so the walk spreads splits evenly over all GPUs across the
  // whole scan stage instead of restarting per scan operator. A provider stamps
  // the chosen device onto its splits' operating data; the task creator reads
  // it back when building the pipeline task.
  std::vector<int> device_ids;
  device_ids.reserve(gpu_memory_spaces.size());
  for (auto const& [device_id, space] : gpu_memory_spaces) {
    device_ids.push_back(device_id);
  }
  std::sort(device_ids.begin(), device_ids.end());
  auto round_robin = std::make_shared<round_robin_strategy>(std::move(device_ids));

  for (auto const& pipeline : query.get_pipelines()) {
    if (!pipeline) { continue; }
    auto source = pipeline->get_source();
    if (!source) { continue; }
    if (source->type != ::sirius::op::SiriusPhysicalOperatorType::GPU_SCAN) { continue; }

    auto* op = &source->Cast<op::scan::sirius_gpu_scan_operator>();
    if (_providers_by_op.find(op) != _providers_by_op.end()) { continue; }

    auto table_info = op->take_table_info();
    if (!table_info) { continue; }

    auto ingestible =
      _factory.produce(std::move(table_info), *this, gpu_memory_spaces, op->get_operator_id());
    if (!ingestible) { continue; }
    op->install_ingestible(std::move(ingestible));

    // Operator is now the sole shared_ptr owner; provider borrows the
    // ingestible by reference. enable_shared_from_this lets any consumer
    // promote to shared_ptr on demand.
    auto provider = std::make_unique<split_provider>(op->get_ingestible());
    // Spread fresh-read splits across GPUs for every scan source — parquet and
    // duckdb-native alike both emit non-resident scan_operator_input splits.
    // Installing the strategy on every provider is safe: resident pinned-cache
    // splits are left untouched by split_provider::apply_balancing's
    // is_resident() guard, so their data-locality placement is preserved.
    provider->set_balancing_strategy(round_robin, pipeline->get_pipeline_id());
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
  // Since the scan-manager cleanup (#913) the scan_manager OWNS the IO
  // backends, the S3 thread pool and the prefetch buffer pool; they are torn
  // down in the destructor (_io_ctxs.clear() -> S3 pool stop -> buffer pool
  // reset). stop() only halts the scan-orchestration pool so it stays safe to
  // call while backends may still serve in-flight reads.
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

void sirius_scan_manager::insert_pinned_entry_host_compressed(
  const std::string& name,
  const std::vector<std::string>& column_names,
  const std::vector<std::string>& file_paths,
  std::vector<std::shared_ptr<sirius::compressed_host_representation>> compressed_chunks,
  cucascade::memory::memory_space& memory_space,
  bool is_partial)
{
  std::size_t new_num_rows = 0;
  for (auto const& chunk : compressed_chunks) {
    if (chunk) { new_num_rows += static_cast<std::size_t>(chunk->num_rows()); }
  }

  pinned_entry entry;
  entry.column_names           = column_names;
  entry.file_paths             = file_paths;
  entry.tier                   = cucascade::memory::Tier::HOST;
  entry.memory_space           = &memory_space;
  entry.num_rows               = new_num_rows;
  entry.compressed_host_chunks = std::move(compressed_chunks);
  entry.is_partial             = is_partial;

  SIRIUS_LOG_DEBUG(
    "[sirius_scan_manager::insert_pinned_entry_host_compressed] '{}' chunks={} rows={}",
    name,
    entry.compressed_host_chunks.size(),
    new_num_rows);

  _pinned_entries[name] = std::move(entry);
}

std::shared_ptr<sirius::io::sirius_datasource> sirius_scan_manager::create_datasource(
  std::string_view path) const
{
  auto file_path = normalize_path(std::string(path));
  // use_sirius_datasource=false keeps LOCAL paths on cudf's bundled datasource
  // (KvikIO fallback): they are deliberately left unclaimed so slices carry a
  // null datasource and reads fall back to cudf::io::datasource::create.
  // Object-store paths (s3:// etc.) always resolve through their backend —
  // this flag does not disable S3.
  if (!_config.use_sirius_datasource && file_path.find("://") == std::string::npos) {
    return nullptr;
  }
  for (auto const& ctx : _io_ctxs) {
    if (ctx && ctx->supports(file_path)) {
      auto io_object = ctx->create_io_object(file_path.data());
      return ctx->make_datasource(io_object);
    }
  }
  return nullptr;
}

void sirius_scan_manager::remove_pinned_entry(const std::string& name)
{
  _pinned_entries.erase(name);
}

}  // namespace sirius::scan_manager
