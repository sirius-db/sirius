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

#include "cucascade/data/gpu_data_representation.hpp"
#include "data/data_batch_utils.hpp"
#include "exec/thread_pool.hpp"
#include "io/cache/prefetching_cache.hpp"
#include "io/kvikio/kvikio_context.hpp"
#include "io/parquet_helpers.hpp"
#include "io/sirius_datasource.hpp"
#include "io/uring/uring_ioctx.hpp"
#include "log/logging.hpp"
#include "memory/topology_index.hpp"
#include "op/scan/parquet_metadata.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "planner/query.hpp"
#include "scan_manager/round_robin_strategy.hpp"
#include "scan_manager/split_connector.hpp"
#include "scan_manager/split_provider.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <utility>

namespace sirius::scan_manager {

namespace {

struct cached_databatch_provider : public databatch_provider {
  explicit cached_databatch_provider(pinned_entry const& entry, std::span<size_t> selected_columns)
    : _entry(entry)
  {
    auto entry_column_names = _entry.table_info->column_names();
    std::ranges::for_each(selected_columns, [this, entry_column_names](size_t idx) {
      _column_names.emplace_back(entry_column_names[idx]);
      _column_indices.push_back(idx);
    });

    if (_entry.tier == cucascade::memory::Tier::GPU) {
      if (_entry.data_batches_by_column.empty()) {
        _n_chunks = 0;
      } else {
        _n_chunks = _entry.data_batches_by_column.begin()->second.size();
      }
    } else if (_entry.tier == cucascade::memory::Tier::HOST) {
      _n_chunks = _entry.host_chunks.size();
    }
  }

  std::shared_ptr<cucascade::data_batch> get_next_batch() override
  {
    auto index = _index.fetch_add(1);
    if (index >= _n_chunks) { return nullptr; }
    if (_entry.tier == cucascade::memory::Tier::GPU) {
      return get_device_databatch(index);
    } else if (_entry.tier == cucascade::memory::Tier::HOST) {
      return get_host_databatch(index);
    }
    return nullptr;
  }

 private:
  std::shared_ptr<cucascade::data_batch> get_host_databatch(std::size_t index)
  {
    if (index >= _entry.host_chunks.size()) { return nullptr; }
    const auto& chunk = _entry.host_chunks.at(index);
    if (!chunk) { return nullptr; }
    auto data_rep = chunk->slice(_column_indices);
    return std::make_shared<cucascade::data_batch>(get_next_batch_id(), std::move(data_rep));
  }

  std::shared_ptr<cucascade::data_batch> get_device_databatch(std::size_t index)
  {
    std::vector<std::shared_ptr<cudf::column>> columns;
    std::vector<cudf::column_view> column_views;
    std::size_t alloc_size = 0;
    for (const auto& col_idx : _column_names) {
      const auto& col_chunks = _entry.data_batches_by_column.at(col_idx);
      if (index >= col_chunks.size()) { return nullptr; }
      columns.push_back(col_chunks.at(index));
      column_views.emplace_back(columns.back()->view());
      alloc_size += columns.back()->alloc_size();
    }
    cudf::table_view view(column_views);
    auto* chunk_space = !_entry.chunk_memory_spaces.empty() ? _entry.chunk_memory_spaces.at(index)
                                                            : _entry.memory_space;
    auto gpu_repr     = std::make_unique<::cucascade::gpu_table_representation>(
      view, std::move(columns), alloc_size, *chunk_space, rmm::cuda_stream_view{});
    return std::make_shared<::cucascade::data_batch>(::sirius::get_next_batch_id(),
                                                     std::move(gpu_repr));
  }

  std::size_t _n_chunks;
  std::vector<std::string> _column_names;
  std::vector<size_t> _column_indices;
  const pinned_entry& _entry;
  std::atomic<std::size_t> _index{0};
};

std::unique_ptr<databatch_provider> make_provider_for_pinned_entry(
  pinned_entry const& entry, std::span<size_t> selected_columns)
{
  return std::make_unique<cached_databatch_provider>(entry, selected_columns);
}

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
  const scan_manager_config& config,
  cucascade::memory::memory_reservation_manager& reservation_manager,
  std::shared_ptr<const sirius::memory::topology_index> topology_index)
  : _config(config),
    _topology_index(std::move(topology_index)),
    _thread_pool(_config.thread_pool.num_threads + 1,
                 _config.thread_pool.thread_name_prefix,
                 _config.thread_pool.cpu_affinity_list),
    _dispatcher(
      std::make_unique<exec::scoped_dispatcher>(_thread_pool, _config.thread_pool.num_threads + 1)),
    _ioctx_registry(config)
{
  if (!_topology_index) {
    throw std::invalid_argument("[sirius_scan_manager] topology_index must be non-null");
  }

  auto host_spaces = reservation_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  std::vector<cucascade::memory::fixed_size_host_memory_resource*> host_mrs;
  std::ranges::transform(host_spaces, std::back_inserter(host_mrs), [](auto* sp) {
    return sp->template get_memory_resource_of<cucascade::memory::Tier::HOST>();
  });

  if (host_mrs.empty()) {
    throw std::runtime_error(
      "[sirius_scan_manager] use_sirius_datasource is true but the reservation "
      "manager has no HOST-tier fixed_size_host_memory_resource");
  }

  // scan_manager always owns an io_ctx: sirius_datasource (uring) on the
  // fast path, kvikio_context as the universal fallback so the rest of the
  // scan path (parquet_split_provider, scan tasks) always has an ioctx to
  // talk to.  kvikio_context wraps cudf::io::datasource so the read path
  // is identical from the caller's point of view.
  if (_config.use_sirius_datasource) {
    _io_ctx = std::make_shared<sirius::io::uring::uring_ioctx>(
      _config.uring_n_reactors, *host_mrs.front(), _config.use_odirect);
    SIRIUS_LOG_DEBUG("[sirius_scan_manager] sirius_datasource enabled (uring_ioctx n_reactors={})",
                     _config.uring_n_reactors);
  } else {
    if (_topology_index->gpu_ids().size() > 1) {
      throw std::runtime_error(
        "[sirius_scan_manager] kvikio_context fallback (use_sirius_datasource=false) "
        "does not support multi-GPU; topology reports " +
        std::to_string(_topology_index->gpu_ids().size()) +
        " GPUs.  Enable use_sirius_datasource for multi-GPU runs.");
    }
    _io_ctx = std::make_shared<sirius::io::kvikio_context>();
    SIRIUS_LOG_DEBUG(
      "[sirius_scan_manager] sirius_datasource disabled — using kvikio_context fallback");
  }

  // Reactors are built parked; start() launches their worker threads and
  // allocates per-reactor staging.  No-op for the kvikio fallback (no reactors).
  _io_ctx->start();

  // Build the prefetching cache on the ioctx.  Budget=0 keeps the
  // cache unarmed (no background threads); we pass that whenever the
  // user has disabled prefetching so the construction is always
  // unconditional and there's no "is the cache present" branch to
  // worry about in callers.
  if (_config.enable_prefetch_cache) {
    // Total slab budget for the cache's pool; the cache builds and owns the
    // pool from the reservation manager's HOST-tier spaces.
    auto const slab_bytes =
      host_mrs.front()->get_block_size() *
      static_cast<std::size_t>(sirius::io::cache::buffer_pool::CHUNKS_PER_SLAB);
    auto const max_slabs =
      static_cast<uint32_t>((_config.prefetch_buffer_pool_bytes + slab_bytes - 1) / slab_bytes);
    _io_ctx->initialize_cache(
      reservation_manager, _config.prefetch_inflight_budget_chunks, max_slabs, _topology_index);
  }

  if (_io_ctx->cache() && _io_ctx->cache()->is_armed()) {
    SIRIUS_LOG_DEBUG("[sirius_scan_manager] prefetch cache armed (inflight_chunks={})",
                     _config.prefetch_inflight_budget_chunks);
  } else {
    SIRIUS_LOG_DEBUG("[sirius_scan_manager] prefetch cache unarmed");
  }
}

sirius_scan_manager::~sirius_scan_manager()
{
  if (_io_ctx && _io_ctx->cache()) {
    SIRIUS_LOG_INFO("[sirius_scan_manager] cache summary: {}", _io_ctx->cache()->summary());
  }
  // Drain the dispatcher (and the worker pool) first so no in-flight
  // metadata-scan / sequencer task can still be reaching into the
  // cache via _io_ctx when we tear it down below.
  stop();
  // Tear down the cache (which owns its buffer_pool).  shutdown_cache drains
  // in-flight IO before the pool is destroyed, so callbacks release their
  // chunks safely.
  if (_io_ctx) { _io_ctx->shutdown_cache(); }
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

  // Reuse a previously parsed footer when present — a prior bind or scan of the
  // same file parks it in the ioctx metadata store, which lives for the ioctx's
  // lifetime. On a miss, fetch + Thrift-parse the footer once and park it so the
  // subsequent scan reuses it. Mirrors parquet_gpu_ingestible::build_file_scan_info,
  // so the footer is parsed exactly once per file per process.
  std::shared_ptr<cudf::io::parquet::FileMetaData const> file_metadata;
  if (auto cached = datasource->metadata()) {
    if (auto pm = std::dynamic_pointer_cast<op::scan::parquet_metadata>(std::move(cached))) {
      file_metadata = pm->file_metadata();
    }
  }
  if (!file_metadata) {
    auto footer_buffer         = cudf::io::parquet::fetch_footer_to_host(*datasource);
    auto const footer_byte_len = footer_buffer->size();
    auto reader_options        = cudf::io::parquet_reader_options::builder().build();
    cudf::io::parquet::experimental::hybrid_scan_reader reader{
      cudf::host_span<std::uint8_t const>(footer_buffer->data(), footer_buffer->size()),
      reader_options};
    file_metadata =
      std::make_shared<cudf::io::parquet::FileMetaData const>(reader.parquet_metadata());
    [[maybe_unused]] auto const stored = datasource->store_metadata(
      std::make_shared<op::scan::parquet_metadata>(file_metadata, footer_byte_len));
  }

  auto schema = sirius::io::parquet_helpers::extract_schema(*file_metadata);

  parquet_bind_result result;
  result.return_types   = std::move(schema.types);
  result.names          = std::move(schema.names);
  result.object_size    = datasource->size();
  result.total_num_rows = static_cast<std::size_t>(file_metadata->num_rows);
  return result;
}

void sirius_scan_manager::prepare_for_query(const sirius::planner::query& query)
{
  reset();

  if (_io_ctx && _io_ctx->cache()) {
    SIRIUS_LOG_INFO("[sirius_scan_manager] cache summary: {}", _io_ctx->cache()->summary());
    _io_ctx->cache()->prepare_for_query(query);
  }

  auto const gpu_ids = _topology_index->gpu_ids();
  auto round_robin =
    std::make_shared<round_robin_strategy>(std::vector<int>(gpu_ids.begin(), gpu_ids.end()));

  // Advance the cache age so the evictor can score this query's inserts
  // against entries left over from prior queries.
  // refresh_cache() removed — the cache no longer has a query-epoch
  // notion.  Per-file aging in the eviction queue handles staleness
  // without scan_manager involvement.

  SIRIUS_LOG_INFO("[sirius_scan_manager::prepare_for_query] scan_operators={}",
                  query.get_scan_operators().size());

  // Build a fresh sequencer for this query.  Slots are added by
  // create_provider_for() whenever it chooses the parquet path; the
  // cached path skips slot allocation since there's no IO to prefetch.
  // The sequencer task piggy-backs on the dispatcher's injected
  // stop_token, so reset()/request_stop on the dispatcher tears it
  // down without a side-channel stop_source.
  _metadata_processor = std::make_unique<load_balancing_scan_batch_coalecer>();

  for (auto const& scan_op : query.get_scan_operators()) {
    if (scan_op->type != ::sirius::op::SiriusPhysicalOperatorType::GPU_SCAN) { continue; }
    auto* op = &scan_op->Cast<op::scan::sirius_gpu_scan_operator>();
    if (_providers_by_op.find(op) != _providers_by_op.end()) { continue; }
    auto provider = std::make_unique<split_provider>(op->get_ingestible(), *_io_ctx);
    _metadata_processor->register_pipeline(op, round_robin);
    try_assign_cached_entries(op);
    _providers_by_op.emplace(op, std::move(provider));
    _scan_op_order.push_back(op);
  }

  if (_scan_op_order.empty()) {
    spdlog::warn("[sirius_scan_manager::prepare_for_query] no GPU scan operators found in query");
    return;
  }

  start_metadata_processing();
}

void sirius_scan_manager::start_metadata_processing()
{
  _metadata_processor->spawn_workers(*_dispatcher);
  for (auto* op : _scan_op_order) {
    auto it = _providers_by_op.find(op);
    if (it == _providers_by_op.end()) { continue; }
    it->second->run(*_dispatcher, _metadata_processor->get_split_provider_bridge(op));
  }
}

std::shared_ptr<sirius::io::sirius_datasource> sirius_scan_manager::create_datasource(
  std::string_view path) const noexcept
{
  auto file_path = normalize_path(std::string(path));
  if (!_io_ctx) { return nullptr; }
  return _io_ctx->open_datasource(file_path);
}

void sirius_scan_manager::reset()
{
  _dispatcher->request_stop();
  _dispatcher->wait_for_all();
  _scan_op_order.clear();
  _providers_by_op.clear();
  _metadata_processor.reset();
  _dispatcher =
    std::make_unique<exec::scoped_dispatcher>(_thread_pool, _config.thread_pool.num_threads + 1);
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
  std::unique_ptr<ingestible_table_info> table_info,
  std::vector<std::unique_ptr<cudf::table>> data_tables,
  std::vector<cucascade::memory::memory_space*> chunk_memory_spaces)
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

  // Column identity for this pin lives on its table_info. The span stays valid
  // after table_info is moved into the entry below (moving the unique_ptr does
  // not relocate the pointed-to ingestible_table_info).
  std::span<std::string const> column_names =
    table_info ? table_info->column_names() : std::span<std::string const>{};

  auto existing_it = _pinned_entries.find(name);
  if (existing_it != _pinned_entries.end()) {
    // Same-row-count merge only applies when the completeness contracts match.
    // Mixing a full pin with a partial pin produces an entry whose columns came
    // from different row coverage — drop and rebuild instead.
    if (existing_it->second.num_rows == new_num_rows) {
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
          entry.data_batches_by_column[std::string{column_names[i]}].emplace_back(
            std::move(cols[i]));
        }
      }
      // The union of pinned column names is reflected by entry.table_info; the
      // merged columns are keyed into data_batches_by_column above.
      return;
    }
    // Row count or completeness contract differs → drop the stale entry and rebuild below.
    _pinned_entries.erase(existing_it);
  }

  pinned_entry entry;
  entry.table_info          = std::move(table_info);
  entry.chunk_memory_spaces = std::move(chunk_memory_spaces);
  entry.tier                = cucascade::memory::Tier::GPU;
  entry.num_rows            = new_num_rows;

  for (auto& table : data_tables) {
    if (!table) { continue; }
    auto cols = table->release();
    if (cols.size() != column_names.size()) {
      throw std::runtime_error("[sirius_scan_manager::insert_pinned_entry] table column count " +
                               std::to_string(cols.size()) + " does not match column_names size " +
                               std::to_string(column_names.size()));
    }
    for (std::size_t i = 0; i < cols.size(); ++i) {
      entry.data_batches_by_column[std::string{column_names[i]}].emplace_back(std::move(cols[i]));
    }
  }

  _pinned_entries[name] = std::move(entry);
}

void sirius_scan_manager::insert_pinned_entry_host(
  const std::string& name,
  std::unique_ptr<ingestible_table_info> table_info,
  std::vector<std::shared_ptr<cucascade::host_data_representation>> host_chunks,
  cucascade::memory::memory_space& memory_space)
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
  entry.table_info   = std::move(table_info);
  entry.tier         = cucascade::memory::Tier::HOST;
  entry.memory_space = &memory_space;
  entry.num_rows     = new_num_rows;
  entry.host_chunks  = std::move(host_chunks);

  _pinned_entries[name] = std::move(entry);
}

void sirius_scan_manager::remove_pinned_entry(const std::string& name)
{
  _pinned_entries.erase(name);
}

void sirius_scan_manager::try_assign_cached_entries(op::scan::sirius_gpu_scan_operator* op)
{
  const auto& table_info = op->get_ingestible().table_info();

  try {
    for (auto const& [pinned_name, entry] : _pinned_entries) {
      if (auto cols = entry.table_info->can_serve_with_columns(table_info); !cols.empty()) {
        auto provider = make_provider_for_pinned_entry(entry, cols);
        _metadata_processor->use_cached_entries_for_pipeline(op, std::move(provider));
        spdlog::info("[sirius_scan_manager] assigned pinned entry '{}' to operator '{}'",
                     pinned_name,
                     op->get_operator_id());
      }
    }
  } catch (...) {
    spdlog::error(
      "[sirius_scan_manager] error while trying to assign cached entries to "
      "operator '{}'",
      op->get_operator_id());
  }
}

}  // namespace sirius::scan_manager
