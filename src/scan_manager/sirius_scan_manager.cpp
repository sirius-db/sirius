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

#include "data/data_batch_utils.hpp"
#include "exec/thread_pool.hpp"
#include "io/cache/prefetching_cache.hpp"
#include "io/io_context.hpp"
#include "io/parquet_helpers.hpp"
#include "io/rest/rest_ioctx.hpp"
#include "io/sirius_datasource.hpp"
#include "log/logging.hpp"
#include "memory/topology_index.hpp"
#include "op/scan/duckdb_native_gpu_ingestible.hpp"
#include "op/scan/gpu_ingestible.hpp"
#include "op/scan/parquet_gpu_ingestible.hpp"
#include "op/scan/parquet_metadata.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "op/scan/sirius_gpu_scan_operator_data.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "planner/query.hpp"
#include "scan_manager/round_robin_strategy.hpp"

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

#include <rmm/cuda_device.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace sirius::scan_manager {

namespace {

struct cached_databatch_provider : public databatch_provider {
  cached_databatch_provider(pinned_entry const& entry,
                            std::span<std::size_t const> selected_columns,
                            cached_scan_plan plan,
                            const telemetry::batch_telemetry_info& telemetry_info,
                            std::shared_ptr<mvcc_chunk_mask_set const> mvcc_masks)
    : _plan(std::move(plan)),
      _entry(entry),
      _telemetry_info(telemetry_info),
      _mvcc_masks(std::move(mvcc_masks))
  {
    auto const& entry_column_names = _entry.cache_info.column_names();
    std::ranges::for_each(selected_columns, [this, &entry_column_names](size_t idx) {
      _column_names.emplace_back(entry_column_names[idx]);
      _column_indices.push_back(idx);
    });
  }

  databatch_provider::batch get_next_batch() override
  {
    // The atomic cursor walks survivor positions, each mapping to a chunk index.
    auto const cursor = _index.fetch_add(1);
    if (cursor >= _plan.survivor_chunk_indices.size()) { return {}; }
    auto const index = _plan.survivor_chunk_indices[cursor];
    std::shared_ptr<cucascade::data_batch> data;
    if (_entry.tier == cucascade::memory::Tier::GPU) {
      data = get_device_databatch(index);
    } else if (_entry.tier == cucascade::memory::Tier::HOST) {
      data = get_host_databatch(index);
    }
    if (!data) { return {}; }
    auto mask = (_mvcc_masks && index < _mvcc_masks->size()) ? (*_mvcc_masks)[index] : nullptr;
    return {std::move(data), std::move(mask)};
  }

 private:
  std::shared_ptr<cucascade::data_batch> get_host_databatch(std::size_t index)
  {
    if (index >= _entry.host_chunks.size()) { return nullptr; }
    const auto& chunk = _entry.host_chunks.at(index);
    if (!chunk) { return nullptr; }
    auto data_rep       = chunk->slice(_column_indices);
    const auto batch_id = get_next_batch_id();
    return cucascade::data_batch::make(
      batch_id,
      std::move(data_rep),
      telemetry::quent_data_batch_probe::create(_telemetry_info, batch_id));
  }

  std::shared_ptr<cucascade::data_batch> get_device_databatch(std::size_t index)
  {
    if (index >= _entry.chunk_memory_spaces.size()) { return nullptr; }
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
    const auto batch_id = ::sirius::get_next_batch_id();
    return ::cucascade::data_batch::make(
      batch_id,
      std::move(gpu_repr),
      telemetry::quent_data_batch_probe::create(_telemetry_info, batch_id));
  }

  cached_scan_plan _plan;
  std::vector<std::string> _column_names;
  std::vector<size_t> _column_indices;
  const pinned_entry& _entry;
  telemetry::batch_telemetry_info _telemetry_info;
  std::shared_ptr<mvcc_chunk_mask_set const> _mvcc_masks;
  std::atomic<std::size_t> _index{0};
};

/// Filter view extracted from the scan's ingestible info: the pushed-down TableFilterSet plus the
/// scan's column_ids its keys index into.
struct scan_filter_view {
  duckdb::TableFilterSet const* table_filters{nullptr};
  duckdb::vector<duckdb::ColumnIndex> const* column_ids{nullptr};
};

scan_filter_view extract_scan_filters(op::scan::ingestible_table_info const& info)
{
  if (auto const* p = dynamic_cast<op::scan::parquet_ingestible_table_info const*>(&info)) {
    return {p->table_filters.get(), &p->column_ids};
  } else if (auto const* p =
               dynamic_cast<op::scan::duckdb_native_ingestible_table_info const*>(&info)) {
    return {p->table_filters.get(), &p->column_ids};
  }
  return {};
}

std::unique_ptr<databatch_provider> make_provider_for_pinned_entry(
  pinned_entry const& entry,
  std::span<std::size_t const> selected_columns,
  cached_scan_plan plan,
  const telemetry::batch_telemetry_info& telemetry_info,
  std::shared_ptr<mvcc_chunk_mask_set const> mvcc_masks)
{
  return std::make_unique<cached_databatch_provider>(
    entry, selected_columns, std::move(plan), telemetry_info, std::move(mvcc_masks));
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
    _reservation_manager(reservation_manager),
    _topology_index(std::move(topology_index)),
    _thread_pool(_config.thread_pool.num_threads + 1,
                 _config.thread_pool.thread_name_prefix,
                 _config.thread_pool.cpu_affinity_list),
    _dispatcher(
      std::make_unique<exec::scoped_dispatcher>(_thread_pool, _thread_pool.num_threads())),
    _ioctx_registry(config, reservation_manager)
{
  if (!_topology_index) {
    throw std::invalid_argument("[sirius_scan_manager] topology_index must be non-null");
  }

  // scan_manager always owns an io_ctx: sirius_datasource (uring) on the
  // fast path, kvikio_context as the universal fallback so the rest of the
  // scan path (parquet_split_provider, scan tasks) always has an ioctx to
  // talk to.  kvikio_context wraps cudf::io::datasource so the read path
  // is identical from the caller's point of view.  Both are built by the
  // ioctx registry, which sources the reactor staging resource from the
  // reservation manager it was constructed with.
  if (_config.use_sirius_datasource) {
    _io_ctx = _ioctx_registry.make_ioctx(sirius::io::io_context_type::uring);
    if (!_io_ctx) {
      throw std::runtime_error("[sirius_scan_manager] failed to create uring io_context");
    }
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
    _io_ctx = _ioctx_registry.make_ioctx(sirius::io::io_context_type::kvikio);
    if (!_io_ctx) {
      throw std::runtime_error("[sirius_scan_manager] failed to create kvikio io_context");
    }
    SIRIUS_LOG_DEBUG(
      "[sirius_scan_manager] sirius_datasource disabled — using kvikio_context fallback");
  }

  // Build the prefetching cache on the ioctx.  Budget=0 keeps the
  // cache unarmed (no background threads); we pass that whenever the
  // user has disabled prefetching so the construction is always
  // unconditional and there's no "is the cache present" branch to
  // worry about in callers.
  if (_config.enable_prefetch_cache && _io_ctx->can_use_prefetching_cache()) {
    _io_ctx->initialize_cache(reservation_manager, _config.cache, _topology_index);
  }

  // Reactors are built parked; start() launches their worker threads and
  // allocates per-reactor staging.  No-op for the kvikio fallback (no reactors).
  _io_ctx->start();
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
  // Same drain for any path-routed ioctxs; their reactors stop when the
  // shared_ptrs in _routed_io_ctxs are released (member destruction below).
  std::lock_guard lk{_routed_io_ctxs_mtx};
  for (auto& [type, io_ctx] : _routed_io_ctxs) {
    if (io_ctx) { io_ctx->shutdown_cache(); }
  }
}

parquet_bind_result sirius_scan_manager::describe_parquet(std::string const& uri)
{
  // Footer-probe only when we will actually read + parse the footer.  On a warm
  // re-bind the metadata_store already holds the parsed footer, so a suffix GET
  // would download footer bytes we won't reuse — a plain HEAD resolves the size.
  auto const cache_key     = normalize_path(uri);
  auto const io_ctx        = ioctx_for_path(uri);
  bool const footer_cached = io_ctx && io_ctx->metadata_store().get_metadata(cache_key) != nullptr;
  auto const hint =
    footer_cached ? sirius::io::open_hint::generic : sirius::io::open_hint::parquet_footer_probe;

  auto datasource = create_datasource(uri, hint);
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

void sirius_scan_manager::prepare_for_query(const sirius::planner::query& query,
                                            bool enable_pinned_zone_map_pruning)
{
  _pruning_enabled = enable_pinned_zone_map_pruning;
  reset();

  if (_io_ctx && _io_ctx->cache()) {
    SIRIUS_LOG_INFO("[sirius_scan_manager] cache summary: {}", _io_ctx->cache()->summary());
    _io_ctx->cache()->prepare_for_query(query);
  }

  // Routed ioctxs (e.g. the restful context serving s3://) are built lazily and
  // reused across queries; advance their caches to this query too, or a routed
  // cache's epoch freezes at build time and a later query serves the prior
  // query's cached chunks as current.
  {
    std::lock_guard lk{_routed_io_ctxs_mtx};
    for (auto& [type, io_ctx] : _routed_io_ctxs) {
      if (io_ctx && io_ctx->cache()) { io_ctx->cache()->prepare_for_query(query); }
    }
  }

  auto const gpu_ids = _topology_index->gpu_ids();
  auto round_robin =
    std::make_shared<round_robin_strategy>(std::vector<int>(gpu_ids.begin(), gpu_ids.end()));

  _metadata_processor = std::make_unique<load_balancing_scan_batch_coalescer>();

  for (auto const& scan_op : query.get_scan_operators()) {
    if (scan_op->type != ::sirius::op::SiriusPhysicalOperatorType::GPU_SCAN) { continue; }
    auto* op = &scan_op->Cast<op::scan::sirius_gpu_scan_operator>();
    if (_providers_by_op.find(op) != _providers_by_op.end()) { continue; }
    _metadata_processor->register_pipeline(op, round_robin);
    // On a pinned-cache hit the coalescer serves this operator from the cached
    // batch_provider (process_cached_entries); skip the disk-reading
    // split_provider entirely so no read is issued for the cached scan.
    if (try_assign_cached_entries(op)) {
      _scan_op_order.push_back(op);
      continue;
    }
    auto provider = std::make_unique<split_provider>(
      op->get_ingestible(),
      [this](std::string_view file_path) -> std::shared_ptr<io::sirius_ioctx> {
        auto io_ctx = ioctx_for_path(file_path);
        if (!io_ctx) {
          throw std::runtime_error("scan_manager: no backend supports path: " +
                                   std::string(file_path));
        }
        return io_ctx;
      });
    _providers_by_op.emplace(op, std::move(provider));
    _scan_op_order.push_back(op);
  }

  if (_scan_op_order.empty()) {
    SIRIUS_LOG_WARN(
      "[sirius_scan_manager::prepare_for_query] no GPU scan operators found in query");
    return;
  }

  // #819: compute this query's duckdb MVCC keep-masks BEFORE serving starts —
  // block-in-prepare, so masks are finished plain buffers when the sequencer
  // runs (it walks all ops' slots serially; a wait there would be head-of-line
  // blocking across every scan op). The dispatcher is fresh and otherwise idle
  // here. Errors are loud: past the plan-time gate there is no CPU fallback,
  // and the alternative to failing is serving rows a concurrent DELETE removed.
  if (!_pending_mvcc_mask_jobs.empty()) {
    run_mvcc_mask_jobs(
      _pending_mvcc_mask_jobs, *_dispatcher, _reservation_manager, *_topology_index);
    _pending_mvcc_mask_jobs.clear();
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
  std::string_view path, sirius::io::open_hint hint)
{
  auto file_path = normalize_path(std::string(path));
  auto io_ctx    = ioctx_for_path(file_path);
  if (!io_ctx) { return nullptr; }  // no backend supports the path
  // Real I/O / HEAD / auth / missing-object errors propagate as exceptions;
  // only "no backend" is reported as nullptr (callers map it to that message).
  return io_ctx->open_datasource(file_path, hint);
}

void sirius_scan_manager::list_objects_paged(
  std::string const& s3_prefix_uri,
  std::size_t page_size,
  std::function<bool(sirius::io::s3::list_objects_v2_page const&)> const& sink,
  std::optional<std::size_t> max_scanned)
{
  // Hand-split rather than uri_parser::parse — a LIST prefix URI legitimately
  // has an EMPTY key part (bucket-root glob: "s3://bucket/"), which parse()
  // rejects as an object URI.
  constexpr std::string_view k_scheme = "s3://";
  if (s3_prefix_uri.size() <= k_scheme.size() ||
      s3_prefix_uri.compare(0, k_scheme.size(), k_scheme) != 0) {
    throw std::runtime_error("sirius_scan_manager::list_objects_paged: malformed prefix URI '" +
                             s3_prefix_uri + "'");
  }
  auto const rest_uri     = std::string_view{s3_prefix_uri}.substr(k_scheme.size());
  auto const bucket_slash = rest_uri.find('/');
  auto const bucket       = rest_uri.substr(0, bucket_slash);
  auto const prefix =
    bucket_slash == std::string_view::npos ? std::string_view{} : rest_uri.substr(bucket_slash + 1);
  if (bucket.empty()) {
    throw std::runtime_error("sirius_scan_manager::list_objects_paged: malformed prefix URI '" +
                             s3_prefix_uri + "'");
  }
  // Routing probe only (scheme dispatch, never touches the network): the
  // backend checkers parse() their input and reject an empty object key, so a
  // bucket-root prefix routes via a placeholder key.
  auto const route_probe =
    "s3://" + std::string(bucket) + "/" + (prefix.empty() ? "_" : std::string(prefix));
  auto io_ctx = ioctx_for_path(route_probe);
  auto* rest  = dynamic_cast<sirius::io::rest::rest_ioctx*>(io_ctx.get());
  if (rest == nullptr) {
    throw std::runtime_error("sirius_scan_manager::list_objects_paged: '" + s3_prefix_uri +
                             "' does not route to an object-store backend that supports LIST");
  }
  rest->list_objects_paged(bucket, prefix, page_size, sink, max_scanned);
}

std::size_t sirius_scan_manager::s3_list_max_matches(std::string const& s3_uri)
{
  // Route by scheme only (no LIST call) to read the backend's configured cap.
  // Same bucket-root-safe placeholder-key probe as list_objects_paged.
  constexpr std::string_view k_scheme = "s3://";
  if (s3_uri.size() <= k_scheme.size() || s3_uri.compare(0, k_scheme.size(), k_scheme) != 0) {
    throw std::runtime_error("sirius_scan_manager::s3_list_max_matches: malformed URI '" + s3_uri +
                             "'");
  }
  auto const rest_uri     = std::string_view{s3_uri}.substr(k_scheme.size());
  auto const bucket_slash = rest_uri.find('/');
  auto const bucket       = rest_uri.substr(0, bucket_slash);
  auto const prefix =
    bucket_slash == std::string_view::npos ? std::string_view{} : rest_uri.substr(bucket_slash + 1);
  auto const route_probe =
    "s3://" + std::string(bucket) + "/" + (prefix.empty() ? "_" : std::string(prefix));
  auto io_ctx = ioctx_for_path(route_probe);
  auto* rest  = dynamic_cast<sirius::io::rest::rest_ioctx*>(io_ctx.get());
  if (rest == nullptr) {
    throw std::runtime_error("sirius_scan_manager::s3_list_max_matches: '" + s3_uri +
                             "' does not route to an object-store backend that supports LIST");
  }
  return rest->list_max_matches();
}

std::shared_ptr<sirius::io::sirius_ioctx> sirius_scan_manager::ioctx_for_path(std::string_view path)
{
  // Normalize here so every caller (incl. the scan resolver, which forwards raw
  // ingestible paths) routes `file://` the same way create_datasource does.
  auto file_path = normalize_path(std::string(path));
  auto type      = _ioctx_registry.lookup_path(file_path);
  if (!type) { return nullptr; }
  // The local default `_io_ctx` already serves uring/kvikio; only an off-default
  // backend (e.g. s3:// -> restful) needs a separate, lazily-built context.
  if (_io_ctx && _io_ctx->type() == *type) { return _io_ctx; }

  {
    std::lock_guard lk{_routed_io_ctxs_mtx};
    if (auto it = _routed_io_ctxs.find(*type); it != _routed_io_ctxs.end()) { return it->second; }
  }
  // Build outside the map mutex: make_ioctx/start spawn reactor threads and
  // initialize_cache allocates, so holding _routed_io_ctxs_mtx across them would
  // park every concurrent lookup behind one long critical section. The build
  // mutex serializes builders instead, so two first-touches of the same type
  // never construct twice (a losing ioctx would need drain/stop teardown).
  std::lock_guard build_lk{_routed_io_ctxs_build_mtx};
  {
    std::lock_guard lk{_routed_io_ctxs_mtx};
    if (auto it = _routed_io_ctxs.find(*type); it != _routed_io_ctxs.end()) { return it->second; }
  }
  auto io_ctx = _ioctx_registry.make_ioctx(*type);
  if (!io_ctx) { return nullptr; }
  io_ctx->start();
  if (_config.enable_prefetch_cache && io_ctx->can_use_prefetching_cache()) {
    io_ctx->initialize_cache(_reservation_manager, _config.cache, _topology_index);
  }
  std::lock_guard lk{_routed_io_ctxs_mtx};
  auto [it, inserted] = _routed_io_ctxs.emplace(*type, std::move(io_ctx));
  return it->second;
}

void sirius_scan_manager::reset()
{
  _dispatcher->request_stop();
  _dispatcher->wait_for_all();
  _scan_op_order.clear();
  _providers_by_op.clear();
  _pending_mvcc_mask_jobs.clear();
  _metadata_processor.reset();
  _dispatcher = std::make_unique<exec::scoped_dispatcher>(_thread_pool, _thread_pool.num_threads());
}

void sirius_scan_manager::start() {}

void sirius_scan_manager::stop()
{
  reset();
  _thread_pool.stop();
}

namespace {

// Gather positions into @p cached_ids for each requested primary (storage) index, in the
// given order. Empty when any requested column is absent — i.e. the cache is not a superset.
std::vector<std::size_t> gather_by_primary_index(
  duckdb::vector<duckdb::ColumnIndex> const& cached_ids,
  std::vector<std::size_t> const& requested_primary_indices)
{
  std::unordered_map<duckdb::idx_t, std::size_t> pos;
  pos.reserve(cached_ids.size());
  for (std::size_t i = 0; i < cached_ids.size(); ++i) {
    pos.emplace(cached_ids[i].GetPrimaryIndex(), i);
  }
  std::vector<std::size_t> projection;
  projection.reserve(requested_primary_indices.size());
  for (auto const primary_idx : requested_primary_indices) {
    auto it = pos.find(primary_idx);
    if (it == pos.end()) { return {}; }  // cache lacks a requested column
    projection.push_back(it->second);
  }
  return projection;
}

// Gather projection that lets a cache holding @p cached_ids (by primary/storage
// index) serve a scan requesting @p requested_ids: for each requested column,
// its position within @p cached_ids, in the requested order. Empty when any
// requested column is absent — i.e. the cache is not a column superset.
std::vector<std::size_t> column_superset_projection(
  duckdb::vector<duckdb::ColumnIndex> const& cached_ids,
  duckdb::vector<duckdb::ColumnIndex> const& requested_ids)
{
  std::vector<std::size_t> requested_primary_indices;
  requested_primary_indices.reserve(requested_ids.size());
  for (auto const& c : requested_ids) {
    requested_primary_indices.push_back(c.GetPrimaryIndex());
  }
  return gather_by_primary_index(cached_ids, requested_primary_indices);
}

// column_ids-aligned names: for each column_ids[i], the full-schema name at its
// primary (storage) index — the keys data_batches_by_column / the gather use.
std::vector<std::string> aligned_column_names(duckdb::vector<std::string> const& full_names,
                                              duckdb::vector<duckdb::ColumnIndex> const& column_ids)
{
  std::vector<std::string> out;
  out.reserve(column_ids.size());
  for (auto const& c : column_ids) {
    auto const p = static_cast<std::size_t>(c.GetPrimaryIndex());
    out.push_back(p < full_names.size() ? full_names[p] : std::string{});
  }
  return out;
}

}  // namespace

cache_entry_info cache_entry_info::from(const op::scan::ingestible_table_info& info)
{
  cache_entry_info ci;
  if (auto const* p = dynamic_cast<op::scan::parquet_ingestible_table_info const*>(&info)) {
    ci.resolved_file_paths = p->resolved_file_paths;
    ci.column_ids          = p->column_ids;
    ci.names               = aligned_column_names(p->names, p->column_ids);
  } else if (auto const* d =
               dynamic_cast<op::scan::duckdb_native_ingestible_table_info const*>(&info)) {
    ci.catalog_name = d->catalog_name;
    ci.schema_name  = d->schema_name;
    ci.table_name   = d->table_name;
    ci.column_ids   = d->column_ids;
    ci.names        = aligned_column_names(d->names, d->column_ids);
  }
  return ci;
}

std::vector<std::size_t> cache_entry_info::can_serve_with_columns(
  const op::scan::ingestible_table_info& other) const
{
  // A parquet pin serves a parquet scan over the same file set; a duckdb pin
  // serves a duckdb scan over the same catalog.schema.table. A cache of one format
  // never serves a scan of the other — the identity check below falls through (a
  // duckdb cache has empty resolved_file_paths; a parquet cache has an empty table_name).
  if (auto const* p = dynamic_cast<op::scan::parquet_ingestible_table_info const*>(&other)) {
    if (resolved_file_paths.size() != p->resolved_file_paths.size()) { return {}; }
    auto these_files = resolved_file_paths;
    auto those_files = p->resolved_file_paths;
    std::sort(these_files.begin(), these_files.end());
    std::sort(those_files.begin(), those_files.end());
    if (these_files != those_files) { return {}; }
    return column_superset_projection(column_ids, p->column_ids);
  }
  if (auto const* d = dynamic_cast<op::scan::duckdb_native_ingestible_table_info const*>(&other)) {
    // Same duckdb table by qualified name (catalog.schema.table), derived on both
    // pin and query sides from the resolved DuckTableEntry — so the stored casing is
    // the table's canonical (case-preserved) name on both sides and a byte-exact
    // compare is correct. (If a future site ever populates these from parsed input
    // rather than the resolved entry, switch to a case-insensitive compare.)
    // A parquet cache has an empty table_name, so it never matches a duckdb scan.
    if (table_name.empty()) { return {}; }
    if (catalog_name != d->catalog_name || schema_name != d->schema_name ||
        table_name != d->table_name) {
      return {};
    }
    return column_superset_projection(column_ids, d->column_ids);
  }
  return {};
}

void sirius_scan_manager::insert_pinned_entry(
  const std::string& name,
  cache_entry_info cache_info,
  std::vector<std::unique_ptr<cudf::table>> data_tables,
  std::vector<cucascade::memory::memory_space*> chunk_memory_spaces,
  duckdb::vector<duckdb::LogicalType> column_types,
  std::vector<std::vector<duckdb::unique_ptr<duckdb::BaseStatistics>>> chunk_stats)
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

  // Column names (aligned with the cached column_ids) key data_batches_by_column.
  // Copied out before cache_info is moved into the entry below.
  std::vector<std::string> column_names = cache_info.column_names();

  // column_ids and names within cache_info are built aligned 1:1 by
  // cache_entry_info::from; the merge path below indexes column_ids by the same
  // position as the column names, so reject any misalignment loudly rather than
  // risk an out-of-bounds access.
  if (cache_info.column_ids.size() != column_names.size()) {
    throw std::invalid_argument(
      "[sirius_scan_manager::insert_pinned_entry] cache_info.column_ids.size() (" +
      std::to_string(cache_info.column_ids.size()) + ") must equal column_names size (" +
      std::to_string(column_names.size()) + ")");
  }

  // Normalize the optional zone-map capture.
  bool const stats_supplied = !column_types.empty() && !chunk_stats.empty();
  auto pin_zone_maps        = pinned_zone_maps::from_capture(std::move(column_types),
                                                      std::move(chunk_stats),
                                                      cache_info.column_ids.size(),
                                                      data_tables.size());
  if (stats_supplied && !pin_zone_maps.has_stats()) {
    SIRIUS_LOG_WARN(
      "[sirius_scan_manager::insert_pinned_entry] zone-map capture failure; pinning '{}' without "
      "statistics",
      name);
  }

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
      // Per-chunk ROW counts must match too: the byte-budget coalescer can slice
      // the same table at different row-group boundaries for a different column
      // set (varchar byte budgets are data-dependent), which still yields equal
      // chunk counts and — because round-robin placement is a function of chunk
      // index alone — identical memory_spaces. Merging such chunks would corrupt
      // the entry positionally (columns disagreeing on chunk boundaries) and
      // silently invalidate the per-chunk MVCC row-count map a duckdb pin stamps
      // via attach_mvcc_metadata. Reject loudly instead.
      if (!entry.data_batches_by_column.empty()) {
        auto const& existing_chunks = entry.data_batches_by_column.begin()->second;
        if (existing_chunks.size() != data_tables.size()) {
          throw std::runtime_error(
            "[sirius_scan_manager::insert_pinned_entry] merge mismatch — existing entry has " +
            std::to_string(existing_chunks.size()) + " chunks but the new materialization has " +
            std::to_string(data_tables.size()));
        }
        for (std::size_t i = 0; i < data_tables.size(); ++i) {
          if (!data_tables[i]) { continue; }
          if (existing_chunks[i]->size() != data_tables[i]->num_rows()) {
            throw std::runtime_error(
              "[sirius_scan_manager::insert_pinned_entry] merge mismatch — chunk " +
              std::to_string(i) + " has " + std::to_string(existing_chunks[i]->size()) +
              " rows in the existing entry but " + std::to_string(data_tables[i]->num_rows()) +
              " in the new materialization (same total, different chunk boundaries)");
          }
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
      // Reflect the merged columns in cache_info so can_serve_with_columns'
      // superset match — and the gather it drives — actually see them. Append
      // only columns that received data above (an empty data_tables call must
      // not list a column with no backing chunks in data_batches_by_column).
      // column_ids and names grow together and we only append, so the projection
      // positions already handed out for existing columns stay valid.
      bool const entry_had_stats = entry.zone_maps.has_stats();
      for (std::size_t i = 0; i < is_new_col.size(); ++i) {
        if (!is_new_col[i]) { continue; }
        if (!entry.data_batches_by_column.contains(column_names[i])) { continue; }
        entry.cache_info.column_ids.push_back(cache_info.column_ids[i]);
        entry.cache_info.names.push_back(column_names[i]);
        entry.zone_maps.append_column_from(pin_zone_maps, i);
      }
      if (entry_had_stats && !entry.zone_maps.has_stats()) {
        // append_from_column() cleared the sidecar stats because the new column's zone-map shape
        // disagreed with the existing entry's shape. Warn but still pin the entry.
        SIRIUS_LOG_WARN(
          "[sirius_scan_manager::insert_pinned_entry] merge appended columns without "
          "statistics; dropping zone-map statistics for entry '{}'",
          name);
      }
      // A statless entry adopts the incoming capture when it covers every entry column (by
      // primary index) — the re-pin recovery path for entries pinned while capture was off.
      // Safe because the merge guards above proved identical chunk boundaries.
      if (!entry.zone_maps.has_stats() && pin_zone_maps.has_stats()) {
        std::unordered_map<duckdb::idx_t, std::size_t> incoming_pos_by_primary;
        incoming_pos_by_primary.reserve(cache_info.column_ids.size());
        for (std::size_t i = 0; i < cache_info.column_ids.size(); ++i) {
          incoming_pos_by_primary.emplace(cache_info.column_ids[i].GetPrimaryIndex(), i);
        }
        std::vector<std::size_t> incoming_pos_by_entry_pos;
        incoming_pos_by_entry_pos.reserve(entry.cache_info.column_ids.size());
        for (auto const& col : entry.cache_info.column_ids) {
          auto it = incoming_pos_by_primary.find(col.GetPrimaryIndex());
          if (it == incoming_pos_by_primary.end()) { break; }
          incoming_pos_by_entry_pos.push_back(it->second);
        }
        if (incoming_pos_by_entry_pos.size() == entry.cache_info.column_ids.size()) {
          entry.zone_maps =
            pinned_zone_maps::remap(std::move(pin_zone_maps), incoming_pos_by_entry_pos);
        }
      }
      return;
    }
    // Row count or completeness contract differs → drop the stale entry and rebuild below.
    _pinned_entries.erase(existing_it);
  }

  pinned_entry entry;
  entry.cache_info          = std::move(cache_info);
  entry.chunk_memory_spaces = std::move(chunk_memory_spaces);
  entry.tier                = cucascade::memory::Tier::GPU;
  entry.num_rows            = new_num_rows;
  entry.zone_maps           = std::move(pin_zone_maps);

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
  cache_entry_info cache_info,
  std::vector<std::shared_ptr<cucascade::host_data_representation>> host_chunks,
  cucascade::memory::memory_space& memory_space,
  duckdb::vector<duckdb::LogicalType> column_types,
  std::vector<std::vector<duckdb::unique_ptr<duckdb::BaseStatistics>>> chunk_stats)
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

  // Normalize the optional zone-map capture
  bool const stats_supplied = !column_types.empty() && !chunk_stats.empty();
  auto pin_zone_maps        = pinned_zone_maps::from_capture(std::move(column_types),
                                                      std::move(chunk_stats),
                                                      cache_info.column_ids.size(),
                                                      host_chunks.size());
  if (stats_supplied && !pin_zone_maps.has_stats()) {
    // Only reason stats failed to append is a shape mismatch between the captured stats and the
    // pinned entry; warn but still pin the entry.
    SIRIUS_LOG_WARN(
      "[sirius_scan_manager::insert_pinned_entry_host] zone-map capture shape mismatch; "
      "pinning '{}' without statistics",
      name);
  }

  pinned_entry entry;
  entry.cache_info   = std::move(cache_info);
  entry.tier         = cucascade::memory::Tier::HOST;
  entry.memory_space = &memory_space;
  entry.num_rows     = new_num_rows;
  entry.host_chunks  = std::move(host_chunks);
  entry.zone_maps    = std::move(pin_zone_maps);

  _pinned_entries[name] = std::move(entry);
}

void sirius_scan_manager::attach_mvcc_metadata(const std::string& name,
                                               duckdb_mvcc_metadata metadata)
{
  auto it = _pinned_entries.find(name);
  if (it == _pinned_entries.end()) {
    throw std::invalid_argument("[attach_mvcc_metadata] no pinned entry named '" + name + "'");
  }
  it->second.mvcc = std::make_unique<duckdb_mvcc_metadata>(std::move(metadata));
}

void sirius_scan_manager::remove_pinned_entry(const std::string& name)
{
  _pinned_entries.erase(name);
}

void sirius_scan_manager::visit_pinned_entries(
  const std::function<bool(std::string_view, const pinned_entry&)>& visitor) const
{
  for (auto const& [name, entry] : _pinned_entries) {
    if (!visitor(name, entry)) { break; }
  }
}

void validate_pinned_entry_for_serving(pinned_entry const& entry,
                                       std::span<std::size_t const> selected_columns)
{
  auto const& entry_column_names = entry.cache_info.column_names();

  if (entry.tier == cucascade::memory::Tier::GPU) {
    if (entry.data_batches_by_column.empty()) { return; }  // legitimate zero-chunk entry
    auto const n_chunks = entry.data_batches_by_column.begin()->second.size();
    if (entry.chunk_memory_spaces.size() != n_chunks) {
      throw std::runtime_error(
        "pinned entry's chunk_memory_spaces does not cover every chunk of the entry");
    }
    for (auto const* space : entry.chunk_memory_spaces) {
      if (!space) { throw std::runtime_error("pinned entry has a null chunk memory_space"); }
    }
    for (auto idx : selected_columns) {
      auto const& name = entry_column_names.at(idx);
      auto it          = entry.data_batches_by_column.find(name);
      if (it == entry.data_batches_by_column.end() || it->second.size() != n_chunks) {
        throw std::runtime_error("pinned column '" + name +
                                 "' does not cover every chunk of the entry");
      }
      for (auto const& chunk : it->second) {
        if (!chunk) { throw std::runtime_error("pinned column '" + name + "' has a null chunk"); }
      }
    }
    return;
  }

  if (entry.tier == cucascade::memory::Tier::HOST) {
    for (auto const& chunk : entry.host_chunks) {
      if (!chunk) { throw std::runtime_error("pinned entry has a null host chunk"); }
    }
    return;
  }

  throw std::runtime_error("pinned entry has an unsupported tier");
}

cached_scan_plan build_cached_scan_plan(pinned_entry const& entry,
                                        duckdb::TableFilterSet const* table_filters,
                                        duckdb::vector<duckdb::ColumnIndex> const* column_ids)
{
  std::size_t n_chunks = 0;
  if (entry.tier == cucascade::memory::Tier::GPU) {
    n_chunks = entry.data_batches_by_column.empty()
                 ? 0
                 : entry.data_batches_by_column.begin()->second.size();
  } else if (entry.tier == cucascade::memory::Tier::HOST) {
    n_chunks = entry.host_chunks.size();
  }

  // Identity plan (serve everything) until proven otherwise.
  cached_scan_plan plan;
  plan.survivor_chunk_indices.reserve(n_chunks);
  for (std::size_t c = 0; c < n_chunks; ++c) {
    plan.survivor_chunk_indices.push_back(c);
  }
  if (n_chunks == 0 || !entry.zone_maps.has_stats() || table_filters == nullptr ||
      table_filters->filters.empty() || column_ids == nullptr) {
    return plan;
  }

  // Filter keys are positions into the QUERY's column_ids (remapped at plan time by
  // create_table_filter_set); map each to its primary/storage index, then to the entry's positional
  // column.
  std::unordered_map<duckdb::idx_t, std::size_t> entry_pos_by_primary;
  entry_pos_by_primary.reserve(entry.cache_info.column_ids.size());
  for (std::size_t i = 0; i < entry.cache_info.column_ids.size(); ++i) {
    entry_pos_by_primary.emplace(entry.cache_info.column_ids[i].GetPrimaryIndex(), i);
  }

  struct usable_filter {
    duckdb::TableFilter const* filter;
    std::size_t entry_pos;
  };
  std::vector<usable_filter> usable;
  for (auto const& [col_idx, filter] : table_filters->filters) {
    if (!filter) { continue; }
    if (col_idx >= column_ids->size()) { continue; }  // defensive
    auto const& column_id = (*column_ids)[col_idx];
    // rowid / empty / virtual sentinels have no storage stats.
    if (!column_id.HasPrimaryIndex() || column_id.IsRowIdColumn() || column_id.IsEmptyColumn() ||
        column_id.IsVirtualColumn()) {
      continue;
    }
    auto it = entry_pos_by_primary.find(column_id.GetPrimaryIndex());
    if (it == entry_pos_by_primary.end()) { continue; }
    auto const pos = it->second;
    if (pos >= entry.zone_maps.column_count()) { continue; }  // absent for this column
    if (!filter_safe_for_stats(*filter, entry.zone_maps.column_type(pos))) { continue; }
    usable.push_back({filter.get(), pos});
  }
  if (usable.empty()) { return plan; }

  plan.survivor_chunk_indices.clear();
  for (std::size_t c = 0; c < n_chunks; ++c) {
    bool pruned = false;
    for (auto const& uf : usable) {
      auto const* cell = entry.zone_maps.cell(uf.entry_pos, c);
      if (cell != nullptr && chunk_provably_empty(*uf.filter, *cell)) {
        pruned = true;
        break;
      }
    }
    if (!pruned) { plan.survivor_chunk_indices.push_back(c); }
  }
  plan.pruned = n_chunks - plan.survivor_chunk_indices.size();

  // Sentinel chunk: an all-pruned scan must not become a zero-batch scan (zero
  // splits => zero tasks => pipeline completion never fires — the hazard both
  // disk paths guard against). Keep chunk 0; the GPU filter empties it.
  if (plan.survivor_chunk_indices.empty()) {
    plan.survivor_chunk_indices.push_back(0);
    plan.pruned = n_chunks - 1;
  }
  return plan;
}

bool sirius_scan_manager::try_assign_cached_entries(op::scan::sirius_gpu_scan_operator* op)
{
  const auto& table_info = op->get_ingestible().table_info();

  for (auto const& [pinned_name, entry] : _pinned_entries) {
    // Identity + serviceability gate: empty when this cache cannot serve the scan
    // (wrong format / file-set / table, or missing a requested column).
    if (entry.cache_info.can_serve_with_columns(table_info).empty()) { continue; }
    // #819 cache-or-CPU: for duckdb pins with MVCC metadata, the plan-time
    // guards promised this scan serves from the pin — the disk-native path is
    // MVCC-blind and increasingly stale under the pin's checkpoint
    // suppression, so any failure past the identity gate is a loud error,
    // never a silent disk fallback.
    bool const mvcc_strict = entry.mvcc != nullptr;
    try {
      // Serve cached columns in the ingestible's materialized (disk-decode) order rather
      // than raw column_ids order, so post_filter_and_project's index-based filter and
      // projection bind to the same columns they would on the disk read path.
      auto cols = gather_by_primary_index(entry.cache_info.column_ids,
                                          op->get_ingestible().materialized_column_order());
      if (cols.empty()) {
        throw std::runtime_error("materialized column set is not a subset of the cached columns");
      }
      // Serve-time defense: a malformed entry would end the cached batch stream
      // early (nullptr mid-stream reads as end-of-stream) and silently truncate
      // the scan; validate up front so it throws here and the catch below falls
      // back to the disk read instead.
      validate_pinned_entry_for_serving(entry, cols);
      // Zone-map survivor plan
      auto const filter_view =
        _pruning_enabled ? extract_scan_filters(table_info) : scan_filter_view{};
      auto plan = build_cached_scan_plan(entry, filter_view.table_filters, filter_view.column_ids);
      auto const total_chunks = plan.survivor_chunk_indices.size() + plan.pruned;
      if (plan.pruned > 0) {
        SIRIUS_LOG_INFO(
          "[sirius_scan_manager] zone-map pruning for pinned entry '{}' ({} tier): {}/{} chunks "
          "pruned, serving {}",
          pinned_name,
          entry.tier == cucascade::memory::Tier::GPU ? "GPU" : "HOST",
          plan.pruned,
          total_chunks,
          plan.survivor_chunk_indices.size());
      } else {
        SIRIUS_LOG_DEBUG(
          "[sirius_scan_manager] zone-map pruning for pinned entry '{}': nothing pruned ({} "
          "chunks served)",
          pinned_name,
          total_chunks);
      }

      std::shared_ptr<mvcc_chunk_mask_set const> provider_masks;  // stays null for parquet pins
      if (entry.mvcc) {
        auto const* duckdb_info =
          dynamic_cast<op::scan::duckdb_native_ingestible_table_info const*>(&table_info);
        if (duckdb_info == nullptr || duckdb_info->storage == nullptr ||
            duckdb_info->context == nullptr) {
          throw std::runtime_error("mvcc-pinned entry matched a scan without duckdb table state");
        }
        auto const n_chunks      = entry.mvcc->base_row_count_per_chunk.size();
        auto const stored_chunks = entry.tier == cucascade::memory::Tier::GPU
                                     ? (entry.data_batches_by_column.empty()
                                          ? 0
                                          : entry.data_batches_by_column.begin()->second.size())
                                     : entry.host_chunks.size();
        if (stored_chunks != n_chunks) {
          // PR1 keeps mvcc counts and entry chunks in lock-step (merge-path
          // guard); a mismatch here means masks would bind to the wrong rows.
          throw std::runtime_error("mvcc metadata covers " + std::to_string(n_chunks) +
                                   " chunk(s) but the entry stores " +
                                   std::to_string(stored_chunks));
        }
        auto masks     = std::make_shared<mvcc_chunk_mask_set>(n_chunks);
        provider_masks = masks;
        mvcc_mask_job_request request;
        request.masks    = std::move(masks);
        request.metadata = *entry.mvcc;
        request.storage  = duckdb_info->storage;
        request.context  = duckdb_info->context;
        request.chunk_spaces =
          entry.tier == cucascade::memory::Tier::GPU
            ? entry.chunk_memory_spaces
            : std::vector<cucascade::memory::memory_space*>(n_chunks, entry.memory_space);
        request.entry_name = pinned_name;
        _pending_mvcc_mask_jobs.push_back(std::move(request));
      }

      auto provider = make_provider_for_pinned_entry(
        entry, cols, std::move(plan), op->batch_telemetry(), std::move(provider_masks));
      _metadata_processor->use_cached_entries_for_pipeline(op, std::move(provider));
      SIRIUS_LOG_INFO("[sirius_scan_manager] assigned pinned entry '{}' to operator '{}'",
                      pinned_name,
                      op->get_operator_id());
      return true;
    } catch (std::exception const& e) {
      if (mvcc_strict) {
        throw std::runtime_error("[sirius_scan_manager] mvcc-pinned entry '" + pinned_name +
                                 "' matched operator '" + std::to_string(op->get_operator_id()) +
                                 "' but cannot serve from the cache: " + e.what());
      }
      SIRIUS_LOG_ERROR(
        "[sirius_scan_manager] error while trying to assign cached entries to "
        "operator '{}': {}",
        op->get_operator_id(),
        e.what());
      return false;
    } catch (...) {
      if (mvcc_strict) { throw; }
      SIRIUS_LOG_ERROR(
        "[sirius_scan_manager] error while trying to assign cached entries to "
        "operator '{}'",
        op->get_operator_id());
      return false;
    }
  }
  return false;
}

}  // namespace sirius::scan_manager
