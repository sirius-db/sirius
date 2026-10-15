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

#include "vss/kmeans_functions.hpp"

#include "cuda/vss/cudf_raft_interop.hpp"
#include "data/data_batch_utils.hpp"
#include "duckdb/common/exception.hpp"
#include "pipeline/batch_lock_utils.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_context.hpp"
#include "telemetry/data_batch_probe.hpp"
#include "vss/bound_gemm.hpp"
#include "vss/cluster_fold.hpp"
#include "vss/cluster_lists.hpp"
#include "vss/cuvs_index_cache.hpp"
#include "vss/device_context_guard.hpp"
#include "vss/distance_metric.hpp"
#include "vss/pinned_column.hpp"
#include "vss/vector_search_internal.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/filling.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <raft/core/device_resources.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>

#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/cudf/host_table.hpp>
#include <cucascade/memory/column_metadata.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace sirius::vss {

namespace {

/// Handles every kmeans function needs before it can touch a pinned column.
struct kmeans_context {
  cucascade::memory::memory_space* space;
  const cucascade::memory::memory_space* host_space;
  std::shared_ptr<const scan_manager::pinned_entry> pin;
  int target_gpu;
};

kmeans_context resolve_context(duckdb::SiriusContext& ctx,
                               const std::string& fn,
                               const std::string& catalog,
                               const std::string& schema,
                               const std::string& table)
{
  auto& memory_manager = ctx.get_memory_manager();
  auto gpu_spaces      = memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  if (gpu_spaces.empty()) {
    throw duckdb::InvalidInputException(fn + ": no GPU memory space available");
  }
  auto host_spaces = memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  if (host_spaces.empty()) {
    throw duckdb::InvalidInputException(fn + ": no HOST memory space available");
  }

  auto pin = ctx.get_scan_manager().find_pinned_entry_for_duckdb_table(catalog, schema, table);
  if (pin == nullptr) {
    throw duckdb::InvalidInputException(fn + ": table '" + table +
                                        "' must be pinned (GPU or HOST tier)");
  }

  auto* space = const_cast<cucascade::memory::memory_space*>(gpu_spaces.front());
  return kmeans_context{space, host_spaces.front(), pin, space->get_device_id()};
}

/// The clustering stored under @p name, or nullptr when the name is free or holds something
/// that is not a clustering.
const pinned_index_entry* find_clustering_entry(duckdb::SiriusContext& ctx, const std::string& name)
{
  auto const entry = ctx.get_cuvs_index_cache().find(name);
  if (entry == nullptr || entry->meta.kind != index_kind::kmeans_centroids) { return nullptr; }
  return entry.get();
}

/// The lists live beside their clustering under a derived name, so re-fitting the clustering
/// can drop them without the cache knowing the two are related.
std::string lists_key(const std::string& clustering) { return clustering + "::lists"; }

/// SIRIUS_VSS_EXACT_COPY=1: FLOAT16 and INT8 lists keep their own FP32 copy instead of reading the
/// pin.
bool exact_copy_asked()
{
  auto const* v = std::getenv("SIRIUS_VSS_EXACT_COPY");
  return v != nullptr && std::string_view{v} == "1";
}

}  // namespace

const cudf::column* find_clustering_centroids(duckdb::SiriusContext& ctx, const std::string& name)
{
  const auto* entry = find_clustering_entry(ctx, name);
  if (entry == nullptr) { return nullptr; }
  auto* held = entry->index_as<std::unique_ptr<cudf::column>>();
  return held != nullptr ? held->get() : nullptr;
}

kmeans_fit_result run_kmeans_fit(duckdb::SiriusContext& ctx, const kmeans_fit_request& req)
{
  static const std::string fn = "sirius_kmeans_fit";
  auto const c                = resolve_context(ctx, fn, req.catalog, req.schema, req.table);
  device_context_guard device_guard{c.target_gpu};
  // The centroids outlive this call by design -- they stay in the index cache until the session
  // ends. rmm::device_buffer records the stream it was allocated on and deallocates on that same
  // stream, so an owned stream here would be destroyed long before the cache frees the column,
  // and the free would run on a dead stream. The default stream outlives both.
  auto stream = cudf::get_default_stream();

  auto const n_rows = static_cast<std::int64_t>(c.pin->num_rows);
  if (n_rows == 0) {
    throw duckdb::InvalidInputException(fn + ": table '" + req.table + "' is empty");
  }

  auto const metric     = ann_distance_type_from_metric(req.metric);
  auto const n_clusters = resolve_n_clusters(req.spec.n_clusters, n_rows);
  auto const train_rows = resolve_train_rows(req.spec.train_rows, n_rows, n_clusters);

  // Over-reserved to cover the sample, the concatenated copy train_centroids makes of it, and
  // k-means' own scratch; shrunk to the centroids alone once the fit is done.
  auto const vec_bytes        = static_cast<std::size_t>(req.dim) * sizeof(float);
  std::size_t const footprint = static_cast<std::size_t>(train_rows) * vec_bytes * 3 +
                                static_cast<std::size_t>(n_clusters) * vec_bytes * 2 +
                                (std::size_t{1} << 20);

  auto& index_cache = ctx.get_cuvs_index_cache();
  // Free the old clustering's reservation before asking for the new one, so re-fitting under
  // the same name does not have to fit both at once.
  index_cache.erase(req.name);
  index_cache.erase(lists_key(req.name));
  auto reservation = index_cache.reserve_index_memory(footprint, c.target_gpu);
  if (!reservation) {
    throw duckdb::InvalidInputException(fn + ": not enough free GPU memory to train " +
                                        std::to_string(n_clusters) + " centroids: need ~" +
                                        std::to_string(footprint >> 20) + " MiB");
  }
  // The reservation is the admission check for the training footprint; the centroids themselves
  // are ordinary GPU memory the cache entry owns, so they are drawn from the space, not from it.
  auto const reserved_bytes = reservation->size();
  reservation.reset();
  auto const mr = c.space->get_default_allocator();

  telemetry::batch_telemetry_info const telemetry_info{};
  auto const n_chunks = pinned_column_chunk_count(*c.pin, req.column);

  std::unique_ptr<cudf::column> centroids;
  std::int64_t sampled = 0;
  {
    // Each chunk is staged, reduced to its share of the sample, and released before the next
    // is staged, so a corpus larger than device memory can still be clustered.
    std::vector<std::unique_ptr<cudf::column>> samples;
    auto const share = [&](std::int64_t rows) {
      auto const take = std::min<std::int64_t>(rows, ((train_rows * rows) + n_rows - 1) / n_rows);
      return std::min<std::int64_t>(take, train_rows - sampled);
    };
    // A host-tier pin gives up its sample in place: the device reads the strided rows through the
    // pin's mapping. They are a few percent of the column, which staging moved whole to take them.
    auto const map = c.pin->tier == cucascade::memory::Tier::HOST
                       ? map_pinned_rows(*c.pin, req.column, req.dim, *c.space, stream, mr)
                       : std::nullopt;
    for (std::size_t i = 0; map && i < n_chunks; ++i) {
      auto const first = map->chunk_first_row[i];
      auto const rows  = map->chunk_first_row[i + 1] - first;
      if (rows == 0) { continue; }
      auto const take = share(rows);
      if (take <= 0) { break; }
      auto const pick = strided_row_sample(rows, take, req.spec.seed);
      samples.push_back(gather_pinned_rows(
        map->view(), first + pick.start, pick.stride, pick.count, req.dim, stream, mr));
      sampled += take;
    }
    for (std::size_t i = 0; !map && i < n_chunks; ++i) {
      auto staged =
        stage_pinned_column_chunk(*c.pin, req.column, i, *c.space, stream, telemetry_info);
      auto const rows = static_cast<std::int64_t>(staged.view.size());
      if (rows == 0) { continue; }

      auto const take = share(rows);
      if (take <= 0) { break; }
      samples.push_back(sample_vector_rows(staged.view, take, req.spec.seed, stream, mr));
      // The sample borrows the staged chunk, which this iteration is about to release.
      stream.synchronize();
      sampled += take;
    }
    if (samples.empty()) {
      throw duckdb::InvalidInputException(fn + ": column '" + req.column + "' has no rows");
    }

    std::vector<cudf::column_view> views;
    views.reserve(samples.size());
    for (auto const& s : samples) {
      views.push_back(s->view());
    }

    // Both counts are already resolved against the full table; passing them through stops
    // train_centroids from re-deriving them from the sample it is handed.
    clustering_spec spec = req.spec;
    spec.n_clusters      = n_clusters;
    spec.train_rows      = sampled;
    spec.metric          = metric;
    centroids            = train_centroids(views, req.dim, spec, stream, mr);
  }
  index_metadata meta;
  meta.kind           = index_kind::kmeans_centroids;
  meta.catalog_name   = req.catalog;
  meta.schema_name    = req.schema;
  meta.table_name     = req.table;
  meta.column_name    = req.column;
  meta.dim            = req.dim;
  meta.num_rows       = n_rows;
  meta.n_lists        = n_clusters;
  meta.metric         = metric;
  meta.resident_bytes = static_cast<std::size_t>(centroids->size()) * sizeof(float);
  (void)reserved_bytes;
  index_cache.insert(
    req.name, std::move(meta), make_cuvs_index(std::move(centroids)), rmm::cuda_stream{});

  return kmeans_fit_result{n_clusters, req.dim, sampled, n_rows};
}

std::unique_ptr<cucascade::host_data_representation> run_kmeans_assign(
  duckdb::SiriusContext& ctx, const kmeans_assign_request& req)
{
  static const std::string fn = "sirius_kmeans_assign";
  auto const c                = resolve_context(ctx, fn, req.catalog, req.schema, req.table);
  device_context_guard device_guard{c.target_gpu};
  rmm::cuda_stream stream_owner;
  auto stream   = stream_owner.view();
  auto const mr = c.space->get_default_allocator();

  const auto* entry = find_clustering_entry(ctx, req.clustering);
  if (entry == nullptr) {
    throw duckdb::InvalidInputException(fn + ": no clustering named '" + req.clustering +
                                        "'; run sirius_kmeans_fit first");
  }
  if (entry->meta.dim != req.dim) {
    throw duckdb::InvalidInputException(fn + ": clustering '" + req.clustering +
                                        "' is over FLOAT[" + std::to_string(entry->meta.dim) +
                                        "] vectors but column '" + req.column + "' is FLOAT[" +
                                        std::to_string(req.dim) + "]");
  }
  const auto* centroids = find_clustering_centroids(ctx, req.clustering);
  if (centroids == nullptr) {
    throw duckdb::InvalidInputException(fn + ": clustering '" + req.clustering +
                                        "' holds no centroids");
  }

  telemetry::batch_telemetry_info const telemetry_info{};
  auto const n_chunks = pinned_column_chunk_count(*c.pin, req.column);

  raft::device_resources res{stream};
  std::vector<std::unique_ptr<cudf::column>> row_ids;
  std::vector<std::unique_ptr<cudf::column>> cluster_ids;
  std::vector<std::unique_ptr<cudf::column>> distances;
  std::int64_t base = 0;
  for (std::size_t i = 0; i < n_chunks; ++i) {
    auto staged =
      stage_pinned_column_chunk(*c.pin, req.column, i, *c.space, stream, telemetry_info);
    auto const rows = static_cast<std::int64_t>(staged.view.size());
    if (rows == 0) { continue; }

    auto assignment = assign_to_centroids(
      res, staged.view, centroids->view(), req.dim, req.spec, base, entry->meta.metric, stream, mr);
    // The assignment reads the staged chunk, which is released at the end of this iteration.
    stream.synchronize();
    row_ids.push_back(std::move(assignment.row_ids));
    cluster_ids.push_back(std::move(assignment.cluster_ids));
    distances.push_back(std::move(assignment.distances));
    base += rows;
  }
  if (row_ids.empty()) {
    throw duckdb::InvalidInputException(fn + ": column '" + req.column + "' has no rows");
  }

  auto merge = [&](std::vector<std::unique_ptr<cudf::column>> const& parts) {
    if (parts.size() == 1) {
      return std::make_unique<cudf::column>(parts.front()->view(), stream, mr);
    }
    std::vector<cudf::column_view> views;
    views.reserve(parts.size());
    for (auto const& p : parts) {
      views.push_back(p->view());
    }
    return cudf::concatenate(views, stream, mr);
  };

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(merge(row_ids));
  columns.push_back(merge(cluster_ids));
  columns.push_back(merge(distances));
  auto table = std::make_unique<cudf::table>(std::move(columns));

  return vss_table_to_host(*c.space, *c.host_space, stream, std::move(table));
}

const cluster_lists* find_cluster_lists(duckdb::SiriusContext& ctx, const std::string& clustering)
{
  auto const entry = ctx.get_cuvs_index_cache().find(lists_key(clustering));
  if (entry == nullptr || entry->meta.kind != index_kind::cluster_lists) { return nullptr; }
  return entry->index_as<cluster_lists>();
}

std::optional<exact_lists_choice> find_exact_lists(duckdb::SiriusContext& ctx,
                                                   const std::string& catalog,
                                                   const std::string& schema,
                                                   const std::string& table,
                                                   const std::string& column,
                                                   bool cosine,
                                                   std::int64_t n_rows)
{
  for (auto const& name : ctx.get_cuvs_index_cache().names_on_column(
         catalog, schema, table, column, index_kind::kmeans_centroids)) {
    auto const* lists = find_cluster_lists(ctx, name);
    if (lists == nullptr || lists->n_rows != n_rows || lists->unit_rows != cosine) { continue; }
    bool const exact = lists->encoding == list_encoding::float32 ||
                       lists->encoding == list_encoding::uint8 ||
                       (lists->encoding == list_encoding::float16 && lists->has_exact_rows() &&
                        bound_filter_f16_supports(lists->dim));
    if (!exact) { continue; }
    auto const entry = find_clustering_entry(ctx, name);
    if (entry == nullptr) { continue; }
    auto const clusters = static_cast<std::int64_t>(lists->offsets.size()) - 1;
    return exact_lists_choice{name,
                              entry->meta.n_lists,
                              lists->encoding,
                              lists->tier == cucascade::memory::Tier::GPU,
                              clusters > 0 && lists->offsets.back() <= clusters * 20 * kSeedSample};
  }
  return std::nullopt;
}

std::optional<exact_lists_choice> find_named_lists(duckdb::SiriusContext& ctx,
                                                   const std::string& clustering,
                                                   const std::string& catalog,
                                                   const std::string& schema,
                                                   const std::string& table,
                                                   const std::string& column,
                                                   bool cosine,
                                                   std::int64_t n_rows)
{
  auto const on_column = ctx.get_cuvs_index_cache().names_on_column(
    catalog, schema, table, column, index_kind::kmeans_centroids);
  if (std::find(on_column.begin(), on_column.end(), clustering) == on_column.end()) {
    return std::nullopt;
  }
  auto const* lists = find_cluster_lists(ctx, clustering);
  auto const entry  = find_clustering_entry(ctx, clustering);
  if (lists == nullptr || entry == nullptr || lists->n_rows != n_rows ||
      lists->unit_rows != cosine) {
    return std::nullopt;
  }
  auto const clusters = static_cast<std::int64_t>(lists->offsets.size()) - 1;
  return exact_lists_choice{clustering,
                            entry->meta.n_lists,
                            lists->encoding,
                            lists->tier == cucascade::memory::Tier::GPU,
                            clusters > 0 && lists->offsets.back() <= clusters * 20 * kSeedSample};
}

void erase_cluster_lists(duckdb::SiriusContext& ctx, const std::string& clustering)
{
  ctx.get_cuvs_index_cache().erase(lists_key(clustering));
}

cluster_lists_result run_kmeans_build_lists(duckdb::SiriusContext& ctx,
                                            const kmeans_assign_request& req,
                                            list_storage storage,
                                            bool host_tier,
                                            bool unit_rows)
{
  static const std::string fn = "sirius_kmeans_build_lists";
  if (unit_rows && (storage == list_storage::uint8 || storage == list_storage::int8)) {
    throw duckdb::InvalidInputException(
      fn + ": metric => 'cosine' stores unit rows, which need storage => 'float16' or 'float32'");
  }
  auto const c = resolve_context(ctx, fn, req.catalog, req.schema, req.table);
  device_context_guard device_guard{c.target_gpu};
  // The lists outlive this call in the index cache, and a device_buffer frees on the stream it
  // was allocated on, so the persistent buffers use the default stream (as the fit's centroids
  // do); per-chunk scratch uses the owned one.
  auto const persistent = cudf::get_default_stream();
  rmm::cuda_stream stream_owner;
  auto stream       = stream_owner.view();
  auto const mr     = c.space->get_default_allocator();
  auto& index_cache = ctx.get_cuvs_index_cache();

  const auto* entry     = find_clustering_entry(ctx, req.clustering);
  const auto* centroids = find_clustering_centroids(ctx, req.clustering);
  if (entry == nullptr || centroids == nullptr) {
    throw duckdb::InvalidInputException(fn + ": no clustering named '" + req.clustering +
                                        "'; run sirius_kmeans_fit first");
  }
  if (entry->meta.dim != req.dim) {
    throw duckdb::InvalidInputException(fn + ": clustering '" + req.clustering +
                                        "' is over FLOAT[" + std::to_string(entry->meta.dim) +
                                        "] vectors but column '" + req.column + "' is FLOAT[" +
                                        std::to_string(req.dim) + "]");
  }
  // INT8 and FLOAT16 lists only filter; exact answers come from the bounded search's FP32
  // re-check, which needs these widths. Without it every join would refuse the lists (INT8) or
  // answer from rounded rows (FLOAT16), so refuse here, before the old lists are dropped.
  if (storage == list_storage::int8 && !bound_filter_int8_supports(req.dim)) {
    throw duckdb::InvalidInputException(fn +
                                        ": storage => 'int8' needs a vector width that is a "
                                        "multiple of 16 and at most 256, not " +
                                        std::to_string(req.dim) + "; use 'float16' or 'float32'");
  }
  if ((storage == list_storage::float16 || storage == list_storage::exact) &&
      !bound_filter_f16_supports(req.dim)) {
    throw duckdb::InvalidInputException(fn +
                                        ": storage => 'float16' needs a vector width that is "
                                        "a multiple of 16, not " +
                                        std::to_string(req.dim) + "; use 'float32'");
  }

  // Built afresh every time, so the old copy is freed before the new one is allocated.
  index_cache.erase(lists_key(req.clustering));

  auto const n_rows     = static_cast<std::int64_t>(c.pin->num_rows);
  auto const dim        = req.dim;
  auto const n_clusters = entry->meta.n_lists;
  if (n_rows == 0) {
    throw duckdb::InvalidInputException(fn + ": table '" + req.table + "' is empty");
  }

  telemetry::batch_telemetry_info const telemetry_info{};
  auto const n_chunks = pinned_column_chunk_count(*c.pin, req.column);
  raft::device_resources res{stream};
  // Same switch as the join's phase timer; synchronizes, so only for attribution.
  auto const dbg = std::getenv("SIRIUS_VECTOR_JOIN_PHASE_DEBUG") != nullptr;
  auto phase_t0  = std::chrono::steady_clock::now();
  auto phase     = [&](const char* name) {
    if (!dbg) { return; }
    stream.synchronize();
    auto const now = std::chrono::steady_clock::now();
    std::fprintf(stderr,
                 "[vecjoin-lists] %-18s %8.3f s\n",
                 name,
                 std::chrono::duration<double>(now - phase_t0).count());
    phase_t0 = now;
  };

  // Pass 1: every row's nearest centroid, to the host. 4 bytes a row, so even 100M rows is
  // 400 MB, and the host is where the counting sort below is cheapest to run deterministically.
  std::vector<std::int32_t> labels(static_cast<std::size_t>(n_rows));
  // Whether every component is a byte, counted while the chunks are resident anyway: that is
  // what decides if the lists may be stored as UINT8 without changing a single value.
  bool const check_uint8 =
    !unit_rows && (storage == list_storage::automatic || storage == list_storage::uint8 ||
                   storage == list_storage::exact);
  rmm::device_uvector<unsigned long long> non_uint8(1, stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(non_uint8.data(), 0, sizeof(unsigned long long), stream.value()));
  // INT8 codes need each component's range over every row, taken while the chunks are resident.
  bool const want_int8 = storage == list_storage::int8;
  std::optional<rmm::device_uvector<unsigned int>> col_min, col_max;
  if (want_int8) {
    col_min.emplace(static_cast<std::size_t>(dim), stream, mr);
    col_max.emplace(static_cast<std::size_t>(dim), stream, mr);
    CUDF_CUDA_TRY(cudaMemsetAsync(
      col_min->data(), 0xff, static_cast<std::size_t>(dim) * sizeof(unsigned int), stream.value()));
    CUDF_CUDA_TRY(cudaMemsetAsync(
      col_max->data(), 0, static_cast<std::size_t>(dim) * sizeof(unsigned int), stream.value()));
  }
  {
    centroid_labeler labeler{res, centroids->view(), dim, entry->meta.metric, mr};
    pinned_chunk_walk walk{*c.pin, req.column, *c.space, stream, telemetry_info};
    std::int64_t base = 0;
    for (std::size_t i = 0; i < n_chunks; ++i) {
      auto staged     = walk.take(i);
      auto const rows = static_cast<std::int64_t>(staged.view.size());
      if (rows == 0) { continue; }
      if (base + rows > n_rows) {
        throw std::runtime_error(fn + ": pinned column holds more rows than the pin reports");
      }
      rmm::device_uvector<std::int32_t> chunk_labels(static_cast<std::size_t>(rows), stream, mr);
      labeler.label(staged.view, chunk_labels.data());
      if (check_uint8) {
        count_non_uint8(list_column_as_dataset_view(staged.view, dim).data_handle(),
                        rows * dim,
                        non_uint8.data(),
                        stream);
      }
      if (want_int8) {
        column_min_max(list_column_as_dataset_view(staged.view, dim).data_handle(),
                       rows,
                       dim,
                       col_min->data(),
                       col_max->data(),
                       stream);
      }
      // Before the copy below, which waits for the labels: into pageable memory it is synchronous.
      walk.prefetch_after(i);
      CUDF_CUDA_TRY(cudaMemcpyAsync(labels.data() + base,
                                    chunk_labels.data(),
                                    static_cast<std::size_t>(rows) * sizeof(std::int32_t),
                                    cudaMemcpyDeviceToHost,
                                    stream.value()));
      stream.synchronize();
      base += rows;
    }
    if (base != n_rows) {
      throw std::runtime_error(fn + ": pinned column holds " + std::to_string(base) +
                               " rows but the pin reports " + std::to_string(n_rows));
    }
  }

  phase("pass 1 labels");

  unsigned long long non_uint8_host = 0;
  CUDF_CUDA_TRY(cudaMemcpyAsync(&non_uint8_host,
                                non_uint8.data(),
                                sizeof(non_uint8_host),
                                cudaMemcpyDeviceToHost,
                                stream.value()));
  stream.synchronize();
  if (storage == list_storage::uint8 && non_uint8_host != 0) {
    throw duckdb::InvalidInputException(fn + ": storage => 'uint8' needs every value of '" +
                                        req.column + "' to be an integer in [0, 255]; " +
                                        std::to_string(non_uint8_host) + " are not");
  }
  // 'auto' is the planner's choice, the one a statement that builds its own lists makes ('exact'):
  // bytes as UINT8, anything else as FLOAT16 that the bounded search re-scores in FP32 -- where the
  // width allows it and the FP32 rows can be read, in place from the pin or as a host copy that
  // fits. Where they cannot, FP32 lists, which answer without a re-score.
  std::optional<pinned_rows_map> early_pin;
  if (storage == list_storage::automatic) {
    bool const bytes = check_uint8 && non_uint8_host == 0;
    bool f16         = !bytes && bound_filter_f16_supports(req.dim);
    if (f16 && !unit_rows && !exact_copy_asked()) {
      early_pin = map_pinned_rows(
        *c.pin, req.column, dim, *c.space, persistent, c.space->get_default_allocator());
      if (early_pin && early_pin->n_rows() != n_rows) { early_pin.reset(); }
    }
    if (f16 && !early_pin) {
      auto const copy_bytes =
        static_cast<std::size_t>(n_rows) * static_cast<std::size_t>(dim) * sizeof(float);
      f16 = c.host_space->get_available_memory() >= copy_bytes + (std::size_t{1} << 28);
    }
    storage = bytes ? list_storage::uint8 : f16 ? list_storage::exact : list_storage::float32;
  }
  auto const encoding = storage == list_storage::float16     ? list_encoding::float16
                        : want_int8                          ? list_encoding::int8
                        : check_uint8 && non_uint8_host == 0 ? list_encoding::uint8
                        : storage == list_storage::exact     ? list_encoding::float16
                                                             : list_encoding::float32;
  // INT8 codes: each component centered on the middle of its range, one scale sized so the
  // widest component's range fills [-127, 127]. A single scale keeps code dot products
  // proportional to FP32 ones, which is what lets the int8 GEMM filter.
  std::vector<float> code_offset_host;
  float code_scale = 0.f;
  if (want_int8) {
    std::vector<unsigned int> lo(static_cast<std::size_t>(dim)), hi(static_cast<std::size_t>(dim));
    CUDF_CUDA_TRY(cudaMemcpyAsync(lo.data(),
                                  col_min->data(),
                                  lo.size() * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost,
                                  stream.value()));
    CUDF_CUDA_TRY(cudaMemcpyAsync(hi.data(),
                                  col_max->data(),
                                  hi.size() * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost,
                                  stream.value()));
    stream.synchronize();
    code_offset_host.resize(static_cast<std::size_t>(dim));
    float half_range = 0.f;
    for (std::size_t j = 0; j < lo.size(); ++j) {
      auto const a        = decode_ordered_float(lo[j]);
      auto const b        = decode_ordered_float(hi[j]);
      code_offset_host[j] = 0.5f * (a + b);
      half_range          = std::max(half_range, 0.5f * (b - a));
    }
    code_scale = half_range > 0.f ? half_range / 127.f : 1.f;
  }

  // A stable counting sort, split over threads by row range: per-thread histograms, then each
  // thread's start inside each list is the list's start plus the rows earlier threads put
  // there. Rows keep their pin order within a list, so the result does not depend on the
  // thread count.
  auto const n_threads =
    static_cast<std::int64_t>(std::clamp<unsigned>(std::thread::hardware_concurrency(), 1u, 16u));
  auto const span = (n_rows + n_threads - 1) / n_threads;
  std::vector<std::vector<std::int64_t>> hist(
    static_cast<std::size_t>(n_threads),
    std::vector<std::int64_t>(static_cast<std::size_t>(n_clusters), 0));
  auto parallel = [&](auto&& body) {
    std::vector<std::thread> threads;
    for (std::int64_t t = 0; t < n_threads; ++t) {
      threads.emplace_back(
        [&, t] { body(t, std::min(n_rows, t * span), std::min(n_rows, (t + 1) * span)); });
    }
    for (auto& th : threads) {
      th.join();
    }
  };
  std::atomic<bool> bad_label{false};
  parallel([&](std::int64_t t, std::int64_t lo, std::int64_t hi) {
    auto& h = hist[static_cast<std::size_t>(t)];
    for (auto r = lo; r < hi; ++r) {
      auto const l = labels[static_cast<std::size_t>(r)];
      if (l < 0 || l >= n_clusters) {
        bad_label = true;
        return;
      }
      ++h[static_cast<std::size_t>(l)];
    }
  });
  if (bad_label) {
    throw std::runtime_error(fn + ": assignment produced a label outside the clustering");
  }

  cluster_lists lists;
  lists.pin        = c.pin.get();
  lists.table      = req.table;
  lists.column     = req.column;
  lists.n_rows     = n_rows;
  lists.dim        = dim;
  lists.n_clusters = n_clusters;
  lists.offsets.assign(static_cast<std::size_t>(n_clusters) + 1, 0);
  std::vector<std::vector<std::int64_t>> start = hist;
  {
    std::int64_t run = 0;
    for (std::int64_t k = 0; k < n_clusters; ++k) {
      lists.offsets[static_cast<std::size_t>(k)] = run;
      for (std::int64_t t = 0; t < n_threads; ++t) {
        start[static_cast<std::size_t>(t)][static_cast<std::size_t>(k)] = run;
        run += hist[static_cast<std::size_t>(t)][static_cast<std::size_t>(k)];
      }
    }
    lists.offsets.back() = run;
  }
  std::vector<std::int64_t> dest(static_cast<std::size_t>(n_rows));
  parallel([&](std::int64_t t, std::int64_t lo, std::int64_t hi) {
    auto& s = start[static_cast<std::size_t>(t)];
    for (auto r = lo; r < hi; ++r) {
      dest[static_cast<std::size_t>(r)] =
        s[static_cast<std::size_t>(labels[static_cast<std::size_t>(r)])]++;
    }
  });
  phase("counting sort");

  // Storage. The row map is device-resident on both tiers because the fold reads it per slice;
  // the vectors go to the device when the pin is there and they fit, else to pinned host blocks.
  auto const elem_bytes = list_encoding_bytes(encoding);
  auto const vec_bytes =
    static_cast<std::size_t>(n_rows) * static_cast<std::size_t>(dim) * elem_bytes;
  auto const row_bytes = static_cast<std::size_t>(n_rows) * sizeof(std::int64_t);
  auto* host_mr =
    c.host_space->get_memory_resource_as<cucascade::memory::fixed_size_host_memory_resource>();
  auto const block_bytes = host_mr != nullptr ? host_mr->get_block_size() : std::size_t{0};
  // A staged chunk is ~64 MiB of FP32 whatever the encoding. An FP32 chunk on the device is a
  // view and needs no bound but cudf's element count; a UINT8 one is widened into a buffer when
  // staged, so it is bounded like a host chunk.
  constexpr std::size_t kStagedChunkBytes = std::size_t{64} << 20;
  auto const staged_rows                  = std::max<std::int64_t>(
    1,
    static_cast<std::int64_t>(kStagedChunkBytes / (static_cast<std::size_t>(dim) * sizeof(float))));
  auto const max_list_elements =
    static_cast<std::int64_t>(std::numeric_limits<cudf::size_type>::max());

  std::unique_ptr<cucascade::memory::reservation> reservation;
  // FP32 lists live where the pin does; UINT8 ones are a quarter of the corpus and go to the
  // device whenever they fit, which is what takes the corpus stream off the query path.
  bool on_device = !host_tier && (c.pin->tier == cucascade::memory::Tier::GPU ||
                                  encoding != list_encoding::float32);
  if (on_device) {
    reservation = index_cache.reserve_index_memory(
      vec_bytes + row_bytes + row_bytes / 2 + (std::size_t{1} << 24), c.target_gpu);
    // Lists on the device are memory the engine cannot spill, so they go there only if what is left
    // can still run joins over them: a quarter of the device, and at least 2 GiB (one task over a
    // 200k-row probe chunk at 16 probes needs ~1.4 GiB). Squeezed in tighter, a join's tasks run
    // out of memory that nothing will free.
    auto const headroom =
      std::max<std::size_t>(std::size_t{2} << 30, c.space->get_max_memory() / 4);
    if (reservation != nullptr && c.space->get_available_memory() < headroom) {
      reservation.reset();
    }
    on_device = reservation != nullptr;
  }
  if (!on_device) {
    if (host_mr == nullptr || block_bytes == 0) {
      throw duckdb::InvalidInputException(fn + ": no pinned host memory pool to hold the lists");
    }
    reservation =
      index_cache.reserve_index_memory(row_bytes + (std::size_t{1} << 24), c.target_gpu);
    if (!reservation) {
      throw duckdb::InvalidInputException(fn + ": not enough free GPU memory for the lists' " +
                                          std::to_string(row_bytes >> 20) + " MiB row map");
    }
  }
  // The reservation above is only the admission check. A reservation charges the allocations
  // made on a stream it is attached to, and these buffers outlive every stream here, so drawing
  // them through it would count each byte twice -- reserved and allocated -- which at 100M rows is
  // enough to exhaust the pool. They are allocated from the space directly and counted once.
  auto const reserved_bytes = reservation->size();
  reservation.reset();
  auto const persistent_mr = c.space->get_default_allocator();

  lists.encoding  = encoding;
  lists.unit_rows = unit_rows;
  // HOST tier: the chunks' pinned blocks, filled by the scatter below and wrapped as data
  // batches once it is done.
  std::vector<cucascade::memory::fixed_multiple_blocks_allocation> host_chunks;
  if (on_device) {
    lists.tier       = cucascade::memory::Tier::GPU;
    lists.chunk_rows = encoding != list_encoding::float32
                         ? std::min<std::int64_t>(n_rows, staged_rows)
                         : std::min<std::int64_t>(n_rows, max_list_elements / dim);
    // Four rows of slack past the end: the int8 search rounds a slice up to a multiple of 4 rows.
    lists.device_vectors = std::make_unique<rmm::device_buffer>(
      vec_bytes + 4 * static_cast<std::size_t>(dim) * elem_bytes, persistent, persistent_mr);
  } else {
    lists.tier       = cucascade::memory::Tier::HOST;
    lists.host_space = const_cast<cucascade::memory::memory_space*>(c.host_space);
    lists.chunk_rows =
      std::min<std::int64_t>(std::min<std::int64_t>(n_rows, staged_rows), max_list_elements / dim);
    auto const n_chunks_h = static_cast<std::size_t>(lists.num_chunks());
    host_chunks.reserve(n_chunks_h);
    for (std::size_t ch = 0; ch < n_chunks_h; ++ch) {
      host_chunks.push_back(host_mr->allocate_multiple_blocks(
        static_cast<std::size_t>(lists.rows_in_chunk(static_cast<std::int64_t>(ch))) *
        static_cast<std::size_t>(dim) * elem_bytes));
    }
  }
  lists.row_ids = std::make_unique<rmm::device_buffer>(row_bytes, persistent, persistent_mr);
  if (encoding == list_encoding::uint8 || encoding == list_encoding::int8) {
    lists.row_sq = std::make_unique<rmm::device_buffer>(
      static_cast<std::size_t>(n_rows) * sizeof(std::int32_t), persistent, persistent_mr);
  }
  // FLOAT16 and INT8 lists need the FP32 rows for the bounded search to re-score against. Without
  // them FLOAT16 lists could only answer from rounded rows and INT8 lists not at all. They read
  // the pin in place, through row_ids, unless its rows are not the ones searched (cosine lists hold
  // unit rows) or it cannot be mapped; then they keep their own copy in layout order on the host,
  // and a build that cannot hold it is refused. SIRIUS_VSS_EXACT_COPY=1 asks for the copy anyway.
  std::vector<std::byte*> exact_ptrs;
  std::optional<rmm::device_uvector<unsigned int>> max_norm_bits;
  auto const exact_rows_per_block =
    host_mr != nullptr
      ? static_cast<std::int64_t>(block_bytes / (static_cast<std::size_t>(dim) * sizeof(float)))
      : std::int64_t{0};
  if (early_pin) {
    lists.exact_pin = std::move(early_pin);
  } else if ((encoding == list_encoding::float16 || encoding == list_encoding::int8) &&
             !unit_rows && !exact_copy_asked()) {
    lists.exact_pin = map_pinned_rows(*c.pin, req.column, dim, *c.space, persistent, persistent_mr);
    if (lists.exact_pin && lists.exact_pin->n_rows() != n_rows) { lists.exact_pin.reset(); }
  }
  if ((encoding == list_encoding::float16 || encoding == list_encoding::int8) && !lists.exact_pin) {
    auto const exact_bytes =
      static_cast<std::size_t>(n_rows) * static_cast<std::size_t>(dim) * sizeof(float);
    auto const refuse = [&](std::string const& why) {
      throw duckdb::InvalidInputException(
        fn + ": storage => '" + (encoding == list_encoding::int8 ? "int8" : "float16") +
        "' keeps an FP32 copy of the column on the host for exact answers (" +
        std::to_string(exact_bytes) + " bytes), and " + why +
        "; raise the host memory capacity or use storage => 'float32'");
    };
    if (exact_rows_per_block <= 0) { refuse("the host memory pool cannot hold one row of it"); }
    auto const n_blocks = (n_rows + exact_rows_per_block - 1) / exact_rows_per_block;
    try {
      lists.exact_vectors =
        host_mr->allocate_multiple_blocks(static_cast<std::size_t>(n_blocks) * block_bytes);
    } catch (std::exception const& e) {
      refuse(std::string{"the host memory pool could not allocate it ("} + e.what() + ")");
    }
    if (lists.exact_vectors) {
      lists.exact_rows_per_block = exact_rows_per_block;
      std::vector<float const*> device_ptrs;
      exact_ptrs.reserve(static_cast<std::size_t>(n_blocks));
      device_ptrs.reserve(static_cast<std::size_t>(n_blocks));
      for (std::int64_t b = 0; b < n_blocks; ++b) {
        auto* host = lists.exact_vectors->at(static_cast<std::size_t>(b)).data();
        void* dev  = nullptr;
        CUDF_CUDA_TRY(cudaHostGetDevicePointer(&dev, host, 0));
        exact_ptrs.push_back(host);
        device_ptrs.push_back(static_cast<float const*>(dev));
      }
      lists.exact_blocks = std::make_unique<rmm::device_buffer>(
        device_ptrs.size() * sizeof(float const*), persistent, persistent_mr);
      CUDF_CUDA_TRY(cudaMemcpyAsync(lists.exact_blocks->data(),
                                    device_ptrs.data(),
                                    device_ptrs.size() * sizeof(float const*),
                                    cudaMemcpyHostToDevice,
                                    persistent.value()));
    }
  }
  if (lists.has_exact_rows()) {
    lists.row_sq_f32 = std::make_unique<rmm::device_buffer>(
      static_cast<std::size_t>(n_rows) * sizeof(float), persistent, persistent_mr);
    max_norm_bits.emplace(1, persistent, persistent_mr);
    CUDF_CUDA_TRY(
      cudaMemsetAsync(max_norm_bits->data(), 0, sizeof(unsigned int), persistent.value()));
  }
  lists.list_offsets = std::make_unique<rmm::device_buffer>(
    static_cast<std::size_t>(lists.chunk_rows + 1) * sizeof(std::int32_t),
    persistent,
    persistent_mr);
  fill_list_offsets(
    static_cast<std::int32_t*>(lists.list_offsets->data()), lists.chunk_rows, dim, persistent);
  persistent.synchronize();

  std::optional<rmm::device_uvector<unsigned int>> code_error_bits, half_error_bits;
  if (encoding == list_encoding::float16 && lists.has_exact_rows()) {
    half_error_bits.emplace(1, stream, mr);
    CUDF_CUDA_TRY(
      cudaMemsetAsync(half_error_bits->data(), 0, sizeof(unsigned int), stream.value()));
  }
  if (encoding == list_encoding::int8) {
    lists.code_scale  = code_scale;
    lists.code_offset = std::make_unique<rmm::device_buffer>(
      code_offset_host.size() * sizeof(float), persistent, persistent_mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(lists.code_offset->data(),
                                  code_offset_host.data(),
                                  code_offset_host.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  persistent.value()));
    code_error_bits.emplace(1, stream, mr);
    CUDF_CUDA_TRY(
      cudaMemsetAsync(code_error_bits->data(), 0, sizeof(unsigned int), stream.value()));
    persistent.synchronize();
  }

  phase("allocate");

  // Pass 2: stage each chunk again and move its rows into the lists. A list's rows keep pin
  // order, so the rows one chunk contributes to a list are one contiguous run of that list, and
  // the chunk is grouped by cluster on the device so that they are one run of the chunk too.
  // What the lists keep on the device takes the grouped chunk in one scatter. What they keep on
  // the host takes it in one transfer to pinned staging and a host copy per run, made while the
  // device works on the next chunk. A device copy per run costs microseconds whatever its size,
  // which grows with the cluster count and was most of the build at 16,384 clusters; writing
  // rows through mapped host memory from a kernel ran at ~5 GB/s, one scattered PCIe write per
  // row.
  {
    struct run {
      std::int64_t first;  // position in the grouped chunk
      std::int64_t dest;   // first layout row
      std::int64_t rows;
    };
    auto const row_bytes_v     = lists.row_bytes();
    auto const exact_row_bytes = static_cast<std::size_t>(dim) * sizeof(float);
    // Where a byte of the host-resident layout lives, and how many bytes follow it in its block.
    using piece         = std::pair<std::byte*, std::size_t>;
    auto const exact_at = [&](std::size_t at) -> piece {
      auto const row      = static_cast<std::int64_t>(at / exact_row_bytes);
      auto const in_block = static_cast<std::size_t>(row % exact_rows_per_block) * exact_row_bytes +
                            at % exact_row_bytes;
      return {exact_ptrs[static_cast<std::size_t>(row / exact_rows_per_block)] + in_block,
              static_cast<std::size_t>(exact_rows_per_block) * exact_row_bytes - in_block};
    };
    auto const chunk_at = [&](std::size_t at) -> piece {
      auto const chunk_bytes = static_cast<std::size_t>(lists.chunk_rows) * row_bytes_v;
      auto const chunk       = at / chunk_bytes;
      auto const in_chunk    = at % chunk_bytes;
      auto const in_block    = in_chunk % block_bytes;
      auto const this_chunk =
        static_cast<std::size_t>(lists.rows_in_chunk(static_cast<std::int64_t>(chunk))) *
        row_bytes_v;
      return {host_chunks[chunk]->at(in_chunk / block_bytes).data() + in_block,
              std::min(block_bytes - in_block, this_chunk - in_chunk)};
    };

    // One grouped chunk on its way to the host. `copies` is declared last so that an unwinding
    // build waits for them before the blocks they read are returned.
    struct staging {
      cucascade::memory::fixed_multiple_blocks_allocation blocks;
      std::vector<std::future<void>> copies;
      void wait()
      {
        for (auto& c : copies) {
          c.get();
        }
        copies.clear();
      }
    };
    auto const download = [&](staging& s, std::byte const* src, std::size_t bytes) {
      if (!s.blocks || s.blocks->size_bytes() < bytes) {
        s.blocks.reset();
        s.blocks = host_mr->allocate_multiple_blocks(bytes);
      }
      for (std::size_t at = 0; at < bytes; at += block_bytes) {
        CUDF_CUDA_TRY(cudaMemcpyAsync(s.blocks->at(at / block_bytes).data(),
                                      src + at,
                                      std::min(block_bytes, bytes - at),
                                      cudaMemcpyDeviceToHost,
                                      stream.value()));
      }
    };
    auto const copy_threads =
      static_cast<std::size_t>(std::clamp<unsigned>(std::thread::hardware_concurrency(), 1u, 4u));
    // Each run of the chunk staged in @p s to where @p at puts it.
    auto const copy_out = [&](staging& s,
                              std::shared_ptr<std::vector<run> const> const& runs,
                              std::size_t width,
                              auto const& at) {
      auto const* blocks = s.blocks.get();
      auto const n_tasks = std::min(copy_threads, runs->size());
      for (std::size_t t = 0; t < n_tasks; ++t) {
        auto const lo = runs->size() * t / n_tasks;
        auto const hi = runs->size() * (t + 1) / n_tasks;
        s.copies.push_back(
          std::async(std::launch::async, [runs, lo, hi, width, at, blocks, block_bytes] {
            for (auto i = lo; i < hi; ++i) {
              auto const& g = (*runs)[i];
              auto from     = static_cast<std::size_t>(g.first) * width;
              auto to       = static_cast<std::size_t>(g.dest) * width;
              auto left     = static_cast<std::size_t>(g.rows) * width;
              while (left > 0) {
                auto const in_block    = from % block_bytes;
                auto const [out, room] = at(to);
                auto const now         = std::min({left, block_bytes - in_block, room});
                std::memcpy(out, blocks->at(from / block_bytes).data() + in_block, now);
                from += now;
                to += now;
                left -= now;
              }
            }
          }));
      }
    };
    // Two of each, used in turn: the copies out of one overlap the transfer into the other.
    std::array<staging, 2> exact_stages, row_stages;

    std::vector<std::int64_t> order, grouped_dest;
    std::vector<std::int64_t> count(static_cast<std::size_t>(n_clusters));
    pinned_chunk_walk walk{*c.pin, req.column, *c.space, stream, telemetry_info};
    std::int64_t base = 0;
    for (std::size_t i = 0, turn = 0; i < n_chunks; ++i) {
      auto staged     = walk.take(i);
      auto const rows = static_cast<std::int64_t>(staged.view.size());
      if (rows == 0) { continue; }
      auto const vectors       = list_column_as_dataset_view(staged.view, dim);
      auto const* chunk_labels = labels.data() + base;

      std::fill(count.begin(), count.end(), 0);
      for (std::int64_t r = 0; r < rows; ++r) {
        ++count[static_cast<std::size_t>(chunk_labels[r])];
      }
      auto const runs_owner = std::make_shared<std::vector<run>>();
      auto& runs            = *runs_owner;
      std::int64_t pos      = 0;
      for (std::int64_t k = 0; k < n_clusters; ++k) {
        auto const n = count[static_cast<std::size_t>(k)];
        if (n == 0) { continue; }
        runs.push_back(run{pos, -1, n});
        count[static_cast<std::size_t>(k)] = pos;  // now the next slot of cluster k
        pos += n;
      }
      order.resize(static_cast<std::size_t>(rows));
      grouped_dest.resize(static_cast<std::size_t>(rows));
      for (std::int64_t r = 0; r < rows; ++r) {
        auto const slot =
          static_cast<std::size_t>(count[static_cast<std::size_t>(chunk_labels[r])]++);
        order[slot]        = r;
        grouped_dest[slot] = dest[static_cast<std::size_t>(base + r)];
      }
      for (auto& g : runs) {
        g.dest = grouped_dest[static_cast<std::size_t>(g.first)];
      }

      auto& exact_stage = exact_stages[turn % 2];
      auto& row_stage   = row_stages[turn % 2];
      ++turn;
      // The chunk before last was staged here; its copies have had a whole chunk to finish.
      exact_stage.wait();
      row_stage.wait();

      rmm::device_uvector<std::int64_t> order_d(static_cast<std::size_t>(rows), stream, mr);
      rmm::device_uvector<std::int64_t> chunk_dest(static_cast<std::size_t>(rows), stream, mr);
      rmm::device_uvector<std::int64_t> grouped_dest_d(static_cast<std::size_t>(rows), stream, mr);
      rmm::device_uvector<float> grouped(static_cast<std::size_t>(rows * dim), stream, mr);
      CUDF_CUDA_TRY(cudaMemcpyAsync(order_d.data(),
                                    order.data(),
                                    static_cast<std::size_t>(rows) * sizeof(std::int64_t),
                                    cudaMemcpyHostToDevice,
                                    stream.value()));
      CUDF_CUDA_TRY(cudaMemcpyAsync(chunk_dest.data(),
                                    dest.data() + base,
                                    static_cast<std::size_t>(rows) * sizeof(std::int64_t),
                                    cudaMemcpyHostToDevice,
                                    stream.value()));
      CUDF_CUDA_TRY(cudaMemcpyAsync(grouped_dest_d.data(),
                                    grouped_dest.data(),
                                    static_cast<std::size_t>(rows) * sizeof(std::int64_t),
                                    cudaMemcpyHostToDevice,
                                    stream.value()));
      // Per-row values of the grouped chunk into a device-resident part of the layout.
      auto const scatter = [&](void const* src, std::size_t width, void* out) {
        scatter_rows(static_cast<std::byte const*>(src),
                     static_cast<std::int64_t>(width),
                     grouped_dest_d.data(),
                     rows,
                     static_cast<std::byte*>(out),
                     stream);
      };
      gather_rows(vectors.data_handle(), dim, order_d.data(), rows, grouped.data(), stream);
      if (unit_rows) { normalize_rows(grouped.data(), rows, dim, grouped.data(), stream); }
      // The copies below read the grouped rows in the list encoding.
      std::optional<rmm::device_uvector<std::uint16_t>> grouped_f16;
      std::optional<rmm::device_uvector<std::int8_t>> grouped_i8;
      std::optional<rmm::device_uvector<std::int32_t>> grouped_sq;
      auto const* grouped_bytes = reinterpret_cast<std::byte const*>(grouped.data());
      if (encoding == list_encoding::float16) {
        grouped_f16.emplace(static_cast<std::size_t>(rows * dim), stream, mr);
        narrow_to_float16(grouped.data(), rows * dim, grouped_f16->data(), stream);
        grouped_bytes = reinterpret_cast<std::byte const*>(grouped_f16->data());
      }
      std::optional<rmm::device_uvector<float>> grouped_sqf;
      if (lists.has_exact_rows()) {
        grouped_sqf.emplace(static_cast<std::size_t>(rows), stream, mr);
        if (grouped_f16) {
          // The FP16 search compares rounded rows, so it needs their norms and how far each is
          // from the row it stands for.
          half_rows_norms(grouped.data(),
                          grouped_f16->data(),
                          rows,
                          dim,
                          grouped_sqf->data(),
                          nullptr,
                          half_error_bits->data(),
                          max_norm_bits->data(),
                          stream);
        } else {
          float_row_sq_norms(
            grouped.data(), rows, dim, grouped_sqf->data(), max_norm_bits->data(), stream);
        }
        scatter(grouped_sqf->data(), sizeof(float), lists.row_sq_f32->data());
        if (lists.exact_vectors) {
          download(exact_stage,
                   reinterpret_cast<std::byte const*>(grouped.data()),
                   static_cast<std::size_t>(rows) * exact_row_bytes);
        }
      }
      if (encoding == list_encoding::int8) {
        grouped_i8.emplace(static_cast<std::size_t>(rows * dim), stream, mr);
        grouped_sq.emplace(static_cast<std::size_t>(rows), stream, mr);
        quantize_rows_int8(grouped.data(),
                           rows,
                           dim,
                           static_cast<float const*>(lists.code_offset->data()),
                           lists.code_scale,
                           grouped_i8->data(),
                           nullptr,
                           code_error_bits->data(),
                           stream);
        int8_row_sq_norms(grouped_i8->data(), rows, dim, grouped_sq->data(), stream);
        grouped_bytes = reinterpret_cast<std::byte const*>(grouped_i8->data());
        scatter(grouped_sq->data(), sizeof(std::int32_t), lists.row_sq->data());
      }
      if (encoding == list_encoding::uint8) {
        grouped_i8.emplace(static_cast<std::size_t>(rows * dim), stream, mr);
        grouped_sq.emplace(static_cast<std::size_t>(rows), stream, mr);
        narrow_to_shifted_int8(grouped.data(), rows * dim, grouped_i8->data(), stream);
        int8_row_sq_norms(grouped_i8->data(), rows, dim, grouped_sq->data(), stream);
        grouped_bytes = reinterpret_cast<std::byte const*>(grouped_i8->data());
        scatter(grouped_sq->data(), sizeof(std::int32_t), lists.row_sq->data());
      }
      scatter_row_ids(
        chunk_dest.data(), rows, base, static_cast<std::int64_t*>(lists.row_ids->data()), stream);
      if (on_device) {
        scatter(grouped_bytes, row_bytes_v, lists.device_vectors->data());
      } else {
        download(row_stage, grouped_bytes, static_cast<std::size_t>(rows) * row_bytes_v);
      }
      walk.prefetch_after(i);
      // The staged chunk, the grouped rows and the pageable sources of the copies above are
      // released or reused once this iteration ends.
      stream.synchronize();
      if (lists.exact_vectors) { copy_out(exact_stage, runs_owner, exact_row_bytes, exact_at); }
      if (!on_device) { copy_out(row_stage, runs_owner, row_bytes_v, chunk_at); }
      base += rows;
    }
    for (auto* stages : {&exact_stages, &row_stages}) {
      for (auto& s : *stages) {
        s.wait();
      }
    }
  }
  labels.clear();
  labels.shrink_to_fit();
  if (!on_device) {
    // One flat column per chunk: the H2D converter reconstructs it from this metadata when the
    // engine stages the chunk, and the join reads the same bytes directly while a chunk is on
    // the host.
    auto const type_id = encoding == list_encoding::float32   ? cudf::type_id::FLOAT32
                         : encoding == list_encoding::float16 ? cudf::type_id::INT16
                                                              : cudf::type_id::INT8;
    lists.chunk_batches.reserve(host_chunks.size());
    for (std::size_t ch = 0; ch < host_chunks.size(); ++ch) {
      auto const rows_c = lists.rows_in_chunk(static_cast<std::int64_t>(ch));
      auto const bytes  = static_cast<std::size_t>(rows_c) * lists.row_bytes();
      cucascade::memory::column_metadata meta{};
      meta.type_id     = static_cast<std::int32_t>(type_id);
      meta.num_rows    = static_cast<std::int32_t>(rows_c * dim);
      meta.null_count  = 0;
      meta.has_data    = true;
      meta.data_offset = 0;
      meta.data_size   = bytes;
      auto table       = cucascade::memory::host_table_allocation::create(
        std::move(host_chunks[ch]), std::vector<cucascade::memory::column_metadata>{meta}, bytes);
      lists.chunk_batches.push_back(cucascade::data_batch::make(
        get_next_batch_id(),
        std::make_unique<cucascade::host_data_representation>(
          std::move(table), const_cast<cucascade::memory::memory_space*>(c.host_space))));
    }
  }
  if (max_norm_bits) {
    unsigned int bits = 0;
    CUDF_CUDA_TRY(cudaMemcpy(&bits, max_norm_bits->data(), sizeof(bits), cudaMemcpyDeviceToHost));
    float max_sq = 0;
    std::memcpy(&max_sq, &bits, sizeof(max_sq));
    lists.max_row_norm = std::sqrt(max_sq);
    // A component past half's range would have become infinity; a norm this large is the only
    // way to have one.
    if (encoding == list_encoding::float16 && !(lists.max_row_norm < 65504.f)) {
      throw duckdb::InvalidInputException(fn + ": a row norm of " +
                                          std::to_string(lists.max_row_norm) +
                                          " is outside float16's range; use storage => 'float32'");
    }
  }

  if (half_error_bits) {
    unsigned int bits = 0;
    CUDF_CUDA_TRY(cudaMemcpy(&bits, half_error_bits->data(), sizeof(bits), cudaMemcpyDeviceToHost));
    std::memcpy(&lists.half_error, &bits, sizeof(bits));
  }
  if (code_error_bits) {
    unsigned int bits = 0;
    CUDF_CUDA_TRY(cudaMemcpy(&bits, code_error_bits->data(), sizeof(bits), cudaMemcpyDeviceToHost));
    std::memcpy(&lists.code_error, &bits, sizeof(bits));
  }

  phase("pass 2 scatter");

  cluster_lists_result result;
  result.n_rows     = n_rows;
  result.n_clusters = n_clusters;
  result.tier       = on_device ? "gpu" : "host";
  result.encoding   = encoding == list_encoding::uint8     ? "uint8"
                      : encoding == list_encoding::float16 ? "float16"
                      : encoding == list_encoding::int8    ? "int8"
                                                           : "float32";
  result.exact_rows = lists.exact_pin ? "pin" : lists.exact_vectors ? "copy" : "none";
  result.min_list   = std::numeric_limits<std::int64_t>::max();
  for (std::int64_t k = 0; k < n_clusters; ++k) {
    auto const size =
      lists.offsets[static_cast<std::size_t>(k) + 1] - lists.offsets[static_cast<std::size_t>(k)];
    result.min_list = std::min(result.min_list, size);
    result.max_list = std::max(result.max_list, size);
    if (size == 0) { ++result.empty_lists; }
  }

  index_metadata meta;
  meta.kind           = index_kind::cluster_lists;
  meta.table_name     = req.table;
  meta.column_name    = req.column;
  meta.dim            = dim;
  meta.num_rows       = n_rows;
  meta.n_lists        = n_clusters;
  meta.metric         = entry->meta.metric;
  meta.resident_bytes = reserved_bytes;
  index_cache.insert(lists_key(req.clustering),
                     std::move(meta),
                     make_cuvs_index(std::move(lists)),
                     rmm::cuda_stream{});
  return result;
}

cucascade::read_only_data_batch lock_host_list_chunk(const cluster_lists& lists,
                                                     std::size_t chunk,
                                                     rmm::cuda_stream_view stream)
{
  auto const& batch = lists.chunk_batches.at(chunk);
  {
    auto ro = batch->to_read_only();
    if (ro.get_current_tier() == cucascade::memory::Tier::HOST) { return ro; }
    std::ignore = cucascade::data_batch::to_idle(std::move(ro));
  }
  // The downgrade executor moved it on; the engine brings it back to the lists' host space.
  auto prepared =
    pipeline::lock_and_prepare_batch(batch, lists.host_space, ::cuda::stream_ref{stream.value()});
  if (!prepared) {
    throw std::runtime_error("[vss] cluster list chunk " + std::to_string(chunk) +
                             " could not be brought back to host memory");
  }
  return std::visit([](auto& r) { return std::move(r.ro_lock); }, *prepared);
}

void copy_host_list_rows(const cluster_lists& lists,
                         const cucascade::read_only_data_batch& ro,
                         std::int64_t row0,
                         std::int64_t rows,
                         std::byte* dst,
                         rmm::cuda_stream_view stream)
{
  auto const* data = ro.get_data();
  if (data == nullptr) { throw std::runtime_error("[vss] cluster list chunk has no data"); }
  auto const& table      = data->cast<cucascade::host_data_representation>().get_host_table();
  auto const& alloc      = *table->allocation;
  auto const block_bytes = alloc.block_size();
  auto offset            = static_cast<std::size_t>(row0) * lists.row_bytes();
  auto remaining         = static_cast<std::size_t>(rows) * lists.row_bytes();
  while (remaining > 0) {
    auto const block    = offset / block_bytes;
    auto const in_block = offset % block_bytes;
    auto const bytes    = std::min(remaining, block_bytes - in_block);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      dst, alloc.at(block).data() + in_block, bytes, cudaMemcpyHostToDevice, stream.value()));
    dst += bytes;
    offset += bytes;
    remaining -= bytes;
  }
}

std::unique_ptr<cucascade::host_data_representation> run_kmeans_centroids(
  duckdb::SiriusContext& ctx, const std::string& clustering)
{
  static const std::string fn = "sirius_kmeans_centroids";

  auto& memory_manager = ctx.get_memory_manager();
  auto gpu_spaces      = memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  auto host_spaces     = memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  if (gpu_spaces.empty() || host_spaces.empty()) {
    throw duckdb::InvalidInputException(fn + ": no GPU or HOST memory space available");
  }
  auto* space = const_cast<cucascade::memory::memory_space*>(gpu_spaces.front());
  device_context_guard device_guard{space->get_device_id()};
  rmm::cuda_stream stream_owner;
  auto stream   = stream_owner.view();
  auto const mr = space->get_default_allocator();

  const auto* entry = find_clustering_entry(ctx, clustering);
  if (entry == nullptr) {
    throw duckdb::InvalidInputException(fn + ": no clustering named '" + clustering + "'");
  }
  const auto* centroids = find_clustering_centroids(ctx, clustering);
  if (centroids == nullptr) {
    throw duckdb::InvalidInputException(fn + ": clustering '" + clustering +
                                        "' holds no centroids");
  }

  auto const dim    = entry->meta.dim;
  auto const values = cudf::lists_column_view(centroids->view()).child();
  auto const total  = values.size();

  cudf::numeric_scalar<std::int32_t> const zero(0, true, stream);
  cudf::numeric_scalar<std::int32_t> const one(1, true, stream);
  cudf::numeric_scalar<std::int32_t> const width(static_cast<std::int32_t>(dim), true, stream);

  // Position in the flattened [n_clusters, dim] buffer splits into its two coordinates by
  // division and remainder, which is what puts the centroids in long form.
  auto const positions = cudf::sequence(total, zero, one, stream, mr);
  auto cluster_ids     = cudf::binary_operation(positions->view(),
                                            width,
                                            cudf::binary_operator::DIV,
                                            cudf::data_type{cudf::type_id::INT32},
                                            stream,
                                            mr);
  auto dim_index       = cudf::binary_operation(positions->view(),
                                          width,
                                          cudf::binary_operator::MOD,
                                          cudf::data_type{cudf::type_id::INT32},
                                          stream,
                                          mr);

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(cluster_ids));
  columns.push_back(std::move(dim_index));
  columns.push_back(std::make_unique<cudf::column>(values, stream, mr));
  auto table = std::make_unique<cudf::table>(std::move(columns));

  return vss_table_to_host(*space, *host_spaces.front(), stream, std::move(table));
}

}  // namespace sirius::vss
