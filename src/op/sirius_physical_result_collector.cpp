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

// sirius

#include <nvtx3/nvtx3.hpp>

#include <cudf/cudf_utils.hpp>
#include <data/sirius_converter_registry.hpp>
#include <helper/type_conversions.hpp>
#include <op/result/host_table_chunk_reader.hpp>
#include <op/sirius_physical_result_collector.hpp>
#include <pipeline/sirius_meta_pipeline.hpp>
#include <pipeline/sirius_pipeline.hpp>
#include <sirius_interface.hpp>

// cucascade
#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>

// sirius exceptions
#include "sirius/exception.hpp"

// cudf
#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/utilities/span.hpp>
#include <rmm/device_buffer.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

// duckdb
#include <duckdb/main/materialized_query_result.hpp>
#include <duckdb/main/prepared_statement_data.hpp>

// nixl exchange support
#include <last_gpu_buffers.hpp>

// standard library
#include <algorithm>
#include <cassert>
#include <optional>

namespace sirius {
namespace op {

namespace {

struct normalized_exchange_view {
  std::vector<std::unique_ptr<cudf::column>> owned_columns;
  std::vector<cudf::column_view> columns;

  cudf::table_view view() const { return cudf::table_view(columns); }
};

struct prepared_exchange_view {
  normalized_exchange_view normalized;
  bool projection_already_applied = false;
};

cudf::table_view apply_exchange_projection(
  cudf::table_view input,
  const std::vector<int>& projection_indices);

std::optional<normalized_exchange_view> build_normalized_exchange_view(
  cudf::table_view input,
  const duckdb::vector<sirius::logical_type>& expected_types,
  rmm::cuda_stream_view stream)
{
  if (input.num_columns() != static_cast<cudf::size_type>(expected_types.size())) {
    SIRIUS_LOG_WARN(
      "[result_collector] exchange capture disabled: output column count {} != expected {}",
      input.num_columns(),
      expected_types.size());
    return std::nullopt;
  }

  normalized_exchange_view normalized;
  normalized.columns.reserve(input.num_columns());

  try {
    for (cudf::size_type col_idx = 0; col_idx < input.num_columns(); ++col_idx) {
      auto col_view = input.column(col_idx);
      auto expected_type = sirius::get_cudf_type(expected_types[col_idx]);
      if (col_view.type() == expected_type) {
        normalized.columns.push_back(col_view);
        continue;
      }

      SIRIUS_LOG_INFO(
        "[result_collector] normalizing exchange column {} from cudf type {} scale {} to {} scale {}",
        col_idx,
        static_cast<int>(col_view.type().id()),
        static_cast<int>(col_view.type().scale()),
        static_cast<int>(expected_type.id()),
        static_cast<int>(expected_type.scale()));
      auto casted = cudf::cast(col_view, expected_type, stream);
      normalized.columns.push_back(casted->view());
      normalized.owned_columns.push_back(std::move(casted));
    }
  } catch (const std::exception& e) {
    SIRIUS_LOG_WARN(
      "[result_collector] exchange capture disabled: failed to normalize output types: {}",
      e.what());
    return std::nullopt;
  }

  return normalized;
}

std::optional<prepared_exchange_view> build_prepared_exchange_view(
  cudf::table_view input,
  const duckdb::vector<sirius::logical_type>& expected_types,
  const std::vector<int>& projection_indices,
  rmm::cuda_stream_view stream)
{
  auto projected_input = apply_exchange_projection(input, projection_indices);
  if (projected_input.num_columns() == static_cast<cudf::size_type>(expected_types.size())) {
    if (!projection_indices.empty()) {
      SIRIUS_LOG_INFO(
        "[result_collector] applying exchange sender projection before normalization: input_cols={} projected_cols={} expected_cols={}",
        input.num_columns(),
        projected_input.num_columns(),
        expected_types.size());
    }
    if (auto normalized =
          build_normalized_exchange_view(projected_input, expected_types, stream)) {
      return prepared_exchange_view{std::move(*normalized), true};
    }
    return std::nullopt;
  }

  if (auto normalized = build_normalized_exchange_view(input, expected_types, stream)) {
    return prepared_exchange_view{std::move(*normalized), false};
  }
  return std::nullopt;
}

cudf::table_view apply_exchange_projection(
  cudf::table_view input,
  const std::vector<int>& projection_indices)
{
  if (projection_indices.empty()) {
    return input;
  }

  std::vector<cudf::column_view> projected_columns;
  projected_columns.reserve(projection_indices.size());
  for (auto idx : projection_indices) {
    if (idx < 0 || idx >= input.num_columns()) {
      throw duckdb::InvalidInputException(
        "[result_collector] exchange capture projection index out of range: {} for {} columns",
        idx,
        input.num_columns());
    }
    projected_columns.push_back(input.column(idx));
  }
  return cudf::table_view(projected_columns);
}

}  // namespace

sirius_physical_result_collector::sirius_physical_result_collector(
  ::sirius::sirius_prepared_statement_data& data)
  : sirius_physical_operator(SiriusPhysicalOperatorType::RESULT_COLLECTOR,
                             {sirius::logical_type::make(sirius::type_id::BOOLEAN)},
                             0),
    statement_type(data.prepared->statement_type),
    properties(data.prepared->properties),
    plan(*data.sirius_physical_plan),
    names(data.prepared->names)
{
  this->types = sirius::from_duckdb_vec(data.prepared->types);
}

std::unique_ptr<operator_data> sirius_physical_result_collector::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_result_collector::execute"};
  return std::make_unique<pipelineable_operator_data>(
    dynamic_cast<const pipelineable_operator_data&>(input_data).get_data_batches());
}

duckdb::vector<duckdb::const_reference<sirius_physical_operator>>
sirius_physical_result_collector::get_children() const
{
  return {plan};
}

void sirius_physical_result_collector::build_pipelines(
  pipeline::sirius_pipeline& current, pipeline::sirius_meta_pipeline& meta_pipeline)
{
  // operator is a sink, build a pipeline
  D_ASSERT(children.empty());

  // single operator: the operator becomes the data source of the current pipeline
  auto& state = meta_pipeline.get_state();
  state.set_pipeline_source(current, *this);

  // we create a new pipeline starting from the child
  auto& child_meta_pipeline = meta_pipeline.create_child_meta_pipeline(current, *this);
  child_meta_pipeline.build(plan);
}

sirius_physical_materialized_collector::sirius_physical_materialized_collector(
  ::sirius::sirius_prepared_statement_data& data, duckdb::ClientContext& client_ctx)
  : sirius_physical_result_collector(data),
    _client_ctx(client_ctx),
    result_collection(
      duckdb::make_uniq<duckdb::ColumnDataCollection>(client_ctx, sirius::to_duckdb_vec(types)))
{
}

duckdb::unique_ptr<duckdb::QueryResult> sirius_physical_materialized_collector::get_result()
{
  auto props = _client_ctx.GetClientProperties();

  std::lock_guard<std::mutex> guard(lock);
  // Return an empty result collection if the result_collection is null (from a move)
  if (!result_collection) {
    result_collection =
      duckdb::make_uniq<duckdb::ColumnDataCollection>(_client_ctx, sirius::to_duckdb_vec(types));
  }

  return duckdb::make_uniq<duckdb::MaterializedQueryResult>(
    statement_type, properties, names, std::move(result_collection), props);
}

void sirius_physical_materialized_collector::sink(const operator_data& input_data,
                                                  rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_materialized_collector::sink"};
  auto& pipelineable_input      = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches     = pipelineable_input.get_data_batches();
  using host_table_chunk_reader = ::sirius::op::result::host_table_chunk_reader;

  if (input_batches.empty()) {
    return;  // todo(kevin) we should handle this case properly
  }

  // Check if GPU buffer retention is requested (for nixl GPU-direct exchange).
  bool should_retain = duckdb::LastGPUBuffers::GetInstance().ShouldRetain();
  auto capture_mode = duckdb::LastGPUBuffers::GetInstance().GetCaptureMode();
  bool exchange_only = capture_mode == duckdb::ExchangeCaptureMode::ExchangeOnly;
  SIRIUS_LOG_INFO("[result_collector] sink called: {} batches, should_retain={}", input_batches.size(), should_retain);

  for (auto const& input_batch : input_batches) {
    auto* data = input_batch->get_data();
    std::shared_ptr<cucascade::data_batch> clone_batch;
    if (!data) {
      throw invalid_input_exception(
        "[GPUPhysicalMaterializedCollector] data_batch has no data representation");
    }
    if (data->get_size_in_bytes() == 0) { continue; }

    SIRIUS_LOG_INFO("[result_collector] batch: tier={} size={}",
                    static_cast<int>(data->get_current_tier()), data->get_size_in_bytes());

    // If data is in GPU tier, convert to HOST tier first
    if (data->get_current_tier() == cucascade::memory::Tier::GPU) {
      auto& gpu_rep = data->cast<cucascade::gpu_table_representation>();
      cudf::table_view view = gpu_rep.get_table().view();
      auto& lgb = duckdb::LastGPUBuffers::GetInstance();
      auto projection_indices = lgb.GetProjectionIndices();
      std::optional<prepared_exchange_view> exchange_view_storage;
      cudf::table_view exchange_view = view;

      if (should_retain) {
        exchange_view_storage =
          build_prepared_exchange_view(view, types, projection_indices, stream);
        if (exchange_view_storage) {
          exchange_view = exchange_view_storage->normalized.view();
          if (exchange_view_storage->projection_already_applied) {
            projection_indices.clear();
          }
        } else {
          SIRIUS_LOG_WARN(
            "[result_collector] retaining disabled for this batch; falling back to CPU exchange");
        }
      }

      if (should_retain && exchange_view_storage) {
        // Keep the source GPU batch alive until exchange transfer completes.
        // The packed exchange path may still depend on buffers originating from
        // this batch after the pipeline releases its local references.
        lgb.RetainData(std::static_pointer_cast<void>(input_batch));
        auto [staging_addr, staging_size] = lgb.GetStagingBuffer();
        auto [part_num, part_cols] = lgb.GetPartitionConfig();

        SIRIUS_LOG_INFO(
          "[result_collector] retain path: part_num={} part_cols={} projection_cols={} staging_addr=0x{:x}",
          part_num,
          part_cols.size(),
          projection_indices.size(),
          staging_addr);

        if (part_num > 0 && !part_cols.empty() && staging_addr != 0) {
            // GPU hash partition + per-partition pack into staging.
            // For a single destination, hash partitioning is a no-op. Bypass the
            // cuDF partitioner entirely and pack the original view as one
            // partition to avoid unnecessary ownership churn on the partitioned
            // table.
            std::unique_ptr<cudf::table> partitioned_table;
            cudf::table_view partitioned_view = exchange_view;
            std::vector<cudf::size_type> offsets{0};
            if (part_num > 1) {
              std::vector<cudf::size_type> col_indices(part_cols.begin(), part_cols.end());
              auto partition_result = cudf::hash_partition(
                  exchange_view, col_indices, part_num, cudf::hash_id::HASH_MURMUR3,
                  cudf::DEFAULT_HASH_SEED, stream);
              partitioned_table = std::move(partition_result.first);
              offsets = std::move(partition_result.second);
              partitioned_view = partitioned_table->view();

              SIRIUS_LOG_INFO("[result_collector] GPU hash_partition: {} partitions, {} rows → offsets: [{}]",
                              part_num, exchange_view.num_rows(),
                              [&]() { std::string s; for (auto o : offsets) { if (!s.empty()) s += ","; s += std::to_string(o); } return s; }());

              // Validate hash_partition output: check first INT32 value and STRING offsets.
              for (cudf::size_type c = 0; c < partitioned_view.num_columns(); c++) {
                auto col = partitioned_view.column(c);
                if (col.type().id() == cudf::type_id::INT32 && col.size() > 0) {
                  int32_t first_val = 0;
                  cudaMemcpy(&first_val, col.data<int32_t>(), 4, cudaMemcpyDeviceToHost);
                  SIRIUS_LOG_INFO("[result_collector] hash_part col {} INT32 first_val={}", c, first_val);
                }
                if (col.type().id() == cudf::type_id::STRING && col.num_children() > 0 && col.size() > 0) {
                  auto child = col.child(0);
                  int32_t first_off = 0, last_off = 0;
                  cudaMemcpy(&first_off, child.data<int32_t>(), 4, cudaMemcpyDeviceToHost);
                  cudaMemcpy(&last_off, child.data<int32_t>() + child.size() - 1, 4, cudaMemcpyDeviceToHost);
                  SIRIUS_LOG_INFO("[result_collector] hash_part col {} STRING first_off={} last_off={} offset={}",
                                  c, first_off, last_off, col.offset());
                }
              }
            } else {
              SIRIUS_LOG_INFO("[result_collector] single-destination exchange: bypassing hash_partition for {} rows",
                              exchange_view.num_rows());
            }

            auto projected_partitioned_view =
              apply_exchange_projection(partitioned_view, projection_indices);

            std::vector<duckdb::PackedPartition> packed_parts;

            // Estimate total staging space needed: full table packed size as upper bound.
            // Reserve atomically to prevent concurrent sink() calls from overlapping.
            // Compute exact per-partition packed sizes upfront to get accurate staging reservation.
            // The full-table estimate is unreliable because per-partition alignment padding differs.
            size_t exact_total = 0;
            for (int i = 0; i < part_num; i++) {
              auto start = offsets[i];
              auto end =
                (i + 1 < part_num) ? offsets[i + 1] : projected_partitioned_view.num_rows();
              if (end <= start) continue;
              auto pslice = cudf::slice(projected_partitioned_view, {start, end});
              auto psz = cudf::chunked_pack::create(pslice[0], 1UL << 20, stream)->get_total_contiguous_size();
              exact_total += (psz + 255) & ~255UL;  // 256-byte align per partition
            }
            size_t aligned_estimate = exact_total;
            size_t staging_offset = lgb.ReserveStagingRegion(aligned_estimate, staging_size);
            size_t actual_total_packed = 0; // Track actual vs estimate

            for (int i = 0; i < part_num; i++) {
              auto start = offsets[i];
              auto end =
                (i + 1 < part_num) ? offsets[i + 1] : projected_partitioned_view.num_rows();
              auto num_rows = end - start;
              if (num_rows == 0) {
                { duckdb::PackedPartition pp; pp.staging_offset = staging_offset; pp.packed_size = 0; pp.num_rows = 0; packed_parts.push_back(std::move(pp)); }
                continue;
              }
              auto slice = cudf::slice(projected_partitioned_view, {start, end});
              auto total = cudf::chunked_pack::create(slice[0], 1UL << 20, stream)->get_total_contiguous_size();
              if (staging_offset + total > staging_size) {
                // Overflow: pack into separate rmm::device_buffer instead of dropping data.
                SIRIUS_LOG_WARN("[result_collector] partition {} ({} bytes) overflows staging, using rmm::device_buffer", i, total);
                auto packed = cudf::pack(slice[0]);
                auto addr = reinterpret_cast<uintptr_t>(packed.gpu_data->data());
                auto size = packed.gpu_data->size();
                duckdb::PackedPartition pp;
                pp.staging_offset = 0;
                pp.packed_size = size;
                pp.metadata = std::move(packed.metadata);
                pp.num_rows = static_cast<int32_t>(num_rows);
                pp.overflow_gpu_addr = addr;
                pp.overflow_data = std::make_shared<rmm::device_buffer>(std::move(*packed.gpu_data));
                packed_parts.push_back(std::move(pp));
              } else {
                // Use cudf::pack (not chunked_pack) — chunked_pack has a bug where
                // STRING chars data overwrites fixed-width column positions for specific
                // hash_partition + slice outputs. cudf::pack produces correct output.
                // Synchronize the stream to ensure the hash_partition + slice is complete.
                stream.synchronize();
                auto packed = cudf::pack(slice[0], stream);
                // Synchronize again to ensure pack is complete before D2D copy.
                stream.synchronize();
                auto total = packed.gpu_data->size();
                // D2D copy to staging — use the SAME stream to maintain ordering.
                cudaMemcpyAsync(reinterpret_cast<void*>(staging_addr + staging_offset),
                               packed.gpu_data->data(), total,
                               cudaMemcpyDeviceToDevice, stream.value());
                stream.synchronize();
                auto md = std::move(packed.metadata);
                SIRIUS_LOG_INFO("[result_collector] partition {}: {} rows, {} bytes at staging+{}",
                                i, num_rows, total, staging_offset);
                // Validate: read first 4 bytes of packed buffer — should be INT32 if col 0 is INT32
                {
                  int32_t packed_first4 = 0;
                  cudaMemcpy(&packed_first4, reinterpret_cast<void*>(staging_addr + staging_offset),
                             4, cudaMemcpyDeviceToHost);
                  auto first_col_type = slice[0].column(0).type().id();
                  bool looks_ascii = (packed_first4 > 0x20000000 && packed_first4 < 0x7f7f7f7f);
                  SIRIUS_LOG_INFO("[result_collector] partition {} packed first4={} (0x{:08x}) col0_type={} ascii={}",
                                  i, packed_first4, static_cast<uint32_t>(packed_first4),
                                  static_cast<int>(first_col_type), looks_ascii);
                  if (looks_ascii && cudf::is_fixed_width(cudf::data_type{first_col_type})) {
                    // SENDER-SIDE CORRUPTION DETECTED! Also pack with cudf::pack for comparison.
                    SIRIUS_LOG_ERROR("[result_collector] SENDER CORRUPTION: partition {} has ASCII at INT32 offset 0!", i);
                    auto ref_packed = cudf::pack(slice[0]);
                    int32_t ref_first4 = 0;
                    cudaMemcpy(&ref_first4, ref_packed.gpu_data->data(), 4, cudaMemcpyDeviceToHost);
                    SIRIUS_LOG_ERROR("[result_collector] cudf::pack first4={} (0x{:08x}) — {}",
                                    ref_first4, static_cast<uint32_t>(ref_first4),
                                    (ref_first4 > 0x20000000 && ref_first4 < 0x7f7f7f7f) ? "ALSO ASCII" : "OK (INT32)");
                    // Dump the pre-pack slice info
                    for (cudf::size_type c = 0; c < slice[0].num_columns() && c < 3; c++) {
                      auto col = slice[0].column(c);
                      SIRIUS_LOG_ERROR("[result_collector] slice col {} type={} rows={} offset={} data=0x{:x}",
                                      c, static_cast<int>(col.type().id()), col.size(), col.offset(),
                                      reinterpret_cast<uintptr_t>(col.head<uint8_t>()));
                    }
                  }
                }
                duckdb::PackedPartition pp;
                pp.staging_offset = staging_offset;
                pp.packed_size = total;
                pp.metadata = std::move(md);
                pp.num_rows = static_cast<int32_t>(num_rows);
                packed_parts.push_back(std::move(pp));
                actual_total_packed += (total + 255) & ~255UL;
                staging_offset += (total + 255) & ~255UL;  // 256-byte align
              }
            }
            if (actual_total_packed > aligned_estimate) {
              SIRIUS_LOG_ERROR("[result_collector] OVERFLOW: actual_total_packed={} > estimated={} (diff={})",
                              actual_total_packed, aligned_estimate, actual_total_packed - aligned_estimate);
            } else {
              SIRIUS_LOG_INFO("[result_collector] staging estimate OK: actual={} estimated={} (slack={})",
                              actual_total_packed, aligned_estimate, aligned_estimate - actual_total_packed);
            }
            lgb.AccumulatePackedPartitions(std::move(packed_parts));
        } else if (staging_addr != 0 && staging_size > 0) {
            auto projected_exchange_view =
              apply_exchange_projection(exchange_view, projection_indices);
            // Broadcast path: accumulate each batch as a separate entry.
            // Reserve a staging region atomically BEFORE packing, to prevent
            // concurrent sink() calls from overlapping in the staging buffer.
            // Use cudf::pack (not chunked_pack) to get correct packed output,
            // then D2D copy to staging. chunked_pack has a bug with STRING columns.
            auto packed_probe = cudf::pack(projected_exchange_view, stream);
            auto total_size = packed_probe.gpu_data->size();
            size_t aligned_size = (total_size + 255) & ~255UL;
            size_t staging_offset = lgb.ReserveStagingRegion(aligned_size, staging_size);

            if (staging_offset + total_size <= staging_size) {
              // Fits in staging buffer — copy packed data to staging (sync).
              cudaMemcpy(reinterpret_cast<void*>(staging_addr + staging_offset),
                        packed_probe.gpu_data->data(), total_size,
                        cudaMemcpyDeviceToDevice);
              auto metadata = std::move(packed_probe.metadata);
              SIRIUS_LOG_INFO("[result_collector] broadcast batch: {} rows, {} bytes at staging+{}",
                              projected_exchange_view.num_rows(), total_size, staging_offset);
              duckdb::PackedBroadcastEntry entry;
              entry.staging_offset = staging_offset;
              entry.packed_size = total_size;
              entry.metadata = std::move(metadata);
              entry.num_rows = static_cast<int32_t>(projected_exchange_view.num_rows());
              lgb.AccumulatePackedBroadcast(std::move(entry));
            } else {
              // Overflow: pack into separate rmm::device_buffer.
              SIRIUS_LOG_WARN("[result_collector] broadcast batch ({} bytes) overflows staging, using rmm::device_buffer", total_size);
              auto packed = cudf::pack(projected_exchange_view);
              auto gpu_addr = reinterpret_cast<uintptr_t>(packed.gpu_data->data());
              auto gpu_size = packed.gpu_data->size();
              duckdb::PackedBroadcastEntry entry;
              entry.staging_offset = 0;
              entry.packed_size = gpu_size;
              entry.metadata = std::move(packed.metadata);
              entry.num_rows = static_cast<int32_t>(projected_exchange_view.num_rows());
              entry.overflow_gpu_addr = gpu_addr;
              entry.overflow_data = std::make_shared<rmm::device_buffer>(std::move(*packed.gpu_data));
              lgb.AccumulatePackedBroadcast(std::move(entry));
            }
        } else {
            auto projected_exchange_view =
              apply_exchange_projection(exchange_view, projection_indices);
            // No staging buffer — pack into rmm::device_buffer.
            auto packed = cudf::pack(projected_exchange_view);
            auto gpu_addr = reinterpret_cast<uintptr_t>(packed.gpu_data->data());
            auto gpu_size = packed.gpu_data->size();
            SIRIUS_LOG_INFO("[result_collector] cudf::pack (no staging): {} bytes at 0x{:x}", gpu_size, gpu_addr);
            duckdb::PackedBroadcastEntry entry;
            entry.staging_offset = 0;
            entry.packed_size = gpu_size;
            entry.metadata = std::move(packed.metadata);
            entry.num_rows = static_cast<int32_t>(projected_exchange_view.num_rows());
            entry.overflow_gpu_addr = gpu_addr;
            entry.overflow_data = std::make_shared<rmm::device_buffer>(std::move(*packed.gpu_data));
            lgb.AccumulatePackedBroadcast(std::move(entry));
        }
      }

      if (should_retain && exchange_only && exchange_view_storage) {
        SIRIUS_LOG_INFO(
          "[result_collector] exchange-only capture: skipping host materialization for {} rows",
          exchange_view.num_rows());
        continue;
      }

      // Make the HOST memory reservation
      auto sirius_ctx  = _client_ctx.registered_state->Get<duckdb::SiriusContext>("sirius_state");
      auto& memory_mgr = sirius_ctx->get_memory_manager();
      /// TODO: Find the closest memory space, not just any memory space, in HOST tier
      auto reservation = memory_mgr.request_reservation(
        cucascade::memory::any_memory_space_in_tier{cucascade::memory::Tier::HOST},
        data->get_size_in_bytes());
      if (!reservation) {
        throw internal_exception(
          "[GPUPhysicalMaterializedCollector] Failed to reserve host memory for result collection");
      }

      // Convert to host representation
      auto& registry      = sirius::converter_registry::get();
      auto& mem_space     = reservation->get_memory_space();
      auto& data_repo_mgr = sirius_ctx->get_data_repository_manager();
      auto next_batch_id  = data_repo_mgr.get_next_data_batch_id();
      clone_batch         = input_batch->clone(next_batch_id, stream);
      // todo (bobbi) pass stream to sink
      clone_batch->convert_to<cucascade::host_data_representation>(registry, &mem_space, stream);
      data = clone_batch->get_data();
    } else if (data->get_current_tier() != cucascade::memory::Tier::HOST) {
      // Data must be in HOST tier (i.e., cannot currently reside in DISK tier)
      throw invalid_input_exception(
        "[GPUPhysicalMaterializedCollector] Expected host_data_representation in HOST tier");
    }

    // Only accepting host_data_representation for now
    assert(dynamic_cast<cucascade::host_data_representation*>(data) != nullptr);

    // Push chunks to result collection
    auto const& host_table = data->cast<cucascade::host_data_representation>();
    // host_table_chunk_reader expects get_host_table() and ->allocation to be non-null;
    // otherwise it will dereference a null unique_ptr (e.g. in column_reader::initialize).
    auto const* ht = host_table.get_host_table().get();
    if (!ht) {
      throw invalid_input_exception(
        "[GPUPhysicalMaterializedCollector] host_data_representation has null "
        "get_host_table()");
    }
    if (!ht->allocation) {
      throw invalid_input_exception(
        "[GPUPhysicalMaterializedCollector] host_table allocation is null (cannot read chunks)");
    }
    host_table_chunk_reader chunk_reader(_client_ctx, host_table, types);

    // Push chunks to result collection
    while (true) {
      // TODO(amin): it is fishy that append take a mutable reference to the chunk reader and we are
      // passing local variable chunk reader by reference. We should investigate if this can cause
      // any issues (e.g., if duckdb does not consume all data from the chunk reader in append and
      // we move to the next chunk reader, then the previous chunk reader's state will be lost).
      duckdb::DataChunk chunk;
      if (!chunk_reader.get_next_chunk(chunk)) { break; }

      std::lock_guard<std::mutex> guard(lock);
      // Initialize result collection if it is null (from a move)
      if (!result_collection) {
        result_collection = duckdb::make_uniq<duckdb::ColumnDataCollection>(
          _client_ctx, sirius::to_duckdb_vec(types));
      }
      result_collection->Append(chunk);
    }
  }

}

}  // namespace op
}  // namespace sirius
