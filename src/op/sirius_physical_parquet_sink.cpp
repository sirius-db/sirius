// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "op/sirius_physical_parquet_sink.hpp"

#include "data/data_batch_utils.hpp"
#include "log/logging.hpp"
#include "memory/size_arithmetic.hpp"
#include "output/parquet_writer.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"

namespace sirius::op {
sirius_physical_parquet_sink::sirius_physical_parquet_sink(duckdb::vector<logical_type> input_types,
                                                           output::parquet_export_spec spec)
  : sirius_physical_operator(TYPE, std::move(input_types), 0), session_(std::move(spec))
{
}
void sirius_physical_parquet_sink::build_pipelines(pipeline::sirius_pipeline& current,
                                                   pipeline::sirius_meta_pipeline& meta)
{
  if (children.size() != 1) throw internal_exception("Parquet sink requires one producer");
  meta.get_state().add_pipeline_operator(current, *this);
  auto& child = meta.create_child_meta_pipeline(current, *this);
  child.build(*children[0]);
}
std::unique_ptr<operator_data> sirius_physical_parquet_sink::execute(operator_data const& data,
                                                                     ::cuda::stream_ref stream)
{
  auto const& input = dynamic_cast<pipelineable_operator_data const&>(data);
  auto locked       = input.get_read_only_batches();
  std::vector<cudf::table_view> tables;
  std::vector<std::uint64_t> rows;
  for (auto const& batch : locked) {
    auto table = get_cudf_table_view(batch);
    if (table.num_rows() == 0) continue;
    tables.push_back(table);
    rows.push_back(table.num_rows());
  }
  session_.write(
    input.original_batch_ids(), input.get_position(), rows, [&](std::size_t i, auto const& path) {
      output::write_parquet_part(session_.spec(), tables[i], path, stream);
    });
  return std::make_unique<pipelineable_operator_data>();
}
std::size_t sirius_physical_parquet_sink::no_history_peak_memory_estimate(
  input_stats const& stats) const
{
  return memory::saturating_mul(stats.bytes, 2);
}
output::parquet_export_result sirius_physical_parquet_sink::finish(
  std::function<void(output::parquet_export_result const&)> const& prepare_completion)
{
  auto result = session_.finish(
    [&](std::size_t, auto const& path) { output::write_empty_parquet(session_.spec(), path); },
    prepare_completion);
  try {
    SIRIUS_LOG_INFO("[parquet_export] rows={} files={} bytes={} codec={} directory={}",
                    result.rows,
                    result.files,
                    result.bytes,
                    output::compression_name(session_.spec().compression),
                    session_.spec().directory.string());
  } catch (...) {
  }
  return result;
}
}  // namespace sirius::op
