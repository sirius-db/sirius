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

#include "pipeline/data_size_estimator.hpp"

#include "op/sirius_physical_operator.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace sirius {
namespace pipeline {

namespace {

op::sirius_physical_operator::port* resolve_data_port(op::sirius_physical_operator& op,
                                                      std::string_view id)
{
  auto* p = op.try_get_port(id);
  if (p == nullptr || p->repo == nullptr || !p->src_pipeline) { return nullptr; }
  return p;
}

struct producer_scan {
  std::size_t count      = 0;
  sirius_pipeline* first = nullptr;
  bool output_capped     = false;
};

producer_scan scan_producer_pipelines(sirius_pipeline& pipeline)
{
  producer_scan result;
  std::vector<const op::sirius_physical_operator::port*> seen;
  seen.reserve(4);

  auto collect_from = [&](op::sirius_physical_operator* candidate) {
    if (candidate == nullptr) { return; }
    result.output_capped |= candidate->caps_pipeline_output();
    for (auto port_id : candidate->get_port_ids()) {
      auto* p = resolve_data_port(*candidate, port_id);
      if (p == nullptr) { continue; }
      if (std::find(seen.begin(), seen.end(), p) != seen.end()) { continue; }
      seen.push_back(p);
      if (result.count == 0) { result.first = p->src_pipeline.get(); }
      result.count++;
    }
  };

  for (auto& op_ref : pipeline.get_operators()) {
    collect_from(&op_ref.get());
  }
  // Source and sink may also appear in get_operators(); port-pointer dedup prevents false fan-in.
  collect_from(pipeline.get_source().get());
  collect_from(pipeline.get_sink().get());

  return result;
}

// Zero means no ratio has been applied yet.
std::size_t weaker_sample_count(std::size_t so_far, std::size_t candidate)
{
  return so_far == 0 ? candidate : std::min(so_far, candidate);
}

bool ratio_is_trusted(const history_totals& totals, const size_estimate_options& options)
{
  return totals.ratio_records >= options.min_ratio_samples && totals.ratio().has_value();
}

std::optional<data_size_estimate> apply_pipeline_ratio(const history_totals& totals,
                                                       data_size_estimate input,
                                                       const size_estimate_options& options)
{
  bool const trust_measured = ratio_is_trusted(totals, options);
  if (!trust_measured && !options.assume_unit_ratio) { return std::nullopt; }

  auto const scaled = scale_bytes_checked(input.bytes, trust_measured ? *totals.ratio() : 1.0);
  if (!scaled.has_value()) { return std::nullopt; }
  return data_size_estimate{
    .bytes = *scaled,
    .exact = false,
    .hops  = input.hops,
    // A substituted unit ratio contributes no measured support.
    .ratio_samples = trust_measured ? weaker_sample_count(input.ratio_samples, totals.ratio_records)
                                    : input.ratio_samples,
    // Preserve planner-derived provenance through measured ratios.
    .planner_derived = input.planner_derived,
  };
}

std::optional<data_size_estimate> estimate_output_bytes_impl(sirius_pipeline& pipeline,
                                                             const size_estimate_options& options,
                                                             std::size_t hops)
{
  if (hops > options.max_hops) { return std::nullopt; }

  // Finished implies every task recorded output before completion. Use all output records, not
  // only ratio-eligible records, because zero-basis tasks may still emit bytes.
  if (pipeline.is_pipeline_finished()) {
    auto const totals = pipeline.get_memory_history().totals();
    if (totals.output_records == 0) {
      // No tasks means exact empty output; created tasks with no records lost measurement.
      if (pipeline.get_tasks_created() != 0) { return std::nullopt; }
      return data_size_estimate{.bytes = 0, .exact = true, .hops = hops};
    }
    return data_size_estimate{
      .bytes = totals.output_bytes,
      .exact = true,
      .hops  = hops,
    };
  }

  auto const producers = scan_producer_pipelines(pipeline);

  // A capped pipeline may stop before draining its source, so only a finished total is valid.
  if (producers.output_capped) { return std::nullopt; }

  // Fan-in: use the nominated primary input. A join's task-level input basis repeats borrowed
  // batches across pairings and is not an input volume.
  if (producers.count > 1) {
    auto* source = pipeline.get_source().get();
    if (source == nullptr) { return std::nullopt; }

    auto const port_name = source->primary_input_port();
    if (!port_name.has_value()) { return std::nullopt; }

    // Read output before consumed input. A concurrent task can then only bias the ratio low;
    // reversing the order could omit its input while including its output.
    auto const totals = pipeline.get_memory_history().totals();
    if (totals.output_records < options.min_fan_in_ratio_samples) { return std::nullopt; }

    auto const consumed = source->consumed_primary_input_bytes();
    if (!consumed.has_value() || *consumed == 0) { return std::nullopt; }

    auto* p = resolve_data_port(*source, *port_name);
    if (p == nullptr) { return std::nullopt; }

    auto upstream = estimate_output_bytes_impl(*p->src_pipeline, options, hops + 1);
    if (!upstream.has_value()) { return std::nullopt; }

    // The sample threshold bounds the conservative in-flight bias.
    auto const ratio  = static_cast<double>(totals.output_bytes) / static_cast<double>(*consumed);
    auto const scaled = scale_bytes_checked(upstream->bytes, ratio);
    if (!scaled.has_value()) { return std::nullopt; }
    return data_size_estimate{
      .bytes           = *scaled,
      .exact           = false,
      .hops            = upstream->hops,
      .ratio_samples   = weaker_sample_count(upstream->ratio_samples, totals.output_records),
      .planner_derived = upstream->planner_derived,
    };
  }

  // Leaf: anchor on the source's total.
  if (producers.count == 0) {
    auto* source = pipeline.get_source().get();
    if (source == nullptr) { return std::nullopt; }

    if (auto input_bytes = source->total_source_input_bytes()) {
      return apply_pipeline_ratio(
        pipeline.get_memory_history().totals(),
        data_size_estimate{.bytes = *input_bytes, .exact = true, .hops = hops},
        options);
    }
    // A source-output estimate is also the pipeline output only for a lone source. Scaling it
    // through downstream operators would double-count scan pushdown selectivity.
    if (pipeline.get_sink().get() == source) {
      // GPU scan cardinality projection is planner-derived.
      if (auto output_bytes = source->total_source_output_bytes()) {
        return data_size_estimate{
          .bytes = *output_bytes, .exact = false, .hops = hops, .planner_derived = true};
      }
    }
    return std::nullopt;
  }

  // Single producer: reject an unsupported local ratio before recursing.
  if (producers.first == nullptr) { return std::nullopt; }
  auto const totals = pipeline.get_memory_history().totals();
  if (!ratio_is_trusted(totals, options) && !options.assume_unit_ratio) { return std::nullopt; }

  auto upstream = estimate_output_bytes_impl(*producers.first, options, hops + 1);
  if (!upstream.has_value()) { return std::nullopt; }
  return apply_pipeline_ratio(totals, *upstream, options);
}

}  // namespace

std::optional<std::size_t> scale_bytes_checked(std::size_t bytes, double ratio)
{
  if (!std::isfinite(ratio) || ratio < 0.0) { return std::nullopt; }
  double const scaled = static_cast<double>(bytes) * ratio;
  if (!std::isfinite(scaled) || scaled < 0.0) { return std::nullopt; }
  // llround is undefined outside the integral type's range.
  constexpr double kMaxBytes = 9.0e18;  // Within int64_t and size_t.
  if (scaled > kMaxBytes) { return std::nullopt; }
  return static_cast<std::size_t>(std::llround(scaled));
}

std::optional<std::size_t> project_source_output_bytes(std::size_t estimated_cardinality,
                                                       std::size_t emitted_rows,
                                                       std::size_t emitted_bytes)
{
  if (emitted_rows == 0 || emitted_bytes == 0) { return std::nullopt; }
  auto const bytes_per_row = static_cast<double>(emitted_bytes) / static_cast<double>(emitted_rows);
  auto const projected     = scale_bytes_checked(estimated_cardinality, bytes_per_row);
  if (!projected.has_value()) { return std::nullopt; }
  // Emitted bytes are a hard lower bound on the total.
  return std::max(*projected, emitted_bytes);
}

std::optional<data_size_estimate> estimate_pipeline_total_output_bytes(
  sirius_pipeline& pipeline, size_estimate_options options)
{
  return estimate_output_bytes_impl(pipeline, options, /*hops=*/0);
}

std::optional<data_size_estimate> estimate_port_total_input_bytes(op::sirius_physical_operator& op,
                                                                  std::string_view port_id,
                                                                  size_estimate_options options)
{
  auto* p = resolve_data_port(op, port_id);
  if (p == nullptr) { return std::nullopt; }
  return estimate_pipeline_total_output_bytes(*p->src_pipeline, options);
}

}  // namespace pipeline
}  // namespace sirius
