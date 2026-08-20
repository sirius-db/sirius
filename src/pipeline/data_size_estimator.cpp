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

/// The port @p id on @p op if it carries data from a producer, else nullptr. Folds the three
/// conditions every caller here needs: the name must exist (get_port throws, and a throw escaping
/// into get_next_task_hint is not recoverable), the port must carry bytes rather than being a
/// dependency-only edge (null repo), and it must have a producing pipeline to walk into.
op::sirius_physical_operator::port* resolve_data_port(op::sirius_physical_operator& op,
                                                      std::string_view id)
{
  auto* p = op.try_get_port(id);
  if (p == nullptr || p->repo == nullptr || !p->src_pipeline) { return nullptr; }
  return p;
}

/// How many distinct data-carrying producer pipelines feed a pipeline, and (when there is
/// exactly one) which. Also reports whether any operator caps the pipeline's output, since that
/// falls out of the same walk. See @ref scan_producer_pipelines.
struct producer_scan {
  std::size_t count      = 0;
  sirius_pipeline* first = nullptr;
  bool output_capped     = false;
};

/**
 * @brief Count the data-carrying ports feeding @p pipeline and identify the first producer.
 *
 * Mirrors sirius_pipeline::get_ingress_ports_info, which returns the producer *operator* where
 * we need the producer *pipeline*. `operators` spans source through sink after is_ready(), so
 * walking it reaches build-side ports on the sink too. Dependency-only ports (null repo) and
 * ports with no producer carry no bytes and are skipped.
 *
 * Deduplication by port pointer is load-bearing: `operators` normally already contains the
 * source and sink, so without it every port is seen twice and a single-input pipeline reads as
 * a fan-in. A reserved vector rather than a hash set because this runs on every task-creation
 * poll until a projection latches, and port counts are single digits.
 */
producer_scan scan_producer_pipelines(sirius_pipeline& pipeline)
{
  producer_scan result;
  std::vector<const op::sirius_physical_operator::port*> seen;
  seen.reserve(4);

  auto collect_from = [&](op::sirius_physical_operator* candidate) {
    if (candidate == nullptr) { return; }
    // Idempotent, so unlike the port count this needs no dedup against a second visit.
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
  // Sink-only pipelines leave `operators` empty; source/sink are still set.
  collect_from(pipeline.get_source().get());
  collect_from(pipeline.get_sink().get());

  return result;
}

/// The weaker of two sample counts, treating 0 as "no ratio applied yet" rather than as a
/// minimum. Keeps @ref data_size_estimate::ratio_samples reporting the least-supported ratio in
/// the chain.
std::size_t weaker_sample_count(std::size_t so_far, std::size_t candidate)
{
  return so_far == 0 ? candidate : std::min(so_far, candidate);
}

/// Whether @p totals has enough support for its measured ratio to be used. Too few completed
/// tasks is treated exactly as "no ratio yet": the number exists but one or two batches are not
/// evidence that it describes the pipeline, and the consumer latches the first estimate it is
/// given rather than refining it later. See size_estimate_options::min_ratio_samples.
bool ratio_is_trusted(const history_totals& totals, const size_estimate_options& options)
{
  return totals.ratio_records >= options.min_ratio_samples && totals.ratio().has_value();
}

/// Scale @p input by the pipeline ratio in @p totals. Takes the snapshot rather than the pipeline
/// so callers that already needed it (to decide whether recursing is worthwhile) do not lock the
/// history a second time.
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
    // A learned ratio is a projection, never a measurement.
    .exact = false,
    .hops  = input.hops,
    // A substituted unit ratio carries no measured support, so it neither adds to nor weakens
    // the chain's: pass the upstream count through rather than overwriting it, which would
    // discard a real measurement made further up.
    .ratio_samples = trust_measured ? weaker_sample_count(input.ratio_samples, totals.ratio_records)
                                    : input.ratio_samples,
    // Applying a measured ratio does not launder a planner guess out of the chain. This is the
    // hop where ratio_samples stops being able to carry the provenance, hence the separate flag.
    .planner_derived = input.planner_derived,
  };
}

std::optional<data_size_estimate> estimate_output_bytes_impl(sirius_pipeline& pipeline,
                                                             const size_estimate_options& options,
                                                             std::size_t hops)
{
  if (hops > options.max_hops) { return std::nullopt; }

  // 1. Finished pipeline: its recorded output total is the exact answer. Safe to read because
  //    pipeline_finished is only set once tasks_created == tasks_completed, and each task
  //    records its output before mark_task_completed() runs in its destructor.
  //
  //    Keyed on output_records, not the ratio terms: a task with no a-priori size estimate has
  //    a zero basis and cannot inform a ratio, but its emitted bytes are still part of this
  //    pipeline's total. Reporting only the ratio-eligible tasks' output would under-count while
  //    still claiming exact.
  //
  //    With no record at all, the task count separates two lookalikes: a pipeline that finished
  //    having never created a task had nothing to measure and emitted exactly zero (an empty
  //    scan), while one whose tasks all recorded nothing lost the measurement and yields nullopt.
  //    Finishing needs the source exhausted, not just the counters balanced, so 0 == 0 here means
  //    drained rather than not yet started.
  if (pipeline.is_pipeline_finished()) {
    auto const totals = pipeline.get_memory_history().totals();
    if (totals.output_records == 0) {
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

  // A row limit caps output independently of input, so no measured ratio extrapolates through
  // this pipeline: output stops growing once the cap binds, and the pipeline may then finish
  // without its source draining. Only the finished case above can answer for it.
  if (producers.output_capped) { return std::nullopt; }

  // 2. Fan-in: follow only the source's nominated primary input.
  //
  //    The recorded input_basis is unusable here: a STANDARD join pairs each probe batch with
  //    every build batch and borrows rather than pops, so the same bytes enter input_basis once
  //    per pairing and its sum is a cross product, not an input volume. Ask the operator for
  //    probe bytes counted once per batch instead, and divide the pipeline's output total by
  //    that. An operator nominating no primary port (CTE, delim-join wiring) yields nullopt.
  if (producers.count > 1) {
    auto* source = pipeline.get_source().get();
    if (source == nullptr) { return std::nullopt; }

    auto const port_name = source->primary_input_port();
    if (!port_name.has_value()) { return std::nullopt; }

    // Numerator before denominator, and keep it that way: `consumed` advances at task creation,
    // output_bytes at completion, and both only grow, so reading output first guarantees every
    // task in it is also in the `consumed` read below. The other order lets a task created and
    // completed between the two reads contribute output with no matching input, inflating the
    // ratio; this way the residual error is always low, which is the direction the in-flight bias
    // already points. See min_fan_in_ratio_samples and data-size-estimation.md.
    auto const totals = pipeline.get_memory_history().totals();
    if (totals.output_records < options.min_fan_in_ratio_samples) { return std::nullopt; }

    auto const consumed = source->consumed_primary_input_bytes();
    if (!consumed.has_value() || *consumed == 0) { return std::nullopt; }

    // A bad nomination is "no estimate", not an error: the name comes from an operator override.
    auto* p = resolve_data_port(*source, *port_name);
    if (p == nullptr) { return std::nullopt; }

    auto upstream = estimate_output_bytes_impl(*p->src_pipeline, options, hops + 1);
    if (!upstream.has_value()) { return std::nullopt; }

    // The residual in-flight bias is bounded by min_fan_in_ratio_samples above, not corrected
    // here: task counts cannot be used to discount it, because `consumed` does not advance once
    // per task. See data-size-estimation.md.
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

  // 3. Leaf: anchor on what the source operator knows about its own total.
  if (producers.count == 0) {
    auto* source = pipeline.get_source().get();
    if (source == nullptr) { return std::nullopt; }

    if (auto input_bytes = source->total_source_input_bytes()) {
      return apply_pipeline_ratio(
        pipeline.get_memory_history().totals(),
        data_size_estimate{.bytes = *input_bytes, .exact = true, .hops = hops},
        options);
    }
    // An output quantity for the *source*, which is the pipeline's output only when nothing
    // follows it — get_operators() runs source..sink, so source == sink means a lone operator.
    // With a FILTER, PROJECTION or DYNAMIC_FILTER in between there is no way to bridge the gap:
    // the pipeline ratio's denominator is pre-filter input bytes, so scaling by it would count
    // the scan's pushdown selectivity twice, and returning it unscaled would ignore the
    // downstream operators entirely.
    if (pipeline.get_sink().get() == source) {
      // GPU_SCAN's estimated_cardinality projection, the one planner-derived number in the
      // design, so this is where the provenance flag originates.
      if (auto output_bytes = source->total_source_output_bytes()) {
        return data_size_estimate{
          .bytes = *output_bytes, .exact = false, .hops = hops, .planner_derived = true};
      }
    }
    return std::nullopt;
  }

  // 4. Single producer: recurse, then scale by this pipeline's ratio.
  //
  //    Check our own ratio first. Without it the whole multi-hop walk runs and is then thrown
  //    away whenever the ratio would be rejected — a cost paid on every task-creation poll early
  //    in a query, which is exactly when no pipeline has enough samples yet. The fan-in branch
  //    gates the same way.
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
  // llround is UB outside the integral type's range; cap at what a byte count can represent.
  constexpr double kMaxBytes = 9.0e18;  // comfortably inside both int64 and size_t
  if (scaled > kMaxBytes) { return std::nullopt; }
  return static_cast<std::size_t>(std::llround(scaled));
}

std::optional<std::size_t> project_source_output_bytes(std::size_t estimated_cardinality,
                                                       std::size_t emitted_rows,
                                                       std::size_t emitted_bytes)
{
  // Nothing measured yet: there is no bytes/row factor, and no floor either.
  if (emitted_rows == 0 || emitted_bytes == 0) { return std::nullopt; }
  auto const bytes_per_row = static_cast<double>(emitted_bytes) / static_cast<double>(emitted_rows);
  auto const projected     = scale_bytes_checked(estimated_cardinality, bytes_per_row);
  if (!projected.has_value()) { return std::nullopt; }
  // Bytes already emitted are a hard lower bound on the total. See the header for why this fires
  // in practice rather than only in principle.
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
