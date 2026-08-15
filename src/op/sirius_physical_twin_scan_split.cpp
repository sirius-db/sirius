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

#include "op/sirius_physical_twin_scan_split.hpp"

#include "data/data_batch_utils.hpp"
#include "expression_evaluator/expression_evaluator.hpp"
#include "log/logging.hpp"
#include "sirius/exception.hpp"

#include <cudf/cudf_utils.hpp>
#include <cudf/table/table.hpp>

#include <nvtx3/nvtx3.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>

#include <format>
#include <iterator>
#include <type_traits>
#include <variant>

namespace sirius {
namespace op {

namespace {

/**
 * @brief Twin-scan task output: both fan-out halves in one pipelineable container.
 *
 * The base class holds out-A batches followed by out-B batches so the task machinery's
 * generic accounting (output bytes, telemetry subscription) sees every produced batch;
 * `split_point` lets sink() route the two halves to their respective consumer pipelines.
 */
class twin_split_output_data : public pipelineable_operator_data {
 public:
  twin_split_output_data(std::vector<std::shared_ptr<cucascade::data_batch>> batches_a_then_b,
                         std::size_t split_point)
    : pipelineable_operator_data(std::move(batches_a_then_b)), _split_point(split_point)
  {
  }

  [[nodiscard]] std::size_t split_point() const noexcept { return _split_point; }

 private:
  std::size_t _split_point;
};

}  // namespace

sirius_physical_twin_scan_split::sirius_physical_twin_scan_split(
  duckdb::vector<sirius::logical_type> types_a,
  std::vector<cudf::size_type> output_indices_a,
  std::unique_ptr<sirius::ast::node> residual,
  output_mask output_columns_b,
  duckdb::vector<sirius::logical_type> types_b,
  std::size_t estimated_cardinality,
  sirius_physical_twin_scan_ref* twin_ref)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::TWIN_SCAN_SPLIT, std::move(types_a), estimated_cardinality),
    _output_indices_a(std::move(output_indices_a)),
    _residual(std::move(residual)),
    _output_columns_b(std::move(output_columns_b)),
    _types_b(std::move(types_b)),
    _twin_ref(twin_ref)
{
  if (_residual == nullptr) {
    throw internal_exception("twin_scan_split constructed without a residual predicate");
  }
  if (_twin_ref == nullptr) {
    throw internal_exception("twin_scan_split constructed without a twin ref anchor");
  }
  if (_output_indices_a.empty()) {
    throw internal_exception("twin_scan_split constructed with an empty out-A projection");
  }
}

std::string sirius_physical_twin_scan_split::params_to_string() const
{
  return std::format("out_a_cols={} out_b_cols={}", _output_indices_a.size(), _types_b.size());
}

std::unique_ptr<operator_data> sirius_physical_twin_scan_split::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_twin_scan_split::execute"};
  const auto& input         = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches = input.get_read_only_batches();

  sirius::expression_evaluator evaluator(
    *_residual, cudf::get_current_device_resource_ref(), stream);

  std::vector<std::shared_ptr<cucascade::data_batch>> batches_a;
  std::vector<std::shared_ptr<cucascade::data_batch>> batches_b;
  batches_a.reserve(input_batches.size());
  batches_b.reserve(input_batches.size());

  std::size_t rows_in = 0;
  std::size_t rows_a  = 0;
  std::size_t rows_b  = 0;

  for (auto const& batch : input_batches) {
    auto view = batch.get_data()->cast<cucascade::gpu_table_representation>().get_table_view();
    rows_in += static_cast<std::size_t>(view.num_rows());

    // out-B first: the residual gather reads the shared columns, so it must not observe a
    // half-moved batch (out-A below is a fresh copy, so ordering is for clarity only).
    auto filtered_b = std::visit(
      [&](const auto& indices) {
        using IndicesType = std::decay_t<decltype(indices)>;
        if constexpr (std::is_same_v<IndicesType, passthrough>) {
          return evaluator.select(view);
        } else {
          return evaluator.select(view, indices);
        }
      },
      _output_columns_b);
    rows_b += static_cast<std::size_t>(filtered_b->num_rows());
    batches_b.push_back(sirius::make_data_batch(
      std::move(filtered_b), *batch.get_memory_space(), stream, batch_telemetry()));

    // out-A: an unfiltered gather of the first scan's columns from the shared stream.
    auto table_a = std::make_unique<cudf::table>(
      view.select(_output_indices_a), stream, cudf::get_current_device_resource_ref());
    rows_a += static_cast<std::size_t>(table_a->num_rows());
    batches_a.push_back(sirius::make_data_batch(
      std::move(table_a), *batch.get_memory_space(), stream, batch_telemetry()));
  }

  _rows_in.fetch_add(rows_in, std::memory_order_relaxed);
  _rows_a.fetch_add(rows_a, std::memory_order_relaxed);
  _rows_b.fetch_add(rows_b, std::memory_order_relaxed);

  const std::size_t split_point = batches_a.size();
  auto combined                 = std::move(batches_a);
  combined.insert(combined.end(),
                  std::make_move_iterator(batches_b.begin()),
                  std::make_move_iterator(batches_b.end()));
  return std::make_unique<twin_split_output_data>(std::move(combined), split_point);
}

void sirius_physical_twin_scan_split::sink(const operator_data& input_data,
                                           rmm::cuda_stream_view stream)
{
  auto const* output = dynamic_cast<const twin_split_output_data*>(&input_data);
  if (output == nullptr) {
    throw internal_exception("twin_scan_split::sink expects twin_split_output_data");
  }
  auto* consumer_b = _twin_ref->get_parent_op();
  if (consumer_b == nullptr) {
    throw internal_exception("twin_scan_split: twin ref has no tree parent to route out-B to");
  }
  // Exactly one A edge (the tree parent's pipeline) and one B edge (the twin ref's consumer)
  // are emitted by the converter; anything else means the wiring pass and this operator
  // disagree — fail loudly rather than mis-route data.
  std::size_t b_edges = 0;
  for (auto const& port_info : next_port_after_sink) {
    b_edges += port_info.next_operator == consumer_b ? 1 : 0;
  }
  if (next_port_after_sink.size() != 2 || b_edges != 1) {
    throw internal_exception(
      "twin_scan_split: expected exactly one out-A and one out-B edge, got {} edges ({} to the "
      "twin consumer)",
      next_port_after_sink.size(),
      b_edges);
  }

  const auto& batches = output->get_data_batches();
  for (auto const& port_info : next_port_after_sink) {
    const bool is_b         = port_info.next_operator == consumer_b;
    const std::size_t begin = is_b ? output->split_point() : 0;
    const std::size_t end   = is_b ? batches.size() : output->split_point();
    for (std::size_t i = begin; i < end; ++i) {
      port_info.next_operator->push_data_batch(port_info.next_operator_port_name, batches[i]);
    }
  }
}

void sirius_physical_twin_scan_split::on_finalize_operator()
{
  SIRIUS_LOG_INFO("[twin_scan_split] fused stream rows={} out_a rows={} out_b rows={}",
                  _rows_in.load(std::memory_order_relaxed),
                  _rows_a.load(std::memory_order_relaxed),
                  _rows_b.load(std::memory_order_relaxed));
}

}  // namespace op
}  // namespace sirius
