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

#pragma once

#include "op/partition/partition_exchange.hpp"
#include "op/partition/partition_input_observer.hpp"
#include "pipeline/data_size_estimator.hpp"

#include <atomic>
#include <optional>

namespace duckdb {
class SiriusContext;
}

namespace sirius::op {

/// Routes rows using explicit keys and a layout resolved before the first input claim.
/// Consumers supply sizing/algorithm policy through a plan-bound partition_exchange.
class sirius_physical_partition : public sirius_physical_operator {
 public:
  static constexpr auto TYPE = SiriusPhysicalOperatorType::PARTITION;
  sirius_physical_partition(duckdb::vector<sirius::logical_type> types,
                            std::size_t estimated_cardinality,
                            partition_spec spec,
                            MemoryBarrierType input_barrier = MemoryBarrierType::FULL,
                            duckdb::SiriusContext* observer = nullptr,
                            bool enable_size_estimation     = false);

  std::string get_name() const override;
  bool is_source() const override;
  bool is_sink() const override;
  MemoryBarrierType input_barrier_for(sirius_physical_operator const& producer) const override;
  void build_pipelines(pipeline::sirius_pipeline& current,
                       pipeline::sirius_meta_pipeline& meta_pipeline) override;
  std::unique_ptr<operator_data> execute(operator_data const&, ::cuda::stream_ref) override;
  void sink(operator_data const&, ::cuda::stream_ref) override;
  std::optional<task_creation_hint> get_next_task_hint() override;
  std::unique_ptr<operator_data> get_next_task_input_data() override;
  std::size_t no_history_peak_memory_estimate(input_stats const&) const override;

  /// Plan-time bindings. No tasks or arrivals may precede these calls.
  void set_exchange(std::shared_ptr<partition_exchange> exchange)
  {
    _exchange = std::move(exchange);
  }
  void set_input_observer(std::unique_ptr<partition_input_observer> observer)
  {
    _input_observer = std::move(observer);
  }
  void set_input_barrier(MemoryBarrierType barrier) { _input_barrier = barrier; }
  [[nodiscard]] MemoryBarrierType input_barrier() const { return _input_barrier; }
  [[nodiscard]] bool is_size_estimation_enabled() const { return _enable_size_estimation; }
  [[nodiscard]] std::vector<int> const& partition_keys() const { return _spec.key_columns; }
  [[nodiscard]] partition_spec const& specification() const { return _spec; }

  /// Fixed layouts need no consumer or negotiation. Also used by a bound exchange's transaction.
  void set_layout(partition_layout layout);
  void set_fixed_layout(partition_layout layout)
  {
    _exchange.reset();
    set_layout(std::move(layout));
  }
  [[nodiscard]] std::shared_ptr<partition_exchange> exchange() const { return _exchange; }
  [[nodiscard]] std::shared_ptr<partition_layout const> layout() const
  {
    return _layout.load(std::memory_order_acquire);
  }

  /// Input policy queries; these never claim batches.
  std::optional<task_creation_hint> sizing_hint();
  [[nodiscard]] bool input_complete() const;
  [[nodiscard]] bool input_empty() const;
  partition_input_stats input_statistics(bool use_estimate = false);

 protected:
  void on_finalize_operator() override;
  void on_input_batch_pushed(std::string_view, cucascade::data_batch&) override;

 private:
  partition_input_stats compute_input_totals();
  uint64_t compute_total_bytes() { return compute_input_totals().bytes; }
  std::optional<uint64_t> estimated_total_input_bytes();

  partition_spec _spec;
  MemoryBarrierType _input_barrier;
  std::atomic<std::shared_ptr<partition_layout const>> _layout;
  std::shared_ptr<partition_exchange> _exchange;
  std::unique_ptr<partition_input_observer> _input_observer;
  duckdb::SiriusContext* _compressed_materialization_observer;
  bool _enable_size_estimation;
  std::optional<pipeline::data_size_estimate> _size_estimate;
  enum class sizing_basis : uint8_t { measured, upstream_complete, projected };
  static char const* sizing_basis_name(sizing_basis);
  sizing_basis _sizing_basis{sizing_basis::measured};
  uint64_t _sizing_bytes{0};
  std::atomic<uint64_t> _actual_bytes{0};
};

}  // namespace sirius::op
