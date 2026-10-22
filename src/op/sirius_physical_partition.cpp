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

#include "op/sirius_physical_partition.hpp"

#include "config.hpp"
#include "cudf/cudf_utils.hpp"
#include "data/data_batch_utils.hpp"
#include "log/logging.hpp"
#include "memory/size_arithmetic.hpp"
#include "op/partition/gpu_partition_impl.hpp"
#include "op/sirius_physical_partition_consumer_operator.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "sirius/exception.hpp"
#include "sirius_context.hpp"
#include "telemetry/nvtx.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <string_view>

namespace sirius {
namespace op {

namespace {

constexpr bool producer_requires_full_partition_input(SiriusPhysicalOperatorType type) noexcept
{
  switch (type) {
    case SiriusPhysicalOperatorType::PARTITION:
    case SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE:
    case SiriusPhysicalOperatorType::TOP_N:
    case SiriusPhysicalOperatorType::SORT_PARTITION: return true;
    default: return false;
  }
}

}  // namespace

const char* sirius_physical_partition::sizing_basis_name(sizing_basis basis)
{
  switch (basis) {
    case sizing_basis::measured: return "measured";
    case sizing_basis::upstream_complete: return "upstream-complete";
    case sizing_basis::projected: return "projected";
  }
  return "unknown";
}

sirius_physical_partition::sirius_physical_partition(duckdb::vector<sirius::logical_type> types,
                                                     std::size_t estimated_cardinality,
                                                     partition_spec spec,
                                                     MemoryBarrierType input_barrier,
                                                     duckdb::SiriusContext* observer,
                                                     bool enable_size_estimation)
  : sirius_physical_operator(TYPE, std::move(types), estimated_cardinality),
    _spec(std::move(spec)),
    _input_barrier(input_barrier),
    _compressed_materialization_observer(observer),
    _enable_size_estimation(enable_size_estimation)
{
  if (_spec.mode != partition_mode::hash && _spec.mode != partition_mode::single) {
    throw std::invalid_argument("partition input specification must be hash or single");
  }
  if (_spec.mode == partition_mode::hash && _spec.key_columns.empty()) {
    throw std::invalid_argument("hash partition requires keys");
  }
  if (_spec.mode == partition_mode::single && !_spec.key_columns.empty()) {
    throw std::invalid_argument("single-destination partition cannot have hash keys");
  }
  if (!_spec.key_cast_types.empty() && _spec.key_cast_types.size() != _spec.key_columns.size()) {
    throw std::invalid_argument("partition key/cast count mismatch");
  }
  for (auto key : _spec.key_columns) {
    if (key < 0 || static_cast<std::size_t>(key) >= this->types.size()) {
      throw std::invalid_argument("partition key is outside the input schema");
    }
  }
}

std::string sirius_physical_partition::get_name() const { return "PARTITION"; }

bool sirius_physical_partition::is_source() const { return true; }

bool sirius_physical_partition::is_sink() const { return true; }

void sirius_physical_partition::build_pipelines(pipeline::sirius_pipeline& current,
                                                pipeline::sirius_meta_pipeline& meta_pipeline)
{
  // PARTITION is always its own single-operator pipeline. The child is guaranteed to be
  // a sink (its `_parent_op` is this PARTITION, so the base `is_sink()` returns true), so
  // create our own meta and let the child's protocol build its own boundary.
  D_ASSERT(children.size() == 1);
  D_ASSERT(children[0]->is_sink());
  auto& partition_meta = meta_pipeline.create_child_meta_pipeline(current, *this);
  children[0]->build_pipelines(*partition_meta.get_base_pipeline(), partition_meta);
}

MemoryBarrierType sirius_physical_partition::input_barrier_for(
  sirius_physical_operator const& producer) const
{
  // Producer protocol takes precedence over the explicitly declared ingress preference.
  if (producer.type == SiriusPhysicalOperatorType::ORDER_BY) {
    return sirius_physical_operator::input_barrier_for(producer);
  }
  if (producer_requires_full_partition_input(producer.type)) { return MemoryBarrierType::FULL; }
  return _input_barrier;
}

std::unique_ptr<operator_data> sirius_physical_partition::execute(const operator_data& input_data,
                                                                  ::cuda::stream_ref stream)
{
  nvtx_scoped_range nvtx_range{"sirius_physical_partition::execute"};
  auto& input               = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches = input.get_read_only_batches();
  if (input_batches.size() != 1) {
    throw std::runtime_error("We expect only one input batch for partition operator " +
                             std::to_string(this->get_operator_id()));
  }
  auto const routing = layout();
  if (!routing) {
    throw std::runtime_error("Num partitions was not set in sirius_physical_partition operator " +
                             std::to_string(this->get_operator_id()));
  }

  auto const count           = static_cast<int>(routing->placement->num_partitions());
  auto const& input_batch_ro = input_batches[0];
  auto* space                = input_batch_ro.get_memory_space();

  auto record_actual_bytes = [&] {
    // Only successful executions contribute to the final projection-error report. In particular,
    // an OOM-rescheduled task must not count the same input once for every attempt.
    if (_enable_size_estimation && input_batch_ro.get_data()) {
      _actual_bytes.fetch_add(input_batch_ro.get_data()->get_size_in_bytes(),
                              std::memory_order_relaxed);
    }
  };

  // Broadcast mode never hash-partitions: the build side replicates its (small) batch to every
  // slot and the probe side streams through unpartitioned. In both cases execute() just forwards
  // the input batches; the fan-out to slots happens in sink().
  if (routing->mode != partition_mode::hash || count < 2) {
    std::vector<std::size_t> destinations;
    std::vector<std::shared_ptr<cucascade::data_batch>> batches;
    auto add = [&](std::size_t destination) {
      destinations.push_back(destination);
      batches.push_back(input.get_data_batches().front());
    };
    if (routing->mode == partition_mode::broadcast) {
      for (std::size_t i = 0; i < routing->placement->num_partitions(); ++i) {
        add(i);
      }
    } else if (routing->mode == partition_mode::local) {
      std::optional<std::size_t> device;
      if (space != nullptr && space->get_tier() == cucascade::memory::Tier::GPU) {
        device = routing->placement->first_partition_for_device(space->get_device_id());
        if (!device) {
          SIRIUS_LOG_WARN(
            "sirius_physical_partition id {}: batch on GPU {} has no slot in local placement {}; "
            "using slot 0",
            this->get_operator_id(),
            space->get_device_id(),
            routing->placement->to_string());
        }
      }
      add(device.value_or(0));
    } else {
      add(0);
    }
    auto result =
      std::make_unique<routed_operator_data>(std::move(batches), std::move(destinations));
    record_actual_bytes();
    return result;
  }

  std::vector<std::shared_ptr<cucascade::data_batch>> partitioned_results;
  std::optional<std::uint64_t> publication_id;
  // Narrow-passthrough observability: count input columns whose actual carrier is narrower
  // than the native mapping of this operator's logical schema. The event reports actual batch
  // types, so a regression anywhere in the narrow-carrier chain drops it to zero.
  if (has_physical_overrides() && _compressed_materialization_observer != nullptr) {
    auto& publisher = _compressed_materialization_observer->get_event_publisher();
    // Inspect columns only when someone observes this activity.
    if (publisher.has_subscribers(sirius::event::event_type::compressed_materialization)) {
      auto const view = get_cudf_table_view(input_batch_ro);
      auto const width =
        std::min<std::size_t>(static_cast<std::size_t>(view.num_columns()), types.size());
      uint64_t narrow_columns = 0;
      for (std::size_t column_idx = 0; column_idx < width; ++column_idx) {
        auto const actual = view.column(static_cast<cudf::size_type>(column_idx)).type();
        // An alternate carrier (a widened SUM partial) is wider, not narrow.
        if (actual != sirius::require_cudf_type(types[column_idx]) &&
            !is_alternate_physical_type(column_idx, actual)) {
          ++narrow_columns;
        }
      }
      if (narrow_columns > 0) {
        publisher.publish_compressed_materialization(
          sirius::event::compressed_materialization_activity::partition_narrow_columns,
          narrow_columns);
      }
    }
  }
  // Contribute to the accumulated filter before scattering to partitions.
  if (_input_observer) {
    publication_id = _input_observer->before_transform(input, input_batch_ro, stream);
  }
  partitioned_results = gpu_partition_impl::hash_partition(input_batch_ro,
                                                           _spec.key_columns,
                                                           _spec.key_cast_types,
                                                           count,
                                                           stream,
                                                           *space,
                                                           batch_telemetry());
  std::vector<std::size_t> destinations(partitioned_results.size());
  std::iota(destinations.begin(), destinations.end(), std::size_t{0});
  auto result =
    std::make_unique<routed_operator_data>(std::move(partitioned_results), std::move(destinations));
  if (publication_id) { _input_observer->after_transform(*publication_id, *space, stream); }
  record_actual_bytes();
  return result;
}

void sirius_physical_partition::sink(operator_data const& input_data, ::cuda::stream_ref)
{
  nvtx_scoped_range nvtx_range{"sirius_physical_partition::sink"};
  auto const& output = dynamic_cast<routed_operator_data const&>(input_data);
  auto const routing = layout();
  if (!routing) { throw std::logic_error("partition sink has no layout"); }
  auto const& batches = output.get_data_batches();
  for (std::size_t i = 0; i < batches.size(); ++i) {
    auto const destination = output.destinations()[i];
    if (destination >= routing->placement->num_partitions()) {
      throw std::out_of_range("partition output destination is outside the layout");
    }
    for (auto const& next : next_port_after_sink) {
      next.next_operator->push_data_batch_partitioned(
        next.next_operator_port_name, batches[i], destination);
    }
  }
}

namespace {

/// `rows * (to_bytes / from_bytes)`, saturating; zero when `from_bytes` is zero.
uint64_t scale_rows(uint64_t rows, uint64_t to_bytes, uint64_t from_bytes)
{
  if (from_bytes == 0 || rows == 0) { return 0; }
  double const scaled =
    static_cast<double>(rows) * (static_cast<double>(to_bytes) / static_cast<double>(from_bytes));
  constexpr double kMaxRows = 9.0e18;  // Within uint64_t.
  if (!(scaled < kMaxRows)) { return static_cast<uint64_t>(kMaxRows); }
  return static_cast<uint64_t>(std::llround(scaled));
}

}  // namespace

partition_input_stats sirius_physical_partition::compute_input_totals()
{
  if (ports.find("default") == ports.end()) {
    throw std::runtime_error(
      "sirius_physical_partition::compute_input_totals() did not find default repo for id " +
      std::to_string(this->get_operator_id()));
  }
  auto& repo                   = ports.at("default")->repo;
  auto batch_ids               = repo->get_batch_ids(0);
  partition_input_stats totals = {};
  uint64_t counted_bytes       = 0;
  uint64_t uncounted_bytes     = 0;
  for (auto batch_id : batch_ids) {
    auto batch = repo->get_data_batch_by_id(batch_id, 0);
    if (batch) {
      auto ro = batch->to_read_only();
      if (ro.get_data()) {
        auto const bytes = ro.get_data()->get_size_in_bytes();
        totals.bytes += bytes;
        if (auto const rows = representation_num_rows(ro.get_data())) {
          totals.rows += *rows;
          counted_bytes += bytes;
        } else {
          uncounted_bytes += bytes;
        }
      }
    }
  }
  totals.rows += scale_rows(totals.rows, uncounted_bytes, counted_bytes);
  return totals;
}

std::optional<uint64_t> sirius_physical_partition::estimated_total_input_bytes()
{
  if (!_enable_size_estimation) { return std::nullopt; }
  if (ports.size() != 1) { return std::nullopt; }

  if (!_size_estimate.has_value()) {
    _size_estimate = pipeline::estimate_port_total_input_bytes(*this, ports.begin()->first);
    if (!_size_estimate.has_value()) { return std::nullopt; }
    SIRIUS_LOG_DEBUG(
      "sirius_physical_partition id {} projected {} bytes on its input port (exact={}, hops={}, "
      "samples={})",
      this->get_operator_id(),
      _size_estimate->bytes,
      _size_estimate->exact,
      _size_estimate->hops,
      _size_estimate->ratio_samples);
  }

  // An estimate cannot invalidate bytes already received.
  return std::max(static_cast<uint64_t>(_size_estimate->bytes), compute_total_bytes());
}

void sirius_physical_partition::set_layout(partition_layout resolved)
{
  std::lock_guard guard(lock);
  if (resolved.placement->num_partitions() >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("partition destination count exceeds the kernel limit");
  }
  if (resolved.mode == partition_mode::hash && _spec.mode != partition_mode::hash) {
    throw std::invalid_argument("hash layout requires hash input keys");
  }
  auto const installed = layout();
  if (installed) {
    if (installed->mode != resolved.mode || *installed->placement != *resolved.placement) {
      throw std::logic_error("partition layout is already resolved differently");
    }
    // A fixed layout may have been declared before repository materialization. Reapply its
    // original placement to endpoints once they exist, retaining the immutable routing decision.
    resolved = *installed;
  }
  auto placement = resolved.placement;
  for (auto const& next : next_port_after_sink) {
    if (auto* receiver =
          dynamic_cast<sirius_physical_partition_consumer_operator*>(next.next_operator)) {
      receiver->set_placement(placement);
    }
    auto* port = next.next_operator->try_get_port(next.next_operator_port_name);
    if (port && port->repo && placement->num_partitions() > port->repo->num_partitions()) {
      port->repo->set_num_partitions(placement->num_partitions());
    }
  }
  if (!installed) {
    _layout.store(std::make_shared<partition_layout const>(std::move(resolved)),
                  std::memory_order_release);
  }
}

bool sirius_physical_partition::input_complete() const
{
  return std::all_of(get_ports().begin(), get_ports().end(), [](auto const& p) {
    return !p->src_pipeline || p->src_pipeline->is_pipeline_finished();
  });
}

bool sirius_physical_partition::input_empty() const
{
  return std::all_of(get_ports().begin(), get_ports().end(), [](auto const& p) {
    return !p->repo || p->repo->total_size() == 0;
  });
}

std::optional<task_creation_hint> sirius_physical_partition::sizing_hint()
{
  std::lock_guard guard(lock);
  if (_enable_size_estimation && !estimated_total_input_bytes() && !input_complete()) {
    auto* source = ports.begin()->second->src_pipeline.get();
    if (source != nullptr) {
      return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA,
                                &source->get_operators()[0].get()};
    }
  }
  return sirius_physical_operator::get_next_task_hint();
}

partition_input_stats sirius_physical_partition::input_statistics(bool use_estimate)
{
  std::lock_guard guard(lock);
  auto received  = compute_input_totals();
  auto estimated = use_estimate ? estimated_total_input_bytes() : std::nullopt;
  _sizing_bytes  = estimated.value_or(received.bytes);
  _sizing_basis  = !estimated              ? sizing_basis::measured
                   : _size_estimate->exact ? sizing_basis::upstream_complete
                                           : sizing_basis::projected;
  return {_sizing_bytes,
          estimated ? scale_rows(received.rows, *estimated, received.bytes) : received.rows};
}

std::optional<task_creation_hint> sirius_physical_partition::get_next_task_hint()
{
  if (_exchange) {
    // A right-family build can publish before the complete probe fixes the layout. Follow only
    // the installed build stage and do not add a dependency through the sizing input.
    if (!layout() && !_exchange->is_sizing_input(*this) && children.size() == 1 &&
        children.front()->type == SiriusPhysicalOperatorType::BUILD_FILTER && ports.size() == 1) {
      auto const* input = ports.begin()->second;
      if (input->src_pipeline && !input->src_pipeline->is_pipeline_finished()) {
        auto* producer = &(input->src_pipeline->get_operators()[0].get());
        return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA, producer};
      }
    }
    if (auto readiness = _exchange->hint()) { return readiness; }
  } else {
    auto const fixed = layout();
    if (!fixed) {
      throw std::logic_error("partition has neither a fixed layout nor an input policy");
    }
    set_layout(*fixed);
  }
  return sirius_physical_operator::get_next_task_hint();
}

std::unique_ptr<operator_data> sirius_physical_partition::get_next_task_input_data()
{
  if (_exchange) {
    if (_exchange->prepare()) { return nullptr; }
  } else {
    auto const fixed = layout();
    if (!fixed) {
      throw std::logic_error("partition has neither a fixed layout nor an input policy");
    }
    set_layout(*fixed);
  }
  if (_input_observer && !_input_observer->before_claim()) { return nullptr; }
  return sirius_physical_operator::get_next_task_input_data();
}

void sirius_physical_partition::on_input_batch_pushed(std::string_view port,
                                                      cucascade::data_batch& batch)
{
  if (_input_observer) { _input_observer->on_arrival(port, batch); }
}

void sirius_physical_partition::on_finalize_operator()
{
  if (_input_observer) { _input_observer->finalize_input(); }
  if (!_enable_size_estimation) { return; }
  std::lock_guard<std::mutex> guard(lock);
  auto const actual = _actual_bytes.load(std::memory_order_relaxed);

  if (_sizing_basis == sizing_basis::measured) {
    SIRIUS_LOG_INFO(
      "sirius_physical_partition id {} size estimate: sized from measured input ({} bytes); "
      "no projection was available; partitions {}",
      this->get_operator_id(),
      _sizing_bytes,
      layout() ? layout()->placement->num_partitions() : 0);
    return;
  }

  double const error_pct = actual == 0
                             ? 0.0
                             : (static_cast<double>(_sizing_bytes) - static_cast<double>(actual)) /
                                 static_cast<double>(actual) * 100.0;
  SIRIUS_LOG_INFO(
    "sirius_physical_partition id {} size estimate: sized from {} {} bytes (exact={}, hops={}, "
    "samples={}), actual {} bytes, error {:+.1f}%, partitions {}",
    this->get_operator_id(),
    sizing_basis_name(_sizing_basis),
    _sizing_bytes,
    _size_estimate.has_value() && _size_estimate->exact,
    _size_estimate.has_value() ? _size_estimate->hops : 0,
    _size_estimate.has_value() ? _size_estimate->ratio_samples : 0,
    actual,
    error_pct,
    layout() ? layout()->placement->num_partitions() : 0);
}

std::size_t sirius_physical_partition::no_history_peak_memory_estimate(
  input_stats const& stats) const
{
  auto const resolved = layout();
  if (resolved &&
      (resolved->mode != partition_mode::hash || resolved->placement->num_partitions() == 1)) {
    return 0;
  }
  return sirius::memory::saturating_mul(stats.bytes, std::size_t{2});
}

}  // namespace op
}  // namespace sirius
