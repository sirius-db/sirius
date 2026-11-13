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

#include "op/sirius_physical_discard_sink.hpp"

#include "data/data_batch_utils.hpp"
#include "helper/type_conversions.hpp"
#include "log/logging.hpp"
#include "sirius/exception.hpp"
#include "telemetry/nvtx.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cucascade/memory/memory_space.hpp>

#include <cmath>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace sirius {
namespace op {

namespace {

//! One sort key of one row, on the host. `std::monostate` is NULL.
using host_key =
  std::variant<std::monostate, std::int64_t, std::uint64_t, double, __int128_t, std::string>;
//! The sort keys of one row, in `discard_sink_options::key_columns` order.
using host_key_row = std::vector<host_key>;

//! First and last key rows of the rows that arrived at one position.
struct key_range {
  host_key_row first;
  host_key_row last;
};

//! Task input of the discard sink. Its batches may be anywhere, so it keeps the GPU memory space
//! the task prepared it for: that is where the count row is allocated.
class discard_sink_input final : public pipelineable_operator_data {
 public:
  using pipelineable_operator_data::pipelineable_operator_data;

  void prepare_for_processing(const ::cucascade::memory::memory_space* requested_memory_space,
                              ::cuda::stream_ref stream) override
  {
    _task_space = const_cast<::cucascade::memory::memory_space*>(requested_memory_space);
    pipelineable_operator_data::prepare_for_processing(requested_memory_space, stream);
  }

  //! The memory space of the task, or nullptr before preparation or when the task named none.
  [[nodiscard]] ::cucascade::memory::memory_space* task_space() const noexcept
  {
    return _task_space;
  }

  [[nodiscard]] std::unique_ptr<pipelineable_operator_data> with_batches_preserving_rows(
    std::vector<std::shared_ptr<::cucascade::data_batch>> batches) const override
  {
    auto replacement = std::make_unique<discard_sink_input>(std::move(batches), get_position());
    replacement->adopt_row_identity(*this);
    replacement->set_materialize_on_gpu(materializes_on_gpu());
    replacement->_task_space = _task_space;
    return replacement;
  }

 private:
  ::cucascade::memory::memory_space* _task_space{nullptr};
};

//! Copies a valid scalar's value to the host.
struct host_key_reader {
  template <typename T>
  host_key operator()(const cudf::scalar& value, ::cuda::stream_ref stream) const
  {
    if constexpr (std::is_same_v<T, cudf::string_view>) {
      return static_cast<const cudf::string_scalar&>(value).to_string(stream);
    } else if constexpr (cudf::is_fixed_point<T>()) {
      return static_cast<__int128_t>(
        static_cast<const cudf::fixed_point_scalar<T>&>(value).value(stream));
    } else if constexpr (cudf::is_timestamp<T>()) {
      return static_cast<std::int64_t>(static_cast<const cudf::timestamp_scalar<T>&>(value)
                                         .value(stream)
                                         .time_since_epoch()
                                         .count());
    } else if constexpr (cudf::is_duration<T>()) {
      return static_cast<std::int64_t>(
        static_cast<const cudf::duration_scalar<T>&>(value).value(stream).count());
    } else if constexpr (std::is_floating_point_v<T>) {
      return static_cast<double>(static_cast<const cudf::numeric_scalar<T>&>(value).value(stream));
    } else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
      return static_cast<std::int64_t>(
        static_cast<const cudf::numeric_scalar<T>&>(value).value(stream));
    } else if constexpr (std::is_integral_v<T>) {
      return static_cast<std::uint64_t>(
        static_cast<const cudf::numeric_scalar<T>&>(value).value(stream));
    } else {
      throw not_implemented_exception(
        "sirius_physical_discard_sink: order check does not support this key type");
    }
  }
};

host_key_row read_key_row(const cudf::table_view& keys,
                          cudf::size_type row,
                          ::cuda::stream_ref stream,
                          rmm::device_async_resource_ref mr)
{
  host_key_row result;
  result.reserve(static_cast<std::size_t>(keys.num_columns()));
  for (auto const& column : keys) {
    auto const value = cudf::get_element(column, row, stream, mr);
    if (!value->is_valid(stream)) {
      result.emplace_back(std::monostate{});
      continue;
    }
    result.push_back(cudf::type_dispatcher(column.type(), host_key_reader{}, *value, stream));
  }
  return result;
}

//! Three-way comparison of two non-null values of one column, as cudf orders them: NaN after
//! every other value, strings by byte.
int compare_values(const host_key& lhs, const host_key& rhs)
{
  if (lhs.index() != rhs.index()) {
    throw internal_exception("sirius_physical_discard_sink: key type changed between batches");
  }
  return std::visit(
    [&rhs](auto const& left) -> int {
      using value_type  = std::decay_t<decltype(left)>;
      auto const& right = std::get<value_type>(rhs);
      if constexpr (std::is_same_v<value_type, std::monostate>) {
        return 0;
      } else if constexpr (std::is_same_v<value_type, std::string>) {
        auto const order = left.compare(right);
        return order < 0 ? -1 : (order > 0 ? 1 : 0);
      } else {
        if constexpr (std::is_same_v<value_type, double>) {
          auto const left_nan  = std::isnan(left);
          auto const right_nan = std::isnan(right);
          if (left_nan || right_nan) {
            return static_cast<int>(left_nan) - static_cast<int>(right_nan);
          }
        }
        return left < right ? -1 : (right < left ? 1 : 0);
      }
    },
    lhs);
}

//! Whether row @p lhs sorts strictly before row @p rhs. Follows cudf's row comparator, so it agrees
//! with `cudf::is_sorted`: the null placement is applied first and a descending column then
//! reverses the whole comparison, nulls included.
bool orders_before(const host_key_row& lhs,
                   const host_key_row& rhs,
                   const discard_sink_options& options)
{
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    auto const null_precedence =
      options.null_precedence.empty() ? cudf::null_order::BEFORE : options.null_precedence[i];
    auto const ascending =
      options.column_order.empty() || options.column_order[i] == cudf::order::ASCENDING;
    auto const lhs_null = std::holds_alternative<std::monostate>(lhs[i]);
    auto const rhs_null = std::holds_alternative<std::monostate>(rhs[i]);
    int order           = 0;
    if (lhs_null && rhs_null) {
      order = 0;
    } else if (lhs_null) {
      order = null_precedence == cudf::null_order::BEFORE ? -1 : 1;
    } else if (rhs_null) {
      order = null_precedence == cudf::null_order::AFTER ? -1 : 1;
    } else {
      order = compare_values(lhs[i], rhs[i]);
    }
    if (order == 0) { continue; }
    return ascending ? order < 0 : order > 0;
  }
  return false;
}

}  // namespace

struct sirius_physical_discard_sink::order_check_state {
  std::mutex mutex;
  //! Position -> key range. Tasks that share a position keep their arrival order.
  std::multimap<std::size_t, key_range> ranges;
  bool finished{false};
};

sirius_physical_discard_sink::sirius_physical_discard_sink(
  duckdb::vector<sirius::logical_type> input_types,
  discard_sink_options options,
  std::size_t estimated_cardinality)
  : sirius_physical_operator(SiriusPhysicalOperatorType::DISCARD_SINK,
                             {sirius::from_duckdb(duckdb::LogicalType::BIGINT)},
                             estimated_cardinality),
    _options(std::move(options)),
    _input_types(std::move(input_types)),
    _state(std::make_unique<order_check_state>())
{
  auto const num_keys = _options.key_columns.size();
  if ((!_options.column_order.empty() && _options.column_order.size() != num_keys) ||
      (!_options.null_precedence.empty() && _options.null_precedence.size() != num_keys)) {
    throw std::invalid_argument(
      "sirius_physical_discard_sink: column_order and null_precedence must be empty or have one "
      "entry per key column");
  }
}

sirius_physical_discard_sink::~sirius_physical_discard_sink() = default;

std::string sirius_physical_discard_sink::get_name() const { return "DISCARD_SINK"; }

std::unique_ptr<operator_data> sirius_physical_discard_sink::execute(
  const operator_data& input_data, ::cuda::stream_ref stream)
{
  nvtx_scoped_range nvtx_range{"sirius_physical_discard_sink::execute"};
  const auto* input = dynamic_cast<const pipelineable_operator_data*>(&input_data);
  if (input == nullptr) {
    throw internal_exception(
      "sirius_physical_discard_sink::execute: expected pipelineable_operator_data");
  }
  auto const check_order = _options.check_order && !_options.key_columns.empty();

  // The read locks are the prepared ones when the task prepared its input; a batch that was not
  // prepared is locked in the tier it is in and counted from its metadata.
  auto const batches = input->get_read_only_batches();

  // The count row goes to the task's GPU memory space, or failing that (an input that was not
  // prepared) to the space of the first input batch that is on the GPU.
  ::cucascade::memory::memory_space* gpu_space = nullptr;
  if (const auto* own_input = dynamic_cast<const discard_sink_input*>(input)) {
    auto* task_space = own_input->task_space();
    if (task_space != nullptr && task_space->get_tier() == ::cucascade::memory::Tier::GPU) {
      gpu_space = task_space;
    }
  }

  std::uint64_t rows       = 0;
  std::uint64_t bytes      = 0;
  std::uint64_t violations = 0;
  std::optional<key_range> range;
  for (auto const& batch : batches) {
    auto const* data = batch.get_data();
    if (data == nullptr) { continue; }
    auto* space         = batch.get_memory_space();
    auto const on_gpu   = batch.get_current_tier() == ::cucascade::memory::Tier::GPU;
    auto const num_rows = representation_num_rows(*data);
    if (!num_rows) {
      throw internal_exception("sirius_physical_discard_sink: cannot determine batch row count");
    }
    rows += *num_rows;
    bytes += data->get_uncompressed_data_size_in_bytes();
    if (on_gpu && gpu_space == nullptr) { gpu_space = space; }

    if (!check_order || *num_rows == 0) { continue; }
    if (!on_gpu) {
      throw internal_exception(
        "sirius_physical_discard_sink: the order check needs its input on the GPU");
    }
    auto const keys = get_cudf_table_view(batch).select(_options.key_columns);
    if (!cudf::is_sorted(keys, _options.column_order, _options.null_precedence, stream)) {
      ++violations;
    }
    auto const mr = space->get_default_allocator();
    auto first    = read_key_row(keys, 0, stream, mr);
    auto last     = read_key_row(keys, keys.num_rows() - 1, stream, mr);
    if (range) {
      // Batches of one task arrive in order: check the seam between them now.
      if (orders_before(first, range->last, _options)) { ++violations; }
      range->last = std::move(last);
    } else {
      range.emplace(key_range{std::move(first), std::move(last)});
    }
  }

  std::vector<std::shared_ptr<::cucascade::data_batch>> output;
  if (!batches.empty()) {
    if (gpu_space == nullptr) {
      // Reached only by an input that did not come from get_next_task_input_data() or was not
      // prepared.
      throw internal_exception(
        "sirius_physical_discard_sink: the task has no GPU memory space and no input batch is on "
        "the GPU, so there is nowhere to put the count row");
    }
    auto const mr = gpu_space->get_default_allocator();
    cudf::numeric_scalar<std::int64_t> count(static_cast<std::int64_t>(rows), true, stream, mr);
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(cudf::make_column_from_scalar(count, 1, stream, mr));
    output.push_back(sirius::make_data_batch(
      std::make_unique<cudf::table>(std::move(columns)), *gpu_space, stream, batch_telemetry()));
  }

  // Record only once nothing can fail any more, so a task that is retried is counted once.
  if (range && input->get_position()) {
    // Without a position there is no order between tasks to check.
    std::lock_guard<std::mutex> guard(_state->mutex);
    _state->ranges.emplace(*input->get_position(), std::move(*range));
  }
  _rows_discarded.fetch_add(rows);
  _bytes_discarded.fetch_add(bytes);
  if (violations != 0) { _order_violations.fetch_add(violations); }
  return std::make_unique<pipelineable_operator_data>(std::move(output), input->get_position());
}

MemoryBarrierType sirius_physical_discard_sink::input_barrier_for(
  sirius_physical_operator const& /*producer*/) const
{
  return MemoryBarrierType::PIPELINE;
}

port_residency_policy sirius_physical_discard_sink::input_residency_policy(
  std::string_view /*port_id*/) const
{
  auto const check_order = _options.check_order && !_options.key_columns.empty();
  return {.materialize_inputs_on_gpu = check_order, .downgrade_priority = -100};
}

std::unique_ptr<operator_data> sirius_physical_discard_sink::get_next_task_input_data()
{
  auto input = sirius_physical_operator::get_next_task_input_data();
  if (input == nullptr) { return nullptr; }
  auto const& popped = dynamic_cast<const pipelineable_operator_data&>(*input);
  auto own = std::make_unique<discard_sink_input>(popped.get_data_batches(), popped.get_position());
  own->set_materialize_on_gpu(popped.materializes_on_gpu());
  if (auto const device = popped.get_preferred_device_id()) {
    own->set_preferred_device_id(*device);
  }
  return own;
}

std::size_t sirius_physical_discard_sink::no_history_peak_memory_estimate(
  const op::input_stats& /*stats*/) const
{
  // The count row, plus the key scalars read by the order check.
  return std::size_t{1} << 20;
}

sirius::OrderPreservationType sirius_physical_discard_sink::output_order_preservation() const
{
  return sirius::OrderPreservationType::INSERTION_ORDER;
}

std::uint64_t sirius_physical_discard_sink::rows_discarded() const noexcept
{
  return _rows_discarded.load();
}

std::uint64_t sirius_physical_discard_sink::bytes_discarded() const noexcept
{
  return _bytes_discarded.load();
}

std::uint64_t sirius_physical_discard_sink::order_violations() const noexcept
{
  return _order_violations.load();
}

void sirius_physical_discard_sink::finish()
{
  std::lock_guard<std::mutex> guard(_state->mutex);
  if (_state->finished) { return; }
  _state->finished = true;

  std::uint64_t violations  = 0;
  const key_range* previous = nullptr;
  for (auto const& [position, range] : _state->ranges) {
    if (previous != nullptr && orders_before(range.first, previous->last, _options)) {
      ++violations;
    }
    previous = &range;
  }
  if (violations != 0) { _order_violations.fetch_add(violations); }
  _state->ranges.clear();

  SIRIUS_LOG_INFO("[discard_sink] rows={} bytes={} order_violations={}",
                  _rows_discarded.load(),
                  _bytes_discarded.load(),
                  _order_violations.load());
}

}  // namespace op
}  // namespace sirius
