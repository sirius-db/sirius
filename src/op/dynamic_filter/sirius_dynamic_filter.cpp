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

#include <cudf/ast/expressions.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/search.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <rmm/cuda_device.hpp>

#include <cucascade/memory/memory_space.hpp>
#include <log/logging.hpp>
#include <op/dynamic_filter/dynamic_filter_device.hpp>
#include <op/dynamic_filter/sirius_dynamic_filter.hpp>

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace sirius::op {

namespace {

cudf::ast::expression const& and_join(cudf::ast::tree& tree,
                                      cudf::ast::expression const& lhs,
                                      cudf::ast::expression const& rhs)
{
  return tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::LOGICAL_AND, lhs, rhs);
}

cudf::ast::expression const& or_join(cudf::ast::tree& tree,
                                     cudf::ast::expression const& lhs,
                                     cudf::ast::expression const& rhs)
{
  return tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::LOGICAL_OR, lhs, rhs);
}

template <typename ScalarT>
auto scalar_value_to_host(cudf::scalar const& source,
                          int source_device,
                          ::cuda::stream_ref source_stream)
{
  rmm::cuda_set_device_raii source_guard{rmm::cuda_device_id{source_device}};
  auto const& typed = static_cast<ScalarT const&>(source);
  auto const valid  = typed.is_valid(source_stream);
  return std::pair{typed.value(source_stream), valid};
}

std::unique_ptr<cudf::scalar> clone_scalar_to_device(cudf::scalar const& source,
                                                     int source_device,
                                                     ::cuda::stream_ref source_stream,
                                                     ::cuda::stream_ref target_stream,
                                                     rmm::device_async_resource_ref target_mr)
{
  switch (source.type().id()) {
    case cudf::type_id::INT8: {
      auto const [value, valid] =
        scalar_value_to_host<cudf::numeric_scalar<int8_t>>(source, source_device, source_stream);
      return std::make_unique<cudf::numeric_scalar<int8_t>>(value, valid, target_stream, target_mr);
    }
    case cudf::type_id::INT16: {
      auto const [value, valid] =
        scalar_value_to_host<cudf::numeric_scalar<int16_t>>(source, source_device, source_stream);
      return std::make_unique<cudf::numeric_scalar<int16_t>>(
        value, valid, target_stream, target_mr);
    }
    case cudf::type_id::INT32: {
      auto const [value, valid] =
        scalar_value_to_host<cudf::numeric_scalar<int32_t>>(source, source_device, source_stream);
      return std::make_unique<cudf::numeric_scalar<int32_t>>(
        value, valid, target_stream, target_mr);
    }
    case cudf::type_id::INT64: {
      auto const [value, valid] =
        scalar_value_to_host<cudf::numeric_scalar<int64_t>>(source, source_device, source_stream);
      return std::make_unique<cudf::numeric_scalar<int64_t>>(
        value, valid, target_stream, target_mr);
    }
    case cudf::type_id::UINT8: {
      auto const [value, valid] =
        scalar_value_to_host<cudf::numeric_scalar<uint8_t>>(source, source_device, source_stream);
      return std::make_unique<cudf::numeric_scalar<uint8_t>>(
        value, valid, target_stream, target_mr);
    }
    case cudf::type_id::UINT16: {
      auto const [value, valid] =
        scalar_value_to_host<cudf::numeric_scalar<uint16_t>>(source, source_device, source_stream);
      return std::make_unique<cudf::numeric_scalar<uint16_t>>(
        value, valid, target_stream, target_mr);
    }
    case cudf::type_id::UINT32: {
      auto const [value, valid] =
        scalar_value_to_host<cudf::numeric_scalar<uint32_t>>(source, source_device, source_stream);
      return std::make_unique<cudf::numeric_scalar<uint32_t>>(
        value, valid, target_stream, target_mr);
    }
    case cudf::type_id::UINT64: {
      auto const [value, valid] =
        scalar_value_to_host<cudf::numeric_scalar<uint64_t>>(source, source_device, source_stream);
      return std::make_unique<cudf::numeric_scalar<uint64_t>>(
        value, valid, target_stream, target_mr);
    }
    case cudf::type_id::FLOAT32: {
      auto const [value, valid] =
        scalar_value_to_host<cudf::numeric_scalar<float>>(source, source_device, source_stream);
      return std::make_unique<cudf::numeric_scalar<float>>(value, valid, target_stream, target_mr);
    }
    case cudf::type_id::FLOAT64: {
      auto const [value, valid] =
        scalar_value_to_host<cudf::numeric_scalar<double>>(source, source_device, source_stream);
      return std::make_unique<cudf::numeric_scalar<double>>(value, valid, target_stream, target_mr);
    }
    case cudf::type_id::BOOL8: {
      auto const [value, valid] =
        scalar_value_to_host<cudf::numeric_scalar<bool>>(source, source_device, source_stream);
      return std::make_unique<cudf::numeric_scalar<bool>>(value, valid, target_stream, target_mr);
    }
    case cudf::type_id::TIMESTAMP_DAYS: {
      auto const [value, valid] = scalar_value_to_host<cudf::timestamp_scalar<cudf::timestamp_D>>(
        source, source_device, source_stream);
      return std::make_unique<cudf::timestamp_scalar<cudf::timestamp_D>>(
        value, valid, target_stream, target_mr);
    }
    case cudf::type_id::TIMESTAMP_SECONDS: {
      auto const [value, valid] = scalar_value_to_host<cudf::timestamp_scalar<cudf::timestamp_s>>(
        source, source_device, source_stream);
      return std::make_unique<cudf::timestamp_scalar<cudf::timestamp_s>>(
        value, valid, target_stream, target_mr);
    }
    case cudf::type_id::TIMESTAMP_MILLISECONDS: {
      auto const [value, valid] = scalar_value_to_host<cudf::timestamp_scalar<cudf::timestamp_ms>>(
        source, source_device, source_stream);
      return std::make_unique<cudf::timestamp_scalar<cudf::timestamp_ms>>(
        value, valid, target_stream, target_mr);
    }
    case cudf::type_id::TIMESTAMP_MICROSECONDS: {
      auto const [value, valid] = scalar_value_to_host<cudf::timestamp_scalar<cudf::timestamp_us>>(
        source, source_device, source_stream);
      return std::make_unique<cudf::timestamp_scalar<cudf::timestamp_us>>(
        value, valid, target_stream, target_mr);
    }
    case cudf::type_id::TIMESTAMP_NANOSECONDS: {
      auto const [value, valid] = scalar_value_to_host<cudf::timestamp_scalar<cudf::timestamp_ns>>(
        source, source_device, source_stream);
      return std::make_unique<cudf::timestamp_scalar<cudf::timestamp_ns>>(
        value, valid, target_stream, target_mr);
    }
    case cudf::type_id::DECIMAL32: {
      auto const [value, valid] =
        scalar_value_to_host<cudf::fixed_point_scalar<numeric::decimal32>>(
          source, source_device, source_stream);
      return std::make_unique<cudf::fixed_point_scalar<numeric::decimal32>>(
        value, numeric::scale_type{source.type().scale()}, valid, target_stream, target_mr);
    }
    case cudf::type_id::DECIMAL64: {
      auto const [value, valid] =
        scalar_value_to_host<cudf::fixed_point_scalar<numeric::decimal64>>(
          source, source_device, source_stream);
      return std::make_unique<cudf::fixed_point_scalar<numeric::decimal64>>(
        value, numeric::scale_type{source.type().scale()}, valid, target_stream, target_mr);
    }
    case cudf::type_id::DECIMAL128: {
      auto const [value, valid] =
        scalar_value_to_host<cudf::fixed_point_scalar<numeric::decimal128>>(
          source, source_device, source_stream);
      return std::make_unique<cudf::fixed_point_scalar<numeric::decimal128>>(
        value, numeric::scale_type{source.type().scale()}, valid, target_stream, target_mr);
    }
    case cudf::type_id::STRING: {
      std::string value;
      bool valid = false;
      {
        rmm::cuda_set_device_raii source_guard{rmm::cuda_device_id{source_device}};
        auto const& scalar = static_cast<cudf::string_scalar const&>(source);
        valid              = scalar.is_valid(source_stream);
        value              = scalar.to_string(source_stream);
      }
      return std::make_unique<cudf::string_scalar>(value, valid, target_stream, target_mr);
    }
    default:
      throw std::runtime_error(
        "[sirius_dynamic_zone_map_filter] Unsupported scalar type for device replication");
  }
}

cudf::ast::expression const& emplace_literal_from_scalar(cudf::ast::tree& tree, cudf::scalar& s)
{
  switch (s.type().id()) {
    case cudf::type_id::INT8:
      return tree.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<int8_t>&>(s));
    case cudf::type_id::INT16:
      return tree.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<int16_t>&>(s));
    case cudf::type_id::INT32:
      return tree.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<int32_t>&>(s));
    case cudf::type_id::INT64:
      return tree.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<int64_t>&>(s));
    case cudf::type_id::UINT8:
      return tree.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<uint8_t>&>(s));
    case cudf::type_id::UINT16:
      return tree.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<uint16_t>&>(s));
    case cudf::type_id::UINT32:
      return tree.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<uint32_t>&>(s));
    case cudf::type_id::UINT64:
      return tree.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<uint64_t>&>(s));
    case cudf::type_id::FLOAT32:
      return tree.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<float>&>(s));
    case cudf::type_id::FLOAT64:
      return tree.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<double>&>(s));
    case cudf::type_id::BOOL8:
      return tree.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<bool>&>(s));
    case cudf::type_id::TIMESTAMP_DAYS:
      return tree.emplace<cudf::ast::literal>(
        static_cast<cudf::timestamp_scalar<cudf::timestamp_D>&>(s));
    case cudf::type_id::TIMESTAMP_SECONDS:
      return tree.emplace<cudf::ast::literal>(
        static_cast<cudf::timestamp_scalar<cudf::timestamp_s>&>(s));
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      return tree.emplace<cudf::ast::literal>(
        static_cast<cudf::timestamp_scalar<cudf::timestamp_ms>&>(s));
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      return tree.emplace<cudf::ast::literal>(
        static_cast<cudf::timestamp_scalar<cudf::timestamp_us>&>(s));
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      return tree.emplace<cudf::ast::literal>(
        static_cast<cudf::timestamp_scalar<cudf::timestamp_ns>&>(s));
    case cudf::type_id::DECIMAL32:
      return tree.emplace<cudf::ast::literal>(
        static_cast<cudf::fixed_point_scalar<numeric::decimal32>&>(s));
    case cudf::type_id::DECIMAL64:
      return tree.emplace<cudf::ast::literal>(
        static_cast<cudf::fixed_point_scalar<numeric::decimal64>&>(s));
    case cudf::type_id::DECIMAL128:
      return tree.emplace<cudf::ast::literal>(
        static_cast<cudf::fixed_point_scalar<numeric::decimal128>&>(s));
    case cudf::type_id::STRING:
      return tree.emplace<cudf::ast::literal>(static_cast<cudf::string_scalar&>(s));
    default:
      throw std::runtime_error(
        "[sirius_dynamic_zone_map_filter] Unsupported scalar type for AST literal");
  }
}

}  // namespace

cudf::ast::tree sirius_ast_lowerable::to_standalone_ast(
  std::function<cudf::ast::expression const&(cudf::ast::tree&)> const& column_ref_factory) const
{
  cudf::ast::tree tree;
  auto const& col_ref = column_ref_factory(tree);
  (void)to_ast(tree, col_ref);
  return tree;
}

// The allowlist is exactly the type set clone_scalar_to_device and emplace_literal_from_scalar
// handle, minus the floating-point ids the declaration excludes.
bool sirius_dynamic_zone_map_filter::supports(cudf::data_type t) noexcept
{
  switch (t.id()) {
    case cudf::type_id::INT8:
    case cudf::type_id::INT16:
    case cudf::type_id::INT32:
    case cudf::type_id::INT64:
    case cudf::type_id::UINT8:
    case cudf::type_id::UINT16:
    case cudf::type_id::UINT32:
    case cudf::type_id::UINT64:
    case cudf::type_id::BOOL8:
    case cudf::type_id::TIMESTAMP_DAYS:
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
    case cudf::type_id::DECIMAL32:
    case cudf::type_id::DECIMAL64:
    case cudf::type_id::DECIMAL128:
    case cudf::type_id::STRING: return true;
    default: return false;
  }
}

struct sirius_dynamic_zone_map_filter::device_zones {
  int device_id = -1;
  std::vector<zone_map_entry> zones;

  ~device_zones() noexcept
  {
    if (device_id < 0) { return; }
    try {
      rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device_id}};
      zones.clear();
    } catch (...) {
      // Leak scalars rather than free them on the wrong device.
      for (auto& zone : zones) {
        (void)zone.min.release();
        (void)zone.max.release();
      }
      zones.clear();
    }
  }
};

sirius_dynamic_zone_map_filter::sirius_dynamic_zone_map_filter(std::vector<zone_map_entry> zones,
                                                               bool inclusive_min,
                                                               bool inclusive_max)
  : _zones(std::move(zones)), _inclusive_min(inclusive_min), _inclusive_max(inclusive_max)
{
  if (_zones.empty()) {
    throw std::invalid_argument("[sirius_dynamic_zone_map_filter] At least one zone is required");
  }
  for (auto const& z : _zones) {
    if (!z.min || !z.max) {
      throw std::invalid_argument(
        "[sirius_dynamic_zone_map_filter] Every zone must have non-null min and max");
    }
    if (z.min->type() != z.max->type()) {
      throw std::invalid_argument(
        "[sirius_dynamic_zone_map_filter] Zone min and max must share one type");
    }
    // to_ast lowers every zone against the same column reference, so one type spans all zones.
    if (z.min->type() != _zones.front().min->type()) {
      throw std::invalid_argument(
        "[sirius_dynamic_zone_map_filter] All zones must share one bound type");
    }
    if (!supports(z.min->type())) {
      throw std::invalid_argument("[sirius_dynamic_zone_map_filter] Unsupported zone bound type");
    }
  }
  if (cudaGetDevice(&_source_device) != cudaSuccess) {
    throw std::runtime_error("[sirius_dynamic_zone_map_filter] failed to identify source device.");
  }
}

sirius_dynamic_zone_map_filter::~sirius_dynamic_zone_map_filter() noexcept
{
  // Replica destructors select their owning devices.
  _replicas.clear();
  if (_source_device >= 0) {
    try {
      rmm::cuda_set_device_raii guard{rmm::cuda_device_id{_source_device}};
      _zones.clear();
    } catch (...) {
      // Leak scalars rather than free them on the wrong device.
      for (auto& zone : _zones) {
        (void)zone.min.release();
        (void)zone.max.release();
      }
      _zones.clear();
    }
  }
}

bool sirius_dynamic_zone_map_filter::is_available_on_device(int device_id) const noexcept
{
  device_id = detail::resolve_dynamic_filter_device_id(device_id);
  if (device_id == _source_device) { return true; }
  return std::any_of(_replicas.begin(), _replicas.end(), [device_id](auto const& replica) {
    return replica->device_id == device_id;
  });
}

void sirius_dynamic_zone_map_filter::replicate_to_devices(
  std::span<dynamic_filter_replica_space const> spaces)
{
  if (spaces.empty()) { return; }

  auto const source = std::find_if(spaces.begin(), spaces.end(), [this](auto const& target) {
    return target.get_gpu_space().get_device_id() == _source_device;
  });
  if (source == spaces.end()) {
    SIRIUS_LOG_WARN(
      "[sirius_dynamic_zone_map_filter] source GPU {} has no planned memory space; remote "
      "replicas are unavailable and will be skipped.",
      _source_device);
    return;
  }

  // Publication synchronizes construction before replication.
  rmm::cuda_set_device_raii source_guard{rmm::cuda_device_id{_source_device}};
  auto const source_stream = source->get_gpu_space().acquire_stream();

  for (auto const& target : spaces) {
    auto const& target_space = target.get_gpu_space();
    auto const device_id     = target_space.get_device_id();
    if (is_available_on_device(device_id)) { continue; }
    try {
      auto replica       = std::make_unique<device_zones>();
      replica->device_id = device_id;

      {
        rmm::cuda_set_device_raii target_guard{rmm::cuda_device_id{device_id}};
        auto const target_stream = target_space.acquire_stream();
        auto const target_mr     = target_space.get_default_allocator();
        replica->zones.reserve(_zones.size());
        for (auto const& zone : _zones) {
          replica->zones.push_back(
            {clone_scalar_to_device(
               *zone.min, _source_device, source_stream, target_stream, target_mr),
             clone_scalar_to_device(
               *zone.max, _source_device, source_stream, target_stream, target_mr)});
        }
        target_stream.sync();
      }
      SIRIUS_LOG_DEBUG("[sirius_dynamic_zone_map_filter] replicated {} zone(s) GPU {} -> GPU {}.",
                       _zones.size(),
                       _source_device,
                       device_id);
      _replicas.push_back(std::move(replica));
    } catch (std::exception const& e) {
      SIRIUS_LOG_WARN(
        "[sirius_dynamic_zone_map_filter] replica GPU {} -> GPU {} unavailable: {}. "
        "That GPU will skip this optional filter.",
        _source_device,
        device_id,
        e.what());
    }
  }
}

cudf::ast::expression const& sirius_dynamic_zone_map_filter::to_ast(
  cudf::ast::tree& tree, cudf::ast::expression const& column_ref, int device_id) const
{
  auto const lower_op =
    _inclusive_min ? cudf::ast::ast_operator::GREATER_EQUAL : cudf::ast::ast_operator::GREATER;
  auto const upper_op =
    _inclusive_max ? cudf::ast::ast_operator::LESS_EQUAL : cudf::ast::ast_operator::LESS;

  device_id                   = detail::resolve_dynamic_filter_device_id(device_id);
  auto const* device_zone_map = &_zones;
  if (device_id != _source_device) {
    auto const it = std::find_if(_replicas.begin(), _replicas.end(), [device_id](auto const& r) {
      return r->device_id == device_id;
    });
    if (it == _replicas.end()) {
      throw std::runtime_error("[sirius_dynamic_zone_map_filter] no replica for consumer device");
    }
    device_zone_map = &(*it)->zones;
  }

  cudf::ast::expression const* result = nullptr;
  for (auto const& z : *device_zone_map) {
    auto const& min_lit   = emplace_literal_from_scalar(tree, *z.min);
    auto const& max_lit   = emplace_literal_from_scalar(tree, *z.max);
    auto const& lo        = tree.emplace<cudf::ast::operation>(lower_op, column_ref, min_lit);
    auto const& hi        = tree.emplace<cudf::ast::operation>(upper_op, column_ref, max_lit);
    auto const& zone_pred = and_join(tree, lo, hi);
    result                = (result == nullptr) ? &zone_pred : &or_join(tree, *result, zone_pred);
  }
  return *result;
}

//===----------sirius_dynamic_filter_set::state----------===//
/// @brief The shared channel state of a dynamic filter set
struct sirius_dynamic_filter_set::state {
  struct producer_state {
    bool terminal     = false;
    completion result = completion::skipped;
  };

  mutable std::mutex mutex;
  std::unordered_map<std::size_t, std::vector<std::shared_ptr<sirius_dynamic_filter const>>>
    filters;
  std::unordered_set<std::size_t> ignored_columns;
  std::set<std::size_t> planned_columns;
  std::vector<producer_state> producers;
  std::size_t completed_producers = 0;
  bool registration_frozen        = false;
  std::atomic<std::size_t> filter_count{0};
  std::atomic<std::size_t> producer_count{0};
  std::atomic<bool> unscoped{false};
  std::atomic<bool> accepting{true};
};

//===----------sirius_dynamic_filter_set----------===//
sirius_dynamic_filter_set::sirius_dynamic_filter_set() : _state(std::make_shared<state>()) {}

sirius_dynamic_filter_set::producer::producer(std::shared_ptr<state> channel,
                                              std::size_t index) noexcept
  : _channel(std::move(channel)), _index(index)
{
}

sirius_dynamic_filter_set::producer::producer(producer&& other) noexcept = default;

sirius_dynamic_filter_set::producer& sirius_dynamic_filter_set::producer::operator=(
  producer&& other) noexcept
{
  if (this != &other) {
    finish();
    _channel = std::move(other._channel);
    _index   = other._index;
  }
  return *this;
}

sirius_dynamic_filter_set::producer::~producer()
{
  // finish() with skipped completion
  finish();
}

bool sirius_dynamic_filter_set::producer::push_filter(
  std::size_t col_idx, std::shared_ptr<sirius_dynamic_filter const> filter) const
{
  if (!_channel || !filter) { return false; }
  std::scoped_lock lock(_channel->mutex);
  if (_channel->producers[_index].terminal ||
      !_channel->accepting.load(std::memory_order_relaxed) ||
      _channel->ignored_columns.contains(col_idx)) {
    return false;
  }
  _channel->filters[col_idx].push_back(std::move(filter));
  _channel->filter_count.fetch_add(1, std::memory_order_release);
  return true;
}

void sirius_dynamic_filter_set::producer::finish(completion result) const noexcept
{
  if (!_channel) { return; }
  std::scoped_lock lock(_channel->mutex);
  auto& slot = _channel->producers[_index];
  if (slot.terminal) { return; }
  slot.result   = result;
  slot.terminal = true;
  ++_channel->completed_producers;
}

dynamic_filter_snapshot sirius_dynamic_filter_set::snapshot() const
{
  std::scoped_lock lock(_state->mutex);
  dynamic_filter_snapshot result;
  result._entries.reserve(_state->filter_count.load(std::memory_order_relaxed));
  for (auto const& [column, filters] : _state->filters) {
    for (auto const& filter : filters) {
      result._entries.push_back({column, filter});
    }
  }
  result._terminal =
    _state->registration_frozen && _state->completed_producers == _state->producers.size();
  return result;
}

void sirius_dynamic_filter_set::ignore_columns(std::vector<std::size_t> const& cols)
{
  std::scoped_lock lock(_state->mutex);
  _state->ignored_columns.insert(cols.begin(), cols.end());
}

sirius_dynamic_filter_set::producer sirius_dynamic_filter_set::register_producer(
  std::vector<std::size_t> planned_target_columns)
{
  std::scoped_lock lock(_state->mutex);
  if (_state->registration_frozen) {
    throw std::logic_error("Dynamic-filter producers must register before execution");
  }
  if (planned_target_columns.empty()) {
    _state->unscoped.store(true, std::memory_order_release);
  } else {
    _state->planned_columns.insert(planned_target_columns.begin(), planned_target_columns.end());
  }
  auto const index = _state->producers.size();
  _state->producers.emplace_back();
  _state->producer_count.fetch_add(1, std::memory_order_release);
  return producer{_state, index};
}

void sirius_dynamic_filter_set::freeze_registration() noexcept
{
  std::scoped_lock lock(_state->mutex);
  _state->registration_frozen = true;
}

bool sirius_dynamic_filter_set::has_producers() const noexcept
{
  return _state->producer_count.load(std::memory_order_acquire) != 0;
}

bool sirius_dynamic_filter_set::has_unscoped_producer() const noexcept
{
  return _state->unscoped.load(std::memory_order_acquire);
}

bool sirius_dynamic_filter_set::accepting_filters() const noexcept
{
  return _state->accepting.load(std::memory_order_acquire);
}

bool sirius_dynamic_filter_set::has_filters() const noexcept { return filter_count() != 0; }

std::size_t sirius_dynamic_filter_set::filter_count() const noexcept
{
  return _state->filter_count.load(std::memory_order_acquire);
}

std::vector<std::size_t> sirius_dynamic_filter_set::planned_target_columns() const
{
  std::scoped_lock lock(_state->mutex);
  return {_state->planned_columns.begin(), _state->planned_columns.end()};
}

void sirius_dynamic_filter_set::close_for_new_filters()
{
  std::scoped_lock lock(_state->mutex);
  _state->accepting.store(false, std::memory_order_release);
}

std::vector<std::shared_ptr<sirius_dynamic_filter const>>
sirius_dynamic_filter_set::filters_for_column(std::size_t col_idx) const
{
  std::scoped_lock lock(_state->mutex);
  auto it = _state->filters.find(col_idx);
  if (it == _state->filters.end()) { return {}; }
  return it->second;
}

std::vector<std::size_t> sirius_dynamic_filter_set::filtered_columns() const
{
  std::scoped_lock lock(_state->mutex);
  std::vector<std::size_t> out;
  out.reserve(_state->filters.size());
  for (auto const& [k, _] : _state->filters) {
    out.push_back(k);
  }
  return out;
}

bool sirius_dynamic_filter_set::empty() const { return !has_filters(); }

cudf::ast::expression const& merge_ast_dynamic_filters_into_tree(
  cudf::ast::tree& tree,
  cudf::ast::expression const& existing_root,
  dynamic_filter_snapshot const& filters,
  column_ref_resolver_fn const& column_ref_resolver)
{
  if (filters.empty()) { return existing_root; }
  auto const device_id = detail::resolve_dynamic_filter_device_id(-1);

  std::vector<std::size_t> cols;
  for (auto const& entry : filters.entries()) {
    cols.push_back(entry.column_index);
  }
  std::ranges::sort(cols);
  cols.erase(std::unique(cols.begin(), cols.end()), cols.end());

  cudf::ast::expression const* dynamic_root = nullptr;
  for (auto const col_idx : cols) {
    cudf::ast::expression const* column_ref = nullptr;
    cudf::ast::expression const* per_col    = nullptr;
    for (auto const& entry : filters.entries()) {
      if (entry.column_index != col_idx) { continue; }
      auto const& f = entry.filter;
      if (!f->is_available_on_device(device_id)) { continue; }
      auto const* lowerable = dynamic_cast<sirius_ast_lowerable const*>(f.get());
      if (!lowerable) { continue; }
      if (!column_ref) { column_ref = &column_ref_resolver(col_idx); }
      auto const& fragment = lowerable->to_ast(tree, *column_ref, device_id);
      per_col              = per_col ? &and_join(tree, *per_col, fragment) : &fragment;
    }
    if (!per_col) { continue; }
    dynamic_root = dynamic_root ? &and_join(tree, *dynamic_root, *per_col) : per_col;
  }
  if (!dynamic_root) { return existing_root; }
  return and_join(tree, existing_root, *dynamic_root);
}

}  // namespace sirius::op
