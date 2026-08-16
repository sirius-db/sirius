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
#include <op/dynamic_filter/dynamic_filter_device.hpp>
#include <op/dynamic_filter/dynamic_filter_replica_reservation.hpp>
#include <op/dynamic_filter/sirius_dynamic_filter.hpp>

// cudf
#include <cudf/ast/expressions.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/search.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/wrappers/timestamps.hpp>

// cucascade
#include <cucascade/memory/memory_space.hpp>

// rmm
#include <rmm/cuda_device.hpp>

// sirius
#include <log/logging.hpp>

// standard library
#include <algorithm>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

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
                          rmm::cuda_stream_view source_stream)
{
  rmm::cuda_set_device_raii source_guard{rmm::cuda_device_id{source_device}};
  auto const& typed = static_cast<ScalarT const&>(source);
  auto const valid  = typed.is_valid(source_stream);
  return std::pair{typed.value(source_stream), valid};
}

std::unique_ptr<cudf::scalar> clone_scalar_to_device(cudf::scalar const& source,
                                                     int source_device,
                                                     rmm::cuda_stream_view source_stream,
                                                     rmm::cuda_stream_view target_stream,
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

//===--------------------------------------------------------------------===//
// Top-N boundary filter helpers
//===--------------------------------------------------------------------===//

/// @brief Whether an exact host boundary component's storage type is on the Top-N allowlist
/// (main doc, "Range and lexicographic filters").
bool is_admitted_boundary_type(cudf::data_type type) noexcept
{
  switch (type.id()) {
    case cudf::type_id::INT8:
    case cudf::type_id::INT16:
    case cudf::type_id::INT32:
    case cudf::type_id::INT64:
    case cudf::type_id::TIMESTAMP_DAYS:
    // Fixed-point keys are admitted as their scaled integer; `DECIMAL128` rides the variant's
    // `__int128_t` alternative and the kernel's width-16 load.
    case cudf::type_id::DECIMAL32:
    case cudf::type_id::DECIMAL64:
    case cudf::type_id::DECIMAL128: return true;
    default: return false;
  }
}

/// @brief Construct one device scalar holding @p value on the current device. The variant is
/// widened and re-narrowed through the storage type, so the alternative's own width never
/// decides the scalar's type.
std::unique_ptr<cudf::scalar> make_boundary_scalar(sirius::op::exact_host_scalar const& value,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  auto const widened = value.widened();
  switch (value.storage_type().id()) {
    case cudf::type_id::INT8:
      return std::make_unique<cudf::numeric_scalar<std::int8_t>>(
        static_cast<std::int8_t>(widened), true, stream, mr);
    case cudf::type_id::INT16:
      return std::make_unique<cudf::numeric_scalar<std::int16_t>>(
        static_cast<std::int16_t>(widened), true, stream, mr);
    case cudf::type_id::INT32:
      return std::make_unique<cudf::numeric_scalar<std::int32_t>>(
        static_cast<std::int32_t>(widened), true, stream, mr);
    case cudf::type_id::INT64:
      return std::make_unique<cudf::numeric_scalar<std::int64_t>>(
        static_cast<std::int64_t>(widened), true, stream, mr);
    case cudf::type_id::TIMESTAMP_DAYS:
      return std::make_unique<cudf::timestamp_scalar<cudf::timestamp_D>>(
        cudf::timestamp_D{cudf::timestamp_D::duration{static_cast<std::int32_t>(widened)}},
        true,
        stream,
        mr);
    // The scale travels with the storage type, so the device scalar carries the same scale the
    // key column does and cuDF compares the two as like-typed fixed-point values.
    case cudf::type_id::DECIMAL32:
      return std::make_unique<cudf::fixed_point_scalar<numeric::decimal32>>(
        static_cast<std::int32_t>(widened),
        numeric::scale_type{value.storage_type().scale()},
        true,
        stream,
        mr);
    case cudf::type_id::DECIMAL64:
      return std::make_unique<cudf::fixed_point_scalar<numeric::decimal64>>(
        static_cast<std::int64_t>(widened),
        numeric::scale_type{value.storage_type().scale()},
        true,
        stream,
        mr);
    case cudf::type_id::DECIMAL128:
      return std::make_unique<cudf::fixed_point_scalar<numeric::decimal128>>(
        widened, numeric::scale_type{value.storage_type().scale()}, true, stream, mr);
    default:
      throw std::invalid_argument(
        "[top_n boundary filter] boundary storage type is outside the admitted allowlist");
  }
}

/// @brief One device's AST-literal scalars. Frees them with its owning device current, mirroring
/// @c sirius_dynamic_zone_map_filter::device_zones.
struct device_scalar_replica {
  int device_id = -1;
  /// One entry per boundary component, in component order; null for a disengaged component,
  /// which owns no scalar on any device.
  std::vector<std::unique_ptr<cudf::scalar>> scalars;

  ~device_scalar_replica() noexcept
  {
    if (device_id < 0) { return; }
    try {
      rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device_id}};
      scalars.clear();
    } catch (...) {
      // Do not destroy device scalars after CUDA device selection fails; releasing ownership
      // avoids freeing them in the wrong device context.
      for (auto& scalar : scalars) {
        (void)scalar.release();
      }
      scalars.clear();
    }
  }
};

}  // namespace

//===----------------------------------------------------------------------===//
// AST mix-in
//===----------------------------------------------------------------------===//
cudf::ast::tree sirius_ast_lowerable::to_standalone_ast(
  std::function<cudf::ast::expression const&(cudf::ast::tree&)> const& column_ref_factory) const
{
  cudf::ast::tree tree;
  auto const& col_ref = column_ref_factory(tree);
  (void)to_ast(tree, col_ref);
  return tree;
}

//===----------------------------------------------------------------------===//
// sirius_dynamic_zone_map_filter
//===----------------------------------------------------------------------===//
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
      // Do not destroy device scalars after CUDA device selection fails; releasing ownership
      // avoids freeing them in the wrong device context.
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
  }
  if (cudaGetDevice(&_source_device) != cudaSuccess) {
    throw std::runtime_error("[sirius_dynamic_zone_map_filter] failed to identify source device.");
  }
}

sirius_dynamic_zone_map_filter::~sirius_dynamic_zone_map_filter() noexcept
{
  // device_zones selects each replica's device before releasing its scalars.
  _replicas.clear();
  // Release source scalars while their source device is current.
  if (_source_device >= 0) {
    try {
      rmm::cuda_set_device_raii guard{rmm::cuda_device_id{_source_device}};
      _zones.clear();
    } catch (...) {
      // Do not destroy device scalars after CUDA device selection fails; releasing ownership
      // avoids freeing them in the wrong device context.
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

  // The publisher synchronized source construction before replication. Use the source memory
  // space's pooled stream to read exact scalar values.
  rmm::cuda_set_device_raii source_guard{rmm::cuda_device_id{_source_device}};
  auto const source_stream = source->get_gpu_space().acquire_stream();

  for (auto const& target : spaces) {
    auto const& target_space = target.get_gpu_space();
    auto const device_id     = target_space.get_device_id();
    if (is_available_on_device(device_id)) { continue; }
    try {
      auto replica       = std::make_unique<device_zones>();
      replica->device_id = device_id;

      // Target scalars use the target space's allocator and pooled stream.
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
        target_stream.synchronize();
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

//===----------------------------------------------------------------------===//
// sirius_dynamic_in_list_filter --
//   implemented in src/cuda/sirius_dynamic_in_list_filter.cu (the
//   persistent cuco::static_set is device code, PIMPL'd behind set_impl).
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Top-N boundary filters: shared replication
//===----------------------------------------------------------------------===//
namespace {

/// @brief Bytes CuCascade charges for one device's AST-literal scalars.
std::size_t boundary_replica_bytes(std::span<cudf::data_type const> engaged_types) noexcept
{
  std::size_t bytes = 0;
  for (auto const type : engaged_types) {
    bytes +=
      detail::tracked_replica_allocation_bytes(static_cast<std::size_t>(cudf::size_of(type)));
  }
  return bytes;
}

/**
 * @brief Materialize one AST-literal replica per planned device, all-or-nothing
 *
 * Diverges deliberately from the join filters' best-effort per-target policy: any reservation,
 * construction, or completion failure throws and nothing is installed, so a caller can never
 * publish a boundary that is missing on one consumer device (main doc, "Multi-GPU publication").
 * Devices that already own a replica are skipped. Only engaged components allocate; a null
 * component owns no scalar on any device.
 *
 * @param[in,out] replicas The filter's replica store; appended to only on complete success
 */
template <typename ReplicaT>
void replicate_boundary_scalars(std::span<dynamic_filter_replica_space const> spaces,
                                std::vector<std::optional<exact_host_scalar>> const& components,
                                std::vector<std::unique_ptr<ReplicaT>>& replicas,
                                char const* filter_name)
{
  if (spaces.empty()) { return; }

  std::vector<cudf::data_type> engaged_types;
  for (auto const& component : components) {
    if (component) { engaged_types.push_back(component->storage_type()); }
  }
  // Both filters require an engaged head component, so at least one scalar is always allocated;
  // a hypothetical all-null boundary would be rejected by `scoped_replica_reservation`'s
  // non-empty precondition rather than yielding a scalar-less replica.
  auto const bytes = boundary_replica_bytes(engaged_types);

  auto const already_present = [&replicas](int device_id) {
    return std::any_of(replicas.begin(), replicas.end(), [device_id](auto const& replica) {
      return replica->device_id == device_id;
    });
  };

  std::vector<std::unique_ptr<ReplicaT>> pending;
  pending.reserve(spaces.size());
  for (auto const& target : spaces) {
    auto const& target_space = target.get_gpu_space();
    auto const device_id     = target_space.get_device_id();
    if (already_present(device_id)) { continue; }

    auto replica       = std::make_unique<ReplicaT>();
    replica->device_id = device_id;

    rmm::cuda_set_device_raii target_guard{rmm::cuda_device_id{device_id}};
    auto const target_stream = target_space.acquire_stream();
    auto reservation =
      detail::scoped_replica_reservation::try_acquire(target, bytes, target_stream);
    if (!reservation) {
      throw std::runtime_error(std::string{"["} + filter_name + "] replica reservation for GPU " +
                               std::to_string(device_id) +
                               " was denied; installing nothing (all-or-nothing).");
    }
    replica->scalars.reserve(components.size());
    for (auto const& component : components) {
      replica->scalars.push_back(
        component ? make_boundary_scalar(*component, target_stream, reservation->allocator())
                  : nullptr);
    }
    target_stream.synchronize();
    pending.push_back(std::move(replica));
  }

  // Commit only after every planned device succeeded.
  replicas.reserve(replicas.size() + pending.size());
  for (auto& replica : pending) {
    replicas.push_back(std::move(replica));
  }
}

/// @brief The replica serving @p device_id, or null when that device has none.
template <typename ReplicaT>
ReplicaT const* find_boundary_replica(std::vector<std::unique_ptr<ReplicaT>> const& replicas,
                                      int device_id) noexcept
{
  auto const it = std::find_if(replicas.begin(), replicas.end(), [device_id](auto const& replica) {
    return replica->device_id == device_id;
  });
  return it == replicas.end() ? nullptr : it->get();
}

}  // namespace

//===----------------------------------------------------------------------===//
// sirius_dynamic_range_filter
//===----------------------------------------------------------------------===//
struct sirius_dynamic_range_filter::device_scalars : device_scalar_replica {};

detail::boundary_filter_params sirius_dynamic_range_filter::make_boundary_filter_params(
  exact_host_scalar const& bound,
  range_bound_side side,
  bool inclusive,
  dynamic_filter_null_policy null_policy)
{
  // Validate before marshalling: every admitted type is fixed-width at 1/2/4/8/16 bytes, so the
  // width below is always one the kernel's widened loads read. An unvetted type must still throw
  // here rather than marshal -- a non-fixed-width type would make cudf::size_of throw the wrong
  // exception type. Checking here also lets the constructor marshal in its member-init list while
  // still honoring its documented std::invalid_argument contract.
  if (!is_admitted_boundary_type(bound.storage_type())) {
    throw std::invalid_argument(
      "[sirius_dynamic_range_filter] boundary type is outside the admitted allowlist");
  }
  // The kernel keeps rows that are strictly better in output order, so orienting the frame by the
  // bounded side makes "better" mean "on the kept side": LOWER keeps col > B / col >= B (better ==
  // larger, i.e. descending), UPPER keeps col < B / col <= B (better == smaller, ascending).
  detail::boundary_filter_params params{};
  params.count          = 1;
  params.strict         = !inclusive;
  auto& component       = params.components[0];
  component.engaged     = true;
  component.value       = bound.widened();
  component.descending  = side == range_bound_side::LOWER;
  component.nulls_first = null_policy == dynamic_filter_null_policy::ADMIT;
  component.width       = static_cast<std::uint8_t>(cudf::size_of(bound.storage_type()));
  return params;
}

sirius_dynamic_range_filter::sirius_dynamic_range_filter(exact_host_scalar bound,
                                                         range_bound_side side,
                                                         bool inclusive,
                                                         dynamic_filter_null_policy null_policy)
  : _bound(bound),
    _side(side),
    _inclusive(inclusive),
    _null_policy(null_policy),
    // The marshaller validates the allowlist before touching the type, so this init-list call is
    // also the constructor's documented std::invalid_argument gate.
    _compaction_params(make_boundary_filter_params(bound, side, inclusive, null_policy))
{
  int source_device = -1;
  if (cudaGetDevice(&source_device) != cudaSuccess) {
    throw std::runtime_error("[sirius_dynamic_range_filter] failed to identify source device.");
  }
  // The constructing device is served before replication; every other planned device is covered
  // by replicate_to_devices. Only the AST path consults these scalars.
  auto replica       = std::make_unique<device_scalars>();
  replica->device_id = source_device;
  replica->scalars.push_back(make_boundary_scalar(
    _bound, cudf::get_default_stream(), cudf::get_current_device_resource_ref()));
  _replicas.push_back(std::move(replica));
}

sirius_dynamic_range_filter::~sirius_dynamic_range_filter() noexcept
{
  // device_scalars selects each replica's device before releasing its scalars.
  _replicas.clear();
}

bool sirius_dynamic_range_filter::is_available_on_device(int device_id) const noexcept
{
  // Gates the AST path only: apply_compact carries the boundary in launch parameters and needs
  // no replica on any device.
  return find_boundary_replica(_replicas, detail::resolve_dynamic_filter_device_id(device_id)) !=
         nullptr;
}

void sirius_dynamic_range_filter::replicate_to_devices(
  std::span<dynamic_filter_replica_space const> spaces)
{
  std::vector<std::optional<exact_host_scalar>> const components{_bound};
  replicate_boundary_scalars(spaces, components, _replicas, "sirius_dynamic_range_filter");
}

cudf::ast::expression const& sirius_dynamic_range_filter::to_ast(
  cudf::ast::tree& tree, cudf::ast::expression const& column_ref, int device_id) const
{
  auto const* replica =
    find_boundary_replica(_replicas, detail::resolve_dynamic_filter_device_id(device_id));
  if (replica == nullptr) {
    throw std::runtime_error("[sirius_dynamic_range_filter] no replica for consumer device");
  }

  auto const op =
    _side == range_bound_side::LOWER
      ? (_inclusive ? cudf::ast::ast_operator::GREATER_EQUAL : cudf::ast::ast_operator::GREATER)
      : (_inclusive ? cudf::ast::ast_operator::LESS_EQUAL : cudf::ast::ast_operator::LESS);

  auto const& literal    = emplace_literal_from_scalar(tree, *replica->scalars.front());
  auto const& comparison = tree.emplace<cudf::ast::operation>(op, column_ref, literal);
  if (_null_policy == dynamic_filter_null_policy::REJECT) { return comparison; }
  // ADMIT: a null probe value passes, so the comparison's null verdict must not drop the row.
  auto const& is_null =
    tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::IS_NULL, column_ref);
  return tree.emplace<cudf::ast::operation>(
    cudf::ast::ast_operator::NULL_LOGICAL_OR, is_null, comparison);
}

detail::boundary_filter_result sirius_dynamic_range_filter::apply_compact(
  cudf::table_view const& batch,
  std::span<cudf::size_type const> key_columns,
  int /*device_id*/,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  // Consumer-leg type check: the fused kernel reads the key column at the boundary's width and
  // scale, so a batch whose key column differs must never be compared, only passed through (see
  // the header's apply_compact contract). data_type equality includes the scale.
  if (key_columns.empty() || batch.column(key_columns.front()).type() != _bound.storage_type()) {
    return detail::boundary_filter_result{.filtered = nullptr, .rows_kept = batch.num_rows()};
  }
  // No replica lookup: the boundary travels in launch parameters, so this works on any device.
  return detail::apply_boundary_filter(batch, key_columns, _compaction_params, stream, mr);
}

//===----------------------------------------------------------------------===//
// sirius_dynamic_lex_range_filter
//===----------------------------------------------------------------------===//
struct sirius_dynamic_lex_range_filter::device_scalars : device_scalar_replica {};

sirius_dynamic_lex_range_filter::sirius_dynamic_lex_range_filter(
  exact_host_key_tuple boundary, std::vector<lex_component_semantics> components, bool inclusive)
  : _boundary(std::move(boundary)), _components(std::move(components)), _inclusive(inclusive)
{
  if (_components.size() < 2 || _boundary.size() != _components.size()) {
    throw std::invalid_argument(
      "[sirius_dynamic_lex_range_filter] requires at least two components and one boundary "
      "component per key");
  }
  if (!_boundary.component(0).has_value()) {
    throw std::invalid_argument(
      "[sirius_dynamic_lex_range_filter] the first boundary component must be non-null");
  }
  _referenced_ordinals.reserve(_components.size());
  _key_semantics.reserve(_components.size());
  for (std::size_t i = 0; i < _components.size(); ++i) {
    auto const& component = _components[i];
    if (!is_admitted_boundary_type(component.key.storage_type)) {
      throw std::invalid_argument(
        "[sirius_dynamic_lex_range_filter] component type is outside the admitted allowlist");
    }
    _referenced_ordinals.push_back(component.consumer_ordinal);
    _key_semantics.push_back(component.key);
  }
  _compaction_params = detail::make_boundary_filter_params(
    _boundary, _key_semantics, _key_semantics.size(), !_inclusive);

  int source_device = -1;
  if (cudaGetDevice(&source_device) != cudaSuccess) {
    throw std::runtime_error("[sirius_dynamic_lex_range_filter] failed to identify source device.");
  }
  auto replica       = std::make_unique<device_scalars>();
  replica->device_id = source_device;
  replica->scalars.reserve(_components.size());
  for (std::size_t i = 0; i < _components.size(); ++i) {
    auto const& component = _boundary.component(i);
    replica->scalars.push_back(
      component ? make_boundary_scalar(
                    *component, cudf::get_default_stream(), cudf::get_current_device_resource_ref())
                : nullptr);
  }
  _replicas.push_back(std::move(replica));
}

sirius_dynamic_lex_range_filter::~sirius_dynamic_lex_range_filter() noexcept { _replicas.clear(); }

std::span<std::size_t const> sirius_dynamic_lex_range_filter::referenced_ordinals() const noexcept
{
  return _referenced_ordinals;
}

bool sirius_dynamic_lex_range_filter::is_available_on_device(int device_id) const noexcept
{
  // Gates the AST path only; apply_compact needs no replica (see the class comment).
  return find_boundary_replica(_replicas, detail::resolve_dynamic_filter_device_id(device_id)) !=
         nullptr;
}

void sirius_dynamic_lex_range_filter::replicate_to_devices(
  std::span<dynamic_filter_replica_space const> spaces)
{
  std::vector<std::optional<exact_host_scalar>> components;
  components.reserve(_components.size());
  for (std::size_t i = 0; i < _components.size(); ++i) {
    components.push_back(_boundary.component(i));
  }
  replicate_boundary_scalars(spaces, components, _replicas, "sirius_dynamic_lex_range_filter");
}

detail::boundary_filter_result sirius_dynamic_lex_range_filter::apply_compact(
  cudf::table_view const& batch,
  std::span<cudf::size_type const> key_columns,
  int /*device_id*/,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  // Consumer-leg type check, per component: an arity or type mismatch would make the kernel read
  // a wrong column at a wrong width, so one bad component degrades the whole batch to all-pass
  // (see the header's apply_compact contract). data_type equality includes the scale.
  auto const cannot_apply = [&]() {
    if (key_columns.size() != _key_semantics.size()) { return true; }
    for (std::size_t i = 0; i < _key_semantics.size(); ++i) {
      if (batch.column(key_columns[i]).type() != _key_semantics[i].storage_type) { return true; }
    }
    return false;
  };
  if (cannot_apply()) {
    return detail::boundary_filter_result{.filtered = nullptr, .rows_kept = batch.num_rows()};
  }
  return detail::apply_boundary_filter(batch, key_columns, _compaction_params, stream, mr);
}

cudf::ast::expression const& sirius_dynamic_lex_range_filter::to_ast(
  cudf::ast::tree& tree, column_ref_resolver_fn const& resolver, int device_id) const
{
  using cudf::ast::ast_operator;
  using cudf::ast::operation;

  auto const* replica =
    find_boundary_replica(_replicas, detail::resolve_dynamic_filter_device_id(device_id));
  if (replica == nullptr) {
    throw std::runtime_error("[sirius_dynamic_lex_range_filter] no replica for consumer device");
  }

  // Prefix disjunction T0 OR (E0 AND T1) OR ..., per the design doc's per-component derivations;
  // the inclusive form appends the all-equal disjunct so boundary-tied rows are never dropped.
  cudf::ast::expression const* root      = nullptr;
  cudf::ast::expression const* eq_prefix = nullptr;
  auto const component_count             = _components.size();
  for (std::size_t i = 0; i < component_count; ++i) {
    auto const& bound      = _boundary.component(i);
    auto const& key        = _components[i].key;
    auto const nulls_first = detail::nulls_first_in_output(key);
    auto const& column_ref = resolver(_components[i].consumer_ordinal);
    cudf::ast::expression const* literal =
      bound ? &emplace_literal_from_scalar(tree, *replica->scalars[i]) : nullptr;

    // T_i: strictly better than the boundary at key i. A null boundary component under
    // nulls-first admits nothing strictly better -- a constant-false T_i drops its disjunct.
    cudf::ast::expression const* strict = nullptr;
    if (bound) {
      auto const op =
        key.order == cudf::order::DESCENDING ? ast_operator::GREATER : ast_operator::LESS;
      strict = &tree.emplace<operation>(op, column_ref, *literal);
      if (nulls_first) {
        auto const& is_null = tree.emplace<operation>(ast_operator::IS_NULL, column_ref);
        strict = &tree.emplace<operation>(ast_operator::NULL_LOGICAL_OR, is_null, *strict);
      }
    } else if (!nulls_first) {
      strict = &tree.emplace<operation>(ast_operator::NOT,
                                        tree.emplace<operation>(ast_operator::IS_NULL, column_ref));
    }
    if (strict) {
      auto const* disjunct =
        eq_prefix ? &tree.emplace<operation>(ast_operator::NULL_LOGICAL_AND, *eq_prefix, *strict)
                  : strict;
      root =
        root ? &tree.emplace<operation>(ast_operator::NULL_LOGICAL_OR, *root, *disjunct) : disjunct;
    }

    // E_i: sort-equal at key i. The last component's E_i is built only for the inclusive form,
    // whose all-equal disjunct needs it.
    if (i + 1 < component_count || _inclusive) {
      auto const& equal = bound ? tree.emplace<operation>(ast_operator::EQUAL, column_ref, *literal)
                                : tree.emplace<operation>(ast_operator::IS_NULL, column_ref);
      eq_prefix         = eq_prefix
                            ? &tree.emplace<operation>(ast_operator::NULL_LOGICAL_AND, *eq_prefix, equal)
                            : &equal;
    }
  }
  if (_inclusive && eq_prefix) {
    root =
      root ? &tree.emplace<operation>(ast_operator::NULL_LOGICAL_OR, *root, *eq_prefix) : eq_prefix;
  }
  if (root == nullptr) {
    // Unreachable while the constructor requires an engaged head: an engaged component always
    // emits a comparison, so the first disjunct always exists. Reaching this means a head-null
    // boundary arrived without its derivation (`k0 IS NULL` under NULLS FIRST, exclusion under
    // NULLS LAST) -- a case publication deliberately suppresses today. Fail loudly instead of
    // lowering an always-false predicate, which a reader checkpoint would apply by dropping
    // every row.
    throw std::logic_error(
      "[sirius_dynamic_lex_range_filter] no disjunct lowered: a null head component needs the "
      "head-null derivations, which this stage does not implement");
  }
  return *root;
}

//===----------------------------------------------------------------------===//
// sirius_dynamic_filter_set
//===----------------------------------------------------------------------===//
bool sirius_dynamic_filter_set::push_filter(std::size_t col_idx,
                                            std::shared_ptr<sirius_dynamic_filter const> f)
{
  if (!f) { return false; }
  {
    std::scoped_lock lk(_mu);
    if (!_accepting_filters.load(std::memory_order_relaxed)) { return false; }
    if (_ignored_columns.count(col_idx) != 0) { return false; }
    _filters[col_idx].push_back(std::move(f));
    // Under _mu so a coherent snapshot can never pair the new filter with the old generation.
    _generation.fetch_add(1, std::memory_order_release);
  }
  _filter_count.fetch_add(1, std::memory_order_release);
  return true;
}

dynamic_filter_refinement_publisher sirius_dynamic_filter_set::register_refinement_slot(
  std::size_t primary_ordinal, std::vector<std::size_t> referenced_ordinals)
{
  auto self              = shared_from_this();
  std::size_t slot_index = 0;
  {
    std::scoped_lock lk(_mu);
    slot_index = _slots.size();
    _slots.push_back({.primary_ordinal     = primary_ordinal,
                      .referenced_ordinals = std::move(referenced_ordinals),
                      .filter              = nullptr,
                      .revision            = 0});
  }
  register_producer();
  return dynamic_filter_refinement_publisher{std::move(self), slot_index};
}

std::vector<sirius_dynamic_filter_set::refinement_slot_view>
sirius_dynamic_filter_set::refinement_slots() const
{
  std::scoped_lock lk(_mu);
  std::vector<refinement_slot_view> views;
  views.reserve(_slots.size());
  for (auto const& slot : _slots) {
    views.push_back(
      {.primary_ordinal = slot.primary_ordinal, .referenced_ordinals = slot.referenced_ordinals});
  }
  return views;
}

dynamic_filter_snapshot sirius_dynamic_filter_set::snapshot() const
{
  std::scoped_lock lk(_mu);
  dynamic_filter_snapshot snap;
  snap.generation           = _generation.load(std::memory_order_relaxed);
  snap.logical_filter_count = _filter_count.load(std::memory_order_relaxed);

  std::unordered_map<std::size_t, std::size_t> column_positions;
  auto const column_for = [&](std::size_t col) -> column_filter_snapshot& {
    auto const [it, inserted] = column_positions.try_emplace(col, snap.columns.size());
    if (inserted) { snap.columns.push_back({.column = col, .filters = {}}); }
    return snap.columns[it->second];
  };
  for (auto const& [col, filters] : _filters) {
    auto& entry = column_for(col);
    entry.filters.insert(entry.filters.end(), filters.begin(), filters.end());
  }
  // Populated slot values follow the appended filters, in slot-registration order.
  for (auto const& slot : _slots) {
    if (slot.filter) { column_for(slot.primary_ordinal).filters.push_back(slot.filter); }
  }
  return snap;
}

refinement_publish_result dynamic_filter_refinement_publisher::publish(
  std::uint64_t producer_revision, std::shared_ptr<sirius_dynamic_filter const> ready_filter) const
{
  if (!ready_filter) { return refinement_publish_result::IGNORED; }
  auto& set = *_channel;

  std::scoped_lock lk(set._mu);
  if (!set._accepting_filters.load(std::memory_order_relaxed)) {
    return refinement_publish_result::CLOSED;
  }
  auto& slot = set._slots[_slot_index];
  if (set._ignored_columns.count(slot.primary_ordinal) != 0) {
    return refinement_publish_result::IGNORED;
  }
  for (auto const referenced : slot.referenced_ordinals) {
    if (set._ignored_columns.count(referenced) != 0) { return refinement_publish_result::IGNORED; }
  }
  // Sequencing only: a fresh slot holds revision 0, so the first value needs revision >= 1.
  if (producer_revision <= slot.revision) { return refinement_publish_result::STALE; }

  bool const first_value = slot.filter == nullptr;
  slot.filter            = std::move(ready_filter);
  slot.revision          = producer_revision;
  if (first_value) { set._filter_count.fetch_add(1, std::memory_order_release); }
  set._generation.fetch_add(1, std::memory_order_release);
  return refinement_publish_result::ACCEPTED;
}

void sirius_dynamic_filter_set::ignore_columns(std::vector<std::size_t> const& cols)
{
  std::scoped_lock lk(_mu);
  _ignored_columns.insert(cols.begin(), cols.end());
}

void sirius_dynamic_filter_set::register_producer()
{
  _producer_count.fetch_add(1, std::memory_order_release);
}

void sirius_dynamic_filter_set::close_for_new_filters()
{
  std::scoped_lock lk(_mu);
  _accepting_filters.store(false, std::memory_order_release);
}

std::vector<std::shared_ptr<sirius_dynamic_filter const>>
sirius_dynamic_filter_set::filters_for_column(std::size_t col_idx) const
{
  std::scoped_lock lk(_mu);
  auto it = _filters.find(col_idx);
  if (it == _filters.end()) { return {}; }
  return it->second;
}

std::vector<std::size_t> sirius_dynamic_filter_set::filtered_columns() const
{
  std::scoped_lock lk(_mu);
  std::vector<std::size_t> out;
  out.reserve(_filters.size());
  for (auto const& [k, _] : _filters) {
    out.push_back(k);
  }
  return out;
}

bool sirius_dynamic_filter_set::empty() const
{
  std::scoped_lock lk(_mu);
  return _filters.empty();
}

}  // namespace sirius::op
