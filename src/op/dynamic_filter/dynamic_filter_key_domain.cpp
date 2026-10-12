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

#include "op/dynamic_filter/dynamic_filter_key_domain.hpp"

#include "helper/numeric_carrier_rule.hpp"
#include "helper/numeric_narrowing.hpp"

#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace sirius::op {

namespace {

// The rep is the narrowest listed rep that holds the carrier: 32-bit for carriers up to 4 bytes,
// 64-bit above. DECIMAL128 (16 bytes) lands on the 64-bit rep provisionally, see the header.
membership_key_rep rep_for_width(std::size_t carrier_bytes, bool is_signed) noexcept
{
  if (carrier_bytes <= sizeof(std::int32_t)) {
    return is_signed ? membership_key_rep::i32 : membership_key_rep::u32;
  }
  return is_signed ? membership_key_rep::i64 : membership_key_rep::u64;
}

}  // namespace

// The numeric families follow the carrier domains of compressed materialization
// (sirius::narrow_domain_of), so the one list of which type ids are signed, unsigned, fixed-point,
// or DATE carriers lives in helper/numeric_narrowing.cpp. The temporal and string arms are
// membership-only. Each family owns a matching adapter arm in cuda/dynamic_filter_probe.cuh's
// dispatch_probe_adapter.
std::optional<membership_key_domain> classify_membership_key(cudf::data_type build_type) noexcept
{
  using family = membership_key_family;
  // Only reached for fixed-width types: cudf::size_of throws on variable-width ones.
  auto const domain_of = [&](family f, bool is_signed, std::int32_t scale = 0) {
    auto const width = static_cast<std::size_t>(cudf::size_of(integer_storage_type(build_type)));
    return membership_key_domain{rep_for_width(width, is_signed), f, build_type, scale};
  };
  switch (sirius::narrow_domain_of(build_type)) {
    // Signed integers, including the INT8/INT16 carriers compressed materialization narrows a
    // wider key to: the set is built at the narrowest rep that holds the carrier.
    case narrow_domain::SIGNED_INTEGER: return domain_of(family::signed_int, true);
    case narrow_domain::UNSIGNED_INTEGER: return domain_of(family::unsigned_int, false);
    // Fixed-point keys are their unscaled integer storage at one scale. A DECIMAL64 key pinned
    // narrow arrives as DECIMAL32 at the same scale and builds a 32-bit set, exactly like an
    // INT16 carrier of an INTEGER key. DECIMAL128 has no 16-byte rep (cuco's static_set caps keys
    // at 8 bytes); it classifies onto i64 provisionally and membership_build_fits_rep decides.
    case narrow_domain::DECIMAL: return domain_of(family::decimal, true, build_type.scale());
    // Temporal keys sit on integer storage (int32 epoch days) and use the signed integral adapter
    // over it; the family restricts which probe types are comparable.
    case narrow_domain::DATE: return domain_of(family::date_days, true);
    case narrow_domain::NONE: break;
  }
  switch (build_type.id()) {
    // Sub-day timestamps: int64 ticks, comparable only at their own unit.
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS: return domain_of(family::timestamp, true);
    // Strings have no bitwise-comparable fixed-width form a set could hold; the set stores a
    // 64-bit XXHash_64 fingerprint per key instead (no false negatives, see the header).
    case cudf::type_id::STRING:
      return membership_key_domain{membership_key_rep::u64, family::string_hash, build_type};
    // Every other type (duration, floating-point, nested) declines.
    default: return std::nullopt;
  }
}

bool membership_key_supported(cudf::data_type t) noexcept
{
  return classify_membership_key(t).has_value();
}

bool membership_same_family(cudf::data_type recorded, cudf::data_type arrived) noexcept
{
  auto const a = classify_membership_key(recorded);
  auto const b = classify_membership_key(arrived);
  return a.has_value() && b.has_value() && a->family == b->family && a->scale == b->scale;
}

cudf::data_type membership_rep_type(membership_key_domain const& domain) noexcept
{
  bool const wide = domain.rep == membership_key_rep::i64 || domain.rep == membership_key_rep::u64;
  if (domain.family == membership_key_family::decimal) {
    return cudf::data_type{wide ? cudf::type_id::DECIMAL64 : cudf::type_id::DECIMAL32,
                           domain.scale};
  }
  switch (domain.rep) {
    case membership_key_rep::i32: return cudf::data_type{cudf::type_id::INT32};
    case membership_key_rep::i64: return cudf::data_type{cudf::type_id::INT64};
    case membership_key_rep::u32: return cudf::data_type{cudf::type_id::UINT32};
    case membership_key_rep::u64: return cudf::data_type{cudf::type_id::UINT64};
  }
  return cudf::data_type{cudf::type_id::INT64};
}

bool membership_build_fits_rep(cudf::column_view const& keys,
                               ::cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr)
{
  auto const domain = classify_membership_key(keys.type());
  if (!domain.has_value()) { return false; }
  // A carrier no wider than its rep always fits (the string family stores fingerprints, not the
  // column). Only a carrier wider than the rep, DECIMAL128 on the int64 rep, has to be checked;
  // the check is the same one compressed materialization applies when narrowing a carrier.
  if (domain->family == membership_key_family::string_hash) { return true; }
  auto const rep_type = membership_rep_type(*domain);
  if (cudf::size_of(integer_storage_type(keys.type())) <= cudf::size_of(rep_type)) { return true; }
  return sirius::column_values_fit(keys, rep_type, stream, mr);
}

// The one acceptance rule for probe types. dispatch_probe_adapter (cuda/dynamic_filter_probe.cuh)
// declines whatever this declines, then reads the accepted probe through its integer storage.
bool membership_probe_compatible(membership_key_domain const& domain,
                                 cudf::data_type probe) noexcept
{
  switch (domain.family) {
    case membership_key_family::signed_int:
      return sirius::narrow_domain_of(probe) == narrow_domain::SIGNED_INTEGER;
    case membership_key_family::unsigned_int:
      return sirius::narrow_domain_of(probe) == narrow_domain::UNSIGNED_INTEGER;
    // A DATE probe arrives native from the post-decode cascade, or at the INT8/INT16 carrier a
    // pinned chunk stored it in (fused decode re-tags the decoded column with the stored type).
    // INT32 is the storage width itself, so it is accepted as well; INT64 is not a DATE carrier.
    case membership_key_family::date_days:
      return probe.id() == cudf::type_id::TIMESTAMP_DAYS ||
             (sirius::narrow_domain_of(probe) == narrow_domain::SIGNED_INTEGER &&
              cudf::size_of(probe) <= cudf::size_of(integer_storage_type(domain.native)));
    // Sub-day timestamps are never narrowed, and a unit change is a planner cast that blocks the
    // key upstream, so only the identical unit is comparable.
    case membership_key_family::timestamp: return probe.id() == domain.native.id();
    case membership_key_family::decimal:
      return sirius::narrow_domain_of(probe) == narrow_domain::DECIMAL &&
             probe.scale() == domain.scale;
    // The in-kernel hash reads cudf::string_view elements, so the probe must be a materialized
    // STRING column; a dictionary-encoded carrier declines rather than hashing its codes.
    case membership_key_family::string_hash: return probe.id() == cudf::type_id::STRING;
  }
  return false;
}

std::size_t membership_rep_bytes(membership_key_rep rep) noexcept
{
  switch (rep) {
    case membership_key_rep::i32:
    case membership_key_rep::u32: return sizeof(std::int32_t);
    case membership_key_rep::i64:
    case membership_key_rep::u64: return sizeof(std::int64_t);
  }
  return sizeof(std::int64_t);
}

membership_build_keys prepare_membership_build(std::string_view filter_name,
                                               cudf::column_view const& keys,
                                               ::cuda::stream_ref stream,
                                               rmm::device_async_resource_ref mr)
{
  auto const prefix = std::string{filter_name};
  auto const domain = classify_membership_key(keys.type());
  if (!domain.has_value()) {
    throw std::invalid_argument(prefix + " unsupported key type (see membership_key_supported).");
  }
  // A DECIMAL128 build whose unscaled values exceed the int64 rep cannot be stored exactly.
  if (!membership_build_fits_rep(keys, stream, mr)) {
    throw std::invalid_argument(
      prefix + " build keys do not fit the key rep (DECIMAL128 values outside int64).");
  }
  membership_build_keys build;
  build.domain = *domain;
  build.keys   = keys;
  // Null build keys match nothing under the join's null_equality::UNEQUAL, so they are dropped
  // exactly. The compacted table stays alive in the result until the caller has enqueued its
  // copy or insert on `stream`; its stream-ordered free then follows that work.
  if (keys.null_count() > 0) {
    build.compacted = cudf::drop_nulls(cudf::table_view{{keys}}, {0}, stream, mr);
    build.keys      = build.compacted->view().column(0);
  }
  if (cudaGetDevice(&build.source_device) != cudaSuccess) {
    throw std::runtime_error(prefix + " failed to identify source device.");
  }
  return build;
}

}  // namespace sirius::op
