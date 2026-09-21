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

#include "helper/numeric_narrowing.hpp"

namespace sirius::op {

// Each key family owns one arm here and a matching adapter arm in
// cuda/dynamic_filter_probe.cuh's dispatch_probe_adapter. Keep the arms explicit; a
// cudf::type_dispatcher here would silently widen the correctness surface.
std::optional<membership_key_domain> classify_membership_key(cudf::data_type build_type) noexcept
{
  using rep    = membership_key_rep;
  using family = membership_key_family;
  switch (build_type.id()) {
    // Signed integers, including the INT8/INT16 carriers compressed materialization narrows a
    // wider key to: the set is built at the narrowest rep that holds the carrier.
    case cudf::type_id::INT8:
    case cudf::type_id::INT16:
    case cudf::type_id::INT32:
      return membership_key_domain{rep::i32, family::signed_int, build_type};
    case cudf::type_id::INT64:
      return membership_key_domain{rep::i64, family::signed_int, build_type};
    case cudf::type_id::UINT8:
    case cudf::type_id::UINT16:
    case cudf::type_id::UINT32:
      return membership_key_domain{rep::u32, family::unsigned_int, build_type};
    case cudf::type_id::UINT64:
      return membership_key_domain{rep::u64, family::unsigned_int, build_type};
    // Temporal keys sit on integer storage (int32 epoch days, int64 ticks) and use the signed
    // integral adapter over it; the family restricts which probe types are comparable.
    case cudf::type_id::TIMESTAMP_DAYS:
      return membership_key_domain{rep::i32, family::date_days, build_type};
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      return membership_key_domain{rep::i64, family::timestamp, build_type};
    // Fixed-point keys are their unscaled integer storage at one scale. A DECIMAL64 key pinned
    // narrow arrives as DECIMAL32 at the same scale and builds a 32-bit set, exactly like an
    // INT16 carrier of an INTEGER key. DECIMAL128 has no 16-byte rep (cuco's static_set caps keys
    // at 8 bytes); it classifies onto i64 provisionally and membership_build_fits_rep decides.
    case cudf::type_id::DECIMAL32:
      return membership_key_domain{rep::i32, family::decimal, build_type, build_type.scale()};
    case cudf::type_id::DECIMAL64:
    case cudf::type_id::DECIMAL128:
      return membership_key_domain{rep::i64, family::decimal, build_type, build_type.scale()};
    // Strings have no bitwise-comparable fixed-width form a set could hold; the set stores a
    // 64-bit XXHash_64 fingerprint per key instead (no false negatives, see the header).
    case cudf::type_id::STRING:
      return membership_key_domain{rep::u64, family::string_hash, build_type};
    // Every other type (duration, floating-point, nested) declines.
    default: return std::nullopt;
  }
}

bool membership_key_supported(cudf::data_type t) noexcept
{
  return classify_membership_key(t).has_value();
}

bool membership_build_fits_rep(cudf::column_view const& keys,
                               ::cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr)
{
  auto const domain = classify_membership_key(keys.type());
  if (!domain.has_value()) { return false; }
  if (keys.type().id() != cudf::type_id::DECIMAL128) { return true; }
  // No non-null value: nothing can be truncated (every filter compacts null build keys out).
  if (keys.size() == 0 || keys.null_count() == keys.size()) { return true; }
  // Exact unscaled bounds over the valid rows; nullopt only for a scale outside the SQL range,
  // which DuckDB never produces, and is treated as not fitting so nothing is built on an
  // unverified column. Reuse the carrier-fit helper so decimal domain, scale, and bounds follow
  // the same rules as compressed materialization.
  auto const range = sirius::compute_exact_numeric_range(keys, stream, mr);
  return range.has_value() &&
         sirius::numeric_range_fits(cudf::data_type{cudf::type_id::DECIMAL64, keys.type().scale()},
                                    *range);
}

// Mirrors dispatch_probe_adapter: the probe carriers each family's adapter accepts.
bool membership_probe_compatible(membership_key_domain const& domain,
                                 cudf::data_type probe) noexcept
{
  switch (domain.family) {
    case membership_key_family::signed_int:
      switch (probe.id()) {
        case cudf::type_id::INT8:
        case cudf::type_id::INT16:
        case cudf::type_id::INT32:
        case cudf::type_id::INT64: return true;
        default: return false;
      }
    case membership_key_family::unsigned_int:
      switch (probe.id()) {
        case cudf::type_id::UINT8:
        case cudf::type_id::UINT16:
        case cudf::type_id::UINT32:
        case cudf::type_id::UINT64: return true;
        default: return false;
      }
    // A DATE probe arrives native from the post-decode cascade, or at the INT8/INT16 carrier a
    // pinned chunk stored it in (fused decode re-tags the decoded column with the stored type).
    // INT32 is the storage width itself, so it is accepted as well.
    case membership_key_family::date_days:
      switch (probe.id()) {
        case cudf::type_id::TIMESTAMP_DAYS:
        case cudf::type_id::INT8:
        case cudf::type_id::INT16:
        case cudf::type_id::INT32: return true;
        default: return false;
      }
    // Sub-day timestamps are never narrowed, and a unit change is a planner cast that blocks the
    // key upstream, so only the identical unit is comparable.
    case membership_key_family::timestamp: return probe.id() == domain.native.id();
    case membership_key_family::decimal:
      switch (probe.id()) {
        case cudf::type_id::DECIMAL32:
        case cudf::type_id::DECIMAL64:
        case cudf::type_id::DECIMAL128: return probe.scale() == domain.scale;
        default: return false;
      }
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

cudf::data_type membership_storage_type(cudf::data_type t) noexcept
{
  switch (t.id()) {
    case cudf::type_id::TIMESTAMP_DAYS: return cudf::data_type{cudf::type_id::INT32};
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS: return cudf::data_type{cudf::type_id::INT64};
    default: return t;
  }
}

}  // namespace sirius::op
