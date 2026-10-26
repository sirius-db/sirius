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

#include "ducklake/unit_descriptor.hpp"

#include "io/uri_parser.hpp"

#include <stdexcept>

namespace sirius::ducklake {
unit_gate assess_local_scope(unit_descriptor const& unit)
{
  unit_gate result;
  auto reject = [&](unit_gate_reason reason) {
    result.outside_local_scope = true;
    result.reasons.push_back(reason);
  };
  auto location = [&](std::string const& path, unit_gate_reason remote) {
    try {
      auto uri = io::parse(path);
      if (uri.scheme != "file" || !uri.host.empty()) reject(remote);
    } catch (std::invalid_argument const&) {
      reject(unit_gate_reason::invalid_locator);
    }
  };
  switch (unit.kind) {
    case unit_kind::parquet: location(unit.input.path, unit_gate_reason::remote_data); break;
    case unit_kind::inline_table:
    case unit_kind::transaction_local_inline: break;
    default: reject(unit_gate_reason::unknown_unit_kind); break;
  }
  if (!unit.delete_path.empty()) location(unit.delete_path, unit_gate_reason::remote_delete);
  if (unit.encrypted) reject(unit_gate_reason::encrypted_data);
  if (unit.delete_encrypted) reject(unit_gate_reason::encrypted_delete);
  if (unit.snapshot_min || unit.snapshot_max) reject(unit_gate_reason::partial_data);
  if (unit.snapshot_min && unit.snapshot_max && *unit.snapshot_min > *unit.snapshot_max)
    reject(unit_gate_reason::invalid_snapshot_range);
  if (unit.delete_path.empty() != (unit.deletes == delete_format::none)) {
    reject(unit_gate_reason::unknown_delete_format);
  } else if (unit.deletes == delete_format::parquet) {
    // Neither the file-list entry nor its suffix proves absence of per-row
    // snapshot_ids. Payload validation must establish the exact delete semantics.
    result.reasons.push_back(unit_gate_reason::parquet_delete_semantics_required);
  } else if (unit.deletes == delete_format::puffin) {
    // Container/bare, file association and embedded snapshot semantics are
    // metadata evidence; a single blob is not evidence of a simple DV.
    result.reasons.push_back(unit_gate_reason::dv_metadata_required);
  } else if (unit.deletes == delete_format::unknown) {
    reject(unit_gate_reason::unknown_delete_format);
  }
  return result;
}
}  // namespace sirius::ducklake
