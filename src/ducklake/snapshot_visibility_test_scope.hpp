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

#pragma once
#include "sirius/export.hpp"
namespace duckdb {
class DatabaseInstance;
}
namespace sirius::ducklake::testing {
// Internal qualification hook; the database must outlive its scopes. Keep the
// registry out of headers so shared-library clients observe the engine's state.
SIRIUS_EXPORT bool snapshot_visibility_probe_enabled(duckdb::DatabaseInstance const&);
class SIRIUS_EXPORT scoped_snapshot_visibility_probe {
 public:
  explicit scoped_snapshot_visibility_probe(duckdb::DatabaseInstance const&);
  ~scoped_snapshot_visibility_probe();
  scoped_snapshot_visibility_probe(scoped_snapshot_visibility_probe const&)            = delete;
  scoped_snapshot_visibility_probe& operator=(scoped_snapshot_visibility_probe const&) = delete;

 private:
  duckdb::DatabaseInstance const* database_;
};
}  // namespace sirius::ducklake::testing
