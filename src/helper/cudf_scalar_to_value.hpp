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

#include <cudf/scalar/scalar.hpp>

#include <cuda/stream>

#include <duckdb/common/types/value.hpp>

namespace sirius {

// Convert integer, date, timestamp, and string scalars using the supplied stream.
// Returns a NULL Value for unsupported types. The caller must validate the scalar
// and enforce its own type allowlist before calling; conversion support does not
// imply that a type is eligible for boundary publication or statistics capture.
duckdb::Value cudf_scalar_to_value(cudf::scalar const& scalar, ::cuda::stream_ref stream);

}  // namespace sirius
