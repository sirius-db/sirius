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

// The Arrow C Device Data Interface struct `cudf::to_arrow_host` returns. DuckDB's Arrow header
// (the one definition of ArrowArray/ArrowSchema this library has, see arrow_host_import.cpp)
// defines only the C Data Interface, and cudf/interop.hpp forward-declares ArrowDeviceArray and
// typedefs ArrowDeviceType. The interface is meant to be vendored — arrow/c/abi.h says so in its
// preamble — so this is the spec's definition under the spec's guard, shared by the FFI and its
// tests so the two never drift.

#include "duckdb/common/arrow/arrow.hpp"  // ArrowArray, ArrowSchema (ARROW_C_DATA_INTERFACE)

#include <cudf/interop.hpp>  // ArrowDeviceType

#include <cstdint>

#ifndef ARROW_C_DEVICE_DATA_INTERFACE
#define ARROW_C_DEVICE_DATA_INTERFACE
#define ARROW_DEVICE_CPU 1
struct ArrowDeviceArray {
  struct ArrowArray array;
  int64_t device_id;
  ArrowDeviceType device_type;
  void* sync_event;
  int64_t reserved[3];
};
#endif
