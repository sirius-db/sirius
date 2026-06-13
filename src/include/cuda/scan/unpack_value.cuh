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

//! @file
//! Compatibility entry point for the shared bit-unpacking primitive. The
//! implementation lives in `detail/bit_unpack.cuh`; this header re-exports it
//! into `sirius::cuda::scan` so every codec decoder can call `unpack_value`
//! through one include.

#pragma once

#include <cuda/scan/detail/bit_unpack.cuh>

namespace sirius::cuda::scan
{
using detail::unpack_value;
} // namespace sirius::cuda::scan
