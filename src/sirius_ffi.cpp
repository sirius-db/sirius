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

// Implementation of the public FFI surface (sirius_ffi.hpp). This is the one
// translation unit that sees the heavy internal types, so consumers (e.g. the
// Rust bindings) never include sirius_context.hpp.

#include "sirius_ffi.hpp"

#include "sirius_config.hpp"   // sirius::sirius_config
#include "sirius_context.hpp"  // duckdb::SiriusContext

namespace sirius::ffi {

Context::Context() : context_(std::make_unique<duckdb::SiriusContext>())
{
  sirius::sirius_config config;
  config.apply_defaults();  // populate default GPU/host/disk memory spaces
  context_->initialize(config);
}

// Defined here, where duckdb::SiriusContext is complete: destroying `context_`
// runs ~SiriusContext (which tears down the initialized engine).
Context::~Context() = default;

std::unique_ptr<Context> make_context() { return std::make_unique<Context>(); }

}  // namespace sirius::ffi
