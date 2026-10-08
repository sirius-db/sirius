// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "context_runtime.hpp"

#include "sirius_context.hpp"

namespace sirius {

struct context_runtime::Impl {
  duckdb::SiriusContext engine;
};

context_runtime::context_runtime(const parsed_sirius_config& config)
  // The default engine constructor initializes host state only.
  : impl_(std::make_unique<Impl>())
{
  impl_->engine.initialize(config);
}

context_runtime::~context_runtime() noexcept = default;

}  // namespace sirius
