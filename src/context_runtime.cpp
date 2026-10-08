// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "context_runtime.hpp"

#include "sirius_context.hpp"

#include <rmm/error.hpp>

namespace sirius {

struct context_runtime::Impl {
  duckdb::SiriusContext engine;
};

context_runtime::context_runtime(const parsed_sirius_config& config)
  : impl_(std::make_unique<Impl>())
{
  try {
    impl_->engine.initialize(config);
  } catch (const rmm::bad_alloc& e) {
    // GPU pool exhaustion is a construction error, not a host allocation failure.
    throw std::runtime_error(e.what());
  }
}

context_runtime::~context_runtime() noexcept = default;

}  // namespace sirius
