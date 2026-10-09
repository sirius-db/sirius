// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "c_api/config_internal.hpp"
#include "c_api/error_internal.hpp"
#include "sirius_context.hpp"

#include <sirius/c/context/context.h>

struct sirius_context {
  duckdb::SiriusContext engine;
};

sirius_status sirius_context_create(const sirius_context_config* config,
                                    sirius_context** out,
                                    sirius_error** error) noexcept
{
  if (out) { *out = nullptr; }
  return sirius::c_api::invoke(
    error,
    [&] {
      if (!config || !out) {
        return sirius::c_api::fail(
          SIRIUS_INVALID_ARGUMENT, "Configuration or output pointer is NULL", error);
      }
      // Allocate all handle storage before committing engine initialization.
      auto context = std::make_unique<sirius_context>();
      context->engine.initialize(config->config);
      *out = context.release();
      return sirius_status{SIRIUS_SUCCESS};
    },
    SIRIUS_CONTEXT_INITIALIZATION);
}

void sirius_context_destroy(sirius_context* context) noexcept { delete context; }
