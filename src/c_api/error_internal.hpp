// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <sirius/c/error.h>

#include <new>

namespace sirius::c_api {
sirius_status fail(sirius_status status, const char* message, sirius_error** error) noexcept;

// Diagnostics are optional: their allocation must never replace the original status.
template <typename F>
sirius_status with_diagnostic(sirius_status status, sirius_error** error, F&& make) noexcept
{
  if (error) {
    try {
      *error = make();
    } catch (...) {
      *error = nullptr;
    }
  }
  return status;
}

template <typename F>
sirius_status invoke(sirius_error** error, F&& operation) noexcept
{
  if (error) { *error = nullptr; }
  try {
    return operation();
  } catch (const std::bad_alloc&) {
    return SIRIUS_ALLOCATION_FAILURE;
  } catch (const std::exception& e) {
    return fail(SIRIUS_INTERNAL_ERROR, e.what(), error);
  } catch (...) {
    return fail(SIRIUS_INTERNAL_ERROR, "Unknown Sirius failure", error);
  }
}
}  // namespace sirius::c_api
