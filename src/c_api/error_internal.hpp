// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "config_loading.hpp"

#include <sirius/c/error.h>

#include <new>

namespace sirius::c_api {
sirius_status fail(sirius_status status, const char* message, sirius_error** error) noexcept;

template <typename F>
sirius_status invoke(sirius_error** error, F&& operation) noexcept
{
  if (error) { *error = nullptr; }
  try {
    return operation();
  } catch (const std::bad_alloc&) {
    return SIRIUS_ALLOCATION_FAILURE;
  } catch (const configuration_load_error& e) {
    return fail(e.status, e.what(), error);
  } catch (const std::exception& e) {
    return fail(SIRIUS_INTERNAL_ERROR, e.what(), error);
  } catch (...) {
    return fail(SIRIUS_INTERNAL_ERROR, "Unknown Sirius failure", error);
  }
}
}  // namespace sirius::c_api
