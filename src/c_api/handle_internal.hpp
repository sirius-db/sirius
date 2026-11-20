// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>

namespace sirius::c_api {
// Reference counting of the immutable handles, which hold `std::atomic<std::size_t> references`
// starting at one.
template <typename T>
void retain(T* value) noexcept
{
  if (value) { value->references.fetch_add(1, std::memory_order_relaxed); }
}
template <typename T>
void release(T* value) noexcept
{
  if (value && value->references.fetch_sub(1, std::memory_order_acq_rel) == 1) { delete value; }
}
}  // namespace sirius::c_api
