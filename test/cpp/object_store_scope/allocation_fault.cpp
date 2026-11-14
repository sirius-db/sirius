// Copyright 2026 Sirius Contributors
// SPDX-License-Identifier: Apache-2.0
#include "allocation_fault.hpp"

#include <cstddef>
#include <cstdlib>
#include <new>

#if defined(__SANITIZE_THREAD__)
#define SIRIUS_TEST_ALLOCATION_FAULTS 0
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define SIRIUS_TEST_ALLOCATION_FAULTS 0
#endif
#endif
#ifndef SIRIUS_TEST_ALLOCATION_FAULTS
#define SIRIUS_TEST_ALLOCATION_FAULTS 1
#endif

namespace {

struct allocation_state {
  bool enabled{false};
  bool hit{false};
  std::size_t ordinal{0};
  std::size_t requests{0};
};

thread_local allocation_state state;

#if SIRIUS_TEST_ALLOCATION_FAULTS
void* allocate(std::size_t size, std::size_t alignment)
{
  if (state.enabled) {
    ++state.requests;
    if (state.ordinal != 0 && state.requests == state.ordinal) {
      state.hit     = true;
      state.enabled = false;
      throw std::bad_alloc{};
    }
  }
  size         = size == 0 ? 1 : size;
  void* result = nullptr;
  if (alignment <= alignof(std::max_align_t)) {
    result = std::malloc(size);
  } else if (::posix_memalign(&result, alignment, size) != 0) {
    result = nullptr;
  }
  if (!result) { throw std::bad_alloc{}; }
  return result;
}
#endif

}  // namespace

namespace sirius::test::allocation_fault {

bool supported() noexcept { return SIRIUS_TEST_ALLOCATION_FAULTS; }

void arm(std::size_t ordinal) noexcept
{
  state = {.enabled = true, .hit = false, .ordinal = ordinal, .requests = 0};
}

observation disarm() noexcept
{
  state.enabled = false;
  return {.requests = state.requests, .hit = state.hit};
}

}  // namespace sirius::test::allocation_fault

#if SIRIUS_TEST_ALLOCATION_FAULTS
void* operator new(std::size_t size) { return allocate(size, alignof(std::max_align_t)); }
void* operator new[](std::size_t size) { return ::operator new(size); }
void* operator new(std::size_t size, std::align_val_t alignment)
{
  return allocate(size, static_cast<std::size_t>(alignment));
}
void* operator new[](std::size_t size, std::align_val_t alignment)
{
  return ::operator new(size, alignment);
}
void* operator new(std::size_t size, std::nothrow_t const&) noexcept
{
  try {
    return ::operator new(size);
  } catch (...) {
    return nullptr;
  }
}
void* operator new[](std::size_t size, std::nothrow_t const& tag) noexcept
{
  return ::operator new(size, tag);
}
void* operator new(std::size_t size, std::align_val_t alignment, std::nothrow_t const&) noexcept
{
  try {
    return ::operator new(size, alignment);
  } catch (...) {
    return nullptr;
  }
}
void* operator new[](std::size_t size,
                     std::align_val_t alignment,
                     std::nothrow_t const& tag) noexcept
{
  return ::operator new(size, alignment, tag);
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { ::operator delete(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { ::operator delete(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { ::operator delete(pointer); }
void operator delete(void* pointer, std::align_val_t) noexcept { ::operator delete(pointer); }
void operator delete[](void* pointer, std::align_val_t) noexcept { ::operator delete(pointer); }
void operator delete(void* pointer, std::size_t, std::align_val_t) noexcept
{
  ::operator delete(pointer);
}
void operator delete[](void* pointer, std::size_t, std::align_val_t) noexcept
{
  ::operator delete(pointer);
}
void operator delete(void* pointer, std::nothrow_t const&) noexcept { ::operator delete(pointer); }
void operator delete[](void* pointer, std::nothrow_t const&) noexcept
{
  ::operator delete(pointer);
}
void operator delete(void* pointer, std::align_val_t, std::nothrow_t const&) noexcept
{
  ::operator delete(pointer);
}
void operator delete[](void* pointer, std::align_val_t, std::nothrow_t const&) noexcept
{
  ::operator delete(pointer);
}
#endif
