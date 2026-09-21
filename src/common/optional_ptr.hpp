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

#include <stdexcept>

namespace sirius {

//! Non-owning nullable pointer wrapper. Identical API to duckdb::optional_ptr but depends only on
//! standard headers. Throws std::runtime_error on null dereference via operator* / operator->.
template <class T>
class optional_ptr {  // NOLINT: mimic std casing
 public:
  optional_ptr() noexcept : ptr(nullptr) {}
  optional_ptr(T* ptr_p) : ptr(ptr_p) {}  // NOLINT: allow implicit creation from pointer
  optional_ptr(T& ref) : ptr(&ref) {}     // NOLINT: allow implicit creation from reference

  explicit operator bool() const { return ptr; }

  T& operator*()
  {
    check_valid();
    return *ptr;
  }
  const T& operator*() const
  {
    check_valid();
    return *ptr;
  }
  T* operator->()
  {
    check_valid();
    return ptr;
  }
  const T* operator->() const
  {
    check_valid();
    return ptr;
  }

  T* get() { return ptr; }              // NOLINT: mimic std casing
  const T* get() const { return ptr; }  // NOLINT: mimic std casing
  // Returns mutable pointer even from a const optional_ptr — mirrors duckdb::optional_ptr behavior
  T* get_mutable() const { return ptr; }  // NOLINT: mimic std casing

  bool operator==(const optional_ptr& rhs) const { return ptr == rhs.ptr; }
  bool operator!=(const optional_ptr& rhs) const { return ptr != rhs.ptr; }

 private:
  T* ptr;

  void check_valid() const
  {
    if (!ptr) { throw std::runtime_error("Dereferencing null optional_ptr"); }
  }
};

}  // namespace sirius
