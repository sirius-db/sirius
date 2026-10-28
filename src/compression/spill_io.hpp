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

#pragma once

#include <cucascade/cudf/disk_io_context.hpp>
#include <cucascade/exec/semi_future.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace sirius::compression::detail {

inline std::shared_ptr<cucascade::io::ioctx> spill_context(
  cucascade::memory::memory_space const& space)
{
  return space.get_or_init_io_context(cucascade::default_disk_io_context);
}

// Declare after the object, file guard and all buffer owners. Even an unexpected exception
// between submission and get() must drain physical IO before any of those owners unwind.
class spill_request {
 public:
  explicit spill_request(cucascade::exec::semi_future<std::size_t>&& future) noexcept
    : _future(std::move(future))
  {
  }
  ~spill_request()
  {
    if (_future.valid()) {
      try {
        _future.wait();
      } catch (...) {
      }
    }
  }
  spill_request(spill_request const&)            = delete;
  spill_request& operator=(spill_request const&) = delete;

  std::size_t get() { return std::move(_future).get(); }

 private:
  cucascade::exec::semi_future<std::size_t> _future;
};

inline void check_spill_bytes(std::size_t actual, std::size_t expected)
{
  if (actual != expected) {
    throw std::runtime_error("compressed spill IO transferred " + std::to_string(actual) + " of " +
                             std::to_string(expected) + " bytes");
  }
}

inline cucascade::io::write_open_options spill_open_options(std::size_t size)
{
  // generate_disk_file_path already created the file. Never recreate a removed spill.
  return {.mode = cucascade::io::write_mode::open_existing, .size_hint = size};
}

}  // namespace sirius::compression::detail
