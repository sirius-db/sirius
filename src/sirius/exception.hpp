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

#include <cuda_runtime_api.h>

#include <array>
#include <cstdio>
#include <exception>
#include <format>
#include <stdexcept>
#include <string>

namespace sirius {

enum class gpu_completion_point { unknown, prepare, execute, failure, sink, retry, shutdown };

inline char const* completion_point_name(gpu_completion_point point) noexcept
{
  switch (point) {
    case gpu_completion_point::prepare: return "prepare";
    case gpu_completion_point::execute: return "execute";
    case gpu_completion_point::failure: return "failure";
    case gpu_completion_point::sink: return "sink";
    case gpu_completion_point::retry: return "retry";
    case gpu_completion_point::shutdown: return "shutdown";
    default: return "unknown";
  }
}

class internal_exception : public std::runtime_error {
 public:
  explicit internal_exception(const std::string& msg) : std::runtime_error(msg) {}

  template <typename... Args>
  explicit internal_exception(std::format_string<Args...> fmt, Args&&... args)
    : std::runtime_error(std::format(fmt, std::forward<Args>(args)...))
  {
  }
};

class not_implemented_exception : public std::runtime_error {
 public:
  explicit not_implemented_exception(const std::string& msg) : std::runtime_error(msg) {}

  template <typename... Args>
  explicit not_implemented_exception(std::format_string<Args...> fmt, Args&&... args)
    : std::runtime_error(std::format(fmt, std::forward<Args>(args)...))
  {
  }
};

class invalid_input_exception : public std::runtime_error {
 public:
  explicit invalid_input_exception(const std::string& msg) : std::runtime_error(msg) {}

  template <typename... Args>
  explicit invalid_input_exception(std::format_string<Args...> fmt, Args&&... args)
    : std::runtime_error(std::format(fmt, std::forward<Args>(args)...))
  {
  }
};

/**
 * @brief Reports that submitted GPU work could not be confirmed complete.
 *
 * Query cleanup retains resources reachable by this work. Status, stage and a bounded
 * earlier diagnostic are copied without allocation. Default construction remains usable
 * by scan paths that prepare the exception before submitting GPU work.
 */
class unjoined_gpu_work : public std::exception {
 public:
  explicit unjoined_gpu_work(cudaError_t status         = cudaSuccess,
                             gpu_completion_point point = gpu_completion_point::unknown) noexcept
    : status_(status), point_(point)
  {
  }
  [[nodiscard]] cudaError_t error_code() const noexcept { return status_; }
  [[nodiscard]] gpu_completion_point point() const noexcept { return point_; }
  [[nodiscard]] char const* prior_error() const noexcept { return prior_error_.data(); }

  void remember_prior_error(std::exception_ptr error) noexcept
  {
    try {
      if (error) std::rethrow_exception(error);
    } catch (std::exception const& e) {
      std::snprintf(prior_error_.data(), prior_error_.size(), "%s", e.what());
    } catch (...) {
      std::snprintf(prior_error_.data(), prior_error_.size(), "%s", "non-standard exception");
    }
  }
  [[nodiscard]] char const* what() const noexcept override
  {
    return "GPU work may still run after a failed host join";
  }

 private:
  cudaError_t status_;
  gpu_completion_point point_;
  std::array<char, 256> prior_error_{};
};

}  // namespace sirius
