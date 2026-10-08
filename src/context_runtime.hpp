// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>

namespace sirius {

class parsed_sirius_config;

// Keeps engine headers and their C++20 dependencies behind the public API implementation.
class context_runtime {
 public:
  explicit context_runtime(const parsed_sirius_config& config);
  ~context_runtime() noexcept;
  context_runtime(const context_runtime&)            = delete;
  context_runtime& operator=(const context_runtime&) = delete;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sirius
