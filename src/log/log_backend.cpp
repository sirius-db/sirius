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

#include "log/log_backend.hpp"

#include <memory>

namespace sirius {

namespace {

class noop_backend final : public log_backend {
 public:
  void log(log_level, const std::source_location&, std::string_view) override {}
  // Vacuously reliable: nothing is ever buffered.
  bool flush() override { return true; }
};

}  // namespace

std::shared_ptr<log_backend> make_noop_backend() { return std::make_shared<noop_backend>(); }

}  // namespace sirius
