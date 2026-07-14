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

#include <chrono>
#include <memory>
#include <optional>
#include <string>

namespace sirius::log {

namespace {

class noop_backend final : public sink {
 public:
  void set_level(level) override {}
  // Discards everything; reporting "never" also lets the facade skip formatting.
  bool should_log(level) const override { return false; }
  void log(level, const std::source_location&, std::string_view) override {}
  // Vacuously reliable: nothing is ever buffered.
  bool flush() override { return true; }
};

}  // namespace

std::shared_ptr<sink> make_noop_backend() { return std::make_shared<noop_backend>(); }

std::shared_ptr<sink> make_backend(std::string_view level_str,
                                   std::string_view log_dir,
                                   uint32_t flush_ms,
                                   backend_type type)
{
  std::shared_ptr<sink> s;
  switch (type) {
    case backend_type::spdlog: {
      auto flush_interval =
        flush_ms == 0 ? std::nullopt : std::optional{std::chrono::milliseconds{flush_ms}};
      s = make_spdlog_backend({std::string{log_dir}, flush_interval});
      break;
    }
    case backend_type::noop: s = make_noop_backend(); break;
  }

  level lvl = level::info;  // unknown names default to info
  string_to_enum(level_str, lvl);
  s->set_level(lvl);
  return s;
}

}  // namespace sirius::log
