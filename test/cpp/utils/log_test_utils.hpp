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

#include "config.hpp"
#include "log/logging.hpp"
#include "log/spdlog_owning_sink.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <source_location>
#include <string>
#include <string_view>
#include <vector>

namespace sirius::test {

/// In-memory log sink recording every message at or above its level.
class recording_log_sink final : public sirius::log::sink {
 public:
  struct record {
    sirius::log::level level;
    std::string file;
    uint32_t line;
    std::string message;
  };

  void set_level(sirius::log::level level) override
  {
    _level.store(level, std::memory_order_relaxed);
  }

  bool should_log(sirius::log::level level) const override
  {
    return static_cast<int>(level) >= static_cast<int>(_level.load(std::memory_order_relaxed));
  }

  void log(sirius::log::level level,
           const std::source_location& loc,
           std::string_view message) override
  {
    if (!should_log(level)) { return; }
    std::lock_guard lock(_mutex);
    _records.push_back({level, loc.file_name(), loc.line(), std::string{message}});
  }

  bool flush() override
  {
    std::lock_guard lock(_mutex);
    ++_flush_count;
    return true;
  }

  [[nodiscard]] std::vector<record> records() const
  {
    std::lock_guard lock(_mutex);
    return _records;
  }

  [[nodiscard]] size_t count() const
  {
    std::lock_guard lock(_mutex);
    return _records.size();
  }

  [[nodiscard]] int flush_count() const
  {
    std::lock_guard lock(_mutex);
    return _flush_count;
  }

 private:
  std::atomic<sirius::log::level> _level{sirius::log::level::off};
  mutable std::mutex _mutex;
  std::vector<record> _records;
  int _flush_count = 0;
};

/// Swaps the global logging sink for a recording one for the scope of a
/// test and restores the configured logger (initialized in unittest.cpp's
/// main from the Config:: values) on exit.
class scoped_recording_log_sink {
 public:
  explicit scoped_recording_log_sink(std::string_view level = "trace")
    : _sink(std::make_shared<recording_log_sink>())
  {
    _sink->set_level(sirius::log::string_to_enum(level).value_or(sirius::log::level::info));
    sirius::log::set_sink(_sink);
  }

  ~scoped_recording_log_sink()
  {
    using duckdb::Config;
    auto lvl = sirius::log::string_to_enum(Config::LOG_LEVEL).value_or(sirius::log::level::info);
    auto flush =
      Config::LOG_FLUSH_SECONDS <= 0
        ? std::nullopt
        : std::optional<std::chrono::milliseconds>{std::chrono::seconds{Config::LOG_FLUSH_SECONDS}};
    auto sink = sirius::log::make_spdlog_owning_sink({Config::LOG_DIR, flush});
    sink->set_level(lvl);
    sirius::log::set_sink(std::move(sink));
  }

  scoped_recording_log_sink(const scoped_recording_log_sink&)            = delete;
  scoped_recording_log_sink& operator=(const scoped_recording_log_sink&) = delete;

  [[nodiscard]] recording_log_sink& sink() { return *_sink; }
  [[nodiscard]] std::shared_ptr<recording_log_sink> sink_ptr() { return _sink; }
  [[nodiscard]] std::vector<recording_log_sink::record> records() const { return _sink->records(); }

 private:
  std::shared_ptr<recording_log_sink> _sink;
};

}  // namespace sirius::test
