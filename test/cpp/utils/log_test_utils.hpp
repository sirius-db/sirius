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
#include "log/log_backend.hpp"
#include "log/logging.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <source_location>
#include <string>
#include <string_view>
#include <vector>

namespace sirius::test {

/// In-memory log_backend recording every message at or above its level.
class recording_log_backend final : public sirius::log_backend {
 public:
  struct record {
    sirius::log_level level;
    std::string file;
    uint32_t line;
    std::string message;
  };

  void set_level(sirius::log_level level) override
  {
    _level.store(level, std::memory_order_relaxed);
  }

  bool should_log(sirius::log_level level) const override
  {
    return static_cast<int>(level) >= static_cast<int>(_level.load(std::memory_order_relaxed));
  }

  void log(sirius::log_level level,
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
  std::atomic<sirius::log_level> _level{sirius::log_level::off};
  mutable std::mutex _mutex;
  std::vector<record> _records;
  int _flush_count = 0;
};

/// Swaps the global logging backend for a recording one for the scope of a
/// test and restores the configured logger (initialized in unittest.cpp's
/// main from the Config:: values) on exit.
class scoped_recording_log_backend {
 public:
  explicit scoped_recording_log_backend(std::string_view level = "trace")
    : _backend(std::make_shared<recording_log_backend>())
  {
    sirius::InitGlobalLogger(_backend, level);
  }

  ~scoped_recording_log_backend()
  {
    using duckdb::Config;
    sirius::InitGlobalLogger(
      Config::LOG_LEVEL, Config::LOG_DIR, Config::LOG_FLUSH_MS, Config::LOG_BACKEND);
  }

  scoped_recording_log_backend(const scoped_recording_log_backend&)            = delete;
  scoped_recording_log_backend& operator=(const scoped_recording_log_backend&) = delete;

  [[nodiscard]] recording_log_backend& backend() { return *_backend; }
  [[nodiscard]] std::shared_ptr<recording_log_backend> backend_ptr() { return _backend; }
  [[nodiscard]] std::vector<recording_log_backend::record> records() const
  {
    return _backend->records();
  }

 private:
  std::shared_ptr<recording_log_backend> _backend;
};

}  // namespace sirius::test
