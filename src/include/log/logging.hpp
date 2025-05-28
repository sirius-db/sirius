#pragma once

#ifndef SPDLOG_ACTIVE_LEVEL
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#endif

#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <string>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/daily_file_sink.h>

#define LOG_TRACE(...) SPDLOG_LOGGER_TRACE(spdlog::default_logger_raw(), __VA_ARGS__)
#define LOG_DEBUG(...) SPDLOG_LOGGER_DEBUG(spdlog::default_logger_raw(), __VA_ARGS__)
#define LOG_INFO(...)  SPDLOG_LOGGER_INFO(spdlog::default_logger_raw(), __VA_ARGS__)
#define LOG_WARN(...)  SPDLOG_LOGGER_WARN(spdlog::default_logger_raw(), __VA_ARGS__)
#define LOG_ERROR(...) SPDLOG_LOGGER_ERROR(spdlog::default_logger_raw(), __VA_ARGS__)
#define LOG_FATAL(...) SPDLOG_LOGGER_CRITICAL(spdlog::default_logger_raw(), __VA_ARGS__)

namespace duckdb {

inline constexpr int LOG_FLUSH_SEC = 3;

inline std::optional<std::string> GetEnvVar(const std::string& name) {
    const char* val = std::getenv(name.c_str());
    if (val) {
        return std::string(val);
    } else {
        return std::nullopt;
    }
}

inline spdlog::level::level_enum GetLogLevel() {
  auto log_level_str = GetEnvVar("SIRIUS_LOG_LEVEL");
  if (log_level_str.has_value()) {
    if (*log_level_str == "trace") return spdlog::level::trace;
    if (*log_level_str == "debug") return spdlog::level::debug;
    if (*log_level_str == "info") return spdlog::level::info;
    if (*log_level_str == "warn") return spdlog::level::warn;
    if (*log_level_str == "error") return spdlog::level::err;
    if (*log_level_str == "critical") return spdlog::level::critical;
    if (*log_level_str == "off") return spdlog::level::off;
  }
  return spdlog::level::info;
}

inline std::string GetLogDir() {
  auto log_dir_str = GetEnvVar("SIRIUS_LOG_DIR");
  if (log_dir_str.has_value()) {
    return *log_dir_str;
  }
  return SIRIUS_DEFAULT_LOG_DIR;
}

inline void InitGlobalLogger() {
  // Log file
  auto log_dir = GetLogDir();
  auto log_file = log_dir + "/sirius.log";
  auto file_sink = std::make_shared<spdlog::sinks::daily_file_sink_mt>(log_file, 0, 0, false);
  file_sink->set_pattern("[%Y-%m-%d %T.%e] [%l] [%s:%#] %v");

  // Logger
  auto logger = std::make_shared<spdlog::logger>("", spdlog::sinks_init_list{file_sink});
  auto log_level = GetLogLevel();
  logger->set_level(log_level);
  spdlog::flush_every(std::chrono::seconds(LOG_FLUSH_SEC));
  spdlog::set_default_logger(logger);
}

}
