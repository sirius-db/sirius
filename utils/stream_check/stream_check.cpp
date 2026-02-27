// =============================================================================
// Copyright 2025, Sirius Contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
// =============================================================================

#include "stream_check.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <absl/debugging/stacktrace.h>
#include <absl/debugging/symbolize.h>

#include <atomic>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_set>

// Thread-local atomic boolean to control logging behavior
thread_local std::atomic<bool> g_log_on_default_stream{false};

// Global state for logging
static std::mutex g_log_mutex;
static std::string g_log_file_path = "default_stream_traces.log";
static std::unordered_set<std::string> g_logged_traces;
static bool g_symbolize_initialized = false;

extern "C" {

void enable_log_on_default_stream()
{
  // Initialize symbolization on first use
  if (!g_symbolize_initialized) {
    absl::InitializeSymbolizer(nullptr);
    g_symbolize_initialized = true;
  }
  g_log_on_default_stream.store(true, std::memory_order_relaxed);
}

void disable_log_on_default_stream()
{
  g_log_on_default_stream.store(false, std::memory_order_relaxed);
}

void set_stream_check_log_file(const char* path)
{
  std::lock_guard<std::mutex> lock(g_log_mutex);
  if (path && path[0] != '\0') {
    g_log_file_path = path;
  } else {
    g_log_file_path = "default_stream_traces.log";
  }
}

}  // extern "C"

// Helper function to get a string representation of a stack trace
static std::string get_stack_trace_string()
{
  constexpr int kMaxStackDepth = 64;
  void* stack[kMaxStackDepth];

  // Get the raw stack trace
  int depth = absl::GetStackTrace(stack, kMaxStackDepth, 1);  // Skip this function

  std::ostringstream oss;
  char symbol_buf[1024];

  for (int i = 0; i < depth; ++i) {
    // Try to symbolize the address
    if (absl::Symbolize(stack[i], symbol_buf, sizeof(symbol_buf))) {
      oss << "  [" << i << "] " << symbol_buf << " (" << stack[i] << ")\n";
    } else {
      oss << "  [" << i << "] " << stack[i] << "\n";
    }
  }

  return oss.str();
}

// Helper function to get current timestamp
static std::string get_timestamp()
{
  auto now = std::time(nullptr);
  auto tm  = *std::localtime(&now);
  std::ostringstream oss;
  oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
  return oss.str();
}

// Helper function to log a default stream access
static void log_default_stream_access()
{
  std::string stack_trace = get_stack_trace_string();

  std::lock_guard<std::mutex> lock(g_log_mutex);

  // Check if we've already logged this exact stack trace
  if (g_logged_traces.find(stack_trace) != g_logged_traces.end()) {
    return;  // Already logged, skip to avoid spam
  }

  // Mark this trace as logged
  g_logged_traces.insert(stack_trace);

  // Open log file in append mode
  std::ofstream log_file(g_log_file_path, std::ios::app);
  if (!log_file) {
    return;  // Can't log, silently continue
  }

  // Write the log entry
  log_file << "================================================================================\n";
  log_file << "Default stream access detected at: " << get_timestamp() << "\n";
  log_file << "Stack trace:\n";
  log_file << stack_trace;
  log_file
    << "================================================================================\n\n";
  log_file.flush();
}

// Namespace wrapper to match cudf's namespace structure
namespace cudf {

/**
 * @brief Override of cudf::get_default_stream() that logs stack traces
 *
 * This function intercepts calls to cudf::get_default_stream(). When
 * enable_log_on_default_stream() has been called for the current thread,
 * this will log a stack trace to the configured file. Otherwise, it returns
 * the RMM default stream without logging.
 */
rmm::cuda_stream_view get_default_stream()
{
  if (g_log_on_default_stream.load(std::memory_order_relaxed)) { log_default_stream_access(); }

  // Always return the RMM default stream
  return rmm::cuda_stream_view{};
}

}  // namespace cudf
