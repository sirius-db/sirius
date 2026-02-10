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

#include "util/stream_check_wrapper.hpp"

#include <dlfcn.h>

#include <iostream>
#include <mutex>

namespace sirius {
namespace util {

namespace {

// Function pointer types matching stream_check API
using enable_log_fn_t   = void (*)();
using disable_log_fn_t  = void (*)();
using set_log_file_fn_t = void (*)(const char*);

// Lazy-loaded function pointers
enable_log_fn_t g_enable_log_fn     = nullptr;
disable_log_fn_t g_disable_log_fn   = nullptr;
set_log_file_fn_t g_set_log_file_fn = nullptr;

// Handle to the loaded library
void* g_stream_check_handle = nullptr;

// For thread-safe one-time initialization
std::once_flag g_init_flag;
bool g_init_success = false;

/**
 * @brief Attempt to dynamically load the stream_check library and resolve symbols
 *
 * This is called once on first use via std::call_once. If the library can't be
 * found or symbols can't be resolved, subsequent calls will be no-ops.
 */
void lazy_init() noexcept
{
  // Try to load the stream_check library
  // Search in standard locations only (not build directory)
  // This allows controlling whether it's loaded without recompiling
  const char* lib_names[] = {
    "libstream_check.so",    // Standard location (LD_LIBRARY_PATH, install dir)
    "./libstream_check.so",  // Current directory
    nullptr};

  // Check for environment variable override
  const char* env_path = std::getenv("SIRIUS_STREAM_CHECK_LIB");
  if (env_path && env_path[0] != '\0') {
    g_stream_check_handle = dlopen(env_path, RTLD_LAZY | RTLD_LOCAL);
    if (g_stream_check_handle) {
      std::cerr << "Stream check: loaded from SIRIUS_STREAM_CHECK_LIB: " << env_path << std::endl;
      // Continue to resolve symbols below
    } else {
      std::cerr << "Stream check: failed to load from SIRIUS_STREAM_CHECK_LIB: " << env_path
                << " - " << dlerror() << std::endl;
    }
  }

  // If not loaded from environment variable, try standard search paths
  if (!g_stream_check_handle) {
    for (int i = 0; lib_names[i] != nullptr; ++i) {
      g_stream_check_handle = dlopen(lib_names[i], RTLD_LAZY | RTLD_LOCAL);
      if (g_stream_check_handle) {
        std::cerr << "Stream check: loaded from: " << lib_names[i] << std::endl;
        break;
      }
    }
  }

  if (!g_stream_check_handle) {
    // Not an error - library is optional
    return;
  }

  // Clear any existing errors
  dlerror();

  // Resolve symbols
  g_enable_log_fn =
    reinterpret_cast<enable_log_fn_t>(dlsym(g_stream_check_handle, "enable_log_on_default_stream"));
  if (!g_enable_log_fn) {
    std::cerr << "Stream check: failed to resolve enable_log_on_default_stream: " << dlerror()
              << std::endl;
    dlclose(g_stream_check_handle);
    g_stream_check_handle = nullptr;
    return;
  }

  g_disable_log_fn = reinterpret_cast<disable_log_fn_t>(
    dlsym(g_stream_check_handle, "disable_log_on_default_stream"));
  if (!g_disable_log_fn) {
    std::cerr << "Stream check: failed to resolve disable_log_on_default_stream: " << dlerror()
              << std::endl;
    dlclose(g_stream_check_handle);
    g_stream_check_handle = nullptr;
    g_enable_log_fn       = nullptr;
    return;
  }

  g_set_log_file_fn =
    reinterpret_cast<set_log_file_fn_t>(dlsym(g_stream_check_handle, "set_stream_check_log_file"));
  if (!g_set_log_file_fn) {
    std::cerr << "Stream check: failed to resolve set_stream_check_log_file: " << dlerror()
              << std::endl;
    dlclose(g_stream_check_handle);
    g_stream_check_handle = nullptr;
    g_enable_log_fn       = nullptr;
    g_disable_log_fn      = nullptr;
    return;
  }

  g_init_success = true;
  std::cerr << "Stream check: library loaded successfully" << std::endl;
}

}  // anonymous namespace

void enable_log_on_default_stream() noexcept
{
  std::call_once(g_init_flag, lazy_init);
  if (g_init_success && g_enable_log_fn) { g_enable_log_fn(); }
}

void disable_log_on_default_stream() noexcept
{
  std::call_once(g_init_flag, lazy_init);
  if (g_init_success && g_disable_log_fn) { g_disable_log_fn(); }
}

void set_stream_check_log_file(const char* path) noexcept
{
  std::call_once(g_init_flag, lazy_init);
  if (g_init_success && g_set_log_file_fn) { g_set_log_file_fn(path); }
}

}  // namespace util
}  // namespace sirius
