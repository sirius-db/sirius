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

namespace sirius {
namespace util {

/**
 * @brief Enable logging on default CUDA stream access (no-op if stream_check not available)
 *
 * When ENABLE_STREAM_CHECK is defined at compile time, this will attempt to dynamically
 * load the stream_check library and enable logging. If the library is not found or
 * ENABLE_STREAM_CHECK is not defined, this is a no-op.
 */
void enable_log_on_default_stream() noexcept;

/**
 * @brief Disable logging on default CUDA stream access (no-op if stream_check not available)
 *
 * When ENABLE_STREAM_CHECK is defined at compile time, this will attempt to dynamically
 * load the stream_check library and disable logging. If the library is not found or
 * ENABLE_STREAM_CHECK is not defined, this is a no-op.
 */
void disable_log_on_default_stream() noexcept;

/**
 * @brief Set the log file path for stream check (no-op if stream_check not available)
 *
 * @param path Path to the log file where stack traces will be written
 */
void set_stream_check_log_file(const char* path) noexcept;

}  // namespace util
}  // namespace sirius
