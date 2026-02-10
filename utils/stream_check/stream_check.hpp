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

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Enable logging on default stream access for the current thread
 *
 * After calling this function, any call to cudf::get_default_stream() from
 * the current thread will log a stack trace to the configured file.
 */
void enable_log_on_default_stream();

/**
 * @brief Disable logging on default stream access for the current thread
 *
 * After calling this function, calls to cudf::get_default_stream() from
 * the current thread will not log stack traces.
 */
void disable_log_on_default_stream();

/**
 * @brief Set the file path for logging default stream stack traces
 *
 * @param path File path where stack traces will be logged. If NULL or empty,
 *             defaults to "default_stream_traces.log" in current directory.
 */
void set_stream_check_log_file(const char* path);

#ifdef __cplusplus
}
#endif
