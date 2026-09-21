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
 * Install a handler for SIGSEGV (and optionally SIGBUS) that prints a
 * backtrace to stderr on the thread that received the signal, then re-raises
 * so the process terminates. Safe to call from any thread; the handler runs
 * on the faulting thread when the signal is delivered.
 *
 * No-op on non-Linux / when execinfo is not available.
 */
void install_segfault_backtrace_handler();

}  // namespace util
}  // namespace sirius
