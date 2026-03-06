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

#include "util/segfault_backtrace.hpp"

#if defined(__linux__) && defined(__GLIBC__)

#include <cxxabi.h>
#include <execinfo.h>
#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <array>
#include <csignal>
#include <cstring>
#include <memory>
#include <string>

namespace sirius {
namespace util {

namespace {

constexpr int kBacktraceMaxFrames                                     = 128;
constexpr size_t kSegfaultLogPathMax                                  = 480;
static std::array<char, kSegfaultLogPathMax + 32> s_segfault_log_path = {};

// Write one backtrace line, demangling C++ symbols so output looks like GDB.
static void write_backtrace_line(int fd, int frame_no, const char* raw_line)
{
  char prefix[32];
  int plen = snprintf(prefix, sizeof(prefix), "  #%-2d  ", frame_no);
  if (plen > 0) write(fd, prefix, static_cast<size_t>(plen));

  const char* open_paren  = strchr(raw_line, '(');
  const char* plus        = open_paren ? strchr(open_paren, '+') : nullptr;
  const char* close_paren = open_paren ? strchr(open_paren, ')') : nullptr;

  if (open_paren && close_paren && close_paren > open_paren + 1) {
    size_t mangled_len = (plus ? static_cast<size_t>(plus - (open_paren + 1))
                               : static_cast<size_t>(close_paren - (open_paren + 1)));
    if (mangled_len > 0 && mangled_len < 1024) {
      std::string mangled(open_paren + 1, mangled_len);
      int status = 0;
      std::unique_ptr<char, void (*)(void*)> demangled(
        abi::__cxa_demangle(mangled.c_str(), nullptr, nullptr, &status), std::free);
      if (status == 0 && demangled) {
        write(fd, demangled.get(), strlen(demangled.get()));
        write(fd, " at ", 4);
      }
    }
  }
  write(fd, raw_line, strlen(raw_line));
  write(fd, "\n", 1);
}

static void segfault_handler(int sig)
{
  std::array<void*, kBacktraceMaxFrames> frames{};
  int n = backtrace(frames.data(), kBacktraceMaxFrames);
  if (n <= 0) { _exit(1); }

  char** symbols = backtrace_symbols(frames.data(), n);
  if (!symbols) {
    backtrace_symbols_fd(frames.data(), n, STDERR_FILENO);
    _exit(1);
  }

  long tid           = static_cast<long>(syscall(SYS_gettid));
  const char* header = (sig == SIGSEGV) ? "\n*** SIGSEGV — backtrace from faulting thread ***\n"
                                        : "\n*** SIGBUS — backtrace from faulting thread ***\n";

  auto write_tid = [tid](int fd) {
    write(fd, "Faulting thread id: ", 19);
    std::array<char, 24> buf{};
    char* p = buf.data() + buf.size() - 1;
    *p      = '\n';
    unsigned long u =
      (tid < 0) ? static_cast<unsigned long>(-tid) : static_cast<unsigned long>(tid);
    do {
      *--p = static_cast<char>('0' + (u % 10));
      u /= 10;
    } while (u != 0);
    if (tid < 0) *--p = '-';
    write(fd, p, static_cast<size_t>((buf.data() + buf.size()) - p));
  };

  auto write_backtrace = [&symbols, n](int fd) {
    for (int i = 0; i < n; i++) {
      write_backtrace_line(fd, i, symbols[i]);
    }
  };

  if (s_segfault_log_path[0] != '\0') {
    int log_fd = open(s_segfault_log_path.data(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (log_fd >= 0) {
      write(log_fd, header, __builtin_strlen(header));
      write_tid(log_fd);
      write_backtrace(log_fd);
      const char* tail = "*** end backtrace ***\n";
      write(log_fd, tail, __builtin_strlen(tail));
      close(log_fd);
    }
  }

  write(STDERR_FILENO, header, __builtin_strlen(header));
  write_tid(STDERR_FILENO);
  write_backtrace(STDERR_FILENO);
  const char* tail = "*** end backtrace ***\n";
  write(STDERR_FILENO, tail, __builtin_strlen(tail));
  free(symbols);
  _exit(1);
}

}  // namespace

void install_segfault_backtrace_handler()
{
  const char* log_dir = std::getenv("SIRIUS_LOG_DIR");
  if (log_dir != nullptr) {
    size_t dlen = strlen(log_dir);
    if (dlen > 0 && dlen < kSegfaultLogPathMax) {
      int written = snprintf(s_segfault_log_path.data(),
                             s_segfault_log_path.size(),
                             "%s/segfault_backtrace.txt",
                             log_dir);
      if (written < 0 || static_cast<size_t>(written) >= s_segfault_log_path.size()) {
        s_segfault_log_path[0] = '\0';
      }
    }
  }
  struct sigaction sa{};
  sa.sa_handler = segfault_handler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_RESETHAND;
  sigaction(SIGSEGV, &sa, nullptr);
  sigaction(SIGBUS, &sa, nullptr);
}

}  // namespace util
}  // namespace sirius

#else

namespace sirius {
namespace util {

void install_segfault_backtrace_handler() { (void)0; }

}  // namespace util
}  // namespace sirius

#endif
