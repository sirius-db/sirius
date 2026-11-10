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

// Pull in libc feature macros so __GLIBC__ is defined before we test it below.
// Without a prior libc include, __GLIBC__ is undefined at the #if and the entire
// implementation silently compiles to the no-op fallback (so the handler never
// installs). <cstdlib> transitively includes <features.h> on glibc and is
// harmless elsewhere.
#include <cstdlib>

#if defined(__linux__) && defined(__GLIBC__)

#include <cxxabi.h>
#include <execinfo.h>
#include <fcntl.h>
#include <log/logging.hpp>
#include <sys/syscall.h>
#include <unistd.h>

#include <array>
#include <cctype>
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

/// Hard deadline for the WHOLE handler, armed on entry. See arm_handler_deadline().
constexpr unsigned kHandlerDeadlineSeconds = 10;

// Bound every unsafe step in this handler with a wall-clock deadline.
//
// Only backtrace() and backtrace_symbols_fd() are async-signal-safe. backtrace_symbols(),
// abi::__cxa_demangle() and the log flush all allocate and take locks — backtrace_symbols() goes
// through _dl_addr(), which takes the dynamic loader lock. If the crash happened while any thread
// held the loader lock, the malloc arena lock, or the logging mutex, those calls do not fail:
// they spin or block forever. The process then looks alive (100% CPU, no output) instead of dead,
// which is strictly worse than crashing — it hides the crash, survives SIGTERM, and cannot be
// attached to once the parent has moved on.
//
// That is not hypothetical: it turned an ordinary deadlock in this codebase into an
// unkillable spin that took an hour to diagnose, because every symptom pointed at a live process.
//
// SIGALRM's default disposition terminates, so this converts "hangs forever" into "dies within
// kHandlerDeadlineSeconds". alarm() and signal() are themselves async-signal-safe.
static void arm_handler_deadline()
{
  signal(SIGALRM, SIG_DFL);  // ensure default (terminate) disposition
  alarm(kHandlerDeadlineSeconds);
}

// Best-effort flush of the global logger so the most recent log lines reach disk before we
// terminate. NOT async-signal-safe (the sink takes a mutex and may allocate); covered by the
// handler-wide deadline armed in arm_handler_deadline(), so this must NOT cancel the alarm.
static void flush_logs_best_effort()
{
  SIRIUS_LOG_WARN("SIRIUS signal handler triggered, flushing logs");
  // get_sink() never returns null; the flush itself is the risky part, bounded by the
  // handler-wide deadline and by the handler's re-entrancy guard.
  sirius::log::get_sink()->flush();
}

static const char* signal_name(int sig)
{
  switch (sig) {
    case SIGSEGV: return "SIGSEGV";
    case SIGBUS: return "SIGBUS";
    case SIGABRT: return "SIGABRT";
    case SIGFPE: return "SIGFPE";
    case SIGILL: return "SIGILL";
    default: return "SIGNAL";
  }
}

static void segfault_handler(int sig)
{
  // Re-entrancy guard: SA_RESETHAND resets only the delivered signal, so a
  // *different* fault raised while we run the allocation-heavy backtrace or the
  // log flush would re-enter this handler. On any such re-entry bail out
  // immediately rather than risk looping through those unsafe steps again.
  static volatile sig_atomic_t handling = 0;
  if (handling) { _exit(1); }
  handling = 1;

  // Everything below this line is bounded. Nothing here may hang the process.
  arm_handler_deadline();

  std::array<void*, kBacktraceMaxFrames> frames{};
  int n = backtrace(frames.data(), kBacktraceMaxFrames);
  if (n <= 0) {
    flush_logs_best_effort();
    _exit(1);
  }

  long tid          = static_cast<long>(syscall(SYS_gettid));
  const char* sname = signal_name(sig);

  // Emit the async-signal-safe form FIRST. backtrace_symbols_fd() writes straight to the fd with
  // no allocation and no loader lock, so this reaches stderr even if the demangled pass below
  // blocks and the deadline has to kill us. Un-demangled frames are far less readable, but a raw
  // backtrace beats the silence that a hung handler produces.
  {
    const char* raw_header = "\n*** ";
    write(STDERR_FILENO, raw_header, __builtin_strlen(raw_header));
    write(STDERR_FILENO, sname, strlen(sname));
    const char* raw_suffix = " — raw frames (async-signal-safe) ***\n";
    write(STDERR_FILENO, raw_suffix, __builtin_strlen(raw_suffix));
    backtrace_symbols_fd(frames.data(), n, STDERR_FILENO);
  }

  // From here on the output is nicer but the calls are not signal-safe.
  char** symbols = backtrace_symbols(frames.data(), n);
  if (!symbols) {
    flush_logs_best_effort();
    _exit(1);
  }

  auto write_header = [sname](int fd) {
    const char* suffix = " — backtrace from faulting thread ***\n";
    write(fd, "\n*** ", 5);
    write(fd, sname, strlen(sname));
    write(fd, suffix, __builtin_strlen(suffix));
  };

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
      write_header(log_fd);
      write_tid(log_fd);
      write_backtrace(log_fd);
      const char* tail = "*** end backtrace ***\n";
      write(log_fd, tail, __builtin_strlen(tail));
      close(log_fd);
    }
  }

  write_header(STDERR_FILENO);
  write_tid(STDERR_FILENO);
  write_backtrace(STDERR_FILENO);
  const char* tail = "*** end backtrace ***\n";
  write(STDERR_FILENO, tail, __builtin_strlen(tail));
  free(symbols);

  // Backtrace is safely out on both sinks; now attempt the (riskier) log flush.
  flush_logs_best_effort();
  _exit(1);
}

}  // namespace

// Returns true for "1", "true", "yes", "on" (case-insensitive). Used to let
// developers opt out of the crash handler so the OS can write a core dump.
static bool env_flag_enabled(const char* name)
{
  const char* v = std::getenv(name);
  if (v == nullptr || v[0] == '\0') { return false; }
  if (std::strcmp(v, "1") == 0) { return true; }
  auto ieq = [](const char* a, const char* b) {
    for (; *a && *b; ++a, ++b) {
      if (std::tolower(static_cast<unsigned char>(*a)) !=
          std::tolower(static_cast<unsigned char>(*b))) {
        return false;
      }
    }
    return *a == *b;
  };
  return ieq(v, "true") || ieq(v, "yes") || ieq(v, "on");
}

void install_segfault_backtrace_handler()
{
  // Escape hatch: when DISABLE_SIRIUS_SIGNAL_HANDLER is set, do NOT install the
  // handler. The handler calls _exit(1) on a crash, which prevents the OS from
  // writing a core dump; leaving it uninstalled lets the default disposition
  // produce a core (with `ulimit -c unlimited`). See docs/super-sirius/debugging.md.
  if (env_flag_enabled("DISABLE_SIRIUS_SIGNAL_HANDLER")) {
    SIRIUS_LOG_WARN(
      "Sirius crash backtrace handler DISABLED via DISABLE_SIRIUS_SIGNAL_HANDLER — "
      "the OS default disposition is in effect (core dumps enabled if `ulimit -c` allows)");
    return;
  }

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
  // Install an alternate signal stack so a stack-overflow SIGSEGV — which
  // leaves no room on the normal stack — can still run the handler. Static
  // storage so it outlives this function; sized generously since
  // backtrace()/demangle() allocate on the heap, not here. Note: sigaltstack
  // is per-thread, so this only covers stack-overflow crashes on the thread
  // that called install (typically main). SIGABRT/SIGFPE/SIGILL do not exhaust
  // the stack, so they are handled correctly on any thread regardless.
  static std::array<char, 1 << 16> s_altstack;  // 64 KiB
  stack_t ss{};
  ss.ss_sp    = s_altstack.data();
  ss.ss_size  = s_altstack.size();
  ss.ss_flags = 0;
  sigaltstack(&ss, nullptr);

  struct sigaction sa{};
  sa.sa_handler = segfault_handler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_RESETHAND | SA_ONSTACK;
  sigaction(SIGSEGV, &sa, nullptr);
  sigaction(SIGBUS, &sa, nullptr);
  sigaction(SIGABRT, &sa, nullptr);
  sigaction(SIGFPE, &sa, nullptr);
  sigaction(SIGILL, &sa, nullptr);
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
