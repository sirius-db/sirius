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

#include "io/types.hpp"  // file_descriptor

#include <curl/curl.h>
#include <curl/multi.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/timerfd.h>

#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace sirius::io::rest {

// ---------------------------------------------------------------------------
// Error-checking macros
// ---------------------------------------------------------------------------

/// Evaluate a libcurl easy-interface call and throw on a non-OK code.
#define SIRIUS_CURL_CHECK(call)                                                         \
  do {                                                                                  \
    CURLcode _ec = (call);                                                              \
    if (_ec != CURLE_OK) {                                                              \
      throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
                               " libcurl error: " + curl_easy_strerror(_ec));           \
    }                                                                                   \
  } while (false)

/// Evaluate a libcurl multi-interface call and throw on a non-OK code.
#define SIRIUS_CURLM_CHECK(call)                                                        \
  do {                                                                                  \
    CURLMcode _mc = (call);                                                             \
    if (_mc != CURLM_OK) {                                                              \
      throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
                               " libcurl-multi error: " + curl_multi_strerror(_mc));    \
    }                                                                                   \
  } while (false)

// ---------------------------------------------------------------------------
// RAII handle wrappers
// ---------------------------------------------------------------------------

struct curl_easy_deleter {
  void operator()(CURL* h) const noexcept
  {
    if (h != nullptr) curl_easy_cleanup(h);
  }
};
/// Owns a @c CURL* easy handle; cleans up on destruction.
using curl_easy_ptr = std::unique_ptr<CURL, curl_easy_deleter>;

struct curl_multi_deleter {
  void operator()(CURLM* m) const noexcept
  {
    if (m != nullptr) curl_multi_cleanup(m);
  }
};
/// Owns a @c CURLM* multi handle; cleans up on destruction.  All easy handles
/// must be removed (@c curl_multi_remove_handle) before the multi is destroyed.
using curl_multi_ptr = std::unique_ptr<CURLM, curl_multi_deleter>;

struct curl_slist_deleter {
  void operator()(curl_slist* l) const noexcept
  {
    if (l != nullptr) curl_slist_free_all(l);
  }
};
/// Owns a @c curl_slist* header list; frees the whole list on destruction.
using curl_slist_ptr = std::unique_ptr<curl_slist, curl_slist_deleter>;

struct curl_share_deleter {
  void operator()(CURLSH* s) const noexcept
  {
    if (s != nullptr) curl_share_cleanup(s);
  }
};
/// Owns a @c CURLSH* share handle; cleans up on destruction.
using curl_share_ptr = std::unique_ptr<CURLSH, curl_share_deleter>;

// ---------------------------------------------------------------------------
// epoll / timerfd / eventfd factories (RAII via file_descriptor)
// ---------------------------------------------------------------------------
//
// All three are ordinary file descriptors closed with close(), so the existing
// file_descriptor RAII wrapper owns them directly.  These factories create the
// fd with the flags the reactor needs and throw on failure.

/// Create an epoll instance (@c EPOLL_CLOEXEC).
[[nodiscard]] inline file_descriptor make_epoll_fd()
{
  int fd = ::epoll_create1(EPOLL_CLOEXEC);
  if (fd < 0) {
    throw std::runtime_error(std::string("rest: epoll_create1 failed: ") + std::strerror(errno));
  }
  return file_descriptor{fd};
}

/// Create a monotonic, non-blocking timerfd (@c TFD_NONBLOCK | @c TFD_CLOEXEC).
[[nodiscard]] inline file_descriptor make_timer_fd()
{
  int fd = ::timerfd_create(CLOCK_MONOTONIC, TFD_NONBLOCK | TFD_CLOEXEC);
  if (fd < 0) {
    throw std::runtime_error(std::string("rest: timerfd_create failed: ") + std::strerror(errno));
  }
  return file_descriptor{fd};
}

/// Create a non-blocking eventfd (@c EFD_NONBLOCK | @c EFD_CLOEXEC), initial 0.
/// Used as the cross-thread wakeup that bridges the lock-free request queue and
/// CUDA copy-completion callbacks into the epoll loop.
[[nodiscard]] inline file_descriptor make_event_fd()
{
  int fd = ::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  if (fd < 0) {
    throw std::runtime_error(std::string("rest: eventfd failed: ") + std::strerror(errno));
  }
  return file_descriptor{fd};
}

// ---------------------------------------------------------------------------
// curl_share
// ---------------------------------------------------------------------------

/**
 * @brief A @c CURLSH cache shared across a set of easy handles.
 *
 * Always shares DNS resolutions, TLS session tickets, and cookies (avoiding
 * repeated lookups / full handshakes).  When @c share_connections is true it
 * also pools the connection cache (@c CURL_LOCK_DATA_CONNECT), so idle
 * connections survive across handles and are reachable by @c curl_easy_upkeep.
 *
 * @warning A connection-sharing @c curl_share must be confined to a single
 *          thread.  Connection checkout and @c curl_easy_upkeep walk the cache
 *          and do socket I/O on its connections; two threads sharing one
 *          connection cache would race on that I/O (the lock only guards the
 *          cache structure, not the per-connection traffic).  The DNS/TLS/cookie
 *          caches are safe to share across threads (the lock fully guards them).
 */
class curl_share {
 public:
  explicit curl_share(bool share_connections);

  [[nodiscard]] CURLSH* get() const noexcept { return _share.get(); }

  curl_share(curl_share const&)            = delete;
  curl_share& operator=(curl_share const&) = delete;

 private:
  static void lock_cb(CURL* handle, curl_lock_data data, curl_lock_access access, void* userp);
  static void unlock_cb(CURL* handle, curl_lock_data data, void* userp);

  std::mutex _mtx;
  curl_share_ptr _share;
};

// ---------------------------------------------------------------------------
// global_curl_context
// ---------------------------------------------------------------------------

/**
 * @brief Process-wide libcurl initialization + a thread-safe DNS/TLS/cookie
 *        share (no connection sharing).
 *
 * Performs @c curl_global_init exactly once (and deliberately never calls
 * @c curl_global_cleanup — it races with late static destructors in
 * third-party libraries; the bounded one-time leak is the accepted trade-off).
 * Its share pools DNS, TLS sessions, and cookies across every handle in the
 * process; it intentionally does NOT pool connections (that must stay
 * single-threaded — see @c curl_share).  Used by the reactor's synchronous,
 * one-shot easy handles (host_read / head_object_size).
 */
class global_curl_context {
 public:
  /// Lazily-constructed process singleton (thread-safe first-use init).
  static global_curl_context& instance();

  [[nodiscard]] CURLSH* share_handle() const noexcept { return _share.get(); }

  global_curl_context(global_curl_context const&)            = delete;
  global_curl_context& operator=(global_curl_context const&) = delete;

 private:
  global_curl_context();

  // Runs curl_global_init before the share below is constructed (member init
  // order = declaration order).
  struct global_init_guard {
    global_init_guard();
  };
  global_init_guard _init;
  curl_share _share{/*share_connections=*/false};
};

// ---------------------------------------------------------------------------
// Easy-handle configuration
// ---------------------------------------------------------------------------

/**
 * @brief Apply the standard high-performance options to a fresh easy handle.
 *
 * Sets HTTP/2, the shared DNS/TLS/cookie (and optionally connection) cache, TCP
 * tuning (NODELAY, keepalive), @c NOSIGNAL (required for multithreaded use),
 * receive buffer size, connect/transfer timeouts, disabled redirect following
 * (presigned URLs cannot survive a redirect — see the .cpp), connection
 * max-age, DNS cache timeout, and the @c curl_easy_upkeep interval.
 * Per-request options (URL, Range, write/header callbacks, TLS verification)
 * are set by the reactor at submit time.
 *
 * @param handle            the easy handle to configure.
 * @param share_handle      the @c CURLSH to attach (DNS/TLS/cookie [+ connect]).
 * @param upkeep_interval_ms minimum gap between @c curl_easy_upkeep PINGs per
 *                           connection; 0 leaves the curl default and means the
 *                           caller does not drive upkeep on this handle.
 * @param conn_max_age_s    @c CURLOPT_MAXAGE_CONN — max age (seconds) of a
 *                           pooled connection still eligible for reuse; 0 leaves
 *                           the curl default.
 */
void configure_easy_handle(CURL* handle,
                           CURLSH* share_handle,
                           long upkeep_interval_ms = 0,
                           long conn_max_age_s     = 20);

}  // namespace sirius::io::rest
