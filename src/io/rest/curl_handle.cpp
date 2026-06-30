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

#include "io/rest/curl_handle.hpp"

#include <stdexcept>

namespace sirius::io::rest {

namespace {

// Receive buffer handed to libcurl for each transfer; larger than the default
// 16 KiB to cut write-callback round trips on multi-MiB ranged GETs.
constexpr long kRecvBufferSize     = 128L * 1024L;
constexpr long kConnectTimeoutMs   = 5'000L;
constexpr long kTransferTimeoutMs  = 30'000L;
constexpr long kDnsCacheTimeoutSec = 600L;

// curl_global_init must run exactly once per process, before any handle is
// created.  curl_global_cleanup is intentionally never paired with it (see the
// class doc): it races with late static destructors in third-party libraries.
std::once_flag g_global_init_once;

void global_init_once()
{
  std::call_once(g_global_init_once, [] {
    if (curl_global_init(CURL_GLOBAL_DEFAULT) != 0) {
      throw std::runtime_error("rest: curl_global_init failed");
    }
  });
}

}  // namespace

curl_share::curl_share(bool share_connections)
{
  CURLSH* sh = curl_share_init();
  if (sh == nullptr) { throw std::runtime_error("rest: curl_share_init failed"); }
  _share.reset(sh);

  // Each setopt can fail (e.g. unsupported build), so check.
  auto share_set = [sh](CURLSHoption opt, auto value) {
    if (curl_share_setopt(sh, opt, value) != CURLSHE_OK) {
      throw std::runtime_error("rest: curl_share_setopt failed");
    }
  };
  share_set(CURLSHOPT_LOCKFUNC, &curl_share::lock_cb);
  share_set(CURLSHOPT_UNLOCKFUNC, &curl_share::unlock_cb);
  share_set(CURLSHOPT_USERDATA, this);
  share_set(CURLSHOPT_SHARE, CURL_LOCK_DATA_DNS);
  share_set(CURLSHOPT_SHARE, CURL_LOCK_DATA_SSL_SESSION);
  share_set(CURLSHOPT_SHARE, CURL_LOCK_DATA_COOKIE);
  if (share_connections) {
    // Pool idle connections so they survive across handles and are reachable by
    // curl_easy_upkeep.  Caller guarantees single-thread use of this share.
    share_set(CURLSHOPT_SHARE, CURL_LOCK_DATA_CONNECT);
  }
}

void curl_share::lock_cb(CURL* /*handle*/,
                         curl_lock_data /*data*/,
                         curl_lock_access /*access*/,
                         void* userp)
{
  // Coarse single-mutex serialization across all shared data classes.
  static_cast<curl_share*>(userp)->_mtx.lock();
}

void curl_share::unlock_cb(CURL* /*handle*/, curl_lock_data /*data*/, void* userp)
{
  static_cast<curl_share*>(userp)->_mtx.unlock();
}

global_curl_context::global_init_guard::global_init_guard() { global_init_once(); }

global_curl_context& global_curl_context::instance()
{
  static global_curl_context ctx;
  return ctx;
}

global_curl_context::global_curl_context() = default;

void configure_easy_handle(CURL* handle,
                           CURLSH* share_handle,
                           long upkeep_interval_ms,
                           long conn_max_age_s)
{
  if (handle == nullptr) { throw std::runtime_error("rest: configure_easy_handle: null handle"); }

  // Connection reuse / sharing.
  if (share_handle != nullptr) {
    SIRIUS_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_SHARE, share_handle));
  }
  if (conn_max_age_s > 0) {
    SIRIUS_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_MAXAGE_CONN, conn_max_age_s));
  }
  SIRIUS_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_DNS_CACHE_TIMEOUT, kDnsCacheTimeoutSec));

  // TCP tuning.
  SIRIUS_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_TCP_KEEPALIVE, 1L));
  SIRIUS_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_TCP_NODELAY, 1L));

  // Multithreaded safety: no SIGALRM-based DNS timeouts.
  SIRIUS_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_NOSIGNAL, 1L));

  // HTTP behavior.
  SIRIUS_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_2_0));
  // Never follow redirects.  A presigned SigV4 URL is signed for one exact
  // host/path/query; transparently following an S3 region-mismatch 301/307 to a
  // different endpoint would re-issue the GET with an invalid signature (and
  // curl drops the custom Range header across the redirect), so a 3xx must
  // surface as an explicit error rather than silently producing a 403 or wrong
  // bytes.  Region selection belongs at the authorizer/endpoint level.
  SIRIUS_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_FOLLOWLOCATION, 0L));
  SIRIUS_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_BUFFERSIZE, kRecvBufferSize));

  // Default timeouts; the reactor may override the whole-transfer timeout per
  // request based on its configuration.
  SIRIUS_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_CONNECTTIMEOUT_MS, kConnectTimeoutMs));
  SIRIUS_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_TIMEOUT_MS, kTransferTimeoutMs));

  // Minimum gap between curl_easy_upkeep PINGs per connection (the reactor
  // drives the actual upkeep calls on an idle timer).
  if (upkeep_interval_ms > 0) {
    SIRIUS_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_UPKEEP_INTERVAL_MS, upkeep_interval_ms));
  }
}

}  // namespace sirius::io::rest
