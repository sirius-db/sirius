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

#include "io/io_errors.hpp"
#include "io/rest/authorizer.hpp"

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <utility>

namespace sirius::io::rest {

/**
 * @brief Test-only @c request_authorizer that returns a canned
 *        @c authorized_request or throws.
 *
 * Header-only so the S3 reactor PR (PR3) and downstream test suites can
 * include it without a separate library target. Produces deterministic
 * output for unit tests of code that consumes @c request_authorizer through
 * the abstract base class — typical pattern is:
 *
 * @code
 *   auto provider = std::make_shared<mock_authorizer>(
 *     authorized_request{"https://canned/url", {}});
 *   reactor->set_credentials(provider);
 *   reactor->read(obj);
 *   CHECK(provider->call_count() == 1);
 *   CHECK(provider->last_bucket() == "mybucket");
 * @endcode
 *
 * Default behavior: every call to @c authorize returns the
 * @c authorized_request passed to the constructor verbatim (independent of
 * @p obj / @p method) so tests can verify "the reactor passed our URL +
 * headers to libcurl unchanged". To exercise error paths, call @c set_throw to
 * make subsequent calls throw @c sirius::io::credential_error.
 *
 * Thread safety: counters are atomic; @c last_bucket / @c last_key are
 * guarded by an internal mutex. Safe to share across threads when tests
 * exercise concurrent reactor paths.
 */
class mock_authorizer final : public request_authorizer {
 public:
  explicit mock_authorizer(authorized_request canned) : _canned(std::move(canned)) {}

  authorized_request authorize(object_ref const& obj,
                               request_method method,
                               std::chrono::seconds timeout) override
  {
    ++_call_count;
    if (method == request_method::GET) ++_get_count;
    if (method == request_method::HEAD) ++_head_count;
    {
      std::scoped_lock lk{_last_mtx};
      _last_bucket  = obj.bucket;
      _last_key     = obj.key;
      _last_timeout = timeout;
    }
    if (_should_throw.load()) {
      std::string msg;
      {
        std::scoped_lock lk{_last_mtx};
        msg = _throw_msg.empty() ? std::string{"mock_authorizer: forced failure"} : _throw_msg;
      }
      throw credential_error(msg);
    }
    return _canned;
  }

  /// Subsequent calls throw @c credential_error with @p msg (or default).
  void set_throw(std::string msg = {})
  {
    {
      std::scoped_lock lk{_last_mtx};
      _throw_msg = std::move(msg);
    }
    _should_throw.store(true);
  }

  /// Stop throwing.
  void clear_throw()
  {
    _should_throw.store(false);
    {
      std::scoped_lock lk{_last_mtx};
      _throw_msg.clear();
    }
  }

  [[nodiscard]] int call_count() const noexcept { return _call_count.load(); }
  [[nodiscard]] int get_count() const noexcept { return _get_count.load(); }
  [[nodiscard]] int head_count() const noexcept { return _head_count.load(); }

  [[nodiscard]] std::string last_bucket() const
  {
    std::scoped_lock lk{_last_mtx};
    return _last_bucket;
  }
  [[nodiscard]] std::string last_key() const
  {
    std::scoped_lock lk{_last_mtx};
    return _last_key;
  }
  [[nodiscard]] std::chrono::seconds last_timeout() const
  {
    std::scoped_lock lk{_last_mtx};
    return _last_timeout;
  }

 private:
  authorized_request _canned;
  std::atomic<int> _call_count{0};
  std::atomic<int> _get_count{0};
  std::atomic<int> _head_count{0};
  std::atomic<bool> _should_throw{false};
  mutable std::mutex _last_mtx;
  std::string _last_bucket;
  std::string _last_key;
  std::chrono::seconds _last_timeout{0};
  std::string _throw_msg;
};

}  // namespace sirius::io::rest
