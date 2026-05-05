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
#include "io/s3/credential_provider.hpp"

#include <atomic>
#include <mutex>
#include <string>
#include <utility>

namespace sirius::io::s3 {

/**
 * @brief Test-only @c credential_provider that returns a canned URL or throws.
 *
 * Header-only so the S3 reactor PR (PR3) and downstream test suites can
 * include it without a separate library target. Produces deterministic
 * output for unit tests of code that consumes @c credential_provider through
 * the abstract base class — typical pattern is:
 *
 * @code
 *   auto provider = std::make_shared<mock_credential_provider>("https://canned/url");
 *   reactor->set_credentials(provider);
 *   reactor->read(obj);
 *   CHECK(provider->call_count() == 1);
 *   CHECK(provider->last_bucket() == "mybucket");
 * @endcode
 *
 * Default behavior: every call to @c get_presigned_url returns the URL
 * passed to the constructor verbatim (independent of @p obj / @p method) so
 * tests can verify "the reactor passed our URL to libcurl unchanged". To
 * exercise error paths, call @c set_throw to make subsequent calls throw
 * @c sirius::io::credential_error.
 *
 * Thread safety: counters are atomic; @c last_bucket / @c last_key are
 * guarded by an internal mutex. Safe to share across threads when tests
 * exercise concurrent reactor paths.
 */
class mock_credential_provider final : public credential_provider {
 public:
  explicit mock_credential_provider(std::string url) : _url(std::move(url)) {}

  std::string get_presigned_url(s3_object_ref const& obj, presign_method method) override
  {
    ++_call_count;
    if (method == presign_method::GET) ++_get_count;
    if (method == presign_method::HEAD) ++_head_count;
    {
      std::scoped_lock lk{_last_mtx};
      _last_bucket = obj.bucket;
      _last_key    = obj.key;
    }
    if (_should_throw.load()) {
      std::string msg;
      {
        std::scoped_lock lk{_last_mtx};
        msg =
          _throw_msg.empty() ? std::string{"mock_credential_provider: forced failure"} : _throw_msg;
      }
      throw credential_error(msg);
    }
    return _url;
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

 private:
  std::string _url;
  std::atomic<int> _call_count{0};
  std::atomic<int> _get_count{0};
  std::atomic<int> _head_count{0};
  std::atomic<bool> _should_throw{false};
  mutable std::mutex _last_mtx;
  std::string _last_bucket;
  std::string _last_key;
  std::string _throw_msg;
};

}  // namespace sirius::io::s3
