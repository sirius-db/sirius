// Copyright 2026 Sirius Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "object_store/scope.hpp"
#include "object_store/settings.hpp"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace sirius::test::object_store_test {

namespace core = sirius::object_store;

inline core::settings valid_settings(std::string prefix = "s3://bucket/prefix/")
{
  return {.endpoint       = "https://object-store.example.invalid:443",
          .region         = "core-region",
          .access_key     = "CORE_ACCESS_SENTINEL",
          .secret_key     = "CORE_SECRET_SENTINEL",
          .session_token  = "CORE_TOKEN_SENTINEL",
          .transport      = core::transport_auto,
          .signing_mode   = core::signing_presigned,
          .ca_bundle_path = {},
          .uri_prefix     = std::move(prefix)};
}

inline std::shared_ptr<const core::settings> settings_for(std::string prefix)
{
  return std::make_shared<const core::settings>(valid_settings(std::move(prefix)));
}

inline std::shared_ptr<const core::entry> entry_for(std::string prefix)
{
  return std::make_shared<const core::entry>(
    core::entry{settings_for(std::move(prefix)), std::make_shared<int>(17)});
}

template <class Left, class Right>
bool same_owner(std::shared_ptr<Left> const& left, std::shared_ptr<Right> const& right)
{
  return !left.owner_before(right) && !right.owner_before(left);
}

struct temporary_files {
  temporary_files()
  {
    auto pattern = (std::filesystem::temp_directory_path() / "sirius-core-XXXXXX").string();
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');
    auto const path = ::mkdtemp(buffer.data());
    if (!path) { throw std::runtime_error("cannot create object-store test directory"); }
    directory  = path;
    readable   = directory / "readable-ca.pem";
    unreadable = directory / "unreadable-ca.pem";
    std::ofstream(readable) << "synthetic readable CA fixture\n";
    std::ofstream(unreadable) << "synthetic unreadable CA fixture\n";
    std::filesystem::permissions(unreadable, std::filesystem::perms::none);
  }

  ~temporary_files()
  {
    std::error_code error;
    std::filesystem::permissions(unreadable, std::filesystem::perms::owner_all, error);
    std::filesystem::remove_all(directory, error);
  }

  std::filesystem::path directory;
  std::filesystem::path readable;
  std::filesystem::path unreadable;
};

// A broken lock contract must fail the process rather than hang in a thread destructor.
// The watchdog never calls the test framework.
class operation_deadline {
 public:
  explicit operation_deadline(char const* label)
    : _watchdog([this, label] {
        std::unique_lock lock(_mutex);
        if (!_changed.wait_for(lock, std::chrono::seconds(20), [this] { return _done; })) {
          std::fprintf(stderr, "object-store core timeout: %s\n", label);
          std::fflush(stderr);
          std::_Exit(124);
        }
      })
  {
  }

  ~operation_deadline()
  {
    {
      std::lock_guard lock(_mutex);
      _done = true;
    }
    _changed.notify_one();
    _watchdog.join();
  }

  operation_deadline(operation_deadline const&)            = delete;
  operation_deadline& operator=(operation_deadline const&) = delete;

 private:
  std::mutex _mutex;
  std::condition_variable _changed;
  bool _done{false};
  std::thread _watchdog;
};

}  // namespace sirius::test::object_store_test
