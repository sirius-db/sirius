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

#include "extension_lock.hpp"

#include <fcntl.h>     // for open, O_CREAT, O_CLOEXEC, etc.
#include <sys/file.h>  // for flock
#include <unistd.h>    // for close

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

namespace sirius {

extension_lock::extension_lock(const std::string& extension_name, const std::string& lock_prefix)
  : lock_path_(lock_prefix + "/" + extension_name + ".lock")
{
  fd_ = open(lock_path_.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0666);
  if (fd_ == -1) {
    throw std::runtime_error("Failed to open lock file '" + lock_path_ +
                             "': " + std::strerror(errno));
  }

  // Try to acquire an exclusive lock on the file
  if (flock(fd_, LOCK_EX | LOCK_NB) == -1) {
    int err = errno;
    close(fd_);  // cleanup before throw
    fd_ = -1;

    if (err == EWOULDBLOCK) {
      throw std::runtime_error("Extension '" + extension_name +
                               "' is already loaded in another process." +
                               std::string(" (Lock file: ") + lock_path_ + ")");
    } else {
      throw std::runtime_error("Failed to lock file: " + std::string(std::strerror(err)));
    }
  }
}

extension_lock::~extension_lock()
{
  if (fd_ != -1) {
    flock(fd_, LOCK_UN);
    close(fd_);
    std::filesystem::remove(lock_path_);
  }
}

extension_lock::extension_lock(extension_lock&& other) noexcept
  : fd_(std::exchange(other.fd_, -1)), lock_path_(std::move(other.lock_path_))
{
}

extension_lock& extension_lock::operator=(extension_lock&& other) noexcept
{
  if (this != &other) {
    if (fd_ != -1) {
      flock(fd_, LOCK_UN);
      close(fd_);
    }
    fd_        = other.fd_;
    lock_path_ = std::move(other.lock_path_);
    other.fd_  = -1;
  }
  return *this;
}

}  // namespace sirius
