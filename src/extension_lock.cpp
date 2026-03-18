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
#include <unistd.h>    // for close, getpid

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
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

  // Write our PID to the lock file so we can detect same-process re-acquisition.
  pid_t our_pid = getpid();
  std::string pid_str = std::to_string(our_pid) + "\n";
  (void)write(fd_, pid_str.c_str(), pid_str.size());

  // Try to acquire an exclusive lock on the file
  if (flock(fd_, LOCK_EX | LOCK_NB) == -1) {
    int err = errno;

    if (err == EWOULDBLOCK) {
      // Check if the lock is held by the same process (another DuckDB instance
      // in the same process, e.g. when multiple test files run in one binary).
      // Read the PID from the lock file via a separate fd.
      int check_fd = open(lock_path_.c_str(), O_RDONLY | O_CLOEXEC);
      if (check_fd != -1) {
        char buf[32] = {};
        ssize_t n = read(check_fd, buf, sizeof(buf) - 1);
        ::close(check_fd);
        if (n > 0) {
          pid_t holder_pid = static_cast<pid_t>(std::strtol(buf, nullptr, 10));
          if (holder_pid == our_pid) {
            // Same process — allow it. The existing lock holder will release it
            // when its extension_lock is destroyed.
            close(fd_);
            fd_ = -1;
            return;
          }
        }
      }
      close(fd_);
      fd_ = -1;
      throw std::runtime_error("Extension '" + extension_name +
                               "' is already loaded in another process." +
                               std::string(" (Lock file: ") + lock_path_ + ")");
    } else {
      close(fd_);
      fd_ = -1;
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
