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

#include <string>

namespace sirius {

class extension_lock {
 public:
  explicit extension_lock(const std::string& extension_name);

  ~extension_lock();

  extension_lock(const extension_lock&)            = delete;
  extension_lock& operator=(const extension_lock&) = delete;

  extension_lock(extension_lock&& other) noexcept;
  extension_lock& operator=(extension_lock&& other) noexcept;

 private:
  std::string lock_path_;
  int fd_ = -1;
};

}  // namespace sirius
