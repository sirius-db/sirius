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

#include <dlfcn.h>

#include <cstring>

extern "C" void* dlopen(char const* path, int flags)
{
  using open_t = void* (*)(char const*, int);
  auto open    = reinterpret_cast<open_t>(dlsym(RTLD_NEXT, "dlopen"));
  if (path && std::strcmp(path, "libcuda.so.1") == 0) {
    return open("/sirius-loader-test-missing-driver/libcuda.so.1", flags);
  }
  return open(path, flags);
}
