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

#include <cucascade/cudf/disk_io_context.hpp>
#include <cucascade/io/io_context.hpp>

#include <memory>

namespace sirius::test {

/// A started local-file ioctx for tests that read their fixture files directly.
///
/// One uring reactor with its own private pinned staging, so no reservation
/// manager is needed. The backend is named rather than left @c automatic, which
/// would probe cuFile and leave a @c cufile.log in the working directory. Only
/// existing regular files are supported (see @c uring_ioctx::supports).
inline std::shared_ptr<cucascade::io::ioctx> make_test_local_ioctx()
{
  cucascade::disk_io_context_options options;
  options.backend             = cucascade::disk_io_backend::uring;
  options.io.uring_n_reactors = 1;
  return cucascade::make_disk_io_context(options);
}

}  // namespace sirius::test
