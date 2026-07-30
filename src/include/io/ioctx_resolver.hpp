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

#include <cucascade/io/io_context.hpp>

#include <functional>
#include <memory>
#include <string_view>

namespace sirius::io {

/**
 * @brief Late-bound lookup from a path to the ioctx that can serve it.
 *
 * Split providers and ingestibles are built before the backend for a given file
 * is known — routing depends on the path's scheme, and a scan can span several
 * backends — so they carry this resolver rather than an ioctx.  The scan manager
 * supplies the closure (see @c sirius_scan_manager::ioctx_for_path), which
 * consults the registry and lazily constructs + starts the backend on first use.
 *
 * Sirius-owned: cuCascade's io_context.hpp carries no equivalent typedef.
 */
using ioctx_resolver = std::function<std::shared_ptr<cucascade::io::ioctx>(std::string_view)>;

}  // namespace sirius::io
