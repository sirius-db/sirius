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

namespace sirius::exchange {

/// Whether writes to a peer go through its pinned host mirror instead of NIXL. automatic: over TCP
/// lanes to peers on other hosts, and through shared memory to peers on this host whose GPU lacks
/// peer access to this one (NVML). always: also to every peer on this host. never: NIXL to every
/// peer. remote: TCP lanes to every peer, even on this host (for tests).
enum class host_staging { automatic, always, never, remote };

}  // namespace sirius::exchange
