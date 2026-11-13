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

namespace duckdb {
class DatabaseInstance;
}

namespace sirius::exchange {

/// Registers the functions a distributed gateway uses to restart a rank it did not start:
/// - `sirius_rank_info()`: this process's pid; its boot id, a random id made once per process, so
///   the gateway can tell a restarted rank from the one it asked to exit; and whether it joined
///   (note_exchange_joined), since only a process that has must restart to join other ranks;
/// - `CALL sirius_rank_exit()`: replies, then shuts Sirius down and ends the process, for the
///   rank's launcher to start it again. The exchange's peers are fixed once it connects, so
///   joining a new set of ranks takes a new process. Refused in a process that has not joined.
void register_rank_control_functions(duckdb::DatabaseInstance& instance);

/// Records that this process's exchange connected to its peers, which it cannot change after.
void note_exchange_joined() noexcept;

}  // namespace sirius::exchange
