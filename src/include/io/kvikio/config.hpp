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

#include <kvikio/compat_mode.hpp>

#include <cstddef>
#include <optional>

// NOTE ON NAMESPACING: this lives in `sirius::io`, not `sirius::io::kvikio`,
// deliberately.  A `sirius::io::kvikio` namespace would shadow the upstream
// `::kvikio` namespace for every unqualified `kvikio::` use inside
// `sirius::io` (e.g. `kvikio::FileHandle` in kvikio_context.hpp), forcing
// global qualification everywhere.  The header still lives under io/kvikio/ so
// the file layout matches the uring / rest backends.
namespace sirius::io {

/**
 * @brief Tunables for the kvikIO-backed local-file ioctx (@c kvikio_context).
 *
 * Every field is optional and means "leave kvikIO's own default alone".  kvikIO
 * seeds each setting from an environment variable at first use
 * (@c KVIKIO_NTHREADS, @c KVIKIO_TASK_SIZE, ...), so an unset field here keeps
 * that env-var value; an engaged field overrides it.
 *
 * @warning PROCESS-GLOBAL.  Every field except @c compat_mode maps to a setter
 *          on kvikIO's @c kvikio::defaults singleton, so applying a config
 *          mutates state shared by ALL kvikIO users in the process.  Two
 *          @c kvikio_context instances built with different configs do not get
 *          independent settings; the last one constructed wins.  Treat this as
 *          startup configuration, applied once.  @c nthreads is especially
 *          disruptive: kvikIO's setter waits for all running tasks, destroys
 *          the pool, and rebuilds it.
 *
 * @c compat_mode is the exception — it is passed per @c FileHandle at open
 * time, so it affects only files this ioctx opens and mutates nothing global.
 *
 * Write-side knobs are intentionally absent: @c kvikio_context opens every file
 * read-only, so they would be dead config.
 */
struct kvikio_config {
  /// How many scan tasks the readahead manager may keep in flight against this
  /// backend at once.  Zero — the default — disables readahead for kvikIO:
  /// kvikIO owns its own process-global task pool and does its own splitting,
  /// so a second scheduler stacked on top would only fight it for the same
  /// threads.  Not a kvikIO setting, so unlike the fields below it is a plain
  /// value rather than an optional override.
  std::size_t n_max_concurrent_scans{0};

  /// Threads in kvikIO's task pool — the parallelism bound for a single
  /// @c pread (it splits the read into @c task_size chunks across this pool).
  /// Env: @c KVIKIO_NTHREADS (default 1).  Must be non-zero.
  std::optional<unsigned int> nthreads;

  /// Chunk size a parallel read is split into.  Env: @c KVIKIO_TASK_SIZE
  /// (default 4 MiB).  Must be non-zero.  With @c auto_direct_io_read on, keep
  /// it a multiple of the page size so tasks start page-aligned — otherwise
  /// kvikIO falls back to buffered I/O for the misaligned head/tail.
  std::optional<std::size_t> task_size;

  /// Minimum read size that goes through GDS + the thread pool; smaller reads
  /// take a direct POSIX shortcut that skips the pool.  Env:
  /// @c KVIKIO_GDS_THRESHOLD (default 1 MiB).  Zero is legal (always use GDS).
  std::optional<std::size_t> gds_threshold;

  /// Host staging buffer size for device reads that cannot go straight to GPU
  /// memory.  Env: @c KVIKIO_BOUNCE_BUFFER_SIZE (default 16 MiB).  Must be
  /// non-zero.
  std::optional<std::size_t> bounce_buffer_size;

  /// Use Direct I/O (@c O_DIRECT) for POSIX reads where possible.  Env:
  /// @c KVIKIO_AUTO_DIRECT_IO_READ.  Applies to the POSIX path only — the
  /// cuFile/GDS path manages its own I/O mode — so it matters most in
  /// compatibility mode or below @c gds_threshold.
  std::optional<bool> auto_direct_io_read;

  /// For device reads, align offsets down and sizes up to page boundaries so
  /// the whole transfer is pure Direct I/O, at the cost of reading extra bytes.
  /// When false (kvikIO's default) the unaligned head/tail falls back to
  /// buffered I/O.  Env: @c KVIKIO_AUTO_DIRECT_IO_READ_OVERREAD.  Requires
  /// @c auto_direct_io_read to have any effect; device path only.
  std::optional<bool> auto_direct_io_read_overread;

  /// Give each block device its own thread pool (each sized @c nthreads)
  /// instead of sharing one global pool.  Helps when reads span several
  /// physical devices.  Env: @c KVIKIO_THREAD_POOL_PER_BLOCK_DEVICE (default
  /// false).  Takes effect only for files opened after it is applied.
  std::optional<bool> thread_pool_per_block_device;

  /// cuFile vs POSIX selection, applied PER FILE HANDLE (not global):
  /// @c OFF enforces cuFile/GDS, @c ON enforces POSIX, @c AUTO tries cuFile and
  /// falls back.  Unset leaves it to kvikIO's own default, which honours
  /// @c KVIKIO_COMPAT_MODE.
  std::optional<kvikio::CompatMode> compat_mode;
};

/**
 * @brief Push @p cfg's engaged fields into kvikIO's global @c defaults.
 *
 * Unset fields are left untouched, preserving kvikIO's env-var-seeded values.
 * @c compat_mode is NOT applied here — it is per-handle and consumed at open
 * time by @c kvikio_context::create_io_object.
 *
 * Called once by the @c kvikio_context constructor; exposed so an application
 * that wants to configure kvikIO at startup (before any ioctx exists) can do
 * the same thing explicitly.
 *
 * @throw std::invalid_argument on a zero @c nthreads, @c task_size, or
 *        @c bounce_buffer_size.
 */
void apply_kvikio_defaults(kvikio_config const& cfg);

}  // namespace sirius::io
