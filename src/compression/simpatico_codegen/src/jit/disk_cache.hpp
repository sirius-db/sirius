#pragma once

#include "cache_identity.hpp"

#include <string>
#include <vector>

namespace codegen::jit::detail {

// Private best-effort storage. The record verifies length and payload XXH3-128
// before exposing bytes to the CUDA loader. Errors are misses, never exceptions.
bool read_cubin_file(const std::string& path, std::vector<char>& bytes) noexcept;
void write_cubin_file_atomic(const std::string& path, const std::vector<char>& bytes) noexcept;

// First disk use: touch the current environment; keep it, the three newest
// others, and any used within one hour. Best effort, not a live-process lease.
// Follows the trusted root, but no symlinks below it; no recursive traversal.
void prune_disk_cache(const std::string& root, const Digest& current_environment) noexcept;

// Explicit maintenance only, with writers stopped. Recognizes legacy flat files,
// v2 namespaces, and their abandoned temporaries; follows only the trusted root.
void clear_disk_cache(const std::string& root) noexcept;

}  // namespace codegen::jit::detail
