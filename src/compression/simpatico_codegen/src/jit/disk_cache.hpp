#pragma once

#include <string>
#include <vector>

namespace codegen::jit::detail {

// Private best-effort storage. The record verifies length and payload SHA-256
// before exposing bytes to the CUDA loader. Errors are misses, never exceptions.
bool read_cubin_file(const std::string& path, std::vector<char>& bytes) noexcept;
void write_cubin_file_atomic(const std::string& path, const std::vector<char>& bytes) noexcept;

// Explicit maintenance only, with writers stopped. Recognizes legacy flat files,
// v2 namespaces, and their abandoned temporaries; does not follow symlinks.
void clear_disk_cache(const std::string& root) noexcept;

}  // namespace codegen::jit::detail
