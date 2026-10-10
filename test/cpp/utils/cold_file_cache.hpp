// Copyright 2026, Sirius Contributors. Licensed under the Apache License, Version 2.0.
#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace sirius::test {

struct cold_cache_result {
  size_t files{0}, bytes{0}, pages{0};
};

// Only the listed local files' OS page cache is covered, not device/controller caches.
inline cold_cache_result evict_file_cache(std::string const& manifest)
{
  std::ifstream input(manifest);
  if (!input) throw std::runtime_error("cannot open cold-cache file list");
  struct file {
    int fd;
    size_t bytes;
    std::string path;
  };
  std::vector<file> files;
  struct close_files {
    std::vector<file>& files;
    ~close_files()
    {
      for (auto const& f : files)
        ::close(f.fd);
    }
  } cleanup{files};
  cold_cache_result result;
  auto fail = [](int error, char const* operation) {
    if (error) throw std::system_error(error, std::generic_category(), operation);
  };
  std::string path;
  while (std::getline(input, path)) {
    if (path.empty() || path.front() != '/')
      throw std::runtime_error("cold-cache file list requires absolute local paths");
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) fail(errno, "cold-cache open");
    files.push_back({fd, 0, path});
    struct stat st{};
    if (::fstat(fd, &st)) fail(errno, "cold-cache fstat");
    if (!S_ISREG(st.st_mode)) throw std::runtime_error("cold-cache input is not a regular file");
    files.back().bytes = static_cast<size_t>(st.st_size);
    // DONTNEED is allowed to retain dirty pages. Flush before eviction and verify afterward.
    if (::fsync(fd)) fail(errno, "cold-cache fsync");
    fail(::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED), "cold-cache fadvise");
    ::close(fd);
    files.back().fd = -1;
  }
  if (files.empty()) throw std::runtime_error("empty cold-cache file list");
  auto page_size = ::sysconf(_SC_PAGESIZE);
  if (page_size <= 0) throw std::runtime_error("cannot determine page size");
  for (auto& f : files) {
    ++result.files;
    result.bytes += f.bytes;
    if (!f.bytes) continue;
    f.fd = ::open(f.path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (f.fd < 0) fail(errno, "cold-cache verification open");
    auto pages = (f.bytes + page_size - 1) / page_size;
    std::vector<unsigned char> resident(pages);
    void* address = ::mmap(nullptr, f.bytes, PROT_READ, MAP_SHARED, f.fd, 0);
    if (address == MAP_FAILED) fail(errno, "cold-cache mmap");
    auto rc    = ::mincore(address, f.bytes, resident.data());
    auto error = errno;
    ::munmap(address, f.bytes);
    if (rc) fail(error, "cold-cache mincore");
    for (auto page : resident)
      if (page & 1) throw std::runtime_error("cold-cache eviction left resident pages");
    result.pages += pages;
    ::close(f.fd);
    f.fd = -1;
  }
  return result;
}
}  // namespace sirius::test
