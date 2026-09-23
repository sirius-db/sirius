#include "jit/disk_cache.hpp"

#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs    = std::filesystem;
namespace cache = codegen::jit::detail;

enum class Fault { none, short_write, interrupted, partial_failure, zero_write, close, rename };
thread_local Fault fault = Fault::none;
thread_local int writes  = 0;

extern "C" ssize_t __real_write(int, const void*, size_t);
extern "C" int __real_close(int);
extern "C" int __real_rename(const char*, const char*);

// Link-time fault injection exercises the real production syscall handling,
// without test-only branches or overloads in the production API.
extern "C" ssize_t __wrap_write(int fd, const void* bytes, size_t size)
{
  ++writes;
  if (fault == Fault::interrupted && writes == 1) {
    errno = EINTR;
    return -1;
  }
  if (fault == Fault::partial_failure && writes > 1) {
    errno = ENOSPC;
    return -1;
  }
  if (fault == Fault::zero_write) return 0;
  if (fault == Fault::short_write || fault == Fault::partial_failure)
    size = std::min(size, size_t{7});
  return __real_write(fd, bytes, size);
}
extern "C" int __wrap_close(int fd)
{
  const int result = __real_close(fd);
  if (fault == Fault::close) {
    errno = EIO;
    return -1;
  }
  return result;
}
extern "C" int __wrap_rename(const char* from, const char* to)
{
  if (fault == Fault::rename) {
    errno = EACCES;
    return -1;
  }
  return __real_rename(from, to);
}

void require(bool condition, const char* description)
{
  if (!condition) throw std::runtime_error(description);
}

struct Directory {
  Directory()
  {
    char pattern[]      = "/tmp/simpatico-disk-XXXXXX";
    const char* created = ::mkdtemp(pattern);
    if (!created) throw std::runtime_error("mkdtemp failed");
    path = created;
  }
  ~Directory()
  {
    std::error_code error;
    fs::remove_all(path, error);
  }
  fs::path path;
};

void touch(const fs::path& path, std::string_view bytes = "fixture")
{
  fs::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary);
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  require(bool(output), "fixture write failed");
}

std::vector<char> read(const fs::path& path)
{
  std::vector<char> bytes;
  require(cache::read_cubin_file(path.string(), bytes), "expected readable record");
  return bytes;
}

void publication_and_faults(const fs::path& root)
{
  const auto path = root / "records" / "test.cubin";
  std::vector<char> bytes;
  require(!cache::read_cubin_file(path.string(), bytes), "missing record should miss");
  const std::vector<char> original(4096, 'a'), replacement(4096, 'b');
  cache::write_cubin_file_atomic(path.string(), original);
  require(read(path) == original, "initial roundtrip");
  for (const auto injected :
       {Fault::partial_failure, Fault::zero_write, Fault::close, Fault::rename}) {
    fault  = injected;
    writes = 0;
    cache::write_cubin_file_atomic(path.string(), replacement);
    fault = Fault::none;
    require(read(path) == original, "failed publication replaced the destination");
    require(
      std::distance(fs::directory_iterator(path.parent_path()), fs::directory_iterator{}) == 1,
      "failed publication left its temporary file");
  }
  for (const auto injected : {Fault::short_write, Fault::interrupted}) {
    fault  = injected;
    writes = 0;
    cache::write_cubin_file_atomic(path.string(), replacement);
    fault = Fault::none;
    require(read(path) == replacement, "short/interrupted write should complete");
  }
  // A failed rename must not remove an existing destination directory.
  fs::create_directory(root / "destination-directory");
  cache::write_cubin_file_atomic((root / "destination-directory").string(), replacement);
  require(fs::is_directory(root / "destination-directory"), "rename failure removed directory");
  touch(root / "not-a-directory");
  cache::write_cubin_file_atomic((root / "not-a-directory" / "child").string(), replacement);
  require(fs::is_regular_file(root / "not-a-directory"), "unwritable root modified");

  // Reject damaged records before calling the CUDA loader.
  for (const auto length : {0, 1, 47, 48, 49, 4095}) {
    cache::write_cubin_file_atomic(path.string(), original);
    fs::resize_file(path, length);
    require(!cache::read_cubin_file(path.string(), bytes), "truncated record accepted");
  }
  cache::write_cubin_file_atomic(path.string(), original);
  {
    std::fstream output(path, std::ios::binary | std::ios::in | std::ios::out);
    output.seekp(100);
    output.put('x');
  }
  require(!cache::read_cubin_file(path.string(), bytes), "payload corruption accepted");
  fs::resize_file(path, 300ULL * 1024 * 1024);  // Sparse fixture: must not allocate this size.
  require(!cache::read_cubin_file(path.string(), bytes), "oversized file accepted");
  fs::create_symlink(path, root / "link");
  require(!cache::read_cubin_file((root / "link").string(), bytes), "followed symlink");
  require(::mkfifo((root / "fifo").c_str(), 0600) == 0, "mkfifo failed");
  require(!cache::read_cubin_file((root / "fifo").string(), bytes),
          "FIFO should miss without blocking");
}

void concurrent_writers(const fs::path& root)
{
  const auto path = (root / "concurrent" / "same-key.cubin").string();
  const std::vector<char> initial(65536, 'a');
  cache::write_cubin_file_atomic(path, initial);
  constexpr int count = 6;
  std::barrier start(count + 1);
  std::atomic<int> remaining{count};
  std::atomic<bool> bad{false};
  std::vector<std::thread> threads;
  for (int index = 0; index < count; ++index) {
    threads.emplace_back([&, index] {
      const std::vector<char> bytes(65536, static_cast<char>('a' + index));
      start.arrive_and_wait();
      for (int iteration = 0; iteration < 40; ++iteration)
        cache::write_cubin_file_atomic(path, bytes);
      --remaining;
    });
  }
  start.arrive_and_wait();
  while (remaining) {
    std::vector<char> bytes;
    if (!cache::read_cubin_file(path, bytes) || bytes.size() != initial.size() ||
        !std::all_of(
          bytes.begin(), bytes.end(), [&](char value) { return value == bytes.front(); }))
      bad = true;
  }
  for (auto& thread : threads)
    thread.join();
  require(!bad, "threaded reader observed partial/mixed publication");
  require(std::distance(fs::directory_iterator(fs::path(path).parent_path()),
                        fs::directory_iterator{}) == 1,
          "concurrent writers left temporary files");

  // No threads survive fork; children publish the same key in separate processes.
  std::vector<pid_t> children;
  for (int index = 0; index < 4; ++index) {
    const auto pid = ::fork();
    require(pid >= 0, "fork failed");
    if (pid == 0) {
      const std::vector<char> bytes(65536, static_cast<char>('a' + index));
      for (int iteration = 0; iteration < 30; ++iteration)
        cache::write_cubin_file_atomic(path, bytes);
      std::vector<char> loaded;
      const bool valid = cache::read_cubin_file(path, loaded) && loaded.size() == bytes.size() &&
                         std::all_of(loaded.begin(), loaded.end(), [&](char value) {
                           return value == loaded.front();
                         });
      ::_exit(valid ? 0 : 1);
    }
    children.push_back(pid);
  }
  for (auto child : children) {
    int status = 0;
    require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "process publication failed");
  }
}

void cleanup(const fs::path& root)
{
  const auto location    = root / "cleanup";
  const auto environment = location / "v2" / std::string(64, 'a');
  const auto request     = std::string(64, 'b') + ".cubin";
  const auto legacy      = "0123456789abcdef_a120_c13030_d13030.cubin";
  touch(environment / request);
  touch(environment / (request + ".tmp.Abc123"));
  touch(location / legacy);
  touch(location / (std::string(legacy) + ".tmp.123"));
  const auto outside = root / "outside";
  touch(outside / request);
  fs::create_directory_symlink(outside, location / "v2" / std::string(64, 'c'));
  fs::create_symlink(outside / request, environment / (std::string(64, 'd') + ".cubin"));
  touch(environment / "keep.txt");
  touch(environment / "nested" / request);
  touch(location / "unrelated.cubin");
  touch(location / "v3" / std::string(64, 'a') / request);
  touch(location / "v2" / "unknown-namespace" / request);
  cache::clear_disk_cache(location.string());
  require(!fs::exists(environment / request) && !fs::exists(location / legacy),
          "recognized entries retained");
  require(!fs::exists(environment / (request + ".tmp.Abc123")), "abandoned temporary retained");
  require(!fs::exists(location / (std::string(legacy) + ".tmp.123")), "legacy temporary retained");
  for (const auto& path : {outside / request,
                           environment / "keep.txt",
                           environment / "nested" / request,
                           location / "unrelated.cubin",
                           location / "v3" / std::string(64, 'a') / request,
                           location / "v2" / "unknown-namespace" / request}) {
    require(fs::exists(path), "cleanup removed unrelated content");
  }
  require(fs::is_symlink(location / "v2" / std::string(64, 'c')),
          "cleanup removed namespace symlink");
  require(fs::is_symlink(environment / (std::string(64, 'd') + ".cubin")),
          "cleanup removed file symlink");
  cache::clear_disk_cache("");
  cache::clear_disk_cache((root / "missing").string());
}

int main()
{
  try {
    Directory root;
    publication_and_faults(root.path);
    concurrent_writers(root.path);
    cleanup(root.path);
    std::puts("disk cache: publication, faults, concurrency, cleanup passed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
