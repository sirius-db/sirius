#include "jit/cache_identity.hpp"
#include "jit/disk_cache.hpp"

#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs                      = std::filesystem;
namespace cache                   = codegen::jit::detail;
constexpr std::size_t header_size = 8 + 8 + cache::Digest{}.size();
constexpr std::size_t hex_size    = cache::Digest{}.size() * 2;

enum class Fault {
  none,
  short_write,
  interrupted,
  partial_failure,
  zero_write,
  fchmod,
  close,
  rename
};
thread_local Fault fault = Fault::none;
thread_local int writes  = 0;

extern "C" ssize_t __real_write(int, const void*, size_t);
extern "C" int __real_fchmod(int, mode_t);
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
extern "C" int __wrap_fchmod(int fd, mode_t mode)
{
  if (fault == Fault::fchmod) {
    errno = EACCES;
    return -1;
  }
  return __real_fchmod(fd, mode);
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
    auto pattern = (fs::absolute(fs::temp_directory_path()) / "simpatico-disk-XXXXXX").string();
    const char* created = ::mkdtemp(pattern.data());
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

bool uniform_payload(const std::vector<char>& bytes, std::size_t expected_size)
{
  return bytes.size() == expected_size &&
         std::ranges::all_of(bytes, [&](char value) { return value == bytes.front(); });
}

void publication_permissions(const fs::path& root)
{
  struct RestoreUmask {
    mode_t previous = ::umask(0022);
    ~RestoreUmask() { ::umask(previous); }
  } restore;
  const auto path = root / "permissions" / "test.cubin";
  const std::vector<char> bytes(32, 'p');
  // All mask changes finish before any concurrent tests start.
  for (const mode_t mask : {0022, 0077, 0022}) {
    ::umask(mask);
    cache::write_cubin_file_atomic(path.string(), bytes);
    struct stat metadata{};
    require(::stat(path.c_str(), &metadata) == 0, "published permissions unavailable");
    require((metadata.st_mode & 0777) == (0666 & ~mask),
            "publication did not respect the current umask");
  }
}

void publication_and_faults(const fs::path& root)
{
  const auto path = root / "records" / "test.cubin";
  std::vector<char> bytes;
  require(!cache::read_cubin_file(path.string(), bytes), "missing record should miss");
  const std::vector<char> original(4096, 'a'), replacement(4096, 'b');
  cache::write_cubin_file_atomic(path.string(), original);
  require(read(path) == original, "initial roundtrip");
  require(fs::file_size(path) == header_size + original.size(), "record digest width");
  {
    std::array<char, header_size> header{};
    std::ifstream input(path, std::ios::binary);
    require(bool(input.read(header.data(), header.size())), "record header read");
    require(std::string_view(header.data(), 8) == "SIMPJIT2", "record format marker");
    const auto digest = cache::content_identity({original.data(), original.size()});
    require(
      std::equal(
        digest.begin(), digest.end(), reinterpret_cast<const unsigned char*>(header.data() + 16)),
      "record stores canonical XXH3-128");
  }
  for (const auto injected :
       {Fault::partial_failure, Fault::zero_write, Fault::fchmod, Fault::close, Fault::rename}) {
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
  for (const auto length : {std::size_t{0},
                            std::size_t{1},
                            header_size - 1,
                            header_size,
                            header_size + 1,
                            header_size + original.size() - 1}) {
    cache::write_cubin_file_atomic(path.string(), original);
    fs::resize_file(path, length);
    require(!cache::read_cubin_file(path.string(), bytes), "truncated record accepted");
  }
  for (const auto offset : {0, 7, 8, 16}) {
    cache::write_cubin_file_atomic(path.string(), original);
    {
      std::fstream output(path, std::ios::binary | std::ios::in | std::ios::out);
      output.seekg(offset);
      const auto value = output.get();
      output.seekp(offset);
      output.put(static_cast<char>(value ^ 1));
    }
    require(!cache::read_cubin_file(path.string(), bytes), "corrupt record header accepted");
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
    if (!cache::read_cubin_file(path, bytes) || !uniform_payload(bytes, initial.size())) bad = true;
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
      const bool valid =
        cache::read_cubin_file(path, loaded) && uniform_payload(loaded, bytes.size());
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

void environment_namespaces(const fs::path& root)
{
  const auto location = root / "environments";
  const std::array<std::string_view, 1> options{"-arch=sm_120"};
  const auto request = cache::request_identity({"kernel", "entry", "test.cu", options});
  const std::array<cache::EnvironmentView, 5> environments{{
    {"project", "cccl", 13, 4},
    {"project-edited", "cccl", 13, 4},
    {"project", "cccl-edited", 13, 4},
    {"project", "cccl", 14, 4},
    {"project", "cccl", 13, 5},
  }};
  std::vector<fs::path> paths;
  for (std::size_t index = 0; index < environments.size(); ++index) {
    const cache::CompilationIdentity identity{cache::environment_identity(environments[index]),
                                              request};
    const auto path = location / identity.relative_path();
    require(!fs::exists(path), "different environment selected an existing record");
    cache::write_cubin_file_atomic(path.string(), {static_cast<char>('a' + index)});
    paths.push_back(path);
  }
  for (std::size_t index = 0; index < paths.size(); ++index)
    require(read(paths[index]) == std::vector<char>{static_cast<char>('a' + index)},
            "publishing an environment changed another environment's record");

  const auto unrelated = paths.front().parent_path() / "keep.txt";
  touch(unrelated);
  cache::clear_disk_cache(location.string());
  for (const auto& path : paths)
    require(!fs::exists(path), "cleanup missed the version-based environment layout");
  require(fs::is_regular_file(unrelated), "cleanup removed an unrelated environment file");
}

void cleanup(const fs::path& root)
{
  const auto location    = root / "cleanup";
  const auto environment = location / "v2" / std::string(hex_size, 'a');
  const auto request     = std::string(hex_size, 'b') + ".cubin";
  const auto legacy      = "0123456789abcdef_a120_c13030_d13030.cubin";
  touch(environment / request);
  touch(environment / (request + ".tmp.Abc123"));
  touch(location / legacy);
  touch(location / (std::string(legacy) + ".tmp.123"));
  const auto outside = root / "outside";
  touch(outside / request);
  fs::create_directory_symlink(outside, location / "v2" / std::string(hex_size, 'c'));
  fs::create_symlink(outside / request, environment / (std::string(hex_size, 'd') + ".cubin"));
  touch(environment / "keep.txt");
  touch(environment / "nested" / request);
  touch(location / "unrelated.cubin");
  touch(location / "v3" / std::string(hex_size, 'a') / request);
  touch(location / "v2" / "unknown-namespace" / request);
  const auto root_link = root / "cleanup-root-link";
  fs::create_directory_symlink(location, root_link);
  cache::clear_disk_cache(root_link.string());
  require(fs::is_symlink(root_link), "cleanup removed the trusted root symlink");
  require(!fs::exists(environment / request) && !fs::exists(location / legacy),
          "recognized entries retained");
  require(!fs::exists(environment / (request + ".tmp.Abc123")), "abandoned temporary retained");
  require(!fs::exists(location / (std::string(legacy) + ".tmp.123")), "legacy temporary retained");
  for (const auto& path : {outside / request,
                           environment / "keep.txt",
                           environment / "nested" / request,
                           location / "unrelated.cubin",
                           location / "v3" / std::string(hex_size, 'a') / request,
                           location / "v2" / "unknown-namespace" / request}) {
    require(fs::exists(path), "cleanup removed unrelated content");
  }
  require(fs::is_symlink(location / "v2" / std::string(hex_size, 'c')),
          "cleanup removed namespace symlink");
  require(fs::is_symlink(environment / (std::string(hex_size, 'd') + ".cubin")),
          "cleanup removed file symlink");
  cache::clear_disk_cache("");
  cache::clear_disk_cache((root / "missing").string());
}

fs::path seed_environment(const fs::path& root, std::string_view label, fs::file_time_type modified)
{
  const auto path = root / "v2" / cache::hex_digest(cache::content_identity(label));
  touch(path / (std::string(hex_size, 'b') + ".cubin"));
  fs::last_write_time(path, modified);
  return path;
}

void pruning(const fs::path& root)
{
  using namespace std::chrono;
  const auto location = root / "pruning";
  const auto now      = fs::file_time_type::clock::now();
  const auto current  = seed_environment(location, "current", now - hours(24));
  const auto request  = std::string(hex_size, 'b') + ".cubin";
  std::vector<fs::path> others;
  for (int index = 0; index < 5; ++index)
    others.push_back(
      seed_environment(location, "other-" + std::to_string(index), now - hours(2 + index)));

  // Even stale recognized environments retain unknown files, nested content,
  // and symlinks. Such leftovers can prevent removing their directory.
  const auto preserved = others.back();
  const auto outside   = root / "prune-outside";
  touch(outside / request);
  touch(preserved / "keep.txt");
  touch(preserved / "nested" / request);
  touch(preserved / (request + ".tmp.Abc123"));
  fs::create_symlink(outside / request, preserved / (std::string(hex_size, 'd') + ".cubin"));
  fs::create_directory_symlink(outside, location / "v2" / std::string(hex_size, 'c'));
  touch(location / "v2" / "unknown-namespace" / request);
  touch(location / "v3" / std::string(hex_size, 'e') / request);
  touch(location / "unrelated.cubin");
  const auto legacy = location / "0123456789abcdef_a120_c13030_d13030.cubin";
  touch(legacy);
  touch(legacy.string() + ".tmp.123");
  const auto legacy_link = location / "0123456789abcdef_a121_c13030_d13030.cubin";
  fs::create_symlink(outside / request, legacy_link);
  fs::last_write_time(preserved, now - hours(6));

  const auto root_link = root / "prune-root-link";
  fs::create_directory_symlink(location, root_link);
  cache::prune_disk_cache(root_link.string(), cache::content_identity("current"));
  require(fs::is_symlink(root_link), "pruning removed the trusted root symlink");
  require(fs::exists(current / request), "pruned the current environment");
  require(fs::last_write_time(current) >= now, "first use did not refresh the current environment");
  for (std::size_t index = 0; index < others.size(); ++index)
    require(fs::exists(others[index] / request) == (index < 3), "incorrect environment retention");
  require(!fs::exists(others[3]), "empty stale environment directory retained");
  require(!fs::exists(preserved / (request + ".tmp.Abc123")), "stale temporary retained");
  require(!fs::exists(legacy) && !fs::exists(legacy.string() + ".tmp.123"),
          "automatic maintenance retained legacy records");
  for (const auto& path : {preserved / "keep.txt",
                           preserved / "nested" / request,
                           outside / request,
                           location / "v2" / "unknown-namespace" / request,
                           location / "v3" / std::string(hex_size, 'e') / request,
                           location / "unrelated.cubin"})
    require(fs::exists(path), "automatic pruning removed unrelated content");
  for (const auto& path : {legacy_link,
                           location / "v2" / std::string(hex_size, 'c'),
                           preserved / (std::string(hex_size, 'd') + ".cubin")})
    require(fs::is_symlink(path), "automatic pruning removed a symlink");

  // Grace extends retention past four directories, including future timestamps.
  const auto grace_root = root / "grace";
  std::vector<fs::path> recent;
  for (const int age : {-60, 5, 10, 15, 59, 61})
    recent.push_back(seed_environment(grace_root, std::to_string(age), now - minutes(age)));
  cache::prune_disk_cache(grace_root.string(), cache::content_identity("current"));
  for (std::size_t index = 0; index < recent.size(); ++index)
    require(fs::exists(recent[index] / request) == (index < recent.size() - 1),
            "one-hour grace or future timestamp not respected");

  // Only the trusted root may be a symlink; descendants are never followed.
  fs::last_write_time(outside, now - hours(24));
  const auto outside_time = fs::last_write_time(outside);
  fs::create_directories(root / "version-link");
  fs::create_directory_symlink(outside, root / "version-link" / "v2");
  fs::create_directories(root / "current-link" / "v2");
  fs::create_directory_symlink(
    outside, root / "current-link" / "v2" / cache::hex_digest(cache::content_identity("current")));
  for (const auto& path : {root / "version-link", root / "current-link"}) {
    cache::prune_disk_cache(path.string(), cache::content_identity("current"));
    cache::clear_disk_cache(path.string());
  }
  require(fs::is_symlink(root / "version-link" / "v2"), "maintenance removed version symlink");
  require(fs::is_symlink(root / "current-link" / "v2" /
                         cache::hex_digest(cache::content_identity("current"))),
          "maintenance removed current environment symlink");
  require(fs::last_write_time(outside) == outside_time, "maintenance touched a symlink target");
  require(std::distance(fs::directory_iterator(outside), fs::directory_iterator{}) == 1,
          "maintenance created content through a symlink");
  touch(root / "prune-not-a-directory");
  cache::prune_disk_cache((root / "prune-not-a-directory").string(), cache::Digest{});
  require(fs::is_regular_file(root / "prune-not-a-directory"), "modified unusable cache root");
  cache::prune_disk_cache("", cache::Digest{});
}

void concurrent_pruning(const fs::path& root)
{
  const auto location = root / "concurrent-pruning";
  const auto now      = fs::file_time_type::clock::now();
  const auto old      = seed_environment(location, "old", now - std::chrono::hours(24));
  const auto path     = (old / (std::string(hex_size, 'b') + ".cubin")).string();
  for (int index = 0; index < 3; ++index)
    seed_environment(location, std::to_string(index), now - std::chrono::hours(2 + index));
  cache::write_cubin_file_atomic(path, std::vector<char>(65536, 'a'));
  std::barrier start(3);
  std::atomic<int> remaining{2};
  std::atomic<bool> bad{false};
  std::thread writer([&] {
    start.arrive_and_wait();
    for (int index = 0; index < 100; ++index)
      cache::write_cubin_file_atomic(path, std::vector<char>(65536, 'a' + index % 2));
    --remaining;
  });
  std::thread pruner([&] {
    start.arrive_and_wait();
    for (int index = 0; index < 100; ++index) {
      // Force eligibility to exercise pruning even as publication refreshes
      // directory mtime. A miss/failed publication is allowed; a partial hit is not.
      std::error_code error;
      fs::last_write_time(old, now - std::chrono::hours(24), error);
      cache::prune_disk_cache(location.string(), cache::content_identity("current"));
    }
    --remaining;
  });
  start.arrive_and_wait();
  while (remaining) {
    std::vector<char> bytes;
    if (cache::read_cubin_file(path, bytes) && !uniform_payload(bytes, 65536)) bad = true;
  }
  writer.join();
  pruner.join();
  require(!bad, "concurrent pruning exposed a partial/mixed record");
  const std::vector<char> repaired(65536, 'z');
  cache::write_cubin_file_atomic(path, repaired);
  require(read(path) == repaired, "publication did not recover after pruning");
}

int main()
{
  try {
    Directory root;
    publication_permissions(root.path);
    publication_and_faults(root.path);
    concurrent_writers(root.path);
    environment_namespaces(root.path);
    cleanup(root.path);
    pruning(root.path);
    concurrent_pruning(root.path);
    std::puts("disk cache: publication, faults, concurrency, cleanup, pruning passed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
