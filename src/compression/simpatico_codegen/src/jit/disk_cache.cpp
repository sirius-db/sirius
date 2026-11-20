#include "disk_cache.hpp"

#include "cache_identity.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <regex>
#include <span>
#include <string_view>
#include <utility>

namespace codegen::jit::detail {
namespace {

constexpr std::string_view magic = "SIMPJIT2";
static_assert(magic.substr(7) == kCacheFormatVersion.substr(1),
              "record marker must match the cache format version");
constexpr std::size_t length_offset = magic.size();
constexpr std::size_t digest_offset = length_offset + sizeof(uint64_t);
constexpr std::size_t header_size   = digest_offset + Digest{}.size();
// Corrupt lengths must not cause unbounded allocation. Larger kernels still
// compile and remain reusable in memory; only their persistence is skipped.
constexpr uint64_t max_cubin_bytes = 256ULL * 1024 * 1024;
// Retention policy from #2088, applied to input-environment namespaces.
constexpr std::size_t kept_environments = 4;
constexpr auto environment_grace        = std::chrono::hours(1);

class Descriptor {
 public:
  explicit Descriptor(int fd) : fd_(fd) {}
  ~Descriptor()
  {
    if (fd_ >= 0) ::close(fd_);
  }
  Descriptor(const Descriptor&)            = delete;
  Descriptor& operator=(const Descriptor&) = delete;
  int get() const { return fd_; }
  bool close() { return ::close(std::exchange(fd_, -1)) == 0; }

 private:
  int fd_;
};

struct Temporary {
  std::string path;
  ~Temporary()
  {
    if (!path.empty()) ::unlink(path.c_str());
  }
};

bool read_all(int fd, std::span<char> bytes)
{
  while (!bytes.empty()) {
    const auto size = ::read(fd, bytes.data(), bytes.size());
    if (size < 0 && errno == EINTR) continue;
    if (size <= 0) return false;
    bytes = bytes.subspan(static_cast<std::size_t>(size));
  }
  return true;
}

bool write_all(int fd, std::span<const char> bytes)
{
  while (!bytes.empty()) {
    const auto size = ::write(fd, bytes.data(), bytes.size());
    if (size < 0 && errno == EINTR) continue;
    if (size <= 0) return false;
    bytes = bytes.subspan(static_cast<std::size_t>(size));
  }
  return true;
}

std::optional<mode_t> record_permissions()
{
  // Read per publication: never change the host process's umask or cache a
  // value that could become stale. Unavailable/malformed status keeps the
  // owner-only temporary permissions unchanged.
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    const std::string_view field = line;
    if (!field.starts_with("Umask:")) continue;
    const auto first = field.find_first_not_of(" \t", 6);
    if (first == std::string_view::npos) break;
    unsigned mask = 0;
    const auto [end, error] =
      std::from_chars(field.data() + first, field.data() + field.size(), mask, 8);
    if (error == std::errc{} && end == field.data() + field.size() && mask <= 0777)
      return static_cast<mode_t>(0666 & ~mask);
    break;
  }
  return std::nullopt;
}

bool is_directory_no_follow(const std::filesystem::path& path)
{
  std::error_code error;
  return std::filesystem::is_directory(std::filesystem::symlink_status(path, error));
}

template <class Visit>
bool visit_directory(const std::filesystem::path& path, Visit visit)
{
  if (!is_directory_no_follow(path)) return false;
  std::error_code error;
  std::filesystem::directory_iterator current(path, error), end;
  while (!error && current != end) {
    visit(*current);
    current.increment(error);
  }
  return !error;
}

bool is_regular(const std::filesystem::directory_entry& entry)
{
  std::error_code error;
  return std::filesystem::is_regular_file(entry.symlink_status(error));
}

void remove_entry(const std::filesystem::path& path)
{
  std::error_code error;
  std::filesystem::remove(path, error);  // Directories are removed only if empty.
}

bool is_environment(const std::filesystem::directory_entry& entry)
{
  if (!is_directory_no_follow(entry.path())) return false;
  const auto name = entry.path().filename().string();
  return name.size() == Digest{}.size() * 2 && std::ranges::all_of(name, [](char value) {
           return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
         });
}

void remove_matching_files(const std::filesystem::path& path, const std::regex& pattern)
{
  visit_directory(path, [&](const auto& entry) {
    if (is_regular(entry) && std::regex_match(entry.path().filename().string(), pattern))
      remove_entry(entry.path());
  });
}

void clear_legacy_files(const std::filesystem::path& root)
{
  const std::regex legacy("[0-9a-f]{16}_a[0-9]+_c[0-9]+_d[0-9]+\\.cubin(\\.tmp\\.[0-9]+)?");
  remove_matching_files(root, legacy);
}

void clear_environment(const std::filesystem::path& path)
{
  const auto hex = "[0-9a-f]{" + std::to_string(Digest{}.size() * 2) + "}";
  const std::regex request(hex + "\\.cubin(\\.tmp\\.[A-Za-z0-9]{6})?");
  remove_matching_files(path, request);
  if (is_directory_no_follow(path)) remove_entry(path);
}

}  // namespace

bool read_cubin_file(const std::string& path, std::vector<char>& bytes) noexcept
{
  bytes.clear();
  try {
    Descriptor file(::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK));
    struct stat metadata{};
    if (file.get() < 0 || ::fstat(file.get(), &metadata) != 0 || !S_ISREG(metadata.st_mode) ||
        metadata.st_size <= static_cast<off_t>(header_size) ||
        static_cast<uint64_t>(metadata.st_size) > max_cubin_bytes + header_size)
      return false;
    std::array<char, header_size> header{};
    if (!read_all(file.get(), header) || !std::equal(magic.begin(), magic.end(), header.begin()))
      return false;
    uint64_t size = 0;
    for (std::size_t i = length_offset; i < digest_offset; ++i)
      size = (size << 8) | static_cast<unsigned char>(header[i]);
    if (size != static_cast<uint64_t>(metadata.st_size) - header.size()) return false;
    std::vector<char> payload(static_cast<std::size_t>(size));
    if (!read_all(file.get(), payload) || !file.close()) return false;
    const auto digest = content_identity({payload.data(), payload.size()});
    if (!std::equal(digest.begin(),
                    digest.end(),
                    reinterpret_cast<const unsigned char*>(header.data() + digest_offset)))
      return false;
    bytes = std::move(payload);
    return true;
  } catch (...) {
    return false;
  }
}

void write_cubin_file_atomic(const std::string& path, const std::vector<char>& bytes) noexcept
{
  try {
    if (bytes.empty() || bytes.size() > max_cubin_bytes) return;
    std::array<char, header_size> header{};
    std::ranges::copy(magic, header.begin());
    auto size = static_cast<uint64_t>(bytes.size());
    for (std::size_t i = digest_offset; i > length_offset; --i) {
      header[i - 1] = static_cast<char>(size & 255);
      size >>= 8;
    }
    const auto digest = content_identity({bytes.data(), bytes.size()});
    std::ranges::copy(digest, header.begin() + digest_offset);
    std::error_code error;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path(), error);
    if (error) return;
    Temporary temporary{path + ".tmp.XXXXXX"};
    Descriptor file(::mkostemp(temporary.path.data(), O_CLOEXEC));
    if (file.get() < 0) {
      temporary.path.clear();
      return;
    }
    if (!write_all(file.get(), header) || !write_all(file.get(), bytes)) return;
    if (const auto mode = record_permissions(); mode && ::fchmod(file.get(), *mode) != 0) return;
    if (!file.close()) return;
    // No fsync: this is a recoverable cache, not durable user data. Rename
    // publishes a complete record; crash/torn-write damage becomes a miss.
    if (::rename(temporary.path.c_str(), path.c_str()) == 0) temporary.path.clear();
  } catch (...) {
    // Disk caching must never prevent compilation or execution.
  }
}

void prune_disk_cache(const std::string& root, const Digest& current_environment) noexcept
{
  if (root.empty()) return;
  try {
    namespace fs = std::filesystem;
    std::error_code error;
    fs::create_directories(root, error);
    if (error) return;
    const auto base = fs::canonical(root, error);
    if (error || !is_directory_no_follow(base)) return;
    const auto spaces  = base / kCacheFormatVersion;
    const auto current = spaces / hex_digest(current_environment);
    // The configured root is trusted; do not follow symlinks below it.
    for (const auto& directory : {spaces, current}) {
      fs::create_directories(directory, error);
      if (error || !is_directory_no_follow(directory)) return;
    }
    const auto now = fs::file_time_type::clock::now();
    fs::last_write_time(current, now, error);
    if (error) return;
    clear_legacy_files(base);

    std::vector<std::pair<fs::file_time_type, fs::path>> others;
    if (!visit_directory(spaces, [&](const auto& entry) {
          if (entry.path() == current || !is_environment(entry)) return;
          const auto modified = entry.last_write_time(error);
          if (!error) others.emplace_back(modified, entry.path());
        }))
      return;
    std::ranges::sort(others, std::greater<>{});
    const auto cutoff = now - environment_grace;
    for (std::size_t index = kept_environments - 1; index < others.size(); ++index) {
      const auto& [modified, path] = others[index];
      if (modified >= cutoff || !is_directory_no_follow(path)) continue;
      // Another process may have started using this environment during the
      // scan. Recheck its timestamp, without claiming a lease or a strict cap.
      const auto latest = fs::last_write_time(path, error);
      if (!error && latest < cutoff) clear_environment(path);
    }
  } catch (...) {
    // Retention must never prevent cache lookup, compilation or publication.
  }
}

void clear_disk_cache(const std::string& root) noexcept
{
  if (root.empty()) return;
  try {
    std::error_code error;
    const auto base = std::filesystem::canonical(root, error);
    if (error || !is_directory_no_follow(base)) return;
    clear_legacy_files(base);
    const auto spaces = base / kCacheFormatVersion;
    visit_directory(spaces, [&](const auto& entry) {
      if (is_environment(entry)) clear_environment(entry.path());
    });
    if (is_directory_no_follow(spaces)) remove_entry(spaces);
  } catch (...) {
    // An unavailable root or individual entry is not fatal.
  }
}

}  // namespace codegen::jit::detail
