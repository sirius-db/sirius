#include "disk_cache.hpp"

#include "cache_identity.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <regex>
#include <string_view>
#include <utility>

namespace codegen::jit::detail {
namespace {

constexpr std::string_view magic    = "SIMPJIT3";
constexpr std::size_t length_offset = magic.size();
constexpr std::size_t digest_offset = length_offset + sizeof(uint64_t);
constexpr std::size_t header_size   = digest_offset + Digest{}.size();
// Corrupt lengths must not cause unbounded allocation. Larger kernels still
// compile and remain reusable in memory; only their persistence is skipped.
constexpr uint64_t max_cubin_bytes = 256ULL * 1024 * 1024;

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

bool read_all(int fd, char* bytes, std::size_t count)
{
  while (count) {
    const auto size = ::read(fd, bytes, count);
    if (size < 0 && errno == EINTR) continue;
    if (size <= 0) return false;
    bytes += size;
    count -= static_cast<std::size_t>(size);
  }
  return true;
}

bool write_all(int fd, const char* bytes, std::size_t count)
{
  while (count) {
    const auto size = ::write(fd, bytes, count);
    if (size < 0 && errno == EINTR) continue;
    if (size <= 0) return false;
    bytes += size;
    count -= static_cast<std::size_t>(size);
  }
  return true;
}

template <class Visit>
void visit_directory(const std::filesystem::path& path, Visit visit)
{
  std::error_code error;
  if (!std::filesystem::is_directory(std::filesystem::symlink_status(path, error))) return;
  std::filesystem::directory_iterator current(path, error), end;
  while (!error && current != end) {
    visit(*current);
    current.increment(error);
  }
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
    if (!read_all(file.get(), header.data(), header.size()) ||
        !std::equal(magic.begin(), magic.end(), header.begin()))
      return false;
    uint64_t size = 0;
    for (std::size_t i = length_offset; i < digest_offset; ++i)
      size = (size << 8) | static_cast<unsigned char>(header[i]);
    if (size != static_cast<uint64_t>(metadata.st_size) - header.size()) return false;
    std::vector<char> payload(static_cast<std::size_t>(size));
    if (!read_all(file.get(), payload.data(), payload.size()) || !file.close()) return false;
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
    std::copy(magic.begin(), magic.end(), header.begin());
    auto size = static_cast<uint64_t>(bytes.size());
    for (std::size_t i = digest_offset; i > length_offset; --i) {
      header[i - 1] = static_cast<char>(size & 255);
      size >>= 8;
    }
    const auto digest = content_identity({bytes.data(), bytes.size()});
    std::copy(digest.begin(), digest.end(), header.begin() + digest_offset);
    std::error_code error;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path(), error);
    if (error) return;
    Temporary temporary{path + ".tmp.XXXXXX"};
    Descriptor file(::mkostemp(temporary.path.data(), O_CLOEXEC));
    if (file.get() < 0) {
      temporary.path.clear();
      return;
    }
    if (!write_all(file.get(), header.data(), header.size()) ||
        !write_all(file.get(), bytes.data(), bytes.size()) || !file.close())
      return;
    // No fsync: this is a recoverable cache, not durable user data. Rename
    // publishes a complete record; crash/torn-write damage becomes a miss.
    if (::rename(temporary.path.c_str(), path.c_str()) == 0) temporary.path.clear();
  } catch (...) {
    // Disk caching must never prevent compilation or execution.
  }
}

void clear_disk_cache(const std::string& root) noexcept
{
  if (root.empty()) return;
  try {
    const std::regex legacy("[0-9a-f]{16}_a[0-9]+_c[0-9]+_d[0-9]+\\.cubin(\\.tmp\\.[0-9]+)?");
    const auto hex = "[0-9a-f]{" + std::to_string(Digest{}.size() * 2) + "}";
    const std::regex environment(hex);
    const std::regex request(hex + "\\.cubin(\\.tmp\\.[A-Za-z0-9]{6})?");
    visit_directory(root, [&](const auto& entry) {
      const auto name = entry.path().filename().string();
      if (is_regular(entry) && std::regex_match(name, legacy)) {
        remove_entry(entry.path());
      } else if (name == "v2") {
        visit_directory(entry.path(), [&](const auto& space) {
          if (!std::regex_match(space.path().filename().string(), environment)) return;
          // Check the directory separately: never unlink or traverse a symlink.
          std::error_code error;
          if (!std::filesystem::is_directory(space.symlink_status(error))) return;
          visit_directory(space.path(), [&](const auto& file) {
            if (is_regular(file) && std::regex_match(file.path().filename().string(), request))
              remove_entry(file.path());
          });
          remove_entry(space.path());
        });
        std::error_code error;
        if (std::filesystem::is_directory(entry.symlink_status(error))) remove_entry(entry.path());
      }
    });
  } catch (...) {
    // An unavailable root or individual entry is not fatal.
  }
}

}  // namespace codegen::jit::detail
