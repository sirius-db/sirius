// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "output/parquet_session.hpp"

#include <fcntl.h>
#include <linux/fs.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace sirius::output {
parquet_compression parse_compression(std::string name)
{
  std::transform(
    name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::tolower(c); });
  if (name == "uncompressed" || name == "none") return parquet_compression::uncompressed;
  if (name == "snappy") return parquet_compression::snappy;
  if (name == "gzip") return parquet_compression::gzip;
  if (name == "brotli") return parquet_compression::brotli;
  if (name == "zstd") return parquet_compression::zstd;
  if (name == "lz4" || name == "lz4_raw") return parquet_compression::lz4;
  throw std::invalid_argument("Unsupported Parquet compression: " + name);
}
std::string compression_name(parquet_compression codec)
{
  switch (codec) {
    case parquet_compression::uncompressed: return "uncompressed";
    case parquet_compression::snappy: return "snappy";
    case parquet_compression::gzip: return "gzip";
    case parquet_compression::brotli: return "brotli";
    case parquet_compression::zstd: return "zstd";
    case parquet_compression::lz4: return "lz4";
  }
  throw std::invalid_argument("Invalid Parquet codec");
}
namespace {
std::string index_name(std::uint64_t index)
{
  std::ostringstream out;
  out << std::setfill('0') << std::setw(20) << index;
  return out.str();
}
void remove_parts(std::vector<std::filesystem::path> const& files) noexcept
{
  for (auto const& file : files) {
    std::error_code ignored;
    std::filesystem::remove(file, ignored);
  }
}
}  // namespace
parquet_session::parquet_session(parquet_export_spec spec) : spec_(std::move(spec))
{
  if (spec_.directory.empty()) throw std::invalid_argument("Parquet destination is empty");
  spec_.directory = std::filesystem::absolute(spec_.directory).lexically_normal();
  while (spec_.directory != spec_.directory.root_path() && spec_.directory.filename().empty())
    spec_.directory = spec_.directory.parent_path();
}
parquet_session::~parquet_session() { abort(); }
void parquet_session::start_locked()
{
  if (published_) throw std::runtime_error("Parquet export is already published");
  if (!staging_.empty()) return;
  if (std::filesystem::exists(spec_.directory) || std::filesystem::is_symlink(spec_.directory))
    throw std::runtime_error("Parquet export requires a new directory: " +
                             spec_.directory.string());
  auto pattern = (spec_.directory.parent_path() / ".sirius-parquet-XXXXXX").string();
  if (!::mkdtemp(pattern.data()))
    throw std::system_error(errno, std::generic_category(), "Creating Parquet staging directory");
  staging_ = pattern;
}
void parquet_session::write(input_id const& id,
                            std::optional<std::size_t> position,
                            std::vector<std::uint64_t> const& rows,
                            write_part const& writer)
{
  if (rows.empty()) return;  // Empty input never fixes the ordering mode.
  input_record* record;
  {
    std::lock_guard guard(mutex_);
    start_locked();
    if (id.empty()) throw std::runtime_error("Parquet input requires stable retry identity");
    if (positioned_ && *positioned_ != position.has_value())
      throw std::runtime_error("Cannot mix positioned and unpositioned Parquet inputs");
    positioned_         = position.has_value();
    auto [it, inserted] = inputs_.try_emplace(id);
    record              = &it->second;
    if (inserted) {
      if (position && !positions_.insert(*position).second) {
        inputs_.erase(it);
        throw std::runtime_error("Duplicate Parquet input position");
      }
      record->position = position;
      record->rows     = rows;
      for (std::size_t i = 0; i < rows.size(); ++i) {
        auto name =
          position ? index_name(*position) + "-" + index_name(i) : index_name(next_.fetch_add(1));
        record->files.push_back(staging_ / ("part-" + name + ".parquet"));
      }
    } else if (record->rows != rows || record->position != position) {
      throw std::runtime_error("Parquet retry changed its input layout");
    }
    if (record->active) throw std::runtime_error("Concurrent duplicate Parquet input");
    if (record->committed) return;
    record->active = true;
  }
  try {
    for (std::size_t i = 0; i < rows.size(); ++i)
      writer(i, record->files[i]);
  } catch (...) {
    remove_parts(record->files);
    std::lock_guard guard(mutex_);
    record->active = false;
    throw;
  }
  std::lock_guard guard(mutex_);
  record->active    = false;
  record->committed = true;
}
parquet_export_result parquet_session::publish(
  write_part const& write_empty,
  std::function<void(parquet_export_result const&)> const& before_publish)
{
  std::lock_guard guard(mutex_);
  start_locked();
  parquet_export_result result;
  for (auto const& [id, record] : inputs_) {
    if (!record.committed || record.active)
      throw std::runtime_error("Incomplete Parquet export input");
    for (std::size_t i = 0; i < record.rows.size(); ++i) {
      if (record.rows[i] > std::uint64_t(std::numeric_limits<std::int64_t>::max()) - result.rows)
        throw std::overflow_error("Parquet row count exceeds BIGINT");
      result.rows += record.rows[i];
      ++result.files;
      result.bytes += std::filesystem::file_size(record.files[i]);
    }
  }
  if (result.files == 0) {
    auto path = staging_ / "part-00000000000000000000.parquet";
    write_empty(0, path);
    result.files = 1;
    result.bytes = std::filesystem::file_size(path);
  }
  if (before_publish) before_publish(result);
  // Linux local filesystems only. Unlike rename(), never replace a concurrently
  // created destination (including an empty directory or a symlink).
  if (::syscall(SYS_renameat2,
                AT_FDCWD,
                staging_.c_str(),
                AT_FDCWD,
                spec_.directory.c_str(),
                RENAME_NOREPLACE) != 0)
    throw std::system_error(
      errno, std::generic_category(), "Publishing Parquet directory without replacement");
  published_ = true;
  staging_.clear();
  return result;
}
void parquet_session::abort() noexcept
{
  std::error_code ignored;
  if (!staging_.empty()) std::filesystem::remove_all(staging_, ignored);
  staging_.clear();
}
}  // namespace sirius::output
