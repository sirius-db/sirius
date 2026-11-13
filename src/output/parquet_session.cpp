// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "output/parquet_session.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
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
  if (successful_ || aborted_) throw std::runtime_error("Parquet export is already finished");
  if (started_) return;
  if (std::filesystem::is_symlink(std::filesystem::symlink_status(spec_.directory)))
    throw std::runtime_error("Parquet destination must not be a symlink");
  // Like DuckDB, allow existing directory trees provided they contain no files.
  // create_directory is nonrecursive: the destination's parent must exist.
  created_directory_ = std::filesystem::create_directory(spec_.directory) || created_directory_;
  for (auto const& entry : std::filesystem::recursive_directory_iterator(spec_.directory)) {
    if (entry.is_symlink() || !entry.is_directory())
      throw std::runtime_error("Parquet output directory is not empty: " +
                               spec_.directory.string());
  }
  started_ = true;
}
void parquet_session::reserve_part_locked(std::filesystem::path const& path)
{
  // Reserve ownership before handing the filename to cuDF (which opens with
  // truncation). A racing export must not overwrite this export's files.
  auto [it, inserted] = owned_files_.insert(path);
  if (!inserted) throw std::runtime_error("Parquet part is still owned by a previous attempt");
  auto fd = ::open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0666);
  if (fd < 0) {
    auto error = errno;
    owned_files_.erase(it);
    throw std::system_error(error, std::generic_category(), "Reserving Parquet output file");
  }
  if (::close(fd) != 0)
    throw std::system_error(errno, std::generic_category(), "Closing Parquet file reservation");
}
void parquet_session::remove_part_locked(std::filesystem::path const& path) noexcept
{
  auto it = owned_files_.find(path);
  if (it == owned_files_.end()) return;
  if (::unlink(path.c_str()) == 0 || errno == ENOENT) owned_files_.erase(it);
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
        record->files.push_back(spec_.directory / ("part-" + name + ".parquet"));
      }
    } else if (record->rows != rows || record->position != position) {
      throw std::runtime_error("Parquet retry changed its input layout");
    }
    if (record->active) throw std::runtime_error("Concurrent duplicate Parquet input");
    if (record->committed) return;
    record->active = true;
  }
  try {
    for (std::size_t i = 0; i < rows.size(); ++i) {
      {
        std::lock_guard guard(mutex_);
        reserve_part_locked(record->files[i]);
      }
      writer(i, record->files[i]);
    }
  } catch (...) {
    std::lock_guard guard(mutex_);
    for (auto const& path : record->files)
      remove_part_locked(path);
    record->active = false;
    throw;
  }
  std::lock_guard guard(mutex_);
  record->active    = false;
  record->committed = true;
}
parquet_export_result parquet_session::finish(
  write_part const& write_empty,
  std::function<void(parquet_export_result const&)> const& prepare_completion)
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
    auto path = spec_.directory / "part-00000000000000000000.parquet";
    reserve_part_locked(path);
    write_empty(0, path);
    result.files = 1;
    result.bytes = std::filesystem::file_size(path);
  }
  if (prepare_completion) prepare_completion(result);
  // Files are already visible. Only mark success after completion is prepared,
  // so a completion-allocation failure still cleans up this export's output.
  successful_ = true;
  return result;
}
void parquet_session::abort() noexcept
{
  if (successful_) return;
  aborted_ = true;
  for (auto it = owned_files_.begin(); it != owned_files_.end();) {
    // Advance before removal; failed cleanup keeps ownership for another attempt.
    auto const& path = *it++;
    remove_part_locked(path);
  }
  // Never recursively remove a directory: concurrent or preexisting contents
  // belong to someone else. Preserve directories that predate this export.
  if (created_directory_ && (::rmdir(spec_.directory.c_str()) == 0 || errno == ENOENT))
    created_directory_ = false;
}
}  // namespace sirius::output
