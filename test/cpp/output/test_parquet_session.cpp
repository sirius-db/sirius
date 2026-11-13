// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "catch.hpp"
#include "output/parquet_session.hpp"

#include <unistd.h>

#include <fstream>
#include <thread>

namespace {
struct temporary_directory {
  std::filesystem::path path;
  temporary_directory()
  {
    std::string pattern =
      (std::filesystem::temp_directory_path() / "sirius-export-test-XXXXXX").string();
    auto p = ::mkdtemp(pattern.data());
    if (!p) throw std::runtime_error("mkdtemp failed");
    path = p;
  }
  ~temporary_directory() { std::filesystem::remove_all(path); }
};
auto writer = [](std::size_t i, std::filesystem::path const& path) {
  std::ofstream out(path);
  out << i;
};
}  // namespace
using namespace sirius::output;
TEST_CASE("Parquet parts are visible before export completion", "[parquet_export][cpu]")
{
  temporary_directory root;
  auto target = root.path / "result";
  parquet_session session({.directory = target});
  session.write(1, 42, {3, 4}, writer);
  session.write(2, 2, {5}, writer);
  REQUIRE(std::filesystem::is_directory(target));
  REQUIRE(std::distance(std::filesystem::directory_iterator(target),
                        std::filesystem::directory_iterator{}) == 3);
  auto result = session.finish(writer);
  REQUIRE(result.rows == 12);
  REQUIRE(result.files == 3);
  REQUIRE(
    std::filesystem::exists(target / "part-00000000000000000002-00000000000000000000.parquet"));
  REQUIRE(
    std::filesystem::exists(target / "part-00000000000000000042-00000000000000000001.parquet"));
}
TEST_CASE("Parquet retries neither retain partial parts nor duplicate committed rows",
          "[parquet_export][cpu]")
{
  temporary_directory root;
  parquet_session session({.directory = root.path / "result"});
  REQUIRE_THROWS(session.write(1, 7, {2, 3}, [](auto i, auto const& path) {
    writer(i, path);
    if (i == 1) throw std::runtime_error("encoder failed");
  }));
  REQUIRE(std::filesystem::is_empty(root.path / "result"));
  session.write(1, 7, {2, 3}, writer);
  session.write(1, 7, {2, 3}, writer);
  REQUIRE_THROWS(session.write(1, 7, {2}, writer));
  auto result = session.finish(writer);
  REQUIRE(result.rows == 5);
  REQUIRE(result.files == 2);
}
TEST_CASE("Parquet ordering mode ignores empty inputs and rejects mixed or duplicate positions",
          "[parquet_export][cpu]")
{
  temporary_directory root;
  parquet_session session({.directory = root.path / "result"});
  session.write({}, std::nullopt, {}, writer);
  session.write(1, 9, {2}, writer);
  REQUIRE_THROWS(session.write(2, std::nullopt, {2}, writer));
  REQUIRE_THROWS(session.write(2, 9, {2}, writer));
  session.abort();
  REQUIRE(std::filesystem::is_empty(root.path));
}
TEST_CASE("Parquet cleanup preserves files created by another writer during execution",
          "[parquet_export][cpu]")
{
  temporary_directory root;
  auto target = root.path / "result";
  parquet_session session({.directory = target});
  session.write(1, std::nullopt, {2}, writer);
  auto foreign = target / "another-writer.txt";
  std::ofstream(foreign) << "preserve";
  session.abort();
  REQUIRE(std::filesystem::exists(foreign));
  REQUIRE(std::distance(std::filesystem::directory_iterator(target),
                        std::filesystem::directory_iterator{}) == 1);
}
TEST_CASE("Concurrent unordered Parquet producers get unique parts", "[parquet_export][cpu]")
{
  temporary_directory root;
  parquet_session session({.directory = root.path / "result"});
  std::vector<std::thread> producers;
  for (std::uint64_t i = 0; i < 32; ++i)
    producers.emplace_back([&, i] { session.write(i, std::nullopt, {1, 1}, writer); });
  for (auto& t : producers)
    t.join();
  auto result = session.finish(writer);
  REQUIRE(result.rows == 64);
  REQUIRE(result.files == 64);
}
TEST_CASE("Empty Parquet export emits a schema-only part", "[parquet_export][cpu]")
{
  temporary_directory root;
  parquet_session session({.directory = root.path / "result"});
  auto result = session.finish(writer);
  REQUIRE(result.rows == 0);
  REQUIRE(result.files == 1);
}

TEST_CASE("Failed Parquet completion preparation cleans up already visible parts",
          "[parquet_export][cpu]")
{
  temporary_directory root;
  auto target = root.path / "result";
  parquet_session session({.directory = target});
  session.write(1, std::nullopt, {1}, writer);
  REQUIRE_THROWS(session.finish(writer, [](auto const&) { throw std::bad_alloc(); }));
  REQUIRE_FALSE(std::filesystem::is_empty(target));
  session.abort();
  REQUIRE(std::filesystem::is_empty(root.path));
}

TEST_CASE("Parquet codec aliases are validated without GPU initialization", "[parquet_export][cpu]")
{
  REQUIRE(parse_compression("ZSTD") == parquet_compression::zstd);
  REQUIRE(parse_compression("none") == parquet_compression::uncompressed);
  REQUIRE(parse_compression("lz4_raw") == parquet_compression::lz4);
  REQUIRE_THROWS(parse_compression("not-a-codec"));
}

TEST_CASE("Parquet export accepts an existing directory tree without files",
          "[parquet_export][cpu]")
{
  temporary_directory root;
  auto target = root.path / "result";
  std::filesystem::create_directories(target / "empty-subdirectory");
  {
    parquet_session session({.directory = target});
    session.write(1, std::nullopt, {2}, writer);
    REQUIRE(session.finish(writer).rows == 2);
  }
  REQUIRE(std::filesystem::is_directory(target / "empty-subdirectory"));
  REQUIRE(std::filesystem::exists(target / "part-00000000000000000000.parquet"));
}

TEST_CASE("Parquet abort preserves preexisting empty directories", "[parquet_export][cpu]")
{
  temporary_directory root;
  auto target = root.path / "result";
  std::filesystem::create_directories(target / "empty-subdirectory");
  {
    parquet_session session({.directory = target});
    session.write(1, std::nullopt, {2}, writer);
    REQUIRE_THROWS(session.write(2, std::nullopt, {2}, [](auto i, auto const& path) {
      writer(i, path);
      throw std::runtime_error("encoder failed");
    }));
  }
  REQUIRE(std::filesystem::is_directory(target));
  REQUIRE(std::filesystem::is_empty(target / "empty-subdirectory"));
  REQUIRE(std::distance(std::filesystem::directory_iterator(target),
                        std::filesystem::directory_iterator{}) == 1);
}

TEST_CASE("Parquet rejects existing files including files in nested directories",
          "[parquet_export][cpu]")
{
  temporary_directory root;
  auto target = root.path / "result";
  std::filesystem::create_directories(target / "subdirectory");
  auto original = target / "subdirectory" / "original.txt";
  std::ofstream(original) << "preserve";
  {
    parquet_session session({.directory = target});
    REQUIRE_THROWS(session.write(1, std::nullopt, {2}, writer));
  }
  std::ifstream in(original);
  std::string value;
  in >> value;
  REQUIRE(value == "preserve");
}

TEST_CASE("A Parquet filename collision preserves the competing file and cleans only owned parts",
          "[parquet_export][cpu]")
{
  temporary_directory root;
  auto target  = root.path / "result";
  auto foreign = target / "part-00000000000000000001.parquet";
  parquet_session session({.directory = target});
  REQUIRE_THROWS(session.write(1, std::nullopt, {1, 1}, [&](auto i, auto const& path) {
    writer(i, path);
    if (i == 0) std::ofstream(foreign) << "competing-writer";
  }));
  session.abort();
  REQUIRE(std::filesystem::exists(foreign));
  REQUIRE_FALSE(std::filesystem::exists(target / "part-00000000000000000000.parquet"));
  std::ifstream in(foreign);
  std::string value;
  in >> value;
  REQUIRE(value == "competing-writer");
}

TEST_CASE("Empty Parquet writer failures clean up their reserved output file",
          "[parquet_export][cpu]")
{
  temporary_directory root;
  auto target = root.path / "result";
  std::filesystem::create_directory(target);
  {
    parquet_session session({.directory = target});
    REQUIRE_THROWS(session.finish([](auto i, auto const& path) {
      writer(i, path);
      throw std::runtime_error("schema writer failed");
    }));
  }
  REQUIRE(std::filesystem::is_empty(target));
}

TEST_CASE("Parquet rejects file and symlink destinations without modifying them",
          "[parquet_export][cpu]")
{
  temporary_directory root;
  auto target = root.path / "file";
  std::ofstream(target) << "preserve";
  auto link = root.path / "link";
  std::filesystem::create_directory_symlink(root.path, link);
  for (auto const& path : {target, link}) {
    parquet_session session({.directory = path});
    REQUIRE_THROWS(session.finish(writer));
  }
  REQUIRE(std::filesystem::file_size(target) == 8);
  REQUIRE(std::filesystem::is_symlink(link));
}

TEST_CASE("Parquet retry identity follows the original task including task zero",
          "[parquet_export][cpu]")
{
  temporary_directory root;
  parquet_session session({.directory = root.path / "result"});
  constexpr parquet_session::input_id original_task = 0;
  REQUIRE_THROWS(session.write(original_task, 12, {3, 4}, [](auto i, auto const& path) {
    writer(i, path);
    if (i == 1) throw std::runtime_error("writer failed after one part");
  }));
  REQUIRE(std::filesystem::is_empty(root.path / "result"));
  session.write(original_task, 12, {3, 4}, writer);
  session.write(original_task, 12, {3, 4}, [](auto, auto const&) {
    FAIL("committed logical task must not be written again");
  });
  auto result = session.finish(writer);
  REQUIRE(result.rows == 7);
  REQUIRE(result.files == 2);
}

TEST_CASE("Distinct logical tasks with identical layouts are separate Parquet deliveries",
          "[parquet_export][cpu]")
{
  temporary_directory root;
  parquet_session session({.directory = root.path / "result"});
  session.write(1, std::nullopt, {1}, writer);
  session.write(2, std::nullopt, {1}, writer);
  auto result = session.finish(writer);
  REQUIRE(result.rows == 2);
  REQUIRE(result.files == 2);
}

TEST_CASE("Parquet delivery IDs are scoped to each export session", "[parquet_export][cpu]")
{
  temporary_directory root;
  parquet_session first({.directory = root.path / "first"});
  parquet_session second({.directory = root.path / "second"});
  first.write(0, std::nullopt, {2}, writer);
  second.write(0, std::nullopt, {3}, writer);
  REQUIRE(first.finish(writer).rows == 2);
  REQUIRE(second.finish(writer).rows == 3);
}
