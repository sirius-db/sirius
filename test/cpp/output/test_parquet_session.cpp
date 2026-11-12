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
TEST_CASE("Parquet export is private until atomic publication", "[parquet_export][cpu]")
{
  temporary_directory root;
  auto target = root.path / "result";
  parquet_session session({.directory = target});
  session.write({1}, 42, {3, 4}, writer);
  session.write({2}, 2, {5}, writer);
  REQUIRE_FALSE(std::filesystem::exists(target));
  auto result = session.publish(writer);
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
  REQUIRE_THROWS(session.write({1}, 7, {2, 3}, [](auto i, auto const& path) {
    writer(i, path);
    if (i == 1) throw std::runtime_error("encoder failed");
  }));
  session.write({1}, 7, {2, 3}, writer);
  session.write({1}, 7, {2, 3}, writer);
  REQUIRE_THROWS(session.write({1}, 7, {2}, writer));
  auto result = session.publish(writer);
  REQUIRE(result.rows == 5);
  REQUIRE(result.files == 2);
}
TEST_CASE("Parquet ordering mode ignores empty inputs and rejects mixed or duplicate positions",
          "[parquet_export][cpu]")
{
  temporary_directory root;
  parquet_session session({.directory = root.path / "result"});
  session.write({}, std::nullopt, {}, writer);
  session.write({1}, 9, {2}, writer);
  REQUIRE_THROWS(session.write({2}, std::nullopt, {2}, writer));
  REQUIRE_THROWS(session.write({2}, 9, {2}, writer));
  session.abort();
  REQUIRE(std::filesystem::is_empty(root.path));
}
TEST_CASE("Parquet publication never overwrites a destination created during execution",
          "[parquet_export][cpu]")
{
  temporary_directory root;
  auto target = root.path / "result";
  parquet_session session({.directory = target});
  session.write({1}, std::nullopt, {2}, writer);
  std::filesystem::create_directory(target);
  REQUIRE_THROWS(session.publish(writer));
  session.abort();
  REQUIRE(std::filesystem::is_empty(target));
}
TEST_CASE("Concurrent unordered Parquet producers get unique parts", "[parquet_export][cpu]")
{
  temporary_directory root;
  parquet_session session({.directory = root.path / "result"});
  std::vector<std::thread> producers;
  for (std::uint64_t i = 0; i < 32; ++i)
    producers.emplace_back([&, i] { session.write({i}, std::nullopt, {1, 1}, writer); });
  for (auto& t : producers)
    t.join();
  auto result = session.publish(writer);
  REQUIRE(result.rows == 64);
  REQUIRE(result.files == 64);
}
TEST_CASE("Empty Parquet export emits a schema-only part", "[parquet_export][cpu]")
{
  temporary_directory root;
  parquet_session session({.directory = root.path / "result"});
  auto result = session.publish(writer);
  REQUIRE(result.rows == 0);
  REQUIRE(result.files == 1);
}

TEST_CASE("Parquet completion preparation fails before publication", "[parquet_export][cpu]")
{
  temporary_directory root;
  auto target = root.path / "result";
  parquet_session session({.directory = target});
  session.write({1}, std::nullopt, {1}, writer);
  REQUIRE_THROWS(session.publish(writer, [](auto const&) { throw std::bad_alloc(); }));
  REQUIRE_FALSE(std::filesystem::exists(target));
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
