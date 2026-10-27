/*
 * Copyright 2026, Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "io/object_identity.hpp"
#include "io/rest/rest_reactor.hpp"

#include <catch2/catch_test_macros.hpp>
#include <fcntl.h>
#include <unistd.h>

using namespace sirius::io;

TEST_CASE("Object identity separates version kinds and unambiguous locators", "[object_identity]")
{
  object_identity a{identity_kind::object_tag, 100, "tag"};
  auto b = a;
  CHECK(a == b);
  b.version = "new";
  CHECK(a.cache_key("file") != b.cache_key("file"));
  b = a;
  b.size++;
  CHECK(a != b);
  b      = a;
  b.kind = identity_kind::local_stat;
  CHECK(a != b);
  CHECK(a.cache_key("file:1") != a.cache_key("file"));
  CHECK(next_open_generation() != next_open_generation());
}

TEST_CASE("Local identity separates replacement with preserved mtime", "[object_identity]")
{
  char name[]             = "/tmp/sirius-identity-XXXXXX";
  char replacement_name[] = "/tmp/sirius-identity-replacement-XXXXXX";
  int first_fd            = ::mkstemp(name);
  REQUIRE(first_fd >= 0);
  REQUIRE(::write(first_fd, "abc", 3) == 3);
  timespec times[2]{{123, 456}, {123, 456}};
  REQUIRE(::futimens(first_fd, times) == 0);
  object_identity first;
  REQUIRE(capture_local_identity(first_fd, first));
  ::close(first_fd);

  int replacement_fd = ::mkstemp(replacement_name);
  REQUIRE(replacement_fd >= 0);
  REQUIRE(::write(replacement_fd, "xyz", 3) == 3);
  REQUIRE(::futimens(replacement_fd, times) == 0);
  ::close(replacement_fd);
  REQUIRE(::rename(replacement_name, name) == 0);

  int fd = ::open(name, O_RDONLY);
  REQUIRE(fd >= 0);
  struct cleanup {
    int fd;
    char* name;
    char* replacement;
    ~cleanup()
    {
      ::close(fd);
      ::unlink(name);
      ::unlink(replacement);
    }
  } owner{fd, name, replacement_name};
  object_identity replaced;
  REQUIRE(capture_local_identity(fd, replaced));
  CHECK(replaced.size == first.size);
  CHECK(replaced != first);
  CHECK((replaced.inode != first.inode || replaced.ctime_ns != first.ctime_ns));
  REQUIRE(::unlink(name) == 0);
  object_identity held;
  REQUIRE(capture_local_identity(fd, held));
  // Removing a directory entry can change ctime as the link count drops. The open descriptor
  // still holds the same object, but its captured version need not match the pre-unlink one.
  CHECK(held.device == replaced.device);
  CHECK(held.inode == replaced.inode);
  CHECK(held.size == replaced.size);
  // io_uring reopens the owned descriptor for O_DIRECT, so pathname removal
  // cannot switch the data descriptor to another object.
  auto proc_path = "/proc/self/fd/" + std::to_string(fd);
  int reopened   = ::open(proc_path.c_str(), O_RDONLY);
  REQUIRE(reopened >= 0);
  object_identity reopened_identity;
  bool captured = capture_local_identity(reopened, reopened_identity);
  ::close(reopened);
  CHECK(captured);
  CHECK(reopened_identity == held);
  CHECK_FALSE(capture_local_identity(-1, held));
}

TEST_CASE("REST identity only reuses strong ETags", "[object_identity]")
{
  using sirius::io::identity_kind;
  using sirius::io::rest::rest_io_object;

  rest_io_object weak("s3://bucket/key", "bucket", "key", 3, "W/\"v1\"");
  CHECK(weak.identity().kind == identity_kind::opened_handle);
  CHECK(weak.validation_tag() == "W/\"v1\"");

  rest_io_object invalid("s3://bucket/key", "bucket", "key", 3, "v1");
  CHECK(invalid.identity().kind == identity_kind::opened_handle);

  rest_io_object strong("s3://bucket/key", "bucket", "key", 3, "\"v1\"");
  CHECK(strong.identity().kind == identity_kind::object_tag);
}
