/*
 * Copyright 2026, Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "io/object_identity.hpp"

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

TEST_CASE("Local identity observes mtime through the owned fd", "[object_identity]")
{
  char name[] = "/tmp/sirius-identity-XXXXXX";
  int fd      = ::mkstemp(name);
  REQUIRE(fd >= 0);
  struct cleanup {
    int fd;
    char* name;
    ~cleanup()
    {
      ::close(fd);
      ::unlink(name);
    }
  } owner{fd, name};
  REQUIRE(::write(fd, "abc", 3) == 3);
  object_identity first;
  REQUIRE(capture_local_identity(fd, first));
  CHECK(first.size == 3);
  CHECK(first.kind == identity_kind::local_stat);
  timespec times[2]{{123, 456}, {123, 456}};
  REQUIRE(::futimens(fd, times) == 0);
  object_identity changed;
  REQUIRE(capture_local_identity(fd, changed));
  CHECK(changed.size == first.size);
  CHECK(changed != first);
  REQUIRE(::unlink(name) == 0);
  object_identity held;
  REQUIRE(capture_local_identity(fd, held));
  CHECK(held == changed);
  CHECK_FALSE(capture_local_identity(-1, held));
}
