/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudf/io/text/byte_range_info.hpp>

#include <catch.hpp>
#include <io/rest/mock_authorizer.hpp>
#include <io/rest/rest_reactor.hpp>

#include <array>
#include <optional>
#include <span>
#include <vector>

using cudf::io::text::byte_range_info;
using sirius::io::grouped_coordinator;
using sirius::io::grouped_io_request;
using sirius::io::host_buffer;
using sirius::io::prepared_io_slice;
using sirius::io::range;
using sirius::io::rest::mock_authorizer;
using sirius::io::rest::rest_io_object;
using sirius::io::rest::rest_reactor;

namespace {

std::vector<byte_range_info> coalesce(std::vector<byte_range_info> ranges,
                                      std::optional<std::size_t> alignment = std::nullopt)
{
  return rest_reactor::align_and_coalesce(ranges, alignment);
}

std::unique_ptr<rest_reactor> make_reactor()
{
  auto authorizer = std::make_shared<mock_authorizer>(
    sirius::io::rest::authorized_request{"http://127.0.0.1/unused", {}});
  sirius::io::rest::config config;
  config.max_connections = 1;
  auto context =
    std::make_shared<rest_reactor::reactor_context>(config, std::move(authorizer), nullptr);
  return std::make_unique<rest_reactor>(std::move(context), "rest_lifecycle_test");
}

sirius::exec::semi_future<std::size_t> enqueue_one(rest_reactor& reactor, std::uint8_t* destination)
{
  auto object      = std::make_shared<rest_io_object>("s3://bucket/object", "bucket", "object", 1);
  auto coordinator = std::make_shared<grouped_coordinator>(1, 1);
  auto future      = coordinator->get_future();
  std::vector<prepared_io_slice> slices;
  slices.emplace_back(range{0, 1}, host_buffer{destination});
  reactor.enqueue(grouped_io_request::create(std::move(object), std::move(slices), coordinator));
  return future;
}

}  // namespace

TEST_CASE("rest_reactor::supports only accepts s3 URLs", "[rest]")
{
  CHECK(rest_reactor::supports("s3://bucket/key"));
  CHECK(rest_reactor::supports("s3://bucket/path/to/obj.parquet"));
  CHECK_FALSE(rest_reactor::supports("file:///tmp/x"));
  CHECK_FALSE(rest_reactor::supports("https://host/obj"));
  CHECK_FALSE(rest_reactor::supports("/local/abs/path"));
  CHECK_FALSE(rest_reactor::supports("not a uri"));
}

TEST_CASE("align_and_coalesce coalesces without alignment by default", "[rest]")
{
  SECTION("empty input") { CHECK(coalesce({}).empty()); }
  SECTION("zero-size ranges dropped") { CHECK(coalesce({byte_range_info{100, 0}}).empty()); }
  SECTION("disjoint ranges stay separate and sorted")
  {
    auto out = coalesce({byte_range_info{200, 50}, byte_range_info{0, 50}});
    REQUIRE(out.size() == 2);
    CHECK(out[0].offset() == 0);
    CHECK(out[0].size() == 50);
    CHECK(out[1].offset() == 200);
    CHECK(out[1].size() == 50);
  }
  SECTION("overlapping ranges merge")
  {
    auto out = coalesce({byte_range_info{0, 100}, byte_range_info{50, 100}});
    REQUIRE(out.size() == 1);
    CHECK(out[0].offset() == 0);
    CHECK(out[0].size() == 150);
  }
  SECTION("adjacent ranges merge")
  {
    auto out = coalesce({byte_range_info{0, 100}, byte_range_info{100, 100}});
    REQUIRE(out.size() == 1);
    CHECK(out[0].offset() == 0);
    CHECK(out[0].size() == 200);
  }
}

TEST_CASE("align_and_coalesce honors a caller alignment as a lower bound", "[rest]")
{
  auto out = coalesce({byte_range_info{100, 100}, byte_range_info{9000, 100}}, 4096);
  REQUIRE(out.size() == 2);
  CHECK(out[0].offset() == 0);
  CHECK(out[0].size() == 4096);
  CHECK(out[1].offset() == 8192);
  CHECK(out[1].size() == 4096);

  auto merged = coalesce({byte_range_info{100, 100}, byte_range_info{3000, 100}}, 4096);
  REQUIRE(merged.size() == 1);
  CHECK(merged[0].offset() == 0);
  CHECK(merged[0].size() == 4096);
}

TEST_CASE("rest_reactor rejects work outside its running lifetime", "[rest]")
{
  std::array<std::uint8_t, 1> destination{};

  SECTION("before start")
  {
    auto reactor = make_reactor();
    auto future  = enqueue_one(*reactor, destination.data());
    CHECK(future.is_ready());
    CHECK_THROWS(std::move(future).get());
    CHECK(reactor->queued_bytes() == 0);
  }

  SECTION("after shutdown")
  {
    auto reactor = make_reactor();
    reactor->start();
    reactor->shutdown();
    auto future = enqueue_one(*reactor, destination.data());
    CHECK(future.is_ready());
    CHECK_THROWS(std::move(future).get());
    CHECK(reactor->queued_bytes() == 0);
  }
}
