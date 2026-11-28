// Copyright 2026 Sirius Contributors
// SPDX-License-Identifier: Apache-2.0
#include "object_store_scope/scope.hpp"
#include "test_helpers.hpp"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

using namespace sirius::test::object_store_test;

void expect_refusal(std::optional<core::refusal> const& actual,
                    core::refusal_kind kind,
                    std::string const& item)
{
  REQUIRE(actual.has_value());
  CHECK(actual->kind == kind);
  CHECK(actual->item == item);
  CHECK(actual->item.find("CORE_SECRET_SENTINEL") == std::string::npos);
  CHECK(actual->item.find("CORE_TOKEN_SENTINEL") == std::string::npos);
}

}  // namespace

TEST_CASE("Object-store scope resolves longest byte prefixes with scheme-only folding",
          "[object_store_scope]")
{
  auto const deep     = entry_for("s3://Bucket/Path/deep/");
  auto const parent   = entry_for("s3://Bucket/Path/");
  auto const bucket   = entry_for("s3://Bucket/");
  auto const fallback = entry_for("");
  core::scope_set const scopes{deep, parent, bucket, fallback};
  struct row {
    char const* uri;
    std::shared_ptr<const core::entry> expected;
  };
  std::array const rows{
    row{"S3://Bucket/Path/deep/file", deep},
    row{"s3://Bucket/Path/deeper/file", parent},
    row{"s3://Bucket/Path/file", parent},
    row{"s3://Bucket/path/file", bucket},
    row{"s3://bucket/Path/file", fallback},
    row{"http://elsewhere/file", nullptr},
    row{"https://elsewhere/file", nullptr},
    row{"gs://elsewhere/file", nullptr},
    row{"a-local-path", nullptr},
    row{"", nullptr},
  };
  for (auto const& test : rows) {
    INFO(test.uri);
    auto const result = core::resolve(scopes, test.uri);
    CHECK_FALSE(result.refused);
    if (!test.expected) {
      CHECK_FALSE(result.hit);
      continue;
    }
    REQUIRE(result.hit == test.expected);
    CHECK(result.hit->value == test.expected->value);
    CHECK(result.hit->owner.get() == test.expected->owner.get());
    CHECK(same_owner(result.hit->owner, test.expected->owner));
  }
  auto const miss = core::resolve(core::scope_set{deep}, "s3://Bucket/Path/other");
  CHECK_FALSE(miss.hit);
  CHECK_FALSE(miss.refused);
  auto const second = entry_for("s3://Bucket/Path/");
  CHECK(core::resolve(core::scope_set{parent, second}, "s3://Bucket/Path/file").hit == parent);
  CHECK(core::resolve(core::scope_set{second, parent}, "s3://Bucket/Path/file").hit == second);
}

TEST_CASE("Object-store scope gates raw keys before scopes but after an empty-set miss",
          "[object_store_scope]")
{
  core::scope_set const empty;
  core::scope_set const default_only{entry_for("")};
  core::scope_set const scoped{entry_for("s3://bucket/private/")};
  for (auto const uri : {"s3://bucket/",
                         "s3://bucket/private/file/",
                         "s3://bucket/private//file",
                         "s3://bucket/private/./file",
                         "s3://bucket/private/../file",
                         "s3://bucket/private/%2efile",
                         "s3://bucket/private/raw%file"}) {
    INFO(uri);
    auto const absent = core::resolve(empty, uri);
    CHECK_FALSE(absent.hit);
    CHECK_FALSE(absent.refused);
    for (auto const* set : {&default_only, &scoped}) {
      auto const refused = core::resolve(*set, uri);
      CHECK_FALSE(refused.hit);
      expect_refusal(refused.refused, core::refusal_kind::unmatchable_key, "s3://<redacted>");
    }
  }
  auto const redacted = core::resolve(
    default_only, "S3://CORE_SECRET_SENTINEL@bucket/a/../b?CORE_TOKEN_SENTINEL#fragment");
  CHECK_FALSE(redacted.hit);
  expect_refusal(redacted.refused, core::refusal_kind::unmatchable_key, "s3://<redacted>");
  auto const ambiguous =
    core::resolve(default_only, "s3://bucket/a/../b?CORE_TOKEN_SENTINEL@hidden");
  expect_refusal(ambiguous.refused, core::refusal_kind::unmatchable_key, "s3://<redacted>");
  auto const other = core::resolve(default_only, "https://other/a%2Fb?CORE_TOKEN_SENTINEL");
  CHECK_FALSE(other.hit);
  expect_refusal(other.refused, core::refusal_kind::unmatchable_key, "https://other/a%2Fb");
}

TEST_CASE("Object-store scope classifies a complete plan in item order", "[object_store_scope]")
{
  using core::refusal_kind;
  core::scope_set const empty;
  core::scope_set const scopes{entry_for("s3://bucket/allowed/")};
  std::vector<std::string> const no_items;
  CHECK_FALSE(core::classify_remote_reads(empty, no_items));
  CHECK_FALSE(core::classify_remote_reads(scopes, no_items));
  auto classify = [&](core::scope_set const& set, std::vector<std::string> const& items) {
    return core::classify_remote_reads(set, items);
  };
  expect_refusal(classify(empty, {"s3://bucket/a/../b?CORE_TOKEN_SENTINEL"}),
                 refusal_kind::unconfigured,
                 "s3://<redacted>");
  expect_refusal(
    classify(scopes, {"s3://bucket/allowed/first?CORE_TOKEN_SENTINEL", "s3://bucket/allowed/last"}),
    refusal_kind::route_unavailable,
    "s3://<redacted>");
  expect_refusal(classify(scopes,
                          {"s3://bucket/allowed/first",
                           "s3://elsewhere/miss?CORE_TOKEN_SENTINEL",
                           "s3://bucket/allowed/a/../b"}),
                 refusal_kind::out_of_scope,
                 "s3://<redacted>");
  expect_refusal(classify(scopes,
                          {"s3://bucket/allowed/first",
                           "s3://bucket/allowed/a/../b?CORE_TOKEN_SENTINEL",
                           "s3://elsewhere/miss"}),
                 refusal_kind::unmatchable_key,
                 "s3://<redacted>");
  expect_refusal(classify(scopes, {"s3://CORE_SECRET_SENTINEL@elsewhere/file?CORE_TOKEN_SENTINEL"}),
                 refusal_kind::out_of_scope,
                 "s3://<redacted>");
  expect_refusal(classify(empty, {"https://first/x?CORE_TOKEN_SENTINEL", "s3://b/file"}),
                 refusal_kind::unconfigured,
                 "https://first/x");
  expect_refusal(classify(empty, {"invalid scheme://CORE_SECRET_SENTINEL@host/key"}),
                 refusal_kind::unconfigured,
                 "<redacted-uri>");
}

TEST_CASE("Object-store scope refusal tokens are stable and nonthrowing", "[object_store_scope]")
{
  using core::refusal_kind;
  static_assert(noexcept(core::token_of(refusal_kind::unconfigured)));
  CHECK(core::token_of(refusal_kind::unconfigured) == "object_store_unconfigured");
  CHECK(core::token_of(refusal_kind::unmatchable_key) == "object_store_scope");
  CHECK(core::token_of(refusal_kind::out_of_scope) == "object_store_scope");
  CHECK(core::token_of(refusal_kind::route_unavailable) == "object_store_route_unavailable");
  CHECK(core::token_of(static_cast<refusal_kind>(255)).empty());
}

TEST_CASE("Object-store scope derives ordered sets without changing earlier sets",
          "[object_store_scope]")
{
  auto const root        = entry_for("");
  auto const before      = entry_for("s3://bucket/data/");
  auto const replacement = entry_for("s3://bucket/data/");
  auto const nested      = entry_for("s3://bucket/data/nested/");
  core::scope_set const original{before, root};
  auto added = core::with_entry(original, nested);
  REQUIRE(added);
  auto replaced = core::with_entry(*added, replacement);
  REQUIRE(replaced);
  auto removed = core::without_entry(*replaced, before->value->uri_prefix);
  REQUIRE(removed);
  auto absent = core::without_entry(*removed, before->value->uri_prefix);
  REQUIRE(absent);

  CHECK((original == core::scope_set{before, root}));
  CHECK((*added == core::scope_set{nested, before, root}));
  CHECK((*replaced == core::scope_set{nested, replacement, root}));
  CHECK((*removed == core::scope_set{nested, root}));
  CHECK(*absent == *removed);
}
