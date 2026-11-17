// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0

#include "catch.hpp"
#include "config/option.hpp"

#include <chrono>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace sirius::config;
using namespace sirius::config::size_literals;
using namespace std::chrono_literals;
using std::chrono::milliseconds;

namespace {

/// A mapping of scalar entries.
node map_of(std::initializer_list<std::pair<std::string, std::string>> entries)
{
  node block = node::mapping();
  for (auto const& [key, text] : entries) {
    block.set(key, node::scalar(text));
  }
  return block;
}

/// The two options of the original sketch.
struct memory_options {
  config_option<bytes, double> capacity = option<bytes>("capacity")
                                            .validate([](bytes v) { return v >= 0_B; })
                                            .or_type<double>("capacity_fraction")
                                            .validate(between<double>{0.0, 1.0})
                                            .fallback(0.95)
                                            .as_optional()
                                            .for_prod()
                                            .build();

  config_option<milliseconds> timeout = option<milliseconds>("timeout")
                                          .validate([](milliseconds v) { return v > 0ms; })
                                          .fallback(1500ms)
                                          .as_optional()
                                          .scope(option_scope::query)
                                          .build();

  void load(const node& block, environment env = environment::prod)
  {
    capacity.load(block, "memory", env);
    timeout.load(block, "memory", env);
  }
};

std::string load_error(memory_options& options, const node& block)
{
  try {
    options.load(block);
  } catch (const config_error& e) {
    return e.what();
  }
  FAIL("expected a config_error");
  return {};
}

}  // namespace

TEST_CASE("an optional option without a fallback stays unset", "[config][config_option]")
{
  auto limit = option<int>("limit").as_optional().build();
  REQUIRE(limit.is_unset());
  limit.load(node::mapping(), "", environment::prod);
  REQUIRE(limit.is_unset());
  REQUIRE_FALSE(limit.has_value());
}

TEST_CASE("value_source tracks file vs fallback", "[config][config_option]")
{
  auto limit = option<int>("limit").fallback(42).as_optional().build();
  REQUIRE(limit.is_unset());  // declaring a fallback does not load the option

  limit.load(node::mapping(), "", environment::prod);
  REQUIRE(limit.is_fallback());
  REQUIRE(limit.get() == 42);

  limit.load(map_of({{"limit", "42"}}), "", environment::prod);
  REQUIRE(limit.is_from_file());  // same value as the fallback, different source

  limit.load(node::mapping(), "", environment::prod);
  REQUIRE(limit.source() == value_source::fallback);

  // A present null counts as absent.
  node null_value = node::mapping();
  null_value.set("limit", node{});
  limit.load(null_value, "", environment::prod);
  REQUIRE(limit.is_fallback());

  REQUIRE_THROWS_AS(limit.load(map_of({{"limit", "oops"}}), "", environment::prod), config_error);
  REQUIRE(limit.is_unset());  // a failed load leaves the option unset
  REQUIRE_FALSE(limit.has_value());
}

TEST_CASE("for_prod options are skipped outside prod", "[config][config_option]")
{
  memory_options options;
  options.load(node::mapping(), environment::prod);
  REQUIRE(options.capacity.is_fallback());
  REQUIRE(options.capacity.get<double>() == 0.95);

  options.load(node::mapping(), environment::dev);
  REQUIRE(options.capacity.is_unset());
  REQUIRE_FALSE(options.capacity.has_value());
  REQUIRE(options.timeout.is_fallback());  // not for_prod
}

TEST_CASE("alternatives hold the type of the key that was present", "[config][config_option]")
{
  memory_options options;
  options.load(map_of({{"capacity", "1 GiB"}}));
  REQUIRE(options.capacity.is_from_file());
  REQUIRE(options.capacity.get<bytes>() == 1_GiB);
  REQUIRE(options.capacity.get_if<double>() == nullptr);
  REQUIRE_THROWS_AS(options.capacity.get<double>(), std::logic_error);
  REQUIRE(options.timeout.is_fallback());

  options.load(map_of({{"capacity_fraction", "0.8"}, {"timeout", "250 ms"}}));
  REQUIRE(options.capacity.get<double>() == 0.8);
  REQUIRE(options.timeout.get() == 250ms);
  REQUIRE(options.capacity.keys() ==
          std::vector<std::string_view>{"capacity", "capacity_fraction"});
}

TEST_CASE("a fallback does not satisfy a mandatory option", "[config][config_option]")
{
  auto limit = option<int>("limit").fallback(42).as_mandatory().build();
  REQUIRE_THROWS_WITH(limit.load(node::mapping(), "sirius.limits", environment::prod),
                      "'sirius.limits.limit': required but missing");
  REQUIRE(limit.is_unset());
  limit.load(map_of({{"limit", "7"}}), "sirius.limits", environment::prod);
  REQUIRE(limit.get() == 7);
  REQUIRE(limit.is_from_file());
}

TEST_CASE("present but invalid values never fall back", "[config][config_option]")
{
  memory_options options;
  REQUIRE(load_error(options, map_of({{"capacity", "1 B"}, {"capacity_fraction", "0.8"}})) ==
          "'memory': 'capacity' and 'capacity_fraction' are mutually exclusive");
  REQUIRE(load_error(options, map_of({{"capacity_fraction", "1.5"}})) ==
          "'memory.capacity_fraction': value out of range: '1.5' (must be between 0.0 and 1.0)");
  REQUIRE(load_error(options, map_of({{"timeout", "oops"}})) ==
          "'memory.timeout': invalid time value: 'oops'");
  REQUIRE(load_error(options, map_of({{"timeout", "0 ms"}})) ==
          "'memory.timeout': value out of range: '0 ms'");
  REQUIRE(load_error(options, map_of({{"timeout", "0.5 ms"}})) ==
          "'memory.timeout': '0.5 ms' is not a whole number of ms (it would need rounding)");
  REQUIRE(load_error(options, map_of({{"capacity", "999999999999999999 GiB"}})) ==
          "'memory.capacity': byte value out of range: '999999999999999999 GiB'");

  node nested = node::mapping();
  nested.set("timeout", node::mapping());
  REQUIRE(load_error(options, nested) == "'memory.timeout': expected a scalar, got a mapping");
}

TEST_CASE("scope and description are declared per option", "[config][config_option]")
{
  memory_options options;
  REQUIRE(options.capacity.scope == option_scope::session);
  REQUIRE(options.timeout.scope == option_scope::query);
  REQUIRE(to_string(option_scope::query) == "query");

  auto described = option<bool>("enable").describe("Turn it on.").build();
  REQUIRE(described.description == "Turn it on.");
}

TEST_CASE("or_type keeps the option-wide policy declared before it", "[config][config_option]")
{
  auto limit = option<bytes>("limit_bytes")
                 .fallback(1_MiB)
                 .scope(option_scope::query)
                 .or_type<double>("limit_fraction")
                 .build();
  REQUIRE(limit.scope == option_scope::query);
  limit.load(node::mapping(), "", environment::prod);
  REQUIRE(limit.get<bytes>() == 1_MiB);
}

TEST_CASE("an invalid fallback is a declaration error", "[config][config_option]")
{
  auto builder = option<double>("fraction").validate(between<double>{0.0, 1.0}).fallback(2.0);
  REQUIRE_THROWS_AS(builder.build(), std::logic_error);
}

TEST_CASE("validators may word their own rejection", "[config][config_option]")
{
  auto threads = option<int>("num_threads")
                   .validate([](int v) {
                     return v > 0 ? parse_status{} : parse_status::failure("must be positive");
                   })
                   .build();
  REQUIRE_THROWS_WITH(threads.load(map_of({{"num_threads", "0"}}), "pool", environment::prod),
                      "'pool.num_threads': must be positive");

  auto ratio =
    option<double>("ratio").validate([](double v) { return v > 0; }, "must be > 0").build();
  REQUIRE_THROWS_WITH(ratio.load(map_of({{"ratio", "-1"}}), "", environment::prod),
                      "'ratio': value out of range: '-1.0' (must be > 0)");
}

TEST_CASE("vector options read sequences", "[config][config_option]")
{
  auto affinity = option<std::vector<int>>("cpu_affinity").build();
  node block    = node::mapping();
  block.set("cpu_affinity",
            node::sequence({node::scalar("0"), node::scalar("2"), node::scalar("4")}));
  affinity.load(block, "pool", environment::prod);
  REQUIRE(affinity.get() == std::vector<int>{0, 2, 4});

  block.set("cpu_affinity", node::sequence({node::scalar("0"), node::scalar("x")}));
  REQUIRE_THROWS_WITH(affinity.load(block, "pool", environment::prod),
                      "'pool.cpu_affinity': element 1: invalid integer value: 'x'");
  REQUIRE_THROWS_WITH(affinity.load(map_of({{"cpu_affinity", "3"}}), "pool", environment::prod),
                      "'pool.cpu_affinity': expected a sequence, got a scalar");
}

TEST_CASE("custom parsers and writers replace the defaults", "[config][config_option]")
{
  // A closed set of spellings for a type without converters.
  auto mode = option<int>("mode")
                .parser([](std::string_view text, int& out) {
                  if (text == "epoll") {
                    out = 0;
                  } else if (text == "uring") {
                    out = 1;
                  } else {
                    return false;
                  }
                  return true;
                })
                .writer([](int v) { return v == 0 ? "epoll" : "uring"; })
                .build();
  mode.load(map_of({{"mode", "uring"}}), "rest", environment::prod);
  REQUIRE(mode.get() == 1);
  REQUIRE_THROWS_WITH(mode.load(map_of({{"mode", "poll"}}), "rest", environment::prod),
                      "'rest.mode': cannot parse 'poll'");

  mode.assign(0);
  node out = node::mapping();
  mode.write(out, false);
  REQUIRE(out == map_of({{"mode", "epoll"}}));
}

TEST_CASE("write emits file values, and fallbacks only on request", "[config][config_option]")
{
  memory_options options;
  options.load(map_of({{"capacity_fraction", "0.5"}}));

  node file_only = node::mapping();
  options.capacity.write(file_only, false);
  options.timeout.write(file_only, false);
  REQUIRE(file_only == map_of({{"capacity_fraction", "0.5"}}));

  node everything = node::mapping();
  options.capacity.write(everything, true);
  options.timeout.write(everything, true);
  REQUIRE(everything == map_of({{"capacity_fraction", "0.5"}, {"timeout", "1500 ms"}}));

  // What was written reads back to the same values and sources.
  memory_options reread;
  reread.load(everything);
  REQUIRE(reread.capacity.get<double>() == 0.5);
  REQUIRE(reread.timeout.get() == 1500ms);
  REQUIRE(reread.timeout.is_from_file());
}

TEST_CASE("assign sets a value as if read from the file", "[config][config_option]")
{
  memory_options options;
  options.capacity.assign(2_GiB);
  REQUIRE(options.capacity.is_from_file());
  REQUIRE(options.capacity.get<bytes>() == 2_GiB);
  options.capacity.reset();
  REQUIRE(options.capacity.is_unset());
}
