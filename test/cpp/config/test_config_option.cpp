// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0

#include "catch.hpp"
#include "config/option.hpp"
#include "config/validators.hpp"

#include <chrono>
#include <cstdint>
#include <initializer_list>
#include <optional>
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
struct sample_memory {
  config_option<bytes, double> capacity = option<bytes>("capacity")
                                            .validate([](bytes v) { return v >= 0_B; })
                                            .or_type<double>("capacity_fraction")
                                            .validate(between<double>{0.0, 1.0})
                                            .fallback(0.95)
                                            .mark_optional()
                                            .build();

  config_option<milliseconds> timeout = option<milliseconds>("timeout")
                                          .validate([](milliseconds v) { return v > 0ms; })
                                          .fallback(1500ms)
                                          .mark_optional()
                                          .build();

  void load(const node& block)
  {
    capacity.load(block, "memory");
    timeout.load(block, "memory");
  }
};

std::string load_error(sample_memory& options, const node& block)
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
  auto limit = option<int>("limit").mark_optional().build();
  REQUIRE(limit.is_unset());
  limit.load(node::mapping(), "");
  REQUIRE(limit.is_unset());
  REQUIRE_FALSE(limit.has_value());
}

TEST_CASE("value_source tracks file vs fallback", "[config][config_option]")
{
  auto limit = option<int>("limit").fallback(42).mark_optional().build();
  REQUIRE(limit.is_unset());  // declaring a fallback does not load the option

  limit.load(node::mapping(), "");
  REQUIRE(limit.is_fallback());
  REQUIRE(limit.get() == 42);

  limit.load(map_of({{"limit", "42"}}), "");
  REQUIRE(limit.user_defined());  // same value as the fallback, different source

  limit.load(node::mapping(), "");
  REQUIRE(limit.source() == value_source::fallback);

  // A present null counts as absent.
  node null_value = node::mapping();
  null_value.set("limit", node{});
  limit.load(null_value, "");
  REQUIRE(limit.is_fallback());

  REQUIRE_THROWS_AS(limit.load(map_of({{"limit", "oops"}}), ""), config_error);
  REQUIRE(limit.is_unset());  // a failed load leaves the option unset
  REQUIRE_FALSE(limit.has_value());
}

TEST_CASE("alternatives hold the type of the key that was present", "[config][config_option]")
{
  sample_memory options;
  options.load(map_of({{"capacity", "1 GiB"}}));
  REQUIRE(options.capacity.user_defined());
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
  auto limit = option<int>("limit").fallback(42).mark_mandatory().build();
  REQUIRE_THROWS_WITH(limit.load(node::mapping(), "sirius.limits"),
                      "'sirius.limits.limit': required but missing");
  REQUIRE(limit.is_unset());
  limit.load(map_of({{"limit", "7"}}), "sirius.limits");
  REQUIRE(limit.get() == 7);
  REQUIRE(limit.user_defined());
}

TEST_CASE("present but invalid values never fall back", "[config][config_option]")
{
  sample_memory options;
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

TEST_CASE("every error about an option ends with its description", "[config][config_option]")
{
  auto limit = option<int>("limit")
                 .validate(positive<int>{})
                 .or_type<double>("limit_fraction")
                 .describe("the share of memory a query may use")
                 .build();
  REQUIRE(limit.description() == "the share of memory a query may use");
  REQUIRE_THROWS_WITH(
    limit.load(map_of({{"limit", "x"}}), "sirius.pool"),
    "'sirius.pool.limit': invalid integer value: 'x'; the share of memory a query "
    "may use");
  REQUIRE_THROWS_WITH(limit.load(map_of({{"limit", "1"}, {"limit_fraction", "0.5"}}), "p"),
                      "'p': 'limit' and 'limit_fraction' are mutually exclusive; the share of "
                      "memory a query may use");
  REQUIRE_THROWS_WITH(limit.assign(0),
                      Catch::Matchers::EndsWith("; the share of memory a query may use"));

  auto required = option<int>("n").mark_mandatory().describe("how many").build();
  REQUIRE_THROWS_WITH(required.load(node::mapping(), "p"), "'p.n': required but missing; how many");

  auto plain = option<int>("n").validate(positive<int>{}).build();
  REQUIRE_THROWS_WITH(plain.load(map_of({{"n", "0"}}), "p"),
                      Catch::Matchers::EndsWith("(must be greater than zero)"));
}

TEST_CASE("or_type keeps the option-wide policy declared before it", "[config][config_option]")
{
  auto limit = option<bytes>("limit_bytes")
                 .fallback(1_MiB)
                 .mark_mandatory()
                 .or_type<double>("limit_fraction")
                 .build();
  REQUIRE(limit.is_required());
  limit.load(map_of({{"limit_bytes", "1 MiB"}}), "");
  REQUIRE(limit.get<bytes>() == 1_MiB);
  limit.load(node{}, "");  // an absent block requires nothing
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
  REQUIRE_THROWS_WITH(threads.load(map_of({{"num_threads", "0"}}), "pool"),
                      "'pool.num_threads': must be positive");

  auto ratio =
    option<double>("ratio").validate([](double v) { return v > 0; }, "must be > 0").build();
  REQUIRE_THROWS_WITH(ratio.load(map_of({{"ratio", "-1"}}), ""),
                      "'ratio': value out of range: '-1.0' (must be > 0)");
}

TEST_CASE("vector options read sequences", "[config][config_option]")
{
  auto affinity = option<std::vector<int>>("cpu_affinity").build();
  node block    = node::mapping();
  block.set("cpu_affinity",
            node::sequence({node::scalar("0"), node::scalar("2"), node::scalar("4")}));
  affinity.load(block, "pool");
  REQUIRE(affinity.get() == std::vector<int>{0, 2, 4});

  block.set("cpu_affinity", node::sequence({node::scalar("0"), node::scalar("x")}));
  REQUIRE_THROWS_WITH(affinity.load(block, "pool"),
                      "'pool.cpu_affinity': element 1: invalid integer value: 'x'");
  REQUIRE_THROWS_WITH(affinity.load(map_of({{"cpu_affinity", "3"}}), "pool"),
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
  mode.load(map_of({{"mode", "uring"}}), "rest");
  REQUIRE(mode.get() == 1);
  REQUIRE_THROWS_WITH(mode.load(map_of({{"mode", "poll"}}), "rest"),
                      "'rest.mode': cannot parse 'poll'");

  mode.assign(0);
  node out = node::mapping();
  mode.write(out, false);
  REQUIRE(out == map_of({{"mode", "epoll"}}));
}

TEST_CASE("write emits file values, and fallbacks only on request", "[config][config_option]")
{
  sample_memory options;
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
  sample_memory reread;
  reread.load(everything);
  REQUIRE(reread.capacity.get<double>() == 0.5);
  REQUIRE(reread.timeout.get() == 1500ms);
  REQUIRE(reread.timeout.user_defined());
}

TEST_CASE("assign sets a value as if read from the file", "[config][config_option]")
{
  sample_memory options;
  options.capacity.assign(2_GiB);
  REQUIRE(options.capacity.user_defined());
  REQUIRE(options.capacity.get<bytes>() == 2_GiB);
  options.capacity.reset();
  REQUIRE(options.capacity.is_unset());
}

// ---- node ----------------------------------------------------------------------------------

TEST_CASE("node mappings keep order and replace in place", "[config][config_option]")
{
  node block;
  block.set("b", node::scalar("1"));  // a null node becomes a mapping
  block.set("a", node::scalar("2"));
  block.set("b", node::scalar("3"));
  REQUIRE(block.entries().size() == 2);
  REQUIRE(block.entries()[0].first == "b");
  REQUIRE(block.find("b")->text() == "3");
  REQUIRE(block != map_of({{"a", "2"}, {"b", "3"}}));  // order matters
  REQUIRE(block == map_of({{"b", "3"}, {"a", "2"}}));

  block.erase("b");
  block.erase("missing");
  REQUIRE(block == map_of({{"a", "2"}}));

  node copy = block;
  copy.set("a", node::scalar("changed"));
  REQUIRE(block.find("a")->text() == "2");  // copies are independent

  block.set("null", node{});
  REQUIRE(block.find("null") != nullptr);
  REQUIRE_FALSE(block.has_value("null"));
  REQUIRE_FALSE(block.has_value("missing"));
}

TEST_CASE("node accessors check the kind", "[config][config_option]")
{
  node scalar = node::scalar("x");
  REQUIRE(scalar.find("x") == nullptr);
  REQUIRE_FALSE(scalar.has_value("x"));
  REQUIRE_THROWS_WITH(scalar.entries(), "config::node: expected a mapping, got a scalar");
  REQUIRE_THROWS_AS(scalar.set("k", node{}), std::logic_error);
  REQUIRE_THROWS_WITH(node{}.text(), "config::node: expected a scalar, got null");
  REQUIRE(node::sequence().kind_name() == "a sequence");
}

// ---- loading -------------------------------------------------------------------------------

TEST_CASE("a null alternative does not conflict with a present one", "[config][config_option]")
{
  sample_memory options;
  node block = map_of({{"capacity_fraction", "0.5"}});
  block.set("capacity", node{});
  options.load(block);
  REQUIRE(options.capacity.get<double>() == 0.5);

  // Only nulls: absent, so a mandatory option is missing.
  auto pair  = option<int>("a").or_type<double>("b").mark_mandatory().build();
  node nulls = node::mapping();
  nulls.set("a", node{});
  nulls.set("b", node{});
  REQUIRE_THROWS_WITH(pair.load(nulls, "p"), "'p': one of 'a' or 'b' is required");
}

TEST_CASE("messages at the root have no path prefix", "[config][config_option]")
{
  auto pair = option<int>("a").or_type<double>("b").build();
  REQUIRE_THROWS_WITH(pair.load(map_of({{"a", "1"}, {"b", "2"}}), ""),
                      "'a' and 'b' are mutually exclusive");
  auto triple = option<int>("a").or_type<double>("b").or_type<bool>("c").mark_mandatory().build();
  REQUIRE_THROWS_WITH(triple.load(map_of({{"a", "1"}, {"b", "2"}, {"c", "y"}}), ""),
                      "'a', 'b' and 'c' are mutually exclusive");
  REQUIRE_THROWS_WITH(triple.load(node::mapping(), ""), "one of 'a', 'b' or 'c' is required");
  auto single = option<int>("a").mark_mandatory().build();
  REQUIRE_THROWS_WITH(single.load(node::mapping(), ""), "'a': required but missing");
}

TEST_CASE("every failed load leaves the option unset", "[config][config_option]")
{
  sample_memory options;
  auto failing = [&](const node& block) {
    options.load(map_of({{"capacity", "1 GiB"}}));  // populated first
    REQUIRE_THROWS_AS(options.capacity.load(block, "memory"), config_error);
    REQUIRE(options.capacity.is_unset());
    REQUIRE_FALSE(options.capacity.has_value());
  };
  failing(map_of({{"capacity", "1 GiB"}, {"capacity_fraction", "0.5"}}));  // exclusive
  failing(map_of({{"capacity_fraction", "2"}}));                           // validator
  failing(map_of({{"capacity", "x"}}));                                    // parser

  auto required = option<int>("limit").mark_mandatory().build();
  required.assign(3);
  REQUIRE_THROWS_AS(required.load(node::mapping(), ""), config_error);
  REQUIRE(required.is_unset());
}

TEST_CASE("exceptions from custom parsers and validators are framed", "[config][config_option]")
{
  auto throwing = option<int>("n")
                    .parser([](std::string_view text, int& out) -> parse_status {
                      out = std::stoi(std::string(text));  // throws std::invalid_argument
                      return {};
                    })
                    .build();
  throwing.assign(1);
  REQUIRE_THROWS_AS(throwing.load(map_of({{"n", "x"}}), "p"), config_error);
  REQUIRE(throwing.is_unset());

  auto node_misuse = option<int>("n")
                       .parser([](const node& n, int&) -> parse_status {
                         (void)n.entries();  // a scalar: throws std::logic_error
                         return {};
                       })
                       .build();
  REQUIRE_THROWS_WITH(node_misuse.load(map_of({{"n", "1"}}), "p"),
                      "'p.n': config::node: expected a mapping, got a scalar");
}

// ---- assign and fallback conversions --------------------------------------------------------

static_assert(detail::selects_one_of<mebibytes, bytes, double>);  // lossless class conversion
static_assert(detail::selects_one_of<float, bytes, double>);      // widening
static_assert(!detail::selects_one_of<int, bytes, double>);       // 1 is not 1.0 of a fraction
static_assert(!detail::selects_one_of<const char (&)[6], bool>);  // no pointer to bool
static_assert(!detail::selects_one_of<double, int>);              // no truncation
static_assert(!detail::selects_one_of<int, std::size_t>);         // no sign change
static_assert(detail::selects_one_of<int, std::int64_t>);         // widening, same sign
static_assert(!detail::selects_one_of<bool, int>);                // bool stays bool
static_assert(detail::selects_one_of<const char (&)[4], std::string>);  // string literal
static_assert(detail::selects_one_of<int, bool, int>);                  // exact match wins

TEST_CASE("assign converts losslessly and validates", "[config][config_option]")
{
  auto size = option<bytes>("size").validate([](bytes v) { return v > 0_B; }).build();
  size.assign(1_MiB);
  REQUIRE(size.get() == 1_MiB);

  size.assign(2_MiB);
  REQUIRE_THROWS_WITH(size.assign(0_B), "'size': value out of range: '0 B'");
  REQUIRE(size.get() == 2_MiB);  // unchanged after a rejected assign
  REQUIRE(size.user_defined());

  auto name = option<std::string>("name").fallback("telemetry_data").build();
  name.load(node::mapping(), "");
  REQUIRE(name.get() == "telemetry_data");
}

// ---- custom parsers and writers -------------------------------------------------------------

namespace {
struct point {
  int x = 0, y = 0;
  bool operator==(const point&) const = default;
};
}  // namespace

static_assert(!config_value<point>);

TEST_CASE("a type without parser<T> works with a custom parser and writer",
          "[config][config_option]")
{
  auto origin = option<point>("origin")
                  .parser([](const node& n, point& out) -> parse_status {
                    if (!n.is_sequence() || n.elements().size() != 2) {
                      return parse_status::failure("expected [x, y]");
                    }
                    point p;
                    if (!parser<int>{}(n.elements()[0].text(), p.x) ||
                        !parser<int>{}(n.elements()[1].text(), p.y)) {
                      return parse_status::failure("expected integers");
                    }
                    out = p;
                    return {};
                  })
                  .writer([](const point& p) {
                    return node::sequence(
                      {node::scalar(std::to_string(p.x)), node::scalar(std::to_string(p.y))});
                  })
                  .build();
  node block = node::mapping();
  block.set("origin", node::sequence({node::scalar("3"), node::scalar("4")}));
  origin.load(block, "");
  REQUIRE(origin.get() == point{3, 4});
  node out = node::mapping();
  origin.write(out, false);
  REQUIRE(out == block);

  // Without both, the declaration is rejected.
  REQUIRE_THROWS_AS(option<point>("origin").build(), std::logic_error);
}

TEST_CASE("text parsers may take std::string and return parse_status", "[config][config_option]")
{
  auto upper = option<std::string>("name")
                 .parser([](const std::string& text, std::string& out) -> parse_status {
                   if (text.empty()) { return parse_status::failure("must not be empty"); }
                   out = text;
                   return {};
                 })
                 .build();
  REQUIRE_THROWS_WITH(upper.load(map_of({{"name", ""}}), "t"), "'t.name': must not be empty");
}

TEST_CASE("validator descriptions", "[config][config_option]")
{
  // An explicit description wins over the callable's description().
  auto fraction = option<double>("f").validate(between<double>{0.0, 1.0}, "a fraction").build();
  REQUIRE_THROWS_WITH(fraction.load(map_of({{"f", "2"}}), ""),
                      "'f': value out of range: '2.0' (a fraction)");

  // A parse_status validator keeps its reason and gains the description.
  auto even =
    option<int>("n")
      .validate([](int v) { return v % 2 == 0 ? parse_status{} : parse_status::failure("odd"); },
                "must be even")
      .build();
  REQUIRE_THROWS_WITH(even.load(map_of({{"n", "3"}}), ""), "'n': odd (must be even)");

  // A vector value is not echoed.
  auto list = option<std::vector<int>>("ids")
                .validate([](const std::vector<int>& v) { return !v.empty(); }, "must not be empty")
                .build();
  node block = node::mapping();
  block.set("ids", node::sequence());
  REQUIRE_THROWS_WITH(list.load(block, ""), "'ids': value out of range (must not be empty)");
}

TEST_CASE("declaration mistakes are logic errors", "[config][config_option]")
{
  REQUIRE_THROWS_WITH(option<int>("k").or_type<double>("k").build(),
                      "config option 'k': repeated key");
  REQUIRE_THROWS_WITH(option<int>("").build(), "config option: empty key");
  REQUIRE_THROWS_WITH(
    option<int>("a").or_type<double>("b").validate(between<double>{0.0, 1.0}).fallback(2.0).build(),
    "config option 'b': invalid fallback: value out of range: '2.0' (must be between 0.0 and 1.0)");
}

TEST_CASE("or_type carries every option-wide setting", "[config][config_option]")
{
  auto pair = option<bool>("on")
                .fallback(true)
                .mark_mandatory()
                .describe("text")
                .or_type<int>("level")
                .build();
  REQUIRE(pair.is_required());
  REQUIRE(pair.description() == "text");
  REQUIRE(*pair.fallback() == decltype(pair)::value_type(true));
}

TEST_CASE("vector values", "[config][config_option]")
{
  auto names = option<std::vector<std::string>>("names").build();
  node block = node::mapping();
  block.set("names", node::sequence());
  names.load(block, "");
  REQUIRE(names.get().empty());

  block.set("names", node::sequence({node::scalar("a"), node::mapping()}));
  REQUIRE_THROWS_WITH(names.load(block, ""),
                      "'names': element 1: expected a scalar, got a mapping");

  names.assign(std::vector<std::string>{"x", "y"});
  node out = node::mapping();
  names.write(out, false);
  names.load(out, "");
  REQUIRE(names.get() == std::vector<std::string>{"x", "y"});
}

TEST_CASE("write replaces a sibling alternative and skips unset options", "[config][config_option]")
{
  sample_memory options;
  node block = map_of({{"capacity", "1 GiB"}});
  options.load(block);
  options.capacity.assign(0.5);
  options.capacity.write(block, false);
  REQUIRE(block == map_of({{"capacity_fraction", "0.5"}}));

  auto unset = option<int>("n").build();
  unset.load(node::mapping(), "");
  node out = map_of({{"n", "7"}});
  unset.write(out, true);
  REQUIRE(out == map_of({{"n", "7"}}));
}

TEST_CASE("accessors report what they hold", "[config][config_option]")
{
  auto triple = option<int>("a").or_type<double>("b").or_type<bool>("c").build();
  REQUIRE(triple.keys() == std::vector<std::string_view>{"a", "b", "c"});
  REQUIRE(triple.get_if<int>() == nullptr);
  REQUIRE_THROWS_WITH(triple.get<int>(), "config option 'a': has no value");
  triple.assign(true);
  REQUIRE_THROWS_WITH(triple.get<int>(), "config option 'a': holds another alternative");
  REQUIRE(to_string(value_source::fallback) == "fallback");
}

TEST_CASE("a mandatory option is not required when its block is absent", "[config][config_option]")
{
  auto cert = option<std::string>("cert").mark_mandatory().build();
  cert.load(node{}, "tls");
  REQUIRE(cert.is_unset());
  REQUIRE_THROWS_WITH(cert.load(node::mapping(), "tls"), "'tls.cert': required but missing");
}

TEST_CASE("choices read and write a closed set of spellings", "[config][config_option]")
{
  enum class mode { epoll, uring, poll };
  auto choice =
    option<mode>("wait_mode")
      .choices({{"epoll", mode::epoll}, {"uring", mode::uring}, {"io_uring", mode::uring}})
      .build();
  choice.load(map_of({{"wait_mode", "io_uring"}}), "rest");  // an alias reads
  REQUIRE(choice.get() == mode::uring);
  node out = node::mapping();
  choice.write(out, false);
  REQUIRE(out == map_of({{"wait_mode", "uring"}}));  // the first spelling writes

  REQUIRE_THROWS_WITH(
    choice.load(map_of({{"wait_mode", "Epoll"}}), "rest"),
    "'rest.wait_mode': invalid enum value 'Epoll' (expected epoll, uring, io_uring)");
  REQUIRE_THROWS_WITH(
    choice.load(map_of({{"wait_mode", " epoll"}}), "rest"),
    "'rest.wait_mode': invalid enum value ' epoll' (expected epoll, uring, io_uring)");
  node seq = node::mapping();
  seq.set("wait_mode", node::sequence());
  REQUIRE_THROWS_WITH(choice.load(seq, "rest"),
                      "'rest.wait_mode': expected a scalar, got a sequence");

  choice.assign(mode::poll);  // a value the table cannot spell
  node unspellable = node::mapping();
  REQUIRE_THROWS_AS(choice.write(unspellable, false), std::logic_error);

  auto bound = option<std::optional<int>>("max_version")
                 .choices({{"auto", std::nullopt}, {"12", 12}, {"13", 13}})
                 .fallback(std::optional<int>{})
                 .build();
  bound.load(map_of({{"max_version", "auto"}}), "");
  REQUIRE(bound.get() == std::nullopt);
  bound.load(map_of({{"max_version", "13"}}), "");
  REQUIRE(bound.get() == 13);
}
