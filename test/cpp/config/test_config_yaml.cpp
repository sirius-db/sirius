// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0

#include "catch.hpp"
#include "config/block.hpp"
#include "config/yaml.hpp"

#include <chrono>
#include <string>
#include <vector>

using namespace sirius::config;
using namespace sirius::config::size_literals;
using namespace std::chrono_literals;

namespace {

struct pool_options : block_base<pool_options> {
  config_option<int> num_threads               = option<int>("num_threads").fallback(4).build();
  config_option<std::vector<int>> cpu_affinity = option<std::vector<int>>("cpu_affinity").build();
  config_option<std::string> name              = option<std::string>("name").build();
  auto fields() { return field_list(num_threads, cpu_affinity, name); }
};

struct disk_options : block_base<disk_options> {
  config_option<bytes, double> capacity =
    option<bytes>("capacity_bytes").or_type<double>("capacity_fraction").fallback(1_TiB).build();
  config_option<std::chrono::milliseconds> period =
    option<std::chrono::milliseconds>("monitor_period").fallback(100ms).build();
  auto fields() { return field_list(capacity, period); }
};

struct root_options : block_base<root_options> {
  config_option<int> max_concurrent_queries =
    option<int>("max_concurrent_queries").fallback(1).build();
  pool_options pipeline;
  disk_options disk;
  std::vector<disk_options> spaces;
  auto fields()
  {
    return field_list(
      max_concurrent_queries, sub("pipeline", pipeline), sub("disk", disk), list("space", spaces));
  }
};

std::string read_error(std::string_view text)
{
  root_options root;
  try {
    read_yaml(root, text, "sirius", environment::prod);
  } catch (const config_error& e) {
    return e.what();
  }
  FAIL("expected a config_error");
  return {};
}

}  // namespace

TEST_CASE("parse_yaml keeps scalar text and maps nulls", "[config][config_yaml]")
{
  auto const doc = parse_yaml(R"(
a: 1.5 GiB
b: "quoted text"
c:
d: ~
e: null
f: [1, 2]
g: {h: x}
)");
  REQUIRE(doc.find("a")->text() == "1.5 GiB");
  REQUIRE(doc.find("b")->text() == "quoted text");
  REQUIRE(doc.find("c")->is_null());
  REQUIRE(doc.find("d")->is_null());
  REQUIRE(doc.find("e")->is_null());
  REQUIRE(doc.find("f")->elements().size() == 2);
  REQUIRE(doc.find("g")->find("h")->text() == "x");
  REQUIRE(parse_yaml("").is_null());
}

TEST_CASE("parse_yaml rejects malformed text and duplicate keys", "[config][config_yaml]")
{
  REQUIRE_THROWS_AS(parse_yaml("a: [1, 2"), yaml_syntax_error);
  REQUIRE_THROWS_WITH(parse_yaml("sirius:\n  a: 1\n  a: 2\n"),
                      "duplicate config key: 'a' in sirius");
}

TEST_CASE("emit_yaml reads back to the same tree", "[config][config_yaml]")
{
  node tree = node::mapping();
  for (std::string text :
       {"",         "~",        "null",    "Null", "true", "123",    " lead",
        "trail ",   "a: b",     "- x",     "#c",   "[1]",  "{k: v}", "multi\nline",
        "quote\"s", "'single'", "1.5 GiB", "@at",  "%pct", "&anchor"}) {
    tree.set("k" + std::to_string(tree.entries().size()), node::scalar(text));
  }
  tree.set("null_value", node{});
  tree.set("empty_map", node::mapping());
  tree.set("empty_seq", node::sequence());
  tree.set("nested",
           node::sequence({node::scalar("1"), node::mapping({{"x", node::scalar("y")}})}));
  auto const text = emit_yaml(tree);
  INFO(text);
  REQUIRE(parse_yaml(text) == tree);
}

TEST_CASE("read_yaml fills a block tree from the root key", "[config][config_yaml]")
{
  root_options root;
  read_yaml(root,
            R"(
sirius:
  max_concurrent_queries: 2
  pipeline:
    num_threads: 8
    cpu_affinity: [0, 2]
  disk:
    capacity_fraction: 0.5
  space:
    - capacity_bytes: 4 GiB
    - monitor_period: 1s
)",
            "sirius",
            environment::prod);
  REQUIRE(root.max_concurrent_queries.get() == 2);
  REQUIRE(root.pipeline.cpu_affinity.get() == std::vector<int>{0, 2});
  REQUIRE(root.pipeline.name.is_unset());
  REQUIRE(root.disk.capacity.get<double>() == 0.5);
  REQUIRE(root.disk.period.is_fallback());
  REQUIRE(root.spaces.size() == 2);
  REQUIRE(root.spaces[0].capacity.get<bytes>() == 4_GiB);
  REQUIRE(root.spaces[1].period.get() == 1000ms);
}

TEST_CASE("read_yaml checks the document shape", "[config][config_yaml]")
{
  REQUIRE(read_error("other: 1\n") == "unknown config key: 'other'");
  REQUIRE(read_error("") == "expected a mapping at the top level, got null");
  REQUIRE(read_error("- 1\n") == "expected a mapping at the top level, got a sequence");
  REQUIRE(read_error("{}") == "missing top-level 'sirius' key");
  REQUIRE(read_error("sirius:\n  pipeline:\n    num_threads: x\n") ==
          "'sirius.pipeline.num_threads': invalid integer value: 'x'");
  REQUIRE(
    read_error("sirius:\n  space:\n    - capacity_bytes: 1 KB\n      capacity_fraction: 0.1\n") ==
    "'sirius.space[0]': 'capacity_bytes' and 'capacity_fraction' are mutually exclusive");

  // An empty root block is valid: every option takes its fallback.
  root_options root;
  read_yaml(root, "sirius:\n", "sirius", environment::prod);
  REQUIRE(root.max_concurrent_queries.is_fallback());
}

TEST_CASE("write_yaml round-trips file values and optionally fallbacks", "[config][config_yaml]")
{
  constexpr std::string_view text = R"(sirius:
  pipeline:
    num_threads: 8
    cpu_affinity:
      - 0
      - 2
    name: "~"
  disk:
    capacity_bytes: 1536 MiB
  space:
    - monitor_period: 2 min
)";
  root_options root;
  read_yaml(root, text, "sirius", environment::prod);
  REQUIRE(root.pipeline.name.get() == "~");

  auto const written = write_yaml(root, "sirius");
  INFO(written);
  REQUIRE(parse_yaml(written) == parse_yaml(text));

  root_options all;
  read_yaml(all, write_yaml(root, "sirius", true), "sirius", environment::prod);
  REQUIRE(all.max_concurrent_queries.user_defined());
  REQUIRE(all.disk.period.get() == 100ms);
  REQUIRE(all.spaces.at(0).capacity.get<bytes>() == 1_TiB);
  REQUIRE(all.spaces.at(0).period.get() == 120000ms);
}
