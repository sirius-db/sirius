// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0

#include "catch.hpp"
#include "config/block.hpp"
#include "config/yaml.hpp"

#include <chrono>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

using namespace sirius::config;
using namespace sirius::config::size_literals;
using namespace std::chrono_literals;

namespace {

struct sample_pool : block_base<sample_pool> {
  config_option<int> num_threads               = option<int>("num_threads").fallback(4).build();
  config_option<std::vector<int>> cpu_affinity = option<std::vector<int>>("cpu_affinity").build();
  config_option<std::string> name              = option<std::string>("name").build();
  auto fields() { return field_list(num_threads, cpu_affinity, name); }
};

struct sample_disk : block_base<sample_disk> {
  config_option<bytes, double> capacity =
    option<bytes>("capacity_bytes").or_type<double>("capacity_fraction").fallback(1_TiB).build();
  config_option<std::chrono::milliseconds> period =
    option<std::chrono::milliseconds>("monitor_period").fallback(100ms).build();
  auto fields() { return field_list(capacity, period); }
};

struct sample_root : block_base<sample_root> {
  config_option<int> max_concurrent_queries =
    option<int>("max_concurrent_queries").fallback(1).build();
  sample_pool pipeline;
  sample_disk disk;
  std::vector<sample_disk> spaces;
  auto fields()
  {
    return field_list(
      max_concurrent_queries, sub("pipeline", pipeline), sub("disk", disk), list("space", spaces));
  }
};

std::string read_error(std::string_view text)
{
  sample_root root;
  try {
    read_yaml(root, text, "sirius");
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
                      "duplicate config key: 'a' in sirius (line 3, column 3)");
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
  sample_root root;
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
            "sirius");
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
  REQUIRE(read_error("") == "missing top-level 'sirius' key");
  REQUIRE(read_error("# only a comment\n") == "missing top-level 'sirius' key");
  REQUIRE(read_error("sirius:\n") == "missing top-level 'sirius' key");  // null root, as before
  REQUIRE(read_error("sirius: 5\n") == "'sirius': expected a mapping, got a scalar");
  REQUIRE(read_error("- 1\n") == "expected a mapping at the top level, got a sequence");
  REQUIRE(read_error("{}") == "missing top-level 'sirius' key");
  REQUIRE(read_error("sirius:\n  pipeline:\n    num_threads: x\n") ==
          "'sirius.pipeline.num_threads': invalid integer value: 'x'");
  REQUIRE(
    read_error("sirius:\n  space:\n    - capacity_bytes: 1 KB\n      capacity_fraction: 0.1\n") ==
    "'sirius.space[0]': 'capacity_bytes' and 'capacity_fraction' are mutually exclusive");

  // An empty root mapping is valid: every option takes its fallback.
  sample_root root;
  read_yaml(root, "sirius: {}\n", "sirius");
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
  sample_root root;
  read_yaml(root, text, "sirius");
  REQUIRE(root.pipeline.name.get() == "~");

  auto const written = write_yaml(root, "sirius");
  INFO(written);
  REQUIRE(parse_yaml(written) == parse_yaml(text));

  sample_root all;
  read_yaml(all, write_yaml(root, "sirius", true), "sirius");
  REQUIRE(all.max_concurrent_queries.user_defined());
  REQUIRE(all.disk.period.get() == 100ms);
  REQUIRE(all.spaces.at(0).capacity.get<bytes>() == 1_TiB);
  REQUIRE(all.spaces.at(0).period.get() == 120000ms);
}

TEST_CASE("aliases are bounded", "[config][config_yaml]")
{
  // yaml-cpp loads an alias inside the value it names as a cycle.
  for (std::string_view text : {"a: &x [*x]\n", "&x {a: *x}\n", "a: &x\n  b: *x\n"}) {
    INFO(text);
    REQUIRE_THROWS_AS(parse_yaml(text), yaml_syntax_error);
  }

  // Nested aliases expand exponentially; the value budget stops them early.
  std::string laughs = "a0: &a0 x\n";
  for (int level = 1; level <= 9; ++level) {
    auto const name = "a" + std::to_string(level);
    auto const prev = "*a" + std::to_string(level - 1);
    laughs += name + ": &" + name + " [";
    for (int i = 0; i < 9; ++i) {
      laughs += (i == 0 ? "" : ", ") + prev;
    }
    laughs += "]\n";
  }
  REQUIRE_THROWS_WITH(parse_yaml(laughs),
                      "YAML document has more than 100000 values once its aliases are expanded");

  auto const doc = parse_yaml("base: &b {x: 1}\ncopy: *b\n");
  REQUIRE(doc.find("copy")->find("x")->text() == "1");
}

TEST_CASE("one document only", "[config][config_yaml]")
{
  REQUIRE_THROWS_WITH(parse_yaml("a: 1\n---\na: 2\n"), "expected one YAML document, got 2");
  REQUIRE_THROWS_AS(parse_yaml("a: 1\n---\n[oops\n"), yaml_syntax_error);
  REQUIRE(parse_yaml("a: 1\n---\n").find("a")->text() == "1");  // trailing empty document
  REQUIRE(parse_yaml("   \n").is_null());
}

TEST_CASE("null-like text stays text when quoted", "[config][config_yaml]")
{
  auto const doc = parse_yaml("a: \"null\"\nb: '~'\nc: \"\"\nd: !!str ~\ne: NULL\nf: nUll\n");
  REQUIRE(doc.find("a")->text() == "null");
  REQUIRE(doc.find("b")->text() == "~");
  REQUIRE(doc.find("c")->text().empty());
  REQUIRE(doc.find("d")->text() == "~");
  REQUIRE(doc.find("e")->is_null());
  REQUIRE(doc.find("f")->text() == "nUll");
  // Merge keys are not merged.
  REQUIRE(parse_yaml("b: &b {x: 1}\nc:\n  <<: *b\n").find("c")->find("<<") != nullptr);
}

TEST_CASE("key errors carry their position", "[config][config_yaml]")
{
  REQUIRE_THROWS_WITH(parse_yaml("? [a, b]\n: v\n"),
                      "mapping keys must be non-null scalars (line 1, column 3)");
  REQUIRE_THROWS_WITH(parse_yaml("s:\n  - a: 1\n    a: 2\n"),
                      "duplicate config key: 'a' in s[0] (line 3, column 5)");

  // A duplicate key is a bad configuration, not a syntax error; both are config errors.
  try {
    (void)parse_yaml("a: 1\na: 2\n");
    FAIL("expected an error");
  } catch (const yaml_syntax_error&) {
    FAIL("a duplicate key is not a syntax error");
  } catch (const config_error&) {
  }
  REQUIRE_THROWS_AS(parse_yaml("a: [1"), config_error);
  REQUIRE_THROWS_WITH(parse_yaml(std::string(600, '[')), "YAML document nested too deeply");
}

TEST_CASE("emit_yaml round-trips keys and nested shapes", "[config][config_yaml]")
{
  node tree = node::mapping();
  for (auto const& key : std::vector<std::string>{"a: b",
                                                  "#c",
                                                  "- x",
                                                  "",
                                                  "multi\nline",
                                                  std::string(1100, 'k'),
                                                  "--- x",
                                                  "... y",
                                                  "tab\there",
                                                  std::string("nul\0byte", 8)}) {
    tree.set(key, node::scalar("v"));
  }
  tree.set("seq",
           node::sequence({node{},
                           node::mapping(),
                           node::sequence(),
                           node::sequence({node::scalar("1"), node::sequence({node::scalar("2")})}),
                           node::scalar(std::string(5000, 'x')),
                           node::scalar("\x01\x7f")}));
  auto const text = emit_yaml(tree);
  INFO(text);
  REQUIRE(parse_yaml(text) == tree);

  for (auto const& root : {node::scalar("---"), node::scalar("..."), node::scalar("plain")}) {
    REQUIRE(parse_yaml(emit_yaml(root)) == root);
  }
}

TEST_CASE("read_yaml leaves the block as it was on failure", "[config][config_yaml]")
{
  sample_root root;
  read_yaml(root, "sirius:\n  max_concurrent_queries: 3\n", "sirius");
  REQUIRE_THROWS_AS(read_yaml(root, "sirius: [oops", "sirius"), config_error);
  REQUIRE_THROWS_AS(read_yaml(root, "other: 1\n", "sirius"), config_error);
  REQUIRE_THROWS_AS(read_yaml(root, "sirius:\n  max_concurrent_queries: x\n", "sirius"),
                    config_error);
  REQUIRE(root.max_concurrent_queries.get() == 3);
}

TEST_CASE("write_yaml of a block nobody read", "[config][config_yaml]")
{
  sample_root root;
  REQUIRE(parse_yaml(write_yaml(root, "sirius")) == node::mapping({{"sirius", node::mapping()}}));
  sample_root reread;
  read_yaml(reread, write_yaml(root, "sirius"), "sirius");
  REQUIRE(reread.max_concurrent_queries.is_fallback());
}
