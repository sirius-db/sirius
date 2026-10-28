/*
 * Copyright 2025, Sirius Contributors.
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

#include "catch.hpp"
#include "sirius_config.hpp"
#include "yaml_reader.hpp"

#include <cucascade/memory/config.hpp>
#include <cucascade/memory/topology_discovery.hpp>
#include <yaml-cpp/yaml.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

using namespace sirius;

namespace {
struct allocation_failure {
  static void from_yaml(const YAML::Node&, allocation_failure&) { throw std::bad_alloc(); }
};

sirius_config resolve_remote_config(const YAML::Node& node)
{
  cucascade::memory::system_topology_info topology{};
  topology.num_gpus       = 1;
  topology.num_numa_nodes = 1;
  topology.gpus           = {{.id = 0, .numa_node = 0}};
  topology.numa_nodes     = {{.id = 0, .memory_capacity = 1ULL << 30, .has_cpus = true}};
  return parsed_sirius_config::from_node(node).resolve(topology);
}

std::vector<cucascade::memory::remote_memory_space_config> remote_spaces(
  const sirius_config& config)
{
  std::vector<cucascade::memory::remote_memory_space_config> result;
  for (const auto& space : config.get_memory_space_configs()) {
    if (const auto* remote = std::get_if<cucascade::memory::remote_memory_space_config>(&space)) {
      result.push_back(*remote);
    }
  }
  return result;
}
}  // namespace

TEST_CASE("yaml reader preserves allocation failures", "[config_opt][allocation]")
{
  yaml::reader reader(YAML::Load("value: {}"));
  allocation_failure value;
  SECTION("optional") { REQUIRE_THROWS_AS(reader.optional("value", value), std::bad_alloc); }
  SECTION("optional value")
  {
    std::optional<allocation_failure> optional;
    REQUIRE_THROWS_AS(reader.optional("value", optional), std::bad_alloc);
  }
  SECTION("validated")
  {
    REQUIRE_THROWS_AS(reader.optional("value", value, [](const auto&) { return true; }),
                      std::bad_alloc);
  }
  SECTION("required") { REQUIRE_THROWS_AS(reader.required("value", value), std::bad_alloc); }
}

TEST_CASE("operator parameters start with hardware-independent defaults", "[config_opt][defaults]")
{
  const operator_params params;
  CHECK(params.scan_task_batch_size == config::DEFAULT_BATCH_SIZE);
  CHECK(params.hash_partition_bytes == config::DEFAULT_BATCH_SIZE);
  CHECK(params.concat_batch_bytes == config::DEFAULT_BATCH_SIZE);
  CHECK(params.sort_sample_bytes == config::DEFAULT_BATCH_SIZE);
  CHECK(params.max_build_hash_table_bytes == 8 * config::DEFAULT_BATCH_SIZE);
  CHECK(params.max_build_probe_resident_bytes == 8 * config::DEFAULT_BATCH_SIZE);
}

TEST_CASE("yaml reader basic types", "[config_opt][basic]")
{
  auto node = YAML::Load(R"(
    int_value: 100
    double_value: 6.28
    string_value: "config setter test"
  )");

  int int_value       = 0;
  double double_value = 0.0;
  std::string string_value;

  yaml::reader r(node);
  r.optional("int_value", int_value);
  r.optional("double_value", double_value);
  r.optional("string_value", string_value);
  r.reject_unknown();

  REQUIRE(int_value == 100);
  REQUIRE(double_value == Approx(6.28));
  REQUIRE(string_value == "config setter test");
}

TEST_CASE("yaml reader optional missing field uses default", "[config_opt][optional]")
{
  auto node = YAML::Load("int_value: 100");

  int int_value       = 0;
  double double_value = 3.14;  // should remain unchanged

  yaml::reader r(node);
  r.optional("int_value", int_value);
  r.optional("double_value", double_value);
  r.reject_unknown();

  REQUIRE(int_value == 100);
  REQUIRE(double_value == 3.14);
}

TEST_CASE("yaml reader required field throws if missing", "[config_opt][required]")
{
  auto node = YAML::Load("int_value: 100");

  int int_value = 0;
  std::string name;

  yaml::reader r(node);
  r.optional("int_value", int_value);
  REQUIRE_THROWS_AS(r.required("name", name), std::runtime_error);
}

TEST_CASE("yaml reader validation rejects out-of-range", "[config_opt][conditional]")
{
  auto node = YAML::Load("value: 100");

  int value = 0;
  yaml::reader r(node);
  REQUIRE_THROWS_AS(r.optional("value", value, yaml::greater_than<int>{150}), std::runtime_error);
  REQUIRE(value == 0);  // value should not be changed due to validation failure
}

TEST_CASE("yaml reader validation with fraction", "[config_opt][conditional]")
{
  auto node = YAML::Load("f: 1.5");
  double f  = 0.5;

  yaml::reader r(node);
  REQUIRE_THROWS_AS(r.optional("f", f, yaml::fraction<double>{}), std::runtime_error);
  REQUIRE(f == 0.5);  // unchanged
}

TEST_CASE("yaml reader byte suffix parsing", "[config_opt][bytes]")
{
  SECTION("plain integers work with bytes()")
  {
    auto node          = YAML::Load("size: 1024");
    std::uint64_t size = 0;
    yaml::reader r(node);
    r.optional("size", yaml::bytes(size));
    REQUIRE(size == 1024);
  }

  SECTION("binary suffixes (Ki/KiB = 1024)")
  {
    auto node       = YAML::Load(R"(a: "4Ki"
b: "4KiB"
c: "1GiB"
d: "2MiB")");
    std::uint64_t a = 0, b = 0, c = 0, d = 0;
    yaml::reader r(node);
    r.optional("a", yaml::bytes(a));
    r.optional("b", yaml::bytes(b));
    r.optional("c", yaml::bytes(c));
    r.optional("d", yaml::bytes(d));
    REQUIRE(a == 4 * 1024ULL);
    REQUIRE(b == 4 * 1024ULL);
    REQUIRE(c == 1ULL * 1024 * 1024 * 1024);
    REQUIRE(d == 2ULL * 1024 * 1024);
  }

  SECTION("decimal suffixes (K/KB/G/GB = 1000)")
  {
    auto node       = YAML::Load(R"(a: "4K"
b: "4KB"
c: "1G"
d: "1T")");
    std::uint64_t a = 0, b = 0, c = 0, d = 0;
    yaml::reader r(node);
    r.optional("a", yaml::bytes(a));
    r.optional("b", yaml::bytes(b));
    r.optional("c", yaml::bytes(c));
    r.optional("d", yaml::bytes(d));
    REQUIRE(a == 4000ULL);
    REQUIRE(b == 4000ULL);
    REQUIRE(c == 1000ULL * 1000 * 1000);
    REQUIRE(d == 1000ULL * 1000 * 1000 * 1000);
  }

  SECTION("fractional values")
  {
    auto node        = YAML::Load(R"(si: "1.5G"
bi: "1.5Gi")");
    std::uint64_t si = 0, bi = 0;
    yaml::reader r(node);
    r.optional("si", yaml::bytes(si));
    r.optional("bi", yaml::bytes(bi));
    REQUIRE(si == static_cast<std::uint64_t>(1.5 * 1000 * 1000 * 1000));
    REQUIRE(bi == static_cast<std::uint64_t>(1.5 * 1024 * 1024 * 1024));
  }

  SECTION("required byte values reject negative integers without mutation")
  {
    auto node          = YAML::Load("size: -1");
    std::uint64_t size = 4096;
    yaml::reader r(node);
    REQUIRE_THROWS_WITH(r.required("size", yaml::bytes(size)),
                        Catch::Matchers::ContainsSubstring("byte value must be non-negative"));
    REQUIRE(size == 4096);
  }

  SECTION("required byte values reject negative suffixed strings without mutation")
  {
    auto node          = YAML::Load(R"(size: "-1GiB")");
    std::uint64_t size = 4096;
    yaml::reader r(node);
    REQUIRE_THROWS_WITH(r.required("size", yaml::bytes(size)),
                        Catch::Matchers::ContainsSubstring("byte value must be non-negative"));
    REQUIRE(size == 4096);
  }

  SECTION("optional byte values reject negative integers without mutation")
  {
    auto node                         = YAML::Load("size: -1");
    std::optional<std::uint64_t> size = 4096;
    yaml::reader r(node);
    REQUIRE_THROWS_WITH(r.optional("size", yaml::bytes(size)),
                        Catch::Matchers::ContainsSubstring("byte value must be non-negative"));
    REQUIRE(size == 4096);
  }

  SECTION("optional byte values reject negative suffixed strings without mutation")
  {
    auto node                         = YAML::Load(R"(size: "-1GiB")");
    std::optional<std::uint64_t> size = 4096;
    yaml::reader r(node);
    REQUIRE_THROWS_WITH(r.optional("size", yaml::bytes(size)),
                        Catch::Matchers::ContainsSubstring("byte value must be non-negative"));
    REQUIRE(size == 4096);
  }

  SECTION("zero byte values remain valid")
  {
    auto node                                  = YAML::Load("size: 0");
    std::uint64_t size                         = 4096;
    std::optional<std::uint64_t> optional_size = 4096;
    yaml::reader r(node);
    r.required("size", yaml::bytes(size));
    REQUIRE(size == 0);

    yaml::reader optional_reader(node);
    optional_reader.optional("size", yaml::bytes(optional_size));
    REQUIRE(optional_size == 0);
  }

  SECTION("string suffix on plain integer field is rejected")
  {
    auto node = YAML::Load(R"(count: "4Ki")");
    int count = 0;
    yaml::reader r(node);
    REQUIRE_THROWS(r.optional("count", count));
  }

  SECTION("invalid suffix throws")
  {
    auto node          = YAML::Load(R"(size: "8X")");
    std::uint64_t size = 0;
    yaml::reader r(node);
    REQUIRE_THROWS_AS(r.optional("size", size), std::runtime_error);
  }

  SECTION("integers above 2^53 parse exactly")
  {
    // 2^53 + 1 is not representable as a double, so parsing through double would round it.
    constexpr std::uint64_t exact = 9007199254740993ULL;
    REQUIRE(yaml::parse_bytes("9007199254740993") == exact);

    auto node          = YAML::Load(R"(size: "9007199254740993")");
    std::uint64_t size = 0;
    yaml::reader r(node);
    r.optional("size", yaml::bytes(size));
    REQUIRE(size == exact);
  }

  SECTION("uint64 max parses exactly")
  {
    constexpr auto exact = std::numeric_limits<std::uint64_t>::max();
    REQUIRE(yaml::parse_bytes("18446744073709551615") == exact);
    REQUIRE(yaml::parse_bytes("18446744073709551615B") == exact);
    REQUIRE(yaml::parse_bytes("18446744073709551615b") == exact);

    auto node          = YAML::Load("size: 18446744073709551615");
    std::uint64_t size = 0;
    yaml::reader r(node);
    r.required("size", yaml::bytes(size));
    REQUIRE(size == exact);

    auto quoted               = YAML::Load(R"(size: "18446744073709551615")");
    std::uint64_t quoted_size = 1;
    yaml::reader quoted_reader(quoted);
    quoted_reader.optional("size", yaml::bytes(quoted_size));
    REQUIRE(quoted_size == exact);
  }

  SECTION("unsuffixed integers above uint64 max are rejected without mutation")
  {
    REQUIRE_THROWS_WITH(yaml::parse_bytes("18446744073709551616"),
                        Catch::Matchers::ContainsSubstring("byte value out of range"));

    auto node          = YAML::Load("size: 18446744073709551616");
    std::uint64_t size = 4096;
    yaml::reader r(node);
    REQUIRE_THROWS_WITH(r.required("size", yaml::bytes(size)),
                        Catch::Matchers::ContainsSubstring("byte value out of range"));
    REQUIRE(size == 4096);

    std::optional<std::uint64_t> optional_size = 4096;
    yaml::reader optional_reader(node);
    REQUIRE_THROWS_WITH(optional_reader.optional("size", yaml::bytes(optional_size)),
                        Catch::Matchers::ContainsSubstring("byte value out of range"));
    REQUIRE(optional_size == 4096);
  }

  SECTION("suffixed values that do not fit in uint64 are rejected without mutation")
  {
    REQUIRE_THROWS_WITH(yaml::parse_bytes("16777216TiB"),
                        Catch::Matchers::ContainsSubstring("byte value out of range"));

    auto node          = YAML::Load(R"(size: "16777216TiB")");
    std::uint64_t size = 4096;
    yaml::reader r(node);
    REQUIRE_THROWS_WITH(r.required("size", yaml::bytes(size)),
                        Catch::Matchers::ContainsSubstring("byte value out of range"));
    REQUIRE(size == 4096);

    std::optional<std::uint64_t> optional_size = 4096;
    yaml::reader optional_reader(node);
    REQUIRE_THROWS_WITH(optional_reader.optional("size", yaml::bytes(optional_size)),
                        Catch::Matchers::ContainsSubstring("byte value out of range"));
    REQUIRE(optional_size == 4096);
  }

  SECTION("a scaled value that overflows double is rejected")
  {
    // 300 nines fit in double; times TiB the product is non-finite.
    REQUIRE_THROWS_WITH(yaml::parse_bytes(std::string(300, '9') + ".0TiB"),
                        Catch::Matchers::ContainsSubstring("byte value out of range"));
    // 400 nines do not fit in double, so stod itself fails.
    REQUIRE_THROWS_WITH(yaml::parse_bytes(std::string(400, '9') + ".0TiB"),
                        Catch::Matchers::ContainsSubstring("byte value out of range"));
  }

  SECTION("a numeric token must be fully consumed")
  {
    REQUIRE_THROWS_WITH(yaml::parse_bytes("1.2.3G"),
                        Catch::Matchers::ContainsSubstring("invalid byte value"));
    REQUIRE_THROWS_WITH(yaml::parse_bytes("."),
                        Catch::Matchers::ContainsSubstring("invalid byte value"));
  }

  SECTION("largest integer GiB that fits parses exactly")
  {
    // 17179869183 GiB == (2^34 - 1) * 2^30 == 2^64 - 2^30, the largest integer GiB count that still
    // fits. One more GiB is exactly 2^64 and must be rejected.
    constexpr std::uint64_t exact = 17179869183ULL * 1024ULL * 1024ULL * 1024ULL;
    static_assert(exact == std::numeric_limits<std::uint64_t>::max() - (1ULL << 30) + 1);

    REQUIRE(yaml::parse_bytes("17179869183GiB") == exact);
    REQUIRE(yaml::parse_bytes("17179869183 GiB") == exact);

    auto node          = YAML::Load(R"(size: "17179869183GiB")");
    std::uint64_t size = 0;
    yaml::reader r(node);
    r.optional("size", yaml::bytes(size));
    REQUIRE(size == exact);
  }
}

TEST_CASE("parse_duration time suffix parsing", "[config_opt][duration]")
{
  using namespace std::chrono_literals;

  SECTION("all supported units")
  {
    REQUIRE(yaml::parse_duration("500ns") == 500ns);
    REQUIRE(yaml::parse_duration("500us") == 500us);
    REQUIRE(yaml::parse_duration("10ms") == 10ms);
    REQUIRE(yaml::parse_duration("2s") == 2s);
    REQUIRE(yaml::parse_duration("3min") == 3min);
    REQUIRE(yaml::parse_duration("1h") == 1h);
  }

  SECTION("long-form and alias suffixes")
  {
    REQUIRE(yaml::parse_duration("500nsec") == 500ns);
    REQUIRE(yaml::parse_duration("500usec") == 500us);
    REQUIRE(yaml::parse_duration("10msec") == 10ms);
    REQUIRE(yaml::parse_duration("2sec") == 2s);
    REQUIRE(yaml::parse_duration("2seconds") == 2s);
    REQUIRE(yaml::parse_duration("3m") == 3min);
    REQUIRE(yaml::parse_duration("3minutes") == 3min);
    REQUIRE(yaml::parse_duration("1hr") == 1h);
    REQUIRE(yaml::parse_duration("1hours") == 1h);
  }

  SECTION("case-insensitive suffixes")
  {
    REQUIRE(yaml::parse_duration("10MS") == 10ms);
    REQUIRE(yaml::parse_duration("2S") == 2s);
    REQUIRE(yaml::parse_duration("1H") == 1h);
  }

  SECTION("whitespace between number and unit") { REQUIRE(yaml::parse_duration("10 ms") == 10ms); }

  SECTION("fractional values round down to nanoseconds")
  {
    REQUIRE(yaml::parse_duration("1.5s") == 1500ms);
    REQUIRE(yaml::parse_duration("2.5ms") == 2500us);
  }

  SECTION("bare numbers are rejected")
  {
    REQUIRE_THROWS_AS(yaml::parse_duration("10"), std::runtime_error);
  }

  SECTION("empty value is rejected")
  {
    REQUIRE_THROWS_AS(yaml::parse_duration(""), std::runtime_error);
  }

  SECTION("unknown suffix is rejected")
  {
    REQUIRE_THROWS_AS(yaml::parse_duration("10years"), std::runtime_error);
  }
}

TEST_CASE("yaml reader duration parsing", "[config_opt][duration]")
{
  using namespace std::chrono_literals;

  SECTION("reads a suffixed string into a milliseconds field")
  {
    auto node = YAML::Load(R"(period: "250ms")");
    std::chrono::milliseconds period{0};
    yaml::reader r(node);
    r.optional("period", period);
    REQUIRE(period == 250ms);
  }

  SECTION("converts units to the target duration type")
  {
    auto node = YAML::Load(R"(period: "2s")");
    std::chrono::milliseconds period{0};
    yaml::reader r(node);
    r.optional("period", period);
    REQUIRE(period == 2000ms);
  }

  SECTION("finer-grained input truncates to target resolution")
  {
    auto node = YAML::Load(R"(period: "1500us")");
    std::chrono::milliseconds period{0};
    yaml::reader r(node);
    r.optional("period", period);
    REQUIRE(period == 1ms);  // 1500us -> 1ms after duration_cast truncation
  }

  SECTION("missing optional field leaves default untouched")
  {
    auto node = YAML::Load(R"(other: 5)");
    std::chrono::milliseconds period{10};
    yaml::reader r(node);
    r.optional("period", period);
    REQUIRE(period == 10ms);
  }

  SECTION("bare number is interpreted in the field's native unit")
  {
    auto node = YAML::Load(R"(period: 10)");
    std::chrono::milliseconds ms_period{0};
    std::chrono::seconds s_period{0};
    yaml::reader r(node);
    r.optional("period", ms_period);
    REQUIRE(ms_period == 10ms);
    yaml::reader r2(node);
    r2.optional("period", s_period);
    REQUIRE(s_period == 10s);
  }
}

TEST_CASE("yaml reader arrays", "[config_opt][array]")
{
  auto node = YAML::Load(R"(
    int_value: [1, 2, 3, 4, 5]
    double_value: [6.28, 3.14]
    string_value: ["config setter test", "another string"]
  )");

  std::vector<int> int_values;
  std::vector<double> double_values;
  std::vector<std::string> string_values;

  yaml::reader r(node);
  r.optional("int_value", int_values);
  r.optional("double_value", double_values);
  r.optional("string_value", string_values);
  r.reject_unknown();

  REQUIRE(int_values == std::vector<int>{1, 2, 3, 4, 5});
  REQUIRE(double_values == std::vector<double>{6.28, 3.14});
  REQUIRE(string_values == std::vector<std::string>{"config setter test", "another string"});
}

// ================ Struct with from_yaml ================= //

struct complex_config {
  int int_value   = 0;
  bool bool_value = false;
  std::vector<std::string> string_values;

  static void from_yaml(const YAML::Node& node, complex_config& opt)
  {
    yaml::reader r(node);
    r.optional("int_value", opt.int_value);
    r.optional("bool_value", opt.bool_value);
    r.optional("string_values", opt.string_values);
    r.reject_unknown();
  }
};

TEST_CASE("yaml reader nested struct via from_yaml", "[config_opt][complex]")
{
  auto node = YAML::Load(R"(
    cfg:
      int_value: 100
      bool_value: true
      string_values: ["config setter test", "another string"]
  )");

  complex_config cfg;
  yaml::reader r(node);
  r.optional("cfg", cfg);
  r.reject_unknown();

  REQUIRE(cfg.int_value == 100);
  REQUIRE(cfg.bool_value == true);
  REQUIRE(cfg.string_values == std::vector<std::string>{"config setter test", "another string"});
}

TEST_CASE("yaml reader list of structs", "[config_opt][complex]")
{
  auto node = YAML::Load(R"(
    cfgs:
      - int_value: 100
        bool_value: true
        string_values: ["config setter test", "another string"]
      - int_value: 200
        bool_value: false
        string_values: ["second config", "more strings"]
  )");

  std::vector<complex_config> cfgs;
  yaml::reader r(node);
  r.optional("cfgs", cfgs);
  r.reject_unknown();

  REQUIRE(cfgs.size() == 2);
  REQUIRE(cfgs[0].int_value == 100);
  REQUIRE(cfgs[0].bool_value == true);
  REQUIRE(cfgs[0].string_values ==
          std::vector<std::string>{"config setter test", "another string"});
  REQUIRE(cfgs[1].int_value == 200);
  REQUIRE(cfgs[1].bool_value == false);
  REQUIRE(cfgs[1].string_values == std::vector<std::string>{"second config", "more strings"});
}

// ================ Enums ================= //

namespace ee {

enum class fruit { apple, banana, orange };

enum class color { red, green, blue };

bool string_to_enum(std::string_view sv, color& c)
{
  static const std::unordered_map<std::string_view, color> str_to_color = {
    {"red", color::red},
    {"green", color::green},
    {"blue", color::blue},
  };
  auto it = str_to_color.find(sv);
  if (it != str_to_color.end()) {
    c = it->second;
    return true;
  }
  return false;
}

bool enum_to_string(color c, std::string& sv)
{
  static const std::unordered_map<color, std::string> color_to_str = {
    {color::red, "red"},
    {color::green, "green"},
    {color::blue, "blue"},
  };
  auto it = color_to_str.find(c);
  if (it != color_to_str.end()) {
    sv = it->second;
    return true;
  }
  return false;
}
}  // namespace ee

TEST_CASE("yaml reader enum arrays", "[config_opt][enum]")
{
  auto node = YAML::Load(R"(
    colors: ["red", "green", "blue"]
    fruits: [0, 1, 2]
  )");

  std::vector<ee::color> colors;
  std::vector<ee::fruit> fruits;

  yaml::reader r(node);
  r.optional("colors", colors);
  r.optional("fruits", fruits);
  r.reject_unknown();

  REQUIRE(colors.size() == 3);
  REQUIRE(colors[0] == ee::color::red);
  REQUIRE(colors[1] == ee::color::green);
  REQUIRE(colors[2] == ee::color::blue);

  REQUIRE(fruits.size() == 3);
  REQUIRE(fruits[0] == ee::fruit::apple);
  REQUIRE(fruits[1] == ee::fruit::banana);
  REQUIRE(fruits[2] == ee::fruit::orange);
}

// ================ Nested structs ================= //

struct nested_config {
  int int_value = 0;
  complex_config cfg;

  static void from_yaml(const YAML::Node& node, nested_config& opt)
  {
    yaml::reader r(node);
    r.optional("int_value", opt.int_value);
    r.optional("inner", opt.cfg);
    r.reject_unknown();
  }
};

TEST_CASE("yaml reader deeply nested struct", "[config_opt][nested]")
{
  auto node = YAML::Load(R"(
    cfg:
      int_value: 100
      inner:
        int_value: 200
        bool_value: true
        string_values: ["nested config test", "another nested string"]
  )");

  nested_config cfg;
  yaml::reader r(node);
  r.optional("cfg", cfg);
  r.reject_unknown();

  REQUIRE(cfg.int_value == 100);
  REQUIRE(cfg.cfg.int_value == 200);
  REQUIRE(cfg.cfg.bool_value == true);
  REQUIRE(cfg.cfg.string_values ==
          std::vector<std::string>{"nested config test", "another nested string"});
}

// ================ Unknown key detection ================= //

TEST_CASE("yaml reader rejects unknown keys", "[config_opt][unregistered]")
{
  SECTION("single unregistered key throws")
  {
    auto node = YAML::Load(R"(
      int_value: 42
      unknown_key: "surprise"
    )");

    int int_value = 0;
    yaml::reader r(node);
    r.optional("int_value", int_value);
    REQUIRE_THROWS_AS(r.reject_unknown(), std::runtime_error);
  }

  SECTION("all keys registered does not throw")
  {
    auto node = YAML::Load(R"(
      int_value: 42
      double_value: 3.14
    )");

    int int_value       = 0;
    double double_value = 0.0;
    yaml::reader r(node);
    r.optional("int_value", int_value);
    r.optional("double_value", double_value);
    REQUIRE_NOTHROW(r.reject_unknown());
  }
}

TEST_CASE("yaml reader nested struct rejects unknown keys", "[config_opt][unknown_key]")
{
  auto node = YAML::Load(R"(
    cfg:
      int_value: 100
      typo_field: 999
  )");

  complex_config cfg;
  yaml::reader r(node);
  REQUIRE_THROWS_AS(r.optional("cfg", cfg), std::runtime_error);
}

// ================ optional_node ================= //

TEST_CASE("yaml reader optional_node", "[config_opt][optional_node]")
{
  auto node = YAML::Load(R"(
    present: 42
  )");

  yaml::reader r(node);
  auto present = r.optional_node("present");
  auto missing = r.optional_node("missing");
  r.reject_unknown();

  REQUIRE(present.has_value());
  REQUIRE(present->as<int>() == 42);
  REQUIRE_FALSE(missing.has_value());
}

TEST_CASE("yaml reader has_value distinguishes null from scalar values", "[config_opt][has_value]")
{
  struct test_case {
    const char* yaml;
    bool expected;
  };

  const test_case cases[] = {
    {"{}", false},
    {"usage_limit_bytes: null", false},
    {"usage_limit_fraction: null", false},
    {"usage_limit_bytes: null\nusage_limit_fraction: null", false},
    {"usage_limit_bytes: 0", true},
    {"usage_limit_fraction: 0.5", true},
    {"usage_limit_bytes: 0\nusage_limit_fraction: null", true},
    {"usage_limit_bytes: null\nusage_limit_fraction: 0.5", true},
  };

  for (auto const& test : cases) {
    INFO("yaml=" << test.yaml);
    auto node = YAML::Load(test.yaml);
    yaml::reader r(node);
    auto const has_explicit_gpu_limit =
      r.has_value("usage_limit_bytes") || r.has_value("usage_limit_fraction");
    REQUIRE(has_explicit_gpu_limit == test.expected);
  }
}

TEST_CASE("yaml reader has distinguishes missing, null, and scalar keys", "[config_opt][has]")
{
  auto node = YAML::Load("null_key: null\nscalar_key: 0");
  yaml::reader r(node);

  CHECK_FALSE(r.has("missing_key"));
  CHECK(r.has("null_key"));
  CHECK(r.has("scalar_key"));
}

// ================ error context ================= //

TEST_CASE("yaml reader error messages include context", "[config_opt][errors]")
{
  auto node = YAML::Load("value: not_a_number");

  int value = 0;
  yaml::reader r(node, "test.section");
  try {
    r.required("value", value);
    FAIL("expected exception");
  } catch (const std::runtime_error& e) {
    std::string msg = e.what();
    REQUIRE(msg.find("test.section.value") != std::string::npos);
  }
}

TEST_CASE("the domain-coverage threshold is validated where it enters the engine",
          "[config_opt][conditional][dynamic_filter]")
{
  // YAML and SQL configuration share this ingress validator.
  config::valid_domain_coverage_threshold const accepts;

  REQUIRE(accepts(0.9));
  REQUIRE(accepts(1.5));  // values above 1.0 disable the gate

  REQUIRE_FALSE(accepts(0.0));  // would suppress every filter
  REQUIRE_FALSE(accepts(-0.5));
  REQUIRE_FALSE(accepts(std::numeric_limits<double>::quiet_NaN()));
  REQUIRE_FALSE(accepts(std::numeric_limits<double>::infinity()));

  // On rejection the YAML surface throws and leaves the default untouched.
  auto node    = YAML::Load("dynamic_filter_domain_coverage_threshold: 0");
  double value = 0.9;
  yaml::reader r(node);
  REQUIRE_THROWS_AS(r.optional("dynamic_filter_domain_coverage_threshold", value, accepts),
                    std::runtime_error);
  REQUIRE(value == 0.9);
}

TEST_CASE("the dynamic-filter switch is consumed from the operator_params YAML section",
          "[config_opt][dynamic_filter]")
{
  auto const path = std::filesystem::temp_directory_path() / "sirius_dynamic_filter.yaml";
  {
    std::ofstream out(path);
    out << "sirius:\n"
           "  operator_params:\n"
           "    enable_dynamic_filter: false\n";
  }

  // Parsing false is the non-vacuous direction because the default is true.
  CHECK(operator_params{}.enable_dynamic_filter);

  sirius_config cfg;
  cfg.load_from_file(path);
  CHECK_FALSE(cfg.get_operator_params().enable_dynamic_filter);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST_CASE("multi-partition dynamic-filter settings preserve defaults and parse byte budgets",
          "[config_opt][dynamic_filter][multi_partition]")
{
  auto const budget         = GENERATE(std::string{"32MiB"}, std::string{"1024"});
  auto const expected_bytes = budget == "32MiB" ? 32ULL * 1024 * 1024 : std::stoull(budget);
  auto const path = std::filesystem::temp_directory_path() / "sirius_dynamic_filter_multi.yaml";
  {
    std::ofstream out(path);
    out << "sirius:\n"
           "  operator_params:\n"
           "    enable_dynamic_filter_multi_partition: false\n"
           "    max_dynamic_filter_bloom_bytes_per_gpu: "
        << budget << '\n';
  }

  sirius_config cfg;
  CHECK(cfg.get_operator_params().enable_dynamic_filter_multi_partition);
  CHECK(cfg.get_operator_params().max_dynamic_filter_bloom_bytes_per_gpu == 256ULL * 1024 * 1024);
  cfg.load_from_file(path);
  CHECK_FALSE(cfg.get_operator_params().enable_dynamic_filter_multi_partition);
  CHECK(cfg.get_operator_params().max_dynamic_filter_bloom_bytes_per_gpu == expected_bytes);
  CHECK(cfg.get_operator_params().enable_dynamic_filter);
  CHECK(cfg.get_operator_params().dynamic_filter_domain_coverage_threshold == Approx(0.9));
  cfg.apply_defaults();
  CHECK(cfg.get_operator_params().enable_dynamic_filter_multi_partition);
  CHECK(cfg.get_operator_params().max_dynamic_filter_bloom_bytes_per_gpu == 256ULL * 1024 * 1024);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST_CASE("multi-partition dynamic-filter byte budgets reject zero and negative YAML",
          "[config_opt][dynamic_filter][multi_partition]")
{
  auto const budget = GENERATE(std::string{"-1"}, std::string{"0"});
  auto const path = std::filesystem::temp_directory_path() / "sirius_dynamic_filter_multi_bad.yaml";
  {
    std::ofstream out(path);
    out << "sirius:\n"
           "  operator_params:\n"
           "    max_dynamic_filter_bloom_bytes_per_gpu: "
        << budget << '\n';
  }
  sirius_config cfg;
  if (budget == "0") {
    // Zero is not a way to disable accumulation; the message names the switch that is.
    CHECK_THROWS_WITH(cfg.load_from_file(path),
                      Catch::Matchers::ContainsSubstring("enable_dynamic_filter_multi_partition"));
  } else {
    CHECK_THROWS(cfg.load_from_file(path));
  }
  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST_CASE("REMOTE spill is disabled by default and by an empty URI", "[sirius][config][remote]")
{
  auto node = YAML::Load("sirius: {}");
  SECTION("omitted remote configuration") {}
  SECTION("empty remote configuration")
  {
    node["sirius"]["memory"]["remote"] = YAML::Node(YAML::NodeType::Map);
  }
  SECTION("explicit empty URI") { node["sirius"]["memory"]["remote"]["uri"] = ""; }
  SECTION("zero capacity is allowed only while disabled")
  {
    node["sirius"]["memory"]["remote"]["uri"]            = "";
    node["sirius"]["memory"]["remote"]["capacity_bytes"] = 0;
  }

  auto const config = resolve_remote_config(node);
  CHECK(remote_spaces(config).empty());
  auto const& transport = config.get_remote_spill_config();
  CHECK(transport.endpoint.empty());
  CHECK(transport.region.empty());
  CHECK_FALSE(transport.signing_mode.has_value());
  CHECK(transport.n_reactors == 2);
  CHECK(transport.staging_bytes == 256ULL * 1024 * 1024);
}

TEST_CASE("REMOTE high-level YAML resolves an S3 URI with default capacity",
          "[sirius][config][remote]")
{
  auto const uri = GENERATE("s3://spill-bucket", "s3://spill-bucket/spills/run-1");
  auto node      = YAML::Load("sirius: {}");
  node["sirius"]["memory"]["remote"]["uri"] = uri;

  auto const config = resolve_remote_config(node);
  auto const spaces = remote_spaces(config);
  REQUIRE(spaces.size() == 1);
  CHECK(spaces.front().remote_id == 0);
  CHECK(spaces.front().uri_prefix == uri);
  CHECK(spaces.front().memory_capacity == (1ULL << 50));
  CHECK(spaces.front().io_context == nullptr);
}

TEST_CASE("REMOTE YAML normalizes S3 schemes and trailing slashes", "[sirius][config][remote]")
{
  auto node = YAML::Load("sirius: {}");
  SECTION("high-level")
  {
    node["sirius"]["memory"]["remote"]["uri"] = "S3://spill-bucket/prefix///";
  }
  SECTION("low-level")
  {
    node["sirius"]["space"]["remote"] = YAML::Load("[{uri_prefix: 'S3://spill-bucket/prefix///'}]");
  }
  auto const config = resolve_remote_config(node);
  auto const spaces = remote_spaces(config);
  REQUIRE(spaces.size() == 1);
  CHECK(spaces.front().uri_prefix == "s3://spill-bucket/prefix");
}

TEST_CASE("REMOTE high-level YAML preserves transport overrides and byte units",
          "[sirius][config][remote]")
{
  using signing_mode = cucascade::io::object_store_config::signing_mode;
  auto const mode    = GENERATE("presigned", "header");
  auto node          = YAML::Load(R"(
    sirius:
      memory:
        remote:
          remote_id: 7
          uri: s3://spill-bucket/spills
          capacity_bytes: 1.5GiB
          endpoint: http://localhost:8333
          region: us-west-2
          n_reactors: 3
          staging_bytes: 64MiB
  )");
  node["sirius"]["memory"]["remote"]["signing_mode"] = mode;

  auto const config = resolve_remote_config(node);
  auto const spaces = remote_spaces(config);
  REQUIRE(spaces.size() == 1);
  CHECK(spaces.front().remote_id == 7);
  CHECK(spaces.front().uri_prefix == "s3://spill-bucket/spills");
  CHECK(spaces.front().memory_capacity == 1536ULL * 1024 * 1024);
  CHECK(spaces.front().io_context == nullptr);
  auto const& transport = config.get_remote_spill_config();
  CHECK(transport.endpoint == "http://localhost:8333");
  CHECK(transport.region == "us-west-2");
  REQUIRE(transport.signing_mode.has_value());
  CHECK(*transport.signing_mode ==
        (std::string(mode) == "header" ? signing_mode::header : signing_mode::presigned));
  CHECK(transport.n_reactors == 3);
  CHECK(transport.staging_bytes == 64ULL * 1024 * 1024);
  CHECK(transport.shutdown_removal_timeout_ms == 10000);
}

TEST_CASE("REMOTE low-level YAML resolves multiple spaces and default capacity",
          "[sirius][config][remote]")
{
  auto const config = resolve_remote_config(YAML::Load(R"(
    sirius:
      space:
        remote:
          - uri_prefix: s3://spill-bucket/first
          - remote_id: 9
            uri_prefix: s3://other-bucket/second
            memory_capacity: 48GiB
  )"));
  auto const spaces = remote_spaces(config);
  REQUIRE(spaces.size() == 2);
  CHECK(spaces[0].remote_id == 0);
  CHECK(spaces[0].uri_prefix == "s3://spill-bucket/first");
  CHECK(spaces[0].memory_capacity == (1ULL << 50));
  CHECK(spaces[0].io_context == nullptr);
  CHECK(spaces[1].remote_id == 9);
  CHECK(spaces[1].uri_prefix == "s3://other-bucket/second");
  CHECK(spaces[1].memory_capacity == 48ULL * 1024 * 1024 * 1024);
  CHECK(spaces[1].io_context == nullptr);
  auto const& transport = config.get_remote_spill_config();
  CHECK(transport.endpoint.empty());
  CHECK(transport.region.empty());
  CHECK_FALSE(transport.signing_mode.has_value());
  CHECK(transport.n_reactors == 2);
  CHECK(transport.staging_bytes == 256ULL * 1024 * 1024);
  CHECK(transport.shutdown_removal_timeout_ms == 10000);
}

TEST_CASE("REMOTE shutdown removal timeout accepts nonnegative milliseconds",
          "[sirius][config][remote]")
{
  auto const timeout = GENERATE(0, 250);
  auto node          = YAML::Load("sirius: {memory: {remote: {uri: 's3://spill-bucket/spills'}}}");
  node["sirius"]["memory"]["remote"]["shutdown_removal_timeout_ms"] = timeout;
  CHECK(resolve_remote_config(node).get_remote_spill_config().shutdown_removal_timeout_ms ==
        timeout);
}

TEST_CASE("REMOTE shutdown removal timeout rejects negative milliseconds",
          "[sirius][config][remote]")
{
  auto node = YAML::Load("sirius: {memory: {remote: {uri: 's3://spill-bucket/spills'}}}");
  node["sirius"]["memory"]["remote"]["shutdown_removal_timeout_ms"] = -1;
  CHECK_THROWS_WITH(
    parsed_sirius_config::from_node(node),
    Catch::Matchers::ContainsSubstring("memory.remote.shutdown_removal_timeout_ms") &&
      Catch::Matchers::ContainsSubstring("value out of range"));
}

TEST_CASE("REMOTE YAML participates in global high-level and low-level exclusivity",
          "[sirius][config][remote]")
{
  auto node = YAML::Load("sirius: {}");
  SECTION("high-level remote excludes every non-empty low-level tier")
  {
    auto const uri  = GENERATE("", "s3://spill-bucket/spills");
    auto const tier = GENERATE("gpu", "host", "disk", "remote");
    CAPTURE(uri, tier);
    node["sirius"]["memory"]["remote"]["uri"] = uri;
    auto space                                = YAML::Load("{}");
    if (std::string(tier) == "remote") { space["uri_prefix"] = "s3://spill-bucket/low"; }
    node["sirius"]["space"][tier].push_back(space);
  }
  SECTION("low-level remote excludes every other high-level tier")
  {
    auto const tier = GENERATE("gpu", "host", "disk");
    CAPTURE(tier);
    node["sirius"]["memory"][tier]    = YAML::Node(YAML::NodeType::Map);
    node["sirius"]["space"]["remote"] = YAML::Load("[{uri_prefix: 's3://spill-bucket/low'}]");
  }

  CHECK_THROWS_WITH(parsed_sirius_config::from_node(node),
                    Catch::Matchers::ContainsSubstring("sirius.memory") &&
                      Catch::Matchers::ContainsSubstring("sirius.space") &&
                      Catch::Matchers::ContainsSubstring("mutually exclusive"));
}

TEST_CASE("REMOTE high-level YAML permits empty low-level space lists", "[sirius][config][remote]")
{
  auto const config = resolve_remote_config(YAML::Load(R"(
    sirius:
      memory:
        remote:
          uri: s3://spill-bucket/spills
      space:
        gpu: []
        host: []
        disk: []
        remote: []
  )"));
  auto const spaces = remote_spaces(config);
  REQUIRE(spaces.size() == 1);
  CHECK(spaces.front().uri_prefix == "s3://spill-bucket/spills");
}

TEST_CASE("REMOTE YAML rejects unknown keys and credential material", "[sirius][config][remote]")
{
  auto const field = GENERATE("capacitty_bytes",
                              "access_key",
                              "secret_key",
                              "access_key_id",
                              "secret_access_key",
                              "session_token",
                              "credentials",
                              "object_store",
                              "AWS_ACCESS_KEY_ID",
                              "AWS_SECRET_ACCESS_KEY");
  CAPTURE(field);
  auto node = YAML::Load("sirius: {}");
  SECTION("high-level")
  {
    auto remote   = node["sirius"]["memory"]["remote"];
    remote["uri"] = "s3://spill-bucket/spills";
    remote[field] = "not-a-credential";
  }
  SECTION("low-level")
  {
    auto remote   = YAML::Load("{uri_prefix: 's3://spill-bucket/spills'}");
    remote[field] = "not-a-credential";
    node["sirius"]["space"]["remote"].push_back(remote);
  }

  CHECK_THROWS_WITH(parsed_sirius_config::from_node(node),
                    Catch::Matchers::ContainsSubstring("unknown config key") &&
                      Catch::Matchers::ContainsSubstring(field));
}

TEST_CASE("REMOTE low-level YAML rejects high-level transport keys", "[sirius][config][remote]")
{
  auto const field = GENERATE("uri",
                              "capacity_bytes",
                              "endpoint",
                              "region",
                              "signing_mode",
                              "n_reactors",
                              "staging_bytes",
                              "shutdown_removal_timeout_ms");
  CAPTURE(field);
  auto node                                   = YAML::Load(R"(
    sirius:
      space:
        remote:
          - uri_prefix: s3://spill-bucket/spills
  )");
  node["sirius"]["space"]["remote"][0][field] = "invalid";
  CHECK_THROWS_WITH(parsed_sirius_config::from_node(node),
                    Catch::Matchers::ContainsSubstring("unknown config key") &&
                      Catch::Matchers::ContainsSubstring(field));
}

TEST_CASE("REMOTE YAML rejects malformed S3 URIs before topology resolution",
          "[sirius][config][remote]")
{
  auto const uri = GENERATE("https://spill-bucket/spills",
                            "file:///tmp/spills",
                            "spill-bucket/spills",
                            "s3://",
                            "s3:///spills",
                            "s3://spill-bucket?query=1",
                            "s3://spill-bucket#fragment",
                            "s3://spill-bucket/spills?query=1",
                            "s3://spill-bucket/spills#fragment",
                            "s3://user@spill-bucket/spills",
                            "s3://user:secret@spill-bucket/spills",
                            "s3://spill-bucket:9000/spills",
                            "s3://spill bucket/spills",
                            "s3://spill-bucket/spill path",
                            " s3://spill-bucket/spills",
                            "s3://spill-bucket/spills ",
                            "s3://spill-bucket/spills\tnext",
                            "s3://spill-bucket/spills\nnext");
  CAPTURE(uri);
  auto node = YAML::Load("sirius: {}");
  SECTION("high-level") { node["sirius"]["memory"]["remote"]["uri"] = uri; }
  SECTION("low-level")
  {
    auto remote          = YAML::Load("{}");
    remote["uri_prefix"] = uri;
    node["sirius"]["space"]["remote"].push_back(remote);
  }
  CHECK_THROWS_AS(parsed_sirius_config::from_node(node), configuration_input_error);
}

TEST_CASE("REMOTE high-level YAML rejects invalid capacity and transport values",
          "[sirius][config][remote]")
{
  auto const [field, value] = GENERATE(table<std::string, std::string>({
    {"remote_id", "-1"},
    {"remote_id", "1.5"},
    {"capacity_bytes", "0"},
    {"capacity_bytes", "-1"},
    {"capacity_bytes", "-1GiB"},
    {"n_reactors", "0"},
    {"n_reactors", "-1"},
    {"n_reactors", "1.5"},
    {"staging_bytes", "0"},
    {"staging_bytes", "-1"},
    {"staging_bytes", "-1MiB"},
    {"signing_mode", "anonymous"},
    {"endpoint", "http://user:secret@localhost:8333"},
  }));
  CAPTURE(field, value);
  auto node = YAML::Load("sirius: {memory: {remote: {uri: 's3://spill-bucket/spills'}}}");
  node["sirius"]["memory"]["remote"][field] = YAML::Load(value);
  CHECK_THROWS_WITH(parsed_sirius_config::from_node(node),
                    Catch::Matchers::ContainsSubstring(field));
}

TEST_CASE("REMOTE low-level YAML requires a URI and positive capacity", "[sirius][config][remote]")
{
  auto const remote = GENERATE("{}",
                               "{uri_prefix: ''}",
                               "{uri_prefix: []}",
                               "{uri_prefix: 's3://spill-bucket', memory_capacity: 0}",
                               "{uri_prefix: 's3://spill-bucket', memory_capacity: -1}",
                               "{uri_prefix: 's3://spill-bucket', memory_capacity: -1GiB}",
                               "{uri_prefix: 's3://spill-bucket', remote_id: -1}",
                               "{uri_prefix: 's3://spill-bucket', remote_id: 1.5}");
  CAPTURE(remote);
  auto node = YAML::Load("sirius: {}");
  node["sirius"]["space"]["remote"].push_back(YAML::Load(remote));
  CHECK_THROWS_AS(parsed_sirius_config::from_node(node), configuration_input_error);
}
