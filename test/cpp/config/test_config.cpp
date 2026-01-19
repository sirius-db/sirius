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
#include "config_option.hpp"

#include <libconfig.h++>

#include <cstdlib>
#include <exception>
#include <optional>
#include <variant>

TEST_CASE("use configuration basic setters", "[config_opt][basic]")
{
  using namespace sirius;
  config::configuration_setter setter;
  int int_value       = 0;
  double double_value = 0.0;
  std::string string_value;
  setter.add_config("int_value", int_value);
  setter.add_config("double_value", double_value);
  setter.add_config("string_value", string_value);

  // Create a libconfig config object
  libconfig::Config libconfig;
  libconfig.readString(R"(
          int_value = 100;
          double_value = 6.28;
          string_value = "config setter test";
    )");

  try {
    setter.apply(libconfig.getRoot());
  } catch (const std::exception& e) {
    std::cerr << "Setting not found: " << e.what() << std::endl;
  }

  REQUIRE(int_value == 100);
  REQUIRE(double_value == Approx(6.28));
  REQUIRE(string_value == "config setter test");

  libconfig::Config root;
  setter.write(root.getRoot());

  // reset values and test write
  int_value    = 0;
  double_value = 0.0;
  string_value = "";

  try {
    setter.apply(root.getRoot());
  } catch (const std::exception& e) {
    std::cerr << "Setting not found: " << e.what() << std::endl;
  }

  REQUIRE(int_value == 100);
  REQUIRE(double_value == Approx(6.28));
  REQUIRE(string_value == "config setter test");
}

TEST_CASE("use configuration basic optional setters", "[config_opt][optional]")
{
  using namespace sirius;
  config::configuration_setter setter;
  std::optional<int> int_value = std::nullopt;
  setter.add_optional_config("int_value", int_value);

  // Create a libconfig config object
  libconfig::Config libconfig;
  libconfig.readString(R"(
          int_value = 100;
          double_value = 6.28;
          string_value = "config setter test";
    )");

  try {
    setter.apply(libconfig.getRoot());
  } catch (const std::exception& e) {
    std::cerr << "Setting not found: " << e.what() << std::endl;
  }

  REQUIRE(int_value.value() == 100);

  int_value.reset();
  libconfig::Config root;
  setter.write(root.getRoot());

  // reset values and test write

  try {
    setter.apply(root.getRoot());
  } catch (const std::exception& e) {
    std::cerr << "Setting not found: " << e.what() << std::endl;
  }

  REQUIRE_FALSE(int_value.has_value());
}

TEST_CASE("use configuration basic variant setters", "[config_opt][variant]")
{
  using namespace sirius;
  config::configuration_setter setter;
  std::variant<std::monostate, int, double, std::string> one_of_options;
  setter.add_variant_config<int>("int_value", one_of_options);
  setter.add_variant_config<double>("double_value", one_of_options);
  setter.add_variant_config<std::string>("string_value", one_of_options);

  {  // Create a libconfig config object
    one_of_options = std::monostate{};
    libconfig::Config libconfig;
    libconfig.readString(R"(
          int_value = 100;
    )");

    try {
      setter.apply(libconfig.getRoot());
    } catch (const std::exception& e) {
      std::cerr << "Setting not found: " << e.what() << std::endl;
    }

    REQUIRE(std::get<int>(one_of_options) == 100);
  }
  {  // Create a libconfig config object
    one_of_options = std::monostate{};
    libconfig::Config libconfig;
    libconfig.readString(R"(
          double_value = 10.20;
    )");

    try {
      setter.apply(libconfig.getRoot());
    } catch (const std::exception& e) {
      std::cerr << "Setting not found: " << e.what() << std::endl;
    }

    REQUIRE(std::get<double>(one_of_options) == 10.20);
  }
  {  // Create a libconfig config object
    one_of_options = std::monostate{};
    libconfig::Config libconfig;
    libconfig.readString(R"(
          string_value = "test string";
    )");

    try {
      setter.apply(libconfig.getRoot());
    } catch (const std::exception& e) {
      std::cerr << "Setting not found: " << e.what() << std::endl;
    }

    REQUIRE(std::get<std::string>(one_of_options) == "test string");
  }
}

TEST_CASE("use configuration basic setters with condition", "[config_opt][conditional]")
{
  using namespace sirius;
  config::configuration_setter setter;
  int int_value = 0;
  setter.add_config("int_value", int_value, sirius::config::greater_than<int>{50});

  // Create a libconfig config object
  libconfig::Config libconfig;
  libconfig.readString(R"(
          int_value = 100;
          double_value = 6.28;
          string_value = "config setter test";
    )");

  REQUIRE_THROWS_AS(setter.apply(libconfig.getRoot()), std::invalid_argument);
  REQUIRE(int_value == 0);  // value should not be changed due to validation failure
}

TEST_CASE("use configuration array setters", "[config_opt][array]")
{
  using namespace sirius;
  config::configuration_setter setter;
  std::vector<int> int_values;
  std::vector<double> double_values;
  std::vector<std::string> string_values;
  setter.add_config("int_value", int_values);
  setter.add_config("double_value", double_values);
  setter.add_config("string_value", string_values);
  // Create a libconfig config object
  libconfig::Config libconfig;
  libconfig.readString(R"(
          int_value = [1, 2, 3, 4, 5];
          double_value = [6.28, 3.14];
          string_value = ["config setter test", "another string"];
    )");

  try {
    setter.apply(libconfig.getRoot());
  } catch (const std::exception& e) {
    std::cerr << "Setting not found: " << e.what() << std::endl;
  }

  REQUIRE(int_values == std::vector<int>{1, 2, 3, 4, 5});
  REQUIRE(double_values == std::vector<double>{6.28, 3.14});
  REQUIRE(string_values == std::vector<std::string>{"config setter test", "another string"});

  libconfig::Config root;
  setter.write(root.getRoot());

  int_values.clear();
  double_values.clear();
  string_values.clear();

  try {
    setter.apply(root.getRoot());
  } catch (const std::exception& e) {
    std::cerr << "Setting not found: " << e.what() << std::endl;
  }

  CHECK(int_values == std::vector<int>{1, 2, 3, 4, 5});
  CHECK(double_values == std::vector<double>{6.28, 3.14});
  CHECK(string_values == std::vector<std::string>{"config setter test", "another string"});
}

struct complex_config {
  int int_value   = 0;
  bool bool_value = false;
  std::vector<std::string> string_values;
};

template <>
struct sirius::config::custom_config_registrar<complex_config> {
  static void config(sirius::config::configuration_setter& setter, complex_config& opt)
  {
    setter.add_config("int_value", opt.int_value);
    setter.add_config("bool_value", opt.bool_value);
    setter.add_config("string_values", opt.string_values);
  }
};

TEST_CASE("use configuration class setters with registered type", "[config_opt][complex]")
{
  using namespace sirius;
  config::configuration_setter setter;
  complex_config cfg;
  setter.add_config("cfg", cfg);
  // Create a libconfig config object
  libconfig::Config libconfig;
  libconfig.readString(R"(
          cfg = {
            int_value = 100;
            bool_value = true;
            string_values = ["config setter test", "another string"];
          };
    )");

  try {
    setter.apply(libconfig.getRoot());
  } catch (const std::exception& e) {
    std::cerr << "Setting not found: " << e.what() << std::endl;
  }

  REQUIRE(cfg.int_value == 100);
  REQUIRE(cfg.bool_value == true);
  REQUIRE(cfg.string_values == std::vector<std::string>{"config setter test", "another string"});

  libconfig::Config root;
  setter.write(root.getRoot());
  cfg = complex_config{};
  try {
    setter.apply(root.getRoot());
  } catch (const std::exception& e) {
    std::cerr << "Setting not found: " << e.what() << std::endl;
  }

  REQUIRE(cfg.int_value == 100);
  REQUIRE(cfg.bool_value == true);
  REQUIRE(cfg.string_values == std::vector<std::string>{"config setter test", "another string"});
}

TEST_CASE("use configuration class setters array  with registered type", "[config_opt][complex]")
{
  using namespace sirius;
  config::configuration_setter setter;
  std::vector<complex_config> cfgs;
  setter.add_config("cfgs", cfgs);
  // Create a libconfig config object
  libconfig::Config libconfig;
  try {
    libconfig.readString(R"(
        cfgs = (
          {
            int_value = 100;
            bool_value = true;
            string_values = ["config setter test", "another string"];
          },
          {
            int_value = 200;
            bool_value = false;
            string_values = ["second config", "more strings"];
          }
        );
    )");
    setter.apply(libconfig.getRoot());
  } catch (const std::exception& e) {
    std::cerr << "Setting not found: " << e.what() << std::endl;
  }

  REQUIRE(cfgs.size() == 2);
  REQUIRE(cfgs[0].int_value == 100);
  REQUIRE(cfgs[0].bool_value == true);
  REQUIRE(cfgs[0].string_values ==
          std::vector<std::string>{"config setter test", "another string"});
  REQUIRE(cfgs[1].int_value == 200);
  REQUIRE(cfgs[1].bool_value == false);
  REQUIRE(cfgs[1].string_values == std::vector<std::string>{"second config", "more strings"});
}
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

TEST_CASE("use configuration class setters custom array of enum", "[config_opt][enum]")
{
  using namespace sirius;
  config::configuration_setter setter;
  std::vector<ee::color> colors;
  std::vector<ee::fruit> fruits;

  setter.add_config("colors", colors);
  setter.add_config("fruits", fruits);

  libconfig::Config libconfig;
  try {
    libconfig.readString(R"(
        colors = ["red", "green", "blue"];
        fruits = [0, 1 ,2];
    )");
    setter.apply(libconfig.getRoot());
  } catch (const std::exception& e) {
    std::cerr << "Setting not found: " << e.what() << std::endl;
  }

  REQUIRE(colors.size() == 3);
  REQUIRE(colors[0] == ee::color::red);
  REQUIRE(colors[1] == ee::color::green);
  REQUIRE(colors[2] == ee::color::blue);

  REQUIRE(fruits.size() == 3);
  REQUIRE(fruits[0] == ee::fruit::apple);
  REQUIRE(fruits[1] == ee::fruit::banana);
  REQUIRE(fruits[2] == ee::fruit::orange);
}

struct nested_config {
  int int_value = 0;
  complex_config cfg;
};

template <>
struct sirius::config::custom_config_registrar<nested_config> {
  static void config(sirius::config::configuration_setter& setter, nested_config& opt)
  {
    setter.add_config("int_value", opt.int_value);
    setter.add_config("inner", opt.cfg);
  }
};

TEST_CASE("use configuration class setters nested  with registered type", "[config_opt][nested]")
{
  using namespace sirius;
  config::configuration_setter setter;
  nested_config cfg;
  setter.add_config("cfg", cfg);
  // Create a libconfig config object
  libconfig::Config libconfig;
  try {
    libconfig.readString(R"(
        cfg =
          {
            int_value = 100;
            inner = {
              int_value = 200;
              bool_value = true;
              string_values = ["nested config test", "another nested string"];
    };
          };
    )");
    setter.apply(libconfig.getRoot());
  } catch (const std::exception& e) {
    std::cerr << "Setting not found: " << e.what() << std::endl;
  }

  REQUIRE(cfg.int_value == 100);
  REQUIRE(cfg.cfg.int_value == 200);
  REQUIRE(cfg.cfg.bool_value == true);
  REQUIRE(cfg.cfg.string_values ==
          std::vector<std::string>{"nested config test", "another nested string"});

  // Test writing back
  libconfig::Config root;
  setter.write(root.getRoot());
  cfg = nested_config{};
  try {
    setter.apply(root.getRoot());
  } catch (const std::exception& e) {
    std::cerr << "Setting not found: " << e.what() << std::endl;
  }
  REQUIRE(cfg.int_value == 100);
  REQUIRE(cfg.cfg.int_value == 200);
  REQUIRE(cfg.cfg.bool_value == true);
  REQUIRE(cfg.cfg.string_values ==
          std::vector<std::string>{"nested config test", "another nested string"});
}

TEST_CASE("use env variable to set variables of a registered class", "[config_opt][required]")
{
  using namespace sirius;
  config::configuration_setter setter;
  ee::color favorite_color = ee::color::red;
  int int_value            = 0;
  setter.add_config("favorite_color", favorite_color, config::config_requirement::required);
  setter.add_config("int_value", int_value);

  libconfig::Config libconfig;
  libconfig.readString(R"(
        int_value = 100;
    )");

  REQUIRE_THROWS_AS(setter.apply(libconfig.getRoot()), std::exception);
}

TEST_CASE("use nested naming with config", "[config_opt][nested_naming]")
{
  using namespace sirius;
  config::configuration_setter setter;
  ee::color favorite_color = ee::color::red;
  int int_value            = 0;
  setter.add_config("color.favorite", favorite_color, config::config_requirement::required);
  setter.add_config("int.value", int_value);

  libconfig::Config libconfig;
  libconfig.readString(R"(
        int = {
          value = 100;
        };
        color = {
          favorite = "green";
        };
    )");

  setter.apply(libconfig.getRoot());

  REQUIRE(int_value == 100);
  REQUIRE(favorite_color == ee::color::green);
}
