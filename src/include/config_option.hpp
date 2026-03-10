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

#pragma once

#include "log/logging.hpp"

#include <libconfig.h++>

#include <concepts>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

namespace sirius {
namespace config {

// Concepts to identify different types of configuration options

template <typename T>
concept HasStringToEnum = std::is_enum_v<T> && requires(std::string_view sv, T& t, std::string& s) {
  { string_to_enum(sv, t) } -> std::same_as<bool>;
  { enum_to_string(t, s) } -> std::same_as<bool>;  // Just check it compiles
};

template <typename T>
concept IsSimpleEnum = std::is_enum_v<T> && !HasStringToEnum<T>;

template <typename T>
concept IsBackInsertableWithValue = requires(T& t) {
  typename std::enable_if_t<!std::is_same_v<T, std::string>>;
  // 1. Check if 'value_type' member type exists
  typename T::value_type;

  // 2. Check if std::back_inserter(t) is valid
  // This implicitly checks if the type has a 'push_back' method
  std::back_inserter(t);
};

template <typename T>
concept IsBasicConfig =
  std::disjunction_v<std::is_integral<T>, std::is_floating_point<T>, std::is_same<T, std::string>>;

// ================ config type_traits ================= //

template <typename T>
struct config_to_type_traits {
  static constexpr libconfig::Setting::Type type = libconfig::Setting::TypeGroup;
  static T parse_value(std::string_view str_value)
  {
    throw std::invalid_argument("No parse_value implementation for this type.");
  }
};

template <IsSimpleEnum T>
struct config_to_type_traits<T> : public config_to_type_traits<std::underlying_type_t<T>> {
  static T parse_value(std::string_view str_value)
  {
    using underlying_type = std::underlying_type_t<T>;
    underlying_type value = config_to_type_traits<underlying_type>::parse_value(str_value);
    return static_cast<T>(value);
  }
};

template <HasStringToEnum T>
struct config_to_type_traits<T> {
  static constexpr libconfig::Setting::Type type = libconfig::Setting::TypeString;

  static T parse_value(std::string_view str_value)
  {
    T enum_value;
    if (!string_to_enum(str_value, enum_value)) {
      throw std::invalid_argument(
        fmt::format("Invalid configuration string for enum {}", str_value));
    }
    return enum_value;
  }
};

template <std::integral T>
struct config_to_type_traits<T> {
  static constexpr libconfig::Setting::Type type =
    (sizeof(T) > 4) ? libconfig::Setting::TypeInt64 : libconfig::Setting::TypeInt;

  static T parse_value(std::string_view str_value)
  {
    if constexpr (std::is_signed_v<T>) {
      return static_cast<T>(std::stoll(std::string(str_value)));
    } else {
      return static_cast<T>(std::stoull(std::string(str_value)));
    }
  }
};

template <std::floating_point T>
struct config_to_type_traits<T> {
  static constexpr libconfig::Setting::Type type = libconfig::Setting::TypeFloat;
  static T parse_value(std::string_view str_value)
  {
    return static_cast<T>(std::stod(std::string(str_value)));
  }
};

template <>
struct config_to_type_traits<bool> {
  static constexpr libconfig::Setting::Type type = libconfig::Setting::TypeBoolean;
  static bool parse_value(std::string_view str_value)
  {
    if (str_value == "true" || str_value == "1" || str_value == "on") {
      return true;
    } else if (str_value == "false" || str_value == "0" || str_value == "off") {
      return false;
    } else {
      throw std::invalid_argument("Invalid boolean string representation.");
    }
  }
};

template <>
struct config_to_type_traits<std::string> {
  static constexpr libconfig::Setting::Type type = libconfig::Setting::TypeString;
  static std::string parse_value(std::string_view str_value) { return std::string(str_value); }
};

template <IsBackInsertableWithValue T>
struct config_to_type_traits<T> {
  static constexpr libconfig::Setting::Type type = (IsBasicConfig<typename T::value_type>)
                                                     ? libconfig::Setting::TypeArray
                                                     : libconfig::Setting::TypeList;
  static T parse_value(std::string_view str_value)
  {
    throw std::invalid_argument("No parse_value implementation for this type.");
  }
};

// ================ config setter ================= //

struct configuration_setter;

template <typename T>
struct custom_config_registrar {
  static_assert(sizeof(T) == 0, "registrar not specialized for this type");

  static void config(configuration_setter& setter,
                     T& var);  // just to trigger static assert
};

// ================ config assigner implementations ================= //

template <typename ValueType>
struct config_value_applicator {
  static void assign(ValueType& opt, const libconfig::Setting& value);
};

template <std::integral ValueType>
struct config_value_applicator<ValueType> {
  static void assign(ValueType& opt, const libconfig::Setting& value)
  {
    if (value.getType() == libconfig::Setting::TypeInt) {
      int temp = static_cast<int>(value);
      opt      = static_cast<ValueType>(temp);
    } else if (value.getType() == libconfig::Setting::TypeInt64) {
      long long temp = static_cast<long long>(value);
      opt            = static_cast<ValueType>(temp);
    } else {
      throw std::runtime_error("Config setting is not an integer or int64.");
    }
  }
};

template <std::floating_point ValueType>
struct config_value_applicator<ValueType> {
  static void assign(ValueType& opt, const libconfig::Setting& value)
  {
    opt = static_cast<ValueType>(value);
  }
};

template <>
struct config_value_applicator<bool> {
  static void assign(bool& opt, const libconfig::Setting& value) { opt = static_cast<bool>(value); }
};

template <>
struct config_value_applicator<std::string> {
  static void assign(std::string& opt, const libconfig::Setting& value)
  {
    opt = std::string(value.c_str());
  }
};

template <IsSimpleEnum ValueType>
struct config_value_applicator<ValueType> {
  static void assign(ValueType& opt, const libconfig::Setting& value)
  {
    opt = static_cast<ValueType>(static_cast<std::underlying_type_t<ValueType>>(value));
  }
};

template <HasStringToEnum ValueType>
struct config_value_applicator<ValueType> {
  static void assign(ValueType& opt, const libconfig::Setting& value)
  {
    std::string_view str_value = value.c_str();
    // Use the string converter found via ADL
    if (!string_to_enum(str_value, opt)) {
      throw std::invalid_argument(fmt::format("Invalid configuration string for enum {}", value));
    }
  }
};

template <IsBackInsertableWithValue ValueType>
struct config_value_applicator<ValueType> {
  static void assign(ValueType& opt, const libconfig::Setting& value)
  {
    using value_type = typename ValueType::value_type;
    using assigner   = config_value_applicator<value_type>;
    for (const auto& item : value) {
      value_type v{};
      assigner::assign(v, item);
      opt.push_back(std::move(v));
    }
  }
};

// ================ configuration writers implementation ================= //

template <typename ValueType>
struct config_value_exporter {
  static void write(libconfig::Setting& cfg, const ValueType& opt);
};

template <>
struct config_value_exporter<bool> {
  static void write(libconfig::Setting& cfg, bool opt) { cfg = opt; }
};

template <>
struct config_value_exporter<std::string> {
  static void write(libconfig::Setting& cfg, const std::string& opt) { cfg = opt; }
};

template <std::integral ValueType>
struct config_value_exporter<ValueType> {
  static void write(libconfig::Setting& cfg, const ValueType& opt)
  {
    if constexpr (sizeof(ValueType) <= 4)
      cfg = static_cast<int32_t>(opt);
    else
      cfg = static_cast<int64_t>(opt);
  }
};

template <std::floating_point ValueType>
struct config_value_exporter<ValueType> {
  static void write(libconfig::Setting& cfg, const ValueType& opt)
  {
    cfg = static_cast<double>(opt);
  }
};

template <IsSimpleEnum EnumType>
struct config_value_exporter<EnumType> {
  static void write(libconfig::Setting& cfg, const EnumType& opt)
  {
    cfg = static_cast<std::underlying_type_t<EnumType>>(opt);
  }
};

template <HasStringToEnum EnumType>
struct config_value_exporter<EnumType> {
  static void write(libconfig::Setting& cfg, const EnumType& opt)
  {
    std::string sv;
    if (!enum_to_string(opt, sv)) {
      throw std::invalid_argument("Invalid enum value for configuration writing.");
    }
    cfg = sv;
  }
};

template <IsBackInsertableWithValue ListItemType>
struct config_value_exporter<ListItemType> {
  static void write(libconfig::Setting& iterable, const ListItemType& options)
  {
    using value_type                     = typename ListItemType::value_type;
    using writer                         = config_value_exporter<value_type>;
    libconfig::Setting::Type value_ctype = config_to_type_traits<value_type>::type;
    for (const auto& opt : options) {
      auto& cfg = iterable.add(value_ctype);
      writer::write(cfg, opt);
    }
  }
};

// ================ configuration options implementation ================= //

struct config_base {
  virtual ~config_base()                                               = default;
  virtual void apply(const libconfig::Setting& setting)                = 0;
  virtual void write(libconfig::Setting& setting) const                = 0;
  [[nodiscard]] virtual std::string_view path() const noexcept         = 0;
  [[nodiscard]] virtual libconfig::Setting::Type type() const noexcept = 0;
  [[nodiscard]] virtual bool is_required() const noexcept { return false; }
};

template <typename T>
struct value_holder {
  explicit value_holder(T& var) : var_(var) {}

  [[nodiscard]] T* get_or_null() const noexcept { return &var_.get(); }

  [[nodiscard]] T* get_or_null() noexcept { return &var_.get(); }

  [[nodiscard]] T& get_or_create() noexcept { return var_.get(); }

 private:
  std::reference_wrapper<T> var_;
};

template <typename T>
struct optional_value_holder {
  explicit optional_value_holder(std::optional<T>& var) : var_(var) {}

  [[nodiscard]] T* get_or_null() const noexcept
  {
    auto& opt = var_.get();
    if (opt.has_value()) { return &opt.value(); }
    return nullptr;
  }

  [[nodiscard]] T* get_or_null() noexcept
  {
    auto& opt = var_.get();
    if (opt.has_value()) { return &opt.value(); }
    return nullptr;
  }

  [[nodiscard]] T& get_or_create() noexcept
  {
    auto& opt = var_.get();
    if (!opt.has_value()) { opt = T{}; }
    return opt.value();
  }

 private:
  std::reference_wrapper<std::optional<T>> var_;
};

template <typename T, typename... Args>
struct variant_value_holder {
  explicit variant_value_holder(std::variant<Args...>& var) : var_(var) {}

  [[nodiscard]] T* get_or_null() const noexcept
  {
    auto& opt = var_.get();
    return std::get_if<T>(&opt);
  }

  [[nodiscard]] T* get_or_null() noexcept
  {
    auto& opt = var_.get();
    return std::get_if<T>(&opt);
  }

  [[nodiscard]] T& get_or_create() noexcept
  {
    auto& opt = var_.get();
    if (!std::holds_alternative<T>(opt)) { opt = T{}; }
    return std::get<T>(opt);
  }

 private:
  std::reference_wrapper<std::variant<Args...>> var_;
};

// Helper function to create nested groups and return the parent setting
// For path "server.network.port", creates "server" and "server.network" groups
// and returns reference to "network" group along with final component "port"
static inline std::pair<libconfig::Setting*, std::string> ensure_parent_path(
  libconfig::Setting& setting, std::string_view path)
{
  libconfig::Setting* current = &setting;
  std::string path_str(path);
  size_t pos = 0;
  std::string last_component;

  while (pos < path_str.length()) {
    size_t dot_pos = path_str.find('.', pos);
    std::string component =
      (dot_pos == std::string::npos) ? path_str.substr(pos) : path_str.substr(pos, dot_pos - pos);

    // If this is the last component, save it and return
    if (dot_pos == std::string::npos) {
      last_component = component;
      break;
    }

    // Create or get the group for this component
    if (!current->exists(component)) {
      current = &current->add(component, libconfig::Setting::TypeGroup);
    } else {
      current = &(*current)[component];
    }

    pos = dot_pos + 1;
  }

  return {current, last_component};
}

template <typename T, typename holder = value_holder<T>>
struct registered_config : config_base {
  template <typename ConfigType>
  explicit registered_config(std::string_view name, ConfigType& var, bool is_required = false)
    : path_(name.data()), var_(var), is_required_(is_required)
  {
  }

  template <typename ConfigType>
  explicit registered_config(std::string_view name,
                             ConfigType& var,
                             std::function<bool(const T&)> validator,
                             bool is_required = false)
    : path_(name.data()), var_(var), predicate_(std::move(validator)), is_required_(is_required)
  {
  }

  void apply(const libconfig::Setting& cfg) override
  {
    if (predicate_) {
      T temp_value{};
      config_value_applicator<T>::assign(temp_value, cfg);
      if (!predicate_(temp_value)) {
        throw std::invalid_argument(
          fmt::format("Invalid configuration value for option {}", path_.data()));
      }
      var_.get_or_create() = std::move(temp_value);
    } else {
      config_value_applicator<T>::assign(var_.get_or_create(), cfg);
    }
  }

  void write(libconfig::Setting& setting) const override
  {
    auto* ptr = var_.get_or_null();
    if (ptr) {
      auto [parent, name]     = ensure_parent_path(setting, path_);
      libconfig::Setting& cfg = parent->add(name, type());
      config_value_exporter<T>::write(cfg, *ptr);
    }
  }

  [[nodiscard]] std::string_view path() const noexcept override { return path_; }

  [[nodiscard]] libconfig::Setting::Type type() const noexcept override
  {
    return config_to_type_traits<T>::type;
  }

  [[nodiscard]] bool is_required() const noexcept override { return is_required_; }

 private:
  std::string path_;
  holder var_;
  std::function<bool(const T&)> predicate_{nullptr};
  bool is_required_{false};
};

// Specialized registered_config for variant types with validation
template <typename T, typename... Args>
struct registered_config_variant : config_base {
  explicit registered_config_variant(std::string_view name,
                                     std::variant<Args...>& var,
                                     std::function<bool(const T&)> validator,
                                     bool is_required = false)
    : path_(name.data()), var_(var), predicate_(std::move(validator)), is_required_(is_required)
  {
  }

  void apply(const libconfig::Setting& cfg) override
  {
    if (predicate_) {
      T temp_value{};
      config_value_applicator<T>::assign(temp_value, cfg);
      if (!predicate_(temp_value)) {
        throw std::invalid_argument(
          fmt::format("Invalid configuration value for variant option {}", path_.data()));
      }
      var_.get_or_create() = std::move(temp_value);
    } else {
      config_value_applicator<T>::assign(var_.get_or_create(), cfg);
    }
  }

  void write(libconfig::Setting& setting) const override
  {
    auto* ptr = var_.get_or_null();
    if (ptr) {
      auto [parent, name]     = ensure_parent_path(setting, path_);
      libconfig::Setting& cfg = parent->add(name, type());
      config_value_exporter<T>::write(cfg, *ptr);
    }
  }

  [[nodiscard]] std::string_view path() const noexcept override { return path_; }

  [[nodiscard]] libconfig::Setting::Type type() const noexcept override
  {
    return config_to_type_traits<T>::type;
  }

  [[nodiscard]] bool is_required() const noexcept override { return is_required_; }

 private:
  std::string path_;
  variant_value_holder<T, Args...> var_;
  std::function<bool(const T&)> predicate_{nullptr};
  bool is_required_{false};
};

// Specialized registered_config for iterable types with element validation
template <IsBackInsertableWithValue T, typename holder>
struct registered_config_iterable : config_base {
  using value_type = typename T::value_type;

  template <typename ConfigType>
  explicit registered_config_iterable(std::string_view name,
                                      ConfigType& var,
                                      std::function<bool(const value_type&)> element_validator,
                                      bool is_required = false)
    : path_(name.data()),
      var_(var),
      element_predicate_(std::move(element_validator)),
      is_required_(is_required)
  {
  }

  void apply(const libconfig::Setting& cfg) override
  {
    T& container   = var_.get_or_create();
    using assigner = config_value_applicator<value_type>;

    for (const auto& item : cfg) {
      value_type v{};
      assigner::assign(v, item);

      // Validate each element if predicate is provided
      if (element_predicate_ && !element_predicate_(v)) {
        throw std::invalid_argument(
          fmt::format("Invalid element value in configuration array for option {}", path_.data()));
      }

      container.push_back(std::move(v));
    }
  }

  void write(libconfig::Setting& setting) const override
  {
    auto* ptr = var_.get_or_null();
    if (ptr) {
      auto [parent, name]     = ensure_parent_path(setting, path_);
      libconfig::Setting& cfg = parent->add(name, type());
      config_value_exporter<T>::write(cfg, *ptr);
    }
  }

  [[nodiscard]] std::string_view path() const noexcept override { return path_; }

  [[nodiscard]] libconfig::Setting::Type type() const noexcept override
  {
    return config_to_type_traits<T>::type;
  }

  [[nodiscard]] bool is_required() const noexcept override { return is_required_; }

 private:
  std::string path_;
  holder var_;
  std::function<bool(const value_type&)> element_predicate_{nullptr};
  bool is_required_{false};
};

// ================ validators ================= //

template <typename T>
struct less_than {
  T threshold;
  bool operator()(const T& value) const noexcept { return value < threshold; }
};

template <typename T>
struct greater_than {
  T threshold;
  bool operator()(const T& value) const noexcept { return value > threshold; }
};

template <typename T>
struct between {
  T lower;
  T upper;
  bool operator()(const T& value) const noexcept { return (value >= lower) && (value <= upper); }
};

template <typename T>
struct fraction {
  bool operator()(const T& value) const noexcept
  {
    return (value >= static_cast<T>(0)) && (value <= static_cast<T>(1));
  }
};

struct path_exists {
  bool operator()(const std::string& value) const noexcept
  {
    return std::filesystem::exists(value);
  }
};

// ================ configuration_setter ================= //

enum class config_requirement { required, optional };

struct configuration_setter {
  template <typename T>
    requires IsBasicConfig<T>
  void add_config(std::string_view name,
                  T& opt,
                  std::predicate<const T&> auto validator,
                  config_requirement requirement = config_requirement::optional)
  {
    bool is_required = (requirement == config_requirement::required);
    configs_.emplace_back(
      std::make_unique<registered_config<T>>(name, opt, std::move(validator), is_required));
  }

  template <typename ConfigType>
  void add_config(std::string_view name,
                  ConfigType& instance,
                  config_requirement requirement = config_requirement::optional)
  {
    bool is_required = (requirement == config_requirement::required);
    configs_.emplace_back(
      std::make_unique<registered_config<ConfigType>>(name, instance, is_required));
  }

  template <typename ConfigType, typename... Args>
  void add_variant_config(std::string_view name, std::variant<Args...>& instance)
  {
    configs_.emplace_back(
      std::make_unique<registered_config<ConfigType, variant_value_holder<ConfigType, Args...>>>(
        name, instance));
  }

  // Add variant config with validation
  template <typename ConfigType, typename... Args>
  void add_variant_config(std::string_view name,
                          std::variant<Args...>& instance,
                          std::predicate<const ConfigType&> auto validator,
                          config_requirement requirement = config_requirement::optional)
  {
    bool is_required = (requirement == config_requirement::required);
    configs_.emplace_back(std::make_unique<registered_config_variant<ConfigType, Args...>>(
      name, instance, std::move(validator), is_required));
  }

  // Add iterable config with element-level validation
  template <IsBackInsertableWithValue T>
  void add_config(std::string_view name,
                  T& container,
                  std::predicate<const typename T::value_type&> auto element_validator,
                  config_requirement requirement = config_requirement::optional)
  {
    bool is_required = (requirement == config_requirement::required);
    configs_.emplace_back(std::make_unique<registered_config_iterable<T, value_holder<T>>>(
      name, container, std::move(element_validator), is_required));
  }

  template <typename T>
    requires IsBasicConfig<T>
  void add_optional_config(std::string_view name,
                           std::optional<T>& opt,
                           std::predicate<const T&> auto validator)
  {
    configs_.emplace_back(std::make_unique<registered_config<T, optional_value_holder<T>>>(
      name, opt, std::move(validator)));
  }

  template <typename T>
  void add_optional_config(std::string_view name, std::optional<T>& opt)
  {
    configs_.emplace_back(
      std::make_unique<registered_config<T, optional_value_holder<T>>>(name, opt));
  }

  // Add optional iterable config with element-level validation
  template <IsBackInsertableWithValue T>
  void add_optional_config(std::string_view name,
                           std::optional<T>& container,
                           std::predicate<const typename T::value_type&> auto element_validator)
  {
    configs_.emplace_back(std::make_unique<registered_config_iterable<T, optional_value_holder<T>>>(
      name, container, std::move(element_validator)));
  }

  static const libconfig::Setting* safe_lookup(const libconfig::Setting& setting,
                                               std::string_view path)
  {
    const libconfig::Setting* current = &setting;
    std::string path_str(path);
    size_t pos = 0;

    while (pos < path_str.length()) {
      size_t dot_pos = path_str.find('.', pos);
      std::string component =
        (dot_pos == std::string::npos) ? path_str.substr(pos) : path_str.substr(pos, dot_pos - pos);

      if (!current->exists(component)) { return nullptr; }

      current = &(*current)[component];

      if (dot_pos == std::string::npos) { break; }
      pos = dot_pos + 1;
    }

    return current;
  }

  void apply(const libconfig::Setting& setting)
  {
    std::for_each(configs_.begin(), configs_.end(), [&](auto& setter) {
      auto path                     = setter->path();
      const libconfig::Setting* cfg = safe_lookup(setting, path);

      if (cfg) {
        setter->apply(*cfg);
      } else if (setter->is_required()) {
        throw std::invalid_argument(
          fmt::format("Missing required configuration option: {}", path.data()));
      }
    });
  }

  void write(libconfig::Setting& setting) const
  {
    for (const auto& exporter : configs_) {
      exporter->write(setting);
    }
  }

 private:
  std::vector<std::unique_ptr<config_base>> configs_;
};

// ================ config specialization registrations ================= //

template <typename ValueType>
void config_value_applicator<ValueType>::assign(ValueType& opt, const libconfig::Setting& value)
{
  configuration_setter setter;
  custom_config_registrar<ValueType>::config(setter, opt);
  setter.apply(value);
}

template <typename ValueType>
void config_value_exporter<ValueType>::write(libconfig::Setting& cfg, const ValueType& opt)
{
  auto& mutable_opt = const_cast<ValueType&>(opt);
  configuration_setter setter;
  custom_config_registrar<ValueType>::config(setter, mutable_opt);
  setter.write(cfg);
}

}  // namespace config
}  // namespace sirius
