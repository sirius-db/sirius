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

#include <yaml-cpp/yaml.h>

#include <chrono>
#include <concepts>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace sirius::yaml {

// ================ Concepts ================= //

/// Enum with ADL-found string_to_enum / enum_to_string converters.
template <typename T>
concept StringEnum = std::is_enum_v<T> && requires(std::string_view sv, T& t, std::string& s) {
  { string_to_enum(sv, t) } -> std::same_as<bool>;
  { enum_to_string(t, s) } -> std::same_as<bool>;
};

/// Enum without string converters (uses underlying integer).
template <typename T>
concept IntEnum = std::is_enum_v<T> && !StringEnum<T>;

/// Type with a static from_yaml(const YAML::Node&, T&) method.
template <typename T>
concept HasFromYaml = requires(const YAML::Node& n, T& t) {
  { T::from_yaml(n, t) };
};

// ================ Validators ================= //

template <typename T>
struct fraction {
  bool operator()(const T& v) const noexcept { return v >= T{0} && v <= T{1}; }
  static constexpr const char* description() { return "must be between 0.0 and 1.0"; }
};

template <typename T>
struct greater_than {
  T threshold;
  bool operator()(const T& v) const noexcept { return v > threshold; }
};

template <typename T>
struct between {
  T lo, hi;
  bool operator()(const T& v) const noexcept { return v >= lo && v <= hi; }
};

struct path_exists {
  bool operator()(const std::string& v) const noexcept { return std::filesystem::exists(v); }
};

// ================ Byte-suffix parsing ================= //

/// Parse a string with an optional byte suffix into a byte count.
///
/// Binary (powers of 1024): Ki/KiB, Mi/MiB, Gi/GiB, Ti/TiB
/// Decimal (powers of 1000): K/KB, M/MB, G/GB, T/TB
/// Plain integers without suffix are returned as-is.
///
/// Follows the Kubernetes/systemd convention where K=1000, Ki=1024.
inline std::uint64_t parse_bytes(std::string_view sv)
{
  if (sv.empty()) { throw std::runtime_error("empty byte value"); }

  // Find where the numeric part ends
  size_t pos = 0;
  while (pos < sv.size() && (std::isdigit(sv[pos]) || sv[pos] == '.' || sv[pos] == '-')) {
    ++pos;
  }

  if (pos == 0) { throw std::runtime_error("invalid byte value: '" + std::string(sv) + "'"); }

  double number = std::stod(std::string(sv.substr(0, pos)));
  if (number < 0) { throw std::runtime_error("byte value must be non-negative"); }
  auto suffix = sv.substr(pos);

  // Strip leading whitespace from suffix
  while (!suffix.empty() && suffix[0] == ' ') {
    suffix.remove_prefix(1);
  }

  if (suffix.empty() || suffix == "B" || suffix == "b") {
    return static_cast<std::uint64_t>(number);
  }

  // Determine if binary (Ki, Mi, Gi, Ti, KiB, MiB, GiB, TiB) or decimal (K, KB, M, MB, ...)
  char unit   = static_cast<char>(std::toupper(static_cast<unsigned char>(suffix[0])));
  bool binary = (suffix.size() >= 2 && suffix[1] == 'i');

  std::uint64_t base = binary ? 1024ULL : 1000ULL;
  std::uint64_t multiplier;
  switch (unit) {
    case 'K': multiplier = base; break;
    case 'M': multiplier = base * base; break;
    case 'G': multiplier = base * base * base; break;
    case 'T': multiplier = base * base * base * base; break;
    default: throw std::runtime_error("unknown byte suffix: '" + std::string(suffix) + "'");
  }

  return static_cast<std::uint64_t>(number * static_cast<double>(multiplier));
}

// ================ Time-suffix parsing ================= //

/// Parse a string with a time-unit suffix into a std::chrono::nanoseconds count.
///
/// Supported suffixes (case-insensitive):
///   ns / nsec           -> nanoseconds
///   us / usec           -> microseconds
///   ms / msec           -> milliseconds
///   s  / sec / seconds  -> seconds
///   m  / min / minutes  -> minutes
///   h  / hr  / hours    -> hours
///
/// A unit suffix is required: bare numbers are rejected because the intended
/// unit would be ambiguous. Fractional values are allowed (e.g. "1.5s").
inline std::chrono::nanoseconds parse_duration(std::string_view sv)
{
  using namespace std::chrono_literals;

  if (sv.empty()) { throw std::runtime_error("empty time value"); }

  // Find where the numeric part ends
  size_t pos = 0;
  while (pos < sv.size() &&
         (std::isdigit(static_cast<unsigned char>(sv[pos])) || sv[pos] == '.' || sv[pos] == '-')) {
    ++pos;
  }

  if (pos == 0) { throw std::runtime_error("invalid time value: '" + std::string(sv) + "'"); }

  double number = std::stod(std::string(sv.substr(0, pos)));
  auto suffix   = sv.substr(pos);

  // Strip leading whitespace from suffix
  while (!suffix.empty() && suffix[0] == ' ') {
    suffix.remove_prefix(1);
  }

  // Normalize the suffix to lowercase for case-insensitive matching.
  std::string unit;
  unit.reserve(suffix.size());
  for (char c : suffix) {
    unit.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }

  // Scale the (possibly fractional) count by the chrono literal for the unit,
  // then round down to whole nanoseconds.
  auto scale = [number](auto literal_unit) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(number * literal_unit);
  };

  if (unit == "ns" || unit == "nsec") { return scale(1ns); }
  if (unit == "us" || unit == "usec") { return scale(1us); }
  if (unit == "ms" || unit == "msec") { return scale(1ms); }
  if (unit == "s" || unit == "sec" || unit == "seconds") { return scale(1s); }
  if (unit == "m" || unit == "min" || unit == "minutes") { return scale(1min); }
  if (unit == "h" || unit == "hr" || unit == "hours") { return scale(1h); }

  throw std::runtime_error("unknown or missing time suffix: '" + std::string(suffix) + "'");
}

// ================ read_yaml — type-dispatched YAML→C++ ================= //

/// Read a YAML scalar into a C++ value. Overloaded for each supported type.

inline void read_yaml(const YAML::Node& node, bool& out) { out = node.as<bool>(); }

inline void read_yaml(const YAML::Node& node, std::string& out) { out = node.as<std::string>(); }

template <std::floating_point T>
void read_yaml(const YAML::Node& node, T& out)
{
  out = static_cast<T>(node.as<double>());
}

template <std::integral T>
  requires(!std::is_same_v<T, bool>)
void read_yaml(const YAML::Node& node, T& out)
{
  if constexpr (sizeof(T) > 4)
    out = static_cast<T>(node.as<long long>());
  else
    out = static_cast<T>(node.as<int>());
}

/// Wrapper that marks an integral field as accepting byte suffixes (e.g. "8Gi", "512M").
/// Use with reader::optional/required: `r.optional("capacity_bytes", bytes(cfg.capacity));`
template <std::integral T>
struct bytes_value {
  T& ref;
};

template <std::integral T>
bytes_value<T> bytes(T& v)
{
  return {v};
}

template <std::integral T>
void read_yaml(const YAML::Node& node, bytes_value<T>& out)
{
  T val{};
  try {
    if constexpr (sizeof(T) > 4) {
      auto const parsed = node.as<long long>();
      if (parsed < 0) { throw std::runtime_error("byte value must be non-negative"); }
      val = static_cast<T>(parsed);
    } else {
      auto const parsed = node.as<int>();
      if (parsed < 0) { throw std::runtime_error("byte value must be non-negative"); }
      val = static_cast<T>(parsed);
    }
  } catch (const YAML::BadConversion&) {
    val = static_cast<T>(parse_bytes(node.as<std::string>()));
  }
  out.ref = val;
}

/// Wrapper for optional byte values.
template <std::integral T>
struct optional_bytes_value {
  std::optional<T>& ref;
};

template <std::integral T>
optional_bytes_value<T> bytes(std::optional<T>& v)
{
  return {v};
}

template <std::integral T>
void read_yaml(const YAML::Node& node, optional_bytes_value<T>& out)
{
  T val{};
  try {
    if constexpr (sizeof(T) > 4) {
      auto const parsed = node.as<long long>();
      if (parsed < 0) { throw std::runtime_error("byte value must be non-negative"); }
      val = static_cast<T>(parsed);
    } else {
      auto const parsed = node.as<int>();
      if (parsed < 0) { throw std::runtime_error("byte value must be non-negative"); }
      val = static_cast<T>(parsed);
    }
  } catch (const YAML::BadConversion&) {
    val = static_cast<T>(parse_bytes(node.as<std::string>()));
  }
  out.ref = val;
}

/// Read a YAML scalar into a std::chrono::duration.
///
/// Accepts either a bare number — interpreted in the duration's native unit
/// (milliseconds for std::chrono::milliseconds, seconds for std::chrono::seconds,
/// etc.) — or a string with a time-unit suffix (e.g. "10ms", "1.5s", "500us"),
/// which is parsed via parse_duration and cast to the field's resolution.
template <typename Rep, typename Period>
void read_yaml(const YAML::Node& node, std::chrono::duration<Rep, Period>& out)
{
  using target = std::chrono::duration<Rep, Period>;
  try {
    out = target{node.as<Rep>()};
  } catch (const YAML::BadConversion&) {
    out = std::chrono::duration_cast<target>(parse_duration(node.as<std::string>()));
  }
}

template <StringEnum T>
void read_yaml(const YAML::Node& node, T& out)
{
  auto str = node.as<std::string>();
  if (!string_to_enum(std::string_view{str}, out)) {
    throw std::runtime_error("invalid enum value '" + str + "'");
  }
}

template <IntEnum T>
void read_yaml(const YAML::Node& node, T& out)
{
  out = static_cast<T>(node.as<int>());
}

/// Struct with a static from_yaml method.
template <HasFromYaml T>
void read_yaml(const YAML::Node& node, T& out)
{
  T::from_yaml(node, out);
}

/// Vector of any supported type.
template <typename T>
void read_yaml(const YAML::Node& node, std::vector<T>& out)
{
  if (!node.IsSequence()) { throw std::runtime_error("expected a sequence"); }
  for (const auto& item : node) {
    T val{};
    read_yaml(item, val);
    out.push_back(std::move(val));
  }
}

// ================ reader ================= //

/// Reads fields from a YAML map node with validation and unknown-key detection.
class reader {
 public:
  explicit reader(const YAML::Node& node, std::string context = "")
    : node_(node), context_(std::move(context))
  {
    if (node_.IsDefined() && !node_.IsNull() && !node_.IsMap()) {
      throw std::runtime_error(context_.empty()
                                 ? "expected a mapping, got " + type_name(node_)
                                 : context_ + ": expected a mapping, got " + type_name(node_));
    }
  }

  /// Read an optional field. If the key exists, deserialize into `out`.
  template <typename T>
  void optional(const std::string& key, T&& out)
  {
    consumed_.insert(key);
    auto child = find(key);
    if (!child.IsDefined() || child.IsNull()) return;
    try {
      read_yaml(child, out);
    } catch (const std::exception& e) {
      throw std::runtime_error(format_error(key, e.what()));
    }
  }

  /// Read an optional field into a std::optional. Sets to the value if present, leaves as-is if
  /// not.
  template <typename T>
  void optional(const std::string& key, std::optional<T>& out)
  {
    consumed_.insert(key);
    auto child = find(key);
    if (!child.IsDefined() || child.IsNull()) return;
    try {
      T val{};
      read_yaml(child, val);
      out = std::move(val);
    } catch (const std::exception& e) {
      throw std::runtime_error(format_error(key, e.what()));
    }
  }

  /// Read an optional field with validation.
  template <typename T, typename Validator>
  void optional(const std::string& key, T& out, Validator validator)
  {
    consumed_.insert(key);
    auto child = find(key);
    if (!child.IsDefined() || child.IsNull()) return;
    try {
      T temp{};
      read_yaml(child, temp);
      if (!validator(temp)) { throw std::runtime_error("value out of range"); }
      out = std::move(temp);
    } catch (const std::exception& e) {
      throw std::runtime_error(format_error(key, e.what()));
    }
  }

  /// Read a required field. Throws if missing.
  template <typename T>
  void required(const std::string& key, T&& out)
  {
    consumed_.insert(key);
    auto child = find(key);
    if (!child.IsDefined() || child.IsNull()) {
      throw std::runtime_error(format_error(key, "required but missing"));
    }
    try {
      read_yaml(child, out);
    } catch (const std::exception& e) {
      throw std::runtime_error(format_error(key, e.what()));
    }
  }

  /// Get a child node without deserializing. Returns std::nullopt if missing.
  std::optional<YAML::Node> optional_node(const std::string& key)
  {
    consumed_.insert(key);
    auto child = find(key);
    if (!child.IsDefined() || child.IsNull()) return std::nullopt;
    return child;
  }

  /// Check whether a key exists in the map.
  [[nodiscard]] bool has(const std::string& key) const
  {
    if (!node_.IsMap()) return false;
    for (auto it = node_.begin(); it != node_.end(); ++it) {
      if (it->first.as<std::string>() == key) return true;
    }
    return false;
  }

  /// Check whether a key exists in the map and has a non-null value.
  [[nodiscard]] bool has_value(const std::string& key) const
  {
    auto child = find(key);
    return child.IsDefined() && !child.IsNull();
  }

  /// Throw if the map contains keys that were not consumed by optional/required calls.
  void reject_unknown() const
  {
    if (!node_.IsMap()) return;
    for (auto it = node_.begin(); it != node_.end(); ++it) {
      auto key = it->first.as<std::string>();
      if (!consumed_.contains(key)) {
        throw std::runtime_error(context_.empty()
                                   ? "unknown config key: '" + key + "'"
                                   : "unknown config key: '" + key + "' in " + context_);
      }
    }
  }

 private:
  /// Find a child by key using iteration (avoids YAML::Node::operator[] mutation).
  [[nodiscard]] YAML::Node find(const std::string& key) const
  {
    if (!node_.IsMap()) return {};
    for (auto it = node_.begin(); it != node_.end(); ++it) {
      if (it->first.as<std::string>() == key) return it->second;
    }
    return {};
  }

  [[nodiscard]] std::string format_error(const std::string& key, const std::string& msg) const
  {
    if (context_.empty()) return "'" + key + "': " + msg;
    return "'" + context_ + "." + key + "': " + msg;
  }

  [[nodiscard]] static std::string type_name(const YAML::Node& node)
  {
    switch (node.Type()) {
      case YAML::NodeType::Scalar: return "a scalar";
      case YAML::NodeType::Sequence: return "a sequence";
      case YAML::NodeType::Map: return "a mapping";
      case YAML::NodeType::Null: return "null";
      default: return "undefined";
    }
  }

  YAML::Node node_;
  std::string context_;
  std::set<std::string> consumed_;
};

}  // namespace sirius::yaml
