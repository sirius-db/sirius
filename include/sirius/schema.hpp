// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
/** @file
 * @brief Immutable typed schemas for Sirius physical plans.
 */
#pragma once
#include <sirius-c/schema.h>
#include <sirius/error.hpp>

#include <cstddef>
#include <expected>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace sirius {
/// Physical type of a schema field or expression. Values may be copied between threads.
/// @code{.cpp}
/// auto type = sirius::DataType::int64;
/// @endcode
enum class DataType : sirius_data_type { boolean = SIRIUS_TYPE_BOOLEAN, int64 = SIRIUS_TYPE_INT64 };

/// One named column. Schema::create copies the descriptor, including its name.
/// Concurrent readers are safe while its members are unmodified.
/// @code{.cpp}
/// sirius::Field id_field{"id", sirius::DataType::int64, false};
/// @endcode
struct Field {
  std::string name;                  ///< Nonempty, case-sensitive UTF-8 name.
  DataType type{DataType::boolean};  ///< Physical value type.
  bool nullable{false};              ///< Whether the field admits nulls.
};

/** @brief Validated ordered fields with unique names.
 * @see @ref cpp_plan_construction "Construct a physical plan" for a complete example.
 * @par Thread safety
 * Copies share immutable storage. Concurrent const operations and copying are safe
 * while the source is alive. Assignment and destruction require exclusive access
 * to that wrapper. Copies can be transferred and released on any thread.
 */
class Schema {
 public:
  /// Share ownership of the same immutable object.
  /// @code{.cpp}
  /// sirius::Schema copy = *schema;
  /// @endcode
  Schema(const Schema& other) noexcept : handle_(other.handle_) { sirius_schema_retain(handle_); }
  /// Replace this reference with another owning reference.
  /// @code{.cpp}
  /// copy = *schema;
  /// @endcode
  Schema& operator=(const Schema& other) noexcept
  {
    sirius_schema_retain(other.handle_);
    sirius_schema_release(handle_);
    handle_ = other.handle_;
    return *this;
  }
  /// Release this owning reference.
  /// @code{.cpp}
  /// { auto copy = *schema; }
  /// @endcode
  ~Schema() noexcept { sirius_schema_release(handle_); }

  /// Copy the fields in their supplied order. Column indices follow this order.
  /// Returns ErrorCode::invalid_argument for an empty schema, unsupported types, duplicate or
  /// invalid names, or a field list too large to represent. Names must be nonempty UTF-8 without
  /// embedded NUL bytes. Name comparison is case-sensitive and compares the exact bytes. Allocation
  /// and unexpected native failures return ErrorCode::allocation_failure and
  /// ErrorCode::internal_error, respectively.
  /// @code{.cpp}
  /// sirius::Field fields[] = {{"id", sirius::DataType::int64, false}};
  /// auto schema = sirius::Schema::create(fields);
  /// if (!schema) return std::unexpected(schema.error());
  /// @endcode
  [[nodiscard]] static std::expected<Schema, Error> create(std::span<const Field> fields) noexcept
  {
    try {
      std::vector<sirius_field> descriptors;
      descriptors.reserve(fields.size());
      for (const auto& field : fields)
        descriptors.push_back({field.name.data(),
                               field.name.size(),
                               static_cast<sirius_data_type>(field.type),
                               field.nullable});
      sirius_schema* result    = nullptr;
      sirius_error* diagnostic = nullptr;
      const auto status =
        sirius_schema_create(descriptors.data(), descriptors.size(), &result, &diagnostic);
      if (status != SIRIUS_SUCCESS) return std::unexpected(Error::from_c(status, diagnostic));
      return Schema(result);
    } catch (const std::bad_alloc&) {
      return std::unexpected(Error{ErrorCode::allocation_failure, {}});
    } catch (const std::length_error&) {
      return std::unexpected(Error{ErrorCode::invalid_argument, {}});
    }
  }

  /// Return the number of fields.
  /// @code{.cpp}
  /// auto columns = schema->size();
  /// @endcode
  [[nodiscard]] std::size_t size() const noexcept { return sirius_schema_field_count(handle_); }

  /// Return an owning copy of the field at a zero-based position. The field and its
  /// name remain usable after the schema is destroyed. An out-of-range index or a name too large
  /// for std::string returns ErrorCode::invalid_argument. Copying the name can return
  /// ErrorCode::allocation_failure. Unexpected native failures return ErrorCode::internal_error.
  /// @code{.cpp}
  /// auto first = schema->field(0);
  /// @endcode
  [[nodiscard]] std::expected<Field, Error> field(std::size_t index) const noexcept
  {
    sirius_field value{};
    sirius_error* diagnostic = nullptr;
    const auto status        = sirius_schema_field(handle_, index, &value, &diagnostic);
    if (status != SIRIUS_SUCCESS) return std::unexpected(Error::from_c(status, diagnostic));
    try {
      return Field{std::string(value.name, value.name_size),
                   static_cast<DataType>(value.type),
                   value.nullable != 0};
    } catch (const std::bad_alloc&) {
      return std::unexpected(Error{ErrorCode::allocation_failure, {}});
    } catch (const std::length_error&) {
      return std::unexpected(Error{ErrorCode::invalid_argument, {}});
    }
  }

 private:
  explicit Schema(sirius_schema* owned) noexcept : handle_(owned) {}
  sirius_schema* handle_;
  friend class PlanNode;
};
}  // namespace sirius
