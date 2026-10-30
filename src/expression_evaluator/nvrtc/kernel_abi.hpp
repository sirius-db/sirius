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

#include <cudf/table/table_view.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <expression_evaluator/nvrtc/translation_context.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

/**
 * @file
 * @brief The kernel ABI of the `NVRTC_JIT` sinks: one ordered table of kernel parameters that both
 *        the generated signature and the `cuLaunchKernel` argument array are derived from, so a
 *        parameter can never be emitted in one order and packed in another.
 *
 * The identifier spelling of translation-owned parameters is defined here and used by
 * @c translation_context when it emits loads, so a rename touches one place.
 */

namespace sirius::nvrtc {

/** @brief `in_{id}_{field}`: a field of bound input column @p binding_id. */
[[nodiscard]] std::string input_param_name(std::size_t binding_id, std::string_view field);
/** @brief `c_{id}_{field}`: a field of bound constant @p binding_id. */
[[nodiscard]] std::string constant_param_name(std::size_t binding_id, std::string_view field);
/** @brief `lp_{id}`: the by-value SWAR descriptor of bound LIKE pattern @p binding_id. */
[[nodiscard]] std::string like_pattern_param_name(std::size_t binding_id);

/** @brief One kernel parameter: its CUDA type spelling and identifier. */
struct kernel_parameter {
  std::string cuda_type;
  std::string identifier;
};

/**
 * @brief Ordered kernel parameter table.
 *
 * Construction appends the parameters a @p translation_context needs, in this order:
 *   - per bound input column: a STRING column is `in_N_offsets` (the column's INT32 or INT64
 *     offset type), `in_N_chars`, `in_N_mask` and `in_N_bit_offset`; any other column is
 *     `in_N_data` at its physical type, `in_N_mask` and `in_N_bit_offset`;
 *   - per bound LIKE pattern: `like_pattern_desc lp_N` by value;
 *   - per non-NULL constant: a string is `c_N_ptr` and `c_N_len`, anything else `c_N_val`. NULL
 *     constants have no parameter because their invalidity is written into the prelude.
 * A sink then appends its own parameters with @c add_sink_parameter and supplies their values in
 * the same order through @c kernel_arguments::add_sink_argument.
 */
class kernel_abi {
 public:
  explicit kernel_abi(translation_context const& ctx);

  /** @brief Appends a sink-owned parameter after the translation-owned ones. */
  void add_sink_parameter(std::string cuda_type, std::string identifier);

  /** @brief Appends the parenthesized parameter list of the kernel signature to @p out. */
  void emit_parameter_list(std::string& out) const;

  /**
   * @brief Appends the per-thread definitions derived from the parameters to @p out: for every
   *        string column, `in_N_chars_end` (the end of the slice's characters).
   */
  void emit_kernel_prelude(std::string& out) const;

  [[nodiscard]] std::span<kernel_parameter const> parameters() const noexcept
  {
    return _parameters;
  }

  /** @brief Returns the number of sink-owned parameters. */
  [[nodiscard]] std::size_t sink_parameter_count() const noexcept { return _sink_parameter_count; }

 private:
  friend class kernel_arguments;

  /** @brief The source of a parameter. */
  enum class source : std::uint8_t {
    INPUT_DATA,
    INPUT_OFFSETS,
    INPUT_CHARS,
    INPUT_MASK,
    INPUT_BIT_OFFSET,
    LIKE_PATTERN,
    CONSTANT_VALUE,
    CONSTANT_STRING_PTR,
    CONSTANT_STRING_LEN,
    SINK,
  };

  /** @brief A slot in the parameter table. */
  struct slot {
    source from;
    std::size_t index;  ///< Binding ordinal for translation-owned sources; unused for `sink`.
  };

  /** @brief Adds a parameter to the table. */
  void add(source from, std::size_t index, std::string cuda_type, std::string identifier);

  std::vector<kernel_parameter> _parameters;        ///< The parameters in table order.
  std::vector<slot> _slots;                         ///< The slots in table order.
  std::vector<std::size_t> _string_input_bindings;  ///< The string input bindings.
  std::size_t _sink_parameter_count{0};             ///< The number of sink-owned parameters.
};

/**
 * @brief Host storage for the values of every parameter of a @c kernel_abi, in table order, and
 *        the @c void** array @c cuLaunchKernel takes.
 *
 * The translation-owned values are bound on construction from the input table and the context's
 * constants (string constants are copied to one device buffer owned here). The object, the
 * @c translation_context, and the input table must outlive the launch.
 */
class kernel_arguments {
 public:
  kernel_arguments(kernel_abi const& abi,
                   translation_context const& ctx,
                   cudf::table_view const& input,
                   ::cuda::stream_ref stream,
                   rmm::device_async_resource_ref mr);

  /** @brief Supplies the next sink-owned parameter value; @p host_value must outlive the launch. */
  void add_sink_argument(void* host_value);

  /**
   * @brief The argument array for @c cuLaunchKernel.
   * @throws internal_exception if fewer sink arguments were added than the ABI declares.
   */
  [[nodiscard]] void** data();

 private:
  /** @brief Constant storage. */
  struct alignas(16) constant_storage {
    std::array<std::uint8_t, 16> val_bytes{0};
  };

  /** @brief Input (column reference) storage. */
  struct input_storage {
    void const* data_ptr{nullptr};
    void const* offsets_ptr{nullptr};
    char const* chars_ptr{nullptr};
    std::uint32_t const* mask_ptr{nullptr};
    std::int32_t bit_offset{0};
  };

  /** @brief String constant storage. */
  struct string_constant_storage {
    char const* ptr{nullptr};
    std::int32_t len{0};
  };

  kernel_abi const& _abi;                                  ///< The kernel ABI.
  std::vector<input_storage> _inputs;                      ///< The input storage.
  std::vector<constant_storage> _constants;                ///< The constant storage.
  std::vector<string_constant_storage> _string_constants;  ///< The string constant storage.
  rmm::device_uvector<char> _constant_chars;               ///< The constant characters.
  std::vector<void*> _pointers;                            ///< The pointers.
  std::size_t _sink_arguments_added{0};                    ///< The number of sink arguments added.
};

}  // namespace sirius::nvrtc
