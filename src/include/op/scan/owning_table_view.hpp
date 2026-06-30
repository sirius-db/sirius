/*
 * Copyright 2026, Sirius Contributors.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <functional>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace sirius::op::scan {

namespace detail {

/**
 * @brief An owner whose pointee can surrender its columns as owned
 *        @c std::unique_ptr<cudf::column> objects (the canonical case is
 *        @c std::unique_ptr<cudf::table>).
 *
 * When this holds, a view built over the owner can be materialized into a
 * @c cudf::table by *moving* the surviving column buffers out of the owner
 * instead of copying them — no new device buffers are allocated. Owners that
 * do not satisfy this concept fall back to a copying materialization.
 */
template <typename Owner>
concept no_alloc_materializable = requires(Owner& owner) {
  { (*owner).release() } -> std::same_as<std::vector<std::unique_ptr<cudf::column>>>;
};

//===----------------------------------------------------------------------===//
// my_view
//===----------------------------------------------------------------------===//
/**
 * @brief A type-erased data owner paired with a column selection describing the
 *        currently-exposed @c cudf::table_view.
 *
 * The selection is stored as indices into the owner's *original* columns so
 * that @ref reorder_columns / @ref drop_columns are pure index manipulations
 * (no allocation) and so that @ref materialize can, when the owner supports it
 * (@ref no_alloc_materializable), move the surviving column buffers out of the
 * owner rather than copying them.
 */
class my_view {
 public:
  /// Type-erase @p owner (kept alive for the view's lifetime) together with the
  /// @p base_view that names its columns. The initial selection exposes every
  /// column of @p base_view in order.
  template <typename Owner>
  my_view(Owner&& owner, cudf::table_view base_view)
    : _model(
        std::make_unique<owner_model<std::decay_t<Owner>>>(std::forward<Owner>(owner), base_view)),
      _selection(identity_selection(base_view.num_columns()))
  {
  }

  [[nodiscard]] cudf::table_view view() const { return _model->full_view().select(_selection); }

  [[nodiscard]] std::size_t n_columns() const { return _selection.size(); }

  /// Row count of the underlying data, independent of which columns are
  /// currently exposed.
  [[nodiscard]] cudf::size_type num_rows() const { return _model->full_view().num_rows(); }

  /// Swap the columns at the given current positions, applying each swap in
  /// order. Pure index manipulation; allocates nothing. Throws
  /// @c std::out_of_range if any position is outside the current view.
  void reorder_columns(std::span<const std::pair<std::size_t, std::size_t>> swaps)
  {
    for (auto const& [a, b] : swaps) {
      check_position(a);
      check_position(b);
      std::swap(_selection[a], _selection[b]);
    }
  }

  /// Drop the columns at the given current positions. Pure index manipulation;
  /// allocates nothing. Throws @c std::out_of_range if any position is outside
  /// the current view.
  void drop_columns(std::span<const std::size_t> positions)
  {
    std::vector<std::size_t> sorted(positions.begin(), positions.end());
    std::sort(sorted.begin(), sorted.end(), std::greater<>{});
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    for (std::size_t p : sorted) {
      check_position(p);
      _selection.erase(_selection.begin() + static_cast<std::ptrdiff_t>(p));
    }
  }

  /// Replace the current view with exactly the columns at @p positions, in the
  /// given order (projection + reorder in one). Pure index manipulation;
  /// allocates nothing. Throws @c std::out_of_range if any position is outside
  /// the current view, or @c std::invalid_argument on a duplicate position
  /// (the no-alloc materialization moves each column out at most once).
  void select_columns(std::span<const std::size_t> positions)
  {
    std::vector<cudf::size_type> next;
    next.reserve(positions.size());
    for (std::size_t p : positions) {
      check_position(p);
      next.push_back(_selection[p]);
    }
    std::vector<cudf::size_type> sorted(next);
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
      throw std::invalid_argument("owning_table_view: duplicate column position in select_columns");
    }
    _selection = std::move(next);
  }

  /// Realize the currently-exposed columns into an owned @c cudf::table. Moves
  /// buffers out of the owner when it is @ref no_alloc_materializable, otherwise
  /// copies the selected view (allocating).
  [[nodiscard]] std::unique_ptr<cudf::table> materialize(rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
  {
    return _model->materialize(_selection, stream, mr);
  }

 private:
  struct owner_concept {
    virtual ~owner_concept() = default;

    [[nodiscard]] virtual cudf::table_view full_view() const = 0;

    [[nodiscard]] virtual std::unique_ptr<cudf::table> materialize(
      std::span<const cudf::size_type> selection,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) = 0;
  };

  template <typename Owner>
  struct owner_model final : owner_concept {
    owner_model(Owner owner, cudf::table_view base_view)
      : _owner(std::move(owner)), _base_view(std::move(base_view))
    {
    }

    [[nodiscard]] cudf::table_view full_view() const override { return _base_view; }

    [[nodiscard]] std::unique_ptr<cudf::table> materialize(
      std::span<const cudf::size_type> selection,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) override
    {
      if constexpr (no_alloc_materializable<Owner>) {
        // Move the selected columns out of the owner; no device buffer copy.
        // Precondition: the selection contains no duplicate original indices
        // (guaranteed by the swap/drop operations, which only permute/subset an
        // identity selection). Moving the same column twice would be a bug, so
        // assert each slot is still occupied before moving it out.
        auto columns = (*_owner).release();
        std::vector<std::unique_ptr<cudf::column>> out;
        out.reserve(selection.size());
        for (auto idx : selection) {
          auto& column = columns[static_cast<std::size_t>(idx)];
          assert(column != nullptr && "owning_table_view: duplicate column index in selection");
          out.push_back(std::move(column));
        }
        return std::make_unique<cudf::table>(std::move(out));
      } else {
        // Generic owner: copy the selected view into a fresh table.
        std::vector<cudf::size_type> sel(selection.begin(), selection.end());
        return std::make_unique<cudf::table>(_base_view.select(sel), stream, mr);
      }
    }

    Owner _owner;
    cudf::table_view _base_view;
  };

  /// Validate a caller-supplied column position against the current view.
  void check_position(std::size_t position) const
  {
    if (position >= _selection.size()) {
      throw std::out_of_range("owning_table_view: column position " + std::to_string(position) +
                              " is out of range (n_columns=" + std::to_string(_selection.size()) +
                              ")");
    }
  }

  static std::vector<cudf::size_type> identity_selection(cudf::size_type n)
  {
    std::vector<cudf::size_type> v(static_cast<std::size_t>(n));
    std::iota(v.begin(), v.end(), 0);
    return v;
  }

  std::unique_ptr<owner_concept> _model;
  std::vector<cudf::size_type> _selection;
};

}  // namespace detail

//===----------------------------------------------------------------------===//
// owning_table_view
//===----------------------------------------------------------------------===//
/**
 * @brief A handle that exposes a @c cudf::table_view while owning the data
 *        behind it, regardless of whether that data is a fully-materialized
 *        @c cudf::table or a view into some other type-erased owner.
 *
 * The handle is in one of three states: an owned @c cudf::table, a
 * @ref detail::my_view (type-erased owner + column selection), or empty.
 *
 * Only @ref materialize and @ref release may allocate device memory (the
 * view -> table transition). Every other mutating operation —
 * @ref reorder_columns and @ref drop_columns — is a pure view manipulation:
 * if the handle currently owns a @c cudf::table it is first demoted into a
 * @ref detail::my_view (a @c unique_ptr move, not a copy), after which the
 * column selection is permuted/subset in place. Because of this they are
 * @c const and operate on a @c mutable state member.
 */
class owning_table_view {
 public:
  /// Construct empty (no valid state).
  owning_table_view() = default;

  /// Take ownership of an already-materialized table.
  explicit owning_table_view(std::unique_ptr<cudf::table> table) : _state(std::move(table)) {}

  /// Take ownership of a table by move.
  explicit owning_table_view(cudf::table&& table)
    : _state(std::make_unique<cudf::table>(std::move(table)))
  {
  }

  /**
   * @brief Construct from a type-erased @p owner that keeps the data alive and
   *        an explicit @p base_view naming its columns.
   *
   * If @p Owner is @ref detail::no_alloc_materializable, materialization will
   * move column buffers out of the owner; otherwise it copies @p base_view.
   */
  template <typename Owner>
    requires(!std::same_as<std::decay_t<Owner>, owning_table_view> &&
             !std::same_as<std::decay_t<Owner>, std::unique_ptr<cudf::table>>)
  owning_table_view(Owner&& owner, cudf::table_view base_view)
    : _state(std::make_unique<detail::my_view>(std::forward<Owner>(owner), base_view))
  {
  }

  /// The currently-exposed view. Empty view when the handle has no valid state.
  [[nodiscard]] cudf::table_view view() const;

  /// Number of columns in the currently-exposed view.
  [[nodiscard]] std::size_t n_columns() const;

  /// Number of rows in the data. Independent of the exposed column selection;
  /// 0 when the handle has no valid state.
  [[nodiscard]] cudf::size_type num_rows() const;

  /// The column at the given current position. Throws @c std::out_of_range if
  /// @p index is outside the currently-exposed view.
  [[nodiscard]] cudf::column_view column(std::size_t index) const;

  /// The data types of the currently-exposed columns, in order.
  [[nodiscard]] std::vector<cudf::data_type> column_types() const;

  /// Swap columns at the given current positions (applied in order). Never
  /// allocates; demotes an owned table to a view first if necessary.
  void reorder_columns(std::span<const std::pair<std::size_t, std::size_t>> swaps) const;

  /// Drop columns at the given current positions. Never allocates; demotes an
  /// owned table to a view first if necessary.
  void drop_columns(std::span<const std::size_t> positions) const;

  /// Keep exactly the columns at the given current positions, in the given
  /// order (projection + reorder). Never allocates; demotes an owned table to a
  /// view first if necessary. Throws on out-of-range or duplicate positions.
  void select_columns(std::span<const std::size_t> positions) const;

  /// Realize a view state into an owned table. No-op if already materialized or
  /// empty. May allocate (copying path) depending on the underlying owner.
  void materialize(rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                   rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /// Materialize if needed and surrender the owned table, leaving the handle
  /// empty. Returns nullptr if the handle had no valid state.
  [[nodiscard]] std::unique_ptr<cudf::table> release(
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /// Release all held data, leaving the handle empty.
  void drop();

  /// True when the handle holds a valid state (a table or a view).
  explicit operator bool() const { return !std::holds_alternative<std::monostate>(_state); }

 private:
  /// Demote an owned-table state into an equivalent view state. No-op for the
  /// other states. Allocates nothing (moves the table's unique_ptr).
  void ensure_view() const;

  mutable std::
    variant<std::unique_ptr<cudf::table>, std::unique_ptr<detail::my_view>, std::monostate>
      _state{std::monostate{}};
};

}  // namespace sirius::op::scan
