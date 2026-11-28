/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <algorithm>
#include <compare>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace sirius::op::scan {
/// View of elements stored in equal-sized blocks that need not be adjacent in memory; a
/// contiguous range is the one-block case. Owns neither the blocks nor the block table.
template <typename T>
class block_span {
 public:
  class iterator;
  using value_type = std::remove_cv_t<T>;

  block_span() = default;
  /// A contiguous range, with the lifetime rules of std::span.
  template <typename Range>
    requires std::convertible_to<Range&&, std::span<T>> &&
             (!std::same_as<std::remove_cvref_t<Range>, block_span>)
  block_span(Range&& range) noexcept  // NOLINT(google-explicit-constructor)
  {
    std::span<T> values(std::forward<Range>(range));
    data_ = values.data();
    size_ = values.size();
  }
  /// @p size elements laid out in order over @p blocks of @p block_bytes each. An element never
  /// straddles two blocks, so a block size that is not a multiple of the element size holds less.
  block_span(std::span<std::byte* const> blocks, std::size_t block_bytes, std::size_t size)
    : size_(size)
  {
    auto const per_block = block_bytes / sizeof(T);
    if (size > blocks.size() * per_block)
      throw std::invalid_argument("blocks are too small for their elements");
    if (blocks.size() == 1) {
      data_ = reinterpret_cast<T*>(blocks.front());
    } else if (size) {
      table_     = blocks.data();
      per_block_ = per_block;
    }
  }

  /// The same elements, read-only.
  template <typename U>
    requires std::same_as<U const, T>
  block_span(block_span<U> const& other) noexcept  // NOLINT(google-explicit-constructor)
    : table_(other.table_), data_(other.data_), per_block_(other.per_block_), size_(other.size_)
  {
  }

  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] bool empty() const noexcept { return !size_; }
  [[nodiscard]] T& operator[](std::size_t index) const noexcept
  {
    return table_ ? block(index / per_block_)[index % per_block_] : data_[index];
  }
  [[nodiscard]] T& front() const noexcept { return (*this)[0]; }
  [[nodiscard]] T& back() const noexcept { return (*this)[size_ - 1]; }
  /// The elements from @p index up to the end of its block, or of the span.
  [[nodiscard]] std::span<T> run(std::size_t index) const noexcept
  {
    if (index >= size_) return {};
    if (!table_) return {data_ + index, size_ - index};
    auto const offset = index % per_block_;
    return {block(index / per_block_) + offset, std::min(per_block_ - offset, size_ - index)};
  }
  [[nodiscard]] iterator begin() const noexcept { return {*this, 0}; }
  [[nodiscard]] iterator end() const noexcept { return {*this, size_}; }

 private:
  template <typename U>
  friend class block_span;
  [[nodiscard]] T* block(std::size_t index) const noexcept
  {
    return reinterpret_cast<T*>(table_[index]);
  }
  std::byte* const* table_ = nullptr;  // null: contiguous at data_
  T* data_                 = nullptr;
  std::size_t per_block_   = 0;
  std::size_t size_        = 0;
};

/// Random access; stepping forward stays inside the current block until its end.
template <typename T>
class block_span<T>::iterator {
 public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type        = block_span::value_type;
  using difference_type   = std::ptrdiff_t;
  using pointer           = T*;
  using reference         = T&;

  iterator() = default;
  reference operator*() const noexcept { return *current_; }
  pointer operator->() const noexcept { return current_; }
  reference operator[](difference_type n) const noexcept
  {
    return span_[index_ + static_cast<std::size_t>(n)];
  }
  iterator& operator++() noexcept
  {
    ++index_;
    if (++current_ == run_end_) seek();
    return *this;
  }
  iterator operator++(int) noexcept
  {
    auto previous = *this;
    ++*this;
    return previous;
  }
  iterator& operator--() noexcept { return *this += -1; }
  iterator operator--(int) noexcept
  {
    auto previous = *this;
    --*this;
    return previous;
  }
  iterator& operator+=(difference_type n) noexcept
  {
    index_ += static_cast<std::size_t>(n);
    seek();
    return *this;
  }
  iterator& operator-=(difference_type n) noexcept { return *this += -n; }
  friend iterator operator+(iterator it, difference_type n) noexcept { return it += n; }
  friend iterator operator+(difference_type n, iterator it) noexcept { return it += n; }
  friend iterator operator-(iterator it, difference_type n) noexcept { return it -= n; }
  friend difference_type operator-(iterator const& a, iterator const& b) noexcept
  {
    return static_cast<difference_type>(a.index_) - static_cast<difference_type>(b.index_);
  }
  friend bool operator==(iterator const& a, iterator const& b) noexcept
  {
    return a.index_ == b.index_;
  }
  friend std::strong_ordering operator<=>(iterator const& a, iterator const& b) noexcept
  {
    return a.index_ <=> b.index_;
  }

 private:
  friend class block_span;
  iterator(block_span span, std::size_t index) noexcept : span_(span), index_(index) { seek(); }
  void seek() noexcept
  {
    auto const run = span_.run(index_);
    current_       = run.data();
    run_end_       = run.data() + run.size();
  }
  block_span span_;
  std::size_t index_ = 0;
  T* current_        = nullptr;
  T* run_end_        = nullptr;
};
}  // namespace sirius::op::scan
