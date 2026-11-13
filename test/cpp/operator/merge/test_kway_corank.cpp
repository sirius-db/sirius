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
#include "op/merge/detail/kway_merge_detail.hpp"
#include "operator/merge/merge_test_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace sirius::op::merge::detail;
namespace merge_utils = sirius::test::merge_utils;

namespace {

/// One key on the host: an INT64 value or, for STRING windows, a text.
struct host_key {
  bool valid{true};
  int64_t value{0};
  std::string text;
};

using host_window = std::vector<host_key>;

/// k sorted key windows of one INT64 or STRING column, their comparator and the host reference.
class corank_fixture {
 public:
  corank_fixture(std::vector<host_window> windows,
                 bool strings,
                 cudf::order order           = cudf::order::ASCENDING,
                 cudf::null_order null_order = cudf::null_order::AFTER)
    : _windows(std::move(windows)), _order(order), _null_order(null_order)
  {
    for (auto& window : _windows) {
      std::stable_sort(window.begin(), window.end(), [&](const host_key& a, const host_key& b) {
        return compare(a, b) < 0;
      });
      _tables.push_back(merge_utils::make_table(make_column(window, strings)));
      _views.push_back(_tables.back()->view());
      _lengths.push_back(static_cast<cudf::size_type>(window.size()));
      _total += static_cast<int64_t>(window.size());
    }
    const std::vector<cudf::order> orders           = {order};
    const std::vector<cudf::null_order> null_orders = {null_order};
    _cmp.emplace(kway_key_comparator::create(_views,
                                             orders,
                                             null_orders,
                                             cudf::get_default_stream(),
                                             cudf::get_current_device_resource_ref()));

    // The merged order: every (input, row), sorted by key, then input, then row.
    for (std::size_t i = 0; i < _windows.size(); ++i) {
      for (std::size_t r = 0; r < _windows[i].size(); ++r) {
        _merged.emplace_back(static_cast<int32_t>(i), static_cast<cudf::size_type>(r));
      }
    }
    std::sort(_merged.begin(), _merged.end(), [&](const auto& a, const auto& b) {
      const int cmp = compare(_windows[a.first][a.second], _windows[b.first][b.second]);
      if (cmp != 0) { return cmp < 0; }
      return a < b;
    });
  }

  [[nodiscard]] std::size_t k() const { return _windows.size(); }
  [[nodiscard]] int64_t total() const { return _total; }
  [[nodiscard]] const std::vector<cudf::size_type>& lengths() const { return _lengths; }

  /// Reference: per-input counts of the first @p rank rows of the merged order.
  [[nodiscard]] std::vector<cudf::size_type> expected_counts(cudf::size_type rank) const
  {
    std::vector<cudf::size_type> counts(k(), 0);
    for (cudf::size_type p = 0; p < rank; ++p) {
      ++counts[static_cast<std::size_t>(_merged[static_cast<std::size_t>(p)].first)];
    }
    return counts;
  }

  /// Reference: min(max_rows, r_safe, Σ lengths).
  [[nodiscard]] cudf::size_type expected_target(const std::vector<uint8_t>& truncated,
                                                cudf::size_type max_rows) const
  {
    int64_t target = std::min<int64_t>(max_rows, _total);
    for (std::size_t i = 0; i < k(); ++i) {
      if (truncated[i] == 0) { continue; }
      if (_windows[i].empty()) { return 0; }
      const std::pair<int32_t, cudf::size_type> last{static_cast<int32_t>(i), _lengths[i] - 1};
      const auto position = std::find(_merged.begin(), _merged.end(), last) - _merged.begin();
      target              = std::min<int64_t>(target, position + 1);
    }
    return static_cast<cudf::size_type>(target);
  }

  /// corank_split on the device: out[r * k + i].
  [[nodiscard]] std::vector<cudf::size_type> corank(const std::vector<cudf::size_type>& ranks) const
  {
    auto stream    = cudf::get_default_stream();
    auto d_lengths = upload(_lengths);
    auto d_ranks   = upload(ranks);
    rmm::device_uvector<cudf::size_type> out(
      ranks.size() * k(), stream, cudf::get_current_device_resource_ref());
    corank_split(*_cmp,
                 cudf::device_span<const cudf::size_type>(d_lengths.data(), d_lengths.size()),
                 cudf::device_span<const cudf::size_type>(d_ranks.data(), d_ranks.size()),
                 cudf::device_span<cudf::size_type>(out.data(), out.size()),
                 stream);
    stream.sync();
    std::vector<cudf::size_type> host(out.size());
    if (!host.empty()) {
      REQUIRE(cudaMemcpy(host.data(),
                         out.data(),
                         host.size() * sizeof(cudf::size_type),
                         cudaMemcpyDeviceToHost) == cudaSuccess);
    }
    return host;
  }

  /// safe_target on the device.
  [[nodiscard]] cudf::size_type target(const std::vector<uint8_t>& truncated,
                                       cudf::size_type max_rows) const
  {
    auto d_lengths   = upload(_lengths);
    auto d_truncated = upload(truncated);
    return safe_target(*_cmp,
                       cudf::device_span<const cudf::size_type>(d_lengths.data(), d_lengths.size()),
                       cudf::device_span<const uint8_t>(d_truncated.data(), d_truncated.size()),
                       max_rows,
                       cudf::get_default_stream());
  }

  /// Checks corank_split at @p ranks against the reference prefix of the merged order.
  void check_ranks(const std::vector<cudf::size_type>& ranks) const
  {
    const auto got = corank(ranks);
    REQUIRE(got.size() == ranks.size() * k());
    for (std::size_t r = 0; r < ranks.size(); ++r) {
      const std::vector<cudf::size_type> counts(
        got.begin() + static_cast<std::ptrdiff_t>(r * k()),
        got.begin() + static_cast<std::ptrdiff_t>((r + 1) * k()));
      INFO("rank " << ranks[r] << " of " << _total << ", k = " << k());
      CHECK(counts == expected_counts(ranks[r]));
    }
  }

  /// The ranks every test splits at: 0, 1, mid, Σ − 1, Σ (those that exist), plus random ones.
  [[nodiscard]] std::vector<cudf::size_type> interesting_ranks(std::mt19937& rng,
                                                               int random_ranks = 12) const
  {
    std::vector<cudf::size_type> ranks = {0};
    const auto total                   = static_cast<cudf::size_type>(_total);
    if (total >= 1) {
      ranks.push_back(1);
      ranks.push_back(total / 2);
      ranks.push_back(total - 1);
      ranks.push_back(total);
      std::uniform_int_distribution<cudf::size_type> any_rank(0, total);
      for (int i = 0; i < random_ranks; ++i) {
        ranks.push_back(any_rank(rng));
      }
    }
    return ranks;
  }

 private:
  /// Three-way comparison of two keys under the fixture's order and null order.
  [[nodiscard]] int compare(const host_key& a, const host_key& b) const
  {
    int cmp = 0;
    if (!a.valid || !b.valid) {
      if (!a.valid && !b.valid) { return 0; }
      const bool null_is_less = _null_order == cudf::null_order::BEFORE;
      cmp                     = (!a.valid == null_is_less) ? -1 : 1;
    } else if (a.value != b.value) {
      cmp = a.value < b.value ? -1 : 1;
    } else {
      const int text_cmp = a.text.compare(b.text);
      cmp                = static_cast<int>(text_cmp > 0) - static_cast<int>(text_cmp < 0);
    }
    return _order == cudf::order::ASCENDING ? cmp : -cmp;
  }

  static std::unique_ptr<cudf::column> make_column(const host_window& window, bool strings)
  {
    if (strings) {
      if (window.empty()) { return cudf::make_empty_column(cudf::type_id::STRING); }
      std::vector<std::string> values;
      for (const auto& key : window) {
        values.push_back(key.text);
      }
      return merge_utils::make_string_column(values);
    }
    std::vector<int64_t> values;
    std::vector<bool> valids;
    bool has_nulls = false;
    for (const auto& key : window) {
      values.push_back(key.value);
      valids.push_back(key.valid);
      has_nulls = has_nulls || !key.valid;
    }
    return merge_utils::make_column(
      values, cudf::type_id::INT64, has_nulls ? valids : std::vector<bool>{});
  }

  template <typename T>
  static rmm::device_uvector<T> upload(const std::vector<T>& host)
  {
    rmm::device_uvector<T> out(
      host.size(), cudf::get_default_stream(), cudf::get_current_device_resource_ref());
    if (!host.empty()) {
      REQUIRE(
        cudaMemcpy(out.data(), host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice) ==
        cudaSuccess);
    }
    return out;
  }

  std::vector<host_window> _windows;
  cudf::order _order;
  cudf::null_order _null_order;
  std::vector<std::unique_ptr<cudf::table>> _tables;
  std::vector<cudf::table_view> _views;
  std::vector<cudf::size_type> _lengths;
  int64_t _total{0};
  std::optional<kway_key_comparator> _cmp;
  std::vector<std::pair<int32_t, cudf::size_type>> _merged;
};

/// INT64 keys drawn from @p domain values, so equal keys are frequent.
host_window random_int_window(cudf::size_type rows,
                              int64_t domain,
                              double null_probability,
                              std::mt19937& rng)
{
  std::uniform_int_distribution<int64_t> value(0, domain - 1);
  std::bernoulli_distribution is_null(null_probability);
  host_window window;
  for (cudf::size_type r = 0; r < rows; ++r) {
    host_key key;
    key.valid = !is_null(rng);
    key.value = value(rng);
    window.push_back(std::move(key));
  }
  return window;
}

/// STRING keys from a small set that includes the empty string and shared prefixes.
host_window random_string_window(cudf::size_type rows, std::mt19937& rng)
{
  static const std::vector<std::string> domain = {
    "", "a", "ab", "abc", "b", "ba", "a longer string value", "a longer string"};
  host_window window;
  for (cudf::size_type r = 0; r < rows; ++r) {
    host_key key;
    key.text = domain[rng() % domain.size()];
    window.push_back(std::move(key));
  }
  return window;
}

/// Window lengths for k inputs: random up to @p max_rows, with empty windows mixed in.
std::vector<cudf::size_type> random_lengths(std::size_t k,
                                            cudf::size_type max_rows,
                                            std::mt19937& rng)
{
  std::uniform_int_distribution<cudf::size_type> length(1, max_rows);
  std::vector<cudf::size_type> lengths;
  for (std::size_t i = 0; i < k; ++i) {
    // Every fifth window is empty (not the first, so that k = 1 has rows).
    lengths.push_back(i % 5 == 3 ? 0 : length(rng));
  }
  return lengths;
}

std::vector<host_window> random_int_windows(const std::vector<cudf::size_type>& lengths,
                                            int64_t domain,
                                            double null_probability,
                                            std::mt19937& rng)
{
  std::vector<host_window> windows;
  for (const auto rows : lengths) {
    windows.push_back(random_int_window(rows, domain, null_probability, rng));
  }
  return windows;
}

}  // namespace

TEST_CASE("corank_split matches the merged prefix for random INT64 windows", "[kway_corank]")
{
  const auto k = GENERATE(std::size_t{1}, std::size_t{2}, std::size_t{7}, std::size_t{64});
  // A domain of 6 values makes almost every comparison a tie on the key; the wide domain has
  // few duplicates.
  const auto domain = GENERATE(int64_t{6}, int64_t{100000});

  std::mt19937 rng(static_cast<unsigned>(17 * k + static_cast<std::size_t>(domain)));
  const corank_fixture fixture(random_int_windows(random_lengths(k, 400, rng), domain, 0.0, rng),
                               false);
  fixture.check_ranks(fixture.interesting_ranks(rng));
}

TEST_CASE("corank_split handles windows of very different lengths", "[kway_corank]")
{
  std::mt19937 rng(4242);
  const std::vector<cudf::size_type> lengths = {20000, 0, 1, 3, 9000, 0, 150, 1};
  const corank_fixture fixture(random_int_windows(lengths, 50, 0.0, rng), false);
  fixture.check_ranks(fixture.interesting_ranks(rng, 40));
}

TEST_CASE("corank_split matches the merged prefix for STRING windows", "[kway_corank]")
{
  std::mt19937 rng(77);
  std::vector<host_window> windows;
  for (const auto rows : random_lengths(7, 300, rng)) {
    windows.push_back(random_string_window(rows, rng));
  }
  const auto order = GENERATE(cudf::order::ASCENDING, cudf::order::DESCENDING);
  const corank_fixture fixture(std::move(windows), true, order);
  fixture.check_ranks(fixture.interesting_ranks(rng));
}

TEST_CASE("corank_split takes equal keys in input order", "[kway_corank]")
{
  // Every key is 5, so the merged order is window 0, then window 1, ...
  const std::vector<cudf::size_type> lengths = {4, 0, 3, 1, 0, 5};
  std::vector<host_window> windows;
  for (const auto rows : lengths) {
    host_key five;
    five.value = 5;
    windows.emplace_back(static_cast<std::size_t>(rows), five);
  }
  const corank_fixture fixture(std::move(windows), false);

  std::vector<cudf::size_type> ranks;
  for (cudf::size_type r = 0; r <= static_cast<cudf::size_type>(fixture.total()); ++r) {
    ranks.push_back(r);
  }
  const auto got = fixture.corank(ranks);
  for (std::size_t r = 0; r < ranks.size(); ++r) {
    // The first `rank` rows fill the windows front to back.
    cudf::size_type remaining = ranks[r];
    std::vector<cudf::size_type> expected;
    for (const auto rows : lengths) {
      expected.push_back(std::min(rows, remaining));
      remaining -= expected.back();
    }
    const std::vector<cudf::size_type> counts(
      got.begin() + static_cast<std::ptrdiff_t>(r * lengths.size()),
      got.begin() + static_cast<std::ptrdiff_t>((r + 1) * lengths.size()));
    INFO("rank " << ranks[r]);
    CHECK(counts == expected);
  }
  fixture.check_ranks(ranks);
}

TEST_CASE("corank_split follows descending order and null precedence", "[kway_corank]")
{
  const auto order      = GENERATE(cudf::order::ASCENDING, cudf::order::DESCENDING);
  const auto null_order = GENERATE(cudf::null_order::BEFORE, cudf::null_order::AFTER);
  const auto k          = GENERATE(std::size_t{3}, std::size_t{40});

  std::mt19937 rng(static_cast<unsigned>(900 + k + static_cast<unsigned>(order) * 2 +
                                         static_cast<unsigned>(null_order)));
  // A fifth of the keys are NULL; NULLs tie with each other across windows.
  const corank_fixture fixture(
    random_int_windows(random_lengths(k, 200, rng), 8, 0.2, rng), false, order, null_order);
  fixture.check_ranks(fixture.interesting_ranks(rng));
}

TEST_CASE("corank_split accepts no ranks and rejects mismatched spans", "[kway_corank]")
{
  std::mt19937 rng(5);
  const corank_fixture fixture(random_int_windows({10, 20, 30}, 4, 0.0, rng), false);
  CHECK(fixture.corank({}).empty());

  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();
  std::vector<cudf::table_view> views;
  auto table = merge_utils::make_table(
    merge_utils::make_column(std::vector<int64_t>{1, 2, 3}, cudf::type_id::INT64));
  views.push_back(table->view());
  views.push_back(table->view());
  const std::vector<cudf::order> orders           = {cudf::order::ASCENDING};
  const std::vector<cudf::null_order> null_orders = {cudf::null_order::AFTER};
  auto cmp = kway_key_comparator::create(views, orders, null_orders, stream, mr);

  rmm::device_uvector<cudf::size_type> lengths(2, stream, mr);
  rmm::device_uvector<cudf::size_type> ranks(1, stream, mr);
  rmm::device_uvector<cudf::size_type> out(2, stream, mr);
  rmm::device_uvector<uint8_t> truncated(2, stream, mr);
  const cudf::device_span<const cudf::size_type> lengths_span(lengths.data(), lengths.size());
  const cudf::device_span<const cudf::size_type> short_lengths(lengths.data(), 1);
  const cudf::device_span<const cudf::size_type> ranks_span(ranks.data(), ranks.size());
  const cudf::device_span<const uint8_t> truncated_span(truncated.data(), truncated.size());

  // lengths must have k entries. corank_split sizes out from lengths (ranks * lengths.size()), so
  // the 1-entry out matches the short lengths and only the lengths check can throw here.
  CHECK_THROWS_AS(
    corank_split(
      cmp, short_lengths, ranks_span, cudf::device_span<cudf::size_type>(out.data(), 1), stream),
    std::invalid_argument);
  // out must have ranks * k entries.
  CHECK_THROWS_AS(
    corank_split(
      cmp, lengths_span, ranks_span, cudf::device_span<cudf::size_type>(out.data(), 1), stream),
    std::invalid_argument);
  CHECK_THROWS_AS(safe_target(cmp, short_lengths, truncated_span, 10, stream),
                  std::invalid_argument);
  CHECK_THROWS_AS(
    safe_target(
      cmp, lengths_span, cudf::device_span<const uint8_t>(truncated.data(), 1), 10, stream),
    std::invalid_argument);
}

namespace {

host_window int_window(const std::vector<int64_t>& values)
{
  host_window window;
  for (const auto value : values) {
    host_key key;
    key.value = value;
    window.push_back(std::move(key));
  }
  return window;
}

}  // namespace

TEST_CASE("safe_target without truncated windows is bounded by max_rows and the rows",
          "[kway_corank]")
{
  std::mt19937 rng(11);
  const corank_fixture fixture(random_int_windows({30, 0, 12, 25}, 5, 0.0, rng), false);
  const std::vector<uint8_t> none(fixture.k(), 0);
  REQUIRE(fixture.total() == 67);
  CHECK(fixture.target(none, 1000) == 67);
  CHECK(fixture.target(none, 67) == 67);
  CHECK(fixture.target(none, 40) == 40);
  CHECK(fixture.target(none, 0) == 0);
}

TEST_CASE("safe_target stops after the last row of a short truncated window", "[kway_corank]")
{
  // Merged order (key:window): 0:1 1:0 2:0 2:1 2:1 3:0 5:1 9:1 9:1 9:2 9:2
  const corank_fixture fixture(
    {int_window({1, 2, 3}), int_window({0, 2, 2, 5, 9, 9}), int_window({9, 9})}, false);

  SECTION("the window's last key is small")
  {
    // Row (0, 2) holds key 3: two rows of window 0 and three of window 1 precede it.
    CHECK(fixture.target({1, 0, 0}, 1000) == 6);
    CHECK(fixture.target({1, 0, 0}, 4) == 4);
  }

  SECTION("equal keys of a lower input order before the window's last row")
  {
    // Row (2, 1) is the last row of the merged order: the 9s of window 1 precede it.
    CHECK(fixture.target({0, 0, 1}, 1000) == 11);
    // Row (1, 5) holds key 9: the 9s of window 2 follow it.
    CHECK(fixture.target({0, 1, 0}, 1000) == 9);
  }

  SECTION("the smallest bound of several truncated windows wins")
  {
    CHECK(fixture.target({1, 1, 1}, 1000) == 6);
    CHECK(fixture.target({0, 1, 1}, 1000) == 9);
  }
}

TEST_CASE("safe_target is zero when a truncated window has no rows", "[kway_corank]")
{
  const corank_fixture fixture({int_window({1, 2, 3}), int_window({}), int_window({4, 5})}, false);
  CHECK(fixture.target({0, 1, 0}, 1000) == 0);
  CHECK(fixture.target({1, 1, 1}, 1000) == 0);
  // An empty window that is not truncated (an exhausted run) does not limit the target.
  CHECK(fixture.target({0, 0, 0}, 1000) == 5);
  CHECK(fixture.target({0, 0, 1}, 1000) == 5);
}

TEST_CASE("safe_target matches the host reference for random windows", "[kway_corank]")
{
  const auto strings = GENERATE(false, true);
  const auto order   = GENERATE(cudf::order::ASCENDING, cudf::order::DESCENDING);
  std::mt19937 rng(static_cast<unsigned>(300 + (strings ? 10 : 0) + static_cast<unsigned>(order)));

  std::vector<host_window> windows;
  for (const auto rows : random_lengths(9, 120, rng)) {
    // Non-empty windows only: a truncated empty window is covered above.
    const auto non_empty = std::max<cudf::size_type>(rows, 2);
    windows.push_back(strings ? random_string_window(non_empty, rng)
                              : random_int_window(non_empty, 6, 0.1, rng));
  }
  const corank_fixture fixture(std::move(windows), strings, order, cudf::null_order::BEFORE);

  for (int round = 0; round < 20; ++round) {
    std::vector<uint8_t> truncated;
    for (std::size_t i = 0; i < fixture.k(); ++i) {
      truncated.push_back(rng() % 3 == 0 ? 1 : 0);
    }
    const auto max_rows = static_cast<cudf::size_type>(rng() % 1200);
    INFO("round " << round << ", max_rows " << max_rows);
    CHECK(fixture.target(truncated, max_rows) == fixture.expected_target(truncated, max_rows));
  }
}

TEST_CASE("safe_target matches the host reference across fan-ins", "[kway_corank]")
{
  // k = 2 leaves most lanes of a warp without an input, 64 gives every lane two and 256 eight.
  const auto k = GENERATE(std::size_t{2}, std::size_t{64}, std::size_t{256});
  std::mt19937 rng(static_cast<unsigned>(900 + k));

  // Windows without rows are included; a truncated one forces the target to 0.
  const corank_fixture fixture(random_int_windows(random_lengths(k, 40, rng), 50, 0.1, rng), false);

  for (int round = 0; round < 12; ++round) {
    // Few truncated windows in the first rounds, so the bound is not always the same small one.
    const unsigned one_in = round < 6 ? 40 : 3;
    std::vector<uint8_t> truncated;
    for (std::size_t i = 0; i < fixture.k(); ++i) {
      // The first rounds only truncate windows with rows; the others also hit the empty ones.
      const bool allowed = round >= 9 || fixture.lengths()[i] > 0;
      truncated.push_back(allowed && rng() % one_in == 0 ? 1 : 0);
    }
    const auto max_rows = static_cast<cudf::size_type>(rng() % (2 * fixture.total() + 1));
    INFO("k " << k << ", round " << round << ", max_rows " << max_rows);
    CHECK(fixture.target(truncated, max_rows) == fixture.expected_target(truncated, max_rows));
  }
}
