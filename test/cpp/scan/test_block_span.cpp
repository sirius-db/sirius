/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "op/scan/block_span.hpp"

#include <catch.hpp>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <vector>

using sirius::op::scan::block_span;

namespace {
/// The values 0..count-1 laid out over blocks of @p per_block values that are neither adjacent
/// nor in address order.
struct scattered_values {
  scattered_values(std::size_t count, std::size_t per_block) : storage(2 * (count + per_block), -1)
  {
    // Every other block-sized slot, highest address first.
    for (auto i = (count + per_block - 1) / per_block; i-- > 0;)
      table.push_back(reinterpret_cast<std::byte*>(storage.data() + 2 * i * per_block));
    view = {table, per_block * sizeof(int64_t), count};
    std::iota(view.begin(), view.end(), int64_t{0});
  }
  std::vector<int64_t> storage;
  std::vector<std::byte*> table;
  block_span<int64_t> view;
};
}  // namespace

TEST_CASE("block_span reads blocks that are not adjacent as one ordered range",
          "[scan][block_span]")
{
  // 10 over 4 leaves a partial last block, 12 over 4 ends on a block boundary, 1 is one value
  // per block.
  auto const count     = GENERATE(std::size_t{0}, std::size_t{1}, std::size_t{10}, std::size_t{12});
  auto const per_block = GENERATE(std::size_t{1}, std::size_t{4});
  CAPTURE(count, per_block);
  scattered_values values(count, per_block);
  block_span<int64_t const> const view = values.view;
  std::vector<int64_t> expected(count);
  std::iota(expected.begin(), expected.end(), int64_t{0});

  REQUIRE(view.size() == count);
  CHECK(view.empty() == (count == 0));
  CHECK(std::vector<int64_t>(view.begin(), view.end()) == expected);
  CHECK(std::vector<int64_t>(std::make_reverse_iterator(view.end()),
                             std::make_reverse_iterator(view.begin())) ==
        std::vector<int64_t>(expected.rbegin(), expected.rend()));
  for (std::size_t i = 0; i < count; ++i) {
    auto const at = view.begin() + static_cast<std::ptrdiff_t>(i);
    CHECK(view[i] == expected[i]);
    CHECK(*at == expected[i]);
    CHECK(view.end() - at == static_cast<std::ptrdiff_t>(count - i));
    CHECK(*(view.end() - static_cast<std::ptrdiff_t>(count - i)) == expected[i]);
    CHECK(std::lower_bound(view.begin(), view.end(), expected[i]) == at);
    CHECK(view.run(i).size() == std::min(per_block - i % per_block, count - i));
    CHECK(view.run(i).front() == expected[i]);
  }
  CHECK(std::lower_bound(view.begin(), view.end(), static_cast<int64_t>(count)) == view.end());
  CHECK(view.run(count).empty());
}

TEST_CASE("block_span orders and deduplicates values across block boundaries", "[scan][block_span]")
{
  scattered_values values(11, 4);
  auto const view = values.view;
  for (std::size_t i = 0; i < view.size(); ++i)
    view[i] = static_cast<int64_t>((view.size() - i) / 2);
  std::sort(view.begin(), view.end());
  CHECK(std::is_sorted(view.begin(), view.end()));
  auto const unique = std::unique(view.begin(), view.end()) - view.begin();
  CHECK(std::vector<int64_t>(view.begin(), view.begin() + unique) ==
        std::vector<int64_t>{0, 1, 2, 3, 4, 5});
}

TEST_CASE("block_span over one buffer is the span it came from", "[scan][block_span]")
{
  std::vector<int64_t> const values{3, 5, 8};
  block_span<int64_t const> const contiguous = values;
  CHECK(&contiguous.front() == values.data());
  CHECK(contiguous.run(1).size() == 2);
  CHECK(std::vector<int64_t>(contiguous.begin(), contiguous.end()) == values);

  auto* block = reinterpret_cast<std::byte*>(const_cast<int64_t*>(values.data()));
  block_span<int64_t const> const one_block({&block, 1}, 3 * sizeof(int64_t), 3);
  CHECK(std::vector<int64_t>(one_block.begin(), one_block.end()) == values);
  CHECK(block_span<int64_t const>{}.begin() == block_span<int64_t const>{}.end());
}

TEST_CASE("block_span never splits an element across two blocks", "[scan][block_span]")
{
  // 12-byte blocks hold one int64_t each, so two of them hold two values, not three.
  std::vector<int64_t> storage(8);
  std::vector<std::byte*> table{reinterpret_cast<std::byte*>(storage.data() + 4),
                                reinterpret_cast<std::byte*>(storage.data())};
  CHECK_THROWS_AS(block_span<int64_t>(table, 12, 3), std::invalid_argument);
  block_span<int64_t> const view(table, 12, 2);
  view[0] = 7;
  view[1] = 9;
  CHECK(storage[4] == 7);
  CHECK(storage[0] == 9);
}
