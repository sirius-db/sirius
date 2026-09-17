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

// [late_mat][origin] — the origin metadata, on the CPU. No GPU required.
//
// The property worth testing here is that a stale origin fails CLOSED. A
// deferred column outlives the operator that deferred it, so by the time it
// resolves, its pinned entry may have been unpinned, replaced or merged. Every
// one of those has a wrong answer available to it — a dangling pointer, or a
// name lookup that finds a different entry with the same name — and none of
// those wrong answers looks like a failure at the point it happens. So each
// lifecycle transition gets a case, and each requires nullptr rather than
// merely "not the old pointer".

#include <catch.hpp>
#include <late_mat/column_origin.hpp>

#include <cstdlib>
#include <memory>
#include <string>

using sirius::late_mat::column_origin;
using sirius::late_mat::pin_entry_handle;
using sirius::late_mat::row_range;
using sirius::late_mat::row_selection;
using sirius::late_mat::row_selection_kind;

namespace {

// The handle only ever stores and compares this pointer, so a distinct address
// is all a test needs; constructing a real pinned_entry would drag in the whole
// scan manager to prove nothing extra.
sirius::scan_manager::pinned_entry const* fake_entry()
{
  static int marker = 0;
  return reinterpret_cast<sirius::scan_manager::pinned_entry const*>(&marker);
}

column_origin origin_against(std::shared_ptr<pin_entry_handle> const& handle,
                             std::uint32_t column_pos = 0)
{
  column_origin o;
  o.handle     = handle;
  o.column_pos = column_pos;
  o.generation = handle->generation();
  return o;
}

}  // namespace

TEST_CASE("a live origin resolves to its entry", "[late_mat][origin]")
{
  auto handle = std::make_shared<pin_entry_handle>("lineitem", 7);
  handle->set_entry(fake_entry());

  auto const origin = origin_against(handle, 3);
  REQUIRE(origin.has_origin());
  REQUIRE(origin.resolve() == fake_entry());
  REQUIRE(origin.column_pos == 3);
}

TEST_CASE("a default-constructed origin fails closed", "[late_mat][origin]")
{
  // Generation 0 is the invalidated value precisely so that a zero-initialized
  // origin cannot resolve — the failure mode of a forgotten field.
  column_origin const origin;
  REQUIRE_FALSE(origin.has_origin());
  REQUIRE(origin.resolve() == nullptr);
}

TEST_CASE("an origin captured before an unpin fails closed", "[late_mat][origin]")
{
  auto handle = std::make_shared<pin_entry_handle>("lineitem", 7);
  handle->set_entry(fake_entry());
  auto const origin = origin_against(handle);
  REQUIRE(origin.resolve() != nullptr);

  handle->invalidate();
  REQUIRE(origin.resolve() == nullptr);
  REQUIRE(handle->generation() == 0);
}

TEST_CASE("an origin captured before an in-place merge fails closed", "[late_mat][origin]")
{
  auto handle = std::make_shared<pin_entry_handle>("lineitem", 7);
  handle->set_entry(fake_entry());
  auto const before = origin_against(handle);

  handle->bump_generation(8);
  REQUIRE(before.resolve() == nullptr);

  // An origin captured after the merge is live against the same entry, which
  // is what distinguishes a bump from an invalidate.
  auto const after = origin_against(handle);
  REQUIRE(after.resolve() == fake_entry());
}

TEST_CASE("a zero generation never resolves, even against a live handle", "[late_mat][origin]")
{
  auto handle = std::make_shared<pin_entry_handle>("lineitem", 0);
  handle->set_entry(fake_entry());

  column_origin origin;
  origin.handle     = handle;
  origin.generation = 0;
  REQUIRE(origin.resolve() == nullptr);
}

TEST_CASE("a row range spans exactly its own rows", "[late_mat][origin]")
{
  row_range const r{1000, 250};
  REQUIRE(r.end() == 1250);
  REQUIRE(r.contains(1000));
  REQUIRE(r.contains(1249));
  REQUIRE_FALSE(r.contains(999));
  REQUIRE_FALSE(r.contains(1250));
}

TEST_CASE("live_rows follows the selection's own form", "[late_mat][origin]")
{
  row_range const range{4096, 3000};

  auto const dense = row_selection::make_dense(range);
  REQUIRE(dense.kind == row_selection_kind::dense);
  REQUIRE(dense.live_rows() == 3000);

  row_selection masked;
  masked.kind           = row_selection_kind::mask;
  masked.range          = range;
  masked.survivor_count = 42;
  REQUIRE(masked.live_rows() == 42);

  row_selection listed;
  listed.kind    = row_selection_kind::id_list;
  listed.range   = range;
  listed.num_ids = 17;
  REQUIRE(listed.live_rows() == 17);
}

TEST_CASE("the gate reads its environment once and the same way", "[late_mat][origin]")
{
  // Whatever the environment says, every caller must agree — the point of the
  // single reader is that two of them cannot disagree.
  bool const enabled = sirius::late_mat::late_mat_enabled();
  REQUIRE(sirius::late_mat::late_mat_enabled() == enabled);
}
