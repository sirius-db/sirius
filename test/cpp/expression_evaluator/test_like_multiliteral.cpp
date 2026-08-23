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

// Tests for the multi-literal LIKE SWAR fast path (expression_evaluator/like_multiliteral.hpp):
//   1. the pattern classifier — exactly `%lit1%lit2%...%litN%` shapes are accepted, everything
//      else (anchored patterns, `_`, escapes, non-ASCII, over-caps) is refused;
//   2. the query cache — concurrent lookups share one immutable value-keyed classification;
//   3. the kernel — differential against cudf::strings::like on adversarial fixed rows (word
//      alignment sweeps, literal-at-row-edges, greedy/overlap traps, UTF-8 neighbours, nulls,
//      sliced views) and on seeded random near-miss data;
//   4. the host launcher contract — caps-violating patterns that bypass the classifier throw
//      sirius::internal_exception instead of silently falling back.

#include "catch.hpp"

// sirius
#include <expression_evaluator/like_multiliteral.hpp>
#include <sirius/exception.hpp>

// cudf
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

// rmm
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

// standard library
#include <cstdint>
#include <future>
#include <random>
#include <string>
#include <vector>

using sirius::classify_like_multiliteral;
using sirius::like_multiliteral;

namespace {

/// Builds a STRING column from host rows; valids empty = all valid (null rows contribute
/// 0 bytes). int64_offsets selects the large-strings layout (INT64 offsets child).
std::unique_ptr<cudf::column> make_strings(std::vector<std::string> const& rows,
                                           std::vector<bool> const& valids = {},
                                           bool int64_offsets              = false)
{
  auto stream  = cudf::get_default_stream();
  auto mr      = cudf::get_current_device_resource_ref();
  auto const n = static_cast<cudf::size_type>(rows.size());

  std::string chars;
  std::vector<int64_t> offsets(rows.size() + 1, 0);
  for (size_t i = 0; i < rows.size(); ++i) {
    if (valids.empty() || valids[i]) { chars += rows[i]; }
    offsets[i + 1] = static_cast<int64_t>(chars.size());
  }

  rmm::device_buffer chars_buf(chars.data(), chars.size(), stream, mr);
  std::unique_ptr<cudf::column> offsets_col;
  if (int64_offsets) {
    offsets_col = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT64}, n + 1, cudf::mask_state::UNALLOCATED, stream, mr);
    cudaMemcpyAsync(offsets_col->mutable_view().data<int64_t>(),
                    offsets.data(),
                    offsets.size() * sizeof(int64_t),
                    cudaMemcpyHostToDevice,
                    stream.value());
  } else {
    std::vector<int32_t> offsets32(offsets.begin(), offsets.end());
    offsets_col = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32}, n + 1, cudf::mask_state::UNALLOCATED, stream, mr);
    cudaMemcpyAsync(offsets_col->mutable_view().data<int32_t>(),
                    offsets32.data(),
                    offsets32.size() * sizeof(int32_t),
                    cudaMemcpyHostToDevice,
                    stream.value());
    stream.synchronize();  // offsets32 goes out of scope here
  }

  rmm::device_buffer mask_buf{};
  cudf::size_type null_count = 0;
  if (!valids.empty()) {
    // cudf null masks must be padded to bitmask_allocation_size_bytes (64B multiples).
    std::vector<cudf::bitmask_type> mask_words(
      cudf::bitmask_allocation_size_bytes(n) / sizeof(cudf::bitmask_type), 0);
    for (cudf::size_type i = 0; i < n; ++i) {
      if (valids[i]) {
        mask_words[i / 32] |= (cudf::bitmask_type{1} << (i % 32));
      } else {
        ++null_count;
      }
    }
    mask_buf = rmm::device_buffer(
      mask_words.data(), mask_words.size() * sizeof(cudf::bitmask_type), stream, mr);
  }

  auto col = cudf::make_strings_column(
    n, std::move(offsets_col), std::move(chars_buf), null_count, std::move(mask_buf));
  stream.synchronize();  // host vectors go out of scope
  return col;
}

std::vector<uint8_t> to_host_values(cudf::column_view const& col)
{
  std::vector<uint8_t> out(col.size());
  cudaMemcpy(out.data(), col.data<bool>(), col.size(), cudaMemcpyDeviceToHost);
  return out;
}

std::vector<bool> to_host_validity(cudf::column_view const& col)
{
  std::vector<bool> out(col.size(), true);
  if (col.null_mask() == nullptr) { return out; }
  std::vector<cudf::bitmask_type> words(cudf::num_bitmask_words(col.size() + col.offset()));
  cudaMemcpy(words.data(),
             col.null_mask(),
             words.size() * sizeof(cudf::bitmask_type),
             cudaMemcpyDeviceToHost);
  for (cudf::size_type i = 0; i < col.size(); ++i) {
    out[i] = cudf::bit_is_set(words.data(), col.offset() + i);
  }
  return out;
}

/// Runs the fast path and cudf::strings::like on @p view and requires identical results
/// (values on valid rows, validity everywhere) for both LIKE and NOT LIKE.
void require_matches_cudf(cudf::column_view const& view, std::string const& pattern)
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();
  auto scv    = cudf::strings_column_view(view);

  auto const parsed = classify_like_multiliteral(pattern, {});
  REQUIRE(parsed.has_value());

  auto ref             = cudf::strings::like(scv, pattern, std::string_view(), stream, mr);
  auto const ref_vals  = to_host_values(ref->view());
  auto const ref_valid = to_host_validity(ref->view());

  for (bool const invert : {false, true}) {
    auto fast = like_multiliteral(scv, *parsed, invert, stream, mr);
    REQUIRE(fast != nullptr);
    REQUIRE(fast->size() == view.size());
    auto const fast_vals  = to_host_values(fast->view());
    auto const fast_valid = to_host_validity(fast->view());

    int64_t value_mismatches = 0, validity_mismatches = 0;
    for (cudf::size_type i = 0; i < view.size(); ++i) {
      if (fast_valid[i] != ref_valid[i]) { ++validity_mismatches; }
      if (ref_valid[i] &&
          (fast_vals[i] != 0) != (invert ? !(ref_vals[i] != 0) : ref_vals[i] != 0)) {
        ++value_mismatches;
      }
    }
    INFO("pattern=" << pattern << " invert=" << invert);
    REQUIRE(validity_mismatches == 0);
    REQUIRE(value_mismatches == 0);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Classifier
// ---------------------------------------------------------------------------

TEST_CASE("like_multiliteral classifier accepts %lit...% shapes",
          "[expression_evaluator][like_multiliteral]")
{
  auto p = classify_like_multiliteral("%special%requests%", {});
  REQUIRE(p.has_value());
  REQUIRE(p->literals() == std::vector<std::string>{"special", "requests"});

  p = classify_like_multiliteral("%abc%", {});
  REQUIRE(p.has_value());
  REQUIRE(p->literals() == std::vector<std::string>{"abc"});

  // Consecutive '%' collapse per SQL semantics.
  p = classify_like_multiliteral("%%a%%%b%%", {});
  REQUIRE(p.has_value());
  REQUIRE(p->literals() == std::vector<std::string>{"a", "b"});

  // Max caps: 4 literals, 32 bytes each.
  p = classify_like_multiliteral("%a%b%c%d%", {});
  REQUIRE(p.has_value());
  REQUIRE(p->literals().size() == 4);
  std::string const lit32(32, 'x');
  p = classify_like_multiliteral("%" + lit32 + "%", {});
  REQUIRE(p.has_value());
  REQUIRE(p->literals() == std::vector<std::string>{lit32});

  // Punctuation and spaces are plain ASCII literal bytes.
  p = classify_like_multiliteral("% a.b-c! %", {});
  REQUIRE(p.has_value());
  REQUIRE(p->literals() == std::vector<std::string>{" a.b-c! "});
}

TEST_CASE("like_multiliteral classifier refuses everything else",
          "[expression_evaluator][like_multiliteral]")
{
  // Anchored / non-%-delimited shapes (q16's 'PROMO POLISHED%' shape included).
  CHECK_FALSE(classify_like_multiliteral("PROMO POLISHED%", {}).has_value());
  CHECK_FALSE(classify_like_multiliteral("%MEDIUM POLISHED", {}).has_value());
  CHECK_FALSE(classify_like_multiliteral("exact", {}).has_value());
  CHECK_FALSE(classify_like_multiliteral("", {}).has_value());
  CHECK_FALSE(classify_like_multiliteral("%", {}).has_value());
  CHECK_FALSE(classify_like_multiliteral("%%", {}).has_value());

  // '_' wildcard anywhere.
  CHECK_FALSE(classify_like_multiliteral("%a_b%", {}).has_value());
  CHECK_FALSE(classify_like_multiliteral("%_%", {}).has_value());

  // Escape character in play.
  CHECK_FALSE(classify_like_multiliteral("%a%", "\\").has_value());

  // Non-ASCII or NUL literal bytes.
  CHECK_FALSE(classify_like_multiliteral("%calf\xC3\xA9%", {}).has_value());
  CHECK_FALSE(classify_like_multiliteral(std::string_view("%a\0b%", 5), {}).has_value());

  // Over caps: 5 literals / 33-byte literal.
  CHECK_FALSE(classify_like_multiliteral("%a%b%c%d%e%", {}).has_value());
  CHECK_FALSE(classify_like_multiliteral("%" + std::string(33, 'x') + "%", {}).has_value());
}

TEST_CASE("like_multiliteral cache classifies a pattern once under concurrency",
          "[expression_evaluator][like_multiliteral][cache]")
{
  auto cache = std::make_shared<sirius::like_multiliteral_cache>();
  std::vector<std::future<sirius::like_multiliteral_cache::entry_ptr>> lookups;
  for (int i = 0; i < 8; ++i) {
    lookups.push_back(std::async(std::launch::async,
                                 [cache] { return cache->get_or_classify("%special%requests%"); }));
  }

  auto const first = lookups.front().get();
  REQUIRE(first->has_value());
  for (std::size_t i = 1; i < lookups.size(); ++i) {
    auto const entry = lookups[i].get();
    REQUIRE(entry.get() == first.get());
  }
  REQUIRE(cache->classification_count_for_testing() == 1);
}

// ---------------------------------------------------------------------------
// Kernel: differential vs cudf::strings::like
// ---------------------------------------------------------------------------

TEST_CASE("like_multiliteral matches cudf on adversarial fixed rows",
          "[expression_evaluator][like_multiliteral]")
{
  std::vector<std::string> rows = {
    "",                        // empty row
    "special",                 // first literal only
    "requests",                // second literal only
    "specialrequests",         // adjacent literals ('%' matches empty)
    "requestsspecial",         // wrong order
    "specialrequest",          // truncated second literal
    "specia",                  // truncated first literal
    "pecialrequests",          // truncated head
    "special requests",        //
    "specialspecialrequests",  // repeated first literal
    "specialrequestsrequests",
    "special special requests",
    "sspecialrrequests",  // candidate-dense near-misses
    "spspspspecialrererequests",
    "xxrequestsxxspecialxxrequestsxx",  // second literal both before and after the first
    "specialxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxrequests",  // literals words apart
    "the special package with regular requests handling",
    "no keywords here at all",
    "s",
    "sp",
    "spe re sp req reque request specialx",
  };
  // Alignment sweep: every chars-buffer phase for literal start/end, row ends exactly at the
  // literal's last byte (verify-window edge) and digrams straddling word boundaries.
  for (int k = 0; k < 16; ++k) {
    rows.push_back(std::string(k, 'x') + "special" + std::string(k % 3, 'y') + "requests");
    rows.push_back(std::string(k, 's') + "pecial" + std::string(k, 'r') +
                   "equests");  // codespell:ignore equests
  }
  // Greedy/overlap traps for %aa%aa%.
  rows.insert(rows.end(), {"a", "aa", "aaa", "aaaa", "aaaaa", "aabaa", "aab", "baa", "aba"});
  // Valid UTF-8 neighbours around ASCII literals (byte-level == char-level for ASCII literals).
  rows.insert(rows.end(),
              {"\xC3\xA9 special \xC3\xBC requests \xE2\x9C\x93",
               "\xE3\x82\xB9\xE3\x83\x9A special \xE3\x83\xAA requests",
               "\xC3\xA9special\xC3\xA9requests\xC3\xA9",
               "calf\xC3\xA9"});
  // 32-byte literal present / absent / truncated.
  std::string const lit32 = "abcdefghijklmnopqrstuvwxyz012345";
  rows.insert(rows.end(), {lit32, "x" + lit32 + "x", lit32.substr(0, 31), "yy" + lit32});

  auto const col = make_strings(rows);

  std::vector<std::string> const patterns = {
    "%special%requests%",          // the q13 pattern
    "%special%",                   // single literal
    "%s%",                         // single 1-byte literal
    "%aa%aa%",                     // repeated-literal greedy semantics
    "%s%p%e%c%",                   // four 1-byte literals
    "%al requ%",                   // literal spanning the gap between the q13 words
    "%" + lit32 + "%",             // 32-byte literal (4 verify chunks)
    "%special package with%",      // 20-byte literal: partially-masked verify chunk at index 2
    "%special%requests%special%",  // literal count > occurrences in most rows
  };
  for (auto const& pattern : patterns) {
    require_matches_cudf(col->view(), pattern);
  }

  SECTION("null rows: mask copied, semantics identical")
  {
    std::vector<bool> valids(rows.size(), true);
    for (size_t i = 0; i < rows.size(); i += 3) {
      valids[i] = false;
    }
    auto const with_nulls = make_strings(rows, valids);
    for (auto const& pattern : patterns) {
      require_matches_cudf(with_nulls->view(), pattern);
    }
  }

  SECTION("sliced views: offsets/positions stay buffer-absolute")
  {
    auto const slices = cudf::slice(
      col->view(), {3, static_cast<cudf::size_type>(rows.size()) - 5}, cudf::get_default_stream());
    require_matches_cudf(slices.front(), "%special%requests%");
    require_matches_cudf(slices.front(), "%aa%aa%");

    // Sliced view WITH nulls pins the kernel's `mask_offset + row` null-mask read and the
    // copy_bitmask rebase of the sliced result mask. Detection property: the slice offset (4)
    // must not be a multiple of the null period (3), so an un-offset mask read yields a different
    // validity pattern; and each tested pattern needs a genuinely-valid MATCHING row at a slice
    // position r with r % 3 == 0, which such a read mis-classifies as null and flips from !invert
    // to invert. Witnesses: original row 10 "specialrequestsrequests" -> r = 6 for
    // %special%requests%, original row 58 "aabaa" -> r = 54 for %aa%aa%. Re-verify these
    // witnesses when editing the row list, the valids period, or the slice bounds.
    std::vector<bool> valids(rows.size(), true);
    for (size_t i = 0; i < rows.size(); i += 3) {
      valids[i] = false;
    }
    auto const with_nulls        = make_strings(rows, valids);
    auto const slices_with_nulls = cudf::slice(with_nulls->view(),
                                               {4, static_cast<cudf::size_type>(rows.size()) - 5},
                                               cudf::get_default_stream());
    require_matches_cudf(slices_with_nulls.front(), "%special%requests%");
    require_matches_cudf(slices_with_nulls.front(), "%aa%aa%");
  }
}

TEST_CASE("like_multiliteral matches cudf on seeded random near-miss data",
          "[expression_evaluator][like_multiliteral]")
{
  // Near-miss-heavy alphabet: mostly the literals' own bytes so digram candidates fire often.
  std::string const alphabet           = "specialrequst xyzSPECIALREQUST.,-";
  std::vector<std::string> const words = {
    "special", "requests", "sp", "re", "specia", "equests"};  // codespell:ignore equests

  std::mt19937 gen(20260817);
  std::uniform_int_distribution<int> len_dist(0, 120);
  std::uniform_int_distribution<size_t> alpha_dist(0, alphabet.size() - 1);
  std::uniform_int_distribution<size_t> word_dist(0, words.size() - 1);
  std::uniform_int_distribution<int> action_dist(0, 9);

  constexpr int num_rows = 50000;
  std::vector<std::string> rows;
  rows.reserve(num_rows);
  std::vector<bool> valids;
  valids.reserve(num_rows);
  for (int i = 0; i < num_rows; ++i) {
    std::string row;
    int const target = len_dist(gen);
    while (static_cast<int>(row.size()) < target) {
      if (action_dist(gen) == 0) {
        row += words[word_dist(gen)];  // inject whole (near-)literals
      } else {
        row += alphabet[alpha_dist(gen)];
      }
    }
    rows.push_back(std::move(row));
    valids.push_back(action_dist(gen) != 9);  // ~10% nulls
  }
  auto const col = make_strings(rows, valids);

  for (auto const& pattern : {"%special%requests%", "%sp%re%sp%re%", "%aa%aa%", "%e%"}) {
    require_matches_cudf(col->view(), pattern);
  }
}

// ---------------------------------------------------------------------------
// Kernel: column-layout edges (INT64 offsets, misaligned base, empty, NUL data)
// ---------------------------------------------------------------------------

TEST_CASE("like_multiliteral handles column-layout edges",
          "[expression_evaluator][like_multiliteral]")
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  SECTION("INT64 offsets (large-strings layout) execute the int64 kernel instantiation")
  {
    std::vector<std::string> rows = {
      "special requests", "requests special", "aaaa", "aaa", "", "just text"};
    std::vector<bool> valids = {true, true, true, false, true, true};
    auto const col           = make_strings(rows, valids, /*int64_offsets=*/true);
    REQUIRE(cudf::strings_column_view(col->view()).offsets().type().id() == cudf::type_id::INT64);
    require_matches_cudf(col->view(), "%special%requests%");
    require_matches_cudf(col->view(), "%aa%aa%");

    // A sliced view pins the `data<int64_t>() + input.offset()` offsets-pointer rebase in the
    // int64 launch branch: without it, slice row 0 ("requests special", no match) reads original
    // row 0's bounds ("special requests", match) and diverges from cudf.
    auto const slices = cudf::slice(col->view(), {1, 5}, stream);
    require_matches_cudf(slices.front(), "%special%requests%");
  }

  SECTION("tail words use bounded loads for every remainder and offset width")
  {
    std::string const literal = "abcdefghijklmnopqrst";
    for (bool const int64_offsets : {false, true}) {
      for (int remainder = 1; remainder < 8; ++remainder) {
        CAPTURE(int64_offsets, remainder);
        auto const col = make_strings(
          {std::string(remainder, 'x'), "zz" + literal, "miss", "yy" + literal}, {}, int64_offsets);
        auto const scv = cudf::strings_column_view(col->view());
        REQUIRE(scv.chars_size(stream) % 8 == remainder);
        require_matches_cudf(col->view(), "%" + literal + "%");

        auto const slice = cudf::slice(col->view(), {1, 3}, stream);
        require_matches_cudf(slice.front(), "%" + literal + "%");
      }
    }
  }

  SECTION("misaligned chars base returns nullptr and the cudf fallback still works")
  {
    // Hand-build a strings view whose chars base sits at buffer+1 (odd address).
    std::vector<std::string> const rows = {"xxabyy", "zzz", "ab", "ba"};
    std::string chars;
    std::vector<int32_t> offsets(rows.size() + 1, 0);
    for (size_t i = 0; i < rows.size(); ++i) {
      chars += rows[i];
      offsets[i + 1] = static_cast<int32_t>(chars.size());
    }
    auto const n = static_cast<cudf::size_type>(rows.size());
    rmm::device_buffer chars_buf(chars.size() + 8, stream, mr);
    cudaMemcpyAsync(static_cast<char*>(chars_buf.data()) + 1,
                    chars.data(),
                    chars.size(),
                    cudaMemcpyHostToDevice,
                    stream.value());
    auto offsets_col = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32}, n + 1, cudf::mask_state::UNALLOCATED, stream, mr);
    cudaMemcpyAsync(offsets_col->mutable_view().data<int32_t>(),
                    offsets.data(),
                    offsets.size() * sizeof(int32_t),
                    cudaMemcpyHostToDevice,
                    stream.value());
    stream.synchronize();

    cudf::column_view const strings_view(cudf::data_type{cudf::type_id::STRING},
                                         n,
                                         static_cast<char const*>(chars_buf.data()) + 1,
                                         nullptr,
                                         0,
                                         0,
                                         {offsets_col->view()});
    auto const scv    = cudf::strings_column_view(strings_view);
    auto const parsed = classify_like_multiliteral("%ab%", {});
    REQUIRE(parsed.has_value());
    REQUIRE(like_multiliteral(scv, *parsed, false, stream, mr) == nullptr);

    // The caller's cudf fallback evaluates the same view correctly.
    auto ref            = cudf::strings::like(scv, "%ab%", std::string_view(), stream, mr);
    auto const ref_vals = to_host_values(ref->view());
    for (size_t i = 0; i < rows.size(); ++i) {
      REQUIRE((ref_vals[i] != 0) == (rows[i].find("ab") != std::string::npos));
    }
  }

  SECTION("empty column returns an empty BOOL8 column")
  {
    auto const empty  = cudf::make_empty_column(cudf::type_id::STRING);
    auto const parsed = classify_like_multiliteral("%ab%", {});
    REQUIRE(parsed.has_value());
    for (bool const invert : {false, true}) {
      auto out =
        like_multiliteral(cudf::strings_column_view(empty->view()), *parsed, invert, stream, mr);
      REQUIRE(out != nullptr);
      REQUIRE(out->size() == 0);
      REQUIRE(out->type().id() == cudf::type_id::BOOL8);
    }
  }

  SECTION("all-null column: zero-byte chars buffer, validity gate does all the work")
  {
    // With every row null the chars buffer holds 0 bytes and its base may be nullptr, which
    // passes the 8-byte-alignment check by design; the null mask alone decides every output.
    auto const col = make_strings({"special requests", "x", ""}, {false, false, false});
    require_matches_cudf(col->view(), "%special%requests%");
  }

  SECTION("embedded NUL bytes in data rows (literals cannot contain NUL, data can)")
  {
    using namespace std::string_literals;
    std::vector<std::string> const rows = {
      "special\0requests"s,      // NUL between the literals — still a match
      "\0special\0requests\0"s,  // NUL-framed
      "spec\0ial requests"s,     // NUL splits the first literal — no match
      "\0"s,
      "\0\0\0"s,                 // NUL-only rows
      "aa\0aa"s,                 // %aa%aa% across a NUL
      "a\0a"s,                   // NUL breaks the digram
      "plain special requests",  // control row
    };
    auto const col = make_strings(rows);
    require_matches_cudf(col->view(), "%special%requests%");
    require_matches_cudf(col->view(), "%aa%aa%");
  }
}

// ---------------------------------------------------------------------------
// Host launcher contract: caps violations throw, they never fall back
// ---------------------------------------------------------------------------

TEST_CASE("like_multiliteral throws on patterns violating the classifier caps",
          "[expression_evaluator][like_multiliteral]")
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  // Hand-built patterns bypass the classifier; a caps-violating pattern reaching the launcher is
  // a contract violation and must throw (nullptr exclusively means layout ineligibility).
  auto const col   = make_strings({"x"});
  auto const empty = cudf::make_empty_column(cudf::type_id::STRING);

  sirius::like_multiliteral_pattern const five_literals{{"a", "b", "c", "d", "e"}};
  sirius::like_multiliteral_pattern const oversized_literal{{std::string(33, 'x')}};
  sirius::like_multiliteral_pattern const empty_literal{{""}};

  auto require_throws = [&](cudf::strings_column_view const& input) {
    REQUIRE_THROWS_AS(like_multiliteral(input, five_literals, false, stream, mr),
                      sirius::internal_exception);
    REQUIRE_THROWS_AS(like_multiliteral(input, oversized_literal, false, stream, mr),
                      sirius::internal_exception);
    REQUIRE_THROWS_AS(like_multiliteral(input, empty_literal, false, stream, mr),
                      sirius::internal_exception);
  };

  require_throws(cudf::strings_column_view(col->view()));
  require_throws(cudf::strings_column_view(empty->view()));

  rmm::device_buffer chars_buf(9, stream, mr);
  auto offsets_col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 2, cudf::mask_state::UNALLOCATED, stream, mr);
  std::vector<int32_t> const offsets{0, 1};
  cudaMemcpyAsync(offsets_col->mutable_view().data<int32_t>(),
                  offsets.data(),
                  offsets.size() * sizeof(int32_t),
                  cudaMemcpyHostToDevice,
                  stream.value());
  stream.synchronize();
  cudf::column_view const misaligned_view(cudf::data_type{cudf::type_id::STRING},
                                          1,
                                          static_cast<char const*>(chars_buf.data()) + 1,
                                          nullptr,
                                          0,
                                          0,
                                          {offsets_col->view()});
  require_throws(cudf::strings_column_view(misaligned_view));
}
