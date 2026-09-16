#pragma once

#include <cudf/io/text/byte_range_info.hpp>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <span>
#include <vector>

namespace io_utils {

using range = cudf::io::text::byte_range_info;

/**
 * @brief Expands each range outward so its start is aligned down and its end
 *        is aligned up to @p alignment bytes.
 *
 * Ranges are processed independently; overlaps introduced by alignment are not
 * merged here — call @c coalesce afterwards if desired.
 *
 * @param ranges    Input ranges (any order).
 * @param alignment Alignment in bytes; must be a power of two.
 * @return One aligned range per input range, in the same order.
 */
inline std::vector<range> align_ranges(std::span<range const> ranges, size_t alignment)
{
  std::vector<range> out;
  out.reserve(ranges.size());
  for (auto const& r : ranges) {
    size_t start       = static_cast<size_t>(r.offset());
    size_t end         = start + static_cast<size_t>(r.size());
    size_t align_start = (start / alignment) * alignment;
    size_t align_end   = ((end + alignment - 1) / alignment) * alignment;
    out.emplace_back(static_cast<int64_t>(align_start),
                     static_cast<int64_t>(align_end - align_start));
  }
  return out;
}

/**
 * @brief Merges adjacent or near-adjacent ranges into fewer, larger ranges.
 *
 * Two consecutive ranges (sorted by offset) are merged when the gap between
 * them is at most @p max_gap bytes.  Overlapping ranges are always merged.
 *
 * @param ranges  Input ranges (need not be sorted).
 * @param max_gap Maximum byte gap that still triggers a merge.
 * @return Sorted, non-overlapping, coalesced ranges.
 */
inline std::vector<range> coalesce_ranges(std::span<range const> ranges, size_t max_gap)
{
  if (ranges.empty()) return {};

  std::vector<range> sorted(ranges.begin(), ranges.end());
  std::sort(sorted.begin(), sorted.end(), [](range const& a, range const& b) {
    return a.offset() < b.offset();
  });

  std::vector<range> out;
  size_t cur_start = static_cast<size_t>(sorted[0].offset());
  size_t cur_end   = cur_start + static_cast<size_t>(sorted[0].size());

  for (size_t i = 1; i < sorted.size(); ++i) {
    size_t next_start = static_cast<size_t>(sorted[i].offset());
    size_t next_end   = next_start + static_cast<size_t>(sorted[i].size());

    if (next_start <= cur_end + max_gap) {
      cur_end = std::max(cur_end, next_end);
    } else {
      out.emplace_back(static_cast<int64_t>(cur_start), static_cast<int64_t>(cur_end - cur_start));
      cur_start = next_start;
      cur_end   = next_end;
    }
  }
  out.emplace_back(static_cast<int64_t>(cur_start), static_cast<int64_t>(cur_end - cur_start));
  return out;
}

inline std::vector<range> coalesce_sorted_ranges(std::span<range const> sorted_ranges,
                                                 size_t max_gap)
{
  if (sorted_ranges.empty()) return {};

  std::vector<range> out;
  size_t cur_start = static_cast<size_t>(sorted_ranges[0].offset());
  size_t cur_end   = cur_start + static_cast<size_t>(sorted_ranges[0].size());

  for (size_t i = 1; i < sorted_ranges.size(); ++i) {
    size_t next_start = static_cast<size_t>(sorted_ranges[i].offset());
    size_t next_end   = next_start + static_cast<size_t>(sorted_ranges[i].size());

    if (next_start <= cur_end + max_gap) {
      cur_end = std::max(cur_end, next_end);
    } else {
      out.emplace_back(static_cast<int64_t>(cur_start), static_cast<int64_t>(cur_end - cur_start));
      cur_start = next_start;
      cur_end   = next_end;
    }
  }
  out.emplace_back(static_cast<int64_t>(cur_start), static_cast<int64_t>(cur_end - cur_start));
  return out;
}

/**
 * @brief Aligns then coalesces ranges in one step.
 *
 * Equivalent to @c coalesce_ranges(align_ranges(ranges, alignment), max_gap).
 * After alignment neighbouring ranges often become adjacent or overlapping, so
 * the coalescing pass is most useful when called together with alignment.
 *
 * @param ranges    Input ranges.
 * @param alignment Alignment in bytes; must be a power of two.
 * @param max_gap   Maximum byte gap between aligned ranges to still merge.
 * @return Sorted, aligned, coalesced ranges.
 */
inline std::vector<range> align_and_coalesce(std::span<range const> ranges,
                                             size_t alignment,
                                             size_t max_gap = 0)
{
  return coalesce_ranges(align_ranges(ranges, alignment), max_gap);
}

/**
 * @brief Aligns then coalesces ranges in one step.
 *
 * Equivalent to @c coalesce_ranges(align_ranges(ranges, alignment), max_gap).
 * After alignment neighbouring ranges often become adjacent or overlapping, so
 * the coalescing pass is most useful when called together with alignment.
 *
 * @param ranges    Input ranges.
 * @param alignment Alignment in bytes; must be a power of two.
 * @param max_gap   Maximum byte gap between aligned ranges to still merge.
 * @return Sorted, aligned, coalesced ranges.
 */
inline std::vector<range> align_and_coalesce_sorted(std::span<range const> ranges,
                                                    size_t alignment,
                                                    size_t max_gap = 0)
{
  return coalesce_sorted_ranges(align_ranges(ranges, alignment), max_gap);
}

/**
 * @brief Merges two sorted range groups into a single sorted, coalesced group.
 *
 * @param a       First sorted range group.
 * @param b       Second sorted range group.
 * @param max_gap Maximum byte gap between ranges to still merge.
 * @return Sorted, coalesced union of both groups.
 */
inline std::vector<range> union_sorted_ranges(std::span<range const> a,
                                              std::span<range const> b,
                                              size_t max_gap = 0)
{
  std::vector<range> merged;
  merged.reserve(a.size() + b.size());
  std::merge(a.begin(),
             a.end(),
             b.begin(),
             b.end(),
             std::back_inserter(merged),
             [](range const& x, range const& y) { return x.offset() < y.offset(); });
  return coalesce_sorted_ranges(merged, max_gap);
}

/**
 * @brief Returns the byte ranges covered by both @p a and @p b.
 *
 * Partial overlaps are included — only the overlapping portion is kept.
 * Both inputs must be sorted by offset and non-overlapping within themselves.
 *
 * @param a First sorted range group.
 * @param b Second sorted range group.
 * @return Sorted ranges present in both groups (intersection).
 */
inline std::vector<range> common_sorted_ranges(std::span<range const> a, std::span<range const> b)
{
  std::vector<range> out;
  size_t i = 0, j = 0;
  while (i < a.size() && j < b.size()) {
    size_t a_start = static_cast<size_t>(a[i].offset());
    size_t a_end   = a_start + static_cast<size_t>(a[i].size());
    size_t b_start = static_cast<size_t>(b[j].offset());
    size_t b_end   = b_start + static_cast<size_t>(b[j].size());

    size_t ovlp_start = std::max(a_start, b_start);
    size_t ovlp_end   = std::min(a_end, b_end);
    if (ovlp_start < ovlp_end)
      out.emplace_back(static_cast<int64_t>(ovlp_start),
                       static_cast<int64_t>(ovlp_end - ovlp_start));

    if (a_end < b_end)
      ++i;
    else if (b_end < a_end)
      ++j;
    else {
      ++i;
      ++j;
    }
  }
  return out;
}

/**
 * @brief Returns the parts of @p source not covered by @p coverage.
 *
 * Partial coverage is handled: if a coverage range clips only one end of a
 * source range, the uncovered portion is still returned.
 * Both inputs must be sorted by offset and non-overlapping within themselves.
 *
 * @param source   Ranges to examine.
 * @param coverage Ranges that are already present / satisfied.
 * @return Sorted sub-ranges of @p source with no coverage.
 */
inline std::vector<range> missing_sorted_ranges(std::span<range const> source,
                                                std::span<range const> coverage)
{
  std::vector<range> out;
  size_t j = 0;
  for (auto const& s : source) {
    size_t cur = static_cast<size_t>(s.offset());
    size_t end = cur + static_cast<size_t>(s.size());

    // Advance j past coverage ranges that end before this source range starts.
    while (j < coverage.size() &&
           static_cast<size_t>(coverage[j].offset()) + static_cast<size_t>(coverage[j].size()) <=
             cur)
      ++j;

    // Walk coverage ranges that overlap this source range with a local cursor
    // so j stays valid for the next source range.
    size_t k = j;
    while (k < coverage.size() && static_cast<size_t>(coverage[k].offset()) < end) {
      size_t c_start = static_cast<size_t>(coverage[k].offset());
      size_t c_end   = c_start + static_cast<size_t>(coverage[k].size());

      if (cur < c_start)
        out.emplace_back(static_cast<int64_t>(cur), static_cast<int64_t>(c_start - cur));
      cur = std::max(cur, c_end);
      ++k;
    }

    if (cur < end) out.emplace_back(static_cast<int64_t>(cur), static_cast<int64_t>(end - cur));
  }
  return out;
}

}  // namespace io_utils
