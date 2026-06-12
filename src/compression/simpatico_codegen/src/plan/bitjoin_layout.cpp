// SPDX-License-Identifier: Apache-2.0
#include "codegen/plan/bitjoin_layout.hpp"

#include "codegen/plan/representation.hpp"  // parse_bitjoin_spec, launch_bitjoin_field, launch_check_truncation

#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <algorithm>
#include <cstdio>
#include <string_view>
#include <utility>

namespace simpatico {

bool resolve_bitjoin_layout(std::string const& compressor_name,
                            std::vector<std::string> const& input_paths,
                            std::vector<std::optional<bit_range>> const& input_ranges,
                            bitjoin_layout* layout,
                            std::string* error_out)
{
  static constexpr std::string_view kBitjoinPfx = "bitjoin_";
  if (compressor_name.size() <= kBitjoinPfx.size() ||
      compressor_name.compare(0, kBitjoinPfx.size(), kBitjoinPfx) != 0) {
    if (error_out)
      *error_out = "only bitjoin_* supports multiple inputs (step: '" + compressor_name + "')";
    return false;
  }
  std::string_view suffix(compressor_name);
  suffix.remove_prefix(kBitjoinPfx.size());
  auto spec = parse_bitjoin_spec(suffix);
  if (spec.output_type.id() == cudf::type_id::EMPTY) {
    if (error_out) *error_out = "bitjoin: bad spec '" + compressor_name + "'";
    return false;
  }

  size_t n_fields = input_paths.size();
  bool any_range  = false;
  for (auto const& r : input_ranges) {
    if (r.has_value()) {
      any_range = true;
      break;
    }
  }
  layout->widths.resize(n_fields);
  layout->src_los.resize(n_fields);

  if (any_range) {
    for (size_t fi = 0; fi < n_fields; ++fi) {
      if (!input_ranges[fi].has_value()) {
        if (error_out) *error_out = "bitjoin: mixing ranged and unranged inputs is not allowed";
        return false;
      }
      auto rng = input_ranges[fi].value();
      if (rng.first < rng.second) {
        if (error_out) *error_out = "bitjoin: input range hi < lo";
        return false;
      }
      layout->widths[fi]  = rng.first - rng.second + 1;
      layout->src_los[fi] = rng.second;
    }
  } else {
    if (spec.fields.empty()) {
      if (error_out) *error_out = "bitjoin: no field widths (use an alias or provide ranges)";
      return false;
    }
    if (spec.fields.size() != n_fields) {
      if (error_out)
        *error_out = "bitjoin: input count (" + std::to_string(n_fields) + ") != field count (" +
                     std::to_string(spec.fields.size()) + ")";
      return false;
    }
    for (size_t fi = 0; fi < n_fields; ++fi) {
      layout->widths[fi]  = spec.fields[fi].bits;
      layout->src_los[fi] = 0;
    }
  }

  uint32_t total_bits = 0;
  for (auto w : layout->widths)
    total_bits += w;
  uint32_t out_width = static_cast<uint32_t>(cudf::size_of(spec.output_type)) * 8;
  if (total_bits > out_width) {
    if (error_out)
      *error_out = "bitjoin: total bits (" + std::to_string(total_bits) +
                   ") exceeds output type width (" + std::to_string(out_width) + ")";
    return false;
  }

  layout->dst_los.resize(n_fields);
  uint32_t offset_from_msb = 0;
  for (size_t fi = 0; fi < n_fields; ++fi) {
    layout->dst_los[fi] = out_width - offset_from_msb - layout->widths[fi];
    offset_from_msb += layout->widths[fi];
  }
  layout->output_type = spec.output_type;
  return true;
}

void bitjoin_warn_on_truncation(std::unordered_map<std::string, cudf::column_view> const& columns,
                                bitjoin_layout const& layout,
                                std::vector<std::string> const& input_paths,
                                std::string const& compressor_name,
                                cudaStream_t stream)
{
  std::vector<std::pair<std::string, uint64_t>> per_input;
  std::unordered_map<std::string, size_t> path_to_idx;
  for (size_t fi = 0; fi < input_paths.size(); ++fi) {
    uint64_t field_mask =
      (layout.widths[fi] >= 64) ? ~uint64_t{0} : ((uint64_t{1} << layout.widths[fi]) - 1);
    uint64_t mask = field_mask << layout.src_los[fi];
    auto it       = path_to_idx.find(input_paths[fi]);
    if (it == path_to_idx.end()) {
      path_to_idx[input_paths[fi]] = per_input.size();
      per_input.emplace_back(input_paths[fi], mask);
    } else {
      per_input[it->second].second |= mask;
    }
  }

  rmm::device_buffer flag_buf;
  try {
    flag_buf = rmm::device_buffer(
      sizeof(uint32_t), rmm::cuda_stream_view{stream}, rmm::mr::get_current_device_resource_ref());
  } catch (...) {
    return;  // best-effort: skip the check if allocation fails
  }
  auto* d_flag = static_cast<uint32_t*>(flag_buf.data());
  cudaMemsetAsync(d_flag, 0, sizeof(uint32_t), stream);
  size_t n_checked = std::min<size_t>(per_input.size(), 32);
  for (size_t b = 0; b < n_checked; ++b) {
    launch_check_truncation(
      columns.at(per_input[b].first), per_input[b].second, d_flag, uint32_t{1} << b, stream);
  }
  uint32_t h_flag = 0;
  cudaMemcpyAsync(&h_flag, d_flag, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  for (size_t b = 0; b < n_checked; ++b) {
    if (h_flag & (uint32_t{1} << b)) {
      std::fprintf(stderr,
                   "WARNING: bitjoin '%s' has non-zero bits outside the selected range "
                   "in input '%s'; compression is lossy.\n",
                   compressor_name.c_str(),
                   per_input[b].first.c_str());
    }
  }
}

}  // namespace simpatico
