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

#include <cudf/column/column_view.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>

#include <rmm/aligned.hpp>

#include <cub/device/device_scan.cuh>
#include <cuda/accumulation_partial.cuh>

#include <op/dynamic_filter/detail/exact_key_stager.hpp>
#include <op/dynamic_filter/dynamic_filter_key_domain.hpp>
#include <op/dynamic_filter/dynamic_filter_replica_transfer.hpp>
#include <op/dynamic_filter/prepared_build_keys.hpp>
#include <sirius/exception.hpp>
#include <telemetry/nvtx.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <mutex>
#include <utility>

namespace sirius::op::detail {

namespace {

/// Throws `accumulation_invariant_error` with @p mismatch unless @p stream belongs to @p device.
void require_stream_on(::cuda::stream_ref stream, rmm::cuda_device_id device, char const* mismatch)
{
  int stream_device = -1;
  check_cuda(cudaStreamGetDevice(stream.get(), &stream_device), "cudaStreamGetDevice");
  if (stream_device != device.value()) { throw accumulation_invariant_error(mismatch); }
}

/// Arrays of one partial: key `k` owns its buffer at `2k` and its overflow word at `2k + 1`.
constexpr std::size_t buffer_of(std::size_t key) noexcept { return 2 * key; }
constexpr std::size_t word_of(std::size_t key) noexcept { return 2 * key + 1; }

}  // namespace

struct exact_key_stager::impl {
  impl(std::vector<key> staged, std::vector<membership_key_domain> key_domains)
    : keys{std::move(staged)}, domains{std::move(key_domains)}
  {
  }

  /// Bytes of one row of key @p index at its rep.
  [[nodiscard]] std::size_t rep_bytes(std::size_t index) const noexcept
  {
    return membership_rep_bytes(domains[index].rep);
  }

  std::vector<key> keys;
  std::vector<membership_key_domain> domains;  // Per key, classified from its type
  std::atomic<bool> leak_storage{false};
  std::vector<std::unique_ptr<accumulation_partial>> partials;  // Destroyed before leak_storage.
  std::mutex written_mutex;  // Guards `written`; never held across CUDA work.
  std::vector<std::vector<std::vector<slice>>> written;  // [partial][key]: slices written there
};

exact_key_stager::exact_key_stager(std::unique_ptr<impl> state) noexcept : _impl{std::move(state)}
{
}

exact_key_stager::~exact_key_stager()                                      = default;
exact_key_stager::exact_key_stager(exact_key_stager&&) noexcept            = default;
exact_key_stager& exact_key_stager::operator=(exact_key_stager&&) noexcept = default;

std::optional<exact_key_stager> exact_key_stager::try_create(
  std::vector<key> keys, std::span<dynamic_filter_replica_space const> targets)
{
  std::vector<membership_key_domain> domains;
  std::vector<std::size_t> array_bytes;
  domains.reserve(keys.size());
  array_bytes.reserve(2 * keys.size());
  for (auto const& staged : keys) {
    auto const domain = classify_membership_key(staged.type);
    bool const small  = staged.kind == membership_filter_kind::small_in_list;
    if (!domain || (!small && staged.kind != membership_filter_kind::hash_in_list) ||
        (small && (staged.valid_rows == 0 ||
                   staged.valid_rows > sirius_dynamic_small_in_list_filter::k_max_keys))) {
      throw accumulation_invariant_error{"[exact_key_stager] invalid staged key"};
    }
    auto const rep_bytes = membership_rep_bytes(domain->rep);
    if (staged.valid_rows > std::numeric_limits<std::size_t>::max() / rep_bytes ||
        staged.valid_rows > static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max())) {
      throw accumulation_invariant_error{"[exact_key_stager] staged key too large"};
    }
    domains.push_back(*domain);
    // A key without valid rows stages nothing, so it takes no buffer and no overflow word.
    array_bytes.push_back(staged.valid_rows * rep_bytes);
    array_bytes.push_back(staged.valid_rows == 0 ? 0 : sizeof(std::uint32_t));
  }
  auto state    = std::make_unique<impl>(std::move(keys), std::move(domains));
  auto partials = lease_partials(targets, array_bytes, /*zero_fill=*/true, state->leak_storage);
  if (!partials) { return std::nullopt; }
  state->partials = std::move(*partials);
  state->written.assign(state->partials.size(),
                        std::vector<std::vector<slice>>(state->keys.size()));
  return exact_key_stager{std::move(state)};
}

std::size_t exact_key_stager::contribution_scratch_bytes(std::size_t rows)
{
  // prepare_build_keys allocates one rank per row and the temporaries of one exclusive scan.
  auto const items = static_cast<int>(std::min<std::size_t>(rows, std::numeric_limits<int>::max()));
  std::size_t scan = 0;
  check_cuda(cub::DeviceScan::ExclusiveSum(nullptr,
                                           scan,
                                           static_cast<cudf::size_type const*>(nullptr),
                                           static_cast<cudf::size_type*>(nullptr),
                                           items),
             "cub::DeviceScan::ExclusiveSum(size)");
  // One allocation alignment of slack covers the scan's own bookkeeping allocation.
  return tracked_replica_allocation_bytes(rows * sizeof(cudf::size_type)) +
         tracked_replica_allocation_bytes(scan) + rmm::CUDA_ALLOCATION_ALIGNMENT;
}

bool exact_key_stager::enqueue_add(std::span<slice const> slices,
                                   cudf::table_view const& input,
                                   rmm::cuda_device_id device,
                                   ::cuda::stream_ref stream,
                                   rmm::device_async_resource_ref scratch)
{
  nvtx_scoped_range range{"dynfilter::stage::contribute"};
  check_cuda(cudaGetLastError(), "pending CUDA error before staged contribution");
  auto const found = std::ranges::find(
    _impl->partials, device.value(), [](auto const& entry) { return entry->device_id; });
  if (found == _impl->partials.end()) { return false; }
  auto& target       = **found;
  auto const partial = static_cast<std::size_t>(found - _impl->partials.begin());
  require_stream_on(
    stream,
    device,
    "[exact_key_stager::enqueue_add] the contribution stream belongs to another GPU");
  auto const& keys = _impl->keys;
  if (slices.size() != keys.size()) {
    throw accumulation_invariant_error{"[exact_key_stager] one slice per key is required"};
  }
  for (std::size_t index = 0; index < keys.size(); ++index) {
    if (keys[index].valid_rows == 0) { continue; }
    auto const& [offset, rows] = slices[index];
    auto const column_type     = input.column(keys[index].ordinal).type();
    bool const carrier         = _impl->domains[index].family == membership_key_family::string_hash
                                   ? column_type.id() == cudf::type_id::STRING
                                   : membership_probe_compatible(_impl->domains[index], column_type);
    if (offset > keys[index].valid_rows || rows > keys[index].valid_rows - offset || !carrier) {
      throw accumulation_invariant_error{"[exact_key_stager] a slice does not fit its key"};
    }
  }
  rmm::cuda_set_device_raii guard{device};
  std::array const joined{stream};
  try {
    join_on_failure(joined, [&] {
      target.enqueue_contribution(stream, [&] {
        for (std::size_t index = 0; index < keys.size(); ++index) {
          if (keys[index].valid_rows == 0) { continue; }
          auto const rep_bytes = _impl->rep_bytes(index);
          auto* const first    = static_cast<std::byte*>(target.arrays[buffer_of(index)]->data()) +
                              slices[index].offset * rep_bytes;
          cudf::mutable_column_view const out{membership_rep_type(_impl->domains[index]),
                                              static_cast<cudf::size_type>(slices[index].rows),
                                              first,
                                              nullptr,
                                              0};
          prepare_build_keys(_impl->domains[index],
                             input.column(keys[index].ordinal),
                             out,
                             static_cast<std::uint32_t*>(target.arrays[word_of(index)]->data()),
                             stream,
                             scratch);
        }
      });
    });
  } catch (unjoined_gpu_work const&) {
    _impl->leak_storage.store(true);
    throw;
  }
  std::scoped_lock lock(_impl->written_mutex);
  for (std::size_t index = 0; index < keys.size(); ++index) {
    if (keys[index].valid_rows != 0 && slices[index].rows != 0) {
      _impl->written[partial][index].push_back(slices[index]);
    }
  }
  return true;
}

exact_key_stager::gathered::gathered(impl const& state,
                                     accumulation_partial const& root,
                                     std::vector<bool> overflowed)
  : _state{&state}, _root{&root}, _overflowed{std::move(overflowed)}
{
}

bool exact_key_stager::gathered::overflowed(std::size_t key_index) const
{
  return _overflowed.at(key_index);
}

std::shared_ptr<sirius_dynamic_filter> exact_key_stager::gathered::build(
  std::size_t key_index, ::cuda::stream_ref stream, rmm::device_async_resource_ref mr) const
{
  if (overflowed(key_index)) {
    throw accumulation_invariant_error{"[exact_key_stager] an overflowed key builds no list"};
  }
  auto const& staged = _state->keys[key_index];
  auto const& domain = _state->domains[key_index];
  cudf::column_view const prepared{
    membership_rep_type(domain),
    static_cast<cudf::size_type>(staged.valid_rows),
    staged.valid_rows == 0 ? nullptr : _root->arrays[buffer_of(key_index)]->data(),
    nullptr,
    0};
  if (staged.kind == membership_filter_kind::small_in_list) {
    return std::make_shared<sirius_dynamic_small_in_list_filter>(domain, prepared, stream, mr);
  }
  return std::make_shared<sirius_dynamic_in_list_filter>(domain, prepared, stream, mr);
}

std::optional<exact_key_stager::gathered> exact_key_stager::gather(rmm::cuda_device_id root_device,
                                                                   ::cuda::stream_ref stream)
{
  nvtx_scoped_range range{"dynfilter::stage::gather"};
  check_cuda(cudaGetLastError(), "pending CUDA error before staged publication");
  int current_device = -1;
  check_cuda(cudaGetDevice(&current_device), "cudaGetDevice(staged publication)");
  if (current_device != root_device.value()) {
    throw accumulation_invariant_error(
      "[exact_key_stager::gather] the root GPU is not the current device");
  }
  require_stream_on(stream,
                    root_device,
                    "[exact_key_stager::gather] the publication stream belongs to another GPU");
  auto* root = find_partial(_impl->partials, root_device.value());
  if (root == nullptr) { return std::nullopt; }
  auto const& keys     = _impl->keys;
  auto const& partials = _impl->partials;
  std::vector<::cuda::stream_ref> joined{stream};
  for (auto const& entry : partials) {
    joined.push_back(entry->stream_ref());
  }
  auto words =
    cudf::detail::make_pinned_vector_async<std::uint32_t>(partials.size() * keys.size(), stream);
  std::fill(words.begin(), words.end(), std::uint32_t{0});
  try {
    join_on_failure(joined, [&] {
      // A copy staged through the host reads its source there, so another GPU's writes complete
      // first. Only GPUs other than the root that wrote slices are waited for.
      for (std::size_t partial = 0; partial < partials.size(); ++partial) {
        auto const& source = *partials[partial];
        bool const wrote   = std::ranges::any_of(_impl->written[partial],
                                               [](auto const& slices) { return !slices.empty(); });
        if (&source == root || !wrote) { continue; }
        check_cuda(cudaStreamSynchronize(source.stream->value()),
                   "cudaStreamSynchronize(staged contributions)");
        for (std::size_t index = 0; index < keys.size(); ++index) {
          auto const rep_bytes = _impl->rep_bytes(index);
          for (auto const& written : _impl->written[partial][index]) {
            auto const offset = written.offset * rep_bytes;
            (void)enqueue_replica_copy(
              static_cast<std::byte*>(root->arrays[buffer_of(index)]->data()) + offset,
              root_device,
              static_cast<std::byte const*>(source.arrays[buffer_of(index)]->data()) + offset,
              *source.space,
              written.rows * rep_bytes,
              stream,
              *root->host_space);
          }
        }
      }
      // Every overflow word, each read after its GPU's contributions; `stream` then follows every
      // GPU's work, including the root's own writes that `gathered::build` reads.
      for (std::size_t partial = 0; partial < partials.size(); ++partial) {
        auto const& source = *partials[partial];
        rmm::cuda_set_device_raii guard{rmm::cuda_device_id{source.device_id}};
        for (std::size_t index = 0; index < keys.size(); ++index) {
          if (keys[index].valid_rows == 0) { continue; }
          check_cuda(cudaMemcpyAsync(&words[partial * keys.size() + index],
                                     source.arrays[word_of(index)]->data(),
                                     sizeof(std::uint32_t),
                                     cudaMemcpyDeviceToHost,
                                     source.stream->value()),
                     "cudaMemcpyAsync(overflow word)");
        }
        cuda_event read;
        check_cuda(cudaEventRecord(read.get(), source.stream->value()),
                   "cudaEventRecord(overflow words)");
        check_cuda(cudaStreamWaitEvent(stream.get(), read.get(), 0),
                   "cudaStreamWaitEvent(overflow words)");
      }
      check_cuda(cudaStreamSynchronize(stream.get()), "cudaStreamSynchronize(staging gather)");
    });
  } catch (unjoined_gpu_work const&) {
    _impl->leak_storage.store(true);
    throw;
  }
  std::vector<bool> overflowed(keys.size(), false);
  for (std::size_t index = 0; index < keys.size(); ++index) {
    for (std::size_t partial = 0; partial < partials.size(); ++partial) {
      if (words[partial * keys.size() + index] != 0) { overflowed[index] = true; }
    }
  }
  return gathered{*_impl, *root, std::move(overflowed)};
}

bool exact_key_stager::releasable_here() const noexcept
{
  if (!_impl) { return true; }
  try {
    return std::ranges::all_of(_impl->partials,
                               [](auto const& entry) { return entry->releasable_here(); });
  } catch (...) {
    return false;
  }
}

std::optional<std::vector<std::byte>> exact_key_stager::inspect_buffer(
  std::size_t key_index, rmm::cuda_device_id device) const
{
  auto const* partial = _impl ? find_partial(_impl->partials, device.value()) : nullptr;
  if (partial == nullptr || key_index >= _impl->keys.size() ||
      !partial->arrays[buffer_of(key_index)]) {
    return std::nullopt;
  }
  rmm::cuda_set_device_raii guard{device};
  auto const& buffer = *partial->arrays[buffer_of(key_index)];
  std::vector<std::byte> bytes(buffer.size());
  check_cuda(
    cudaMemcpyAsync(
      bytes.data(), buffer.data(), bytes.size(), cudaMemcpyDeviceToHost, partial->stream->value()),
    "cudaMemcpyAsync(staged buffer)");
  check_cuda(cudaStreamSynchronize(partial->stream->value()),
             "cudaStreamSynchronize(staged buffer)");
  return bytes;
}

}  // namespace sirius::op::detail
