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

#include <cuda/accumulation_partial.cuh>

#include <log/logging.hpp>
#include <op/dynamic_filter/detail/exact_key_stager.hpp>
#include <op/dynamic_filter/dynamic_filter_key_domain.hpp>
#include <op/dynamic_filter/dynamic_filter_replica_transfer.hpp>
#include <op/dynamic_filter/prepared_build_keys.hpp>
#include <sirius/exception.hpp>
#include <telemetry/nvtx.hpp>

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

std::optional<exact_key_stager::publication> exact_key_stager::finish(
  rmm::cuda_device_id root_device, ::cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  nvtx_scoped_range range{"dynfilter::stage::finish"};
  check_cuda(cudaGetLastError(), "pending CUDA error before staged publication");
  int current_device = -1;
  check_cuda(cudaGetDevice(&current_device), "cudaGetDevice(staged publication)");
  if (current_device != root_device.value()) {
    throw accumulation_invariant_error(
      "[exact_key_stager::finish] the root GPU is not the current device");
  }
  require_stream_on(stream,
                    root_device,
                    "[exact_key_stager::finish] the publication stream belongs to another GPU");
  auto* root = find_partial(_impl->partials, root_device.value());
  if (root == nullptr) { return std::nullopt; }
  auto const& keys = _impl->keys;
  std::vector<::cuda::stream_ref> joined{stream};
  for (auto const& entry : _impl->partials) {
    joined.push_back(entry->stream_ref());
  }
  publication result;
  result.filters.resize(keys.size());
  try {
    join_on_failure(joined, [&] {
      // Every write is complete before any slice is read: a HOST-staged gather reads on the host.
      for (auto const& entry : _impl->partials) {
        check_cuda(cudaStreamSynchronize(entry->stream->value()),
                   "cudaStreamSynchronize(staged contributions)");
      }
      for (std::size_t partial = 0; partial < _impl->partials.size(); ++partial) {
        auto const& source = *_impl->partials[partial];
        if (&source == root) { continue; }
        for (std::size_t index = 0; index < keys.size(); ++index) {
          if (keys[index].valid_rows == 0) { continue; }
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
      // One read of every overflow word; a set word on any GPU drops the key.
      std::vector<std::uint32_t> words(_impl->partials.size() * keys.size(), 0);
      for (std::size_t partial = 0; partial < _impl->partials.size(); ++partial) {
        auto const& source = *_impl->partials[partial];
        for (std::size_t index = 0; index < keys.size(); ++index) {
          if (keys[index].valid_rows == 0) { continue; }
          check_cuda(cudaMemcpyAsync(&words[partial * keys.size() + index],
                                     source.arrays[word_of(index)]->data(),
                                     sizeof(std::uint32_t),
                                     cudaMemcpyDeviceToHost,
                                     source.stream->value()),
                     "cudaMemcpyAsync(overflow word)");
        }
        check_cuda(cudaStreamSynchronize(source.stream->value()),
                   "cudaStreamSynchronize(overflow words)");
      }
      check_cuda(cudaStreamSynchronize(stream.get()), "cudaStreamSynchronize(staging gather)");
      for (std::size_t index = 0; index < keys.size(); ++index) {
        bool overflowed = false;
        for (std::size_t partial = 0; partial < _impl->partials.size(); ++partial) {
          overflowed = overflowed || words[partial * keys.size() + index] != 0;
        }
        if (overflowed) {
          ++result.counts.filters_skipped_construction_invariant;
          SIRIUS_LOG_ERROR(
            "[exact_key_stager] key {} staged more valid rows than its inventory recorded; its "
            "exact filter is dropped.",
            index);
          continue;
        }
        auto const& domain = _impl->domains[index];
        cudf::column_view const prepared{
          membership_rep_type(domain),
          static_cast<cudf::size_type>(keys[index].valid_rows),
          keys[index].valid_rows == 0 ? nullptr : root->arrays[buffer_of(index)]->data(),
          nullptr,
          0};
        if (keys[index].kind == membership_filter_kind::small_in_list) {
          result.filters[index] =
            std::make_shared<sirius_dynamic_small_in_list_filter>(domain, prepared, stream, mr);
        } else {
          result.filters[index] =
            std::make_shared<sirius_dynamic_in_list_filter>(domain, prepared, stream, mr);
        }
      }
      // The lists have read the root buffers before they are visible or released.
      check_cuda(cudaStreamSynchronize(stream.get()), "cudaStreamSynchronize(staged lists)");
    });
  } catch (unjoined_gpu_work const&) {
    _impl->leak_storage.store(true);
    throw;
  }
  return result;
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
