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

#include "op/scan/arrow/arrow_batch_producer.hpp"

#include "op/scan/arrow/arrow_staging_resource.hpp"
#include "op/scan/arrow/arrow_string_offsets.hpp"

#include <duckdb/common/exception.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <system_error>

namespace sirius::op::scan {
namespace sm = scan_manager;
scan_manager::scan_envelope arrow_preparation_envelope(arrow_batch_producer const& producer)
{
  auto minimum = producer.minimum_permits();
  return {
    std::vector<scan_manager::unit_envelope>(std::max(1U, minimum), producer.limits().envelope),
    {},
    true,
    minimum};
}
namespace {
uint64_t add(uint64_t a, uint64_t b)
{
  if (b > UINT64_MAX - a) throw std::overflow_error("Arrow copy bound overflow");
  return a + b;
}
uint64_t aligned(uint64_t n) { return add(n, 15) / 16 * 16; }
struct copied_column {
  ArrowSchema schema{};
  ArrowArray array{};
  void const* buffers[3]{};
};
uint64_t copy_size(offered_arrow_input const& input, std::vector<cudf::data_type> const& types)
{
  auto rows      = static_cast<uint64_t>(input.array->length);
  uint64_t bytes = 0;
  auto reserve   = [&](uint64_t n) { bytes = add(bytes, aligned(n)); };
  reserve(types.size() * sizeof(ArrowSchema*));
  reserve(types.size() * sizeof(ArrowArray*));
  reserve(sizeof(void*));
  for (size_t i = 0; i < types.size(); ++i) {
    auto const& source = *input.array->children[i];
    auto begin         = source.offset + input.array->offset;
    reserve(sizeof(copied_column));
    reserve(std::strlen(input.schema->children[i]->format) + 1);
    if (rows && source.null_count != 0 && source.buffers[0]) reserve((rows + 7) / 8);
    if (types[i].id() == cudf::type_id::STRING) {
      detail::arrow_string_offsets offsets(input.schema->children[i]->format, source.buffers[1]);
      reserve((rows + 1) * offsets.width());
      reserve(rows ? detail::arrow_string_window_size(offsets[begin], offsets[begin + rows]) : 0);
    } else if (types[i].id() == cudf::type_id::BOOL8) {
      reserve((rows + 7) / 8);
    } else {
      reserve(rows * cudf::size_of(types[i]));
    }
  }
  return bytes;
}
struct copy_owner {
  std::shared_ptr<void> claim;
  sm::charged_block block;
  arrow_staging_resource staging;
  copy_owner(std::shared_ptr<void> c, sm::charged_block b, size_t copy_bytes, size_t staging_bytes)
    : claim(std::move(c)), block(std::move(b)), staging(block.data() + copy_bytes, staging_bytes)
  {
  }
};
// Allocate all arrays, format strings and selected payload through one charged block.
std::shared_ptr<owned_arrow_batch> copy_input(offered_arrow_input const& input,
                                              materialization_program const& program,
                                              arrow_push_limits const& limits,
                                              admitted_arrow_input admitted)
{
  auto const& root  = *input.array;
  auto const& types = *program.input_types;
  auto copy_bytes   = aligned(limits.max_copy_bytes);
  auto storage =
    std::make_shared<copy_owner>(admitted.lifetime,
                                 admitted.allocator.allocate(add(copy_bytes, limits.staging_bytes)),
                                 copy_bytes,
                                 limits.staging_bytes);
  size_t cursor = 0;
  auto alloc    = [&](uint64_t bytes) -> std::byte* {
    auto n = aligned(bytes);
    if (n > copy_bytes - cursor) throw std::invalid_argument("Arrow copy exceeds declared bound");
    auto* result = storage->block.data() + cursor;
    std::memset(result, 0, n);
    cursor += n;
    return result;
  };
  auto** schemas      = reinterpret_cast<ArrowSchema**>(alloc(types.size() * sizeof(ArrowSchema*)));
  auto** arrays       = reinterpret_cast<ArrowArray**>(alloc(types.size() * sizeof(ArrowArray*)));
  auto** root_buffers = reinterpret_cast<void const**>(alloc(sizeof(void*)));
  for (size_t i = 0; i < types.size(); ++i) {
    auto* column       = new (alloc(sizeof(copied_column))) copied_column{};
    auto const& source = *root.children[i];
    auto const& schema = *input.schema->children[i];
    auto& out          = column->array;
    schemas[i]         = &column->schema;
    arrays[i]          = &out;
    auto format_bytes  = std::strlen(schema.format) + 1;
    auto* format       = alloc(format_bytes);
    std::memcpy(format, schema.format, format_bytes);
    column->schema.format  = reinterpret_cast<char*>(format);
    column->schema.flags   = schema.flags;
    column->schema.release = [](ArrowSchema* s) { s->release = nullptr; };
    out.length             = root.length;
    out.release            = [](ArrowArray* a) { a->release = nullptr; };
    out.buffers            = column->buffers;
    auto begin             = source.offset + root.offset;
    auto rows              = static_cast<uint64_t>(root.length);
    if (rows && source.null_count != 0 && source.buffers[0]) {
      auto* bits           = reinterpret_cast<uint8_t*>(alloc((rows + 7) / 8));
      auto const* original = static_cast<uint8_t const*>(source.buffers[0]);
      for (uint64_t row = 0; row < rows; ++row) {
        auto value = (original[(begin + row) / 8] >> ((begin + row) % 8)) & 1;
        bits[row / 8] |= value << (row % 8);
        out.null_count += !value;
      }
      column->buffers[0] = bits;
    }
    bool strings  = types[i].id() == cudf::type_id::STRING;
    out.n_buffers = strings ? 3 : 2;
    if (strings) {
      detail::arrow_string_offsets original(schema.format, source.buffers[1]);
      auto* offsets = alloc((rows + 1) * original.width());
      auto first    = rows ? original[begin] : 0;
      auto bytes    = rows ? detail::arrow_string_window_size(first, original[begin + rows]) : 0;
      auto* chars   = alloc(bytes);
      if (bytes)
        std::memcpy(chars, static_cast<std::byte const*>(source.buffers[2]) + first, bytes);
      for (uint64_t row = 0; rows && row <= rows; ++row) {
        auto offset = detail::arrow_string_window_size(first, original[begin + row]);
        // Preserve the Arrow storage format in the detached input. The importer
        // separately normalizes both formats to its INT32 destination offsets.
        if (original.width() == sizeof(int64_t))
          reinterpret_cast<int64_t*>(offsets)[row] = offset;
        else
          reinterpret_cast<int32_t*>(offsets)[row] = offset;
      }
      column->buffers[1] = offsets;
      column->buffers[2] = chars;
    } else if (types[i].id() == cudf::type_id::BOOL8) {
      auto* bits           = reinterpret_cast<uint8_t*>(alloc((rows + 7) / 8));
      auto const* original = static_cast<uint8_t const*>(source.buffers[1]);
      for (uint64_t row = 0; row < rows; ++row)
        bits[row / 8] |= ((original[(begin + row) / 8] >> ((begin + row) % 8)) & 1) << (row % 8);
      column->buffers[1] = bits;
    } else {
      auto width   = cudf::size_of(types[i]);
      auto* values = alloc(rows * width);
      if (rows)
        std::memcpy(
          values, static_cast<std::byte const*>(source.buffers[1]) + begin * width, rows * width);
      column->buffers[1] = values;
    }
  }
  ArrowSchema schema{};
  schema.format     = "+s";
  schema.n_children = types.size();
  schema.children   = schemas;
  schema.release    = [](ArrowSchema* s) { s->release = nullptr; };
  ArrowArray array{};
  array.length     = root.length;
  array.n_children = types.size();
  array.children   = arrays;
  array.n_buffers  = 1;
  array.buffers    = root_buffers;
  array.release    = [](ArrowArray* a) { a->release = nullptr; };
  return std::make_shared<owned_arrow_batch>(schema, array, storage, storage->staging);
}
}  // namespace
struct arrow_push_queue::impl {
  materialization_program program;
  arrow_push_limits bounds;
  std::function<void(std::exception_ptr)> report;
  mutable std::mutex mutex;
  std::condition_variable cv;
  sm::preparation_wakeup wake;
  std::optional<offered_arrow_input> input;
  bool offering = false, consumed = false, finished = false, closed = false, cancelled = false;
  std::exception_ptr error;
  bool qualified = false, slot_held = false;
  uint32_t accepted               = 0;
  uint64_t allocation_granularity = 1;
  size_t head = 0, queued = 0;
  std::array<std::shared_ptr<owned_arrow_batch>, 2> pending;
  struct allocation_set {
    std::vector<arrow_allocation> allocations;
    arrow_source_memory memory = arrow_source_memory::pageable_host;
    int device                 = -1;
    uint64_t control           = 0;
    bool occupied              = false;
  };
  std::array<allocation_set, 2> live;
  allocation_set offered;
  struct input_credit {
    std::shared_ptr<impl> state;
    size_t index = 0;
    bool armed   = false;
    ~input_credit()
    {
      if (!armed) return;
      sm::preparation_wakeup wake;
      {
        std::lock_guard lock(state->mutex);
        state->live[index] = {};
        --state->accepted;
        wake = state->wake;
      }
      state->cv.notify_all();
      wake();
    }
  };
  impl(materialization_program p, arrow_push_limits b, std::function<void(std::exception_ptr)> f)
    : program(std::move(p)), bounds(b), report(std::move(f))
  {
    if (!program.contract || !program.generation || !program.input_types || !report)
      throw std::invalid_argument(
        "Arrow push queue requires a bound program and failure authority");
    qualified = bool(program.arrow_schema);
    if (qualified) validate_arrow_binding(*program.arrow_schema, program);
    if (b.max_rows < 0 || b.max_rows > std::numeric_limits<cudf::size_type>::max() ||
        !b.max_offered_bytes || !b.max_copy_bytes || !b.staging_bytes)
      throw transparent::classified_execution_error(
        transparent::late_failure_cause::resource,
        "arrow_envelope_unavailable: Arrow push queue requires finite positive byte bounds");
    try {
      (void)add(add(b.max_offered_bytes, aligned(b.max_copy_bytes)), b.staging_bytes);
    } catch (std::overflow_error const&) {
      throw transparent::classified_execution_error(
        transparent::late_failure_cause::resource,
        "arrow_envelope_unavailable: Arrow envelope overflow");
    }
  }
  capacity_result terminal() const
  {
    if (error || cancelled) return capacity_result::failed;
    if (finished || closed) return capacity_result::closed;
    return capacity_result::available;
  }
  void validate(offered_arrow_input const& value) const
  {
    if (!value.schema || !value.array || !value.owner || !value.retained_bytes ||
        value.array->n_children != static_cast<int64_t>(program.input_types->size()))
      throw std::invalid_argument(
        "arrow_layout_unsupported: Arrow offered input lacks a valid root");
    auto types = validate_arrow_host_flat(*value.schema, *value.array, value.metadata_lengths);
    if (types != *program.input_types)
      throw std::invalid_argument(
        "arrow_schema_drift: Arrow offered schema differs from its binding");
    for (size_t i = 0; i < types.size(); ++i) {
      if (types[i].id() != cudf::type_id::DECIMAL128) continue;
      int precision = 0;
      std::string_view format(value.schema->children[i]->format);
      (void)std::from_chars(format.data() + 2, format.data() + format.size(), precision);
      if (program.decimal_precisions.size() != types.size() ||
          program.decimal_precisions[i] != precision)
        throw std::invalid_argument("arrow_schema_drift: Arrow offered decimal precision drift");
    }
    // An independently malformed root must never become a resource refusal just
    // because it also exceeds a configured admission limit.
    if (value.retained_bytes > bounds.max_offered_bytes || value.array->length > bounds.max_rows ||
        copy_size(value, types) > aligned(bounds.max_copy_bytes) ||
        arrow_import_staging_bytes(*value.array, types) > bounds.staging_bytes)
      throw transparent::classified_execution_error(
        transparent::late_failure_cause::resource,
        "arrow_batch_too_large: Arrow input exceeds retained, row, copy or normalization bounds");
  }
};
arrow_push_queue::arrow_push_queue(materialization_program p,
                                   arrow_push_limits b,
                                   std::function<void(std::exception_ptr)> f)
  : impl_(std::make_shared<impl>(std::move(p), b, std::move(f)))
{
}
arrow_push_queue::arrow_push_queue(materialization_program p,
                                   uint64_t bytes,
                                   std::function<void(std::exception_ptr)> f)
  : arrow_push_queue(std::move(p),
                     {std::numeric_limits<cudf::size_type>::max(), bytes, bytes, bytes},
                     std::move(f))
{
  if (!impl_->qualified)
    throw std::invalid_argument("Arrow bounded queue requires a storage schema");
}
arrow_push_queue::~arrow_push_queue() { cancel(); }
arrow_push_queue::offered_slot::offered_slot(offered_slot&& other) noexcept
  : state_(std::move(other.state_)), input_(std::move(other.input_)), status_(other.status_)
{
}
arrow_push_queue::offered_slot& arrow_push_queue::offered_slot::operator=(
  offered_slot&& other) noexcept
{
  if (this != &other) {
    reset();
    state_  = std::move(other.state_);
    input_  = std::move(other.input_);
    status_ = other.status_;
  }
  return *this;
}
arrow_push_queue::offered_slot::~offered_slot() { reset(); }
void arrow_push_queue::offered_slot::reset() noexcept
{
  input_.reset();  // Foreign callbacks and leases, before returning the offered permit.
  if (!state_) return;
  auto state = std::move(state_);
  {
    std::lock_guard lock(state->mutex);
    state->offered   = {};
    state->slot_held = false;
  }
  state->cv.notify_all();
}
void arrow_push_queue::offered_slot::set_input(ArrowSchema& schema,
                                               ArrowArray& array,
                                               std::shared_ptr<void> owner,
                                               arrow_source_profile profile,
                                               arrow_metadata_lengths lengths)
{
  if (!state_ || input_) throw std::logic_error("Arrow root needs an unused offered slot");
  auto source = std::make_shared<arrow_source_profile const>(std::move(profile));
  auto input  = std::make_shared<owned_arrow_batch>(
    schema, array, std::move(owner), *std::pmr::null_memory_resource(), std::move(lengths), source);
  input_ = std::move(input);
  impl::allocation_set allocation{source->allocations, source->memory, source->device, 0, true};
  std::lock_guard lock(state_->mutex);
  state_->offered = std::move(allocation);
}
arrow_push_queue::offered_slot arrow_push_queue::acquire_offered_slot()
{
  if (!impl_->qualified)
    throw std::logic_error("Offered permits require the qualified Arrow queue");
  offered_slot result;
  std::unique_lock lock(impl_->mutex);
  impl_->cv.wait(
    lock, [&] { return !impl_->slot_held || impl_->terminal() != capacity_result::available; });
  result.status_ = impl_->terminal();
  if (result.status_ == capacity_result::available) {
    impl_->slot_held = true;
    result.state_    = impl_;
  }
  return result;
}
arrow_push_queue::offer_outcome arrow_push_queue::offer(offered_slot& slot)
{
  if (slot.state_ != impl_ || !slot.input_)
    throw std::logic_error("Arrow offer has no root/permit");
  arrow_input_estimate estimate;
  impl::allocation_set allocation;
  auto credit   = std::make_shared<impl::input_credit>();
  credit->state = impl_;
  try {
    slot.input_->validate_for(impl_->program);
    estimate     = slot.input_->estimate();
    auto quantum = impl_->allocation_granularity;
    if (slot.input_->row_count() > impl_->bounds.max_rows ||
        estimate.total() > impl_->bounds.max_offered_bytes ||
        (estimate.staging && quantum > impl_->bounds.max_offered_bytes))
      throw transparent::classified_execution_error(
        transparent::late_failure_cause::resource,
        "arrow_batch_too_large: source exceeds the scan row or byte target");
    auto staging_charge =
      estimate.staging ? add(estimate.staging, quantum - 1) / quantum * quantum : 0;
    auto total = add(estimate.total() - estimate.staging, staging_charge);
    if (slot.input_->row_count() > impl_->bounds.max_rows ||
        total > impl_->bounds.max_offered_bytes)
      throw transparent::classified_execution_error(
        transparent::late_failure_cause::resource,
        "arrow_batch_too_large: retained input, normalization and destination exceed scan batch "
        "target");
    auto const& source = *slot.input_->source_profile();
    allocation         = {source.allocations, source.memory, source.device, estimate.control, true};
  } catch (std::invalid_argument const& e) {
    auto error = std::make_exception_ptr(transparent::classified_execution_error(
      transparent::late_failure_cause::physical_input, e.what()));
    fail(error);
    std::rethrow_exception(error);
  } catch (...) {
    fail(std::current_exception());
    throw;
  }
  sm::preparation_wakeup wake;
  {
    std::unique_lock lock(impl_->mutex);
    if (impl_->terminal() != capacity_result::available) return offer_outcome::terminal;
    // The same allocation ID means the same retained parent, across every root.
    for (auto const& live : impl_->live)
      for (auto const& prior : live.allocations)
        for (auto const& current : allocation.allocations) {
          bool same_domain = live.memory == allocation.memory && live.device == allocation.device;
          auto a           = reinterpret_cast<uintptr_t>(prior.base),
               b           = reinterpret_cast<uintptr_t>(current.base);
          if ((prior.id == current.id &&
               (!same_domain || a != b || prior.bytes != current.bytes)) ||
              (same_domain && prior.id != current.id && a < b + current.bytes &&
               b < a + prior.bytes)) {
            lock.unlock();
            auto error = std::make_exception_ptr(transparent::classified_execution_error(
              transparent::late_failure_cause::physical_input,
              "arrow_layout_unsupported: conflicting shared parent identity"));
            fail(error);
            std::rethrow_exception(error);
          }
        }
    if (impl_->accepted == 2) {
      impl_->offered = std::move(allocation);
      return offer_outcome::capacity_wait;
    }
    auto index    = impl_->live[0].occupied ? 1U : 0U;
    credit->index = index;
    slot.input_->bind_credit(credit);
    impl_->live[index]                                = std::move(allocation);
    impl_->pending[(impl_->head + impl_->queued) % 2] = std::move(slot.input_);
    ++impl_->queued;
    ++impl_->accepted;
    credit->armed    = true;
    impl_->offered   = {};
    impl_->slot_held = false;
    slot.state_.reset();
    wake = impl_->wake;
  }
  impl_->cv.notify_all();
  wake();
  return offer_outcome::accepted;
}
arrow_push_queue::capacity_result arrow_push_queue::wait_for_capacity_or_terminal(
  offered_slot const& slot)
{
  if (slot.state_ != impl_) throw std::logic_error("Arrow capacity wait needs its offered slot");
  std::unique_lock lock(impl_->mutex);
  impl_->cv.wait(
    lock, [&] { return impl_->accepted < 2 || impl_->terminal() != capacity_result::available; });
  return impl_->terminal();
}
arrow_push_queue::observation arrow_push_queue::observe() const
{
  std::lock_guard lock(impl_->mutex);
  observation result{impl_->slot_held ? 1U : 0U, impl_->accepted};
  result.reserved_bytes = (result.offered + result.accepted) * impl_->bounds.max_offered_bytes;
  std::vector<uint64_t> seen;
  auto count = [&](impl::allocation_set const& set) {
    result.control_reservation_bytes = add(result.control_reservation_bytes, set.control);
    for (auto const& a : set.allocations)
      if (std::find(seen.begin(), seen.end(), a.id) == seen.end()) {
        seen.push_back(a.id);
        result.unique_retained_bytes = add(result.unique_retained_bytes, a.bytes);
      }
  };
  for (auto const& set : impl_->live)
    count(set);
  count(impl_->offered);
  return result;
}
bool arrow_push_queue::releases_gate_after_next() const noexcept { return impl_->qualified; }
uint32_t arrow_push_queue::minimum_permits() const noexcept { return impl_->qualified ? 2 : 1; }
void arrow_push_queue::bind_allocation_granularity(uint64_t quantum)
{
  if (!quantum) throw std::invalid_argument("Arrow staging granularity must be positive");
  std::lock_guard lock(impl_->mutex);
  if (!impl_->qualified || quantum == impl_->allocation_granularity) return;
  if (impl_->slot_held || impl_->accepted)
    throw std::logic_error("Bind Arrow staging granularity before acquiring roots");
  impl_->allocation_granularity = quantum;
}
void arrow_push_queue::check_status() const
{
  std::lock_guard lock(impl_->mutex);
  if (impl_->error) std::rethrow_exception(impl_->error);
  if (impl_->cancelled) throw duckdb::InterruptException();
}
bool arrow_push_queue::offer(offered_arrow_input const& input)
{
  if (impl_->qualified)
    throw std::logic_error("Acquire an offered slot before producing an Arrow root");
  // Validate before accepting ownership or waiting for capacity. A rejected root
  // remains owned by the caller, but the attempt must still observe physical drift.
  try {
    impl_->validate(input);
  } catch (std::invalid_argument const& e) {
    auto error = std::make_exception_ptr(transparent::classified_execution_error(
      transparent::late_failure_cause::physical_input, e.what()));
    fail(error);
    std::rethrow_exception(error);
  } catch (...) {
    // Preserve resource refusals, allocation failures and any other typed cause.
    fail(std::current_exception());
    throw;
  }
  std::unique_lock lock(impl_->mutex);
  if (impl_->offering || impl_->finished || impl_->closed || impl_->cancelled || impl_->error)
    return false;
  impl_->offering = true;
  impl_->consumed = false;
  impl_->input    = input;
  auto wake       = impl_->wake;
  lock.unlock();
  wake();
  lock.lock();
  impl_->cv.wait(
    lock, [&] { return impl_->consumed || impl_->closed || impl_->cancelled || impl_->error; });
  bool accepted   = impl_->consumed;
  impl_->offering = false;
  return accepted;
}
void arrow_push_queue::finish()
{
  sm::preparation_wakeup wake;
  {
    std::lock_guard lock(impl_->mutex);
    impl_->finished = true;
    wake            = impl_->wake;
  }
  impl_->cv.notify_all();
  wake();
}
void arrow_push_queue::fail(std::exception_ptr error)
{
  if (!error) throw std::invalid_argument("Arrow failure requires an exception");
  sm::preparation_wakeup wake;
  bool first;
  std::optional<offered_arrow_input> refused;
  std::array<std::shared_ptr<owned_arrow_batch>, 2> discarded;
  {
    std::lock_guard lock(impl_->mutex);
    first = !impl_->error;
    if (first) impl_->error = error;
    refused = std::move(impl_->input);
    impl_->input.reset();
    discarded.swap(impl_->pending);
    impl_->queued = 0;
    wake          = impl_->wake;
  }
  impl_->cv.notify_all();
  wake();
  if (first) impl_->report(error);
}
sm::preparation_execution_domain arrow_push_queue::execution_domain() const noexcept
{
  return sm::preparation_execution_domain::metadata_worker;
}
arrow_producer_limits arrow_push_queue::limits() const
{
  if (impl_->qualified) return {impl_->bounds.max_rows, {{impl_->bounds.max_offered_bytes}, 0}};
  // Offered foreign backing is bounded independently; reserve overlap headroom
  // while the admitted copy and its normalization storage are live.
  return {impl_->bounds.max_rows,
          {{add(aligned(impl_->bounds.max_copy_bytes), impl_->bounds.staging_bytes),
            impl_->bounds.max_offered_bytes},
           0}};
}
void arrow_push_queue::preflight() {}
arrow_readiness arrow_push_queue::readiness() const noexcept
{
  std::lock_guard lock(impl_->mutex);
  return impl_->input || impl_->queued || impl_->finished || impl_->closed || impl_->cancelled ||
             impl_->error
           ? arrow_readiness::ready
           : arrow_readiness::pending;
}
void arrow_push_queue::bind_wakeup(sm::preparation_wakeup wake)
{
  std::lock_guard lock(impl_->mutex);
  impl_->wake = std::move(wake);
}
arrow_production arrow_push_queue::next(admitted_arrow_input admitted,
                                        sm::preparation_stop_token stop)
{
  if (impl_->qualified) {
    std::shared_ptr<owned_arrow_batch> input;
    {
      std::lock_guard lock(impl_->mutex);
      if (impl_->error) std::rethrow_exception(impl_->error);
      if (stop.stop_requested() || impl_->cancelled)
        throw std::system_error(std::make_error_code(std::errc::operation_canceled));
      if (!impl_->queued) {
        if (impl_->finished || impl_->closed) return arrow_end_of_stream{};
        throw std::logic_error("Arrow queue claimed while pending");
      }
      input       = std::move(impl_->pending[impl_->head]);
      impl_->head = (impl_->head + 1) % 2;
      --impl_->queued;
    }
    try {
      struct staging {
        std::shared_ptr<void> claim;
        sm::charged_block block;
        arrow_staging_resource arena;
        staging(admitted_arrow_input admitted, uint64_t bytes)
          : claim(std::move(admitted.lifetime)),
            block(bytes ? admitted.allocator.allocate(bytes) : sm::charged_block{}),
            arena(block.data(), bytes)
        {
        }
      };
      auto owner = std::make_shared<staging>(std::move(admitted), input->estimate().staging);
      input->bind_staging(owner, owner->arena);
      {
        std::lock_guard lock(impl_->mutex);
        if (impl_->error) std::rethrow_exception(impl_->error);
        if (stop.stop_requested() || impl_->cancelled)
          throw std::system_error(std::make_error_code(std::errc::operation_canceled));
      }
      return produced_arrow_input{std::move(input), {}};
    } catch (std::system_error const& error) {
      if (error.code() == std::errc::operation_canceled)
        cancel();
      else
        fail(std::current_exception());
      throw;
    } catch (...) {
      fail(std::current_exception());
      throw;
    }
  }
  offered_arrow_input input;
  {
    std::lock_guard lock(impl_->mutex);
    if (impl_->error) std::rethrow_exception(impl_->error);
    if (stop.stop_requested())
      throw std::system_error(std::make_error_code(std::errc::operation_canceled));
    if (impl_->cancelled) throw duckdb::InterruptException();
    if (impl_->closed) throw std::logic_error("Arrow queue is closed");
    if (!impl_->input) {
      if (impl_->finished) return arrow_end_of_stream{};
      throw std::logic_error("Arrow queue claimed without ready input");
    }
    input = std::move(*impl_->input);
    impl_->input.reset();
  }
  try {
    auto batch = copy_input(input, impl_->program, impl_->bounds, admitted);
    {
      std::lock_guard lock(impl_->mutex);
      if (impl_->error) std::rethrow_exception(impl_->error);
      if (stop.stop_requested())
        throw std::system_error(std::make_error_code(std::errc::operation_canceled));
      if (impl_->cancelled) throw duckdb::InterruptException();
      if (impl_->closed) throw std::logic_error("Arrow queue closed during copy");
      impl_->consumed = true;
    }
    impl_->cv.notify_all();
    return produced_arrow_input{std::move(batch), {}};
  } catch (duckdb::InterruptException const&) {
    cancel();
    throw;
  } catch (std::system_error const& error) {
    if (stop.stop_requested() && error.code() == std::errc::operation_canceled)
      cancel();
    else
      fail(std::current_exception());
    throw;
  } catch (...) {
    fail(std::current_exception());
    throw;
  }
}
void arrow_push_queue::close()
{
  if (impl_->qualified) {
    finish();
    return;
  }
  cancel();
  std::lock_guard lock(impl_->mutex);
  impl_->closed = true;
}
void arrow_push_queue::cancel() noexcept
{
  sm::preparation_wakeup wake;
  std::optional<offered_arrow_input> refused;
  std::array<std::shared_ptr<owned_arrow_batch>, 2> discarded;
  {
    std::lock_guard lock(impl_->mutex);
    impl_->cancelled = true;
    refused          = std::move(impl_->input);
    impl_->input.reset();
    discarded.swap(impl_->pending);
    impl_->queued = 0;
    wake          = impl_->wake;
  }
  impl_->cv.notify_all();
  wake();
}
}  // namespace sirius::op::scan
