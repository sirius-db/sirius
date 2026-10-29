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

#include <cudf/utilities/traits.hpp>

#include <duckdb/common/exception.hpp>

#include <charconv>
#include <condition_variable>
#include <cstring>
#include <limits>

namespace sirius::op::scan {
namespace sm = scan_manager;
scan_manager::scan_envelope arrow_preparation_envelope(arrow_batch_producer const& producer)
{
  return {{producer.limits().envelope}, {}, true, 1};
}
namespace {
uint64_t add(uint64_t a, uint64_t b)
{
  if (b > UINT64_MAX - a) throw std::overflow_error("Arrow copy bound overflow");
  return a + b;
}
uint64_t aligned(uint64_t n) { return add(n, 15) / 16 * 16; }
int32_t offset_at(void const* data, int64_t i)
{
  int32_t result;
  std::memcpy(&result, static_cast<std::byte const*>(data) + i * sizeof(result), sizeof(result));
  return result;
}
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
      reserve((rows + 1) * sizeof(int32_t));
      reserve(rows
                ? offset_at(source.buffers[1], begin + rows) - offset_at(source.buffers[1], begin)
                : 0);
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
      auto* offsets = reinterpret_cast<int32_t*>(alloc((rows + 1) * sizeof(int32_t)));
      auto first    = rows ? offset_at(source.buffers[1], begin) : 0;
      auto bytes    = rows ? offset_at(source.buffers[1], begin + rows) - first : 0;
      auto* chars   = alloc(bytes);
      if (bytes)
        std::memcpy(chars, static_cast<std::byte const*>(source.buffers[2]) + first, bytes);
      for (uint64_t row = 0; rows && row <= rows; ++row)
        offsets[row] = offset_at(source.buffers[1], begin + row) - first;
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
  impl(materialization_program p, arrow_push_limits b, std::function<void(std::exception_ptr)> f)
    : program(std::move(p)), bounds(b), report(std::move(f))
  {
    if (!program.contract || !program.generation || !program.input_types || !report ||
        b.max_rows < 0 || b.max_rows > std::numeric_limits<cudf::size_type>::max() ||
        !b.max_offered_bytes || !b.max_copy_bytes || !b.staging_bytes)
      throw std::invalid_argument(
        "Arrow push queue requires finite bounds and a failure authority");
    (void)add(add(b.max_offered_bytes, aligned(b.max_copy_bytes)), b.staging_bytes);
  }
  void validate(offered_arrow_input const& value) const
  {
    if (!value.schema || !value.array || !value.owner || !value.retained_bytes ||
        value.retained_bytes > bounds.max_offered_bytes || value.array->length > bounds.max_rows ||
        value.array->n_children != static_cast<int64_t>(program.input_types->size()))
      throw std::invalid_argument("Arrow offered input exceeds its declared bounds");
    auto types = validate_arrow_host_flat(*value.schema, *value.array, value.metadata_lengths);
    if (types != *program.input_types) throw std::invalid_argument("Arrow offered schema drift");
    if (copy_size(value, types) > aligned(bounds.max_copy_bytes) ||
        arrow_import_staging_bytes(*value.array, types) > bounds.staging_bytes)
      throw std::invalid_argument("Arrow offered input exceeds copy or normalization bounds");
    for (size_t i = 0; i < types.size(); ++i) {
      if (types[i].id() != cudf::type_id::DECIMAL128) continue;
      int precision = 0;
      std::string_view format(value.schema->children[i]->format);
      (void)std::from_chars(format.data() + 2, format.data() + format.size(), precision);
      if (program.decimal_precisions.size() != types.size() ||
          program.decimal_precisions[i] != precision)
        throw std::invalid_argument("Arrow offered decimal precision drift");
    }
  }
};
arrow_push_queue::arrow_push_queue(materialization_program p,
                                   arrow_push_limits b,
                                   std::function<void(std::exception_ptr)> f)
  : impl_(std::make_unique<impl>(std::move(p), b, std::move(f)))
{
}
arrow_push_queue::~arrow_push_queue() = default;
bool arrow_push_queue::offer(offered_arrow_input const& input)
{
  // Validate before accepting ownership or waiting for capacity. A rejected root
  // remains owned by the caller, but the attempt must still observe physical drift.
  try {
    impl_->validate(input);
  } catch (std::exception const& e) {
    auto error = std::make_exception_ptr(transparent::classified_execution_error(
      transparent::late_failure_cause::physical_input, e.what()));
    fail(error);
    std::rethrow_exception(error);
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
  {
    std::lock_guard lock(impl_->mutex);
    first = !impl_->error;
    if (first) impl_->error = error;
    refused = std::move(impl_->input);
    impl_->input.reset();
    wake = impl_->wake;
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
  return impl_->input || impl_->finished || impl_->closed || impl_->cancelled || impl_->error
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
  offered_arrow_input input;
  {
    std::lock_guard lock(impl_->mutex);
    if (impl_->error) std::rethrow_exception(impl_->error);
    if (impl_->cancelled || stop.stop_requested()) throw duckdb::InterruptException();
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
      if (impl_->cancelled || stop.stop_requested()) throw duckdb::InterruptException();
      if (impl_->closed) throw std::logic_error("Arrow queue closed during copy");
      impl_->consumed = true;
    }
    impl_->cv.notify_all();
    return produced_arrow_input{std::move(batch), {}};
  } catch (duckdb::InterruptException const&) {
    cancel();
    throw;
  } catch (...) {
    fail(std::current_exception());
    throw;
  }
}
void arrow_push_queue::close()
{
  cancel();
  std::lock_guard lock(impl_->mutex);
  impl_->closed = true;
}
void arrow_push_queue::cancel() noexcept
{
  sm::preparation_wakeup wake;
  std::optional<offered_arrow_input> refused;
  {
    std::lock_guard lock(impl_->mutex);
    impl_->cancelled = true;
    refused          = std::move(impl_->input);
    impl_->input.reset();
    wake = impl_->wake;
  }
  impl_->cv.notify_all();
  wake();
}
}  // namespace sirius::op::scan
