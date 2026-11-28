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

#include "op/scan/paimon/paimon_arrow_producer.hpp"

#include "op/scan/arrow/arrow_input_copy.hpp"
#include "transparent/replay_admission.hpp"

#include <atomic>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <system_error>

namespace sirius::op::scan {
namespace sm = scan_manager;
namespace {
uint64_t copy_charge(arrow_copy_limits const& limits)
{
  if (limits.max_copy_bytes > UINT64_MAX - 15)
    throw std::overflow_error("Paimon copy limit overflow");
  auto copy = (limits.max_copy_bytes + 15) / 16 * 16;
  if (limits.staging_bytes > UINT64_MAX - copy)
    throw std::overflow_error("Paimon staging limit overflow");
  return copy + limits.staging_bytes;
}
std::shared_ptr<owned_arrow_batch> detach(ArrowSchema const& schema,
                                          ArrowArray const& array,
                                          materialization_program const& program,
                                          arrow_copy_limits const& bounds,
                                          admitted_arrow_input admitted)
{
  auto fields = program.input_types->size();
  if (!schema.format || std::strcmp(schema.format, "+s") || array.offset || array.null_count ||
      schema.n_children != static_cast<int64_t>(fields + 1) ||
      schema.n_children != array.n_children || !schema.children || !array.children ||
      !schema.children[0] || !array.children[0])
    throw std::runtime_error("Paimon batch has an unsupported root layout");
  auto const& kind_schema = *schema.children[0];
  auto const& kind        = *array.children[0];
  if (!kind_schema.name || std::strcmp(kind_schema.name, "_VALUE_KIND") || !kind_schema.format ||
      std::strcmp(kind_schema.format, "c") || !kind_schema.release || !kind.release ||
      kind_schema.dictionary || kind.dictionary || kind_schema.n_children || kind.n_children ||
      kind.offset || kind.null_count || kind.length != array.length || kind.n_buffers != 2 ||
      !kind.buffers || (kind.length && !kind.buffers[1]))
    throw std::runtime_error("Paimon batch has no valid row-kind column");
  auto kinds = static_cast<int8_t const*>(kind.buffers[1]);
  for (int64_t row = 0; row < kind.length; ++row)
    if (kinds[row] != 0) throw std::runtime_error("Paimon batch contains non-insert row kinds");
  std::vector<ArrowSchema> projected(fields);
  std::vector<ArrowSchema*> schema_children(fields);
  for (size_t i = 0; i < fields; ++i) {
    if (!schema.children[i + 1]) throw std::runtime_error("Paimon batch is missing a field");
    projected[i] = *schema.children[i + 1];
    // Native field-id metadata is retained in the bound read. It carries no
    // conversion semantics for this flat, typed SDK output; C Data has no byte lengths.
    projected[i].metadata = nullptr;
    schema_children[i]    = &projected[i];
  }
  ArrowSchema selected_schema = schema;
  ArrowArray selected_array   = array;
  selected_schema.metadata    = nullptr;
  selected_schema.n_children = selected_array.n_children = fields;
  selected_schema.children                               = schema_children.data();
  selected_array.children                                = array.children + 1;
  try {
    detail::validate_arrow_copy_input(selected_schema, selected_array, program, bounds);
  } catch (transparent::arrow_schema_drift_error const& error) {
    throw transparent::classified_execution_error(transparent::late_failure_cause::physical_input,
                                                  error.what(),
                                                  transparent::kArrowSchemaDrift);
  }
  return detail::copy_admitted_arrow_input(
    selected_schema, selected_array, program, bounds, std::move(admitted));
}
}  // namespace
struct paimon_arrow_producer::impl {
  materialization_program program;
  arrow_copy_limits bounds;
  read_factory open;
  std::unique_ptr<paimon_batch_cursor> cursor;
  std::exception_ptr failure;
  std::atomic<bool> cancelled{false};
  bool opened = false, eof = false, closed = false;
  void check_stop(sm::preparation_stop_token stop) const
  {
    if (cancelled || stop.stop_requested())
      throw std::system_error(std::make_error_code(std::errc::operation_canceled));
  }
};
paimon_arrow_producer::paimon_arrow_producer(materialization_program program,
                                             arrow_copy_limits bounds,
                                             read_factory open)
  : impl_(std::make_unique<impl>())
{
  if (!program.contract || !program.generation || !program.input_types || !open ||
      bounds.max_rows <= 0 || bounds.max_rows > std::numeric_limits<cudf::size_type>::max() ||
      !bounds.max_copy_bytes || !bounds.staging_bytes)
    throw std::invalid_argument(
      "Paimon producer requires a typed program, native factory and finite copy bounds");
  if (program.arrow_schema)
    throw std::invalid_argument("Paimon producer does not supply an Arrow source profile");
  (void)copy_charge(bounds);
  impl_->program = std::move(program);
  impl_->bounds  = bounds;
  impl_->open    = std::move(open);
}
paimon_arrow_producer::~paimon_arrow_producer() = default;
sm::preparation_execution_domain paimon_arrow_producer::execution_domain() const noexcept
{
  return sm::preparation_execution_domain::native_reader;
}
arrow_producer_limits paimon_arrow_producer::limits() const
{
  return {impl_->bounds.max_rows, {{copy_charge(impl_->bounds)}, 0}};
}
void paimon_arrow_producer::preflight() {}
arrow_readiness paimon_arrow_producer::readiness() const noexcept { return arrow_readiness::ready; }
void paimon_arrow_producer::bind_wakeup(sm::preparation_wakeup) {}
arrow_production paimon_arrow_producer::next(admitted_arrow_input admitted,
                                             sm::preparation_stop_token stop)
{
  if (impl_->failure) std::rethrow_exception(impl_->failure);
  if (impl_->closed) throw std::logic_error("Paimon producer is closed");
  try {
    impl_->check_stop(stop);
    if (impl_->eof) return arrow_end_of_stream{};
    if (!impl_->opened) {
      impl_->opened = true;
      impl_->cursor = impl_->open(impl_->bounds.max_rows, stop);
      impl_->check_stop(stop);
      if (!impl_->cursor) {
        impl_->eof = true;
        return arrow_end_of_stream{};
      }
    }
    std::shared_ptr<owned_arrow_batch> output;
    bool data = impl_->cursor->consume(
      stop, impl_->bounds.max_rows, [&](ArrowSchema const& schema, ArrowArray const& array) {
        impl_->check_stop(stop);
        output = detach(schema, array, impl_->program, impl_->bounds, admitted);
      });
    impl_->check_stop(stop);
    if (!data) {
      impl_->eof = true;
      return arrow_end_of_stream{};
    }
    return produced_arrow_input{std::move(output), {}};
  } catch (...) {
    impl_->failure = std::current_exception();
    throw;
  }
}
void paimon_arrow_producer::close()
{
  if (impl_->closed) return;
  impl_->closed = true;
  auto open     = std::move(impl_->open);
  auto cursor   = std::move(impl_->cursor);
  try {
    if (cursor) cursor->close();
  } catch (...) {
    if (!impl_->failure) impl_->failure = std::current_exception();
    throw;
  }
}
void paimon_arrow_producer::cancel() noexcept { impl_->cancelled = true; }
}  // namespace sirius::op::scan
