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

#include "ducklake/scan_ingestible.hpp"

#include "ducklake/binding_bridge.hpp"
#include "ducklake/dv_visibility.hpp"
#include "ducklake/field_mapping.hpp"
#include "ducklake/inline_batch_producer.hpp"
#include "ducklake/parquet_unit_ingestible.hpp"
#include "ducklake/preparation_gate.hpp"
#include "log/logging.hpp"
#include "op/scan/table_scan/split_validation.hpp"
#include "scan_manager/preparation_slot.hpp"

#include <cudf/column/column_factories.hpp>

#include <cucascade/memory/memory_space.hpp>
#include <duckdb/common/exception.hpp>

#include <numeric>
namespace sirius::ducklake {
namespace {
namespace scan = op::scan;
namespace sm   = scan_manager;
struct scan_info final : scan::ingestible_table_info {
  std::vector<std::string> names;
  std::span<std::string const> column_names() const override { return names; }
  std::span<std::string const> file_paths() const override { return {}; }
  std::string display_name() const override { return "DuckLake bound scan"; }
  bool allows_cached_rows() const override { return false; }
};
}  // namespace
struct scan_ingestible::impl {
  std::shared_ptr<binding_handle> binding;
  scan_info info;
  uint64_t contract, generation;
  std::shared_ptr<sm::preparation_admission> admission;
  std::shared_ptr<sm::preparation_ledger> ledger;
  std::vector<size_t> selected, outputs;
  std::vector<cudf::data_type> output_types;
  unit_scan_options options;
  bool count_only;
  uint64_t next_file = 0;
  impl(std::shared_ptr<binding_handle> b,
       uint64_t contract,
       uint64_t generation,
       std::shared_ptr<sm::preparation_admission> a,
       std::vector<size_t> columns,
       bool count,
       unit_scan_options opts)
    : binding(std::move(b)),
      contract(contract),
      generation(generation),
      admission(std::move(a)),
      options(std::move(opts)),
      count_only(count)
  {
    if (!binding || !contract || !generation || !admission || !admission->decision.deferred)
      throw std::invalid_argument("DuckLake scan requires original binding and admitted resources");
    auto const& identity = *binding->view().identity;
    if (count_only && (!columns.empty() || options.filters || options.output_columns))
      throw std::invalid_argument("DuckLake count-only cannot request filter columns");
    if (!count_only)
      selected = required_field_indices(identity.bound_types.size(), std::move(columns));
    validate_scan_options(options, selected.size());
    info.contract_id = contract;
    if (options.profiles) info.profiles = options.profiles;
    options.profiles    = info.profiles;
    options.identities  = std::make_shared<preparation_identity>();
    options.inline_gate = std::make_shared<preparation_gate>();
    outputs.resize(selected.size());
    std::iota(outputs.begin(), outputs.end(), 0);
    if (options.output_columns) outputs = *options.output_columns;
    for (auto column : selected)
      info.names.push_back(identity.bound_names.at(column));
    for (auto column : outputs) {
      auto type = scan::field_mapping_type(identity.bound_types.at(selected.at(column)));
      if (!type) throw std::invalid_argument("DuckLake scan output type is not qualified");
      output_types.push_back(*type);
    }
  }
  std::optional<scan::composite_scan_child> discover()
  {
    auto descriptor = binding->unit_at(next_file);
    while (descriptor && binding->can_prune_unit(next_file)) {
      SIRIUS_LOG_DEBUG("DuckLake Top-N pruned file before open: {}", descriptor->input.path);
      if (next_file == std::numeric_limits<uint64_t>::max())
        throw std::overflow_error("DuckLake file index exhausted");
      descriptor = binding->unit_at(++next_file);
    }
    if (!descriptor) { return {}; }
    scan::composite_scan_child state;
    unit_scan_options opts;
    opts.scan_task_batch_size = options.scan_task_batch_size;
    if (options.filters) opts.filters = options.filters->Copy();
    opts.output_columns = options.output_columns;
    opts.profiles       = options.profiles;
    opts.identities     = options.identities;
    opts.inline_gate    = options.inline_gate;
    opts.parquet_slot   = options.parquet_slot;
    opts.owns_admission = false;
    if (descriptor->kind == unit_kind::parquet) {
      auto source = binding->make_parquet_unit(
        next_file, contract, generation, admission, selected, count_only, std::move(opts));
      state.validator = source->materializer().validator;
      state.checks    = scan::check_bit(scan::later_check::footer_per_file) |
                     scan::check_bit(scan::later_check::profile_per_file) |
                     scan::check_bit(scan::later_check::schema_per_file);
      state.source = std::move(source);
    } else {
      auto source = binding->make_inline_unit(
        next_file, contract, generation, admission, count_only, selected, std::move(opts));
      state.validator = source->materializer().validator;
      state.checks    = scan::check_bit(scan::later_check::host_staged);
      state.source    = std::move(source);
    }
    if (next_file == std::numeric_limits<uint64_t>::max())
      throw std::overflow_error("DuckLake original file index exhausted");
    ++next_file;
    return state;
  }
};
scan_ingestible::scan_ingestible(std::shared_ptr<binding_handle> binding,
                                 uint64_t contract,
                                 uint64_t generation,
                                 std::shared_ptr<sm::preparation_admission> admission,
                                 std::vector<size_t> columns,
                                 bool count,
                                 unit_scan_options options)
  : impl_(std::make_shared<impl>(std::move(binding),
                                 contract,
                                 generation,
                                 std::move(admission),
                                 std::move(columns),
                                 count,
                                 std::move(options)))
{
  auto supported = scan::check_bit(scan::later_check::footer_per_file) |
                   scan::check_bit(scan::later_check::profile_per_file) |
                   scan::check_bit(scan::later_check::schema_per_file) |
                   scan::check_bit(scan::later_check::host_staged);
  driver_ = std::make_unique<scan::composite_scan_driver>(
    contract, generation, supported, impl_->options.scan_task_batch_size, [state = impl_] {
      return state->discover();
    });
}
scan_ingestible::~scan_ingestible() = default;
sm::unit_envelope scan_ingestible::envelope(size_t columns, size_t string_columns)
{
  auto result              = inline_batch_producer::envelope(columns, string_columns);
  result.temporary.front() = std::max<uint64_t>(dv_max_file_bytes, result.temporary.front());
  result.retained = 14 * dv_max_footer_bytes + 512 + 4 * dv_max_positions * sizeof(int64_t);
  return result;
}

void scan_ingestible::preflight_for_query()
{
  impl_->binding->bind_execution_owner();
  impl_->binding->configure_scan(impl_->selected, impl_->options.filters.get());
}
bool scan_ingestible::has_processed_all_metadata() const { return driver_->exhausted(); }
scan_ingestible::metadata_scan_task_t scan_ingestible::next_split_provider(io::ioctx_resolver)
{
  throw std::logic_error("DuckLake scan requires admitted multi-lane preparation");
}
std::vector<scan_ingestible::preparation_source> scan_ingestible::make_preparation_sources(
  io::ioctx_resolver resolve)
{
  return driver_->make_preparation_sources(
    resolve, impl_->ledger, _execution_completion, [state = impl_](sm::preparation_wakeup wake) {
      state->options.inline_gate->wake = wake;
    });
}
std::unique_ptr<scan::batch_coalescer> scan_ingestible::create_batch_coalescer() const
{
  return driver_->create_batch_coalescer();
}
std::shared_ptr<sm::preparation_admission> scan_ingestible::preparation_admission() const
{
  return impl_->admission;
}
void scan_ingestible::attach_preparation_ledger(std::shared_ptr<sm::preparation_ledger> ledger)
{
  if (!ledger || impl_->ledger) throw std::logic_error("DuckLake scan ledger attachment invalid");
  impl_->options.parquet_slot = std::make_shared<preparation_slot>(
    ledger,
    sm::unit_key{impl_->contract, impl_->options.identities->next()},
    envelope(0, 0).retained);
  impl_->ledger = std::move(ledger);
}
void scan_ingestible::stop_preparation() noexcept
{
  driver_->stop_preparation();
  impl_->ledger.reset();
}
void scan_ingestible::finish_preparation() noexcept
{
  driver_->finish_preparation();
  stop_preparation();
  impl_->options.parquet_slot.reset();
  impl_->admission->finish();
}
scan::ingestible_table_info const& scan_ingestible::table_info() const noexcept
{
  return impl_->info;
}
std::vector<size_t> scan_ingestible::materialized_column_order() const { return impl_->outputs; }
bool scan_ingestible::has_row_filter() const noexcept { return bool(impl_->options.filters); }
scan::materializer_contract_identity scan_ingestible::materializer() const
{
  return {scan::materializer_kind::stream, "ducklake.bound-scan.v1", driver_->validator()};
}
scan::filtered_table scan_ingestible::materialize_metadata_to_table(
  scan::scan_info const& input,
  cucascade::memory::memory_space const& space,
  ::cuda::stream_ref stream,
  bool like,
  std::shared_ptr<sirius::like_multiliteral_cache const> cache,
  expression_query_policy policy)
{
  scan::validate_split_for_gpu(
    impl_->contract, {}, {}, input, *driver_->validator(), impl_->generation, impl_->info.profiles);
  auto batch = driver_->materialize(input, space, stream, like, std::move(cache), policy);
  if (!batch) {
    std::vector<std::unique_ptr<cudf::column>> columns;
    for (auto type : impl_->output_types)
      columns.push_back(cudf::make_empty_column(type));
    if (columns.empty())
      columns.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::INT8}));
    return {scan::owning_table_view(std::make_unique<cudf::table>(std::move(columns))),
            scan::filter_state::ROW_FILTERED_AND_PROJECTED};
  }
  return std::move(*batch);
}
std::unique_ptr<cudf::table> scan_ingestible::post_filter_and_project(
  scan::filtered_table&& input,
  cucascade::memory::memory_space const& space,
  ::cuda::stream_ref stream,
  bool,
  std::shared_ptr<sirius::like_multiliteral_cache const>,
  std::unique_ptr<cudf::column>*,
  std::span<size_t const> elided,
  expression_query_policy)
{
  if (!elided.empty()) throw std::invalid_argument("DuckLake scan elision is not qualified");
  input.table.materialize(stream, space.get_default_allocator());
  return input.table.release(stream, space.get_default_allocator());
}
}  // namespace sirius::ducklake
