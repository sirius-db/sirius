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

#include "ducklake/inline_unit_ingestible.hpp"

#include "ducklake/binding_bridge.hpp"
#include "ducklake/field_mapping.hpp"
#include "ducklake/inline_reader.hpp"
#include "ducklake/inline_staging.hpp"
#include "ducklake/inline_varchar.hpp"
#include "ducklake/preparation_gate.hpp"
#include "expression/ast/from_duckdb.hpp"
#include "expression_evaluator/expression_evaluator.hpp"
#include "helper/type_conversions.hpp"
#include "op/scan/detail/read_retirement.hpp"
#include "op/scan/scan_plan.hpp"
#include "op/scan/scan_utils.hpp"
#include "op/scan/table_scan/owned_arrow_scan.hpp"

#include <cucascade/memory/memory_space.hpp>
#include <duckdb/common/exception.hpp>
#include <duckdb/common/types/data_chunk.hpp>
#include <duckdb/common/types/date.hpp>
#include <duckdb/common/types/timestamp.hpp>

#include <array>
#include <cstring>
#include <numeric>
#include <thread>

namespace sirius::ducklake {
namespace {
namespace scan           = op::scan;
namespace sm             = scan_manager;
constexpr size_t columns = inline_unit_ingestible::max_columns;
struct arrow_tree {
  ArrowSchema schema{}, children_schema[columns]{};
  ArrowArray array{}, children[columns]{};
  ArrowSchema* schemas[columns]{};
  ArrowArray* arrays[columns]{};
  void const* root_buffers[1]{nullptr};
  void const* buffers[columns][3]{};
};
constexpr size_t tree_bytes = (sizeof(arrow_tree) + 7) / 8 * 8;
size_t column_bytes(size_t rows) { return rows * sizeof(int64_t) + rows / 8; }
static_assert(inline_unit_ingestible::max_batch_cells / inline_unit_ingestible::max_columns >= 64);
static_assert(inline_unit_ingestible::max_string_bytes <= INT32_MAX);
template <class T>
void copy_scalar(std::byte* output, size_t row, duckdb::Value const& value)
{
  auto typed = value.GetValue<T>();
  std::memcpy(output + row * sizeof(T), &typed, sizeof(T));
}
void copy_fixed_scalar(std::byte* output, size_t row, duckdb::Value const& value)
{
  using D = duckdb::LogicalTypeId;
  switch (value.type().id()) {
    case D::BOOLEAN:
      if (value.GetValue<bool>()) reinterpret_cast<uint8_t*>(output)[row / 8] |= 1u << (row % 8);
      break;
    case D::TINYINT: copy_scalar<int8_t>(output, row, value); break;
    case D::SMALLINT: copy_scalar<int16_t>(output, row, value); break;
    case D::INTEGER: copy_scalar<int32_t>(output, row, value); break;
    case D::BIGINT: copy_scalar<int64_t>(output, row, value); break;
    case D::UTINYINT: copy_scalar<uint8_t>(output, row, value); break;
    case D::USMALLINT: copy_scalar<uint16_t>(output, row, value); break;
    case D::UINTEGER: copy_scalar<uint32_t>(output, row, value); break;
    case D::UBIGINT: copy_scalar<uint64_t>(output, row, value); break;
    case D::FLOAT: copy_scalar<float>(output, row, value); break;
    case D::DOUBLE: copy_scalar<double>(output, row, value); break;
    case D::DATE: {
      auto days = duckdb::DateValue::Get(value).days;
      std::memcpy(output + row * sizeof(days), &days, sizeof(days));
      break;
    }
    case D::TIMESTAMP: {
      auto micros = duckdb::TimestampValue::Get(value).value;
      std::memcpy(output + row * sizeof(micros), &micros, sizeof(micros));
      break;
    }
    default: throw std::invalid_argument("DuckLake inline scalar type is outside its profile");
  }
}
struct arrow_capacity {
  std::shared_ptr<void> claim;
  sm::charged_block backing, characters;
  inline_staging_resource staging;
  arrow_capacity(std::shared_ptr<void> owner,
                 sm::charged_block b,
                 sm::charged_block chars,
                 size_t payload_bytes,
                 size_t staging_bytes)
    : claim(std::move(owner)),
      backing(std::move(b)),
      characters(std::move(chars)),
      staging(backing.data() + payload_bytes, staging_bytes)
  {
  }
  std::byte* data() { return backing.data(); }
};
struct inline_info final : scan::ingestible_table_info {
  std::vector<std::string> names;
  std::span<std::string const> column_names() const override { return names; }
  std::span<std::string const> file_paths() const override { return {}; }
  std::string display_name() const override { return "DuckLake inline unit"; }
  bool allows_cached_rows() const override { return false; }
};
struct inline_binding {
  unit_descriptor unit;
};
struct inline_evidence final : scan::prepared_visibility {
  std::shared_ptr<inline_binding const> binding;
  std::weak_ptr<scan::owned_arrow_batch> batch;
  uint64_t sequence;
};
class inline_validator final : public scan::split_validator {
  std::shared_ptr<scan::split_validator const> arrow;
  std::shared_ptr<inline_binding const> binding;

 public:
  inline_validator(scan::materialization_program program, std::shared_ptr<inline_binding const> b)
    : arrow(scan::owned_arrow_split_validator(std::move(program))), binding(std::move(b))
  {
  }
  bool accepts(scan::scan_info const& info) const noexcept override { return arrow->accepts(info); }
  scan::split_validation_evidence inspect(scan::split_validation_context const& context,
                                          scan::scan_info const& info) const override
  {
    auto checked      = arrow->inspect(context, info);
    auto const& input = dynamic_cast<scan::owned_arrow_scan_info const&>(info);
    auto proof =
      std::dynamic_pointer_cast<inline_evidence const>(info.dependencies()[0].visibility);
    if (!proof || proof->binding != binding || proof->batch.lock() != input.batch() ||
        proof->sequence != input.sequence())
      throw scan::certificate_incomplete(
        context.contract, {}, "DuckLake inline batch provenance mismatch");
    return checked;
  }
};
struct inline_coalescer final : scan::batch_coalescer {
  size_t bytes;
  explicit inline_coalescer(size_t b) : bytes(b) {}
  size_t target_batch_bytes() const noexcept override { return bytes; }
  cursor_step advance(scan::scan_info& input, size_t&, size_t) override
  {
    return {dynamic_cast<scan::owned_arrow_scan_info&>(input).coalesced_copy(), true};
  }
  std::vector<std::unique_ptr<scan::scan_info>> push(std::unique_ptr<scan::scan_info>) override
  {
    throw std::logic_error("DuckLake inline requires bounded coalescing");
  }
  std::vector<std::unique_ptr<scan::scan_info>> flush() override { return {}; }
};
struct batch_claim {
  std::shared_ptr<sm::preparation_ledger> ledger;
  std::shared_ptr<preparation_gate> gate;
  sm::unit_key key;
  sm::w_permit permit;
  bool acquired = false;
  ~batch_claim()
  {
    if (acquired) {
      ledger->retire_unit(key);
      permit.release();
      gate->active = false;
      gate->wake();
    }
  }
};
}  // namespace
struct inline_unit_ingestible::impl {
  std::shared_ptr<binding_handle> handle;
  std::shared_ptr<inline_binding const> binding;
  inline_info info;
  scan::materialization_program program;
  std::shared_ptr<scan::split_validator const> validator;
  std::shared_ptr<sm::preparation_admission> admission;
  std::shared_ptr<sm::preparation_ledger> ledger;
  std::shared_ptr<preparation_gate> gate = std::make_shared<preparation_gate>();
  std::unique_ptr<inline_reader> reader;  // query-owner lane only; reset by close.
  duckdb::DataChunk const* chunk = nullptr;
  size_t offset = 0, width = 0, row_limit = 0, column_stride = 0;
  std::vector<size_t> selected, outputs;
  std::shared_ptr<sirius::ast::node const> predicate;
  std::vector<size_t> strings;
  uint64_t sequence = 0, bytes = 0;
  std::shared_ptr<preparation_identity> identities;
  bool owns_admission = true, owns_gate = true;
  std::optional<uint64_t> pending_identity;
  std::atomic<bool> done{false};
  std::thread::id owner = std::this_thread::get_id();
  impl(std::shared_ptr<binding_handle> h,
       unit_descriptor unit,
       uint64_t contract,
       uint64_t generation,
       std::shared_ptr<sm::preparation_admission> a,
       bool count_only,
       std::vector<size_t> required_columns,
       unit_scan_options options)
    : handle(std::move(h)),
      binding(std::make_shared<inline_binding const>(inline_binding{std::move(unit)})),
      admission(std::move(a))
  {
    auto const& u = binding->unit;
    if (!contract || !generation || !u.read_view || u.kind == unit_kind::parquet ||
        !assess_local_scope(u).reasons.empty() || u.max_row_count || u.inline_delete_count ||
        u.transaction_local_delete || !admission || !admission->decision.deferred)
      throw std::invalid_argument("DuckLake inline unit has unqualified dependencies/admission");
    auto const& types = u.read_view->bound_types;
    if (count_only && (!required_columns.empty() || options.filters || options.output_columns))
      throw std::invalid_argument(
        "DuckLake count-only cannot request user columns or a filter projection");
    if (!count_only) {
      selected = required_field_indices(types.size(), std::move(required_columns));
      if (selected.empty() || selected.size() > max_columns ||
          std::any_of(selected.begin(), selected.end(), [&](auto index) {
            return !fixed_scalar_arrow_format(types[index]) &&
                   types[index] != duckdb::LogicalType::VARCHAR;
          }))
        throw std::invalid_argument(
          "DuckLake initial inline profile requires 1..32 selected fixed scalar/VARCHAR columns");
    }
    width         = count_only ? 0 : selected.size();
    row_limit     = batch_rows(width);
    column_stride = column_bytes(row_limit);
    for (size_t i = 0; i < width; ++i)
      if (types[selected[i]] == duckdb::LogicalType::VARCHAR) strings.push_back(i);
    bytes            = batch_capacity(width, strings.size());
    info.contract_id = contract;
    if (options.profiles) info.profiles = std::move(options.profiles);
    identities     = std::move(options.identities);
    owns_admission = options.owns_admission;
    owns_gate      = !options.inline_gate;
    if (options.inline_gate) gate = std::move(options.inline_gate);
    if (!count_only)
      for (auto index : selected)
        info.names.push_back(u.read_view->bound_names.at(index));
    validate_scan_options(options, width);
    outputs.resize(width);
    std::iota(outputs.begin(), outputs.end(), 0);
    if (options.output_columns) outputs = std::move(*options.output_columns);
    program.contract   = contract;
    program.generation = generation;
    program.input_types.emplace();
    duckdb::vector<duckdb::ColumnIndex> column_ids;
    duckdb::vector<sirius::logical_type> returned_types;
    std::vector<std::optional<size_t>> positions;
    for (size_t i = 0; i < width; ++i) {
      program.input_types->push_back(*scan::field_mapping_type(types[selected[i]]));
      column_ids.emplace_back(i);
      returned_types.push_back(sirius::from_duckdb(types[selected[i]]));
      positions.emplace_back(i);
    }
    for (auto i : outputs)
      program.output_types.push_back(program.input_types->at(i));
    if (options.filters) {
      auto expression = op::convert_table_filters_to_expression(
        *options.filters, column_ids, returned_types, positions);
      if (expression) {
        predicate = sirius::ast::from_duckdb(*expression);
        if (!predicate)
          throw std::invalid_argument("DuckLake inline predicate is not supported on GPU");
      }
    }
    program.predicates = predicate ? 1 : 0;
    validator          = std::make_shared<inline_validator>(program, binding);
  }
  void close()
  {
    if (std::this_thread::get_id() != owner)
      throw std::runtime_error("DuckLake inline close requires query owner");
    chunk = nullptr;
    reader.reset();
    done = true;
  }
  std::unique_ptr<scan::scan_info> produce(std::shared_ptr<batch_claim> const& claim,
                                           sm::preparation_stop_token stop)
  {
    handle->validate_statement();
    if (stop.stop_requested()) throw duckdb::InterruptException();
    if (!reader) reader = handle->open_inline_unit(binding->unit.original_file_index);
    if (!chunk || offset == chunk->size()) {
      chunk  = reader->next();
      offset = 0;
    }
    if (!chunk) {
      done = true;
      return nullptr;
    }
    // Empty provider output advances the reader without creating a GPU batch.
    // The composite scan supplies an empty carrier only if the whole scan is empty.
    if (!chunk->size()) return nullptr;
    size_t count = std::min(row_limit, static_cast<size_t>(chunk->size()) - offset);
    std::array<size_t, columns> string_sizes{};
    if (!strings.empty()) {
      size_t accepted = 0, used = 0;
      for (; accepted < count; ++accepted) {
        size_t row_bytes = 0;
        std::array<size_t, columns> lengths{};
        bool fits = true;
        for (auto col : strings) {
          auto value  = borrow_inline_varchar(chunk->data[selected[col]], offset + accepted);
          auto length = value ? value->size() : 0;
          if (length > max_string_bytes - used - row_bytes) {
            fits = false;
            break;
          }
          row_bytes += length;
          lengths[col] = length;
        }
        if (!fits) {
          if (!accepted)
            throw sm::preparation_resource_error(
              "DuckLake inline row exceeds the 1 MiB VARCHAR payload budget", true);
          break;
        }
        used += row_bytes;
        for (auto col : strings)
          string_sizes[col] += lengths[col];
      }
      count = accepted;
    }
    // Temporary charged backing intentionally holds its permit through the
    // last GPU input read. Pending then needs neither a cursor advance nor a
    // second retained-payload budget or custom wakeup scheduler.
    auto allocator  = ledger->allocator(claim->permit);
    auto base_bytes = batch_capacity(width);
    auto allocation = allocator.allocate(base_bytes);
    // Arrow buffers are independently owned. Character storage need not be
    // adjacent to the schema, fixed-width values or import staging arena.
    auto characters = strings.empty() ? sm::charged_block{} : allocator.allocate(max_string_bytes);
    auto staging_bytes      = width ? max_staging_bytes : 0;
    auto capacity           = std::make_shared<arrow_capacity>(claim,
                                                     std::move(allocation),
                                                     std::move(characters),
                                                     base_bytes - staging_bytes,
                                                     staging_bytes);
    auto* tree              = new (capacity->data()) arrow_tree{};
    tree->schema.format     = "+s";
    tree->schema.n_children = width;
    tree->schema.children   = tree->schemas;
    tree->schema.release    = [](ArrowSchema* p) { p->release = nullptr; };
    tree->array.length      = count;
    tree->array.n_buffers   = 1;
    tree->array.buffers     = tree->root_buffers;
    tree->array.n_children  = width;
    tree->array.children    = tree->arrays;
    tree->array.release     = [](ArrowArray* p) { p->release = nullptr; };
    auto* payload           = capacity->data() + tree_bytes;
    auto* chars             = capacity->characters.data();
    for (size_t col = 0; col < width; ++col) {
      auto& schema = tree->children_schema[col];
      auto& array  = tree->children[col];
      bool is_string =
        binding->unit.read_view->bound_types[selected[col]] == duckdb::LogicalType::VARCHAR;
      schema.format =
        is_string ? "u"
                  : fixed_scalar_arrow_format(binding->unit.read_view->bound_types[selected[col]]);
      schema.flags   = ARROW_FLAG_NULLABLE;
      schema.release = [](ArrowSchema* p) { p->release = nullptr; };
      auto* values   = payload + col * column_stride;
      auto* valid    = reinterpret_cast<uint8_t*>(values + row_limit * sizeof(int64_t));
      std::memset(values, 0, column_stride);
      auto* offsets      = reinterpret_cast<int32_t*>(values);
      size_t char_offset = 0;
      for (size_t row = 0; row < count; ++row) {
        if (is_string) {
          auto value = borrow_inline_varchar(chunk->data[selected[col]], offset + row);
          if (value) {
            if (value->size() > string_sizes[col] - char_offset)
              throw std::logic_error("DuckLake borrowed VARCHAR changed during detach");
            if (!value->empty()) std::memcpy(chars + char_offset, value->data(), value->size());
            char_offset += value->size();
            valid[row / 8] |= 1u << (row % 8);
          } else
            ++array.null_count;
          offsets[row + 1] = static_cast<int32_t>(char_offset);
        } else {
          auto value = chunk->GetValue(selected[col], offset + row);
          if (value.IsNull())
            ++array.null_count;
          else {
            if (value.type() != binding->unit.read_view->bound_types[selected[col]])
              throw std::invalid_argument("DuckLake inline value type differs from its binding");
            copy_fixed_scalar(values, row, value);
            valid[row / 8] |= 1u << (row % 8);
          }
        }
      }
      if (is_string) {
        if (char_offset != string_sizes[col])
          throw std::logic_error("DuckLake borrowed VARCHAR changed during detach");
        tree->buffers[col][2] = chars;
        chars += char_offset;
      }
      tree->buffers[col][0] = valid;
      tree->buffers[col][1] = values;
      array.length          = count;
      array.n_buffers       = is_string ? 3 : 2;
      array.buffers         = tree->buffers[col];
      array.release         = [](ArrowArray* p) { p->release = nullptr; };
      tree->schemas[col]    = &schema;
      tree->arrays[col]     = &array;
    }
    offset += count;
    auto batch = std::make_shared<scan::owned_arrow_batch>(
      tree->schema, tree->array, capacity, capacity->staging);
    auto result = std::make_unique<scan::owned_arrow_scan_info>(
      batch, info.contract_id, program.generation, claim->key.unit_id, info.profiles, bytes);
    auto proof         = std::make_shared<inline_evidence>();
    proof->binding     = binding;
    proof->batch       = batch;
    proof->sequence    = claim->key.unit_id;
    auto deps          = std::vector(result->dependencies().begin(), result->dependencies().end());
    deps[0].visibility = std::move(proof);
    result->set_contract_payload(info.contract_id,
                                 {result->certificates().begin(), result->certificates().end()},
                                 std::move(deps));
    return result;
  }
};
size_t inline_unit_ingestible::batch_rows(size_t columns)
{
  if (columns > max_columns) throw std::invalid_argument("DuckLake inline column limit");
  // Keep the widest batch's cell and normalization bounds. Whole validity words
  // also preserve alignment of each following column's fixed values/offsets.
  return columns ? std::min(max_batch_rows, (max_batch_cells / columns / 64) * 64) : max_batch_rows;
}
uint64_t inline_unit_ingestible::batch_capacity(size_t n, size_t string_columns)
{
  if (n > max_columns || string_columns > n)
    throw std::invalid_argument("DuckLake inline column limit");
  return tree_bytes + n * column_bytes(batch_rows(n)) + (string_columns ? max_string_bytes : 0) +
         (n ? max_staging_bytes : 0);
}
sm::unit_envelope inline_unit_ingestible::envelope(size_t columns, size_t string_columns)
{
  (void)batch_capacity(columns, string_columns);
  sm::unit_envelope result{{batch_capacity(columns)}, 0};
  if (string_columns) result.temporary.push_back(max_string_bytes);
  return result;
}
inline_unit_ingestible::inline_unit_ingestible(std::shared_ptr<binding_handle> h,
                                               unit_descriptor unit,
                                               uint64_t contract,
                                               uint64_t generation,
                                               std::shared_ptr<sm::preparation_admission> admission,
                                               bool count_only,
                                               std::vector<size_t> required_columns,
                                               unit_scan_options options)
  : impl_(std::make_unique<impl>(std::move(h),
                                 std::move(unit),
                                 contract,
                                 generation,
                                 std::move(admission),
                                 count_only,
                                 std::move(required_columns),
                                 std::move(options)))
{
}
inline_unit_ingestible::~inline_unit_ingestible() = default;
void inline_unit_ingestible::preflight_for_query() { impl_->handle->validate_statement(); }
bool inline_unit_ingestible::has_processed_all_metadata() const { return impl_->done.load(); }
inline_unit_ingestible::metadata_scan_task_t inline_unit_ingestible::next_split_provider(
  io::ioctx_resolver)
{
  throw std::logic_error("DuckLake inline requires admitted claims");
}
inline_unit_ingestible::metadata_claim inline_unit_ingestible::try_claim_split(io::ioctx_resolver)
{
  if (!impl_->ledger) throw std::logic_error("DuckLake inline ledger is not attached");
  if (impl_->done) return sm::preparation_exhausted{};
  if (impl_->gate->active) return sm::preparation_pending{};
  if (impl_->sequence == UINT64_MAX)
    throw std::overflow_error("DuckLake inline sequence exhausted");
  auto state    = std::make_shared<batch_claim>();
  state->ledger = impl_->ledger;
  state->gate   = impl_->gate;
  if (!impl_->pending_identity)
    impl_->pending_identity = impl_->identities ? impl_->identities->next() : impl_->sequence;
  state->key    = {impl_->info.contract_id, *impl_->pending_identity};
  state->permit = impl_->ledger->try_acquire_unit(state->key, 0);
  if (!state->permit) return sm::preparation_pending{};
  impl_->pending_identity.reset();
  state->acquired     = true;
  impl_->gate->active = true;
  metadata_job job{
    [this, state](sm::preparation_stop_token stop) { return impl_->produce(state, stop); }};
  ++impl_->sequence;
  return job;
}
inline_unit_ingestible::preparation_source inline_unit_ingestible::make_preparation_source(
  io::ioctx_resolver)
{
  preparation_source source;
  source.execution   = sm::preparation_execution_domain::query_owner;
  source.try_claim   = [this] { return try_claim_split({}); };
  source.close       = [this] { impl_->close(); };
  source.bind_wakeup = [this](sm::preparation_wakeup wake) {
    bind_preparation_wakeup(std::move(wake));
  };
  source.coalescer = create_batch_coalescer();
  return source;
}
std::unique_ptr<scan::batch_coalescer> inline_unit_ingestible::create_batch_coalescer() const
{
  return std::make_unique<inline_coalescer>(impl_->bytes);
}
std::shared_ptr<sm::preparation_admission> inline_unit_ingestible::preparation_admission() const
{
  return impl_->admission;
}
void inline_unit_ingestible::attach_preparation_ledger(
  std::shared_ptr<sm::preparation_ledger> ledger)
{
  if (!ledger || impl_->ledger) throw std::logic_error("DuckLake inline ledger attachment invalid");
  impl_->ledger = std::move(ledger);
}
void inline_unit_ingestible::bind_preparation_wakeup(sm::preparation_wakeup wake)
{
  if (impl_->owns_gate) impl_->gate->wake = wake;
  if (impl_->ledger) impl_->ledger->bind_wakeup(std::move(wake));
}
void inline_unit_ingestible::stop_preparation() noexcept { impl_->ledger.reset(); }
void inline_unit_ingestible::finish_preparation() noexcept
{
  stop_preparation();
  if (impl_->owns_admission) impl_->admission->finish();
}
scan::ingestible_table_info const& inline_unit_ingestible::table_info() const noexcept
{
  return impl_->info;
}
std::vector<size_t> inline_unit_ingestible::materialized_column_order() const
{
  return impl_->outputs;
}
scan::materializer_contract_identity inline_unit_ingestible::materializer() const
{
  return {scan::materializer_kind::stream, "ducklake.inline-scalar.v1", impl_->validator};
}
bool inline_unit_ingestible::has_row_filter() const noexcept { return bool(impl_->predicate); }
unit_descriptor const& inline_unit_ingestible::unit() const noexcept
{
  return impl_->binding->unit;
}
scan::filtered_table inline_unit_ingestible::materialize_metadata_to_table(
  scan::scan_info const& input,
  cucascade::memory::memory_space const& space,
  ::cuda::stream_ref stream,
  bool like,
  std::shared_ptr<sirius::like_multiliteral_cache const> cache,
  expression_query_policy policy)
{
  scan::validate_split_for_gpu(impl_->info.contract_id,
                               scan::check_bit(scan::later_check::host_staged),
                               {},
                               input,
                               *impl_->validator,
                               impl_->program.generation,
                               impl_->info.profiles);
  auto const& arrow   = dynamic_cast<scan::owned_arrow_scan_info const&>(input);
  auto mr             = space.get_default_allocator();
  auto const unjoined = scan::detail::prepare_unjoined_failure();
  struct import_state {
    exec::retire_lane lane;
    scan::materialized_batch batch;
    explicit import_state(cudaStream_t stream) : lane(stream) {}
  };
  auto imported = std::make_unique<import_state>(stream.get());
  try {
    imported->batch = scan::import_arrow_host(arrow.batch(), impl_->program, imported->lane, mr);
  } catch (...) {
    // Import unwinding may enqueue output frees after commit recovery. Always
    // establish another barrier while the caller's output reservation is alive.
    if (imported->lane.quiesce() != cudaSuccess) {
      imported.release();
      std::rethrow_exception(unjoined);
    }
    throw;
  }
  if (imported->lane.quiesce() != cudaSuccess) {
    imported.release();
    std::rethrow_exception(unjoined);
  }
  if (auto counters = impl_->info.profiles->counters; counters && counters->track_units) {
    ++counters->arrow_batches_imported;
    counters->arrow_rows_imported += imported->batch.rows;
  }
  auto batch = std::move(imported->batch);
  struct selection_state {
    scan::owning_table_view input;
    std::shared_ptr<sirius::ast::node const> predicate;
    std::unique_ptr<sirius::expression_evaluator> evaluator;
  };
  auto state       = std::make_unique<selection_state>();
  state->predicate = impl_->predicate;
  scan::materialization_steps steps;
  steps.residual = [&](scan::materialized_batch& current) {
    state->input     = std::move(current.table);
    state->evaluator = std::make_unique<sirius::expression_evaluator>(
      state->predicate.get(),
      mr,
      stream,
      policy,
      sirius::expression_evaluator::default_min_ast_size,
      like,
      std::move(cache));
    auto filtered = state->evaluator->select(state->input.view());
    scan::detail::check_read_checkpoint(scan::detail::read_checkpoint::PAYLOAD_GATHER);
    scan::detail::check_read_checkpoint(scan::detail::read_checkpoint::READER_EVENT);
    state->input.record_reader_event(stream);
    current.table = scan::owning_table_view(std::move(filtered));
    current.rows  = current.table.num_rows();
  };
  steps.projection = [&](scan::materialized_batch& current) {
    if (impl_->program.input_types->empty()) return;
    state->input = std::move(current.table);
    if (impl_->outputs.empty()) {
      current.table = scan::owning_table_view(
        std::make_unique<cudf::table>(std::vector<std::unique_ptr<cudf::column>>{}));
      scan::ensure_row_count_carrier(current, stream, mr);
    } else {
      current.table =
        scan::select_scan_columns(std::move(state->input), impl_->outputs, stream, mr);
    }
  };
  try {
    batch = scan::finish_materialization(std::move(batch), impl_->program, steps);
  } catch (...) {
    scan::detail::retire_reads(state, stream, unjoined);
    throw;
  }
  return {std::move(batch.table), scan::filter_state::ROW_FILTERED_AND_PROJECTED};
}
std::unique_ptr<cudf::table> inline_unit_ingestible::post_filter_and_project(
  scan::filtered_table&& input,
  cucascade::memory::memory_space const& space,
  ::cuda::stream_ref stream,
  bool,
  std::shared_ptr<sirius::like_multiliteral_cache const>,
  std::unique_ptr<cudf::column>*,
  std::span<size_t const> elided,
  expression_query_policy)
{
  if (!elided.empty()) throw std::invalid_argument("DuckLake inline elision is not qualified");
  input.table.materialize(stream, space.get_default_allocator());
  return input.table.release(stream, space.get_default_allocator());
}
}  // namespace sirius::ducklake
