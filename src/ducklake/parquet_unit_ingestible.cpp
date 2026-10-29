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

#include "ducklake/parquet_unit_ingestible.hpp"

#include "ducklake/binding_bridge.hpp"
#include "ducklake/dv_visibility.hpp"
#include "ducklake/field_mapping.hpp"
#include "ducklake/preparation_gate.hpp"
#include "ducklake/preparation_slot.hpp"
#include "helper/type_conversions.hpp"
#include "io/io_errors.hpp"
#include "op/scan/parquet_batch_layout.hpp"
#include "op/scan/positional_visibility.hpp"
#include "op/scan/table_scan/split_validation.hpp"

#include <cudf/io/parquet_schema.hpp>
#include <cudf/table/table.hpp>

#include <cucascade/memory/memory_space.hpp>
#include <duckdb/common/exception.hpp>

namespace sirius::ducklake {
struct parquet_visibility_state {
  std::atomic<std::shared_ptr<dv_visibility const>> value;
};
namespace {
namespace scan = op::scan;
namespace sm   = scan_manager;
struct mapping_witness {
  std::atomic<std::shared_ptr<scan::field_id_mapping const>> value;
};
struct plain_info final : scan::parquet_ingestible_table_info {
  std::shared_ptr<preparation_identity> identities;
  std::shared_ptr<preparation_slot> retained_slot;
  std::shared_ptr<preparation_gate> gate;
  bool owns_admission                      = true;
  std::shared_ptr<mapping_witness> mapping = std::make_shared<mapping_witness>();
  bool allows_cached_rows() const override { return false; }
};
scan::later_check_set checks()
{
  return scan::check_bit(scan::later_check::footer_per_file) |
         scan::check_bit(scan::later_check::profile_per_file) |
         scan::check_bit(scan::later_check::schema_per_file);
}
std::unique_ptr<plain_info> make_info(unit_descriptor const& unit,
                                      std::vector<scan::field_requirement> fields,
                                      std::vector<size_t> const& required_columns,
                                      scan::scan_contract_id contract,
                                      uint64_t generation,
                                      bool count_only,
                                      unit_scan_options options)
{
  if (!contract || !generation || !unit.read_view || (fields.empty() && !count_only) ||
      (count_only && (!fields.empty() || !required_columns.empty())))
    throw std::invalid_argument(
      "DuckLake Parquet unit has an invalid bound schema or scan identity");
  validate_parquet_unit_dependencies(unit);
  auto info                    = std::make_unique<plain_info>();
  info->approximate_batch_size = options.scan_task_batch_size;
  info->contract_id            = contract;
  if (options.profiles) info->profiles = std::move(options.profiles);
  info->identities = std::move(options.identities);
  if (parquet_unit_envelope(unit).retained) {
    info->retained_slot = std::move(options.parquet_slot);
    info->gate          = std::move(options.inline_gate);
  }
  info->owns_admission      = options.owns_admission;
  info->resolved_file_paths = {unit.input.path};
  for (auto column : required_columns) {
    info->bound_types.push_back(unit.read_view->bound_types.at(column));
    info->names.push_back(unit.read_view->bound_names.at(column));
  }
  if (fields.size() != info->names.size() || fields.size() != info->bound_types.size())
    throw std::invalid_argument("DuckLake field requirements do not match the bound schema");
  for (size_t i = 0; i < fields.size(); ++i) {
    if (!scan::field_mapping_type(fields[i].type))
      throw std::invalid_argument("DuckLake plain Parquet requires qualified flat types");
    info->returned_types.push_back(sirius::from_duckdb(fields[i].type));
    info->column_ids.emplace_back(i);
  }
  validate_scan_options(options, fields.size());
  info->projected_output_count = fields.size();
  if (options.output_columns) {
    info->projected_output_count = options.output_columns->size();
    std::vector<bool> included(fields.size(), false);
    for (auto column : *options.output_columns) {
      info->projection_ids.push_back(column);
      included[column] = true;
    }
    // Every required field remains available to the residual, including fields
    // absent from the output. Repeated output columns are materialized once.
    for (size_t i = 0; i < fields.size(); ++i)
      if (!included[i]) info->projection_ids.push_back(i);
  }
  info->table_filters                  = std::move(options.filters);
  info->physical_contract.contract_id  = contract;
  info->physical_contract.profiles     = info->profiles;
  info->physical_contract.output_types = info->bound_types;
  info->field_mapping_requirements     = fields;
  info->prepare_field_mapping          = [fields, generation, witness = info->mapping](
                                  std::shared_ptr<scan::parquet_metadata const> metadata,
                                  std::string const& identity,
                                  scan::materialization_program program,
                                  std::vector<scan::field_requirement> selected) {
    std::vector<scan::field_requirement> required;
    for (auto const& requested : selected) {
      auto found = std::find_if(fields.begin(), fields.end(), [&](auto const& bound) {
        return bound.id == requested.id && bound.type == requested.type;
      });
      if (found == fields.end())
        throw std::invalid_argument("DuckLake required field is outside its bound schema");
      required.push_back(*found);
    }
    program.generation = generation;
    // Compile original provider defaults/cast rules for the shared required set.
    auto mapping = std::make_shared<scan::field_id_mapping const>(compile_field_mapping(
      std::move(metadata), identity, std::move(program), std::move(required)));
    std::shared_ptr<scan::field_id_mapping const> absent;
    if (!witness->value.compare_exchange_strong(absent, mapping))
      throw std::logic_error("DuckLake unit mapping was already sealed");
    return mapping;
  };
  return info;
}
class plain_validator final : public scan::split_validator {
 public:
  plain_validator(unit_descriptor unit,
                  uint64_t generation,
                  std::shared_ptr<parquet_visibility_state> visibility,
                  std::shared_ptr<mapping_witness> mapping)
    : unit_(std::move(unit)),
      generation_(generation),
      visibility_(std::move(visibility)),
      mapping_(std::move(mapping))
  {
  }
  bool accepts(scan::scan_info const& split) const noexcept override
  {
    return scan::parquet_split_validator()->accepts(split);
  }
  scan::split_validation_evidence inspect(scan::split_validation_context const& context,
                                          scan::scan_info const& split) const override
  {
    auto evidence       = scan::parquet_split_validator()->inspect(context, split);
    auto const& parquet = dynamic_cast<scan::parquet_split_info const&>(split);
    if (!parquet.field_mapping || parquet.field_mapping != mapping_->value.load() ||
        parquet.field_mapping->program().generation != generation_ ||
        parquet.field_mapping->program().contract != context.contract ||
        parquet.rg_slices.size() != split.dependencies().size())
      throw scan::certificate_incomplete(context.contract, {}, "DuckLake mapping evidence missing");
    auto visibility = visibility_->value.load();
    if (!visibility ||
        (unit_.deletes != delete_format::none &&
         (visibility->delete_identity.empty() ||
          (unit_.deletes == delete_format::puffin ? !visibility->validated_blobs
                                                  : !visibility->parquet_validated))) ||
        (visibility->inline_count != unit_.inline_delete_count) ||
        (!visibility->positions.empty() && !parquet.disable_filter_pushdown))
      throw scan::certificate_incomplete(context.contract, {}, "DuckLake visibility is incomplete");
    for (size_t i = 0; i < parquet.rg_slices.size(); ++i) {
      auto const& slice = parquet.rg_slices[i];
      if (slice.file_index != unit_.original_file_index || slice.file_path != unit_.input.path ||
          split.dependencies()[i].visibility != visibility ||
          visibility->data_identity != parquet.field_mapping->identity() ||
          parquet.field_mapping->identity() !=
            split.dependencies()[i].parquet_approval->object_identity)
        throw scan::certificate_incomplete(
          context.contract, {}, "DuckLake unit/visibility evidence mismatch");
    }
    evidence.generation = generation_;
    return evidence;
  }

 private:
  unit_descriptor unit_;
  uint64_t generation_;
  std::shared_ptr<parquet_visibility_state> visibility_;
  std::shared_ptr<mapping_witness> mapping_;
};
struct claimed_unit {
  std::shared_ptr<preparation_slot> slot;
  std::shared_ptr<preparation_gate> gate;
  std::shared_ptr<sm::preparation_ledger> ledger;
  sm::unit_key key;
  sm::w_permit permit;
  ~claimed_unit()
  {
    bool const acquired = bool(permit);
    if (acquired && !slot) ledger->retire_unit(key);
    permit.release();
    if (gate) {
      gate->active = false;
      if (acquired) gate->wake();
    }
  }
};
std::shared_ptr<claimed_unit> claim_unit(plain_info const& info,
                                         unit_descriptor const& unit,
                                         uint64_t preparation_id,
                                         std::shared_ptr<sm::preparation_ledger> const& ledger)
{
  auto owned    = std::make_shared<claimed_unit>();
  owned->ledger = ledger;
  owned->slot   = info.retained_slot;
  owned->key    = owned->slot ? owned->slot->key() : sm::unit_key{info.contract_id, preparation_id};
  if (owned->slot) {
    if (!info.gate) throw std::logic_error("DuckLake retained slot lacks a scan progress gate");
    bool idle = false;
    if (!info.gate->active.compare_exchange_strong(idle, true)) return {};
    owned->gate   = info.gate;
    owned->permit = owned->slot->try_acquire();
  } else {
    owned->permit = ledger->try_acquire_unit(owned->key, parquet_unit_envelope(unit).retained);
  }
  return owned->permit ? std::move(owned) : nullptr;
}
// Destroy visibility backing before returning its permit and waking another file.
struct visibility_owner {
  std::shared_ptr<claimed_unit> claim;
  std::shared_ptr<dv_visibility const> visibility;
};
}  // namespace
struct inline_delete_preparation {
  struct copied_positions {
    std::shared_ptr<claimed_unit> claim;
    scan_manager::charged_block backing;
    std::span<uint64_t> positions;
    std::shared_ptr<dv_visibility const> ordinary;
  };
  bool claimed = false;
  std::atomic<std::shared_ptr<copied_positions>> ready;
  scan_manager::preparation_wakeup wake;
};
parquet_unit_ingestible::parquet_unit_ingestible(
  std::shared_ptr<binding_handle> binding,
  unit_descriptor unit,
  std::vector<op::scan::field_requirement> fields,
  std::vector<size_t> required_columns,
  op::scan::scan_contract_id contract,
  uint64_t generation,
  std::shared_ptr<scan_manager::preparation_admission> admission,
  bool count_only,
  unit_scan_options options)
  : parquet_gpu_ingestible(make_info(unit,
                                     std::move(fields),
                                     required_columns,
                                     contract,
                                     generation,
                                     count_only,
                                     std::move(options))),
    binding_(std::move(binding)),
    unit_(std::move(unit)),
    generation_(generation),
    visibility_(std::make_shared<parquet_visibility_state>()),
    inline_deletes_(std::make_shared<inline_delete_preparation>()),
    validator_(std::make_shared<plain_validator>(
      unit_, generation_, visibility_, static_cast<plain_info const&>(table_info()).mapping)),
    admission_(std::move(admission))
{
  auto const& identities = static_cast<plain_info const&>(table_info()).identities;
  preparation_id_        = identities ? identities->next() : unit_.original_file_index;
  if (!admission_ || !admission_->decision.deferred)
    throw std::invalid_argument("DuckLake Parquet unit requires statement preparation admission");
}
scan_manager::unit_envelope parquet_unit_ingestible::envelope(unit_descriptor const& unit)
{
  return parquet_unit_envelope(unit);
}
void parquet_unit_ingestible::preflight_for_query() { binding_->validate_statement(); }
op::scan::gpu_ingestible::metadata_scan_task_t parquet_unit_ingestible::next_split_provider(
  io::ioctx_resolver)
{
  throw std::logic_error("DuckLake Parquet requires the admitted claim path");
}
op::scan::gpu_ingestible::metadata_claim parquet_unit_ingestible::try_claim_split(
  io::ioctx_resolver resolve)
{
  if (!ledger_) throw std::logic_error("DuckLake preparation ledger is not attached");
  if (has_processed_all_metadata()) return scan_manager::preparation_exhausted{};
  std::shared_ptr<inline_delete_preparation::copied_positions> copied;
  std::shared_ptr<claimed_unit> owned;
  // The owner job transfers its admitted permit; a second acquisition could
  // deadlock an attempt that has capacity for only one preparation job.
  if (unit_.inline_delete_count || unit_.deletes == delete_format::parquet) {
    copied = inline_deletes_->ready.exchange({});
    if (!copied) return scan_manager::preparation_pending{};
    owned = copied->claim;
  } else {
    owned =
      claim_unit(static_cast<plain_info const&>(table_info()), unit_, preparation_id_, ledger_);
    if (!owned) return scan_manager::preparation_pending{};
  }
  auto work = parquet_gpu_ingestible::next_split_provider(resolve);
  if (!work) throw std::logic_error("DuckLake unit cursor exhausted during claim");
  return metadata_job{[this, owned, copied, work = std::move(work), resolve = std::move(resolve)](
                        scan_manager::preparation_stop_token stop) {
    if (stop.stop_requested()) throw duckdb::InterruptException();
    auto result = work(stop);
    auto& file  = dynamic_cast<op::scan::parquet_file_scan_info&>(*result);
    if (file.file_path != unit_.input.path || file.certificates().size() != 1)
      throw std::logic_error("DuckLake Parquet unit preparation changed input");
    file.file_index   = unit_.original_file_index;
    auto certs        = std::vector(file.certificates().begin(), file.certificates().end());
    auto deps         = std::vector(file.dependencies().begin(), file.dependencies().end());
    certs[0].split_id = unit_.original_file_index;
    std::shared_ptr<dv_visibility const> visibility;
    auto data_identity = deps.at(0).parquet_approval->object_identity;
    if (unit_.deletes == delete_format::puffin) {
      auto allocator = owned->ledger->allocator(owned->permit);
      try {
        visibility = read_dv_visibility(
          unit_, file.file_metadata->num_rows, data_identity, resolve, allocator, stop);
      } catch (transparent::classified_execution_error const&) {
        throw;
      } catch (io::object_changed_error const& error) {
        throw transparent::classified_execution_error(transparent::late_failure_cause::reader_io,
                                                      error.what());
      } catch (io::descriptor_mismatch_error const& error) {
        throw transparent::classified_execution_error(transparent::late_failure_cause::reader_io,
                                                      error.what());
      } catch (std::system_error const& error) {
        if (error.code() == std::errc::operation_canceled) throw;
        throw transparent::classified_execution_error(transparent::late_failure_cause::reader_io,
                                                      error.what());
      } catch (std::bad_alloc const&) {
        throw scan_manager::preparation_resource_error(
          "DuckLake DV control allocation failed", false, std::current_exception());
      } catch (std::runtime_error const& error) {
        throw op::scan::unsupported_physical_input(table_info().contract_id,
                                                   unit_.delete_path,
                                                   op::scan::verdict_reason::evidence_missing,
                                                   error.what());
      }
    } else if (unit_.deletes == delete_format::parquet) {
      try {
        visibility =
          bind_parquet_visibility(copied->ordinary, data_identity, file.file_metadata->num_rows);
      } catch (std::runtime_error const& error) {
        throw scan::unsupported_physical_input(table_info().contract_id,
                                               unit_.delete_path,
                                               scan::verdict_reason::evidence_missing,
                                               error.what());
      }
    } else {
      auto empty           = std::make_shared<dv_visibility>();
      empty->data_identity = data_identity;
      visibility           = std::move(empty);
    }
    if (copied) {
      auto allocator = owned->ledger->allocator(owned->permit);
      try {
        visibility = combine_inline_visibility(
          visibility, copied->positions, file.file_metadata->num_rows, allocator);
      } catch (transparent::classified_execution_error const&) {
        throw;
      } catch (std::bad_alloc const&) {
        throw sm::preparation_resource_error(
          "DuckLake visibility allocation failed", false, std::current_exception());
      } catch (std::runtime_error const& error) {
        throw scan::unsupported_physical_input(table_info().contract_id,
                                               unit_.input.path,
                                               scan::verdict_reason::evidence_missing,
                                               error.what());
      }
    }
    if (owned->slot) {
      auto retained = std::make_shared<visibility_owner>(owned, std::move(visibility));
      visibility    = std::shared_ptr<dv_visibility const>(retained, retained->visibility.get());
    }
    std::shared_ptr<dv_visibility const> absent;
    if (!visibility_->value.compare_exchange_strong(absent, visibility))
      throw std::logic_error("DuckLake visibility was already sealed");
    file.disable_filter_pushdown |= !visibility->positions.empty();
    deps[0].visibility = std::move(visibility);
    file.set_contract_payload(table_info().contract_id, std::move(certs), std::move(deps));
    return result;
  }};
}
std::vector<parquet_unit_ingestible::preparation_source>
parquet_unit_ingestible::make_preparation_sources(io::ioctx_resolver resolve)
{
  std::vector<preparation_source> sources;
  if (unit_.inline_delete_count || unit_.deletes == delete_format::parquet) {
    preparation_source owner;
    owner.execution = sm::preparation_execution_domain::query_owner;
    owner.coalescer = create_batch_coalescer();
    owner.try_claim = [this]() -> metadata_claim {
      if (inline_deletes_->claimed) return sm::preparation_exhausted{};
      auto owned =
        claim_unit(static_cast<plain_info const&>(table_info()), unit_, preparation_id_, ledger_);
      if (!owned) return sm::preparation_pending{};
      inline_deletes_->claimed = true;
      return metadata_job{[this, owned](
                            sm::preparation_stop_token stop) -> std::unique_ptr<scan::scan_info> {
        auto allocator  = owned->ledger->allocator(owned->permit);
        auto result     = std::make_shared<inline_delete_preparation::copied_positions>();
        result->claim   = owned;
        result->backing = allocator.allocate_retained(unit_.inline_delete_count * sizeof(uint64_t));
        result->positions = {reinterpret_cast<uint64_t*>(result->backing.data()),
                             unit_.inline_delete_count};
        size_t offset     = 0;
        std::optional<uint64_t> after;
        while (offset < result->positions.size()) {
          if (stop.stop_requested())
            throw std::system_error(std::make_error_code(std::errc::operation_canceled));
          auto page = binding_->inline_delete_positions(
            unit_.original_file_index,
            result->positions.subspan(offset,
                                      std::min<size_t>(256, result->positions.size() - offset)),
            after);
          if (!page.size || page.size > result->positions.size() - offset)
            throw std::runtime_error("DuckLake inline delete page changed size");
          offset += page.size;
          after = result->positions[offset - 1];
          if (page.more != (offset < result->positions.size()))
            throw std::runtime_error("DuckLake inline delete coverage changed");
        }
        if (unit_.deletes == delete_format::parquet) {
          try {
            result->ordinary =
              binding_->read_parquet_deletes(unit_.original_file_index, allocator, stop);
          } catch (transparent::classified_execution_error const&) {
            throw;
          } catch (io::object_changed_error const& e) {
            throw transparent::classified_execution_error(
              transparent::late_failure_cause::reader_io, e.what());
          } catch (io::descriptor_mismatch_error const& e) {
            throw transparent::classified_execution_error(
              transparent::late_failure_cause::reader_io, e.what());
          } catch (std::system_error const&) {
            throw;
          } catch (std::bad_alloc const&) {
            throw sm::preparation_resource_error(
              "DuckLake native delete allocation failed", false, std::current_exception());
          } catch (duckdb::OutOfMemoryException const&) {
            throw sm::preparation_resource_error(
              "DuckLake native delete allocation failed", false, std::current_exception());
          } catch (duckdb::IOException const& e) {
            throw transparent::classified_execution_error(
              transparent::late_failure_cause::reader_io, e.what());
          } catch (std::exception const& e) {
            throw scan::unsupported_physical_input(table_info().contract_id,
                                                   unit_.delete_path,
                                                   scan::verdict_reason::evidence_missing,
                                                   e.what());
          }
        }
        inline_deletes_->ready.store(std::move(result));
        inline_deletes_->wake();
        return {};
      }};
    };
    // No provider reader survives the copy. The transferred backing is kept
    // until metadata consumes it, or finish_preparation releases it after drain.
    owner.close       = [] {};
    owner.bind_wakeup = [this](sm::preparation_wakeup wake) {
      inline_deletes_->wake = std::move(wake);
    };
    sources.push_back(std::move(owner));
  }
  auto metadata      = gpu_ingestible::make_preparation_source(std::move(resolve));
  metadata.coalescer = create_batch_coalescer();
  sources.push_back(std::move(metadata));
  return sources;
}
std::shared_ptr<scan_manager::preparation_admission>
parquet_unit_ingestible::preparation_admission() const
{
  return admission_;
}
void parquet_unit_ingestible::attach_preparation_ledger(
  std::shared_ptr<scan_manager::preparation_ledger> ledger)
{
  if (ledger_ || !ledger) throw std::logic_error("DuckLake preparation ledger attachment invalid");
  ledger_ = std::move(ledger);
}
void parquet_unit_ingestible::bind_preparation_wakeup(scan_manager::preparation_wakeup wake)
{
  if (ledger_) ledger_->bind_wakeup(std::move(wake));
}
void parquet_unit_ingestible::stop_preparation() noexcept { ledger_.reset(); }
void parquet_unit_ingestible::finish_preparation() noexcept
{
  stop_preparation();
  visibility_->value.store({});
  inline_deletes_->ready.store({});
  if (static_cast<plain_info const&>(table_info()).owns_admission) admission_->finish();
}
op::scan::materializer_contract_identity parquet_unit_ingestible::materializer() const
{
  return {op::scan::materializer_kind::parquet, "ducklake.plain-parquet-unit.v1", validator_};
}
op::scan::filtered_table parquet_unit_ingestible::materialize_metadata_to_table(
  op::scan::scan_info const& info,
  cucascade::memory::memory_space const& space,
  ::cuda::stream_ref stream,
  bool like,
  std::shared_ptr<sirius::like_multiliteral_cache const> cache,
  expression_query_policy policy)
{
  op::scan::validate_split_for_gpu(
    table_info().contract_id, checks(), {}, info, *validator_, generation_, table_info().profiles);
  auto visibility = visibility_->value.load();
  auto base       = parquet_gpu_ingestible::materialize_metadata_to_table(
    info, space, stream, like, std::move(cache), policy);
  if (visibility->positions.empty()) return base;
  if (base.state != op::scan::filter_state::UNFILTERED)
    throw std::logic_error("DuckLake DV requires unfiltered original file rows");
  auto const& split = dynamic_cast<op::scan::parquet_split_info const&>(info);
  auto layout       = op::scan::build_batch_layout(split);
  rmm::device_async_resource_ref mr(space.get_default_allocator());
  auto table = base.table.release(stream, mr);
  if (!table) throw std::logic_error("DuckLake DV requires an owned decoded table");
  auto filtered = op::scan::apply_positional_visibility(
    std::move(table),
    layout,
    [&](std::string const& path) {
      if (path != unit_.input.path) throw std::logic_error("DuckLake DV row domain changed file");
      return visibility->positions;
    },
    stream,
    mr);
  return {op::scan::owning_table_view{std::move(filtered)}, op::scan::filter_state::UNFILTERED};
}
}  // namespace sirius::ducklake
