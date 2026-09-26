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

#include "op/scan/duckdb_native_gpu_ingestible.hpp"
#include "op/scan/gpu_ingestible_types.hpp"
#include "op/scan/parquet_gpu_ingestible.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "op/scan/sirius_gpu_scan_operator_data.hpp"
#include "op/scan/table_scan/bound_read_view.hpp"
#include "op/scan/table_scan/scan_contract.hpp"
#include "sirius_context.hpp"
#include "test_utils.hpp"
#include "transparent/read_view_registry.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/duck_table_entry.hpp>
#include <duckdb/execution/physical_plan_generator.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/planner/table_filter.hpp>
#include <duckdb/storage/data_table.hpp>
#include <duckdb/storage/storage_manager.hpp>
#include <io/kvikio/kvikio_context.hpp>
#include <io/sirius_datasource.hpp>
#include <unistd.h>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/parquet_fixture_utils.hpp>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <future>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using namespace sirius::op::scan;

std::filesystem::path project_root()
{
#ifdef SIRIUS_PROJECT_ROOT
  return std::filesystem::path{SIRIUS_PROJECT_ROOT};
#else
  return std::filesystem::current_path();
#endif
}

void exec_ok(duckdb::Connection& con, std::string const& query)
{
  auto result = con.Query(query);
  REQUIRE(result);
  if (result->HasError()) { INFO(result->GetError()); }
  REQUIRE_FALSE(result->HasError());
}

struct native_database : sirius::test::GpuExecutionFixture {
  std::filesystem::path path = temp_db_path;
  std::unique_ptr<duckdb::SiriusContext::StandaloneQueryScope> window;
  decltype(con)& connection = con;

  void lease(duckdb::AttachedDatabase& database)
  {
    auto context = sirius::test::get_registered_sirius_context(*con);
    if (!window) {
      window = std::make_unique<duckdb::SiriusContext::StandaloneQueryScope>(
        *context, *con->context, "native_metadata_test");
    }
    if (!context->get_scan_manager().holds_checkpoint_key(database)) {
      context->get_scan_manager().acquire_checkpoint_key(window->query_id(), database);
    }
  }
};

struct temporary_directory {
  std::filesystem::path path = std::filesystem::temp_directory_path() /
                               ("sirius_split_certificate_files_" + std::to_string(::getpid()) +
                                "_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)));

  temporary_directory() { std::filesystem::create_directories(path); }
  ~temporary_directory()
  {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
};

std::unique_ptr<duckdb_native_ingestible_table_info> native_info(native_database& fixture,
                                                                 scan_contract_id contract_id,
                                                                 bool all_pruned = false)
{
  auto& con = *fixture.connection;
  if (!con.context->transaction.HasActiveTransaction()) { exec_ok(con, "BEGIN TRANSACTION"); }
  auto& context = *con.context;
  auto& catalog = duckdb::Catalog::GetCatalog(context, "");
  duckdb::CatalogTransaction transaction(catalog, context);
  auto& schema = catalog.GetSchema(transaction, "main");
  auto entry   = schema.GetEntry(transaction, duckdb::CatalogType::TABLE_ENTRY, "items");
  REQUIRE(entry);
  auto& storage = entry->Cast<duckdb::DuckTableEntry>().GetStorage();
  fixture.lease(storage.GetAttached());

  auto info          = std::make_unique<duckdb_native_ingestible_table_info>();
  info->contract_id  = contract_id;
  info->storage      = &storage;
  info->context      = con.context.get();
  info->db_path      = storage.GetAttached().GetStorageManager().GetDBPath();
  info->catalog_name = "memory";
  info->schema_name  = "main";
  info->table_name   = "items";
  info->projected_cols.push_back({duckdb::StorageIndex(0), false});
  info->column_ids.push_back(duckdb::ColumnIndex(0));
  info->names.push_back("id");
  auto type = sirius::logical_type::make(sirius::type_id::INTEGER);
  info->projected_types.push_back(type);
  info->returned_types.push_back(type);
  info->output_types.push_back(type);
  if (all_pruned) {
    info->table_filters             = duckdb::make_uniq<duckdb::TableFilterSet>();
    info->table_filters->filters[0] = duckdb::make_uniq<duckdb::ConstantFilter>(
      duckdb::ExpressionType::COMPARE_LESSTHAN, duckdb::Value::INTEGER(-1));
  }
  return info;
}

std::unique_ptr<parquet_ingestible_table_info> parquet_info(scan_contract_id contract_id)
{
  auto info                 = std::make_unique<parquet_ingestible_table_info>();
  info->contract_id         = contract_id;
  info->resolved_file_paths = {
    (project_root() / "test/cpp/integration/data/parquet/nation.parquet").string()};
  info->names = {"n_nationkey", "n_name", "n_regionkey", "n_comment"};
  info->returned_types.push_back(sirius::logical_type::make(sirius::type_id::INTEGER));
  info->returned_types.push_back(sirius::logical_type::make(sirius::type_id::VARCHAR));
  info->returned_types.push_back(sirius::logical_type::make(sirius::type_id::INTEGER));
  info->returned_types.push_back(sirius::logical_type::make(sirius::type_id::VARCHAR));
  info->column_ids.push_back(duckdb::ColumnIndex(0));
  info->scan_output_arity = 1;
  return info;
}

struct parquet_certificate_fixture {
  static constexpr scan_contract_id contract_id = 53;
  temporary_directory files;
  std::shared_ptr<sirius::io::kvikio_context> ioctx =
    std::make_shared<sirius::io::kvikio_context>();
  std::shared_ptr<gpu_ingestible> ingestible;

  parquet_certificate_fixture()
  {
    auto info         = parquet_info(contract_id);
    auto const source = info->resolved_file_paths.front();
    info->resolved_file_paths.clear();
    for (auto const* name : {"first.parquet", "second.parquet", "third.parquet"}) {
      auto path = (files.path / name).string();
      std::filesystem::copy_file(source, path);
      info->resolved_file_paths.push_back(std::move(path));
    }
    info->approximate_batch_size = 64 * 1024 * 1024;
    ingestible                   = make_ingestible(std::move(info));
  }

  std::unique_ptr<parquet_file_scan_info> next_file()
  {
    auto provider = ingestible->next_split_provider(
      [this](std::string_view) -> std::shared_ptr<sirius::io::ioctx> { return ioctx; });
    REQUIRE(provider);
    auto info  = provider();
    auto* file = dynamic_cast<parquet_file_scan_info*>(info.get());
    REQUIRE(file);
    REQUIRE(file->certificates().size() == 1);
    REQUIRE(file->dependencies().size() == 1);
    REQUIRE_FALSE(file->row_groups.empty());
    info.release();
    return std::unique_ptr<parquet_file_scan_info>(file);
  }
};

// Duplicate only the metadata under test; each input keeps fresh prefetch state.
std::unique_ptr<parquet_file_scan_info> copy_parquet_file_metadata(
  parquet_file_scan_info const& file)
{
  auto copy                     = std::make_unique<parquet_file_scan_info>();
  copy->file_metadata           = file.file_metadata;
  copy->file_path               = file.file_path;
  copy->file_index              = file.file_index;
  copy->datasource              = file.datasource;
  copy->row_groups              = file.row_groups;
  copy->reader_options          = file.reader_options;
  copy->partition_values        = file.partition_values;
  copy->disable_filter_pushdown = file.disable_filter_pushdown;
  copy->carrier_unavailable     = file.carrier_unavailable;
  copy->set_contract_payload(
    file.contract_id(),
    std::vector<split_materializer_certificate>(file.certificates().begin(),
                                                file.certificates().end()),
    std::vector<split_dependencies>(file.dependencies().begin(), file.dependencies().end()));
  return copy;
}

void check_parquet_file_split(scan_info const& info,
                              parquet_file_scan_info const& file,
                              uint64_t split_id)
{
  auto const* split = dynamic_cast<parquet_split_info const*>(&info);
  REQUIRE(split);
  REQUIRE(split->rg_slices.size() == 1);
  REQUIRE(split->certificates().size() == 1);
  REQUIRE(split->dependencies().size() == 1);
  CHECK(split->contract_id() == file.contract_id());
  CHECK(split->certificates()[0].input_identity == file.certificates()[0].input_identity);
  CHECK(split->certificates()[0].split_id == split_id);
  CHECK(split->dependencies()[0].footer == file.file_metadata);
  CHECK(split->rg_slices[0].file_metadata == file.file_metadata);
  CHECK(split->rg_slices[0].file_path == file.file_path);
  CHECK(split->rg_slices[0].file_index == file.file_index);
  std::vector<cudf::size_type> row_groups;
  for (auto const& group : file.row_groups)
    row_groups.push_back(group.index);
  CHECK(split->rg_slices[0].row_group_indices == row_groups);
  CHECK(split->estimated_bytes() == file.estimated_bytes());
  CHECK(split->estimated_working_set_bytes() == file.estimated_working_set_bytes());
  CHECK(split->partition_values == file.partition_values);
  CHECK(split->disable_filter_pushdown == file.disable_filter_pushdown);
  CHECK(split->reader_options == file.reader_options);
  REQUIRE_NOTHROW(validate_split_for_gpu(
    file.contract_id(),
    check_bit(later_check::footer_per_file) | check_bit(later_check::profile_per_file),
    {},
    *split));
}

bound_read_view test_view()
{
  bound_read_identity identity;
  identity.source      = {"read_parquet", source_kind::parquet_local, "duckdb.read_parquet.v1"};
  identity.data_view   = file_inventory{1};
  identity.bound_types = {duckdb::LogicalType::INTEGER};
  identity.bound_names = {"id"};
  bound_read_view view;
  std::vector<std::string> paths{"one.parquet"};
  view.identity = make_bound_read_identity(std::move(identity), paths);
  return view;
}

class certificate_test_split final : public scan_info {
 public:
  explicit certificate_test_split(scan_contract_id id, bool omit_dependency = false) : id_(id)
  {
    certificates_.push_back(split_materializer_certificate{
      id, 7, "one.parquet|footer=128", 0, check_bit(later_check::host_staged)});
    if (!omit_dependency) dependencies_.emplace_back();
  }

  [[nodiscard]] std::span<split_materializer_certificate const> certificates() const override
  {
    return certificates_;
  }

  [[nodiscard]] std::span<split_dependencies const> dependencies() const override
  {
    return dependencies_;
  }

  [[nodiscard]] scan_contract_id contract_id() const override { return id_; }

 private:
  scan_contract_id id_;
  std::vector<split_materializer_certificate> certificates_;
  std::vector<split_dependencies> dependencies_;
};

}  // namespace

TEST_CASE("Scan contract handles are process-wide monotonic and name immutable entries",
          "[scan][certificate]")
{
  sirius::transparent::read_view_registry first_registry;
  sirius::transparent::read_view_registry second_registry;
  auto view = std::make_shared<bound_read_view const>(test_view());

  auto const first  = allocate_scan_contract(first_registry,
                                            /*window_id=*/11,
                                            /*finalize_generation=*/0,
                                            /*scan_node_id=*/3,
                                            view,
                                            column_requirements{},
                                            predicate_contract{},
                                             {materializer_kind::parquet, "parquet.v1"});
  auto const second = allocate_scan_contract(second_registry,
                                             /*window_id=*/12,
                                             /*finalize_generation=*/0,
                                             /*scan_node_id=*/4,
                                             view,
                                             column_requirements{},
                                             predicate_contract{},
                                             {materializer_kind::parquet, "parquet.v1"});

  REQUIRE(first != 0);
  REQUIRE(second > first);
  REQUIRE(contract_of(first_registry, first).contract_id == first);
  REQUIRE(contract_of(first_registry, first).scan_node_id == 3);
  REQUIRE(first_registry.entry_for_scan_node(3).contract.contract_id == first);
  REQUIRE(first_registry.entries().size() == 1);
  CHECK(first_registry.entries()[0].window_id == 11);
  CHECK(first_registry.entries()[0].finalize_generation == 0);
  CHECK(first_registry.entries()[0].eligibility.verdict == eligibility_verdict::not_evaluated);
  CHECK(first_registry.entries()[0].eligibility.evidence_scope == certificate_evidence_scope::none);
  REQUIRE_THROWS(contract_of(first_registry, second));
  REQUIRE_THROWS(first_registry.entry_for_scan_node(4));
  REQUIRE_THROWS_WITH(allocate_scan_contract(first_registry,
                                             /*window_id=*/11,
                                             /*finalize_generation=*/0,
                                             /*scan_node_id=*/3,
                                             view,
                                             column_requirements{},
                                             predicate_contract{},
                                             {materializer_kind::parquet, "parquet.v1"}),
                      Catch::Matchers::ContainsSubstring("duplicate scan node"));
}

TEST_CASE("Split certificates expose parallel dependencies and unevaluated eligibility",
          "[scan][certificate]")
{
  certificate_test_split split(41);
  REQUIRE(split.contract_id() == 41);
  REQUIRE(split.certificates().size() == 1);
  REQUIRE(split.dependencies().size() == split.certificates().size());
  CHECK(split.certificates()[0].contract_id == split.contract_id());
  CHECK(split.certificates()[0].input_identity == "one.parquet|footer=128");

  eligibility_certificate eligibility;
  eligibility.contract_id = split.contract_id();
  CHECK(eligibility.verdict == eligibility_verdict::not_evaluated);
  CHECK(eligibility.evidence_scope == certificate_evidence_scope::none);
}

TEST_CASE("Certificate diagnostic reports retained checks and profiles without identity",
          "[scan][certificate][diagnostic]")
{
  scan_info split;
  auto profiles = std::make_shared<physical_profile_table>();
  physical_profile profile;
  profile.storage_version = 7;
  profile.columns.push_back(
    physical_column_profile{.type = 2, .data_codecs = 4, .validity_or_encodings = 8});
  auto id = profiles->add(std::move(profile));
  split_materializer_certificate certificate;
  certificate.contract_id    = 41;
  certificate.input_identity = "sensitive-source-path";
  certificate.profile        = id;
  certificate.validation =
    check_bit(later_check::segments_per_range) | check_bit(later_check::matrix_per_range);
  split_dependencies dependency;
  dependency.profiles = profiles;
  split.set_contract_payload(41, {certificate}, {dependency});
  auto text = format_split_certificates_for_dump(split);
  CHECK(text.find("validation=segments_per_range,matrix_per_range") != std::string::npos);
  CHECK(text.find("storage_version:7") != std::string::npos);
  CHECK(text.find("data_codecs:4") != std::string::npos);
  CHECK(text.find("sensitive-source-path") == std::string::npos);

  certificate.profile = 0;
  split.set_contract_payload(41, {certificate}, {dependency});
  CHECK(format_split_certificates_for_dump(split).find("profile=missing") != std::string::npos);
}

TEST_CASE("Scan contract later checks are materializer-specific", "[scan][certificate]")
{
  struct expectation {
    materializer_kind kind;
    later_check_set later_checks;
  };
  auto const expectations = std::vector<expectation>{
    {materializer_kind::duckdb_native, check_bit(later_check::segments_per_range)},
    {materializer_kind::parquet, check_bit(later_check::footer_per_file)},
    {materializer_kind::iceberg, check_bit(later_check::footer_per_file)},
    {materializer_kind::stream, {}},
  };

  for (std::size_t index = 0; index < expectations.size(); ++index) {
    sirius::transparent::read_view_registry registry;
    auto const& expected = expectations[index];
    auto const id        = allocate_scan_contract(registry,
                                           /*window_id=*/101,
                                           /*finalize_generation=*/0,
                                           /*scan_node_id=*/index,
                                           std::make_shared<bound_read_view const>(test_view()),
                                                  {},
                                                  {},
                                                  {expected.kind, "test.v1"});
    CHECK(registry.entry(id).eligibility.later_checks == expected.later_checks);
  }
}

TEST_CASE("Certification budget charges only explicit added work", "[scan][certificate]")
{
  certification_budget budget(std::chrono::milliseconds{5}, 1024, true);
  budget.charge(std::chrono::microseconds{2500}, 512);
  CHECK_FALSE(budget.exceeded());
  std::this_thread::sleep_for(std::chrono::milliseconds{6});
  CHECK_FALSE(budget.exceeded());
  budget.charge(std::chrono::microseconds{2500}, 512);
  CHECK_FALSE(budget.exceeded());  // Exact allowance is admitted.
  budget.charge(std::chrono::microseconds{1}, 0);
  CHECK(budget.exceeded());
  CHECK(budget.declines());
  CHECK(budget.consumed().added_time_us == 5001);

  certification_budget production(std::chrono::milliseconds{5}, 1024);
  production.charge(std::chrono::microseconds{0}, 1025);
  CHECK(production.exceeded());
  CHECK_FALSE(production.declines());
}

TEST_CASE("Fresh Parquet slices carry physical input certificates", "[scan][certificate]")
{
  constexpr scan_contract_id contract_id = 51;
  auto const combine_files               = GENERATE(false, true);
  auto const warm_cache                  = GENERATE(false, true);
  auto const iceberg_schema              = GENERATE(false, true);
  CAPTURE(combine_files, warm_cache, iceberg_schema);
  temporary_directory files;
  auto make_info = [&] {
    auto result = parquet_info(contract_id);
    if (iceberg_schema) {
      result->resolved_file_paths = {
        (project_root() /
         "test/cpp/integration/data/iceberg_conformance/rename_col/conf/rename_col/data/"
         "00000-0-571c62d9-f336-4fc6-b0c0-6ee9232f603c.parquet")
          .string()};
      result->names          = {"id", "value"};
      result->returned_types = {sirius::logical_type::make(sirius::type_id::INTEGER),
                                sirius::logical_type::make(sirius::type_id::VARCHAR)};
      result->physical_schema =
        iceberg_table_schema{{{"id", 1, "INTEGER"}, {"value", 2, "VARCHAR"}}};
    }
    return result;
  };
  auto info         = make_info();
  auto const source = info->resolved_file_paths.front();
  std::vector<std::string> paths;
  for (auto const* name : {"first.parquet", "second.parquet"}) {
    auto path = (files.path / name).string();
    std::filesystem::copy_file(source, path);
    paths.push_back(std::move(path));
  }
  info->resolved_file_paths    = paths;
  info->approximate_batch_size = 64 * 1024 * 1024;
  auto ingestible              = make_ingestible(std::move(info));
  auto ioctx                   = std::make_shared<sirius::io::kvikio_context>();
  if (warm_cache) {
    auto warm_info                 = make_info();
    warm_info->resolved_file_paths = paths;
    auto warmer                    = make_ingestible(std::move(warm_info));
    for (auto const& path : paths) {
      auto provider = warmer->next_split_provider(
        [ioctx](std::string_view) -> std::shared_ptr<sirius::io::ioctx> { return ioctx; });
      REQUIRE(provider);
      REQUIRE(provider());
      REQUIRE(ioctx->open_datasource(path)->metadata());
    }
  }
  auto coalescer = ingestible->create_batch_coalescer();
  std::vector<std::unique_ptr<scan_info>> splits;
  std::vector<std::string> identities;
  auto collect = [&](auto emitted) {
    for (auto& split : emitted)
      splits.push_back(std::move(split));
  };
  for (auto const& path : paths) {
    auto provider = ingestible->next_split_provider(
      [ioctx](std::string_view) -> std::shared_ptr<sirius::io::ioctx> { return ioctx; });
    REQUIRE(provider);
    auto file = provider();
    REQUIRE(file);
    REQUIRE(file->certificates().size() == 1);
    REQUIRE(file->dependencies().size() == 1);
    auto const identity = file->certificates().front().input_identity;
    CHECK(identity.find(path + "|footer=") == 0);
    CHECK(identity.find("size=") == std::string::npos);
    CHECK(identity.find("etag=") == std::string::npos);
    identities.push_back(identity);
    collect(coalescer->push(std::move(file)));
    if (!combine_files) collect(coalescer->flush());
  }
  collect(coalescer->flush());
  REQUIRE(splits.size() == (combine_files ? 1 : 2));
  std::size_t position = 0;
  for (auto const& split : splits) {
    auto const* parquet_split = dynamic_cast<parquet_split_info const*>(split.get());
    REQUIRE(parquet_split);
    REQUIRE(split->contract_id() == contract_id);
    REQUIRE(split->certificates().size() == parquet_split->rg_slices.size());
    REQUIRE(split->dependencies().size() == split->certificates().size());
    REQUIRE(parquet_split->rg_slices.size() == (combine_files ? 2 : 1));
    for (std::size_t slice = 0; slice < parquet_split->rg_slices.size(); ++slice) {
      auto const& certificate = split->certificates()[slice];
      CHECK(certificate.contract_id == contract_id);
      REQUIRE(position < identities.size());
      CHECK(certificate.input_identity == identities[position++]);
      CHECK(certificate.input_identity.find(parquet_split->rg_slices[slice].file_path +
                                            "|footer=") == 0);
      auto const& dependency = split->dependencies()[slice];
      auto const& group      = parquet_split->rg_slices[slice];
      REQUIRE(dependency.datasource);
      REQUIRE(group.datasource);
      CHECK(dependency.footer == group.file_metadata);
      CHECK(&dependency.datasource->get_io_object() == &group.datasource->get_io_object());
      auto required =
        check_bit(later_check::footer_per_file) | check_bit(later_check::profile_per_file);
      if (iceberg_schema) required |= check_bit(later_check::schema_per_file);
      CHECK(certificate.validation == required);
      REQUIRE(dependency.profiles);
      REQUIRE(certificate.profile != 0);
      CHECK_FALSE(dependency.profiles->get(certificate.profile).columns.empty());
    }
  }
  CHECK(position == paths.size());
}

TEST_CASE("Parquet coalescing rejects malformed files without changing pending work",
          "[scan][certificate][parquet_certificate]")
{
  auto const defect  = std::string{GENERATE("missing", "extra", "foreign", "uncertified_foreign")};
  auto const pending = std::string{GENERATE("empty", "rows", "pruned")};
  auto const all_pruned = GENERATE(false, true);
  CAPTURE(defect, pending, all_pruned);
  parquet_certificate_fixture fixture;
  auto coalescer = fixture.ingestible->create_batch_coalescer();
  auto saved     = fixture.next_file();
  if (pending == "pruned") saved->row_groups.clear();
  if (pending != "empty") { REQUIRE(coalescer->push(copy_parquet_file_metadata(*saved)).empty()); }

  auto malformed = fixture.next_file();
  if (all_pruned) malformed->row_groups.clear();
  // A rejected file must not flush a pending batch or replace its metadata.
  malformed->partition_values        = {"rejected"};
  malformed->disable_filter_pushdown = !saved->disable_filter_pushdown;
  malformed->reader_options =
    std::make_shared<cudf::io::parquet_reader_options>(*saved->reader_options);
  auto id = fixture.contract_id;
  std::vector<split_materializer_certificate> certificates(malformed->certificates().begin(),
                                                           malformed->certificates().end());
  std::vector<split_dependencies> dependencies(malformed->dependencies().begin(),
                                               malformed->dependencies().end());
  if (defect == "missing" || defect == "uncertified_foreign") {
    certificates.clear();
    dependencies.clear();
  } else if (defect == "extra") {
    certificates.push_back(certificates.front());
    dependencies.push_back(dependencies.front());
  }
  if (defect == "foreign" || defect == "uncertified_foreign") {
    ++id;
    for (auto& certificate : certificates)
      certificate.contract_id = id;
  }
  malformed->set_contract_payload(id, std::move(certificates), std::move(dependencies));
  REQUIRE_THROWS_AS(coalescer->push(std::move(malformed)), std::invalid_argument);

  auto preserved = coalescer->flush();
  REQUIRE(preserved.size() == (pending == "empty" ? 0 : 1));
  if (!preserved.empty()) check_parquet_file_split(*preserved.front(), *saved, 1);
  REQUIRE(coalescer->flush().empty());

  auto next = fixture.next_file();
  REQUIRE(coalescer->push(copy_parquet_file_metadata(*next)).empty());
  auto reused = coalescer->flush();
  REQUIRE(reused.size() == 1);
  check_parquet_file_split(*reused.front(), *next, pending == "empty" ? 1 : 2);
  CHECK(coalescer->flush().empty());
}

TEST_CASE("Parquet split validation requires one matching certificate per slice",
          "[scan][certificate][parquet_certificate]")
{
  auto const defect = std::string{GENERATE("missing", "partial", "extra", "swapped_footers")};
  CAPTURE(defect);
  parquet_certificate_fixture fixture;
  auto coalescer = fixture.ingestible->create_batch_coalescer();
  REQUIRE(coalescer->push(fixture.next_file()).empty());
  REQUIRE(coalescer->push(fixture.next_file()).empty());
  auto splits = coalescer->flush();
  REQUIRE(splits.size() == 1);
  auto* split = dynamic_cast<parquet_split_info*>(splits.front().get());
  REQUIRE(split);
  REQUIRE(split->rg_slices.size() == 2);
  REQUIRE(split->certificates().size() == 2);
  REQUIRE(split->dependencies().size() == 2);
  auto const required =
    check_bit(later_check::footer_per_file) | check_bit(later_check::profile_per_file);
  REQUIRE_NOTHROW(validate_split_for_gpu(fixture.contract_id, required, {}, *split));

  std::vector<split_materializer_certificate> certificates(split->certificates().begin(),
                                                           split->certificates().end());
  std::vector<split_dependencies> dependencies(split->dependencies().begin(),
                                               split->dependencies().end());
  if (defect == "missing") {
    certificates.clear();
    dependencies.clear();
  } else if (defect == "partial") {
    certificates.pop_back();
    dependencies.pop_back();
  } else if (defect == "extra") {
    certificates.push_back(certificates.front());
    dependencies.push_back(dependencies.front());
  } else {
    REQUIRE(dependencies[0].footer != dependencies[1].footer);
    std::swap(dependencies[0].footer, dependencies[1].footer);
  }
  split->set_contract_payload(
    fixture.contract_id, std::move(certificates), std::move(dependencies));
  REQUIRE_THROWS_AS(validate_split_for_gpu(fixture.contract_id, required, {}, *split),
                    std::runtime_error);
}

TEST_CASE("Parquet certificates include physical-original file evidence after comparison",
          "[scan][certificate][read_view]")
{
  auto const source =
    (project_root() / "test/cpp/integration/data/parquet/nation.parquet").string();
  temporary_directory files;
  auto const first  = (files.path / "z-nation.parquet").string();
  auto const second = (files.path / "a-nation.parquet").string();
  std::filesystem::copy_file(source, first);
  std::filesystem::copy_file(source, second);
  std::vector<std::string> paths{first, second};
  auto const sorted        = GENERATE(false, true);
  auto const with_evidence = GENERATE(false, true);
  CAPTURE(sorted, with_evidence);
  if (sorted) { std::swap(paths[0], paths[1]); }
  bound_read_identity identity;
  identity.source      = {"read_parquet", source_kind::parquet_local, "duckdb.read_parquet.v1"};
  identity.data_view   = file_inventory{2};
  identity.bound_types = {duckdb::LogicalType::INTEGER};
  identity.bound_names = {"id"};
  bound_read_view candidate;
  candidate.identity = make_bound_read_identity(std::move(identity), paths);

  auto registry = std::make_shared<sirius::transparent::read_view_registry>();
  auto const contract_id =
    allocate_scan_contract(*registry,
                           /*window_id=*/std::nullopt,
                           /*finalize_generation=*/1,
                           /*scan_node_id=*/9,
                           std::make_shared<bound_read_view const>(candidate),
                           column_requirements{},
                           predicate_contract{},
                           {materializer_kind::parquet, "parquet.v1"});
  bound_read_view physical = candidate;
  auto evidence            = std::make_shared<file_evidence_arrays>();
  // Evidence arrays use canonical sorted-path order: a-nation, then z-nation.
  evidence->size                  = {1234, 4321};
  evidence->last_modified         = {5678, 8765};
  evidence->size_present          = {1, 1};
  evidence->last_modified_present = {1, 1};
  evidence->etag                  = {"a-tag", "z-tag"};
  if (with_evidence) {
    physical.evidence = evidence;
    physical.depth    = evidence_depth::path_size_and_tag;
  }
  std::vector<bound_read_view> physical_original{physical};

  auto info                 = parquet_info(contract_id);
  info->resolved_file_paths = paths;
  info->read_views          = registry;
  auto ingestible           = make_ingestible(std::move(info));
  // Finalize can publish evidence after constructing the ingestible, before metadata dispatch.
  registry->publish_correspondence(
    certificate_evidence_scope::binding_correspondence, "table_index", physical_original);
  auto ioctx   = std::make_shared<sirius::io::kvikio_context>();
  auto resolve = [ioctx](std::string_view) -> std::shared_ptr<sirius::io::ioctx> { return ioctx; };
  auto first_provider  = ingestible->next_split_provider(resolve);
  auto second_provider = ingestible->next_split_provider(resolve);
  REQUIRE(first_provider);
  REQUIRE(second_provider);
  // Metadata tasks share the ingestible and may initialize its evidence index concurrently.
  auto first_file  = std::async(std::launch::async, std::move(first_provider));
  auto second_file = std::async(std::launch::async, std::move(second_provider));
  std::vector<std::unique_ptr<scan_info>> splits;
  splits.push_back(first_file.get());
  splits.push_back(second_file.get());
  for (std::size_t i = 0; i < splits.size(); ++i) {
    REQUIRE(splits[i]);
    REQUIRE(splits[i]->certificates().size() == 1);
    auto const& input_identity = splits[i]->certificates().front().input_identity;
    CHECK(input_identity.find(paths[i] + "|footer=") == 0);
    if (with_evidence) {
      auto const is_first = paths[i] == first;
      CHECK(input_identity.find(is_first ? "|size=4321" : "|size=1234") != std::string::npos);
      CHECK(input_identity.find(is_first ? "|last_modified=8765" : "|last_modified=5678") !=
            std::string::npos);
      CHECK(input_identity.find(is_first ? "|etag=z-tag" : "|etag=a-tag") != std::string::npos);
    } else {
      CHECK(input_identity.find("|size=") == std::string::npos);
      CHECK(input_identity.find("|last_modified=") == std::string::npos);
      CHECK(input_identity.find("|etag=") == std::string::npos);
    }
  }
  auto const& entry = registry->entry(contract_id);
  CHECK(entry.eligibility.depth == physical.depth);
  CHECK(entry.eligibility.correspondence == "table_index");
  CHECK(entry.physical_evidence == physical.evidence);
}

TEST_CASE("Local glob evidence reaches Parquet certificates through physical comparison",
          "[scan][certificate][read_view][integration]")
{
  native_database fixture;
  auto& con = *fixture.connection;
  exec_ok(con, "SET gpu_execution=false");
  temporary_directory files;
  auto const source = project_root() / "test/cpp/integration/data/parquet/nation.parquet";
  for (auto const* name : {"a.parquet", "z.parquet"})
    std::filesystem::copy_file(source, files.path / name);
  exec_ok(con, "BEGIN");
  auto plan = con.ExtractPlan("SELECT n_nationkey FROM read_parquet(" +
                              sirius::test::sql_literal((files.path / "*.parquet").string()) + ")");
  logical_bound_read_view_capture logical;
  logical.views = capture_bound_read_views(*plan, *con.context);
  REQUIRE(logical.views.size() == 1);
  auto candidate = logical.views.front().view;
  duckdb::PhysicalPlanGenerator generator(*con.context);
  auto physical_plan = generator.Plan(std::move(plan));
  auto physical      = capture_bound_read_views(physical_plan->Root(), *con.context);
  REQUIRE(physical.size() == 1);
  auto registry   = std::make_shared<sirius::transparent::read_view_registry>();
  auto const id   = allocate_scan_contract(*registry,
                                         std::nullopt,
                                         1,
                                         logical.views.front().table_index,
                                         std::make_shared<bound_read_view const>(candidate),
                                         column_requirements{},
                                         predicate_contract{},
                                           {materializer_kind::parquet, "parquet.v1"},
                                           {},
                                         logical.views.front().table_index);
  auto comparison = sirius::transparent::compare_read_views(
    sirius::transparent::candidate_origin::copy, &logical, physical, *registry);
  INFO(sirius::transparent::describe_read_view_mismatch(comparison));
  REQUIRE(comparison.equal);
  registry->publish_correspondence(
    certificate_evidence_scope::binding_correspondence, comparison.correspondence, physical);
  auto const evidence = registry->entry(id).physical_evidence;
  REQUIRE(evidence);
  REQUIRE(evidence->size.size() == 2);
  auto info = parquet_info(id);
  // Consume the glob's physical evidence in reverse path order.
  info->resolved_file_paths = {(files.path / "z.parquet").string(),
                               (files.path / "a.parquet").string()};
  info->read_views          = registry;
  auto ingestible           = make_ingestible(std::move(info));
  auto ioctx                = std::make_shared<sirius::io::kvikio_context>();
  for (auto const index : {1, 0}) {
    auto provider = ingestible->next_split_provider(
      [ioctx](std::string_view) -> std::shared_ptr<sirius::io::ioctx> { return ioctx; });
    REQUIRE(provider);
    auto file = provider();
    REQUIRE(file);
    REQUIRE(file->certificates().size() == 1);
    auto const& identity = file->certificates().front().input_identity;
    REQUIRE(evidence->size_present[index]);
    CHECK(evidence->size[index] == std::filesystem::file_size(source));
    CHECK(identity.find("|size=" + std::to_string(evidence->size[index])) != std::string::npos);
    REQUIRE(evidence->last_modified_present[index]);
    CHECK(identity.find("|last_modified=" + std::to_string(evidence->last_modified[index])) !=
          std::string::npos);
  }
  exec_ok(con, "ROLLBACK");
}

TEST_CASE("Native decode rejects metadata from an earlier checkpoint",
          "[scan][certificate][integration]")
{
  native_database fixture;
  auto& con = *fixture.connection;
  exec_ok(con, "SET gpu_execution=false");
  exec_ok(con, "CREATE TABLE items AS SELECT range::INTEGER AS id FROM range(100)");
  exec_ok(con, "CHECKPOINT");
  auto info       = native_info(fixture, 62);
  auto* database  = &info->storage->GetAttached();
  auto ingestible = make_ingestible(std::move(info));
  auto ioctx      = std::make_shared<sirius::io::kvikio_context>();
  auto provider   = ingestible->next_split_provider(
    [ioctx](std::string_view) -> std::shared_ptr<sirius::io::ioctx> { return ioctx; });
  REQUIRE(provider);
  auto split = provider();
  REQUIRE(split);
  REQUIRE_FALSE(split->dependencies().empty());
  REQUIRE(split->dependencies().front().checkpoint_iteration);
  // End the original window, advance storage, then deliberately present the stale split.
  fixture.window->finish();
  fixture.window.reset();
  exec_ok(con, "COMMIT");
  exec_ok(con, "INSERT INTO items VALUES (100)");
  exec_ok(con, "CHECKPOINT");
  exec_ok(con, "BEGIN");
  fixture.lease(*database);
  auto context = sirius::test::get_registered_sirius_context(con);
  auto* space =
    sirius::scan_test_utils::get_space(context->get_memory_manager(), cucascade::memory::Tier::GPU);
  REQUIRE(space);
  rmm::cuda_stream stream;
  scan_operator_input input(std::move(split));
  input.gpu_memory_space = space;
  auto const before      = context->get_transparent_execution_stats();
  REQUIRE_THROWS_WITH(ingestible->materialize_table(input, stream),
                      "duckdb-native checkpoint iteration changed before metadata materialization");
  auto const after = context->get_transparent_execution_stats();
  CHECK(after.checkpoint_revalidation_failures == before.checkpoint_revalidation_failures + 1);
  CHECK(after.certificate_mismatches == before.certificate_mismatches);
  fixture.window->finish();
  fixture.window.reset();
  exec_ok(con, "ROLLBACK");
}

TEST_CASE("Fresh native ranges and coalesced splits preserve every certificate",
          "[scan][certificate][integration]")
{
  constexpr scan_contract_id contract_id = 61;
  native_database fixture;
  exec_ok(*fixture.connection, "CREATE TABLE items(id INTEGER)");
  exec_ok(*fixture.connection, "INSERT INTO items SELECT range FROM range(1000000)");
  exec_ok(*fixture.connection, "CHECKPOINT");

  auto info                    = native_info(fixture, contract_id);
  info->approximate_batch_size = 1;
  auto ingestible              = make_ingestible(std::move(info));
  auto ioctx                   = std::make_shared<sirius::io::kvikio_context>();
  auto coalescer               = ingestible->create_batch_coalescer();
  std::size_t input_slices     = 0;
  std::size_t input_ranges     = 0;
  std::vector<std::unique_ptr<scan_info>> splits;
  while (auto provider = ingestible->next_split_provider(
           [ioctx](std::string_view) -> std::shared_ptr<sirius::io::ioctx> { return ioctx; })) {
    auto range = provider();
    ++input_ranges;
    REQUIRE(range);
    auto const* native_range = dynamic_cast<duckdb_native_scan_info const*>(range.get());
    REQUIRE(native_range);
    REQUIRE(range->contract_id() == contract_id);
    REQUIRE(range->certificates().size() == native_range->row_groups.size());
    REQUIRE(range->dependencies().size() == range->certificates().size());
    input_slices += range->certificates().size();
    auto emitted = coalescer->push(std::move(range));
    for (auto& split : emitted) {
      splits.push_back(std::move(split));
    }
  }
  auto tail = coalescer->flush();
  for (auto& split : tail) {
    splits.push_back(std::move(split));
  }

  REQUIRE(input_slices > 1);
  REQUIRE(input_ranges > 1);
  REQUIRE(splits.size() > 1);
  std::size_t output_slices = 0;
  for (auto const& split : splits) {
    auto const* native_split = dynamic_cast<duckdb_native_scan_info const*>(split.get());
    REQUIRE(native_split);
    REQUIRE(split->contract_id() == contract_id);
    REQUIRE(split->certificates().size() == native_split->row_groups.size());
    REQUIRE(split->dependencies().size() == split->certificates().size());
    for (std::size_t i = 0; i < split->certificates().size(); ++i) {
      auto const& certificate = split->certificates()[i];
      auto const& dependency  = split->dependencies()[i];
      CHECK(certificate.contract_id == contract_id);
      auto const& group = native_split->row_groups[i];
      CHECK(certificate.split_id == static_cast<uint64_t>(group.row_group_index));
      REQUIRE(dependency.checkpoint_iteration.has_value());
      CHECK(certificate.input_identity == fixture.path.string() + "|checkpoint=" +
                                            std::to_string(*dependency.checkpoint_iteration) +
                                            "|row_group=" + std::to_string(group.row_group_index));
      CHECK(certificate.validation == (check_bit(later_check::segments_per_range) |
                                       check_bit(later_check::matrix_per_range)));
      REQUIRE(dependency.profiles);
      REQUIRE(certificate.profile != 0);
      auto profile = dependency.profiles->get(certificate.profile);
      CHECK(profile.storage_version != 0);
      CHECK(profile.columns.size() == group.columns.size());
      REQUIRE(dependency.datasource);
      REQUIRE(native_split->datasource);
      CHECK(&dependency.datasource->get_io_object() == &native_split->datasource->get_io_object());
      CHECK(certificate.input_identity.find(
              "|checkpoint=" + std::to_string(*dependency.checkpoint_iteration) + "|row_group=") !=
            std::string::npos);
    }
    output_slices += split->certificates().size();
  }
  CHECK(output_slices == input_slices);
}

TEST_CASE("Native coalescing rejects missing or surplus row-group certificates",
          "[scan][certificate][integration]")
{
  auto const certificate_count           = GENERATE(0, 1, 3);
  constexpr scan_contract_id contract_id = 63;
  native_database fixture;
  exec_ok(*fixture.connection, "CREATE TABLE items(id INTEGER)");
  exec_ok(*fixture.connection, "INSERT INTO items VALUES (1)");
  exec_ok(*fixture.connection, "CHECKPOINT");
  auto ingestible = make_ingestible(native_info(fixture, contract_id));
  auto coalescer  = ingestible->create_batch_coalescer();
  auto malformed  = std::make_unique<duckdb_native_scan_info>();
  malformed->row_groups.resize(2);
  std::vector<split_materializer_certificate> certificates(certificate_count);
  for (auto& certificate : certificates) {
    certificate.contract_id = contract_id;
  }
  malformed->set_contract_payload(
    contract_id, std::move(certificates), std::vector<split_dependencies>(certificate_count));
  REQUIRE_THROWS_WITH(coalescer->push(std::move(malformed)),
                      "native split requires one certificate and dependency per row group");

  // Rejection must leave the coalescer ready for a valid provider.
  auto ioctx    = std::make_shared<sirius::io::kvikio_context>();
  auto provider = ingestible->next_split_provider(
    [ioctx](std::string_view) -> std::shared_ptr<sirius::io::ioctx> { return ioctx; });
  REQUIRE(provider);
  REQUIRE(coalescer->push(provider()).empty());
  auto splits = coalescer->flush();
  REQUIRE(splits.size() == 1);
  REQUIRE(splits.front()->certificates().size() == 1);
  REQUIRE(splits.front()->dependencies().size() == 1);
  CHECK(splits.front()->contract_id() == contract_id);
}

TEST_CASE("An all-pruned native scan keeps its contract on the empty fallback split",
          "[scan][certificate][integration]")
{
  constexpr scan_contract_id contract_id = 62;
  native_database fixture;
  exec_ok(*fixture.connection, "CREATE TABLE items(id INTEGER)");
  exec_ok(*fixture.connection, "INSERT INTO items SELECT range FROM range(300000)");
  exec_ok(*fixture.connection, "CHECKPOINT");

  auto ingestible = make_ingestible(native_info(fixture, contract_id, /*all_pruned=*/true));
  auto ioctx      = std::make_shared<sirius::io::kvikio_context>();
  auto coalescer  = ingestible->create_batch_coalescer();
  std::vector<std::unique_ptr<scan_info>> splits;
  while (auto provider = ingestible->next_split_provider(
           [ioctx](std::string_view) -> std::shared_ptr<sirius::io::ioctx> { return ioctx; })) {
    auto emitted = coalescer->push(provider());
    for (auto& split : emitted) {
      splits.push_back(std::move(split));
    }
  }
  auto tail = coalescer->flush();
  for (auto& split : tail) {
    splits.push_back(std::move(split));
  }

  REQUIRE(splits.size() == 1);
  auto const* empty = dynamic_cast<duckdb_native_scan_info const*>(splits.front().get());
  REQUIRE(empty);
  CHECK(empty->row_groups.empty());
  CHECK(empty->contract_id() == contract_id);
  CHECK(empty->certificates().empty());
  CHECK(empty->dependencies().empty());
  CHECK_NOTHROW(validate_split_for_gpu(
    contract_id,
    check_bit(later_check::segments_per_range) | check_bit(later_check::matrix_per_range),
    {},
    *empty));
}

TEST_CASE("GPU scan construction requires an explicit contract",
          "[scan][certificate][scan_contract_required]")
{
  CHECK_FALSE((std::is_constructible_v<sirius_gpu_scan_operator,
                                       duckdb::vector<sirius::logical_type>,
                                       duckdb::idx_t,
                                       std::shared_ptr<gpu_ingestible>>));
}

TEST_CASE("GPU scan construction rejects a zero contract",
          "[scan][certificate][scan_contract_required]")
{
  REQUIRE_THROWS_AS((sirius_gpu_scan_operator{{}, 0, nullptr, /*contract_id=*/0}),
                    std::invalid_argument);
  CHECK_NOTHROW((sirius_gpu_scan_operator{{}, 0, nullptr, /*contract_id=*/71}));
}

TEST_CASE("Split consumption rejects a foreign projection contract", "[scan][certificate]")
{
  duckdb::SiriusContext observer;
  sirius_gpu_scan_operator scan{/*types=*/{},
                                /*estimated_cardinality=*/0,
                                /*ingestible=*/nullptr,
                                /*contract_id=*/71,
                                /*compressed_materialization_observer=*/&observer};
  scan_operator_input foreign_input(std::make_unique<certificate_test_split>(72));
  auto const before = observer.get_transparent_execution_stats();

  REQUIRE_THROWS_WITH(scan.execute(foreign_input, rmm::cuda_stream_view{}),
                      Catch::Matchers::ContainsSubstring("contract"));
  auto const after = observer.get_transparent_execution_stats();
  CHECK(after.certificate_mismatches == before.certificate_mismatches + 1);
}

TEST_CASE("Consumption requires per-unit coverage and matching dependencies",
          "[scan][certificate][consumption]")
{
  auto info       = parquet_info(81);
  auto ingestible = make_ingestible(std::move(info));
  auto ioctx      = std::make_shared<sirius::io::kvikio_context>();
  auto provider   = ingestible->next_split_provider(
    [ioctx](std::string_view) -> std::shared_ptr<sirius::io::ioctx> { return ioctx; });
  auto coalescer = ingestible->create_batch_coalescer();
  auto splits    = coalescer->push(provider());
  auto tail      = coalescer->flush();
  for (auto& s : tail)
    splits.push_back(std::move(s));
  REQUIRE(splits.size() == 1);
  auto& split   = *splits[0];
  auto* parquet = dynamic_cast<parquet_split_info*>(&split);
  REQUIRE(parquet);
  auto required =
    check_bit(later_check::footer_per_file) | check_bit(later_check::profile_per_file);
  REQUIRE_NOTHROW(validate_split_for_gpu(81, required, {}, split));
  std::vector<split_materializer_certificate> certificates(split.certificates().begin(),
                                                           split.certificates().end());
  std::vector<split_dependencies> dependencies(split.dependencies().begin(),
                                               split.dependencies().end());
  SECTION("missing coverage") { certificates[0].validation.reset(); }
  SECTION("missing profile") { certificates[0].profile = 0; }
  SECTION("wrong footer identity")
  {
    certificates[0].input_identity = parquet->rg_slices[0].file_path + "|footer=0";
  }
  SECTION("missing file approval") { dependencies[0].parquet_approval.reset(); }
  SECTION("row group outside approved set")
  {
    auto approval = std::make_shared<parquet_input_approval>(*dependencies[0].parquet_approval);
    approval->row_groups.clear();
    dependencies[0].parquet_approval = std::move(approval);
  }
  SECTION("foreign file") { certificates[0].input_identity = "other.parquet|footer=1"; }
  SECTION("foreign footer")
  {
    dependencies[0].footer = std::make_shared<cudf::io::parquet::FileMetaData>();
  }
  SECTION("missing datasource") { dependencies[0].datasource.reset(); }
  SECTION("non-parallel dependency")
  {
    certificate_test_split malformed(81, true);
    REQUIRE_THROWS_WITH(validate_split_for_gpu(81, required, {}, malformed),
                        "scan certificate incomplete: non-parallel dependencies");
    return;
  }
  SECTION("nonempty payload with empty certificates")
  {
    certificates.clear();
    dependencies.clear();
  }
  SECTION("surplus certificate")
  {
    certificates.push_back(certificates[0]);
    dependencies.push_back(dependencies[0]);
  }
  SECTION("invalid row group") { parquet->rg_slices[0].row_group_indices = {999999}; }
  SECTION("schema check required") { required |= check_bit(later_check::schema_per_file); }
  SECTION("zero-work empty certificates")
  {
    for (auto& slice : parquet->rg_slices)
      slice.row_group_indices.clear();
    split.set_contract_payload(81, {}, {});
    REQUIRE_NOTHROW(validate_split_for_gpu(81, required, {}, split));
    return;
  }
  split.set_contract_payload(81, std::move(certificates), std::move(dependencies));
  REQUIRE_THROWS_AS(validate_split_for_gpu(81, required, {}, split), certificate_incomplete);
}

TEST_CASE("Delta consumption derives coverage from actual segment lanes and expected key",
          "[scan][certificate][consumption]")
{
  auto const kind = GENERATE(0, 1, 2);  // transient, persistent blockless, mixed
  CAPTURE(kind);
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  exec_ok(con, "BEGIN TRANSACTION");
  auto& database = duckdb::Catalog::GetCatalog(*con.context, "").GetAttached();
  key_held_witness key{&database, "fixture.db", 31};
  duckdb_native_scan_info split;
  split.is_insert_delta  = true;
  split.host_backed_only = true;  // must NOT decide coverage
  duckdb_row_group_metadata group{};
  group.row_group_index = 4;
  group.row_count       = 2;
  duckdb_column_metadata column{};
  std::uint8_t staged[8]{};
  duckdb_segment_descriptor segment{};
  segment.block_id     = -1;
  segment.is_transient = true;
  segment.host_ptr     = staged;
  auto required        = check_bit(later_check::key_held);
  if (kind != 1) {
    column.data_segments.push_back(segment);
    required |= check_bit(later_check::host_staged);
  }
  if (kind != 0) {
    segment.is_transient = false;
    segment.host_ptr     = nullptr;
    column.array_child_validity_segments.push_back(segment);
    required |=
      check_bit(later_check::segments_per_range) | check_bit(later_check::matrix_per_range);
  }
  group.columns.push_back(column);
  split.row_groups.push_back(group);
  auto profiles = std::make_shared<physical_profile_table>();
  auto id       = profiles->add({});
  split_materializer_certificate cert{81, 4, "table|row_group=4", id, required, key};
  auto install = [&] {
    split.set_contract_payload(81, {cert}, {{nullptr, nullptr, {}, profiles}});
  };
  install();
  auto fresh_required =
    check_bit(later_check::segments_per_range) | check_bit(later_check::matrix_per_range);
  REQUIRE_NOTHROW(validate_split_for_gpu(81, fresh_required, key, split));
  SECTION("missing union bit") { cert.validation.reset(); }
  SECTION("wrong query") { cert.key_held->query_token++; }
  SECTION("wrong database") { cert.key_held->database = nullptr; }
  SECTION("wrong path") { cert.key_held->db_path = "other.db"; }
  SECTION("missing witness") { cert.key_held.reset(); }
  SECTION("wrong group") { cert.split_id++; }
  SECTION("wrong identity") { cert.input_identity = "table|row_group=5"; }
  SECTION("no expected key")
  {
    REQUIRE_THROWS_AS(validate_split_for_gpu(81, fresh_required, {}, split),
                      certificate_incomplete);
    return;
  }
  install();
  REQUIRE_THROWS_AS(validate_split_for_gpu(81, fresh_required, key, split), certificate_incomplete);
}

TEST_CASE("Resident admission requires pin facts and this query's applicable checks",
          "[scan][certificate][consumption]")
{
  pin_validation v;
  v.identity = v.layout = v.structure = {true, true};
  v.query_token                       = 17;
  REQUIRE_NOTHROW(admit_resident_batch(81, v, 17));  // Parquet N/A
  v.iteration = v.visibility = {true, true};
  REQUIRE_NOTHROW(admit_resident_batch(81, v, 17));
  SECTION("identity missing") { v.identity.applies = false; }
  SECTION("identity failed") { v.identity.passed = false; }
  SECTION("layout missing") { v.layout.applies = false; }
  SECTION("layout failed") { v.layout.passed = false; }
  SECTION("structure missing") { v.structure.applies = false; }
  SECTION("structure failed") { v.structure.passed = false; }
  SECTION("iteration failed") { v.iteration.passed = false; }
  SECTION("visibility failed") { v.visibility.passed = false; }
  SECTION("applicability mismatch") { v.iteration.applies = false; }
  SECTION("stale query") { v.query_token++; }
  try {
    admit_resident_batch(81, v, 17);
    FAIL("invalid resident witness admitted");
  } catch (certificate_incomplete const& error) {
    CHECK(error.contract == 81);
    CHECK(error.missing.none());
  }
}

TEST_CASE("Native consumption rejects shuffled groups and missing iteration evidence",
          "[scan][certificate][consumption][integration]")
{
  native_database fixture;
  exec_ok(*fixture.connection,
          "CREATE TABLE items AS SELECT i::INTEGER id FROM range(300000) t(i)");
  exec_ok(*fixture.connection, "CHECKPOINT");
  auto ingestible = make_ingestible(native_info(fixture, 81));
  auto ioctx      = std::make_shared<sirius::io::kvikio_context>();
  auto provider   = ingestible->next_split_provider(
    [ioctx](std::string_view) -> std::shared_ptr<sirius::io::ioctx> { return ioctx; });
  auto split = provider();
  auto required =
    check_bit(later_check::segments_per_range) | check_bit(later_check::matrix_per_range);
  REQUIRE_NOTHROW(validate_split_for_gpu(81, required, {}, *split));
  std::vector<split_materializer_certificate> certificates(split->certificates().begin(),
                                                           split->certificates().end());
  std::vector<split_dependencies> dependencies(split->dependencies().begin(),
                                               split->dependencies().end());
  REQUIRE(certificates.size() > 1);
  SECTION("shuffled certificates") { std::swap(certificates[0], certificates[1]); }
  SECTION("missing iteration") { dependencies[0].checkpoint_iteration.reset(); }
  SECTION("wrong iteration") { ++*dependencies[0].checkpoint_iteration; }
  SECTION("missing datasource") { dependencies[0].datasource.reset(); }
  SECTION("missing matrix")
  {
    certificates[0].validation &= ~check_bit(later_check::matrix_per_range);
  }
  SECTION("missing profile") { dependencies[0].profiles.reset(); }
  SECTION("nonempty payload empty list")
  {
    certificates.clear();
    dependencies.clear();
  }
  SECTION("missing group certificate")
  {
    certificates.pop_back();
    dependencies.pop_back();
  }
  split->set_contract_payload(81, std::move(certificates), std::move(dependencies));
  REQUIRE_THROWS_AS(validate_split_for_gpu(81, required, {}, *split), certificate_incomplete);
}
