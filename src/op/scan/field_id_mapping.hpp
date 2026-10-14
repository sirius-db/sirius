/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "op/scan/materialization_contract.hpp"
#include "op/scan/parquet_materialize.hpp"
#include "op/scan/parquet_metadata.hpp"

#include <duckdb/common/types/value.hpp>

#include <unordered_map>

namespace sirius::op::scan {
// Shared ID index; names are physical reader addresses, never logical identities.
std::unordered_map<int32_t, std::size_t> index_parquet_field_ids(
  cudf::io::parquet::FileMetaData const&);

// The qualified flat scalar mapping profile; nested transformations remain separate.
std::optional<cudf::data_type> field_mapping_type(duckdb::LogicalType const&);

struct field_requirement {
  int32_t id;
  duckdb::LogicalType type;
  std::optional<duckdb::Value> missing_value;
  bool allow_int32_to_int64  = false;
  bool allow_float_to_double = false;
};

// Immutable, scan-local approval. Only compile() can mint its operations; callers cannot
// attach a cast to an unrelated footer or change its target after physical validation.
class field_id_mapping {
 public:
  static field_id_mapping compile(std::shared_ptr<parquet_metadata const> metadata,
                                  std::string const& opened_identity,
                                  materialization_program program,
                                  std::vector<field_requirement> fields);
  [[nodiscard]] std::vector<std::string> const& read_names() const { return read_names_; }
  [[nodiscard]] std::vector<cudf::size_type> const& read_indices() const { return read_indices_; }
  [[nodiscard]] std::vector<duckdb::LogicalType> const& read_types() const { return read_types_; }
  [[nodiscard]] materialization_program const& program() const { return program_; }
  [[nodiscard]] std::string const& identity() const { return metadata_->built_for_identity; }
  [[nodiscard]] std::shared_ptr<parquet_metadata const> const& metadata() const
  {
    return metadata_;
  }
  [[nodiscard]] bool compatible_with(field_id_mapping const&) const;
  // Incremental peak over decoded input: defaults, cast outputs and duplicate
  // outputs coexist with decoder buffers. Caller adds this to its task reservation.
  [[nodiscard]] std::size_t additional_bytes(std::size_t rows) const;
  materialized_batch apply(owning_table_view decoded,
                           int64_t rows,
                           ::cuda::stream_ref,
                           rmm::device_async_resource_ref) const;

 private:
  field_id_mapping() = default;
  struct operation {
    field_requirement field;
    std::optional<std::size_t> physical_slot;
    cudf::data_type physical_type;
    cudf::data_type target_type;
  };
  std::shared_ptr<parquet_metadata const> metadata_;
  materialization_program program_;
  std::vector<std::string> read_names_;
  std::vector<cudf::size_type> read_indices_;
  std::vector<duckdb::LogicalType> read_types_;
  std::vector<operation> operations_;
};

// Prepared-unit entry point. It consumes the same source, footer and R2a physical
// checks as the legacy path, then applies the immutable mapping before semantics.
materialized_batch materialize_mapped_parquet(parquet_source const&,
                                              field_id_mapping const&,
                                              bound_table_scan const&,
                                              std::span<std::size_t const> row_groups,
                                              ::cuda::stream_ref,
                                              rmm::device_async_resource_ref);
// Group only already-admitted units. Incompatible mapping/default/original-schema
// boundaries flush a run; distinct object identities do not prevent coalescing.
// The caller reserves all retained outputs in this bounded unit span, plus the
// largest active run temporary bound; the returned vector owns every result.
struct mapped_parquet_unit {
  parquet_source source;
  field_id_mapping const* mapping;
};
std::vector<materialized_batch> materialize_mapped_parquet_runs(
  std::span<mapped_parquet_unit const>,
  bound_table_scan const&,
  ::cuda::stream_ref,
  rmm::device_async_resource_ref);
}  // namespace sirius::op::scan
