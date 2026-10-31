/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "expression_evaluator/expression_evaluator.hpp"
#include "op/scan/field_id_mapping.hpp"
#include "op/scan/gpu_ingestible_types.hpp"
#include "op/scan/iceberg_delete_set.hpp"
#include "op/scan/row_group_metadata.hpp"
#include "op/scan/scan_plan.hpp"

#include <stop_token>

namespace sirius::op::scan {
struct prepared_split {
  std::shared_ptr<scan_info const> info;
  std::shared_ptr<void const> metadata_owner;
  // Views into info (or the direct caller's retained input); copying the shared
  // task state must not deep-copy every row-group list.
  std::span<split_dependencies const> dependencies;
  std::span<row_group_slice const> slices;
  std::shared_ptr<scan_plan const> plan;
  std::vector<std::string> partition_values;
  std::shared_ptr<lineage_domain const> input_domain;
  // Bound query file indices in the lineage file namespace, only for FILE_INDEX virtuals.
  std::vector<uint64_t> virtual_file_indices;
  uint64_t attempt       = 1;
  uint64_t logical_input = 0;
  std::stop_token stop;
};
// Adapter certification is frozen before decode and tied to the exact program
// and plan. Non-Parquet adapters retain their own proof, without a fake footer.
class mapping_certification {
 public:
  mapping_certification(materialization_program const& program,
                        std::shared_ptr<scan_plan const> plan,
                        std::shared_ptr<void const> owner,
                        std::shared_ptr<field_id_mapping const> normalization = {})
    : contract(program.contract),
      generation(program.generation),
      layout(program.input_layout),
      plan(std::move(plan)),
      owner(std::move(owner)),
      normalization(std::move(normalization))
  {
    if (!contract || !generation || !layout || !this->plan || !this->owner)
      throw std::invalid_argument("mapping certification requires bound evidence");
  }
  uint64_t const contract, generation;
  std::shared_ptr<void const> const layout;
  std::shared_ptr<scan_plan const> const plan;
  std::shared_ptr<void const> const owner;
  std::shared_ptr<field_id_mapping const> const normalization;
};
struct certified_mapping {
  std::shared_ptr<materialization_program const> program;
  std::shared_ptr<field_id_mapping const> normalization;
  // Adapter-owned evidence for independently qualified non-Parquet input.
  // The default Parquet entry also retains/checks its per-slice approvals.
  std::shared_ptr<mapping_certification const> evidence;
};
struct required_visibility_input {
  applied_visibility binding;
  std::shared_ptr<iceberg_delete_set const> owner;
};
enum class validation_policy { legacy_iceberg, qualified_lineage };
enum class materialization_phase { decoded, complete };
struct materializer_inputs;
struct decode_result {
  filtered_table batch;
  row_lineage lineage;
};
// Borrowed decoder staging must be owned by the prepared input, certification,
// owned callback or result. Deferred replacement success joins the caller stream before releasing
// these owners; default readers keep their own completion protocol.
using decode_fn = std::function<decode_result(prepared_split const&,
                                              materializer_inputs const&,
                                              cucascade::memory::memory_space const&,
                                              ::cuda::stream_ref,
                                              rmm::device_async_resource_ref)>;
struct reader_task {
  // Owns the default adapter's options, AST, snapshot and borrowed input owners
  // through the callback. Only the default decoder consumes this task.
  decode_fn decode;
};
struct coverage_facts {
  std::shared_ptr<static_predicate_catalogue const> catalogue;
  std::vector<std::size_t> reader_requested;
  bool reader_pushdown_complete = false;
};
struct materializer_inputs {
  coverage_facts coverage;
  certified_mapping mapping;
  std::vector<required_visibility_input> visibility;
  validation_policy validation = validation_policy::qualified_lineage;
  std::string diagnostic_context;
  std::shared_ptr<reader_task const> reader;
  bool needs_lineage = false;
  expression_query_policy expression_policy;
  bool like_swar_fastpath = false;
  std::shared_ptr<like_multiliteral_cache const> like_cache;
  // The pipeline may defer residual/projection for late materialization. Both
  // entries still validate the same decode facts and immutable bindings.
  materialization_phase phase = materialization_phase::complete;
};

// Production semantic steps used by both the pipeline bridge and replacement
// decoders. The observer only reports work performed inside these functions.
struct semantic_options {
  expression_query_policy policy;
  bool like_swar_fastpath = false;
  std::shared_ptr<like_multiliteral_cache const> like_cache;
  std::vector<cudf::size_type> filter_projection;
  std::unique_ptr<cudf::column>* survivors = nullptr;
  std::span<std::size_t const> elided;
  uint64_t attempt       = 1;
  uint64_t logical_input = 0;
};
void apply_materialization_visibility(
  std::span<applied_visibility const> required,
  visibility_bindings& applied,
  bool& complete,
  std::function<void(std::span<applied_visibility const>)> const& apply,
  uint64_t contract,
  uint64_t attempt,
  uint64_t logical_input);
void apply_materialization_residual(materialized_batch&,
                                    materialization_program const&,
                                    semantic_options const&,
                                    ::cuda::stream_ref,
                                    rmm::device_async_resource_ref);
void apply_materialization_projection(materialized_batch&,
                                      materialization_program const&,
                                      scan_plan const&,
                                      std::vector<std::string> const& partition_values,
                                      semantic_options const&,
                                      ::cuda::stream_ref,
                                      rmm::device_async_resource_ref);
decode_result parquet_cudf_decode(prepared_split const&,
                                  materializer_inputs const&,
                                  cucascade::memory::memory_space const&,
                                  ::cuda::stream_ref,
                                  rmm::device_async_resource_ref);
decode_result materialize_shared(prepared_split const&,
                                 materializer_inputs const&,
                                 decode_fn const&,
                                 cucascade::memory::memory_space const&,
                                 ::cuda::stream_ref,
                                 rmm::device_async_resource_ref);

// Extra peak for canonical selection, validation and visibility, above decoder
// buffers. payload_bytes includes normalized data and requested virtual columns;
// adapter-specific decode/mapping temporaries remain the caller's responsibility.
std::size_t shared_materialization_workspace_bytes(std::size_t rows,
                                                   std::size_t payload_bytes,
                                                   std::size_t deletes,
                                                   std::size_t intervals) noexcept;

struct step_identity {
  uint64_t contract, attempt, logical_input;
};
struct step_observer {
  virtual void on_visibility(step_identity const&, applied_visibility const&) {}
  virtual void on_residual(step_identity const&, std::size_t) {}
  virtual void on_assemble(step_identity const&) {}
  virtual ~step_observer() = default;
};
class scoped_step_observer {
 public:
  explicit scoped_step_observer(step_observer&);
  ~scoped_step_observer() noexcept;
  scoped_step_observer(scoped_step_observer const&)            = delete;
  scoped_step_observer& operator=(scoped_step_observer const&) = delete;

 private:
  uint64_t registration_;
};
}  // namespace sirius::op::scan
