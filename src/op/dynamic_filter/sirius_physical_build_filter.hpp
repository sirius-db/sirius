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

#pragma once

#include "op/dynamic_filter/complete_build_inventory.hpp"
#include "op/sirius_physical_operator.hpp"

#include <atomic>
#include <mutex>

namespace sirius::op {

class dynamic_filter_publication_session;

/**
 * @brief Publishes a complete build's optional filter before join partition sizing.
 *
 * A FULL input certifies all original batches. Scheduled GPU tasks accumulate their keys and
 * forward the same batches; the last contribution publishes before this pipeline completes.
 * The owning join retains its publication session and its original partitioning strategy.
 */
class sirius_physical_build_filter final : public sirius_physical_operator {
 public:
  static constexpr SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::BUILD_FILTER;

  sirius_physical_build_filter(duckdb::vector<sirius::logical_type> types,
                               std::size_t estimated_cardinality,
                               dynamic_filter_publication_session& session);

  bool is_source() const override { return true; }
  bool is_sink() const override { return true; }
  [[nodiscard]] MemoryBarrierType input_barrier_for(
    sirius_physical_operator const& producer) const override;
  void build_pipelines(pipeline::sirius_pipeline& current,
                       pipeline::sirius_meta_pipeline& meta_pipeline) override;
  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    input_stats const& stats) const override;

  std::unique_ptr<operator_data> get_next_task_input_data() override;
  std::unique_ptr<operator_data> execute(operator_data const& input,
                                         ::cuda::stream_ref stream) override;

 protected:
  void on_input_batch_pushed(std::string_view port_id, cucascade::data_batch& batch) override;
  void on_finalize_operator() override;

 private:
  // Starts once on the untracked creator thread, before the first batch leaves the repository.
  // Returns false while the source is still producing; unsupported inputs remain pass-through.
  [[nodiscard]] bool decide_accumulation();

  dynamic_filter_publication_session& _session;
  build_arrival_ledger _arrival_ledger;
  std::mutex _decision_mutex;
  bool _decided = false;
  std::atomic<bool> _accumulating{false};
};

}  // namespace sirius::op
