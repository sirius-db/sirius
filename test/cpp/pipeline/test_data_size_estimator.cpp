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

#include "catch.hpp"
#include "pipeline/data_size_estimator.hpp"
#include "pipeline/pipeline_build_context.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <cucascade/data/data_repository.hpp>

#include <cstddef>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using sirius::op::MemoryBarrierType;
using sirius::op::sirius_physical_operator;
using sirius::op::SiriusPhysicalOperatorType;
using sirius::pipeline::estimate_pipeline_total_output_bytes;
using sirius::pipeline::estimate_port_total_input_bytes;
using sirius::pipeline::pipeline_build_context;
using sirius::pipeline::sirius_pipeline;
using sirius::pipeline::sirius_pipeline_build_state;
using sirius::pipeline::size_estimate_options;
using sirius::pipeline::task_memory_record;

namespace {

constexpr std::size_t kMiB = 1024ull * 1024ull;

struct test_source_operator : sirius_physical_operator {
  test_source_operator() : sirius_physical_operator(SiriusPhysicalOperatorType::PROJECTION, {}, 0)
  {
  }

  [[nodiscard]] std::optional<std::size_t> total_source_input_bytes() const override
  {
    return input_total;
  }
  [[nodiscard]] std::optional<std::size_t> total_source_output_bytes() const override
  {
    return output_total;
  }

  // Fan-in primary input and the bytes consumed from it.
  [[nodiscard]] std::optional<std::string_view> primary_input_port() const override
  {
    if (primary_port.empty()) { return std::nullopt; }
    return std::string_view{primary_port};
  }
  [[nodiscard]] std::optional<std::size_t> consumed_primary_input_bytes() const override
  {
    // The hook simulates a task completing between estimator reads.
    if (on_consumed_read) { on_consumed_read(); }
    return consumed_primary;
  }

  [[nodiscard]] bool caps_pipeline_output() const override { return caps_output; }

  std::optional<std::size_t> input_total;
  std::optional<std::size_t> output_total;
  std::string primary_port;
  std::optional<std::size_t> consumed_primary;
  bool caps_output = false;
  std::function<void()> on_consumed_read;
};

/// Pipeline with a directly controlled finished state.
struct test_pipeline : sirius_pipeline {
  explicit test_pipeline(const pipeline_build_context& ctx) : sirius_pipeline(ctx) {}

  [[nodiscard]] bool is_pipeline_finished() const override { return finished; }

  bool finished = false;
};

/// Builds test pipeline DAGs with one operator per pipeline by default.
class estimator_dag {
 public:
  test_pipeline& add()
  {
    auto pipeline = duckdb::make_shared_ptr<test_pipeline>(_ctx);
    auto op       = std::make_unique<test_source_operator>();
    op->set_pipeline(pipeline);
    _bs.set_pipeline_source(*pipeline, *op);
    _bs.set_pipeline_sink(*pipeline, sirius::optional_ptr<sirius_physical_operator>(op.get()), 0);
    auto& ref = *pipeline;
    _ops.push_back(std::move(op));
    _pipelines.push_back(std::move(pipeline));
    return ref;
  }

  /// `with_repo=false` creates a dependency-only port.
  void connect(test_pipeline& from,
               test_pipeline& to,
               MemoryBarrierType barrier    = MemoryBarrierType::FULL,
               bool with_repo               = true,
               const std::string& port_name = "")
  {
    auto* consumer_op = to.get_source().get();
    auto* producer_op = from.get_sink().get();

    _names.push_back(port_name.empty() ? "e" + std::to_string(_names.size()) : port_name);
    std::string_view name = _names.back();

    auto port  = std::make_unique<sirius_physical_operator::port>();
    port->type = barrier;
    if (with_repo) {
      _repos.push_back(std::make_unique<cucascade::shared_data_repository>());
      port->repo = _repos.back().get();
    } else {
      port->repo = nullptr;
    }
    port->src_pipeline  = shared_for(from);
    port->dest_pipeline = shared_for(to);
    consumer_op->add_port(name, std::move(port));

    producer_op->add_next_port_after_sink(
      sirius_physical_operator::next_port_info{consumer_op, name, uuid::now_v7()});
  }

  static test_source_operator& source_of(test_pipeline& pipeline)
  {
    return static_cast<test_source_operator&>(*pipeline.get_source());
  }

  /// Models a pipeline with an operator after its source.
  test_source_operator& add_distinct_sink(test_pipeline& pipeline)
  {
    auto op = std::make_unique<test_source_operator>();
    op->set_pipeline(shared_for(pipeline));
    _bs.set_pipeline_sink(pipeline, sirius::optional_ptr<sirius_physical_operator>(op.get()), 0);
    auto& ref = *op;
    _ops.push_back(std::move(op));
    return ref;
  }

 private:
  duckdb::shared_ptr<sirius_pipeline> shared_for(test_pipeline& pipeline) const
  {
    for (const auto& p : _pipelines) {
      if (p.get() == &pipeline) { return p; }
    }
    return nullptr;
  }

  pipeline_build_context _ctx{nullptr, true};
  sirius_pipeline_build_state _bs;
  std::vector<std::unique_ptr<test_source_operator>> _ops;
  std::vector<std::unique_ptr<cucascade::shared_data_repository>> _repos;
  std::deque<std::string> _names;  // stable storage backing the port-name string_views
  std::vector<duckdb::shared_ptr<sirius_pipeline>> _pipelines;
};

void record_task(test_pipeline& pipeline, std::size_t in_bytes, std::size_t out_bytes)
{
  pipeline.get_memory_history().record(task_memory_record{in_bytes, in_bytes * 2, out_bytes});
}

/// Records an aggregate output/input ratio with enough samples to be trusted by default.
void record_ratio(test_pipeline& pipeline,
                  std::size_t in_bytes,
                  std::size_t out_bytes,
                  std::size_t samples = size_estimate_options{}.min_ratio_samples)
{
  for (std::size_t i = 0; i < samples; ++i) {
    record_task(pipeline, in_bytes / samples, out_bytes / samples);
  }
}

}  // namespace

TEST_CASE("data_size_estimator: a finished pipeline reports its exact recorded output",
          "[data_size_estimator][estimation]")
{
  estimator_dag dag;
  auto& producer = dag.add();
  record_ratio(producer, 100 * kMiB, 40 * kMiB);
  record_ratio(producer, 100 * kMiB, 60 * kMiB);
  producer.finished = true;

  auto est = estimate_pipeline_total_output_bytes(producer);
  REQUIRE(est.has_value());
  CHECK(est->bytes == 100 * kMiB);
  CHECK(est->exact);
  CHECK(est->hops == 0);
}

TEST_CASE("data_size_estimator: a finished pipeline whose tasks recorded nothing is unknown",
          "[data_size_estimator][estimation]")
{
  estimator_dag dag;
  auto& producer    = dag.add();
  producer.finished = true;
  // Treat failed measurement as unknown, not as zero output.
  producer.mark_task_created();
  producer.mark_task_created();

  CHECK_FALSE(estimate_pipeline_total_output_bytes(producer).has_value());
}

TEST_CASE("data_size_estimator: a finished pipeline that never created a task is exactly zero",
          "[data_size_estimator][estimation]")
{
  estimator_dag dag;
  auto& producer    = dag.add();
  producer.finished = true;
  // A finished pipeline with no tasks has drained without producing output.

  auto est = estimate_pipeline_total_output_bytes(producer);
  REQUIRE(est.has_value());
  CHECK(est->bytes == 0);
  CHECK(est->exact);
  CHECK(est->hops == 0);
}

TEST_CASE("data_size_estimator: a finished pipeline counts output from zero-basis tasks",
          "[data_size_estimator][estimation]")
{
  estimator_dag dag;
  auto& producer = dag.add();
  record_task(producer, 100 * kMiB, 40 * kMiB);
  record_task(producer, 0, 60 * kMiB);
  producer.finished = true;

  auto est = estimate_pipeline_total_output_bytes(producer);
  REQUIRE(est.has_value());
  CHECK(est->bytes == 100 * kMiB);
  CHECK(est->exact);
}

TEST_CASE("data_size_estimator: a finished pipeline that emitted nothing reports zero, exactly",
          "[data_size_estimator][estimation]")
{
  estimator_dag dag;
  auto& producer = dag.add();
  record_task(producer, 100 * kMiB, 0);
  producer.finished = true;

  auto est = estimate_pipeline_total_output_bytes(producer);
  REQUIRE(est.has_value());
  CHECK(est->bytes == 0);
  CHECK(est->exact);
}

TEST_CASE("data_size_estimator: an unfinished leaf scales its known total by the measured ratio",
          "[data_size_estimator][estimation]")
{
  estimator_dag dag;
  auto& producer                                 = dag.add();
  estimator_dag::source_of(producer).input_total = 1024 * kMiB;
  record_ratio(producer, 100 * kMiB, 25 * kMiB);

  auto est = estimate_pipeline_total_output_bytes(producer);
  REQUIRE(est.has_value());
  CHECK(est->bytes == 256 * kMiB);
  CHECK_FALSE(est->exact);
  // A leaf applies its own ratio without recursing, so the count is 1 at zero recursion depth.
  CHECK(est->hops == 1);
}

TEST_CASE("data_size_estimator: no ratio yet means no estimate, unless unit ratio is allowed",
          "[data_size_estimator][estimation]")
{
  estimator_dag dag;
  auto& producer                                 = dag.add();
  estimator_dag::source_of(producer).input_total = 512 * kMiB;

  CHECK_FALSE(estimate_pipeline_total_output_bytes(producer).has_value());

  auto est = estimate_pipeline_total_output_bytes(producer,
                                                  size_estimate_options{
                                                    .assume_unit_ratio = true,
                                                  });
  REQUIRE(est.has_value());
  CHECK(est->bytes == 512 * kMiB);
  CHECK_FALSE(est->exact);
}

TEST_CASE("data_size_estimator: a substituted unit ratio claims no measured support",
          "[data_size_estimator][estimation]")
{
  auto const min_samples = size_estimate_options{}.min_ratio_samples;
  REQUIRE(min_samples > 1);

  estimator_dag dag;
  auto& scan = dag.add();
  auto& mid  = dag.add();
  dag.connect(scan, mid);

  estimator_dag::source_of(scan).input_total = 1024 * kMiB;
  record_ratio(scan, 100 * kMiB, 100 * kMiB, /*samples=*/8);
  record_task(mid, 100 * kMiB, 50 * kMiB);

  auto est = estimate_pipeline_total_output_bytes(mid,
                                                  size_estimate_options{
                                                    .assume_unit_ratio = true,
                                                  });
  REQUIRE(est.has_value());
  CHECK(est->ratio_samples != 1);
  CHECK(est->ratio_samples == 8);
  CHECK_FALSE(est->exact);
  CHECK(est->bytes == 1024 * kMiB);
}

TEST_CASE("data_size_estimator: a single-input ratio from too few tasks is not trusted",
          "[data_size_estimator][estimation]")
{
  auto const min_samples = size_estimate_options{}.min_ratio_samples;
  REQUIRE(min_samples > 1);

  estimator_dag dag;
  auto& producer                                 = dag.add();
  estimator_dag::source_of(producer).input_total = 1024 * kMiB;

  // A single batch may not represent selectivity over clustered data.
  for (std::size_t i = 0; i < min_samples - 1; ++i) {
    record_task(producer, 100 * kMiB, 25 * kMiB);
  }
  CHECK_FALSE(estimate_pipeline_total_output_bytes(producer).has_value());

  auto assumed = estimate_pipeline_total_output_bytes(producer,
                                                      size_estimate_options{
                                                        .assume_unit_ratio = true,
                                                      });
  REQUIRE(assumed.has_value());
  CHECK(assumed->bytes == 1024 * kMiB);
  CHECK_FALSE(assumed->exact);

  auto lowered = estimate_pipeline_total_output_bytes(producer,
                                                      size_estimate_options{
                                                        .min_ratio_samples = 1,
                                                      });
  REQUIRE(lowered.has_value());
  CHECK(lowered->bytes == 256 * kMiB);

  record_task(producer, 100 * kMiB, 25 * kMiB);
  auto est = estimate_pipeline_total_output_bytes(producer);
  REQUIRE(est.has_value());
  CHECK(est->bytes == 256 * kMiB);
  CHECK(est->ratio_samples == min_samples);
}

TEST_CASE("data_size_estimator: a source that knows nothing yields no estimate",
          "[data_size_estimator][estimation]")
{
  estimator_dag dag;
  auto& producer = dag.add();
  record_ratio(producer, 100 * kMiB, 50 * kMiB);

  CHECK_FALSE(estimate_pipeline_total_output_bytes(producer).has_value());
  CHECK_FALSE(estimate_pipeline_total_output_bytes(producer,
                                                   size_estimate_options{
                                                     .assume_unit_ratio = true,
                                                   })
                .has_value());
}

TEST_CASE("data_size_estimator: an output-level source total bypasses the pipeline ratio",
          "[data_size_estimator][estimation]")
{
  estimator_dag dag;
  auto& producer                                  = dag.add();
  estimator_dag::source_of(producer).output_total = 300 * kMiB;
  // Applying the pre-filter ratio would count selectivity twice.
  record_ratio(producer, 100 * kMiB, 10 * kMiB);

  auto est = estimate_pipeline_total_output_bytes(producer);
  REQUIRE(est.has_value());
  CHECK(est->bytes == 300 * kMiB);
  CHECK_FALSE(est->exact);
  CHECK(est->planner_derived);
}

TEST_CASE("data_size_estimator: an output-level source total is unusable when operators follow it",
          "[data_size_estimator][estimation]")
{
  estimator_dag dag;
  auto& producer                                  = dag.add();
  estimator_dag::source_of(producer).output_total = 300 * kMiB;
  record_ratio(producer, 100 * kMiB, 10 * kMiB);
  // The source total cannot represent output after another operator.
  dag.add_distinct_sink(producer);

  CHECK_FALSE(estimate_pipeline_total_output_bytes(producer).has_value());
}

TEST_CASE("data_size_estimator: a capped pipeline is not extrapolated while unfinished",
          "[data_size_estimator][estimation]")
{
  estimator_dag dag;
  auto& producer                                 = dag.add();
  estimator_dag::source_of(producer).input_total = 1024 * kMiB;
  record_ratio(producer, 100 * kMiB, 50 * kMiB);

  auto before = estimate_pipeline_total_output_bytes(producer);
  REQUIRE(before.has_value());
  CHECK(before->bytes == 512 * kMiB);

  // A capped pipeline may finish without draining its source.
  estimator_dag::source_of(producer).caps_output = true;
  CHECK_FALSE(estimate_pipeline_total_output_bytes(producer).has_value());
  CHECK_FALSE(estimate_pipeline_total_output_bytes(producer,
                                                   size_estimate_options{
                                                     .assume_unit_ratio = true,
                                                   })
                .has_value());
}

TEST_CASE("data_size_estimator: a finished capped pipeline still reports its measured output",
          "[data_size_estimator][estimation]")
{
  estimator_dag dag;
  auto& producer                                 = dag.add();
  estimator_dag::source_of(producer).caps_output = true;
  record_ratio(producer, 100 * kMiB, 10 * kMiB);
  producer.finished = true;

  auto est = estimate_pipeline_total_output_bytes(producer);
  REQUIRE(est.has_value());
  CHECK(est->bytes == 10 * kMiB);
  CHECK(est->exact);
}

TEST_CASE("data_size_estimator: a cap anywhere in the chain stops the walk",
          "[data_size_estimator][estimation]")
{
  estimator_dag dag;
  auto& scan = dag.add();
  auto& mid  = dag.add();
  dag.connect(scan, mid);

  estimator_dag::source_of(scan).input_total = 1024 * kMiB;
  record_ratio(scan, 100 * kMiB, 100 * kMiB);
  record_ratio(mid, 100 * kMiB, 50 * kMiB);
  estimator_dag::source_of(mid).caps_output = true;

  CHECK_FALSE(estimate_pipeline_total_output_bytes(mid).has_value());
}

TEST_CASE("data_size_estimator: a cardinality projection never falls below bytes emitted",
          "[data_size_estimator][estimation]")
{
  using sirius::pipeline::project_source_output_bytes;

  constexpr std::size_t kRows  = 10;
  constexpr std::size_t kBytes = 1000;

  SECTION("a cardinality above what is emitted projects normally")
  {
    CHECK(project_source_output_bytes(50, kRows, kBytes) == 5000);
  }

  SECTION("a cardinality below the rows already emitted is floored at the observed bytes")
  {
    CHECK(project_source_output_bytes(4, kRows, kBytes) == kBytes);
  }

  SECTION("a zero cardinality does not claim the scan emits nothing")
  {
    // DuckDB can report zero cardinality for a nonempty scan.
    CHECK(project_source_output_bytes(0, kRows, kBytes) == kBytes);
  }

  SECTION("no measurement yet yields no projection")
  {
    CHECK_FALSE(project_source_output_bytes(1000, 0, 0).has_value());
    CHECK_FALSE(project_source_output_bytes(1000, kRows, 0).has_value());
    CHECK_FALSE(project_source_output_bytes(1000, 0, kBytes).has_value());
  }

  SECTION("an unrepresentable product yields no projection rather than a wrapped one")
  {
    CHECK_FALSE(
      project_source_output_bytes(std::numeric_limits<std::size_t>::max(), 1, kBytes).has_value());
  }
}

TEST_CASE("data_size_estimator: a measured anchor is not marked planner-derived",
          "[data_size_estimator][estimation]")
{
  estimator_dag dag;
  auto& producer                                 = dag.add();
  estimator_dag::source_of(producer).input_total = 1024 * kMiB;
  record_ratio(producer, 100 * kMiB, 50 * kMiB);

  auto est = estimate_pipeline_total_output_bytes(producer);
  REQUIRE(est.has_value());
  CHECK_FALSE(est->planner_derived);
}

TEST_CASE("data_size_estimator: planner provenance survives a downstream ratio",
          "[data_size_estimator][estimation]")
{
  // Regression: downstream ratios must preserve planner-derived provenance.
  estimator_dag dag;
  auto& scan = dag.add();
  auto& mid  = dag.add();
  dag.connect(scan, mid);

  estimator_dag::source_of(scan).output_total = 300 * kMiB;
  record_ratio(mid, 100 * kMiB, 50 * kMiB);

  auto est = estimate_pipeline_total_output_bytes(mid);
  REQUIRE(est.has_value());
  CHECK(est->bytes == 150 * kMiB);
  CHECK_FALSE(est->exact);
  CHECK(est->ratio_samples == size_estimate_options{}.min_ratio_samples);
  CHECK(est->planner_derived);
}

TEST_CASE("data_size_estimator: a projection never falls below the bytes already emitted",
          "[data_size_estimator][estimation]")
{
  estimator_dag dag;
  auto& producer                                 = dag.add();
  estimator_dag::source_of(producer).input_total = 500 * kMiB;

  // Ratio-eligible tasks: 400 MiB in, 40 MiB out, so the measured ratio is 0.1.
  record_ratio(producer, 400 * kMiB, 40 * kMiB);
  // Zero-basis tasks emit real bytes but cannot inform a ratio, so the projection is blind to
  // them however accurate the upstream total is.
  record_task(producer, 0, 200 * kMiB);

  // 500 MiB x 0.1 = 50 MiB, which is less than the 240 MiB already emitted.
  auto est = estimate_pipeline_total_output_bytes(producer);
  REQUIRE(est.has_value());
  CHECK(est->bytes == 240 * kMiB);
  CHECK_FALSE(est->exact);
}

TEST_CASE("data_size_estimator: ratios compose along a multi-hop chain",
          "[data_size_estimator][estimation]")
{
  estimator_dag dag;
  auto& scan     = dag.add();
  auto& mid      = dag.add();
  auto& consumer = dag.add();
  dag.connect(scan, mid);
  dag.connect(mid, consumer);

  estimator_dag::source_of(scan).input_total = 1024 * kMiB;
  record_ratio(scan, 100 * kMiB, 50 * kMiB);
  record_ratio(mid, 100 * kMiB, 25 * kMiB);

  auto& consumer_op = *consumer.get_source();
  auto est = estimate_port_total_input_bytes(consumer_op, consumer_op.get_port_ids().front());
  REQUIRE(est.has_value());
  CHECK(est->bytes == 128 * kMiB);
  CHECK_FALSE(est->exact);
  // scan's own ratio, then mid's.
  CHECK(est->hops == 2);
}

TEST_CASE("data_size_estimator: the walk stops at the first finished pipeline",
          "[data_size_estimator][estimation]")
{
  estimator_dag dag;
  auto& scan     = dag.add();
  auto& mid      = dag.add();
  auto& consumer = dag.add();
  dag.connect(scan, mid);
  dag.connect(mid, consumer);

  // Leaving scan unknown verifies that the walk stops at finished mid.
  record_ratio(mid, 100 * kMiB, 200 * kMiB);
  mid.finished = true;

  auto& consumer_op = *consumer.get_source();
  auto est = estimate_port_total_input_bytes(consumer_op, consumer_op.get_port_ids().front());
  REQUIRE(est.has_value());
  CHECK(est->bytes == 200 * kMiB);
  CHECK(est->exact);
  CHECK(est->hops == 0);
}

namespace {

/// A `probe -> join <- build`, `join -> consumer` test DAG.
struct fan_in_dag {
  estimator_dag dag;
  test_pipeline* probe    = nullptr;
  test_pipeline* build    = nullptr;
  test_pipeline* join     = nullptr;
  test_pipeline* consumer = nullptr;

  fan_in_dag()
  {
    probe    = &dag.add();
    build    = &dag.add();
    join     = &dag.add();
    consumer = &dag.add();
    dag.connect(*probe, *join, MemoryBarrierType::PARTIAL, true, "default");
    dag.connect(*build, *join, MemoryBarrierType::FULL, true, "build");
    dag.connect(*join, *consumer);
  }

  /// Records the join ratio across exactly `n` tasks.
  void record_join_ratio(std::size_t in_bytes, std::size_t out_bytes, std::size_t n = 16)
  {
    for (std::size_t i = 0; i < n; ++i) {
      record_task(*join, in_bytes / n, out_bytes / n);
    }
  }

  void set_join_task_counters(std::size_t created, std::size_t completed)
  {
    for (std::size_t i = 0; i < created; ++i) {
      join->mark_task_created();
    }
    for (std::size_t i = 0; i < completed; ++i) {
      join->mark_task_completed();
    }
  }

  void give_both_sides_totals()
  {
    estimator_dag::source_of(*probe).input_total = 1024 * kMiB;
    record_ratio(*probe, 100 * kMiB, 100 * kMiB);
    estimator_dag::source_of(*build).input_total = 1 * kMiB;
    record_ratio(*build, 100 * kMiB, 100 * kMiB);
  }

  std::optional<sirius::pipeline::data_size_estimate> estimate()
  {
    auto& op = *consumer->get_source();
    return estimate_port_total_input_bytes(op, op.get_port_ids().front());
  }
};

}  // namespace

TEST_CASE("data_size_estimator: a fan-in that nominates no primary port yields no estimate",
          "[data_size_estimator][estimation][fan_in]")
{
  fan_in_dag f;
  f.give_both_sides_totals();
  f.record_join_ratio(100 * kMiB, 50 * kMiB);
  estimator_dag::source_of(*f.join).consumed_primary = 100 * kMiB;

  CHECK_FALSE(f.estimate().has_value());
}

TEST_CASE("data_size_estimator: a fan-in with no consumed primary bytes yields no estimate",
          "[data_size_estimator][estimation][fan_in]")
{
  fan_in_dag f;
  f.give_both_sides_totals();
  f.record_join_ratio(100 * kMiB, 50 * kMiB);
  estimator_dag::source_of(*f.join).primary_port     = "default";
  estimator_dag::source_of(*f.join).consumed_primary = 0;

  CHECK_FALSE(f.estimate().has_value());
}

TEST_CASE("data_size_estimator: a fan-in scales the primary upstream by output-per-primary-byte",
          "[data_size_estimator][estimation][fan_in]")
{
  fan_in_dag f;
  f.give_both_sides_totals();
  // STANDARD joins must use consumed primary bytes, not re-paired input basis.
  f.record_join_ratio(999 * kMiB, 50 * kMiB);
  estimator_dag::source_of(*f.join).primary_port     = "default";
  estimator_dag::source_of(*f.join).consumed_primary = 200 * kMiB;

  auto est = f.estimate();
  REQUIRE(est.has_value());
  CHECK(est->bytes == 256 * kMiB);
  CHECK_FALSE(est->exact);
  // probe side's ratio, then the join's.
  CHECK(est->hops == 2);
  CHECK_FALSE(est->planner_derived);
}

TEST_CASE("data_size_estimator: a fan-in projection never falls below the bytes already emitted",
          "[data_size_estimator][estimation][fan_in]")
{
  fan_in_dag f;
  // Probe projects 8 MiB and has emitted only 4, so its own floor does not bind and the join
  // scales a genuinely small upstream.
  estimator_dag::source_of(*f.probe).input_total = 8 * kMiB;
  record_ratio(*f.probe, 4 * kMiB, 4 * kMiB);
  // Consumed primary bytes are deliberately over-counted, so the ratio can read far below the
  // join's true expansion: 8 MiB x (64/512) = 1 MiB, against 64 MiB already emitted.
  f.record_join_ratio(999 * kMiB, 64 * kMiB);
  estimator_dag::source_of(*f.join).primary_port     = "default";
  estimator_dag::source_of(*f.join).consumed_primary = 512 * kMiB;

  auto est = f.estimate();
  REQUIRE(est.has_value());
  CHECK(est->bytes == 64 * kMiB);
  CHECK_FALSE(est->exact);
}

TEST_CASE("data_size_estimator: planner provenance survives a fan-in hop",
          "[data_size_estimator][estimation][fan_in]")
{
  fan_in_dag f;
  estimator_dag::source_of(*f.probe).output_total = 1024 * kMiB;
  f.record_join_ratio(999 * kMiB, 50 * kMiB);
  estimator_dag::source_of(*f.join).primary_port     = "default";
  estimator_dag::source_of(*f.join).consumed_primary = 200 * kMiB;

  auto est = f.estimate();
  REQUIRE(est.has_value());
  CHECK(est->bytes == 256 * kMiB);
  CHECK(est->ratio_samples == size_estimate_options{}.min_fan_in_ratio_samples);
  CHECK(est->planner_derived);
}

TEST_CASE("data_size_estimator: a fan-in samples its numerator before its denominator",
          "[data_size_estimator][estimation][fan_in]")
{
  // Reading output first prevents an interleaved completion from inflating the ratio.
  fan_in_dag f;
  f.give_both_sides_totals();
  f.record_join_ratio(999 * kMiB, 50 * kMiB);
  auto& src            = estimator_dag::source_of(*f.join);
  src.primary_port     = "default";
  src.consumed_primary = 200 * kMiB;

  // Simulate a task completing when the denominator is sampled.
  bool fired           = false;
  src.on_consumed_read = [&] {
    if (fired) { return; }
    fired = true;
    record_task(*f.join, 0, 50 * kMiB);
  };

  auto est = f.estimate();
  REQUIRE(fired);  // Ensure the simulated race occurred.
  REQUIRE(est.has_value());
  CHECK(est->bytes == 256 * kMiB);
}

TEST_CASE("data_size_estimator: a fan-in with too few completed tasks yields no estimate",
          "[data_size_estimator][estimation][fan_in]")
{
  fan_in_dag f;
  f.give_both_sides_totals();
  estimator_dag::source_of(*f.join).primary_port     = "default";
  estimator_dag::source_of(*f.join).consumed_primary = 200 * kMiB;

  // In-flight tasks can temporarily depress the fan-in ratio.
  f.record_join_ratio(100 * kMiB, 50 * kMiB, /*n=*/1);
  CHECK_FALSE(f.estimate().has_value());

  // A unit ratio is unsafe for joins that may greatly change data volume.
  {
    auto& op = *f.consumer->get_source();
    CHECK_FALSE(estimate_port_total_input_bytes(op,
                                                op.get_port_ids().front(),
                                                size_estimate_options{
                                                  .assume_unit_ratio = true,
                                                })
                  .has_value());
  }

  f.record_join_ratio(100 * kMiB, 50 * kMiB, /*n=*/16);
  CHECK(f.estimate().has_value());
}

TEST_CASE("data_size_estimator: a fan-in is withheld while probe tasks are in flight",
          "[data_size_estimator][estimation][fan_in]")
{
  fan_in_dag f;
  f.give_both_sides_totals();
  f.record_join_ratio(100 * kMiB, 50 * kMiB);
  estimator_dag::source_of(*f.join).primary_port     = "default";
  estimator_dag::source_of(*f.join).consumed_primary = 200 * kMiB;

  auto const quiescent = f.estimate();
  REQUIRE(quiescent.has_value());
  CHECK(quiescent->bytes == 256 * kMiB);

  // One enormous in-flight task could hold the ratio near zero however many small tasks have
  // completed, so in-flight snapshots are withheld rather than sample-floored.
  f.set_join_task_counters(/*created=*/25, /*completed=*/20);
  CHECK_FALSE(f.estimate().has_value());

  // Quiescent again: the totals cover the same tasks and the estimate returns.
  f.set_join_task_counters(/*created=*/0, /*completed=*/5);
  auto const after = f.estimate();
  REQUIRE(after.has_value());
  CHECK(after->bytes == quiescent->bytes);
}

TEST_CASE("data_size_estimator: a fan-in rejects a task racing the ratio reads",
          "[data_size_estimator][estimation][fan_in]")
{
  fan_in_dag f;
  f.give_both_sides_totals();
  f.record_join_ratio(100 * kMiB, 50 * kMiB);
  auto& src            = estimator_dag::source_of(*f.join);
  src.primary_port     = "default";
  src.consumed_primary = 200 * kMiB;

  // A task starts and completes while `consumed` is read: the counters end equal, so only a
  // comparison against the pre-read snapshot can catch it.
  bool fired           = false;
  src.on_consumed_read = [&] {
    if (fired) { return; }
    fired = true;
    f.join->mark_task_created();
    f.join->mark_task_completed();
  };
  CHECK_FALSE(f.estimate().has_value());
  REQUIRE(fired);

  // Quiescent and stable again: the estimate returns.
  auto est = f.estimate();
  REQUIRE(est.has_value());
  CHECK(est->bytes == 256 * kMiB);
}

TEST_CASE("data_size_estimator: a zero fan-in floor still requires completed output",
          "[data_size_estimator][estimation][fan_in]")
{
  fan_in_dag f;
  f.give_both_sides_totals();
  estimator_dag::source_of(*f.join).primary_port     = "default";
  estimator_dag::source_of(*f.join).consumed_primary = 200 * kMiB;

  // No join task has completed; a zero floor must not turn 0 / consumed into a confident zero.
  auto& op = *f.consumer->get_source();
  CHECK_FALSE(estimate_port_total_input_bytes(op,
                                              op.get_port_ids().front(),
                                              size_estimate_options{
                                                .min_fan_in_ratio_samples = 0,
                                              })
                .has_value());
}

TEST_CASE("data_size_estimator: a projection that would overflow is reported as unknown",
          "[data_size_estimator][estimation]")
{
  estimator_dag dag;
  auto& scan = dag.add();

  estimator_dag::source_of(scan).input_total = std::numeric_limits<std::size_t>::max();
  record_ratio(scan, 1 * kMiB, 1024 * kMiB);

  CHECK_FALSE(estimate_pipeline_total_output_bytes(scan).has_value());

  estimator_dag::source_of(scan).input_total = 1024 * kMiB;
  CHECK(estimate_pipeline_total_output_bytes(scan).has_value());
}

TEST_CASE("data_size_estimator: ratio_samples reports the weakest ratio in the chain",
          "[data_size_estimator][estimation][fan_in]")
{
  fan_in_dag f;
  f.give_both_sides_totals();
  f.record_join_ratio(100 * kMiB, 50 * kMiB, /*n=*/40);
  estimator_dag::source_of(*f.join).primary_port     = "default";
  estimator_dag::source_of(*f.join).consumed_primary = 200 * kMiB;

  auto est = f.estimate();
  REQUIRE(est.has_value());
  CHECK(est->ratio_samples == size_estimate_options{}.min_ratio_samples);
}

TEST_CASE("data_size_estimator: a fan-in ignores the non-primary side entirely",
          "[data_size_estimator][estimation][fan_in]")
{
  fan_in_dag f;
  f.give_both_sides_totals();
  f.record_join_ratio(100 * kMiB, 50 * kMiB);
  estimator_dag::source_of(*f.join).primary_port     = "default";
  estimator_dag::source_of(*f.join).consumed_primary = 200 * kMiB;

  auto const before = f.estimate();
  REQUIRE(before.has_value());

  estimator_dag::source_of(*f.build).input_total = 4096 * kMiB;
  record_ratio(*f.build, 100 * kMiB, 800 * kMiB);

  auto const after = f.estimate();
  REQUIRE(after.has_value());
  CHECK(after->bytes == before->bytes);
}

TEST_CASE("data_size_estimator: a fan-in whose primary upstream is unknown yields no estimate",
          "[data_size_estimator][estimation][fan_in]")
{
  fan_in_dag f;
  estimator_dag::source_of(*f.build).input_total = 1 * kMiB;
  record_ratio(*f.build, 100 * kMiB, 100 * kMiB);
  f.record_join_ratio(100 * kMiB, 50 * kMiB);
  estimator_dag::source_of(*f.join).primary_port     = "default";
  estimator_dag::source_of(*f.join).consumed_primary = 200 * kMiB;

  CHECK_FALSE(f.estimate().has_value());
}

TEST_CASE("data_size_estimator: max_hops bounds the upstream walk",
          "[data_size_estimator][estimation]")
{
  estimator_dag dag;
  auto& scan = dag.add();
  auto& a    = dag.add();
  auto& b    = dag.add();
  dag.connect(scan, a);
  dag.connect(a, b);

  estimator_dag::source_of(scan).input_total = 100 * kMiB;
  record_ratio(scan, 100 * kMiB, 100 * kMiB);
  record_ratio(a, 100 * kMiB, 100 * kMiB);
  record_ratio(b, 100 * kMiB, 100 * kMiB);

  CHECK(estimate_pipeline_total_output_bytes(b, size_estimate_options{.max_hops = 2}).has_value());
  CHECK_FALSE(
    estimate_pipeline_total_output_bytes(b, size_estimate_options{.max_hops = 1}).has_value());
}

TEST_CASE("data_size_estimator: unknown and dependency-only ports yield no estimate",
          "[data_size_estimator][estimation]")
{
  estimator_dag dag;
  auto& producer = dag.add();
  auto& consumer = dag.add();
  dag.connect(producer, consumer, MemoryBarrierType::FULL, /*with_repo=*/false);

  producer.finished = true;
  record_ratio(producer, 100 * kMiB, 100 * kMiB);

  auto& consumer_op = *consumer.get_source();
  CHECK_FALSE(estimate_port_total_input_bytes(consumer_op, "no_such_port").has_value());
  CHECK_FALSE(
    estimate_port_total_input_bytes(consumer_op, consumer_op.get_port_ids().front()).has_value());
}
