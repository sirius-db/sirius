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

#include "utils/cuda_status_fault.hpp"
#include "utils/dynamic_filter_publication_fixture.hpp"

using namespace sirius::test::publication;

TEST_CASE("whole-build readiness failures retain CUDA status and finish the publication",
          "[dynamic_filter][publication_claim][cuda_status_fault]")
{
  using fault       = sirius::test::scoped_cuda_status_fault;
  auto const status = GENERATE(cudaErrorIllegalAddress, cudaErrorUnknown, cudaErrorInvalidValue);
  bool const with_event = GENERATE(false, true);
  claim_fixture fixture;
  fixture.hash_join->set_build_arrives_whole(true);
  auto owner = fixture.make_build_batch();
  // Wrap the finished table with either an explicit writer event or the legacy
  // null-stream representation to exercise both readiness calls.
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  rmm::cuda_stream writer;
  auto batch = [&] {
    auto ro    = owner->to_read_only();
    auto* data = dynamic_cast<cucascade::gpu_table_representation const*>(ro.get_data());
    REQUIRE(data != nullptr);
    auto rep = std::make_unique<cucascade::gpu_table_representation>(
      data->get_table_view(),
      owner,
      data->get_size_in_bytes(),
      *fixture.gpu_space,
      with_event ? ::cuda::stream_ref{writer} : ::cuda::stream_ref{cudaStream_t{}});
    REQUIRE((rep->get_writer_event() != nullptr) == with_event);
    return cucascade::data_batch::make(0, std::move(rep));
  }();
  fault injection{with_event ? fault::wait_event : fault::synchronize_device, status};
  try {
    fixture.hash_join->push_data_batch_partitioned("build", batch, 0);
    FAIL("readiness failure must propagate");
  } catch (cudf::cuda_error const& error) {
    CHECK(error.error_code() == status);
    CHECK(std::string(error.what()).find(cudaGetErrorName(status)) != std::string::npos);
  }
  REQUIRE(injection.consumed());
  CHECK(fixture.stats.publication_attempts.load() == 1);
  CHECK(fixture.stats.publications_failed.load() == 1);
  CHECK(fixture.stats.publications_finished.load() == 0);
  CHECK(fixture.channel->snapshot().terminal());
  CHECK(fixture.channel->snapshot().empty());
}
