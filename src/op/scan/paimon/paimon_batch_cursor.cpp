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

#include "op/scan/paimon/paimon_batch_cursor.hpp"

#include "scan_manager/preparation_ledger.hpp"

#include <duckdb/common/arrow/arrow.hpp>
#include <paimon/reader/batch_reader.h>
#include <paimon/status.h>

#include <stdexcept>
#include <system_error>
#include <utility>
namespace sirius::op::scan {
namespace {
struct native_roots {
  paimon::BatchReader::ReadBatch value;
  ~native_roots() { release(); }
  void release() noexcept
  {
    if (value.first && value.first->release) value.first->release(value.first.get());
    if (value.second && value.second->release) value.second->release(value.second.get());
    value = {};
  }
};
void check_stop(scan_manager::preparation_stop_token stop)
{
  if (stop.stop_requested())
    throw std::system_error(std::make_error_code(std::errc::operation_canceled));
}
}  // namespace
void detail::throw_paimon_status(paimon::Status const& status)
{
  if (status.IsCapacityError() || status.IsOutOfMemory())
    throw scan_manager::preparation_resource_error(status.ToString(), status.IsCapacityError());
  if (status.IsCancelled())
    throw std::system_error(std::make_error_code(std::errc::operation_canceled), status.ToString());
  throw std::runtime_error(status.ToString());
}
paimon_batch_cursor::paimon_batch_cursor(std::unique_ptr<paimon::BatchReader> reader,
                                         std::shared_ptr<void> owners)
  : native_owners_(std::move(owners)), reader_(std::move(reader))
{
  if (!reader_ || !native_owners_) {
    close();
    throw std::invalid_argument("Paimon cursor requires a reader and retained owners");
  }
}
paimon_batch_cursor::~paimon_batch_cursor()
{
  try {
    close();
  } catch (...) {
  }
}
bool paimon_batch_cursor::consume(scan_manager::preparation_stop_token stop,
                                  int64_t max_rows,
                                  consumer const& detach)
{
  if (failure_) std::rethrow_exception(failure_);
  if (phase_ == phase::eof) return false;
  if (phase_ == phase::closed) throw std::logic_error("Paimon cursor closed before EOF");
  try {
    if (!detach || max_rows <= 0)
      throw std::invalid_argument("Paimon cursor requires a consumer and positive row limit");
    check_stop(stop);
    auto result = reader_->NextBatch();
    if (!result.ok()) detail::throw_paimon_status(result.status());
    native_roots roots{std::move(result).value()};
    check_stop(stop);
    if (paimon::BatchReader::IsEofBatch(roots.value)) {
      roots.release();
      close();
      phase_ = phase::eof;
      return false;
    }
    auto const& [array, schema] = roots.value;
    if (!array || !schema || !array->release || !schema->release || array->length < 0)
      throw std::runtime_error("Paimon reader returned an invalid batch profile");
    if (array->length > max_rows)
      throw scan_manager::preparation_resource_error("Paimon batch exceeds its declared row bound",
                                                     true);
    detach(*schema, *array);
    check_stop(stop);
    return true;
  } catch (...) {
    failure_ = std::current_exception();
    throw;
  }
}
void paimon_batch_cursor::close()
{
  if (!reader_) return;
  phase_      = phase::closed;
  auto reader = std::move(reader_);
  try {
    reader->Close();
  } catch (...) {
    if (!failure_) failure_ = std::current_exception();
    throw;
  }
}
}  // namespace sirius::op::scan
