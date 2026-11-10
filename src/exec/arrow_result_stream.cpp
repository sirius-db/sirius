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

#include "arrow_result_stream.hpp"

#include <duckdb/common/arrow/arrow_appender.hpp>
#include <duckdb/common/arrow/arrow_wrapper.hpp>
#include <duckdb/function/table/arrow/arrow_duck_schema.hpp>
#include <duckdb/main/client_context.hpp>

#include <algorithm>
#include <cerrno>
#include <exception>
#include <memory>
#include <utility>

namespace sirius::exec {
namespace {

struct arrow_result_state {
  explicit arrow_result_state(duckdb::unique_ptr<duckdb::QueryResult> result_p,
                              duckdb::idx_t batch_size_p)
    : result(std::move(result_p)), batch_size(batch_size_p)
  {
    if (!result || batch_size == 0) {
      throw duckdb::InvalidInputException(
        "Arrow export requires a result and a nonzero batch size");
    }
    result->ThrowIfError();
    auto client = result->client_properties.client_context;
    if (!client) { throw duckdb::InvalidInputException("Arrow export requires a client context"); }
    context = client->shared_from_this();
    result->Collection();
    extension_types =
      duckdb::ArrowTypeExtensionData::GetExtensionTypes(*client, result->GetTypes());
  }

  // Arrow conversion can consult the connection even after query execution ends.
  duckdb::shared_ptr<duckdb::ClientContext> context;
  duckdb::unique_ptr<duckdb::QueryResult> result;
  duckdb::unordered_map<duckdb::idx_t, const duckdb::shared_ptr<duckdb::ArrowTypeExtensionData>>
    extension_types;
  duckdb::unique_ptr<duckdb::DataChunk> chunk;
  duckdb::idx_t offset = 0;
  duckdb::idx_t batch_size;
  bool finished = false;
  std::exception_ptr error;
};

template <typename F>
int with_stream(ArrowArrayStream* stream, F&& callback) noexcept
{
  if (!stream || !stream->release) { return EINVAL; }
  auto& state = *static_cast<arrow_result_state*>(stream->private_data);
  if (state.error) { return EIO; }
  try {
    callback(state);
    return 0;
  } catch (...) {
    state.error = std::current_exception();
    return EIO;
  }
}

int get_schema(ArrowArrayStream* stream, ArrowSchema* out) noexcept
{
  if (!out) { return EINVAL; }
  *out = {};
  return with_stream(stream, [&](arrow_result_state& state) {
    duckdb::ArrowSchemaWrapper schema;
    duckdb::ArrowConverter::ToArrowSchema(&schema.arrow_schema,
                                          state.result->GetTypes(),
                                          duckdb::IdentifiersToStrings(state.result->GetNames()),
                                          state.result->client_properties);
    *out                        = schema.arrow_schema;
    schema.arrow_schema.release = nullptr;
  });
}

int get_next(ArrowArrayStream* stream, ArrowArray* out) noexcept
{
  if (!out) { return EINVAL; }
  *out = {};
  return with_stream(stream, [&](arrow_result_state& state) {
    if (state.finished) { return; }
    duckdb::ArrowAppender appender(state.result->GetTypes(),
                                   std::min(state.batch_size, state.result->RowCount()),
                                   state.result->client_properties,
                                   state.extension_types);
    duckdb::idx_t rows = 0;
    while (rows < state.batch_size) {
      if (!state.chunk || state.offset == state.chunk->size()) {
        state.chunk  = state.result->Fetch();
        state.offset = 0;
        if (!state.chunk || state.chunk->size() == 0) {
          state.finished = true;
          break;
        }
      }
      auto const count = std::min(state.batch_size - rows, state.chunk->size() - state.offset);
      appender.Append(*state.chunk, state.offset, state.offset + count, state.chunk->size());
      state.offset += count;
      rows += count;
    }
    if (rows != 0) { *out = appender.Finalize(); }
  });
}

const char* get_last_error(ArrowArrayStream* stream) noexcept
{
  if (!stream || !stream->release) { return "Arrow stream was released"; }
  auto& state = *static_cast<arrow_result_state*>(stream->private_data);
  if (!state.error) { return nullptr; }
  try {
    std::rethrow_exception(state.error);
  } catch (std::exception const& error) {
    return error.what();
  } catch (...) {
    return "Arrow result conversion failed";
  }
}

void release(ArrowArrayStream* stream) noexcept
{
  if (!stream || !stream->release) { return; }
  stream->release = nullptr;
  delete static_cast<arrow_result_state*>(stream->private_data);
  stream->private_data = nullptr;
}

}  // namespace

void export_arrow_result(duckdb::unique_ptr<duckdb::QueryResult> result,
                         ArrowArrayStream& out,
                         duckdb::idx_t batch_size)
{
  out        = {};
  auto state = std::make_unique<arrow_result_state>(std::move(result), batch_size);
  out        = {get_schema, get_next, get_last_error, release, state.release()};
}

}  // namespace sirius::exec
