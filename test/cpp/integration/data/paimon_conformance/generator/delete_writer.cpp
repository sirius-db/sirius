/*
 * Copyright 2026, Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Uses the public writer/commit API illustrated by Apache Paimon's
 * examples/read_write_demo.cpp (Apache-2.0). No storage files are fabricated.
 */
#include <arrow/api.h>
#include <arrow/c/bridge.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <paimon/api.h>
#include <paimon/catalog/catalog.h>

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

template <typename T>
T checked(paimon::Result<T> result)
{
  if (!result.ok()) { throw std::runtime_error(result.status().ToString()); }
  return std::move(result).value();
}

template <typename T>
T checked(arrow::Result<T> result)
{
  if (!result.ok()) { throw std::runtime_error(result.status().ToString()); }
  return std::move(result).ValueOrDie();
}

template <typename Status>
void require_ok(Status status)
{
  if (!status.ok()) { throw std::runtime_error(status.ToString()); }
}

const std::map<std::string, std::string> options = {{"file.format", "parquet"},
                                                    {"manifest.format", "avro"},
                                                    {"file-system", "local"},
                                                    {"write-only", "true"},
                                                    {"snapshot.num-retained.min", "100"},
                                                    {"snapshot.num-retained.max", "100"}};

void metadata(std::string const& path)
{
  paimon::ScanContextBuilder builder(path);
  builder.SetOptions(options);
  auto scanner = checked(paimon::TableScan::Create(checked(builder.Finish())));
  auto plan    = checked(scanner->CreatePlan());
  std::cout << "{\"snapshot_id\":";
  if (plan->SnapshotId()) {
    std::cout << *plan->SnapshotId();
  } else {
    std::cout << "null";
  }
  std::cout << "}\n";
}

int main(int argc, char** argv)
{
  try {
    // A deliberately small fixture utility, not a general-purpose Paimon writer.
    if (argc != 5 && argc != 7) {
      throw std::runtime_error(
        "Usage: delete_writer create|write warehouse table arrow-ipc-file [insert|delete "
        "commit-label]");
    }
    std::string command = argv[1], root = argv[2], table = argv[3];
    if (table != "orders_pk" ||
        !((command == "create" && argc == 5) || (command == "write" && argc == 7))) {
      throw std::runtime_error(
        "Only the orders_pk fixture and create/write commands are supported");
    }
    auto path   = root + "/reference.db/" + table;
    auto input  = checked(arrow::io::ReadableFile::Open(argv[4]));
    auto reader = checked(arrow::ipc::RecordBatchFileReader::Open(input));
    if (reader->num_record_batches() != 1) {
      throw std::runtime_error("Fixture input must contain exactly one Arrow batch");
    }
    auto batch = checked(reader->ReadRecordBatch(0));
    if (command == "create") {
      auto catalog = checked(paimon::Catalog::Create(root, options));
      require_ok(catalog->CreateDatabase("reference", options, true));
      ArrowSchema schema{};
      require_ok(arrow::ExportSchema(*batch->schema(), &schema));
      auto table_options            = options;
      table_options["bucket"]       = "1";
      table_options["merge-engine"] = "deduplicate";
      auto status                   = catalog->CreateTable(paimon::Identifier("reference", table),
                                         &schema,
                                         std::vector<std::string>{},
                                         std::vector<std::string>{"id"},
                                         table_options,
                                         false);
      if (schema.release) { schema.release(&schema); }
      require_ok(status);
    } else {
      std::string kind = argv[5];
      if (kind != "insert" && kind != "delete") {
        throw std::runtime_error("Row kind must be insert or delete");
      }
      std::string commit_user = "sirius-reference-" + std::string(argv[6]);
      paimon::WriteContextBuilder write_context(path, commit_user);
      auto writer = checked(
        paimon::FileStoreWrite::Create(checked(write_context.SetOptions(options).Finish())));
      auto array = checked(batch->ToStructArray());
      ArrowArray exported{};
      require_ok(arrow::ExportArray(*array, &exported));
      paimon::RecordBatchBuilder builder(&exported);
      builder.SetBucket(0);
      builder.SetRowKinds(std::vector<paimon::RecordBatch::RowKind>(
        batch->num_rows(),
        kind == "delete" ? paimon::RecordBatch::RowKind::DELETE
                         : paimon::RecordBatch::RowKind::INSERT));
      require_ok(writer->Write(checked(builder.Finish())));
      auto messages = checked(writer->PrepareCommit());
      paimon::CommitContextBuilder commit_context(path, commit_user);
      auto committer = checked(
        paimon::FileStoreCommit::Create(checked(commit_context.SetOptions(options).Finish())));
      require_ok(committer->Commit(messages));
      require_ok(writer->Close());
    }
    metadata(path);
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "Fixture generation failed: " << error.what() << '\n';
    return 1;
  }
}
