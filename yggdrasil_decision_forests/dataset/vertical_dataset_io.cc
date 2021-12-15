/*
 * Copyright 2021 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"

#include <memory>
#include <type_traits>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/example_reader.h"
#include "yggdrasil_decision_forests/dataset/example_reader_interface.h"
#include "yggdrasil_decision_forests/dataset/example_writer.h"
#include "yggdrasil_decision_forests/dataset/example_writer_interface.h"
#include "yggdrasil_decision_forests/dataset/formats.h"
#include "yggdrasil_decision_forests/dataset/formats.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/concurrency_streamprocessor.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/sharded_io.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace dataset {
namespace {

// Loads the datasets using a single thread. This solution is more memory
// efficient that per-shard loading as examples are directly integrated into the
// vertical representation.
absl::Status LoadVerticalDatasetSingleThread(
    const absl::string_view typed_path,
    const proto::DataSpecification& data_spec, VerticalDataset* dataset,
    absl::optional<std::vector<int>> ensure_non_missing,
    const LoadConfig& config) {
  // Initialize dataset.
  dataset->set_data_spec(data_spec);
  RETURN_IF_ERROR(dataset->CreateColumnsFromDataspec());
  dataset->set_nrow(0);

  // Read and record the examples.
  ASSIGN_OR_RETURN(auto reader, CreateExampleReader(typed_path, data_spec,
                                                    ensure_non_missing));

  // Number of skipped example because of "config.load_example".
  std::size_t skipped_examples = 0;

  proto::Example example;
  utils::StatusOr<bool> status;
  while ((status = reader->Next(&example)).ok() && status.value()) {
    if (config.load_example.has_value() &&
        !config.load_example.value()(example)) {
      skipped_examples++;
      continue;
    }
    dataset->AppendExample(example, config.load_columns);
    if ((dataset->nrow() % 100) == 0) {
      LOG_INFO_EVERY_N_SEC(30, _ << dataset->nrow() << " examples scanned.");
    }
  }

  dataset->ShrinkToFit();

  LOG_INFO_EVERY_N_SEC(
      30, _ << dataset->nrow() << " examples read. Memory: "
            << dataset->MemorySummary() << ". " << skipped_examples << " ("
            << 100 * skipped_examples /
                   std::max<size_t>(1, dataset->nrow() + skipped_examples)
            << "%) examples have been skipped.");

  return status.status();
}

// Set of examples extracted by a worker.
struct BlockOfExamples {
  // List of examples. These messages are allocated in "arena".
  std::vector<proto::Example*> examples;
  google::protobuf::Arena arena;
};

// Reads a shard.
utils::StatusOr<std::unique_ptr<BlockOfExamples>> LoadShard(
    const proto::DataSpecification& data_spec, const absl::string_view prefix,
    const absl::optional<std::vector<int>>& ensure_non_missing,
    const absl::string_view shard) {
  auto block = absl::make_unique<BlockOfExamples>();
  ASSIGN_OR_RETURN(auto reader,
                   CreateExampleReader(absl::StrCat(prefix, ":", shard),
                                       data_spec, ensure_non_missing));
  auto* example = google::protobuf::Arena::CreateMessage<proto::Example>(&block->arena);
  utils::StatusOr<bool> status;
  while ((status = reader->Next(example)).ok() && status.value()) {
    block->examples.push_back(example);
    example = google::protobuf::Arena::CreateMessage<proto::Example>(&block->arena);
  }
  return block;
}

}  // namespace

absl::Status LoadVerticalDataset(
    const absl::string_view typed_path,
    const proto::DataSpecification& data_spec, VerticalDataset* dataset,
    absl::optional<std::vector<int>> ensure_non_missing,
    const LoadConfig& config) {
  // Extract the shards from the dataset path.
  std::string path, prefix;
  ASSIGN_OR_RETURN(std::tie(prefix, path), SplitTypeAndPath(typed_path));
  std::vector<std::string> shards;
  CHECK_OK(utils::ExpandInputShards(path, &shards));

  if (shards.size() <= 1 || config.num_threads <= 1) {
    // Loading in a single thread.
    return LoadVerticalDatasetSingleThread(typed_path, data_spec, dataset,
                                           ensure_non_missing, config);
  }

  // Initialize dataset.
  dataset->set_data_spec(data_spec);
  RETURN_IF_ERROR(dataset->CreateColumnsFromDataspec());
  dataset->set_nrow(0);

  // Reads the examples in a shard.
  const auto load_shard = [&](const std::string shard)
      -> utils::StatusOr<std::unique_ptr<BlockOfExamples>> {
    return LoadShard(data_spec, prefix, ensure_non_missing, shard);
  };

  utils::concurrency::StreamProcessor<
      std::string, utils::StatusOr<std::unique_ptr<BlockOfExamples>>>
      processor("DatasetLoader",
                std::min<int>(shards.size(), config.num_threads), load_shard,
                /*result_in_order=*/true);

  // Configure the shard loading jobs.
  processor.StartWorkers();
  for (const auto& shard : shards) {
    processor.Submit(shard);
  }
  processor.CloseSubmits();

  // Number of skipped example because of "config.load_example".
  std::size_t skipped_examples = 0;

  // Ingest the examples in the vertical dataset.
  int loaded_shards = 0;
  while (true) {
    auto examples = processor.GetResult();
    if (!examples.has_value()) {
      // All the shards have been read.
      break;
    }
    RETURN_IF_ERROR(examples.value().status());
    auto block = std::move(examples.value().value());

    if (loaded_shards == 0) {
      // Reserve the vertical dataset memory by assuming that all the shards
      // have ~ the same number of examples.

      // Count the number of examples in the first shard.
      std::size_t num_examples_in_shard;
      if (config.load_example.has_value()) {
        num_examples_in_shard = 0;
        for (const auto* example : block->examples) {
          if (config.load_example.value()(*example)) {
            num_examples_in_shard++;
          }
        }
      } else {
        num_examples_in_shard = block->examples.size();
      }

      if (num_examples_in_shard > 100) {
        // The number of examples in the first shard is representative.
        const auto reserved_examples = num_examples_in_shard * shards.size();
        const auto num_features = config.load_columns.has_value()
                                      ? config.load_columns->size()
                                      : data_spec.columns_size();
        const auto approx_memory_usage_mb =
            4 * num_features * reserved_examples / 1000000;
        if (approx_memory_usage_mb > 100) {
          LOG_INFO_EVERY_N_SEC(30, _ << "Reserving " << reserved_examples
                                     << " examples and " << num_features
                                     << " features for ~"
                                     << approx_memory_usage_mb << "MB");
        }
        dataset->Reserve(reserved_examples, config.load_columns);
      }
    }
    for (const auto* example : block->examples) {
      if (config.load_example.has_value() &&
          !config.load_example.value()(*example)) {
        skipped_examples++;
        continue;
      }
      dataset->AppendExample(*example, config.load_columns);
    }
    LOG_INFO_EVERY_N_SEC(30, _ << dataset->nrow() << " examples scanned.");
    loaded_shards++;
  }

  if (loaded_shards != shards.size()) {
    return absl::InternalError("Unexpected number of shards.");
  }

  dataset->ShrinkToFit();

  processor.JoinAllAndStopThreads();
  LOG_INFO_EVERY_N_SEC(
      30, _ << dataset->nrow() << " examples and " << loaded_shards
            << " shards scanned in total. Memory: " << dataset->MemorySummary()
            << ". " << skipped_examples << " ("
            << 100 * skipped_examples /
                   std::max<size_t>(1, dataset->nrow() + skipped_examples)
            << "%) examples have been skipped.");
  return absl::OkStatus();
}

absl::Status SaveVerticalDataset(const VerticalDataset& dataset,
                                 const absl::string_view typed_path,
                                 int64_t num_records_by_shard) {
  ASSIGN_OR_RETURN(auto writer,
                   CreateExampleWriter(typed_path, dataset.data_spec(),
                                       num_records_by_shard));
  proto::Example example;
  for (VerticalDataset::row_t row = 0; row < dataset.nrow(); row++) {
    dataset.ExtractExample(row, &example);
    RETURN_IF_ERROR(writer->Write(example));
  }
  return absl::OkStatus();
}

}  // namespace dataset
}  // namespace yggdrasil_decision_forests
