/*
 * Copyright 2022 Google LLC.
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

// Converts a dataset from one format to another. The dataspec of the dataset
// should be available.
//
// Usage example:
//   bazel run -c opt :infer_dataspec -- \
//     --dataset=csv:/my/dataset.csv \
//     --output=spec.pbtxt
//
//   bazel run -c opt :convert_dataset -- \
//     --input=csv:/my/dataset.csv \
//     --dataspec= spec.pbtxt \
//     --output=tfrecord+tfe:/my/dataset.tfrecord-tfe
//
#include "absl/flags/flag.h"
#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example_reader.h"
#include "yggdrasil_decision_forests/dataset/example_writer.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"

ABSL_FLAG(std::string, input, "",
          "Input dataset specified with [type]:[path] format.");

ABSL_FLAG(std::string, output, "",
          "Output dataset specified with [type]:[path] format.");

ABSL_FLAG(int, shard_size, -1,
          "Number of record per output shards. Only valid if the output "
          "path is sharded (e.g. contains @10). This flag is required as this "
          "conversion is greedy. If num_records_by_shard is too low, all the "
          "remaining examples will be put in the last shard.");

ABSL_FLAG(std::string, dataspec, "",
          "Input data specification path. This file is generally created "
          "with :infer_dataspec and inspected with :show_dataspec.");

ABSL_FLAG(bool, dataspec_is_binary, false,
          "If true, the dataspec is a binary proto. If false (default), the "
          "dataspec is a text proto. The :infer_dataspec cli generates text "
          "proto dataspec, while the dataspec contained in a model is encoded "
          "as a binary proto.");

ABSL_FLAG(bool, ignore_missing_columns, false,
          "If false (default), fails if one of the column in the dataspec is "
          "missing. If true, fill missing columns with \"missing values\".");

constexpr char kUsageMessage[] =
    "Converts a dataset from one format to another. The dataspec of the "
    "dataset should be available.";

namespace yggdrasil_decision_forests {
namespace cli {

void ConvertDataset() {
  // Check required flags.
  QCHECK(!absl::GetFlag(FLAGS_input).empty());
  QCHECK(!absl::GetFlag(FLAGS_dataspec).empty());
  QCHECK(!absl::GetFlag(FLAGS_output).empty());

  // Load the dataspec.
  dataset::proto::DataSpecification data_spec;

  if (absl::GetFlag(FLAGS_dataspec_is_binary)) {
    QCHECK_OK(file::GetBinaryProto(absl::GetFlag(FLAGS_dataspec), &data_spec,
                                   file::Defaults()));
  } else {
    QCHECK_OK(file::GetTextProto(absl::GetFlag(FLAGS_dataspec), &data_spec,
                                 file::Defaults()));
  }

  absl::optional<std::vector<int>> required_columns;
  if (absl::GetFlag(FLAGS_ignore_missing_columns)) {
    // Ignore all missing columns.
    required_columns = std::vector<int>{};
  }

  // Create the reader.
  auto reader = dataset::CreateExampleReader(absl::GetFlag(FLAGS_input),
                                             data_spec, required_columns)
                    .value();

  // Create the writer.
  auto writer =
      dataset::CreateExampleWriter(absl::GetFlag(FLAGS_output), data_spec,
                                   absl::GetFlag(FLAGS_shard_size))
          .value();

  dataset::proto::Example example;
  int64_t nrow = 0;
  absl::StatusOr<bool> status;
  while ((status = reader->Next(&example)).ok() && status.value()) {
    LOG_INFO_EVERY_N_SEC(30, _ << nrow << " examples converted.");
    QCHECK_OK(writer->Write(example));
    nrow++;
  }

  YDF_LOG(INFO) << "Converting done. " << nrow << " example(s) converted.";
}

}  // namespace cli
}  // namespace yggdrasil_decision_forests

int main(int argc, char** argv) {
  InitLogging(kUsageMessage, &argc, &argv, true);
  yggdrasil_decision_forests::cli::ConvertDataset();
  return 0;
}
