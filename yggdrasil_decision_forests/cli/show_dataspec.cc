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

// Print a human readable representation of a dataspec.
//
// Usage example:
//   bazel run -c opt :show_dataspec --dataspec=spec.pb
//
#include "absl/flags/flag.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"

ABSL_FLAG(std::string, dataspec, "",
          "Path to dataset specification (dataspec).");

ABSL_FLAG(bool, sort_by_column_names, true,
          "If true, sort the columns by names. If false, sort the columns by "
          "column index.");

ABSL_FLAG(bool, is_text_proto, true,
          "If true, the dataset is read as a text proto. If false, the "
          "dataspec is read as a binary proto.");

constexpr char kUsageMessage[] =
    "Print a human readable representation of a dataspec.";

namespace yggdrasil_decision_forests {
namespace cli {
namespace {

void ShowDataspec() {
  QCHECK(!absl::GetFlag(FLAGS_dataspec).empty());
  dataset::proto::DataSpecification data_spec;
  if (absl::GetFlag(FLAGS_is_text_proto)) {
    QCHECK_OK(file::GetTextProto(absl::GetFlag(FLAGS_dataspec), &data_spec,
                                 file::Defaults()));
  } else {
    QCHECK_OK(file::GetBinaryProto(absl::GetFlag(FLAGS_dataspec), &data_spec,
                                   file::Defaults()));
  }
  std::cout << dataset::PrintHumanReadable(
      data_spec, absl::GetFlag(FLAGS_sort_by_column_names));
}

}  // namespace
}  // namespace cli
}  // namespace yggdrasil_decision_forests

int main(int argc, char** argv) {
  InitLogging(kUsageMessage, &argc, &argv, true);
  yggdrasil_decision_forests::cli::ShowDataspec();
  return 0;
}
