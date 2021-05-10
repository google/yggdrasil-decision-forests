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

// Infers the dataspec of a dataset.
//
// A dataspec is the list of features, their type and meta data. A dataspec is
// used for all dataset IO operations.
//
// Usage example:
//   bazel run -c opt :infer_dataspec -- \
//     --dataset=csv:data.csv \
//     --output=spec.pbtxt
//
//   You can then visualize "spec.pbtxt" directly or using :show_dataspec.
//
#include "absl/flags/flag.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"

ABSL_FLAG(std::string, output, "", "Output dataspec path.");

ABSL_FLAG(std::string, dataset, "",
          "Typed path to training dataset i.e. [type]:[path] format.");

ABSL_FLAG(std::string, guide, "",
          "Path to an optional dataset specification guide "
          "(DataSpecificationGuide Text proto). Use to override the automatic "
          "type detection of the columns.");

constexpr char kUsageMessage[] =
    "Infers the dataspec of a dataset i.e. the name, type and meta-data of the "
    "dataset columns.";

namespace yggdrasil_decision_forests {
namespace cli {
namespace {

void InferDataspec() {
  QCHECK(!absl::GetFlag(FLAGS_dataset).empty());
  QCHECK(!absl::GetFlag(FLAGS_output).empty());

  dataset::proto::DataSpecificationGuide guide;
  if (!absl::GetFlag(FLAGS_guide).empty()) {
    QCHECK_OK(file::GetTextProto(absl::GetFlag(FLAGS_guide), &guide,
                                 file::Defaults()));
  }

  dataset::proto::DataSpecification data_spec;
  dataset::CreateDataSpec(absl::GetFlag(FLAGS_dataset), false, guide,
                          &data_spec);

  QCHECK_OK(file::SetTextProto(absl::GetFlag(FLAGS_output), data_spec,
                               file::Defaults()));
}

}  // namespace
}  // namespace cli
}  // namespace yggdrasil_decision_forests

int main(int argc, char** argv) {
  InitLogging(kUsageMessage, &argc, &argv, true);
  yggdrasil_decision_forests::cli::InferDataspec();
  return 0;
}
