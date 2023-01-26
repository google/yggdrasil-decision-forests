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

// Create an HTML report containing interpretation information about a model and
// a dataset.
//
// Usage example:
//
//  bazel run :analyze_model_and_dataset -- \
//    --model=/tmp/model\
//    --dataset=csv:/tmp/dataset.csv \
//    --output=/tmp/analyis_report
//

#include <memory>

#include "absl/flags/flag.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/model_analysis.h"
#include "yggdrasil_decision_forests/utils/protobuf.h"

// Input/output flags.
ABSL_FLAG(std::string, model, "", "Input model directory. Optional.");
ABSL_FLAG(std::string, dataset, "",
          "Input typed dataset e.g. \"csv:my_file.csv\". Optional.");
ABSL_FLAG(std::string, output, "", "Output directory for the HTML report.");
ABSL_FLAG(std::string, options, "",
          "Serialized text proto utils::model_analysis::proto::Options");

constexpr char kUsageMessage[] =
    "Create an HTML report containing interpretation information about a model "
    "and a dataset.";

namespace yggdrasil_decision_forests {
namespace cli {

void AnalyseDatasetAndModel() {
  YDF_LOG(INFO) << "Warning: The :analyze_model_and_dataset is experimental";

  // Check required flags.
  QCHECK(!absl::GetFlag(FLAGS_model).empty());
  QCHECK(!absl::GetFlag(FLAGS_dataset).empty());
  QCHECK(!absl::GetFlag(FLAGS_output).empty());
  utils::model_analysis::proto::Options options;
  if (!absl::GetFlag(FLAGS_options).empty()) {
    options = utils::ParseTextProto<utils::model_analysis::proto::Options>(
                  absl::GetFlag(FLAGS_options))
                  .value();
  }

  YDF_LOG(INFO) << "Load model";
  std::unique_ptr<model::AbstractModel> model;
  QCHECK_OK(model::LoadModel(absl::GetFlag(FLAGS_model), &model));

  YDF_LOG(INFO) << "Load dataset";
  dataset::VerticalDataset dataset;
  QCHECK_OK(dataset::LoadVerticalDataset(
      absl::GetFlag(FLAGS_dataset), model->data_spec(), &dataset,
      /*ensure_non_missing=*/model->input_features()));

  QCHECK_OK(utils::model_analysis::AnalyseAndCreateHtmlReport(
      *model, dataset, absl::GetFlag(FLAGS_model), absl::GetFlag(FLAGS_dataset),
      absl::GetFlag(FLAGS_output), options));
}

}  // namespace cli
}  // namespace yggdrasil_decision_forests

int main(int argc, char** argv) {
  InitLogging(kUsageMessage, &argc, &argv, true);
  yggdrasil_decision_forests::cli::AnalyseDatasetAndModel();
  return 0;
}
