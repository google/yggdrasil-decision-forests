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

// Computes various variable importances, and add the result to the model
// metadata. The results can be seen with the ":show_model" tool.
//
// See the documentation for the definition of variable importances:
// https://ydf.readthedocs.io/en/latest/cli_user_manual.html#variable-importances
//
// This tool is useful for programmatic analysis of pre-trained model on large
// datasets.
//
// Usage example:
//
// :train ... --dataset=csv:train.csv --output=model
// :compute_variable_importances --input_model=model \
//   --output_model=model_with_vi --dataset=csv:test.csv
// :show_model --model=model
//
#include <memory>

#include "absl/flags/flag.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/utils/feature_importance.h"
#include "yggdrasil_decision_forests/utils/logging.h"

ABSL_FLAG(std::string, input_model, "", "Input model.");
ABSL_FLAG(std::string, output_model, "", "Output model.");
ABSL_FLAG(std::string, dataset, "",
          "Typed path to dataset i.e. [type]:[path] format.");
ABSL_FLAG(std::string, variable_importance_prefix, "",
          "Prefix added in front of the variable importance names.");
ABSL_FLAG(int, num_io_threads, 10,
          "Number of threads used to read the dataset.");
ABSL_FLAG(int, num_compute_threads, 6,
          "Number of threads used for the computation.");
ABSL_FLAG(int, num_repetitions, 1,
          "Number of times variable importances are evaluated and averaged. "
          "Larger numbers increase the quality of the estimate at the expense "
          "of the computation time.");

constexpr char kUsageMessage[] =
    "Compute the variable importances and add them to a model metadata. The "
    "results can be seen with the :show_model tool.";

namespace yggdrasil_decision_forests {
namespace cli {

void PermutationVI() {
  // Check required flags.
  QCHECK(!absl::GetFlag(FLAGS_input_model).empty());
  QCHECK(!absl::GetFlag(FLAGS_output_model).empty());
  QCHECK(!absl::GetFlag(FLAGS_dataset).empty());

  YDF_LOG(INFO) << "Load model";
  std::unique_ptr<model::AbstractModel> model;
  QCHECK_OK(model::LoadModel(absl::GetFlag(FLAGS_input_model), &model));

  YDF_LOG(INFO) << "Load dataset";
  dataset::LoadConfig read_dataset_options;
  read_dataset_options.num_threads = absl::GetFlag(FLAGS_num_io_threads);
  dataset::VerticalDataset dataset;
  QCHECK_OK(dataset::LoadVerticalDataset(
      absl::GetFlag(FLAGS_dataset), model->data_spec(), &dataset,
      /*ensure_non_missing=*/model->input_features(), read_dataset_options));

  YDF_LOG(INFO) << "Compute the permutation variable importances";
  utils::ComputeFeatureImportanceOptions options;
  options.num_threads = absl::GetFlag(FLAGS_num_compute_threads);
  options.num_rounds = absl::GetFlag(FLAGS_num_repetitions);
  QCHECK_OK(utils::ComputePermutationFeatureImportance(
      dataset, model.get(), model->mutable_precomputed_variable_importances(),
      options));

  YDF_LOG(INFO) << "Save model";
  QCHECK_OK(model::SaveModel(absl::GetFlag(FLAGS_output_model), model.get()));
}

}  // namespace cli
}  // namespace yggdrasil_decision_forests

int main(int argc, char** argv) {
  InitLogging(kUsageMessage, &argc, &argv, true);
  yggdrasil_decision_forests::cli::PermutationVI();
  return 0;
}
