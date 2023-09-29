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

// Edit a model.
//
// Usage example:
//
//   # Change the name of the label
//   bazel run -c opt :edit_model -- \
//     --input=/path/to/original_model \
//     --output=/path/to/final_model \
//     --new_label_name="NEW_LABEL"
//
// The available edit actions are:
//
//   If new_label_name is set:
//     Changes the label's name to --new_label_name. For example, this operation
//     is useful to set a specific label name in a TensorFlow Decision Forests
//     model (as in TF-DF, the label name is always "__LABEL").
//
//   If new_weights_name is set:
//     Changes the weight name to --new_weights_name. For example, this
//     operation is useful to set a specific label name in a TensorFlow Decision
//     Forests model (as in TF-DF, the label name is always "__WEIGHTS").
//
//   If new_file_prefix is set:
//     Changes the model filename prefix (i.e. the prefix string added to all
//     the model filenames) with "new_file_prefix". Set "new_file_prefix" to the
//     empty string (i.e. --new_file_prefix=) to remove the model prefix.
//
#include "absl/flags/flag.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/utils/logging.h"

// Default string flag values. Used to detect if a flag is set by the user for
// flag where the empty value is possible.
constexpr char kStringNoSet[] = "__NO__SET__";

ABSL_FLAG(std::string, input, kStringNoSet, "Input model directory.");
ABSL_FLAG(std::string, output, kStringNoSet, "Output model directory.");

ABSL_FLAG(std::string, new_label_name, kStringNoSet, "New label name.");
ABSL_FLAG(std::string, new_weights_name, kStringNoSet, "New weights name.");
ABSL_FLAG(std::string, new_file_prefix, kStringNoSet,
          "New prefix in the filenames.");
ABSL_FLAG(std::string, pure_serving, kStringNoSet,
          "Clear the model from any information that is not required for model "
          "serving.This includes debugging, model interpretation and other "
          "meta-data. Can reduce significantly the size of the model.");

constexpr char kUsageMessage[] = "Edits a trained model.";

namespace yggdrasil_decision_forests {
namespace cli {

void EditModel() {
  // Check required flags.
  const auto input = absl::GetFlag(FLAGS_input);
  const auto output = absl::GetFlag(FLAGS_output);

  if (input == kStringNoSet) {
    YDF_LOG(FATAL) << "--input required";
  }
  if (output == kStringNoSet) {
    YDF_LOG(FATAL) << "--output required";
  }

  YDF_LOG(INFO) << "Loading model";
  std::unique_ptr<model::AbstractModel> model;
  QCHECK_OK(model::LoadModel(input, &model));
  auto* label_column =
      model->mutable_data_spec()->mutable_columns(model->label_col_idx());

  YDF_LOG(INFO) << "Apply action";

  // Change the name of the label.
  if (absl::GetFlag(FLAGS_new_label_name) != kStringNoSet) {
    label_column->set_name(absl::GetFlag(FLAGS_new_label_name));
  }

  // Change the name of the weights.
  if (absl::GetFlag(FLAGS_new_weights_name) != kStringNoSet) {
    auto weights = model->weights();
    if (!weights.has_value()) {
      YDF_LOG(FATAL)
          << "Cannot apply --new_weights_name because the model is not "
             "weighted.";
    }
    auto* weight_column = model->mutable_data_spec()->mutable_columns(
        weights.value().attribute_idx());
    weight_column->set_name(absl::GetFlag(FLAGS_new_weights_name));
  }

  // Pure serving
  if (absl::GetFlag(FLAGS_pure_serving) != kStringNoSet) {
    QCHECK_OK(model->MakePureServing());
  }

  // Change how the model is exported.
  model::ModelIOOptions output_options;
  if (absl::GetFlag(FLAGS_new_file_prefix) != kStringNoSet) {
    output_options.file_prefix = absl::GetFlag(FLAGS_new_file_prefix);
  }
  YDF_LOG(INFO) << "Saving model";
  QCHECK_OK(model::SaveModel(output, model.get(), output_options));
}

}  // namespace cli
}  // namespace yggdrasil_decision_forests

int main(int argc, char** argv) {
  InitLogging(kUsageMessage, &argc, &argv, true);
  yggdrasil_decision_forests::cli::EditModel();
  return 0;
}
