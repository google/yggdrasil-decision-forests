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

// Apply a model on a dataset and export the predictions to disk.

#include <memory>
#include <vector>

#include "absl/flags/flag.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/model/prediction.pb.h"
#include "yggdrasil_decision_forests/utils/evaluation.h"
#include "yggdrasil_decision_forests/utils/logging.h"

ABSL_FLAG(std::string, model, "", "Model directory.");

ABSL_FLAG(std::string, dataset, "",
          "Typed path to dataset i.e. [type]:[path] format.");

ABSL_FLAG(std::string, output, "",
          "Output prediction specified with [type]:[path] format. e.g. "
          "\"csv:/path/to/dataset.csv\".");

ABSL_FLAG(int, num_records_by_shard_in_output, -1,
          "Number of records per output shards. Only valid if the output "
          "path is sharded (e.g. contains @10).");

constexpr char kUsageMessage[] =
    "Apply a model on a dataset and export the predictions to disk.";

namespace yggdrasil_decision_forests {
namespace cli {

void ConvertDataset() {
  // Check required flags.
  QCHECK(!absl::GetFlag(FLAGS_dataset).empty());
  QCHECK(!absl::GetFlag(FLAGS_output).empty());
  QCHECK(!absl::GetFlag(FLAGS_model).empty());

  // Load the model
  std::unique_ptr<model::AbstractModel> model;
  QCHECK_OK(model::LoadModel(absl::GetFlag(FLAGS_model), &model));
  const auto& label_column = model->data_spec().columns(model->label_col_idx());

  // Load dataset.
  dataset::VerticalDataset dataset;
  QCHECK_OK(LoadVerticalDataset(absl::GetFlag(FLAGS_dataset),
                                model->data_spec(), &dataset));

  // Compute the predictions.
  std::vector<model::proto::Prediction> predictions;
  predictions.resize(dataset.nrow());

  auto engine_or = model->BuildFastEngine();
  if (engine_or.ok()) {
    LOG(INFO) << "Run predictions with semi-fast engine";

    // Convert dataset to efficient format.
    auto engine = std::move(engine_or.value());
    auto examples = engine->AllocateExamples(dataset.nrow());
    QCHECK_OK(serving::CopyVerticalDatasetToAbstractExampleSet(
        dataset, 0, dataset.nrow(), engine->features(), examples.get()));

    // Apply the model.
    std::vector<float> fast_predictions;
    engine->Predict(*examples, dataset.nrow(), &fast_predictions);
    examples.reset();

    // Convert the prediction to the expected format.
    const int num_prediction_dimensions = engine->NumPredictionDimension();
    for (dataset::VerticalDataset::row_t example_idx = 0;
         example_idx < dataset.nrow(); example_idx++) {
      model::FloatToProtoPrediction(fast_predictions, example_idx,
                                    model->task(), num_prediction_dimensions,
                                    &predictions[example_idx]);
    }
  } else {
    LOG(INFO) << "Run predictions with slow engine";
    for (dataset::VerticalDataset::row_t example_idx = 0;
         example_idx < dataset.nrow(); example_idx++) {
      LOG_INFO_EVERY_N_SEC(30, _ << example_idx << "/" << dataset.nrow()
                                 << " predictions computed.");
      model->Predict(dataset, example_idx, &predictions[example_idx]);
    }
  }

  // Save the predictions.
  QCHECK_OK(utils::ExportPredictions(
      predictions, model->task(), label_column, absl::GetFlag(FLAGS_output),
      absl::GetFlag(FLAGS_num_records_by_shard_in_output)));
}

}  // namespace cli
}  // namespace yggdrasil_decision_forests

int main(int argc, char** argv) {
  InitLogging(kUsageMessage, &argc, &argv, true);
  yggdrasil_decision_forests::cli::ConvertDataset();
  return 0;
}
