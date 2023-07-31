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

// Apply a model on a dataset and export the predictions to disk.
//
// The output predictions are ordered similarly as to the examples in the input
// dataset. In case of a sharded dataset, predictions are exported in shards in
// numerical order.
//
// The prediction representation depends on the model task.
//
// For example, for a classification model, the predictions consist of the
// predicted probability of each of the classes, and the names of the prediction
// columns are equal to the corresponding classes. Note that if the label does
// not have a dictionary (i.e., the classification labels are integers), the
// name of the column will look like integers e.g. "1", "2", etc.
//
// Usage example:
//  predict \
//    --model=/path/to/my_model \
//    --dataset=csv:/path/to/dataset@10 \
//    --output=csv:/path/to/predictions.csv
//
#include <memory>
#include <vector>

#include "absl/flags/flag.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
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

ABSL_FLAG(std::string, key, "",
          "If set, copies the column \"key\" in the output prediction file. "
          "This key column cannot be an input feature of the model.");

constexpr char kUsageMessage[] =
    "Apply a model on a dataset and export the predictions to disk.";

namespace yggdrasil_decision_forests {
namespace cli {

void Predict() {
  // Check required flags.
  QCHECK(!absl::GetFlag(FLAGS_dataset).empty());
  QCHECK(!absl::GetFlag(FLAGS_output).empty());
  QCHECK(!absl::GetFlag(FLAGS_model).empty());

  // Load the model
  std::unique_ptr<model::AbstractModel> model;
  QCHECK_OK(model::LoadModel(absl::GetFlag(FLAGS_model), &model));

  // Dataspec used to read the dataset.
  auto data_spec = model->data_spec();

  // List of columns that cannot be missing in the dataset.
  auto required_columns = model->input_features();

  // Name of the column to copy along side the predictions.
  const auto key = absl::GetFlag(FLAGS_key);

  // Column index of the key in the dataspec. -1 if there is no key.
  absl::optional<int> key_col_idx;
  if (!key.empty()) {
    key_col_idx = dataset::GetOptionalColumnIdxFromName(key, data_spec);
    if (!key_col_idx.has_value()) {
      // The key column does not exist. Let's create it.
      key_col_idx = data_spec.columns_size();
      dataset::AddColumn(key, dataset::proto::STRING, &data_spec);
    } else {
      // The key column already exists.
      const bool key_is_input_feature =
          std::find(required_columns.begin(), required_columns.end(),
                    key_col_idx) != required_columns.end();
      if (key_is_input_feature) {
        YDF_LOG(FATAL) << "The --key cannot be an input feature of the model.";
      }
      // Turn the column into a raw string to make sure no processing is applied
      // during reading.
      data_spec.mutable_columns(key_col_idx.value())
          ->set_type(dataset::proto::STRING);
    }
    // If specified, the key column should be present.
    required_columns.push_back(key_col_idx.value());
  }

  // Load dataset.
  //
  // The columns corresponding to the model input features are required (i.e.
  // loading the dataset will fail if one of them is missing). The other columns
  // (e.g. label, weights) are optional.
  dataset::VerticalDataset dataset;
  QCHECK_OK(LoadVerticalDataset(absl::GetFlag(FLAGS_dataset), data_spec,
                                &dataset,
                                /*ensure_non_missing=*/required_columns));

  // Compute the predictions.
  std::vector<model::proto::Prediction> predictions;
  predictions.resize(dataset.nrow());

  auto engine_or = model->BuildFastEngine();
  if (engine_or.ok()) {
    YDF_LOG(INFO) << "Run predictions with semi-fast engine";

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
      auto& prediction = predictions[example_idx];
      model::FloatToProtoPrediction(fast_predictions, example_idx,
                                    model->task(), num_prediction_dimensions,
                                    &prediction);

      if (key_col_idx.has_value()) {
        // Copy the key information.
        prediction.set_example_key(
            dataset.ValueToString(example_idx, key_col_idx.value()));
      }
    }
  } else {
    YDF_LOG(INFO) << "Run predictions with slow engine";
    for (dataset::VerticalDataset::row_t example_idx = 0;
         example_idx < dataset.nrow(); example_idx++) {
      LOG_INFO_EVERY_N_SEC(30, _ << example_idx << "/" << dataset.nrow()
                                 << " predictions computed.");
      auto& prediction = predictions[example_idx];
      model->Predict(dataset, example_idx, &prediction);

      if (key_col_idx.has_value()) {
        // Copy the key information.
        prediction.set_example_key(
            dataset.ValueToString(example_idx, key_col_idx.value()));
      }
    }
  }

  // Save the predictions.
  const auto& label_column = model->data_spec().columns(model->label_col_idx());

  absl::optional<std::string> optional_prediction_key;
  if (key_col_idx.has_value()) {
    optional_prediction_key = key;
  }

  QCHECK_OK(utils::ExportPredictions(
      predictions, model->task(), label_column, absl::GetFlag(FLAGS_output),
      absl::GetFlag(FLAGS_num_records_by_shard_in_output),
      optional_prediction_key));
}

}  // namespace cli
}  // namespace yggdrasil_decision_forests

int main(int argc, char** argv) {
  InitLogging(kUsageMessage, &argc, &argv, true);
  yggdrasil_decision_forests::cli::Predict();
  return 0;
}
