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

// Demonstrates how to do fast training and inference using 8 bits numerical
// values.
//
// Usage example:
//   bazel build -c opt \
//   //yggdrasil_decision_forests/examples:fast_8bits_numerical
//
//   bazel-bin/yggdrasil_decision_forests/examples/fast_8bits_numerical\
//   --alsologtostderr
//

#include <cstdint>
#include <random>

#include "absl/flags/flag.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/metric/report.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/serving/decision_forest/8bits_numerical_features.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"

ABSL_FLAG(std::string, work_dir, "/tmp/fast_8bits_numerical",
          "Directory to store the model and dataset. Nothing if saved if not "
          "provided.");

namespace ydf = yggdrasil_decision_forests;

// Creates a dataspec for the the synthetic 8bits dataset.
ydf::dataset::proto::DataSpecification BuildDataspec(const int num_features) {
  ydf::dataset::proto::DataSpecification dataspec;

  for (int feature_idx = 0; feature_idx < num_features; feature_idx++) {
    auto* feature = ydf::dataset::AddColumn(
        absl::StrCat("feature_", feature_idx),
        ydf::dataset::proto::ColumnType::DISCRETIZED_NUMERICAL, &dataspec);

    feature->mutable_discretized_numerical()->set_maximum_num_bins(256);
    for (int boundary_idx = 0; boundary_idx < 255; boundary_idx++) {
      feature->mutable_discretized_numerical()->add_boundaries(
          static_cast<float>(boundary_idx) + 0.5f);
    }
  }

  // The label is the last column.
  auto* label = ydf::dataset::AddColumn(
      "label", ydf::dataset::proto::ColumnType::CATEGORICAL, &dataspec);
  // Binary classification
  label->mutable_categorical()->set_number_of_unique_values(3);
  label->mutable_categorical()->set_is_already_integerized(true);

  return dataspec;
}

ydf::dataset::VerticalDataset CreateSyntheticDataset(
    const ydf::dataset::proto::DataSpecification& dataspec,
    const int num_features, const int num_examples) {
  ydf::dataset::VerticalDataset dataset;
  *dataset.mutable_data_spec() = dataspec;
  CHECK_OK(dataset.CreateColumnsFromDataspec());
  dataset.Resize(num_examples);

  std::mt19937_64 rnd;
  std::uniform_int_distribution<uint8_t> feature_dist(0, 255);

  double max_acc = 255 * num_features * (num_features + 1) / 2;

  for (int example_idx = 0; example_idx < num_examples; example_idx++) {
    double acc = 0;
    for (int feature_idx = 0; feature_idx < num_features; feature_idx++) {
      const uint8_t value = feature_dist(rnd);

      acc += static_cast<double>(value) * (feature_idx + 1);

      auto* column_data = dataset.MutableColumnWithCast<
          ydf::dataset::VerticalDataset::DiscretizedNumericalColumn>(
          feature_idx);
      (*column_data->mutable_values())[example_idx] = value;
    }

    auto* label_data = dataset.MutableColumnWithCast<
        ydf::dataset::VerticalDataset::CategoricalColumn>(num_features);
    (*label_data->mutable_values())[example_idx] = 1 + (acc * 2 >= max_acc);
  }

  return dataset;
}

int main(int argc, char** argv) {
  // Enable the logging. Optional in most cases.
  InitLogging(argv[0], &argc, &argv, true);
  const auto work_dir = absl::GetFlag(FLAGS_work_dir);

  if (!work_dir.empty()) {
    // Create work directory.
    CHECK_OK(file::RecursivelyCreateDir(work_dir, file::Defaults()));
  }

  const int num_features = 28;

  // Specify the features in the dataset.
  const auto dataspec = BuildDataspec(/*num_features=*/num_features);
  YDF_LOG(INFO) << "Dataspec:\n"
                << ydf::dataset::PrintHumanReadable(dataspec, false);

  // Create a synthetic training dataset
  YDF_LOG(INFO) << "Generate dataset";
  const auto train_ds = CreateSyntheticDataset(
      dataspec, /*num_features=*/num_features, /*num_examples=*/10000);

  if (!work_dir.empty()) {
    // Export datasets to disk.
    CHECK_OK(ydf::dataset::SaveVerticalDataset(
        train_ds, absl::StrCat("csv:", file::JoinPath(work_dir, "train.csv"))));
  }

  // Train a model
  YDF_LOG(INFO) << "Train model";
  ydf::model::proto::TrainingConfig train_config;
  train_config.set_learner("GRADIENT_BOOSTED_TREES");
  train_config.set_task(ydf::model::proto::Task::CLASSIFICATION);
  train_config.set_label("label");
  auto* gbt_config = train_config.MutableExtension(
      ydf::model::gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_num_trees(10);
  std::unique_ptr<ydf::model::AbstractLearner> learner;
  CHECK_OK(GetLearner(train_config, &learner));
  learner->set_log_directory(file::JoinPath(work_dir, "train_logs"));

  // Effective training
  auto model = learner->TrainWithStatus(train_ds).value();

  if (!work_dir.empty()) {
    YDF_LOG(INFO) << "Save model";
    CHECK_OK(ydf::model::SaveModel(file::JoinPath(work_dir, "model"),
                                   model.get(), {""}));
  }

  // Compile the model for fast inference using the 8-bits specialized engine.
  namespace num_8bits = ydf::serving::decision_forest::num_8bits;
  num_8bits::GradientBoostedTreesBinaryClassificationModel engine;

  // Make sure the model is a GBT.
  auto* gbt_model = dynamic_cast<
      ydf::model::gradient_boosted_trees::GradientBoostedTreesModel*>(
      model.get());
  CHECK(gbt_model);

  // Effectively compile the model.
  CHECK_OK(num_8bits::GenericToSpecializedModel(*gbt_model, &engine));

  // Generate some more examples.
  const auto serving_ds = CreateSyntheticDataset(
      dataspec, /*num_features=*/num_features, /*num_examples=*/10000);

  if (!work_dir.empty()) {
    // Export datasets to disk.
    CHECK_OK(ydf::dataset::SaveVerticalDataset(
        serving_ds,
        absl::StrCat("csv:", file::JoinPath(work_dir, "serving.csv"))));
  }

  // Copy the example data from the VerticalDaset to the format expected by the
  // engine. In a real setting, you want to populate "examples" directly.
  std::vector<uint8_t> examples(engine.num_features * serving_ds.nrow());
  for (int example_idx = 0; example_idx < serving_ds.nrow(); example_idx++) {
    for (int local_feature_idx = 0; local_feature_idx < engine.num_features;
         local_feature_idx++) {
      const auto feature_index_values =
          serving_ds
              .ColumnWithCast<
                  ydf::dataset::VerticalDataset::DiscretizedNumericalColumn>(
                  engine.features[local_feature_idx])
              ->values();

      // The data is example-major, feature-minor.
      examples[example_idx * engine.num_features + local_feature_idx] =
          feature_index_values[example_idx];
    }
  }

  // Allocate predictions memory.
  std::vector<float> engine_predictions(serving_ds.nrow());

  // Makes the predictions.
  CHECK_OK(Predict(engine, examples, serving_ds.nrow(), &engine_predictions));

  // Print the first predictions.
  YDF_LOG(INFO) << "Predictions: " << engine_predictions[0] << " "
                << engine_predictions[1] << " " << engine_predictions[2];

  return 0;
}
