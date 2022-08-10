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

// Beginner example for the C++ interface.
//
// This program do the following:
//   - Scan the dataset columns to create a dataspec.
//   - Print a human readable report of the dataspec.
//   - Train a Random Forest model.
//   - Export the model to disk.
//   - Print and export a description of the model (meta-data and structure).
//   - Evaluate the model on a test dataset.
//   - Instantiate a serving engine with the model.
//   - Run a couple of predictions with the serving engine.
//
// Most of the sections are equivalent as calling one of the CLI command. This
// is indicated in the comments. For example, the comment "Same as
// :infer_dataspec" indicates that the following section is equivalent as
// running the "infer_dataspec" CLI command.
//
// When converting a CLI pipeline in C++, it is also interesting to look at the
// implementation of each CLI command. Generally, one CLI command contains one
// or a small number of C++ calls.
//
// Usage example:
//  ./compile_and_run.sh
//

#include "absl/flags/flag.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/metric/report.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"

ABSL_FLAG(std::string, dataset_dir,
          "yggdrasil_decision_forests/test_data/dataset",
          "Directory containing the \"adult_train.csv\" and \"adult_test.csv\" "
          "datasets.");

ABSL_FLAG(std::string, output_dir, "/tmp/yggdrasil_decision_forest",
          "Output directory for the model and evaluation");

namespace ygg = yggdrasil_decision_forests;

int main(int argc, char** argv) {
  // Enable the logging. Optional in most cases.
  InitLogging(argv[0], &argc, &argv, true);

  // Path to the training and testing dataset.
  const auto train_dataset_path = absl::StrCat(
      "csv:",
      file::JoinPath(absl::GetFlag(FLAGS_dataset_dir), "adult_train.csv"));

  const auto test_dataset_path = absl::StrCat(
      "csv:",
      file::JoinPath(absl::GetFlag(FLAGS_dataset_dir), "adult_test.csv"));

  // Create the output directory
  QCHECK_OK(file::RecursivelyCreateDir(absl::GetFlag(FLAGS_output_dir),
                                       file::Defaults()));

  // Scan the columns of the dataset to create a dataspec.
  // Same as :infer_dataspec
  LOG(INFO) << "Create dataspec";
  const auto dataspec_path =
      file::JoinPath(absl::GetFlag(FLAGS_output_dir), "dataspec.pbtxt");
  ygg::dataset::proto::DataSpecification dataspec;
  ygg::dataset::CreateDataSpec(train_dataset_path, false, /*guide=*/{},
                               &dataspec);
  QCHECK_OK(file::SetTextProto(dataspec_path, dataspec, file::Defaults()));

  // Display the dataspec in a human readable form.
  // Same as :show_dataspec
  LOG(INFO) << "Nice print of the dataspec";
  const auto dataspec_report =
      ygg::dataset::PrintHumanReadable(dataspec, false);
  QCHECK_OK(
      file::SetContent(absl::StrCat(dataspec_path, ".txt"), dataspec_report));
  LOG(INFO) << "Dataspec:\n" << dataspec_report;

  // Train the model.
  // Same as :train
  LOG(INFO) << "Train model";

  // Configure the learner.
  ygg::model::proto::TrainingConfig train_config;
  train_config.set_learner("RANDOM_FOREST");
  train_config.set_task(ygg::model::proto::Task::CLASSIFICATION);
  train_config.set_label("income");
  std::unique_ptr<ygg::model::AbstractLearner> learner;
  QCHECK_OK(GetLearner(train_config, &learner));

  // Set to export the training logs.
  learner->set_log_directory(absl::GetFlag(FLAGS_output_dir));

  // Effectively train the model.
  auto model = learner->TrainWithStatus(train_dataset_path, dataspec).value();

  // Save the model.
  LOG(INFO) << "Export the model";
  const auto model_path =
      file::JoinPath(absl::GetFlag(FLAGS_output_dir), "model");
  QCHECK_OK(ygg::model::SaveModel(model_path, model.get()));

  // Show information about the model.
  // Like :show_model, but without the list of compatible engines.
  std::string model_description;
  model->AppendDescriptionAndStatistics(/*full_definition=*/false,
                                        &model_description);
  QCHECK_OK(
      file::SetContent(absl::StrCat(model_path, ".txt"), model_description));
  LOG(INFO) << "Model:\n" << model_description;

  // Evaluate the model
  // Same as :evaluate
  ygg::dataset::VerticalDataset test_dataset;
  QCHECK_OK(ygg::dataset::LoadVerticalDataset(
      test_dataset_path, model->data_spec(), &test_dataset));

  ygg::utils::RandomEngine rnd;
  ygg::metric::proto::EvaluationOptions evaluation_options;
  evaluation_options.set_task(model->task());

  // The effective evaluation.
  const ygg::metric::proto::EvaluationResults evaluation =
      model->Evaluate(test_dataset, evaluation_options, &rnd);

  // Export the raw evaluation.
  const auto evaluation_path =
      file::JoinPath(absl::GetFlag(FLAGS_output_dir), "evaluation.pbtxt");
  QCHECK_OK(file::SetTextProto(evaluation_path, evaluation, file::Defaults()));

  // Export the evaluation to a nice text.
  std::string evaluation_report;
  QCHECK_OK(
      ygg::metric::AppendTextReportWithStatus(evaluation, &evaluation_report));
  QCHECK_OK(file::SetContent(absl::StrCat(evaluation_path, ".txt"),
                             evaluation_report));
  LOG(INFO) << "Evaluation:\n" << evaluation_report;

  // Compile the model for fast inference.
  const std::unique_ptr<ygg::serving::FastEngine> serving_engine =
      model->BuildFastEngine().value();
  const auto& features = serving_engine->features();

  // Handle to two features.
  const auto age_feature = features.GetNumericalFeatureId("age").value();
  const auto education_feature =
      features.GetCategoricalFeatureId("education").value();

  // Allocate a batch of 5 examples.
  std::unique_ptr<ygg::serving::AbstractExampleSet> examples =
      serving_engine->AllocateExamples(5);

  // Set all the values as missing. This is only necessary if you don't set all
  // the feature values manually e.g. SetNumerical.
  examples->FillMissing(features);

  // Set the value of "age" and "eduction" for the first example.
  examples->SetNumerical(/*example_idx=*/0, age_feature, 35.f, features);
  examples->SetCategorical(/*example_idx=*/0, education_feature, "HS-grad",
                           features);

  // Run the predictions on the first two examples.
  std::vector<float> batch_of_predictions;
  serving_engine->Predict(*examples, 2, &batch_of_predictions);

  LOG(INFO) << "Predictions:";
  for (const float prediction : batch_of_predictions) {
    LOG(INFO) << "\t" << prediction;
  }

  return 0;
}
