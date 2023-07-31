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

// Simple example of the C++ API.
//
// This program do the following:
//   - Scan the dataset to create a dataspec.
//   - Print the dataspec.
//   - Train a Random Forest model.
//   - Save the model.
//   - Print model details (e.g. meta-data, variable importance, structure).
//   - Evaluate the model on a test dataset.
//   - Convert the model into an engine (i.e. a model optimized for serving).
//   - Generate some predictions with the engine.
//
// Many functions in the C++ API have 1:1 a correspondence with the CLI API
// commands. Those cases are annotated.
//
// The C++ API does not use exceptions. Instead, most functions return an
// `absl::Status` or an `absl::StatusOr` (i.e. a `absl::Status` with some
// result). The macros QCHECK_OK and  absl::StatusOr.value() can be used to
// check such status and extract the result.
//
// This code relies heavily on Absl, notably `absl::StrCat`. This method is a
// simple string concatenation function, equivalent but more efficient than
// absl::StrCat("hello ", "world") <=> std::string("hello ") + string("world").
//
// Usage example:
//   bazel build -c opt \
//   //yggdrasil_decision_forests/examples:beginner_cc
//
//   bazel-bin/yggdrasil_decision_forests/examples/beginner_cc \
//   --alsologtostderr
//
#include "absl/flags/flag.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/metric/report.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"

ABSL_FLAG(std::string, dataset_dir,
          "yggdrasil_decision_forests/test_data/dataset",
          "Input directory containing the datasets: \"adult_train.csv\" and "
          "\"adult_test.csv\"");

ABSL_FLAG(std::string, output_dir, "/tmp/yggdrasil_decision_forest",
          "Output directory to save the model, evaluation and other results.");

// Commonly used alias.
namespace ydf = yggdrasil_decision_forests;

int main(int argc, char** argv) {
  // Enable the logging. Optional.
  InitLogging(argv[0], &argc, &argv, true);

  // Read flags.
  const std::string dataset_dir = absl::GetFlag(FLAGS_dataset_dir);
  const std::string output_dir = absl::GetFlag(FLAGS_output_dir);

  // Training and testing dataset paths.
  //
  // Note: The prefix (e.g."csv:") indicates the format of the dataset.
  const auto train_path =
      absl::StrCat("csv:", file::JoinPath(dataset_dir, "adult_train.csv"));
  const auto test_path =
      absl::StrCat("csv:", file::JoinPath(dataset_dir, "adult_test.csv"));

  // Create output directory.
  QCHECK_OK(file::RecursivelyCreateDir(output_dir, file::Defaults()));

  // Scan dataset to create a dataspec.
  //
  // The dataspec is the list of available columns and their meta-data.
  //
  // This is similar to the "infer_dataspec" CLI command.
  YDF_LOG(INFO) << "Create dataspec";
  const auto dataspec_path = file::JoinPath(output_dir, "dataspec.pbtxt");
  const auto dataspec = ydf::dataset::CreateDataSpec(train_path).value();
  // Save the dataspec.
  QCHECK_OK(file::SetTextProto(dataspec_path, dataspec, file::Defaults()));

  // Print dataspec.
  //
  // This is similar to the "show_dataspec" CLI command.
  YDF_LOG(INFO) << "Print dataspec";
  std::string dataspec_report = ydf::dataset::PrintHumanReadable(dataspec);
  YDF_LOG(INFO) << "Dataspec:\n" << dataspec_report;
  // Save dataspec print in a .txt file.
  QCHECK_OK(
      file::SetContent(absl::StrCat(dataspec_path, ".txt"), dataspec_report));

  // Train model.
  YDF_LOG(INFO) << "Train model";

  // Configure the learner.
  ydf::model::proto::TrainingConfig train_config;
  train_config.set_learner("RANDOM_FOREST");
  train_config.set_task(ydf::model::proto::Task::CLASSIFICATION);
  train_config.set_label("income");
  const auto learner = ydf::model::GetLearner(train_config).value();

  // Effectively train model.
  //
  // This is similar to the "train" CLI command.
  auto model = learner->TrainWithStatus(train_path, dataspec).value();

  // Save the model.
  YDF_LOG(INFO) << "Export the model";
  const auto model_path = file::JoinPath(output_dir, "model");
  QCHECK_OK(ydf::model::SaveModel(model_path, *model));

  // Show details about model.
  //
  // This is similar to the "show_model" CLI command.
  std::string model_description = model->DescriptionAndStatistics();
  YDF_LOG(INFO) << "Model:\n" << model_description;
  // Save details in a .txt file.
  QCHECK_OK(
      file::SetContent(absl::StrCat(model_path, ".txt"), model_description));

  // Evaluate model
  //
  // This is similar to the "evaluate" CLI command.
  ydf::dataset::VerticalDataset test_dataset;
  QCHECK_OK(ydf::dataset::LoadVerticalDataset(test_path, model->data_spec(),
                                              &test_dataset));

  // The effective evaluation.
  ydf::utils::RandomEngine rnd;
  const auto evaluation = model->Evaluate(test_dataset, {}, &rnd);

  // Save the raw evaluation.
  std::string evaluation_path = file::JoinPath(output_dir, "evaluation.pbtxt");
  QCHECK_OK(file::SetTextProto(evaluation_path, evaluation, file::Defaults()));

  // Save the evaluation in a text file.
  std::string evaluation_report = ydf::metric::TextReport(evaluation).value();
  QCHECK_OK(file::SetContent(absl::StrCat(evaluation_path, ".txt"),
                             evaluation_report));
  YDF_LOG(INFO) << "Evaluation:\n" << evaluation_report;

  // Compile the model into an engine for fast inference.
  const auto engine = model->BuildFastEngine().value();

  // At this point, the model is not needed anymore.
  model.reset();

  // Get handle of features about the engine.
  //
  // Note: Feature handles should be extracted one and then saved at not to
  // reacquire them for each inference.
  const auto& features = engine->features();
  const auto age_feature = features.GetNumericalFeatureId("age").value();
  const auto education_feature =
      features.GetCategoricalFeatureId("education").value();

  // Allocate a batch of 5 examples.
  //
  // Note: Batch of examples can be reused to maximize the program efficiency.
  std::unique_ptr<ydf::serving::AbstractExampleSet> examples =
      engine->AllocateExamples(5);

  // Fill the batch with missing values. We will then override the non-missing
  // values.
  //
  // Filling the batch with missing values is only necessary if you don't plan
  // on setting all the feature values manually.
  examples->FillMissing(features);

  // Set the value of "age" and "eduction" for the first example.
  examples->SetNumerical(/*example_idx=*/0, age_feature, 35.f, features);
  examples->SetCategorical(/*example_idx=*/0, education_feature, "HS-grad",
                           features);

  // Compute predictions on the first two examples.
  std::vector<float> batch_of_predictions;
  engine->Predict(*examples, 2, &batch_of_predictions);

  // Print predictions.
  YDF_LOG(INFO) << "Predictions:";
  for (const float prediction : batch_of_predictions) {
    YDF_LOG(INFO) << "\t" << prediction;
  }

  YDF_LOG(INFO) << "The results are available in " << output_dir;

  return 0;
}
