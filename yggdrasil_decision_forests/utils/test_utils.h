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

// Utility functions for end-to-end testing of the model training, evaluation,
// serialization and fast-engine-inference.
//
// One such test should be applied for each learning algorithm and learning
// algorithm variant (e.g. a new hyper-parameter).
//
// Usage example:
//
//   class MyLearningAlgorithm : public utils::TrainAndTestTester {
//     void SetUp() override {
//       train_config_.set_learner(MyLearningAlgorithm::kRegisteredName);
//       train_config_.set_task(model::proto::Task::CLASSIFICATION);
//       train_config_.set_label("LABEL");
//       dataset_filename_ = "dna.csv";
//     }
//   };
//
//   TEST_F(MyLearningAlgorithm, MyConfiguration) {
//     TrainAndEvaluateModel();
//     EXPECT_NEAR(metric::Accuracy(evaluation_), 0.9466, 0.01);
//     EXPECT_NEAR(metric::LogLoss(evaluation_), 0.2973, 0.04);
//   }
//
#ifndef YGGDRASIL_DECISION_FORESTS_TOOL_TEST_UTILS_H_
#define YGGDRASIL_DECISION_FORESTS_TOOL_TEST_UTILS_H_

#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/synthetic_dataset.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/prediction.pb.h"
#include "yggdrasil_decision_forests/serving/decision_forest/decision_forest.h"
#include "yggdrasil_decision_forests/serving/example_set.h"

namespace yggdrasil_decision_forests {
namespace utils {

// Train and test a model on a dataset stored in the "test_data" folder.
class TrainAndTestTester : public ::testing::Test {
 public:
  // Run the training and evaluation of the model. Should be called after
  // "train_config_" is set. After this function is called, "evaluation_"
  // contains the result of the evaluation.
  void TrainAndEvaluateModel(
      absl::optional<absl::string_view> numerical_weight_attribute = {},
      bool emulate_weight_with_duplication = false,
      std::function<void(void)> callback_training_about_to_start = nullptr);

  // Configure the test to run on the synthetic dataset generator.
  void ConfigureForSyntheticDataset();

  // Directory containing the dataset used in the test.
  std::string dataset_root_directory_ =
      "yggdrasil_decision_forests/test_data/dataset";

  std::string EffectiveDatasetRootDirectory();

  // Filename of the dataset. The full dataset path will be
  // Join(dataset_root_directory_, dataset_filename_).
  std::string dataset_filename_;

  // If "dataset_test_filename_" is not specified, the dataset
  // "dataset_filename_" is split into a training and a testing dataset.
  // If "dataset_test_filename_" is specified, all of "dataset_filename_" is
  // used for training, and "dataset_test_filename_" is used for testing.
  std::string dataset_test_filename_;

  dataset::proto::SyntheticDatasetOptions synthetic_dataset_;

  // Filename of the datec guide. The full guide path will be
  // Join(dataset_root_directory_, guide_filename_). If empty, no guide will
  // be used.
  std::string guide_filename_;

  // Training configuration of the model.
  model::proto::TrainingConfig train_config_;

  absl::optional<model::proto::GenericHyperParameters> generic_parameters_;

  model::proto::DeploymentConfig deployment_config_;

  metric::proto::EvaluationResults evaluation_;

  std::unique_ptr<model::AbstractLearner> learner_;

  // Configure the evaluation.
  metric::proto::EvaluationOptions eval_options_;

  // Percentage of the dataset used for the train/test.
  float dataset_sampling_ = 1.f;

  // The trained model.
  std::unique_ptr<model::AbstractModel> model_;

  // Duration of training of the model.
  absl::Duration training_duration_;

  // Directory name containing the model, evaluation and training logs.
  std::string test_dir_;

  // Train and testing datasets.
  dataset::VerticalDataset train_dataset_;
  dataset::VerticalDataset valid_dataset_;
  dataset::VerticalDataset test_dataset_;
  dataset::proto::DataSpecificationGuide guide_;

  // Ratio of the original dataset going into the training fold. The remaining
  // examples are uniformly split between the test and valid dataset (is
  // "pass_validation_dataset_=true").
  //
  // If the value if 0.5f (default), the examples are split deterministically
  // according to their index : even examples are used for training, odd
  // examples are used for test/validation.
  float split_train_ratio_ = 0.5f;

  // Format and extension used to store the temporary dataset generated during
  // the test. The reader and writer of this format need to be registered.
  std::string preferred_format_type = "tfrecord+tfe";
  std::string preferred_format_extension = ".tfr";

  // If true, the dataset is passed to the learner as a path. If false, the
  // dataset is passed to the learner as a VerticalDataset.
  bool pass_training_dataset_as_path_ = false;

  // Number of shards to use if "pass_training_dataset_as_path_" is set to true.
  int num_shards_ = 3;

  // If set, interrupts the training after "interrupt_training_after".
  absl::optional<absl::Duration> interrupt_training_after;

  // If true, the model is checked and the implementation is checked for
  // potential issues e.g. serializing+deserializing, creation of serving
  // engines.
  bool check_model = true;

  // If true, the training method is called with a validation dataset (either a
  // path or a VerticalDataset; depending on "pass_training_dataset_as_path_");
  bool pass_validation_dataset_ = false;

  // If true, show the entire model structure (e.g. show the decision trees) in
  // the logs.
  bool show_full_model_structure_ = false;

 private:
  std::pair<std::string, std::string> GetTrainAndTestDatasetPaths();

  dataset::proto::DataSpecification BuildDataspec(
      const absl::string_view dataset_path);

  void FixConfiguration(
      absl::optional<absl::string_view> numerical_weight_attribute,
      const dataset::proto::DataSpecification& data_spec,
      int32_t* numerical_weight_attribute_idx,
      float* max_numerical_weight_value);

  void BuildTrainValidTestDatasets(
      const dataset::proto::DataSpecification& data_spec,
      const absl::string_view train_path, const absl::string_view test_path,
      int32_t numerical_weight_attribute_idx, float max_numerical_weight_value);
};

// Tests the prediction of the (slow) generic engine and the fast generic
// engine. If the model does not implement a fast generic engine, the model
// succeed.
void TestGenericEngine(const model::AbstractModel& model,
                       const dataset::VerticalDataset& dataset);

// Checks the predictions of an engine vs the slow generic engine.
void ExpectEqualPredictions(const dataset::VerticalDataset& dataset,
                            const model::AbstractModel& model,
                            const serving::FastEngine& engine);

// Checks the predictions of the slow generic engine vs "predictions".
void ExpectEqualPredictions(const dataset::VerticalDataset& dataset,
                            dataset::VerticalDataset::row_t begin_example_idx,
                            dataset::VerticalDataset::row_t end_example_idx,
                            const model::AbstractModel& model,
                            const std::vector<float>& predictions);

// Checks the predictions of a templated engine vs the slow generic engine.
template <typename Engine,
          void (*PredictCall)(const Engine&, const typename Engine::ExampleSet&,
                              int, std::vector<float>*)>
void ExpectEqualPredictionsTemplate(const dataset::VerticalDataset& dataset,
                                    const model::AbstractModel& model,
                                    const Engine& engine);

// Checks the predictions of a templated engine with the old API vs the slow
// generic engine.
template <typename Engine,
          void (*PredictCall)(const Engine&,
                              const std::vector<typename Engine::ValueType>&,
                              int, std::vector<float>*)>
void ExpectEqualPredictionsOldTemplate(
    const dataset::VerticalDataset& dataset, const model::AbstractModel& model,
    const Engine& engine,
    const serving::ExampleFormat example_format =
        serving::ExampleFormat::FORMAT_EXAMPLE_MAJOR);

// Checks that two predictions are equivalent.
void ExpectEqualPredictions(const model::proto::Task task,
                            const model::proto::Prediction& a,
                            const model::proto::Prediction& b);

template <typename Engine,
          void (*PredictCall)(const Engine&, const typename Engine::ExampleSet&,
                              int, std::vector<float>*)>
void ExpectEqualPredictionsTemplate(const dataset::VerticalDataset& dataset,
                                    const model::AbstractModel& model,
                                    const Engine& engine) {
  const int batch_size = 20;
  const int num_batches = (dataset.nrow() + batch_size - 1) / batch_size;
  const auto dataset_as_example_set =
      yggdrasil_decision_forests::serving::VerticalDatasetToExampleSet(dataset,
                                                                       engine)
          .value();
  typename Engine::ExampleSet example_set_batch(batch_size, engine);

  std::vector<float> predictions;

  for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
    // Extract a set of examples.
    const auto begin_idx = batch_idx * batch_size;
    const auto end_idx =
        std::min(begin_idx + batch_size, static_cast<int>(dataset.nrow()));

    EXPECT_OK(dataset_as_example_set.Copy(begin_idx, end_idx, engine,
                                          &example_set_batch));

    // Generate the predictions of the engine.
    PredictCall(engine, example_set_batch, end_idx - begin_idx, &predictions);

    // Check the predictions against the ground truth inference code.
    ExpectEqualPredictions(dataset, begin_idx, end_idx, model, predictions);
  }
}

template <typename Engine,
          void (*PredictCall)(const Engine&,
                              const std::vector<typename Engine::ValueType>&,
                              int, std::vector<float>*)>
void ExpectEqualPredictionsOldTemplate(
    const dataset::VerticalDataset& dataset, const model::AbstractModel& model,
    const Engine& engine, const serving::ExampleFormat example_format) {
  const int batch_size = 20;
  const int num_batches = (dataset.nrow() + batch_size - 1) / batch_size;
  std::vector<typename Engine::ValueType> batch_of_examples;

  std::vector<float> predictions;

  for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
    // Extract a set of examples.
    const auto begin_idx = batch_idx * batch_size;
    const auto end_idx =
        std::min(begin_idx + batch_size, static_cast<int>(dataset.nrow()));

    CHECK_OK(serving::decision_forest::LoadFlatBatchFromDataset(
        dataset, begin_idx, end_idx,
        FeatureNames(engine.features().fixed_length_features()),
        engine.features().fixed_length_na_replacement_values(),
        &batch_of_examples, example_format));

    // Generate the predictions of the engine.
    PredictCall(engine, batch_of_examples, end_idx - begin_idx, &predictions);

    // Check the predictions against the ground truth inference code.
    ExpectEqualPredictions(dataset, begin_idx, end_idx, model, predictions);
  }
}

// Trains and tests a model for each possible predefined hyper-parameters
// values.
void TestPredefinedHyperParameters(
    absl::string_view train_ds_path, absl::string_view test_ds_path,
    const model::proto::TrainingConfig& train_config,
    const int expected_num_preconfigured_parameters,
    absl::optional<float> min_accuracy);

// Runs "TestPredefinedHyperParameters" on the adult dataset.
void TestPredefinedHyperParametersAdultDataset(
    model::proto::TrainingConfig train_config,
    const int expected_num_preconfigured_parameters,
    absl::optional<float> min_accuracy);

// Randomly shards a dataset. Returns the sharded path in the temp directory.
std::string ShardDataset(const dataset::VerticalDataset& dataset,
                         int num_shards, float sampling,
                         absl::string_view format);

// Exports the predictions of a binary treatment uplift model to a csv file with
// the columns: "uplift", "response", "weight", "group".
absl::Status ExportUpliftPredictionsToTFUpliftCsvFormat(
    const model::AbstractModel& model, const dataset::VerticalDataset& dataset,
    absl::string_view output_csv_path);

}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_TOOL_TEST_UTILS_H_
