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

#include "yggdrasil_decision_forests/utils/test_utils.h"

#include <cxxabi.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/synthetic_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/dataset/weight.pb.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/metric/report.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/model_engine_wrapper.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/model/prediction.pb.h"
#include "yggdrasil_decision_forests/serving/example_set.h"
#include "yggdrasil_decision_forests/serving/fast_engine.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/csv.h"
#include "yggdrasil_decision_forests/utils/distribution.pb.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/uid.h"

namespace yggdrasil_decision_forests {
namespace utils {

namespace {

// Shuffles a dataset randomly. Does not rely on a static seed.
void ShuffleDataset(dataset::VerticalDataset* dataset) {
  absl::BitGen bitgen;
  std::vector<dataset::VerticalDataset::row_t> example_idxs(dataset->nrow());
  std::iota(example_idxs.begin(), example_idxs.end(), 0);
  std::shuffle(example_idxs.begin(), example_idxs.end(), bitgen);
  *dataset = dataset->Extract(example_idxs).value();
}

// Generates a random seed. Does not rely on a static seed.
int64_t RandomSeed() {
  absl::BitGen bitgen;
  return bitgen();
}

// Generates a deterministic sequence of boolean value approximating poorly a
// binomial distribution sampling.
class DeterministicBinomial {
 public:
  DeterministicBinomial(const float rate) : rate_(rate) {}

  bool Sample() {
    if (num_total_ == 0) {
      // Always return false first, unless the rate is 1.
      num_total_++;
      if (rate_ == 1) {
        num_pos_++;
        return true;
      }
      return false;
    }

    if (num_pos_ > rate_ * num_total_) {
      num_total_++;
      return false;
    } else {
      num_pos_++;
      num_total_++;
      return true;
    }
  }

 private:
  float rate_;
  int num_pos_ = 0;
  int num_total_ = 0;
};

}  // namespace

void TrainAndTestTester::ConfigureForSyntheticDataset() {
  auto cat_int = guide_.add_column_guides();
  cat_int->set_column_name_pattern("^cat_int_.*$");
  cat_int->set_type(dataset::proto::CATEGORICAL);

  auto cat_set_int = guide_.add_column_guides();
  cat_set_int->set_column_name_pattern("^cat_set_int_.*$");
  cat_set_int->set_type(dataset::proto::CATEGORICAL_SET);

  train_config_.set_label("LABEL");

  switch (train_config_.task()) {
    case model::proto::CLASSIFICATION:
      synthetic_dataset_.mutable_classification();
      break;
    case model::proto::REGRESSION:
      synthetic_dataset_.mutable_regression();
      break;
    default:
      YDF_LOG(FATAL) << "Non supported task";
  }
}

std::string TrainAndTestTester::EffectiveDatasetRootDirectory() {
  return file::JoinPath(test::DataRootDirectory(), dataset_root_directory_);
}

void TrainAndTestTester::TrainAndEvaluateModel(
    absl::optional<absl::string_view> numerical_weight_attribute,
    const bool emulate_weight_with_duplication,
    std::function<void(void)> callback_training_about_to_start) {
  model::ModelIOOptions model_io;
  model_io.file_prefix = "";

  // Path to dataset(s).
  std::string dataset_path;
  std::string test_dataset_path;
  std::tie(dataset_path, test_dataset_path) = GetTrainAndTestDatasetPaths();

  // Build dataspec.
  const auto data_spec = BuildDataspec(dataset_path);

  // Check and update the configuration.
  int32_t numerical_weight_attribute_idx;
  float max_numerical_weight_value;
  FixConfiguration(numerical_weight_attribute, data_spec,
                   &numerical_weight_attribute_idx,
                   &max_numerical_weight_value);

  // Instantiate the train and test datasets.
  BuildTrainValidTestDatasets(data_spec, dataset_path, test_dataset_path,
                              numerical_weight_attribute_idx,
                              max_numerical_weight_value);

  // Configure the learner.
  CHECK_OK(model::GetLearner(train_config_, &learner_, deployment_config_));

  if (change_random_seed_ && !learner_->training_config().has_random_seed()) {
    learner_->mutable_training_config()->set_random_seed(RandomSeed());
  }

  if (generic_parameters_.has_value()) {
    CHECK_OK(learner_->SetHyperParameters(generic_parameters_.value()));
  }
  const auto log_dir = file::JoinPath(test::TmpDirectory(), test_dir_, "logs");

  YDF_LOG(INFO) << "Set log directory: " << log_dir;
  learner_->set_log_directory(log_dir);

  if (callback_training_about_to_start) {
    callback_training_about_to_start();
  }

  std::atomic<bool> stop_training{false};
  std::unique_ptr<utils::concurrency::Thread> interrupter_thread;
  if (interrupt_training_after.has_value()) {
    learner_->set_stop_training_trigger(&stop_training);
    interrupter_thread = absl::make_unique<utils::concurrency::Thread>([&]() {
      absl::SleepFor(interrupt_training_after.value());
      YDF_LOG(INFO)
          << "Interrupt the training. Waiting for the learner to return "
             "a model.";
      stop_training = true;
    });
  }

  const auto begin_training = absl::Now();

  absl::StatusOr<std::unique_ptr<model::AbstractModel>> model_or;
  // Train the model.
  if (pass_training_dataset_as_path_) {
    // Export the training dataset into a set of sharded files.
    const auto train_dataset_path = ShardDataset(
        train_dataset_, num_shards_, 1.f, preferred_format_type, "training");
    if (pass_validation_dataset_) {
      // Export the validation dataset into a set of sharded files.
      const auto valid_dataset_path =
          ShardDataset(valid_dataset_, num_shards_, 1.f, preferred_format_type,
                       "validation");
      model_or = learner_->TrainWithStatus(train_dataset_path, data_spec,
                                           valid_dataset_path);
    } else {
      model_or = learner_->TrainWithStatus(train_dataset_path, data_spec);
    }
  } else if (pass_validation_dataset_) {
    model_or = learner_->TrainWithStatus(train_dataset_, valid_dataset_);
  } else {
    model_or = learner_->TrainWithStatus(train_dataset_);
  }

  CHECK_OK(model_or.status());
  model_ = std::move(model_or).value();

  const auto end_training = absl::Now();
  training_duration_ = end_training - begin_training;

  if (interrupter_thread) {
    interrupter_thread->Join();
    interrupter_thread.reset();
  }

  YDF_LOG(INFO) << "Training duration: " << training_duration_;

  // Evaluate the model.
  utils::RandomEngine rnd(1234);  // Not used
  evaluation_ = model_->Evaluate(test_dataset_, eval_options_, &rnd);

  // Print the model evaluation.
  const auto evaluation_description = metric::TextReport(evaluation_).value();
  YDF_LOG(INFO) << "Evaluation:\n" << evaluation_description;

  // Export the evaluation to a html file.
  std::string html_evaluation_report;
  CHECK_OK(metric::AppendHtmlReport(evaluation_, &html_evaluation_report));
  const auto html_report_path =
      file::JoinPath(test::TmpDirectory(), test_dir_, "evaluation.html");
  YDF_LOG(INFO) << "Export html report to: " << html_report_path;
  CHECK_OK(file::SetContent(html_report_path, html_evaluation_report));

  if (!check_model) {
    return;
  }

  // Export the model to drive.
  const std::string model_path =
      file::JoinPath(test::TmpDirectory(), test_dir_, "model");
  EXPECT_OK(SaveModel(model_path, model_.get(), model_io));

  YDF_LOG(INFO) << "Description:\n"
                << model_->DescriptionAndStatistics(show_full_model_structure_);

  const auto check_evaluation_is_equal =
      [this](const metric::proto::EvaluationResults& e1,
             const metric::proto::EvaluationResults& e2) {
        switch (train_config_.task()) {
          case model::proto::Task::CLASSIFICATION:
            // Note: On small dataset, the accuracy can change if the prediction
            // value for one example is near the decision boundary.
            //
            // Note: In the next test (see "TestGenericEngine"), we ensure that
            // predictions are equal with a margin of 0.0002.
            EXPECT_NEAR(metric::Accuracy(e1), metric::Accuracy(e2), 0.002);
            EXPECT_NEAR(metric::LogLoss(e1), metric::LogLoss(e2), 0.05);
            break;
          case model::proto::Task::REGRESSION:
            EXPECT_NEAR(metric::RMSE(e1), metric::RMSE(e2), 0.001);
            break;
          case model::proto::Task::RANKING:
            EXPECT_NEAR(metric::NDCG(e1), metric::NDCG(e2), 0.001);
            break;
          case model::proto::Task::CATEGORICAL_UPLIFT:
          case model::proto::Task::NUMERICAL_UPLIFT:
            EXPECT_NEAR(metric::AUUC(e1), metric::AUUC(e2), 0.001);
            EXPECT_NEAR(metric::Qini(e1), metric::Qini(e2), 0.001);
            break;
          default:
            YDF_LOG(FATAL) << "Not implemented";
        }
      };

  // Evaluate the exported model.
  YDF_LOG(INFO) << "Evaluate the exported model";
  std::unique_ptr<model::AbstractModel> loaded_model;
  EXPECT_OK(LoadModel(model_path, &loaded_model, model_io));
  rnd.seed(1234);
  const auto evaluation_loaded_model =
      loaded_model->Evaluate(test_dataset_, eval_options_, &rnd);

  // Test that the exported model evaluation is the same as the original model
  // evaluation.
  check_evaluation_is_equal(evaluation_, evaluation_loaded_model);

  // Ensure that the predictions of the semi-fast engine are similar as the
  // predictions of the generic engine.
  YDF_LOG(INFO) << "Test generic engine";
  TestGenericEngine(*model_, test_dataset_);

  // Evaluation with disabled semi-fast engine.
  YDF_LOG(INFO) << "Evaluate model without fast engine";
  loaded_model->SetAllowFastEngine(false);
  rnd.seed(1234);
  const auto evaluation_loaded_model_no_fast_engine =
      loaded_model->Evaluate(test_dataset_, eval_options_, &rnd);
  check_evaluation_is_equal(evaluation_loaded_model,
                            evaluation_loaded_model_no_fast_engine);

  // Test that the pure version of the model is equal to the non-pure version.
  EXPECT_OK(loaded_model->MakePureServing());
  rnd.seed(1234);
  const auto evaluation_loaded_and_pure_model =
      loaded_model->Evaluate(test_dataset_, eval_options_, &rnd);
  check_evaluation_is_equal(evaluation_, evaluation_loaded_and_pure_model);
}

std::pair<std::string, std::string>
TrainAndTestTester::GetTrainAndTestDatasetPaths() {
  std::string dataset_path;
  std::string test_dataset_path;
  if (dataset_filename_.empty()) {
    dataset_path = absl::StrCat(
        preferred_format_type, ":",
        file::JoinPath(
            test::TmpDirectory(),
            absl::StrCat("synthetic_dataset", preferred_format_extension)));
    CHECK_OK(
        dataset::GenerateSyntheticDataset(synthetic_dataset_, dataset_path));
  } else {
    dataset_path = absl::StrCat(
        "csv:",
        file::JoinPath(EffectiveDatasetRootDirectory(), dataset_filename_));
    if (!dataset_test_filename_.empty()) {
      test_dataset_path =
          absl::StrCat("csv:", file::JoinPath(EffectiveDatasetRootDirectory(),
                                              dataset_test_filename_));
    }
  }
  return {dataset_path, test_dataset_path};
}

dataset::proto::DataSpecification TrainAndTestTester::BuildDataspec(
    const absl::string_view dataset_path) {
  // Infer the dataspec.
  if (!guide_filename_.empty()) {
    dataset::proto::DataSpecificationGuide loaded_guide;
    const std::string guide_path =
        file::JoinPath(EffectiveDatasetRootDirectory(), guide_filename_);
    QCHECK_OK(file::GetTextProto(guide_path, &loaded_guide, file::Defaults()));
    guide_.MergeFrom(loaded_guide);
  }
  dataset::proto::DataSpecification data_spec;
  dataset::CreateDataSpec(dataset_path, false, guide_, &data_spec);
  YDF_LOG(INFO) << "Dataspec:\n"
                << dataset::PrintHumanReadable(data_spec, false);
  return data_spec;
}

void TrainAndTestTester::FixConfiguration(
    absl::optional<absl::string_view> numerical_weight_attribute,
    const dataset::proto::DataSpecification& data_spec,
    int32_t* numerical_weight_attribute_idx,
    float* max_numerical_weight_value) {
  eval_options_.set_bootstrapping_samples(100);
  eval_options_.set_task(train_config_.task());

  *numerical_weight_attribute_idx = -1;
  *max_numerical_weight_value = -1;
  if (numerical_weight_attribute.has_value()) {
    eval_options_.mutable_weights()->mutable_numerical();
    eval_options_.mutable_weights()->set_attribute(
        std::string(numerical_weight_attribute.value()));
    *train_config_.mutable_weight_definition() = eval_options_.weights();
    CHECK_OK(dataset::GetSingleColumnIdxFromName(
        numerical_weight_attribute.value(), data_spec,
        numerical_weight_attribute_idx));
    *max_numerical_weight_value =
        data_spec.columns(*numerical_weight_attribute_idx)
            .numerical()
            .max_value();
    CHECK_GT(*max_numerical_weight_value, 0);
    CHECK_GT(data_spec.columns(*numerical_weight_attribute_idx)
                 .numerical()
                 .min_value(),
             0);
  }
}

void TrainAndTestTester::BuildTrainValidTestDatasets(
    const dataset::proto::DataSpecification& data_spec,
    const absl::string_view train_path, const absl::string_view test_path,
    int32_t numerical_weight_attribute_idx, float max_numerical_weight_value) {
  if (!test_path.empty()) {
    CHECK_OK(LoadVerticalDataset(train_path, data_spec, &train_dataset_));
    CHECK_OK(LoadVerticalDataset(test_path, data_spec, &test_dataset_));

    if (pass_validation_dataset_) {
      std::vector<dataset::VerticalDataset::row_t> train_example_idxs,
          valid_example_idxs;
      for (dataset::VerticalDataset::row_t example_idx = 0;
           example_idx < train_dataset_.nrow(); example_idx++) {
        // Split deterministically, 1/2 of the examples in training, 1/2 of the
        // examples for validation.
        if ((example_idx % 2) == 0) {
          train_example_idxs.push_back(example_idx);
        } else {
          valid_example_idxs.push_back(example_idx);
        }
      }
      valid_dataset_ = train_dataset_.Extract(valid_example_idxs).value();
      train_dataset_ = train_dataset_.Extract(train_example_idxs).value();
    }

    if (inject_random_noise_) {
      ShuffleDataset(&train_dataset_);
      ShuffleDataset(&test_dataset_);
      if (pass_validation_dataset_) {
        ShuffleDataset(&valid_dataset_);
      }
    }

    return;
  }

  dataset::VerticalDataset dataset;
  CHECK_OK(LoadVerticalDataset(train_path, data_spec, &dataset));

  // Split the dataset in two folds: training and testing.
  std::vector<dataset::VerticalDataset::row_t> train_example_idxs;
  std::vector<dataset::VerticalDataset::row_t> test_example_idxs;
  std::vector<dataset::VerticalDataset::row_t> valid_example_idxs;

  DeterministicBinomial sampling(dataset_sampling_);
  DeterministicBinomial train_test_split(split_train_ratio_);
  DeterministicBinomial test_valid_split(0.5f);

  // TODO: Make deterministic.
  utils::RandomEngine rnd(1234);
  std::uniform_real_distribution<double> dist_01;

  for (dataset::VerticalDataset::row_t example_idx = 0;
       example_idx < dataset.nrow(); example_idx++) {
    // Down-sampling of examples.
    if (!sampling.Sample()) {
      continue;
    }

    // Down-sampling of examples according of a numerical attribute.
    if (numerical_weight_attribute_idx != -1) {
      const float weight = dataset
                               .ColumnWithCastWithStatus<
                                   dataset::VerticalDataset::NumericalColumn>(
                                   numerical_weight_attribute_idx)
                               .value()
                               ->values()[example_idx];
      const float proba_reject = weight / max_numerical_weight_value;
      if (dist_01(rnd) < proba_reject) {
        continue;
      }
    }

    const bool is_training_example = train_test_split.Sample();

    if (is_training_example) {
      train_example_idxs.push_back(example_idx);
    } else {
      if (pass_validation_dataset_ && test_valid_split.Sample()) {
        valid_example_idxs.push_back(example_idx);
      } else {
        test_example_idxs.push_back(example_idx);
      }
    }
  }

  if (split_train_ratio_ == 1.0) {
    YDF_LOG(INFO) << "Using the same dataset for training and evaluation";
    test_example_idxs = train_example_idxs;
  }

  train_dataset_ = dataset.Extract(train_example_idxs).value();
  test_dataset_ = dataset.Extract(test_example_idxs).value();
  valid_dataset_ = dataset.Extract(valid_example_idxs).value();

  if (inject_random_noise_) {
    ShuffleDataset(&train_dataset_);
    ShuffleDataset(&test_dataset_);
    ShuffleDataset(&valid_dataset_);
  }

  YDF_LOG(INFO) << "Number of examples: train:" << train_dataset_.nrow()
                << " valid:" << valid_dataset_.nrow()
                << " test:" << test_dataset_.nrow();
}

void TestGenericEngine(const model::AbstractModel& model,
                       const dataset::VerticalDataset& dataset) {
  auto engine_or = model.BuildFastEngine();
  if (!engine_or.ok()) {
    YDF_LOG(INFO) << "Model " << model.name()
                  << " does not implement any fast generic engine: "
                  << engine_or.status().message();
    return;
  }
  YDF_LOG(INFO) << "Testing fast generic engine.";
  auto engine = std::move(engine_or.value());
  ExpectEqualPredictions(dataset, model, *engine);
  YDF_LOG(INFO) << "Fast generic and generic engine predictions match";

  YDF_LOG(INFO) << "Check engine wrapper";
  const model::EngineWrapperModel wrapper_engine(&model, std::move(engine));
  for (int example_idx = 0; example_idx < dataset.nrow(); example_idx++) {
    model::proto::Prediction generic, vertical_dataset, proto_example;

    model.Predict(dataset, example_idx, &generic);
    wrapper_engine.Predict(dataset, example_idx, &vertical_dataset);

    dataset::proto::Example example;
    dataset.ExtractExample(example_idx, &example);
    wrapper_engine.Predict(example, &proto_example);

    ExpectEqualPredictions(model.task(), generic, vertical_dataset);
    ExpectEqualPredictions(model.task(), generic, proto_example);
  }
}

void ExpectEqualPredictions(const dataset::VerticalDataset& dataset,
                            const model::AbstractModel& model,
                            const serving::FastEngine& engine) {
  const int batch_size = 20;
  const int num_batches = (dataset.nrow() + batch_size - 1) / batch_size;
  const auto examples = engine.AllocateExamples(batch_size);

  std::vector<float> predictions;
  for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
    // Extract a set of examples.
    const auto begin_idx = batch_idx * batch_size;
    const auto end_idx =
        std::min(begin_idx + batch_size, static_cast<int>(dataset.nrow()));

    CHECK_OK(serving::CopyVerticalDatasetToAbstractExampleSet(
        dataset, begin_idx, end_idx, engine.features(), examples.get()));

    // Generate the predictions of the engine.
    engine.Predict(*examples, end_idx - begin_idx, &predictions);
    CHECK_EQ(predictions.size(),
             engine.NumPredictionDimension() * (end_idx - begin_idx))
        << "with NumPredictionDimension=" << engine.NumPredictionDimension()
        << " end_idx=" << end_idx << " and begin_idx=" << begin_idx;

    // Check the predictions against the ground truth inference code.
    ExpectEqualPredictions(dataset, begin_idx, end_idx, model, predictions);
  }
}

void ExpectEqualPredictions(const model::proto::Task task,
                            const model::proto::Prediction& a,
                            const model::proto::Prediction& b) {
  constexpr float epsilon = 0.0002f;
  EXPECT_EQ(a.type_case(), b.type_case());
  switch (task) {
    case model::proto::Task::CLASSIFICATION: {
      EXPECT_EQ(a.classification().distribution().counts().size(),
                b.classification().distribution().counts().size());
      for (int class_idx = 0;
           class_idx < a.classification().distribution().counts().size();
           class_idx++) {
        double p1 = a.classification().distribution().sum() > 0
                        ? (a.classification().distribution().counts(class_idx) /
                           a.classification().distribution().sum())
                        : 0;
        double p2 = b.classification().distribution().sum() > 0
                        ? (b.classification().distribution().counts(class_idx) /
                           b.classification().distribution().sum())
                        : 0;
        EXPECT_NEAR(p1, p2, epsilon);
      }
    } break;

    case model::proto::Task::REGRESSION:
      EXPECT_NEAR(a.regression().value(), b.regression().value(), epsilon);
      break;

    case model::proto::Task::RANKING:
      EXPECT_NEAR(a.ranking().relevance(), b.ranking().relevance(), epsilon);
      break;

    case model::proto::Task::CATEGORICAL_UPLIFT:
    case model::proto::Task::NUMERICAL_UPLIFT: {
      EXPECT_EQ(a.uplift().treatment_effect().size(),
                b.uplift().treatment_effect().size());
      for (int effect_idx = 0;
           effect_idx < a.uplift().treatment_effect().size(); effect_idx++) {
        EXPECT_NEAR(a.uplift().treatment_effect(effect_idx),
                    b.uplift().treatment_effect(effect_idx), epsilon);
      }
    } break;

    default:
      YDF_LOG(FATAL) << "Not supported task";
  }
}

void ExpectEqualPredictions(
    const dataset::VerticalDataset& dataset,
    const dataset::VerticalDataset::row_t begin_example_idx,
    const dataset::VerticalDataset::row_t end_example_idx,
    const model::AbstractModel& model, const std::vector<float>& predictions) {
  // Maximum difference between two supposedly equal values.
  constexpr float epsilon = 0.0002f;

  // Container of predictions from the ground truth engine.
  model::proto::Prediction generic_prediction;

  const auto num_examples = end_example_idx - begin_example_idx;

  // Current prediction being checked.
  int prediction_idx = 0;

  const auto get_probability = [](const model::proto::Prediction& prediction,
                                  const int class_idx) {
    return prediction.classification().distribution().counts(class_idx) /
           prediction.classification().distribution().sum();
  };

  for (dataset::VerticalDataset::row_t row_idx = begin_example_idx;
       row_idx < end_example_idx; row_idx++) {
    // Compute the prediction with the generic engine.
    model.Predict(dataset, row_idx, &generic_prediction);

    switch (model.task()) {
      case model::proto::Task::CLASSIFICATION: {
        // Determine the format of the predictions.
        const int num_classes =
            generic_prediction.classification().distribution().counts().size() -
            1;
        bool compact_format;
        if (predictions.size() == num_examples) {
          compact_format = true;
        } else if (predictions.size() == num_examples * num_classes) {
          compact_format = false;
        } else {
          YDF_LOG(FATAL)
              << "predictions for classification are expected to be of "
                 "size \"num_row\" (compact format) or \"num_rows * "
                 "num_classes\" (classical format). Got num_classes="
              << num_classes << " predictions.size()=" << predictions.size()
              << " num_examples=" << num_examples << ".";
        }

        if (compact_format) {
          CHECK_EQ(num_classes, 2)
              << "Compact format only compatible with binary predictions.";
          // Generic predictions.
          const float pos_probability = get_probability(generic_prediction, 2);
          EXPECT_NEAR(pos_probability, predictions[prediction_idx], epsilon)
              << "Predictions don't match.";
        } else {
          // Precomputed predictions.
          for (int class_idx = 0; class_idx < num_classes; class_idx++) {
            const float probability =
                get_probability(generic_prediction, class_idx + 1);
            EXPECT_NEAR(probability,
                        predictions[prediction_idx * num_classes + class_idx],
                        epsilon)
                << "Predictions don't match.";
          }
        }
      } break;

      case model::proto::Task::REGRESSION:
        EXPECT_NEAR(generic_prediction.regression().value(),
                    predictions[prediction_idx], epsilon)
            << "Predictions don't match.";
        break;

      case model::proto::Task::RANKING:
        EXPECT_NEAR(generic_prediction.ranking().relevance(),
                    predictions[prediction_idx], epsilon)
            << "Predictions don't match.";
        break;

      case model::proto::Task::CATEGORICAL_UPLIFT:
      case model::proto::Task::NUMERICAL_UPLIFT: {
        // Precomputed predictions.
        const int num_effects =
            generic_prediction.uplift().treatment_effect_size();
        for (int effect_idx = 0;
             effect_idx < generic_prediction.uplift().treatment_effect_size();
             effect_idx++) {
          EXPECT_NEAR(generic_prediction.uplift().treatment_effect(effect_idx),
                      predictions[prediction_idx * num_effects + effect_idx],
                      epsilon)
              << "Predictions don't match.";
        }
      } break;

      default:
        YDF_LOG(FATAL) << "Not supported task";
    }
    prediction_idx++;
  }
}

void TestPredefinedHyperParameters(
    const absl::string_view train_ds_path, const absl::string_view test_ds_path,
    const model::proto::TrainingConfig& train_config,
    dataset::proto::DataSpecification data_spec,
    const int expected_num_preconfigured_parameters,
    absl::optional<float> min_accuracy) {
  dataset::VerticalDataset train_ds;
  CHECK_OK(LoadVerticalDataset(train_ds_path, data_spec, &train_ds));
  dataset::VerticalDataset test_ds;
  CHECK_OK(LoadVerticalDataset(test_ds_path, data_spec, &test_ds));

  // Retrieve the preconfigured parameters.
  std::unique_ptr<model::AbstractLearner> base_learner;
  CHECK_OK(model::GetLearner(train_config, &base_learner, {}));
  const auto predefined_hyper_parameters =
      base_learner->PredefinedHyperParameters();

  CHECK_EQ(expected_num_preconfigured_parameters,
           predefined_hyper_parameters.size());

  for (const auto& hyper_parameters : predefined_hyper_parameters) {
    YDF_LOG(INFO) << "Testing hyper-parameters " << hyper_parameters.name();
    // Configure a learner
    std::unique_ptr<model::AbstractLearner> learner;
    CHECK_OK(model::GetLearner(train_config, &learner, {}));
    CHECK_OK(learner->SetHyperParameters(hyper_parameters.parameters()));

    // Train a model.
    const auto model = learner->TrainWithStatus(train_ds).value();

    // Evaluate the model.
    if (min_accuracy.has_value()) {
      utils::RandomEngine rnd(1234);  // Not used.
      const auto evaluation = model->Evaluate(test_ds, {}, &rnd);
      EXPECT_GE(metric::Accuracy(evaluation), min_accuracy.value());
    }
  }
}

void TestPredefinedHyperParametersAdultDataset(
    model::proto::TrainingConfig train_config,
    const int expected_num_preconfigured_parameters,
    absl::optional<float> min_accuracy) {
  const auto base_ds_path = absl::StrCat(
      "csv:", file::JoinPath(
                  test::DataRootDirectory(),
                  "yggdrasil_decision_forests/test_data/dataset"));
  const auto train_ds_path = file::JoinPath(base_ds_path, "adult_train.csv");
  const auto test_ds_path = file::JoinPath(base_ds_path, "adult_test.csv");
  // Create a dataspec.
  dataset::proto::DataSpecification data_spec;
  ASSERT_OK(
      dataset::CreateDataSpecWithStatus(train_ds_path, false, {}, &data_spec));

  train_config.set_label("income");

  TestPredefinedHyperParameters(
      train_ds_path, test_ds_path, train_config, data_spec,
      expected_num_preconfigured_parameters, min_accuracy);
}

void TestPredefinedHyperParametersRankingDataset(
    model::proto::TrainingConfig train_config,
    const int expected_num_preconfigured_parameters,
    absl::optional<float> min_accuracy) {
  const auto base_ds_path = absl::StrCat(
      "csv:", file::JoinPath(
                  test::DataRootDirectory(),
                  "yggdrasil_decision_forests/test_data/dataset"));
  const auto train_ds_path =
      file::JoinPath(base_ds_path, "synthetic_ranking_train.csv");
  const auto test_ds_path =
      file::JoinPath(base_ds_path, "synthetic_ranking_test.csv");

  train_config.set_label("LABEL");
  train_config.set_ranking_group("GROUP");
  train_config.set_task(model::proto::Task::RANKING);

  // Create a dataspec.
  dataset::proto::DataSpecificationGuide guide;
  auto* group_guide = guide.add_column_guides();
  group_guide->set_column_name_pattern("^GROUP$");
  group_guide->set_type(dataset::proto::HASH);

  dataset::proto::DataSpecification data_spec;
  ASSERT_OK(dataset::CreateDataSpecWithStatus(train_ds_path, false, guide,
                                              &data_spec));

  TestPredefinedHyperParameters(
      train_ds_path, test_ds_path, train_config, data_spec,
      expected_num_preconfigured_parameters, min_accuracy);
}

std::string ShardDataset(const dataset::VerticalDataset& dataset,
                         const int num_shards, const float sampling,
                         const absl::string_view format,
                         const absl::string_view name) {
  const auto sharded_dir = file::JoinPath(test::TmpDirectory(), name);
  const auto sharded_path =
      file::JoinPath(sharded_dir, absl::StrCat("dataset@", num_shards));
  const auto typed_sharded_path = absl::StrCat(format, ":", sharded_path);
  CHECK_OK(file::RecursivelyCreateDir(sharded_dir, file::Defaults()));
  std::vector<std::string> shards;
  CHECK_OK(utils::ExpandOutputShards(sharded_path, &shards));

  // Down-sample the number of examples.
  std::vector<dataset::VerticalDataset::row_t> examples(dataset.nrow());
  std::iota(examples.begin(), examples.end(), 0);
  std::mt19937 rnd;
  std::shuffle(examples.begin(), examples.end(), rnd);
  examples.resize(std::lround(sampling * dataset.nrow()));

  for (int shard_idx = 0; shard_idx < num_shards; shard_idx++) {
    std::vector<dataset::VerticalDataset::row_t> idxs;
    for (int i = shard_idx; i < examples.size(); i += num_shards) {
      idxs.push_back(examples[i]);
    }
    CHECK_OK(dataset::SaveVerticalDataset(
        dataset.Extract(idxs).value(),
        absl::StrCat(format, ":", shards[shard_idx])));
  }
  return typed_sharded_path;
}

absl::Status ExportUpliftPredictionsToTFUpliftCsvFormat(
    const model::AbstractModel& model, const dataset::VerticalDataset& dataset,
    absl::string_view output_csv_path) {
  ASSIGN_OR_RETURN(auto output_handle, file::OpenOutputFile(output_csv_path));
  utils::csv::Writer writer(output_handle.get());
  RETURN_IF_ERROR(writer.WriteRow({"uplift", "response", "weight", "group"}));

  for (int example_idx = 0; example_idx < dataset.nrow(); example_idx++) {
    model::proto::Prediction prediction;
    model.Predict(dataset, example_idx, &prediction);
    RETURN_IF_ERROR(model.SetGroundTruth(dataset, example_idx, &prediction));

    if (prediction.uplift().treatment_effect_size() != 1) {
      return absl::InvalidArgumentError("Only binary effect supported");
    }
    const auto uplift_str =
        absl::StrCat(prediction.uplift().treatment_effect(0));
    if (!prediction.uplift().has_outcome_categorical()) {
      return absl::InvalidArgumentError("Only categorical outcome supported");
    }
    const auto response_str =
        absl::StrCat(prediction.uplift().outcome_categorical() - 1);
    const auto group_str = absl::StrCat(prediction.uplift().treatment() - 1);
    const auto weight_str = absl::StrCat(prediction.weight());

    RETURN_IF_ERROR(
        writer.WriteRow({uplift_str, response_str, weight_str, group_str}));
  }
  RETURN_IF_ERROR(output_handle->Close());
  YDF_LOG(INFO) << "Uplift predictions exported to: " << output_csv_path;
  return absl::OkStatus();
}

void InternalExportMetricCondition(const absl::string_view test,
                                   const double value, const double center,
                                   const double margin, const double golden,
                                   const absl::string_view metric,
                                   const int line,
                                   const absl::string_view file) {
  // Margin of error when comparing golden metric values.
  constexpr double kGoldenMargin = 0.0001;

  const auto filename = file::GetBasename(file);
  const bool golden_test = kYdfTestMetricCheckGold && !std::isnan(golden);

  double abs_diff_margin = std::abs(value - center);
  double abs_diff_golden = std::abs(value - golden);
  bool success_margin = abs_diff_margin < margin;
  bool success_golden = abs_diff_golden < kGoldenMargin;

  if (strlen(kYdfTestMetricDumpDir) > 0) {
    // Export metric to csv file.
    const auto uid = GenUniqueId();
    const auto path =
        file::JoinPath(kYdfTestMetricDumpDir, absl::StrCat(uid, ".csv"));
    std::string content = absl::StrCat(
        "test,value,center,margin,metric,line,filename,success_margin,success_"
        "golden,golden\n",
        test, ",", value, ",", center, ",", margin, ",", metric, ",", line, ",",
        filename, ",", success_margin, ",", success_golden, ",", golden);
    CHECK_OK(file::SetContent(path, content));
  } else {
    if (!success_margin) {
      EXPECT_TRUE(false) << "Non satisfied range condition for " << metric
                         << " in " << test << "\ndefined at\n"
                         << file << ":" << line << "\nThe metric value "
                         << value << " is not in " << center << " +- " << margin
                         << ".\ni.e. not in [" << (center - margin) << " , "
                         << (center + margin)
                         << "].\nThe absolute value of the difference is "
                         << abs_diff_margin << ".";
    }

    if (golden_test && !success_golden) {
      EXPECT_TRUE(false) << "Non satisfied golden value condition for "
                         << metric << " in " << test << "\ndefined at\n"
                         << file << ":" << line << "\nThe metric value "
                         << value << " is different from " << golden
                         << " (margin:" << kGoldenMargin
                         << ").\nThe absolute value of the difference is "
                         << abs_diff_golden << ".";
    }
  }
}

int GetVariableImportanceRank(
    const absl::string_view attribute,
    const dataset::proto::DataSpecification& data_spec,
    const std::vector<model::proto::VariableImportance>& variable_importance) {
  const int attribute_idx = dataset::GetColumnIdxFromName(attribute, data_spec);
  const auto found_iterator = std::find_if(
      variable_importance.begin(), variable_importance.end(),
      [attribute_idx](const model::proto::VariableImportance& var) {
        return var.attribute_idx() == attribute_idx;
      });
  CHECK(found_iterator != variable_importance.end());
  return std::distance(variable_importance.begin(), found_iterator);
}

}  // namespace utils
}  // namespace yggdrasil_decision_forests
