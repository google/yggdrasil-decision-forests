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

// Benchmark the training speed of a set of models.
//
// Usage example:
// yggdrasil_decision_forests/cli/monitoring/run_benchmark_training.sh

#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/synthetic_dataset.h"
#include "yggdrasil_decision_forests/dataset/synthetic_dataset.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/learner/random_forest/random_forest.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/utils/csv.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/usage.h"

constexpr char kUsageMessage[] =
    "Benchmarks the training time of various learners.";

ABSL_FLAG(std::string, output, "",
          "Output csv file with results. If not specified, results are only "
          "printed in the std out.");

ABSL_FLAG(std::string, tmp, "/tmp/benchmark",
          "Directory used to store the intermediate results");

namespace yggdrasil_decision_forests {

struct Result {
  std::string name;
  absl::Duration training_time;
};

struct SyntheticParams {
  // Number of training examples.
  int num_examples;
  // Number of features.
  // Note: Currently, only numerical features are generated.
  int num_features;
  // Number of label classes
  int num_classes = 2;
  // Resolution of the numerical features. If -1, features have infinite
  // resolution (this is not allowed for DISCRETIZED_NUMERICAL features).
  int resolution = -1;
  // If true, create DISCRETIZED_NUMERICAL features. If false, create NUMERICAL
  // features.
  bool use_discretized_numerical_features = false;

  std::string to_string() const {
    std::string v = absl::StrCat("e:", num_examples, " f:", num_features,
                                 " c:", num_classes);
    if (use_discretized_numerical_features) {
      absl::StrAppend(&v, " disc");
    }
    if (resolution != -1) {
      absl::StrAppend(&v, " res:", resolution);
    }
    return v;
  }
};

// Utility class for the training of a model.
class Trainer {
 public:
  Trainer() {
    // By default, train on 12 threads.
    // Note: Some experiments will change the number of threads.
    deployment_config_.set_num_threads(12);
  }

  // Loads the dataset in memory, and prepare the training.
  absl::Status PrepareTraining() {
    if (!dataset_filename_.empty()) {
      const std::string dataset_path =
          absl::StrCat(dataset_format_, ":",
                       file::JoinPath(dataset_directory_, dataset_filename_));

      // Infer the dataspec.
      if (!dataspec_guide_filename_.empty()) {
        dataset::proto::DataSpecificationGuide loaded_guide;
        const std::string guide_path =
            file::JoinPath(dataset_directory_, dataspec_guide_filename_);
        RETURN_IF_ERROR(
            file::GetTextProto(guide_path, &loaded_guide, file::Defaults()));
        guide_.MergeFrom(loaded_guide);
      }

      dataset::proto::DataSpecification data_spec;
      dataset::CreateDataSpec(dataset_path, false, guide_, &data_spec);
      YDF_LOG(INFO) << "Dataspec:\n"
                    << dataset::PrintHumanReadable(data_spec, false);

      // Load the dataset in memory.
      RETURN_IF_ERROR(
          LoadVerticalDataset(dataset_path, data_spec, &train_dataset_));
    }

    // Configure the trainer.
    RETURN_IF_ERROR(model::GetLearner(train_config_, &learner_));
    *learner_->mutable_deployment() = deployment_config_;

    return absl::OkStatus();
  }

  // Trains a model. Calling this function multiple times will train multiple
  // models.
  absl::Status Run(const std::string& name, std::vector<Result>* results) {
    int idx = results->size() + 1;

    std::cout << "[" << idx << "] Prepare \"" << name << "\"" << std::endl;
    RETURN_IF_ERROR(PrepareTraining());

    std::cout << "[" << idx << "] Running \"" << name << "\"" << std::endl;

    const auto start_time = absl::Now();
    ASSIGN_OR_RETURN(const auto model,
                     learner_->TrainWithStatus(train_dataset_));
    const auto end_time = absl::Now();

    // Record results.
    const auto training_time = end_time - start_time;

    results->push_back({/*name=*/name, /*training_time=*/training_time});

    std::cout << "[" << idx << "] Trained \"" << name << "\" in "
              << absl::StrFormat("%.5g", absl::ToDoubleSeconds(training_time))
              << " seconds" << std::endl;
    return absl::OkStatus();
  }

  // Create a synthetic dataset.
  //
  // TODO: Replace with dataset/synthetic_generator.
  absl::Status GenSyntheticDataset(const SyntheticParams& option) {
    // Set dataspec.
    const std::string label = "label";
    auto* label_col =
        dataset::AddColumn(label, dataset::proto::ColumnType::CATEGORICAL,
                           train_dataset_.mutable_data_spec());
    label_col->mutable_categorical()->set_number_of_unique_values(
        option.num_classes + 1);
    label_col->mutable_categorical()->set_is_already_integerized(true);

    std::vector<std::string> feature_names;
    std::vector<float> feature_weights;
    feature_names.reserve(option.num_features);
    feature_weights.reserve(option.num_features);
    for (int feature_idx = 0; feature_idx < option.num_features;
         feature_idx++) {
      const auto feature_name = absl::StrCat("x", feature_idx);
      feature_names.push_back(feature_name);
      feature_weights.push_back(
          static_cast<float>(option.num_features - feature_idx) /
          option.num_features);

      if (option.use_discretized_numerical_features) {
        auto* column = dataset::AddColumn(
            feature_name, dataset::proto::ColumnType::DISCRETIZED_NUMERICAL,
            train_dataset_.mutable_data_spec());
        auto* boundaries =
            column->mutable_discretized_numerical()->mutable_boundaries();
        if (option.resolution <= 0) {
          return absl::InvalidArgumentError(
              "numerical_feature_resolution should be >0 for "
              "DISCRETIZED_NUMERICAL features");
        }
        for (int idx = 0; idx < option.resolution - 1; idx++) {
          boundaries->Add(idx);
        }
      } else {
        dataset::AddColumn(feature_name, dataset::proto::ColumnType::NUMERICAL,
                           train_dataset_.mutable_data_spec());
      }
    }
    RETURN_IF_ERROR(train_dataset_.CreateColumnsFromDataspec());

    const float sum_feature_weights =
        std::accumulate(feature_weights.begin(), feature_weights.end(), 0.);

    // Link to dataset data.
    std::vector<dataset::VerticalDataset::NumericalColumn*> numerical_features;
    std::vector<dataset::VerticalDataset::DiscretizedNumericalColumn*>
        discretized_numerical_features;
    if (option.use_discretized_numerical_features) {
      discretized_numerical_features.reserve(feature_names.size());
      for (const auto& feature_name : feature_names) {
        ASSIGN_OR_RETURN(
            const auto mutable_col,
            train_dataset_.MutableColumnWithCastWithStatus<
                dataset::VerticalDataset::DiscretizedNumericalColumn>(
                train_dataset_.ColumnNameToColumnIdx(feature_name)));
        discretized_numerical_features.push_back(mutable_col);
      }
    } else {
      numerical_features.reserve(feature_names.size());
      for (const auto& feature_name : feature_names) {
        ASSIGN_OR_RETURN(
            const auto mutable_col,
            train_dataset_.MutableColumnWithCastWithStatus<
                dataset::VerticalDataset::NumericalColumn>(
                train_dataset_.ColumnNameToColumnIdx(feature_name)));
        numerical_features.push_back(mutable_col);
      }
    }
    ASSIGN_OR_RETURN(auto* label_data,
                     train_dataset_.MutableColumnWithCastWithStatus<
                         dataset::VerticalDataset::CategoricalColumn>(
                         train_dataset_.ColumnNameToColumnIdx(label)));

    // Fill dataset.
    utils::RandomEngine random;
    random.seed(1234);
    std::uniform_real_distribution<float> unif_01;
    for (int example_idx = 0; example_idx < option.num_examples;
         example_idx++) {
      double accumulator = 0.;
      for (int feature_idx = 0; feature_idx < feature_names.size();
           feature_idx++) {
        float value = unif_01(random);
        accumulator += value * feature_weights[feature_idx];
        if (option.resolution != -1) {
          value = std::round(value * option.resolution);
        }
        if (option.use_discretized_numerical_features) {
          discretized_numerical_features[feature_idx]->Add(
              static_cast<int>(value));
        } else {
          numerical_features[feature_idx]->Add(value);
        }
      }

      const int label_value = 1 + std::floor(accumulator / sum_feature_weights *
                                             option.num_classes);
      label_data->Add(label_value);
    }
    train_dataset_.set_nrow(option.num_examples);

    // Config trainer.
    train_config().set_task(model::proto::Task::CLASSIFICATION);
    train_config().set_label(label);

    return absl::OkStatus();
  }

  // Accessors to the configuration fields. Should be set before calling
  // "PrepareTraining".
  model::proto::TrainingConfig& train_config() { return train_config_; }
  model::proto::DeploymentConfig& deployment_config() {
    return deployment_config_;
  }
  std::string& dataset_filename() { return dataset_filename_; }
  std::string& dataset_format() { return dataset_format_; }
  std::string& dataspec_guide_filename() { return dataspec_guide_filename_; }
  dataset::proto::DataSpecificationGuide& guide() { return guide_; }
  std::string& dataset_directory() { return dataset_directory_; }

 private:
  // Training configuration fields.
  model::proto::TrainingConfig train_config_;
  model::proto::DeploymentConfig deployment_config_;
  std::string dataset_filename_;
  std::string dataset_format_ = "csv";
  std::string dataspec_guide_filename_;
  dataset::proto::DataSpecificationGuide guide_;
  std::string dataset_directory_ =
      "yggdrasil_decision_forests/test_data/dataset";

  dataset::VerticalDataset train_dataset_;
  std::unique_ptr<model::AbstractLearner> learner_;
};

// Variations for GBT models.
std::vector<std::pair<
    std::string, std::function<void(model::gradient_boosted_trees::proto::
                                        GradientBoostedTreesTrainingConfig*)>>>
GBTVariations() {
  return {
      {"",
       [](model::gradient_boosted_trees::proto::
              GradientBoostedTreesTrainingConfig* gbt_config) {
         // Leave all defaults.
         // Note: Use sorting_strategy=PRESORTED which use PRESORTED except when
         // note interesting.
         // Note: Ideally, PRESORTED should be renamed to AUTO.
       }},

      // In node training is the naive way to train individual trees. It is
      // generally very slow (this is why is it not tested by default).
      /*
      {" in_node",
       [](model::gradient_boosted_trees::proto::
              GradientBoostedTreesTrainingConfig* gbt_config) {
         gbt_config->mutable_decision_tree()
             ->mutable_internal()
             ->set_sorting_strategy(
                 model::decision_tree::proto::DecisionTreeTrainingConfig::
                     Internal::IN_NODE);
       }},
       */

      // Make sure PRESORTED is always used.
      // Note: Ideally, FORCE_PRESORTED should be renamed to PRESORTED.
      {" force_presorted",
       [](model::gradient_boosted_trees::proto::
              GradientBoostedTreesTrainingConfig* gbt_config) {
         gbt_config->mutable_decision_tree()
             ->mutable_internal()
             ->set_sorting_strategy(
                 model::decision_tree::proto::DecisionTreeTrainingConfig::
                     Internal::FORCE_PRESORTED);
       }},
  };
}

// Variations for RF models.
std::vector<
    std::pair<std::string,
              std::function<void(
                  model::random_forest::proto::RandomForestTrainingConfig*)>>>
RFVariations() {
  return {
      {"",
       [](model::random_forest::proto::RandomForestTrainingConfig* rf_config) {
         // Leave all defaults.
         // Note: Use sorting_strategy=PRESORTED.
       }},
  };
}

absl::Status Benchmark_RandomForest_Adult(std::vector<Result>* results) {
  for (const auto& variation : RFVariations()) {
    // Configure benchmark.
    Trainer trainer;
    trainer.train_config().set_learner(
        model::random_forest::RandomForestLearner::kRegisteredName);

    trainer.train_config().set_task(model::proto::Task::CLASSIFICATION);
    trainer.train_config().set_label("income");
    trainer.dataset_filename() = "adult.csv";

    auto* rf_config = trainer.train_config().MutableExtension(
        model::random_forest::proto::random_forest_config);
    variation.second(rf_config);

    // Run benchmark.
    RETURN_IF_ERROR(
        trainer.Run(absl::StrCat("RF Adult", variation.first), results));
  }
  return absl::OkStatus();
}

absl::Status Benchmark_RandomForest_Adult_1Thread(
    std::vector<Result>* results) {
  for (const auto& variation : RFVariations()) {
    // Configure benchmark.
    Trainer trainer;
    trainer.train_config().set_learner(
        model::random_forest::RandomForestLearner::kRegisteredName);
    trainer.deployment_config().set_num_threads(1);
    trainer.train_config().set_task(model::proto::Task::CLASSIFICATION);
    trainer.train_config().set_label("income");
    trainer.dataset_filename() = "adult.csv";

    auto* rf_config = trainer.train_config().MutableExtension(
        model::random_forest::proto::random_forest_config);
    variation.second(rf_config);

    // Run benchmark.
    RETURN_IF_ERROR(trainer.Run(
        absl::StrCat("RF Adult 1-thread", variation.first), results));
  }
  return absl::OkStatus();
}

absl::Status Benchmark_RandomForest_Synthetic(std::vector<Result>* results) {
  for (const auto& synthetic : std::vector<SyntheticParams>{
           // A small dataset with few features.
           {.num_examples = 100'000, .num_features = 20},
           // A small dataset with some features.
           {.num_examples = 100'000, .num_features = 100},
           // A small dataset with some features and many classes.
           // Having classes is expensive for YDF's GBT implementation.
           {.num_examples = 100'000, .num_features = 100, .num_classes = 10},
           // Large datasets with some features.
           // Note: A 2-3 millions examples (depending on the parameters), we
           // observe a point of inflection of the training speed.
           {.num_examples = 1'000'000, .num_features = 200},
           {.num_examples = 1'000'000, .num_features = 200, .resolution = 1000},
       }) {
    for (const auto& variation : RFVariations()) {
      // Configure benchmark.
      Trainer trainer;
      trainer.train_config().set_learner(
          model::random_forest::RandomForestLearner::kRegisteredName);

      auto* rf_config = trainer.train_config().MutableExtension(
          model::random_forest::proto::random_forest_config);
      rf_config->set_num_trees(10);
      variation.second(rf_config);

      RETURN_IF_ERROR(trainer.GenSyntheticDataset(synthetic));

      // Run benchmark.
      RETURN_IF_ERROR(
          trainer.Run(absl::StrCat("RF Synthetic t:10 ", synthetic.to_string(),
                                   variation.first),
                      results));
    }
  }
  return absl::OkStatus();
}

absl::Status Benchmark_GBT_Adult(std::vector<Result>* results) {
  for (const auto& variation : GBTVariations()) {
    // Configure benchmark.
    Trainer trainer;
    trainer.train_config().set_learner(
        model::gradient_boosted_trees::GradientBoostedTreesLearner::
            kRegisteredName);

    trainer.train_config().set_task(model::proto::Task::CLASSIFICATION);
    trainer.train_config().set_label("income");
    trainer.dataset_filename() = "adult.csv";

    auto* gbt_config = trainer.train_config().MutableExtension(
        model::gradient_boosted_trees::proto::gradient_boosted_trees_config);
    variation.second(gbt_config);

    // Run benchmark.
    RETURN_IF_ERROR(
        trainer.Run(absl::StrCat("GBT Adult", variation.first), results));
  }
  return absl::OkStatus();
}

absl::Status Benchmark_GBT_Adult_NoEarlyStop(std::vector<Result>* results) {
  for (const auto& variation : GBTVariations()) {
    // Configure benchmark.
    Trainer trainer;
    trainer.train_config().set_learner(
        model::gradient_boosted_trees::GradientBoostedTreesLearner::
            kRegisteredName);
    trainer.train_config().set_task(model::proto::Task::CLASSIFICATION);
    trainer.train_config().set_label("income");
    trainer.dataset_filename() = "adult.csv";

    auto* gbt_config = trainer.train_config().MutableExtension(
        model::gradient_boosted_trees::proto::gradient_boosted_trees_config);
    gbt_config->set_early_stopping(
        model::gradient_boosted_trees::proto::
            GradientBoostedTreesTrainingConfig::NONE);
    variation.second(gbt_config);

    // Run benchmark.
    RETURN_IF_ERROR(trainer.Run(
        absl::StrCat("GBT Adult NoEarlyStop", variation.first), results));
  }

  return absl::OkStatus();
}

absl::Status Benchmark_GBT_Adult_NoEarlyStop_Oblique(
    std::vector<Result>* results) {
  for (const auto& variation : GBTVariations()) {
    // Configure benchmark.
    Trainer trainer;
    trainer.train_config().set_learner(
        model::gradient_boosted_trees::GradientBoostedTreesLearner::
            kRegisteredName);
    trainer.train_config().set_task(model::proto::Task::CLASSIFICATION);
    trainer.train_config().set_label("income");
    trainer.dataset_filename() = "adult.csv";

    auto* gbt_config = trainer.train_config().MutableExtension(
        model::gradient_boosted_trees::proto::gradient_boosted_trees_config);
    gbt_config->set_early_stopping(
        model::gradient_boosted_trees::proto::
            GradientBoostedTreesTrainingConfig::NONE);
    gbt_config->mutable_decision_tree()->mutable_sparse_oblique_split();
    variation.second(gbt_config);

    // Run benchmark.
    RETURN_IF_ERROR(trainer.Run(
        absl::StrCat("GBT Adult NoEarlyStop Oblique", variation.first),
        results));
  }
  return absl::OkStatus();
}

absl::Status Benchmark_GBT_Adult_Hessian(std::vector<Result>* results) {
  for (const auto& variation : GBTVariations()) {
    // Configure benchmark.
    Trainer trainer;
    trainer.train_config().set_learner(
        model::gradient_boosted_trees::GradientBoostedTreesLearner::
            kRegisteredName);

    trainer.train_config().set_task(model::proto::Task::CLASSIFICATION);
    trainer.train_config().set_label("income");
    trainer.dataset_filename() = "adult.csv";

    auto* gbt_config = trainer.train_config().MutableExtension(
        model::gradient_boosted_trees::proto::gradient_boosted_trees_config);
    gbt_config->set_use_hessian_gain(true);
    variation.second(gbt_config);

    // Run benchmark.
    RETURN_IF_ERROR(trainer.Run(
        absl::StrCat("GBT Adult Hessian", variation.first), results));
  }
  return absl::OkStatus();
}

absl::Status Benchmark_GBT_Adult_Hessian_NoEarlyStop(
    std::vector<Result>* results) {
  for (const auto& variation : GBTVariations()) {
    // Configure benchmark.
    Trainer trainer;
    trainer.train_config().set_learner(
        model::gradient_boosted_trees::GradientBoostedTreesLearner::
            kRegisteredName);
    trainer.train_config().set_task(model::proto::Task::CLASSIFICATION);
    trainer.train_config().set_label("income");
    trainer.dataset_filename() = "adult.csv";

    auto* gbt_config = trainer.train_config().MutableExtension(
        model::gradient_boosted_trees::proto::gradient_boosted_trees_config);
    gbt_config->set_use_hessian_gain(true);
    gbt_config->set_early_stopping(
        model::gradient_boosted_trees::proto::
            GradientBoostedTreesTrainingConfig::NONE);
    variation.second(gbt_config);

    // Run benchmark.
    RETURN_IF_ERROR(trainer.Run(
        absl::StrCat("GBT Adult Hessian NoEarlyStop", variation.first),
        results));
  }
  return absl::OkStatus();
}

absl::Status Benchmark_GBT_Adult_Discretized(std::vector<Result>* results) {
  for (const auto& variation : GBTVariations()) {
    // Configure benchmark.
    Trainer trainer;
    trainer.train_config().set_learner(
        model::gradient_boosted_trees::GradientBoostedTreesLearner::
            kRegisteredName);
    trainer.guide().set_detect_numerical_as_discretized_numerical(true);
    trainer.train_config().set_task(model::proto::Task::CLASSIFICATION);
    trainer.train_config().set_label("income");
    trainer.dataset_filename() = "adult.csv";

    auto* gbt_config = trainer.train_config().MutableExtension(
        model::gradient_boosted_trees::proto::gradient_boosted_trees_config);
    variation.second(gbt_config);

    // Run benchmark.
    RETURN_IF_ERROR(trainer.Run(
        absl::StrCat("GBT Adult Discretized", variation.first), results));
  }
  return absl::OkStatus();
}

absl::Status Benchmark_GBT_Adult_Discretized_NoEarlyStop(
    std::vector<Result>* results) {
  for (const auto& variation : GBTVariations()) {
    // Configure benchmark.
    Trainer trainer;
    trainer.train_config().set_learner(
        model::gradient_boosted_trees::GradientBoostedTreesLearner::
            kRegisteredName);
    trainer.guide().set_detect_numerical_as_discretized_numerical(true);
    trainer.train_config().set_task(model::proto::Task::CLASSIFICATION);
    trainer.train_config().set_label("income");
    trainer.dataset_filename() = "adult.csv";

    auto* gbt_config = trainer.train_config().MutableExtension(
        model::gradient_boosted_trees::proto::gradient_boosted_trees_config);
    gbt_config->set_early_stopping(
        model::gradient_boosted_trees::proto::
            GradientBoostedTreesTrainingConfig::NONE);

    variation.second(gbt_config);

    // Run benchmark.
    RETURN_IF_ERROR(trainer.Run(
        absl::StrCat("GBT Adult Discretized NoEarlyStop", variation.first),
        results));
  }
  return absl::OkStatus();
}

absl::Status Benchmark_GBT_Adult_1Thread(std::vector<Result>* results) {
  for (const auto& variation : GBTVariations()) {
    // Configure benchmark.
    Trainer trainer;
    trainer.train_config().set_learner(
        model::gradient_boosted_trees::GradientBoostedTreesLearner::
            kRegisteredName);
    trainer.deployment_config().set_num_threads(1);

    trainer.train_config().set_task(model::proto::Task::CLASSIFICATION);
    trainer.train_config().set_label("income");
    trainer.dataset_filename() = "adult.csv";

    auto* gbt_config = trainer.train_config().MutableExtension(
        model::gradient_boosted_trees::proto::gradient_boosted_trees_config);
    variation.second(gbt_config);

    // Run benchmark.
    RETURN_IF_ERROR(trainer.Run(
        absl::StrCat("GBT Adult 1-thread", variation.first), results));
  }
  return absl::OkStatus();
}

absl::Status Benchmark_GBT_Synthetic(std::vector<Result>* results) {
  for (const auto& synthetic : std::vector<SyntheticParams>{
           // A small dataset with few features.
           {.num_examples = 100'000, .num_features = 20},
           // A small dataset with some features.
           {.num_examples = 100'000, .num_features = 100},
           // A small dataset with some features and many classes.
           // Having classes is expensive for YDF's GBT implementation.
           {.num_examples = 100'000, .num_features = 100, .num_classes = 10},
           // Large datasets with some features.
           // Note: A 2-3 millions examples (depending on the parameters), we
           // observe a point of inflection of the training speed.
           {.num_examples = 1'000'000, .num_features = 200},
           {.num_examples = 1'000'000, .num_features = 200, .resolution = 1000},
           {.num_examples = 1'000'000,
            .num_features = 200,
            .resolution = 1000,
            .use_discretized_numerical_features = true},
           {.num_examples = 4'000'000, .num_features = 200},
           {.num_examples = 4'000'000, .num_features = 200, .resolution = 1000},
           {.num_examples = 4'000'000,
            .num_features = 200,
            .resolution = 1000,
            .use_discretized_numerical_features = true},
       }) {
    for (const auto& strategy : GBTVariations()) {
      // Configure benchmark.
      Trainer trainer;
      trainer.train_config().set_learner(
          model::gradient_boosted_trees::GradientBoostedTreesLearner::
              kRegisteredName);

      auto* gbt_config = trainer.train_config().MutableExtension(
          model::gradient_boosted_trees::proto::gradient_boosted_trees_config);
      gbt_config->set_num_trees(4);  // Note: Trees are trained sequentially.
      gbt_config->set_validation_set_ratio(0.f);
      gbt_config->set_early_stopping(
          model::gradient_boosted_trees::proto::
              GradientBoostedTreesTrainingConfig::NONE);
      strategy.second(gbt_config);

      RETURN_IF_ERROR(trainer.GenSyntheticDataset(synthetic));

      // Run benchmark.
      RETURN_IF_ERROR(
          trainer.Run(absl::StrCat("GBT Synthetic t:4 ", synthetic.to_string(),
                                   strategy.first),
                      results));
    }
  }
  return absl::OkStatus();
}

// TODO: Generating examples to tfexamples is *very* slow (it takes order
// of magnitude more time that training). Generate examples as YDF dataset to
// YDF examples instead.
absl::Status Benchmark_GBT_SyntheticV2(std::vector<Result>* results) {
  std::vector<std::pair<std::string, dataset::proto::SyntheticDatasetOptions>>
      synthetics;

  // Disable most features.
  dataset::proto::SyntheticDatasetOptions default_synthetic;
  default_synthetic.set_num_numerical(50);
  default_synthetic.set_num_categorical(20);
  default_synthetic.set_num_categorical_set(0);
  default_synthetic.set_num_boolean(20);
  default_synthetic.set_num_multidimensional_numerical(0);

  {
    auto synthetic = default_synthetic;
    synthetic.set_num_examples(100'000);
    synthetics.push_back({"e:100k", synthetic});
  }

  {
    auto synthetic = default_synthetic;
    synthetic.set_num_examples(1'000'000);
    synthetics.push_back({"e:1M", synthetic});
  }

  {
    auto synthetic = default_synthetic;
    synthetic.set_num_examples(4'000'000);
    synthetics.push_back({"e:4M", synthetic});
  }

  for (const auto& synthetic : synthetics) {
    const auto dataset_format = "tfrecord+tfe";
    const auto dataset_filename = "dataset.tfe";
    const auto dataset_directory = absl::GetFlag(FLAGS_tmp);
    RETURN_IF_ERROR(
        file::RecursivelyCreateDir(dataset_directory, file::Defaults()));
    const std::string dst_path =
        absl::StrCat(dataset_format, ":",
                     file::JoinPath(dataset_directory, dataset_filename));
    YDF_LOG(INFO) << "Generating synthetic dataset";
    RETURN_IF_ERROR(
        dataset::GenerateSyntheticDataset(synthetic.second, dst_path));

    for (const auto& strategy : GBTVariations()) {
      // Configure benchmark.
      Trainer trainer;
      trainer.train_config().set_learner(
          model::gradient_boosted_trees::GradientBoostedTreesLearner::
              kRegisteredName);

      auto* gbt_config = trainer.train_config().MutableExtension(
          model::gradient_boosted_trees::proto::gradient_boosted_trees_config);
      gbt_config->set_num_trees(4);  // Note: Trees are trained sequentially.
      gbt_config->set_validation_set_ratio(0.f);
      gbt_config->set_early_stopping(
          model::gradient_boosted_trees::proto::
              GradientBoostedTreesTrainingConfig::NONE);
      strategy.second(gbt_config);

      trainer.train_config().set_label(synthetic.second.label_name());
      trainer.dataset_filename() = dataset_filename;
      trainer.dataset_directory() = dataset_directory;
      trainer.dataset_format() = dataset_format;

      // Run benchmark.
      RETURN_IF_ERROR(trainer.Run(
          absl::StrCat("GBT SyntheticV2 t:4 ", synthetic.first, strategy.first),
          results));
    }
  }
  return absl::OkStatus();
}

absl::Status Benchmark_GBT_Synthetic_Oblique(std::vector<Result>* results) {
  for (const auto& synthetic : std::vector<SyntheticParams>{
           {/*num_examples=*/10'000, /*num_features=*/20, /*num_classes=*/2},
           {/*num_examples=*/100'000, /*num_features=*/20, /*num_classes=*/2},
       }) {
    for (const auto& strategy : GBTVariations()) {
      // Configure benchmark.
      Trainer trainer;
      trainer.train_config().set_learner(
          model::gradient_boosted_trees::GradientBoostedTreesLearner::
              kRegisteredName);

      auto* gbt_config = trainer.train_config().MutableExtension(
          model::gradient_boosted_trees::proto::gradient_boosted_trees_config);
      gbt_config->set_num_trees(10);
      gbt_config->set_early_stopping(
          model::gradient_boosted_trees::proto::
              GradientBoostedTreesTrainingConfig::NONE);
      gbt_config->mutable_decision_tree()->mutable_sparse_oblique_split();
      strategy.second(gbt_config);

      RETURN_IF_ERROR(trainer.GenSyntheticDataset(synthetic));

      // Run benchmark.
      RETURN_IF_ERROR(
          trainer.Run(absl::StrCat("GBT Synthetic Oblique t:10 ",
                                   synthetic.to_string(), strategy.first),
                      results));
    }
  }
  return absl::OkStatus();
}

absl::Status Benchmark_GBT_Iris(std::vector<Result>* results) {
  for (const auto& strategy : GBTVariations()) {
    // Configure benchmark.
    Trainer trainer;
    trainer.train_config().set_learner(
        model::gradient_boosted_trees::GradientBoostedTreesLearner::
            kRegisteredName);

    trainer.train_config().set_task(model::proto::Task::CLASSIFICATION);
    trainer.train_config().set_label("class");
    trainer.dataset_filename() = "iris.csv";

    auto* gbt_config = trainer.train_config().MutableExtension(
        model::gradient_boosted_trees::proto::gradient_boosted_trees_config);
    strategy.second(gbt_config);

    // Run benchmark.
    RETURN_IF_ERROR(
        trainer.Run(absl::StrCat("GBT Iris", strategy.first), results));
  }
  return absl::OkStatus();
}

absl::Status Benchmark_GBT_Abalone(std::vector<Result>* results) {
  for (const auto& strategy : GBTVariations()) {
    // Configure benchmark.
    Trainer trainer;
    trainer.train_config().set_learner(
        model::gradient_boosted_trees::GradientBoostedTreesLearner::
            kRegisteredName);

    trainer.train_config().set_task(model::proto::Task::REGRESSION);
    trainer.train_config().set_label("Rings");
    trainer.dataset_filename() = "abalone.csv";

    auto* gbt_config = trainer.train_config().MutableExtension(
        model::gradient_boosted_trees::proto::gradient_boosted_trees_config);
    strategy.second(gbt_config);

    // Run benchmark.
    RETURN_IF_ERROR(
        trainer.Run(absl::StrCat("GBT Abalone", strategy.first), results));
  }
  return absl::OkStatus();
}

absl::Status Benchmark_GBT_DNA(std::vector<Result>* results) {
  for (const auto& strategy : GBTVariations()) {
    // Configure benchmark.
    Trainer trainer;
    trainer.train_config().set_learner(
        model::gradient_boosted_trees::GradientBoostedTreesLearner::
            kRegisteredName);

    trainer.train_config().set_task(model::proto::Task::CLASSIFICATION);
    trainer.train_config().set_label("LABEL");
    trainer.dataset_filename() = "dna.csv";

    auto* gbt_config = trainer.train_config().MutableExtension(
        model::gradient_boosted_trees::proto::gradient_boosted_trees_config);
    strategy.second(gbt_config);

    // Run benchmark.
    RETURN_IF_ERROR(
        trainer.Run(absl::StrCat("GBT DNA", strategy.first), results));
  }
  return absl::OkStatus();
}

std::string ResultsToString(const std::vector<Result>& results) {
  std::string report;
  absl::StrAppendFormat(&report, "----------------------------------------\n");
  absl::StrAppendFormat(&report, " Training time           Benchmark      \n");
  absl::StrAppendFormat(&report, "----------------------------------------\n");
  for (const auto& result : results) {
    absl::StrAppendFormat(&report, "%10.5g  %s\n",
                          absl::ToDoubleSeconds(result.training_time),
                          result.name);
  }
  absl::StrAppendFormat(&report, "----------------------------------------\n");
  return report;
}

absl::Status ResultsToCsv(const std::vector<Result>& results,
                          const std::string& path) {
  ASSIGN_OR_RETURN(auto file_handle, file::OpenOutputFile(path));
  file::OutputFileCloser result_file_closer(std::move(file_handle));
  utils::csv::Writer result_writer(result_file_closer.stream());
  RETURN_IF_ERROR(result_writer.WriteRow({"benchmark", "training_time"}));
  for (const auto& result : results) {
    RETURN_IF_ERROR(result_writer.WriteRowStrings(
        {result.name,
         absl::StrFormat("%f", absl::ToDoubleSeconds(result.training_time))}));
  }

  return absl::OkStatus();
}

absl::Status Benchmark() {
  utils::usage::EnableUsage(false);

  std::vector<Result> results;

  RETURN_IF_ERROR(Benchmark_RandomForest_Adult(&results));
  RETURN_IF_ERROR(Benchmark_RandomForest_Adult_1Thread(&results));
  RETURN_IF_ERROR(Benchmark_RandomForest_Synthetic(&results));

  RETURN_IF_ERROR(Benchmark_GBT_Adult(&results));
  RETURN_IF_ERROR(Benchmark_GBT_Adult_1Thread(&results));
  RETURN_IF_ERROR(Benchmark_GBT_Synthetic(&results));

  // TODO: Enable by default when fast enough.
  // RETURN_IF_ERROR(Benchmark_GBT_SyntheticV2(&results));

  RETURN_IF_ERROR(Benchmark_GBT_Adult_NoEarlyStop(&results));
  RETURN_IF_ERROR(Benchmark_GBT_Adult_Hessian_NoEarlyStop(&results));
  RETURN_IF_ERROR(Benchmark_GBT_Adult_Discretized_NoEarlyStop(&results));

  RETURN_IF_ERROR(Benchmark_GBT_Adult_NoEarlyStop_Oblique(&results));
  RETURN_IF_ERROR(Benchmark_GBT_Synthetic_Oblique(&results));

  RETURN_IF_ERROR(Benchmark_GBT_Adult_Discretized(&results));
  RETURN_IF_ERROR(Benchmark_GBT_Adult_Hessian(&results));
  RETURN_IF_ERROR(Benchmark_GBT_Iris(&results));
  RETURN_IF_ERROR(Benchmark_GBT_Abalone(&results));
  RETURN_IF_ERROR(Benchmark_GBT_DNA(&results));

  std::cout << ResultsToString(results) << std::endl;

  if (!absl::GetFlag(FLAGS_output).empty()) {
    RETURN_IF_ERROR(ResultsToCsv(results, absl::GetFlag(FLAGS_output)));
  }
  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests

int main(int argc, char** argv) {
  std::cout << "Initialize logging";
  InitLogging(kUsageMessage, &argc, &argv, true);

  std::cout << "Start benchmark";
  const auto status = yggdrasil_decision_forests::Benchmark();
  if (!status.ok()) {
    YDF_LOG(INFO) << "The benchmark failed with the following error: "
                  << status;
    return 1;
  }
  return 0;
}
