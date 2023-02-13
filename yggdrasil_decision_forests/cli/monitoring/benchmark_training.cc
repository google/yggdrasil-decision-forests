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

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/learner/random_forest/random_forest.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/utils/csv.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/usage.h"

constexpr char kUsageMessage[] =
    "Benchmarks the training time of various learners.";

ABSL_FLAG(std::string, output, "",
          "Output csv file with results. If not specified, results are only "
          "printed in the std out.");

namespace yggdrasil_decision_forests {

struct Result {
  std::string name;
  absl::Duration training_time;
};

struct SyntheticParams {
  int num_examples;
  int num_numerical_features;
  int num_classes;

  std::string to_string() const {
    return absl::StrCat("e:", num_examples, " f:", num_numerical_features,
                        " c:", num_classes);
  }
};

// Utility class for the training of a model.
class Trainer {
 public:
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
    RETURN_IF_ERROR(learner_->TrainWithStatus(train_dataset_).status());
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
    feature_names.reserve(option.num_numerical_features);
    for (int feature_idx = 0; feature_idx < option.num_numerical_features;
         feature_idx++) {
      const auto feature_name = absl::StrCat("x", feature_idx);
      feature_names.push_back(feature_name);
      dataset::AddColumn(feature_name, dataset::proto::ColumnType::NUMERICAL,
                         train_dataset_.mutable_data_spec());
    }
    RETURN_IF_ERROR(train_dataset_.CreateColumnsFromDataspec());

    // Link to dataset data.
    std::vector<dataset::VerticalDataset::NumericalColumn*> numerical_features;
    numerical_features.reserve(feature_names.size());
    for (const auto& feature_name : feature_names) {
      ASSIGN_OR_RETURN(const auto mutable_col,
                       train_dataset_.MutableColumnWithCastWithStatus<
                           dataset::VerticalDataset::NumericalColumn>(
                           train_dataset_.ColumnNameToColumnIdx(feature_name)));
      numerical_features.push_back(mutable_col);
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
        const float value = unif_01(random);
        accumulator += value;
        numerical_features[feature_idx]->Add(value);
      }
      const int label_value = 1 + std::floor(accumulator * option.num_classes /
                                             feature_names.size());
      label_data->Add(label_value);
    }
    train_dataset_.set_nrow(option.num_examples);

    // Config trainer.
    train_config()->set_task(model::proto::Task::CLASSIFICATION);
    train_config()->set_label(label);

    return absl::OkStatus();
  }

  // Accessors to the configuration fields. Should be set before calling
  // "PrepareTraining".
  model::proto::TrainingConfig* train_config() { return &train_config_; }
  model::proto::DeploymentConfig* deployment_config() {
    return &deployment_config_;
  }
  std::string* mutable_dataset_directory() { return &dataset_directory_; }
  std::string* dataset_filename() { return &dataset_filename_; }
  std::string* dataset_format() { return &dataset_format_; }
  std::string* dataspec_guide_filename() { return &dataspec_guide_filename_; }
  dataset::proto::DataSpecificationGuide* guide() { return &guide_; }

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

absl::Status Benchmark_RandomForest_Adult(std::vector<Result>* results) {
  // Configure benchmark.
  Trainer trainer;
  trainer.train_config()->set_learner(
      model::random_forest::RandomForestLearner::kRegisteredName);

  trainer.train_config()->set_task(model::proto::Task::CLASSIFICATION);
  trainer.train_config()->set_label("income");
  *trainer.dataset_filename() = "adult.csv";

  // Run benchmark.
  return trainer.Run("RF Adult", results);
}

absl::Status Benchmark_RandomForest_Adult_1Thread(
    std::vector<Result>* results) {
  // Configure benchmark.
  Trainer trainer;
  trainer.train_config()->set_learner(
      model::random_forest::RandomForestLearner::kRegisteredName);
  trainer.deployment_config()->set_num_threads(1);

  trainer.train_config()->set_task(model::proto::Task::CLASSIFICATION);
  trainer.train_config()->set_label("income");
  *trainer.dataset_filename() = "adult.csv";

  // Run benchmark.
  return trainer.Run("RF Adult 1-thread", results);
}

absl::Status Benchmark_RandomForest_Synthetic(std::vector<Result>* results) {
  for (const auto& synthetic : std::vector<SyntheticParams>{
           {/*num_examples=*/100'000, /*num_features=*/10, /*num_classes=*/2},
           {/*num_examples=*/100'000, /*num_features=*/50, /*num_classes=*/2},
           {/*num_examples=*/100'000, /*num_features=*/10, /*num_classes=*/10},
           {/*num_examples=*/1'000'000, /*num_features=*/5, /*num_classes=*/2},
       }) {
    // Configure benchmark.
    Trainer trainer;
    trainer.train_config()->set_learner(
        model::random_forest::RandomForestLearner::kRegisteredName);
    auto* rf_config = trainer.train_config()->MutableExtension(
        model::random_forest::proto::random_forest_config);
    rf_config->set_num_trees(10);
    RETURN_IF_ERROR(trainer.GenSyntheticDataset(synthetic));

    // Run benchmark.
    RETURN_IF_ERROR(trainer.Run(
        absl::StrCat("RF Synthetic t:10 ", synthetic.to_string()), results));
  }
  return absl::OkStatus();
}

absl::Status Benchmark_GBT_Adult(std::vector<Result>* results) {
  // Configure benchmark.
  Trainer trainer;
  trainer.train_config()->set_learner(
      model::gradient_boosted_trees::GradientBoostedTreesLearner::
          kRegisteredName);

  trainer.train_config()->set_task(model::proto::Task::CLASSIFICATION);
  trainer.train_config()->set_label("income");
  *trainer.dataset_filename() = "adult.csv";

  // Run benchmark.
  return trainer.Run("GBT Adult", results);
}

absl::Status Benchmark_GBT_Adult_NoEarlyStop(std::vector<Result>* results) {
  // Configure benchmark.
  Trainer trainer;
  trainer.train_config()->set_learner(
      model::gradient_boosted_trees::GradientBoostedTreesLearner::
          kRegisteredName);
  auto* gbt_config = trainer.train_config()->MutableExtension(
      model::gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_early_stopping(model::gradient_boosted_trees::proto::
                                     GradientBoostedTreesTrainingConfig::NONE);

  trainer.train_config()->set_task(model::proto::Task::CLASSIFICATION);
  trainer.train_config()->set_label("income");
  *trainer.dataset_filename() = "adult.csv";

  // Run benchmark.
  return trainer.Run("GBT Adult NoEarlyStop", results);
}

absl::Status Benchmark_GBT_Adult_NoEarlyStop_Oblique(
    std::vector<Result>* results) {
  // Configure benchmark.
  Trainer trainer;
  trainer.train_config()->set_learner(
      model::gradient_boosted_trees::GradientBoostedTreesLearner::
          kRegisteredName);
  auto* gbt_config = trainer.train_config()->MutableExtension(
      model::gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_early_stopping(model::gradient_boosted_trees::proto::
                                     GradientBoostedTreesTrainingConfig::NONE);
  gbt_config->mutable_decision_tree()->mutable_sparse_oblique_split();

  trainer.train_config()->set_task(model::proto::Task::CLASSIFICATION);
  trainer.train_config()->set_label("income");
  *trainer.dataset_filename() = "adult.csv";

  // Run benchmark.
  return trainer.Run("GBT Adult NoEarlyStop Oblique", results);
}

absl::Status Benchmark_GBT_Adult_Hessian(std::vector<Result>* results) {
  // Configure benchmark.
  Trainer trainer;
  trainer.train_config()->set_learner(
      model::gradient_boosted_trees::GradientBoostedTreesLearner::
          kRegisteredName);
  auto* gbt_config = trainer.train_config()->MutableExtension(
      model::gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_use_hessian_gain(true);

  trainer.train_config()->set_task(model::proto::Task::CLASSIFICATION);
  trainer.train_config()->set_label("income");
  *trainer.dataset_filename() = "adult.csv";

  // Run benchmark.
  return trainer.Run("GBT Adult Hessian", results);
}

absl::Status Benchmark_GBT_Adult_Hessian_NoEarlyStop(
    std::vector<Result>* results) {
  // Configure benchmark.
  Trainer trainer;
  trainer.train_config()->set_learner(
      model::gradient_boosted_trees::GradientBoostedTreesLearner::
          kRegisteredName);
  auto* gbt_config = trainer.train_config()->MutableExtension(
      model::gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_use_hessian_gain(true);
  gbt_config->set_early_stopping(model::gradient_boosted_trees::proto::
                                     GradientBoostedTreesTrainingConfig::NONE);

  trainer.train_config()->set_task(model::proto::Task::CLASSIFICATION);
  trainer.train_config()->set_label("income");
  *trainer.dataset_filename() = "adult.csv";

  // Run benchmark.
  return trainer.Run("GBT Adult Hessian NoEarlyStop", results);
}

absl::Status Benchmark_GBT_Adult_Discretized(std::vector<Result>* results) {
  // Configure benchmark.
  Trainer trainer;
  trainer.train_config()->set_learner(
      model::gradient_boosted_trees::GradientBoostedTreesLearner::
          kRegisteredName);
  trainer.guide()->set_detect_numerical_as_discretized_numerical(true);
  trainer.train_config()->set_task(model::proto::Task::CLASSIFICATION);
  trainer.train_config()->set_label("income");
  *trainer.dataset_filename() = "adult.csv";

  // Run benchmark.
  return trainer.Run("GBT Adult Discretized", results);
}

absl::Status Benchmark_GBT_Adult_Discretized_NoEarlyStop(
    std::vector<Result>* results) {
  // Configure benchmark.
  Trainer trainer;
  trainer.train_config()->set_learner(
      model::gradient_boosted_trees::GradientBoostedTreesLearner::
          kRegisteredName);
  auto* gbt_config = trainer.train_config()->MutableExtension(
      model::gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_early_stopping(model::gradient_boosted_trees::proto::
                                     GradientBoostedTreesTrainingConfig::NONE);

  trainer.guide()->set_detect_numerical_as_discretized_numerical(true);
  trainer.train_config()->set_task(model::proto::Task::CLASSIFICATION);
  trainer.train_config()->set_label("income");
  *trainer.dataset_filename() = "adult.csv";

  // Run benchmark.
  return trainer.Run("GBT Adult Discretized NoEarlyStop", results);
}

absl::Status Benchmark_GBT_Adult_1Thread(std::vector<Result>* results) {
  // Configure benchmark.
  Trainer trainer;
  trainer.train_config()->set_learner(
      model::gradient_boosted_trees::GradientBoostedTreesLearner::
          kRegisteredName);
  trainer.deployment_config()->set_num_threads(1);

  trainer.train_config()->set_task(model::proto::Task::CLASSIFICATION);
  trainer.train_config()->set_label("income");
  *trainer.dataset_filename() = "adult.csv";

  // Run benchmark.
  return trainer.Run("GBT Adult 1-thread", results);
}

absl::Status Benchmark_GBT_Synthetic(std::vector<Result>* results) {
  for (const auto& synthetic : std::vector<SyntheticParams>{
           {/*num_examples=*/100'000, /*num_features=*/10, /*num_classes=*/2},
           {/*num_examples=*/100'000, /*num_features=*/50, /*num_classes=*/2},
           {/*num_examples=*/100'000, /*num_features=*/10, /*num_classes=*/10},
           {/*num_examples=*/1'000'000, /*num_features=*/5, /*num_classes=*/2},
           {/*num_examples=*/1'000'000, /*num_features=*/50, /*num_classes=*/2},
       }) {
    // Configure benchmark.
    Trainer trainer;
    trainer.train_config()->set_learner(
        model::gradient_boosted_trees::GradientBoostedTreesLearner::
            kRegisteredName);
    auto* gbt_config = trainer.train_config()->MutableExtension(
        model::gradient_boosted_trees::proto::gradient_boosted_trees_config);
    gbt_config->set_num_trees(20);
    gbt_config->set_early_stopping(
        model::gradient_boosted_trees::proto::
            GradientBoostedTreesTrainingConfig::NONE);

    RETURN_IF_ERROR(trainer.GenSyntheticDataset(synthetic));

    // Run benchmark.
    RETURN_IF_ERROR(trainer.Run(
        absl::StrCat("GBT Synthetic t:20 ", synthetic.to_string()), results));
  }
  return absl::OkStatus();
}

absl::Status Benchmark_GBT_Synthetic_Oblique(std::vector<Result>* results) {
  for (const auto& synthetic : std::vector<SyntheticParams>{
           {/*num_examples=*/10'000, /*num_features=*/20, /*num_classes=*/2},
           {/*num_examples=*/100'000, /*num_features=*/20, /*num_classes=*/2},
       }) {
    // Configure benchmark.
    Trainer trainer;
    trainer.train_config()->set_learner(
        model::gradient_boosted_trees::GradientBoostedTreesLearner::
            kRegisteredName);
    auto* gbt_config = trainer.train_config()->MutableExtension(
        model::gradient_boosted_trees::proto::gradient_boosted_trees_config);
    gbt_config->set_num_trees(10);
    gbt_config->set_early_stopping(
        model::gradient_boosted_trees::proto::
            GradientBoostedTreesTrainingConfig::NONE);
    gbt_config->mutable_decision_tree()->mutable_sparse_oblique_split();

    RETURN_IF_ERROR(trainer.GenSyntheticDataset(synthetic));

    // Run benchmark.
    RETURN_IF_ERROR(trainer.Run(
        absl::StrCat("GBT Synthetic Oblique t:10 ", synthetic.to_string()),
        results));
  }
  return absl::OkStatus();
}

absl::Status Benchmark_GBT_Iris(std::vector<Result>* results) {
  // Configure benchmark.
  Trainer trainer;
  trainer.train_config()->set_learner(
      model::gradient_boosted_trees::GradientBoostedTreesLearner::
          kRegisteredName);

  trainer.train_config()->set_task(model::proto::Task::CLASSIFICATION);
  trainer.train_config()->set_label("class");
  *trainer.dataset_filename() = "iris.csv";

  // Run benchmark.
  return trainer.Run("GBT Iris", results);
}

absl::Status Benchmark_GBT_Abalone(std::vector<Result>* results) {
  // Configure benchmark.
  Trainer trainer;
  trainer.train_config()->set_learner(
      model::gradient_boosted_trees::GradientBoostedTreesLearner::
          kRegisteredName);

  trainer.train_config()->set_task(model::proto::Task::REGRESSION);
  trainer.train_config()->set_label("Rings");
  *trainer.dataset_filename() = "abalone.csv";

  // Run benchmark.
  return trainer.Run("GBT Abalone", results);
}

absl::Status Benchmark_GBT_DNA(std::vector<Result>* results) {
  // Configure benchmark.
  Trainer trainer;
  trainer.train_config()->set_learner(
      model::gradient_boosted_trees::GradientBoostedTreesLearner::
          kRegisteredName);

  trainer.train_config()->set_task(model::proto::Task::CLASSIFICATION);
  trainer.train_config()->set_label("LABEL");
  *trainer.dataset_filename() = "dna.csv";

  // Run benchmark.
  return trainer.Run("GBT DNA", results);
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
