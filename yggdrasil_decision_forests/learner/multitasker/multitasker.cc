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

#include "yggdrasil_decision_forests/learner/multitasker/multitasker.h"

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/learner/multitasker/multitasker.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/multitasker/multitasker.h"
#include "yggdrasil_decision_forests/serving/example_set.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/regex.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/synchronization_primitives.h"
#include "yggdrasil_decision_forests/utils/usage.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace multitasker {

constexpr char MultitaskerLearner::kRegisteredName[];

namespace {

// Extracts the examples of "src" with non-missing value for the column
// "label_idx" into "dst".
absl::Status ExtractExamplesWithLabels(const int label_idx,
                                       const dataset::VerticalDataset& src,
                                       dataset::VerticalDataset* dst) {
  *dst->mutable_data_spec() = src.data_spec();
  dst->mutable_data_spec()->mutable_columns(label_idx)->set_count_nas(0);
  RETURN_IF_ERROR(dst->CreateColumnsFromDataspec());
  for (dataset::VerticalDataset::row_t row_idx = 0; row_idx < src.nrow();
       row_idx++) {
    dataset::proto::Example example;
    src.ExtractExample(row_idx, &example);
    if (example.attributes(label_idx).type_case() !=
        dataset::proto::Example::Attribute::TYPE_NOT_SET) {
      dst->AppendExample(example);
    }
  }
  return absl::OkStatus();
}

std::string PredictionColNames(const model::AbstractModel& model,
                               const int prediction_idx) {
  return absl::StrCat(model.label(), ":", prediction_idx);
}

absl::Status AddPredictionToDataset(const dataset::VerticalDataset& src,
                                    const model::AbstractModel& model,
                                    dataset::VerticalDataset* dst,
                                    std::vector<std::string>* new_outputs) {
  ASSIGN_OR_RETURN(const auto engine, model.BuildFastEngine());
  auto examples = engine->AllocateExamples(1);
  const int num_predictions = engine->NumPredictionDimension();
  std::vector<std::vector<float>*> prediction_columns;
  for (int prediction_idx = 0; prediction_idx < num_predictions;
       prediction_idx++) {
    const auto new_name = PredictionColNames(model, prediction_idx);
    RETURN_IF_ERROR(
        dst->AddColumn(new_name, dataset::proto::ColumnType::NUMERICAL)
            .status());
    ASSIGN_OR_RETURN(auto* new_col,
                     dst->MutableColumnWithCastWithStatus<
                         dataset::VerticalDataset::NumericalColumn>(
                         dst->ColumnNameToColumnIdx(new_name)));
    prediction_columns.push_back(new_col->mutable_values());
    if (new_outputs) {
      new_outputs->push_back(new_name);
    }
  }
  std::vector<float> predictions;
  for (dataset::VerticalDataset::row_t example_idx = 0;
       example_idx < src.nrow(); example_idx++) {
    RETURN_IF_ERROR(serving::CopyVerticalDatasetToAbstractExampleSet(
        src, example_idx, example_idx + 1, engine->features(), examples.get()));
    engine->Predict(*examples, 1, &predictions);
    for (int pred_idx = 0; pred_idx < num_predictions; pred_idx++) {
      (*prediction_columns[pred_idx])[example_idx] = predictions[pred_idx];
    }
  }
  return absl::OkStatus();
}

std::string QuoteFeatureName(const std::string& feature_name) {
  return utils::QuoteRegex(feature_name, /*full_match=*/true);
}

}  // namespace

MultitaskerLearner::MultitaskerLearner(
    const model::proto::TrainingConfig& training_config)
    : AbstractLearner(training_config) {}

absl::StatusOr<std::unique_ptr<AbstractModel>>
MultitaskerLearner::TrainWithStatus(
    const dataset::VerticalDataset& train_dataset,
    absl::optional<std::reference_wrapper<const dataset::VerticalDataset>>
        valid_dataset) const {
  const auto& mt_config =
      training_config().GetExtension(proto::multitasker_config);
  if (mt_config.subtasks_size() == 0) {
    return absl::InvalidArgumentError("At least one task required");
  }

  auto model = absl::make_unique<MultitaskerModel>();
  model->set_data_spec(train_dataset.data_spec());
  STATUS_CHECK_LE(model->models_.size(), mt_config.subtasks_size());
  model->models_.resize(mt_config.subtasks_size());

  // Initialize the model. Use the first task for meta-data.
  model::proto::TrainingConfigLinking config_link;
  {
    ASSIGN_OR_RETURN(const auto first_config, BuildSubTrainingConfig(0));
    RETURN_IF_ERROR(AbstractLearner::LinkTrainingConfig(
        first_config, train_dataset.data_spec(), &config_link));
    InitializeModelWithAbstractTrainingConfig(training_config(), config_link,
                                              model.get());
  }

  // List the user specified input features.
  std::set<int> all_features;
  for (const int feature : config_link.features()) {
    all_features.insert(feature);
  }

  // Remove all the labels (and other special columns) from the set of the input
  // features.
  for (int task_idx = 0; task_idx < mt_config.subtasks_size(); task_idx++) {
    ASSIGN_OR_RETURN(const auto subtraining_config,
                     BuildSubTrainingConfig(task_idx));
    model::proto::TrainingConfigLinking subtraining_config_link;
    RETURN_IF_ERROR(AbstractLearner::LinkTrainingConfig(
        subtraining_config, train_dataset.data_spec(),
        &subtraining_config_link));
    if (subtraining_config_link.has_label()) {
      all_features.erase(subtraining_config_link.label());
    }
    if (subtraining_config_link.cv_group() >= 0) {
      all_features.erase(subtraining_config_link.cv_group());
    }
    if (subtraining_config_link.ranking_group() >= 0) {
      all_features.erase(subtraining_config_link.ranking_group());
    }
  }

  utils::concurrency::Mutex mutex;
  absl::Status status;

  // Index the primary and secondary tasks.
  std::vector<int> primary_task_idxs;
  std::vector<int> secondary_task_idxs;
  for (int task_idx = 0; task_idx < mt_config.subtasks_size(); task_idx++) {
    const auto& task = mt_config.subtasks(task_idx);
    if (task.primary()) {
      primary_task_idxs.push_back(task_idx);
    } else {
      secondary_task_idxs.push_back(task_idx);
    }
  }

  // Placeholder for primary datasets.
  dataset::VerticalDataset primary_train_dataset =
      train_dataset.ShallowNonOwningClone();
  dataset::VerticalDataset primary_valid_dataset;
  if (valid_dataset.has_value()) {
    primary_valid_dataset = valid_dataset.value().get().ShallowNonOwningClone();
  }
  std::vector<std::string> secondary_model_output;

  const auto train_subtask = [&](const int subtask_idx,
                                 const bool primary) -> absl::Status {
    STATUS_CHECK_GE(subtask_idx, 0);
    ASSIGN_OR_RETURN(auto sublearner, BuildSubLearner(subtask_idx));

    // Add the input features.
    sublearner->mutable_training_config()->clear_features();
    for (const int input_feature : all_features) {
      sublearner->mutable_training_config()->add_features(QuoteFeatureName(
          train_dataset.data_spec().columns()[input_feature].name()));
    }

    std::unique_ptr<AbstractModel> submodel;
    if (primary) {
      // Add the output of the secondary models as input features.
      for (const auto& new_feature : secondary_model_output) {
        sublearner->mutable_training_config()->add_features(
            QuoteFeatureName(new_feature));
      }
    }

    int32_t label_idx;
    RETURN_IF_ERROR(dataset::GetSingleColumnIdxFromName(
        sublearner->training_config().label(),
        primary_train_dataset.data_spec(), &label_idx));

    // Extract the training / validation dataset
    dataset::VerticalDataset local_train_ds;
    RETURN_IF_ERROR(ExtractExamplesWithLabels(label_idx, primary_train_dataset,
                                              &local_train_ds));
    absl::optional<dataset::VerticalDataset> local_valid_ds;
    if (valid_dataset.has_value()) {
      local_valid_ds = dataset::VerticalDataset();
      RETURN_IF_ERROR(ExtractExamplesWithLabels(
          label_idx, primary_valid_dataset, &local_valid_ds.value()));
    }

    // Train
    ASSIGN_OR_RETURN(
        submodel, sublearner->TrainWithStatus(local_train_ds, local_valid_ds));

    utils::concurrency::MutexLock lock(&mutex);
    STATUS_CHECK_LT(subtask_idx, model->models_.size());
    model->models_[subtask_idx] = std::move(submodel);
    return absl::OkStatus();
  };

  const auto train_subtask_nostatus = [&](const int subtask_idx,
                                          const bool primary) {
    {
      utils::concurrency::MutexLock lock(&mutex);
      if (!status.ok()) {
        return;
      }
    }
    const auto substatus = train_subtask(subtask_idx, primary);
    utils::concurrency::MutexLock lock(&mutex);
    if (!substatus.ok()) {
      status.Update(substatus);
    }
  };

  if (!secondary_task_idxs.empty()) {
    YDF_LOG(INFO) << "Train multitasker secondary tasks with "
                  << secondary_task_idxs.size() << " model(s)";

    {
      utils::concurrency::ThreadPool pool("multitasker",
                                          deployment().num_threads());
      pool.StartWorkers();
      for (const auto subtask_idx : secondary_task_idxs) {
        pool.Schedule([train_subtask_nostatus, subtask_idx]() {
          train_subtask_nostatus(subtask_idx, /*primary=*/false);
        });
      }
    }
    RETURN_IF_ERROR(status);

    // Use the secondary models to generated the training dataset for the
    // primary tasks.
    YDF_LOG(INFO) << "Generate signals for primary tasks";
    for (const auto subtask_idx : secondary_task_idxs) {
      RETURN_IF_ERROR(AddPredictionToDataset(
          train_dataset, *model->models_[subtask_idx], &primary_train_dataset,
          &secondary_model_output));
      if (valid_dataset.has_value()) {
        RETURN_IF_ERROR(AddPredictionToDataset(
            valid_dataset.value(), *model->models_[subtask_idx],
            &primary_valid_dataset, nullptr));
      }
    }
    YDF_LOG(INFO) << "New features: "
                  << absl::StrJoin(secondary_model_output, " ");
  }

  if (!primary_task_idxs.empty()) {
    YDF_LOG(INFO) << "Train multitasker primary tasks with "
                  << primary_task_idxs.size() << " model(s)";
    utils::concurrency::ThreadPool pool("multitasker",
                                        deployment().num_threads());
    pool.StartWorkers();
    for (const auto subtask_idx : primary_task_idxs) {
      pool.Schedule([train_subtask_nostatus, subtask_idx]() {
        train_subtask_nostatus(subtask_idx, /*primary=*/true);
      });
    }
  }
  RETURN_IF_ERROR(status);

  std::set<int> all_input_features;
  const auto& feature_source =
      secondary_task_idxs.empty() ? primary_task_idxs : secondary_task_idxs;
  for (const auto task_idx : feature_source) {
    const auto& sub_model = model->models_[task_idx];
    all_input_features.insert(sub_model->input_features().begin(),
                              sub_model->input_features().end());
  }

  *model->mutable_input_features() = {all_input_features.begin(),
                                      all_input_features.end()};
  std::sort(model->mutable_input_features()->begin(),
            model->mutable_input_features()->end());

  return model;
}

absl::Status MultitaskerLearner::SetHyperParameters(
    const model::proto::GenericHyperParameters& generic_hyper_params) {
  generic_hyper_params_ = generic_hyper_params;
  return absl::OkStatus();
}

absl::StatusOr<model::proto::TrainingConfig>
MultitaskerLearner::BuildSubTrainingConfig(const int learner_idx) const {
  const auto& mt_config =
      training_config().GetExtension(proto::multitasker_config);

  model::proto::TrainingConfig sub_learner_config = mt_config.base_learner();
  RETURN_IF_ERROR(CopyProblemDefinition(training_config_, &sub_learner_config));

  if (learner_idx >= mt_config.subtasks_size()) {
    return absl::InvalidArgumentError("Invalid learner idx");
  }
  sub_learner_config.MergeFrom(mt_config.subtasks(learner_idx).train_config());

  if (training_config().has_maximum_training_duration_seconds() &&
      !sub_learner_config.has_maximum_training_duration_seconds()) {
    sub_learner_config.set_maximum_training_duration_seconds(
        training_config().maximum_training_duration_seconds());
  }

  if (training_config().has_maximum_model_size_in_memory_in_bytes() &&
      !sub_learner_config.has_maximum_model_size_in_memory_in_bytes()) {
    sub_learner_config.set_maximum_model_size_in_memory_in_bytes(
        training_config().maximum_model_size_in_memory_in_bytes());
  }

  return sub_learner_config;
}

absl::StatusOr<std::unique_ptr<AbstractLearner>>
MultitaskerLearner::BuildSubLearner(const int learner_idx) const {
  const auto& mt_config =
      training_config().GetExtension(proto::multitasker_config);

  ASSIGN_OR_RETURN(const auto sub_learner_config,
                   BuildSubTrainingConfig(learner_idx));

  // Build sub-learner
  std::unique_ptr<AbstractLearner> sub_learner;
  RETURN_IF_ERROR(GetLearner(sub_learner_config, &sub_learner));
  *sub_learner->mutable_deployment() = mt_config.base_learner_deployment();
  RETURN_IF_ERROR(sub_learner->SetHyperParameters(generic_hyper_params_));
  return sub_learner;
}

absl::StatusOr<model::proto::GenericHyperParameterSpecification>
MultitaskerLearner::GetGenericHyperParameterSpecification() const {
  const auto& mt_config =
      training_config().GetExtension(proto::multitasker_config);
  if (mt_config.subtasks_size() == 0) {
    YDF_LOG(WARNING) << "Sub-learner not set. This is only expected during the "
                        "automatic documentation generation.";
    return AbstractLearner::GetGenericHyperParameterSpecification();
  }

  ASSIGN_OR_RETURN(auto sub_learner, BuildSubLearner(0));
  return sub_learner->GetGenericHyperParameterSpecification();
}

}  // namespace multitasker
}  // namespace model
}  // namespace yggdrasil_decision_forests
