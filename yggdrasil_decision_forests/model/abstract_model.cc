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

#include "yggdrasil_decision_forests/model/abstract_model.h"

#include <stddef.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/weight.h"
#include "yggdrasil_decision_forests/dataset/weight.pb.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/metric/report.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/fast_engine_factory.h"
#include "yggdrasil_decision_forests/model/hyperparameter.pb.h"
#include "yggdrasil_decision_forests/model/prediction.pb.h"
#include "yggdrasil_decision_forests/serving/example_set.h"
#include "yggdrasil_decision_forests/serving/fast_engine.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/distribution.h"
#include "yggdrasil_decision_forests/utils/distribution.pb.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/protobuf.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace model {

namespace {

absl::Status CheckCompatibleEvaluationTask(const proto::Task model_task,
                                           const proto::Task evaluation_task) {
  if (model_task == proto::ANOMALY_DETECTION) {
    if (evaluation_task == proto::ANOMALY_DETECTION) {
      // We may revisit this decision in the future, so disallowing it for now
      // with a clear error message instead of re-routing to classification.
      return absl::InvalidArgumentError(
          "Anomaly detection models don't have direct evaluation. Instead, "
          "evaluate the model as a classification model e.g. "
          "`anomaly_detection_model.evaluate(task=ydf.Task.CLASSIFICATION)");
    } else if (evaluation_task == proto::CLASSIFICATION) {
      return absl::OkStatus();
    }
  }
  if (model_task == evaluation_task) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(absl::Substitute(
      "Model task $0 and evaluation task $1 are incompatible",
      proto::Task_Name(model_task), proto::Task_Name(evaluation_task)));
}

}  // namespace

void AbstractModel::ExportProto(const AbstractModel& model,
                                proto::AbstractModel* proto) {
  proto->set_name(model.name_);
  proto->set_task(model.task_);
  proto->set_label_col_idx(model.label_col_idx_);
  proto->set_ranking_group_col_idx(model.ranking_group_col_idx_);
  proto->set_uplift_treatment_col_idx(model.uplift_treatment_col_idx_);
  proto->set_label_entry_age_col_idx(model.label_entry_age_col_idx_);
  proto->set_label_event_observed_col_idx(model.label_event_observed_col_idx_);
  proto->set_is_pure_model(model.is_pure_model_);
  *proto->mutable_input_features() = {model.input_features_.begin(),
                                      model.input_features_.end()};
  if (model.weights_.has_value()) {
    *proto->mutable_weights() = model.weights_.value();
  }
  proto->mutable_precomputed_variable_importances()->insert(
      model.precomputed_variable_importances_.begin(),
      model.precomputed_variable_importances_.end());
  proto->set_classification_outputs_probabilities(
      model.classification_outputs_probabilities_);

  model.metadata().Export(proto->mutable_metadata());

  if (model.hyperparameter_optimizer_logs_.has_value()) {
    *proto->mutable_hyperparameter_optimizer_logs() =
        model.hyperparameter_optimizer_logs_.value();
  }

  if (model.feature_selection_logs_.has_value()) {
    *proto->mutable_feature_selection_logs() =
        model.feature_selection_logs_.value();
  }
}

void AbstractModel::ImportProto(const proto::AbstractModel& proto,
                                AbstractModel* model) {
  model->name_ = proto.name();
  model->task_ = proto.task();
  model->label_col_idx_ = proto.label_col_idx();
  model->ranking_group_col_idx_ = proto.ranking_group_col_idx();
  model->uplift_treatment_col_idx_ = proto.uplift_treatment_col_idx();
  model->label_entry_age_col_idx_ = proto.label_entry_age_col_idx();
  model->label_event_observed_col_idx_ = proto.label_event_observed_col_idx();
  model->is_pure_model_ = proto.is_pure_model();
  model->input_features_.assign(proto.input_features().begin(),
                                proto.input_features().end());
  if (proto.has_weights()) {
    model->weights_ = proto.weights();
  }
  model->precomputed_variable_importances_.insert(
      proto.precomputed_variable_importances().begin(),
      proto.precomputed_variable_importances().end());
  model->classification_outputs_probabilities_ =
      proto.classification_outputs_probabilities();

  model->mutable_metadata()->Import(proto.metadata());

  if (proto.has_hyperparameter_optimizer_logs()) {
    model->hyperparameter_optimizer_logs_ =
        proto.hyperparameter_optimizer_logs();
  }
  if (proto.has_feature_selection_logs()) {
    model->feature_selection_logs_ = proto.feature_selection_logs();
  }
}

metric::proto::EvaluationResults AbstractModel::Evaluate(
    const dataset::VerticalDataset& dataset,
    const metric::proto::EvaluationOptions& option, utils::RandomEngine* rnd,
    std::vector<model::proto::Prediction>* predictions) const {
  // TODO: Fix.
  return EvaluateWithStatus(dataset, option, rnd, predictions).value();
}

absl::StatusOr<metric::proto::EvaluationResults>
AbstractModel::EvaluateWithStatus(
    const dataset::VerticalDataset& dataset,
    const metric::proto::EvaluationOptions& option, utils::RandomEngine* rnd,
    std::vector<model::proto::Prediction>* predictions) const {
  // TODO: Consider moving the checks into InitializeEvaluation.
  RETURN_IF_ERROR(CheckCompatibleEvaluationTask(task(), option.task()));
  if (label_col_idx_ == -1) {
    if (task() == proto::Task::ANOMALY_DETECTION) {
      STATUS_FATAL(
          "Cannot evaluate an anomaly detection model without a label.");
    } else {
      STATUS_FATAL("A model cannot be evaluated without a label.");
    }
  }
  metric::proto::EvaluationResults eval;
  RETURN_IF_ERROR(
      metric::InitializeEvaluation(option, LabelColumnSpec(), &eval));
  RETURN_IF_ERROR(AppendEvaluation(dataset, option, rnd, &eval, predictions));
  RETURN_IF_ERROR(metric::FinalizeEvaluation(option, LabelColumnSpec(), &eval));
  return eval;
}

absl::StatusOr<metric::proto::EvaluationResults>
AbstractModel::EvaluateWithEngine(
    const serving::FastEngine& engine, const dataset::VerticalDataset& dataset,
    const metric::proto::EvaluationOptions& option, utils::RandomEngine* rnd,
    std::vector<model::proto::Prediction>* predictions) const {
  RETURN_IF_ERROR(CheckCompatibleEvaluationTask(task(), option.task()));
  if (label_col_idx_ == -1) {
    STATUS_FATAL("A model cannot be evaluated without a label.");
  }
  metric::proto::EvaluationResults eval;
  RETURN_IF_ERROR(
      metric::InitializeEvaluation(option, LabelColumnSpec(), &eval));

  dataset::proto::LinkedWeightDefinition weight_links;
  if (option.has_weights()) {
    RETURN_IF_ERROR(dataset::GetLinkedWeightDefinition(
        option.weights(), data_spec_, &weight_links));
  }
  if (dataset.nrow() == 0) {
    STATUS_FATAL("The dataset is empty. Cannot evaluate model.");
  }

  RETURN_IF_ERROR(AppendEvaluationWithEngine(dataset, option, weight_links,
                                             engine, rnd, predictions, &eval));

  eval.set_num_folds(eval.num_folds() + 1);

  RETURN_IF_ERROR(metric::FinalizeEvaluation(option, LabelColumnSpec(), &eval));
  return eval;
}

absl::StatusOr<metric::proto::EvaluationResults>
AbstractModel::EvaluateOverrideType(
    const dataset::VerticalDataset& dataset,
    const metric::proto::EvaluationOptions& option,
    const proto::Task override_task, const int override_label_col_idx,
    const int override_group_col_idx, utils::RandomEngine* rnd,
    std::vector<model::proto::Prediction>* predictions) const {
  RETURN_IF_ERROR(CheckCompatibleEvaluationTask(override_task, option.task()));
  metric::proto::EvaluationResults eval;
  const auto& label_col_spec =
      dataset.data_spec().columns(override_label_col_idx);
  RETURN_IF_ERROR(metric::InitializeEvaluation(option, label_col_spec, &eval));
  RETURN_IF_ERROR(AppendEvaluationOverrideType(
      dataset, option, override_task, override_label_col_idx,
      override_group_col_idx, rnd, &eval, predictions));
  RETURN_IF_ERROR(metric::FinalizeEvaluation(option, label_col_spec, &eval));
  return eval;
}

absl::StatusOr<metric::proto::EvaluationResults>
AbstractModel::EvaluateWithEngineOverrideType(
    const serving::FastEngine& engine, const dataset::VerticalDataset& dataset,
    const metric::proto::EvaluationOptions& option,
    const proto::Task override_task, const int override_label_col_idx,
    const int override_group_col_idx, utils::RandomEngine* rnd,
    std::vector<model::proto::Prediction>* predictions) const {
  RETURN_IF_ERROR(CheckCompatibleEvaluationTask(override_task, option.task()));
  if (label_col_idx_ == -1) {
    STATUS_FATAL("A model cannot be evaluated without a label.");
  }
  const auto& label_col_spec =
      dataset.data_spec().columns(override_label_col_idx);
  metric::proto::EvaluationResults eval;
  RETURN_IF_ERROR(metric::InitializeEvaluation(option, label_col_spec, &eval));
  dataset::proto::LinkedWeightDefinition weight_links;
  if (option.has_weights()) {
    RETURN_IF_ERROR(dataset::GetLinkedWeightDefinition(
        option.weights(), dataset.data_spec(), &weight_links));
  }
  if (dataset.nrow() == 0) {
    STATUS_FATAL("The dataset is empty. Cannot evaluate model.");
  }
  RETURN_IF_ERROR(AppendEvaluationWithEngineOverrideType(
      dataset, option, override_task, override_label_col_idx,
      override_group_col_idx, weight_links, engine, rnd, predictions, &eval));
  RETURN_IF_ERROR(metric::FinalizeEvaluation(option, label_col_spec, &eval));
  return eval;
}

absl::Status AbstractModel::AppendPredictions(
    const dataset::VerticalDataset& dataset, const bool add_ground_truth,
    std::vector<model::proto::Prediction>* predictions) const {
  DCHECK(predictions);

  predictions->reserve(predictions->size() + dataset.nrow());
  auto engine_or_status = BuildFastEngine();
  if (engine_or_status.ok()) {
    const auto& engine = engine_or_status.value();
    // Evaluate using the semi-fast generic engine.

    const auto& engine_features = engine->features();
    const int num_prediction_dimensions = engine->NumPredictionDimension();

    const int64_t total_num_examples = dataset.nrow();
    constexpr int64_t kMaxBatchSize = 100;
    const int64_t batch_size = std::min(kMaxBatchSize, total_num_examples);

    auto batch_of_examples = engine->AllocateExamples(batch_size);
    const int64_t num_batches =
        (total_num_examples + batch_size - 1) / batch_size;

    std::vector<float> batch_of_predictions;
    proto::Prediction prediction;
    for (int64_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
      const int64_t begin_example_idx = batch_idx * batch_size;
      const int64_t end_example_idx =
          std::min(begin_example_idx + batch_size, total_num_examples);
      const int effective_batch_size = end_example_idx - begin_example_idx;
      RETURN_IF_ERROR(CopyVerticalDatasetToAbstractExampleSet(
          dataset, begin_example_idx, end_example_idx, engine_features,
          batch_of_examples.get()));
      engine->Predict(*batch_of_examples, effective_batch_size,
                      &batch_of_predictions);
      for (int sub_example_idx = 0; sub_example_idx < effective_batch_size;
           sub_example_idx++) {
        FloatToProtoPrediction(batch_of_predictions, sub_example_idx, task(),
                               num_prediction_dimensions, &prediction);
        RETURN_IF_ERROR(SetGroundTruth(
            dataset, begin_example_idx + sub_example_idx, &prediction));
        if (predictions) {
          predictions->push_back(prediction);
        }
      }
    }

  } else {
    // Evaluate using the (slow) generic inference.
    proto::Prediction prediction;
    for (dataset::VerticalDataset::row_t test_row_idx = 0;
         test_row_idx < dataset.nrow(); test_row_idx++) {
      LOG_EVERY_N_SEC(INFO, 30) << (test_row_idx + 1) << "/" << dataset.nrow()
                                << " predictions generated.";
      Predict(dataset, test_row_idx, &prediction);
      if (add_ground_truth) {
        RETURN_IF_ERROR(SetGroundTruth(dataset, test_row_idx, &prediction));
      }
      predictions->push_back(prediction);
    }
  }

  return absl::OkStatus();
}

void FloatToProtoPrediction(const std::vector<float>& src_prediction,
                            const int example_idx, const proto::Task task,
                            const int num_prediction_dimensions,
                            proto::Prediction* dst_prediction) {
  switch (task) {
    case proto::UNDEFINED:
      LOG(WARNING) << "Undefined task";
      break;
    case proto::CLASSIFICATION: {
      auto* classification = dst_prediction->mutable_classification();
      auto* distribution = classification->mutable_distribution();
      if (num_prediction_dimensions == 1) {
        const float proba_class2 = src_prediction[example_idx];
        classification->set_value((proba_class2 > 0.5f) ? 2 : 1);
        distribution->mutable_counts()->Resize(3, 0.f);
        distribution->mutable_counts()->Set(0, 0.f);
        distribution->mutable_counts()->Set(1, 1.f - proba_class2);
        distribution->mutable_counts()->Set(2, proba_class2);
        distribution->set_sum(1.f);
      } else {
        distribution->mutable_counts()->Resize(num_prediction_dimensions + 1,
                                               0.f);
        float sum_predictions = 0.f;
        int top_class = 0;
        float top_class_proba = 0.f;
        for (int dim_idx = 0; dim_idx < num_prediction_dimensions; dim_idx++) {
          const float proba =
              src_prediction[example_idx * num_prediction_dimensions + dim_idx];
          distribution->mutable_counts()->Set(dim_idx + 1, proba);
          sum_predictions += proba;
          if (proba > top_class_proba) {
            top_class_proba = proba;
            top_class = dim_idx + 1;
          }
        }
        distribution->set_sum(sum_predictions);
        classification->set_value(top_class);
      }
    } break;

    case proto::REGRESSION:
      DCHECK_EQ(num_prediction_dimensions, 1);
      dst_prediction->mutable_regression()->set_value(
          src_prediction[example_idx]);
      break;

    case proto::RANKING:
      DCHECK_EQ(num_prediction_dimensions, 1);
      dst_prediction->mutable_ranking()->set_relevance(
          src_prediction[example_idx]);
      break;

    case proto::CATEGORICAL_UPLIFT:
    case proto::NUMERICAL_UPLIFT:
      DCHECK_EQ(num_prediction_dimensions, 1);
      *dst_prediction->mutable_uplift()->mutable_treatment_effect() = {
          src_prediction.begin() + example_idx * num_prediction_dimensions,
          src_prediction.begin() +
              (example_idx + 1) * num_prediction_dimensions};
      break;

    case proto::ANOMALY_DETECTION:
      DCHECK_EQ(num_prediction_dimensions, 1);
      dst_prediction->mutable_anomaly_detection()->set_value(
          src_prediction[example_idx]);
      break;

    case proto::SURVIVAL_ANALYSIS:
      DCHECK_EQ(num_prediction_dimensions, 1);
      dst_prediction->mutable_survival_analysis()->set_log_hazard_ratio(
          src_prediction[example_idx]);
      break;
  }
}

void ProtoToFloatPrediction(const proto::Prediction& src_prediction,
                            proto::Task task,
                            absl::Span<float> dst_prediction) {
  size_t num_prediction_dimensions = dst_prediction.size();
  switch (task) {
    case proto::UNDEFINED:
      LOG(WARNING) << "Undefined task";
      break;
    case proto::CLASSIFICATION: {
      const auto& classification = src_prediction.classification();
      if (num_prediction_dimensions == 1) {
        // Binary classification, only need the probability of class 2
        float proba_class2 = classification.distribution().counts(2) /
                             classification.distribution().sum();
        dst_prediction[0] = proba_class2;
      } else {
        // Multi-class classification, copy all probabilities
        for (int dim_idx = 0; dim_idx < num_prediction_dimensions; dim_idx++) {
          dst_prediction[dim_idx] =
              classification.distribution().counts(dim_idx + 1) /
              classification.distribution().sum();
        }
      }
    } break;

    case proto::REGRESSION:
      DCHECK_EQ(num_prediction_dimensions, 1);
      dst_prediction[0] = src_prediction.regression().value();
      break;

    case proto::RANKING:
      DCHECK_EQ(num_prediction_dimensions, 1);
      dst_prediction[0] = src_prediction.ranking().relevance();
      break;

    case proto::CATEGORICAL_UPLIFT:
    case proto::NUMERICAL_UPLIFT:
      DCHECK_EQ(num_prediction_dimensions,
                src_prediction.uplift().treatment_effect().size());
      std::copy(src_prediction.uplift().treatment_effect().begin(),
                src_prediction.uplift().treatment_effect().end(),
                dst_prediction.begin());
      break;

    case proto::ANOMALY_DETECTION:
      DCHECK_EQ(num_prediction_dimensions, 1);
      dst_prediction[0] = src_prediction.anomaly_detection().value();
      break;

    case proto::SURVIVAL_ANALYSIS:
      DCHECK_EQ(num_prediction_dimensions, 1);
      dst_prediction[0] = src_prediction.survival_analysis().log_hazard_ratio();
      break;
  }
}

absl::Status AbstractModel::AppendEvaluationWithEngine(
    const dataset::VerticalDataset& dataset,
    const metric::proto::EvaluationOptions& option,
    const dataset::proto::LinkedWeightDefinition& weight_links,
    const serving::FastEngine& engine, utils::RandomEngine* rnd,
    std::vector<model::proto::Prediction>* predictions,
    metric::proto::EvaluationResults* eval) const {
  const auto& engine_features = engine.features();
  const int num_prediction_dimensions = engine.NumPredictionDimension();
  const size_t num_examples = dataset.nrow();

  std::vector<float> raw_predictions(num_prediction_dimensions * num_examples);

  size_t initial_prediction_size = 0;
  if (predictions) {
    initial_prediction_size = predictions->size();
    predictions->resize(initial_prediction_size + num_examples);
  }

  struct Cache {
    std::unique_ptr<serving::AbstractExampleSet> batch_of_examples;
    std::vector<float> batch_of_predictions;
  };

  const auto create_cache = [&](size_t thread_idx, size_t num_threads,
                                size_t block_size) -> Cache {
    Cache cache;
    cache.batch_of_examples = engine.AllocateExamples(block_size);
    cache.batch_of_predictions.resize(block_size);
    return cache;
  };

  const auto run = [&, num_prediction_dimensions](
                       size_t block_idx, size_t begin_item_idx,
                       size_t end_item_idx, Cache* cache) -> absl::Status {
    // Compute predictions
    const size_t effective_batch_size = end_item_idx - begin_item_idx;
    RETURN_IF_ERROR(CopyVerticalDatasetToAbstractExampleSet(
        dataset, begin_item_idx, end_item_idx, engine_features,
        cache->batch_of_examples.get()));
    engine.Predict(*cache->batch_of_examples, effective_batch_size,
                   &cache->batch_of_predictions);
    std::copy(
        cache->batch_of_predictions.begin(),
        cache->batch_of_predictions.begin() +
            effective_batch_size * num_prediction_dimensions,
        raw_predictions.begin() + begin_item_idx * num_prediction_dimensions);
    return absl::OkStatus();
  };

  RETURN_IF_ERROR(utils::concurrency::ConcurrentForLoopWithWorker<Cache>(
      /*num_items=*/dataset.nrow(),
      /*max_num_threads=*/option.num_threads(),
      /*min_block_size=*/100,    // At least 100 examples in a batch
      /*max_block_size=*/10000,  // No more than 10k examples in a batch
      create_cache, run));

  // Evaluate predictions
  proto::Prediction proto_prediction;
  for (size_t example_idx = 0; example_idx < num_examples; example_idx++) {
    FloatToProtoPrediction(raw_predictions, example_idx, task(),
                           num_prediction_dimensions, &proto_prediction);
    RETURN_IF_ERROR(SetGroundTruth(dataset, example_idx, &proto_prediction));
    if (option.has_weights()) {
      ASSIGN_OR_RETURN(
          const float weight,
          dataset::GetWeightWithStatus(dataset, example_idx, weight_links));
      proto_prediction.set_weight(weight);
    }
    RETURN_IF_ERROR(metric::AddPrediction(option, proto_prediction, rnd, eval));
    if (predictions) {
      (*predictions)[initial_prediction_size + example_idx] = proto_prediction;
    }
  }
  return absl::OkStatus();
}

absl::Status AbstractModel::AppendEvaluation(
    const dataset::VerticalDataset& dataset,
    const metric::proto::EvaluationOptions& option, utils::RandomEngine* rnd,
    metric::proto::EvaluationResults* eval,
    std::vector<model::proto::Prediction>* predictions) const {
  dataset::proto::LinkedWeightDefinition weight_links;
  if (option.has_weights()) {
    RETURN_IF_ERROR(dataset::GetLinkedWeightDefinition(
        option.weights(), data_spec_, &weight_links));
  }
  if (dataset.nrow() == 0) {
    return absl::InvalidArgumentError(
        "The dataset is empty. Cannot evaluate model.");
  }

  absl::StatusOr<std::unique_ptr<serving::FastEngine>> engine_or_status;
  if (!option.force_slow_engine()) {
    engine_or_status = BuildFastEngine();
  }
  if (engine_or_status.ok()) {
    RETURN_IF_ERROR(AppendEvaluationWithEngine(dataset, option, weight_links,
                                               *engine_or_status.value(), rnd,
                                               predictions, eval));
  } else {
    // Evaluate using the (slow) generic inference.

    proto::Prediction prediction;
    for (dataset::VerticalDataset::row_t test_row_idx = 0;
         test_row_idx < dataset.nrow(); test_row_idx++) {
      LOG_EVERY_N_SEC(INFO, 30) << (test_row_idx + 1) << "/" << dataset.nrow()
                                << " predictions evaluated.";
      Predict(dataset, test_row_idx, &prediction);
      RETURN_IF_ERROR(SetGroundTruth(dataset, test_row_idx, &prediction));
      if (option.has_weights()) {
        ASSIGN_OR_RETURN(
            const float weight,
            dataset::GetWeightWithStatus(dataset, test_row_idx, weight_links));
        prediction.set_weight(weight);
      }
      RETURN_IF_ERROR(metric::AddPrediction(option, prediction, rnd, eval));
      if (predictions) {
        predictions->push_back(prediction);
      }
    }
  }

  eval->set_num_folds(eval->num_folds() + 1);
  return absl::OkStatus();
}

absl::Status AbstractModel::AppendEvaluationOverrideType(
    const dataset::VerticalDataset& dataset,
    const metric::proto::EvaluationOptions& option,
    const proto::Task override_task, const int override_label_col_idx,
    const int override_group_col_idx, utils::RandomEngine* rnd,
    metric::proto::EvaluationResults* eval,
    std::vector<model::proto::Prediction>* predictions) const {
  dataset::proto::LinkedWeightDefinition weight_links;
  if (option.has_weights()) {
    RETURN_IF_ERROR(dataset::GetLinkedWeightDefinition(
        option.weights(), data_spec_, &weight_links));
  }

  auto engine_or_status = BuildFastEngine();
  if (engine_or_status.ok()) {
    RETURN_IF_ERROR(AppendEvaluationWithEngineOverrideType(
        dataset, option, override_task, override_label_col_idx,
        override_group_col_idx, weight_links, *engine_or_status.value(), rnd,
        predictions, eval));
  } else {
    proto::Prediction original_prediction;
    proto::Prediction overridden_prediction;
    for (dataset::VerticalDataset::row_t test_row_idx = 0;
         test_row_idx < dataset.nrow(); test_row_idx++) {
      LOG_EVERY_N_SEC(INFO, 30) << (test_row_idx + 1) << "/" << dataset.nrow()
                                << " predictions evaluated.";
      Predict(dataset, test_row_idx, &original_prediction);
      RETURN_IF_ERROR(metric::ChangePredictionType(
          task(), override_task, original_prediction, &overridden_prediction));
      RETURN_IF_ERROR(model::SetGroundTruth(
          dataset, test_row_idx,
          model::GroundTruthColumnIndices(
              override_label_col_idx, override_group_col_idx,
              uplift_treatment_col_idx_, label_entry_age_col_idx_,
              label_event_observed_col_idx_),
          override_task, &overridden_prediction));
      if (option.has_weights()) {
        ASSIGN_OR_RETURN(
            const float weight,
            dataset::GetWeightWithStatus(dataset, test_row_idx, weight_links));
        overridden_prediction.set_weight(weight);
      }
      RETURN_IF_ERROR(
          metric::AddPrediction(option, overridden_prediction, rnd, eval));
      if (predictions) {
        predictions->push_back(overridden_prediction);
      }
    }
  }
  return absl::OkStatus();
}

absl::Status AbstractModel::AppendEvaluationWithEngineOverrideType(
    const dataset::VerticalDataset& dataset,
    const metric::proto::EvaluationOptions& option, proto::Task override_task,
    int override_label_col_idx, int override_group_col_idx,
    const dataset::proto::LinkedWeightDefinition& weight_links,
    const serving::FastEngine& engine, utils::RandomEngine* rnd,
    std::vector<model::proto::Prediction>* predictions,
    metric::proto::EvaluationResults* eval) const {
  const auto& engine_features = engine.features();
  const int num_prediction_dimensions = engine.NumPredictionDimension();

  proto::Prediction original_prediction;
  proto::Prediction overridden_prediction;

  // Evaluate using the semi-fast generic engine.
  const int64_t total_num_examples = dataset.nrow();
  const int64_t batch_size =
      std::min(static_cast<int64_t>(100), total_num_examples);

  auto batch_of_examples = engine.AllocateExamples(batch_size);
  const int64_t num_batches =
      (total_num_examples + batch_size - 1) / batch_size;

  std::vector<float> batch_of_predictions;
  for (int64_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
    const int64_t begin_example_idx = batch_idx * batch_size;
    const int64_t end_example_idx =
        std::min(begin_example_idx + batch_size, total_num_examples);
    const int effective_batch_size = end_example_idx - begin_example_idx;
    RETURN_IF_ERROR(CopyVerticalDatasetToAbstractExampleSet(
        dataset, begin_example_idx, end_example_idx, engine_features,
        batch_of_examples.get()));
    engine.Predict(*batch_of_examples, effective_batch_size,
                   &batch_of_predictions);
    for (int sub_example_idx = 0; sub_example_idx < effective_batch_size;
         sub_example_idx++) {
      FloatToProtoPrediction(batch_of_predictions, sub_example_idx, task(),
                             num_prediction_dimensions, &original_prediction);

      RETURN_IF_ERROR(metric::ChangePredictionType(
          task(), override_task, original_prediction, &overridden_prediction));
      RETURN_IF_ERROR(model::SetGroundTruth(
          dataset, begin_example_idx + sub_example_idx,
          model::GroundTruthColumnIndices(
              override_label_col_idx, override_group_col_idx,
              uplift_treatment_col_idx_, label_entry_age_col_idx_,
              label_event_observed_col_idx_),
          override_task, &overridden_prediction));

      if (option.has_weights()) {
        ASSIGN_OR_RETURN(
            const float weight,
            dataset::GetWeightWithStatus(
                dataset, begin_example_idx + sub_example_idx, weight_links));
        overridden_prediction.set_weight(weight);
      }
      RETURN_IF_ERROR(
          metric::AddPrediction(option, overridden_prediction, rnd, eval));
      if (predictions) {
        predictions->push_back(overridden_prediction);
      }
    }
  }
  return absl::OkStatus();
}

absl::Status AbstractModel::SetGroundTruth(
    const dataset::VerticalDataset& dataset,
    dataset::VerticalDataset::row_t row_idx,
    proto::Prediction* prediction) const {
  return model::SetGroundTruth(
      dataset, row_idx,
      GroundTruthColumnIndices(
          label_col_idx_, ranking_group_col_idx_, uplift_treatment_col_idx_,
          label_entry_age_col_idx_, label_event_observed_col_idx_),
      task_, prediction);
}

absl::Status AbstractModel::SetGroundTruth(
    const dataset::proto::Example& example,
    proto::Prediction* prediction) const {
  return model::SetGroundTruth(
      example,
      GroundTruthColumnIndices(
          label_col_idx_, ranking_group_col_idx_, uplift_treatment_col_idx_,
          label_entry_age_col_idx_, label_event_observed_col_idx_),
      task_, prediction);
}

absl::Status SetGroundTruth(const dataset::VerticalDataset& dataset,
                            const dataset::VerticalDataset::row_t row_idx,
                            const GroundTruthColumnIndices& columns,
                            const proto::Task task,
                            proto::Prediction* prediction) {
  switch (task) {
    case proto::Task::CLASSIFICATION: {
      STATUS_CHECK_EQ(columns.group_col_idx, kNoRankingGroup);
      STATUS_CHECK_EQ(columns.uplift_treatment_col_idx,
                      kNoUpliftTreatmentGroup);
      STATUS_CHECK_EQ(columns.entry_age_col_idx, -1);
      STATUS_CHECK_EQ(columns.event_observed_col_idx, -1);
      ASSIGN_OR_RETURN(const auto* classification_labels,
                       dataset.ColumnWithCastWithStatus<
                           dataset::VerticalDataset::CategoricalColumn>(
                           columns.label_col_idx));
      prediction->mutable_classification()->set_ground_truth(
          classification_labels->values()[row_idx]);
    } break;
    case proto::Task::REGRESSION: {
      STATUS_CHECK_EQ(columns.group_col_idx, kNoRankingGroup);
      STATUS_CHECK_EQ(columns.uplift_treatment_col_idx,
                      kNoUpliftTreatmentGroup);
      STATUS_CHECK_EQ(columns.entry_age_col_idx, -1);
      STATUS_CHECK_EQ(columns.event_observed_col_idx, -1);
      ASSIGN_OR_RETURN(const auto* regression_labels,
                       dataset.ColumnWithCastWithStatus<
                           dataset::VerticalDataset::NumericalColumn>(
                           columns.label_col_idx));
      prediction->mutable_regression()->set_ground_truth(
          regression_labels->values()[row_idx]);
    } break;
    case proto::Task::RANKING: {
      STATUS_CHECK_NE(columns.group_col_idx, kNoRankingGroup);
      STATUS_CHECK_EQ(columns.uplift_treatment_col_idx,
                      kNoUpliftTreatmentGroup);
      STATUS_CHECK_EQ(columns.entry_age_col_idx, -1);
      STATUS_CHECK_EQ(columns.event_observed_col_idx, -1);
      ASSIGN_OR_RETURN(const auto* ranking_labels,
                       dataset.ColumnWithCastWithStatus<
                           dataset::VerticalDataset::NumericalColumn>(
                           columns.label_col_idx));
      prediction->mutable_ranking()->set_ground_truth_relevance(
          ranking_labels->values()[row_idx]);

      const auto* categorical_groups = dataset.ColumnWithCastOrNull<
          dataset::VerticalDataset::CategoricalColumn>(columns.group_col_idx);
      const auto* hash_groups =
          dataset.ColumnWithCastOrNull<dataset::VerticalDataset::HashColumn>(
              columns.group_col_idx);
      if (categorical_groups) {
        prediction->mutable_ranking()->set_group_id(
            categorical_groups->values()[row_idx]);
      } else if (hash_groups) {
        prediction->mutable_ranking()->set_group_id(
            hash_groups->values()[row_idx]);
      } else {
        STATUS_FATAL("The group attribute should be CATEGORICAL or HASH");
      }
    } break;
    case proto::Task::CATEGORICAL_UPLIFT: {
      STATUS_CHECK_EQ(columns.group_col_idx, kNoRankingGroup);
      STATUS_CHECK_NE(columns.uplift_treatment_col_idx,
                      kNoUpliftTreatmentGroup);
      STATUS_CHECK_EQ(columns.entry_age_col_idx, -1);
      STATUS_CHECK_EQ(columns.event_observed_col_idx, -1);
      const auto& numerical_outcomes = dataset.ColumnWithCastOrNull<
          dataset::VerticalDataset::CategoricalColumn>(columns.label_col_idx);
      const auto& categorical_outcomes = dataset.ColumnWithCastOrNull<
          dataset::VerticalDataset::CategoricalColumn>(columns.label_col_idx);
      if (categorical_outcomes != nullptr) {
        prediction->mutable_uplift()->set_outcome_categorical(
            categorical_outcomes->values()[row_idx]);
      } else if (numerical_outcomes != nullptr) {
        prediction->mutable_uplift()->set_outcome_numerical(
            numerical_outcomes->values()[row_idx]);
      } else {
        STATUS_FATAL("Not supported outcome type");
      }
      ASSIGN_OR_RETURN(const auto& treatments,
                       dataset.ColumnWithCastWithStatus<
                           dataset::VerticalDataset::CategoricalColumn>(
                           columns.uplift_treatment_col_idx));
      prediction->mutable_uplift()->set_treatment(
          treatments->values()[row_idx]);
    } break;

    case proto::Task::NUMERICAL_UPLIFT: {
      STATUS_CHECK_EQ(columns.group_col_idx, kNoRankingGroup);
      STATUS_CHECK_NE(columns.uplift_treatment_col_idx,
                      kNoUpliftTreatmentGroup);
      STATUS_CHECK_EQ(columns.entry_age_col_idx, -1);
      STATUS_CHECK_EQ(columns.event_observed_col_idx, -1);
      const auto& numerical_outcomes =
          dataset.ColumnWithCast<dataset::VerticalDataset::NumericalColumn>(
              columns.label_col_idx);
      prediction->mutable_uplift()->set_outcome_numerical(
          numerical_outcomes->values()[row_idx]);
      const auto& treatments =
          dataset
              .ColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(
                  columns.uplift_treatment_col_idx)
              ->values();
      prediction->mutable_uplift()->set_treatment(treatments[row_idx]);
    } break;
    case proto::Task::ANOMALY_DETECTION:
      // No ground truth to set.
      break;

    case proto::Task::SURVIVAL_ANALYSIS: {
      STATUS_CHECK_NE(columns.event_observed_col_idx, -1);
      // No ground truth to set.
    } break;

    default:
      STATUS_FATAL("Non supported task.");
      break;
  }
  return absl::OkStatus();
}

absl::Status SetGroundTruth(const dataset::proto::Example& example,
                            const GroundTruthColumnIndices& columns,
                            proto::Task task, proto::Prediction* prediction) {
  switch (task) {
    case proto::Task::CLASSIFICATION: {
      STATUS_CHECK_EQ(columns.group_col_idx, kNoRankingGroup);
      STATUS_CHECK_EQ(columns.uplift_treatment_col_idx,
                      kNoUpliftTreatmentGroup);
      prediction->mutable_classification()->set_ground_truth(
          example.attributes(columns.label_col_idx).categorical());
    } break;
    case proto::Task::REGRESSION: {
      STATUS_CHECK_EQ(columns.group_col_idx, kNoRankingGroup);
      STATUS_CHECK_EQ(columns.uplift_treatment_col_idx,
                      kNoUpliftTreatmentGroup);
      prediction->mutable_regression()->set_ground_truth(
          example.attributes(columns.label_col_idx).numerical());
    } break;
    case proto::Task::RANKING: {
      STATUS_CHECK_NE(columns.group_col_idx, kNoRankingGroup);
      STATUS_CHECK_EQ(columns.uplift_treatment_col_idx,
                      kNoUpliftTreatmentGroup);
      prediction->mutable_ranking()->set_ground_truth_relevance(
          example.attributes(columns.label_col_idx).numerical());
      const auto& group_value = example.attributes(columns.group_col_idx);
      switch (group_value.type_case()) {
        case dataset::proto::Example_Attribute::kCategorical:
          prediction->mutable_ranking()->set_group_id(
              example.attributes(columns.group_col_idx).categorical());
          break;
        case dataset::proto::Example_Attribute::kHash:
          prediction->mutable_ranking()->set_group_id(
              example.attributes(columns.group_col_idx).hash());
          break;
        default:
          STATUS_FATAL("The group attribute should be CATEGORICAL or HASH");
          break;
      }
    } break;
    case proto::Task::ANOMALY_DETECTION:
      // No ground truth to set.
      break;

    case proto::Task::SURVIVAL_ANALYSIS: {
      STATUS_CHECK_NE(columns.event_observed_col_idx, -1);
      // TODO: Implement.
      return absl::InvalidArgumentError("Not implemented");
    } break;

    default:
      STATUS_FATAL("Non supported task.");
      break;
  }
  return absl::OkStatus();
}

std::string AbstractModel::DescriptionAndStatistics(
    bool full_definition) const {
  std::string model_description;
  AppendDescriptionAndStatistics(full_definition, &model_description);
  return model_description;
}

void AbstractModel::AppendDescriptionAndStatistics(
    const bool full_definition, std::string* description) const {
  absl::StrAppendFormat(description, "Type: \"%s\"\n", name());
  absl::StrAppendFormat(description, "Task: %s\n", proto::Task_Name(task()));
  if (label_col_idx_ != -1) {
    absl::StrAppendFormat(description, "Label: \"%s\"\n",
                          data_spec().columns(label_col_idx_).name());
  }
  if (ranking_group_col_idx_ != -1) {
    absl::StrAppendFormat(description, "Rank group: \"%s\"\n",
                          data_spec().columns(ranking_group_col_idx_).name());
  }

  if (full_definition) {
    absl::StrAppend(description, "\nDataSpec:\n",
                    dataset::PrintHumanReadable(data_spec(), false), "\n");
  }

  absl::SubstituteAndAppend(description, "\nInput Features ($0):\n",
                            input_features().size());
  for (const auto input_feature : input_features()) {
    absl::SubstituteAndAppend(description, "\t$0\n",
                              data_spec().columns(input_feature).name());
  }
  absl::StrAppend(description, "\n");

  if (!weights_.has_value()) {
    absl::StrAppend(description, "No weights\n");
  } else {
    absl::StrAppend(description, "Trained with weights\n");
    if (full_definition) {
      absl::StrAppend(
          description, "\nWeights:\n",
          utils::SerializeTextProto(weights_.value()).value_or("ERROR"), "n");
    }
  }

  absl::StrAppend(description, "\n");
  AppendAllVariableImportanceDescription(description);
  absl::StrAppend(description, "\n");

  const auto self_evaluation_description =
      metric::TextReport(ValidationEvaluation());
  if (self_evaluation_description.ok()) {
    LOG(INFO) << "Model self evaluation:\n"
              << self_evaluation_description.value();
  } else {
    absl::StrAppend(description, "Cannot compute model self evaluation:",
                    self_evaluation_description.status().message(), "\n");
  }

  if (hyperparameter_optimizer_logs_.has_value()) {
    AppendHyperparameterOptimizerLogs(description);
  }

  if (feature_selection_logs_.has_value()) {
    AppendFeatureSelectionLogs(description);
  }
}

void AbstractModel::AppendFeatureSelectionLogs(std::string* description) const {
  absl::StrAppend(description, "Feature selection logs:\n\n");
  for (int iteration_idx = 0;
       iteration_idx < feature_selection_logs_->iterations_size();
       iteration_idx++) {
    const auto& iteration =
        feature_selection_logs_->iterations()[iteration_idx];
    absl::StrAppendFormat(
        description,
        "Iteration:%d Score:%g\n\tFeatures: %s\n\tMetrics:", iteration_idx,
        iteration.score(), absl::StrJoin(iteration.features(), ","));
    for (const auto& metric : iteration.metrics()) {
      absl::StrAppendFormat(description, "%s:%g", metric.first, metric.second);
    }
    absl::StrAppend(description, "\n");
  }
}

void AbstractModel::AppendHyperparameterOptimizerLogs(
    std::string* description) const {
  // Converts an hyperparameter set into an inlined human readable
  // representation of the form: "{<field name>:<value>}+}.
  const auto hyperparameter_to_string =
      [](const proto::GenericHyperParameters& ps) -> std::string {
    if (ps.fields_size() == 0) {
      return "*empty*";
    }
    std::string text;
    for (const auto& field : ps.fields()) {
      if (!text.empty()) {
        // Adds a space in between hyperparameters.
        absl::StrAppend(&text, " ");
      }
      absl::StrAppend(&text, field.name(), ":");
      switch (field.value().Type_case()) {
        case proto::GenericHyperParameters::Value::TypeCase::kCategorical:
          absl::StrAppend(&text, field.value().categorical());
          break;
        case proto::GenericHyperParameters::Value::TypeCase::kInteger:
          absl::StrAppend(&text, field.value().integer());
          break;
        case proto::GenericHyperParameters::Value::TypeCase::kReal:
          absl::StrAppend(&text, field.value().real());
          break;
        case proto::GenericHyperParameters::Value::TypeCase::kCategoricalList:
          absl::StrAppend(&text,
                          field.value().categorical_list().DebugString());
          break;
        case proto::GenericHyperParameters::Value::TypeCase::TYPE_NOT_SET:
          absl::StrAppend(&text, "NOT_SET");
          break;
      }
    }
    return text;
  };

  // Title
  absl::StrAppend(description, "Hyperparameter optimizer:\n\n");
  const auto& logs = hyperparameter_optimizer_logs_.value();  // Shorter

  absl::StrAppendFormat(description, "Best parameters: %s\n",
                        hyperparameter_to_string(logs.best_hyperparameters()));
  absl::StrAppendFormat(description, "Num steps: %d\n", logs.steps_size());
  absl::StrAppendFormat(description, "Best score: %f\n", logs.best_score());
  absl::StrAppend(description, "\n");

  // Prints the score and hyper-parameter of each step.
  for (int step_idx = 0; step_idx < logs.steps_size(); step_idx++) {
    const auto& step = logs.steps(step_idx);
    absl::StrAppendFormat(description, "Step #%d score:%f parameters:{ %s }\n",
                          step_idx, step.score(),
                          hyperparameter_to_string(step.hyperparameters()));
  }
  absl::StrAppend(description, "\n");
}

std::vector<std::string> AbstractModel::AvailableVariableImportances() const {
  std::vector<std::string> keys;
  for (const auto& var : precomputed_variable_importances_) {
    keys.push_back(var.first);
  }
  return keys;
}

absl::Status AbstractModel::PrecomputeVariableImportances(
    const std::vector<std::string>& variable_importances) {
  for (const auto& vi_key : variable_importances) {
    if (precomputed_variable_importances_.find(vi_key) !=
        precomputed_variable_importances_.end()) {
      // VI already cached.
      continue;
    }
    ASSIGN_OR_RETURN(const auto src_vi_values, GetVariableImportance(vi_key));
    auto& dst_vi_values = precomputed_variable_importances_[vi_key];
    for (const auto& src_vi_value : src_vi_values) {
      auto* dst_vi_value = dst_vi_values.add_variable_importances();
      dst_vi_value->set_attribute_idx(src_vi_value.attribute_idx());
      dst_vi_value->set_importance(src_vi_value.importance());
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<std::vector<proto::VariableImportance>>
AbstractModel::GetVariableImportance(absl::string_view key) const {
  const auto vi_it = precomputed_variable_importances_.find(key);
  if (vi_it == precomputed_variable_importances_.end()) {
    const auto expected_variable_importances = AvailableVariableImportances();
    if (std::find(expected_variable_importances.begin(),
                  expected_variable_importances.end(),
                  key) != expected_variable_importances.end()) {
      return absl::NotFoundError(absl::Substitute(
          "The variable importance \"$0\" does not exist for this model "
          "\"$1\". However, "
          "this variable is registered in \"AvailableVariableImportances\": "
          "This error is likely do to an implementation error in the model "
          "class.",
          key, name()));
    } else {
      return absl::NotFoundError(absl::Substitute(
          "The variable importance \"$0\" does not exist for "
          "this model \"$1\". Use \"AvailableVariableImportances\" "
          "for the list of available variable importances: $2",
          key, name(), absl::StrJoin(AvailableVariableImportances(), ", ")));
    }
  }
  return std::vector<proto::VariableImportance>{
      vi_it->second.variable_importances().begin(),
      vi_it->second.variable_importances().end()};
}

void AbstractModel::AppendAllVariableImportanceDescription(
    std::string* description) const {
  if (AvailableVariableImportances().empty()) {
    absl::StrAppend(description,
                    "Variable Importance disabled i.e. "
                    "compute_oob_variable_importances=false.");
  }

  for (const auto& variable_importance_key : AvailableVariableImportances()) {
    absl::SubstituteAndAppend(description, "Variable Importance: $0:\n",
                              variable_importance_key);
    const auto variable_importance =
        GetVariableImportance(variable_importance_key);
    if (!variable_importance.ok()) {
      absl::StrAppend(description, "Cannot access the variable importance: ",
                      variable_importance.status().message(), "\n");
    } else {
      AppendVariableImportanceDescription(variable_importance.value(),
                                          data_spec(), 4, description);
      absl::StrAppend(description, "\n");
    }
  }
}

void AppendVariableImportanceDescription(
    const std::vector<proto::VariableImportance>& variable_importances,
    const dataset::proto::DataSpecification& data_spec,
    const int leading_spaces, std::string* description) {
  if (variable_importances.empty()) {
    return;
  }

  const int max_bar_length = 16;
  double min_value = 0;
  double max_value = 0;
  bool first_value = true;
  for (const auto& var : variable_importances) {
    const auto value = var.importance();
    if (!std::isfinite(value)) {
      continue;
    }
    if (first_value) {
      min_value = max_value = value;
      first_value = false;
    } else {
      min_value = std::min(min_value, value);
      max_value = std::max(max_value, value);
    }
  }

  auto safe_delta = max_value - min_value;
  if (safe_delta <= 0) {
    safe_delta = 1;
  }

  size_t max_var_name_length = 1;
  for (const auto& var : variable_importances) {
    const auto& variable_name = data_spec.columns(var.attribute_idx()).name();
    max_var_name_length = std::max(max_var_name_length, variable_name.size());
  }

  for (int var_idx = 0; var_idx < variable_importances.size(); var_idx++) {
    const auto& var = variable_importances[var_idx];
    const auto& variable_name = data_spec.columns(var.attribute_idx()).name();

    int bar_length;
    if (std::isnan(var.importance())) {
      bar_length = 0;
    } else if (!std::isfinite(var.importance())) {
      bar_length = max_bar_length;
    } else {
      bar_length = max_bar_length * (var.importance() - min_value) / safe_delta;
    }

    const std::string bar(bar_length, '#');
    absl::StrAppendFormat(description, "%*d. %*s %9f %s\n", leading_spaces + 1,
                          var_idx + 1, max_var_name_length + 2,
                          absl::StrCat("\"", variable_name, "\""),
                          var.importance(), bar);
  }
}

metric::proto::EvaluationResults AbstractModel::ValidationEvaluation() const {
  LOG(WARNING) << "Validation evaluation not supported for " << name();
  return {};
}

void MergeVariableImportance(const std::vector<proto::VariableImportance>& src,
                             const double weight_src,
                             std::vector<proto::VariableImportance>* dst) {
  absl::flat_hash_map<int, double> new_importance;

  for (const auto& dst_item : *dst) {
    new_importance[dst_item.attribute_idx()] +=
        dst_item.importance() * (1. - weight_src);
  }

  for (const auto& src_item : src) {
    new_importance[src_item.attribute_idx()] +=
        src_item.importance() * weight_src;
  }

  dst->clear();
  for (const auto& item : new_importance) {
    proto::VariableImportance var_importance;
    var_importance.set_attribute_idx(item.first);
    var_importance.set_importance(item.second);
    dst->push_back(var_importance);
  }

  const auto var_importance_comparator =
      [](const model::proto::VariableImportance& a,
         const model::proto::VariableImportance& b) {
        if (a.importance() != b.importance()) {
          return a.importance() > b.importance();
        } else {
          return a.attribute_idx() < b.attribute_idx();
        }
      };
  std::sort(dst->begin(), dst->end(), var_importance_comparator);
}

void PredictionMerger::Add(const proto::Prediction& src,
                           const float src_factor) {
  DCHECK(dst_->type_case() == src.type_case() ||
         dst_->type_case() == proto::Prediction::TYPE_NOT_SET);
  switch (src.type_case()) {
    case proto::Prediction::kClassification: {
      auto* dst_cls = dst_->mutable_classification();
      const auto& src_cls = src.classification();
      const int num_classes = src_cls.distribution().counts_size();
      if (!dst_cls->has_distribution()) {
        dst_cls->mutable_distribution()->mutable_counts()->Resize(num_classes,
                                                                  0);
      }
      const float normalization = src_factor / src_cls.distribution().sum();
      for (int i = 0; i < num_classes; i++) {
        dst_cls->mutable_distribution()->set_counts(
            i, dst_cls->distribution().counts(i) +
                   normalization * src_cls.distribution().counts(i));
      }
      dst_cls->mutable_distribution()->set_sum(dst_cls->distribution().sum() +
                                               src_factor);
    } break;
    case proto::Prediction::kRegression:
      dst_->mutable_regression()->set_value(
          dst_->regression().value() + src_factor * src.regression().value());
      break;
    case proto::Prediction::kRanking:
      dst_->mutable_ranking()->set_relevance(
          dst_->ranking().relevance() + src_factor * src.ranking().relevance());
      break;
    case proto::Prediction::kAnomalyDetection:
      dst_->mutable_anomaly_detection()->set_value(
          dst_->anomaly_detection().value() +
          src_factor * src.anomaly_detection().value());
      break;
    default:
      CHECK(false);
  }
}

void PredictionMerger::Merge() {
  switch (dst_->type_case()) {
    case proto::Prediction::kClassification: {
      auto* dst_cls = dst_->mutable_classification();
      dst_cls->set_value(utils::TopClass(dst_cls->distribution()));
    } break;
    default:
      break;
  }
}

void PredictionMerger::ScalePrediction(const float scale,
                                       proto::Prediction* dst) {
  switch (dst->type_case()) {
    case proto::Prediction::kRegression:
      dst->mutable_regression()->set_value(dst->regression().value() * scale);
      break;
    case proto::Prediction::kRanking:
      dst->mutable_ranking()->set_relevance(dst->ranking().relevance() * scale);
      break;
    case proto::Prediction::kAnomalyDetection:
      dst->mutable_anomaly_detection()->set_value(
          dst->anomaly_detection().value() * scale);
      break;
    default:
      break;
  }
}

void AbstractModel::CopyAbstractModelMetaData(AbstractModel* dst) const {
  dst->set_data_spec(data_spec());
  dst->set_task(task());
  dst->set_label_col_idx(label_col_idx());
  dst->set_ranking_group_col(ranking_group_col_idx());
  if (weights().has_value()) {
    dst->set_weights(weights().value());
  }
  *dst->mutable_input_features() = input_features();
  dst->precomputed_variable_importances_ = precomputed_variable_importances_;
  dst->classification_outputs_probabilities_ =
      classification_outputs_probabilities_;

  if (hyperparameter_optimizer_logs_.has_value()) {
    dst->hyperparameter_optimizer_logs_ = hyperparameter_optimizer_logs_;
  } else {
    dst->hyperparameter_optimizer_logs_ = {};
  }

  if (feature_selection_logs_.has_value()) {
    dst->feature_selection_logs_ = feature_selection_logs_;
  } else {
    dst->feature_selection_logs_ = {};
  }
}

absl::Status AbstractModel::Validate() const {
  if (label_col_idx_ < -1 || label_col_idx_ >= data_spec().columns_size()) {
    return absl::InvalidArgumentError("Invalid label column");
  }

  if (ranking_group_col_idx_ != -1 &&
      (ranking_group_col_idx_ < 0 ||
       ranking_group_col_idx_ >= data_spec().columns_size())) {
    return absl::InvalidArgumentError("Invalid ranking group column");
  }

  for (const int col_idx : input_features_) {
    if (col_idx < 0 || col_idx >= data_spec().columns_size()) {
      return absl::InvalidArgumentError("Invalid feature column");
    }
  }

  for (const auto& var_importance : precomputed_variable_importances_) {
    for (const auto& feature : var_importance.second.variable_importances()) {
      if (feature.attribute_idx() < 0 ||
          feature.attribute_idx() >= data_spec().columns_size()) {
        return absl::InvalidArgumentError("Invalid feature column");
      }
    }
  }

  switch (task()) {
    case model::proto::Task::CLASSIFICATION:
      if (label_col_spec().type() != dataset::proto::CATEGORICAL) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Invalid label type for classification: ",
            dataset::proto::ColumnType_Name(label_col_spec().type())));
      }
      break;
    case model::proto::Task::REGRESSION:
    case model::proto::Task::RANKING:
      if (label_col_spec().type() != dataset::proto::NUMERICAL) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Invalid label type for regression: ",
            dataset::proto::ColumnType_Name(label_col_spec().type())));
      }
      break;
    case model::proto::Task::SURVIVAL_ANALYSIS:
      if (label_col_spec().type() != dataset::proto::NUMERICAL) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Invalid label type for regression: ",
            dataset::proto::ColumnType_Name(label_col_spec().type())));
      }
      if (data_spec().columns(label_event_observed_col_idx()).type() !=
          dataset::proto::BOOLEAN) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Invalid label event observed type for survival analysis: ",
            dataset::proto::ColumnType_Name(
                data_spec().columns(label_event_observed_col_idx()).type())));
      }
      if (label_entry_age_col_idx() != -1 &&
          data_spec().columns(label_entry_age_col_idx()).type() !=
              dataset::proto::NUMERICAL) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Invalid label entry age type for survival analysis: ",
            dataset::proto::ColumnType_Name(
                data_spec().columns(label_entry_age_col_idx()).type())));
      }
      break;
    case model::proto::Task::CATEGORICAL_UPLIFT:
      if (label_col_spec().type() != dataset::proto::CATEGORICAL) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Invalid label type for categorical uplift: ",
            dataset::proto::ColumnType_Name(label_col_spec().type())));
      }
      break;
    case model::proto::Task::NUMERICAL_UPLIFT:
      if (label_col_spec().type() != dataset::proto::NUMERICAL) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Invalid label type for regressive uplift: ",
            dataset::proto::ColumnType_Name(label_col_spec().type())));
      }
      break;
    case model::proto::Task::ANOMALY_DETECTION:
      // Nothing to check
      break;
    default:
      return absl::InvalidArgumentError("Unknown task");
  }
  return absl::Status();
}

std::vector<std::unique_ptr<FastEngineFactory>>
AbstractModel::ListCompatibleFastEngines() const {
  // Index the compatible engines.
  struct Item {
    std::unique_ptr<FastEngineFactory> factory;
    absl::flat_hash_set<std::string> is_better_than;
  };
  std::vector<Item> items;

  for (auto& factory : ListAllFastEngines()) {
    if (!factory->IsCompatible(this)) {
      continue;
    }
    const auto is_better_than = factory->IsBetterThan();
    items.push_back(Item{std::move(factory),
                         {is_better_than.begin(), is_better_than.end()}});
  }

  // Sort the engine by speed (fastest first).
  std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
    if (a.is_better_than.find(b.factory->name()) != a.is_better_than.end()) {
      // "a" is better than "b".
      return true;
    }
    return false;
  });

  std::vector<std::unique_ptr<FastEngineFactory>> compatible_engines;
  compatible_engines.reserve(items.size());
  for (auto& item : items) {
    compatible_engines.push_back(std::move(item.factory));
  }
  return compatible_engines;
}

std::vector<std::string> AbstractModel::ListCompatibleFastEngineNames() const {
  std::vector<std::string> compatible_engines;
  for (auto& factory : ListCompatibleFastEngines()) {
    compatible_engines.push_back(factory->name());
  }
  return compatible_engines;
}

absl::StatusOr<std::unique_ptr<serving::FastEngine>>
AbstractModel::BuildFastEngine(
    const std::optional<std::string>& force_engine_name) const {
  if (!allow_fast_engine_) {
    return absl::NotFoundError("allow_fast_engine is set to false.");
  }

  // List the compatible engines.
  auto sorted_compatible_engines = ListCompatibleFastEngines();

  // How to create the engine.
  std::unique_ptr<FastEngineFactory> engine_factory;
  if (force_engine_name.has_value()) {
    for (auto& compatible_engine : sorted_compatible_engines) {
      if (compatible_engine->name() == *force_engine_name) {
        engine_factory = std::move(compatible_engine);
        break;
      }
    }
    if (!engine_factory) {
      return absl::NotFoundError(absl::StrCat(
          "The forced engine \"", *force_engine_name,
          "\" does not exist or is not compatible with the model"));
    }
  } else {
    if (sorted_compatible_engines.empty()) {
      return absl::NotFoundError(absl::Substitute(
          "No compatible engine available for model $0. 1)interresting Make "
          "sure the "
          "corresponding engine is added as a dependency, 2) use the (slow) "
          "generic engine (i.e. \"model.Predict()\") or 3) use one of the fast "
          "non-generic engines available in ../serving.",
          name()));
    }

    // Select the best engine.
    engine_factory = std::move(sorted_compatible_engines.front());
  }

  auto engine_or = engine_factory->CreateEngine(this);
  if (!engine_or.ok()) {
    LOG(WARNING) << "The engine \"" << engine_factory->name()
                 << "\" is compatible but could not be created: "
                 << engine_or.status().message();
  } else {
    LOG_EVERY_N_SEC(INFO, 10)
        << "Engine \"" << engine_factory->name() << "\" built";
    STATUS_CHECK(engine_or.value());
  }
  return engine_or;
}

std::optional<size_t> AbstractModel::AbstractAttributesSizeInBytes() const {
  if (!utils::ProtoSizeInBytesIsAvailable()) {
    return {};
  }
  size_t size = sizeof(*this) + name_.size() +
                utils::ProtoSizeInBytes(data_spec_).value_or(0);
  size +=
      input_features_.size() * sizeof(decltype(input_features_)::value_type);
  if (weights_.has_value()) {
    size += utils::ProtoSizeInBytes(weights_.value()).value_or(0);
  }
  for (const auto& v : precomputed_variable_importances_) {
    size += sizeof(v) + v.first.size() +
            utils::ProtoSizeInBytes(v.second).value_or(0);
  }
  return size;
}

absl::Status AbstractModel::ValidateModelIOOptions(
    const ModelIOOptions& io_options) {
  if (!io_options.file_prefix) {
    return absl::InvalidArgumentError(
        "No model file prefix given. When using model::LoadModel() and "
        "model::SaveModel(), a prefix is automatically chosen, if possible.");
  }
  return absl::OkStatus();
}

absl::Status AbstractModel::MakePureServing() {
  is_pure_model_ = true;
  precomputed_variable_importances_.clear();
  hyperparameter_optimizer_logs_ = {};
  feature_selection_logs_ = {};
  return Validate();
}

#define STR_CHECK_EQ(a, b)                                       \
  if (a != b) {                                                  \
    return absl::StrCat(#a, " != ", #b, " i.e. ", a, " != ", b); \
  }

#define STR_CHECK_EQ_NO_PRINT(a, b)      \
  if (a != b) {                          \
    return absl::StrCat(#a, " != ", #b); \
  }

std::string AbstractModel::DebugCompare(const AbstractModel& other) const {
  if (data_spec_.DebugString() != other.data_spec().DebugString()) {
    return absl::StrCat("Dataspecs don't match.\n\n", data_spec_.DebugString(),
                        "\nvs\n\n", other.data_spec().DebugString());
  }

  if (weights_.has_value() != other.weights().has_value()) {
    return "Only one of the models has weights";
  }
  if (weights_.has_value() && other.weights().has_value()) {
    if (weights_->DebugString() != other.weights_->DebugString()) {
      return absl::StrCat("Weights don't match.\n\n", data_spec_.DebugString(),
                          "\nvs\n\n", other.data_spec().DebugString());
    }
  }

  STR_CHECK_EQ(name_, other.name_);
  STR_CHECK_EQ(task_, other.task_);
  STR_CHECK_EQ(label_col_idx_, other.label_col_idx_);
  STR_CHECK_EQ(ranking_group_col_idx_, other.ranking_group_col_idx_);
  STR_CHECK_EQ(uplift_treatment_col_idx_, other.uplift_treatment_col_idx_);
  STR_CHECK_EQ_NO_PRINT(input_features_, other.input_features_);
  STR_CHECK_EQ(classification_outputs_probabilities_,
               other.classification_outputs_probabilities_);
  STR_CHECK_EQ(label_entry_age_col_idx_, other.label_entry_age_col_idx_);
  STR_CHECK_EQ(label_event_observed_col_idx_,
               other.label_event_observed_col_idx_);

  // Note: We don't check for equality of meta-data.

  return {};
}

}  // namespace model
}  // namespace yggdrasil_decision_forests
