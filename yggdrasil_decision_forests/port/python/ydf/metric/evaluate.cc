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

#include "ydf/metric/evaluate.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "pybind11_protobuf/native_proto_caster.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "ydf/utils/numpy_data.h"
#include "ydf/utils/status_casters.h"
#include "yggdrasil_decision_forests/utils/distribution.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace py = ::pybind11;

namespace yggdrasil_decision_forests::port::python {

namespace {

absl::Status AddBinaryClassificationPredictions(
    StridedSpanFloat32 raw_predictions, StridedSpanInt32 labels,
    StridedSpanFloat32 weights,
    const metric::proto::EvaluationOptions& eval_options,
    utils::RandomEngine* rnd, metric::proto::EvaluationResults* eval) {
  DCHECK(weights.empty() || weights.size() == labels.size());
  DCHECK_EQ(raw_predictions.size(), labels.size());
  // Categorical value of the "positive" and "negative" classes;
  constexpr int kNegativeValue = 1;
  constexpr int kPositiveValue = 2;

  model::proto::Prediction prediction_proto;
  auto& prediction_distribution =
      *prediction_proto.mutable_classification()->mutable_distribution();
  prediction_distribution.mutable_counts()->Resize(3, 0);
  // TODO: Add multi-thread support.
  for (size_t i = 0; i < raw_predictions.size(); ++i) {
    const float prediction = raw_predictions[i];
    const int label = labels[i];
    if (label < 0 || label >= 2) {
      return absl::InvalidArgumentError(
          absl::Substitute("Label values for binary classification must be 0 "
                           "or 1, got $0 for example $1",
                           label, i));
    }
    if (prediction < 0 || prediction > 1) {
      return absl::InvalidArgumentError(absl::Substitute(
          "Predictions values for binary classification must be between 0 "
          "or 1, got $0 for example $1",
          prediction, i));
    }
    prediction_proto.mutable_classification()->set_ground_truth(
        label ? kPositiveValue : kNegativeValue);
    if (!weights.empty()) {
      prediction_proto.set_weight(weights[i]);
    }
    prediction_proto.mutable_classification()->set_value(
        prediction >= 0.5f ? kPositiveValue : kNegativeValue);
    prediction_distribution.set_sum(1.f);
    prediction_distribution.set_counts(kNegativeValue, 1 - prediction);
    prediction_distribution.set_counts(kPositiveValue, prediction);
    RETURN_IF_ERROR(
        metric::AddPrediction(eval_options, prediction_proto, rnd, eval));
  }
  return absl::OkStatus();
}

absl::Status AddMulticlassClassificationPredictions(
    StridedSpanFloat32 raw_predictions, StridedSpanInt32 labels,
    StridedSpanFloat32 weights, const int num_classes,
    const metric::proto::EvaluationOptions& eval_options,
    utils::RandomEngine* rnd, metric::proto::EvaluationResults* eval) {
  DCHECK_EQ(raw_predictions.size(), num_classes * labels.size());
  model::proto::Prediction prediction_proto;
  auto& prediction_distribution =
      *prediction_proto.mutable_classification()->mutable_distribution();
  prediction_distribution.mutable_counts()->Resize(3, 0);
  utils::IntegerDistributionFloat distribution;
  distribution.SetNumClasses(num_classes + 1);
  for (int i = 0; i < labels.size(); ++i) {
    const int label = labels[i];
    if (label < 0 || label >= num_classes) {
      return absl::InvalidArgumentError(absl::Substitute(
          "Label values for multi-class classification must be integers "
          "in [0, num_classes-1], got $0 for example $1 with num_classes $2",
          label, i, num_classes));
    }
    prediction_proto.mutable_classification()->set_ground_truth(label + 1);

    distribution.Clear();
    for (int label_value = 1; label_value <= num_classes; label_value++) {
      const int score_col_idx = label_value - 1;
      const float probability =
          raw_predictions[i * num_classes + score_col_idx];
      if (probability < 0 || probability > 1) {
        return absl::InvalidArgumentError(
            absl::Substitute("Predictions values for multi-class "
                             "classification must be between 0 "
                             "or 1, got $0 for example $1",
                             probability, i));
      }
      distribution.Add(label_value, probability);
    }
    prediction_proto.mutable_classification()->set_value(
        distribution.TopClass());
    distribution.Save(
        prediction_proto.mutable_classification()->mutable_distribution());
    if (!weights.empty()) {
      prediction_proto.set_weight(weights[i]);
    }
    RETURN_IF_ERROR(
        metric::AddPrediction(eval_options, prediction_proto, rnd, eval));
  }
  return absl::OkStatus();
}

absl::Status AddRegressionPredictions(
    StridedSpanFloat32 raw_predictions, StridedSpanFloat32 labels,
    StridedSpanFloat32 weights,
    const metric::proto::EvaluationOptions& eval_options,
    utils::RandomEngine* rnd, metric::proto::EvaluationResults* eval) {
  DCHECK_EQ(raw_predictions.size(), labels.size());
  model::proto::Prediction prediction_proto;
  for (int i = 0; i < raw_predictions.size(); ++i) {
    prediction_proto.mutable_regression()->set_value(raw_predictions[i]);
    prediction_proto.mutable_regression()->set_ground_truth(labels[i]);
    if (!weights.empty()) {
      prediction_proto.set_weight(weights[i]);
    }
    RETURN_IF_ERROR(
        metric::AddPrediction(eval_options, prediction_proto, rnd, eval));
  }
  return absl::OkStatus();
}

absl::Status AddRankingPredictions(
    const StridedSpanFloat32 raw_predictions, const StridedSpanFloat32 labels,
    const StridedSpan<uint64_t> ranking_groups,
    const StridedSpanFloat32 weights,
    const metric::proto::EvaluationOptions& eval_options,
    utils::RandomEngine* rnd, metric::proto::EvaluationResults* eval) {
  DCHECK_EQ(raw_predictions.size(), labels.size());
  DCHECK_EQ(raw_predictions.size(), ranking_groups.size());
  model::proto::Prediction prediction_proto;
  for (int i = 0; i < raw_predictions.size(); ++i) {
    prediction_proto.mutable_ranking()->set_relevance(raw_predictions[i]);
    prediction_proto.mutable_ranking()->set_ground_truth_relevance(labels[i]);
    if (!weights.empty()) {
      prediction_proto.set_weight(weights[i]);
    }
    prediction_proto.mutable_ranking()->set_group_id(ranking_groups[i]);
    RETURN_IF_ERROR(
        metric::AddPrediction(eval_options, prediction_proto, rnd, eval));
  }
  return absl::OkStatus();
}

template <typename LabelType>
absl::Status AddNumpyPredictions(
    StridedSpanFloat32 raw_predictions, StridedSpan<LabelType> labels,
    const metric::proto::EvaluationOptions& eval_options,
    const dataset::proto::Column& label_column, StridedSpanFloat32 weights,
    StridedSpan<uint64_t> ranking_groups, utils::RandomEngine* rnd,
    metric::proto::EvaluationResults* eval);

template <>
absl::Status AddNumpyPredictions<int>(
    const StridedSpanFloat32 raw_predictions, const StridedSpanInt32 labels,
    const metric::proto::EvaluationOptions& eval_options,
    const dataset::proto::Column& label_column,
    const StridedSpanFloat32 weights,
    const StridedSpan<uint64_t> ranking_groups, utils::RandomEngine* rnd,
    metric::proto::EvaluationResults* eval) {
  DCHECK(weights.empty() || weights.size() == labels.size());
  switch (eval_options.task()) {
    case model::proto::Task::CLASSIFICATION: {
      DCHECK(label_column.has_categorical());
      DCHECK(label_column.categorical().items().contains(
          yggdrasil_decision_forests::dataset::kOutOfDictionaryItemKey));
      const int num_classes = label_column.categorical().items_size() - 1;
      DCHECK_GE(num_classes, 2);
      if (num_classes == 2) {
        return AddBinaryClassificationPredictions(
            raw_predictions, labels, weights, eval_options, rnd, eval);
      } else {
        return AddMulticlassClassificationPredictions(raw_predictions, labels,
                                                      weights, num_classes,
                                                      eval_options, rnd, eval);
      }
    }
    case model::proto::Task::REGRESSION:
    case model::proto::Task::RANKING:
    case model::proto::Task::ANOMALY_DETECTION:
      return absl::InvalidArgumentError(
          "Regression and ranking tasks require float labels.");
    default:
      return absl::InvalidArgumentError("Unsupported task type");
  }
}

template <>
absl::Status AddNumpyPredictions<float>(
    const StridedSpanFloat32 raw_predictions, const StridedSpanFloat32 labels,
    const metric::proto::EvaluationOptions& eval_options,
    const dataset::proto::Column& label_column,
    const StridedSpanFloat32 weights,
    const StridedSpan<uint64_t> ranking_groups, utils::RandomEngine* rnd,
    metric::proto::EvaluationResults* eval) {
  DCHECK(weights.empty() || weights.size() == labels.size());
  switch (eval_options.task()) {
    case model::proto::Task::REGRESSION:
      return AddRegressionPredictions(raw_predictions, labels, weights,
                                      eval_options, rnd, eval);

    case model::proto::Task::RANKING:
      return AddRankingPredictions(raw_predictions, labels, ranking_groups,
                                   weights, eval_options, rnd, eval);
    case model::proto::Task::CLASSIFICATION:
      return absl::InvalidArgumentError(
          "Classification tasks require float labels.");
    default:
      return absl::InvalidArgumentError("Unsupported task type");
  }
  return absl::OkStatus();
}

absl::StatusOr<dataset::proto::Column> CreateLabelColumn(
    const model::proto::Task task,
    const std::optional<std::vector<std::string>>& label_classes) {
  dataset::proto::Column label_column;
  label_column.set_name("Label");
  if (task == model::proto::Task::CLASSIFICATION) {
    if (!label_classes.has_value()) {
      return absl::InvalidArgumentError(
          "Label names are required for classification tasks");
    }
    label_column.set_type(dataset::proto::ColumnType::CATEGORICAL);
    auto& categorical_label = *label_column.mutable_categorical();
    categorical_label.set_number_of_unique_values(label_classes->size() + 1);
    (*categorical_label.mutable_items())
        [yggdrasil_decision_forests::dataset::kOutOfDictionaryItemKey]
            .set_index(0);
    for (int i = 0; i < label_classes->size(); ++i) {
      (*categorical_label.mutable_items())[(*label_classes)[i]].set_index(i +
                                                                          1);
    }
  } else if (task == model::proto::Task::CATEGORICAL_UPLIFT) {
    label_column.set_type(dataset::proto::ColumnType::CATEGORICAL);
  } else {
    DCHECK(!label_classes.has_value());
    label_column.set_type(dataset::proto::ColumnType::NUMERICAL);
  }
  return label_column;
}

template <typename LabelType>
absl::StatusOr<metric::proto::EvaluationResults> EvaluateRawPredictions(
    const StridedSpanFloat32 raw_predictions,
    const StridedSpan<LabelType> labels,
    const metric::proto::EvaluationOptions& eval_options,
    const StridedSpanFloat32 weights,
    const std::optional<std::vector<std::string>>& label_classes,
    const StridedSpan<uint64_t> ranking_groups, const int64_t random_seed) {
  metric::proto::EvaluationResults eval;
  utils::RandomEngine rnd(random_seed);

  ASSIGN_OR_RETURN(const dataset::proto::Column label_column,
                   CreateLabelColumn(eval_options.task(), label_classes));
  RETURN_IF_ERROR(
      metric::InitializeEvaluation(eval_options, label_column, &eval));
  RETURN_IF_ERROR(AddNumpyPredictions(raw_predictions, labels, eval_options,
                                      label_column, weights, ranking_groups,
                                      &rnd, &eval));
  RETURN_IF_ERROR(
      metric::FinalizeEvaluation(eval_options, label_column, &eval));
  return eval;
}

// Evaluate Predictions against labels.
//
// For binary classification, regression and ranking tasks, `predictions` must
// be a 1D array. For multiclass classification, `predictions` must be a 1D
// array flattened from a 2d array with C-ordering (last axis index changing
// fastest, back to the first axis index changing slowest).
//
// If `weights` is empty, unit weights are used. `ranking_groups` must be empty
// for non-ranking tasks and cannot be empty for ranking tasks.
template <typename LabelType>
absl::StatusOr<metric::proto::EvaluationResults> EvaluatePredictions(
    py::array_t<float>& predictions, py::array_t<LabelType>& labels,
    const metric::proto::EvaluationOptions& options,
    py::array_t<float>& weights,
    std::optional<std::vector<std::string>> label_classes,
    py::array_t<uint64_t>& ranking_groups, const int64_t random_seed) {
  STATUS_CHECK(predictions.ndim() == 1);
  DCHECK(ranking_groups.size() == 0 ||
         options.task() == model::proto::Task::RANKING);
  py::gil_scoped_release release;
  StridedSpanFloat32 prediction_values(predictions);
  StridedSpan<LabelType> labels_values(labels);
  StridedSpan<uint64_t> ranking_groups_values(ranking_groups);
  StridedSpanFloat32 weights_values(weights);
  return EvaluateRawPredictions(prediction_values, labels_values, options,
                                weights_values, label_classes,
                                ranking_groups_values, random_seed);
}

}  // namespace

void init_evaluate(py::module_& m) {
  // WARNING: This method releases the Global Interpreter Lock.
  m.def("EvaluatePredictions", WithStatusOr(EvaluatePredictions<int>),
        py::arg("predictions").noconvert(true),
        py::arg("labels").noconvert(true), py::arg("options"),
        py::arg("weights").noconvert(true), py::arg("label_classes"),
        py::arg("ranking_groups").noconvert(true), py::arg("random_seed"));
  // WARNING: This method releases the Global Interpreter Lock.
  m.def("EvaluatePredictions", WithStatusOr(EvaluatePredictions<float>),
        py::arg("predictions").noconvert(true),
        py::arg("labels").noconvert(true), py::arg("options"),
        py::arg("weights").noconvert(true), py::arg("label_classes"),
        py::arg("ranking_groups").noconvert(true), py::arg("random_seed"));
}

}  // namespace yggdrasil_decision_forests::port::python
