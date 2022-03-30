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

// Utility class to compute model performance metrics.

#ifndef YGGDRASIL_DECISION_FORESTS_METRIC_METRIC_H_
#define YGGDRASIL_DECISION_FORESTS_METRIC_METRIC_H_

#include <functional>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/random.h"

namespace yggdrasil_decision_forests {
namespace metric {

// Initialize the evaluation. Should be called before any "AddPrediction".
void InitializeEvaluation(const proto::EvaluationOptions& option,
                          const dataset::proto::Column& label_column,
                          proto::EvaluationResults* eval);

// Add a prediction to the evaluation. The predictions should contain the label
// / ground truth (see the "SetGroundTruth" methods in "abstract_model.h").
void AddPrediction(const proto::EvaluationOptions& option,
                   const model::proto::Prediction& pred,
                   utils::RandomEngine* rnd, proto::EvaluationResults* eval);

// Merge two initialized (with the same options) and non-finalized evaluations.
void MergeEvaluation(const proto::EvaluationOptions& option,
                     const proto::EvaluationResults& src,
                     proto::EvaluationResults* dst);

// Finalize the evaluation. Only then the metrics can be read from the
// evaluation. Should be called after all "AddPrediction".
void FinalizeEvaluation(const proto::EvaluationOptions& option,
                        const dataset::proto::Column& label_column,
                        proto::EvaluationResults* eval);

// Helper function to evaluate binary classification predictions.
//
// Args:
//   labels: Labels. True=positive, False=negative.
//   predictions: Predicted probability of the positive label.
//   option: Evaluation options.
//   rnd: Random generator. Used for bootstrapping and prediction sampling.
//   positive_label: Representation for the positive label.
//   negative_label: Representation for the negative label.
//
// Returns:
//   Evaluation results. See the test "BinaryClassificationEvaluationHelper" for
//   examples of usage of this object.
//
proto::EvaluationResults BinaryClassificationEvaluationHelper(
    const std::vector<bool>& labels, const std::vector<float>& predictions,
    const proto::EvaluationOptions& option, utils::RandomEngine* rnd,
    absl::string_view positive_label = "Positive",
    absl::string_view negative_label = "Negative");

// Compute a "one-vs-other" Receiver Operating Characteristic curve (ROC curve).
// Args:
//    options: The option used to compute "eval".
//    label_column: The dataspec of the label column.
//    eval: Initialized, filled and finalized evaluation.
//    positive_label_value: The value of the "positive" label. All other label
//      values are considered "negative".
//    roc: The output ROC.
void BuildROCCurve(const proto::EvaluationOptions& option,
                   const dataset::proto::Column& label_column,
                   const proto::EvaluationResults& eval,
                   const int positive_label_value, proto::Roc* roc);

// Compute the "X@Y" metrics e.g. recall at given precision or precision at
// given volume. Metrics are computed without interpolation:
//   - Precision @ Recall : Precision with highest threshold such the recall is
//     greater or equal to the limit. Note: Precision is not monotonic with
//     threshold value.
//   - Precision @ Volume : Precision with highest threshold such the volume is
//     greater or equal to the limit.
//   - Recall @ Precision : Highest recall with precision greater or equal to
//     the limit. Note: Recall is monotonic with threshold value.
//   - Recall @ False Positive Rate: Highest recall with false positive rate
//     less or equal to the limit. Note: Recall and FPR are monotonic positive
//     to each other.
//   - False positive rate @ Recall: Smallest (best) false positive rate with
//     recall greater or equal to the limit.
void ComputeXAtYMetrics(
    const proto::EvaluationOptions& option,
    const google::protobuf::RepeatedPtrField<proto::Roc::Point>& curve, proto::Roc* roc);

// Extract key metrics from an evaluation.
// The metric are defined in "metric.proto".

float Accuracy(const proto::EvaluationResults& eval);
float LogLoss(const proto::EvaluationResults& eval);
float RMSE(const proto::EvaluationResults& eval);
float ErrorRate(const proto::EvaluationResults& eval);

// Loss of the model. Can have different semantic for different models.
float Loss(const proto::EvaluationResults& eval);

// Normalized Discounted Cumulative Gain.
float NDCG(const proto::EvaluationResults& eval);

// Mean Reciprocal Rank.
float MRR(const proto::EvaluationResults& eval);

// Equivalent to the fraction of examples were the highest predicted example is
// also the example with the highest relevance value.
float PrecisionAt1(const proto::EvaluationResults& eval);

// Area under the uplift curve.
// Use trapezoidal AUC integration. Examples with similar effects are merged
// together i.e. the order of the examples have not effect on the metric.
double AUUC(const proto::EvaluationResults& eval);

// Qini score : AUUC offset such that a random model gets a Qini score of 0.
double Qini(const proto::EvaluationResults& eval);

// Confidence interval on the Accuracy using the Wilson score interval with
// continuity correction.
std::pair<float, float> AccuracyConfidenceInterval(
    const proto::EvaluationResults& eval, float confidence_level = 0.95f);

// Confidence interval on the AUC as proposed by Hanley et al, "The Meaning and
// Use of the Area under a Receiver Operating Characteristic (ROC) Curve".
std::pair<float, float> AUCConfidenceInterval(const proto::Roc& roc,
                                              float confidence_level = 0.95f);

// Confidence interval of the PR AUC using the logit interval described and
// evaluated in "Area Under the Precision-Recall Curve: Point Estimates and
// Confidence Intervals" (http://pages.cs.wisc.edu/~boyd/aucpr_final.pdf).
std::pair<float, float> PRAUCConfidenceInterval(const proto::Roc& roc,
                                                float confidence_level = 0.95);

// Closed form confidence interval of the RMSE. Assume for the residual
// (i.e. ground_truth - predicted_value) to be sampled from an unbiased normal
// distribution.
std::pair<float, float> RMSEConfidenceInterval(
    const proto::EvaluationResults& eval, float confidence_level = 0.95);

// Performance of the default predictor i.e. the predictor that always returns
// the same most likely answer. Note: The most likely answer is computed in
// the test dataset.
float DefaultAccuracy(const proto::EvaluationResults& eval);
float DefaultLogLoss(const proto::EvaluationResults& eval);
float DefaultErrorRate(const proto::EvaluationResults& eval);
float DefaultRMSE(const proto::EvaluationResults& eval);
float DefaultNDCG(const proto::EvaluationResults& eval);

// Export a set of metrics from a model evaluation.
std::unordered_map<std::string, std::string> ExtractFlatMetrics(
    absl::string_view model_name, const proto::EvaluationResults& evaluation);

// Export a set of metrics from a model evaluation stored in a file. The file
// should store a serialized binary proto::EvaluationResults.
utils::StatusOr<std::unordered_map<std::string, std::string>>
ExtractFlatMetrics(absl::string_view model_name,
                   absl::string_view evaluation_file);

// List of metric names and accessors for the X@Y metrics.
struct XAtYAccessor {
  std::string x_name;
  std::string y_name;
  std::function<const google::protobuf::RepeatedPtrField<proto::Roc::XAtYMetric>&(
      const proto::Roc& roc)>
      const_access;
  std::function<google::protobuf::RepeatedPtrField<proto::Roc::XAtYMetric>*(
      proto::Roc* roc)>
      mutable_access;
};

// Lists the accessors for the X@Y metrics.
std::vector<XAtYAccessor> XAtYMetricsAccessors();

// Prediction and the label for a binary classifier.
struct BinaryPrediction {
  // Prediction that the label is true. This field does not have to be a
  // probability.
  float predict_true;
  // Value of the label.
  bool label;
  // Weight of the example.
  float weight;
};

// Create a string "[Metric] ([label value] vs the other)". For example: "AUC
// (Class1 vs the others)".
std::string GetPerClassComparisonMetricLabel(
    const proto::EvaluationResults& eval, int label_value,
    absl::string_view metric);

// Returns the numerical metric value defined by "metric" and contained in
// "evaluation".
double GetMetric(const proto::EvaluationResults& evaluation,
                 const proto::MetricAccessor& metric);

// If true, a higher value for the metric is generally preferable (e.g.
// accuracy). If false, a lower value is preferable (e.g. loss). Fails if
// unknown.
utils::StatusOr<bool> HigherIsBetter(const proto::MetricAccessor& metric);

// Computes the minimum and maximum value of a stream of values.
template <typename T>
class MinMaxStream {
 public:
  // Visit a new value i.e. updates the bounds.
  void visit(const T& value);

  // Return true iif no value were visited.
  bool empty() const { return empty_; }

  // Minimum and maximum of the visited values.
  const T& min() const {
    CHECK(!empty_);
    return min_;
  }
  const T& max() const {
    CHECK(!empty_);
    return max_;
  }

 private:
  bool empty_ = true;
  T min_;
  T max_;
};

template <typename T>
void MinMaxStream<T>::visit(const T& value) {
  CHECK(!std::isnan(value));
  if (empty_) {
    empty_ = false;
    min_ = max_ = value;
  } else {
    if (value < min_) {
      min_ = value;
    }
    if (value > max_) {
      max_ = value;
    }
  }
}

// Description of a metric as returned by DefaultMetrics.
struct MetricDefinition {
  std::string name;
  proto::MetricAccessor accessor;
  bool higher_is_better;
  // Does the metrics requires sampling of predictions for computation.
  bool require_sampling = false;
};

// Returns the list of basic metrics that make sense to display.
std::vector<MetricDefinition> DefaultMetrics(
    model::proto::Task task, const dataset::proto::Column& label);

// Computes the RMSE of a set of predictions.
double RMSE(const std::vector<float>& labels,
            const std::vector<float>& predictions,
            const std::vector<float>& weights);
double RMSE(const std::vector<float>& labels,
            const std::vector<float>& predictions);

// Gets the threshold on a binary classifier output that maximize accuracy.
float ComputeThresholdForMaxAccuracy(
    const google::protobuf::RepeatedPtrField<proto::Roc::Point>& curve);

// Computes the p-value of the test:
//   h0: mean <= 0
//   h1: mean > 0
// See https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm for more
// details.
float PValueMeanIsGreaterThanZero(const std::vector<float>& sample);

namespace internal {
// Bootstrap the performance metrics for classification.
void ComputeRocConfidenceIntervalsUsingBootstrapping(
    const proto::EvaluationOptions& option,
    const std::vector<BinaryPrediction>& sorted_predictions, proto::Roc* roc);

// Get two quantiles from a set of observations extracted from a vector of
// proto::Roc.
std::pair<double, double> GetQuantiles(
    const std::vector<proto::Roc>& samples,
    const std::function<double(const proto::Roc&)>& getter, float quantile_1,
    float quantile_2);

// Bootstrap the performance metrics for regression. Note: "eval" is both the
// input and the output.
void UpdateRMSEConfidenceIntervalUsingBootstrapping(
    const proto::EvaluationOptions& option, proto::EvaluationResults* eval);

// ROC metrics from a ROC point.
double RocSum(const proto::Roc::Point& point);
double RocFPR(const proto::Roc::Point& point);
double RocTPR(const proto::Roc::Point& point);
double RocAccuracy(const proto::Roc::Point& point);
double RocPrecision(const proto::Roc::Point& point);
// Ratio of positives i.e. percentage of examples classified as positive by the
// model. Also called "volume".
double RocPositiveRatio(const proto::Roc::Point& point);

}  // namespace internal

}  // namespace metric
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_METRIC_METRIC_H_
