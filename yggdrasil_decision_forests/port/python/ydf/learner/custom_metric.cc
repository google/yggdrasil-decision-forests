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

#include "ydf/learner/custom_metric.h"

#include <pybind11/numpy.h>

#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_interface.h"
#include "ydf/utils/numpy_data.h"
#include "ydf/utils/pybind.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests::port::python {

absl::StatusOr<model::gradient_boosted_trees::CustomMetric>
CCBinaryClassificationMetric::ToCustomMetric() const {
  auto safe_evaluation_func = MakeSafeGilHolder(evaluation_func);
  auto cc_evaluation_func =
      [safe_evaluation_func](
          const absl::Span<const int32_t>& labels,
          const absl::Span<const float>& predictions,
          const absl::Span<const float>& weights) -> absl::StatusOr<float> {
    return SafePythonCall<absl::StatusOr<float>>(
        "evaluation_func", [&]() -> absl::StatusOr<float> {
          auto pylabels = SpanToSafeCopy(labels);
          auto pyweights = SpanToSafeCopy(weights);
          auto pypredictions = SpanToSafeCopy(predictions);
          return (*safe_evaluation_func)(pylabels, pypredictions, pyweights);
        });
  };
  return model::gradient_boosted_trees::CustomMetric{name, cc_evaluation_func};
}

absl::StatusOr<model::gradient_boosted_trees::CustomMetric>
CCMultiClassificationMetric::ToCustomMetric() const {
  auto safe_evaluation_func = MakeSafeGilHolder(evaluation_func);
  auto cc_evaluation_func =
      [safe_evaluation_func](
          const absl::Span<const int32_t>& labels,
          const absl::Span<const float>& predictions,
          const absl::Span<const float>& weights) -> absl::StatusOr<float> {
    int num_examples = labels.size();
    DCHECK_GT(num_examples, 0);
    int dimension = predictions.size() / num_examples;
    return SafePythonCall<absl::StatusOr<float>>(
        "evaluation_func", [&]() -> absl::StatusOr<float> {
          auto pylabels = SpanToSafeCopy(labels);
          auto pyweights = SpanToSafeCopy(weights);
          auto pypredictions = SpanToSafeCopy(predictions);
          pypredictions = pypredictions.reshape({num_examples, dimension});
          return (*safe_evaluation_func)(pylabels, pypredictions, pyweights);
        });
  };
  return model::gradient_boosted_trees::CustomMetric{name, cc_evaluation_func};
}

absl::StatusOr<model::gradient_boosted_trees::CustomMetric>
CCRegressionMetric::ToCustomMetric() const {
  auto safe_evaluation_func = MakeSafeGilHolder(evaluation_func);
  auto cc_evaluation_func =
      [safe_evaluation_func](
          const absl::Span<const float>& labels,
          const absl::Span<const float>& predictions,
          const absl::Span<const float>& weights) -> absl::StatusOr<float> {
    return SafePythonCall<absl::StatusOr<float>>(
        "evaluation_func", [&]() -> absl::StatusOr<float> {
          auto pylabels = SpanToSafeCopy(labels);
          auto pyweights = SpanToSafeCopy(weights);
          auto pypredictions = SpanToSafeCopy(predictions);
          return (*safe_evaluation_func)(pylabels, pypredictions, pyweights);
        });
  };
  return model::gradient_boosted_trees::CustomMetric{name, cc_evaluation_func};
}

absl::Status ApplyCustomMetric(
    const std::vector<CCCustomMetric>& custom_metrics,
    model::AbstractLearner* learner) {
  if (custom_metrics.empty()) {
    return absl::OkStatus();
  }
  auto* gbt_learner =
      dynamic_cast<model::gradient_boosted_trees::GradientBoostedTreesLearner*>(
          learner);
  if (!gbt_learner) {
    return absl::InvalidArgumentError(
        "Custom metrics are only compatible with Gradient Boosted Trees.");
  }

  std::vector<model::gradient_boosted_trees::CustomMetric>
      custom_metrics_internal;

  // TODO: b/516878946 - revise implementation once C++20 is supported.
  for (const auto& custom_metric : custom_metrics) {
    auto status_or_metric = std::visit(
        [](const auto& metric)
            -> absl::StatusOr<model::gradient_boosted_trees::CustomMetric> {
          if constexpr (std::is_same_v<std::decay_t<decltype(metric)>,
                                       std::monostate>) {
            return absl::InvalidArgumentError(
                "Unexpected monostate in custom_metric");
          } else {
            return metric.ToCustomMetric();
          }
        },
        custom_metric);
    ASSIGN_OR_RETURN(auto metric_func, std::move(status_or_metric));
    custom_metrics_internal.push_back(std::move(metric_func));
  }

  if (!custom_metrics_internal.empty()) {
    gbt_learner->SetCustomMetrics(custom_metrics_internal);
  }

  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests::port::python
