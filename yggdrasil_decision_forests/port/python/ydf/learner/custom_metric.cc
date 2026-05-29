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
#include <exception>
#include <memory>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_interface.h"
#include "ydf/utils/numpy_data.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests::port::python {

// TODO: b/514306997 - Move helpers to a shared file.
namespace {

// Holder for objects that require the Python GIL to be held upon destruction.
// Make sure any object wrapped by this is destroyed only when the GIL can be
// acquired.
template <typename Func>
std::shared_ptr<Func> MakeSafeGilHolder(const Func& func) {
  return std::shared_ptr<Func>(new Func(func), [](Func* ptr) {
    if (ptr) {
      py::gil_scoped_acquire acquire;
      delete ptr;
    }
  });
}

}  // namespace

absl::StatusOr<model::gradient_boosted_trees::CustomMetric>
CCBinaryClassificationMetric::ToCustomMetric() const {
  // Wrap the Python functions in a std::shared_ptr with a custom deleter
  // that acquires the GIL before deleting the underlying Python function.
  // This avoids dangling references and ensures safe out-of-scope destruction.
  auto safe_evaluation_func = MakeSafeGilHolder(evaluation_func);
  auto cc_evaluation_func =
      [safe_evaluation_func](
          const absl::Span<const float>& predictions,
          const absl::Span<const int32_t>& labels,
          const absl::Span<const float>& weights) -> absl::StatusOr<float> {
    float current_metric;
    {
      py::gil_scoped_acquire acquire;
      auto pylabels = SpanToUnsafeNumpyArray(labels);
      auto pyweights = SpanToUnsafeNumpyArray(weights);
      auto pypredictions = SpanToUnsafeNumpyArray(predictions);
      try {
        current_metric =
            (*safe_evaluation_func)(pylabels, pypredictions, pyweights);
      } catch (const std::exception& e) {
        return absl::UnknownError(e.what());
      }
    }
    return current_metric;
  };
  return model::gradient_boosted_trees::CustomMetric{name, cc_evaluation_func};
}

absl::StatusOr<model::gradient_boosted_trees::CustomMetric>
CCMultiClassificationMetric::ToCustomMetric() const {
  // Wrap the Python functions in a std::shared_ptr with a custom deleter
  // that acquires the GIL before deleting the underlying Python function.
  // This avoids dangling references and ensures safe out-of-scope destruction.
  auto safe_evaluation_func = MakeSafeGilHolder(evaluation_func);
  auto cc_evaluation_func =
      [safe_evaluation_func](
          const absl::Span<const float>& predictions,
          const absl::Span<const int32_t>& labels,
          const absl::Span<const float>& weights) -> absl::StatusOr<float> {
    int num_examples = labels.size();
    DCHECK_GT(num_examples, 0);
    int dimension = predictions.size() / num_examples;
    float current_metric;
    {
      py::gil_scoped_acquire acquire;
      auto pylabels = SpanToUnsafeNumpyArray(labels);
      auto pyweights = SpanToUnsafeNumpyArray(weights);
      auto pypredictions = SpanToUnsafeNumpyArray(predictions);
      pypredictions = pypredictions.reshape({num_examples, dimension});
      try {
        current_metric =
            (*safe_evaluation_func)(pylabels, pypredictions, pyweights);
      } catch (const std::exception& e) {
        return absl::UnknownError(e.what());
      }
    }
    return current_metric;
  };
  return model::gradient_boosted_trees::CustomMetric{name, cc_evaluation_func};
}

absl::StatusOr<model::gradient_boosted_trees::CustomMetric>
CCRegressionMetric::ToCustomMetric() const {
  // Wrap the Python functions in a std::shared_ptr with a custom deleter
  // that acquires the GIL before deleting the underlying Python function.
  // This avoids dangling references and ensures safe out-of-scope destruction.
  auto safe_evaluation_func = MakeSafeGilHolder(evaluation_func);
  auto cc_evaluation_func =
      [safe_evaluation_func](
          const absl::Span<const float>& predictions,
          const absl::Span<const float>& labels,
          const absl::Span<const float>& weights) -> absl::StatusOr<float> {
    float current_metric;
    {
      py::gil_scoped_acquire acquire;
      auto pylabels = SpanToUnsafeNumpyArray(labels);
      auto pyweights = SpanToUnsafeNumpyArray(weights);
      auto pypredictions = SpanToUnsafeNumpyArray(predictions);
      try {
        current_metric =
            (*safe_evaluation_func)(pylabels, pypredictions, pyweights);
      } catch (const std::exception& e) {
        return absl::UnknownError(e.what());
      }
    }
    return current_metric;
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
