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

#include "yggdrasil_decision_forests/utils/feature_importance.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/random.h"

namespace yggdrasil_decision_forests {
namespace utils {

dataset::VerticalDataset ShuffleDatasetColumns(
    const dataset::VerticalDataset& dataset,
    const std::vector<int>& shuffle_column_idxs, utils::RandomEngine* rnd) {
  auto permuted_dataset = dataset.ShallowNonOwningClone();
  for (auto& column_idx : shuffle_column_idxs) {
    auto* dst_permuted_column =
        permuted_dataset
            .ReplaceColumn(column_idx, dataset.data_spec().columns(column_idx))
            .value();
    const auto src_column = dataset.column(column_idx);

    // Compute the permutation of the indices.
    std::vector<dataset::VerticalDataset::row_t> permited_indices(
        dataset.nrow());
    std::iota(permited_indices.begin(), permited_indices.end(), 0);
    std::shuffle(permited_indices.begin(), permited_indices.end(), *rnd);

    // Permute the values.
    for (dataset::VerticalDataset::row_t example_idx = 0;
         example_idx < dataset.nrow(); example_idx++) {
      dataset::proto::Example::Attribute value;
      src_column->ExtractExample(example_idx, &value);
      dst_permuted_column->Set(permited_indices[example_idx], value);
    }
  }
  return permuted_dataset;
}

void ComputePermutationFeatureImportance(
    const metric::proto::EvaluationResults& base_evaluation,
    const std::function<
        absl::optional<metric::proto::EvaluationResults>(const int feature_idx)>
        get_permutation_evaluation,
    model::AbstractModel* model, int num_rounds) {
  const auto metrics =
      metric::DefaultMetrics(model->task(), model->label_col_spec());

  // Generate the permutation feature importance names.
  std::vector<std::string> importance_names;
  importance_names.reserve(metrics.size());
  for (const auto& metric : metrics) {
    importance_names.push_back(absl::Substitute(
        "MEAN_$0_IN_$1", metric.higher_is_better ? "DECREASE" : "INCREASE",
        metric.name));
  }

  // Evaluate the impact of permuting each feature on each metric.
  for (int feature_idx = 0; feature_idx < model->data_spec().columns_size();
       feature_idx++) {
    std::vector<metric::proto::EvaluationResults> permuted_evaluations;
    for (int round_idx = 0; round_idx < num_rounds; round_idx++) {
      auto permuted_evaluation = get_permutation_evaluation(feature_idx);
      if (!permuted_evaluation.has_value()) {
        continue;
      }
      permuted_evaluations.push_back(std::move(permuted_evaluation).value());
    }

    if (permuted_evaluations.empty()) {
      // The feature is not used by the model.
      continue;
    }

    for (int metric_idx = 0; metric_idx < metrics.size(); metric_idx++) {
      const auto metric = metrics[metric_idx];
      const auto baseline_metric_value =
          metric::GetMetric(base_evaluation, metric.accessor);

      double sum_permuted_metric_value = 0;
      for (const auto& permuted_evaluation : permuted_evaluations) {
        sum_permuted_metric_value +=
            metric::GetMetric(permuted_evaluation, metric.accessor);
      }
      const auto permuted_metric_value =
          sum_permuted_metric_value / permuted_evaluations.size();

      auto& feature_importance =
          (*model->mutable_precomputed_variable_importances())
              [importance_names[metric_idx]];
      auto& feature = *feature_importance.mutable_variable_importances()->Add();
      feature.set_importance((baseline_metric_value - permuted_metric_value) *
                             (metric.higher_is_better ? 1.f : -1.f));
      feature.set_attribute_idx(feature_idx);
    }
  }

  // Sort the importance by decreasing order.
  const auto var_importance_comparer =
      [](const model::proto::VariableImportance& a,
         const model::proto::VariableImportance& b) {
        return a.importance() > b.importance();
      };

  for (const auto& importance_name : importance_names) {
    auto& feature_importance =
        (*model->mutable_precomputed_variable_importances())[importance_name];
    std::sort(feature_importance.mutable_variable_importances()->begin(),
              feature_importance.mutable_variable_importances()->end(),
              var_importance_comparer);
  }
}

absl::Status ComputePermutationFeatureImportance(
    const dataset::VerticalDataset& dataset, model::AbstractModel* model,
    int num_rounds) {
  utils::RandomEngine rnd_permutation_vi;

  // Setup the evaluation configuration.
  metric::proto::EvaluationOptions eval_options;
  eval_options.set_bootstrapping_samples(0);
  eval_options.set_task(model->task());

  const auto base_evaluation =
      model->Evaluate(dataset, eval_options, &rnd_permutation_vi);

  const auto permutation_evaluation = [&](const int feature_idx)
      -> absl::optional<metric::proto::EvaluationResults> {
    const auto it_input_feature =
        std::find(model->input_features().begin(),
                  model->input_features().end(), feature_idx);
    if (it_input_feature == model->input_features().end()) {
      return {};
    }
    const auto perturbed_dataset = utils::ShuffleDatasetColumns(
        dataset, {feature_idx}, &rnd_permutation_vi);
    return model->Evaluate(perturbed_dataset, eval_options,
                           &rnd_permutation_vi);
  };

  utils::ComputePermutationFeatureImportance(
      base_evaluation, permutation_evaluation, model, num_rounds);
  return absl::OkStatus();
}

}  // namespace utils
}  // namespace yggdrasil_decision_forests
