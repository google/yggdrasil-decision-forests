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

#include "yggdrasil_decision_forests/utils/feature_importance.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/random.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace {

// Generates the permutation feature importance names.
std::vector<std::string> BuildPermutationVariableImportanceName(
    const std::vector<metric::MetricDefinition>& metrics) {
  std::vector<std::string> importance_names;
  importance_names.reserve(metrics.size());
  for (const auto& metric : metrics) {
    importance_names.push_back(absl::Substitute(
        "MEAN_$0_IN_$1", metric.higher_is_better ? "DECREASE" : "INCREASE",
        metric.name));
  }
  return importance_names;
}

// Sort the variable importances by importance.
void SortVariableImportance(const std::vector<std::string>& importance_names,
                            ResultFeatureImportance* output) {
  // Sort the importance by decreasing order.
  const auto var_importance_comparer =
      [](const model::proto::VariableImportance& a,
         const model::proto::VariableImportance& b) {
        return a.importance() > b.importance();
      };

  for (const auto& importance_name : importance_names) {
    auto& feature_importance = (*output)[importance_name];
    std::sort(feature_importance.mutable_variable_importances()->begin(),
              feature_importance.mutable_variable_importances()->end(),
              var_importance_comparer);
  }
}

}  // namespace

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
    std::vector<dataset::VerticalDataset::row_t> permitted_indices(
        dataset.nrow());
    std::iota(permitted_indices.begin(), permitted_indices.end(), 0);
    std::shuffle(permitted_indices.begin(), permitted_indices.end(), *rnd);

    // Permute the values.
    for (dataset::VerticalDataset::row_t example_idx = 0;
         example_idx < dataset.nrow(); example_idx++) {
      dataset::proto::Example::Attribute value;
      src_column->ExtractExample(example_idx, &value);
      dst_permuted_column->Set(permitted_indices[example_idx], value);
    }
  }
  return permuted_dataset;
}

absl::Status ComputePermutationFeatureImportance(
    const metric::proto::EvaluationResults& base_evaluation,
    const std::function<
        absl::StatusOr<absl::optional<metric::proto::EvaluationResults>>(
            const int feature_idx)>& get_permutation_evaluation,
    const model::AbstractModel* model, ResultFeatureImportance* output,
    const ComputeFeatureImportanceOptions& options) {
  const auto metrics =
      metric::DefaultMetrics(model->task(), model->label_col_spec());

  std::vector<std::string> importance_names =
      BuildPermutationVariableImportanceName(metrics);

  // Baseline metrics.
  std::vector<double> baseline_metric_values(metrics.size());
  for (int metric_idx = 0; metric_idx < metrics.size(); metric_idx++) {
    const auto metric = metrics[metric_idx];
    ASSIGN_OR_RETURN(const auto baseline_metric_value,
                     metric::GetMetric(base_evaluation, metric.accessor));
    baseline_metric_values[metric_idx] = baseline_metric_value;
  }

  // sum_permutation_metrics[i][j] is the sum of the metrics (across the
  // repetitions) for metric i and feature j.
  const int num_features = model->data_spec().columns_size();
  std::vector<std::vector<double>> sum_permutation_metrics(metrics.size());
  std::vector<std::vector<int>> count_permutation_metrics(metrics.size());
  for (auto& item : sum_permutation_metrics) {
    item.assign(num_features, 0.0);
  }
  for (auto& item : count_permutation_metrics) {
    item.assign(num_features, 0);
  }

  // "data_mutex" protects the fields:
  // - sum_permutation_metrics
  // - count_permutation_metrics
  utils::concurrency::Mutex data_mutex;

  // Compute the permutation variable importances.
  const auto process_return_status =
      [&metrics, &get_permutation_evaluation, &data_mutex,
       &sum_permutation_metrics,
       &count_permutation_metrics](const int feature_idx) {
        ASSIGN_OR_RETURN(auto permuted_evaluation,
                         get_permutation_evaluation(feature_idx));
        if (!permuted_evaluation.has_value()) {
          return absl::OkStatus();
        }

        utils::concurrency::MutexLock lock(&data_mutex);
        for (int metric_idx = 0; metric_idx < metrics.size(); metric_idx++) {
          const auto metric = metrics[metric_idx];
          ASSIGN_OR_RETURN(
              auto value,
              metric::GetMetric(permuted_evaluation.value(), metric.accessor));
          sum_permutation_metrics[metric_idx][feature_idx] += value;
          count_permutation_metrics[metric_idx][feature_idx]++;
        }
        return absl::OkStatus();
      };

  // "status_mutex" protects the fields:
  // - status
  absl::Status status;
  utils::concurrency::Mutex status_mutex;

  const auto process = [&](const int feature_idx) {
    {
      utils::concurrency::MutexLock lock(&status_mutex);
      if (!status.ok()) {
        // One of the previous job has already fail. Skip all the remaining
        // jobs.
        return;
      }
    }
    auto sub_status = process_return_status(feature_idx);
    if (!sub_status.ok()) {
      utils::concurrency::MutexLock lock(&status_mutex);
      status.Update(sub_status);
    }
  };

  {
    utils::concurrency::ThreadPool pool("variable_importance",
                                        options.num_threads);
    pool.StartWorkers();
    YDF_LOG(INFO) << "Running " << model->data_spec().columns_size()
                  << " features on " << options.num_threads << " threads with "
                  << options.num_rounds << " rounds";
    for (int feature_idx = 0; feature_idx < model->data_spec().columns_size();
         feature_idx++) {
      for (int round_idx = 0; round_idx < options.num_rounds; round_idx++) {
        pool.Schedule([feature_idx, &process]() { process(feature_idx); });
      }
    }
  }

  // Compute the effective results from the computation results.
  for (int feature_idx = 0; feature_idx < model->data_spec().columns_size();
       feature_idx++) {
    for (int metric_idx = 0; metric_idx < metrics.size(); metric_idx++) {
      const auto count = count_permutation_metrics[metric_idx][feature_idx];
      if (count == 0) {
        continue;
      }
      const auto sum = sum_permutation_metrics[metric_idx][feature_idx];
      const auto permuted_metric_value = sum / count;

      auto& feature_importance = (*output)[importance_names[metric_idx]];
      auto& feature = *feature_importance.mutable_variable_importances()->Add();
      feature.set_importance(
          (baseline_metric_values[metric_idx] - permuted_metric_value) *
          (metrics[metric_idx].higher_is_better ? 1.f : -1.f));
      feature.set_attribute_idx(feature_idx);
    }
  }

  SortVariableImportance(importance_names, output);
  return absl::OkStatus();
}

absl::Status ComputePermutationFeatureImportance(
    const dataset::VerticalDataset& dataset, const model::AbstractModel* model,
    ResultFeatureImportance* output,
    const ComputeFeatureImportanceOptions& options) {
  // Setup the evaluation configuration.
  metric::proto::EvaluationOptions eval_options;
  eval_options.set_bootstrapping_samples(0);
  eval_options.set_task(model->task());

  utils::RandomEngine rng;
  utils::concurrency::Mutex rng_mutex;

  const auto base_evaluation = model->Evaluate(dataset, eval_options, &rng);

  const auto permutation_evaluation = [&dataset, &eval_options, &rng,
                                       &rng_mutex, model](const int feature_idx)
      -> absl::optional<metric::proto::EvaluationResults> {
    const auto it_input_feature =
        std::find(model->input_features().begin(),
                  model->input_features().end(), feature_idx);
    if (it_input_feature == model->input_features().end()) {
      return {};
    }
    utils::RandomEngine sub_rng;
    {
      utils::concurrency::MutexLock lock(&rng_mutex);
      sub_rng.seed(rng());
    }
    const auto perturbed_dataset =
        utils::ShuffleDatasetColumns(dataset, {feature_idx}, &sub_rng);
    return model->Evaluate(perturbed_dataset, eval_options, &sub_rng);
  };

  return utils::ComputePermutationFeatureImportance(
      base_evaluation, permutation_evaluation, model, output, options);
}

absl::Status ComputePermutationFeatureImportance(
    const dataset::VerticalDataset& dataset, const model::AbstractModel* model,
    ResultFeatureImportanceProto* output,
    const ComputeFeatureImportanceOptions& options) {
  ResultFeatureImportance raw_output;
  RETURN_IF_ERROR(ComputePermutationFeatureImportance(dataset, model,
                                                      &raw_output, options));
  for (const auto& item : raw_output) {
    (*output)[item.first] = std::move(item.second);
  }
  return absl::OkStatus();
}

absl::Status ComputePermutationFeatureImportance(
    const metric::proto::EvaluationResults& base_evaluation,
    const std::function<
        absl::StatusOr<absl::optional<metric::proto::EvaluationResults>>(
            const int feature_idx)>& get_permutation_evaluation,
    const model::AbstractModel* model, ResultFeatureImportanceProto* output,
    const ComputeFeatureImportanceOptions& options) {
  ResultFeatureImportance raw_output;
  RETURN_IF_ERROR(ComputePermutationFeatureImportance(
      base_evaluation, get_permutation_evaluation, model, &raw_output,
      options));
  for (const auto& item : raw_output) {
    (*output)[item.first] = std::move(item.second);
  }
  return absl::OkStatus();
}

}  // namespace utils
}  // namespace yggdrasil_decision_forests
