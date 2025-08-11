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
#include <atomic>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/shap.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/synchronization_primitives.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace {

constexpr char kFeatureImportanceShape[] = "SHAP_VALUE";

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

void FeatureImportanceToProto(const ResultFeatureImportance& src,
                              ResultFeatureImportanceProto* dst) {
  for (const auto& item : src) {
    (*dst)[item.first] = std::move(item.second);
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
        absl::StatusOr<std::optional<metric::proto::EvaluationResults>>(
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
    utils::concurrency::ThreadPool pool(
        options.num_threads,
        {.name_prefix = std::string("variable_importance")});
    LOG(INFO) << "Running " << model->data_spec().columns_size()
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
  utils::RandomEngine rng;
  utils::concurrency::Mutex rng_mutex;

  // Setup the evaluation configuration.
  metric::proto::EvaluationOptions eval_options;
  eval_options.set_bootstrapping_samples(0);
  metric::proto::EvaluationResults base_evaluation;
  int label_col_idx = model->label_col_idx();
  if (model->task() == model::proto::ANOMALY_DETECTION) {
    eval_options.set_task(model::proto::CLASSIFICATION);
    if (label_col_idx == -1) {
      return absl::InvalidArgumentError(
          "Feature importance for anomaly detection models requires a label.");
    }
    ASSIGN_OR_RETURN(base_evaluation,
                     model->EvaluateOverrideType(
                         dataset, eval_options, model::proto::CLASSIFICATION,
                         label_col_idx, /*override_group_col_idx=*/-1, &rng));

  } else {
    eval_options.set_task(model->task());
    ASSIGN_OR_RETURN(base_evaluation,
                     model->EvaluateWithStatus(dataset, eval_options, &rng));
  }

  const auto permutation_evaluation = [&dataset, &eval_options, &rng,
                                       &rng_mutex, model,
                                       label_col_idx](const int feature_idx)
      -> std::optional<metric::proto::EvaluationResults> {
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
    if (model->task() == model::proto::ANOMALY_DETECTION) {
      return model
          ->EvaluateOverrideType(perturbed_dataset, eval_options,
                                 model::proto::CLASSIFICATION, label_col_idx,
                                 /*override_group_col_idx=*/-1, &sub_rng)
          .value();
    } else {
      return model->Evaluate(perturbed_dataset, eval_options, &sub_rng);
    }
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

absl::Status ComputeShapFeatureImportance(
    const dataset::VerticalDataset& dataset, const model::AbstractModel* model,
    ResultFeatureImportance* output,
    const ComputeShapFeatureImportanceOptions& options) {
  auto& vas =
      *(*output)[kFeatureImportanceShape].mutable_variable_importances();

  // Configuration
  std::atomic<bool> force_stop = false;
  std::optional<absl::Time> cutoff_time;
  if (options.max_duration_seconds.has_value()) {
    cutoff_time =
        absl::Now() + absl::Seconds(options.max_duration_seconds.value());
  }

  // Sum of abs of shape values.
  const int num_columns = model->data_spec().columns_size();
  std::vector<double> sum_abs_shapes(num_columns, 0.);
  dataset::UnsignedExampleIdx num_shap_values = 0;
  // Protects "sum_abs_shapes" and "num_shap_values".
  concurrency::Mutex mutex;

  struct Cache {
    dataset::proto::Example example;
    shap::ExampleShapValues example_shapes;
    std::vector<double> sum_abs_shapes;
  };

  const auto create_cache = [&](size_t thread_idx, size_t num_threads,
                                size_t block_size) -> Cache {
    Cache cache;
    return cache;
  };

  const auto run = [&force_stop, &cutoff_time, model, &dataset, &sum_abs_shapes,
                    num_columns, &mutex, &num_shap_values,
                    sampling = options.sampling](
                       size_t block_idx, size_t begin_item_idx,
                       size_t end_item_idx, Cache* cache) -> absl::Status {
    if (force_stop) {
      return absl::OkStatus();
    }
    if (cutoff_time.has_value() && cutoff_time.value() < absl::Now()) {
      // Out of time
      LOG(INFO) << "Maximum duration reached";
      force_stop = true;
      return absl::OkStatus();
    }

    utils::RandomEngine rnd;
    if (sampling < 1.f) {
      rnd.seed(block_idx);
    }

    cache->sum_abs_shapes.assign(num_columns, 0.);
    dataset::UnsignedExampleIdx local_num_shap_values = 0;
    for (auto example_idx = begin_item_idx; example_idx < end_item_idx;
         example_idx++) {
      if (sampling < 1.f &&
          sampling < std::uniform_real_distribution<float>()(rnd)) {
        // Randomly skip examples.
        continue;
      }

      // Compute shap on example
      dataset.ExtractExample(example_idx, &cache->example);
      RETURN_IF_ERROR(shap::tree_shap(*model, cache->example,
                                      &cache->example_shapes,
                                      /*compute_bias=*/false));

      // Accumulate shap values
      DCHECK_EQ(cache->sum_abs_shapes.size(),
                cache->example_shapes.num_columns());
      for (size_t attribute_idx = 0;
           attribute_idx < cache->sum_abs_shapes.size(); attribute_idx++) {
        for (size_t output_idx = 0;
             output_idx < cache->example_shapes.num_outputs(); output_idx++) {
          const int index =
              cache->example_shapes.Index(attribute_idx, output_idx);
          cache->sum_abs_shapes[attribute_idx] +=
              std::abs(cache->example_shapes.values()[index]);
        }
      }
      local_num_shap_values++;
    }

    // Sync shap values
    concurrency::MutexLock l(&mutex);
    for (size_t attribute_idx = 0; attribute_idx < cache->sum_abs_shapes.size();
         attribute_idx++) {
      sum_abs_shapes[attribute_idx] += cache->sum_abs_shapes[attribute_idx];
    }
    num_shap_values += local_num_shap_values;
    return absl::OkStatus();
  };

  LOG(INFO) << "Compute SHAP values";
  RETURN_IF_ERROR(utils::concurrency::ConcurrentForLoopWithWorker<Cache>(
      /*num_items=*/dataset.nrow(),
      /*max_num_threads=*/options.num_threads,
      /*min_block_size=*/20,
      /*max_block_size=*/1000, create_cache, run));
  LOG(INFO) << "Done computing SHAP values";

  const auto& input_features = model->input_features();
  const absl::flat_hash_set<int> input_features_set(input_features.begin(),
                                                    input_features.end());

  for (size_t attribute_idx = 0; attribute_idx < sum_abs_shapes.size();
       attribute_idx++) {
    if (!input_features_set.contains(attribute_idx)) {
      continue;
    }
    auto& feature = *vas.Add();
    feature.set_importance(sum_abs_shapes[attribute_idx] / num_shap_values);
    feature.set_attribute_idx(attribute_idx);
  }

  // Sort values
  const auto var_importance_comparer =
      [](const model::proto::VariableImportance& a,
         const model::proto::VariableImportance& b) {
        return a.importance() > b.importance();
      };
  std::sort(vas.begin(), vas.end(), var_importance_comparer);
  return absl::OkStatus();
}

absl::Status ComputeShapFeatureImportance(
    const dataset::VerticalDataset& dataset, const model::AbstractModel* model,
    ResultFeatureImportanceProto* output,
    const ComputeShapFeatureImportanceOptions& options) {
  ResultFeatureImportance raw_output;
  const auto status =
      ComputeShapFeatureImportance(dataset, model, &raw_output, options);
  if (!status.ok()) {
    LOG(WARNING) << "Cannot compute SHAP values:" << status.message();
    return absl::OkStatus();
  }
  FeatureImportanceToProto(raw_output, output);
  return absl::OkStatus();
}

absl::Status ComputePermutationFeatureImportance(
    const metric::proto::EvaluationResults& base_evaluation,
    const std::function<
        absl::StatusOr<std::optional<metric::proto::EvaluationResults>>(
            const int feature_idx)>& get_permutation_evaluation,
    const model::AbstractModel* model, ResultFeatureImportanceProto* output,
    const ComputeFeatureImportanceOptions& options) {
  ResultFeatureImportance raw_output;
  RETURN_IF_ERROR(ComputePermutationFeatureImportance(
      base_evaluation, get_permutation_evaluation, model, &raw_output,
      options));
  FeatureImportanceToProto(raw_output, output);
  return absl::OkStatus();
}

}  // namespace utils
}  // namespace yggdrasil_decision_forests
