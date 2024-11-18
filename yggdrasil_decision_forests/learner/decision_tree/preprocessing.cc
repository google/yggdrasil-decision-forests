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

#include "yggdrasil_decision_forests/learner/decision_tree/preprocessing.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests::model::decision_tree {

bool StrategyRequireFeaturePresorting(
    proto::DecisionTreeTrainingConfig::Internal::SortingStrategy strategy) {
  switch (strategy) {
    case proto::DecisionTreeTrainingConfig::Internal::PRESORTED:
    case proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED:
    case proto::DecisionTreeTrainingConfig::Internal::AUTO:
      return true;
    case proto::DecisionTreeTrainingConfig::Internal::IN_NODE:
      return false;
  };
}

absl::StatusOr<Preprocessing> PreprocessTrainingDataset(
    const dataset::VerticalDataset& train_dataset,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config, const int num_threads) {
  const auto time_begin = absl::Now();
  Preprocessing preprocessing;
  preprocessing.set_num_examples(train_dataset.nrow());

  if (StrategyRequireFeaturePresorting(
          dt_config.internal().sorting_strategy())) {
    RETURN_IF_ERROR(PresortNumericalFeatures(
        train_dataset, config_link, dt_config, num_threads, &preprocessing));
  }

  const auto duration = absl::Now() - time_begin;
  if (duration > absl::Seconds(10)) {
    LOG(INFO) << "Feature index computed in " << absl::FormatDuration(duration);
  }
  return preprocessing;
}

absl::Status PresortNumericalFeatures(
    const dataset::VerticalDataset& train_dataset,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config, const int num_threads,
    Preprocessing* preprocessing) {
  // Check number of examples.
  RETURN_IF_ERROR(dataset::CheckNumExamples(train_dataset.nrow()));

  preprocessing->mutable_presorted_numerical_features()->resize(
      train_dataset.data_spec().columns_size());

  utils::concurrency::ThreadPool pool(
      std::min(num_threads, config_link.features().size()),
      {.name_prefix = std::string("presort_numerical_features")});
  pool.StartWorkers();

  // For all the input features in the model.
  for (const auto feature_idx : config_link.features()) {
    // Skip non numerical features.
    if (train_dataset.data_spec().columns(feature_idx).type() !=
        dataset::proto::NUMERICAL) {
      continue;
    }

    pool.Schedule([feature_idx, &train_dataset, preprocessing]() {
      const UnsignedExampleIdx num_examples = train_dataset.nrow();
      const auto& values =
          train_dataset
              .ColumnWithCastWithStatus<
                  dataset::VerticalDataset::NumericalColumn>(feature_idx)
              .value()
              ->values();
      CHECK_EQ(num_examples, values.size());

      // Global imputation replacement.
      const float na_replacement_value =
          train_dataset.data_spec().columns(feature_idx).numerical().mean();

      std::vector<std::pair<float, SparseItemMeta::ExampleIdx>> items(
          values.size());
      for (UnsignedExampleIdx example_idx = 0; example_idx < num_examples;
           example_idx++) {
        auto value = values[example_idx];
        if (std::isnan(value)) {
          value = na_replacement_value;
        }
        items[example_idx] = {value, example_idx};
      }

      // Sort by feature value and example index.
      std::sort(items.begin(), items.end());

      auto& sorted_values =
          (*preprocessing->mutable_presorted_numerical_features())[feature_idx];
      sorted_values.items.resize(values.size());

      float last_value;
      if (num_examples >= 0) {
        last_value = items.front().first;
      }
      for (UnsignedExampleIdx sorted_example_idx = 0;
           sorted_example_idx < num_examples; sorted_example_idx++) {
        const auto value = items[sorted_example_idx];
        SparseItemMeta::ExampleIdx example_idx = value.second;
        if (value.first != last_value) {
          example_idx |= SparseItemMeta::kMaskDeltaBit;
        }
        last_value = value.first;
        sorted_values.items[sorted_example_idx] = example_idx;
      }
    });
  }
  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests::model::decision_tree
