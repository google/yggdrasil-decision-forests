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

#include "yggdrasil_decision_forests/learner/decision_tree/training.h"

#include <stddef.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/sparse_oblique.h"
#include "yggdrasil_decision_forests/learner/decision_tree/splitter_accumulator.h"
#include "yggdrasil_decision_forests/learner/decision_tree/splitter_scanner.h"
#include "yggdrasil_decision_forests/learner/decision_tree/utils.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/cast.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/distribution.h"
#include "yggdrasil_decision_forests/utils/distribution.pb.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/random.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {

namespace {

// Number of trials to run when learning a categorical split with randomly
// generated masks.
//
// Args:
//   config: A random categorical split learning configuration.
//
// Returns:
//   A function active_dictionary_size -> number_of_trials.
//
// The "active_dictionary_size" is the number of unique categorical values that
// are present at least once in the training examples of the node.
std::function<int(const int active_dictionary_size)>
NumTrialsForRandomCategoricalSplit(const proto::Categorical::Random& config) {
  const auto num_trial_exponent = config.num_trial_exponent();
  const auto max_num_trials = config.max_num_trials();
  return
      [num_trial_exponent, max_num_trials](const int active_dictionary_size) {
        const int num_trials =
            32 + std::pow(active_dictionary_size, num_trial_exponent);
        return std::min(num_trials, max_num_trials);
      };
}

// Set the label value for a classification label on a vertical dataset.
void SetClassificationLabelDistribution(
    const dataset::VerticalDataset& dataset,
    const std::vector<row_t>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfigLinking& config_link, proto::Node* node) {
  const auto* const labels =
      dataset.ColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(
          config_link.label());
  utils::IntegerDistributionDouble label_distribution;
  const int32_t num_classes = dataset.data_spec()
                                  .columns(config_link.label())
                                  .categorical()
                                  .number_of_unique_values();
  label_distribution.SetNumClasses(num_classes);
  for (const row_t example_idx : selected_examples) {
    if (weights.empty()) {
      label_distribution.Add(labels->values()[example_idx]);
    } else {
      label_distribution.Add(labels->values()[example_idx],
                             weights[example_idx]);
    }
  }
  label_distribution.Save(node->mutable_classifier()->mutable_distribution());
  node->mutable_classifier()->set_top_value(label_distribution.TopClass());
}

void SetCategoricalUpliftLabelDistribution(
    const dataset::VerticalDataset& dataset,
    const std::vector<row_t>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfigLinking& config_link, proto::Node* node) {
  DCHECK(!weights.empty());
  const auto* const outcomes =
      dataset.ColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(
          config_link.label());

  const auto* const treatments =
      dataset.ColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(
          config_link.uplift_treatment());

  const auto& outcome_spec = dataset.data_spec().columns(config_link.label());
  const auto& treatment_spec =
      dataset.data_spec().columns(config_link.uplift_treatment());

  UpliftLabelDistribution label_dist;
  label_dist.InitializeAndClearCategoricalOutcome(
      outcome_spec.categorical().number_of_unique_values(),
      treatment_spec.categorical().number_of_unique_values());

  for (const row_t example_idx : selected_examples) {
    label_dist.AddCategoricalOutcome(outcomes->values()[example_idx],
                                     treatments->values()[example_idx],
                                     weights[example_idx]);
  }
  internal::UpliftLabelDistToLeaf(label_dist, node->mutable_uplift());
}

void SetRegressiveUpliftLabelDistribution(
    const dataset::VerticalDataset& dataset,
    const std::vector<row_t>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfigLinking& config_link, proto::Node* node) {
  const auto* const outcomes =
      dataset.ColumnWithCast<dataset::VerticalDataset::NumericalColumn>(
          config_link.label());

  const auto* const treatments =
      dataset.ColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(
          config_link.uplift_treatment());

  const auto& treatment_spec =
      dataset.data_spec().columns(config_link.uplift_treatment());

  UpliftLabelDistribution label_dist;
  label_dist.InitializeAndClearNumericalOutcome(
      treatment_spec.categorical().number_of_unique_values());

  for (const row_t example_idx : selected_examples) {
    label_dist.AddNumericalOutcome(outcomes->values()[example_idx],
                                   treatments->values()[example_idx],
                                   weights[example_idx]);
  }
  internal::UpliftLabelDistToLeaf(label_dist, node->mutable_uplift());
}

// Compute the ratio of true label for all attribute values.
void ComputeTrueLabelValuePerAttributeValue(
    const utils::IntegersConfusionMatrixDouble& confusion,
    const int32_t true_label_value,
    std::vector<std::pair<float, int32_t>>* ratio_true_label_by_attr_value) {
  ratio_true_label_by_attr_value->resize(confusion.ncol());
  for (int32_t attribute_value = 0; attribute_value < confusion.ncol();
       attribute_value++) {
    const float count_true_label =
        confusion.at(true_label_value, attribute_value);
    float count_all_label = 0;
    for (int32_t label_value = 0; label_value < confusion.nrow();
         label_value++) {
      count_all_label += confusion.at(label_value, attribute_value);
    }
    const float ratio_true_label =
        (count_all_label > 0) ? (count_true_label / count_all_label) : 0;
    (*ratio_true_label_by_attr_value)[attribute_value].first = ratio_true_label;
    (*ratio_true_label_by_attr_value)[attribute_value].second = attribute_value;
  }
  // Order the attribute values in increasing order of true label ratio.
  std::sort(ratio_true_label_by_attr_value->begin(),
            ratio_true_label_by_attr_value->end());
}

// Helper function to set a condition statistics. Do not set the following
// fields: "mutable_condition" and "na_value".
void SetConditionHelper(
    const double information_gain, const int32_t attribute_idx,
    const utils::BinaryToIntegerConfusionMatrixDouble& running_confusion,
    const utils::BinaryToIntegerConfusionMatrixInt64&
        running_confusion_no_weights,
    proto::NodeCondition* condition) {
  condition->set_split_score(information_gain);
  condition->set_attribute(attribute_idx);
  condition->set_num_pos_training_examples_without_weight(
      running_confusion_no_weights.pos().NumObservations());
  condition->set_num_pos_training_examples_with_weight(
      running_confusion.pos().NumObservations());
  condition->set_num_training_examples_without_weight(
      running_confusion_no_weights.NumObservations());
  condition->set_num_training_examples_with_weight(
      running_confusion.NumObservations());
}

// Helper function to set a condition statistics.
// Similar as "SetConditionHelper" above, but for a regression problem.
void SetConditionHelper(
    const double variance_reduction, const int32_t attribute_idx,
    const utils::BinaryToNormalDistributionDouble& running_confusion,
    const utils::BinaryToNormalDistributionDouble& running_confusion_no_weights,
    proto::NodeCondition* condition) {
  condition->set_split_score(variance_reduction);
  condition->set_attribute(attribute_idx);
  condition->set_num_pos_training_examples_without_weight(
      running_confusion_no_weights.pos().NumObservations());
  condition->set_num_pos_training_examples_with_weight(
      running_confusion.pos().NumObservations());
  condition->set_num_training_examples_without_weight(
      running_confusion_no_weights.NumObservations());
  condition->set_num_training_examples_with_weight(
      running_confusion.NumObservations());
}

// Computes and set in "na_replacement" the value to use as a replacement of
// missing values when the "local imputation" strategy is used.
//
// Explanation: The "local imputation" strategy to handle missing values
// consists in replacing these missing values by the mean of the feature in the
// training dataset.
//
// If the feature only contains missing values, the "na_replacement" argument is
// left unchanged.
void LocalImputationForNumericalAttribute(
    const std::vector<row_t>& selected_examples,
    const std::vector<float>& weights, const std::vector<float>& attributes,
    float* na_replacement) {
  double na_replacement_value_accumulator = 0;
  double na_replacement_weight_accumulator = 0;
  for (const auto example_idx : selected_examples) {
    const float attribute = attributes[example_idx];
    const float weight = weights.empty() ? 1.f : weights[example_idx];
    if (!std::isnan(attribute)) {
      na_replacement_value_accumulator += attribute * weight;
      na_replacement_weight_accumulator += weight;
    }
  }
  if (na_replacement_weight_accumulator > 0) {
    *na_replacement = static_cast<float>(na_replacement_value_accumulator /
                                         na_replacement_weight_accumulator);
  }
}

// Similar as "LocalImputationForNumericalAttribute", but for a categorical
// attribute. Return the most frequent attribute value.
void LocalImputationForCategoricalAttribute(
    const std::vector<row_t>& selected_examples,
    const std::vector<float>& weights, const std::vector<int32_t>& attributes,
    const int32_t num_attribute_classes, int32_t* na_replacement) {
  utils::IntegerDistributionDouble attribute_distribution;
  attribute_distribution.SetNumClasses(num_attribute_classes);
  for (const auto example_idx : selected_examples) {
    const auto attribute_value = attributes[example_idx];
    if (attribute_value !=
        dataset::VerticalDataset::CategoricalColumn::kNaValue) {
      const float weight = weights.empty() ? 1.f : weights[example_idx];
      attribute_distribution.Add(attribute_value, weight);
    }
  }
  if (attribute_distribution.NumObservations() > 0) {
    *na_replacement = attribute_distribution.TopClass();
  }
}

// Similar to "LocalImputationForCategoricalAttribute", but for a boolean
// attribute. Returns the most frequent attribute value.
void LocalImputationForBooleanAttribute(
    const std::vector<row_t>& selected_examples,
    const std::vector<float>& weights, const std::vector<char>& attributes,
    bool* na_replacement) {
  DCHECK(!weights.empty());
  utils::IntegerDistributionDouble attribute_distribution;
  attribute_distribution.SetNumClasses(2);
  for (const auto example_idx : selected_examples) {
    const auto attribute_value = attributes[example_idx];
    if (attribute_value != dataset::VerticalDataset::BooleanColumn::kNaValue) {
      const float weight = weights.empty() ? 1.f : weights[example_idx];
      attribute_distribution.Add(attribute_value, weight);
    }
  }
  if (attribute_distribution.NumObservations() > 0) {
    *na_replacement = attribute_distribution.TopClass();
  }
}

// Return the minimum and maximum values of a numerical attribute.
// Return false if there is no min-max e.g. selected_examples is empty or all
// the values are NAs.
bool MinMaxNumericalAttribute(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& attributes, float* min_value, float* max_value) {
  float local_min_value = 0;
  float local_max_value = 0;
  bool first = true;
  for (const auto example_idx : selected_examples) {
    const float attribute = attributes[example_idx];
    if (first && !std::isnan(attribute)) {
      local_max_value = local_min_value = attribute;
      first = false;
    } else if (attribute > local_max_value) {
      local_max_value = attribute;
    } else if (attribute < local_min_value) {
      local_min_value = attribute;
    }
  }
  *min_value = local_min_value;
  *max_value = local_max_value;
  return !first;
}

// Search for the attribute item that maximize the immediate variance reduction.
std::pair<int, double> GetAttributeValueWithMaximumVarianceReduction(
    const double variance_reduction, const int32_t num_attribute_classes,
    const utils::BinaryToNormalDistributionDouble& split_label_distribution,
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<bool>& positive_selected_example_bitmap,
    const std::vector<std::pair<size_t, size_t>>& attribute_values,
    const std::vector<int>& attribute_bank, const std::vector<float>& weights,
    const std::vector<float>& labels, const double initial_variance,
    std::vector<dataset::VerticalDataset::row_t>* running_attr_bank_idx,
    std::vector<bool>* candidate_attributes_bitmap) {
  DCHECK_EQ(weights.size(), labels.size());
  // Search for the attribute item that maximize the variance reduction. Note:
  // We ignore attribute that reduce the current variance reduction.
  double best_candidate_variance_reduction = variance_reduction;
  int best_attr_value = -1;
  for (int candidate_attr_value = 0;
       candidate_attr_value < num_attribute_classes; candidate_attr_value++) {
    if (!(*candidate_attributes_bitmap)[candidate_attr_value]) {
      // The attribute value was already selected or was excluded from the
      // search space.
      continue;
    }
    // Compute the variance reduction of the already selected values
    // "positive_attributes_vector" and the candidate value
    // "candidate_attr_value".
    utils::BinaryToNormalDistributionDouble candidate_split_label_distribution =
        split_label_distribution;
    int64_t num_preset_in_negative_set = 0;
    int64_t num_absent_in_negative_set = 0;
    for (size_t select_idx = 0; select_idx < selected_examples.size();
         select_idx++) {
      const auto example_idx = selected_examples[select_idx];
      if (positive_selected_example_bitmap[select_idx]) {
        // The example is already in the positive set.
        continue;
      }

      // Search if X = attribute_bank[ attribute_values[example_idx].first,
      // attribute_values[example_idx].second ] contains the current candidate
      // attribute value "candidate_attr_value".
      //
      // We use "running_attr_bank_idx[select_idx]" that contains the index in
      // "X" of the last tested candidate attribute.
      //
      // Note: "X" does not contains duplicates and is sorted in increasing
      // order. The candidate attributes "candidate_attr_value" are testing in
      // increasing order.
      bool match = false;
      dataset::VerticalDataset::row_t last_attr;
      while (
          (*running_attr_bank_idx)[select_idx] <
              attribute_values[example_idx].second &&
          (last_attr = attribute_bank[(*running_attr_bank_idx)[select_idx]]) <=
              candidate_attr_value) {
        (*running_attr_bank_idx)[select_idx]++;
        if (last_attr == candidate_attr_value) {
          match = true;
          break;
        }
      }

      if (match) {
        num_preset_in_negative_set++;
        // Add the example to the positive set and remove it from the
        // negative.
        candidate_split_label_distribution.mutable_pos()->Add(
            labels[example_idx], weights[example_idx]);
        candidate_split_label_distribution.mutable_neg()->Add(
            labels[example_idx], -weights[example_idx]);
      } else {
        num_absent_in_negative_set++;
      }
    }
    // Remove the attribute from the candidate set if the attribute is pure
    // for the current negative set.
    if (num_absent_in_negative_set == 0 ||
        num_absent_in_negative_set == selected_examples.size()) {
      (*candidate_attributes_bitmap)[candidate_attr_value] = false;
      continue;
    }

    const double candidate_variance_reduction =
        initial_variance - candidate_split_label_distribution.FinalVariance();
    if (candidate_variance_reduction > best_candidate_variance_reduction) {
      // Best score so far.
      best_candidate_variance_reduction = candidate_variance_reduction;
      best_attr_value = candidate_attr_value;
    }
  }
  return std::make_pair(best_attr_value, best_candidate_variance_reduction);
}

// Predicts if the sorting-in-node numerical splitter is slower or faster than
// the numerical splitter using a pre-sorted index. Returns true if learning a
// split on "num_selected_examples" while the entire training dataset contains
// "num_total_examples" is more efficient with a pre-sorted splitter. If false,
// a splitter with in-node sorting is more efficient.
//
// This function is an heuristic that impact the training speed but not the
// model.
//
// This function is used when PRESORTING is enabled (see
// "SetDefaultHyperParameters").
bool IsPresortingOnNumericalSplitMoreEfficient(
    const int64_t num_selected_examples, const int64_t num_total_examples) {
  const float ratio =
      static_cast<float>(num_selected_examples) / num_total_examples;
  return num_selected_examples >= 25 && ratio >= 0.125;
}

}  // namespace

void SetLabelDistribution(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    NodeWithChildren* node) {
  switch (config.task()) {
    case model::proto::Task::CLASSIFICATION:
      SetClassificationLabelDistribution(train_dataset, selected_examples,
                                         weights, config_link,
                                         node->mutable_node());
      break;

    case model::proto::Task::REGRESSION:
      SetRegressionLabelDistribution(train_dataset, selected_examples, weights,
                                     config_link, node->mutable_node());
      break;

    case model::proto::Task::CATEGORICAL_UPLIFT:
      SetCategoricalUpliftLabelDistribution(train_dataset, selected_examples,
                                            weights, config_link,
                                            node->mutable_node());
      break;

    case model::proto::Task::NUMERICAL_UPLIFT:
      SetRegressiveUpliftLabelDistribution(train_dataset, selected_examples,
                                           weights, config_link,
                                           node->mutable_node());
      break;

    default:
      DCHECK(false);
  }
}

// Specialization in the case of classification.
SplitSearchResult FindBestCondition(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<row_t>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const ClassificationLabelStats& label_stats, const int32_t attribute_idx,
    proto::NodeCondition* best_condition, utils::RandomEngine* random,
    SplitterPerThreadCache* cache) {
  const int min_num_obs =
      dt_config.in_split_min_examples_check() ? dt_config.min_examples() : 1;

  const auto& attribute_column_spec =
      train_dataset.data_spec().columns(attribute_idx);

  SplitSearchResult result;

  switch (train_dataset.column(attribute_idx)->type()) {
    case dataset::proto::ColumnType::NUMERICAL: {
      if (!dt_config.has_axis_aligned_split()) {
        return SplitSearchResult::kNoBetterSplitFound;
      }

      const auto& attribute_data =
          train_dataset
              .ColumnWithCast<dataset::VerticalDataset::NumericalColumn>(
                  attribute_idx)
              ->values();
      const auto na_replacement = attribute_column_spec.numerical().mean();
      if (dt_config.numerical_split().type() == proto::NumericalSplit::EXACT) {
        result = FindSplitLabelClassificationFeatureNumericalCart(
            selected_examples, weights, attribute_data, label_stats.label_data,
            label_stats.num_label_classes, na_replacement, min_num_obs,
            dt_config, label_stats.label_distribution, attribute_idx,
            internal_config, best_condition, cache);
      } else {
        result = FindSplitLabelClassificationFeatureNumericalHistogram(
            selected_examples, weights, attribute_data, label_stats.label_data,
            label_stats.num_label_classes, na_replacement, min_num_obs,
            dt_config, label_stats.label_distribution, attribute_idx, random,
            best_condition);
      }
    } break;

    case dataset::proto::ColumnType::DISCRETIZED_NUMERICAL: {
      if (!dt_config.has_axis_aligned_split()) {
        return SplitSearchResult::kNoBetterSplitFound;
      }

      const auto& attribute_data =
          train_dataset
              .ColumnWithCast<
                  dataset::VerticalDataset::DiscretizedNumericalColumn>(
                  attribute_idx)
              ->values();
      const auto na_replacement = attribute_column_spec.numerical().mean();
      const auto num_bins =
          attribute_column_spec.discretized_numerical().boundaries_size() + 1;
      const auto na_replacement_index =
          dataset::NumericalToDiscretizedNumerical(attribute_column_spec,
                                                   na_replacement);
      result = FindSplitLabelClassificationFeatureDiscretizedNumericalCart(
          selected_examples, weights, attribute_data, num_bins,
          label_stats.label_data, label_stats.num_label_classes,
          na_replacement_index, min_num_obs, dt_config,
          label_stats.label_distribution, attribute_idx, best_condition, cache);
    } break;

    case dataset::proto::ColumnType::CATEGORICAL: {
      const auto& attribute_data =
          train_dataset
              .ColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(
                  attribute_idx)
              ->values();
      const auto na_replacement =
          attribute_column_spec.categorical().most_frequent_value();
      const auto num_attribute_classes =
          attribute_column_spec.categorical().number_of_unique_values();
      result = FindSplitLabelClassificationFeatureCategorical(
          selected_examples, weights, attribute_data, label_stats.label_data,
          num_attribute_classes, label_stats.num_label_classes, na_replacement,
          min_num_obs, dt_config, label_stats.label_distribution, attribute_idx,
          random, best_condition, cache);
    } break;

    case dataset::proto::ColumnType::CATEGORICAL_SET: {
      const auto* attribute_data =
          train_dataset
              .ColumnWithCast<dataset::VerticalDataset::CategoricalSetColumn>(
                  attribute_idx);
      const auto num_attribute_classes =
          attribute_column_spec.categorical().number_of_unique_values();
      result = FindSplitLabelClassificationFeatureCategoricalSetGreedyForward(
          selected_examples, weights, *attribute_data, label_stats.label_data,
          num_attribute_classes, label_stats.num_label_classes, min_num_obs,
          dt_config, label_stats.label_distribution, attribute_idx,
          best_condition, random);
    } break;

    case dataset::proto::ColumnType::BOOLEAN: {
      // Condition of the type "Attr is True".
      const auto& attribute_data =
          train_dataset
              .ColumnWithCast<dataset::VerticalDataset::BooleanColumn>(
                  attribute_idx)
              ->values();
      const auto na_replacement =
          attribute_column_spec.boolean().count_true() >=
          attribute_column_spec.boolean().count_false();
      result = FindSplitLabelClassificationFeatureBoolean(
          selected_examples, weights, attribute_data, label_stats.label_data,
          label_stats.num_label_classes, na_replacement, min_num_obs, dt_config,
          label_stats.label_distribution, attribute_idx, best_condition, cache);
    } break;

    default:
      LOG(FATAL) << dataset::proto::ColumnType_Name(
                        train_dataset.column(attribute_idx)->type())
                 << " attribute " << train_dataset.column(attribute_idx)->name()
                 << " is not supported.";
  }

  // Condition of the type "Attr is NA".
  if (dt_config.allow_na_conditions()) {
    const auto na_result = FindSplitLabelClassificationFeatureNA(
        selected_examples, weights, train_dataset.column(attribute_idx),
        label_stats.label_data, label_stats.num_label_classes, min_num_obs,
        dt_config, label_stats.label_distribution, attribute_idx,
        best_condition, cache);
    result = std::min(result, na_result);
  }

  return result;
}

SplitSearchResult FindBestCondition(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<row_t>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const RegressionHessianLabelStats& label_stats, const int32_t attribute_idx,
    proto::NodeCondition* best_condition, utils::RandomEngine* random,
    SplitterPerThreadCache* cache) {
  const int min_num_obs =
      dt_config.in_split_min_examples_check() ? dt_config.min_examples() : 1;

  const auto& attribute_column_spec =
      train_dataset.data_spec().columns(attribute_idx);

  SplitSearchResult result;

  switch (train_dataset.column(attribute_idx)->type()) {
    case dataset::proto::ColumnType::NUMERICAL: {
      if (!dt_config.has_axis_aligned_split()) {
        return SplitSearchResult::kNoBetterSplitFound;
      }

      // Condition of the type "Attr >= threshold".
      const auto& attribute_data =
          train_dataset
              .ColumnWithCast<dataset::VerticalDataset::NumericalColumn>(
                  attribute_idx)
              ->values();
      const auto na_replacement = attribute_column_spec.numerical().mean();
      if (dt_config.numerical_split().type() == proto::NumericalSplit::EXACT) {
        result = FindSplitLabelHessianRegressionFeatureNumericalCart(
            selected_examples, weights, attribute_data,
            label_stats.gradient_data, label_stats.hessian_data, na_replacement,
            min_num_obs, dt_config, label_stats.sum_gradient,
            label_stats.sum_hessian, label_stats.sum_weights, attribute_idx,
            internal_config, best_condition, cache);
      } else {
        LOG(FATAL) << "Only split exact implemented for hessian gains.";
      }
    } break;

    case dataset::proto::ColumnType::DISCRETIZED_NUMERICAL: {
      if (!dt_config.has_axis_aligned_split()) {
        return SplitSearchResult::kNoBetterSplitFound;
      }

      // Condition of the type "Attr >= threshold".
      const auto& attribute_data =
          train_dataset
              .ColumnWithCast<
                  dataset::VerticalDataset::DiscretizedNumericalColumn>(
                  attribute_idx)
              ->values();
      const auto na_replacement = attribute_column_spec.numerical().mean();
      const auto num_bins =
          attribute_column_spec.discretized_numerical().boundaries_size() + 1;
      const auto na_replacement_index =
          dataset::NumericalToDiscretizedNumerical(attribute_column_spec,
                                                   na_replacement);
      result = FindSplitLabelHessianRegressionFeatureDiscretizedNumericalCart(
          selected_examples, weights, attribute_data, num_bins,
          label_stats.gradient_data, label_stats.hessian_data,
          na_replacement_index, min_num_obs, dt_config,
          label_stats.sum_gradient, label_stats.sum_hessian,
          label_stats.sum_weights, attribute_idx, internal_config,
          best_condition, cache);
    } break;

    case dataset::proto::ColumnType::CATEGORICAL: {
      // Condition of the type "Attr \in X".
      const auto& attribute_data =
          train_dataset
              .ColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(
                  attribute_idx)
              ->values();
      const auto na_replacement =
          attribute_column_spec.categorical().most_frequent_value();
      const auto num_attribute_classes =
          attribute_column_spec.categorical().number_of_unique_values();
      result = FindSplitLabelHessianRegressionFeatureCategorical(
          selected_examples, weights, attribute_data, label_stats.gradient_data,
          label_stats.hessian_data, num_attribute_classes, na_replacement,
          min_num_obs, dt_config, label_stats.sum_gradient,
          label_stats.sum_hessian, label_stats.sum_weights, attribute_idx,
          internal_config, best_condition, cache, random);
    } break;

    case dataset::proto::ColumnType::BOOLEAN: {
      // Condition of the type "Attr is True".
      const auto& attribute_data =
          train_dataset
              .ColumnWithCast<dataset::VerticalDataset::BooleanColumn>(
                  attribute_idx)
              ->values();
      const auto na_replacement =
          attribute_column_spec.boolean().count_true() >=
          attribute_column_spec.boolean().count_false();
      result = FindSplitLabelHessianRegressionFeatureBoolean(
          selected_examples, weights, attribute_data, label_stats.gradient_data,
          label_stats.hessian_data, na_replacement, min_num_obs, dt_config,
          label_stats.sum_gradient, label_stats.sum_hessian,
          label_stats.sum_weights, attribute_idx, internal_config,
          best_condition, cache);
    } break;

    default:
      LOG(FATAL) << dataset::proto::ColumnType_Name(
                        train_dataset.column(attribute_idx)->type())
                 << " attribute " << train_dataset.column(attribute_idx)->name()
                 << " is not supported.";
  }

  // Condition of the type "Attr is NA".
  if (dt_config.allow_na_conditions()) {
    const auto na_result = FindSplitLabelHessianRegressionFeatureNA(
        selected_examples, weights, train_dataset.column(attribute_idx),
        label_stats.gradient_data, label_stats.hessian_data, min_num_obs,
        dt_config, label_stats.sum_gradient, label_stats.sum_hessian,
        label_stats.sum_weights, attribute_idx, internal_config, best_condition,
        cache);
    result = std::min(result, na_result);
  }

  return result;
}

// Specialization in the case of regression.
SplitSearchResult FindBestCondition(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<row_t>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const RegressionLabelStats& label_stats, const int32_t attribute_idx,
    proto::NodeCondition* best_condition, utils::RandomEngine* random,
    SplitterPerThreadCache* cache) {
  const int min_num_obs =
      dt_config.in_split_min_examples_check() ? dt_config.min_examples() : 1;

  const auto& attribute_column_spec =
      train_dataset.data_spec().columns(attribute_idx);

  SplitSearchResult result;

  switch (train_dataset.column(attribute_idx)->type()) {
    case dataset::proto::ColumnType::NUMERICAL: {
      if (!dt_config.has_axis_aligned_split()) {
        return SplitSearchResult::kNoBetterSplitFound;
      }

      // Condition of the type "Attr >= threshold".
      const auto& attribute_data =
          train_dataset
              .ColumnWithCast<dataset::VerticalDataset::NumericalColumn>(
                  attribute_idx)
              ->values();
      const auto na_replacement = attribute_column_spec.numerical().mean();
      if (dt_config.numerical_split().type() == proto::NumericalSplit::EXACT) {
        result = FindSplitLabelRegressionFeatureNumericalCart(
            selected_examples, weights, attribute_data, label_stats.label_data,
            na_replacement, min_num_obs, dt_config,
            label_stats.label_distribution, attribute_idx, internal_config,
            best_condition, cache);
      } else {
        result = FindSplitLabelRegressionFeatureNumericalHistogram(
            selected_examples, weights, attribute_data, label_stats.label_data,
            na_replacement, min_num_obs, dt_config,
            label_stats.label_distribution, attribute_idx, random,
            best_condition);
      }
    } break;

    case dataset::proto::ColumnType::DISCRETIZED_NUMERICAL: {
      if (!dt_config.has_axis_aligned_split()) {
        return SplitSearchResult::kNoBetterSplitFound;
      }

      // Condition of the type "Attr >= threshold".
      const auto& attribute_data =
          train_dataset
              .ColumnWithCast<
                  dataset::VerticalDataset::DiscretizedNumericalColumn>(
                  attribute_idx)
              ->values();
      const auto na_replacement = attribute_column_spec.numerical().mean();
      const auto num_bins =
          attribute_column_spec.discretized_numerical().boundaries_size() + 1;
      const auto na_replacement_index =
          dataset::NumericalToDiscretizedNumerical(attribute_column_spec,
                                                   na_replacement);
      result = FindSplitLabelRegressionFeatureDiscretizedNumericalCart(
          selected_examples, weights, attribute_data, num_bins,
          label_stats.label_data, na_replacement_index, min_num_obs, dt_config,
          label_stats.label_distribution, attribute_idx, best_condition, cache);
    } break;

    case dataset::proto::ColumnType::CATEGORICAL: {
      // Condition of the type "Attr \in X".
      const auto& attribute_data =
          train_dataset
              .ColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(
                  attribute_idx)
              ->values();
      const auto na_replacement =
          attribute_column_spec.categorical().most_frequent_value();
      const auto num_attribute_classes =
          attribute_column_spec.categorical().number_of_unique_values();
      result = FindSplitLabelRegressionFeatureCategorical(
          selected_examples, weights, attribute_data, label_stats.label_data,
          num_attribute_classes, na_replacement, min_num_obs, dt_config,
          label_stats.label_distribution, attribute_idx, best_condition, cache,
          random);
    } break;

    case dataset::proto::ColumnType::CATEGORICAL_SET: {
      const auto* attribute_data =
          train_dataset
              .ColumnWithCast<dataset::VerticalDataset::CategoricalSetColumn>(
                  attribute_idx);
      const auto num_attribute_classes =
          attribute_column_spec.categorical().number_of_unique_values();
      result = FindSplitLabelRegressionFeatureCategoricalSetGreedyForward(
          selected_examples, weights, *attribute_data, label_stats.label_data,
          num_attribute_classes, min_num_obs, dt_config,
          label_stats.label_distribution, attribute_idx, best_condition,
          random);
    } break;

    case dataset::proto::ColumnType::BOOLEAN: {
      // Condition of the type "Attr is True".
      const auto& attribute_data =
          train_dataset
              .ColumnWithCast<dataset::VerticalDataset::BooleanColumn>(
                  attribute_idx)
              ->values();
      const auto na_replacement =
          attribute_column_spec.boolean().count_true() >=
          attribute_column_spec.boolean().count_false();
      result = FindSplitLabelRegressionFeatureBoolean(
          selected_examples, weights, attribute_data, label_stats.label_data,
          na_replacement, min_num_obs, dt_config,
          label_stats.label_distribution, attribute_idx, best_condition, cache);
    } break;

    default:
      LOG(FATAL) << dataset::proto::ColumnType_Name(
                        train_dataset.column(attribute_idx)->type())
                 << " attribute " << train_dataset.column(attribute_idx)->name()
                 << " is not supported.";
  }

  // Condition of the type "Attr is NA".
  if (dt_config.allow_na_conditions()) {
    const auto na_result = FindSplitLabelRegressionFeatureNA(
        selected_examples, weights, train_dataset.column(attribute_idx),
        label_stats.label_data, min_num_obs, dt_config,
        label_stats.label_distribution, attribute_idx, best_condition, cache);
    result = std::min(result, na_result);
  }

  return result;
}

// Specialization in the case of uplift with categorical outcome.
SplitSearchResult FindBestCondition(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<row_t>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const CategoricalUpliftLabelStats& label_stats, const int32_t attribute_idx,
    proto::NodeCondition* best_condition, utils::RandomEngine* random,
    SplitterPerThreadCache* cache) {
  const int min_num_obs =
      dt_config.in_split_min_examples_check() ? dt_config.min_examples() : 1;
  const auto& attribute_column_spec =
      train_dataset.data_spec().columns(attribute_idx);

  SplitSearchResult result;

  switch (train_dataset.column(attribute_idx)->type()) {
    case dataset::proto::ColumnType::NUMERICAL: {
      const auto& attribute_data =
          train_dataset
              .ColumnWithCast<dataset::VerticalDataset::NumericalColumn>(
                  attribute_idx)
              ->values();
      const auto na_replacement = attribute_column_spec.numerical().mean();

      result = FindSplitLabelUpliftCategoricalFeatureNumericalCart(
          selected_examples, weights, attribute_data, label_stats,
          na_replacement, min_num_obs, dt_config, attribute_idx,
          internal_config, best_condition, cache);
    } break;

    case dataset::proto::ColumnType::CATEGORICAL: {
      const auto& attribute_data =
          train_dataset
              .ColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(
                  attribute_idx)
              ->values();
      const auto na_replacement =
          attribute_column_spec.categorical().most_frequent_value();
      const auto num_attribute_classes =
          attribute_column_spec.categorical().number_of_unique_values();

      result = FindSplitLabelUpliftCategoricalFeatureCategorical(
          selected_examples, weights, attribute_data, label_stats,
          num_attribute_classes, na_replacement, min_num_obs, dt_config,
          attribute_idx, internal_config, best_condition, cache, random);
    } break;

    default:
      LOG(FATAL) << dataset::proto::ColumnType_Name(
                        train_dataset.column(attribute_idx)->type())
                 << " attribute " << train_dataset.column(attribute_idx)->name()
                 << " is not supported.";
  }

  // Condition of the type "Attr is NA".
  if (dt_config.allow_na_conditions()) {
    LOG(FATAL) << "allow_na_conditions not supported";
  }

  return result;
}

// Specialization in the case of uplift with numerical outcome.
SplitSearchResult FindBestCondition(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<row_t>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const NumericalUpliftLabelStats& label_stats, const int32_t attribute_idx,
    proto::NodeCondition* best_condition, utils::RandomEngine* random,
    SplitterPerThreadCache* cache) {
  const int min_num_obs =
      dt_config.in_split_min_examples_check() ? dt_config.min_examples() : 1;
  const auto& attribute_column_spec =
      train_dataset.data_spec().columns(attribute_idx);

  SplitSearchResult result;

  switch (train_dataset.column(attribute_idx)->type()) {
    case dataset::proto::ColumnType::NUMERICAL: {
      const auto& attribute_data =
          train_dataset
              .ColumnWithCast<dataset::VerticalDataset::NumericalColumn>(
                  attribute_idx)
              ->values();
      const auto na_replacement = attribute_column_spec.numerical().mean();

      result = FindSplitLabelUpliftNumericalFeatureNumericalCart(
          selected_examples, weights, attribute_data, label_stats,
          na_replacement, min_num_obs, dt_config, attribute_idx,
          internal_config, best_condition, cache);
    } break;

    case dataset::proto::ColumnType::CATEGORICAL: {
      const auto& attribute_data =
          train_dataset
              .ColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(
                  attribute_idx)
              ->values();
      const auto na_replacement =
          attribute_column_spec.categorical().most_frequent_value();
      const auto num_attribute_classes =
          attribute_column_spec.categorical().number_of_unique_values();

      result = FindSplitLabelUpliftNumericalFeatureCategorical(
          selected_examples, weights, attribute_data, label_stats,
          num_attribute_classes, na_replacement, min_num_obs, dt_config,
          attribute_idx, internal_config, best_condition, cache, random);
    } break;

    default:
      LOG(FATAL) << dataset::proto::ColumnType_Name(
                        train_dataset.column(attribute_idx)->type())
                 << " attribute " << train_dataset.column(attribute_idx)->name()
                 << " is not supported.";
  }

  // Condition of the type "Attr is NA".
  if (dt_config.allow_na_conditions()) {
    LOG(FATAL) << "allow_na_conditions not supported";
  }

  return result;
}

SplitterWorkResponse FindBestConditionFromSplitterWorkRequest(
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const SplitterConcurrencySetup& splitter_concurrency_setup,
    const InternalTrainConfig& internal_config,
    const SplitterWorkRequest& request) {
  SplitterWorkResponse response;
  response.status_idx = request.status_idx;
  response.condition = request.dst_condition;
  response.condition->set_split_score(request.best_score);
  request.splitter_cache->random.seed(request.seed);

  switch (config.task()) {
    case model::proto::Task::CLASSIFICATION: {
      const auto& label_stats =
          utils::down_cast<const ClassificationLabelStats&>(
              request.common->label_stats);

      response.status = FindBestCondition(
          request.common->train_dataset, request.common->selected_examples,
          weights, config, config_link, dt_config, request.common->parent,
          internal_config, label_stats, request.attribute_idx,
          response.condition, &request.splitter_cache->random,
          request.splitter_cache);
    } break;
    case model::proto::Task::REGRESSION:
      if (internal_config.use_hessian_gain) {
        const auto& label_stats =
            utils::down_cast<const RegressionHessianLabelStats&>(
                request.common->label_stats);

        response.status = FindBestCondition(
            request.common->train_dataset, request.common->selected_examples,
            weights, config, config_link, dt_config, request.common->parent,
            internal_config, label_stats, request.attribute_idx,
            response.condition, &request.splitter_cache->random,
            request.splitter_cache);

      } else {
        const auto& label_stats = utils::down_cast<const RegressionLabelStats&>(
            request.common->label_stats);

        response.status = FindBestCondition(
            request.common->train_dataset, request.common->selected_examples,
            weights, config, config_link, dt_config, request.common->parent,
            internal_config, label_stats, request.attribute_idx,
            response.condition, &request.splitter_cache->random,
            request.splitter_cache);
      }
      break;
    default:
      CHECK(false);
  }

  return response;
}

utils::StatusOr<bool> FindBestConditionSingleThreadManager(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<row_t>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const LabelStats& label_stats, proto::NodeCondition* best_condition,
    utils::RandomEngine* random, PerThreadCache* cache) {
  // Single Thread Setup.
  cache->splitter_cache_list.resize(1);

  // Was a least one good split found?
  bool found_good_condition = false;

  switch (dt_config.split_axis_case()) {
    case proto::DecisionTreeTrainingConfig::SPLIT_AXIS_NOT_SET:
    case proto::DecisionTreeTrainingConfig::kAxisAlignedSplit:
      // Nothing to do.
      break;
    case proto::DecisionTreeTrainingConfig::kSparseObliqueSplit:
      switch (config.task()) {
        case model::proto::Task::CLASSIFICATION: {
          const auto& class_label_stats =
              utils::down_cast<const ClassificationLabelStats&>(label_stats);
          ASSIGN_OR_RETURN(
              found_good_condition,
              FindBestConditionSparseOblique(
                  train_dataset, selected_examples, weights, config,
                  config_link, dt_config, parent, internal_config,
                  class_label_stats, best_condition, random,
                  &cache->splitter_cache_list[0]));
        } break;
        case model::proto::Task::REGRESSION:
          if (internal_config.use_hessian_gain) {
            const auto& reg_label_stats =
                utils::down_cast<const RegressionHessianLabelStats&>(
                    label_stats);
            ASSIGN_OR_RETURN(
                found_good_condition,
                FindBestConditionSparseOblique(
                    train_dataset, selected_examples, weights, config,
                    config_link, dt_config, parent, internal_config,
                    reg_label_stats, best_condition, random,
                    &cache->splitter_cache_list[0]));
          } else {
            const auto& reg_label_stats =
                utils::down_cast<const RegressionLabelStats&>(label_stats);
            ASSIGN_OR_RETURN(
                found_good_condition,
                FindBestConditionSparseOblique(
                    train_dataset, selected_examples, weights, config,
                    config_link, dt_config, parent, internal_config,
                    reg_label_stats, best_condition, random,
                    &cache->splitter_cache_list[0]));
          }
          break;
        default:
          return absl::UnimplementedError("Task not implemented");
      }
      break;
  }

  // Get the indices of the attributes to test.
  int remaining_attributes_to_test;
  std::vector<int32_t>& candidate_attributes = cache->candidate_attributes;
  GetCandidateAttributes(config, config_link, dt_config,
                         &remaining_attributes_to_test, &candidate_attributes,
                         random);

  // Index of the next attribute to be tested in "candidate_attributes".
  int candidate_attribute_idx_in_candidate_list = 0;

  while (remaining_attributes_to_test >= 0 &&
         candidate_attribute_idx_in_candidate_list <
             candidate_attributes.size()) {
    // Get the attribute data.
    const int32_t attribute_idx =
        candidate_attributes[candidate_attribute_idx_in_candidate_list++];
    SplitSearchResult result;

    switch (config.task()) {
      case model::proto::Task::CLASSIFICATION: {
        const auto& class_label_stats =
            utils::down_cast<const ClassificationLabelStats&>(label_stats);

        result =
            FindBestCondition(train_dataset, selected_examples, weights, config,
                              config_link, dt_config, parent, internal_config,
                              class_label_stats, attribute_idx, best_condition,
                              random, &cache->splitter_cache_list[0]);
      } break;
      case model::proto::Task::REGRESSION:
        if (internal_config.use_hessian_gain) {
          const auto& reg_label_stats =
              utils::down_cast<const RegressionHessianLabelStats&>(label_stats);

          result = FindBestCondition(train_dataset, selected_examples, weights,
                                     config, config_link, dt_config, parent,
                                     internal_config, reg_label_stats,
                                     attribute_idx, best_condition, random,
                                     &cache->splitter_cache_list[0]);

        } else {
          const auto& reg_label_stats =
              utils::down_cast<const RegressionLabelStats&>(label_stats);

          result = FindBestCondition(train_dataset, selected_examples, weights,
                                     config, config_link, dt_config, parent,
                                     internal_config, reg_label_stats,
                                     attribute_idx, best_condition, random,
                                     &cache->splitter_cache_list[0]);
        }
        break;

      case model::proto::Task::CATEGORICAL_UPLIFT: {
        const auto& uplift_label_stats =
            utils::down_cast<const CategoricalUpliftLabelStats&>(label_stats);
        result =
            FindBestCondition(train_dataset, selected_examples, weights, config,
                              config_link, dt_config, parent, internal_config,
                              uplift_label_stats, attribute_idx, best_condition,
                              random, &cache->splitter_cache_list[0]);
      } break;

      case model::proto::Task::NUMERICAL_UPLIFT: {
        const auto& uplift_label_stats =
            utils::down_cast<const NumericalUpliftLabelStats&>(label_stats);
        result =
            FindBestCondition(train_dataset, selected_examples, weights, config,
                              config_link, dt_config, parent, internal_config,
                              uplift_label_stats, attribute_idx, best_condition,
                              random, &cache->splitter_cache_list[0]);
      } break;

      default:
        return absl::UnimplementedError("Non implemented");
    }
    if (result != SplitSearchResult::kInvalidAttribute) {
      remaining_attributes_to_test--;
      if (result == SplitSearchResult::kBetterSplitFound) {
        found_good_condition = true;
      }
    }
  }

  return found_good_condition;
}

utils::StatusOr<bool> FindBestConditionConcurrentManager(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<row_t>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const SplitterConcurrencySetup& splitter_concurrency_setup,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const LabelStats& label_stats, proto::NodeCondition* best_condition,
    utils::RandomEngine* random, PerThreadCache* cache) {
  int num_threads = splitter_concurrency_setup.num_threads;
  int num_features = config_link.features().size();

  if (num_features == 0) {
    return false;
  }

  SplitterWorkRequestCommon common{
      /*.train_dataset =*/train_dataset,
      /*.selected_examples =*/selected_examples,
      /*.parent =*/parent,
      /*.label_stats =*/label_stats,
  };

  // Prepare cache.
  cache->splitter_cache_list.resize(num_threads);
  cache->work_status_list.resize(num_features);
  cache->condition_list.resize(num_threads * kConditionPoolGrowthFactor);

  if (dt_config.split_axis_case() !=
          proto::DecisionTreeTrainingConfig::kAxisAlignedSplit &&
      dt_config.split_axis_case() !=
          proto::DecisionTreeTrainingConfig::SPLIT_AXIS_NOT_SET) {
    LOG(FATAL) << "Multi-threaded splitter only support axis aligned splits";
  }

  // Get the ordered indices of the attributes to test.
  int min_to_examine;
  std::vector<int32_t>& candidate_attributes = cache->candidate_attributes;
  GetCandidateAttributes(config, config_link, dt_config, &min_to_examine,
                         &candidate_attributes, random);

  // The following keeps track of free slots in the splitter cache and node
  // condition cache
  cache->available_cache_idxs.clear_and_resize(
      cache->splitter_cache_list.size());
  for (int32_t i = 0; i < cache->splitter_cache_list.size(); i++) {
    cache->available_cache_idxs.push_back(i);
  }

  cache->available_condition_idxs.clear_and_resize(
      cache->condition_list.size());
  for (int32_t i = 0; i < cache->condition_list.size(); i++) {
    cache->available_condition_idxs.push_back(i);
  }

  // Get Channel readers and writers.
  auto& processor = *splitter_concurrency_setup.split_finder_processor.get();

  // Helper function to create a WorkRequest.
  auto produce = [&](const int idx, const float best_score) {
    int32_t cache_idx = cache->available_cache_idxs.back();
    cache->available_cache_idxs.pop_back();
    int32_t condition_idx = cache->available_condition_idxs.back();
    cache->available_condition_idxs.pop_back();

    SplitterWorkRequest request;
    request.status_idx = idx;
    request.attribute_idx = candidate_attributes[idx];
    request.dst_condition = &cache->condition_list[condition_idx];
    request.splitter_cache = &cache->splitter_cache_list[cache_idx];
    request.best_score = best_score;
    request.common = &common;
    request.seed = (*random)();

    cache->work_status_list[idx].cache_idx = cache_idx;
    cache->work_status_list[idx].condition_idx = condition_idx;
    cache->work_status_list[idx].condition = nullptr;

    return request;
  };

  // Create initial workload.
  int32_t next_to_examine = 0;
  while (next_to_examine < std::min(num_threads, num_features)) {
    processor.Submit(produce(next_to_examine, best_condition->split_score()));
    next_to_examine++;
  }

  // Start accumulating results and start more workers if necessary.
  int valid_examined_features = 0;
  int process_idx = 0;
  SplitterWorkStatus* best_status = nullptr;

  while (true) {
    auto maybe_response = processor.GetResult();
    if (!maybe_response.has_value()) {
      break;
    }
    SplitterWorkResponse& response = maybe_response.value();

    cache->work_status_list[response.status_idx].condition = response.condition;
    cache->work_status_list[response.status_idx].status = response.status;

    // Release the cache entry as it may now be reused by other workers.
    cache->available_cache_idxs.push_front(
        cache->work_status_list[response.status_idx].cache_idx);

    while (process_idx < next_to_examine &&
           valid_examined_features < min_to_examine &&
           cache->work_status_list[process_idx].condition != nullptr) {
      auto status = &cache->work_status_list[process_idx];
      process_idx++;

      if (status->status != SplitSearchResult::kInvalidAttribute) {
        valid_examined_features++;

        if (best_status == nullptr) {
          best_status = status;
        } else if (status->condition->split_score() >
                   best_status->condition->split_score()) {
          cache->available_condition_idxs.push_front(
              best_status->condition_idx);
          best_status = status;
        } else {
          cache->available_condition_idxs.push_front(status->condition_idx);
        }
      } else {
        cache->available_condition_idxs.push_front(status->condition_idx);
      }
    }

    if (valid_examined_features >= min_to_examine) {
      break;
    }

    // Create more work.
    while (!cache->available_condition_idxs.empty() &&
           !cache->available_cache_idxs.empty() &&
           next_to_examine < num_features) {
      if (best_status != nullptr && best_status->condition->split_score() >
                                        best_condition->split_score()) {
        processor.Submit(
            produce(next_to_examine, best_status->condition->split_score()));
      } else {
        processor.Submit(
            produce(next_to_examine, best_condition->split_score()));
      }
      next_to_examine++;
    }

    // The following condition means that no work is in the pipeline and no more
    // work will be generated.
    if (cache->available_cache_idxs.full()) {
      break;
    }
  }

  // Drain the response channel.
  while (!cache->available_cache_idxs.full()) {
    auto maybe_response = processor.GetResult();
    if (!maybe_response.has_value()) {
      break;
    }
    SplitterWorkResponse& response = maybe_response.value();
    cache->available_cache_idxs.push_front(
        cache->work_status_list[response.status_idx].cache_idx);
  }

  // Move the random generator state to facilitate deterministic behavior.
  random->discard(num_features - next_to_examine);

  if (best_status != nullptr &&
      best_status->condition->split_score() > best_condition->split_score()) {
    *best_condition = *(best_status->condition);
    return true;
  }
  return false;
}

utils::StatusOr<bool> FindBestConditionManager(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<row_t>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const SplitterConcurrencySetup& splitter_concurrency_setup,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const LabelStats& label_stats, proto::NodeCondition* best_condition,
    utils::RandomEngine* random, PerThreadCache* cache) {
  if (splitter_concurrency_setup.concurrent_execution) {
    return FindBestConditionConcurrentManager(
        train_dataset, selected_examples, weights, config, config_link,
        dt_config, splitter_concurrency_setup, parent, internal_config,
        label_stats, best_condition, random, cache);
  }
  return FindBestConditionSingleThreadManager(
      train_dataset, selected_examples, weights, config, config_link, dt_config,
      parent, internal_config, label_stats, best_condition, random, cache);
}

utils::StatusOr<bool> FindBestCondition(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<row_t>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const SplitterConcurrencySetup& splitter_concurrency_setup,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    proto::NodeCondition* best_condition, utils::RandomEngine* random,
    PerThreadCache* cache) {
  switch (config.task()) {
    case model::proto::Task::CLASSIFICATION: {
      if (internal_config.use_hessian_gain) {
        return absl::InternalError("Expect use_hessian_gain=false");
      }
      ClassificationLabelStats label_stat(
          train_dataset
              .ColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(
                  config_link.label())
              ->values());

      const auto& label_column_spec =
          train_dataset.data_spec().columns(config_link.label());
      label_stat.num_label_classes =
          label_column_spec.categorical().number_of_unique_values();

      label_stat.label_distribution.Load(parent.classifier().distribution());

      if (label_stat.label_distribution.NumClasses() >= 1 &&
          label_stat.label_distribution.count(
              dataset::kOutOfDictionaryItemIndex) > 0) {
        return absl::InternalError(
            absl::StrCat("The training label column \"", config.label(),
                         "\" contain out-of-dictionary (=0) values."));
      }

      return FindBestConditionManager(
          train_dataset, selected_examples, weights, config, config_link,
          dt_config, splitter_concurrency_setup, parent, internal_config,
          label_stat, best_condition, random, cache);
    } break;

    case model::proto::Task::REGRESSION: {
      if (internal_config.use_hessian_gain) {
        RegressionHessianLabelStats label_stat(
            train_dataset
                .ColumnWithCast<dataset::VerticalDataset::NumericalColumn>(
                    config_link.label())
                ->values(),
            train_dataset
                .ColumnWithCast<dataset::VerticalDataset::NumericalColumn>(
                    internal_config.hessian_col_idx)
                ->values());

        label_stat.sum_gradient = parent.regressor().sum_gradients();
        label_stat.sum_hessian = parent.regressor().sum_hessians();
        label_stat.sum_weights = parent.regressor().sum_weights();

        return FindBestConditionManager(
            train_dataset, selected_examples, weights, config, config_link,
            dt_config, splitter_concurrency_setup, parent, internal_config,
            label_stat, best_condition, random, cache);
      } else {
        RegressionLabelStats label_stat(
            train_dataset
                .ColumnWithCast<dataset::VerticalDataset::NumericalColumn>(
                    config_link.label())
                ->values());

        label_stat.label_distribution.Load(parent.regressor().distribution());

        return FindBestConditionManager(
            train_dataset, selected_examples, weights, config, config_link,
            dt_config, splitter_concurrency_setup, parent, internal_config,
            label_stat, best_condition, random, cache);
      }
    } break;

    case model::proto::Task::CATEGORICAL_UPLIFT: {
      if (internal_config.use_hessian_gain) {
        return absl::InternalError("Hessian gain not supported for uplift");
      }
      const auto& outcome_spec =
          train_dataset.data_spec().columns(config_link.label());
      const auto& treatment_spec =
          train_dataset.data_spec().columns(config_link.uplift_treatment());

      CategoricalUpliftLabelStats label_stat(
          train_dataset
              .ColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(
                  config_link.label())
              ->values(),
          outcome_spec.categorical().number_of_unique_values(),
          train_dataset
              .ColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(
                  config_link.uplift_treatment())
              ->values(),
          treatment_spec.categorical().number_of_unique_values());

      internal::UpliftLeafToLabelDist(parent.uplift(),
                                      &label_stat.label_distribution);

      return FindBestConditionManager(
          train_dataset, selected_examples, weights, config, config_link,
          dt_config, splitter_concurrency_setup, parent, internal_config,
          label_stat, best_condition, random, cache);
    } break;

    case model::proto::Task::NUMERICAL_UPLIFT: {
      if (internal_config.use_hessian_gain) {
        return absl::InternalError("Hessian gain not supported for uplift");
      }
      const auto& treatment_spec =
          train_dataset.data_spec().columns(config_link.uplift_treatment());

      NumericalUpliftLabelStats label_stat(
          train_dataset
              .ColumnWithCast<dataset::VerticalDataset::NumericalColumn>(
                  config_link.label())
              ->values(),
          train_dataset
              .ColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(
                  config_link.uplift_treatment())
              ->values(),
          treatment_spec.categorical().number_of_unique_values());

      internal::UpliftLeafToLabelDist(parent.uplift(),
                                      &label_stat.label_distribution);

      return FindBestConditionManager(
          train_dataset, selected_examples, weights, config, config_link,
          dt_config, splitter_concurrency_setup, parent, internal_config,
          label_stat, best_condition, random, cache);
    } break;

    default:
      return absl::UnimplementedError("Non implemented");
  }
  return false;
}

SplitSearchResult FindSplitLabelClassificationFeatureNumericalHistogram(
    const std::vector<row_t>& selected_examples,
    const std::vector<float>& weights, const std::vector<float>& attributes,
    const std::vector<int32_t>& labels, const int32_t num_label_classes,
    float na_replacement, const row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    const int32_t attribute_idx, utils::RandomEngine* random,
    proto::NodeCondition* condition) {
  DCHECK(condition != nullptr);

  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForNumericalAttribute(selected_examples, weights, attributes,
                                         &na_replacement);
  }
  // Determine the minimum and maximum values of the attribute.
  float min_value, max_value;
  if (!MinMaxNumericalAttribute(selected_examples, attributes, &min_value,
                                &max_value)) {
    return SplitSearchResult::kInvalidAttribute;
  }
  // There should be at least two different unique values.
  if (min_value == max_value) {
    return SplitSearchResult::kInvalidAttribute;
  }
  // Randomly select some threshold values.
  struct CandidateSplit {
    float threshold;
    utils::IntegerDistributionDouble pos_label_distribution;
    int64_t num_positive_examples_without_weights = 0;
    bool operator<(const CandidateSplit& other) const {
      return threshold < other.threshold;
    }
  };

  const auto bins =
      internal::GenHistogramBins(dt_config.numerical_split().type(),
                                 dt_config.numerical_split().num_candidates(),
                                 attributes, min_value, max_value, random);

  std::vector<CandidateSplit> candidate_splits(bins.size());
  for (int split_idx = 0; split_idx < candidate_splits.size(); split_idx++) {
    auto& candidate_split = candidate_splits[split_idx];
    candidate_split.pos_label_distribution.SetNumClasses(num_label_classes);
    candidate_split.threshold = bins[split_idx];
  }

  // Compute the split score of each threshold.
  for (const auto example_idx : selected_examples) {
    const int32_t label = labels[example_idx];
    const float weight = weights.empty() ? 1.f : weights[example_idx];
    float attribute = attributes[example_idx];
    if (std::isnan(attribute)) {
      attribute = na_replacement;
    }
    auto it_split = std::upper_bound(
        candidate_splits.begin(), candidate_splits.end(), attribute,
        [](const float a, const CandidateSplit& b) { return a < b.threshold; });
    if (it_split == candidate_splits.begin()) {
      continue;
    }
    --it_split;
    it_split->num_positive_examples_without_weights++;
    it_split->pos_label_distribution.Add(label, weight);
  }

  for (int split_idx = candidate_splits.size() - 2; split_idx >= 0;
       split_idx--) {
    const auto& src = candidate_splits[split_idx + 1];
    auto& dst = candidate_splits[split_idx];
    dst.num_positive_examples_without_weights +=
        src.num_positive_examples_without_weights;
    dst.pos_label_distribution.Add(src.pos_label_distribution);
  }

  const double initial_entropy = label_distribution.Entropy();
  utils::BinaryToIntegerConfusionMatrixDouble confusion;
  confusion.SetNumClassesIntDim(num_label_classes);

  // Select the best threshold.
  bool found_split = false;
  for (auto& candidate_split : candidate_splits) {
    if (selected_examples.size() -
                candidate_split.num_positive_examples_without_weights <
            min_num_obs ||
        candidate_split.num_positive_examples_without_weights < min_num_obs) {
      continue;
    }

    confusion.mutable_neg()->Set(label_distribution);
    confusion.mutable_neg()->Sub(candidate_split.pos_label_distribution);
    confusion.mutable_pos()->Set(candidate_split.pos_label_distribution);

    const double final_entropy = confusion.FinalEntropy();
    const double information_gain = initial_entropy - final_entropy;
    if (information_gain > condition->split_score()) {
      condition->set_split_score(information_gain);
      condition->mutable_condition()->mutable_higher_condition()->set_threshold(
          candidate_split.threshold);
      condition->set_attribute(attribute_idx);
      condition->set_num_training_examples_without_weight(
          selected_examples.size());
      condition->set_num_training_examples_with_weight(
          confusion.NumObservations());
      condition->set_num_pos_training_examples_without_weight(
          candidate_split.num_positive_examples_without_weights);
      condition->set_num_pos_training_examples_with_weight(
          confusion.pos().NumObservations());
      condition->set_na_value(na_replacement >= candidate_split.threshold);
      found_split = true;
    }
  }
  return found_split ? SplitSearchResult::kBetterSplitFound
                     : SplitSearchResult::kNoBetterSplitFound;
}

SplitSearchResult FindSplitLabelClassificationFeatureNumericalCart(
    const std::vector<row_t>& selected_examples,
    const std::vector<float>& weights, const std::vector<float>& attributes,
    const std::vector<int32_t>& labels, const int32_t num_label_classes,
    float na_replacement, const row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    const int32_t attribute_idx, const InternalTrainConfig& internal_config,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache) {
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForNumericalAttribute(selected_examples, weights, attributes,
                                         &na_replacement);
  }

  FeatureNumericalBucket::Filler feature_filler(selected_examples.size(),
                                                na_replacement, attributes);

  const auto sorting_strategy = dt_config.internal().sorting_strategy();

  // "Why ==3" ?
  // Categorical attributes always have one class reserved for
  // "out-of-vocabulary" items. The "num_label_classes" takes into account this
  // class. In case of binary classification, "num_label_classes" is 3 (OOB,
  // False, True).
  if (num_label_classes == 3) {
    // Binary classification.
    if (weights.empty()) {
      LabelUnweightedBinaryCategoricalOneValueBucket::Filler label_filler(
          labels);
      LabelUnweightedBinaryCategoricalOneValueBucket::Initializer initializer(
          label_distribution);

      if (sorting_strategy ==
              proto::DecisionTreeTrainingConfig::Internal::PRESORTED ||
          sorting_strategy ==
              proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED) {
        if (!internal_config.preprocessing) {
          LOG(FATAL) << "Preprocessing missing for PRESORTED sorting "
                        "strategy";
        }
        if (sorting_strategy ==
                proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED ||
            IsPresortingOnNumericalSplitMoreEfficient(
                selected_examples.size(),
                internal_config.preprocessing->num_examples())) {
          const auto& sorted_attributes =
              internal_config.preprocessing
                  ->presorted_numerical_features()[attribute_idx];
          return ScanSplitsPresortedSparse<
              FeatureNumericalLabelUnweightedBinaryCategoricalOneValue,
              LabelBinaryCategoricalScoreAccumulator>(
              internal_config.preprocessing->num_examples(), selected_examples,
              sorted_attributes.items, feature_filler, label_filler,
              initializer, min_num_obs, attribute_idx,
              internal_config.duplicated_selected_examples, condition,
              &cache->cache_v2);
        }
      }

      return FindBestSplit_LabelUnweightedBinaryClassificationFeatureNumerical(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
    } else {
      LabelBinaryCategoricalOneValueBucket::Filler label_filler(labels,
                                                                weights);
      LabelBinaryCategoricalOneValueBucket::Initializer initializer(
          label_distribution);

      if (sorting_strategy ==
              proto::DecisionTreeTrainingConfig::Internal::PRESORTED ||
          sorting_strategy ==
              proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED) {
        if (!internal_config.preprocessing) {
          LOG(FATAL) << "Preprocessing missing for PRESORTED sorting "
                        "strategy";
        }
        if (sorting_strategy ==
                proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED ||
            IsPresortingOnNumericalSplitMoreEfficient(
                selected_examples.size(),
                internal_config.preprocessing->num_examples())) {
          const auto& sorted_attributes =
              internal_config.preprocessing
                  ->presorted_numerical_features()[attribute_idx];
          return ScanSplitsPresortedSparse<
              FeatureNumericalLabelBinaryCategoricalOneValue,
              LabelBinaryCategoricalScoreAccumulator>(
              internal_config.preprocessing->num_examples(), selected_examples,
              sorted_attributes.items, feature_filler, label_filler,
              initializer, min_num_obs, attribute_idx,
              internal_config.duplicated_selected_examples, condition,
              &cache->cache_v2);
        }
      }

      return FindBestSplit_LabelBinaryClassificationFeatureNumerical(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
    }
  } else {
    // Multi-class classification.
    LabelCategoricalOneValueBucket::Filler label_filler(labels, weights);
    LabelCategoricalOneValueBucket::Initializer initializer(label_distribution);

    if (sorting_strategy ==
            proto::DecisionTreeTrainingConfig::Internal::PRESORTED ||
        sorting_strategy ==
            proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED) {
      if (!internal_config.preprocessing) {
        LOG(FATAL) << "Preprocessing missing for PRESORTED sorting "
                      "strategy";
      }
      if (sorting_strategy ==
              proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED ||
          IsPresortingOnNumericalSplitMoreEfficient(
              selected_examples.size(),
              internal_config.preprocessing->num_examples())) {
        const auto& sorted_attributes =
            internal_config.preprocessing
                ->presorted_numerical_features()[attribute_idx];
        return ScanSplitsPresortedSparse<
            FeatureNumericalLabelCategoricalOneValue,
            LabelCategoricalScoreAccumulator>(
            internal_config.preprocessing->num_examples(), selected_examples,
            sorted_attributes.items, feature_filler, label_filler, initializer,
            min_num_obs, attribute_idx,
            internal_config.duplicated_selected_examples, condition,
            &cache->cache_v2);
      }
    }

    return FindBestSplit_LabelClassificationFeatureNumerical(
        selected_examples, feature_filler, label_filler, initializer,
        min_num_obs, attribute_idx, condition, &cache->cache_v2);
  }
}

SplitSearchResult FindSplitLabelClassificationFeatureDiscretizedNumericalCart(
    const std::vector<row_t>& selected_examples,
    const std::vector<float>& weights,
    const std::vector<dataset::DiscretizedNumericalIndex>& attributes,
    const int num_bins, const std::vector<int32_t>& labels,
    const int32_t num_label_classes,
    const dataset::DiscretizedNumericalIndex na_replacement,
    const row_t min_num_obs, const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    const int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache) {
  FeatureDiscretizedNumericalBucket::Filler feature_filler(
      num_bins, na_replacement, attributes);
  if (num_label_classes == 3) {
    // Binary classification.
    if (weights.empty()) {
      LabelUnweightedBinaryCategoricalBucket::Filler label_filler(
          labels, {}, label_distribution);
      LabelUnweightedBinaryCategoricalBucket::Initializer initializer(
          label_distribution);

      return FindBestSplit_LabelUnweightedBinaryClassificationFeatureDiscretizedNumerical(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
    } else {
      LabelBinaryCategoricalBucket::Filler label_filler(labels, weights,
                                                        label_distribution);
      LabelBinaryCategoricalBucket::Initializer initializer(label_distribution);

      return FindBestSplit_LabelBinaryClassificationFeatureDiscretizedNumerical(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
    }
  } else {
    // Multi-class classification.
    LabelCategoricalBucket::Filler label_filler(labels, weights,
                                                label_distribution);
    LabelCategoricalBucket::Initializer initializer(label_distribution);

    return FindBestSplit_LabelClassificationFeatureDiscretizedNumerical(
        selected_examples, feature_filler, label_filler, initializer,
        min_num_obs, attribute_idx, condition, &cache->cache_v2);
  }
}

SplitSearchResult FindSplitLabelRegressionFeatureNumericalHistogram(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights, const std::vector<float>& attributes,
    const std::vector<float>& labels, float na_replacement,
    const dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    const int32_t attribute_idx, utils::RandomEngine* random,
    proto::NodeCondition* condition) {
  DCHECK(condition != nullptr);
  DCHECK_EQ(weights.size(), labels.size());

  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForNumericalAttribute(selected_examples, weights, attributes,
                                         &na_replacement);
  }
  // Determine the minimum and maximum values of the attribute.
  float min_value, max_value;
  if (!MinMaxNumericalAttribute(selected_examples, attributes, &min_value,
                                &max_value)) {
    return SplitSearchResult::kInvalidAttribute;
  }

  // There should be at least two different unique values.
  if (min_value == max_value) {
    return SplitSearchResult::kInvalidAttribute;
  }
  // Randomly select some threshold values.
  struct CandidateSplit {
    float threshold;
    utils::NormalDistributionDouble pos_label_dist;
    int64_t num_positive_examples_without_weights = 0;
    bool operator<(const CandidateSplit& other) const {
      return threshold < other.threshold;
    }
  };
  const auto bins =
      internal::GenHistogramBins(dt_config.numerical_split().type(),
                                 dt_config.numerical_split().num_candidates(),
                                 attributes, min_value, max_value, random);

  std::vector<CandidateSplit> candidate_splits(bins.size());
  for (int split_idx = 0; split_idx < candidate_splits.size(); split_idx++) {
    auto& candidate_split = candidate_splits[split_idx];
    candidate_split.threshold = bins[split_idx];
  }

  // Compute the split score of each threshold.
  for (const auto example_idx : selected_examples) {
    const float label = labels[example_idx];
    const float weight = weights[example_idx];
    float attribute = attributes[example_idx];
    if (std::isnan(attribute)) {
      attribute = na_replacement;
    }

    auto it_split = std::upper_bound(
        candidate_splits.begin(), candidate_splits.end(), attribute,
        [](const float a, const CandidateSplit& b) { return a < b.threshold; });
    if (it_split == candidate_splits.begin()) {
      continue;
    }
    --it_split;
    it_split->num_positive_examples_without_weights++;
    it_split->pos_label_dist.Add(label, weight);
  }

  for (int split_idx = candidate_splits.size() - 2; split_idx >= 0;
       split_idx--) {
    const auto& src = candidate_splits[split_idx + 1];
    auto& dst = candidate_splits[split_idx];
    dst.num_positive_examples_without_weights +=
        src.num_positive_examples_without_weights;
    dst.pos_label_dist.Add(src.pos_label_dist);
  }

  // Select the best threshold.
  const double initial_variance = label_distribution.Var();
  int best_candidate_split_idx = -1;
  double best_variance_reduction = condition->split_score();
  utils::NormalDistributionDouble neg_label_dist;
  for (int candidate_split_idx = 0;
       candidate_split_idx < candidate_splits.size(); candidate_split_idx++) {
    const auto& candidate_split = candidate_splits[candidate_split_idx];
    if (selected_examples.size() -
                candidate_split.num_positive_examples_without_weights <
            min_num_obs ||
        candidate_split.num_positive_examples_without_weights < min_num_obs) {
      continue;
    }
    neg_label_dist = label_distribution;
    neg_label_dist.Sub(candidate_split.pos_label_dist);
    const double frac_pos = candidate_split.pos_label_dist.NumObservations() /
                            (candidate_split.pos_label_dist.NumObservations() +
                             neg_label_dist.NumObservations());
    const double final_variance =
        frac_pos * candidate_split.pos_label_dist.Var() +
        (1 - frac_pos) * neg_label_dist.Var();
    const double variance_reduction = initial_variance - final_variance;
    if (variance_reduction > best_variance_reduction) {
      best_variance_reduction = variance_reduction;
      best_candidate_split_idx = candidate_split_idx;
    }
  }

  if (best_candidate_split_idx == -1) {
    return SplitSearchResult::kNoBetterSplitFound;
  } else {
    const auto& candidate_split = candidate_splits[best_candidate_split_idx];
    condition->set_split_score(best_variance_reduction);
    condition->mutable_condition()->mutable_higher_condition()->set_threshold(
        candidate_split.threshold);
    condition->set_attribute(attribute_idx);
    condition->set_num_training_examples_without_weight(
        selected_examples.size());
    condition->set_num_training_examples_with_weight(
        candidate_split.pos_label_dist.NumObservations() +
        neg_label_dist.NumObservations());
    condition->set_num_pos_training_examples_without_weight(
        candidate_split.num_positive_examples_without_weights);
    condition->set_num_pos_training_examples_with_weight(
        candidate_split.pos_label_dist.NumObservations());
    condition->set_na_value(na_replacement >= candidate_split.threshold);
    return SplitSearchResult::kBetterSplitFound;
  }
}

SplitSearchResult FindSplitLabelHessianRegressionFeatureNumericalCart(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights, const std::vector<float>& attributes,
    const std::vector<float>& gradients, const std::vector<float>& hessians,
    float na_replacement, row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, double sum_gradient,
    double sum_hessian, double sum_weights, int32_t attribute_idx,
    const InternalTrainConfig& internal_config, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache) {
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForNumericalAttribute(selected_examples, weights, attributes,
                                         &na_replacement);
  }

  FeatureNumericalBucket::Filler feature_filler(selected_examples.size(),
                                                na_replacement, attributes);

  LabelHessianNumericalOneValueBucket::Filler label_filler(
      gradients, hessians, weights, internal_config.hessian_l1,
      internal_config.hessian_l2_numerical);

  LabelHessianNumericalOneValueBucket::Initializer initializer(
      sum_gradient, sum_hessian, sum_weights, internal_config.hessian_l1,
      internal_config.hessian_l2_numerical,
      dt_config.internal().hessian_split_score_subtract_parent());

  if (dt_config.internal().sorting_strategy() ==
          proto::DecisionTreeTrainingConfig::Internal::PRESORTED ||
      dt_config.internal().sorting_strategy() ==
          proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED) {
    if (!internal_config.preprocessing) {
      LOG(FATAL) << "Preprocessing missing for PRESORTED sorting "
                    "strategy";
    }
    if (dt_config.internal().sorting_strategy() ==
            proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED ||
        IsPresortingOnNumericalSplitMoreEfficient(
            selected_examples.size(),
            internal_config.preprocessing->num_examples())) {
      const auto& sorted_attributes =
          internal_config.preprocessing
              ->presorted_numerical_features()[attribute_idx];

      return ScanSplitsPresortedSparse<
          FeatureNumericalLabelHessianNumericalOneValue,
          LabelHessianNumericalScoreAccumulator>(
          internal_config.preprocessing->num_examples(), selected_examples,
          sorted_attributes.items, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx,
          internal_config.duplicated_selected_examples, condition,
          &cache->cache_v2);
    }
  }

  return FindBestSplit_LabelHessianRegressionFeatureNumerical(
      selected_examples, feature_filler, label_filler, initializer, min_num_obs,
      attribute_idx, condition, &cache->cache_v2);
}

SplitSearchResult
FindSplitLabelHessianRegressionFeatureDiscretizedNumericalCart(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights,
    const std::vector<dataset::DiscretizedNumericalIndex>& attributes,
    int num_bins, const std::vector<float>& gradients,
    const std::vector<float>& hessians, float na_replacement, row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, double sum_gradient,
    double sum_hessian, double sum_weights, int32_t attribute_idx,
    const InternalTrainConfig& internal_config, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache) {
  FeatureDiscretizedNumericalBucket::Filler feature_filler(
      num_bins, na_replacement, attributes);

  LabelHessianNumericalBucket::Filler label_filler(
      gradients, hessians, weights, internal_config.hessian_l1,
      internal_config.hessian_l2_categorical);

  LabelHessianNumericalBucket::Initializer initializer(
      sum_gradient, sum_hessian, sum_weights, internal_config.hessian_l1,
      internal_config.hessian_l2_categorical,
      dt_config.internal().hessian_split_score_subtract_parent());

  return FindBestSplit_LabelHessianRegressionFeatureDiscretizedNumerical(
      selected_examples, feature_filler, label_filler, initializer, min_num_obs,
      attribute_idx, condition, &cache->cache_v2);
}

SplitSearchResult FindSplitLabelRegressionFeatureNumericalCart(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights, const std::vector<float>& attributes,
    const std::vector<float>& labels, float na_replacement,
    const dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    const int32_t attribute_idx, const InternalTrainConfig& internal_config,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache) {
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForNumericalAttribute(selected_examples, weights, attributes,
                                         &na_replacement);
  }

  FeatureNumericalBucket::Filler feature_filler(selected_examples.size(),
                                                na_replacement, attributes);

  LabelNumericalOneValueBucket::Filler label_filler(labels, weights);

  LabelNumericalOneValueBucket::Initializer initializer(label_distribution);
  const auto sorting_strategy = dt_config.internal().sorting_strategy();
  if (sorting_strategy ==
          proto::DecisionTreeTrainingConfig::Internal::PRESORTED ||
      sorting_strategy ==
          proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED) {
    if (!internal_config.preprocessing) {
      LOG(FATAL) << "Preprocessing missing for PRESORTED sorting "
                    "strategy";
    }

    if (sorting_strategy ==
            proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED ||
        IsPresortingOnNumericalSplitMoreEfficient(
            -selected_examples.size(),
            internal_config.preprocessing->num_examples())) {
      const auto& sorted_attributes =
          internal_config.preprocessing
              ->presorted_numerical_features()[attribute_idx];
      return ScanSplitsPresortedSparse<FeatureNumericalLabelNumericalOneValue,
                                       LabelNumericalScoreAccumulator>(
          internal_config.preprocessing->num_examples(), selected_examples,
          sorted_attributes.items, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx,
          internal_config.duplicated_selected_examples, condition,
          &cache->cache_v2);
    }
  }

  return FindBestSplit_LabelRegressionFeatureNumerical(
      selected_examples, feature_filler, label_filler, initializer, min_num_obs,
      attribute_idx, condition, &cache->cache_v2);
}

SplitSearchResult FindSplitLabelRegressionFeatureDiscretizedNumericalCart(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights,
    const std::vector<dataset::DiscretizedNumericalIndex>& attributes,
    const int num_bins, const std::vector<float>& labels,
    const dataset::DiscretizedNumericalIndex na_replacement,
    const row_t min_num_obs, const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    const int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache) {
  FeatureDiscretizedNumericalBucket::Filler feature_filler(
      num_bins, na_replacement, attributes);

  LabelNumericalBucket::Filler label_filler(labels, weights);

  LabelNumericalBucket::Initializer initializer(label_distribution);

  return FindBestSplit_LabelRegressionFeatureDiscretizedNumerical(
      selected_examples, feature_filler, label_filler, initializer, min_num_obs,
      attribute_idx, condition, &cache->cache_v2);
}

SplitSearchResult FindSplitLabelClassificationFeatureNA(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights,
    const dataset::VerticalDataset::AbstractColumn* attributes,
    const std::vector<int32_t>& labels, const int32_t num_label_classes,
    const dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    const int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache) {
  FeatureIsMissingBucket::Filler feature_filler(attributes);
  if (num_label_classes == 3) {
    // Binary classification.
    if (weights.empty()) {
      LabelUnweightedBinaryCategoricalBucket::Filler label_filler(
          labels, {}, label_distribution);

      LabelUnweightedBinaryCategoricalBucket::Initializer initializer(
          label_distribution);

      return FindBestSplit_LabelUnweightedBinaryClassificationFeatureNACart(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
    } else {
      LabelBinaryCategoricalBucket::Filler label_filler(labels, weights,
                                                        label_distribution);

      LabelBinaryCategoricalBucket::Initializer initializer(label_distribution);

      return FindBestSplit_LabelBinaryClassificationFeatureNACart(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
    }
  } else {
    // Multi-class classification.
    LabelCategoricalBucket::Filler label_filler(labels, weights,
                                                label_distribution);
    LabelCategoricalBucket::Initializer initializer(label_distribution);

    return FindBestSplit_LabelClassificationFeatureNACart(
        selected_examples, feature_filler, label_filler, initializer,
        min_num_obs, attribute_idx, condition, &cache->cache_v2);
  }
}

SplitSearchResult FindSplitLabelRegressionFeatureNA(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights,
    const dataset::VerticalDataset::AbstractColumn* attributes,
    const std::vector<float>& labels,
    const dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    const int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache) {
  FeatureIsMissingBucket::Filler feature_filler(attributes);

  LabelNumericalBucket::Filler label_filler(labels, weights);

  LabelNumericalBucket::Initializer initializer(label_distribution);

  return FindBestSplit_LabelRegressionFeatureNACart(
      selected_examples, feature_filler, label_filler, initializer, min_num_obs,
      attribute_idx, condition, &cache->cache_v2);
}

SplitSearchResult FindSplitLabelHessianRegressionFeatureNA(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights,
    const dataset::VerticalDataset::AbstractColumn* attributes,
    const std::vector<float>& gradients, const std::vector<float>& hessians,
    const dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const double sum_gradient, const double sum_hessian,
    const double sum_weights, const int32_t attribute_idx,
    const InternalTrainConfig& internal_config, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache) {
  FeatureIsMissingBucket::Filler feature_filler(attributes);

  LabelHessianNumericalBucket::Filler label_filler(
      gradients, hessians, weights, internal_config.hessian_l1,
      internal_config.hessian_l2_numerical);

  LabelHessianNumericalBucket::Initializer initializer(
      sum_gradient, sum_hessian, sum_weights, internal_config.hessian_l1,
      internal_config.hessian_l2_numerical,
      dt_config.internal().hessian_split_score_subtract_parent());

  return FindBestSplit_LabelHessianRegressionFeatureNACart(
      selected_examples, feature_filler, label_filler, initializer, min_num_obs,
      attribute_idx, condition, &cache->cache_v2);
}

SplitSearchResult FindSplitLabelClassificationFeatureBoolean(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights, const std::vector<char>& attributes,
    const std::vector<int32_t>& labels, const int32_t num_label_classes,
    bool na_replacement, dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache) {
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForBooleanAttribute(selected_examples, weights, attributes,
                                       &na_replacement);
  }

  FeatureBooleanBucket::Filler feature_filler(na_replacement, attributes);

  if (num_label_classes == 3) {
    // Binary classification.
    if (weights.empty()) {
      // Unweighted classes
      LabelUnweightedBinaryCategoricalBucket::Filler label_filler(
          labels, {}, label_distribution);

      LabelUnweightedBinaryCategoricalBucket::Initializer initializer(
          label_distribution);

      return FindBestSplit_LabelUnweightedBinaryClassificationFeatureBooleanCart(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
    } else {
      LabelBinaryCategoricalBucket::Filler label_filler(labels, weights,
                                                        label_distribution);

      LabelBinaryCategoricalBucket::Initializer initializer(label_distribution);

      return FindBestSplit_LabelBinaryClassificationFeatureBooleanCart(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
    }
  } else {
    // Multi-class classification.
    LabelCategoricalBucket::Filler label_filler(labels, weights,
                                                label_distribution);

    LabelCategoricalBucket::Initializer initializer(label_distribution);

    return FindBestSplit_LabelClassificationFeatureBooleanCart(
        selected_examples, feature_filler, label_filler, initializer,
        min_num_obs, attribute_idx, condition, &cache->cache_v2);
  }
}

SplitSearchResult FindSplitLabelRegressionFeatureBoolean(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights, const std::vector<char>& attributes,
    const std::vector<float>& labels, bool na_replacement,
    dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache) {
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForBooleanAttribute(selected_examples, weights, attributes,
                                       &na_replacement);
  }

  FeatureBooleanBucket::Filler feature_filler(na_replacement, attributes);
  LabelNumericalBucket::Filler label_filler(labels, weights);
  LabelNumericalBucket::Initializer initializer(label_distribution);

  return FindBestSplit_LabelRegressionFeatureBooleanCart(
      selected_examples, feature_filler, label_filler, initializer, min_num_obs,
      attribute_idx, condition, &cache->cache_v2);
}

SplitSearchResult FindSplitLabelHessianRegressionFeatureBoolean(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights, const std::vector<char>& attributes,
    const std::vector<float>& gradients, const std::vector<float>& hessians,
    bool na_replacement, const dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const double sum_gradient, const double sum_hessian,
    const double sum_weights, const int32_t attribute_idx,
    const InternalTrainConfig& internal_config, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache) {
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForBooleanAttribute(selected_examples, weights, attributes,
                                       &na_replacement);
  }

  FeatureBooleanBucket::Filler feature_filler(na_replacement, attributes);
  LabelHessianNumericalBucket::Filler label_filler(
      gradients, hessians, weights, internal_config.hessian_l1,
      internal_config.hessian_l2_numerical);

  LabelHessianNumericalBucket::Initializer initializer(
      sum_gradient, sum_hessian, sum_weights, internal_config.hessian_l1,
      internal_config.hessian_l2_numerical,
      dt_config.internal().hessian_split_score_subtract_parent());

  return FindBestSplit_LabelHessianRegressionFeatureBooleanCart(
      selected_examples, feature_filler, label_filler, initializer, min_num_obs,
      attribute_idx, condition, &cache->cache_v2);
}

SplitSearchResult FindSplitLabelHessianRegressionFeatureCategorical(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights, const std::vector<int32_t>& attributes,
    const std::vector<float>& gradients, const std::vector<float>& hessians,
    const int32_t num_attribute_classes, int32_t na_replacement,
    const dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const double sum_gradient, const double sum_hessian,
    const double sum_weights, const int32_t attribute_idx,
    const InternalTrainConfig& internal_config, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache, utils::RandomEngine* random) {
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForCategoricalAttribute(selected_examples, weights,
                                           attributes, num_attribute_classes,
                                           &na_replacement);
  }

  FeatureCategoricalBucket::Filler feature_filler(num_attribute_classes,
                                                  na_replacement, attributes);
  LabelHessianNumericalBucket::Filler label_filler(
      gradients, hessians, weights, internal_config.hessian_l1,
      internal_config.hessian_l2_categorical);

  LabelHessianNumericalBucket::Initializer initializer(
      sum_gradient, sum_hessian, sum_weights, internal_config.hessian_l1,
      internal_config.hessian_l2_categorical,
      dt_config.internal().hessian_split_score_subtract_parent());

  const auto algorithm =
      (num_attribute_classes < dt_config.categorical().arity_limit_for_random())
          ? dt_config.categorical().algorithm_case()
          : proto::Categorical::kRandom;

  switch (algorithm) {
    case proto::Categorical::ALGORITHM_NOT_SET:
    case proto::Categorical::kCart:
      return FindBestSplit_LabelHessianRegressionFeatureCategoricalCart(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);

    case proto::Categorical::kRandom:
      return FindBestSplit_LabelHessianRegressionFeatureCategoricalRandom(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx,
          NumTrialsForRandomCategoricalSplit(dt_config.categorical().random()),
          condition, &cache->cache_v2, random);

    default:
      LOG(FATAL) << "Non supported";
  }
}

SplitSearchResult FindSplitLabelRegressionFeatureCategorical(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights, const std::vector<int32_t>& attributes,
    const std::vector<float>& labels, const int32_t num_attribute_classes,
    int32_t na_replacement, const dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    const int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache, utils::RandomEngine* random) {
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForCategoricalAttribute(selected_examples, weights,
                                           attributes, num_attribute_classes,
                                           &na_replacement);
  }

  FeatureCategoricalBucket::Filler feature_filler(num_attribute_classes,
                                                  na_replacement, attributes);
  LabelNumericalBucket::Filler label_filler(labels, weights);

  LabelNumericalBucket::Initializer initializer(label_distribution);

  const auto algorithm =
      (num_attribute_classes < dt_config.categorical().arity_limit_for_random())
          ? dt_config.categorical().algorithm_case()
          : proto::Categorical::kRandom;

  switch (algorithm) {
    case proto::Categorical::ALGORITHM_NOT_SET:
    case proto::Categorical::kCart:
      return FindBestSplit_LabelRegressionFeatureCategoricalCart(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);

    case proto::Categorical::kRandom:
      return FindBestSplit_LabelRegressionFeatureCategoricalRandom(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx,
          NumTrialsForRandomCategoricalSplit(dt_config.categorical().random()),
          condition, &cache->cache_v2, random);

    default:
      LOG(FATAL) << "Non supported";
  }
}

SplitSearchResult
FindSplitLabelClassificationFeatureCategoricalSetGreedyForward(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights,
    const dataset::VerticalDataset::CategoricalSetColumn& attributes,
    const std::vector<int32_t>& labels, const int32_t num_attribute_classes,
    const int32_t num_label_classes,
    const dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    const int32_t attribute_idx, proto::NodeCondition* condition,
    utils::RandomEngine* random) {
  // Bitmap of available attribute values. During the course of the algorithm,
  // an attribute value is available if:
  //  - It is selected by the initial random sampling of candidate attribute
  //  values.
  //  - It is not (yet) selected in the positive set.
  //  - It is not pure in the negative examples i.e. it is not present in all
  //  or in none of the non-selected examples (ps: Initially, all the examples
  //  are non-selected).
  std::vector<bool> candidate_attributes_bitmap(num_attribute_classes, true);
  // The "positive attribute set" are the attribute values that, if present
  // in the example, evaluates the node condition as true.
  std::vector<int> positive_attributes_vector;
  // Bitmap of the example that are already in the positive set i.e. for which
  // the condition defined by "positive_attributes_vector" is positive.
  // Instead of being indexed by the example_idx, this bitmap is indexed by
  // "selected_examples" i.e. "positive_selected_example_bitmap[i]==true"
  // means that "selected_examples[i]" is selected.
  std::vector<bool> positive_selected_example_bitmap(selected_examples.size(),
                                                     false);
  // Weighted and non weighted distribution of the labels in the positive and
  // negative sets.
  utils::BinaryToIntegerConfusionMatrixDouble split_label_distribution;
  utils::BinaryToIntegerConfusionMatrixInt64
      split_label_distribution_no_weights;
  split_label_distribution.SetNumClassesIntDim(num_label_classes);
  split_label_distribution_no_weights.SetNumClassesIntDim(num_label_classes);
  // All the examples are initially in the negative set.
  *split_label_distribution.mutable_neg() = label_distribution;
  // Number of example (with weights) where the attribute value (an attribute
  // value is a set of categorical items) that contains the i-th  categorical
  // items.
  std::vector<int64_t> count_examples_without_weights_by_attribute_class(
      num_attribute_classes);
  // Count per categorical item value.
  const auto& attribute_values = attributes.values();
  const auto& attribute_bank = attributes.bank();
  for (const auto example_idx : selected_examples) {
    for (auto bank_idx = attribute_values[example_idx].first;
         bank_idx < attribute_values[example_idx].second; bank_idx++) {
      const auto value = attribute_bank[bank_idx];
      count_examples_without_weights_by_attribute_class[value]++;
    }
    split_label_distribution_no_weights.Add(false, labels[example_idx]);
  }
  // Sample-out items.
  if (!internal::MaskPureSampledOrPrunedItemsForCategoricalSetGreedySelection(
          dt_config, num_attribute_classes, selected_examples,
          count_examples_without_weights_by_attribute_class,
          &candidate_attributes_bitmap, random)) {
    return SplitSearchResult::kInvalidAttribute;
  }

  // Contains, for each example, the index in the "attribute_bank"
  // corresponding to the next candidate attribute value
  // "candidate_attr_value".
  //
  // When initialized, this corresponds to the index of the first values for
  // this particular example: i.e. running_attr_bank_idx[select_idx] ==
  // attribute_values[selected_examples[select_idx]].first.
  std::vector<dataset::VerticalDataset::row_t> running_attr_bank_idx(
      selected_examples.size());

  const int max_iterations =
      dt_config.categorical_set_greedy_forward().max_selected_items();
  int num_iterations = 0;

  const double initial_entropy = split_label_distribution.InitEntropy();
  // Information gain of the current condition i.e.
  // "positive_attributes_vector".
  double information_gain = 0;
  while (true) {
    if (max_iterations > 0 && num_iterations >= max_iterations) {
      break;
    }

    // Initialize the running attribute bank index.
    for (size_t select_idx = 0; select_idx < selected_examples.size();
         select_idx++) {
      const auto example_idx = selected_examples[select_idx];
      running_attr_bank_idx[select_idx] = attribute_values[example_idx].first;
    }
    // Search for the attribute item that maximize the information gain. Note:
    // We ignore attribute that reduce the current information gain.
    double best_candidate_information_gain = information_gain;
    int best_attr_value = -1;
    for (int candidate_attr_value = 0;
         candidate_attr_value < num_attribute_classes; candidate_attr_value++) {
      if (!candidate_attributes_bitmap[candidate_attr_value]) {
        // The attribute value was already selected or was excluded from the
        // search space.
        continue;
      }
      // Compute the information gains of the already selected values
      // "positive_attributes_vector" and the candidate value
      // "candidate_attr_value".
      utils::BinaryToIntegerConfusionMatrixDouble
          candidate_split_label_distribution = split_label_distribution;
      int64_t num_preset_in_negative_set = 0;
      int64_t num_absent_in_negative_set = 0;
      for (size_t select_idx = 0; select_idx < selected_examples.size();
           select_idx++) {
        const auto example_idx = selected_examples[select_idx];
        if (positive_selected_example_bitmap[select_idx]) {
          // The example is already in the positive set.
          continue;
        }

        // Search if X = attribute_bank[ attribute_values[example_idx].first,
        // attribute_values[example_idx].second ] contains the current
        // candidate attribute value "candidate_attr_value".
        //
        // We use "running_attr_bank_idx[select_idx]" that contains the index
        // in "X" of the last tested candidate attribute.
        //
        // Note: "X" does not contains duplicates and is sorted in increasing
        // order. The candidate attributes "candidate_attr_value" are testing
        // in increasing order.
        bool match = false;
        dataset::VerticalDataset::row_t last_attr;
        while (
            running_attr_bank_idx[select_idx] <
                attribute_values[example_idx].second &&
            (last_attr = attribute_bank[running_attr_bank_idx[select_idx]]) <=
                candidate_attr_value) {
          running_attr_bank_idx[select_idx]++;
          if (last_attr == candidate_attr_value) {
            match = true;
            break;
          }
        }

        if (match) {
          num_preset_in_negative_set++;
          // Add the example to the positive set and remove it from the
          // negative.
          float weight = weights.empty() ? 1.f : weights[example_idx];
          candidate_split_label_distribution.mutable_pos()->Add(
              labels[example_idx], weight);
          candidate_split_label_distribution.mutable_neg()->Add(
              labels[example_idx], -weight);
        } else {
          num_absent_in_negative_set++;
        }
      }
      // Remove the attribute from the candidate set if the attribute is pure
      // for the current negative set.
      if (num_absent_in_negative_set == 0 ||
          num_absent_in_negative_set == selected_examples.size()) {
        candidate_attributes_bitmap[candidate_attr_value] = false;
        continue;
      }

      const double candidate_information_gain =
          initial_entropy - candidate_split_label_distribution.FinalEntropy();
      if (candidate_information_gain > best_candidate_information_gain) {
        // Best score so far.
        best_candidate_information_gain = candidate_information_gain;
        best_attr_value = candidate_attr_value;
      }
    }
    // Check if a satisfying attribute item was found.
    if (best_attr_value == -1) {
      break;
    }
    // Add the attribute item to the positive set.
    candidate_attributes_bitmap[best_attr_value] = false;
    positive_attributes_vector.push_back(best_attr_value);
    information_gain = best_candidate_information_gain;
    // Update the label distributions in the positive and negative sets.
    for (size_t select_idx = 0; select_idx < selected_examples.size();
         select_idx++) {
      const auto example_idx = selected_examples[select_idx];
      if (positive_selected_example_bitmap[select_idx]) {
        // The example is already in the positive set.
        continue;
      }
      const bool match =
          (attribute_values[example_idx].first <
           attribute_values[example_idx].second) &&
          std::binary_search(
              attribute_bank.begin() + attribute_values[example_idx].first,
              attribute_bank.begin() + attribute_values[example_idx].second,
              best_attr_value);
      if (match) {
        positive_selected_example_bitmap[select_idx] = true;
        float weight = weights.empty() ? 1.f : weights[example_idx];
        split_label_distribution.mutable_pos()->Add(labels[example_idx],
                                                    weight);
        split_label_distribution.mutable_neg()->Add(labels[example_idx],
                                                    -weight);
        split_label_distribution_no_weights.mutable_pos()->Add(
            labels[example_idx], 1);
        split_label_distribution_no_weights.mutable_neg()->Add(
            labels[example_idx], -1);
      }
    }

    num_iterations++;
  }

  if (information_gain > condition->split_score()) {
    condition->set_na_value(false);
    SetConditionHelper(information_gain, attribute_idx,
                       split_label_distribution,
                       split_label_distribution_no_weights, condition);
    // Assign the positive set to the condition.
    std::sort(positive_attributes_vector.begin(),
              positive_attributes_vector.end());
    SetPositiveAttributeSetOfCategoricalContainsCondition(
        positive_attributes_vector, num_attribute_classes, condition);
    return SplitSearchResult::kBetterSplitFound;
  } else {
    return SplitSearchResult::kNoBetterSplitFound;
  }
}

SplitSearchResult FindSplitLabelRegressionFeatureCategoricalSetGreedyForward(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights,
    const dataset::VerticalDataset::CategoricalSetColumn& attributes,
    const std::vector<float>& labels, int32_t num_attribute_classes,
    row_t min_num_obs, const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    int32_t attribute_idx, proto::NodeCondition* condition,
    utils::RandomEngine* random) {
  DCHECK_EQ(weights.size(), labels.size());
  // Bitmap of available attribute values. During the course of the algorithm,
  // an attribute value is available if:
  //  - It is selected by the initial random sampling of candidate attribute
  //  values.
  //  - It is not (yet) selected in the positive set.
  //  - It is not pure in the negative examples i.e. it is not present in all
  //  or in none of the non-selected examples (ps: Initially, all the examples
  //  are non-selected).
  std::vector<bool> candidate_attributes_bitmap(num_attribute_classes, true);
  // The "positive attribute set" are the attribute values that, if present
  // in the example, evaluates the node condition as true.
  std::vector<int> positive_attributes_vector;
  // Bitmap of the example that are already in the positive set i.e. for which
  // the condition defined by "positive_attributes_vector" is positive.
  // Instead of being indexed by the example_idx, this bitmap is indexed by
  // "selected_examples" i.e. "positive_selected_example_bitmap[i]==true"
  // means that "selected_examples[i]" is selected.
  std::vector<bool> positive_selected_example_bitmap(selected_examples.size(),
                                                     false);
  // Weighted and non weighted distribution of the labels in the positive and
  // negative sets.
  utils::BinaryToNormalDistributionDouble split_label_distribution;
  utils::BinaryToNormalDistributionDouble split_label_distribution_no_weights;
  // All the examples are initially in the negative set.
  *split_label_distribution.mutable_neg() = label_distribution;
  // Number of example (with weights) where the attribute value (an attribute
  // value is a set of categorical items) that contains the i-th  categorical
  // items.
  std::vector<int64_t> count_examples_without_weights_by_attribute_class(
      num_attribute_classes);
  // Count per categorical item value.
  const auto& attribute_values = attributes.values();
  const auto& attribute_bank = attributes.bank();
  for (const auto example_idx : selected_examples) {
    for (auto bank_idx = attribute_values[example_idx].first;
         bank_idx < attribute_values[example_idx].second; bank_idx++) {
      const auto value = attribute_bank[bank_idx];
      count_examples_without_weights_by_attribute_class[value]++;
    }
    split_label_distribution_no_weights.Add(false, labels[example_idx]);
  }

  // Sample-out items.
  if (!internal::MaskPureSampledOrPrunedItemsForCategoricalSetGreedySelection(
          dt_config, num_attribute_classes, selected_examples,
          count_examples_without_weights_by_attribute_class,
          &candidate_attributes_bitmap, random)) {
    return SplitSearchResult::kInvalidAttribute;
  }

  // Contains, for each example, the index in the "attribute_bank"
  // corresponding to the next candidate attribute value
  // "candidate_attr_value".
  //
  // When initialized, this corresponds to the index of the first values for
  // this particular example: i.e. running_attr_bank_idx[select_idx] ==
  // attribute_values[selected_examples[select_idx]].first.
  std::vector<dataset::VerticalDataset::row_t> running_attr_bank_idx(
      selected_examples.size());

  const double initial_variance = label_distribution.Var();
  // Variance reduction of the current condition i.e.
  // "positive_attributes_vector".
  double variance_reduction = 0;
  while (true) {
    // Initialize the running attribute bank index.
    for (size_t select_idx = 0; select_idx < selected_examples.size();
         select_idx++) {
      const auto example_idx = selected_examples[select_idx];
      running_attr_bank_idx[select_idx] = attribute_values[example_idx].first;
    }
    // Find the attribute with best immediate variance reduction.
    int best_attr_value;
    double best_candidate_variance_reduction;
    std::tie(best_attr_value, best_candidate_variance_reduction) =
        GetAttributeValueWithMaximumVarianceReduction(
            variance_reduction, num_attribute_classes, split_label_distribution,
            selected_examples, positive_selected_example_bitmap,
            attribute_values, attribute_bank, weights, labels, initial_variance,
            &running_attr_bank_idx, &candidate_attributes_bitmap);
    // Check if a satisfying attribute item was found.
    if (best_attr_value == -1) {
      break;
    }
    // Add the attribute item to the positive set.
    candidate_attributes_bitmap[best_attr_value] = false;
    positive_attributes_vector.push_back(best_attr_value);
    variance_reduction = best_candidate_variance_reduction;
    // Update the label distributions in the positive and negative sets.
    for (size_t select_idx = 0; select_idx < selected_examples.size();
         select_idx++) {
      const auto example_idx = selected_examples[select_idx];
      if (positive_selected_example_bitmap[select_idx]) {
        // The example is already in the positive set.
        continue;
      }
      const bool match =
          (attribute_values[example_idx].first <
           attribute_values[example_idx].second) &&
          std::binary_search(
              attribute_bank.begin() + attribute_values[example_idx].first,
              attribute_bank.begin() + attribute_values[example_idx].second,
              best_attr_value);
      if (match) {
        positive_selected_example_bitmap[select_idx] = true;
        split_label_distribution.mutable_pos()->Add(labels[example_idx],
                                                    weights[example_idx]);
        split_label_distribution.mutable_neg()->Add(labels[example_idx],
                                                    -weights[example_idx]);
        split_label_distribution_no_weights.mutable_pos()->Add(
            labels[example_idx]);
        split_label_distribution_no_weights.mutable_neg()->Sub(
            labels[example_idx]);
      }
    }
  }

  if (variance_reduction > condition->split_score()) {
    condition->set_na_value(false);
    SetConditionHelper(variance_reduction, attribute_idx,
                       split_label_distribution,
                       split_label_distribution_no_weights, condition);
    // Assign the positive set to the condition.
    std::sort(positive_attributes_vector.begin(),
              positive_attributes_vector.end());
    SetPositiveAttributeSetOfCategoricalContainsCondition(
        positive_attributes_vector, num_attribute_classes, condition);
    return SplitSearchResult::kBetterSplitFound;
  } else {
    return SplitSearchResult::kNoBetterSplitFound;
  }
}

template <typename LabelBucket, typename ExampleBucketSet,
          typename LabelScoreAccumulator>
SplitSearchResult FindSplitLabelClassificationFeatureCategorical(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights, const std::vector<int32_t>& attributes,
    const std::vector<int32_t>& labels, int32_t num_attribute_classes,
    int32_t num_label_classes, int32_t na_replacement, row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    int32_t attribute_idx, utils::RandomEngine* random,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache) {
  FeatureCategoricalBucket::Filler feature_filler(num_attribute_classes,
                                                  na_replacement, attributes);
  typename LabelBucket::Filler label_filler(labels, weights,
                                            label_distribution);

  typename LabelBucket::Initializer initializer(label_distribution);

  // Create buckets.
  ExampleBucketSet& example_set_accumulator =
      *GetCachedExampleBucketSet<ExampleBucketSet>(&cache->cache_v2);
  FillExampleBucketSet<ExampleBucketSet, /*require_label_sorting=*/false>(
      selected_examples, feature_filler, label_filler, &example_set_accumulator,
      &cache->cache_v2);

  // Scanner for the "one label value vs others".
  const auto one_vs_other_scan = [&]() -> SplitSearchResult {
    // Value and index of the buckets.
    auto& bucket_order = cache->cache_v2.bucket_order;
    bucket_order.resize(example_set_accumulator.items.size());

    SplitSearchResult split_status = SplitSearchResult::kInvalidAttribute;
    for (int32_t positive_label_value = 0;
         positive_label_value < num_label_classes; positive_label_value++) {
      if (label_distribution.count(positive_label_value) == 0) {
        // Never observed label value.
        continue;
      }
      if (num_label_classes == 3 && positive_label_value == 1) {
        // "True vs others" or "False vs others" are equivalent for binary
        // classification.
        continue;
      }

#ifdef SIMPLE_ML_DEBUG_DECISION_TREE_SPLITTER
      LOG(INFO) << "Scan CatCat one vs others for positive_label_value:"
                << positive_label_value
                << " Item size :" << sizeof(example_set_accumulator.items[0]);
#endif

      // Order value of the buckets.
      for (int bucket_idx = 0; bucket_idx < bucket_order.size(); bucket_idx++) {
        const auto& bucket = example_set_accumulator.items[bucket_idx];
        const float ratio_positive_label =
            bucket.label.SafeProportionOrMinusInfinity(positive_label_value);
        DCHECK(!std::isnan(ratio_positive_label));
        bucket_order[bucket_idx] = {ratio_positive_label, bucket_idx};
      }

      // Sort the bucket indices.
      std::sort(bucket_order.begin(), bucket_order.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });

      // Scan the buckets in order.
      const auto scan_result =
          ScanSplitsCustomOrder<ExampleBucketSet, LabelScoreAccumulator>(
              bucket_order, feature_filler, initializer,
              example_set_accumulator, selected_examples.size(), min_num_obs,
              attribute_idx, condition, &cache->cache_v2);
      if (scan_result < split_status) {
        split_status = scan_result;
      }
    }
    return split_status;
  };

  // Scanner for the "one hot" type condition i.e. conditions of the type:
  // "attribute == value".
  //
  // Note: In the majority of cases, one-hot (on attribute value) is worst that
  // "one class vs others". This is however a common solution, and this code is
  // present for comparison purpose.
  const auto one_hot_scan = [&]() -> SplitSearchResult {
    CHECK_EQ(example_set_accumulator.items.size(), num_attribute_classes);

    std::uniform_real_distribution<float> sampling_dist;

    auto& neg = *GetCachedLabelScoreAccumulator<LabelScoreAccumulator>(
        false, &cache->cache_v2);
    auto& pos = *GetCachedLabelScoreAccumulator<LabelScoreAccumulator>(
        true, &cache->cache_v2);

    initializer.InitFull(&pos);
    const double weighted_num_examples = pos.WeightedNumExamples();

    double best_score = condition->split_score();
    bool tried_one_split = false;
    int best_bucket_idx = -1;

    for (int attribute_value = 0; attribute_value < num_attribute_classes;
         attribute_value++) {
      if (dt_config.categorical().one_hot().sampling() < 1.f &&
          sampling_dist(*random) >
              dt_config.categorical().one_hot().sampling()) {
        continue;
      }
      const auto bucket_idx = attribute_value;
      const auto& item = example_set_accumulator.items[bucket_idx];

      const int64_t num_pos_examples = item.label.count;
      const int64_t num_neg_examples =
          selected_examples.size() - item.label.count;

      // Enough examples?
      if (num_pos_examples < min_num_obs || num_neg_examples < min_num_obs) {
        continue;
      }

      initializer.InitFull(&neg);
      initializer.InitEmpty(&pos);

      item.label.SubToScoreAcc(&neg);
      item.label.AddToScoreAcc(&pos);

      const auto score = Score<>(initializer, weighted_num_examples, pos, neg);
      tried_one_split = true;

      if (score > best_score) {
        // Memorize the split.
        best_bucket_idx = bucket_idx;
        best_score = score;
        condition->set_num_pos_training_examples_without_weight(
            num_pos_examples);
        condition->set_num_pos_training_examples_with_weight(
            pos.WeightedNumExamples());
      }
    }

    if (best_bucket_idx != -1) {
      // Finalize the best found split.
      condition->set_na_value(na_replacement == best_bucket_idx);
      SetPositiveAttributeSetOfCategoricalContainsCondition(
          {best_bucket_idx}, num_attribute_classes, condition);

      condition->set_attribute(attribute_idx);
      condition->set_num_training_examples_without_weight(
          selected_examples.size());
      condition->set_num_training_examples_with_weight(weighted_num_examples);
      condition->set_split_score(best_score);
      return SplitSearchResult::kBetterSplitFound;
    } else {
      return tried_one_split ? SplitSearchResult::kNoBetterSplitFound
                             : SplitSearchResult::kInvalidAttribute;
    }
  };

  const auto algorithm =
      (num_attribute_classes < dt_config.categorical().arity_limit_for_random())
          ? dt_config.categorical().algorithm_case()
          : proto::Categorical::kRandom;

  switch (algorithm) {
    case proto::Categorical::ALGORITHM_NOT_SET:
    case proto::Categorical::kCart:
      return one_vs_other_scan();

    case proto::Categorical::kOneHot:
      return one_hot_scan();
      break;

    case proto::Categorical::kRandom:
      return ScanSplitsRandomBuckets<ExampleBucketSet, LabelScoreAccumulator>(
          feature_filler, label_filler, initializer, example_set_accumulator,
          selected_examples.size(), min_num_obs, attribute_idx,
          NumTrialsForRandomCategoricalSplit(dt_config.categorical().random()),
          condition, &cache->cache_v2, random);
  }
}

SplitSearchResult FindSplitLabelClassificationFeatureCategorical(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights, const std::vector<int32_t>& attributes,
    const std::vector<int32_t>& labels, int32_t num_attribute_classes,
    int32_t num_label_classes, int32_t na_replacement, row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    int32_t attribute_idx, utils::RandomEngine* random,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache) {
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForCategoricalAttribute(selected_examples, weights,
                                           attributes, num_attribute_classes,
                                           &na_replacement);
  }

  if (num_label_classes == 3) {
    // Binary classification.
    if (weights.empty()) {
      return FindSplitLabelClassificationFeatureCategorical<
          LabelUnweightedBinaryCategoricalBucket,
          FeatureCategoricalLabelUnweightedBinaryCategorical,
          LabelBinaryCategoricalScoreAccumulator>(
          selected_examples, weights, attributes, labels, num_attribute_classes,
          num_label_classes, na_replacement, min_num_obs, dt_config,
          label_distribution, attribute_idx, random, condition, cache);
    } else {
      return FindSplitLabelClassificationFeatureCategorical<
          LabelBinaryCategoricalBucket,
          FeatureCategoricalLabelBinaryCategorical,
          LabelBinaryCategoricalScoreAccumulator>(
          selected_examples, weights, attributes, labels, num_attribute_classes,
          num_label_classes, na_replacement, min_num_obs, dt_config,
          label_distribution, attribute_idx, random, condition, cache);
    }
  } else {
    // Multi-class classification.
    return FindSplitLabelClassificationFeatureCategorical<
        LabelCategoricalBucket, FeatureCategoricalLabelCategorical,
        LabelCategoricalScoreAccumulator>(
        selected_examples, weights, attributes, labels, num_attribute_classes,
        num_label_classes, na_replacement, min_num_obs, dt_config,
        label_distribution, attribute_idx, random, condition, cache);
  }
}

SplitSearchResult FindSplitLabelUpliftCategoricalFeatureNumericalCart(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights, const std::vector<float>& attributes,
    const CategoricalUpliftLabelStats& label_stats, float na_replacement,
    dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, int32_t attribute_idx,
    const InternalTrainConfig& internal_config, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache) {
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForNumericalAttribute(selected_examples, weights, attributes,
                                         &na_replacement);
  }

  FeatureNumericalBucket::Filler feature_filler(selected_examples.size(),
                                                na_replacement, attributes);

  LabelUpliftCategoricalOneValueBucket::Initializer initializer(
      label_stats.label_distribution,
      dt_config.uplift().min_examples_in_treatment(),
      dt_config.uplift().split_score());
  LabelUpliftCategoricalOneValueBucket::Filler label_filler(
      label_stats.outcome_values, label_stats.treatment_values, weights);

  // TODO(gbm): Add support for-presorted splitting.

  return FindBestSplit_LabelUpliftClassificationFeatureNumerical(
      selected_examples, feature_filler, label_filler, initializer, min_num_obs,
      attribute_idx, condition, &cache->cache_v2);
}

SplitSearchResult FindSplitLabelUpliftNumericalFeatureNumericalCart(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights, const std::vector<float>& attributes,
    const NumericalUpliftLabelStats& label_stats, float na_replacement,
    dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, int32_t attribute_idx,
    const InternalTrainConfig& internal_config, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache) {
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForNumericalAttribute(selected_examples, weights, attributes,
                                         &na_replacement);
  }

  FeatureNumericalBucket::Filler feature_filler(selected_examples.size(),
                                                na_replacement, attributes);

  LabelUpliftNumericalOneValueBucket::Initializer initializer(
      label_stats.label_distribution,
      dt_config.uplift().min_examples_in_treatment(),
      dt_config.uplift().split_score());
  LabelUpliftNumericalOneValueBucket::Filler label_filler(
      label_stats.outcome_values, label_stats.treatment_values, weights);

  // TODO(gbm): Add support for pre-sorted splitting.

  return FindBestSplit_LabelUpliftNumericalFeatureNumerical(
      selected_examples, feature_filler, label_filler, initializer, min_num_obs,
      attribute_idx, condition, &cache->cache_v2);
}

SplitSearchResult FindSplitLabelUpliftCategoricalFeatureCategorical(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights, const std::vector<int32_t>& attributes,
    const CategoricalUpliftLabelStats& label_stats, int num_attribute_classes,
    int32_t na_replacement, dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, int32_t attribute_idx,
    const InternalTrainConfig& internal_config, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache, utils::RandomEngine* random) {
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForCategoricalAttribute(selected_examples, weights,
                                           attributes, num_attribute_classes,
                                           &na_replacement);
  }

  FeatureCategoricalBucket::Filler feature_filler(num_attribute_classes,
                                                  na_replacement, attributes);

  LabelUpliftCategoricalBucket::Initializer initializer(
      label_stats.label_distribution,
      dt_config.uplift().min_examples_in_treatment(),
      dt_config.uplift().split_score());
  LabelUpliftCategoricalBucket::Filler label_filler(
      label_stats.label_distribution, label_stats.outcome_values,
      label_stats.treatment_values, weights);

  // TODO(gbm): Add support for pre-sorted splitting.

  const auto algorithm =
      (num_attribute_classes < dt_config.categorical().arity_limit_for_random())
          ? dt_config.categorical().algorithm_case()
          : proto::Categorical::kRandom;

  switch (algorithm) {
    case proto::Categorical::ALGORITHM_NOT_SET:
    case proto::Categorical::kCart:
      return FindBestSplit_LabelUpliftClassificationFeatureCategoricalCart(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);

    case proto::Categorical::kRandom:
      return FindBestSplit_LabelUpliftClassificationFeatureCategoricalRandom(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx,
          NumTrialsForRandomCategoricalSplit(dt_config.categorical().random()),
          condition, &cache->cache_v2, random);

    default:
      LOG(FATAL) << "Non supported";
  }
}

SplitSearchResult FindSplitLabelUpliftNumericalFeatureCategorical(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights, const std::vector<int32_t>& attributes,
    const NumericalUpliftLabelStats& label_stats, int num_attribute_classes,
    int32_t na_replacement, dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, int32_t attribute_idx,
    const InternalTrainConfig& internal_config, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache, utils::RandomEngine* random) {
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForCategoricalAttribute(selected_examples, weights,
                                           attributes, num_attribute_classes,
                                           &na_replacement);
  }

  FeatureCategoricalBucket::Filler feature_filler(num_attribute_classes,
                                                  na_replacement, attributes);

  LabelUpliftNumericalBucket::Initializer initializer(
      label_stats.label_distribution,
      dt_config.uplift().min_examples_in_treatment(),
      dt_config.uplift().split_score());
  LabelUpliftNumericalBucket::Filler label_filler(
      label_stats.label_distribution, label_stats.outcome_values,
      label_stats.treatment_values, weights);

  // TODO(gbm): Add support for pre-sorted splitting.

  const auto algorithm =
      (num_attribute_classes < dt_config.categorical().arity_limit_for_random())
          ? dt_config.categorical().algorithm_case()
          : proto::Categorical::kRandom;

  switch (algorithm) {
    case proto::Categorical::ALGORITHM_NOT_SET:
    case proto::Categorical::kCart:
      return FindBestSplit_LabelUpliftNumericalFeatureCategoricalCart(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);

    case proto::Categorical::kRandom:
      return FindBestSplit_LabelUpliftNumericalFeatureCategoricalRandom(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx,
          NumTrialsForRandomCategoricalSplit(dt_config.categorical().random()),
          condition, &cache->cache_v2, random);

    default:
      LOG(FATAL) << "Non supported";
  }
}

void GetCandidateAttributes(
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    int* num_attributes_to_test, std::vector<int32_t>* candidate_attributes,
    utils::RandomEngine* random) {
  candidate_attributes->assign(config_link.features().begin(),
                               config_link.features().end());
  std::shuffle(candidate_attributes->begin(), candidate_attributes->end(),
               *random);

  // User specified number of candidate attributes.
  if (dt_config.has_num_candidate_attributes_ratio() &&
      dt_config.num_candidate_attributes_ratio() >= 0) {
    if (dt_config.has_num_candidate_attributes() &&
        dt_config.num_candidate_attributes() > 0) {
      LOG(WARNING) << "Both \"num_candidate_attributes\" and "
                      "\"num_candidate_attributes_ratio\" are specified. "
                      "Ignoring \"num_candidate_attributes\".";
    }
    *num_attributes_to_test =
        static_cast<int>(std::ceil(dt_config.num_candidate_attributes_ratio() *
                                   config_link.features_size()));
  } else {
    *num_attributes_to_test = dt_config.num_candidate_attributes();
  }

  // Automatic number of attribute selection logic.
  if (*num_attributes_to_test == 0) {
    switch (config.task()) {
      default:
      case model::proto::Task::CATEGORICAL_UPLIFT:
      case model::proto::Task::CLASSIFICATION:
        *num_attributes_to_test = static_cast<int>(
            ceil(std::sqrt(static_cast<double>(candidate_attributes->size()))));
        break;
      case model::proto::Task::REGRESSION:
        *num_attributes_to_test = static_cast<int>(
            ceil(static_cast<double>(candidate_attributes->size()) / 3));
        break;
    }
  }

  // Special value to use all the available attributes.
  if (*num_attributes_to_test == -1) {
    *num_attributes_to_test = static_cast<int>(candidate_attributes->size());
  }

  // Make sure we don't select more than the available attributes.
  *num_attributes_to_test = std::min(
      *num_attributes_to_test, static_cast<int>(candidate_attributes->size()));
}

void GenerateRandomImputation(
    const dataset::VerticalDataset& src, const std::vector<int>& attributes,
    const std::vector<dataset::VerticalDataset::row_t>& examples,
    dataset::VerticalDataset* dst, utils::RandomEngine* random) {
  CHECK_EQ(dst->ncol(), 0) << "The destination dataset should be empty.";
  dst->set_data_spec(src.data_spec());
  CHECK_OK(dst->CreateColumnsFromDataspec());
  dst->set_nrow(examples.size());
  for (const auto col_idx : attributes) {
    GenerateRandomImputationOnColumn(src.column(col_idx), examples,
                                     dst->mutable_column(col_idx), random);
  }
}

void GenerateRandomImputationOnColumn(
    const dataset::VerticalDataset::AbstractColumn* src,
    const std::vector<dataset::VerticalDataset::row_t>& examples,
    dataset::VerticalDataset::AbstractColumn* dst,
    utils::RandomEngine* random) {
  CHECK_EQ(src->type(), dst->type());
  // Extract the indices of the example with non-na values i.e. the candidate
  // for sampling.
  std::vector<dataset::VerticalDataset::row_t> non_na_examples;
  for (const auto example_idx : examples) {
    if (!src->IsNa(example_idx)) {
      non_na_examples.push_back(example_idx);
    }
  }
  std::uniform_int_distribution<dataset::VerticalDataset::row_t>
      non_na_example_dist(
          0, std::max(static_cast<dataset::VerticalDataset::row_t>(0),
                      static_cast<dataset::VerticalDataset::row_t>(
                          non_na_examples.size()) -
                          1));
  std::vector<row_t> source_indices = examples;
  if (non_na_examples.empty()) {
    src->ExtractAndAppend(source_indices, dst);
    return;
  }

  dataset::VerticalDataset::row_t local_example_idx = 0;
  for (const auto example_idx : examples) {
    if (src->IsNa(example_idx)) {
      // Sample a random non-na value.
      source_indices[local_example_idx] =
          non_na_examples[non_na_example_dist(*random)];
    }
    local_example_idx++;
  }
  src->ExtractAndAppend(source_indices, dst);
}

void SetRegressionLabelDistribution(
    const dataset::VerticalDataset& dataset,
    const std::vector<row_t>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfigLinking& config_link, proto::Node* node) {
  DCHECK(!weights.empty());
  const auto* const labels =
      dataset.ColumnWithCast<dataset::VerticalDataset::NumericalColumn>(
          config_link.label());
  utils::NormalDistributionDouble label_distribution;
  for (const row_t example_idx : selected_examples) {
    label_distribution.Add(labels->values()[example_idx], weights[example_idx]);
  }
  label_distribution.Save(node->mutable_regressor()->mutable_distribution());
  node->mutable_regressor()->set_top_value(label_distribution.Mean());
}

void SetDefaultHyperParameters(proto::DecisionTreeTrainingConfig* config) {
  if (!config->numerical_split().has_num_candidates()) {
    switch (config->numerical_split().type()) {
      case proto::NumericalSplit::HISTOGRAM_RANDOM:
        config->mutable_numerical_split()->set_num_candidates(1);
        break;
      case proto::NumericalSplit::HISTOGRAM_EQUAL_WIDTH:
        config->mutable_numerical_split()->set_num_candidates(255);
        break;
      default:
        break;
    }
  }

  if (config->split_axis_case() ==
      proto::DecisionTreeTrainingConfig::SPLIT_AXIS_NOT_SET) {
    config->mutable_axis_aligned_split();
  }

  // Disable pre-sorting if not supported by the splitters.
  if (config->internal().sorting_strategy() ==
          proto::DecisionTreeTrainingConfig::Internal::PRESORTED ||
      config->internal().sorting_strategy() ==
          proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED) {
    if (config->has_sparse_oblique_split() ||
        config->missing_value_policy() !=
            proto::DecisionTreeTrainingConfig::GLOBAL_IMPUTATION) {
      config->mutable_internal()->set_sorting_strategy(
          proto::DecisionTreeTrainingConfig::Internal::IN_NODE);
    }
  }
}

template <class T, class S, class C>
S& Container(std::priority_queue<T, S, C>& q) {
  struct HackedQueue : private std::priority_queue<T, S, C> {
    static S& Container(std::priority_queue<T, S, C>& q) {
      return q.*&HackedQueue::c;
    }
  };
  return HackedQueue::Container(q);
}

absl::Status GrowTreeBestFirstGlobal(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<row_t>& train_example_idxs,
    const std::vector<row_t>* optional_leaf_examples,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const model::proto::DeploymentConfig& deployment,
    const SplitterConcurrencySetup& splitter_concurrency_setup,
    const std::vector<float>& weights,
    const InternalTrainConfig& internal_config, NodeWithChildren* root,
    utils::RandomEngine* random) {
  if (optional_leaf_examples) {
    return absl::InvalidArgumentError(
        "honest trees are not (yet) supported with "
        "growing_strategy_best_first_global strategy.");
  }

  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::RANDOM_LOCAL_IMPUTATION) {
    return absl::InvalidArgumentError(
        "Random local imputation not supported in best first global "
        "tree growth.");
  }

  PerThreadCache cache;

  struct CandidateSplit {
    // Split.
    proto::NodeCondition condition;
    // Indices of examples in the node.
    std::vector<row_t> example_idxs;
    // Global score of the split.
    float score;
    // The currently leaf node.
    NodeWithChildren* node;
    // Depth of the node.
    int depth;

    bool operator<(const CandidateSplit& other) const {
      return score < other.score;
    }
  };

  // List of candidate splits.
  std::priority_queue<CandidateSplit> candidate_splits;

  // Initialize a node and update the list of candidate splits with a given
  // node.
  const auto ingest_node = [&](const std::vector<row_t>& example_idxs,
                               NodeWithChildren* node,
                               const int depth) -> absl::Status {
    internal_config.set_leaf_value_functor(train_dataset, example_idxs, weights,
                                           config, config_link, node);

    if (example_idxs.size() < dt_config.min_examples() ||
        (dt_config.max_depth() >= 0 && depth >= dt_config.max_depth())) {
      // Stop the grow of the branch.
      node->FinalizeAsLeaf(dt_config.store_detailed_label_distribution());
      return absl::OkStatus();
    }
    proto::NodeCondition condition;
    ASSIGN_OR_RETURN(
        const auto has_better_condition,
        FindBestCondition(train_dataset, example_idxs, weights, config,
                          config_link, dt_config, splitter_concurrency_setup,
                          node->node(), internal_config, &condition, random,
                          &cache));
    if (!has_better_condition) {
      // No good condition found. Close the branch.
      node->FinalizeAsLeaf(dt_config.store_detailed_label_distribution());
      return absl::OkStatus();
    }

    const float score = condition.split_score() * example_idxs.size();
    candidate_splits.push({/*.condition =*/std::move(condition),
                           /*.example_idxs =*/example_idxs,
                           /*.score =*/score,
                           /*.node =*/node,
                           /*.depth =*/depth});
    return absl::OkStatus();
  };

  RETURN_IF_ERROR(ingest_node(train_example_idxs, root, /*depth=*/0));

  // Total number of nodes in the tree.
  int num_nodes = 1;

  const int max_num_nodes =
      dt_config.growing_strategy_best_first_global().max_num_nodes();

  std::vector<row_t> positive_examples;
  std::vector<row_t> negative_examples;

  while (!candidate_splits.empty() &&
         (max_num_nodes < 0 || num_nodes < max_num_nodes) &&
         (!internal_config.timeout.has_value() ||
          internal_config.timeout >= absl::Now())) {
    // Ensure the candidate set is not larger than  "max_num_nodes". Note:
    // There is not need for mode than "max_num_nodes" candidate splits.
    while (max_num_nodes >= 0 && candidate_splits.size() > max_num_nodes) {
      candidate_splits.top().node->FinalizeAsLeaf(
          dt_config.store_detailed_label_distribution());
      candidate_splits.pop();
    }

    // Split the node.
    auto split = candidate_splits.top();
    candidate_splits.pop();

    *split.node->mutable_node()->mutable_condition() = split.condition;
    split.node->CreateChildren();
    split.node->FinalizeAsNonLeaf(
        dt_config.keep_non_leaf_label_distribution(),
        dt_config.store_detailed_label_distribution());

    const auto& condition = split.node->node().condition();

    // Add new candidate splits for children.
    RETURN_IF_ERROR(internal::SplitExamples(
        train_dataset, split.example_idxs, condition,
        /*dataset_is_dense=*/false,
        dt_config.internal_error_on_wrong_splitter_statistics(),
        &positive_examples, &negative_examples));

    RETURN_IF_ERROR(ingest_node(
        positive_examples, split.node->mutable_pos_child(), split.depth + 1));
    RETURN_IF_ERROR(ingest_node(
        negative_examples, split.node->mutable_neg_child(), split.depth + 1));
    num_nodes++;
  }

  // Finalize the remaining candidates.
  while (!candidate_splits.empty()) {
    candidate_splits.top().node->FinalizeAsLeaf(
        dt_config.store_detailed_label_distribution());
    candidate_splits.pop();
  }
  return absl::OkStatus();
}

absl::Status DecisionTreeTrain(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<row_t>& selected_examples,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const model::proto::DeploymentConfig& deployment,
    const std::vector<float>& weights, utils::RandomEngine* random,
    DecisionTree* dt, const InternalTrainConfig& internal_config) {
  // Decide if execution should happen in single-thread or concurrent mode.

  const std::vector<row_t>* effective_selected_examples;
  const std::vector<row_t>* leaf_examples;
  std::vector<row_t> selected_examples_buffer;
  std::vector<row_t> leaf_examples_buffer;

  if (dt_config.has_honest()) {
    // Split the examples in two parts. One ("selected_examples_buffer") will be
    // used to infer the structure of the trees while the second
    // ("leaf_examples_buffer") will be used to determine the leaf values (i.e.
    // the predictions).

    const float leaf_rate = dt_config.honest().ratio_leaf_examples();
    std::uniform_real_distribution<float> dist_01;

    // Reduce the risk of std::vector re-allocations.
    const float error_margin = 1.1f;
    leaf_examples_buffer.reserve(selected_examples.size() * leaf_rate *
                                 error_margin);
    selected_examples_buffer.reserve(selected_examples.size() *
                                     (1.f - leaf_rate) * error_margin);

    auto* effective_random = random;
    utils::RandomEngine fixed_random(12345678);
    if (dt_config.honest().fixed_separation()) {
      effective_random = &fixed_random;
    }

    for (const auto& example : selected_examples) {
      if (dist_01(*effective_random) < leaf_rate) {
        leaf_examples_buffer.push_back(example);
      } else {
        selected_examples_buffer.push_back(example);
      }
    }
    effective_selected_examples = &selected_examples_buffer;
    leaf_examples = &leaf_examples_buffer;
  } else {
    effective_selected_examples = &selected_examples;
    leaf_examples = nullptr;
  }

  const bool force_single_thread =
      internal_config.num_threads > 1 &&
      dt_config.split_axis_case() ==
          proto::DecisionTreeTrainingConfig::kSparseObliqueSplit;

  SplitterConcurrencySetup splitter_concurrency_setup;
  if (internal_config.num_threads <= 1 || force_single_thread) {
    splitter_concurrency_setup.concurrent_execution = false;
    return DecisionTreeCoreTrain(train_dataset, *effective_selected_examples,
                                 leaf_examples, config, config_link, dt_config,
                                 deployment, splitter_concurrency_setup,
                                 weights, random, internal_config, dt);
  } else {
    splitter_concurrency_setup.concurrent_execution = true;
    splitter_concurrency_setup.num_threads = internal_config.num_threads;
  }

  splitter_concurrency_setup.split_finder_processor =
      absl::make_unique<SplitterFinderStreamProcessor>(
          "SplitFinder", internal_config.num_threads,
          [&](SplitterWorkRequest request) -> SplitterWorkResponse {
            return FindBestConditionFromSplitterWorkRequest(
                weights, config, config_link, dt_config,
                splitter_concurrency_setup, internal_config, request);
          });
  splitter_concurrency_setup.split_finder_processor->StartWorkers();

  return DecisionTreeCoreTrain(train_dataset, *effective_selected_examples,
                               leaf_examples, config, config_link, dt_config,
                               deployment, splitter_concurrency_setup, weights,
                               random, internal_config, dt);
}

absl::Status DecisionTreeCoreTrain(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<row_t>& selected_examples,
    const std::vector<dataset::VerticalDataset::row_t>* optional_leaf_examples,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const model::proto::DeploymentConfig& deployment,
    const SplitterConcurrencySetup& splitter_concurrency_setup,
    const std::vector<float>& weights, utils::RandomEngine* random,
    const InternalTrainConfig& internal_config, DecisionTree* dt) {
  dt->CreateRoot();
  PerThreadCache cache;
  switch (dt_config.growing_strategy_case()) {
    case proto::DecisionTreeTrainingConfig::GROWING_STRATEGY_NOT_SET:
    case proto::DecisionTreeTrainingConfig::kGrowingStrategyLocal:
      return NodeTrain(train_dataset, selected_examples, optional_leaf_examples,
                       config, config_link, dt_config, deployment,
                       splitter_concurrency_setup, weights, 1, internal_config,
                       dt->mutable_root(), random, &cache);
      break;
    case proto::DecisionTreeTrainingConfig::kGrowingStrategyBestFirstGlobal:
      return GrowTreeBestFirstGlobal(
          train_dataset, selected_examples, optional_leaf_examples, config,
          config_link, dt_config, deployment, splitter_concurrency_setup,
          weights, internal_config, dt->mutable_root(), random);
      break;
  }
}

absl::Status NodeTrain(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<row_t>& selected_examples,
    const std::vector<row_t>* optional_leaf_examples,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const model::proto::DeploymentConfig& deployment,
    const SplitterConcurrencySetup& splitter_concurrency_setup,
    const std::vector<float>& weights, const int32_t depth,
    const InternalTrainConfig& internal_config, NodeWithChildren* node,
    utils::RandomEngine* random, PerThreadCache* cache) {
  if (selected_examples.empty()) {
    return absl::InternalError("No example feed to the no trainer");
  }
  // Set the node value (i.e. the label distribution).
  internal_config.set_leaf_value_functor(train_dataset, selected_examples,
                                         weights, config, config_link, node);
  node->mutable_node()->set_num_pos_training_examples_without_weight(
      selected_examples.size());

  if (selected_examples.size() < dt_config.min_examples() ||
      (dt_config.max_depth() >= 0 && depth >= dt_config.max_depth()) ||
      (internal_config.timeout.has_value() &&
       internal_config.timeout < absl::Now())) {
    if (optional_leaf_examples) {
      // Override the leaf values.
      internal_config.set_leaf_value_functor(train_dataset,
                                             *optional_leaf_examples, weights,
                                             config, config_link, node);
    }

    // Stop the growth of the branch.
    node->FinalizeAsLeaf(dt_config.store_detailed_label_distribution());
    return absl::OkStatus();
  }
  // Dataset used to train this node.
  const dataset::VerticalDataset* local_train_dataset = &train_dataset;
  const std::vector<row_t>* local_selected_examples = &selected_examples;
  // If true, the entire dataset "local_train_dataset" is composed of training
  // examples for this node. If false, only the subset of
  // "local_train_dataset" indexed by "selected_examples" are to be considered
  // for this node i.e. local_train_dataset[selected_examples[i]].
  bool local_train_dataset_is_compact = false;

  // Extract the random local imputation.
  dataset::VerticalDataset random_local_imputation_train_dataset;
  std::vector<row_t> random_local_imputation_selected_examples;
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::RANDOM_LOCAL_IMPUTATION) {
    std::vector<int> label_and_input_features(config_link.features().begin(),
                                              config_link.features().end());
    label_and_input_features.push_back(config_link.label());
    GenerateRandomImputation(train_dataset, label_and_input_features,
                             selected_examples,
                             &random_local_imputation_train_dataset, random);
    random_local_imputation_selected_examples.resize(selected_examples.size());
    std::iota(random_local_imputation_selected_examples.begin(),
              random_local_imputation_selected_examples.end(), 0);

    local_train_dataset = &random_local_imputation_train_dataset;
    local_selected_examples = &random_local_imputation_selected_examples;
    local_train_dataset_is_compact = true;
  }

  // Determine the best split.
  ASSIGN_OR_RETURN(
      const auto has_better_condition,
      FindBestCondition(
          *local_train_dataset, *local_selected_examples, weights, config,
          config_link, dt_config, splitter_concurrency_setup, node->node(),
          internal_config, node->mutable_node()->mutable_condition(), random,
          cache));
  if (!has_better_condition) {
    // No good condition found. Close the branch.
    node->FinalizeAsLeaf(dt_config.store_detailed_label_distribution());
    return absl::OkStatus();
  }
  CHECK_EQ(selected_examples.size(),
           node->node().condition().num_training_examples_without_weight());
  node->CreateChildren();
  node->FinalizeAsNonLeaf(dt_config.keep_non_leaf_label_distribution(),
                          dt_config.store_detailed_label_distribution());

  // Ensure the per-depth cache is allocated.
  while (cache->per_depth.size() < depth) {
    cache->per_depth.push_back(absl::make_unique<PerThreadCache::PerDepth>());
  }

  // Separate the positive and negative examples.
  auto& per_depth_cache = *cache->per_depth[depth - 1];
  std::vector<row_t>& positive_examples = per_depth_cache.positive_examples;
  std::vector<row_t>& negative_examples = per_depth_cache.negative_examples;
  RETURN_IF_ERROR(internal::SplitExamples(
      *local_train_dataset, selected_examples, node->node().condition(),
      local_train_dataset_is_compact,
      dt_config.internal_error_on_wrong_splitter_statistics(),
      &positive_examples, &negative_examples));

  // Separate the positive and negative examples used only to determine the node
  // value.
  std::vector<row_t>* positive_node_only_examples = nullptr;
  std::vector<row_t>* negative_node_only_examples = nullptr;
  if (optional_leaf_examples) {
    positive_node_only_examples = &per_depth_cache.positive_node_only_examples;
    negative_node_only_examples = &per_depth_cache.negative_node_only_examples;
    RETURN_IF_ERROR(internal::SplitExamples(
        train_dataset, *optional_leaf_examples, node->node().condition(), false,
        dt_config.internal_error_on_wrong_splitter_statistics(),
        positive_node_only_examples, negative_node_only_examples,
        /*examples_are_training_examples=*/false));
  }

  // Positive child.
  RETURN_IF_ERROR(NodeTrain(
      train_dataset, positive_examples, positive_node_only_examples, config,
      config_link, dt_config, deployment, splitter_concurrency_setup, weights,
      depth + 1, internal_config, node->mutable_pos_child(), random, cache));
  // Negative child.
  RETURN_IF_ERROR(NodeTrain(
      train_dataset, negative_examples, negative_node_only_examples, config,
      config_link, dt_config, deployment, splitter_concurrency_setup, weights,
      depth + 1, internal_config, node->mutable_neg_child(), random, cache));
  return absl::OkStatus();
}

utils::StatusOr<Preprocessing> PreprocessTrainingDataset(
    const dataset::VerticalDataset& train_dataset,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config, const int num_threads) {
  const auto time_begin = absl::Now();
  Preprocessing preprocessing;
  preprocessing.set_num_examples(train_dataset.nrow());

  if (dt_config.internal().sorting_strategy() ==
          proto::DecisionTreeTrainingConfig::Internal::PRESORTED ||
      dt_config.internal().sorting_strategy() ==
          proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED) {
    RETURN_IF_ERROR(PresortNumericalFeatures(train_dataset, config_link,
                                             num_threads, &preprocessing));
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
    const int num_threads, Preprocessing* preprocessing) {
  // Check number of examples.
  RETURN_IF_ERROR(CheckNumExamples(train_dataset.nrow()));

  preprocessing->mutable_presorted_numerical_features()->resize(
      train_dataset.data_spec().columns_size());

  utils::concurrency::ThreadPool pool(
      "presort_numerical_features",
      std::min(num_threads, config_link.features().size()));
  pool.StartWorkers();

  // For all the input features in the model.
  for (const auto feature_idx : config_link.features()) {
    // Skip non numerical features.
    if (train_dataset.data_spec().columns(feature_idx).type() !=
        dataset::proto::NUMERICAL) {
      continue;
    }

    pool.Schedule([feature_idx, &train_dataset, preprocessing]() {
      const dataset::VerticalDataset::row_t num_examples = train_dataset.nrow();
      const auto& values =
          train_dataset
              .ColumnWithCast<dataset::VerticalDataset::NumericalColumn>(
                  feature_idx)
              ->values();
      CHECK_EQ(num_examples, values.size());

      // Global imputation replacement.
      const float na_replacement_value =
          train_dataset.data_spec().columns(feature_idx).numerical().mean();

      std::vector<std::pair<float, SparseItem::ExampleIdx>> items(
          values.size());
      for (dataset::VerticalDataset::row_t example_idx = 0;
           example_idx < num_examples; example_idx++) {
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

      for (dataset::VerticalDataset::row_t sorted_example_idx = 0;
           sorted_example_idx < num_examples; sorted_example_idx++) {
        SparseItem::ExampleIdx example_idx = items[sorted_example_idx].second;
        const bool change_value =
            sorted_example_idx > 0 && (items[sorted_example_idx].first !=
                                       items[sorted_example_idx - 1].first);
        if (change_value) {
          example_idx |= ((SparseItem::ExampleIdx)1)
                         << (sizeof(SparseItem::ExampleIdx) * 8 - 1);
        }
        sorted_values.items[sorted_example_idx].example_idx_and_extra =
            example_idx;
      }
    });
  }
  return absl::OkStatus();
}

namespace internal {

bool MaskPureSampledOrPrunedItemsForCategoricalSetGreedySelection(
    const proto::DecisionTreeTrainingConfig& dt_config,
    int32_t num_attribute_classes,
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<int64_t>&
        count_examples_without_weights_by_attribute_class,
    std::vector<bool>* candidate_attributes_bitmap,
    utils::RandomEngine* random) {
  std::uniform_real_distribution<float> sampling_dist;
  int64_t valid_items = 0;
  for (int attr_value = 0; attr_value < num_attribute_classes; attr_value++) {
    if (dt_config.categorical_set_greedy_forward().max_num_items() >= 0 &&
        attr_value >=
            dt_config.categorical_set_greedy_forward().max_num_items()) {
      // Too much candidate items.
      (*candidate_attributes_bitmap)[attr_value] = false;
    } else if (dt_config.categorical_set_greedy_forward().sampling() < 1.f &&
               sampling_dist(*random) >
                   dt_config.categorical_set_greedy_forward().sampling()) {
      // Randomly masked item.
      (*candidate_attributes_bitmap)[attr_value] = false;
    } else if (count_examples_without_weights_by_attribute_class[attr_value] <
                   dt_config.categorical_set_greedy_forward()
                       .min_item_frequency() ||
               count_examples_without_weights_by_attribute_class[attr_value] >
                   selected_examples.size() -
                       dt_config.categorical_set_greedy_forward()
                           .min_item_frequency()) {
      // Pure item.
      (*candidate_attributes_bitmap)[attr_value] = false;
    } else {
      valid_items++;
    }
  }
  return valid_items > 0;
}

std::vector<float> GenHistogramBins(const proto::NumericalSplit::Type type,
                                    const int num_splits,
                                    const std::vector<float>& attributes,
                                    const float min_value,
                                    const float max_value,
                                    utils::RandomEngine* random) {
  CHECK_GE(num_splits, 0);
  std::vector<float> candidate_splits(num_splits);
  switch (type) {
    case proto::NumericalSplit::HISTOGRAM_RANDOM: {
      std::uniform_real_distribution<float> threshold_distribution(min_value,
                                                                   max_value);
      for (auto& candidate_split : candidate_splits) {
        candidate_split = threshold_distribution(*random);
      }
    } break;
    case proto::NumericalSplit::HISTOGRAM_EQUAL_WIDTH: {
      for (int split_idx = 0; split_idx < candidate_splits.size();
           split_idx++) {
        candidate_splits[split_idx] = min_value + (max_value - min_value) *
                                                      (split_idx + 0.5f) /
                                                      candidate_splits.size();
      }
    } break;
    default:
      LOG(FATAL) << "Numerical histogram not implemented";
  }
  std::sort(candidate_splits.begin(), candidate_splits.end());
  return candidate_splits;
}

absl::Status SplitExamples(const dataset::VerticalDataset& dataset,
                           const std::vector<row_t>& examples,
                           const proto::NodeCondition& condition,
                           const bool dataset_is_dense,
                           const bool error_on_wrong_splitter_statistics,
                           std::vector<row_t>* positive_examples,
                           std::vector<row_t>* negative_examples,
                           const bool examples_are_training_examples) {
  if (examples_are_training_examples) {
    positive_examples->reserve(
        condition.num_pos_training_examples_without_weight());
    negative_examples->reserve(
        examples.size() - condition.num_pos_training_examples_without_weight());
  }

  positive_examples->clear();
  negative_examples->clear();

  std::vector<row_t>* example_sets[] = {negative_examples, positive_examples};

  // Index of the example selected for this node.
  const auto column_data = dataset.column(condition.attribute());

  if (!dataset_is_dense) {
    for (const row_t example_idx : examples) {
      const auto dst = example_sets[EvalConditionFromColumn(
          condition, column_data, dataset, example_idx)];
      dst->push_back(example_idx);
    }
  } else {
    row_t dense_example_idx = 0;
    for (const row_t example_idx : examples) {
      const auto dst = example_sets[EvalConditionFromColumn(
          condition, column_data, dataset, dense_example_idx)];
      dense_example_idx++;
      dst->push_back(example_idx);
    }
  }

  // The following test ensure that the effective number of positive examples is
  // equal to the expected number of positive examples. A miss alignment
  // generally indicates that the splitter does not work correctly.
  //
  // Incorrectly working splitters can makes the model worst than expected if
  // the error happen often. It such error happen rarely, the impact is likely
  // insignificant.
  //
  // This generates an error in unit testing and a warning otherwise.
  if (examples_are_training_examples &&
      ABSL_PREDICT_FALSE(
          (positive_examples->size() !=
           condition.num_pos_training_examples_without_weight()) ||
          (negative_examples->size() !=
           examples.size() -
               condition.num_pos_training_examples_without_weight()))) {
    const std::string message = absl::Substitute(
        "The number of positive/negative examples predicted by the splitter "
        "are different from the observations ($1!=$4) for the attribute "
        "\"$5\". This problem is generally caused by extreme floating point "
        "values (e.g. value>=10e30) and might prevent the model from training. "
        "Make sure to check the dataspec Details: eval:examples:$0 "
        "eval:positive_examples:$1 eval:negative_examples:$2 splitter:cond:$3",
        /*$0*/ examples.size(), /*$1*/ positive_examples->size(),
        /*$2*/ negative_examples->size(),
        /*$3*/ condition.DebugString(),
        /*$4*/ condition.num_pos_training_examples_without_weight(),
        /*$5*/ dataset.data_spec().columns(condition.attribute()).name());
    if (error_on_wrong_splitter_statistics) {
      return absl::InternalError(message);
    } else {
      LOG(WARNING) << message;
    }
  }
  return absl::OkStatus();
}

void UpliftLeafToLabelDist(const decision_tree::proto::NodeUpliftOutput& leaf,
                           UpliftLabelDistribution* dist) {
  dist->ImportSetFromLeafProto(leaf);
}

void UpliftLabelDistToLeaf(const UpliftLabelDistribution& dist,
                           decision_tree::proto::NodeUpliftOutput* leaf) {
  dist.ExportToLeafProto(leaf);
}

}  // namespace internal

}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
