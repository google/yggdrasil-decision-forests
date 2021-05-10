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

#include "yggdrasil_decision_forests/learner/decision_tree/sparse_oblique.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/training.h"
#include "yggdrasil_decision_forests/learner/decision_tree/utils.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/logging.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {

namespace {

// A projection is defined as \sum features[projection[i].index] *
// projection[i].weight;
struct AttributeAndWeight {
  int attribute_idx;
  float weight;
};
using Projection = std::vector<AttributeAndWeight>;
using std::is_same;

// Extracts values using an index i.e. returns "values[selected]".
template <typename T>
std::vector<T> Extract(const std::vector<T>& values,
                       const std::vector<row_t>& selected) {
  std::vector<T> extracted(selected.size());
  for (row_t selected_idx = 0; selected_idx < selected.size(); selected_idx++) {
    extracted[selected_idx] = values[selected[selected_idx]];
  }
  return extracted;
}

// Extraction of label values. Different implementations for different types of
// labels.
std::vector<int32_t> ExtractLabels(const ClassificationLabelStats& labels,
                                   const std::vector<row_t>& selected) {
  return Extract(labels.label_data, selected);
}

std::vector<float> ExtractLabels(const RegressionLabelStats& labels,
                                 const std::vector<row_t>& selected) {
  return Extract(labels.label_data, selected);
}

struct GradientAndHessian {
  const std::vector<float> gradient_data;
  const std::vector<float> hessian_data;
};

GradientAndHessian ExtractLabels(const RegressionHessianLabelStats& labels,
                                 const std::vector<row_t>& selected) {
  return {/*.gradient_data =*/Extract(labels.gradient_data, selected),
          /*.hessian_data =*/Extract(labels.hessian_data, selected)};
}

// Randomly generates a projection. A projection cannot be empty. If the
// projection contains only one dimension, the weight is guaranteed to be 1.
void SampleProjection(const proto::DecisionTreeTrainingConfig& dt_config,
                      const dataset::proto::DataSpecification& data_spec,
                      const std::vector<int>& numerical_features,
                      const float projection_density, Projection* projection,
                      utils::RandomEngine* random) {
  projection->clear();
  std::uniform_real_distribution<float> unif01;
  std::uniform_real_distribution<float> unif1m1(-1.f, 1.f);

  const auto gen_weight = [&](const int feature) -> float {
    float weight = unif1m1(*random);
    if (dt_config.sparse_oblique_split().binary_weight()) {
      weight = (weight >= 0) ? 1.f : -1.f;
    }
    const auto& spec = data_spec.columns(feature).numerical();
    switch (dt_config.sparse_oblique_split().normalization()) {
      case proto::DecisionTreeTrainingConfig::SparseObliqueSplit::NONE:
        return weight;
      case proto::DecisionTreeTrainingConfig::SparseObliqueSplit::
          STANDARD_DEVIATION:
        return weight / std::max(1e-6, spec.standard_deviation());
      case proto::DecisionTreeTrainingConfig::SparseObliqueSplit::MIN_MAX:
        return weight / std::max(1e-6f, spec.max_value() - spec.min_value());
    }
  };

  for (const auto feature : numerical_features) {
    if (unif01(*random) < projection_density) {
      projection->push_back({feature, gen_weight(feature)});
    }
  }
  if (projection->empty()) {
    std::uniform_int_distribution<int> unif_feature_idx(
        0, numerical_features.size() - 1);
    projection->push_back(
        {/*.attribute_idx =*/numerical_features[unif_feature_idx(*random)],
         /*.weight =*/1.f});
  } else if (projection->size() == 1) {
    projection->front().weight = 1.f;
  }
}

// Get the list of numerical features.
void GetNumericalFeatures(
    const dataset::VerticalDataset& train_dataset,
    const model::proto::TrainingConfigLinking& config_link,
    std::vector<int>* numerical_features) {
  numerical_features->clear();
  for (const auto feature_idx : config_link.features()) {
    if (train_dataset.column(feature_idx)->type() ==
        dataset::proto::NUMERICAL) {
      numerical_features->push_back(feature_idx);
    }
  }
}

// Converts a Projection object + float threshold into a proto condition of the
// same semantic.
absl::Status SetCondition(const Projection& projection, const float threshold,
                          const float na_replacement,
                          proto::NodeCondition* condition) {
  if (projection.empty()) {
    return absl::InternalError("Empty projection");
  }
  auto& oblique_condition =
      *condition->mutable_condition()->mutable_oblique_condition();
  oblique_condition.set_threshold(threshold);
  oblique_condition.clear_attributes();
  oblique_condition.clear_weights();
  for (const auto& item : projection) {
    oblique_condition.add_attributes(item.attribute_idx);
    oblique_condition.add_weights(item.weight);
  }
  condition->set_attribute(projection.front().attribute_idx);
  condition->set_na_value(na_replacement >= threshold);
  return absl::OkStatus();
}

// Helper for the evaluation of projections.
class ProjectionEvaluator {
 public:
  ProjectionEvaluator(const dataset::VerticalDataset& train_dataset,
                      const std::vector<int>& numerical_features) {
    DCHECK(!numerical_features.empty());
    const int max_feature_idx =
        *std::max_element(numerical_features.begin(), numerical_features.end());
    numerical_attributes_.assign(max_feature_idx + 1, nullptr);
    for (const auto attribute_idx : numerical_features) {
      const std::vector<float>* values =
          &train_dataset
               .ColumnWithCast<dataset::VerticalDataset::NumericalColumn>(
                   attribute_idx)
               ->values();
      numerical_attributes_[attribute_idx] = values;
    }
  }

  void Evaluate(const Projection& projection,
                const std::vector<row_t>& selected_examples,
                std::vector<float>* values) {
    values->resize(selected_examples.size());
    for (row_t selected_idx = 0; selected_idx < selected_examples.size();
         selected_idx++) {
      float value = 0;
      const auto example_idx = selected_examples[selected_idx];
      for (const auto& item : projection) {
        DCHECK_LT(item.attribute_idx, numerical_attributes_.size());
        DCHECK_GE(item.attribute_idx, 0);
        const auto* attribute_values =
            numerical_attributes_[item.attribute_idx];
        DCHECK(attribute_values != nullptr);
        value += (*attribute_values)[example_idx] * item.weight;
      }
      (*values)[selected_idx] = value;
    }
  }

 private:
  // Non-owning pointer to numerical attributes.
  std::vector<const std::vector<float>*> numerical_attributes_;
};

// Computes the number of projections to test i.e.
// num_projections = min(max_num_projections,
// ceil(num_features ^ num_projections_exponent)).
int GetNumProjections(const proto::DecisionTreeTrainingConfig& dt_config,
                      const int num_numerical_features) {
  const auto max_num_projections =
      dt_config.sparse_oblique_split().max_num_projections();
  const auto num_projections_exponent =
      dt_config.sparse_oblique_split().num_projections_exponent();
  return std::min(
      max_num_projections,
      static_cast<int>(0.5 + std::ceil(std::pow(num_numerical_features,
                                                num_projections_exponent))));
}

// Replacement value of the projection when one of the input feature is missing.
float DefaultProjectionValue(
    const Projection& projection,
    const dataset::proto::DataSpecification& dataspec) {
  float value = 0;
  for (const auto& item : projection) {
    const float attribute_value =
        dataspec.columns(item.attribute_idx).numerical().mean();
    value += attribute_value * item.weight;
  }
  return value;
}

}  // namespace

template <typename LabelStats>
utils::StatusOr<bool> FindBestConditionSparseObliqueTemplate(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<row_t>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const LabelStats& label_stats, proto::NodeCondition* best_condition,
    utils::RandomEngine* random, SplitterPerThreadCache* cache) {
  auto& numerical_features = cache->numerical_features;
  GetNumericalFeatures(train_dataset, config_link, &numerical_features);
  if (numerical_features.empty()) {
    return false;
  }

  // Effective number of projections to test.
  const int num_projections =
      GetNumProjections(dt_config, numerical_features.size());

  const float projection_density =
      dt_config.sparse_oblique_split().projection_density_factor() /
      numerical_features.size();
  const int min_num_obs =
      dt_config.in_split_min_examples_check() ? dt_config.min_examples() : 1;

  // Best and current projections.
  Projection best_projection;
  float best_projection_threshold = 0;
  Projection current_projection;
  float best_na_replacement = 0;
  auto& projection_values = cache->projection_values;

  ProjectionEvaluator projection_evaluator(train_dataset, numerical_features);

  const auto selected_labels = ExtractLabels(label_stats, selected_examples);
  const auto selected_weights = Extract(weights, selected_examples);

  std::vector<row_t> dense_example_idxs(selected_examples.size());
  std::iota(dense_example_idxs.begin(), dense_example_idxs.end(), 0);

  for (int projection_idx = 0; projection_idx < num_projections;
       projection_idx++) {
    // Generate a current_projection.
    SampleProjection(dt_config, train_dataset.data_spec(), numerical_features,
                     projection_density, &current_projection, random);

    // Pre-compute the result of the current_projection.
    projection_evaluator.Evaluate(current_projection, selected_examples,
                                  &projection_values);

    const auto na_replacement =
        DefaultProjectionValue(current_projection, train_dataset.data_spec());

    // Find a good split in the current_projection.
    SplitSearchResult result;
    if constexpr (is_same<LabelStats, ClassificationLabelStats>::value) {
      result = FindSplitLabelClassificationFeatureNumericalCart(
          dense_example_idxs, selected_weights, projection_values,
          selected_labels, label_stats.num_label_classes, na_replacement,
          min_num_obs, dt_config, label_stats.label_distribution,
          current_projection.front().attribute_idx, {}, best_condition, cache);
    } else if constexpr (is_same<LabelStats,
                                 RegressionHessianLabelStats>::value) {
      result = FindSplitLabelHessianRegressionFeatureNumericalCart(
          dense_example_idxs, selected_weights, projection_values,
          selected_labels.gradient_data, selected_labels.hessian_data,
          na_replacement, min_num_obs, dt_config, label_stats.sum_gradient,
          label_stats.sum_hessian, label_stats.sum_weights,
          current_projection.front().attribute_idx, internal_config,
          best_condition, cache);
    } else if constexpr (is_same<LabelStats, RegressionLabelStats>::value) {
      result = FindSplitLabelRegressionFeatureNumericalCart(
          dense_example_idxs, selected_weights, projection_values,
          selected_labels, na_replacement, min_num_obs, dt_config,
          label_stats.label_distribution,
          current_projection.front().attribute_idx, {}, best_condition, cache);
    } else {
      static_assert(!is_same<LabelStats, LabelStats>::value,
                    "Not implemented.");
    }

    if (result == SplitSearchResult::kBetterSplitFound) {
      best_projection = current_projection;
      best_projection_threshold =
          best_condition->condition().higher_condition().threshold();
      best_na_replacement = na_replacement;
    }
  }

  // Update with the actual current_projection definition.
  if (!best_projection.empty()) {
    RETURN_IF_ERROR(SetCondition(best_projection, best_projection_threshold,
                                 best_na_replacement, best_condition));
    return true;
  }

  return false;
}

utils::StatusOr<bool> FindBestConditionSparseOblique(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<row_t>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const ClassificationLabelStats& label_stats,
    proto::NodeCondition* best_condition, utils::RandomEngine* random,
    SplitterPerThreadCache* cache) {
  return FindBestConditionSparseObliqueTemplate<ClassificationLabelStats>(
      train_dataset, selected_examples, weights, config, config_link, dt_config,
      parent, internal_config, label_stats, best_condition, random, cache);
}

utils::StatusOr<bool> FindBestConditionSparseOblique(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<row_t>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const RegressionHessianLabelStats& label_stats,
    proto::NodeCondition* best_condition, utils::RandomEngine* random,
    SplitterPerThreadCache* cache) {
  return FindBestConditionSparseObliqueTemplate<RegressionHessianLabelStats>(
      train_dataset, selected_examples, weights, config, config_link, dt_config,
      parent, internal_config, label_stats, best_condition, random, cache);
}

utils::StatusOr<bool> FindBestConditionSparseOblique(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<row_t>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const RegressionLabelStats& label_stats,
    proto::NodeCondition* best_condition, utils::RandomEngine* random,
    SplitterPerThreadCache* cache) {
  return FindBestConditionSparseObliqueTemplate<RegressionLabelStats>(
      train_dataset, selected_examples, weights, config, config_link, dt_config,
      parent, internal_config, label_stats, best_condition, random, cache);
}

}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
