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

#include "yggdrasil_decision_forests/learner/decision_tree/label.h"

#include <cstdint>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/decision_tree/uplift.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/utils/distribution.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests::model::decision_tree {
namespace {
// Set the label value for a classification label on a vertical dataset.
template <bool weighted>
absl::Status SetClassificationLabelDistribution(
    const dataset::VerticalDataset& dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfigLinking& config_link, proto::Node* node) {
  if constexpr (weighted) {
    DCHECK_LE(selected_examples.size(), weights.size());
  } else {
    DCHECK(weights.empty());
  }
  ASSIGN_OR_RETURN(
      const auto* const labels,
      dataset.ColumnWithCastWithStatus<
          dataset::VerticalDataset::CategoricalColumn>(config_link.label()));
  utils::IntegerDistributionDouble label_distribution;
  const int32_t num_classes = dataset.data_spec()
                                  .columns(config_link.label())
                                  .categorical()
                                  .number_of_unique_values();
  label_distribution.SetNumClasses(num_classes);
  for (const UnsignedExampleIdx example_idx : selected_examples) {
    if constexpr (weighted) {
      label_distribution.Add(labels->values()[example_idx],
                             weights[example_idx]);
    } else {
      label_distribution.Add(labels->values()[example_idx]);
    }
  }
  label_distribution.Save(node->mutable_classifier()->mutable_distribution());
  node->mutable_classifier()->set_top_value(label_distribution.TopClass());
  return absl::OkStatus();
}

absl::Status SetCategoricalUpliftLabelDistribution(
    const dataset::VerticalDataset& dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfigLinking& config_link, proto::Node* node) {
  DCHECK(!weights.empty());
  // TODO: Update.
  ASSIGN_OR_RETURN(
      const auto* const outcomes,
      dataset.ColumnWithCastWithStatus<
          dataset::VerticalDataset::CategoricalColumn>(config_link.label()));

  ASSIGN_OR_RETURN(const auto* const treatments,
                   dataset.ColumnWithCastWithStatus<
                       dataset::VerticalDataset::CategoricalColumn>(
                       config_link.uplift_treatment()));

  const auto& outcome_spec = dataset.data_spec().columns(config_link.label());
  const auto& treatment_spec =
      dataset.data_spec().columns(config_link.uplift_treatment());

  UpliftLabelDistribution label_dist;
  label_dist.InitializeAndClearCategoricalOutcome(
      outcome_spec.categorical().number_of_unique_values(),
      treatment_spec.categorical().number_of_unique_values());

  for (const UnsignedExampleIdx example_idx : selected_examples) {
    label_dist.AddCategoricalOutcome(outcomes->values()[example_idx],
                                     treatments->values()[example_idx],
                                     weights[example_idx]);
  }
  UpliftLabelDistToLeaf(label_dist, node->mutable_uplift());
  return absl::OkStatus();
}

absl::Status SetRegressiveUpliftLabelDistribution(
    const dataset::VerticalDataset& dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
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

  for (const UnsignedExampleIdx example_idx : selected_examples) {
    label_dist.AddNumericalOutcome(outcomes->values()[example_idx],
                                   treatments->values()[example_idx],
                                   weights[example_idx]);
  }
  UpliftLabelDistToLeaf(label_dist, node->mutable_uplift());
  return absl::OkStatus();
}

// Set the label value for a regression label on a vertical dataset.
//
// Default policy to set the label value of a leaf in a regression tree i.e. set
// the value to the mean of the labels.
// `weights` may be empty, corresponding to unit weights.
template <bool weighted>
absl::Status SetRegressionLabelDistribution(
    const dataset::VerticalDataset& dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfigLinking& config_link, proto::Node* node) {
  if constexpr (weighted) {
    STATUS_CHECK(weights.size() == dataset.nrow());
  } else {
    STATUS_CHECK(weights.empty());
  }
  ASSIGN_OR_RETURN(
      const auto* const labels,
      dataset
          .ColumnWithCastWithStatus<dataset::VerticalDataset::NumericalColumn>(
              config_link.label()));
  utils::NormalDistributionDouble label_distribution;
  if constexpr (weighted) {
    for (const UnsignedExampleIdx example_idx : selected_examples) {
      label_distribution.Add(labels->values()[example_idx],
                             weights[example_idx]);
    }
  } else {
    for (const UnsignedExampleIdx example_idx : selected_examples) {
      label_distribution.Add(labels->values()[example_idx]);
    }
  }
  label_distribution.Save(node->mutable_regressor()->mutable_distribution());
  node->mutable_regressor()->set_top_value(label_distribution.Mean());
  return absl::OkStatus();
}

}  // namespace

absl::Status SetLabelDistribution(
    const dataset::VerticalDataset& train_dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    NodeWithChildren* node) {
  switch (config.task()) {
    case model::proto::Task::CLASSIFICATION:
      if (weights.empty()) {
        RETURN_IF_ERROR(SetClassificationLabelDistribution</*weighted=*/false>(
            train_dataset, selected_examples, weights, config_link,
            node->mutable_node()));
      } else {
        RETURN_IF_ERROR(SetClassificationLabelDistribution</*weighted=*/true>(
            train_dataset, selected_examples, weights, config_link,
            node->mutable_node()));
      }
      break;
    case model::proto::Task::REGRESSION:
      if (weights.empty()) {
        RETURN_IF_ERROR(SetRegressionLabelDistribution</*weighted=*/false>(
            train_dataset, selected_examples, weights, config_link,
            node->mutable_node()));
        break;
      } else {
        RETURN_IF_ERROR(SetRegressionLabelDistribution</*weighted=*/true>(
            train_dataset, selected_examples, weights, config_link,
            node->mutable_node()));
        break;
      }

    case model::proto::Task::CATEGORICAL_UPLIFT:
      RETURN_IF_ERROR(SetCategoricalUpliftLabelDistribution(
          train_dataset, selected_examples, weights, config_link,
          node->mutable_node()));
      break;

    case model::proto::Task::NUMERICAL_UPLIFT:
      RETURN_IF_ERROR(SetRegressiveUpliftLabelDistribution(
          train_dataset, selected_examples, weights, config_link,
          node->mutable_node()));
      break;

    default:
      NOTREACHED();
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

}  // namespace yggdrasil_decision_forests::model::decision_tree
