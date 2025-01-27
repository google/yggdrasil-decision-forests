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

#include "yggdrasil_decision_forests/learner/decision_tree/vector_sequence.h"

#include <stddef.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/label.h"
#include "yggdrasil_decision_forests/learner/decision_tree/oblique.h"
#include "yggdrasil_decision_forests/learner/decision_tree/training.h"
#include "yggdrasil_decision_forests/learner/decision_tree/utils.h"
#include "yggdrasil_decision_forests/utils/cast.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests::model::decision_tree {
namespace {

// Evaluates, and if better than the current one, records a condition of the
// type "dist(example, anchor) <= threshold").
template <typename LabelType, typename Labels>
absl::Status TryCloserThanCondition(
    absl::Span<const float> anchors, const int num_anchors,
    const dataset::VerticalDataset::NumericalVectorSequenceColumn& attribute,
    const dataset::proto::Column& attribute_spec,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const LabelType& label_stats, const Labels& selected_labels,
    const std::vector<float>& selected_weights, const int32_t attribute_idx,
    std::vector<UnsignedExampleIdx> dense_example_idxs,
    const InternalTrainConfig& internal_config, SplitterPerThreadCache* cache,
    proto::NodeCondition* condition, SplitSearchResult* result_flag,
    std::vector<float>* projections) {
  STATUS_CHECK(internal_config.vector_sequence_computer);
  projections->resize(selected_examples.size() * num_anchors);
  RETURN_IF_ERROR(
      internal_config.vector_sequence_computer->ComputeNegMinSquareDistance(
          attribute_idx, selected_examples, anchors, num_anchors,
          absl::MakeSpan(*projections)));

  DCHECK_EQ(dense_example_idxs.size() * num_anchors, projections->size());
  DCHECK_EQ(dense_example_idxs.size(), selected_examples.size());

  for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
    auto anchor = anchors.subspan(anchor_idx * attribute.vector_length(),
                                  attribute.vector_length());

    auto local_projection = absl::MakeSpan(*projections)
                                .subspan(anchor_idx * selected_examples.size(),
                                         selected_examples.size());

    ASSIGN_OR_RETURN(
        const auto local_result_flag,
        EvaluateProjection(dt_config, label_stats, dense_example_idxs,
                           selected_weights, selected_labels, local_projection,
                           internal_config, attribute_idx, {}, 0, condition,
                           cache));

    if (*result_flag == SplitSearchResult::kInvalidAttribute &&
        local_result_flag == SplitSearchResult::kNoBetterSplitFound) {
      *result_flag = SplitSearchResult::kNoBetterSplitFound;
    } else if (local_result_flag == SplitSearchResult::kBetterSplitFound) {
      *result_flag = SplitSearchResult::kBetterSplitFound;

      const float threshold2 =
          condition->condition().higher_condition().threshold();
      auto* typed_condition = condition->mutable_condition()
                                  ->mutable_numerical_vector_sequence()
                                  ->mutable_closer_than();
      typed_condition->set_threshold2(-threshold2);
      STATUS_CHECK_GT(typed_condition->threshold2(), 0);
      typed_condition->mutable_anchor()->mutable_grounded()->Assign(
          anchor.begin(), anchor.end());
    }
  }
  return absl::OkStatus();
}

// Evaluates, and if better than the current one, records a condition of the
// type "dot(example, anchor) >= threshold").
template <typename LabelType, typename Labels>
absl::Status TryProjectedMoreThanCondition(
    absl::Span<const float> anchors, const int num_anchors,
    const dataset::VerticalDataset::NumericalVectorSequenceColumn& attribute,
    const dataset::proto::Column& attribute_spec,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const LabelType& label_stats, const Labels& selected_labels,
    const std::vector<float>& selected_weights, const int32_t attribute_idx,
    std::vector<UnsignedExampleIdx> dense_example_idxs,
    const InternalTrainConfig& internal_config, SplitterPerThreadCache* cache,
    proto::NodeCondition* condition, SplitSearchResult* result_flag,
    std::vector<float>* projections) {
  STATUS_CHECK(internal_config.vector_sequence_computer);
  projections->resize(selected_examples.size() * num_anchors);
  RETURN_IF_ERROR(
      internal_config.vector_sequence_computer->ComputeMaxDotProduct(
          attribute_idx, selected_examples, anchors, num_anchors,
          absl::MakeSpan(*projections)));

  DCHECK_EQ(dense_example_idxs.size() * num_anchors, projections->size());
  DCHECK_EQ(dense_example_idxs.size(), selected_examples.size());

  for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
    auto anchor = anchors.subspan(anchor_idx * attribute.vector_length(),
                                  attribute.vector_length());
    auto local_projection = absl::MakeSpan(*projections)
                                .subspan(anchor_idx * selected_examples.size(),
                                         selected_examples.size());
    ASSIGN_OR_RETURN(
        const auto local_result_flag,
        EvaluateProjection(dt_config, label_stats, dense_example_idxs,
                           selected_weights, selected_labels, local_projection,
                           internal_config, attribute_idx, {}, 0, condition,
                           cache));

    if (*result_flag == SplitSearchResult::kInvalidAttribute &&
        local_result_flag == SplitSearchResult::kNoBetterSplitFound) {
      *result_flag = SplitSearchResult::kNoBetterSplitFound;
    } else if (local_result_flag == SplitSearchResult::kBetterSplitFound) {
      *result_flag = SplitSearchResult::kBetterSplitFound;

      const float threshold =
          condition->condition().higher_condition().threshold();
      auto* typed_condition = condition->mutable_condition()
                                  ->mutable_numerical_vector_sequence()
                                  ->mutable_projected_more_than();
      typed_condition->set_threshold(threshold);
      typed_condition->mutable_anchor()->mutable_grounded()->Assign(
          anchor.begin(), anchor.end());
    }
  }

  return absl::OkStatus();
}

}  // namespace

template <typename LabelType>
absl::StatusOr<SplitSearchResult>
FindSplitAnyLabelTemplateFeatureNumericalVectorSequence(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const dataset::VerticalDataset::NumericalVectorSequenceColumn& attribute,
    const dataset::proto::Column& attribute_spec, const LabelType& label_stats,
    const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const int32_t attribute_idx, const InternalTrainConfig& internal_config,
    proto::NodeCondition* condition, utils::RandomEngine* random,
    SplitterPerThreadCache* cache) {
  STATUS_CHECK(!selected_examples.empty());

  const bool subsampling =
      selected_examples.size() >
      dt_config.numerical_vector_sequence().max_num_test_examples();

  absl::Span<const UnsignedExampleIdx> effective_selected_examples;
  proto::NodeCondition* effective_condition;

  // TODO: Cache buffers.
  std::vector<UnsignedExampleIdx> selected_examples_buffer;
  proto::NodeCondition condition_buffer;
  std::vector<float> effective_selected_weights;
  std::vector<float> projections;

  if (subsampling) {
    effective_condition = condition;
    // Sample a subset of the selected training examples.
    selected_examples_buffer.resize(
        dt_config.numerical_vector_sequence().max_num_test_examples());
    for (size_t i = 0; i < selected_examples_buffer.size(); i++) {
      selected_examples_buffer[i] = selected_examples[utils::RandomUniformInt(
          selected_examples.size(), random)];
    }
    effective_condition = &condition_buffer;
    effective_selected_examples = absl::MakeConstSpan(selected_examples_buffer);
  } else {
    // All the examples are used for the anchor search.
    effective_selected_examples = absl::MakeConstSpan(selected_examples);
    effective_condition = condition;
  }
  STATUS_CHECK(!effective_selected_examples.empty());

  const auto effective_selected_labels =
      ExtractLabels(label_stats, effective_selected_examples);
  if (!weights.empty()) {
    effective_selected_weights = Extract(weights, effective_selected_examples);
  }

  // TODO: Cache buffers.
  std::vector<UnsignedExampleIdx> dense_example_idxs(
      effective_selected_examples.size());
  std::iota(dense_example_idxs.begin(), dense_example_idxs.end(), 0);

  SplitSearchResult effective_result_flag =
      SplitSearchResult::kInvalidAttribute;

  const auto try_closer_that = [&](absl::Span<const float> anchors,
                                   const int num_anchors) {
    return TryCloserThanCondition(
        anchors, num_anchors, attribute, attribute_spec,
        effective_selected_examples, dt_config, label_stats,
        effective_selected_labels, effective_selected_weights, attribute_idx,
        dense_example_idxs, internal_config, cache, effective_condition,
        &effective_result_flag, &projections);
  };

  const auto try_projected_more_than = [&](absl::Span<const float> anchors,
                                           const int num_anchors) {
    return TryProjectedMoreThanCondition(
        anchors, num_anchors, attribute, attribute_spec,
        effective_selected_examples, dt_config, label_stats,
        effective_selected_labels, effective_selected_weights, attribute_idx,
        dense_example_idxs, internal_config, cache, effective_condition,
        &effective_result_flag, &projections);
  };

  const auto sample_vector_forced =
      [&]() -> absl::StatusOr<std::optional<absl::Span<const float>>> {
    constexpr int max_tries = 10000;
    for (int try_idx = 0; try_idx < max_tries; try_idx++) {
      const auto selected_example = selected_examples[utils::RandomUniformInt(
          selected_examples.size(), random)];
      if (attribute.IsNa(selected_example)) {
        // Example with missing vector sequence.
        continue;
      }
      const auto num_vectors = attribute.SequenceLength(selected_example);
      if (num_vectors == 0) {
        // Example without vector.
        continue;
      }
      const auto vector_idx = utils::RandomUniformInt(num_vectors, random);
      return attribute.GetVector(selected_example, vector_idx);
    }
    return std::nullopt;
  };

  std::vector<float> diff_buffer;

  // Static anchors generated from training examples.
  const auto num_anchors = std::min(
      dt_config.numerical_vector_sequence().num_random_selected_anchors(),
      static_cast<int>(10 * selected_examples.size()));

  const auto num_anchors_per_round =
      internal_config.vector_sequence_computer
          ->MaxNumAnchorsInRequest(selected_examples.size())
          .value_or(1);

  std::vector<float> anchors;
  for (int anchor_idx = 0; anchor_idx < num_anchors;
       anchor_idx += num_anchors_per_round) {
    const int local_num_anchors =
        std::min(num_anchors_per_round, num_anchors - anchor_idx);
    anchors.resize(local_num_anchors * attribute.vector_length());

    for (int i = 0; i < local_num_anchors; i++) {
      // The anchor is the difference between two random values in the dataset.
      ASSIGN_OR_RETURN(const auto vector_1, sample_vector_forced());
      if (!vector_1.has_value()) {
        break;
      }
      ASSIGN_OR_RETURN(const auto vector_2, sample_vector_forced());
      if (!vector_2.has_value()) {
        break;
      }
      for (size_t j = 0; j < vector_1.value().size(); j++) {
        anchors[i * attribute.vector_length() + j] =
            vector_1.value()[j] - vector_2.value()[j];
      }
    }
    RETURN_IF_ERROR(try_projected_more_than(anchors, local_num_anchors));

    for (int i = 0; i < local_num_anchors; i++) {
      // The anchor is a random value in the dataset.
      ASSIGN_OR_RETURN(const auto vector, sample_vector_forced());
      if (!vector.has_value()) {
        break;
      }
      // TODO: Better copy.
      for (size_t j = 0; j < vector.value().size(); j++) {
        anchors[i * attribute.vector_length() + j] = vector.value()[j];
      }
    }
    RETURN_IF_ERROR(try_closer_that(anchors, local_num_anchors));
  }

  if (effective_result_flag == SplitSearchResult::kBetterSplitFound) {
    if (subsampling) {
      // TODO: Cache buffers.
      const auto selected_labels =
          ExtractLabels(label_stats, selected_examples);
      std::vector<float> selected_weights;
      if (!weights.empty()) {
        selected_weights = Extract(weights, selected_examples);
      }
      // TODO: Global iota + use spans
      std::vector<UnsignedExampleIdx> dense_example_idxs(
          selected_examples.size());
      std::iota(dense_example_idxs.begin(), dense_example_idxs.end(), 0);

      SplitSearchResult final_result_flag =
          SplitSearchResult::kInvalidAttribute;

      switch (effective_condition->condition()
                  .numerical_vector_sequence()
                  .type_case()) {
        case proto::Condition::NumericalVectorSequence::kCloserThan: {
          const absl::Span<const float> anchor =
              effective_condition->condition()
                  .numerical_vector_sequence()
                  .closer_than()
                  .anchor()
                  .grounded();

          RETURN_IF_ERROR(TryCloserThanCondition(
              anchor, 1, attribute, attribute_spec, selected_examples,
              dt_config, label_stats, selected_labels, selected_weights,
              attribute_idx, dense_example_idxs, internal_config, cache,
              condition, &final_result_flag, &projections));
        } break;

        case proto::Condition::NumericalVectorSequence::kProjectedMoreThan: {
          const absl::Span<const float> anchor =
              effective_condition->condition()
                  .numerical_vector_sequence()
                  .projected_more_than()
                  .anchor()
                  .grounded();

          RETURN_IF_ERROR(TryProjectedMoreThanCondition(
              anchor, 1, attribute, attribute_spec, selected_examples,
              dt_config, label_stats, selected_labels, selected_weights,
              attribute_idx, dense_example_idxs, internal_config, cache,
              condition, &final_result_flag, &projections));
        } break;

        default:
          return absl::InternalError(
              "Invalid condition type for numerical vector sequence");
      }
      return final_result_flag;
    }
  }
  return effective_result_flag;
}

absl::StatusOr<SplitSearchResult>
FindSplitAnyLabelFeatureNumericalVectorSequence(
    model::proto::Task task,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const dataset::VerticalDataset::NumericalVectorSequenceColumn& attribute,
    const dataset::proto::Column& attribute_spec, const LabelStats& label_stats,
    const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const int32_t attribute_idx, const InternalTrainConfig& internal_config,
    proto::NodeCondition* condition, utils::RandomEngine* random,
    SplitterPerThreadCache* cache) {
  // Dispatch.
  switch (task) {
    case model::proto::Task::CLASSIFICATION: {
      const auto& class_label_stats =
          utils::down_cast<const ClassificationLabelStats&>(label_stats);
      return FindSplitAnyLabelTemplateFeatureNumericalVectorSequence(
          selected_examples, weights, attribute, attribute_spec,
          class_label_stats, min_num_obs, dt_config, attribute_idx,
          internal_config, condition, random, cache);
    } break;
    case model::proto::Task::REGRESSION:
      if (internal_config.hessian_score) {
        const auto& reg_label_stats =
            utils::down_cast<const RegressionHessianLabelStats&>(label_stats);
        return FindSplitAnyLabelTemplateFeatureNumericalVectorSequence(
            selected_examples, weights, attribute, attribute_spec,
            reg_label_stats, min_num_obs, dt_config, attribute_idx,
            internal_config, condition, random, cache);
      } else {
        const auto& reg_label_stats =
            utils::down_cast<const RegressionLabelStats&>(label_stats);
        return FindSplitAnyLabelTemplateFeatureNumericalVectorSequence(
            selected_examples, weights, attribute, attribute_spec,
            reg_label_stats, min_num_obs, dt_config, attribute_idx,
            internal_config, condition, random, cache);
      }
      break;
    default:
      return absl::UnimplementedError(
          "Numerical sequence vector split not implemented for this task");
  }
}

}  // namespace yggdrasil_decision_forests::model::decision_tree
