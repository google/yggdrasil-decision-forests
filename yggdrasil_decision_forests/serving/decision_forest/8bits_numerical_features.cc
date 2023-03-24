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

#include "yggdrasil_decision_forests/serving/decision_forest/8bits_numerical_features.h"

#include <limits>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"

namespace yggdrasil_decision_forests {
namespace serving {
namespace decision_forest {
namespace num_8bits {
namespace {

// Maximum stack size used by the model during inference
constexpr size_t kMaxStackUsageInBytes = 16 * 1024;

using model::decision_tree::NodeWithChildren;
using model::decision_tree::proto::Condition;
using model::gradient_boosted_trees::proto::Loss;

// Activation function for the log likelihood loss.
float ActivationBinomialLogLikelihood(const float value) {
  return utils::clamp(1.f / (1.f + std::exp(-value)), 0.f, 1.f);
}

// Identity activation function.
float ActivationIdentity(const float value) { return value; }

template <float (*Activation)(float)>
absl::Status RawPredict(const RawModel& model,
                        const std::vector<uint8_t>& examples,
                        uint32_t num_examples,
                        std::vector<float>* predictions) {
  predictions->resize(num_examples);

  // Allocate active leaf buffer.
  const size_t active_leaf_buffer_size = model.num_trees * sizeof(LeafMask);
  LeafMask* active_leaf_buffer;
  const bool active_leaf_buffer_uses_stack =
      active_leaf_buffer_size <= kMaxStackUsageInBytes;
  if (active_leaf_buffer_uses_stack) {
    active_leaf_buffer =
        reinterpret_cast<LeafMask*>(alloca(active_leaf_buffer_size));
  } else {
    active_leaf_buffer =
        reinterpret_cast<LeafMask*>(std::malloc(active_leaf_buffer_size));
  }

  const int num_features = model.num_features;
  const int num_trees = model.num_trees;

  const LeafMask* __restrict masks_v2 = model.masks_v2.data();
  const uint32_t* __restrict feature_value_to_mask_list =
      model.feature_value_to_mask_list.data();
  const uint32_t* __restrict feature_to_feature_value =
      model.feature_to_feature_value.data();

  // Run inference
  const uint8_t* example_reader = examples.data();
  for (int example_idx = 0; example_idx < num_examples; ++example_idx) {
    // Zero active node buffer.
    std::memset(active_leaf_buffer, 0xFF, active_leaf_buffer_size);

    // Apply the conditions
    for (int feature_idx = 0; feature_idx < num_features; feature_idx++) {
      const auto feature_value = example_reader[feature_idx];
      DCHECK_LT(feature_value, model.num_buckets[feature_idx]);

      const auto mask_idx =
          feature_value_to_mask_list[feature_to_feature_value[feature_idx] +
                                     feature_value];
      const LeafMask* __restrict leaf_mask_reader = &masks_v2[mask_idx];
      for (int tree_idx = 0; tree_idx < num_trees; tree_idx++) {
        active_leaf_buffer[tree_idx] &= leaf_mask_reader[tree_idx];
      }
    }

    // Get the active leaf.
    float output = model.initial_prediction;
    for (int tree_idx = 0; tree_idx < model.num_trees; tree_idx++) {
      const auto node_idx = absl::countr_zero(active_leaf_buffer[tree_idx]);
      output += model.leaves[model.leaves_tree_index[tree_idx] + node_idx];
    }

    (*predictions)[example_idx] = Activation(output);
    example_reader += model.num_features;
  }

  if (!active_leaf_buffer_uses_stack) {
    // Free memory
    free(active_leaf_buffer);
  }

  return absl::OkStatus();
}

absl::Status Initialize(
    const model::gradient_boosted_trees::GradientBoostedTreesModel& src,
    RawModel* dst, BuildingWorkMemory* working) {
  working->sum_num_buckets = 0;

  dst->num_buckets.assign(dst->num_features, 0);

  for (int local_feature_idx = 0;
       local_feature_idx < src.input_features().size(); local_feature_idx++) {
    const int feature_idx = src.input_features()[local_feature_idx];
    const auto& column_spec = src.data_spec().columns(feature_idx);

    if (column_spec.type() !=
        dataset::proto::ColumnType::DISCRETIZED_NUMERICAL) {
      return absl::InvalidArgumentError(absl::Substitute(
          "Feature \"$0\" hs not DISCRETIZED_NUMERICAL.", column_spec.name()));
    }

    const auto num_boundaries =
        column_spec.discretized_numerical().boundaries_size();
    const auto num_buckets = num_boundaries + 1;

    if (num_boundaries + 1 > 256) {
      return absl::InvalidArgumentError(absl::Substitute(
          "The number of buckets of feature \"$0\" is greater than 256.",
          column_spec.name()));
    }

    if (num_boundaries == 0) {
      return absl::InvalidArgumentError(absl::Substitute(
          "Feature \"$0\" has only one bucket.", column_spec.name()));
    }

    for (int boundary_idx = 0; boundary_idx < num_boundaries; boundary_idx++) {
      const float expected_boundary = static_cast<float>(boundary_idx) + 0.5f;
      const float boundary =
          column_spec.discretized_numerical().boundaries()[boundary_idx];
      if (std::abs(expected_boundary - boundary) >= 0.001) {
        return absl::InvalidArgumentError(absl::Substitute(
            "The boundaries of feature \"$0\" are not [0.5, 1.5, 2.5, ...].",
            column_spec.name()));
      }
    }

    working->features[feature_idx] = {
        /*name=*/column_spec.name(),
        /*local_idx=*/static_cast<uint32_t>(local_feature_idx),
    };

    working->features[feature_idx].value_to_masks.resize(num_buckets);
    dst->num_buckets[local_feature_idx] = num_buckets;
    working->sum_num_buckets += num_buckets;
  }
  return absl::OkStatus();
}

absl::Status FillMaskNode(
    const model::gradient_boosted_trees::GradientBoostedTreesModel& src,
    const uint32_t tree_idx, const NodeWithChildren& src_node, RawModel* dst,
    BuildingWorkMemory* working, int* leaf_idx, int* non_leaf_idx) {
  if (src_node.IsLeaf()) {
    // Store the leaf value.
    if (*leaf_idx >= kMaxLeafs) {
      return absl::InternalError("Too many leaves");
    }

    const auto leaf_value_idx = *leaf_idx + dst->leaves_tree_index[tree_idx];
    dst->leaves[leaf_value_idx] = src_node.node().regressor().top_value();
    (*leaf_idx)++;
  } else {
    // Index of the first leaf in the negative branch.
    const auto begin_neg_leaf_idx = *leaf_idx;

    // Parse the negative branch.
    RETURN_IF_ERROR(FillMaskNode(src, tree_idx, *src_node.neg_child(), dst,
                                 working, leaf_idx, non_leaf_idx));

    // Index of the feature used by the node.
    const int feature_idx = src_node.node().condition().attribute();

    // Compute the bitmap mask i.e. the bitmap that hide the leafs of the
    // negative branch.
    //
    // Example:
    // If begin_neg_leaf_idx=2 and end_neg_leaf_idx = 5, the mask will be:
    //   "1100011111" + 54 * "1" (lower bit on the left).
    const LeafMask end_neg_leaf_idx = *leaf_idx;
    const LeafMask start_leaf_mask = (kOneLeafMask << begin_neg_leaf_idx) - 1;
    const LeafMask after_neg_mask = (kOneLeafMask << end_neg_leaf_idx) - 1;
    const LeafMask mask = ~(after_neg_mask ^ start_leaf_mask);

    const auto& condition = src_node.node().condition().condition();

    if (condition.type_case() !=
        Condition::TypeCase::kDiscretizedHigherCondition) {
      return absl::InvalidArgumentError("Non supported condition");
    }

    const auto it_feature = working->features.find(feature_idx);
    if (it_feature == working->features.end()) {
      return absl::InvalidArgumentError("Feature not found");
    }

    // Process the node's condition.
    it_feature->second
        .value_to_masks[condition.discretized_higher_condition().threshold()]
        .masks.push_back({
            /*tree_idx=*/(int)tree_idx,
            /*mask=*/mask,
        });

    ++(*non_leaf_idx);

    RETURN_IF_ERROR(FillMaskNode(src, tree_idx, *src_node.pos_child(), dst,
                                 working, leaf_idx, non_leaf_idx));
  }

  return absl::OkStatus();
}

absl::Status FillMask(
    const model::gradient_boosted_trees::GradientBoostedTreesModel& src,
    RawModel* dst, BuildingWorkMemory* working) {
  uint32_t sum_num_leaves = 0;
  for (uint32_t tree_idx = 0; tree_idx < dst->num_trees; tree_idx++) {
    const auto& src_tree = src.decision_trees()[tree_idx];
    const auto num_leaves = src_tree->NumLeafs();
    dst->leaves_tree_index[tree_idx] = sum_num_leaves;
    sum_num_leaves += num_leaves;
  }
  dst->leaves.assign(sum_num_leaves, std::numeric_limits<float>::quiet_NaN());

  for (uint32_t tree_idx = 0; tree_idx < dst->num_trees; tree_idx++) {
    const auto& src_tree = src.decision_trees()[tree_idx];
    int leaf_idx = 0;
    int non_leaf_idx = 0;
    RETURN_IF_ERROR(FillMaskNode(src, tree_idx, src_tree->root(), dst, working,
                                 &leaf_idx, &non_leaf_idx));
  }
  return absl::OkStatus();
}

absl::Status FinalizeMask(
    const model::gradient_boosted_trees::GradientBoostedTreesModel& src,
    RawModel* dst, BuildingWorkMemory* working) {
  // Unlike for QS, the inference don't apply all the valid masks. Instead, it
  // only applies the one corresponding to the most specific condition.
  // Therefore, We need to "spread" a mask to all the other most specific
  // masks.

  // The first value is the full mask.
  const uint32_t full_mask_idx = 0;
  std::vector<LeafMask> full_mask(dst->num_trees, kFullLeafMask);
  dst->masks_v2.insert(dst->masks_v2.end(), full_mask.begin(), full_mask.end());

  for (const auto& feature : working->features) {
    dst->feature_to_feature_value.push_back(
        dst->feature_value_to_mask_list.size());

    // Current active mask.
    std::vector<LeafMask> active_mask = full_mask;

    // Currently indexed mask.
    uint32_t indexed_mask_idx = full_mask_idx;
    std::vector<LeafMask> indexed_mask = full_mask;

    for (int bucket_idx = 0; bucket_idx < feature.second.value_to_masks.size();
         bucket_idx++) {
      // Update the active mask.
      for (const auto& mask : feature.second.value_to_masks[bucket_idx].masks) {
        active_mask[mask.tree_idx] &= mask.mask;
      }

      if (indexed_mask != active_mask) {
        // Index a new mask.
        indexed_mask_idx = dst->masks_v2.size();
        dst->masks_v2.insert(dst->masks_v2.end(), active_mask.begin(),
                             active_mask.end());
        indexed_mask = active_mask;
      }
      dst->feature_value_to_mask_list.push_back(indexed_mask_idx);
    }
  }

  return absl::OkStatus();
}

absl::Status RawGenericToSpecializedModel(
    const model::gradient_boosted_trees::GradientBoostedTreesModel& src,
    RawModel* dst) {
  dst->num_trees = src.num_trees();
  dst->features = src.input_features();
  dst->num_features = dst->features.size();

  // Build the feature mapping.
  BuildingWorkMemory working;
  RETURN_IF_ERROR(Initialize(src, dst, &working));

  // Compile the model
  dst->leaves_tree_index.assign(dst->num_trees, 0);
  RETURN_IF_ERROR(FillMask(src, dst, &working));
  RETURN_IF_ERROR(FinalizeMask(src, dst, &working));

  return absl::OkStatus();
}

}  // namespace

absl::Status GenericToSpecializedModel(
    const model::gradient_boosted_trees::GradientBoostedTreesModel& src,
    GradientBoostedTreesBinaryClassificationModel* dst) {
  if (src.loss() != Loss::BINOMIAL_LOG_LIKELIHOOD) {
    return absl::InvalidArgumentError(
        "The GBT is not trained with the binomial log likelihood loss.");
  }
  dst->initial_prediction = src.initial_predictions()[0];
  return RawGenericToSpecializedModel(src, dst);
}

absl::Status GenericToSpecializedModel(
    const model::gradient_boosted_trees::GradientBoostedTreesModel& src,
    GradientBoostedTreesBinaryRegressiveModel* dst) {
  if (src.loss() != Loss::BINOMIAL_LOG_LIKELIHOOD &&
      src.loss() != Loss::SQUARED_ERROR &&
      src.loss() != Loss::LAMBDA_MART_NDCG5 &&
      src.loss() != Loss::XE_NDCG_MART) {
    return absl::InvalidArgumentError(
        "The GBT is not trained with a compatible loss.");
  }
  dst->initial_prediction = src.initial_predictions()[0];
  return RawGenericToSpecializedModel(src, dst);
}

absl::Status Predict(const GradientBoostedTreesBinaryClassificationModel& model,
                     const std::vector<uint8_t>& examples,
                     uint32_t num_examples, std::vector<float>* predictions) {
  return RawPredict<ActivationBinomialLogLikelihood>(model, examples,
                                                     num_examples, predictions);
}

absl::Status Predict(const GradientBoostedTreesBinaryRegressiveModel& model,
                     const std::vector<uint8_t>& examples,
                     uint32_t num_examples, std::vector<float>* predictions) {
  return RawPredict<ActivationIdentity>(model, examples, num_examples,
                                        predictions);
}

template <typename Engine>
std::string GenericEngineDetails(const Engine& model) {
  std::string details;
  absl::StrAppendFormat(&details, "Ram usage (in bytes)\n");
  absl::StrAppendFormat(&details, "\tmasks: %d\n",
                        model.masks_v2.size() * sizeof(LeafMask));
  absl::StrAppendFormat(
      &details, "\tfeature_value_to_mask_list: %d\n",
      model.feature_value_to_mask_list.size() * sizeof(uint32_t));
  absl::StrAppendFormat(
      &details, "\tfeature_to_feature_value: %d\n",
      model.feature_to_feature_value.size() * sizeof(uint32_t));
  absl::StrAppendFormat(&details, "\tleaves: %d\n",
                        model.leaves.size() * sizeof(LeafOutput));
  absl::StrAppendFormat(&details, "\tmasks_feature_index: %d\n",
                        model.leaves_tree_index.size() * sizeof(uint32_t));

  return details;
}

std::string EngineDetails(
    const GradientBoostedTreesBinaryClassificationModel& model) {
  return GenericEngineDetails(model);
}

std::string EngineDetails(
    const GradientBoostedTreesBinaryRegressiveModel& model) {
  return GenericEngineDetails(model);
}

}  // namespace num_8bits
}  // namespace decision_forest
}  // namespace serving
}  // namespace yggdrasil_decision_forests
