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

// Utility methods to find the best splits (i.e. condition).
// See "training.h" header for the definition of the various concepts.
//
#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_DECISION_TREE_SPLITTER_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_DECISION_TREE_SPLITTER_H_

#include "yggdrasil_decision_forests/learner/decision_tree/splitter_accumulator.h"
#include "yggdrasil_decision_forests/learner/decision_tree/splitter_scanner.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/dataset_cache.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/dataset_cache_common.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/dataset_cache_reader.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/label_accessor.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_decision_tree {

// Index of a feature column.
typedef int FeatureIndex;

// Index of an open node within a single tree.
//
// (Currently) the code is optimized for GBT learner. There cannot be more than
// MaxValue(NodeIndex)-1 open nodes. In other words, no more than
// 2xMaxValue(NodeIndex) nodes. GBT generally have ~64-1024 nodes.
typedef uint16_t NodeIndex;

// Mapping between an example index and a node index.
typedef std::vector<NodeIndex> ExampleToNodeMap;

// Special node index indicating that a node is closed.
constexpr NodeIndex kClosedNode = std::numeric_limits<NodeIndex>::max();

// Definition and statistics of a split.
struct Split {
  decision_tree::proto::NodeCondition condition;
  // Label statistics for the negative (index 0) and positive (index 1)
  // branches.
  decision_tree::proto::LabelStatistics label_statistics[2];
};

// A splits+statistics for each of the currently open nodes.
typedef std::vector<Split> SplitPerOpenNode;

// Label statistics for each of the currently open nodes.
typedef std::vector<decision_tree::proto::LabelStatistics> LabelStatsPerNode;

// Common arguments for the "FindBestSplit" methods.
struct FindBestSplitsCommonArgs {
  const std::vector<std::vector<FeatureIndex>>& features_per_open_node;
  const ExampleToNodeMap& example_to_node;
  const dataset::proto::DataSpecification& dataspec;
  const AbstractLabelAccessor& label_accessor;
  const LabelStatsPerNode& label_stats_per_open_node;
  const bool has_multiple_node_idxs_;
  dataset_cache::DatasetCacheReader* const dataset;
  SplitPerOpenNode* best_splits;
};

// Accumulators and other data for the learning of numerical split.
template <typename LabelFiller>
struct NumericalSplitAccumulator {
  // Distribution of the label on the two sides of the split.
  typename LabelFiller::Accumulator pos, neg;

  double weighted_num_examples;

  // Number of examples on the two sides of the split.
  ExampleIndex num_examples_pos;
  ExampleIndex max_num_examples;

  // Number of delta-bit met so far i.e. number of unique feature value.
  ExampleIndex last_num_delta_bits = 0;

  // The "best_*" attributes below are related to the best split found so far.

  // Indices of the previous and next delta values that form the threshold.
  // If "best_next_delta_value_idx==0", no threshold is defined.
  ExampleIndex best_prev_delta_value_idx = 0;
  ExampleIndex best_next_delta_value_idx = 0;

  typename LabelFiller::Accumulator best_pos_accumulator;

  ExampleIndex best_num_examples_pos;
  double best_weighted_num_examples_pos;
  double best_score = 0;
};

template <typename LabelFiller>
absl::Status InitializerNumericalAccumulator(
    const FindBestSplitsCommonArgs& common,
    const std::vector<bool>& is_target_node,
    const std::vector<typename LabelFiller::AccumulatorInitializer>&
        accumulator_initializers,
    ExampleIndex min_num_examples,
    std::vector<NumericalSplitAccumulator<LabelFiller>>* accumulator_per_node) {
  accumulator_per_node->assign(common.features_per_open_node.size(), {});
  for (int node_idx = 0; node_idx < common.features_per_open_node.size();
       node_idx++) {
    if (!is_target_node[node_idx]) {
      continue;
    }

    auto& accumulator = (*accumulator_per_node)[node_idx];
    auto& initializer = accumulator_initializers[node_idx];

    // All the examples are in the "pos" side of the split.
    initializer.InitFull(&accumulator.pos);
    initializer.InitEmpty(&accumulator.neg);

    auto& label_stats = common.label_stats_per_open_node[node_idx];
    accumulator.num_examples_pos = label_stats.num_examples();

    accumulator.best_score =
        (*common.best_splits)[node_idx].condition.split_score();

    accumulator.max_num_examples =
        accumulator.num_examples_pos - min_num_examples;
    accumulator.weighted_num_examples = accumulator.pos.WeightedNumExamples();
  }
  return absl::OkStatus();
}

template <typename LabelFiller, typename ExampleBucketSet>
absl::Status InitializeCategoricalFeatureBuckets(
    const FindBestSplitsCommonArgs& common,
    const std::vector<bool>& is_target_node, const int num_feature_values,
    typename ExampleBucketSet::FeatureBucketType::Filler& feature_filler,
    const LabelFiller& label_filler,
    std::vector<ExampleBucketSet>* example_bucket_set_per_node) {
  example_bucket_set_per_node->resize(common.features_per_open_node.size());
  for (int active_node_idx = 0;
       active_node_idx < common.features_per_open_node.size();
       active_node_idx++) {
    if (!is_target_node[active_node_idx]) {
      continue;
    }
    auto& example_bucket_set = (*example_bucket_set_per_node)[active_node_idx];
    example_bucket_set.items.resize(num_feature_values);
    for (int feature_value = 0; feature_value < num_feature_values;
         feature_value++) {
      feature_filler.InitializeAndZero(
          feature_value, &example_bucket_set.items[feature_value].feature);
      label_filler.InitializeAndZeroBucket(
          &example_bucket_set.items[feature_value].label);
    }
  }
  return absl::OkStatus();
}

template <typename LabelFiller>
absl::Status FillNumericalAccumulator(
    const FindBestSplitsCommonArgs& common, FeatureIndex feature,
    const std::vector<bool>& is_target_node, const LabelFiller& label_filler,
    const std::vector<typename LabelFiller::AccumulatorInitializer>&
        accumulator_initializers,
    ExampleIndex min_num_examples,
    std::vector<NumericalSplitAccumulator<LabelFiller>>* accumulator_per_node) {
  // Scan over the dataset in increasing order of feature value.
  ASSIGN_OR_RETURN(
      auto example_it,
      common.dataset->PresortedNumericalFeatureExampleIterator(feature));

  const auto mask_delta_bit =
      dataset_cache::MaskDeltaBit(common.dataset->num_examples());
  const auto mask_examples_idx =
      dataset_cache::MaskExampleIdx(common.dataset->num_examples());

  const auto has_multiple_node_idxs = common.has_multiple_node_idxs_;

  // Number of times the delta bit was observed.
  ExampleIndex num_delta_bits = 0;

  while (true) {
    RETURN_IF_ERROR(example_it->Next());
    const auto delta_example_idxs = example_it->Values();
    if (delta_example_idxs.empty()) {
      break;
    }

    for (const auto delta_example_idx : delta_example_idxs) {
      const auto example_idx = delta_example_idx & mask_examples_idx;
      DCHECK_GE(example_idx, 0);
      DCHECK_LT(example_idx, common.dataset->num_examples());

      // Check that the current and previous examples are separable.
      if ((delta_example_idx & mask_delta_bit) != 0) {
        num_delta_bits++;
      }

      // TODO(gbm): Maybe. Compile time condition.
      NodeIndex node_idx;
      if (has_multiple_node_idxs) {
        // Retrieve the node containing the example.
        node_idx = common.example_to_node[example_idx];

        // Check that the node is not closed.
        if (node_idx == kClosedNode) {
          continue;
        }

        // Check that the node is a target of this feature.
        if (!is_target_node[node_idx]) {
          continue;
        }
      } else {
        node_idx = 0;
        DCHECK_EQ(common.example_to_node[example_idx], 0);
        DCHECK(is_target_node[0]);
      }

      auto& accumulator = (*accumulator_per_node)[node_idx];

      // New feature values?
      if (num_delta_bits != accumulator.last_num_delta_bits) {
        // Update the delta bit.
        const auto save_last_num_delta_bits = accumulator.last_num_delta_bits;
        accumulator.last_num_delta_bits = num_delta_bits;

        // Enough examples?
        if (accumulator.num_examples_pos >= min_num_examples &&
            accumulator.num_examples_pos <= accumulator.max_num_examples) {
          const auto score =
              decision_tree::Score<>(accumulator_initializers[node_idx],
                                     accumulator.weighted_num_examples,
                                     accumulator.pos, accumulator.neg);
          if (score > accumulator.best_score) {
            // We found a better split.
            accumulator.best_score = score;
            accumulator.best_prev_delta_value_idx = save_last_num_delta_bits;
            accumulator.best_next_delta_value_idx =
                accumulator.last_num_delta_bits;
            accumulator.best_num_examples_pos = accumulator.num_examples_pos;
            accumulator.best_weighted_num_examples_pos =
                accumulator.pos.WeightedNumExamples();
            accumulator.best_pos_accumulator = accumulator.pos;
          }
        }
      }

      // Update the accumulators.
      label_filler.Add(example_idx, &accumulator.neg);
      label_filler.Sub(example_idx, &accumulator.pos);
      DCHECK_GT(accumulator.num_examples_pos, 0);
      accumulator.num_examples_pos--;
    }
  }

  // Check that the counts are as expected.
  for (int active_node_idx = 0;
       active_node_idx < common.features_per_open_node.size();
       active_node_idx++) {
    if (!is_target_node[active_node_idx]) {
      continue;
    }
    auto& accumulator = (*accumulator_per_node)[active_node_idx];
    if (accumulator.num_examples_pos != 0) {
      return absl::InternalError("Unexpected number of training examples");
    }
  }

  return example_it->Close();
}

template <typename LabelFiller, typename ExampleBucketSet>
absl::Status FillCategoricalFeatureBuckets(
    const FindBestSplitsCommonArgs& common, FeatureIndex feature,
    const std::vector<bool>& is_target_node, const LabelFiller& label_filler,
    const int num_feature_values,
    std::vector<ExampleBucketSet>* example_bucket_set_per_node) {
  ASSIGN_OR_RETURN(
      auto value_it,
      common.dataset->InOrderCategoricalFeatureValueIterator(feature));

  const auto has_multiple_node_idxs = common.has_multiple_node_idxs_;

  ExampleIndex example_idx = 0;
  while (true) {
    RETURN_IF_ERROR(value_it->Next());
    const auto values = value_it->Values();
    if (values.empty()) {
      break;
    }

    for (auto value : values) {
      DCHECK_GE(value, 0);
      DCHECK_LT(value, num_feature_values);

      // TODO(gbm): Maybe. Compile time condition.
      NodeIndex node_idx;
      if (has_multiple_node_idxs) {
        // Retrieve the node containing the example.
        node_idx = common.example_to_node[example_idx];

        // Check that the node is not closed and a target feature.
        if (node_idx == kClosedNode || !is_target_node[node_idx]) {
          example_idx++;
          continue;
        }

      } else {
        node_idx = 0;
        DCHECK_EQ(common.example_to_node[example_idx], 0);
        DCHECK(is_target_node[0]);
      }

      auto& example_bucket_set = (*example_bucket_set_per_node)[node_idx];
      auto& bucket = example_bucket_set.items[value];

      // Update the bucket.
      label_filler.Add(example_idx, &bucket.label);

      example_idx++;
    }
  }
  return value_it->Close();
}

template <typename LabelFiller, typename ExampleBucketSet>
absl::Status FillDiscretizedNumericalAccumulatorPartial(
    const FindBestSplitsCommonArgs& common, FeatureIndex feature,
    const std::vector<bool>& is_target_node, const LabelFiller& label_filler,
    const int num_feature_values, size_t begin_idx, size_t end_idx,
    std::vector<ExampleBucketSet>* example_bucket_set_per_node) {
  ASSIGN_OR_RETURN(
      auto value_it,
      common.dataset->InOrderDiscretizedNumericalFeatureValueIterator(
          feature, begin_idx, end_idx));

  const auto has_multiple_node_idxs = common.has_multiple_node_idxs_;

  ExampleIndex example_idx = begin_idx;
  while (true) {
    RETURN_IF_ERROR(value_it->Next());
    const auto values = value_it->Values();
    if (values.empty()) {
      break;
    }

    for (auto value : values) {
      DCHECK_GE(value, 0);
      DCHECK_LT(value, num_feature_values);

      // TODO(gbm): Maybe. Compile time condition.
      NodeIndex node_idx;
      if (has_multiple_node_idxs) {
        // Retrieve the node containing the example.
        node_idx = common.example_to_node[example_idx];

        // Check that the node is not closed and a target feature.
        if (node_idx == kClosedNode || !is_target_node[node_idx]) {
          example_idx++;
          continue;
        }

      } else {
        node_idx = 0;
        DCHECK_EQ(common.example_to_node[example_idx], 0);
        DCHECK(is_target_node[0]);
      }

      auto& example_bucket_set = (*example_bucket_set_per_node)[node_idx];
      auto& bucket = example_bucket_set.items[value];

      // Update the bucket.
      label_filler.Add(example_idx, &bucket.label);

      example_idx++;
    }
  }
  return value_it->Close();
}

template <typename LabelFiller, typename ExampleBucketSet>
absl::Status FillDiscretizedNumericalAccumulator(
    const FindBestSplitsCommonArgs& common, FeatureIndex feature,
    const std::vector<bool>& is_target_node, const LabelFiller& label_filler,
    const int num_feature_values,
    std::vector<ExampleBucketSet>* example_bucket_set_per_node) {
  ASSIGN_OR_RETURN(
      auto value_it,
      common.dataset->InOrderDiscretizedNumericalFeatureValueIterator(feature));

  const auto has_multiple_node_idxs = common.has_multiple_node_idxs_;

  ExampleIndex example_idx = 0;
  while (true) {
    RETURN_IF_ERROR(value_it->Next());
    const auto values = value_it->Values();
    if (values.empty()) {
      break;
    }

    for (auto value : values) {
      DCHECK_GE(value, 0);
      DCHECK_LT(value, num_feature_values);

      // TODO(gbm): Maybe. Compile time condition.
      NodeIndex node_idx;
      if (has_multiple_node_idxs) {
        // Retrieve the node containing the example.
        node_idx = common.example_to_node[example_idx];

        // Check that the node is not closed and a target feature.
        if (node_idx == kClosedNode || !is_target_node[node_idx]) {
          example_idx++;
          continue;
        }

      } else {
        node_idx = 0;
        DCHECK_EQ(common.example_to_node[example_idx], 0);
        DCHECK(is_target_node[0]);
      }

      auto& example_bucket_set = (*example_bucket_set_per_node)[node_idx];
      auto& bucket = example_bucket_set.items[value];

      // Update the bucket.
      label_filler.Add(example_idx, &bucket.label);

      example_idx++;
    }
  }
  return value_it->Close();
}

template <typename LabelFiller, typename ExampleBucketSet>
absl::Status FillBooleanFeatureBuckets(
    const FindBestSplitsCommonArgs& common, FeatureIndex feature,
    const std::vector<bool>& is_target_node, const LabelFiller& label_filler,
    std::vector<ExampleBucketSet>* example_bucket_set_per_node) {
  ASSIGN_OR_RETURN(auto value_it,
                   common.dataset->InOrderBooleanFeatureValueIterator(feature));

  const auto has_multiple_node_idxs = common.has_multiple_node_idxs_;

  ExampleIndex example_idx = 0;
  while (true) {
    RETURN_IF_ERROR(value_it->Next());
    const auto values = value_it->Values();
    if (values.empty()) {
      break;
    }

    for (auto value : values) {
      DCHECK_GE(value, 0);
      DCHECK_LT(value, 2);

      NodeIndex node_idx;
      if (has_multiple_node_idxs) {
        // Retrieve the node containing the example.
        node_idx = common.example_to_node[example_idx];

        // Check that the node is not closed and a target feature.
        if (node_idx == kClosedNode || !is_target_node[node_idx]) {
          example_idx++;
          continue;
        }

      } else {
        node_idx = 0;
        DCHECK_EQ(common.example_to_node[example_idx], 0);
        DCHECK(is_target_node[0]);
      }

      auto& example_bucket_set = (*example_bucket_set_per_node)[node_idx];
      auto& bucket = example_bucket_set.items[value];

      // Update the bucket.
      label_filler.Add(example_idx, &bucket.label);

      example_idx++;
    }
  }
  return value_it->Close();
}

// Set the label statistics for the two children of a categorical split.
template <typename LabelFiller, typename ExampleBucketSet>
absl::Status ComputeSplitLabelStatisticsFromCategoricalSplit(
    const FindBestSplitsCommonArgs& common, FeatureIndex feature,
    const typename LabelFiller::AccumulatorInitializer& initializer,
    const ExampleBucketSet& example_bucket_set, Split* split) {
  // Compute the label statistics in the child nodes.
  const auto positive_values =
      decision_tree::ExactElementsFromContainsCondition(
          common.dataspec.columns(feature)
              .categorical()
              .number_of_unique_values(),
          split->condition.condition());

  // The positive split contains the examples containing the selected
  // items.
  typename LabelFiller::Accumulator pos;
  initializer.InitEmpty(&pos);
  for (const auto value : positive_values) {
    example_bucket_set.items[value].label.AddToScoreAcc(&pos);
  }

  // The negative split is complementary to the positive split.
  typename LabelFiller::Accumulator neg;
  initializer.InitFull(&neg);
  pos.Sub(&neg);

  neg.ExportLabelStats(&split->label_statistics[0]);
  pos.ExportLabelStats(&split->label_statistics[1]);

  split->label_statistics[0].set_num_examples(
      split->condition.num_training_examples_without_weight() -
      split->condition.num_pos_training_examples_without_weight());
  split->label_statistics[1].set_num_examples(
      split->condition.num_pos_training_examples_without_weight());
  return absl::OkStatus();
}

// Set the label statistics for the two children of a discretized numerical
// split.
template <typename LabelFiller, typename ExampleBucketSet>
absl::Status ComputeSplitLabelStatisticsFromDiscretizedNumericalSplit(
    const FindBestSplitsCommonArgs& common, FeatureIndex feature,
    const typename LabelFiller::AccumulatorInitializer& initializer,
    const ExampleBucketSet& example_bucket_set, Split* split) {
  const auto discretized_threshold =
      split->condition.condition().discretized_higher_condition().threshold();

  // The positive split contains all the bucket greater or equal to the
  // threshold.
  typename LabelFiller::Accumulator pos;
  initializer.InitEmpty(&pos);
  for (int positive_bucket_idx = discretized_threshold;
       positive_bucket_idx < example_bucket_set.items.size();
       positive_bucket_idx++) {
    example_bucket_set.items[positive_bucket_idx].label.AddToScoreAcc(&pos);
  }

  // The negative split is complementary to the positive split.
  typename LabelFiller::Accumulator neg;
  initializer.InitFull(&neg);
  pos.Sub(&neg);

  neg.ExportLabelStats(&split->label_statistics[0]);
  pos.ExportLabelStats(&split->label_statistics[1]);

  split->label_statistics[0].set_num_examples(
      split->condition.num_training_examples_without_weight() -
      split->condition.num_pos_training_examples_without_weight());
  split->label_statistics[1].set_num_examples(
      split->condition.num_pos_training_examples_without_weight());
  return absl::OkStatus();
}

template <typename LabelFiller, typename ExampleBucketSet>
absl::Status ComputeSplitLabelStatisticsFromBooleanSplit(
    const FindBestSplitsCommonArgs& common, FeatureIndex feature,
    const typename LabelFiller::AccumulatorInitializer& initializer,
    const ExampleBucketSet& example_bucket_set, Split* split) {
  DCHECK_EQ(example_bucket_set.items.size(), 2);
  typename LabelFiller::Accumulator pos;
  initializer.InitEmpty(&pos);
  example_bucket_set.items[1].label.AddToScoreAcc(&pos);

  // The negative split is complementary to the positive split.
  typename LabelFiller::Accumulator neg;
  initializer.InitFull(&neg);
  pos.Sub(&neg);

  neg.ExportLabelStats(&split->label_statistics[0]);
  pos.ExportLabelStats(&split->label_statistics[1]);

  split->label_statistics[0].set_num_examples(
      split->condition.num_training_examples_without_weight() -
      split->condition.num_pos_training_examples_without_weight());
  split->label_statistics[1].set_num_examples(
      split->condition.num_pos_training_examples_without_weight());
  return absl::OkStatus();
}

template <typename LabelFiller>
absl::Status FindSortedNumericalThreshold(
    const FindBestSplitsCommonArgs& common, FeatureIndex feature,
    const std::vector<bool>& is_target_node,
    const std::vector<NumericalSplitAccumulator<LabelFiller>>&
        accumulator_per_node) {
  // A delta value of interest to determine the threshold of a split.
  struct TargetDeltaValue {
    ExampleIndex delta_value_idx;
    bool is_next;
    int node_idx;
  };

  // Sorted (increasing order) mapping between the num_delta_bits and the node
  // index where this delta bit is the best split.
  std::vector<TargetDeltaValue> target_delta_values;

  // Assemble the final condition (without the thresholds).
  for (int active_node_idx = 0;
       active_node_idx < common.features_per_open_node.size();
       active_node_idx++) {
    if (!is_target_node[active_node_idx]) {
      continue;
    }

    auto& accumulator = accumulator_per_node[active_node_idx];
    if (accumulator.best_next_delta_value_idx == 0) {
      continue;
    }
    auto& label_stats = common.label_stats_per_open_node[active_node_idx];
    auto& split = (*common.best_splits)[active_node_idx];

    // The threshold value is set in the next loop.
    split.condition.mutable_condition()
        ->mutable_higher_condition()
        ->set_threshold(std::numeric_limits<float>::quiet_NaN());

    split.condition.set_attribute(feature);
    split.condition.set_num_pos_training_examples_without_weight(
        accumulator.best_num_examples_pos);
    split.condition.set_num_pos_training_examples_with_weight(
        accumulator.best_weighted_num_examples_pos);
    split.condition.set_num_training_examples_without_weight(
        label_stats.num_examples());
    split.condition.set_num_training_examples_with_weight(
        accumulator.neg.WeightedNumExamples());
    split.condition.set_split_score(accumulator.best_score);

    // Compute the label statistics in the child nodes.
    typename LabelFiller::Accumulator best_neg_accumulator;
    best_neg_accumulator.ImportLabelStats(label_stats);
    accumulator.best_pos_accumulator.Sub(&best_neg_accumulator);
    best_neg_accumulator.ExportLabelStats(&split.label_statistics[0]);
    accumulator.best_pos_accumulator.ExportLabelStats(
        &split.label_statistics[1]);

    split.label_statistics[0].set_num_examples(
        label_stats.num_examples() - accumulator.best_num_examples_pos);
    split.label_statistics[1].set_num_examples(
        accumulator.best_num_examples_pos);

    target_delta_values.push_back(
        {/*delta_value_idx=*/accumulator.best_prev_delta_value_idx,
         /*is_next*/ false, /*node_idx=*/active_node_idx});

    target_delta_values.push_back(
        {/*delta_value_idx=*/accumulator.best_next_delta_value_idx,
         /*is_next*/ true, /*node_idx=*/active_node_idx});
  }

  // Assemble the splitting thresholds.
  if (!target_delta_values.empty()) {
    std::stable_sort(target_delta_values.begin(), target_delta_values.end(),
                     [](const auto& a, const auto& b) {
                       return a.delta_value_idx < b.delta_value_idx;
                     });

    ASSIGN_OR_RETURN(
        auto deta_value_it,
        common.dataset->PresortedNumericalFeatureValueIterator(feature));

    ExampleIndex sum_num_delta = 0;
    RETURN_IF_ERROR(deta_value_it->Next());
    auto delta_values = deta_value_it->Values();

    // Pre-split delta values indexed by node index.
    std::vector<float> prev_delta_values(
        common.features_per_open_node.size(),
        std::numeric_limits<double>::quiet_NaN());

    for (const auto target_value : target_delta_values) {
      // TODO(gbm): Implement a "skip" method in the reader.
      DCHECK(!delta_values.empty());

      while (target_value.delta_value_idx >=
             sum_num_delta + delta_values.size()) {
        sum_num_delta += delta_values.size();
        RETURN_IF_ERROR(deta_value_it->Next());
        delta_values = deta_value_it->Values();
        DCHECK(!delta_values.empty());
      }

      const float delta_value =
          delta_values[target_value.delta_value_idx - sum_num_delta];
      if (target_value.is_next) {
        const auto threshold = decision_tree::MidThreshold(
            prev_delta_values[target_value.node_idx], delta_value);

        (*common.best_splits)[target_value.node_idx]
            .condition.mutable_condition()
            ->mutable_higher_condition()
            ->set_threshold(threshold);

        (*common.best_splits)[target_value.node_idx].condition.set_na_value(
            common.dataset->meta_data()
                .columns(feature)
                .numerical()
                .replacement_missing_value() >= threshold);

      } else {
        prev_delta_values[target_value.node_idx] = delta_value;
      }
    }

    RETURN_IF_ERROR(deta_value_it->Close());

#ifndef NDEBUG
    // Make sure the threshold value is set.
    for (int active_node_idx = 0;
         active_node_idx < common.features_per_open_node.size();
         active_node_idx++) {
      if (!is_target_node[active_node_idx]) {
        continue;
      }
      auto& accumulator = accumulator_per_node[active_node_idx];
      if (accumulator.best_next_delta_value_idx == 0) {
        continue;
      }
      auto& split = (*common.best_splits)[active_node_idx];
      DCHECK(!std::isnan(
          split.condition.condition().higher_condition().threshold()));
    }
#endif
  }

  return absl::OkStatus();
}

template <typename LabelFiller, typename ExampleBucketSet>
absl::Status FindDiscretizedNumericalThreshold(
    const FindBestSplitsCommonArgs& common, FeatureIndex feature,
    const std::vector<bool>& is_target_node, const LabelFiller& label_filler,
    const std::vector<typename LabelFiller::AccumulatorInitializer>&
        accumulator_initializers,
    ExampleIndex min_num_examples,
    const std::vector<ExampleBucketSet>& example_bucket_set_per_node,
    const typename ExampleBucketSet::FeatureBucketType::Filler& feature_filler,
    decision_tree::PerThreadCacheV2* cache) {
  // Test each of the nodes.
  for (int node_idx = 0; node_idx < common.features_per_open_node.size();
       node_idx++) {
    if (!is_target_node[node_idx]) {
      continue;
    }

    const auto& example_bucket_set = example_bucket_set_per_node[node_idx];
    const auto& label_stats = common.label_stats_per_open_node[node_idx];
    const auto& initializer = accumulator_initializers[node_idx];

    auto& split = (*common.best_splits)[node_idx];
    if (decision_tree::ScanSplits<ExampleBucketSet,
                                  typename LabelFiller::Accumulator,
                                  /*bucket_interpolation=*/true>(
            feature_filler, accumulator_initializers[node_idx],
            example_bucket_set, label_stats.num_examples(), min_num_examples,
            feature, &split.condition,
            cache) == decision_tree::SplitSearchResult::kBetterSplitFound) {
      RETURN_IF_ERROR((ComputeSplitLabelStatisticsFromDiscretizedNumericalSplit<
                       LabelFiller, ExampleBucketSet>(
          common, feature, initializer, example_bucket_set, &split)));

      // Transform the condition to a non discretized one.
      const auto discretized_threshold = split.condition.condition()
                                             .discretized_higher_condition()
                                             .threshold();
      const auto& boundaries =
          common.dataset->DiscretizedNumericalFeatureBoundaries(feature);
      const auto threshold = boundaries[discretized_threshold - 1];

      split.condition.mutable_condition()
          ->mutable_higher_condition()
          ->set_threshold(threshold);
    }
  }

  return absl::OkStatus();
}

template <typename LabelFiller, typename ExampleBucketSet>
absl::Status OneVsOtherClassificationAndCategoricalFeatureBuckets(
    const FindBestSplitsCommonArgs& common, FeatureIndex feature,
    const std::vector<bool>& is_target_node, const LabelFiller& label_filler,
    const std::vector<typename LabelFiller::AccumulatorInitializer>&
        accumulator_initializers,
    ExampleIndex min_num_examples, const int num_feature_values,
    const int num_label_values,
    const std::vector<ExampleBucketSet>& example_bucket_set_per_node,
    const typename ExampleBucketSet::FeatureBucketType::Filler& feature_filler,
    decision_tree::PerThreadCacheV2* cache) {
  // Custom ordering of the features.
  std::vector<std::pair<float, int32_t>> feature_order(num_feature_values);

  // Re-order the bucket and test all the splits defined by continuous buckets.
  for (int node_idx = 0; node_idx < common.features_per_open_node.size();
       node_idx++) {
    if (!is_target_node[node_idx]) {
      continue;
    }

    const auto& example_bucket_set = example_bucket_set_per_node[node_idx];
    const auto& label_stats = common.label_stats_per_open_node[node_idx];
    const auto& initializer = accumulator_initializers[node_idx];

    for (int32_t positive_label = 0; positive_label < num_label_values;
         positive_label++) {
      if (accumulator_initializers[node_idx].IsEmpty(positive_label)) {
        // Never observed label value.
        continue;
      }
      if (num_label_values == 3 && positive_label == 1) {
        // "True vs others" or "False vs others" are equivalent for binary
        // classification.
        continue;
      }

      // Order value of the buckets.
      for (int feature_value = 0; feature_value < num_feature_values;
           feature_value++) {
        const auto& bucket = example_bucket_set.items[feature_value];
        const float ratio_positive_label =
            bucket.label.SafeProportionOrMinusInfinity(positive_label);
        DCHECK(!std::isnan(ratio_positive_label));
        feature_order[feature_value] = {ratio_positive_label, feature_value};
      }

      // Sort the bucket indices.
      std::sort(feature_order.begin(), feature_order.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });

      auto& split = (*common.best_splits)[node_idx];
      if (decision_tree::ScanSplitsCustomOrder<
              ExampleBucketSet, typename LabelFiller::Accumulator>(
              feature_order, feature_filler, accumulator_initializers[node_idx],
              example_bucket_set, label_stats.num_examples(), min_num_examples,
              feature, &split.condition,
              cache) == decision_tree::SplitSearchResult::kBetterSplitFound) {
        RETURN_IF_ERROR(
            (ComputeSplitLabelStatisticsFromCategoricalSplit<LabelFiller,
                                                             ExampleBucketSet>(
                common, feature, initializer, example_bucket_set, &split)));
      }
    }
  }

  return absl::OkStatus();
}

template <typename LabelFiller, typename ExampleBucketSet>
absl::Status InOrderRegressionAndCategoricalFeatureBuckets(
    const FindBestSplitsCommonArgs& common, FeatureIndex feature,
    const std::vector<bool>& is_target_node, const LabelFiller& label_filler,
    const std::vector<typename LabelFiller::AccumulatorInitializer>&
        accumulator_initializers,
    ExampleIndex min_num_examples, const int num_feature_values,
    const std::vector<ExampleBucketSet>& example_bucket_set_per_node,
    const typename ExampleBucketSet::FeatureBucketType::Filler& feature_filler,
    decision_tree::PerThreadCacheV2* cache) {
  // Custom ordering of the features.
  std::vector<std::pair<float, int32_t>> feature_order(num_feature_values);

  // Test each of the nodes.
  for (int node_idx = 0; node_idx < common.features_per_open_node.size();
       node_idx++) {
    if (!is_target_node[node_idx]) {
      continue;
    }

    const auto& example_bucket_set = example_bucket_set_per_node[node_idx];
    const auto& label_stats = common.label_stats_per_open_node[node_idx];
    const auto& initializer = accumulator_initializers[node_idx];

    // Order value of the buckets.
    for (int feature_value = 0; feature_value < num_feature_values;
         feature_value++) {
      const auto& bucket = example_bucket_set.items[feature_value];
      feature_order[feature_value] = {bucket.label.value.Mean(), feature_value};
    }

    // Sort the bucket indices.
    std::sort(feature_order.begin(), feature_order.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    auto& split = (*common.best_splits)[node_idx];
    if (decision_tree::ScanSplitsCustomOrder<ExampleBucketSet,
                                             typename LabelFiller::Accumulator>(
            feature_order, feature_filler, accumulator_initializers[node_idx],
            example_bucket_set, label_stats.num_examples(), min_num_examples,
            feature, &split.condition,
            cache) == decision_tree::SplitSearchResult::kBetterSplitFound) {
      RETURN_IF_ERROR(
          (ComputeSplitLabelStatisticsFromCategoricalSplit<LabelFiller,
                                                           ExampleBucketSet>(
              common, feature, initializer, example_bucket_set, &split)));
    }
  }

  return absl::OkStatus();
}

template <typename LabelFiller, typename ExampleBucketSet>
absl::Status InOrderRegressionAndBooleanFeatureBuckets(
    const FindBestSplitsCommonArgs& common, FeatureIndex feature,
    const std::vector<bool>& is_target_node, const LabelFiller& label_filler,
    const std::vector<typename LabelFiller::AccumulatorInitializer>&
        accumulator_initializers,
    ExampleIndex min_num_examples,
    const std::vector<ExampleBucketSet>& example_bucket_set_per_node,
    const typename ExampleBucketSet::FeatureBucketType::Filler& feature_filler,
    decision_tree::PerThreadCacheV2* cache) {
  // Test each of the nodes.
  for (int node_idx = 0; node_idx < common.features_per_open_node.size();
       node_idx++) {
    if (!is_target_node[node_idx]) {
      continue;
    }

    const auto& example_bucket_set = example_bucket_set_per_node[node_idx];
    const auto& label_stats = common.label_stats_per_open_node[node_idx];
    const auto& initializer = accumulator_initializers[node_idx];

    auto& split = (*common.best_splits)[node_idx];
    if (decision_tree::ScanSplits<ExampleBucketSet,
                                  typename LabelFiller::Accumulator>(
            feature_filler, accumulator_initializers[node_idx],
            example_bucket_set, label_stats.num_examples(), min_num_examples,
            feature, &split.condition,
            cache) == decision_tree::SplitSearchResult::kBetterSplitFound) {
      RETURN_IF_ERROR(
          (ComputeSplitLabelStatisticsFromBooleanSplit<LabelFiller,
                                                       ExampleBucketSet>(
              common, feature, initializer, example_bucket_set, &split)));
    }
  }

  return absl::OkStatus();
}

// Splitter for a numerical feature. Support all types of labels.
template <typename LabelFiller>
absl::Status TemplatedFindBestSplitsWithSortedNumericalFeature(
    const FindBestSplitsCommonArgs& common, FeatureIndex feature,
    const std::vector<bool>& is_target_node, const LabelFiller& label_filler,
    const std::vector<typename LabelFiller::AccumulatorInitializer>&
        accumulator_initializers,
    ExampleIndex min_num_examples) {
  // Initialize the score accumulators for the target nodes.
  std::vector<NumericalSplitAccumulator<LabelFiller>> accumulator_per_node;
  RETURN_IF_ERROR(InitializerNumericalAccumulator<>(
      common, is_target_node, accumulator_initializers, min_num_examples,
      &accumulator_per_node));

  // Scan the dataset to find good splits.
  RETURN_IF_ERROR(FillNumericalAccumulator(
      common, feature, is_target_node, label_filler, accumulator_initializers,
      min_num_examples, &accumulator_per_node));

  // Determine the threshold value of the splits.
  RETURN_IF_ERROR(FindSortedNumericalThreshold(common, feature, is_target_node,
                                               accumulator_per_node));

  return absl::OkStatus();
}

// Splitter for a numerical feature. Support all types of labels.
template <typename LabelFiller>
absl::Status TemplatedFindBestSplitsWithDiscretizedNumericalFeature(
    const FindBestSplitsCommonArgs& common, FeatureIndex feature,
    const std::vector<bool>& is_target_node, const LabelFiller& label_filler,
    const std::vector<typename LabelFiller::AccumulatorInitializer>&
        accumulator_initializers,
    ExampleIndex min_num_examples) {
  // Contains the condition conditional label statistics.
  const auto num_discretized_values = common.dataset->meta_data()
                                          .columns(feature)
                                          .numerical()
                                          .num_discretized_values();

  const auto replacement_missing_value =
      common.dataset->meta_data()
          .columns(feature)
          .numerical()
          .discretized_replacement_missing_value();

  using ExampleBucketSet =
      decision_tree::ExampleBucketSet<decision_tree::ExampleBucket<
          decision_tree::FeatureDiscretizedNumericalBucket,
          typename LabelFiller::LabelBucket>>;

  typename ExampleBucketSet::FeatureBucketType::Filler feature_filler(
      num_discretized_values, replacement_missing_value, {});

  // ExampleBucket for each of the open nodes.
  std::vector<ExampleBucketSet> example_bucket_set_per_node;

  RETURN_IF_ERROR(InitializeCategoricalFeatureBuckets<>(
      common, is_target_node, /*num_feature_values=*/num_discretized_values,
      feature_filler, label_filler, &example_bucket_set_per_node));

  // Scan the dataset to find good splits.
  RETURN_IF_ERROR(FillDiscretizedNumericalAccumulator(
      common, feature, is_target_node, label_filler,
      /*num_feature_values=*/num_discretized_values,
      &example_bucket_set_per_node));

  // Determine the threshold value of the splits.
  decision_tree::PerThreadCacheV2 cache;
  RETURN_IF_ERROR(FindDiscretizedNumericalThreshold(
      common, feature, is_target_node, label_filler, accumulator_initializers,
      min_num_examples, example_bucket_set_per_node, feature_filler, &cache));

  return absl::OkStatus();
}

// Same as "TemplatedFindBestSplitsWithDiscretizedNumericalFeature", but with
// multi-threading distribution.
template <typename LabelFiller>
absl::Status
TemplatedFindBestSplitsWithDiscretizedNumericalFeatureMultiThreading(
    const FindBestSplitsCommonArgs& common, FeatureIndex feature,
    const std::vector<bool>& is_target_node, const LabelFiller& label_filler,
    const std::vector<typename LabelFiller::AccumulatorInitializer>&
        accumulator_initializers,
    ExampleIndex min_num_examples, int num_threads) {
  DCHECK_GE(num_threads, 1);

  if (num_threads == 1) {
    return TemplatedFindBestSplitsWithDiscretizedNumericalFeature(
        common, feature, is_target_node, label_filler, accumulator_initializers,
        min_num_examples);
  }

  // Contains the condition conditional label statistics.
  const auto num_discretized_values = common.dataset->meta_data()
                                          .columns(feature)
                                          .numerical()
                                          .num_discretized_values();

  const auto replacement_missing_value =
      common.dataset->meta_data()
          .columns(feature)
          .numerical()
          .discretized_replacement_missing_value();

  using ExampleBucketSet =
      decision_tree::ExampleBucketSet<decision_tree::ExampleBucket<
          decision_tree::FeatureDiscretizedNumericalBucket,
          typename LabelFiller::LabelBucket>>;

  typename ExampleBucketSet::FeatureBucketType::Filler feature_filler(
      num_discretized_values, replacement_missing_value, {});

  // ExampleBucket for each of the threads and each of the open nodes.
  std::vector<std::vector<ExampleBucketSet>> example_bucket_set_per_node(
      num_threads);

  absl::Status status;
  utils::concurrency::Mutex mutex_status;

  {
    utils::concurrency::ThreadPool thread_pool("splitter", num_threads);
    thread_pool.StartWorkers();

    RETURN_IF_ERROR(InitializeCategoricalFeatureBuckets<>(
        common, is_target_node, /*num_feature_values=*/num_discretized_values,
        feature_filler, label_filler, &example_bucket_set_per_node.front()));

    for (int thread_idx = 1; thread_idx < num_threads; thread_idx++) {
      example_bucket_set_per_node[thread_idx] =
          example_bucket_set_per_node.front();
    }

    size_t begin_idx = 0;
    const auto block_size =
        (common.dataset->num_examples() + num_threads - 1) / num_threads;
    for (int thread_idx = 0; thread_idx < num_threads; thread_idx++) {
      const auto end_idx =
          std::min(begin_idx + block_size, common.dataset->num_examples());

      thread_pool.Schedule([&common, &feature, &is_target_node, &label_filler,
                            &example_bucket_set_per_node,
                            &num_discretized_values, &mutex_status, &status,
                            begin_idx, end_idx, thread_idx]() -> void {
        // Scan the dataset to find good splits.
        const auto local_status = FillDiscretizedNumericalAccumulatorPartial(
            common, feature, is_target_node, label_filler,
            /*num_feature_values=*/num_discretized_values, begin_idx, end_idx,
            &example_bucket_set_per_node[thread_idx]);

        utils::concurrency::MutexLock l(&mutex_status);
        status.Update(local_status);
      });

      begin_idx += block_size;
    }
  }

  RETURN_IF_ERROR(status);

  for (int thread_idx = 1; thread_idx < num_threads; thread_idx++) {
    const auto& src = example_bucket_set_per_node[thread_idx];
    auto& dst = example_bucket_set_per_node.front();
    for (size_t bucket_set_idx = 0; bucket_set_idx < src.size();
         bucket_set_idx++) {
      decision_tree::AddLabelBucket<ExampleBucketSet>(src[bucket_set_idx],
                                                      &dst[bucket_set_idx]);
    }
  }

  // Determine the threshold value of the splits.
  decision_tree::PerThreadCacheV2 cache;
  RETURN_IF_ERROR(FindDiscretizedNumericalThreshold(
      common, feature, is_target_node, label_filler, accumulator_initializers,
      min_num_examples, example_bucket_set_per_node.front(), feature_filler,
      &cache));

  return absl::OkStatus();
}

// Splitter for a categorical features and a classification label.
template <typename LabelFiller>
absl::Status TemplatedFindBestSplitsWithClassificationAndCategoricalFeature(
    const FindBestSplitsCommonArgs& common, FeatureIndex feature,
    const std::vector<bool>& is_target_node, const LabelFiller& label_filler,
    const std::vector<typename LabelFiller::AccumulatorInitializer>&
        accumulator_initializers,
    ExampleIndex min_num_examples) {
  decision_tree::PerThreadCacheV2 cache;

  const int num_feature_values =
      common.dataset->meta_data().columns(feature).categorical().num_values();
  const int num_label_values =
      common.dataset->meta_data()
          .columns(common.dataset->meta_data().label_column_idx())
          .categorical()
          .num_values();

  // Contains the condition conditional label statistics.
  using ExampleBucketSet = decision_tree::ExampleBucketSet<
      decision_tree::ExampleBucket<decision_tree::FeatureCategoricalBucket,
                                   typename LabelFiller::LabelBucket>>;

  typename ExampleBucketSet::FeatureBucketType::Filler feature_filler(
      num_feature_values,
      common.dataset->meta_data()
          .columns(feature)
          .categorical()
          .replacement_missing_value(),
      {});

  // ExampleBucket for each of the open nodes.
  std::vector<ExampleBucketSet> example_bucket_set_per_node;

  // Initialize the buckets.
  RETURN_IF_ERROR(InitializeCategoricalFeatureBuckets<>(
      common, is_target_node, num_feature_values, feature_filler, label_filler,
      &example_bucket_set_per_node));

  // Aggregate the label values per bucket and open node.
  RETURN_IF_ERROR(FillCategoricalFeatureBuckets(
      common, feature, is_target_node, label_filler, num_feature_values,
      &example_bucket_set_per_node));

  return OneVsOtherClassificationAndCategoricalFeatureBuckets<>(
      common, feature, is_target_node, label_filler, accumulator_initializers,
      min_num_examples, num_feature_values, num_label_values,
      example_bucket_set_per_node, feature_filler, &cache);
}

// Splitter for a boolean features and a classification label.
template <typename LabelFiller>
absl::Status TemplatedFindBestSplitsWithClassificationAndBooleanFeature(
    const FindBestSplitsCommonArgs& common, FeatureIndex feature,
    const std::vector<bool>& is_target_node, const LabelFiller& label_filler,
    const std::vector<typename LabelFiller::AccumulatorInitializer>&
        accumulator_initializers,
    ExampleIndex min_num_examples) {
  decision_tree::PerThreadCacheV2 cache;

  // Contains the condition conditional label statistics.
  using ExampleBucketSet = decision_tree::ExampleBucketSet<
      decision_tree::ExampleBucket<decision_tree::FeatureBooleanBucket,
                                   typename LabelFiller::LabelBucket>>;

  typename ExampleBucketSet::FeatureBucketType::Filler feature_filler(
      common.dataset->meta_data()
          .columns(feature)
          .boolean()
          .replacement_missing_value(),
      {});

  // ExampleBucket for each of the open nodes.
  std::vector<ExampleBucketSet> example_bucket_set_per_node;

  // Initialize the buckets.
  RETURN_IF_ERROR(InitializeCategoricalFeatureBuckets<>(
      common, is_target_node, /*num_feature_values=*/2, feature_filler,
      label_filler, &example_bucket_set_per_node));

  // Aggregate the label values per bucket and open node.
  RETURN_IF_ERROR(FillBooleanFeatureBuckets(common, feature, is_target_node,
                                            label_filler,
                                            &example_bucket_set_per_node));

  return InOrderRegressionAndBooleanFeatureBuckets<>(
      common, feature, is_target_node, label_filler, accumulator_initializers,
      min_num_examples, example_bucket_set_per_node, feature_filler, &cache);
}

// Splitter for a categorical features and a regression label.
template <typename LabelFiller>
absl::Status TemplatedFindBestSplitsWithRegressionAndCategoricalFeature(
    const FindBestSplitsCommonArgs& common, FeatureIndex feature,
    const std::vector<bool>& is_target_node, const LabelFiller& label_filler,
    const std::vector<typename LabelFiller::AccumulatorInitializer>&
        accumulator_initializers,
    ExampleIndex min_num_examples) {
  decision_tree::PerThreadCacheV2 cache;

  const int num_feature_values =
      common.dataset->meta_data().columns(feature).categorical().num_values();

  // Contains the condition conditional label statistics.
  using ExampleBucketSet = decision_tree::ExampleBucketSet<
      decision_tree::ExampleBucket<decision_tree::FeatureCategoricalBucket,
                                   typename LabelFiller::LabelBucket>>;

  typename ExampleBucketSet::FeatureBucketType::Filler feature_filler(
      num_feature_values,
      common.dataset->meta_data()
          .columns(feature)
          .categorical()
          .replacement_missing_value(),
      {});

  // ExampleBucket for each of the open nodes.
  std::vector<ExampleBucketSet> example_bucket_set_per_node;

  // Initialize the buckets.
  RETURN_IF_ERROR(InitializeCategoricalFeatureBuckets<>(
      common, is_target_node, num_feature_values, feature_filler, label_filler,
      &example_bucket_set_per_node));

  // Aggregate the label values per bucket and open node.
  RETURN_IF_ERROR(FillCategoricalFeatureBuckets(
      common, feature, is_target_node, label_filler, num_feature_values,
      &example_bucket_set_per_node));

  return InOrderRegressionAndCategoricalFeatureBuckets<>(
      common, feature, is_target_node, label_filler, accumulator_initializers,
      min_num_examples, num_feature_values, example_bucket_set_per_node,
      feature_filler, &cache);
}

// Splitter for a boolean features and a regression label.
template <typename LabelFiller>
absl::Status TemplatedFindBestSplitsWithRegressionAndBooleanFeature(
    const FindBestSplitsCommonArgs& common, FeatureIndex feature,
    const std::vector<bool>& is_target_node, const LabelFiller& label_filler,
    const std::vector<typename LabelFiller::AccumulatorInitializer>&
        accumulator_initializers,
    ExampleIndex min_num_examples) {
  decision_tree::PerThreadCacheV2 cache;

  // Contains the condition conditional label statistics.
  using ExampleBucketSet = decision_tree::ExampleBucketSet<
      decision_tree::ExampleBucket<decision_tree::FeatureBooleanBucket,
                                   typename LabelFiller::LabelBucket>>;

  typename ExampleBucketSet::FeatureBucketType::Filler feature_filler(
      common.dataset->meta_data()
          .columns(feature)
          .boolean()
          .replacement_missing_value(),
      {});

  // ExampleBucket for each of the open nodes.
  std::vector<ExampleBucketSet> example_bucket_set_per_node;

  // Initialize the buckets.
  RETURN_IF_ERROR(InitializeCategoricalFeatureBuckets<>(
      common, is_target_node, /*num_feature_values=*/2, feature_filler,
      label_filler, &example_bucket_set_per_node));

  // Aggregate the label values per bucket and open node.
  RETURN_IF_ERROR(FillBooleanFeatureBuckets(common, feature, is_target_node,
                                            label_filler,
                                            &example_bucket_set_per_node));

  return InOrderRegressionAndBooleanFeatureBuckets<>(
      common, feature, is_target_node, label_filler, accumulator_initializers,
      min_num_examples, example_bucket_set_per_node, feature_filler, &cache);
}

}  // namespace distributed_decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_DECISION_TREE_SPLITTER_H_
