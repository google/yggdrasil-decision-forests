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

// Templated methods for the splitter i.e. to find the best split.
//
// Naming convention:
//   selected_examples: Indices of the training examples.
//   ExampleBucket (or bucket) : Accumulation of information for a set of
//     examples. Composed of a FeatureBucket (accumulate the information about
//     the feature) and the LabelBucket (accumulate the information about the
//     label). In some algorithms, the bucket may only contain one example. The
//     examples in a bucket are not separable according to the split being
//     searched i.e. for a categorical feature, all the examples of the same
//     feature value will end in the same bucket.
//   ExampleBucketSet: A set of ExampleBuckets. A ExampleBucketSet is
//     constructed by scanning the training examples, assigning each example to
//     one of the buckets, and accumulating the information about this example
//     (feature and label) to its ExampleBucket. The split is made to optimize
//     the separation of buckets in a bucket-set.
//   score accumulator: Accumulates the statistics from multiple buckets, and
//     propose a split and a split score.
//
// Finding a split is done by calling "FindBestSplit" with the label and feature
// bucket corresponding to the label and feature.
//
// The FindBestSplit algorithm works as follows:
//   - Allocate the buckets.
//   - Iterate over the training examples and fill the buckets.
//   - Optionally, reorder the buckets.
//   - Iterate over the buckets and fill update the score accumulator. At each
//     step, evaluate the score of the split.
//
// If SIMPLE_ML_DEBUG_DECISION_TREE_SPLITTER is set, the algorithm will log the
// detail in the splitter work.
//
#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_SPLITTER_SCANNER_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_SPLITTER_SCANNER_H_

#include <stddef.h>

#include <algorithm>
#include <functional>
#include <iterator>
#include <limits>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/decision_tree/splitter_accumulator.h"
#include "yggdrasil_decision_forests/learner/decision_tree/splitter_structure.h"
#include "yggdrasil_decision_forests/learner/decision_tree/utils.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/random.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {

// TODO(gbm): Explain the expected signature of FeatureBucket and LabelBucket.
template <typename FeatureBucket, typename LabelBucket>
struct ExampleBucket {
  FeatureBucket feature;
  LabelBucket label;

  using FeatureBucketType = FeatureBucket;
  using LabelBucketType = LabelBucket;

  struct SortFeature {
    bool operator()(const ExampleBucket& a, const ExampleBucket& b) {
      return a.feature < b.feature;
    }
  };

  struct SortLabel {
    bool operator()(const ExampleBucket& a, const ExampleBucket& b) {
      return a.label < b.label;
    }
  };
};

template <typename ExampleBucket>
struct ExampleBucketSet {
  std::vector<ExampleBucket> items;

  using ExampleBucketType = ExampleBucket;
  using FeatureBucketType = typename ExampleBucket::FeatureBucketType;
  using LabelBucketType = typename ExampleBucket::LabelBucketType;
};

// Used bucket sets.

// Label: Numerical.

using FeatureNumericalLabelNumericalOneValue = ExampleBucketSet<
    ExampleBucket<FeatureNumericalBucket, LabelNumericalOneValueBucket>>;

using FeatureDiscretizedNumericalLabelNumerical = ExampleBucketSet<
    ExampleBucket<FeatureDiscretizedNumericalBucket, LabelNumericalBucket>>;

using FeatureCategoricalLabelNumerical = ExampleBucketSet<
    ExampleBucket<FeatureCategoricalBucket, LabelNumericalBucket>>;

using FeatureBooleanLabelNumerical =
    ExampleBucketSet<ExampleBucket<FeatureBooleanBucket, LabelNumericalBucket>>;

using FeatureIsMissingLabelNumerical = ExampleBucketSet<
    ExampleBucket<FeatureIsMissingBucket, LabelNumericalBucket>>;

// Label: Hessian Numerical.

using FeatureNumericalLabelHessianNumericalOneValue = ExampleBucketSet<
    ExampleBucket<FeatureNumericalBucket, LabelHessianNumericalOneValueBucket>>;

using FeatureDiscretizedNumericalLabelHessianNumerical =
    ExampleBucketSet<ExampleBucket<FeatureDiscretizedNumericalBucket,
                                   LabelHessianNumericalBucket>>;

using FeatureCategoricalLabelHessianNumerical = ExampleBucketSet<
    ExampleBucket<FeatureCategoricalBucket, LabelHessianNumericalBucket>>;

using FeatureBooleanLabelHessianNumerical = ExampleBucketSet<
    ExampleBucket<FeatureBooleanBucket, LabelHessianNumericalBucket>>;

using FeatureIsMissingLabelHessianNumerical = ExampleBucketSet<
    ExampleBucket<FeatureIsMissingBucket, LabelHessianNumericalBucket>>;

// Label: Categorical.

using FeatureNumericalLabelCategoricalOneValue = ExampleBucketSet<
    ExampleBucket<FeatureNumericalBucket, LabelCategoricalOneValueBucket>>;

using FeatureDiscretizedNumericalLabelCategorical = ExampleBucketSet<
    ExampleBucket<FeatureDiscretizedNumericalBucket, LabelCategoricalBucket>>;

using FeatureCategoricalLabelCategorical = ExampleBucketSet<
    ExampleBucket<FeatureCategoricalBucket, LabelCategoricalBucket>>;

using FeatureBooleanLabelCategorical = ExampleBucketSet<
    ExampleBucket<FeatureBooleanBucket, LabelCategoricalBucket>>;

using FeatureIsMissingLabelCategorical = ExampleBucketSet<
    ExampleBucket<FeatureIsMissingBucket, LabelCategoricalBucket>>;

// Label: Binary Categorical.

using FeatureNumericalLabelBinaryCategoricalOneValue =
    ExampleBucketSet<ExampleBucket<FeatureNumericalBucket,
                                   LabelBinaryCategoricalOneValueBucket>>;

using FeatureDiscretizedNumericalLabelBinaryCategorical =
    ExampleBucketSet<ExampleBucket<FeatureDiscretizedNumericalBucket,
                                   LabelBinaryCategoricalBucket>>;

using FeatureCategoricalLabelBinaryCategorical = ExampleBucketSet<
    ExampleBucket<FeatureCategoricalBucket, LabelBinaryCategoricalBucket>>;

using FeatureBooleanLabelBinaryCategorical = ExampleBucketSet<
    ExampleBucket<FeatureBooleanBucket, LabelBinaryCategoricalBucket>>;

using FeatureIsMissingLabelBinaryCategorical = ExampleBucketSet<
    ExampleBucket<FeatureIsMissingBucket, LabelBinaryCategoricalBucket>>;

// Label: Unweighted Binary Categorical.

using FeatureNumericalLabelUnweightedBinaryCategoricalOneValue =
    ExampleBucketSet<
        ExampleBucket<FeatureNumericalBucket,
                      LabelUnweightedBinaryCategoricalOneValueBucket>>;

using FeatureDiscretizedNumericalLabelUnweightedBinaryCategorical =
    ExampleBucketSet<ExampleBucket<FeatureDiscretizedNumericalBucket,
                                   LabelUnweightedBinaryCategoricalBucket>>;

using FeatureCategoricalLabelUnweightedBinaryCategorical =
    ExampleBucketSet<ExampleBucket<FeatureCategoricalBucket,
                                   LabelUnweightedBinaryCategoricalBucket>>;

using FeatureBooleanLabelUnweightedBinaryCategorical =
    ExampleBucketSet<ExampleBucket<FeatureBooleanBucket,
                                   LabelUnweightedBinaryCategoricalBucket>>;

using FeatureIsMissingLabelUnweightedBinaryCategorical =
    ExampleBucketSet<ExampleBucket<FeatureIsMissingBucket,
                                   LabelUnweightedBinaryCategoricalBucket>>;

// Label: Uplift categorical.

using FeatureNumericalLabelUpliftCategoricalOneValue =
    ExampleBucketSet<ExampleBucket<FeatureNumericalBucket,
                                   LabelUpliftCategoricalOneValueBucket>>;

using FeatureCategoricalLabelUpliftCategorical = ExampleBucketSet<
    ExampleBucket<FeatureCategoricalBucket, LabelUpliftCategoricalBucket>>;

// Label: Uplift numerical.

using FeatureNumericalLabelUpliftNumericalOneValue = ExampleBucketSet<
    ExampleBucket<FeatureNumericalBucket, LabelUpliftNumericalOneValueBucket>>;

using FeatureCategoricalLabelUpliftNumerical = ExampleBucketSet<
    ExampleBucket<FeatureCategoricalBucket, LabelUpliftNumericalBucket>>;

// Memory cache for the splitter.
//
// Used to avoid re-allocating memory each time the splitter is called.
struct PerThreadCacheV2 {
  // Cache for example bucket sets.
  // The postfix digit is only used to differentiate between the objects. There
  // is not special semantic to it.

  FeatureNumericalLabelNumericalOneValue example_bucket_set_num_1;
  FeatureDiscretizedNumericalLabelNumerical example_bucket_set_num_5;
  FeatureCategoricalLabelNumerical example_bucket_set_num_2;
  FeatureIsMissingLabelNumerical example_bucket_set_num_3;
  FeatureBooleanLabelNumerical example_bucket_set_num_4;

  FeatureNumericalLabelCategoricalOneValue example_bucket_set_cat_1;
  FeatureDiscretizedNumericalLabelCategorical example_bucket_set_cat_5;
  FeatureCategoricalLabelCategorical example_bucket_set_cat_2;
  FeatureIsMissingLabelCategorical example_bucket_set_cat_3;
  FeatureBooleanLabelCategorical example_bucket_set_cat_4;

  FeatureNumericalLabelHessianNumericalOneValue example_bucket_set_hnum_1;
  FeatureDiscretizedNumericalLabelHessianNumerical example_bucket_set_hnum_5;
  FeatureCategoricalLabelHessianNumerical example_bucket_set_hnum_2;
  FeatureIsMissingLabelHessianNumerical example_bucket_set_hnum_3;
  FeatureBooleanLabelHessianNumerical example_bucket_set_hnum_4;

  FeatureNumericalLabelBinaryCategoricalOneValue example_bucket_set_bcat_1;
  FeatureDiscretizedNumericalLabelBinaryCategorical example_bucket_set_bcat_5;
  FeatureCategoricalLabelBinaryCategorical example_bucket_set_bcat_2;
  FeatureIsMissingLabelBinaryCategorical example_bucket_set_bcat_3;
  FeatureBooleanLabelBinaryCategorical example_bucket_set_bcat_4;

  FeatureNumericalLabelUnweightedBinaryCategoricalOneValue
      example_bucket_set_ubcat_1;
  FeatureDiscretizedNumericalLabelUnweightedBinaryCategorical
      example_bucket_set_ubcat_5;
  FeatureCategoricalLabelUnweightedBinaryCategorical example_bucket_set_ubcat_2;
  FeatureIsMissingLabelUnweightedBinaryCategorical example_bucket_set_ubcat_3;
  FeatureBooleanLabelUnweightedBinaryCategorical example_bucket_set_ubcat_4;

  FeatureNumericalLabelUpliftCategoricalOneValue example_bucket_set_ul_1;
  FeatureCategoricalLabelUpliftCategorical example_bucket_set_ul_2;

  FeatureNumericalLabelUpliftNumericalOneValue example_bucket_set_ul_n_1;
  FeatureCategoricalLabelUpliftNumerical example_bucket_set_ul_n_2;

  // Cache for the label score accumulator;
  LabelNumericalScoreAccumulator label_numerical_score_accumulator[2];
  LabelCategoricalScoreAccumulator label_categorical_score_accumulator[2];
  LabelHessianNumericalScoreAccumulator
      label_hessian_numerical_score_accumulator[2];
  LabelBinaryCategoricalScoreAccumulator
      label_binary_categorical_score_accumulator[2];
  LabelNumericalWithHessianScoreAccumulator
      label_numerical_with_hessian_score_accumulator[2];
  LabelUpliftCategoricalScoreAccumulator
      label_uplift_categorical_score_accumulator[2];
  LabelUpliftNumericalScoreAccumulator
      label_uplift_numerical_score_accumulator[2];

  std::vector<std::pair<float, int32_t>> bucket_order;

  // Mask of selected examples.
  std::vector<bool> selected_examples_mask;
  std::vector<uint8_t> selected_examples_count;

  // Selected bucket indices;
  std::vector<int> buckets_1;
  std::vector<int> buckets_2;

  // Selected categorical attribute values;
  std::vector<int> categorical_attribute;
};

// Get the example bucket set from the thread cache.
template <typename ExampleBucketSet>
auto* GetCachedExampleBucketSet(PerThreadCacheV2* cache) {
  using utils::is_same_v;
  // Numerical.
  if constexpr (is_same_v<ExampleBucketSet,
                          FeatureNumericalLabelNumericalOneValue>) {
    return &cache->example_bucket_set_num_1;
  } else if constexpr (is_same_v<ExampleBucketSet,
                                 FeatureDiscretizedNumericalLabelNumerical>) {
    return &cache->example_bucket_set_num_5;
  } else if constexpr (is_same_v<ExampleBucketSet,
                                 FeatureCategoricalLabelNumerical>) {
    return &cache->example_bucket_set_num_2;
  } else if constexpr (is_same_v<ExampleBucketSet,
                                 FeatureIsMissingLabelNumerical>) {
    return &cache->example_bucket_set_num_3;
  } else if constexpr (is_same_v<ExampleBucketSet,
                                 FeatureBooleanLabelNumerical>) {
    return &cache->example_bucket_set_num_4;
  } else if constexpr (is_same_v<
                           ExampleBucketSet,
                           FeatureNumericalLabelHessianNumericalOneValue>) {
    // Hessian Numerical.
    return &cache->example_bucket_set_hnum_1;
  } else if constexpr (is_same_v<
                           ExampleBucketSet,
                           FeatureDiscretizedNumericalLabelHessianNumerical>) {
    return &cache->example_bucket_set_hnum_5;
  } else if constexpr (is_same_v<ExampleBucketSet,
                                 FeatureCategoricalLabelHessianNumerical>) {
    return &cache->example_bucket_set_hnum_2;
  } else if constexpr (is_same_v<ExampleBucketSet,
                                 FeatureIsMissingLabelHessianNumerical>) {
    return &cache->example_bucket_set_hnum_3;
  } else if constexpr (is_same_v<ExampleBucketSet,
                                 FeatureBooleanLabelHessianNumerical>) {
    return &cache->example_bucket_set_hnum_4;
  } else if constexpr (is_same_v<ExampleBucketSet,
                                 FeatureNumericalLabelCategoricalOneValue>) {
    // Categorical.
    return &cache->example_bucket_set_cat_1;
  } else if constexpr (is_same_v<ExampleBucketSet,
                                 FeatureDiscretizedNumericalLabelCategorical>) {
    return &cache->example_bucket_set_cat_5;
  } else if constexpr (is_same_v<ExampleBucketSet,
                                 FeatureCategoricalLabelCategorical>) {
    return &cache->example_bucket_set_cat_2;
  } else if constexpr (is_same_v<ExampleBucketSet,
                                 FeatureIsMissingLabelCategorical>) {
    return &cache->example_bucket_set_cat_3;
  } else if constexpr (is_same_v<ExampleBucketSet,
                                 FeatureBooleanLabelCategorical>) {
    return &cache->example_bucket_set_cat_4;
  } else if constexpr (is_same_v<
                           ExampleBucketSet,
                           FeatureNumericalLabelBinaryCategoricalOneValue>) {
    // Binary Categorical.
    return &cache->example_bucket_set_bcat_1;
  } else if constexpr (is_same_v<
                           ExampleBucketSet,
                           FeatureDiscretizedNumericalLabelBinaryCategorical>) {
    return &cache->example_bucket_set_bcat_5;
  } else if constexpr (is_same_v<ExampleBucketSet,
                                 FeatureCategoricalLabelBinaryCategorical>) {
    return &cache->example_bucket_set_bcat_2;
  } else if constexpr (is_same_v<ExampleBucketSet,
                                 FeatureIsMissingLabelBinaryCategorical>) {
    return &cache->example_bucket_set_bcat_3;
  } else if constexpr (is_same_v<ExampleBucketSet,
                                 FeatureBooleanLabelBinaryCategorical>) {
    return &cache->example_bucket_set_bcat_4;
  } else if constexpr (
      is_same_v<ExampleBucketSet,
                FeatureNumericalLabelUnweightedBinaryCategoricalOneValue>) {
    // Unweighted Binary Categorical.
    return &cache->example_bucket_set_ubcat_1;
  } else if constexpr (
      is_same_v<ExampleBucketSet,
                FeatureDiscretizedNumericalLabelUnweightedBinaryCategorical>) {
    return &cache->example_bucket_set_ubcat_5;
  } else if constexpr (
      is_same_v<ExampleBucketSet,
                FeatureCategoricalLabelUnweightedBinaryCategorical>) {
    return &cache->example_bucket_set_ubcat_2;
  } else if constexpr (is_same_v<
                           ExampleBucketSet,
                           FeatureIsMissingLabelUnweightedBinaryCategorical>) {
    return &cache->example_bucket_set_ubcat_3;
  } else if constexpr (is_same_v<
                           ExampleBucketSet,
                           FeatureBooleanLabelUnweightedBinaryCategorical>) {
    return &cache->example_bucket_set_ubcat_4;
  } else if constexpr (is_same_v<
                           ExampleBucketSet,
                           FeatureNumericalLabelUpliftCategoricalOneValue>) {
    // Uplift categorical
    return &cache->example_bucket_set_ul_1;
  } else if constexpr (is_same_v<ExampleBucketSet,
                                 FeatureCategoricalLabelUpliftCategorical>) {
    return &cache->example_bucket_set_ul_2;
  } else if constexpr (is_same_v<
                           ExampleBucketSet,
                           FeatureNumericalLabelUpliftNumericalOneValue>) {
    // Uplift numerical
    return &cache->example_bucket_set_ul_n_1;
  } else if constexpr (is_same_v<ExampleBucketSet,
                                 FeatureCategoricalLabelUpliftNumerical>) {
    return &cache->example_bucket_set_ul_n_2;
  } else {
    static_assert(!is_same_v<ExampleBucketSet, ExampleBucketSet>,
                  "Not implemented.");
  }
}

// Get the label score accumulator from the cache.
template <typename LabelScoreAccumulator>
auto* GetCachedLabelScoreAccumulator(const bool side, PerThreadCacheV2* cache) {
  using utils::is_same_v;
  if constexpr (is_same_v<LabelScoreAccumulator,
                          LabelNumericalScoreAccumulator>) {
    return &cache->label_numerical_score_accumulator[side];
  } else if constexpr (is_same_v<LabelScoreAccumulator,
                                 LabelCategoricalScoreAccumulator>) {
    return &cache->label_categorical_score_accumulator[side];
  } else if constexpr (is_same_v<LabelScoreAccumulator,
                                 LabelBinaryCategoricalScoreAccumulator>) {
    return &cache->label_binary_categorical_score_accumulator[side];
  } else if constexpr (is_same_v<LabelScoreAccumulator,
                                 LabelHessianNumericalScoreAccumulator>) {
    return &cache->label_hessian_numerical_score_accumulator[side];
  } else if constexpr (is_same_v<LabelScoreAccumulator,
                                 LabelNumericalWithHessianScoreAccumulator>) {
    return &cache->label_numerical_with_hessian_score_accumulator[side];
  } else if constexpr (is_same_v<LabelScoreAccumulator,
                                 LabelUpliftCategoricalScoreAccumulator>) {
    return &cache->label_uplift_categorical_score_accumulator[side];
  } else if constexpr (is_same_v<LabelScoreAccumulator,
                                 LabelUpliftNumericalScoreAccumulator>) {
    return &cache->label_uplift_numerical_score_accumulator[side];
  } else {
    static_assert(!is_same_v<LabelScoreAccumulator, LabelScoreAccumulator>,
                  "Not implemented.");
  }
}

template <typename ExampleBucketSet, bool require_label_sorting>
void FillExampleBucketSet(
    const std::vector<row_t>& selected_examples,
    const typename ExampleBucketSet::FeatureBucketType::Filler& feature_filler,
    const typename ExampleBucketSet::LabelBucketType::Filler& label_filler,
    ExampleBucketSet* example_bucket_set, PerThreadCacheV2* cache) {
  // Allocate the buckets.
  example_bucket_set->items.resize(feature_filler.NumBuckets());

  // Initialize the buckets.
  int bucket_idx = 0;
  for (auto& bucket : example_bucket_set->items) {
    feature_filler.InitializeAndZero(bucket_idx, &bucket.feature);
    label_filler.InitializeAndZero(&bucket.label);
    bucket_idx++;
  }

  // Fill the buckets.
  const auto num_selected_examples = selected_examples.size();
  for (size_t select_idx = 0; select_idx < num_selected_examples;
       select_idx++) {
    const row_t example_idx = selected_examples[select_idx];
    const size_t item_idx =
        feature_filler.GetBucketIndex(select_idx, example_idx);
    auto& bucket = example_bucket_set->items[item_idx];
    feature_filler.ConsumeExample(example_idx, &bucket.feature);
    label_filler.ConsumeExample(example_idx, &bucket.label);
  }

  // Finalize the buckets.
  for (auto& bucket : example_bucket_set->items) {
    label_filler.Finalize(&bucket.label);
  }

  //  Sort the buckets.
  static_assert(!(ExampleBucketSet::FeatureBucketType::kRequireSorting &&
                  require_label_sorting),
                "Bucket require sorting");

  if constexpr (ExampleBucketSet::FeatureBucketType::kRequireSorting) {
    std::sort(example_bucket_set->items.begin(),
              example_bucket_set->items.end(),
              typename ExampleBucketSet::ExampleBucketType::SortFeature());
  }

  if constexpr (require_label_sorting) {
    std::sort(example_bucket_set->items.begin(),
              example_bucket_set->items.end(),
              typename ExampleBucketSet::ExampleBucketType::SortLabel());
  }
}

template <typename LabelScoreAccumulator, typename Initializer>
ABSL_ATTRIBUTE_ALWAYS_INLINE double Score(const Initializer& initializer,
                                          const double weighted_num_examples,
                                          const LabelScoreAccumulator& pos,
                                          const LabelScoreAccumulator& neg) {
  const double score_neg = neg.Score();
  const double score_pos = pos.Score();

  if constexpr (LabelScoreAccumulator::kNormalizeByWeight) {
    const double ratio_pos = pos.WeightedNumExamples() / weighted_num_examples;
    return initializer.NormalizeScore(score_pos * ratio_pos +
                                      score_neg * (1. - ratio_pos));
  } else {
    return initializer.NormalizeScore(score_pos + score_neg);
  }
}

// Scans the buckets iteratively. At each iteration evaluate the split that
// could put all the already visited buckets in the negative branch, and the non
// visited buckets in the positive branch.
template <typename ExampleBucketSet, typename LabelScoreAccumulator,
          bool bucket_interpolation = false>
SplitSearchResult ScanSplits(
    const typename ExampleBucketSet::FeatureBucketType::Filler& feature_filler,
    const typename ExampleBucketSet::LabelBucketType::Initializer& initializer,
    const ExampleBucketSet& example_bucket_set, const int64_t num_examples,
    const int min_num_obs, const int attribute_idx,
    proto::NodeCondition* condition, PerThreadCacheV2* cache) {
  using FeatureBucketType = typename ExampleBucketSet::FeatureBucketType;

  if (example_bucket_set.items.size() <= 1) {
    return SplitSearchResult::kInvalidAttribute;
  }

  if (!FeatureBucketType::IsValidAttribute(
          example_bucket_set.items.front().feature,
          example_bucket_set.items.back().feature)) {
    return SplitSearchResult::kInvalidAttribute;
  }

  // Initialize the accumulators.
  // Initially, all the buckets are in the positive accumulators.
  LabelScoreAccumulator& neg =
      *GetCachedLabelScoreAccumulator<LabelScoreAccumulator>(false, cache);
  LabelScoreAccumulator& pos =
      *GetCachedLabelScoreAccumulator<LabelScoreAccumulator>(true, cache);

  initializer.InitEmpty(&neg);
  initializer.InitFull(&pos);

  // Running statistics.
  int64_t num_pos_examples = num_examples;
  int64_t num_neg_examples = 0;
  bool tried_one_split = false;

  const double weighted_num_examples = pos.WeightedNumExamples();
  const int end_bucket_idx = example_bucket_set.items.size() - 1;

  double best_score =
      std::max<double>(condition->split_score(), initializer.MinimumScore());
  int best_bucket_idx = -1;
  int best_bucket_interpolation_idx = -1;

  // If true, a new best split was found ("best_bucket_idx" was set accordingly)
  // but no new examples were observed (i.e. all the bucket visited since the
  // last new best split were empty).
  bool no_new_examples_since_last_new_best_split = false;

  for (int bucket_idx = 0; bucket_idx < end_bucket_idx; bucket_idx++) {
    const auto& item = example_bucket_set.items[bucket_idx];

    if constexpr (bucket_interpolation) {
      if (no_new_examples_since_last_new_best_split && item.label.count > 0) {
        best_bucket_interpolation_idx = bucket_idx;
        no_new_examples_since_last_new_best_split = false;
      }
    }

    // Remove the bucket from the positive accumulator and add it to the
    // negative accumulator.
    item.label.AddToScoreAcc(&neg);
    item.label.SubToScoreAcc(&pos);

    num_pos_examples -= item.label.count;
    num_neg_examples += item.label.count;

    if (!FeatureBucketType::IsValidSplit(
            item.feature, example_bucket_set.items[bucket_idx + 1].feature)) {
      continue;
    }

    // Enough examples?
    if (num_pos_examples < min_num_obs) {
      break;
    }

    if (num_neg_examples < min_num_obs) {
      continue;
    }

    if (!initializer.IsValidSplit(neg, pos)) {
      continue;
    }

    const auto score = Score<>(initializer, weighted_num_examples, pos, neg);
    tried_one_split = true;

    if (score > best_score) {
      // Memorize the split.
      best_bucket_idx = bucket_idx;
      best_score = score;
      condition->set_num_pos_training_examples_without_weight(num_pos_examples);
      condition->set_num_pos_training_examples_with_weight(
          pos.WeightedNumExamples());
      if constexpr (bucket_interpolation) {
        no_new_examples_since_last_new_best_split = true;
        best_bucket_interpolation_idx = -1;
      }
    }
  }

  if (best_bucket_idx != -1) {
    // Finalize the best found split.

    if constexpr (bucket_interpolation) {
      if (best_bucket_interpolation_idx != -1 &&
          best_bucket_interpolation_idx != best_bucket_idx + 1) {
        // Bucket interpolation.
        feature_filler.SetConditionInterpolatedFinal(
            example_bucket_set, best_bucket_idx, best_bucket_interpolation_idx,
            condition);
      } else {
        feature_filler.SetConditionFinal(example_bucket_set, best_bucket_idx,
                                         condition);
      }
    } else {
      feature_filler.SetConditionFinal(example_bucket_set, best_bucket_idx,
                                       condition);
    }

    condition->set_attribute(attribute_idx);
    condition->set_num_training_examples_without_weight(num_examples);
    condition->set_num_training_examples_with_weight(weighted_num_examples);
    condition->set_split_score(best_score);
    return SplitSearchResult::kBetterSplitFound;
  } else {
    return tried_one_split ? SplitSearchResult::kNoBetterSplitFound
                           : SplitSearchResult::kInvalidAttribute;
  }
}

// Scans the buckets (similarly to "ScanSplits"), but in the order specified by
// "bucket_order[i].second" (instead of the bucket order).
template <typename ExampleBucketSet, typename LabelScoreAccumulator,
          typename Initializer =
              typename ExampleBucketSet::LabelBucketType::Initializer>
SplitSearchResult ScanSplitsCustomOrder(
    const std::vector<std::pair<float, int32_t>>& bucket_order,
    const typename ExampleBucketSet::FeatureBucketType::Filler& feature_filler,
    const Initializer& initializer, const ExampleBucketSet& example_bucket_set,
    const int64_t num_examples, const int min_num_obs, const int attribute_idx,
    proto::NodeCondition* condition, PerThreadCacheV2* cache) {
  using FeatureBucketType = typename ExampleBucketSet::FeatureBucketType;

  if (example_bucket_set.items.size() <= 1) {
    return SplitSearchResult::kInvalidAttribute;
  }

  if (!FeatureBucketType::IsValidAttribute(
          example_bucket_set.items.front().feature,
          example_bucket_set.items.back().feature)) {
    return SplitSearchResult::kInvalidAttribute;
  }

  // Initialize the accumulators.
  // Initially, all the buckets are in the positive accumulators.
  LabelScoreAccumulator& neg =
      *GetCachedLabelScoreAccumulator<LabelScoreAccumulator>(false, cache);
  LabelScoreAccumulator& pos =
      *GetCachedLabelScoreAccumulator<LabelScoreAccumulator>(true, cache);

  initializer.InitEmpty(&neg);
  initializer.InitFull(&pos);

  // Running statistics.
  int64_t num_pos_examples = num_examples;
  int64_t num_neg_examples = 0;
  bool tried_one_split = false;

  const double weighted_num_examples = pos.WeightedNumExamples();

  double best_score =
      std::max<double>(condition->split_score(), initializer.MinimumScore());

  int best_bucket_idx = -1;
  int best_order_idx = -1;

  const int end_order_idx = bucket_order.size() - 1;
  for (int order_idx = 0; order_idx < end_order_idx; order_idx++) {
    const auto bucket_idx = bucket_order[order_idx].second;
    const auto& item = example_bucket_set.items[bucket_idx];

    // Remove the bucket from the positive accumulator and add it to the
    // negative accumulator.
    item.label.AddToScoreAcc(&neg);
    item.label.SubToScoreAcc(&pos);

    num_pos_examples -= item.label.count;
    num_neg_examples += item.label.count;

    if (!FeatureBucketType::IsValidSplit(
            item.feature,
            example_bucket_set.items[bucket_order[order_idx + 1].second]
                .feature)) {
      continue;
    }

    // Enough examples?
    if (num_pos_examples < min_num_obs) {
      break;
    }

    if (num_neg_examples < min_num_obs) {
      continue;
    }

    // Compute the split's score.
    const auto score = Score<>(initializer, weighted_num_examples, pos, neg);
    tried_one_split = true;

    if (score > best_score) {
      // Memorize the split.
      best_bucket_idx = bucket_idx;
      best_order_idx = order_idx;
      best_score = score;
      condition->set_num_pos_training_examples_without_weight(num_pos_examples);
      condition->set_num_pos_training_examples_with_weight(
          pos.WeightedNumExamples());
    }
  }

  if (best_bucket_idx != -1) {
    // Finalize the best found split.
    feature_filler.SetConditionFinalWithOrder(bucket_order, example_bucket_set,
                                              best_order_idx, condition);

    condition->set_attribute(attribute_idx);
    condition->set_num_training_examples_without_weight(num_examples);
    condition->set_num_training_examples_with_weight(weighted_num_examples);
    condition->set_split_score(best_score);
    return SplitSearchResult::kBetterSplitFound;
  } else {
    return tried_one_split ? SplitSearchResult::kNoBetterSplitFound
                           : SplitSearchResult::kInvalidAttribute;
  }
}

// Scans the buckets (similarly to "ScanSplits") from a pre-sorted sparse
// collection of buckets. This method is used for features with ordering (e.g.
// numerical features) that are pre-sorted. When applicable, this method is
// equivalent (but more efficient) than the classical "ScanSplits" function
// (that would require extraction + sorting the buckets).
//
// Args:
//   - total_num_examples: Total number of examples in the training dataset.
//   - selected_examples: Index of the active training examples. All values
//     should be in [0,total_num_examples).
//   - sorted_attributes: List of sorted attribute values.
//   - feature_filler: Access to the raw feature data.
//   - label_filler: Access to the raw label data.
//   - min_num_obs: Minimum number of examples (non-weights) in each side of the
//     split.
//   - attribute_idx: Attribute being tested.
//   - condition: Input and output condition.
//   - cache: Utility cache data.
//   - duplicate_examples: If true, "selected_examples" can contain multiple
//     times the same example.
template <typename ExampleBucketSet, typename LabelScoreAccumulator,
          bool duplicate_examples = true>
SplitSearchResult ScanSplitsPresortedSparseDuplicateExampleTemplate(
    const dataset::VerticalDataset::row_t total_num_examples,
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<SparseItem>& sorted_attributes,
    const typename ExampleBucketSet::FeatureBucketType::Filler& feature_filler,
    const typename ExampleBucketSet::LabelBucketType::Filler& label_filler,
    const typename ExampleBucketSet::LabelBucketType::Initializer& initializer,
    const int min_num_obs, const int attribute_idx,
    proto::NodeCondition* condition, PerThreadCacheV2* cache) {
  if (selected_examples.size() <= 1) {
    return SplitSearchResult::kInvalidAttribute;
  }

  // Compute a mask (duplicate_examples=false) or count
  // (duplicate_examples=true) of the selected examples.
  auto get_mask = [&]() -> const auto& {
    if constexpr (duplicate_examples) {
      auto& selected_examples_mask = cache->selected_examples_count;
      selected_examples_mask.assign(total_num_examples, 0);
      for (const auto example_idx : selected_examples) {
        if (selected_examples_mask[example_idx] <
            std::numeric_limits<uint8_t>::max()) {
          selected_examples_mask[example_idx]++;
        }
      }
      return selected_examples_mask;
    } else {
      auto& selected_examples_mask = cache->selected_examples_mask;
      selected_examples_mask.assign(total_num_examples, false);
      for (const auto example_idx : selected_examples) {
        DCHECK(!selected_examples_mask[example_idx])
            << "Duplicated examples. Use duplicate_examples=true";
        selected_examples_mask[example_idx] = true;
      }
      return selected_examples_mask;
    }
  };
  const auto& selected_examples_mask = get_mask();

  // Initialize the accumulators. Initially, all the buckets are in the positive
  // accumulators.
  LabelScoreAccumulator& neg =
      *GetCachedLabelScoreAccumulator<LabelScoreAccumulator>(false, cache);
  LabelScoreAccumulator& pos =
      *GetCachedLabelScoreAccumulator<LabelScoreAccumulator>(true, cache);

  initializer.InitEmpty(&neg);
  initializer.InitFull(&pos);

  // Running statistics.
  int64_t num_pos_examples = selected_examples.size();
  int64_t max_num_pos_examples = selected_examples.size() - min_num_obs;

  // At least one split was tested.
  bool tried_one_split = false;
  // At least one better split was found.
  bool found_split = false;

  const double weighted_num_examples = pos.WeightedNumExamples();

  // Statistics of the best split found so far.
  double best_score =
      std::max<double>(condition->split_score(), initializer.MinimumScore());
  int64_t best_num_pos_training_examples_without_weight;
  int64_t best_num_pos_training_examples_with_weight;
  int64_t best_sorted_example_idx = -1;
  int64_t best_previous_sorted_example_idx = -1;

  constexpr auto new_value_mask = ((SparseItem::ExampleIdx)1)
                                  << (sizeof(SparseItem::ExampleIdx) * 8 - 1);
  constexpr auto example_idx_mask = new_value_mask - 1;

  // A new (i.e. different) attribute value was observed in the scan since the
  // last score test.
  bool new_attribute_value = false;

  // Index of the nearest previous example with a  value different from the
  // current example (i.e. the "sorted_example_idx" example).
  SparseItem::ExampleIdx previous_sorted_example_idx = 0;

  // Iterate over the attribute values in increasing order.
  // Note: For some reasons, the iterator for-loop is faster than the
  // for(auto:sorted_attributes) for loop (test on 10 different compiled
  // binaries).
  for (SparseItem::ExampleIdx sorted_example_idx = 0;
       sorted_example_idx < sorted_attributes.size(); sorted_example_idx++) {
    const auto& sorted_attribute = sorted_attributes[sorted_example_idx];

    auto example_idx =
        sorted_attribute.example_idx_and_extra & example_idx_mask;

    const bool is_new_value =
        sorted_attribute.example_idx_and_extra & new_value_mask;
    new_attribute_value |= is_new_value;

    // Skip non selected examples.
    if constexpr (duplicate_examples) {
      if (selected_examples_mask[example_idx] == 0) {
        continue;
      }
    } else {
      if (!selected_examples_mask[example_idx]) {
        continue;
      }
    }

    // Prefetch the label information needed at the end of the loop body.
    label_filler.Prefetch(example_idx);

    // Test Split
    if (new_attribute_value) {
      if (num_pos_examples >= min_num_obs &&
          num_pos_examples <= max_num_pos_examples) {
        // Compute the split's score.
        const auto score =
            Score<>(initializer, weighted_num_examples, pos, neg);
        tried_one_split = true;

        if (score > best_score) {
          // A better split was found. Memorize the split.
          best_sorted_example_idx = sorted_example_idx;
          best_previous_sorted_example_idx = previous_sorted_example_idx;
          best_score = score;
          best_num_pos_training_examples_without_weight = num_pos_examples;
          best_num_pos_training_examples_with_weight =
              pos.WeightedNumExamples();
          found_split = true;
        }
      }
      previous_sorted_example_idx = sorted_example_idx;
      new_attribute_value = 0;
    }

    // Update positive and negative accumulators.
    // Remove the bucket from the positive accumulator and add it to the
    // negative accumulator.
    if constexpr (duplicate_examples) {
      const int count = selected_examples_mask[example_idx];
      label_filler.AddDirectToScoreAccWithDuplicates(example_idx, count, &neg);
      label_filler.SubDirectToScoreAccWithDuplicates(example_idx, count, &pos);
      num_pos_examples -= count;
    } else {
      label_filler.AddDirectToScoreAcc(example_idx, &neg);
      label_filler.SubDirectToScoreAcc(example_idx, &pos);
      num_pos_examples--;
    }
  }

  if (found_split) {
    // Finalize the best found split.
    const auto best_previous_feature_value = feature_filler.GetValue(
        sorted_attributes[best_previous_sorted_example_idx]
            .example_idx_and_extra &
        example_idx_mask);
    const auto best_feature_value = feature_filler.GetValue(
        sorted_attributes[best_sorted_example_idx].example_idx_and_extra &
        example_idx_mask);
    // TODO(gbm): Experiment with random splits in ]best_previous_feature_value,
    // best_feature_value[.

    feature_filler.SetConditionFinalFromThresholds(
        best_previous_feature_value, best_feature_value, condition);
    condition->set_attribute(attribute_idx);
    condition->set_num_training_examples_without_weight(
        selected_examples.size());
    condition->set_num_training_examples_with_weight(weighted_num_examples);
    condition->set_split_score(best_score);

    condition->set_num_pos_training_examples_without_weight(
        best_num_pos_training_examples_without_weight);
    condition->set_num_pos_training_examples_with_weight(
        best_num_pos_training_examples_with_weight);

    return SplitSearchResult::kBetterSplitFound;
  } else {
    return tried_one_split ? SplitSearchResult::kNoBetterSplitFound
                           : SplitSearchResult::kInvalidAttribute;
  }
}

// Call "ScanSplitsPresortedSparseDuplicateExampleTemplate" after unfolding the
// "duplicate_examples" argument. See
// "ScanSplitsPresortedSparseDuplicateExampleTemplate" for the function
// documentation.
template <typename ExampleBucketSet, typename LabelScoreAccumulator>
SplitSearchResult ScanSplitsPresortedSparse(
    const dataset::VerticalDataset::row_t total_num_examples,
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<SparseItem>& sorted_attributes,
    const typename ExampleBucketSet::FeatureBucketType::Filler& feature_filler,
    const typename ExampleBucketSet::LabelBucketType::Filler& label_filler,
    const typename ExampleBucketSet::LabelBucketType::Initializer& initializer,
    const int min_num_obs, const int attribute_idx,
    const bool duplicate_examples, proto::NodeCondition* condition,
    PerThreadCacheV2* cache) {
  if (duplicate_examples) {
    return ScanSplitsPresortedSparseDuplicateExampleTemplate<
        ExampleBucketSet, LabelScoreAccumulator, true>(
        total_num_examples, selected_examples, sorted_attributes,
        feature_filler, label_filler, initializer, min_num_obs, attribute_idx,
        condition, cache);
  } else {
    return ScanSplitsPresortedSparseDuplicateExampleTemplate<
        ExampleBucketSet, LabelScoreAccumulator, false>(
        total_num_examples, selected_examples, sorted_attributes,
        feature_filler, label_filler, initializer, min_num_obs, attribute_idx,
        condition, cache);
  }
}

// Generates and evaluates random assignments of buckets to the positive or
// negative branches. Used to learn categorical splits of the form "value \in
// mask", where "mask" was selected from a randomly generates set of masks.
template <typename ExampleBucketSet, typename LabelScoreAccumulator>
SplitSearchResult ScanSplitsRandomBuckets(
    const typename ExampleBucketSet::FeatureBucketType::Filler& feature_filler,
    const typename ExampleBucketSet::LabelBucketType::Filler& label_filler,
    const typename ExampleBucketSet::LabelBucketType::Initializer& initializer,
    const ExampleBucketSet& example_bucket_set, const int64_t num_examples,
    const int min_num_obs, const int attribute_idx,
    const std::function<int(const int num_non_empty_buckets)>& num_trials_fn,
    proto::NodeCondition* condition, PerThreadCacheV2* cache,
    utils::RandomEngine* random) {
  using FeatureBucketType = typename ExampleBucketSet::FeatureBucketType;

  if (example_bucket_set.items.size() <= 1) {
    // Not enough examples.
    return SplitSearchResult::kInvalidAttribute;
  }

  if (!FeatureBucketType::IsValidAttribute(
          example_bucket_set.items.front().feature,
          example_bucket_set.items.back().feature)) {
    // Invalid bucket set.
    return SplitSearchResult::kInvalidAttribute;
  }

  // Initialize the accumulators.
  // Initially, all the buckets are in the positive accumulators.
  LabelScoreAccumulator& neg =
      *GetCachedLabelScoreAccumulator<LabelScoreAccumulator>(false, cache);
  LabelScoreAccumulator& pos =
      *GetCachedLabelScoreAccumulator<LabelScoreAccumulator>(true, cache);

  initializer.InitEmpty(&neg);
  initializer.InitFull(&pos);

  // Running statistics.
  int64_t num_pos_examples;
  int64_t num_neg_examples;
  bool tried_one_split = false;

  const double weighted_num_examples = pos.WeightedNumExamples();

  double best_score =
      std::max<double>(condition->split_score(), initializer.MinimumScore());
  std::vector<int>& best_pos_buckets = cache->buckets_1;
  std::vector<int>& pos_buckets = cache->buckets_2;
  std::vector<int>& active_bucket_idxs = cache->categorical_attribute;
  best_pos_buckets.clear();

  // List the non empty buckets.
  active_bucket_idxs.clear();
  const int n = example_bucket_set.items.size();
  for (int bucket_idx = 0; bucket_idx < n; bucket_idx++) {
    if (example_bucket_set.items[bucket_idx].label.count > 0) {
      active_bucket_idxs.push_back(bucket_idx);
    }
  }
  if (active_bucket_idxs.size() <= 1) {
    // All the examples have the same attribute value.
    return SplitSearchResult::kInvalidAttribute;
  }

  const auto num_trials = num_trials_fn(active_bucket_idxs.size());

  for (int trial_idx = 0; trial_idx < num_trials; trial_idx++) {
    pos_buckets.clear();
    num_pos_examples = 0;
    initializer.InitFull(&neg);
    initializer.InitEmpty(&pos);
    for (const int bucket_idx : active_bucket_idxs) {
      if (((*random)() & 1) == 0) {
        const auto& bucket = example_bucket_set.items[bucket_idx];
        num_pos_examples += bucket.label.count;
        bucket.label.SubToScoreAcc(&neg);
        bucket.label.AddToScoreAcc(&pos);
        pos_buckets.push_back(bucket_idx);
      }
    }
    num_neg_examples = num_examples - num_pos_examples;

    if (num_pos_examples < min_num_obs) {
      // Not enough examples in the positive branch.
      continue;
    }

    if (num_neg_examples < min_num_obs) {
      // Not enough examples in the negative branch.
      continue;
    }

    if (!initializer.IsValidSplit(neg, pos)) {
      continue;
    }

    DCHECK(!pos_buckets.empty());

    const auto score = Score<>(initializer, weighted_num_examples, pos, neg);
    tried_one_split = true;

    if (score > best_score) {
      // Better split found. Memorize it.
      best_pos_buckets = pos_buckets;
      best_score = score;
      condition->set_num_pos_training_examples_without_weight(num_pos_examples);
      condition->set_num_pos_training_examples_with_weight(
          pos.WeightedNumExamples());
    }
  }

  if (!best_pos_buckets.empty()) {
    // Finalize the best found split.
    // Note: The bucket are sorted by index i.e. best_pos_buckets[i] ==
    // example_bucket_set.items[i].feature.value.
    feature_filler.SetConditionFinalWithBuckets(best_pos_buckets, condition);
    condition->set_attribute(attribute_idx);
    condition->set_num_training_examples_without_weight(num_examples);
    condition->set_num_training_examples_with_weight(weighted_num_examples);
    condition->set_split_score(best_score);
    return SplitSearchResult::kBetterSplitFound;
  } else {
    return tried_one_split ? SplitSearchResult::kNoBetterSplitFound
                           : SplitSearchResult::kInvalidAttribute;
  }
}

// Find the best possible split (and update the condition accordingly) using
// a simple "scan" of the buckets.  See "ScanSplits".
template <typename ExampleBucketSet, typename LabelBucketSet,
          bool require_label_sorting, bool bucket_interpolation = false>
SplitSearchResult FindBestSplit(
    const std::vector<row_t>& selected_examples,
    const typename ExampleBucketSet::FeatureBucketType::Filler& feature_filler,
    const typename ExampleBucketSet::LabelBucketType::Filler& label_filler,
    const typename ExampleBucketSet::LabelBucketType::Initializer& initializer,
    const int min_num_obs, const int attribute_idx,
    proto::NodeCondition* condition, PerThreadCacheV2* cache) {
  DCHECK(condition != nullptr);

  // Create buckets.
  ExampleBucketSet& example_set_accumulator =
      *GetCachedExampleBucketSet<ExampleBucketSet>(cache);
  FillExampleBucketSet<ExampleBucketSet, require_label_sorting>(
      selected_examples, feature_filler, label_filler, &example_set_accumulator,
      cache);

  // Scan buckets.
  return ScanSplits<ExampleBucketSet, LabelBucketSet, bucket_interpolation>(
      feature_filler, initializer, example_set_accumulator,
      selected_examples.size(), min_num_obs, attribute_idx, condition, cache);
}

// Find the best possible split (and update the condition accordingly) using
// a random scan of the buckets.  See "ScanSplitsRandomBuckets".
template <typename ExampleBucketSet, typename LabelBucketSet>
SplitSearchResult FindBestSplitRandom(
    const std::vector<row_t>& selected_examples,
    const typename ExampleBucketSet::FeatureBucketType::Filler& feature_filler,
    const typename ExampleBucketSet::LabelBucketType::Filler& label_filler,
    const typename ExampleBucketSet::LabelBucketType::Initializer& initializer,
    const int min_num_obs, const int attribute_idx,
    const std::function<int(const int num_non_empty_buckets)>& num_trials_fn,
    proto::NodeCondition* condition, PerThreadCacheV2* cache,
    utils::RandomEngine* random) {
  DCHECK(condition != nullptr);

  // Create buckets.
  ExampleBucketSet& example_set_accumulator =
      *GetCachedExampleBucketSet<ExampleBucketSet>(cache);
  FillExampleBucketSet<ExampleBucketSet, /*require_label_sorting*/ false>(
      selected_examples, feature_filler, label_filler, &example_set_accumulator,
      cache);

  // Scan buckets.
  return ScanSplitsRandomBuckets<ExampleBucketSet, LabelBucketSet>(
      feature_filler, label_filler, initializer, example_set_accumulator,
      selected_examples.size(), min_num_obs, attribute_idx, num_trials_fn,
      condition, cache, random);
}

// Adds the content's of "src" label bucket to "dst"'s label bucket.
template <typename ExampleBucketSet>
void AddLabelBucket(const ExampleBucketSet& src, ExampleBucketSet* dst) {
  DCHECK_EQ(src.items.size(), dst->items.size());
  for (size_t item_idx = 0; item_idx < src.items.size(); item_idx++) {
    src.items[item_idx].label.AddToBucket(&dst->items[item_idx].label);
  }
}

// Pre-defined ExampleBucketSets

// Label: Regression.

constexpr auto FindBestSplit_LabelRegressionFeatureNumerical =
    FindBestSplit<FeatureNumericalLabelNumericalOneValue,
                  LabelNumericalScoreAccumulator,
                  /*require_label_sorting*/ false>;

constexpr auto FindBestSplit_LabelRegressionFeatureDiscretizedNumerical =
    FindBestSplit<FeatureDiscretizedNumericalLabelNumerical,
                  LabelNumericalScoreAccumulator,
                  /*require_label_sorting*/ false,
                  /*bucket_interpolation=*/true>;

constexpr auto FindBestSplit_LabelRegressionFeatureCategoricalCart =
    FindBestSplit<FeatureCategoricalLabelNumerical,
                  LabelNumericalScoreAccumulator,
                  /*require_label_sorting*/ true>;

constexpr auto FindBestSplit_LabelRegressionFeatureCategoricalRandom =
    FindBestSplitRandom<FeatureCategoricalLabelNumerical,
                        LabelNumericalScoreAccumulator>;

constexpr auto FindBestSplit_LabelRegressionFeatureBooleanCart =
    FindBestSplit<FeatureBooleanLabelNumerical, LabelNumericalScoreAccumulator,
                  /*require_label_sorting*/ false>;

constexpr auto FindBestSplit_LabelRegressionFeatureNACart =
    FindBestSplit<FeatureIsMissingLabelNumerical,
                  LabelNumericalScoreAccumulator,
                  /*require_label_sorting*/ false>;

// Label: Classification.

constexpr auto FindBestSplit_LabelClassificationFeatureNumerical =
    FindBestSplit<FeatureNumericalLabelCategoricalOneValue,
                  LabelCategoricalScoreAccumulator,
                  /*require_label_sorting*/ false>;

constexpr auto FindBestSplit_LabelClassificationFeatureDiscretizedNumerical =
    FindBestSplit<FeatureDiscretizedNumericalLabelCategorical,
                  LabelCategoricalScoreAccumulator,
                  /*require_label_sorting*/ false,
                  /*bucket_interpolation=*/true>;

constexpr auto FindBestSplit_LabelClassificationFeatureCategoricalCart =
    FindBestSplit<FeatureCategoricalLabelCategorical,
                  LabelCategoricalScoreAccumulator,
                  /*require_label_sorting*/ false>;

constexpr auto FindBestSplit_LabelClassificationFeatureBooleanCart =
    FindBestSplit<FeatureBooleanLabelCategorical,
                  LabelCategoricalScoreAccumulator,
                  /*require_label_sorting*/ false>;

constexpr auto FindBestSplit_LabelClassificationFeatureNACart =
    FindBestSplit<FeatureIsMissingLabelCategorical,
                  LabelCategoricalScoreAccumulator,
                  /*require_label_sorting*/ false>;

// Label: Binary Classification.

constexpr auto FindBestSplit_LabelBinaryClassificationFeatureNumerical =
    FindBestSplit<FeatureNumericalLabelBinaryCategoricalOneValue,
                  LabelBinaryCategoricalScoreAccumulator,
                  /*require_label_sorting*/ false>;

constexpr auto
    FindBestSplit_LabelBinaryClassificationFeatureDiscretizedNumerical =
        FindBestSplit<FeatureDiscretizedNumericalLabelBinaryCategorical,
                      LabelBinaryCategoricalScoreAccumulator,
                      /*require_label_sorting*/ false,
                      /*bucket_interpolation=*/true>;

constexpr auto FindBestSplit_LabelBinaryClassificationFeatureCategoricalCart =
    FindBestSplit<FeatureCategoricalLabelBinaryCategorical,
                  LabelBinaryCategoricalScoreAccumulator,
                  /*require_label_sorting*/ false>;

constexpr auto FindBestSplit_LabelBinaryClassificationFeatureBooleanCart =
    FindBestSplit<FeatureBooleanLabelBinaryCategorical,
                  LabelBinaryCategoricalScoreAccumulator,
                  /*require_label_sorting*/ false>;

constexpr auto FindBestSplit_LabelBinaryClassificationFeatureNACart =
    FindBestSplit<FeatureIsMissingLabelBinaryCategorical,
                  LabelBinaryCategoricalScoreAccumulator,
                  /*require_label_sorting*/ false>;

constexpr auto
    FindBestSplit_LabelUnweightedBinaryClassificationFeatureNumerical =
        FindBestSplit<FeatureNumericalLabelUnweightedBinaryCategoricalOneValue,
                      LabelBinaryCategoricalScoreAccumulator,
                      /*require_label_sorting*/ false>;

constexpr auto
    FindBestSplit_LabelUnweightedBinaryClassificationFeatureDiscretizedNumerical =
        FindBestSplit<
            FeatureDiscretizedNumericalLabelUnweightedBinaryCategorical,
            LabelBinaryCategoricalScoreAccumulator,
            /*require_label_sorting*/ false,
            /*bucket_interpolation=*/true>;

constexpr auto
    FindBestSplit_LabelUnweightedBinaryClassificationFeatureCategoricalCart =
        FindBestSplit<FeatureCategoricalLabelUnweightedBinaryCategorical,
                      LabelBinaryCategoricalScoreAccumulator,
                      /*require_label_sorting*/ false>;

constexpr auto
    FindBestSplit_LabelUnweightedBinaryClassificationFeatureBooleanCart =
        FindBestSplit<FeatureBooleanLabelUnweightedBinaryCategorical,
                      LabelBinaryCategoricalScoreAccumulator,
                      /*require_label_sorting*/ false>;

constexpr auto FindBestSplit_LabelUnweightedBinaryClassificationFeatureNACart =
    FindBestSplit<FeatureIsMissingLabelUnweightedBinaryCategorical,
                  LabelBinaryCategoricalScoreAccumulator,
                  /*require_label_sorting*/ false>;

// Label: Hessian Regression.

constexpr auto FindBestSplit_LabelHessianRegressionFeatureNumerical =
    FindBestSplit<FeatureNumericalLabelHessianNumericalOneValue,
                  LabelHessianNumericalScoreAccumulator,
                  /*require_label_sorting*/ false>;

constexpr auto FindBestSplit_LabelHessianRegressionFeatureDiscretizedNumerical =
    FindBestSplit<FeatureDiscretizedNumericalLabelHessianNumerical,
                  LabelHessianNumericalScoreAccumulator,
                  /*require_label_sorting*/ false,
                  /*bucket_interpolation=*/true>;

constexpr auto FindBestSplit_LabelHessianRegressionFeatureCategoricalCart =
    FindBestSplit<FeatureCategoricalLabelHessianNumerical,
                  LabelHessianNumericalScoreAccumulator,
                  /*require_label_sorting*/ true>;

constexpr auto FindBestSplit_LabelHessianRegressionFeatureCategoricalRandom =
    FindBestSplitRandom<FeatureCategoricalLabelHessianNumerical,
                        LabelHessianNumericalScoreAccumulator>;

constexpr auto FindBestSplit_LabelHessianRegressionFeatureBooleanCart =
    FindBestSplit<FeatureBooleanLabelHessianNumerical,
                  LabelHessianNumericalScoreAccumulator,
                  /*require_label_sorting*/ false>;

constexpr auto FindBestSplit_LabelHessianRegressionFeatureNACart =
    FindBestSplit<FeatureIsMissingLabelHessianNumerical,
                  LabelHessianNumericalScoreAccumulator,
                  /*require_label_sorting*/ false>;

// Label : Uplift categorical

constexpr auto FindBestSplit_LabelUpliftClassificationFeatureNumerical =
    FindBestSplit<FeatureNumericalLabelUpliftCategoricalOneValue,
                  LabelUpliftCategoricalScoreAccumulator,
                  /*require_label_sorting*/ false>;

constexpr auto FindBestSplit_LabelUpliftClassificationFeatureCategoricalCart =
    FindBestSplit<FeatureCategoricalLabelUpliftCategorical,
                  LabelUpliftCategoricalScoreAccumulator,
                  /*require_label_sorting*/ true>;

constexpr auto FindBestSplit_LabelUpliftClassificationFeatureCategoricalRandom =
    FindBestSplitRandom<FeatureCategoricalLabelUpliftCategorical,
                        LabelUpliftCategoricalScoreAccumulator>;

// Label : Uplift numerical

constexpr auto FindBestSplit_LabelUpliftNumericalFeatureNumerical =
    FindBestSplit<FeatureNumericalLabelUpliftNumericalOneValue,
                  LabelUpliftNumericalScoreAccumulator,
                  /*require_label_sorting*/ false>;

constexpr auto FindBestSplit_LabelUpliftNumericalFeatureCategoricalCart =
    FindBestSplit<FeatureCategoricalLabelUpliftNumerical,
                  LabelUpliftNumericalScoreAccumulator,
                  /*require_label_sorting*/ true>;

constexpr auto FindBestSplit_LabelUpliftNumericalFeatureCategoricalRandom =
    FindBestSplitRandom<FeatureCategoricalLabelUpliftNumerical,
                        LabelUpliftNumericalScoreAccumulator>;

}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_SPLITTER_SCANNER_H_
