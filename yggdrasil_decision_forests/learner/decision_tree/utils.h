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

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_UTILS_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_UTILS_H_

#include <stdlib.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {

// The outcome of a split search.
enum class SplitSearchResult {
  // A new split, better than the current best split, was found.
  kBetterSplitFound,
  // At least a split was found. None of the found splits are better than the
  // current best split.
  kNoBetterSplitFound,
  // No valid split found.
  kInvalidAttribute
};

// Sets the condition to be "contains condition" i.e.
// "is_value_attribute_i in X".
//
// A "Contains" condition tests the presence (or absence) of an example
// categorical attribute value in a set of values. This set, called "positive
// attribute set",  is stored as a sorted array of "positive" values or as a
// bitmap (which ever is the most memory efficient).
//
// Given a "positive attribute set" sorted as a vector, this function set the
// "positive attribute set" of the condition according the most efficient
// solution.
//
// "ratio_true_label_by_attr_value" is expected to be sorted in
// increasing value of positive label ratio i.e. "{x.first for x \in
// ratio_true_label_by_attr_value}" is increasing.
//
// More precisely, it sets the positive attribute set to be
// ratio_true_label_by_attr_value[i].second for i in [
// begin_positive_idx, ratio_true_label_by_attr_value.size()-1].
//
// Note: This function only sets the "positive attribute set" i.e. it does not
// set any of the other fields of a categorical condition (e.g. split_score).
//
void SetPositiveAttributeSetOfCategoricalContainsCondition(
    const std::vector<std::pair<float, int32_t>>&
        ratio_true_label_by_attr_value,
    int32_t begin_positive_idx, int32_t num_attribute_classes,
    proto::NodeCondition* condition);

// Sets the condition to be "contains_condition" or "contains_bitmap_condition"
// with the positive equals to "positive_attribute_value". "contains_condition"
// or "contains_bitmap_condition" is selected to minimize the size in memory of
// the condition.
void SetPositiveAttributeSetOfCategoricalContainsCondition(
    const std::vector<int32_t>& positive_attribute_value,
    int32_t num_attribute_classes, proto::NodeCondition* condition);

// Removes "l1" from the length of "value". Set to 0 is the "length/abs" of
// "value" is lower than "l1".
template <typename T1, typename T2>
T1 l1_threshold(const T1 value, const T2 l1) {
  if (l1 == static_cast<T2>(0)) {
    return value;
  }
  const T1 length =
      std::max(static_cast<T1>(0), std::abs(value) - static_cast<T1>(l1));
  if (value > 0) {
    return length;
  } else {
    return -length;
  }
}

// Returns r := (a+b)/2 for a>b.
// Ensure that:
// - r is finite.
// - r > a.
// - r = b if there is not float number in between a and b.
//
inline float MidThreshold(const float a, const float b) {
  float threshold = a + (b - a) / 2.f;
  if (threshold <= a) {
    threshold = b;
  }
  return threshold;
}

// Utility to apply a function over a range of elements using multi-threading.
//
// Given "num_items" elements divided into "num_blocks" contiguous blocks of
// the same size (except possibly for the last one). This method calls
// "function" on each block in parallel using the thread-pool.
//
// The method is blocking until all the "function" call have returned.
//
// For example, support num_items=10 and num_blocks=3 defines the following
// blocks: [0,4), [4,8), [8,10). Then, "function" will be called in parallel on:
//   function(block_idx=0, begin_item_idx=0, end_item_idx=0)
//   function(block_idx=1, begin_item_idx=4, end_item_idx=8)
//   function(block_idx=2, begin_item_idx=8, end_item_idx=10)
//
void ConcurrentForLoop(
    size_t num_blocks, utils::concurrency::ThreadPool* thread_pool,
    size_t num_items,
    const std::function<void(size_t block_idx, size_t begin_item_idx,
                             size_t end_item_idx)>& function);

}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_UTILS_H_
