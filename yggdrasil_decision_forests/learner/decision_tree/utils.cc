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

#include "yggdrasil_decision_forests/learner/decision_tree/utils.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/bitmap.h"
#include "yggdrasil_decision_forests/utils/logging.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {

void SetPositiveAttributeSetOfCategoricalContainsCondition(
    const std::vector<int32_t>& positive_attribute_value,
    const int32_t num_attribute_classes, proto::NodeCondition* condition) {
  // Estimate the memory usage of both representation.
  const int64_t usage_bitmap = (num_attribute_classes + 7) / 8;
  const int64_t usage_vector =
      sizeof(int32_t) * (positive_attribute_value.size());
  // What is the most compact way to encode the set of positive attribute
  // values?
  if (usage_bitmap <= usage_vector) {
    auto* cond_bitmap = condition->mutable_condition()
                            ->mutable_contains_bitmap_condition()
                            ->mutable_elements_bitmap();
    utils::bitmap::AllocateAndZeroBitMap(num_attribute_classes, cond_bitmap);
    for (const auto attribute_value : positive_attribute_value) {
      utils::bitmap::SetValueBit(attribute_value, cond_bitmap);
    }
  } else {
    auto* cond_vector =
        condition->mutable_condition()->mutable_contains_condition();
    cond_vector->mutable_elements()->Clear();
    cond_vector->mutable_elements()->Reserve(positive_attribute_value.size());
    for (const auto attribute_value : positive_attribute_value) {
      cond_vector->mutable_elements()->Add(attribute_value);
    }
    // The elements are expected to be sorted i.e. sorted by increasing item
    // index value.
    // Note: There are not guarantied correlation between the order of the
    // item indices and the ratio of positive label.
    std::sort(cond_vector->mutable_elements()->begin(),
              cond_vector->mutable_elements()->end());
  }
}

void SetPositiveAttributeSetOfCategoricalContainsCondition(
    const std::vector<std::pair<float, int32_t>>&
        ratio_true_label_by_attr_value,
    const int32_t begin_positive_idx, const int32_t num_attribute_classes,
    proto::NodeCondition* condition) {
  DCHECK_GE(begin_positive_idx, 0);
  DCHECK_LT(begin_positive_idx, ratio_true_label_by_attr_value.size());
  std::vector<int32_t> positive_attribute_value;
  positive_attribute_value.reserve(ratio_true_label_by_attr_value.size() -
                                   begin_positive_idx);
  for (int32_t item_idx = begin_positive_idx;
       item_idx < ratio_true_label_by_attr_value.size(); item_idx++) {
    const auto attribute_value =
        ratio_true_label_by_attr_value[item_idx].second;
    positive_attribute_value.push_back(attribute_value);
  }
  SetPositiveAttributeSetOfCategoricalContainsCondition(
      positive_attribute_value, num_attribute_classes, condition);
}

}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
