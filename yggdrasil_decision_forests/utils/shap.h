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

// Implementation of the TreeSHAP algorithm described in "Consistent
// Individualized Feature Attribution for Tree Ensembles" by Lundberg et al.
// https://arxiv.org/pdf/1802.03888

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_SHAP_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_SHAP_H_

#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"

namespace yggdrasil_decision_forests::utils::shap {
namespace internal {

// PathItem used during in the TreeSHAP computation.
struct PathItem {
  int column_idx;        // "d" in paper.
  double zero_fraction;  // "z" in paper.
  double one_fraction;   // "o" in paper.
  double weight;         // "w" in paper.

  friend bool operator==(const PathItem& a, const PathItem& b);

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const PathItem& p) {
    absl::Format(&sink,
                 "{column_idx:%d zero_fraction:%f one_fraction:%f weight:%f}",
                 p.column_idx, p.zero_fraction, p.one_fraction, p.weight);
  }
};

// A path is a sequence of PathItems.
typedef std::vector<PathItem> Path;

template <typename Sink>
void AbslStringify(Sink& sink, const Path& p) {
  sink.Append(absl::StrCat("[\n", absl::StrJoin(p, ", \n"), "\n]"));
}

// Adds a node to the path buffer.
// Note: The "extend" function the TreeSHAP paper.
void extend(double zero_fraction, double one_fraction, int attribute_idx,
            Path& path);

// Removes a node to the path buffer.
// Note: The "unwind" function the TreeSHAP paper.
void unwind(int path_idx, Path& path);

// Computes the sum of all weights of items on the path after "unwind"-ing.,
// without the expensive operation of calling "unwind". This trick is not used
// in the paper, but instead is used in the SHAP python package.
double unwound_sum(int path_idx, Path& path);

}  // namespace internal

}  // namespace yggdrasil_decision_forests::utils::shap

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_SHAP_H_
