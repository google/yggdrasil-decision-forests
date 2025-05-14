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

#include <stddef.h>

#include <string>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"

namespace yggdrasil_decision_forests::utils::shap {

// Container of SHAP values for a single example.
class ExampleShapValues {
 public:
  // Creates an empty struct. Should be "Initialized" before any other
  // operation.
  ExampleShapValues() {}

  // Initializes with all zero-values.
  void Initialize(int num_columns, int num_outputs, bool compute_bias);

  // Gets the index in "values()" for a given column and output.
  int Index(const int column_idx, const int output_idx) const {
    return output_idx + column_idx * num_outputs_;
  }

  // Shap values array of shape [num columns x num outputs].
  std::vector<double>& values() { return values_; }
  const std::vector<double>& values() const { return values_; }

  // Biases array of shape [num outputs].
  std::vector<double>& bias() { return bias_; }
  const std::vector<double>& bias() const { return bias_; }

  int num_outputs() const { return num_outputs_; }
  int num_columns() const { return num_columns_; }

  // Human readable string representation.
  std::string ToString(
      const dataset::proto::DataSpecification& data_spec) const;

  // Sums the shap values for a given output.
  double SumValues(int output_idx) const;

 private:
  std::vector<double> values_;
  std::vector<double> bias_;
  int num_outputs_ = 0;
  int num_columns_ = 0;
};

// Computes the SHAP values using the TreeSHAP algorithm and save them
// in "shap_values". The biases (e.g. "shap_values.bias()") is only computed
// if "compute_bias = true".
absl::Status tree_shap(const model::AbstractModel& model,
                       const dataset::proto::Example& example,
                       ExampleShapValues* shap_values,
                       bool compute_bias = true);

// Computes the shape of the SHAP values computed by "tree_shap".
struct Shape {
  size_t num_attributes;
  size_t num_outputs;
};
absl::StatusOr<Shape> GetShape(const model::AbstractModel& model);

namespace internal {

// Signature of a function that get the "value" of a leaf node.
typedef absl::FunctionRef<double(
    const model::decision_tree::NodeWithChildren& node, const int output_idx)>
    NodeValueFn;

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

// Explores the structure of a tree and populates the shape values.
// Note: The "recurse" function the TreeSHAP paper.
// Note: "path" should NOT be a reference (this is the algorithm described in
// the paper).
// TODO: Optimize the algorithm and make "path" a reference i.e. avoid
// copies.
absl::Status recurse(
    const model::decision_tree::NodeWithChildren& node,  // "j" in paper
    double zero_fraction,                                // "p_z" in paper.
    double one_fraction,                                 // "p_o" in paper.
    int attribute_idx,                                   // "p_i" in paper.
    NodeValueFn node_value_fn, bool multi_output_trees, int tree_idx,
    const dataset::proto::Example& example,
    Path path,                      // "m" in paper.
    ExampleShapValues& shap_values  // "phi" in paper.
);

}  // namespace internal

}  // namespace yggdrasil_decision_forests::utils::shap

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_SHAP_H_
