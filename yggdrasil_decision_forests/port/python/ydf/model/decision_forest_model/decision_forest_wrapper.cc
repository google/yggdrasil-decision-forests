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

#include "ydf/model/decision_forest_model/decision_forest_wrapper.h"

#include <pybind11/numpy.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/protobuf.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace py = ::pybind11;

namespace yggdrasil_decision_forests::port::python {

absl::StatusOr<py::array_t<int32_t>> DecisionForestCCModel::PredictLeaves(
    const dataset::VerticalDataset& dataset) {
  py::array_t<int32_t, py::array::c_style | py::array::forcecast> leaves;

  const size_t num_examples = dataset.nrow();
  const size_t num_trees = df_model_->num_trees();

  leaves.resize({num_examples, num_trees});
  auto unchecked_leaves = leaves.mutable_unchecked();
  for (size_t example_idx = 0; example_idx < num_examples; example_idx++) {
    auto dst = absl::MakeSpan(unchecked_leaves.mutable_data(example_idx, 0),
                              num_trees);
    RETURN_IF_ERROR(df_model_->PredictGetLeaves(dataset, example_idx, dst));
  }

  return leaves;
}

absl::StatusOr<py::array_t<float>> DecisionForestCCModel::Distance(
    const dataset::VerticalDataset& dataset1,
    const dataset::VerticalDataset& dataset2) {
  py::array_t<float, py::array::c_style | py::array::forcecast> distances;
  const size_t n1 = dataset1.nrow();
  const size_t n2 = dataset2.nrow();
  distances.resize({n1, n2});
  auto dst = absl::MakeSpan(distances.mutable_data(), n1 * n2);
  RETURN_IF_ERROR(df_model_->Distance(dataset1, dataset2, dst));
  return distances;
}

absl::StatusOr<std::vector<model::decision_tree::proto::Node>>
DecisionForestCCModel::GetTree(int tree_idx) const {
  using Node = model::decision_tree::proto::Node;

  if (tree_idx < 0 || tree_idx >= df_model_->num_trees()) {
    return absl::InvalidArgumentError("Invalid tree index");
  }

  struct Writer : utils::ProtoWriterInterface<Node> {
    virtual ~Writer() = default;
    absl::Status Write(const Node& value) override {
      nodes.push_back(value);
      return absl::OkStatus();
    }
    std::vector<Node> nodes;
  } writer;

  RETURN_IF_ERROR(df_model_->decision_trees()[tree_idx]->WriteNodes(&writer));
  return writer.nodes;
}

}  // namespace yggdrasil_decision_forests::port::python
