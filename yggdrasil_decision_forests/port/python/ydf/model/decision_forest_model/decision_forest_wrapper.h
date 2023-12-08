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

#ifndef YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_YDF_MODEL_DECISION_FOREST_MODEL_DECISION_FOREST_WRAPPER_H_
#define YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_YDF_MODEL_DECISION_FOREST_MODEL_DECISION_FOREST_WRAPPER_H_

#include <pybind11/numpy.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_forest_interface.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "ydf/model/model_wrapper.h"
#include "yggdrasil_decision_forests/utils/logging.h"

namespace py = ::pybind11;

namespace yggdrasil_decision_forests::port::python {

class DecisionForestCCModel : public GenericCCModel {
 public:
  int num_trees() const { return df_model_->num_trees(); }

  absl::StatusOr<py::array_t<int32_t>> PredictLeaves(
      const dataset::VerticalDataset& dataset);

  absl::StatusOr<py::array_t<float>> Distance(
      const dataset::VerticalDataset& dataset1,
      const dataset::VerticalDataset& dataset2);

  // Sets the format for saving the model's nodes.
  void set_node_format(const std::string& node_format) {
    df_model_->set_node_format(node_format);
  }

  // Returns the nodes of the tree in a depth-first, negative-first, transversal
  // order.
  absl::StatusOr<std::vector<model::decision_tree::proto::Node>> GetTree(
      int tree_idx) const;

 protected:
  // `model` and `df_model` must correspond to the same object.
  DecisionForestCCModel(std::unique_ptr<model::AbstractModel>&& model,
                        model::DecisionForestInterface* df_model)
      : GenericCCModel(std::move(model)), df_model_(df_model) {}

 private:
  // This is a non-owning pointer to the model held by `model_`.
  model::DecisionForestInterface* df_model_;
};

}  // namespace yggdrasil_decision_forests::port::python

#endif  // YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_YDF_MODEL_DECISION_FOREST_MODEL_DECISION_FOREST_WRAPPER_H_
